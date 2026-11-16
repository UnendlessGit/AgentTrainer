import Foundation
import MLX

/// Depthwise 2-D convolution (NHWC, weight `[C, K, K, 1]`) with a hand-written backward pass.
///
/// The forward is MLX's grouped `conv2d`. MLX's own gradients for grouped convolutions are slow:
/// the input gradient of a strided depthwise conv took ~40× the forward and the weight gradient
/// ~10×. The backward here is the exact same math in fast forms:
/// - input gradient: zero-insert the cotangent for strided convs, then correlate it with the
///   spatially flipped kernel at stride 1 (a fast depthwise conv);
/// - weight gradient: for each kernel tap, the per-channel sum of input × cotangent, in fp32.
enum DepthwiseConv {
    private static let weightGradKernel = MLXFast.metalKernel(
        name: "depthwise_weight_grad",
        inputNames: ["xp", "gy"], outputNames: ["out"],
        source: """
            uint c = thread_position_in_grid.x;
            uint r = thread_position_in_grid.y;
            if (c >= C || r >= uint(gy_shape[0] * HO)) return;
            uint n = r / HO, i = r % HO;
            float acc[K * K];
            for (int t = 0; t < K * K; t++) acc[t] = 0.0f;
            for (int j = 0; j < WO; j++) {
                float g = float(gy[((n * HO + i) * WO + j) * C + c]);
                for (int ky = 0; ky < K; ky++) {
                    uint base = ((n * HP + i * S + ky) * WP + j * S) * C + c;
                    for (int kx = 0; kx < K; kx++) acc[ky * K + kx] += g * float(xp[base + kx * C]);
                }
            }
            uint o = (r * C + c) * (K * K);
            for (int t = 0; t < K * K; t++) out[o + t] = acc[t];
            """)

    nonisolated(unsafe) private static var cache: [String: @Sendable ([MLXArray]) -> [MLXArray]] = [:]
    private static let lock = NSLock()
    /// A/B switch for benchmarks and tests.
    nonisolated(unsafe) static var customBackward = true

    static func apply(_ x: MLXArray, _ w: MLXArray, stride s: Int, padding p: Int) -> MLXArray {
        guard customBackward else { return conv2d(x, w, stride: .init(s), padding: .init(p), groups: x.dim(-1)) }
        let key = "\(s)/\(p)"
        lock.lock()
        let f: @Sendable ([MLXArray]) -> [MLXArray]
        if let c = cache[key] { f = c } else { f = make(stride: s, padding: p); cache[key] = f }
        lock.unlock()
        return f([x, w])[0]
    }

    private static func make(stride s: Int, padding p: Int) -> @Sendable ([MLXArray]) -> [MLXArray] {
        CustomFunction {
            Forward { a in [conv2d(a[0], a[1], stride: .init(s), padding: .init(p), groups: a[0].dim(-1))] }
            VJP { primals, cotangents in
                let x = primals[0], w = primals[1], gy = cotangents[0].asType(x.dtype)
                let N = x.dim(0), H = x.dim(1), W = x.dim(2), C = x.dim(3), K = w.dim(1)
                let Ho = gy.dim(1), Wo = gy.dim(2)
                // Input gradient. Stride 1: MLX's own (fast). Strided: zero-insert the cotangent,
                // then dx[j] = Σ_k up(gy)[j + p − k] · w[k] as a stride-1 depthwise correlation.
                let dx: MLXArray
                if s == 1 {
                    dx = vjp({ a in [conv2d(a[0], w, stride: .init(1), padding: .init(p), groups: C)] }, primals: [x], cotangents: [gy]).1[0]
                } else {
                    let g = padded(gy.reshaped(N, Ho, 1, Wo, 1, C),
                                   widths: [.init(0), .init(0), .init((0, s - 1)), .init(0), .init((0, s - 1)), .init(0)])
                        .reshaped(N, s * Ho, s * Wo, C)
                    let left = K - 1 - p
                    func fit(_ a: MLXArray, axis: Int, length: Int) -> MLXArray {
                        let cur = a.dim(axis)
                        if cur == length { return a }
                        if cur > length { return axis == 1 ? a[0..., 0 ..< length, 0..., 0...] : a[0..., 0..., 0 ..< length, 0...] }
                        var widths: [IntOrPair] = [.init(0), .init(0), .init(0), .init(0)]
                        widths[axis] = .init((0, length - cur))
                        return padded(a, widths: widths)
                    }
                    var gp = padded(g, widths: [.init(0), .init((left, 0)), .init((left, 0)), .init(0)])
                    gp = fit(fit(gp, axis: 1, length: H + K - 1), axis: 2, length: W + K - 1)
                    let wf = w[0..., .stride(by: -1), .stride(by: -1), 0...]
                    dx = conv2d(gp, wf, stride: .init(1), padding: .init(0), groups: C)
                }
                // Weight gradient: dw[c, ky, kx] = Σ_{n,i,j} gy[n,i,j,c] · x_pad[n, s·i+ky, s·j+kx, c],
                // one thread per (channel, output row) accumulating every tap in registers.
                let xp = padded(x, widths: [.init(0), .init((p, p)), .init((p, p)), .init(0)])
                let rows = N * Ho
                let partial = weightGradKernel([xp, gy],
                    template: [("K", K), ("S", s), ("C", C), ("HO", Ho), ("WO", Wo), ("HP", H + 2 * p), ("WP", W + 2 * p)],
                    grid: (C, rows, 1), threadGroup: (min(C, 32), max(1, min(rows, 256 / min(C, 32))), 1),
                    outputShapes: [[rows, C, K * K]], outputDTypes: [.float32])[0]
                let dw = partial.sum(axis: 0).reshaped(C, K, K, 1).asType(w.dtype)
                return [dx, dw]
            }
        }
    }
}
