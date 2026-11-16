import ATCore
import Foundation
import MLX
import MLXNN
import MLXRandom

/// Pixel2Play-style image augmentation on the GPU. Parameters are drawn once per clip and applied
/// to every frame of it (temporal consistency); only sensor noise varies per frame.
struct Augmenter {
    let enabled: Set<Augmentation>
    let strength: Float

    init(_ list: [Augmentation], strength: Double) {
        enabled = Set(list)
        self.strength = Float(strength)
    }

    private func chance(_ p: Double) -> Bool { Double.random(in: 0 ..< 1) < p }
    private func u(_ a: Float, _ b: Float) -> Float { Float.random(in: a ... b) }

    /// `frames`: `[T, S, S, 3]` uint8 (or float 0-255). Returns the same shape in `dtype`, values 0-255.
    func callAsFunction(_ frames: MLXArray, dtype: DType = .float32) -> MLXArray {
        guard strength > 0, !enabled.isEmpty else { return frames.asType(dtype) }
        let s = strength
        let S = frames.dim(1)

        // Geometry: small rotation/scale always (P2P spatial, p = 1), horizontal shift p = 0.25.
        var angle: Float = 0, scale: Float = 1, tx: Float = 0, ty: Float = 0
        if enabled.contains(.spatial) { angle = u(-3, 3) * s * .pi / 180; scale = 1 + u(-0.04, 0.04) * s; ty = u(-0.01, 0.01) * s }
        if enabled.contains(.translation), chance(0.25) { tx = u(-0.03, 0.03) * s }
        let geometry = angle != 0 || scale != 1 || tx != 0 || ty != 0
        // Draw the photometric choices up front, so the common case (geometry only) is a single pass.
        let doColor = enabled.contains(.color) && chance(0.25)
        let doPlanck = enabled.contains(.planckian) && chance(0.25)
        let doBlur = enabled.contains(.blur) && chance(0.2)
        let doSharp = enabled.contains(.sharpness) && chance(0.15)
        let doNoise = enabled.contains(.isoNoise) && chance(0.1)
        let onlyGeometry = !(doColor || doPlanck || doBlur || doSharp || doNoise)

        var x: MLXArray
        if geometry && frames.dtype == .uint8 {
            x = Self.warpUInt8(frames, angle: angle, scale: scale, tx: tx * Float(S), ty: ty * Float(S), final: onlyGeometry, dtype: onlyGeometry ? dtype : .float32)
            if onlyGeometry { return x }
        } else {
            x = frames.asType(.float32)
            if geometry { x = warp(x, angle: angle, scale: scale, tx: tx * Float(S), ty: ty * Float(S)) }
        }

        if doColor || doPlanck {
            // Colour jitter and white balance in one Metal pass (same math as `colorJitter`).
            var b: Float = 1, c: Float = 1, sat: Float = 1, hue: Float = 0
            if doColor { b = 1 + u(-0.2, 0.2) * s; c = 1 + u(-0.2, 0.2) * s; sat = 1 + u(-0.2, 0.2) * s; hue = u(-0.05, 0.05) * s }
            var gains: [Float] = [1, 1, 1]
            if doPlanck {
                let t = u(-1, 1) * min(s, 1.5)
                gains = t > 0 ? [1, 1 - 0.07 * t, 1 - 0.2 * t] : [1 + 0.15 * t, 1 + 0.07 * t, 1]
            }
            x = Self.color(x, brightness: b, contrast: c, saturation: sat, hue: hue, gains: gains, jitter: doColor)
        }
        if doBlur { x = Self.fastBlur(x, sigma: u(0.3, 1.6) * max(s, 0.3)) }
        if doSharp {
            let amount = u(0.5, 1.5)
            x = x + (x - Self.fastBlur(x, sigma: 1)) * (amount - 1)
        }
        if doNoise {
            let sigma = u(2, 12) * s
            x = x + MLXRandom.normal(x.shape) * sigma
        }
        return clip(x, min: 0, max: 255).round().asType(dtype)
    }

    static func hueMatrix(_ hue: Float) -> [Float] {
        let th = hue * 2 * .pi, cs = cos(th), sn = sin(th)
        let toYIQ: [Float] = [0.299, 0.587, 0.114, 0.596, -0.274, -0.322, 0.211, -0.523, 0.312]
        let rot: [Float] = [1, 0, 0, 0, cs, -sn, 0, sn, cs]
        let fromYIQ: [Float] = [1, 0.956, 0.621, 1, -0.272, -0.647, 1, -1.106, 1.703]
        func mul(_ a: [Float], _ b: [Float]) -> [Float] {
            (0 ..< 9).map { i in (0 ..< 3).reduce(Float(0)) { $0 + a[(i / 3) * 3 + $1] * b[$1 * 3 + i % 3] } }
        }
        return mul(fromYIQ, mul(rot, toYIQ))
    }

    static let colorKernel = MLXFast.metalKernel(
        name: "augment_color",
        inputNames: ["src", "means", "params"], outputNames: ["out"],
        source: """
            uint p = thread_position_in_grid.x;
            uint t = thread_position_in_grid.y;
            if (p >= uint(PIXELS)) return;
            uint i = (t * PIXELS + p) * 3;
            float r = src[i], g = src[i + 1], b = src[i + 2];
            if (JITTER) {
                float br = params[0], ct = params[1], st = params[2];
                float m = means[t] * br;
                r *= br; g *= br; b *= br;
                r = (r - m) * ct + m; g = (g - m) * ct + m; b = (b - m) * ct + m;
                float g2 = r * 0.299f + g * 0.587f + b * 0.114f;
                r = (r - g2) * st + g2; g = (g - g2) * st + g2; b = (b - g2) * st + g2;
                if (HUE) {
                    float nr = r * params[3] + g * params[4] + b * params[5];
                    float ng = r * params[6] + g * params[7] + b * params[8];
                    float nb = r * params[9] + g * params[10] + b * params[11];
                    r = nr; g = ng; b = nb;
                }
            }
            out[i] = r * params[12]; out[i + 1] = g * params[13]; out[i + 2] = b * params[14];
            """)

    static func color(_ x: MLXArray, brightness b: Float, contrast c: Float, saturation sat: Float, hue: Float, gains: [Float], jitter: Bool) -> MLXArray {
        let T = x.dim(0), P = x.dim(1) * x.dim(2)
        let means = jitter ? (x[.ellipsis, 0 ..< 1] * 0.299 + x[.ellipsis, 1 ..< 2] * 0.587 + x[.ellipsis, 2 ..< 3] * 0.114).mean(axes: [1, 2, 3]) : MLXArray.zeros([T])
        let params = MLXArray([b, c, sat] + hueMatrix(hue) + gains)
        return colorKernel([x, means, params], template: [("PIXELS", P), ("JITTER", jitter), ("HUE", hue != 0)],
                           grid: (P, T, 1), threadGroup: (256, 1, 1), outputShapes: [[T, x.dim(1), x.dim(2), 3]], outputDTypes: [.float32])[0]
    }

    /// Separable 5-tap Gaussian with edge clamping (same as `blur`), one Metal pass per axis.
    static let blurKernel = MLXFast.metalKernel(
        name: "augment_blur",
        inputNames: ["src", "w"], outputNames: ["out"],
        source: """
            uint p = thread_position_in_grid.x;
            uint t = thread_position_in_grid.y;
            if (p >= uint(S * S)) return;
            int y = p / S, x = p % S;
            uint base = t * S * S * 3;
            float acc0 = 0, acc1 = 0, acc2 = 0;
            for (int k = -2; k <= 2; k++) {
                int yy = VERTICAL ? clamp(y + k, 0, S - 1) : y;
                int xx = VERTICAL ? x : clamp(x + k, 0, S - 1);
                uint j = base + (yy * S + xx) * 3;
                float wk = w[k + 2];
                acc0 += src[j] * wk; acc1 += src[j + 1] * wk; acc2 += src[j + 2] * wk;
            }
            uint o = base + p * 3;
            out[o] = acc0; out[o + 1] = acc1; out[o + 2] = acc2;
            """)

    static func fastBlur(_ x: MLXArray, sigma: Float) -> MLXArray {
        let k = (-2 ... 2).map { exp(-Float($0 * $0) / (2 * sigma * sigma)) }
        let sum = k.reduce(0, +)
        let w = MLXArray(k.map { $0 / sum })
        let T = x.dim(0), S = x.dim(1)
        func pass(_ a: MLXArray, vertical: Bool) -> MLXArray {
            blurKernel([a, w], template: [("S", S), ("VERTICAL", vertical)], grid: (S * S, T, 1), threadGroup: (256, 1, 1),
                       outputShapes: [[T, S, S, 3]], outputDTypes: [.float32])[0]
        }
        return pass(pass(x, vertical: false), vertical: true)
    }

    /// `warp` as one Metal pass straight from uint8 frames (identical bilinear math in fp32).
    /// With `final`, also clips and rounds like the end of the pipeline.
    static let warpKernel = MLXFast.metalKernel(
        name: "augment_warp",
        inputNames: ["src", "params"], outputNames: ["out"],
        source: """
            uint p = thread_position_in_grid.x;
            uint t = thread_position_in_grid.y;
            if (p >= uint(S * S)) return;
            float ca = params[0], sa = params[1], tx = params[2], ty = params[3];
            float c = float(S - 1) / 2.0f;
            float yy = float(p / S) - c, xx = float(p % S) - c;
            float sx = clamp(xx * ca - yy * sa + c - tx, 0.0f, float(S - 1));
            float sy = clamp(xx * sa + yy * ca + c - ty, 0.0f, float(S - 1));
            float x0 = floor(sx), y0 = floor(sy);
            float x1 = min(x0 + 1.0f, float(S - 1)), y1 = min(y0 + 1.0f, float(S - 1));
            float wx = sx - x0, wy = sy - y0;
            uint base = t * S * S * 3;
            uint i00 = base + (uint(y0) * S + uint(x0)) * 3, i01 = base + (uint(y0) * S + uint(x1)) * 3;
            uint i10 = base + (uint(y1) * S + uint(x0)) * 3, i11 = base + (uint(y1) * S + uint(x1)) * 3;
            for (int ch = 0; ch < 3; ch++) {
                float top = float(src[i00 + ch]) * (1.0f - wx) + float(src[i01 + ch]) * wx;
                float bottom = float(src[i10 + ch]) * (1.0f - wx) + float(src[i11 + ch]) * wx;
                float v = top * (1.0f - wy) + bottom * wy;
                if (FINAL) v = rint(clamp(v, 0.0f, 255.0f));
                out[base + p * 3 + ch] = OUT_T(v);
            }
            """)

    static func warpUInt8(_ frames: MLXArray, angle: Float, scale: Float, tx: Float, ty: Float, final: Bool, dtype: DType) -> MLXArray {
        let T = frames.dim(0), S = frames.dim(1)
        let params = MLXArray([cos(-angle) / scale, sin(-angle) / scale, tx, ty])
        return warpKernel([frames, params], template: [("S", S), ("FINAL", final), ("OUT_T", dtype)],
                          grid: (S * S, T, 1), threadGroup: (256, 1, 1),
                          outputShapes: [[T, S, S, 3]], outputDTypes: [dtype])[0]
    }

    /// Bilinear warp about the image center with edge clamping.
    func warp(_ x: MLXArray, angle: Float, scale: Float, tx: Float, ty: Float) -> MLXArray {
        let T = x.dim(0), S = x.dim(1)
        let c = Float(S - 1) / 2
        let idx = MLXArray(0 ..< S).asType(.float32)
        let yy = broadcast(idx.expandedDimensions(axis: 1), to: [S, S]) - c
        let xx = broadcast(idx.expandedDimensions(axis: 0), to: [S, S]) - c
        let ca = cos(-angle) / scale, sa = sin(-angle) / scale
        let sx = clip(xx * ca - yy * sa + c - tx, min: 0, max: Float(S - 1))
        let sy = clip(xx * sa + yy * ca + c - ty, min: 0, max: Float(S - 1))
        let x0 = floor(sx), y0 = floor(sy)
        let x1 = minimum(x0 + 1, MLXArray(Float(S - 1))), y1 = minimum(y0 + 1, MLXArray(Float(S - 1)))
        let wx = (sx - x0).expandedDimensions(axis: -1), wy = (sy - y0).expandedDimensions(axis: -1)
        let flat = x.reshaped(T, S * S, 3)
        func gather(_ yi: MLXArray, _ xi: MLXArray) -> MLXArray {
            let i = (yi * Float(S) + xi).asType(.int32).reshaped(-1)
            return flat.take(i, axis: 1).reshaped(T, S, S, 3)
        }
        let top = gather(y0, x0) * (1 - wx) + gather(y0, x1) * wx
        let bottom = gather(y1, x0) * (1 - wx) + gather(y1, x1) * wx
        return top * (1 - wy) + bottom * wy
    }

    func colorJitter(_ x: MLXArray, brightness b: Float, contrast c: Float, saturation sat: Float, hue: Float) -> MLXArray {
        var y = x * b
        let gray = (y[.ellipsis, 0 ..< 1] * 0.299 + y[.ellipsis, 1 ..< 2] * 0.587 + y[.ellipsis, 2 ..< 3] * 0.114)
        let mean = gray.mean(axes: [1, 2, 3], keepDims: true)
        y = (y - mean) * c + mean
        let g2 = (y[.ellipsis, 0 ..< 1] * 0.299 + y[.ellipsis, 1 ..< 2] * 0.587 + y[.ellipsis, 2 ..< 3] * 0.114)
        y = (y - g2) * sat + g2
        if hue != 0 {
            // Rotate chroma in YIQ space.
            let th = hue * 2 * .pi, cs = cos(th), sn = sin(th)
            let toYIQ = MLXArray([0.299, 0.587, 0.114, 0.596, -0.274, -0.322, 0.211, -0.523, 0.312] as [Float]).reshaped(3, 3)
            let rot = MLXArray([1, 0, 0, 0, cs, -sn, 0, sn, cs] as [Float]).reshaped(3, 3)
            let fromYIQ = MLXArray([1, 0.956, 0.621, 1, -0.272, -0.647, 1, -1.106, 1.703] as [Float]).reshaped(3, 3)
            let m = matmul(fromYIQ, matmul(rot, toYIQ))
            y = matmul(y, m.transposed())
        }
        return y
    }

    /// Separable Gaussian blur via depthwise convolution.
    func blur(_ x: MLXArray, sigma: Float) -> MLXArray {
        let r = 2
        let k = (-r ... r).map { exp(-Float($0 * $0) / (2 * sigma * sigma)) }
        let sum = k.reduce(0, +)
        let kernel = MLXArray(k.map { $0 / sum })
        let kh = broadcast(kernel.reshaped(1, 1, 5, 1), to: [3, 1, 5, 1])
        let kv = broadcast(kernel.reshaped(1, 5, 1, 1), to: [3, 5, 1, 1])
        var y = padded(x, widths: [.init(0), .init((0, 0)), .init((r, r)), .init(0)], mode: .edge)
        y = conv2d(y, kh, groups: 3)
        y = padded(y, widths: [.init(0), .init((r, r)), .init((0, 0)), .init(0)], mode: .edge)
        return conv2d(y, kv, groups: 3)
    }
}

/// Developer check: the Metal warp against the reference MLX warp, and their speed.
public enum AugmentCheck {
    public static func run() {
        let T = 300, S = 192
        let frames = MLXRandom.randInt(low: 0, high: 256, [T, S, S, 3]).asType(.uint8)
        eval(frames)
        let aug = Augmenter([.spatial], strength: 1)
        for (angle, scale, tx, ty) in [(Float(0.03), Float(1.03), Float(2.5), Float(-1.2)), (-0.05, 0.97, -5.1, 1.9)] {
            let ref = clip(aug.warp(frames.asType(.float32), angle: angle, scale: scale, tx: tx, ty: ty), min: 0, max: 255).round()
            let mine = Augmenter.warpUInt8(frames, angle: angle, scale: scale, tx: tx, ty: ty, final: true, dtype: .float32)
            let diff = abs(ref - mine)
            print(String(format: "warp check: max diff %.1f, pixels differing %.4f%%", diff.max().item(Float.self), (diff .> 0).asType(.float32).mean().item(Float.self) * 100))
        }
        func time(_ label: String, _ f: () -> MLXArray) {
            eval(f()); let t0 = Date(); for _ in 0 ..< 5 { eval(f()) }
            print(String(format: "%@: %.1f ms per 300-frame clip", label, Date().timeIntervalSince(t0) / 5 * 1000))
        }
        time("reference warp (MLX ops)") { clip(aug.warp(frames.asType(.float32), angle: 0.03, scale: 1.03, tx: 2.5, ty: -1.2), min: 0, max: 255).round().asType(.bfloat16) }
        time("Metal warp") { Augmenter.warpUInt8(frames, angle: 0.03, scale: 1.03, tx: 2.5, ty: -1.2, final: true, dtype: .bfloat16) }
        let f32 = frames.asType(.float32); eval(f32)
        time("upload 300 frames (Data → MLXArray)") { MLXArray(Data(count: T * S * S * 3), [T, S, S, 3], dtype: .uint8) }
        time("color jitter (incl. hue)") { aug.colorJitter(f32, brightness: 1.1, contrast: 0.9, saturation: 1.1, hue: 0.02) }
        time("color jitter, Metal") { Augmenter.color(f32, brightness: 1.1, contrast: 0.9, saturation: 1.1, hue: 0.02, gains: [1, 1, 1], jitter: true) }
        let cref = aug.colorJitter(f32, brightness: 1.1, contrast: 0.9, saturation: 1.1, hue: 0.02) * MLXArray([Float(1.05), 0.97, 0.9])
        let cm = Augmenter.color(f32, brightness: 1.1, contrast: 0.9, saturation: 1.1, hue: 0.02, gains: [1.05, 0.97, 0.9], jitter: true)
        print(String(format: "color check: max |ref − Metal| %.4f (of 0-255)", abs(cref - cm).max().item(Float.self)))
        let bref = aug.blur(f32, sigma: 1.2), bm = Augmenter.fastBlur(f32, sigma: 1.2)
        print(String(format: "blur check: max |ref − Metal| %.4f (of 0-255)", abs(bref - bm).max().item(Float.self)))
        time("blur, Metal") { Augmenter.fastBlur(f32, sigma: 1) }
        time("planckian gains") { f32 * MLXArray([Float(1), 0.95, 0.9]) }
        time("blur") { aug.blur(f32, sigma: 1) }
        time("noise") { f32 + MLXRandom.normal(f32.shape) * 5 }
        time("final clip+round+cast") { clip(f32, min: 0, max: 255).round().asType(.bfloat16) }
    }
}
