import CoreVideo
import Foundation
import Metal

/// GPU downscaler used for **both** training proxies and live inference, so the model sees
/// pixel-identical inputs in both (Pixel2Play found resize mismatches hurt real play).
///
/// Separable Hamming filter with Pillow / fast_image_resize coefficients (support 1, stretched by
/// the downscale factor), 8-bit rounding between passes, squash to a square output like P2P.
public final class FrameResizer: @unchecked Sendable {
    public let device: MTLDevice
    private let queue: MTLCommandQueue
    private let horizontal: MTLComputePipelineState
    private let vertical: MTLComputePipelineState
    private let marker: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache?
    private var plans: [String: Plan] = [:]
    private let lock = NSLock()

    struct Plan {
        let srcW: Int, srcH: Int, dstW: Int, dstH: Int
        let hTaps: MTLBuffer, hWeights: MTLBuffer, hMax: Int
        let vTaps: MTLBuffer, vWeights: MTLBuffer, vMax: Int
        let mid: MTLBuffer
        let out: MTLBuffer
    }

    public init?(device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        guard let device, let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.queue = queue
        do {
            let lib = try device.makeLibrary(source: Self.shaderSource, options: nil)
            horizontal = try device.makeComputePipelineState(function: lib.makeFunction(name: "resize_h")!)
            vertical = try device.makeComputePipelineState(function: lib.makeFunction(name: "resize_v")!)
            marker = try device.makeComputePipelineState(function: lib.makeFunction(name: "cursor_marker")!)
        } catch {
            return nil
        }
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
    }

    /// Pillow's Hamming filter.
    static func hamming(_ x: Double) -> Double {
        let x = abs(x)
        if x == 0 { return 1 }
        if x >= 1 { return 0 }
        let px = x * Double.pi
        return sin(px) / px * (0.54 + 0.46 * cos(px))
    }

    /// Per-output tap start/count and normalized weights (Pillow `precompute_coeffs`).
    static func coefficients(inSize: Int, outSize: Int) -> (taps: [UInt32], weights: [Float], maxTaps: Int) {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1)
        let support = 1.0 * filterScale
        let maxTaps = Int(ceil(support)) * 2 + 1
        var taps = [UInt32](), weights = [Float](repeating: 0, count: outSize * maxTaps)
        for x in 0 ..< outSize {
            let center = (Double(x) + 0.5) * scale
            let xmin = max(0, Int(center - support + 0.5))
            let xmax = min(inSize, Int(center + support + 0.5)) - xmin
            var w = [Double](repeating: 0, count: max(xmax, 0))
            var total = 0.0
            for i in 0 ..< max(xmax, 0) {
                w[i] = hamming((Double(i + xmin) - center + 0.5) / filterScale)
                total += w[i]
            }
            for i in 0 ..< min(w.count, maxTaps) { weights[x * maxTaps + i] = Float(total != 0 ? w[i] / total : 0) }
            taps += [UInt32(xmin), UInt32(min(max(xmax, 0), maxTaps))]
        }
        return (taps, weights, maxTaps)
    }

    private func plan(srcW: Int, srcH: Int, dstW: Int, dstH: Int) -> Plan {
        let key = "\(srcW)x\(srcH)->\(dstW)x\(dstH)"
        if let p = plans[key] { return p }
        let h = Self.coefficients(inSize: srcW, outSize: dstW), v = Self.coefficients(inSize: srcH, outSize: dstH)
        func buf<T>(_ a: [T]) -> MTLBuffer { device.makeBuffer(bytes: a, length: MemoryLayout<T>.stride * a.count, options: .storageModeShared)! }
        let p = Plan(srcW: srcW, srcH: srcH, dstW: dstW, dstH: dstH,
                     hTaps: buf(h.taps), hWeights: buf(h.weights), hMax: h.maxTaps,
                     vTaps: buf(v.taps), vWeights: buf(v.weights), vMax: v.maxTaps,
                     mid: device.makeBuffer(length: srcH * dstW * 4, options: .storageModePrivate)!,
                     out: device.makeBuffer(length: dstW * dstH * 3, options: .storageModeShared)!)
        plans[key] = p
        return p
    }

    /// Resizes a BGRA pixel buffer to `size`×`size` RGB8. `cursor` (normalized 0-1) draws the
    /// pointer marker used by absolute-pointer models. Calls `body` with the RGB bytes.
    @discardableResult
    public func resize<R>(_ pixelBuffer: CVPixelBuffer, size: Int, cursor: CGPoint? = nil,
                          _ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R? {
        lock.lock(); defer { lock.unlock() }
        let w = CVPixelBufferGetWidth(pixelBuffer), h = CVPixelBufferGetHeight(pixelBuffer)
        guard let cache = textureCache else { return nil }
        var cvTex: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, cache, pixelBuffer, nil, .bgra8Unorm, w, h, 0, &cvTex)
        guard let cvTex, let tex = CVMetalTextureGetTexture(cvTex) else { return nil }
        let p = plan(srcW: w, srcH: h, dstW: size, dstH: size)
        guard let cmd = queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else { return nil }
        var hMax = UInt32(p.hMax), vMax = UInt32(p.vMax), dstW = UInt32(size), dstH = UInt32(size), srcH = UInt32(h)
        enc.setComputePipelineState(horizontal)
        enc.setTexture(tex, index: 0)
        enc.setBuffer(p.mid, offset: 0, index: 0)
        enc.setBuffer(p.hTaps, offset: 0, index: 1)
        enc.setBuffer(p.hWeights, offset: 0, index: 2)
        enc.setBytes(&hMax, length: 4, index: 3)
        enc.setBytes(&dstW, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: size, height: h, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.setComputePipelineState(vertical)
        enc.setBuffer(p.mid, offset: 0, index: 0)
        enc.setBuffer(p.out, offset: 0, index: 1)
        enc.setBuffer(p.vTaps, offset: 0, index: 2)
        enc.setBuffer(p.vWeights, offset: 0, index: 3)
        enc.setBytes(&vMax, length: 4, index: 4)
        enc.setBytes(&dstW, length: 4, index: 5)
        enc.setBytes(&dstH, length: 4, index: 6)
        enc.setBytes(&srcH, length: 4, index: 7)
        enc.dispatchThreads(MTLSize(width: size, height: size, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        if let c = cursor {
            var pos = SIMD2<Float>(Float(c.x) * Float(size), Float(c.y) * Float(size))
            enc.setComputePipelineState(marker)
            enc.setBuffer(p.out, offset: 0, index: 0)
            enc.setBytes(&pos, length: 8, index: 1)
            enc.setBytes(&dstW, length: 4, index: 2)
            enc.setBytes(&dstH, length: 4, index: 3)
            enc.dispatchThreads(MTLSize(width: 9, height: 9, depth: 1), threadsPerThreadgroup: MTLSize(width: 9, height: 9, depth: 1))
        }
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        return try body(UnsafeRawBufferPointer(start: p.out.contents(), count: size * size * 3))
    }

    /// Draws the cursor marker into an existing RGB8 image (CPU path for proxies made offline).
    public static func drawCursorMarker(_ rgb: UnsafeMutableRawBufferPointer, size: Int, at c: CGPoint) {
        let p = rgb.bindMemory(to: UInt8.self)
        let cx = Int((c.x * Double(size)).rounded(.down)), cy = Int((c.y * Double(size)).rounded(.down))
        for dy in -4 ... 4 {
            for dx in -4 ... 4 where abs(dx) <= 1 || abs(dy) <= 1 {
                let x = cx + dx, y = cy + dy
                guard x >= 0, y >= 0, x < size, y < size, abs(dx) + abs(dy) <= 5 else { continue }
                let o = (y * size + x) * 3
                let edge = abs(dx) == 4 || abs(dy) == 4
                p[o] = edge ? 0 : 255; p[o + 1] = edge ? 0 : 0; p[o + 2] = edge ? 0 : 255
            }
        }
    }

    static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void resize_h(texture2d<float, access::read> src [[texture(0)]],
                         device uchar4 *mid [[buffer(0)]],
                         device const uint2 *taps [[buffer(1)]],
                         device const float *weights [[buffer(2)]],
                         constant uint &maxTaps [[buffer(3)]],
                         constant uint &dstW [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dstW || gid.y >= src.get_height()) return;
        uint2 t = taps[gid.x];
        float4 acc = 0;
        for (uint i = 0; i < t.y; i++) acc += src.read(uint2(t.x + i, gid.y)) * weights[gid.x * maxTaps + i];
        mid[gid.y * dstW + gid.x] = uchar4(clamp(rint(acc * 255.0), 0.0, 255.0));
    }

    kernel void resize_v(device const uchar4 *mid [[buffer(0)]],
                         device uchar *out [[buffer(1)]],
                         device const uint2 *taps [[buffer(2)]],
                         device const float *weights [[buffer(3)]],
                         constant uint &maxTaps [[buffer(4)]],
                         constant uint &dstW [[buffer(5)]],
                         constant uint &dstH [[buffer(6)]],
                         constant uint &srcH [[buffer(7)]],
                         uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dstW || gid.y >= dstH) return;
        uint2 t = taps[gid.y];
        float4 acc = 0;
        for (uint i = 0; i < t.y; i++) acc += float4(mid[(t.x + i) * dstW + gid.x]) * weights[gid.y * maxTaps + i];
        uchar4 v = uchar4(clamp(rint(acc), 0.0, 255.0));
        uint o = (gid.y * dstW + gid.x) * 3;
        out[o] = v.r; out[o + 1] = v.g; out[o + 2] = v.b;
    }

    // A small magenta plus with a dark outline: keeps the pointer visible at 192 px.
    kernel void cursor_marker(device uchar *out [[buffer(0)]],
                              constant float2 &pos [[buffer(1)]],
                              constant uint &w [[buffer(2)]],
                              constant uint &h [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
        int dx = int(gid.x) - 4, dy = int(gid.y) - 4;
        if (!(abs(dx) <= 1 || abs(dy) <= 1) || abs(dx) + abs(dy) > 5) return;
        int x = int(floor(pos.x)) + dx, y = int(floor(pos.y)) + dy;
        if (x < 0 || y < 0 || x >= int(w) || y >= int(h)) return;
        bool edge = abs(dx) == 4 || abs(dy) == 4;
        uint o = (uint(y) * w + uint(x)) * 3;
        out[o] = edge ? 0 : 255; out[o + 1] = 0; out[o + 2] = edge ? 0 : 255;
    }
    """
}
