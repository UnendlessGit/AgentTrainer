@preconcurrency import CoreVideo
import AppKit
import Foundation
import Metal
import MLX

final class VisionPreprocessor: @unchecked Sendable {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache?
    private var reusableOutput: MTLBuffer?
    private let lock = NSLock()

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw AgentTrainerError.model("Metal is unavailable on this Mac.")
        }
        self.device = device
        self.queue = queue
        let library = try device.makeLibrary(source: Self.kernelSource, options: nil)
        guard let function = library.makeFunction(name: "packVision") else { throw AgentTrainerError.model("The vision preprocessing kernel could not be created.") }
        pipeline = try device.makeComputePipelineState(function: function)
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
    }

    func process(_ pixelBuffer: CVPixelBuffer, spec: PreprocessingSpec) throws -> Data {
        let spec = try spec.validated()
        lock.lock()
        defer { lock.unlock() }
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else {
            throw AgentTrainerError.model("The shared vision pipeline expects BGRA capture frames.")
        }
        guard let textureCache else { throw AgentTrainerError.model("The Metal texture cache is unavailable.") }
        var cvTexture: CVMetalTexture?
        let sourceWidth = CVPixelBufferGetWidth(pixelBuffer)
        let sourceHeight = CVPixelBufferGetHeight(pixelBuffer)
        let status = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, pixelBuffer, nil, .bgra8Unorm, sourceWidth, sourceHeight, 0, &cvTexture)
        guard status == kCVReturnSuccess, let cvTexture, let texture = CVMetalTextureGetTexture(cvTexture) else {
            throw AgentTrainerError.model("The capture frame could not be mapped into Metal.")
        }
        if reusableOutput?.length ?? 0 < spec.sampleByteCount { reusableOutput = device.makeBuffer(length: spec.sampleByteCount, options: .storageModeShared) }
        guard let output = reusableOutput, let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw AgentTrainerError.model("Metal could not allocate the preprocessing workload.")
        }

        var params = KernelParameters(
            sourceWidth: UInt32(sourceWidth), sourceHeight: UInt32(sourceHeight),
            outputWidth: UInt32(spec.width), outputHeight: UInt32(spec.height),
            bitDepth: UInt32(spec.bitDepth), mode: spec.colorMode == .grayscale ? 0 : 1,
            chroma: UInt32(spec.chroma == .yuv420 ? 0 : spec.chroma == .yuv422 ? 1 : 2),
            resize: UInt32(spec.resizePolicy == .fit ? 0 : spec.resizePolicy == .fill ? 1 : 2)
        )
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBuffer(output, offset: 0, index: 0)
        encoder.setBytes(&params, length: MemoryLayout<KernelParameters>.stride, index: 1)
        let threads = MTLSize(width: 16, height: 16, depth: 1)
        encoder.dispatchThreads(MTLSize(width: spec.width, height: spec.height, depth: 1), threadsPerThreadgroup: threads)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw error }
        return Data(bytes: output.contents(), count: spec.sampleByteCount)
    }

    static func unpackFloats(_ packed: Data, spec: PreprocessingSpec) -> [Float] {
        let w = spec.width, h = spec.height
        let bytes = [UInt8](packed)
        if spec.colorMode == .grayscale { return bytes.prefix(w * h).map { Float($0) / 255 } }
        let yCount = w * h
        let chromaWidth = spec.chroma == .yuv444 ? w : (w + 1) / 2
        let chromaHeight = spec.chroma == .yuv420 ? (h + 1) / 2 : h
        let chromaCount = chromaWidth * chromaHeight
        guard bytes.count >= yCount + 2 * chromaCount else { return [] }
        var result = [Float](repeating: 0, count: w * h * 3)
        for y in 0..<h {
            for x in 0..<w {
                let luma = Float(bytes[y * w + x]) / 255
                let cx = spec.chroma == .yuv444 ? x : x / 2
                let cy = spec.chroma == .yuv420 ? y / 2 : y
                let ci = cy * chromaWidth + cx
                result[(y * w + x) * 3] = luma
                result[(y * w + x) * 3 + 1] = Float(bytes[yCount + ci]) / 255
                result[(y * w + x) * 3 + 2] = Float(bytes[yCount + chromaCount + ci]) / 255
            }
        }
        return result
    }

    /// Builds the model-ready tensor directly from packed UInt8 cache bytes. Chroma
    /// expansion and normalization stay in MLX instead of allocating millions of
    /// Swift Float values for every batch.
    static func mlxTensor(_ packed: Data, batch: Int, spec: PreprocessingSpec) -> MLXArray {
        let raw = MLXArray(packed, [batch, spec.sampleByteCount], dtype: .uint8).asType(.float32) / 255
        let width = spec.width, height = spec.height
        let lumaCount = width * height
        let y = raw[0..., 0..<lumaCount].reshaped([batch, height, width, 1])
        guard spec.colorMode == .color else { return y }
        let chromaWidth = spec.chroma == .yuv444 ? width : (width + 1) / 2
        let chromaHeight = spec.chroma == .yuv420 ? (height + 1) / 2 : height
        let chromaCount = chromaWidth * chromaHeight
        var cb = raw[0..., lumaCount..<(lumaCount + chromaCount)].reshaped([batch, chromaHeight, chromaWidth, 1])
        var cr = raw[0..., (lumaCount + chromaCount)..<(lumaCount + 2 * chromaCount)].reshaped([batch, chromaHeight, chromaWidth, 1])
        if chromaHeight != height {
            cb = repeated(cb, count: 2, axis: 1); cr = repeated(cr, count: 2, axis: 1)
        }
        if chromaWidth != width {
            cb = repeated(cb, count: 2, axis: 2); cr = repeated(cr, count: 2, axis: 2)
        }
        cb = cb[0..., 0..<height, 0..<width, 0...]
        cr = cr[0..., 0..<height, 0..<width, 0...]
        return concatenated([y, cb, cr], axis: -1)
    }

    /// Renders the exact packed Y/Cb/Cr values consumed by the policy. This is only
    /// a presentation conversion; the preview never reprocesses or recaptures a frame.
    static func previewImage(_ packed: Data, spec: PreprocessingSpec) -> NSImage? {
        guard spec.width > 0, spec.height > 0, spec.sampleByteCount < Int.max else { return nil }
        let lumaCount = spec.width * spec.height
        let chromaWidth = spec.chroma == .yuv444 ? spec.width : (spec.width + 1) / 2
        let chromaHeight = spec.chroma == .yuv420 ? (spec.height + 1) / 2 : spec.height
        let chromaCount = chromaWidth * chromaHeight
        let requiredCount = spec.colorMode == .grayscale ? lumaCount : lumaCount + 2 * chromaCount
        guard packed.count >= requiredCount else { return nil }
        var rgba = [UInt8](repeating: 255, count: spec.width * spec.height * 4)
        packed.withUnsafeBytes { raw in
            guard let bytes = raw.bindMemory(to: UInt8.self).baseAddress else { return }
            for y in 0..<spec.height {
                for x in 0..<spec.width {
                    let pixel = y * spec.width + x
                    let luma = Float(bytes[pixel]) / 255
                    let r: Float, g: Float, b: Float
                    if spec.colorMode == .grayscale {
                        r = luma; g = luma; b = luma
                    } else {
                        let cx = spec.chroma == .yuv444 ? x : x / 2
                        let cy = spec.chroma == .yuv420 ? y / 2 : y
                        let chromaIndex = cy * chromaWidth + cx
                        let cb = Float(bytes[lumaCount + chromaIndex]) / 255 - 0.5
                        let cr = Float(bytes[lumaCount + chromaCount + chromaIndex]) / 255 - 0.5
                        r = luma + 1.5748 * cr
                        g = luma - 0.1873 * cb - 0.4681 * cr
                        b = luma + 1.8556 * cb
                    }
                    rgba[pixel * 4] = UInt8(clamping: Int((min(1, max(0, r)) * 255).rounded()))
                    rgba[pixel * 4 + 1] = UInt8(clamping: Int((min(1, max(0, g)) * 255).rounded()))
                    rgba[pixel * 4 + 2] = UInt8(clamping: Int((min(1, max(0, b)) * 255).rounded()))
                }
            }
        }
        guard let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: spec.width, pixelsHigh: spec.height, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: spec.width * 4, bitsPerPixel: 32), let destination = bitmap.bitmapData else { return nil }
        rgba.withUnsafeBytes { source in if let base = source.baseAddress { memcpy(destination, base, rgba.count) } }
        let image = NSImage(size: NSSize(width: spec.width, height: spec.height))
        image.addRepresentation(bitmap)
        return image
    }

    private struct KernelParameters {
        var sourceWidth: UInt32
        var sourceHeight: UInt32
        var outputWidth: UInt32
        var outputHeight: UInt32
        var bitDepth: UInt32
        var mode: UInt32
        var chroma: UInt32
        var resize: UInt32
    }

    private static let kernelSource = #"""
    #include <metal_stdlib>
    using namespace metal;
    struct Params { uint sw, sh, ow, oh, bits, mode, chroma, resize; };

    kernel void packVision(texture2d<float, access::sample> input [[texture(0)]],
                           device uchar *output [[buffer(0)]],
                           constant Params &p [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= p.ow || gid.y >= p.oh) return;
        constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
        float2 outSize = float2(p.ow, p.oh);
        float2 srcSize = float2(p.sw, p.sh);
        float2 uv = (float2(gid) + 0.5) / outSize;
        bool outside = false;
        if (p.resize != 2) {
            float srcAspect = srcSize.x / srcSize.y;
            float outAspect = outSize.x / outSize.y;
            if ((p.resize == 0 && srcAspect > outAspect) || (p.resize == 1 && srcAspect < outAspect)) {
                float used = outAspect / srcAspect;
                outside = p.resize == 0 && (uv.y < (1.0-used)*0.5 || uv.y > 1.0-(1.0-used)*0.5);
                uv.y = (uv.y - (1.0-used)*0.5) / used;
            } else {
                float used = srcAspect / outAspect;
                outside = p.resize == 0 && (uv.x < (1.0-used)*0.5 || uv.x > 1.0-(1.0-used)*0.5);
                uv.x = (uv.x - (1.0-used)*0.5) / used;
            }
        }
        float3 rgb = outside ? float3(0.0) : input.sample(s, uv).rgb;
        float Y = clamp(dot(rgb, float3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
        float Cb = clamp((rgb.b - Y) / 1.8556 + 0.5, 0.0, 1.0);
        float Cr = clamp((rgb.r - Y) / 1.5748 + 0.5, 0.0, 1.0);
        float levels = float((1u << p.bits) - 1u);
        uint yi = gid.y * p.ow + gid.x;
        output[yi] = uchar(round(Y * levels) * (255.0 / levels));
        if (p.mode == 0) return;
        uint cw = p.chroma == 2 ? p.ow : (p.ow + 1) / 2;
        uint ch = p.chroma == 0 ? (p.oh + 1) / 2 : p.oh;
        uint sx = p.chroma == 2 ? 1 : 2;
        uint sy = p.chroma == 0 ? 2 : 1;
        if (gid.x % sx == 0 && gid.y % sy == 0) {
            uint ci = (gid.y / sy) * cw + (gid.x / sx);
            uint base = p.ow * p.oh;
            uint cc = cw * ch;
            output[base + ci] = uchar(round(Cb * levels) * (255.0 / levels));
            output[base + cc + ci] = uchar(round(Cr * levels) * (255.0 / levels));
        }
    }
    """#
}
