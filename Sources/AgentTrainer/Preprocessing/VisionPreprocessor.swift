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
        try withPackedBytes(pixelBuffer, spec: spec) { Data($0) }
    }

    /// Exposes the completed shared Metal buffer only for the duration of the
    /// callback. Cache construction can copy it directly into its large output
    /// buffer instead of allocating and copying an intermediate Data per frame.
    func withPackedBytes<R>(_ pixelBuffer: CVPixelBuffer, spec: PreprocessingSpec, _ body: (UnsafeRawBufferPointer) throws -> R) throws -> R {
        let spec = try spec.validated()
        lock.lock()
        defer { lock.unlock() }
        guard let textureCache else { throw AgentTrainerError.model("The Metal texture cache is unavailable.") }

        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let sourceWidth = CVPixelBufferGetWidth(pixelBuffer)
        let sourceHeight = CVPixelBufferGetHeight(pixelBuffer)
        var cvLuma: CVMetalTexture?
        var cvChroma: CVMetalTexture?
        let sourceFormat: UInt32
        switch pixelFormat {
        case kCVPixelFormatType_32BGRA:
            sourceFormat = 0
            let status = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, pixelBuffer, nil, .bgra8Unorm, sourceWidth, sourceHeight, 0, &cvLuma)
            guard status == kCVReturnSuccess else { throw AgentTrainerError.model("The BGRA capture frame could not be mapped into Metal.") }
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
            sourceFormat = pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ? 1 : 2
            guard CVPixelBufferGetPlaneCount(pixelBuffer) == 2 else { throw AgentTrainerError.model("The decoded video frame has an invalid YUV plane layout.") }
            let lumaStatus = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, pixelBuffer, nil, .r8Unorm, CVPixelBufferGetWidthOfPlane(pixelBuffer, 0), CVPixelBufferGetHeightOfPlane(pixelBuffer, 0), 0, &cvLuma)
            let chromaStatus = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, pixelBuffer, nil, .rg8Unorm, CVPixelBufferGetWidthOfPlane(pixelBuffer, 1), CVPixelBufferGetHeightOfPlane(pixelBuffer, 1), 1, &cvChroma)
            guard lumaStatus == kCVReturnSuccess, chromaStatus == kCVReturnSuccess else { throw AgentTrainerError.model("The decoded YUV video frame could not be mapped into Metal.") }
        default:
            throw AgentTrainerError.model("The shared vision pipeline received an unsupported pixel format.")
        }
        guard let cvLuma, let lumaTexture = CVMetalTextureGetTexture(cvLuma) else { throw AgentTrainerError.model("The vision frame texture is unavailable.") }
        let chromaTexture = cvChroma.flatMap(CVMetalTextureGetTexture)
        if sourceFormat != 0, chromaTexture == nil { throw AgentTrainerError.model("The decoded video chroma texture is unavailable.") }

        if reusableOutput?.length ?? 0 < spec.sampleByteCount { reusableOutput = device.makeBuffer(length: spec.sampleByteCount, options: .storageModeShared) }
        guard let output = reusableOutput, let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw AgentTrainerError.model("Metal could not allocate the preprocessing workload.")
        }

        var params = KernelParameters(
            sourceWidth: UInt32(sourceWidth), sourceHeight: UInt32(sourceHeight),
            outputWidth: UInt32(spec.width), outputHeight: UInt32(spec.height),
            bitDepth: UInt32(spec.bitDepth), mode: spec.colorMode == .grayscale ? 0 : 1,
            chroma: UInt32(spec.chroma == .yuv420 ? 0 : spec.chroma == .yuv422 ? 1 : 2),
            resize: UInt32(spec.resizePolicy == .fit ? 0 : spec.resizePolicy == .fill ? 1 : 2),
            sourceFormat: sourceFormat, sourceMatrix: Self.sourceMatrix(for: pixelBuffer)
        )
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(lumaTexture, index: 0)
        encoder.setTexture(chromaTexture, index: 1)
        encoder.setBuffer(output, offset: 0, index: 0)
        encoder.setBytes(&params, length: MemoryLayout<KernelParameters>.stride, index: 1)
        let threads = MTLSize(width: 16, height: 16, depth: 1)
        encoder.dispatchThreads(MTLSize(width: spec.width, height: spec.height, depth: 1), threadsPerThreadgroup: threads)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw error }
        return try body(UnsafeRawBufferPointer(start: output.contents(), count: spec.sampleByteCount))
    }

    private static func sourceMatrix(for pixelBuffer: CVPixelBuffer) -> UInt32 {
        guard let matrix = CVBufferCopyAttachment(pixelBuffer, kCVImageBufferYCbCrMatrixKey, nil) else { return 0 }
        if CFEqual(matrix, kCVImageBufferYCbCrMatrix_ITU_R_601_4) { return 1 }
        if CFEqual(matrix, kCVImageBufferYCbCrMatrix_ITU_R_2020) { return 2 }
        return 0
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

    /// Builds the temporal vision input used by Policy v4. Supplying both the
    /// current frame and its signed difference from the preceding perception
    /// gives the policy motion/velocity evidence without running the CNN once
    /// per history frame. A missing predecessor deliberately produces a zero
    /// difference at recording and runtime segment boundaries.
    static func mlxTemporalTensor(current: Data, previous: Data?, batch: Int, spec: PreprocessingSpec) -> MLXArray {
        let currentTensor = mlxTensor(current, batch: batch, spec: spec)
        let previousTensor = previous.map { mlxTensor($0, batch: batch, spec: spec) } ?? currentTensor
        return temporalTensor(current: currentTensor, previous: previousTensor)
    }

    static func temporalTensor(current: MLXArray, previous: MLXArray) -> MLXArray {
        concatenated([current, current - previous], axis: -1)
    }

    /// Renders the exact packed Y/Cb/Cr values consumed by the policy. This is only
    /// a presentation conversion; the preview never reprocesses or recaptures a frame.
    static func previewImage(_ packed: Data, spec: PreprocessingSpec, maximumWidth: Int = .max, maximumHeight: Int = .max) -> NSImage? {
        guard spec.width > 0, spec.height > 0, spec.sampleByteCount < Int.max else { return nil }
        let lumaCount = spec.width * spec.height
        let chromaWidth = spec.chroma == .yuv444 ? spec.width : (spec.width + 1) / 2
        let chromaHeight = spec.chroma == .yuv420 ? (spec.height + 1) / 2 : spec.height
        let chromaCount = chromaWidth * chromaHeight
        let requiredCount = spec.colorMode == .grayscale ? lumaCount : lumaCount + 2 * chromaCount
        guard packed.count >= requiredCount else { return nil }
        let scale = min(1, min(Double(max(1, maximumWidth)) / Double(spec.width), Double(max(1, maximumHeight)) / Double(spec.height)))
        let outputWidth = max(1, Int((Double(spec.width) * scale).rounded()))
        let outputHeight = max(1, Int((Double(spec.height) * scale).rounded()))
        let outputPixels = outputWidth.multipliedReportingOverflow(by: outputHeight)
        let outputBytes = outputPixels.partialValue.multipliedReportingOverflow(by: 4)
        guard !outputPixels.overflow, !outputBytes.overflow else { return nil }
        var rgba = [UInt8](repeating: 255, count: outputBytes.partialValue)
        packed.withUnsafeBytes { raw in
            guard let bytes = raw.bindMemory(to: UInt8.self).baseAddress else { return }
            for outputY in 0..<outputHeight {
                let y = min(spec.height - 1, outputY * spec.height / outputHeight)
                for outputX in 0..<outputWidth {
                    let x = min(spec.width - 1, outputX * spec.width / outputWidth)
                    let sourcePixel = y * spec.width + x
                    let outputPixel = outputY * outputWidth + outputX
                    let luma = Float(bytes[sourcePixel]) / 255
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
                    rgba[outputPixel * 4] = UInt8(clamping: Int((min(1, max(0, r)) * 255).rounded()))
                    rgba[outputPixel * 4 + 1] = UInt8(clamping: Int((min(1, max(0, g)) * 255).rounded()))
                    rgba[outputPixel * 4 + 2] = UInt8(clamping: Int((min(1, max(0, b)) * 255).rounded()))
                }
            }
        }
        guard let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: outputWidth, pixelsHigh: outputHeight, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: outputWidth * 4, bitsPerPixel: 32), let destination = bitmap.bitmapData else { return nil }
        rgba.withUnsafeBytes { source in if let base = source.baseAddress { memcpy(destination, base, rgba.count) } }
        let image = NSImage(size: NSSize(width: outputWidth, height: outputHeight))
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
        var sourceFormat: UInt32
        var sourceMatrix: UInt32
    }

    private static let kernelSource = #"""
    #include <metal_stdlib>
    using namespace metal;
    struct Params { uint sw, sh, ow, oh, bits, mode, chroma, resize, sourceFormat, sourceMatrix; };

    kernel void packVision(texture2d<float, access::sample> lumaInput [[texture(0)]],
                           texture2d<float, access::sample> chromaInput [[texture(1)]],
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
        float3 rgb;
        if (outside) {
            rgb = float3(0.0);
        } else if (p.sourceFormat == 0) {
            rgb = lumaInput.sample(s, uv).rgb;
        } else {
            float yCode = lumaInput.sample(s, uv).r;
            float2 chromaCode = chromaInput.sample(s, uv).rg;
            float y = p.sourceFormat == 1 ? (yCode * 255.0 - 16.0) / 219.0 : yCode;
            float cb = p.sourceFormat == 1 ? (chromaCode.x * 255.0 - 128.0) / 224.0 : (chromaCode.x * 255.0 - 128.0) / 255.0;
            float cr = p.sourceFormat == 1 ? (chromaCode.y * 255.0 - 128.0) / 224.0 : (chromaCode.y * 255.0 - 128.0) / 255.0;
            if (p.sourceMatrix == 1) {
                rgb = float3(y + 1.4020 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.7720 * cb);
            } else if (p.sourceMatrix == 2) {
                rgb = float3(y + 1.4746 * cr, y - 0.164553 * cb - 0.571353 * cr, y + 1.8814 * cb);
            } else {
                rgb = float3(y + 1.5748 * cr, y - 0.187324 * cb - 0.468124 * cr, y + 1.8556 * cb);
            }
            rgb = clamp(rgb, 0.0, 1.0);
        }
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
