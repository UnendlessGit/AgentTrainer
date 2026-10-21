import Accelerate
import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

enum HEVCEncodingConfiguration {
    /// A high-quality HEVC target expressed in bits per source pixel. Frame
    /// rates above 30 scale sublinearly because screen recordings commonly
    /// contain repeated regions and the encoder can spend bits where motion
    /// actually occurs. The bounds keep tiny captures useful and extreme
    /// displays from returning to near-lossless-sized files.
    static func targetBitRate(
        width: Int,
        height: Int,
        fps: Double,
        compression: Double = RecordingVisualSettings.defaultCompression
    ) -> Int {
        guard width > 0, height > 0, fps.isFinite, fps > 0 else { return 1_500_000 }
        let pixels = Double(width) * Double(height)
        let boundedFPS = min(240, max(1, fps))
        let effectiveFPS = min(30, boundedFPS) + max(0, boundedFPS - 30) * 0.5
        let visuallyTransparentTarget = pixels * effectiveFPS * 0.055
        let defaultTarget = min(45_000_000, max(1_500_000, visuallyTransparentTarget))
        let boundedCompression = min(1, max(0, compression.isFinite ? compression : RecordingVisualSettings.defaultCompression))
        // Exponential scaling gives the whole slider useful range while the
        // default remains bit-for-bit equivalent to the 2.1 bitrate target.
        let compressionMultiplier = pow(
            0.12,
            boundedCompression - RecordingVisualSettings.defaultCompression
        )
        return Int(min(100_000_000, max(250_000, (defaultTarget * compressionMultiplier).rounded())))
    }
}

/// Legal-range quantization preserves neutral chroma in low-bit recordings.
struct RecordingColorQuantizationTables: Sendable {
    let videoRangeLuma: [UInt8]
    let videoRangeChroma: [UInt8]

    init(bits: Int) {
        let bits = min(8, max(1, bits))
        videoRangeLuma = Self.luma(bits: bits, lower: 16, upper: 235)
        videoRangeChroma = Self.chroma(bits: bits, lower: 16, neutral: 128, upper: 240)
    }

    private static func luma(bits: Int, lower: Int, upper: Int) -> [UInt8] {
        let levels = Double((1 << bits) - 1)
        let extent = Double(upper - lower)
        return (0...255).map { raw in
            let value = min(upper, max(lower, raw))
            let normalized = Double(value - lower) / extent
            let quantized = (normalized * levels).rounded() / levels
            return UInt8(clamping: lower + Int((quantized * extent).rounded()))
        }
    }

    private static func chroma(
        bits: Int,
        lower: Int,
        neutral: Int,
        upper: Int
    ) -> [UInt8] {
        let sideLevels = Double(1 << max(0, bits - 1))
        return (0...255).map { raw in
            let value = min(upper, max(lower, raw))
            let delta = value - neutral
            guard delta != 0 else { return UInt8(neutral) }
            let extent = Double(delta < 0 ? neutral - lower : upper - neutral)
            let normalizedMagnitude = Double(abs(delta)) / extent
            let quantizedMagnitude = (normalizedMagnitude * sideLevels).rounded() / sideLevels
            let signed = delta < 0 ? -quantizedMagnitude : quantizedMagnitude
            return UInt8(clamping: neutral + Int((signed * extent).rounded()))
        }
    }
}

final class HEVCWriter: @unchecked Sendable {
    let url: URL
    let width: Int
    let height: Int
    let requestedFPS: Double

    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private var started = false
    private var firstPTS: CMTime?
    private var lastPTS: CMTime?
    private var sessionEndPTS: CMTime?
    private var generatedFormatDescription: CMVideoFormatDescription?
    private let quantizer: RecordingColorQuantizer?
    private(set) var frameCount = 0
    private(set) var droppedFrameCount = 0

    init(
        url: URL,
        width: Int,
        height: Int,
        fps: Double,
        colorBits: Int = 8,
        compression: Double = RecordingVisualSettings.defaultCompression
    ) throws {
        self.url = url
        self.width = width
        self.height = height
        requestedFPS = fps
        try? FileManager.default.removeItem(at: url)
        writer = try AVAssetWriter(outputURL: url, fileType: .mov)

        guard (1...8).contains(colorBits) else {
            throw AgentTrainerError.invalidConfiguration("Recording color detail must be 1 through 8 bits.")
        }
        quantizer = colorBits < 8
            ? try RecordingColorQuantizer(width: width, height: height, bits: colorBits)
            : nil
        let bitrate = HEVCEncodingConfiguration.targetBitRate(
            width: width,
            height: height,
            fps: fps,
            compression: compression
        )
        let encoderSpecification: [String: Any] = [
            kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder as String: true,
            kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder as String: true
        ]
        let compression: [String: Any] = [
            kVTCompressionPropertyKey_AverageBitRate as String: bitrate,
            AVVideoExpectedSourceFrameRateKey: fps,
            AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main_AutoLevel,
            kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration as String: 6,
            kVTCompressionPropertyKey_AllowTemporalCompression as String: true,
            kVTCompressionPropertyKey_AllowFrameReordering as String: true,
            kVTCompressionPropertyKey_AllowOpenGOP as String: true,
            kVTCompressionPropertyKey_PrioritizeEncodingSpeedOverQuality as String: false,
            kVTCompressionPropertyKey_RealTime as String: true
        ]
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: compression,
            AVVideoEncoderSpecificationKey: encoderSpecification
        ]
        input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true
        guard writer.canAdd(input) else { throw AgentTrainerError.capture("The hardware HEVC writer could not accept the selected capture format.") }
        writer.add(input)
    }

    func append(_ sampleBuffer: CMSampleBuffer) {
        append(sampleBuffer, waitsForCapacity: false, appliesColorQuantization: true)
    }

    private func append(
        _ sampleBuffer: CMSampleBuffer,
        waitsForCapacity: Bool,
        appliesColorQuantization: Bool
    ) {
        guard CMSampleBufferDataIsReady(sampleBuffer) else {
            droppedFrameCount += 1
            return
        }
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        if appliesColorQuantization, let quantizer {
            guard let source = sampleBuffer.imageBuffer,
                  let quantized = quantizer.makeBuffer(from: source) else {
                droppedFrameCount += 1
                return
            }
            append(quantized, presentationTimeStamp: pts)
            return
        }
        if !started {
            guard writer.startWriting() else { return }
            writer.startSession(atSourceTime: pts)
            firstPTS = pts
            started = true
        }
        if waitsForCapacity {
            let deadline = ContinuousClock.now.advanced(by: .seconds(2))
            while !input.isReadyForMoreMediaData,
                  writer.status == .writing,
                  ContinuousClock.now < deadline {
                Thread.sleep(forTimeInterval: 0.001)
            }
        }
        guard input.isReadyForMoreMediaData else {
            droppedFrameCount += 1
            lastPTS = pts
            return
        }
        if input.append(sampleBuffer) {
            frameCount += 1
            lastPTS = pts
        } else {
            droppedFrameCount += 1
        }
    }

    /// Wrap a quantized surface in the original capture timestamp.
    func append(_ pixelBuffer: CVPixelBuffer, presentationTimeStamp: CMTime) {
        if generatedFormatDescription == nil {
            var description: CMVideoFormatDescription?
            guard CMVideoFormatDescriptionCreateForImageBuffer(
                allocator: kCFAllocatorDefault,
                imageBuffer: pixelBuffer,
                formatDescriptionOut: &description
            ) == noErr else {
                droppedFrameCount += 1
                return
            }
            generatedFormatDescription = description
        }
        guard let generatedFormatDescription else {
            droppedFrameCount += 1
            return
        }
        var timing = CMSampleTimingInfo(
            duration: .invalid,
            presentationTimeStamp: presentationTimeStamp,
            decodeTimeStamp: .invalid
        )
        var sample: CMSampleBuffer?
        guard CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: generatedFormatDescription,
            sampleTiming: &timing,
            sampleBufferOut: &sample
        ) == noErr, let sample else {
            droppedFrameCount += 1
            return
        }
        append(sample, waitsForCapacity: true, appliesColorQuantization: false)
    }

    func finish(endPTS requestedEndPTS: CMTime? = nil) async throws -> (duration: Double, deliveredFPS: Double, frames: Int) {
        guard started else {
            if writer.status == .failed {
                throw writer.error ?? AgentTrainerError.capture("HEVC encoding failed before the first frame was written.")
            }
            writer.cancelWriting()
            return (0, 0, 0)
        }
        if writer.status == .failed {
            throw writer.error ?? AgentTrainerError.capture("HEVC encoding failed before finalization.")
        }
        // ScreenCaptureKit intentionally emits no complete frames while the
        // selected content is static. Ending the writer session at the current
        // host time makes the final encoded frame persist for the real recording
        // duration instead of collapsing a long idle recording to one frame.
        let hostEnd = requestedEndPTS ?? CMClockGetTime(CMClockGetHostTimeClock())
        if let firstPTS, hostEnd.isNumeric, CMTimeCompare(hostEnd, firstPTS) > 0 {
            writer.endSession(atSourceTime: hostEnd)
            sessionEndPTS = hostEnd
        }
        input.markAsFinished()
        await writer.finishWriting()
        if writer.status == .failed { throw writer.error ?? AgentTrainerError.capture("HEVC encoding failed.") }
        let duration: Double
        if let firstPTS, let sessionEndPTS {
            duration = max(0, CMTimeGetSeconds(sessionEndPTS - firstPTS))
        } else if let firstPTS, let lastPTS {
            let span = max(0, CMTimeGetSeconds(lastPTS - firstPTS))
            let finalFrameDuration = frameCount > 1 ? span / Double(frameCount - 1) : 1 / max(1, requestedFPS)
            duration = span + finalFrameDuration
        } else { duration = 0 }
        return (duration, duration > 0 ? Double(frameCount) / duration : 0, frameCount)
    }

}

/// Fast lookup-table quantization for ScreenCaptureKit's bi-planar recording
/// surfaces. A CVPixelBufferPool avoids a full-resolution allocation per frame;
/// the 8-bit setting bypasses this path entirely for zero-copy recording.
private final class RecordingColorQuantizer: @unchecked Sendable {
    private let pool: CVPixelBufferPool
    private let lumaTable: [UInt8]
    private let chromaTable: [UInt8]

    init(width: Int, height: Int, bits: Int) throws {
        let poolAttributes: [CFString: Any] = [
            kCVPixelBufferPoolMinimumBufferCountKey: 3
        ]
        let pixelAttributes: [CFString: Any] = [
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        var optionalPool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(
            kCFAllocatorDefault,
            poolAttributes as CFDictionary,
            pixelAttributes as CFDictionary,
            &optionalPool
        ) == kCVReturnSuccess, let optionalPool else {
            throw AgentTrainerError.capture("The recording color-detail buffer pool could not be created.")
        }
        pool = optionalPool
        let tables = RecordingColorQuantizationTables(bits: bits)
        lumaTable = tables.videoRangeLuma
        chromaTable = tables.videoRangeChroma
    }

    func makeBuffer(from source: CVPixelBuffer) -> CVPixelBuffer? {
        guard CVPixelBufferGetPixelFormatType(source) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange else {
            return nil
        }
        var optionalDestination: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(
            kCFAllocatorDefault,
            pool,
            &optionalDestination
        ) == kCVReturnSuccess, let destination = optionalDestination else { return nil }
        CVBufferPropagateAttachments(source, destination)
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard CVPixelBufferGetPlaneCount(source) == 2,
              CVPixelBufferGetPlaneCount(destination) == 2 else { return nil }
        for plane in 0..<2 {
            guard let sourceBase = CVPixelBufferGetBaseAddressOfPlane(source, plane),
                  let destinationBase = CVPixelBufferGetBaseAddressOfPlane(destination, plane) else { return nil }
            let activeWidth = plane == 0
                ? CVPixelBufferGetWidthOfPlane(source, plane)
                : CVPixelBufferGetWidthOfPlane(source, plane) * 2
            var sourceImage = vImage_Buffer(
                data: sourceBase,
                height: vImagePixelCount(CVPixelBufferGetHeightOfPlane(source, plane)),
                width: vImagePixelCount(activeWidth),
                rowBytes: CVPixelBufferGetBytesPerRowOfPlane(source, plane)
            )
            var destinationImage = vImage_Buffer(
                data: destinationBase,
                height: vImagePixelCount(CVPixelBufferGetHeightOfPlane(destination, plane)),
                width: vImagePixelCount(activeWidth),
                rowBytes: CVPixelBufferGetBytesPerRowOfPlane(destination, plane)
            )
            let table = plane == 0 ? lumaTable : chromaTable
            let result = table.withUnsafeBufferPointer {
                vImageTableLookUp_Planar8(&sourceImage, &destinationImage, $0.baseAddress!, vImage_Flags(kvImageNoFlags))
            }
            guard result == kvImageNoError else { return nil }
        }
        return destination
    }
}
