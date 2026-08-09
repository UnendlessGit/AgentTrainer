import AVFoundation
import CoreMedia
import Foundation
import VideoToolbox

enum HEVCEncodingConfiguration {
    /// A high-quality HEVC target expressed in bits per source pixel. Frame
    /// rates above 30 scale sublinearly because screen recordings commonly
    /// contain repeated regions and the encoder can spend bits where motion
    /// actually occurs. The bounds keep tiny captures useful and extreme
    /// displays from returning to near-lossless-sized files.
    static func targetBitRate(width: Int, height: Int, fps: Double) -> Int {
        guard width > 0, height > 0, fps.isFinite, fps > 0 else { return 1_500_000 }
        let pixels = Double(width) * Double(height)
        let boundedFPS = min(240, max(1, fps))
        let effectiveFPS = min(30, boundedFPS) + max(0, boundedFPS - 30) * 0.5
        let visuallyTransparentTarget = pixels * effectiveFPS * 0.055
        return Int(min(45_000_000, max(1_500_000, visuallyTransparentTarget.rounded())))
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
    private(set) var frameCount = 0
    private(set) var droppedFrameCount = 0

    init(url: URL, width: Int, height: Int, fps: Double) throws {
        self.url = url
        self.width = width
        self.height = height
        requestedFPS = fps
        try? FileManager.default.removeItem(at: url)
        writer = try AVAssetWriter(outputURL: url, fileType: .mov)

        let bitrate = HEVCEncodingConfiguration.targetBitRate(width: width, height: height, fps: fps)
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
        guard CMSampleBufferDataIsReady(sampleBuffer) else {
            droppedFrameCount += 1
            return
        }
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        if !started {
            guard writer.startWriting() else { return }
            writer.startSession(atSourceTime: pts)
            firstPTS = pts
            started = true
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

    func finish() async throws -> (duration: Double, deliveredFPS: Double, frames: Int) {
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
        let hostEnd = CMClockGetTime(CMClockGetHostTimeClock())
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
