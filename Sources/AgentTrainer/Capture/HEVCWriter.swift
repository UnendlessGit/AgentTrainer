import AVFoundation
import CoreMedia
import Foundation
import VideoToolbox

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
    private(set) var frameCount = 0

    init(url: URL, width: Int, height: Int, fps: Double) throws {
        self.url = url
        self.width = width
        self.height = height
        requestedFPS = fps
        try? FileManager.default.removeItem(at: url)
        writer = try AVAssetWriter(outputURL: url, fileType: .mov)

        let bitrate = max(2_000_000, min(120_000_000, width * height * 8))
        let encoderSpecification: [String: Any] = [
            kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder as String: true,
            kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder as String: true
        ]
        let compression: [String: Any] = [
            AVVideoAverageBitRateKey: bitrate,
            AVVideoExpectedSourceFrameRateKey: fps,
            AVVideoMaxKeyFrameIntervalDurationKey: 2,
            AVVideoAllowFrameReorderingKey: false,
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
        guard CMSampleBufferDataIsReady(sampleBuffer) else { return }
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        if !started {
            guard writer.startWriting() else { return }
            writer.startSession(atSourceTime: pts)
            firstPTS = pts
            started = true
        }
        guard input.isReadyForMoreMediaData else { return }
        if input.append(sampleBuffer) {
            frameCount += 1
            lastPTS = pts
        }
    }

    func finish() async throws -> (duration: Double, deliveredFPS: Double, frames: Int) {
        guard started else {
            writer.cancelWriting()
            return (0, 0, 0)
        }
        input.markAsFinished()
        await writer.finishWriting()
        if writer.status == .failed { throw writer.error ?? AgentTrainerError.capture("HEVC encoding failed.") }
        let duration: Double
        if let firstPTS, let lastPTS {
            let span = max(0, CMTimeGetSeconds(lastPTS - firstPTS))
            let finalFrameDuration = frameCount > 1 ? span / Double(frameCount - 1) : 1 / max(1, requestedFPS)
            duration = span + finalFrameDuration
        } else { duration = 0 }
        return (duration, duration > 0 ? Double(frameCount) / duration : 0, frameCount)
    }
}
