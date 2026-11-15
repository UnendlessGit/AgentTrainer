import ATCore
import CoreMedia
import CoreVideo
import Foundation
import ScreenCaptureKit

/// One captured frame with its display time on the shared host clock.
public struct CapturedFrame: @unchecked Sendable {
    public let pixelBuffer: CVPixelBuffer
    public let sampleBuffer: CMSampleBuffer
    /// When the frame was displayed (host clock, ns) - the same timeline as input events.
    public let hostNs: UInt64
}

public struct CaptureSettings: Hashable, Sendable {
    public var width: Int
    public var height: Int
    public var fps: Int
    public var showsCursor: Bool
    public init(width: Int, height: Int, fps: Int, showsCursor: Bool) {
        self.width = width; self.height = height; self.fps = fps; self.showsCursor = showsCursor
    }

    /// Output size for a target scaled to fit `maxHeight` (keeps aspect, even dimensions).
    public static func fitting(_ target: CaptureTarget, maxHeight: Int, fps: Int, showsCursor: Bool) -> CaptureSettings {
        let scale = target.pointPixelScale
        let pw = target.frame.width * scale, ph = target.frame.height * scale
        let h = min(Double(maxHeight), ph)
        let w = (pw * h / max(ph, 1)).rounded()
        return CaptureSettings(width: Int(w) & ~1, height: Int(h) & ~1, fps: fps, showsCursor: showsCursor)
    }
}

/// ScreenCaptureKit stream delivering BGRA frames with host-clock display timestamps.
public final class ScreenCapturer: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    public var onFrame: ((CapturedFrame) -> Void)?
    public var onStop: ((Error?) -> Void)?
    private var stream: SCStream?
    private let queue = DispatchQueue(label: "agenttrainer.capture", qos: .userInteractive)
    public private(set) var settings: CaptureSettings?

    public override init() {}

    public func start(_ target: CaptureTarget, settings s: CaptureSettings, excluding own: [SCRunningApplication]) async throws {
        let cfg = SCStreamConfiguration()
        cfg.width = s.width
        cfg.height = s.height
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(s.fps))
        cfg.pixelFormat = kCVPixelFormatType_32BGRA
        cfg.colorSpaceName = CGColorSpace.sRGB
        cfg.showsCursor = s.showsCursor
        cfg.queueDepth = 6
        cfg.scalesToFit = true
        cfg.ignoreShadowsSingleWindow = true
        cfg.capturesAudio = false
        let stream = SCStream(filter: target.filter(excludingSelf: own), configuration: cfg, delegate: self)
        try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: queue)
        try await stream.startCapture()
        self.stream = stream
        settings = s
    }

    public func stop() async {
        guard let s = stream else { return }
        stream = nil
        try? await s.stopCapture()
    }

    public func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, sb.isValid, let pb = sb.imageBuffer else { return }
        guard let attachments = CMSampleBufferGetSampleAttachmentsArray(sb, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
              let info = attachments.first,
              let statusRaw = info[.status] as? Int, SCFrameStatus(rawValue: statusRaw) == .complete else { return }
        let host: UInt64
        if let display = info[.displayTime] as? UInt64 { host = HostClock.ticksToNanoseconds(display) } else { host = HostClock.now() }
        onFrame?(CapturedFrame(pixelBuffer: pb, sampleBuffer: sb, hostNs: host))
    }

    public func stream(_ stream: SCStream, didStopWithError error: Error) {
        self.stream = nil
        onStop?(error)
    }
}
