@preconcurrency import AVFoundation
@preconcurrency import CoreMedia
@preconcurrency import CoreVideo
import Foundation
@preconcurrency import ScreenCaptureKit

struct CaptureSourceOption: Identifiable, Hashable, Sendable {
    enum Kind: Hashable, Sendable { case display, window }
    let id: UInt32
    let kind: Kind
    let name: String
    let detail: String
    let frame: CGRect
}

struct CaptureResult: Sendable {
    let duration: Double
    let deliveredFPS: Double
    let frameCount: Int
    let width: Int
    let height: Int
    let firstFrameHostNanos: UInt64
    let droppedFrameCount: Int
}

enum CaptureFrameDisposition: Equatable, Sendable {
    case deliver
    case reuseLastFrame
    case drop

    static func classify(_ status: SCFrameStatus?) -> Self {
        guard let status else { return .deliver }
        switch status {
        case .complete, .started: return .deliver
        case .idle: return .reuseLastFrame
        case .blank, .suspended, .stopped: return .drop
        @unknown default: return .drop
        }
    }
}

final class CaptureService: NSObject, @unchecked Sendable {
    typealias FrameHandler = @Sendable (CVPixelBuffer, CMTime) -> Void
    typealias IdleFrameHandler = @Sendable (CMTime) -> Void

    private let outputQueue = DispatchQueue(label: "AgentTrainer.ScreenCapture", qos: .userInteractive)
    private var stream: SCStream?
    private var output: StreamOutput?
    private var writer: HEVCWriter?
    private(set) var activeDimensions = CGSize.zero
    private(set) var isRunning = false

    static func availableSources() async throws -> [CaptureSourceOption] {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        var result = content.displays.map {
            CaptureSourceOption(id: $0.displayID, kind: .display, name: "Display \($0.displayID)", detail: "\(Int($0.width)) × \(Int($0.height))", frame: $0.frame)
        }
        result += content.windows.filter { $0.isOnScreen && $0.windowLayer == 0 && $0.frame.width > 40 && $0.frame.height > 40 && $0.owningApplication?.bundleIdentifier != Bundle.main.bundleIdentifier }.map {
            CaptureSourceOption(id: $0.windowID, kind: .window, name: $0.title.flatMap { $0.isEmpty ? nil : $0 } ?? "Untitled Window", detail: $0.owningApplication?.applicationName ?? "Application", frame: $0.frame)
        }
        result.sort { lhs, rhs in
            switch (lhs.kind, rhs.kind) {
            case (.display, .window): return true
            case (.window, .display): return false
            case (.display, .display): return lhs.id < rhs.id
            case (.window, .window):
                let applicationOrder = lhs.detail.localizedStandardCompare(rhs.detail)
                if applicationOrder != .orderedSame { return applicationOrder == .orderedAscending }
                let nameOrder = lhs.name.localizedStandardCompare(rhs.name)
                return nameOrder == .orderedSame ? lhs.id < rhs.id : nameOrder == .orderedAscending
            }
        }
        return result
    }

    func start(spec: CaptureSpec, recordingURL: URL? = nil, exactOutputSize: CGSize? = nil, queueDepth: Int = 8, onFirstFrame: (@Sendable (UInt64) -> Void)? = nil, onFrame: FrameHandler? = nil, onIdle: IdleFrameHandler? = nil, onUnexpectedStop: (@Sendable (Error) -> Void)? = nil) async throws {
        guard !isRunning else { throw AgentTrainerError.capture("A capture stream is already running.") }
        guard CGPreflightScreenCaptureAccess() else {
            CGRequestScreenCaptureAccess()
            throw AgentTrainerError.permission("Screen Recording permission is required. Enable AgentTrainer in System Settings → Privacy & Security → Screen & System Audio Recording.")
        }
        guard spec.requestedFPS.isFinite, spec.requestedFPS > 0, spec.requestedFPS <= 240 else {
            throw AgentTrainerError.invalidConfiguration("Capture FPS must be finite, positive, and no higher than 240.")
        }

        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        let selection = try makeFilterAndGeometry(spec: spec, content: content)
        let outputWidth = exactOutputSize?.width ?? selection.pixelSize.width
        let outputHeight = exactOutputSize?.height ?? selection.pixelSize.height
        guard outputWidth.isFinite, outputHeight.isFinite,
              outputWidth > 0, outputHeight > 0,
              outputWidth <= 16_384, outputHeight <= 16_384 else {
            throw AgentTrainerError.invalidConfiguration("Capture dimensions must be finite, positive, and no larger than 16,384 pixels per side.")
        }
        let config = SCStreamConfiguration()
        let roundedWidth = max(1, Int(outputWidth.rounded()))
        let roundedHeight = max(1, Int(outputHeight.rounded()))
        // Hardware HEVC encoders require an even luma grid on some Apple-silicon
        // generations. Normalize only saved video; exact model capture remains at
        // its immutable requested dimensions.
        config.width = recordingURL == nil ? roundedWidth : max(2, roundedWidth - roundedWidth % 2)
        config.height = recordingURL == nil ? roundedHeight : max(2, roundedHeight - roundedHeight % 2)
        config.minimumFrameInterval = CMTime(seconds: 1 / spec.requestedFPS, preferredTimescale: 1_000_000_000)
        // ScreenCaptureKit documents eight surfaces as the practical upper
        // bound. Keeping the queue bounded avoids runaway IOSurface retention.
        config.queueDepth = min(8, max(1, queueDepth))
        config.showsCursor = spec.showsCursor
        config.capturesAudio = false
        // Saved video goes straight from ScreenCaptureKit's bi-planar 4:2:0
        // surface into the hardware HEVC encoder. Avoiding an intermediate
        // 32-bit BGRA surface cuts capture bandwidth and removes a redundant
        // full-resolution color conversion without changing displayed size.
        config.pixelFormat = recordingURL == nil
            ? kCVPixelFormatType_32BGRA
            : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        config.scalesToFit = exactOutputSize != nil
        config.preservesAspectRatio = exactOutputSize == nil
        if let sourceRect = selection.sourceRect { config.sourceRect = sourceRect }

        let writer = try recordingURL.map { try HEVCWriter(url: $0, width: config.width, height: config.height, fps: spec.requestedFPS) }
        let output = StreamOutput(writer: writer, onFirstFrame: onFirstFrame, onFrame: onFrame, onIdle: onIdle, onUnexpectedStop: onUnexpectedStop)
        let stream = SCStream(filter: selection.filter, configuration: config, delegate: output)
        try stream.addStreamOutput(output, type: .screen, sampleHandlerQueue: outputQueue)
        self.writer = writer
        self.output = output
        self.stream = stream
        activeDimensions = CGSize(width: config.width, height: config.height)
        do {
            try await stream.startCapture()
            isRunning = true
        } catch {
            try? await stream.stopCapture()
            try? stream.removeStreamOutput(output, type: .screen)
            await withCheckedContinuation { continuation in outputQueue.async { continuation.resume() } }
            self.stream = nil; self.output = nil; self.writer = nil; activeDimensions = .zero
            throw error
        }
    }

    func stop() async throws -> CaptureResult {
        guard isRunning, let stream else { return CaptureResult(duration: 0, deliveredFPS: 0, frameCount: 0, width: Int(activeDimensions.width), height: Int(activeDimensions.height), firstFrameHostNanos: 0, droppedFrameCount: 0) }
        var stopError: Error?
        do { try await stream.stopCapture() } catch { stopError = error }
        if let output { try? stream.removeStreamOutput(output, type: .screen) }
        await withCheckedContinuation { continuation in outputQueue.async { continuation.resume() } }
        isRunning = false
        self.stream = nil
        let firstFrameHostNanos = output?.firstFrameHostNanos ?? 0
        let droppedCaptureFrames = output?.droppedFrames ?? 0
        self.output = nil
        let dimensions = activeDimensions
        activeDimensions = .zero
        if let writer {
            // Clear the retained writer before awaiting finalization so a
            // failed encoder cannot leave this CaptureService half-stopped.
            self.writer = nil
            let result = try await writer.finish()
            if let stopError {
                // A ScreenCaptureKit interruption can still leave a complete,
                // finalized HEVC asset. Preserve that recording and surface the
                // interruption through the delegate/log instead of deleting all
                // data received before it.
                AppLog.write(.warning, category: "Capture", "Capture stop reported an error after video finalization", details: stopError.localizedDescription)
            }
            return CaptureResult(duration: result.duration, deliveredFPS: result.deliveredFPS, frameCount: result.frames, width: Int(dimensions.width), height: Int(dimensions.height), firstFrameHostNanos: firstFrameHostNanos, droppedFrameCount: droppedCaptureFrames + writer.droppedFrameCount)
        }
        self.writer = nil
        if let stopError { throw stopError }
        return CaptureResult(duration: 0, deliveredFPS: 0, frameCount: 0, width: Int(dimensions.width), height: Int(dimensions.height), firstFrameHostNanos: firstFrameHostNanos, droppedFrameCount: droppedCaptureFrames)
    }

    private func makeFilterAndGeometry(spec: CaptureSpec, content: SCShareableContent) throws -> (filter: SCContentFilter, pixelSize: CGSize, sourceRect: CGRect?) {
        switch spec.kind {
        case .display, .screenRegion:
            let display = spec.displayID.flatMap { id in content.displays.first { $0.displayID == id } } ?? content.displays.first
            guard let display else { throw AgentTrainerError.capture("No display is available.") }
            let ownApplication = content.applications.first { $0.bundleIdentifier == Bundle.main.bundleIdentifier }
            let excluded = ownApplication.map { [$0] } ?? []
            let filter = SCContentFilter(display: display, excludingApplications: excluded, exceptingWindows: [])
            let scaleX = CGFloat(display.width) / max(1, display.frame.width)
            let scaleY = CGFloat(display.height) / max(1, display.frame.height)
            if spec.kind == .screenRegion, let global = spec.region?.cgRect {
                guard global.hasFiniteComponents else { throw AgentTrainerError.capture("The selected screen region is invalid.") }
                let intersection = global.intersection(display.frame)
                guard !intersection.isNull, intersection.width > 0, intersection.height > 0 else { throw AgentTrainerError.capture("The selected region does not intersect the selected display.") }
                let local = CGRect(x: intersection.minX - display.frame.minX, y: intersection.minY - display.frame.minY, width: intersection.width, height: intersection.height)
                return (filter, CGSize(width: local.width * scaleX, height: local.height * scaleY), local)
            }
            return (filter, CGSize(width: display.width, height: display.height), nil)
        case .window, .windowRegion:
            guard let id = spec.windowID, let window = content.windows.first(where: { $0.windowID == id }) else { throw AgentTrainerError.capture("The selected window is no longer available.") }
            let filter = SCContentFilter(desktopIndependentWindow: window)
            let scale = screenScale(for: window.frame)
            if spec.kind == .windowRegion, let region = spec.region?.cgRect {
                guard region.hasFiniteComponents else { throw AgentTrainerError.capture("The selected window region is invalid.") }
                let bounded = region.intersection(CGRect(origin: .zero, size: window.frame.size))
                guard bounded.width > 0, bounded.height > 0 else { throw AgentTrainerError.capture("The selected window region is empty.") }
                return (filter, CGSize(width: bounded.width * scale, height: bounded.height * scale), bounded)
            }
            return (filter, CGSize(width: window.frame.width * scale, height: window.frame.height * scale), nil)
        }
    }

    private func screenScale(for frame: CGRect) -> CGFloat {
        NSScreen.screens.first { $0.frame.intersects(frame) }?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 2
    }
}

private extension CGRect {
    var hasFiniteComponents: Bool {
        origin.x.isFinite && origin.y.isFinite && width.isFinite && height.isFinite
    }
}

private final class StreamOutput: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    let writer: HEVCWriter?
    let onFirstFrame: (@Sendable (UInt64) -> Void)?
    let onFrame: CaptureService.FrameHandler?
    let onIdle: CaptureService.IdleFrameHandler?
    let onUnexpectedStop: (@Sendable (Error) -> Void)?
    private(set) var droppedFrames = 0
    private(set) var firstFrameHostNanos: UInt64 = 0
    private let firstFrameLock = NSLock()

    init(writer: HEVCWriter?, onFirstFrame: (@Sendable (UInt64) -> Void)?, onFrame: CaptureService.FrameHandler?, onIdle: CaptureService.IdleFrameHandler?, onUnexpectedStop: (@Sendable (Error) -> Void)?) {
        self.writer = writer
        self.onFirstFrame = onFirstFrame
        self.onFrame = onFrame
        self.onIdle = onIdle
        self.onUnexpectedStop = onUnexpectedStop
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen else { return }
        guard sampleBuffer.isValid else { droppedFrames += 1; return }
        let status = (CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]])
            .flatMap { $0.first?[.status] as? Int }
            .flatMap(SCFrameStatus.init(rawValue:))
        switch CaptureFrameDisposition.classify(status) {
        case .reuseLastFrame:
            onIdle?(sampleBuffer.presentationTimeStamp)
            return
        case .drop:
            droppedFrames += 1
            return
        case .deliver:
            break
        }
        guard let pixelBuffer = sampleBuffer.imageBuffer else { droppedFrames += 1; return }
        let pts = sampleBuffer.presentationTimeStamp
        firstFrameLock.lock()
        if firstFrameHostNanos == 0 {
            let hostTime = CMTimeConvertScale(pts, timescale: 1_000_000_000, method: .default)
            if hostTime.isNumeric, hostTime.value > 0 {
                firstFrameHostNanos = UInt64(hostTime.value)
                onFirstFrame?(firstFrameHostNanos)
            }
        }
        firstFrameLock.unlock()
        writer?.append(sampleBuffer)
        onFrame?(pixelBuffer, pts)
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        onUnexpectedStop?(error)
    }
}
