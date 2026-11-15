import AppKit
import ATCore
import AVFoundation
import CoreImage
import Foundation
import ScreenCaptureKit

public struct RecordingConfig: @unchecked Sendable {
    public var target: CaptureTarget
    public var settings: CaptureSettings
    public var input: InputCaptureOptions
    public var proxySize = 192
    public var name: String
    public var task: String
    public var tags: [String] = []
    /// Folder the `.atrec` package is created in.
    public var destination: URL
    public var appVersion: String
    public var ignoredHotkeys: [(code: UInt16, modifiers: CGEventFlags)] = []
    /// Bitrate preference: 0 (small) … 1 (best).
    public var videoQuality: Double = 0.7

    public init(target: CaptureTarget, settings: CaptureSettings, input: InputCaptureOptions, name: String, task: String,
                destination: URL, appVersion: String) {
        self.target = target; self.settings = settings; self.input = input; self.name = name; self.task = task
        self.destination = destination; self.appVersion = appVersion
    }
}

public struct RecordingStats: Sendable, Equatable {
    public var elapsed: Double = 0
    public var frames = 0
    public var events = 0
    public var droppedFrames = 0
    public var bytes: Int64 = 0
    public var averageFPS: Double { elapsed > 0 ? Double(frames) / elapsed : 0 }
    public init() {}
}

/// Records one session into a `.atrec` package: HEVC video, lossless training proxy, frame and
/// input timelines on the shared host clock. Everything is written incrementally.
public final class RecordingSession: @unchecked Sendable {
    public let config: RecordingConfig
    public let live = LiveInputState()
    public let package: URL
    public var onError: ((Error) -> Void)?

    private let capturer = ScreenCapturer()
    private let tap = InputTap()
    private let resizer: FrameResizer
    private var writer: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var proxy: ProxyWriter?
    private var framesHandle: FileHandle?
    private var eventsHandle: FileHandle?
    private var frameBuffer = Data(), eventBuffer = Data()
    private let lock = NSLock()
    private var startHost: UInt64 = 0
    private var firstFrameHost: UInt64?
    private var lastFrameHost: UInt64 = 0
    private var _stats = RecordingStats()
    private var thumbnail: CGImage?
    private var stopped = false
    private var meta: RecordingMeta

    public var stats: RecordingStats {
        lock.lock(); defer { lock.unlock() }
        var s = _stats
        s.elapsed = Double(HostClock.now() - startHost) / 1e9
        return s
    }

    public init(config: RecordingConfig) throws {
        self.config = config
        guard let r = FrameResizer() else { throw NSError(domain: "AgentTrainer", code: 1, userInfo: [NSLocalizedDescriptionKey: "Metal is unavailable."]) }
        resizer = r
        let safe = config.name.replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: ".")
        var url = config.destination.appendingPathComponent(safe).appendingPathExtension(RecordingPackage.fileExtension)
        var n = 2
        while FileManager.default.fileExists(atPath: url.path) {
            url = config.destination.appendingPathComponent("\(safe) \(n)").appendingPathExtension(RecordingPackage.fileExtension); n += 1
        }
        package = url
        meta = RecordingMeta(name: config.name, appVersion: config.appVersion, target: config.target.info,
                             captureWidth: config.settings.width, captureHeight: config.settings.height,
                             sourceRect: CodableRect(config.target.frame), captureFPS: config.settings.fps,
                             showsCursor: config.settings.showsCursor, input: config.input, task: config.task, tags: config.tags)
    }

    public func start(excluding own: [SCRunningApplication]) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: package, withIntermediateDirectories: true)
        try RecordingIO.writeMeta(meta, to: package)
        var ann = RecordingAnnotations()
        if !config.task.trimmingCharacters(in: .whitespaces).isEmpty {
            ann.segments = [TextSegment(start: 0, end: 1e9, text: config.task, source: .task)]
        }
        try RecordingIO.writeAnnotations(ann, to: package)
        for f in [RecordingPackage.frames, RecordingPackage.events] { fm.createFile(atPath: package.appendingPathComponent(f).path, contents: nil) }
        framesHandle = try FileHandle(forWritingTo: package.appendingPathComponent(RecordingPackage.frames))
        eventsHandle = try FileHandle(forWritingTo: package.appendingPathComponent(RecordingPackage.events))
        proxy = try ProxyWriter(url: package.appendingPathComponent(RecordingPackage.proxy(size: config.proxySize)),
                                width: config.proxySize, height: config.proxySize)
        try setUpVideo()
        startHost = HostClock.now()

        tap.options = config.input
        tap.ignoredHotkeys = config.ignoredHotkeys
        if config.input.onlyWhenTargetFocused, !config.target.isDisplay { tap.focusProcessID = config.target.processID }
        tap.onEvent = { [weak self] e in self?.record(e) }
        try tap.start()

        capturer.onFrame = { [weak self] f in self?.handle(f) }
        capturer.onStop = { [weak self] e in if let e { self?.onError?(e) } }
        do {
            try await capturer.start(config.target, settings: config.settings, excluding: own)
        } catch {
            tap.stop()
            throw error
        }
    }

    private func setUpVideo() throws {
        let w = try AVAssetWriter(outputURL: package.appendingPathComponent(RecordingPackage.video), fileType: .mov)
        let s = config.settings
        let bitsPerPixel = 0.05 + 0.25 * config.videoQuality
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: s.width, AVVideoHeightKey: s.height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: Int(Double(s.width * s.height * s.fps) * bitsPerPixel),
                AVVideoExpectedSourceFrameRateKey: s.fps,
                AVVideoMaxKeyFrameIntervalKey: s.fps * 2,
                AVVideoAllowFrameReorderingKey: false,
            ],
        ])
        input.expectsMediaDataInRealTime = true
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: nil)
        guard w.canAdd(input) else { throw NSError(domain: "AgentTrainer", code: 2, userInfo: [NSLocalizedDescriptionKey: "Can't encode video at \(s.width)×\(s.height)."]) }
        w.add(input)
        guard w.startWriting() else { throw w.error ?? NSError(domain: "AgentTrainer", code: 3) }
        writer = w; videoInput = input; self.adaptor = adaptor
    }

    private func handle(_ f: CapturedFrame) {
        lock.lock()
        if stopped { lock.unlock(); return }
        if firstFrameHost == nil {
            firstFrameHost = f.hostNs
            writer?.startSession(atSourceTime: .zero)
        }
        let first = firstFrameHost!
        guard f.hostNs > lastFrameHost || _stats.frames == 0 else { lock.unlock(); return }
        lastFrameHost = f.hostNs
        let index = UInt32(_stats.frames)
        lock.unlock()

        let pts = CMTime(value: CMTimeValue(f.hostNs - first), timescale: 1_000_000_000)
        var dropped = false
        if let input = videoInput, input.isReadyForMoreMediaData {
            if !(adaptor?.append(f.pixelBuffer, withPresentationTime: pts) ?? false) { dropped = true }
        } else { dropped = true }
        resizer.resize(f.pixelBuffer, size: config.proxySize) { proxy?.append($0) }
        if thumbnail == nil, f.hostNs - first > 500_000_000 { thumbnail = Self.makeImage(f.pixelBuffer) }

        let rec = FrameRecord(index: index, flags: 1, timeNs: Int64(f.hostNs) - Int64(startHost), hostNs: f.hostNs)
        lock.lock()
        frameBuffer.append(BinaryRecords.encode(rec))
        _stats.frames += 1
        if dropped { _stats.droppedFrames += 1 }
        _stats.bytes = Int64(_stats.frames) * Int64(config.settings.width * config.settings.height) / 12
        if frameBuffer.count > 64 * FrameRecord.size { framesHandle?.write(frameBuffer); frameBuffer.removeAll(keepingCapacity: true) }
        lock.unlock()
    }

    private func record(_ e: InputEventRecord) {
        live.ingest(e)
        var rel = e
        rel.timeNs = e.timeNs - Int64(startHost)
        lock.lock()
        if !stopped {
            eventBuffer.append(BinaryRecords.encode(rel))
            _stats.events += 1
            if eventBuffer.count > 256 * InputEventRecord.size { eventsHandle?.write(eventBuffer); eventBuffer.removeAll(keepingCapacity: true) }
        }
        lock.unlock()
    }

    static func makeImage(_ pb: CVPixelBuffer) -> CGImage? {
        let ci = CIImage(cvPixelBuffer: pb)
        let scale = 480 / max(ci.extent.width, 1)
        return CIContext().createCGImage(ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale)), from: ci.extent.applying(CGAffineTransform(scaleX: scale, y: scale)))
    }

    /// Stops capture, finalizes every file and returns the package URL.
    @discardableResult
    public func stop() async throws -> URL {
        tap.stop()
        await capturer.stop()
        lock.lock()
        stopped = true
        framesHandle?.write(frameBuffer); eventsHandle?.write(eventBuffer)
        frameBuffer = Data(); eventBuffer = Data()
        let s = _stats
        lock.unlock()
        try framesHandle?.close(); try eventsHandle?.close()
        try proxy?.finish()
        videoInput?.markAsFinished()
        await writer?.finishWriting()

        meta.duration = Double(lastFrameHost - (firstFrameHost ?? lastFrameHost)) / 1e9
        meta.frameCount = s.frames
        meta.eventCount = s.events
        meta.droppedFrames = s.droppedFrames
        meta.proxySizes = [config.proxySize]
        meta.eventClock = tap.clockDomain
        if let frames = try? RecordingIO.readFrames(package), let events = try? RecordingIO.readEvents(package) {
            let steps = TimelineBuilder.build(frames: frames, events: events, meta: meta, annotations: RecordingAnnotations(), rateHz: 20)
            let std = TimelineBuilder.pointerDeltaStd(steps)
            if std.x > 0 { meta.mouseDeltaStd = [std.x, std.y] }
        }
        var ann = RecordingIO.readAnnotations(package)
        for i in ann.segments.indices where ann.segments[i].source == .task { ann.segments[i].end = meta.duration + 1 }
        try RecordingIO.writeAnnotations(ann, to: package)
        try RecordingIO.writeMeta(meta, to: package)
        if let img = thumbnail, let dest = CGImageDestinationCreateWithURL(package.appendingPathComponent(RecordingPackage.thumbnail) as CFURL, "public.jpeg" as CFString, 1, nil) {
            CGImageDestinationAddImage(dest, img, [kCGImageDestinationLossyCompressionQuality: 0.8] as CFDictionary)
            CGImageDestinationFinalize(dest)
        }
        return package
    }
}
