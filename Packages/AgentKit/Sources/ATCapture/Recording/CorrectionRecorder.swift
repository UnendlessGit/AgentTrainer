import ATCore
import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import ScreenCaptureKit

/// Keeps the last seconds of a run ready while an AI plays, so a correction is saved together with
/// the moments that led up to it.
///
/// While it runs it captures the controlled window into short rolling segments (HEVC video plus the
/// lossless training proxy) and keeps the frame and input timelines in memory. `begin()` marks the
/// moment the person takes over and `end()` the moment they hand back. The correction is then cut
/// out of the segments as an ordinary `.atrec`:
///
/// - the `historySeconds` before `begin()` are marked model-controlled: the AI's own actions, kept as
///   context (frames and action history) but not trained on;
/// - from `begin()` to `end()` is the person's correction, which is what the model learns from.
///
/// The AI's input is recorded as synthetic events (`InputEventRecord.isSynthetic`), so the Library
/// can tell the two apart.
public final class CorrectionRecorder: @unchecked Sendable {
    public struct Config: @unchecked Sendable {
        /// The window to capture; nil when frames are fed in with `process` (tests).
        public var target: CaptureTarget?
        public var targetInfo: CaptureTargetInfo
        /// The window's frame in global points.
        public var targetFrame: CGRect
        public var settings: CaptureSettings
        public var input = InputCaptureOptions()
        public var proxySize = 192
        public var videoQuality = 0.7
        /// How much of what led up to a correction is kept with it.
        public var historySeconds = 10.0
        /// Segments are closed and a new one started this often while nobody is correcting.
        public var segmentSeconds = 15.0
        /// Where finished corrections are saved.
        public var destination: URL
        /// Scratch folder for the rolling segments (removed when the recorder stops).
        public var scratch: URL
        public var appVersion: String
        public var modelName: String
        public var task = ""
        /// AgentTrainer's own shortcuts: never recorded.
        public var ignoredHotkeys: [(code: UInt16, modifiers: CGEventFlags)] = []

        public init(target: CaptureTarget, settings: CaptureSettings, destination: URL, scratch: URL, appVersion: String, modelName: String) {
            self.target = target; targetInfo = target.info; targetFrame = target.frame
            self.settings = settings; self.destination = destination; self.scratch = scratch
            self.appVersion = appVersion; self.modelName = modelName
        }
        public init(targetInfo: CaptureTargetInfo, targetFrame: CGRect, settings: CaptureSettings, destination: URL, scratch: URL, appVersion: String, modelName: String) {
            self.target = nil; self.targetInfo = targetInfo; self.targetFrame = targetFrame
            self.settings = settings; self.destination = destination; self.scratch = scratch
            self.appVersion = appVersion; self.modelName = modelName
        }
    }

    public let config: Config
    /// A correction was saved (called off the main thread).
    public var onSaved: ((URL) -> Void)?
    public var onError: ((String) -> Void)?

    private let capturer = ScreenCapturer()
    private let tap = InputTap()
    private let resizer: FrameResizer
    private let lock = NSLock()
    private let builds = DispatchGroup()

    private final class Segment {
        let number: Int
        let folder: URL
        let writer: AVAssetWriter
        let input: AVAssetWriterInput
        let adaptor: AVAssetWriterInputPixelBufferAdaptor
        let proxy: ProxyWriter
        var firstHost: UInt64?
        var lastHost: UInt64 = 0
        var frames = 0
        var closed = false
        /// Corrections still being cut out of this segment.
        var pins = 0
        /// Left once the segment's files are complete.
        let finished = DispatchGroup()

        init(number: Int, folder: URL, writer: AVAssetWriter, input: AVAssetWriterInput,
             adaptor: AVAssetWriterInputPixelBufferAdaptor, proxy: ProxyWriter) {
            self.number = number; self.folder = folder; self.writer = writer; self.input = input
            self.adaptor = adaptor; self.proxy = proxy
            finished.enter()
        }
    }

    private struct FrameRef { var host: UInt64; var segment: Segment; var index: Int }

    private var running = false
    private var segments: [Segment] = []
    private var current: Segment?
    private var segmentCount = 0
    private var frames: [FrameRef] = []
    private var events: [InputEventRecord] = []
    private var lastFrameHost: UInt64 = 0
    private var correctionStart: UInt64?
    private var thumbnail: CGImage?

    public init(config: Config) throws {
        self.config = config
        guard let r = FrameResizer() else { throw NSError(domain: "AgentTrainer", code: 1, userInfo: [NSLocalizedDescriptionKey: "Metal is unavailable."]) }
        resizer = r
    }

    public var isCorrecting: Bool { lock.withLock { correctionStart != nil } }

    /// Seconds of history currently available (up to `historySeconds`).
    public var availableHistory: Double {
        lock.withLock {
            guard let first = frames.first?.host else { return 0 }
            return min(config.historySeconds, Double(HostClock.now() &- first) / 1e9)
        }
    }

    public func start(excluding own: [SCRunningApplication]) async throws {
        guard let target = config.target else { throw NSError(domain: "AgentTrainer", code: 4, userInfo: [NSLocalizedDescriptionKey: "No window to capture."]) }
        try FileManager.default.createDirectory(at: config.scratch, withIntermediateDirectories: true)
        lock.withLock { running = true }
        tap.options = config.input
        tap.ignoredHotkeys = config.ignoredHotkeys
        tap.onEvent = { [weak self] e in self?.record(e) }
        try tap.start()
        capturer.onFrame = { [weak self] f in self?.handle(f) }
        capturer.onStop = { [weak self] e in if let e { self?.onError?(e.localizedDescription) } }
        do {
            try await capturer.start(target, settings: config.settings, excluding: own)
        } catch {
            tap.stop()
            lock.withLock { running = false }
            throw error
        }
    }

    /// Stops recording. A correction in progress is saved first; the rolling segments are deleted.
    public func stop() async {
        if isCorrecting { end() }
        tap.stop()
        await capturer.stop()
        let open: [Segment] = lock.withLock {
            running = false
            let s = segments
            current = nil
            return s
        }
        for s in open where !s.closed { close(s) }
        // let corrections being saved finish before the scratch folder goes
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            builds.notify(queue: .global()) { c.resume() }
        }
        for s in open { await Self.finished(s) }
        try? FileManager.default.removeItem(at: config.scratch)
        lock.withLock { segments = []; frames = []; events = [] }
    }

    // MARK: Corrections

    /// The person takes over now. Returns false when a correction is already running.
    @discardableResult
    public func begin() -> Bool {
        lock.withLock {
            guard running, correctionStart == nil else { return false }
            correctionStart = HostClock.now()
            return true
        }
    }

    /// The person hands back control: the correction (with its history) is saved in the background.
    public func end() { end(at: HostClock.now()) }

    func end(at now: UInt64) {
        let job: Job? = lock.withLock {
            guard let begin = correctionStart else { return nil }
            correctionStart = nil
            let from = begin &- UInt64(config.historySeconds * 1e9)
            let picked = frames.filter { $0.host >= from && $0.host <= now }
            guard let first = picked.first, picked.count >= 2 else { return nil }
            // close the segment being written so every frame of the correction is on disk
            if let c = current { current = nil; close(c) }
            var used: [Segment] = []
            for f in picked where !used.contains(where: { $0 === f.segment }) { used.append(f.segment) }
            for s in used { s.pins += 1 }
            // input from a little before the first frame, so keys already held are known
            let evs = events.filter { $0.timeNs >= Int64(first.host) - 5_000_000_000 && $0.timeNs <= Int64(now) }
            return Job(frames: picked, events: evs, segments: used, begin: begin, end: now)
        }
        guard let job else {
            onError?("The correction was too short to save.")
            return
        }
        builds.enter()
        Task.detached(priority: .utility) { [self] in
            defer { builds.leave() }
            for s in job.segments { await Self.finished(s) }
            do {
                let url = try await build(job)
                onSaved?(url)
            } catch {
                onError?("Saving the correction failed: \(error.localizedDescription)")
            }
            lock.withLock { for s in job.segments { s.pins -= 1 } }
        }
    }

    // MARK: Offline (tests)

    /// Prepares the recorder for `process` / `ingest` without a live capture (tests).
    public func startOffline() throws {
        try FileManager.default.createDirectory(at: config.scratch, withIntermediateDirectories: true)
        lock.withLock { running = true }
    }

    /// Handles one captured frame. Public so the recorder can be exercised without a live capture.
    public func process(_ pixelBuffer: CVPixelBuffer, hostNs: UInt64) { handle(pixelBuffer, hostNs: hostNs) }

    /// Records one input event (host-clock time). Public for tests.
    public func ingest(_ e: InputEventRecord) { record(e) }

    /// `begin` / `end` at given host times (tests).
    func begin(at host: UInt64) { lock.withLock { if running, correctionStart == nil { correctionStart = host } } }

    // MARK: Capture

    private func handle(_ f: CapturedFrame) { handle(f.pixelBuffer, hostNs: f.hostNs) }

    private func handle(_ pixelBuffer: CVPixelBuffer, hostNs: UInt64) {
        let f = (pixelBuffer: pixelBuffer, hostNs: hostNs)
        lock.lock(); defer { lock.unlock() }
        guard running else { return }
        let now = f.hostNs
        if let c = current, correctionStart == nil, let first = c.firstHost, Double(now &- first) / 1e9 >= config.segmentSeconds {
            current = nil
            close(c)
        }
        if current == nil {
            do { current = try makeSegment() } catch { return }
        }
        guard let seg = current else { return }
        guard f.hostNs > lastFrameHost else { return }
        lastFrameHost = f.hostNs
        if seg.firstHost == nil {
            seg.firstHost = f.hostNs
            seg.writer.startSession(atSourceTime: .zero)
        }
        let pts = CMTime(value: CMTimeValue(f.hostNs - seg.firstHost!), timescale: 1_000_000_000)
        if seg.input.isReadyForMoreMediaData { _ = seg.adaptor.append(f.pixelBuffer, withPresentationTime: pts) }
        resizer.resize(f.pixelBuffer, size: config.proxySize) { seg.proxy.append($0) }
        frames.append(FrameRef(host: f.hostNs, segment: seg, index: seg.frames))
        seg.frames += 1
        seg.lastHost = f.hostNs
        if correctionStart != nil, thumbnail == nil { thumbnail = RecordingSession.makeImage(f.pixelBuffer) }
        prune(now: now)
    }

    private func record(_ e: InputEventRecord) {
        lock.withLock { if running { events.append(e) } }
    }

    /// Lock held. Forgets frames, input and finished segments too old to be anyone's history.
    private func prune(now: UInt64) {
        guard correctionStart == nil else { return }
        let keep = UInt64((config.historySeconds + config.segmentSeconds + 2) * 1e9)
        guard now > keep else { return }
        let cutoff = now - keep
        if let i = frames.firstIndex(where: { $0.host >= cutoff }), i > 0 { frames.removeFirst(i) }
        if let i = events.firstIndex(where: { $0.timeNs >= Int64(cutoff) }), i > 0 { events.removeFirst(i) }
        segments.removeAll { s in
            guard s.closed, s.pins == 0, s.lastHost < cutoff else { return false }
            let folder = s.folder
            s.finished.notify(queue: .global()) { try? FileManager.default.removeItem(at: folder) }
            return true
        }
    }

    /// Lock held.
    private func makeSegment() throws -> Segment {
        segmentCount += 1
        let folder = config.scratch.appendingPathComponent("segment-\(segmentCount)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let s = config.settings
        let w = try AVAssetWriter(outputURL: folder.appendingPathComponent(RecordingPackage.video), fileType: .mov)
        let bitsPerPixel = 0.05 + 0.25 * config.videoQuality
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: s.width, AVVideoHeightKey: s.height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: Int(Double(s.width * s.height * s.fps) * bitsPerPixel),
                AVVideoExpectedSourceFrameRateKey: s.fps,
                // short GOPs: a correction can start anywhere inside a segment
                AVVideoMaxKeyFrameIntervalKey: max(1, s.fps / 2),
                AVVideoAllowFrameReorderingKey: false,
            ],
        ])
        input.expectsMediaDataInRealTime = true
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: nil)
        guard w.canAdd(input) else { throw NSError(domain: "AgentTrainer", code: 2, userInfo: [NSLocalizedDescriptionKey: "Can't encode video at \(s.width)×\(s.height)."]) }
        w.add(input)
        guard w.startWriting() else { throw w.error ?? NSError(domain: "AgentTrainer", code: 3) }
        let proxy = try ProxyWriter(url: folder.appendingPathComponent(RecordingPackage.proxy(size: config.proxySize)), width: config.proxySize, height: config.proxySize)
        let seg = Segment(number: segmentCount, folder: folder, writer: w, input: input, adaptor: adaptor, proxy: proxy)
        segments.append(seg)
        return seg
    }

    /// Finishes a segment's files in the background.
    private func close(_ s: Segment) {
        guard !s.closed else { return }
        s.closed = true
        let hadFrames = s.firstHost != nil
        DispatchQueue.global(qos: .utility).async {
            try? s.proxy.finish()
            if hadFrames {
                s.input.markAsFinished()
                s.writer.finishWriting { s.finished.leave() }
            } else {
                s.writer.cancelWriting()
                s.finished.leave()
            }
        }
    }

    // MARK: Saving

    private static func finished(_ s: Segment) async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in s.finished.notify(queue: .global()) { c.resume() } }
    }

    private struct Job: @unchecked Sendable {
        var frames: [FrameRef]
        var events: [InputEventRecord]
        var segments: [Segment]
        var begin: UInt64
        var end: UInt64
    }

    private func build(_ job: Job) async throws -> URL {
        let fm = FileManager.default
        try fm.createDirectory(at: config.destination, withIntermediateDirectories: true)
        let stamp = Date().formatted(.dateTime.year().month(.twoDigits).day(.twoDigits).hour().minute().second())
        let base = "\(config.modelName) correction \(stamp)".replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: ".")
        var url = config.destination.appendingPathComponent(base).appendingPathExtension(RecordingPackage.fileExtension)
        var n = 2
        while fm.fileExists(atPath: url.path) {
            url = config.destination.appendingPathComponent("\(base) \(n)").appendingPathExtension(RecordingPackage.fileExtension); n += 1
        }
        try fm.createDirectory(at: url, withIntermediateDirectories: true)
        do {
            try await write(job, to: url)
            return url
        } catch {
            try? fm.removeItem(at: url)
            throw error
        }
    }

    private func write(_ job: Job, to url: URL) async throws {
        let first = job.frames.first!.host, last = job.frames.last!.host

        // training proxy: the frames, in order, from each segment's proxy
        let proxyOut = try ProxyWriter(url: url.appendingPathComponent(RecordingPackage.proxy(size: config.proxySize)), width: config.proxySize, height: config.proxySize)
        var readers: [Int: ProxyReader] = [:]
        for s in job.segments { readers[s.number] = try ProxyReader(url: s.folder.appendingPathComponent(RecordingPackage.proxy(size: config.proxySize))) }
        var kept: [FrameRef] = []
        for f in job.frames {
            guard let r = readers[f.segment.number], f.index < r.frameCount else { continue }
            let data = try r.frame(f.index)
            data.withUnsafeBytes { proxyOut.append($0) }
            kept.append(f)
        }
        try proxyOut.finish()
        guard kept.count >= 2 else { throw NSError(domain: "AgentTrainer", code: 5, userInfo: [NSLocalizedDescriptionKey: "No frames were captured."]) }

        // timelines, with time zero at the first frame
        var frameData = Data()
        for (i, f) in kept.enumerated() {
            frameData.append(BinaryRecords.encode(FrameRecord(index: UInt32(i), flags: 1, timeNs: Int64(f.host) - Int64(first), hostNs: f.host)))
        }
        try frameData.write(to: url.appendingPathComponent(RecordingPackage.frames))
        var eventData = Data()
        var rebased: [InputEventRecord] = []
        for e in job.events {
            var r = e
            r.timeNs = e.timeNs - Int64(first)
            rebased.append(r)
            eventData.append(BinaryRecords.encode(r))
        }
        try eventData.write(to: url.appendingPathComponent(RecordingPackage.events))

        // video: the segments' pieces, put together without re-encoding
        try await writeVideo(job, first: first, last: last, to: url.appendingPathComponent(RecordingPackage.video))

        // meta and annotations: the history is the model's, the rest is the correction
        let s = config.settings
        var meta = RecordingMeta(name: url.deletingPathExtension().lastPathComponent, appVersion: config.appVersion, target: config.targetInfo,
                                 captureWidth: s.width, captureHeight: s.height, sourceRect: CodableRect(config.targetFrame),
                                 captureFPS: s.fps, showsCursor: s.showsCursor, input: config.input, task: config.task, tags: ["correction"])
        meta.duration = Double(last - first) / 1e9
        meta.frameCount = kept.count
        meta.eventCount = rebased.count
        meta.proxySizes = [config.proxySize]
        meta.eventClock = tap.clockDomain
        meta.isCorrectionRun = true
        meta.modelName = config.modelName
        let historyEnd = max(0, Double(Int64(job.begin) - Int64(first)) / 1e9)
        var ann = RecordingAnnotations()
        ann.modelControlled = historyEnd > 0 ? [TimeRange(start: 0, end: historyEnd)] : []
        if !config.task.trimmingCharacters(in: .whitespaces).isEmpty {
            ann.segments = [TextSegment(start: 0, end: meta.duration + 1, text: config.task, source: .task)]
        }
        let frameRecords = BinaryRecords.decodeFrames(frameData)
        let steps = TimelineBuilder.build(frames: frameRecords, events: rebased, meta: meta, annotations: ann, rateHz: 20).filter { !$0.modelControlled }
        let std = TimelineBuilder.pointerDeltaStd(steps)
        if std.x > 0 { meta.mouseDeltaStd = [std.x, std.y] }
        try RecordingIO.writeAnnotations(ann, to: url)
        try RecordingIO.writeMeta(meta, to: url)

        // thumbnail: the moment the person took over
        let thumb = lock.withLock { () -> CGImage? in let t = thumbnail; thumbnail = nil; return t }
        if let img = thumb, let dest = CGImageDestinationCreateWithURL(url.appendingPathComponent(RecordingPackage.thumbnail) as CFURL, "public.jpeg" as CFString, 1, nil) {
            CGImageDestinationAddImage(dest, img, [kCGImageDestinationLossyCompressionQuality: 0.8] as CFDictionary)
            CGImageDestinationFinalize(dest)
        }
    }

    private func writeVideo(_ job: Job, first: UInt64, last: UInt64, to out: URL) async throws {
        let composition = AVMutableComposition()
        guard let track = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw NSError(domain: "AgentTrainer", code: 6, userInfo: [NSLocalizedDescriptionKey: "Couldn't assemble the video."])
        }
        let frameNs = UInt64(1_000_000_000 / max(1, config.settings.fps))
        let end = last + frameNs
        for s in job.segments {
            guard let segFirst = s.firstHost else { continue }
            let asset = AVURLAsset(url: s.folder.appendingPathComponent(RecordingPackage.video))
            guard let src = try await asset.loadTracks(withMediaType: .video).first else { continue }
            let available = try await src.load(.timeRange)
            let from = max(first, segFirst), to = min(end, s.lastHost + frameNs)
            guard to > from else { continue }
            var range = CMTimeRange(start: CMTime(value: CMTimeValue(from - segFirst), timescale: 1_000_000_000),
                                    duration: CMTime(value: CMTimeValue(to - from), timescale: 1_000_000_000))
            range = range.intersection(available)
            guard range.duration > .zero else { continue }
            try track.insertTimeRange(range, of: src, at: CMTime(value: CMTimeValue(from - first), timescale: 1_000_000_000))
            track.preferredTransform = try await src.load(.preferredTransform)
        }
        guard let export = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetPassthrough) else {
            throw NSError(domain: "AgentTrainer", code: 7, userInfo: [NSLocalizedDescriptionKey: "Couldn't export the video."])
        }
        try await export.export(to: out, as: .mov)
    }
}
