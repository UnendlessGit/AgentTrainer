import ATCore
import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import ScreenCaptureKit

/// Turns Elkborn Training Data replays into ordinary `.atrec` recordings.
///
/// While listening it captures the Roblox window and follows the game's screen protocol
/// (`ElkbornProtocol`): it answers the ready page with F8, reads the timeline pages, answers with
/// F8 again, then records the playback. Each recording is on replay time (frame times come from
/// the timecode strip, not the capture clock), its inputs are the run's own inputs from the
/// timeline (keys converted to the viewer's binds by the game, camera turns converted to mouse
/// movement), and the strip corner is blacked out. A playlist simply produces one recording per
/// replay: after the end page it waits for the next ready page.
///
/// `calibrate()` instead measures how far the camera turns per injected mouse unit, which makes
/// the synthesized mouse movement match what AgentTrainer will inject when it plays.
public final class ElkbornReplayRecorder: @unchecked Sendable {
    public enum Phase: Equatable, Sendable {
        case idle
        case waiting
        case receiving(pages: Int, of: Int)
        case recording(seconds: Double, of: Double)
        case saving
        case calibrating
    }

    public struct Config: @unchecked Sendable {
        /// The window to capture; nil when frames are fed in with `process` (tests).
        public var target: CaptureTarget?
        public var targetInfo: CaptureTargetInfo
        /// The window's frame in global points.
        public var targetFrame: CGRect
        public var settings: CaptureSettings
        public var destination: URL
        public var appVersion: String
        public var proxySize = 192
        public var videoQuality = 0.7
        /// Injected mouse units per Roblox mouse pixel (from calibration).
        public var unitsPerPixel = 1.0
        public init(target: CaptureTarget, settings: CaptureSettings, destination: URL, appVersion: String) {
            self.target = target; targetInfo = target.info; targetFrame = target.frame
            self.settings = settings; self.destination = destination; self.appVersion = appVersion
        }
        public init(targetInfo: CaptureTargetInfo, targetFrame: CGRect, settings: CaptureSettings, destination: URL, appVersion: String) {
            self.target = nil; self.targetInfo = targetInfo; self.targetFrame = targetFrame
            self.settings = settings; self.destination = destination; self.appVersion = appVersion
        }
    }

    public var onPhase: ((Phase, String) -> Void)?
    public var onSaved: ((URL) -> Void)?
    public var onError: ((String) -> Void)?
    /// (units per pixel, the yaw per pixel it was measured at)
    public var onCalibrated: ((Double, Double) -> Void)?
    /// Presses a key in the game (the handshake's F8). Replaceable for tests.
    public var pressKey: (UInt16) -> Void = { code in
        let source = CGEventSource(stateID: .hidSystemState)
        for down in [true, false] {
            let e = CGEvent(keyboardEventSource: source, virtualKey: code, keyDown: down)
            e?.setIntegerValueField(.eventSourceUserData, value: SyntheticInput.userData)
            e?.post(tap: .cghidEventTap)
        }
    }

    /// Moves the mouse by `dx` units, as relative movement at the pointer's current place (the
    /// same events the runtime's "Camera (locked cursor)" output sends). Replaceable for tests.
    public var moveMouse: (Int) -> Void = { dx in
        let source = CGEventSource(stateID: .hidSystemState)
        let at = CGEvent(source: nil)?.location ?? .zero
        let e = CGEvent(mouseEventSource: source, mouseType: .mouseMoved, mouseCursorPosition: at, mouseButton: .left)
        e?.setIntegerValueField(.mouseEventDeltaX, value: Int64(dx))
        e?.setDoubleValueField(.mouseEventDeltaX, value: Double(dx))
        e?.setIntegerValueField(.eventSourceUserData, value: SyntheticInput.userData)
        e?.post(tap: .cghidEventTap)
    }
    /// Puts the pointer at a point in global coordinates. Replaceable for tests.
    public var placePointer: (CGPoint) -> Void = { CGWarpMouseCursorPosition($0) }

    private let config: Config
    private let capturer = ScreenCapturer()
    private let resizer: FrameResizer
    private let lock = NSLock()
    private var calibrationMode = false

    // protocol state (capture queue only)
    private enum Step { case waiting, receiving(session: UInt8), playing(session: UInt8), calibrationReady, calibrating(session: UInt8) }
    private var step = Step.waiting
    private var scanner = ElkbornScanner()
    private var assembler = ElkbornPageAssembler()
    private var timeline: ElkbornTimeline?
    private var lastKey: UInt64 = 0
    private var pageWaitStarted: UInt64 = 0
    private var lastStripHost: UInt64 = 0

    // the recording being written
    private var writer: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var proxy: ProxyWriter?
    private var package: URL?
    private var frames = Data()
    private var frameCount = 0
    private var dropped = 0
    private var lastFrameCounter: UInt16?
    private var lastTimeMs: Int64 = -1
    private var thumbnail: CGImage?
    private var maskRect: CGRect?

    // calibration (the latest camera yaw the game shows, and how many readouts have arrived)
    private var calibration: ElkbornCalibrationInfo?
    private var calibrationYaw: UInt16?
    private var calibrationSamples = 0

    public init(config: Config) throws {
        self.config = config
        guard let r = FrameResizer() else { throw NSError(domain: "AgentTrainer", code: 1, userInfo: [NSLocalizedDescriptionKey: "Metal is unavailable."]) }
        resizer = r
    }

    public func start(excluding own: [SCRunningApplication], calibrate: Bool = false) async throws {
        calibrationMode = calibrate
        step = calibrate ? .calibrationReady : .waiting
        // Frames are handled on the capture queue as they arrive: when one takes long, the stream
        // drops the next instead of building a backlog.
        capturer.onFrame = { [weak self] f in self?.process(f.pixelBuffer, hostNs: f.hostNs) }
        capturer.onStop = { [weak self] e in if let e { self?.onError?(e.localizedDescription) } }
        guard let target = config.target else { throw NSError(domain: "AgentTrainer", code: 4, userInfo: [NSLocalizedDescriptionKey: "No window to capture."]) }
        try await capturer.start(target, settings: config.settings, excluding: own)
        report(calibrate ? .calibrating : .waiting, calibrate ? "Open Training Data in Elkborn and press CALIBRATE MOUSE." : "Waiting for a replay. Press RECORD or PLAYLIST in Elkborn's Training Data.")
    }

    public func stop() async {
        await capturer.stop()
        lock.withLock { discard() }
        report(.idle, "")
    }

    private func report(_ phase: Phase, _ text: String) { onPhase?(phase, text) }

    // MARK: Frames

    /// Handles one captured frame. Public so the protocol can be exercised without a live capture.
    public func process(_ pixelBuffer: CVPixelBuffer, hostNs: UInt64) {
        lock.lock(); defer { lock.unlock() }
        handle(Frame(pixelBuffer: pixelBuffer, hostNs: hostNs))
    }

    /// Prepares a recorder for `process` without starting a capture (tests).
    public func startOffline(calibrate: Bool = false) {
        lock.withLock { calibrationMode = calibrate; step = calibrate ? .calibrationReady : .waiting }
    }

    private struct Frame { let pixelBuffer: CVPixelBuffer; let hostNs: UInt64 }

    /// Called with `lock` held.
    private func handle(_ f: Frame) {
        let pb = f.pixelBuffer
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return }
        let luma = LumaFrame(base: base, width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb), bytesPerRow: CVPixelBufferGetBytesPerRow(pb))

        switch step {
        case .waiting:
            guard let page = scanner.page(luma), page.kind == .beacon else { return }
            assembler = ElkbornPageAssembler()
            step = .receiving(session: page.session)
            pageWaitStarted = f.hostNs
            pressGo(at: f.hostNs)
            report(.receiving(pages: 0, of: 0), "Reading the run's timeline...")

        case .receiving(let session):
            guard let page = scanner.page(luma) else { return }
            if page.kind == .beacon {
                // the go key did not arrive (or a new replay started): answer again
                if page.session != session { step = .receiving(session: page.session); assembler = ElkbornPageAssembler() }
                if f.hostNs - lastKey > 1_000_000_000 { pressGo(at: f.hostNs) }
                return
            }
            guard page.kind == .data, page.session == session else { return }
            let payload = assembler.add(page)
            report(.receiving(pages: assembler.received, of: page.count), "Reading the run's timeline: \(assembler.received) of \(page.count) pages")
            guard let payload else { return }
            do {
                let t = try ElkbornTimeline(payload: payload)
                guard t.session == session else { return }
                try openRecording(for: t) // clears the previous recording's state, so set the timeline after it
                timeline = t
                step = .playing(session: session)
                lastStripHost = f.hostNs
                pressGo(at: f.hostNs)
                report(.recording(seconds: 0, of: t.finalTime), "\(t.runner) on \(t.board): starting")
            } catch {
                onError?("The replay's timeline could not be read (\(error)). Waiting for the next replay.")
                discard()
                step = .waiting
            }

        case .playing(let session):
            if let strip = scanner.strip(luma), strip.session == session {
                lastStripHost = f.hostNs
                if strip.flags.contains(.playing) { record(f, strip: strip) }
                return
            }
            if let page = scanner.page(luma, search: false) {
                if page.kind == .end, page.session == session { finish() }
                else if page.kind == .data, page.session == session, f.hostNs - lastKey > 500_000_000 {
                    // still on the timeline pages: the confirmation was missed
                    pressGo(at: f.hostNs)
                }
                return
            }
            if f.hostNs - lastStripHost > 4_000_000_000 {
                onError?("The replay stopped before it finished. Nothing was saved.")
                discard()
                step = .waiting
                report(.waiting, "Waiting for a replay.")
            }

        case .calibrationReady:
            guard let page = scanner.page(luma), page.kind == .calibration,
                  let info = try? ElkbornCalibrationInfo(payload: page.payload) else { return }
            calibration = info
            calibrationYaw = nil
            calibrationSamples = 0
            step = .calibrating(session: info.session)
            pressGo(at: f.hostNs)
            report(.calibrating, "Measuring the mouse...")
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.8) { [weak self] in self?.runCalibration() }

        case .calibrating(let session):
            if let strip = scanner.strip(luma), strip.session == session, strip.flags.contains(.calibration) {
                calibrationYaw = strip.frame
                calibrationSamples += 1
            }
        }
    }

    private func pressGo(at host: UInt64) {
        lastKey = host
        pressKey(ElkbornProtocol.goKeyCode)
    }

    // MARK: Recording

    private func openRecording(for t: ElkbornTimeline) throws {
        discard()
        let fm = FileManager.default
        try fm.createDirectory(at: config.destination, withIntermediateDirectories: true)
        let stamp = Date().formatted(.dateTime.year().month(.twoDigits).day(.twoDigits).hour().minute().second())
        let base = "\(t.runner) - \(t.board) \(String(format: "%.3f", t.finalTime)) \(stamp)"
            .replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: ".")
        var url = config.destination.appendingPathComponent(base).appendingPathExtension(RecordingPackage.fileExtension)
        var n = 2
        while fm.fileExists(atPath: url.path) {
            url = config.destination.appendingPathComponent("\(base) \(n)").appendingPathExtension(RecordingPackage.fileExtension); n += 1
        }
        try fm.createDirectory(at: url, withIntermediateDirectories: true)
        package = url
        proxy = try ProxyWriter(url: url.appendingPathComponent(RecordingPackage.proxy(size: config.proxySize)), width: config.proxySize, height: config.proxySize)
        let s = config.settings
        let w = try AVAssetWriter(outputURL: url.appendingPathComponent(RecordingPackage.video), fileType: .mov)
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
        w.startSession(atSourceTime: .zero)
        writer = w; videoInput = input; self.adaptor = adaptor
        frames = Data(); frameCount = 0; dropped = 0; lastFrameCounter = nil; lastTimeMs = -1; thumbnail = nil; maskRect = nil
    }

    private func record(_ f: Frame, strip: ElkbornProtocol.Strip) {
        // a frame Roblox drew once can be captured twice; keep it once, in order
        if strip.frame == lastFrameCounter { return }
        let timeMs = Int64(strip.value)
        guard timeMs > lastTimeMs else { return }
        lastFrameCounter = strip.frame
        lastTimeMs = timeMs

        let pb = f.pixelBuffer
        // black out the timecode corner (a little larger than the strip)
        let bounds = CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
        let rect = maskRect ?? strip.rect.insetBy(dx: -strip.cell * 0.5, dy: -strip.cell * 0.5).intersection(bounds)
        maskRect = rect
        blackOut(pb, rect)

        let pts = CMTime(value: CMTimeValue(timeMs), timescale: 1000)
        // The video's timeline starts at the run's zero. A first frame that arrives a few ms
        // after it also stands in at zero, or players show the empty start as a black frame.
        if frameCount == 0, timeMs > 0, let input = videoInput, input.isReadyForMoreMediaData {
            _ = adaptor?.append(pb, withPresentationTime: .zero)
        }
        if let input = videoInput, input.isReadyForMoreMediaData {
            if !(adaptor?.append(pb, withPresentationTime: pts) ?? false) { dropped += 1 }
        } else { dropped += 1 }
        resizer.resize(pb, size: config.proxySize) { proxy?.append($0) }
        if thumbnail == nil, timeMs > 500 { thumbnail = RecordingSession.makeImage(pb) }
        let rec = FrameRecord(index: UInt32(frameCount), flags: 1, timeNs: timeMs * 1_000_000, hostNs: f.hostNs)
        frames.append(BinaryRecords.encode(rec))
        frameCount += 1
        if let t = timeline, frameCount % 15 == 0 {
            report(.recording(seconds: Double(timeMs) / 1000, of: t.finalTime), String(format: "%@ on %@: %.1f of %.1f s", t.runner, t.board, Double(timeMs) / 1000, t.finalTime))
        }
    }

    private func blackOut(_ pb: CVPixelBuffer, _ rect: CGRect) {
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb), bpr = CVPixelBufferGetBytesPerRow(pb)
        let x0 = max(0, Int(rect.minX.rounded(.down))), x1 = min(w, Int(rect.maxX.rounded(.up)))
        let y0 = max(0, Int(rect.minY.rounded(.down))), y1 = min(h, Int(rect.maxY.rounded(.up)))
        guard x1 > x0, y1 > y0 else { return }
        for y in y0 ..< y1 {
            let row = base.advanced(by: y * bpr + x0 * 4).assumingMemoryBound(to: UInt32.self)
            for x in 0 ..< (x1 - x0) { row[x] = 0xFF00_0000 } // BGRA black, opaque
        }
    }

    private func finish() {
        guard let url = package, let t = timeline, let writer else { step = .waiting; return }
        step = .waiting
        report(.saving, "Saving \(t.runner) on \(t.board)...")
        let frames = self.frames, frameCount = self.frameCount, dropped = self.dropped
        let thumbnail = self.thumbnail, mask = self.maskRect
        let s = config.settings
        let proxy = self.proxy
        let input = videoInput
        self.writer = nil; self.videoInput = nil; self.adaptor = nil; self.proxy = nil; self.package = nil; self.timeline = nil

        Task {
            do {
                try frames.write(to: url.appendingPathComponent(RecordingPackage.frames))
                let center = CGPoint(x: config.targetFrame.midX, y: config.targetFrame.midY)
                let events = ElkbornInputs.events(for: t, unitsPerPixel: config.unitsPerPixel, pointer: center)
                var eventData = Data()
                for e in events { eventData.append(BinaryRecords.encode(e)) }
                try eventData.write(to: url.appendingPathComponent(RecordingPackage.events))
                try proxy?.finish()
                input?.markAsFinished()
                await writer.finishWriting()

                var info = config.targetInfo
                info.title = "Elkborn replay"
                var meta = RecordingMeta(name: url.deletingPathExtension().lastPathComponent, appVersion: config.appVersion, target: info,
                                         captureWidth: s.width, captureHeight: s.height, sourceRect: CodableRect(config.targetFrame),
                                         captureFPS: s.fps, showsCursor: false, input: InputCaptureOptions(),
                                         task: "", tags: ["elkborn", t.board])
                let decoded = BinaryRecords.decodeFrames(frames)
                meta.duration = Double((decoded.last?.timeNs ?? 0) - (decoded.first?.timeNs ?? 0)) / 1e9
                meta.frameCount = frameCount
                meta.eventCount = events.count
                meta.droppedFrames = dropped
                meta.proxySizes = [config.proxySize]
                meta.eventClock = "elkborn-replay"
                if let mask, s.width > 0, s.height > 0 {
                    meta.masks = [CodableRect(x: mask.minX / Double(s.width), y: mask.minY / Double(s.height),
                                              width: mask.width / Double(s.width), height: mask.height / Double(s.height))]
                }
                let steps = TimelineBuilder.build(frames: decoded, events: events, meta: meta, annotations: RecordingAnnotations(), rateHz: 20)
                let std = TimelineBuilder.pointerDeltaStd(steps)
                if std.x > 0 { meta.mouseDeltaStd = [std.x, std.y] }
                try RecordingIO.writeMeta(meta, to: url)
                try RecordingIO.writeAnnotations(RecordingAnnotations(), to: url)
                if let img = thumbnail, let dest = CGImageDestinationCreateWithURL(url.appendingPathComponent(RecordingPackage.thumbnail) as CFURL, "public.jpeg" as CFString, 1, nil) {
                    CGImageDestinationAddImage(dest, img, [kCGImageDestinationLossyCompressionQuality: 0.8] as CFDictionary)
                    CGImageDestinationFinalize(dest)
                }
                onSaved?(url)
                report(.waiting, "Saved \(t.runner) on \(t.board). Waiting for the next replay.")
            } catch {
                try? FileManager.default.removeItem(at: url)
                onError?("Saving the replay failed: \(error.localizedDescription)")
                report(.waiting, "Waiting for a replay.")
            }
        }
    }

    private func discard() {
        writer?.cancelWriting()
        try? proxy?.finish()
        if let package { try? FileManager.default.removeItem(at: package) }
        writer = nil; videoInput = nil; adaptor = nil; proxy = nil; package = nil; timeline = nil
        frames = Data(); frameCount = 0
    }

    // MARK: Calibration

    /// Sends known horizontal mouse movements and reads how far the camera turned each time.
    ///
    /// Roblox only re-centres a locked pointer on the first movement after the camera locks (the
    /// snap you also get after switching apps with the pointer off-centre). Before measuring, the
    /// pointer is put in the middle of the window and nudged, so that snap happens first and is
    /// never part of a measurement. Then the camera is turned back and forth three times; the
    /// result is the median, and a calibration whose turns disagree is refused.
    private func runCalibration() {
        guard let info = calibration else { return }
        guard settledYaw(timeout: 5) != nil else {
            finishCalibration(error: "Calibration: the game's camera readout never appeared.")
            return
        }
        report(.calibrating, "Centring the pointer...")
        placePointer(CGPoint(x: config.targetFrame.midX, y: config.targetFrame.midY))
        Thread.sleep(forTimeInterval: 0.3)
        move(dx: 8, steps: 2)
        move(dx: -8, steps: 2)
        guard var before = settledYaw() else {
            finishCalibration(error: "Calibration: the camera never settled. Keep the mouse still while it measures.")
            return
        }
        let units = 400, steps = 20, rounds = 6
        var turns: [Double] = []
        for i in 0 ..< rounds {
            report(.calibrating, "Measuring the mouse (\(i + 1) of \(rounds))...")
            move(dx: i % 2 == 0 ? units : -units, steps: steps)
            guard let after = settledYaw() else {
                finishCalibration(error: "Calibration: the camera never settled. Keep the mouse still while it measures.")
                return
            }
            var d = Int(after) - Int(before)
            if d > 32768 { d -= 65536 } else if d < -32768 { d += 65536 }
            turns.append(abs(Double(d)) * 2 * .pi / 65536)
            before = after
        }
        let sorted = turns.sorted()
        let median = (sorted[rounds / 2 - 1] + sorted[rounds / 2]) / 2
        guard median > 0, info.yawPerPixel > 0 else {
            finishCalibration(error: "Calibration: the camera did not turn. Make sure Roblox is in front and the camera is in shift lock or first person.")
            return
        }
        let spread = (sorted.last! - sorted.first!) / median
        guard spread <= 0.1 else {
            finishCalibration(error: String(format: "Calibration: the six turns disagree by %.0f%%. Keep the mouse still and Roblox in front, then try again.", spread * 100))
            return
        }
        // radians per injected unit / radians per Roblox pixel = Roblox pixels per unit
        let pixelsPerUnit = median / Double(units) / info.yawPerPixel
        finishCalibration(error: nil)
        report(.calibrating, String(format: "Calibrated: %.3f mouse units per Roblox pixel (turns within %.1f%%).", 1 / pixelsPerUnit, spread * 100))
        onCalibrated?(1 / pixelsPerUnit, info.yawPerPixel)
    }

    /// Ends the game's calibration screen (F8) and gets ready for another calibration.
    private func finishCalibration(error: String?) {
        pressGo(at: 0)
        lock.withLock { step = .calibrationReady }
        if let error { onError?(error) }
    }

    /// The camera yaw once it has stopped changing (several fresh readouts in a row agree).
    private func settledYaw(timeout: TimeInterval = 3) -> UInt16? {
        let deadline = Date().addingTimeInterval(timeout)
        var value: UInt16?
        var since = Date(), samplesAtChange = lock.withLock { calibrationSamples }
        while Date() < deadline {
            let (yaw, count) = lock.withLock { (calibrationYaw, calibrationSamples) }
            if yaw != value {
                value = yaw; since = Date(); samplesAtChange = count
            } else if value != nil, Date().timeIntervalSince(since) >= 0.25, count - samplesAtChange >= 4 {
                return value
            }
            Thread.sleep(forTimeInterval: 0.02)
        }
        return nil
    }

    private func move(dx: Int, steps: Int) {
        let per = dx / steps
        for _ in 0 ..< steps {
            moveMouse(per)
            Thread.sleep(forTimeInterval: 0.016)
        }
    }
}
