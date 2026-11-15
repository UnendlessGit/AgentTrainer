import AppKit
import ATCapture
import ATLanguage
import ATCore
import ATModel
import CoreVideo
import Foundation
import MLX
import ScreenCaptureKit

public struct RuntimeOptions: Sendable {
    public var permissions = ActionPermissions()
    public var temperature: Float = 1
    public var mouseSensitivity = 1.0
    public var mouseMode = MouseOutputMode.pointer
    /// Run the model and show its actions in the HUD without sending any input.
    public var dryRun = false
    /// Pause when the person uses the keyboard or mouse.
    public var pauseOnHumanInput = true
    /// Only send input while the controlled window's app is in front.
    public var focusGuard = true
    /// Feed the model its past actions (as in training). Off = decide from the screen only.
    public var useActionHistory = true
    /// Keep the GPU clocked up between steps with a tiny filler kernel. Without it the GPU drops to a
    /// low clock when the model leaves it idle half the time, and steps get 2-3× slower.
    public var keepGPUWarm = true
    public var captureMaxHeight = 720
    public var captureFPS = 60
    public var hotkeys: [(code: UInt16, modifiers: CGEventFlags)] = []
    public init() {}
}

public struct RuntimeStats: Sendable, Equatable {
    public var steps = 0
    public var fps = 0.0
    public var modelMs = 0.0
    public var preprocessMs = 0.0
    public var frameAgeMs = 0.0
    public var endToEndMs = 0.0
    public var droppedTicks = 0
    public var takeovers = 0
    public var paused = false
    public var pauseReason = ""
    public var actionsPerSecond = 0.0
    /// Mean model-step time over steps 50-150 and over the last 100 steps (diagnostics).
    public var earlyStepMs = 0.0
    public var lateStepMs = 0.0
    public init() {}
}

/// Runs a model live against a window or display.
public final class AgentRuntime: @unchecked Sendable {
    public let model: PolicyModel
    public let definition: ModelDefinition
    public let target: CaptureTarget
    public let output = LiveInputState()
    public var options: RuntimeOptions
    public var onVision: ((CGImage) -> Void)?
    /// Instruction / screen-text encoder (EmbeddingGemma) for language-enabled models.
    public var encoder: InstructionEncoder?
    private var feed: LanguageFeed?
    private var instructionText: String?
    private var screenItems: [ScreenTextItem]?
    private let ocrQueue = DispatchQueue(label: "agenttrainer.ocr", qos: .utility)
    private var ocrTimer: DispatchSourceTimer?
    /// Keeps macOS from napping / deprioritizing AgentTrainer while the game is in front.
    private var activity: NSObjectProtocol?
    /// Frontmost app, kept current by workspace notifications (asking NSWorkspace every tick is slow).
    private var frontmostPID: pid_t = NSWorkspace.shared.frontmostApplication?.processIdentifier ?? 0
    private var workspaceObserver: NSObjectProtocol?
    public var onStop: ((String?) -> Void)?

    /// Created on the loop thread itself: MLX work stays on that thread's stream.
    private var stepper: PolicyStepper!
    private let capturer = ScreenCapturer()
    private let tap = InputTap()
    private let executor = ActionExecutor()
    private let resizer: FrameResizer
    private let lock = NSLock()
    /// Signalled when the loop thread has finished (keys released, MLX idle).
    private let loopFinished = DispatchSemaphore(value: 0)
    private var loopStarted = false
    private var latest: CapturedFrame?
    private var running = false
    private var paused = false
    private var pauseReason = ""
    private var thread: Thread?
    private var text: [Float]?
    private var thinkingExtra: MLXArray?
    private var stats = RuntimeStats()
    private var previous: [Int32]?
    // Human input during the current window (for history while paused / takeover detection).
    private var humanKeys = Set<UInt16>()
    private var humanButtons = Set<Int>()
    private var humanDelta = CGVector.zero
    private var humanScroll = CGVector.zero
    private var humanMoved = 0.0
    /// Host-time ranges when the model was in control (for correction recordings).
    public private(set) var controlRanges: [(UInt64, UInt64)] = []
    private var controlStart: UInt64?

    public init(model: PolicyModel, definition: ModelDefinition, target: CaptureTarget, options: RuntimeOptions) throws {
        self.model = model
        self.definition = definition
        self.target = target
        self.options = options
        guard let r = FrameResizer() else { throw NSError(domain: "AgentTrainer", code: 1, userInfo: [NSLocalizedDescriptionKey: "Metal is unavailable"]) }
        resizer = r
    }

    public var currentStats: RuntimeStats { lock.lock(); defer { lock.unlock() }; return stats }
    public var isPaused: Bool { lock.lock(); defer { lock.unlock() }; return paused }

    /// Sets the instruction embedding (nil = no instruction). Takes effect on the next step.
    public func setInstruction(_ embedding: [Float]?, text: String? = nil) { lock.lock(); self.text = embedding; instructionText = text; lock.unlock() }
    public func setThinkingExtra(_ v: MLXArray?) { lock.lock(); thinkingExtra = v; lock.unlock() }
    public func latestFrame() -> CapturedFrame? { lock.lock(); defer { lock.unlock() }; return latest }

    public func start(excluding own: [SCRunningApplication]) async throws {
        guard AXIsProcessTrusted() || options.dryRun else {
            throw NSError(domain: "AgentTrainer", code: 2, userInfo: [NSLocalizedDescriptionKey: "AgentTrainer needs Accessibility permission to press keys and move the mouse."])
        }
        executor.mouseMode = options.mouseMode
        executor.sensitivity = options.mouseSensitivity
        capturer.onFrame = { [weak self] f in
            guard let self else { return }
            self.lock.lock(); self.latest = f; self.lock.unlock()
        }
        capturer.onStop = { [weak self] e in self?.stop(reason: e?.localizedDescription ?? "Capture stopped") }
        let settings = CaptureSettings.fitting(target, maxHeight: options.captureMaxHeight, fps: options.captureFPS,
                                               showsCursor: definition.actions.mouseMode != .absolute)
        try await capturer.start(target, settings: settings, excluding: own)
        tap.ignoredHotkeys = options.hotkeys
        tap.onEvent = { [weak self] e in self?.human(e) }
        // Takeover detection needs the tap (and Input Monitoring); without it the model still runs.
        if options.pauseOnHumanInput { try? tap.start() }
        running = true
        workspaceObserver = NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: nil) { [weak self] n in
            guard let app = n.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication else { return }
            self?.lock.lock(); self?.frontmostPID = app.processIdentifier; self?.lock.unlock()
        }
        activity = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .latencyCritical, .idleSystemSleepDisabled, .idleDisplaySleepDisabled],
                                                         reason: "Running an AI model in real time")
        controlStart = HostClock.now()
        if definition.architecture.usesDecoderMemory { feed = LanguageFeed(architecture: definition.architecture, encoder: encoder) }
        if definition.architecture.screenText {
            let t = DispatchSource.makeTimerSource(queue: ocrQueue)
            t.schedule(deadline: .now() + 0.2, repeating: ScreenTextReader.interval)
            t.setEventHandler { [weak self] in
                guard let self, let f = self.latestFrame() else { return }
                let items = ScreenTextReader.read(f.pixelBuffer, maxItems: self.definition.architecture.screenTextItems)
                self.lock.lock(); self.screenItems = items; self.lock.unlock()
            }
            t.resume()
            ocrTimer = t
        }
        loopStarted = true
        DispatchQueue.global(qos: .userInteractive).async { [weak self] in
            self?.loop()
            self?.loopFinished.signal()
        }
    }

    /// Waits for the loop thread to finish its current step and release held input. Call after
    /// `stop()` before quitting, so the process never exits while the GPU is mid-step.
    @discardableResult
    public func waitUntilStopped(timeout: Double = 2) -> Bool {
        guard loopStarted else { return true }
        let ok = loopFinished.wait(timeout: .now() + timeout) == .success
        if ok { loopStarted = false }
        return ok
    }

    public func stop(reason: String? = nil) {
        lock.lock()
        guard running else { lock.unlock(); return }
        running = false
        if let s = controlStart { controlRanges.append((s, HostClock.now())); controlStart = nil }
        lock.unlock()
        ocrTimer?.cancel()
        ocrTimer = nil
        if let a = activity { ProcessInfo.processInfo.endActivity(a); activity = nil }
        if let o = workspaceObserver { NSWorkspace.shared.notificationCenter.removeObserver(o); workspaceObserver = nil }
        tap.stop()
        Task { await capturer.stop() }
        DispatchQueue.main.async { self.onStop?(reason) }
    }

    public func setPaused(_ p: Bool, reason: String = "") {
        lock.lock()
        paused = p
        pauseReason = reason
        stats.paused = p
        stats.pauseReason = reason
        let now = HostClock.now()
        if p, let s = controlStart { controlRanges.append((s, now)); controlStart = nil }
        if !p, controlStart == nil { controlStart = now }
        lock.unlock()
        if p { executor.releaseAll(); output.clear() }
    }

    // MARK: Human input

    private func human(_ e: InputEventRecord) {
        guard !e.isSynthetic else { return }
        lock.lock()
        switch e.kind {
        case .keyDown: humanKeys.insert(e.code)
        case .keyUp: humanKeys.remove(e.code)
        case .flagsChanged: if e.flags & 2 != 0 { humanKeys.insert(e.code) } else { humanKeys.remove(e.code) }
        case .buttonDown: humanButtons.insert(Int(e.code))
        case .buttonUp: humanButtons.remove(Int(e.code))
        case .mouseMove: humanDelta.dx += CGFloat(e.dx); humanDelta.dy += CGFloat(e.dy); humanMoved += Double(abs(e.dx) + abs(e.dy))
        case .scroll: humanScroll.dx += CGFloat(e.dx); humanScroll.dy += CGFloat(e.dy)
        }
        let takeover = options.pauseOnHumanInput && !paused && (e.kind == .keyDown || e.kind == .buttonDown || e.kind == .scroll || humanMoved > 12)
        lock.unlock()
        if takeover {
            lock.lock(); stats.takeovers += 1; lock.unlock()
            setPaused(true, reason: "You took over. Press the pause shortcut or Resume to hand back control.")
        }
    }

    /// The person's input over the last window, as a model action (history while they're in control).
    private func drainHuman(area: CGRect) -> TimestepAction {
        lock.lock(); defer { lock.unlock() }
        var a = TimestepAction()
        a.keys = Set(humanKeys.compactMap { KeyCatalog.byKeyCode[$0]?.id })
        a.pointer = Set(humanButtons.compactMap { b in PointerEvent.allCases.first { $0.buttonNumber == b } })
        if let s = PointerEvent.fromScroll(lines: humanScroll.dx, humanScroll.dy) { a.pointer.insert(s) }
        a.dx = humanDelta.dx; a.dy = humanDelta.dy
        let loc = CGEvent(source: nil)?.location ?? .zero
        a.absX = min(1, max(0, (loc.x - area.minX) / max(area.width, 1)))
        a.absY = min(1, max(0, (loc.y - area.minY) / max(area.height, 1)))
        humanDelta = .zero; humanScroll = .zero; humanMoved = 0
        return a
    }

    // MARK: Loop

    private func loop() {
        stepper = PolicyStepper(model: model)
        stepper.temperature = options.temperature
        stepper.useActionHistory = options.useActionHistory
        let rate = definition.architecture.stepRateHz
        let period = UInt64(1_000_000_000 / rate)
        let size = definition.architecture.frameSize
        let space = definition.actions
        let absolute = space.mouseMode == .absolute
        let area = target.frame
        var next = HostClock.now()
        var tickTimes: [UInt64] = []
        var actionCount = 0
        var stepTimes: [Double] = []
        let warmer = options.keepGPUWarm ? GPUWarmer() : nil
        defer { warmer?.finish() }
        while true {
            lock.lock(); let isRunning = running; lock.unlock()
            if !isRunning { break }
            next += period
            let now = HostClock.now()
            if now > next + period {
                lock.lock(); stats.droppedTicks += Int((now - next) / period); lock.unlock()
                next = now
            }
            HostClock.sleep(untilNanoseconds: next)
            warmer?.finish()
            guard let frame = latestFrame() else { continue }
            let t0 = HostClock.now()

            let loc = CGEvent(source: nil)?.location ?? .zero
            let cursor = absolute ? CGPoint(x: (loc.x - area.minX) / max(area.width, 1), y: (loc.y - area.minY) / max(area.height, 1)) : nil
            var rgb = Data()
            let input: MLXArray? = resizer.resize(frame.pixelBuffer, size: size, cursor: cursor) { buf in
                rgb = Data(buf)
                return MLXArray(rgb, [size, size, 3], dtype: .uint8)
            }
            guard let input else { continue }
            let t1 = HostClock.now()

            lock.lock(); let text = self.text, extra = self.thinkingExtra, isPaused = paused, prev = previous
            let items = screenItems, insText = instructionText, front = frontmostPID; lock.unlock()
            if let feed {
                feed.setInstruction(insText)
                if let items { feed.setScreenText(items) }
                if case .some(let l) = feed.update() { stepper.setLanguage(l) }
            }
            let focusOK = !options.focusGuard || target.isDisplay || front == target.processID
            let tm = HostClock.now()
            let result = stepper.step(frame: input, text: text, previousAction: prev, thinkingExtra: extra)
            let t2 = HostClock.now()
            var action = space.decode(result.classes)

            var executed = TimestepAction()
            if isPaused || !focusOK || options.dryRun {
                if isPaused || !focusOK { executor.releaseAll() }
                executed = isPaused || !focusOK ? drainHuman(area: area) : action
                if !focusOK && !isPaused { lock.lock(); stats.pauseReason = "Waiting for \(target.info.appName ?? "the target") to be in front"; lock.unlock() }
            } else {
                if !absolute { action.absX = Double(cursor?.x ?? 0.5); action.absY = Double(cursor?.y ?? 0.5) }
                executor.apply(action, absolute: absolute, area: area, permissions: options.permissions)
                executed = action
                executed.keys = executor.heldKeys
                executed.pointer = action.pointer.filter { $0.isScroll ? options.permissions.scroll : options.permissions.mouseButtons }
                if !options.permissions.mouseMovement { executed.dx = 0; executed.dy = 0 }
                _ = drainHuman(area: area)
                lock.lock(); stats.pauseReason = ""; lock.unlock()
            }
            let enc = space.encode(executed)
            if !executed.keys.isEmpty || !executed.pointer.isEmpty || executed.dx != 0 || executed.dy != 0 { actionCount += 1 }
            let t3 = HostClock.now()

            // HUD: what the model is doing (even in dry run).
            output.set(keys: isPaused ? [] : action.keys, buttons: isPaused ? [] : Set(action.pointer.compactMap(\.buttonNumber)),
                       velocity: CGVector(dx: action.dx * Double(rate), dy: action.dy * Double(rate)),
                       scroll: action.pointer.contains(where: \.isScroll) ? CGVector(dx: 0, dy: action.pointer.contains(.scrollUp) || action.pointer.contains(.scrollUpFast) ? 1 : -1) : .zero)
            if stats.steps % 3 == 0, let onVision, let img = Self.image(rgb, size: size, gray: definition.architecture.grayscale) { onVision(img) }

            tickTimes.append(t3)
            if tickTimes.count > rate { tickTimes.removeFirst(tickTimes.count - rate) }
            lock.lock()
            previous = enc
            stats.steps += 1
            stats.preprocessMs = stats.preprocessMs * 0.9 + Double(t1 - t0) / 1e6 * 0.1
            stats.modelMs = stats.modelMs * 0.9 + Double(t2 - tm) / 1e6 * 0.1
            stepTimes.append(Double(t2 - tm) / 1e6)
            if stepTimes.count == 150 { stats.earlyStepMs = stepTimes[50 ..< 150].reduce(0, +) / 100 }
            if stepTimes.count > 150 { stats.lateStepMs = stepTimes.suffix(100).reduce(0, +) / 100 }
            stats.frameAgeMs = stats.frameAgeMs * 0.9 + Double(t0 > frame.hostNs ? t0 - frame.hostNs : 0) / 1e6 * 0.1
            stats.endToEndMs = stats.endToEndMs * 0.9 + Double(t3 > frame.hostNs ? t3 - frame.hostNs : 0) / 1e6 * 0.1
            if tickTimes.count > 1 { stats.fps = Double(tickTimes.count - 1) / (Double(tickTimes.last! - tickTimes.first!) / 1e9) }
            stats.actionsPerSecond = stats.actionsPerSecond * 0.95 + Double(actionCount) * Double(rate) * 0.05
            actionCount = 0
            lock.unlock()
            warmer?.fill(until: next + period)
        }
        executor.releaseAll()
        output.clear()
    }

    static func image(_ rgb: Data, size: Int, gray: Bool) -> CGImage? {
        var bytes = rgb
        if gray {
            bytes.withUnsafeMutableBytes { p in
                let b = p.bindMemory(to: UInt8.self)
                for i in stride(from: 0, to: b.count, by: 3) {
                    let y = UInt8(min(255, Double(b[i]) * 0.299 + Double(b[i + 1]) * 0.587 + Double(b[i + 2]) * 0.114))
                    b[i] = y; b[i + 1] = y; b[i + 2] = y
                }
            }
        }
        guard let provider = CGDataProvider(data: bytes as CFData) else { return nil }
        return CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: size * 3,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider,
                       decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}
