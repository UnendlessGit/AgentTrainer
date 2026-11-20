import ATCapture
import ATCore
import CoreGraphics
import Foundation

/// A recording's input as a timed list of things to send, with time zero at the recording's first
/// video frame, so a replay started on the same spot lines up with the video.
///
/// - Keys and buttons already held when the video starts are pressed at time zero; movement from
///   before the first frame is left out (its effect is already in the first frame).
/// - Nothing after the last frame is sent (that is usually the shortcut that stopped the recording).
/// - The Run tab's allowed actions apply: a blocked key is never pressed, and neither is a key
///   pressed while the recording held a blocked modifier (⌘Q with Command blocked does not become Q).
///   AgentTrainer's own shortcuts are never sent.
public struct ReplayPlan: Sendable {
    public enum Action: Equatable, Sendable {
        /// Modifier keys are sent as flag changes.
        case key(code: UInt16, down: Bool)
        /// `at`: where to press (pointer mode), or nil for wherever the pointer is (camera mode).
        case button(Int, down: Bool, at: CGPoint?)
        /// A relative movement. `to` also moves the pointer there (pointer mode); nil leaves it in place.
        case move(dx: Double, dy: Double, to: CGPoint?)
        /// Lines, y > 0 up.
        case scroll(dx: Double, dy: Double)
    }

    public struct Step: Equatable, Sendable {
        public var timeNs: Int64
        public var action: Action
        public init(timeNs: Int64, action: Action) { self.timeNs = timeNs; self.action = action }
    }

    public struct Options: @unchecked Sendable {
        public var mouseMode = MouseOutputMode.camera
        public var permissions = ActionPermissions()
        /// AgentTrainer's own shortcuts (key code + modifiers): never sent, even if recorded.
        public var hotkeys: [(code: UInt16, modifiers: CGEventFlags)] = []
        /// The target window's frame now, in global points (pointer mode maps positions onto it).
        public var targetFrame: CGRect?
        public init() {}
    }

    public var steps: [Step]
    /// From the first to the last video frame.
    public var durationNs: Int64
    /// The mouse mode actually used (Elkborn replays are always camera movement).
    public var mouseMode: MouseOutputMode
    /// Key and button presses left out (blocked, AgentTrainer shortcuts, chords with a blocked modifier).
    public var skippedPresses: Int

    public var duration: Double { Double(durationNs) / 1e9 }

    /// Elkborn replays carry the run's camera turns as mouse movement at a fixed point, so they
    /// only make sense as camera movement.
    public static func isElkborn(_ meta: RecordingMeta) -> Bool { meta.eventClock == "elkborn-replay" }

    public static func load(_ package: URL, options: Options) throws -> ReplayPlan {
        let meta = try RecordingIO.readMeta(package)
        return make(frames: try RecordingIO.readFrames(package), events: try RecordingIO.readEvents(package), meta: meta, options: options)
    }

    static let modifierIDs: Set<String> = ["shift", "rshift", "control", "rcontrol", "option", "roption", "command", "rcommand"]

    static func flags(_ modifiers: Set<String>) -> CGEventFlags {
        var f = CGEventFlags()
        for m in modifiers {
            switch m {
            case "shift", "rshift": f.insert(.maskShift)
            case "control", "rcontrol": f.insert(.maskControl)
            case "option", "roption": f.insert(.maskAlternate)
            case "command", "rcommand": f.insert(.maskCommand)
            default: break
            }
        }
        return f
    }

    public static func make(frames: [FrameRecord], events: [InputEventRecord], meta: RecordingMeta, options: Options) -> ReplayPlan {
        let ordered = events.enumerated().sorted { $0.element.timeNs != $1.element.timeNs ? $0.element.timeNs < $1.element.timeNs : $0.offset < $1.offset }.map(\.element)
        let t0 = frames.first?.timeNs ?? ordered.first?.timeNs ?? 0
        let end = frames.last?.timeNs ?? ordered.last?.timeNs ?? t0
        let mode: MouseOutputMode = isElkborn(meta) ? .camera : options.mouseMode
        let perms = options.permissions

        // Pointer mode: recorded positions (global points over meta.sourceRect) onto the target window now.
        let source = meta.sourceRect.cgRect
        var map: ((Float, Float) -> CGPoint)?
        var scale = CGVector(dx: 1, dy: 1)
        if mode == .pointer {
            let target = options.targetFrame ?? source
            let sx = source.width > 0 ? target.width / source.width : 1, sy = source.height > 0 ? target.height / source.height : 1
            scale = CGVector(dx: sx, dy: sy)
            map = { x, y in CGPoint(x: target.minX + (CGFloat(x) - source.minX) * sx, y: target.minY + (CGFloat(y) - source.minY) * sy) }
        }

        var steps: [Step] = []
        var skipped = 0
        var recordedMods = Set<String>() // what the human held, blocked or not
        var held = Set<UInt16>() // what the replay holds
        var skippedDown = Set<UInt16>()
        var buttons = Set<Int>()
        var lastPoint: (Float, Float)?
        var startFlushed = false

        func flushStart() {
            guard !startFlushed else { return }
            startFlushed = true
            // Held when the video starts: modifiers first, so the keys after them carry the right flags.
            let ids = held.compactMap { KeyCatalog.byKeyCode[$0] }
            for info in ids.sorted(by: { (modifierIDs.contains($0.id) ? 0 : 1, $0.keyCode) < (modifierIDs.contains($1.id) ? 0 : 1, $1.keyCode) }) {
                steps.append(Step(timeNs: 0, action: .key(code: info.keyCode, down: true)))
            }
            for b in buttons.sorted() { steps.append(Step(timeNs: 0, action: .button(b, down: true, at: lastPoint.flatMap { p in map?(p.0, p.1) }))) }
        }

        for e in ordered {
            if e.timeNs > end { break }
            let pre = e.timeNs < t0
            if !pre { flushStart() }
            let t = max(0, e.timeNs - t0)
            switch e.kind {
            case .flagsChanged, .keyDown, .keyUp:
                guard let info = KeyCatalog.byKeyCode[e.code] else { continue }
                let isModifier = modifierIDs.contains(info.id)
                let down: Bool = e.kind == .flagsChanged ? e.flags & 2 != 0 : e.kind == .keyDown
                if isModifier {
                    if down { recordedMods.insert(info.id) } else { recordedMods.remove(info.id) }
                    guard perms.allows(key: info.id) else { if down { skipped += 1 }; continue }
                } else if e.kind == .flagsChanged {
                    continue // caps lock and fn are toggles/system keys: never replayed
                }
                if down {
                    if held.contains(e.code) || skippedDown.contains(e.code) { continue }
                    if !isModifier {
                        let chordFlags = flags(recordedMods)
                        let blockedChord = recordedMods.contains { !perms.allows(key: $0) }
                        let shortcut = options.hotkeys.contains { $0.code == e.code && !$0.modifiers.isEmpty && chordFlags.contains($0.modifiers) }
                        if !perms.allows(key: info.id) || blockedChord || shortcut {
                            skippedDown.insert(e.code); skipped += 1
                            continue
                        }
                    }
                    held.insert(e.code)
                    if !pre { steps.append(Step(timeNs: t, action: .key(code: e.code, down: true))) }
                } else {
                    if skippedDown.remove(e.code) != nil { continue }
                    if held.remove(e.code) != nil, !pre { steps.append(Step(timeNs: t, action: .key(code: e.code, down: false))) }
                }
            case .buttonDown, .buttonUp:
                lastPoint = (e.x, e.y)
                let b = Int(e.code)
                guard perms.mouseButtons else { if e.kind == .buttonDown { skipped += 1 }; continue }
                if e.kind == .buttonDown {
                    guard buttons.insert(b).inserted else { continue }
                    if !pre { steps.append(Step(timeNs: t, action: .button(b, down: true, at: map?(e.x, e.y)))) }
                } else if buttons.remove(b) != nil, !pre {
                    steps.append(Step(timeNs: t, action: .button(b, down: false, at: map?(e.x, e.y))))
                }
            case .mouseMove:
                lastPoint = (e.x, e.y)
                guard !pre, perms.mouseMovement else { continue }
                steps.append(Step(timeNs: t, action: .move(dx: Double(e.dx) * scale.dx, dy: Double(e.dy) * scale.dy, to: map?(e.x, e.y))))
            case .scroll:
                guard !pre, perms.scroll else { continue }
                steps.append(Step(timeNs: t, action: .scroll(dx: Double(e.dx), dy: Double(e.dy))))
            }
        }
        flushStart()
        return ReplayPlan(steps: steps, durationNs: max(0, end - t0), mouseMode: mode, skippedPresses: skipped)
    }
}

/// Time for a replay: the host clock, or a virtual one in tests.
public struct ReplayClock: @unchecked Sendable {
    /// Nanoseconds.
    public var now: () -> UInt64
    /// Blocks until `deadline` (ns) or until `wake` is signalled, whichever is first.
    public var wait: (_ deadline: UInt64, _ wake: DispatchSemaphore) -> Void

    public init(now: @escaping () -> UInt64, wait: @escaping (UInt64, DispatchSemaphore) -> Void) { self.now = now; self.wait = wait }

    private static let timebase: mach_timebase_info_data_t = {
        var t = mach_timebase_info_data_t()
        mach_timebase_info(&t)
        return t
    }()

    /// Sleeps on the semaphore (so a stop wakes it at once) until ~1.5 ms before the deadline, then
    /// waits the rest out precisely with `mach_wait_until`.
    public static let host = ReplayClock(now: HostClock.now) { deadline, wake in
        let now = HostClock.now()
        guard deadline > now else { return }
        let left = deadline - now
        if left > 2_000_000 {
            if wake.wait(timeout: .now() + .nanoseconds(Int(left - 1_500_000))) == .success { return }
        }
        let tb = timebase
        mach_wait_until(deadline * UInt64(tb.denom) / UInt64(tb.numer))
    }
}

/// Plays a `ReplayPlan` into the system as real keyboard and mouse events, on a dedicated
/// high-priority thread with sub-millisecond timing. Keeps track of what it holds and releases
/// everything when it stops, finishes, or the target loses focus (the clock holds until the
/// target is back, then the keys are pressed again and the replay continues).
public final class InputReplayer: @unchecked Sendable {
    /// One thing to send. `flags` are the modifier flags the replay holds at that moment.
    public enum Output: Equatable, Sendable {
        case key(code: UInt16, down: Bool, flags: UInt64, flagsChanged: Bool)
        /// `at`: nil = the pointer's current place.
        case button(Int, down: Bool, at: CGPoint?, flags: UInt64)
        /// Whole units; `to`: nil = the pointer's current place. `dragging`: held button, if any.
        case move(dx: Int, dy: Int, to: CGPoint?, dragging: Int?, flags: UInt64)
        /// Whole lines, y > 0 up.
        case scroll(dx: Int, dy: Int)
    }

    public enum Ending: Equatable, Sendable { case finished, stopped }

    public struct Status: Equatable, Sendable {
        public var elapsed = 0.0
        public var duration = 0.0
        public var waitingForFocus = false
        public var running = false
    }

    public let plan: ReplayPlan
    /// Sends one output. Posts a real event by default; tests replace it.
    public var emit: (Output) -> Void = { InputReplayer.post($0) }
    /// While this returns false the replay holds: everything is released and the clock stops.
    public var isTargetFrontmost: () -> Bool = { true }
    public var onEnd: ((Ending) -> Void)?
    /// What the replay is pressing, for the HUD.
    public let live = LiveInputState()

    private let clock: ReplayClock
    private let wake = DispatchSemaphore(value: 0)
    private let ended = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var stopRequested = false
    private var _status: Status
    private var thread: Thread?

    private var heldKeys = Set<UInt16>()
    private var heldButtons = Set<Int>()
    private var fraction = CGVector.zero
    private var scrollFraction = CGVector.zero
    private var startNs: UInt64 = 0
    private var pausedNs: UInt64 = 0

    public init(plan: ReplayPlan, clock: ReplayClock = .host) {
        self.plan = plan
        self.clock = clock
        _status = Status(duration: plan.duration)
    }

    public var status: Status { lock.withLock { _status } }

    /// Starts on its own thread.
    public func start() {
        let t = Thread { [weak self] in self?.run() }
        t.qualityOfService = .userInteractive
        t.name = "agenttrainer.replay"
        thread = t
        t.start()
    }

    /// Stops soon (within a millisecond or two); everything held is released and `onEnd(.stopped)` follows.
    public func stop() {
        lock.withLock { stopRequested = true }
        wake.signal()
    }

    /// Stops and waits (up to `seconds`) until everything has been let go, e.g. when the app quits.
    public func stop(waitingUpTo seconds: Double) {
        guard status.running else { return }
        stop()
        _ = ended.wait(timeout: .now() + seconds)
    }

    private var stopping: Bool { lock.withLock { stopRequested } }

    /// Plays the whole plan on the calling thread.
    public func run() {
        lock.withLock { _status.running = true }
        startNs = clock.now()
        for step in plan.steps {
            guard waitUntil(step.timeNs) else { break }
            apply(step.action)
        }
        // Let the run play out to its last frame before letting go.
        if !stopping { _ = waitUntil(plan.durationNs) }
        releaseAll()
        let ending: Ending = stopping ? .stopped : .finished
        lock.withLock {
            _status.running = false
            _status.waitingForFocus = false
            if ending == .finished { _status.elapsed = _status.duration }
        }
        live.clear()
        ended.signal()
        onEnd?(ending)
    }

    /// Waits for plan time `t`; false when stopped.
    private func waitUntil(_ t: Int64) -> Bool {
        while true {
            if stopping { return false }
            if !isTargetFrontmost() { hold(); continue }
            let deadline = startNs &+ pausedNs &+ UInt64(max(0, t))
            let now = clock.now()
            lock.withLock { _status.elapsed = min(_status.duration, Double(now &- startNs &- pausedNs) / 1e9) }
            if now >= deadline { return true }
            // Wake at least every 50 ms to notice focus changes.
            clock.wait(min(deadline, now + 50_000_000), wake)
        }
    }

    /// The target is not in front: let go of everything and stop the clock until it is.
    private func hold() {
        let began = clock.now()
        lock.withLock { _status.waitingForFocus = true }
        let keys = heldKeys, buttons = heldButtons
        releaseAll()
        while !stopping, !isTargetFrontmost() { clock.wait(clock.now() + 50_000_000, wake) }
        pausedNs &+= clock.now() &- began
        lock.withLock { _status.waitingForFocus = false }
        guard !stopping else { return }
        for code in keys.sorted(by: { (Self.isModifier($0) ? 0 : 1, $0) < (Self.isModifier($1) ? 0 : 1, $1) }) { pressKey(code, down: true) }
        for b in buttons.sorted() { pressButton(b, down: true, at: nil) }
    }

    // MARK: Sending

    static let modifierMasks: [UInt16: CGEventFlags] = [
        0x38: .maskShift, 0x3C: .maskShift, 0x3B: .maskControl, 0x3E: .maskControl,
        0x3A: .maskAlternate, 0x3D: .maskAlternate, 0x37: .maskCommand, 0x36: .maskCommand,
    ]

    static func isModifier(_ code: UInt16) -> Bool { modifierMasks[code] != nil }

    private var flags: UInt64 {
        var f = CGEventFlags()
        for k in heldKeys { if let m = Self.modifierMasks[k] { f.insert(m) } }
        return f.rawValue
    }

    private func apply(_ a: ReplayPlan.Action) {
        switch a {
        case let .key(code, down): pressKey(code, down: down)
        case let .button(b, down, at): pressButton(b, down: down, at: at)
        case let .move(dx, dy, to):
            let x = dx + fraction.dx, y = dy + fraction.dy
            let ix = x.rounded(.towardZero), iy = y.rounded(.towardZero)
            fraction = CGVector(dx: x - ix, dy: y - iy)
            if ix == 0, iy == 0, to == nil { return }
            let dragging = heldButtons.contains(0) ? 0 : heldButtons.contains(1) ? 1 : heldButtons.min()
            emit(.move(dx: Int(ix), dy: Int(iy), to: to, dragging: dragging, flags: flags))
            live.ingest(InputEventRecord(timeNs: 0, kind: .mouseMove, dx: Float(ix), dy: Float(iy)))
        case let .scroll(dx, dy):
            let x = dx + scrollFraction.dx, y = dy + scrollFraction.dy
            let ix = x.rounded(.towardZero), iy = y.rounded(.towardZero)
            scrollFraction = CGVector(dx: x - ix, dy: y - iy)
            if ix == 0, iy == 0 { return }
            emit(.scroll(dx: Int(ix), dy: Int(iy)))
            live.ingest(InputEventRecord(timeNs: 0, kind: .scroll, dx: Float(ix), dy: Float(iy)))
        }
    }

    private func pressKey(_ code: UInt16, down: Bool) {
        if down { guard heldKeys.insert(code).inserted else { return } } else { guard heldKeys.remove(code) != nil else { return } }
        let modifier = Self.isModifier(code)
        emit(.key(code: code, down: down, flags: flags, flagsChanged: modifier))
        live.ingest(InputEventRecord(timeNs: 0, kind: modifier ? .flagsChanged : down ? .keyDown : .keyUp, flags: modifier && down ? 2 : 0, code: code))
    }

    private func pressButton(_ b: Int, down: Bool, at: CGPoint?) {
        if down { guard heldButtons.insert(b).inserted else { return } } else { guard heldButtons.remove(b) != nil else { return } }
        emit(.button(b, down: down, at: at, flags: flags))
        live.ingest(InputEventRecord(timeNs: 0, kind: down ? .buttonDown : .buttonUp, code: UInt16(b)))
    }

    /// Lets go of every button, then every key, then the modifiers.
    private func releaseAll() {
        for b in heldButtons.sorted() { pressButton(b, down: false, at: nil) }
        for code in heldKeys.sorted(by: { (Self.isModifier($0) ? 1 : 0, $0) < (Self.isModifier($1) ? 1 : 0, $1) }) { pressKey(code, down: false) }
    }

    // MARK: Real events

    private static let source = CGEventSource(stateID: .hidSystemState)

    /// Posts one output as a real event, marked as AgentTrainer's own (so recordings and takeover
    /// detection can tell it from the user's input).
    public static func post(_ o: Output) {
        let event: CGEvent?
        switch o {
        case let .key(code, down, flags, flagsChanged):
            event = CGEvent(keyboardEventSource: source, virtualKey: code, keyDown: down)
            if flagsChanged { event?.type = .flagsChanged }
            event?.flags = CGEventFlags(rawValue: flags)
        case let .button(b, down, at, flags):
            let type: CGEventType = switch (b, down) {
            case (0, true): .leftMouseDown
            case (0, false): .leftMouseUp
            case (1, true): .rightMouseDown
            case (1, false): .rightMouseUp
            case (_, true): .otherMouseDown
            case (_, false): .otherMouseUp
            }
            event = CGEvent(mouseEventSource: source, mouseType: type, mouseCursorPosition: at ?? currentPointer, mouseButton: CGMouseButton(rawValue: UInt32(b)) ?? .left)
            event?.setIntegerValueField(.mouseEventButtonNumber, value: Int64(b))
            event?.flags = CGEventFlags(rawValue: flags)
        case let .move(dx, dy, to, dragging, flags):
            let type: CGEventType = switch dragging {
            case nil: .mouseMoved
            case 0?: .leftMouseDragged
            case 1?: .rightMouseDragged
            default: .otherMouseDragged
            }
            event = CGEvent(mouseEventSource: source, mouseType: type, mouseCursorPosition: to ?? currentPointer, mouseButton: CGMouseButton(rawValue: UInt32(dragging ?? 0)) ?? .left)
            event?.setIntegerValueField(.mouseEventDeltaX, value: Int64(dx))
            event?.setIntegerValueField(.mouseEventDeltaY, value: Int64(dy))
            event?.setDoubleValueField(.mouseEventDeltaX, value: Double(dx))
            event?.setDoubleValueField(.mouseEventDeltaY, value: Double(dy))
            event?.flags = CGEventFlags(rawValue: flags)
        case let .scroll(dx, dy):
            event = CGEvent(scrollWheelEvent2Source: source, units: .line, wheelCount: 2, wheel1: Int32(dy), wheel2: Int32(dx), wheel3: 0)
        }
        guard let event else { return }
        event.setIntegerValueField(.eventSourceUserData, value: SyntheticInput.userData)
        event.post(tap: .cghidEventTap)
    }

    private static var currentPointer: CGPoint { CGEvent(source: nil)?.location ?? .zero }
}
