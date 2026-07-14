import AppKit
import CoreGraphics
import Foundation

let agentTrainerSyntheticTag: Int64 = 0x4154474E_54524E52

final class InputCaptureService: @unchecked Sendable {
    var onSample: (@Sendable (InputSample) -> Void)?
    var onState: (@Sendable (InputState) -> Void)?

    private var tap: CFMachPort?
    private var source: CFRunLoopSource?
    private var thread: Thread?
    private var threadExit = DispatchSemaphore(value: 0)
    private let lifecycleLock = NSLock()
    private var runLoop: CFRunLoop?
    private let stateLock = NSLock()
    private var state = InputState()
    private var lastStateReportTime = 0.0
    private var lastReportedState = InputState.empty
    private let filterLock = NSLock()
    private var hotkeyFilter = HotkeyInputFilter()
    private var recordingKeyFilter = RecordingKeyFilter()

    var ignoredHotkeys: [HotkeyBinding] {
        get { filterLock.withLock { hotkeyFilter.bindings } }
        set { filterLock.withLock { hotkeyFilter.bindings = newValue } }
    }


    var excludedKeyCodes: Set<UInt16> {
        get { filterLock.withLock { recordingKeyFilter.excludedKeyCodes } }
        set { filterLock.withLock { recordingKeyFilter.excludedKeyCodes = newValue } }
    }

    var isRunning: Bool { tap != nil }

    func start() throws {
        guard tap == nil else { return }
        filterLock.withLock { hotkeyFilter.reset(); recordingKeyFilter.reset() }
        threadExit = DispatchSemaphore(value: 0)
        let mask: CGEventMask = [
            CGEventType.mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged,
            .leftMouseDown, .leftMouseUp, .rightMouseDown, .rightMouseUp, .otherMouseDown, .otherMouseUp,
            .scrollWheel, .keyDown, .keyUp, .flagsChanged
        ].reduce(0) { $0 | (1 << CGEventMask($1.rawValue)) }

        let pointer = Unmanaged.passUnretained(self).toOpaque()
        guard let tap = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap, options: .listenOnly, eventsOfInterest: mask, callback: Self.callback, userInfo: pointer) else {
            throw AgentTrainerError.permission("Input Monitoring permission is required. Enable AgentTrainer in System Settings → Privacy & Security → Input Monitoring, then reopen the app.")
        }
        self.tap = tap
        let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        self.source = source
        let ready = DispatchSemaphore(value: 0)

        let thread = Thread { [weak self] in
            guard let self else { return }
            defer { self.threadExit.signal() }
            guard let source = self.source, let tap = self.tap else { ready.signal(); return }
            let loop = CFRunLoopGetCurrent()
            self.lifecycleLock.withLock { self.runLoop = loop }
            CFRunLoopAddSource(loop, source, .commonModes)
            CGEvent.tapEnable(tap: tap, enable: true)
            ready.signal()
            CFRunLoopRun()
            self.lifecycleLock.withLock { self.runLoop = nil }
        }
        thread.name = "AgentTrainer.InputCapture"
        thread.qualityOfService = .userInteractive
        self.thread = thread
        thread.start()
        guard ready.wait(timeout: .now() + 1) == .success else {
            stop()
            throw AgentTrainerError.capture("The input monitor did not become ready in time. Try starting again.")
        }
    }

    func stop() {
        let worker = thread
        let loop = lifecycleLock.withLock { runLoop }
        if let tap { CGEvent.tapEnable(tap: tap, enable: false) }
        if let source { CFRunLoopSourceInvalidate(source) }
        if let loop { CFRunLoopStop(loop); CFRunLoopWakeUp(loop) }
        tap = nil
        source = nil
        if let worker, Thread.current !== worker { _ = threadExit.wait(timeout: .now() + 1) }
        thread = nil
        stateLock.lock()
        state = .empty; lastReportedState = .empty; lastStateReportTime = 0
        stateLock.unlock()
        filterLock.withLock { hotkeyFilter.reset(); recordingKeyFilter.reset() }
        onState?(.empty)
    }

    private static let callback: CGEventTapCallBack = { _, type, event, userInfo in
        guard let userInfo else { return Unmanaged.passUnretained(event) }
        let service = Unmanaged<InputCaptureService>.fromOpaque(userInfo).takeUnretainedValue()
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            if let tap = service.tap { CGEvent.tapEnable(tap: tap, enable: true) }
            return Unmanaged.passUnretained(event)
        }
        if event.getIntegerValueField(.eventSourceUserData) == agentTrainerSyntheticTag {
            return Unmanaged.passUnretained(event)
        }
        service.consume(type: type, event: event)
        return Unmanaged.passUnretained(event)
    }

    private func consume(type: CGEventType, event: CGEvent) {
        let timestamp = event.timestamp
        let location = event.location
        let modifiers = event.flags.rawValue
        var sample: InputSample?

        switch type {
        case .mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged:
            sample = InputSample(timestampNanos: timestamp, kind: .mouseMove, x: location.x, y: location.y, deltaX: Double(event.getIntegerValueField(.mouseEventDeltaX)), deltaY: Double(event.getIntegerValueField(.mouseEventDeltaY)), modifiers: modifiers)
        case .leftMouseDown, .leftMouseUp:
            sample = InputSample(timestampNanos: timestamp, kind: .mouseButton, x: location.x, y: location.y, button: 0, modifiers: modifiers, isDown: type == .leftMouseDown)
        case .rightMouseDown, .rightMouseUp:
            sample = InputSample(timestampNanos: timestamp, kind: .mouseButton, x: location.x, y: location.y, button: 1, modifiers: modifiers, isDown: type == .rightMouseDown)
        case .otherMouseDown, .otherMouseUp:
            sample = InputSample(timestampNanos: timestamp, kind: .mouseButton, x: location.x, y: location.y, button: UInt8(clamping: event.getIntegerValueField(.mouseEventButtonNumber)), modifiers: modifiers, isDown: type == .otherMouseDown)
        case .scrollWheel:
            sample = InputSample(timestampNanos: timestamp, kind: .scroll, x: location.x, y: location.y, scrollX: event.getDoubleValueField(.scrollWheelEventPointDeltaAxis2), scrollY: event.getDoubleValueField(.scrollWheelEventPointDeltaAxis1), modifiers: modifiers)
        case .keyDown, .keyUp:
            // A held key already remains in the captured state. Discarding the
            // OS-generated repeat keyDowns keeps long recordings smaller and
            // avoids overweighting one key without losing the hold itself.
            if type == .keyDown, event.getIntegerValueField(.keyboardEventAutorepeat) != 0 { return }
            sample = InputSample(timestampNanos: timestamp, kind: .key, keyCode: UInt16(clamping: event.getIntegerValueField(.keyboardEventKeycode)), modifiers: modifiers, isDown: type == .keyDown)
        case .flagsChanged:
            sample = InputSample(timestampNanos: timestamp, kind: .flags, keyCode: UInt16(clamping: event.getIntegerValueField(.keyboardEventKeycode)), modifiers: modifiers)
        default:
            break
        }

        guard let sample else { return }
        let globallySuppressed = HotkeySuppression.shared.suppresses(sample)
        let delivered = filterLock.withLock {
            hotkeyFilter.process(sample, globallySuppressed: globallySuppressed).compactMap { recordingKeyFilter.process($0) }
        }
        for sample in delivered {
            onSample?(sample)
            if onState != nil { updateState(with: sample) }
        }
    }

    private func updateState(with sample: InputSample) {
        stateLock.lock()
        switch sample.kind {
        case .key:
            if sample.isDown { state.keys.insert(sample.keyCode) } else { state.keys.remove(sample.keyCode) }
        case .mouseButton:
            if sample.isDown { state.buttons.insert(sample.button) } else { state.buttons.remove(sample.button) }
        case .mouseMove:
            state.mouseDelta = CGSize(width: sample.deltaX, height: sample.deltaY)
        case .scroll:
            state.scrollDelta = CGSize(width: sample.scrollX, height: sample.scrollY)
        case .flags:
            let mappings: [([UInt16], UInt64)] = [([56, 60], CGEventFlags.maskShift.rawValue), ([59, 62], CGEventFlags.maskControl.rawValue), ([58, 61], CGEventFlags.maskAlternate.rawValue), ([55, 54], CGEventFlags.maskCommand.rawValue)]
            for (keys, mask) in mappings {
                if sample.modifiers & mask != 0 { state.keys.insert(keys[0]) }
                else { state.keys.subtract(keys) }
            }
        }
        state.modifiers = sample.modifiers
        let snapshot = state
        let now = CACurrentMediaTime()
        let controlsChanged = snapshot.keys != lastReportedState.keys || snapshot.buttons != lastReportedState.buttons || snapshot.modifiers != lastReportedState.modifiers
        let shouldReport = controlsChanged || now - lastStateReportTime >= 1.0 / 30.0
        if shouldReport { lastStateReportTime = now; lastReportedState = snapshot }
        stateLock.unlock()
        if shouldReport { onState?(snapshot) }
    }
}

/// Removes user-selected keys from recording data. Modifier flags are also
/// sanitized when either physical side of that modifier is excluded, so an
/// excluded Command/Shift/Option/Control key cannot leak through other samples.
struct RecordingKeyFilter: Sendable {
    var excludedKeyCodes: Set<UInt16> = []

    mutating func process(_ input: InputSample) -> InputSample? {
        if input.kind == .key, excludedKeyCodes.contains(input.keyCode) { return nil }
        var sample = input
        let pairs: [(Set<UInt16>, UInt64)] = [
            ([56, 60], CGEventFlags.maskShift.rawValue),
            ([59, 62], CGEventFlags.maskControl.rawValue),
            ([58, 61], CGEventFlags.maskAlternate.rawValue),
            ([55, 54], CGEventFlags.maskCommand.rawValue)
        ]
        for (keys, flag) in pairs where !excludedKeyCodes.isDisjoint(with: keys) {
            sample.modifiers &= ~flag
        }
        return sample
    }

    mutating func reset() {}
}

/// Buffers modifier transitions until the following event reveals whether they
/// belong to an AgentTrainer shortcut. Normal modifier chords retain their
/// original timestamps and ordering; configured shortcuts are removed entirely.
struct HotkeyInputFilter: Sendable {
    var bindings: [HotkeyBinding] = []
    private var pendingFlags: [InputSample] = []
    private var suppressedKeyCodes: Set<UInt16> = []
    private var suppressingModifierRelease = false

    init(bindings: [HotkeyBinding] = []) { self.bindings = bindings }

    mutating func process(_ sample: InputSample, globallySuppressed: Bool = false) -> [InputSample] {
        if globallySuppressed, sample.kind == .key || sample.kind == .flags {
            pendingFlags.removeAll(keepingCapacity: true)
            return []
        }

        switch sample.kind {
        case .flags:
            if suppressingModifierRelease || !suppressedKeyCodes.isEmpty {
                if sample.modifiers & HotkeyBinding.cgModifierMask == 0 { suppressedKeyCodes.removeAll(); suppressingModifierRelease = false }
                return []
            }
            pendingFlags.append(sample)
            if sample.modifiers & HotkeyBinding.cgModifierMask == 0 { return drainPendingFlags() }
            return []

        case .key:
            if let binding = bindings.first(where: { $0.matches(sample) }) {
                pendingFlags.removeAll(keepingCapacity: true)
                suppressingModifierRelease = binding.cgEventModifiers != 0
                if sample.isDown { suppressedKeyCodes.insert(sample.keyCode) } else { suppressedKeyCodes.remove(sample.keyCode) }
                return []
            }
            if suppressedKeyCodes.contains(sample.keyCode) {
                if !sample.isDown { suppressedKeyCodes.remove(sample.keyCode) }
                return []
            }
            return drainPendingFlags() + [sample]

        default:
            return drainPendingFlags() + [sample]
        }
    }

    mutating func reset() {
        pendingFlags.removeAll(keepingCapacity: false)
        suppressedKeyCodes.removeAll(keepingCapacity: false)
        suppressingModifierRelease = false
    }

    private mutating func drainPendingFlags() -> [InputSample] {
        let result = pendingFlags
        pendingFlags.removeAll(keepingCapacity: true)
        return result
    }
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T { lock(); defer { unlock() }; return try body() }
}
