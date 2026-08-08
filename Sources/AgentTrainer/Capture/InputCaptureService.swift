import AppKit
import CoreGraphics
import Foundation

let agentTrainerSyntheticTag: Int64 = 0x4154474E_54524E52

final class InputCaptureService: @unchecked Sendable {
    var onSample: (@Sendable (InputSample) -> Void)?
    var onState: (@Sendable (InputState, UInt64) -> Void)?

    private final class Session: @unchecked Sendable {
        let tap: CFMachPort
        let source: CFRunLoopSource
        let ready = DispatchSemaphore(value: 0)
        let exit = DispatchGroup()
        weak var thread: Thread?

        private let lock = NSLock()
        private var runLoop: CFRunLoop?
        private var cancelled = false

        init(tap: CFMachPort, source: CFRunLoopSource) {
            self.tap = tap
            self.source = source
            exit.enter()
        }

        var isCancelled: Bool { lock.withLock { cancelled } }

        func prepare(runLoop: CFRunLoop) -> Bool {
            lock.withLock {
                self.runLoop = runLoop
                return !cancelled
            }
        }

        func clearRunLoop() { lock.withLock { runLoop = nil } }

        func cancel() {
            let loop = lock.withLock { () -> CFRunLoop? in
                cancelled = true
                return runLoop
            }
            CGEvent.tapEnable(tap: tap, enable: false)
            CFRunLoopSourceInvalidate(source)
            if let loop { CFRunLoopStop(loop); CFRunLoopWakeUp(loop) }
        }
    }

    private var session: Session?
    private let lifecycleLock = NSLock()
    private let stateLock = NSLock()
    /// Persisted state follows shortcut filtering and is used only to balance
    /// the event file at shutdown. Live state follows physical input immediately
    /// so held modifiers never wait for a later key/release before appearing.
    private var recordedState = InputStateTracker()
    private var liveState = InputStateTracker()
    private var lastLiveStateReportTime = 0.0
    private var lastLiveTransientTime = 0.0
    private var lastReportedLiveState = InputState.empty
    private var liveStateRevision: UInt64 = 0
    private let physicalStateLock = NSLock()
    private var physicalState = PhysicalInputState()
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

    var isRunning: Bool { lifecycleLock.withLock { session != nil } }

    /// Establishes a complete logical state at the first usable screen frame.
    /// Inputs pressed during asynchronous screen-capture startup would otherwise
    /// have only a later release in the file and appear never to have been held.
    func recordingSeedSamples(timestampNanos: UInt64, pointer: CGPoint, excluding shortcut: HotkeyBinding) -> [InputSample] {
        let snapshot = PhysicalInputSnapshot.current()
        let observedAt = DispatchTime.now().uptimeNanoseconds
        physicalStateLock.withLock { physicalState.seed(from: snapshot, timestampNanos: observedAt) }
        let result = filterLock.withLock { () -> [InputSample] in
            let bindings = hotkeyFilter.bindings
            let rawModifiers = snapshot.modifiers
            var modifiers = rawModifiers & ~shortcut.cgEventModifiers
            var result: [InputSample] = []
            if let mouse = recordingKeyFilter.process(InputSample(timestampNanos: timestampNanos, kind: .mouseMove, x: pointer.x, y: pointer.y, modifiers: modifiers)) {
                modifiers = mouse.modifiers
                result.append(mouse)
            }
            if modifiers & HotkeyBinding.cgModifierMask != 0,
               let flags = recordingKeyFilter.process(InputSample(timestampNanos: timestampNanos, kind: .flags, modifiers: modifiers)) {
                result.append(flags)
            }
            for code in snapshot.keys.sorted() {
                let raw = InputSample(timestampNanos: timestampNanos, kind: .key, keyCode: code, modifiers: rawModifiers, isDown: true)
                guard !bindings.contains(where: { $0.matches(raw) }) else { continue }
                var sanitized = raw
                sanitized.modifiers = modifiers
                if let sample = recordingKeyFilter.process(sanitized) { result.append(sample) }
            }
            for button in snapshot.buttons.sorted() {
                let raw = InputSample(timestampNanos: timestampNanos, kind: .mouseButton, x: pointer.x, y: pointer.y, button: button, modifiers: rawModifiers, isDown: true)
                guard !bindings.contains(where: { $0.matches(raw) }) else { continue }
                var sanitized = raw
                sanitized.modifiers = modifiers
                if let sample = recordingKeyFilter.process(sanitized) { result.append(sample) }
            }
            hotkeyFilter.primePersistedModifiers(modifiers)
            return result
        }
        // Seed the same logical state used by the recording HUD and terminal
        // release flush. The event tap may not have completed its first physical
        // reconciliation pass before ScreenCaptureKit produces its first frame.
        // Use the snapshot observation time for in-memory ordering while the
        // returned file samples remain aligned to the first screen frame.
        if onState != nil {
            for sample in result {
                var stateSample = sample
                stateSample.timestampNanos = observedAt
                updateRecordedState(with: stateSample)
                updateLiveState(with: stateSample)
            }
        }
        return result
    }

    func start() throws {
        lifecycleLock.lock()
        guard session == nil else { lifecycleLock.unlock(); return }
        filterLock.withLock { hotkeyFilter.reset(); recordingKeyFilter.reset() }
        let mask: CGEventMask = [
            CGEventType.mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged,
            .leftMouseDown, .leftMouseUp, .rightMouseDown, .rightMouseUp, .otherMouseDown, .otherMouseUp,
            .scrollWheel, .keyDown, .keyUp, .flagsChanged
        ].reduce(0) { $0 | (1 << CGEventMask($1.rawValue)) }

        let pointer = Unmanaged.passUnretained(self).toOpaque()
        guard let tap = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap, options: .listenOnly, eventsOfInterest: mask, callback: Self.callback, userInfo: pointer) else {
            lifecycleLock.unlock()
            throw AgentTrainerError.permission("Input Monitoring permission is required. Enable AgentTrainer in System Settings → Privacy & Security → Input Monitoring, then reopen the app.")
        }
        guard let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0) else {
            CFMachPortInvalidate(tap)
            lifecycleLock.unlock()
            throw AgentTrainerError.capture("The input monitor run-loop source could not be created.")
        }
        let session = Session(tap: tap, source: source)
        self.session = session
        lifecycleLock.unlock()

        let thread = Thread { [weak self, session] in
            defer {
                session.clearRunLoop()
                self?.sessionDidExit(session)
                session.exit.leave()
            }
            guard let loop = CFRunLoopGetCurrent() else { session.ready.signal(); return }
            guard session.prepare(runLoop: loop) else { session.ready.signal(); return }
            CFRunLoopAddSource(loop, session.source, .commonModes)
            let reconciliationTimer = CFRunLoopTimerCreateWithHandler(
                kCFAllocatorDefault,
                CFAbsoluteTimeGetCurrent() + 0.1,
                0.1,
                0,
                0
            ) { [weak self] _ in self?.reconcilePhysicalState() }
            CFRunLoopAddTimer(loop, reconciliationTimer, .commonModes)
            CGEvent.tapEnable(tap: session.tap, enable: true)
            session.ready.signal()
            if !session.isCancelled { CFRunLoopRun() }
            CFRunLoopTimerInvalidate(reconciliationTimer)
            CFRunLoopRemoveTimer(loop, reconciliationTimer, .commonModes)
            CFRunLoopRemoveSource(loop, session.source, .commonModes)
        }
        thread.name = "AgentTrainer.InputCapture"
        thread.qualityOfService = QualityOfService.userInteractive
        session.thread = thread
        thread.start()
        guard session.ready.wait(timeout: .now() + 1) == .success else {
            stop()
            throw AgentTrainerError.capture("The input monitor did not become ready in time. Try starting again.")
        }
        guard lifecycleLock.withLock({ self.session === session }), !session.isCancelled else {
            stop()
            throw CancellationError()
        }
    }

    /// Recording shutdown can request balanced terminal releases. Runtime safety
    /// monitors use the default because stopping a monitor must never masquerade
    /// as new physical human input.
    func stop(flushHeldState: Bool = false) {
        let active = lifecycleLock.withLock { session }
        active?.cancel()
        let calledFromWorker = active?.thread.map { Thread.current === $0 } ?? false
        if let active, !calledFromWorker {
            // Event-tap callbacks are intentionally tiny. Joining here prevents
            // a later recording/run from overlapping a timed-out old run loop
            // or receiving its delayed exit signal.
            active.exit.wait()
            lifecycleLock.withLock {
                if session === active { session = nil }
            }
        }
        if flushHeldState { flushCapturedReleases() }
        stateLock.lock()
        recordedState = InputStateTracker()
        liveState = InputStateTracker()
        lastReportedLiveState = .empty
        lastLiveStateReportTime = 0
        lastLiveTransientTime = 0
        liveStateRevision &+= 1
        let emptyRevision = liveStateRevision
        stateLock.unlock()
        physicalStateLock.withLock { physicalState = PhysicalInputState() }
        filterLock.withLock { hotkeyFilter.reset(); recordingKeyFilter.reset() }
        onState?(.empty, emptyRevision)
    }

    private static let callback: CGEventTapCallBack = { _, type, event, userInfo in
        guard let userInfo else { return Unmanaged.passUnretained(event) }
        let service = Unmanaged<InputCaptureService>.fromOpaque(userInfo).takeUnretainedValue()
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            service.reenableCurrentTap()
            return Unmanaged.passUnretained(event)
        }
        if event.getIntegerValueField(.eventSourceUserData) == agentTrainerSyntheticTag {
            return Unmanaged.passUnretained(event)
        }
        service.consume(type: type, event: event)
        return Unmanaged.passUnretained(event)
    }

    private func reenableCurrentTap() {
        let active = lifecycleLock.withLock { session }
        guard let active, !active.isCancelled else { return }
        CGEvent.tapEnable(tap: active.tap, enable: true)
        // The disabling event replaces the physical transition that timed out.
        // Re-read the HID state immediately so a missed key/button release cannot
        // remain held for the rest of a recording.
        reconcilePhysicalState(scanAllControls: true)
    }

    private func sessionDidExit(_ exited: Session) {
        let endedUnexpectedly = lifecycleLock.withLock { () -> Bool in
            guard session === exited else { return false }
            session = nil
            return !exited.isCancelled
        }
        if endedUnexpectedly {
            AppLog.write(.warning, category: "Input", "Input-monitor run loop exited unexpectedly")
        }
    }

    deinit { stop() }

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
            let continuous = event.getIntegerValueField(.scrollWheelEventIsContinuous) != 0
            let pointX = event.getDoubleValueField(.scrollWheelEventPointDeltaAxis2)
            let pointY = event.getDoubleValueField(.scrollWheelEventPointDeltaAxis1)
            let fixedX = Double(event.getIntegerValueField(.scrollWheelEventFixedPtDeltaAxis2)) / 65_536
            let fixedY = Double(event.getIntegerValueField(.scrollWheelEventFixedPtDeltaAxis1)) / 65_536
            let lineX = Double(event.getIntegerValueField(.scrollWheelEventDeltaAxis2))
            let lineY = Double(event.getIntegerValueField(.scrollWheelEventDeltaAxis1))
            let scrollX = Self.scrollDelta(point: pointX, fixed: fixedX, line: lineX, continuous: continuous)
            let scrollY = Self.scrollDelta(point: pointY, fixed: fixedY, line: lineY, continuous: continuous)
            guard scrollX != 0 || scrollY != 0 else { return }
            sample = InputSample(timestampNanos: timestamp, kind: .scroll, x: location.x, y: location.y, scrollX: scrollX, scrollY: scrollY, modifiers: modifiers)
        case .keyDown, .keyUp:
            // A held key already remains in the captured state. Discarding the
            // OS-generated repeat keyDowns keeps long recordings smaller and
            // avoids overweighting one key without losing the hold itself.
            if type == .keyDown, event.getIntegerValueField(.keyboardEventAutorepeat) != 0 { return }
            sample = InputSample(timestampNanos: timestamp, kind: .key, keyCode: UInt16(clamping: event.getIntegerValueField(.keyboardEventKeycode)), modifiers: modifiers, isDown: type == .keyDown)
        case .flagsChanged:
            let keyCode = UInt16(clamping: event.getIntegerValueField(.keyboardEventKeycode))
            let samples = Self.normalizedFlagsChangedSamples(
                timestampNanos: timestamp,
                keyCode: keyCode,
                modifiers: modifiers,
                isPhysicallyDown: CGEventSource.keyState(.combinedSessionState, key: CGKeyCode(keyCode))
            )
            for flagsSample in samples {
                physicalStateLock.withLock { physicalState.consume(flagsSample) }
                deliver(flagsSample)
            }
            return
        default:
            break
        }

        guard let sample else { return }
        physicalStateLock.withLock { physicalState.consume(sample) }
        deliver(sample)
    }

    private func deliver(_ sample: InputSample) {
        // The HUD represents physical human input, not the delayed persisted
        // shortcut decision. Blacklisted controls remain hidden, but ordinary
        // modifiers and mouse buttons update while they are actually held.
        if onState != nil {
            let liveSample = filterLock.withLock { recordingKeyFilter.process(sample) }
            if let liveSample { updateLiveState(with: liveSample) }
        }

        let delivered: [InputSample]
        switch HotkeySuppression.shared.process(sample) {
        case let .suppress(shortcut):
            delivered = filterLock.withLock {
                hotkeyFilter.processGloballySuppressed(sample, shortcut: shortcut).compactMap { recordingKeyFilter.process($0) }
            }
        case let .pass(sanitized):
            delivered = filterLock.withLock {
                hotkeyFilter.process(sanitized).compactMap { recordingKeyFilter.process($0) }
            }
        }
        for sample in delivered {
            onSample?(sample)
            // Runtime human-input safety does not render or flush logical state.
            // Recording installs onState and tracks filtered file holds.
            if onState != nil { updateRecordedState(with: sample) }
        }
    }

    private func reconcilePhysicalState(scanAllControls: Bool = false) {
        guard isRunning else { return }
        let tracked = physicalStateLock.withLock { (physicalState.keys, physicalState.buttons) }
        let snapshot = PhysicalInputSnapshot.current(
            keyCodes: scanAllControls ? nil : tracked.0,
            buttonNumbers: scanAllControls ? nil : tracked.1
        )
        let now = DispatchTime.now().uptimeNanoseconds
        let samples = physicalStateLock.withLock { physicalState.reconcile(to: snapshot, timestampNanos: now) }
        for sample in samples { deliver(sample) }
        clearLiveTransientStateIfNeeded()
    }

    private func flushCapturedReleases() {
        let releaseSamples = stateLock.withLock { () -> [InputSample] in
            recordedState.terminalReleaseSamples(at: DispatchTime.now().uptimeNanoseconds)
        }
        // These controls already passed shortcut and blacklist filtering. Sending
        // their matching releases directly preserves that exact logical state.
        for sample in releaseSamples {
            onSample?(sample)
            updateRecordedState(with: sample)
        }
    }

    private func updateRecordedState(with sample: InputSample) {
        stateLock.lock()
        _ = recordedState.consume(sample)
        stateLock.unlock()
    }

    private func updateLiveState(with sample: InputSample) {
        stateLock.lock()
        guard liveState.consume(sample) else { stateLock.unlock(); return }
        let snapshot = liveState.state
        let now = CACurrentMediaTime()
        if (sample.kind == .mouseMove && (sample.deltaX != 0 || sample.deltaY != 0)) ||
            (sample.kind == .scroll && (sample.scrollX != 0 || sample.scrollY != 0)) {
            lastLiveTransientTime = now
        }
        let controlsChanged = snapshot.keys != lastReportedLiveState.keys ||
            snapshot.buttons != lastReportedLiveState.buttons ||
            snapshot.modifiers != lastReportedLiveState.modifiers
        let shouldReport = controlsChanged || now - lastLiveStateReportTime >= 1.0 / 30.0
        if shouldReport {
            lastLiveStateReportTime = now
            lastReportedLiveState = snapshot
            liveStateRevision &+= 1
        }
        let revision = liveStateRevision
        stateLock.unlock()
        if shouldReport { onState?(snapshot, revision) }
    }

    private func clearLiveTransientStateIfNeeded() {
        guard onState != nil else { return }
        stateLock.lock()
        let now = CACurrentMediaTime()
        guard lastLiveTransientTime > 0, now - lastLiveTransientTime >= 0.05,
              liveState.clearTransientDeltas() else {
            stateLock.unlock()
            return
        }
        lastLiveTransientTime = 0
        let snapshot = liveState.state
        lastReportedLiveState = snapshot
        lastLiveStateReportTime = now
        liveStateRevision &+= 1
        let revision = liveStateRevision
        stateLock.unlock()
        onState?(snapshot, revision)
    }

    private static func scrollDelta(point: Double, fixed: Double, line: Double, continuous: Bool) -> Double {
        let selected: Double
        if point.isFinite, point != 0 { selected = point }
        else if fixed.isFinite, fixed != 0 { selected = fixed }
        else if line.isFinite { selected = continuous ? line : line * 10 }
        else { selected = 0 }
        return min(10_000, max(-10_000, selected))
    }

    static func normalizedFlagsChangedSamples(
        timestampNanos: UInt64,
        keyCode: UInt16,
        modifiers: UInt64,
        isPhysicallyDown: Bool
    ) -> [InputSample] {
        if keyCode == 57 {
            // Caps Lock reports a toggle transition rather than a normal
            // keyDown/keyUp pair. Persist it as an instantaneous tap so it
            // participates in the ordinary 128-key training/replay layout
            // instead of becoming a held key until the next toggle.
            let releaseTimestamp = timestampNanos == .max ? UInt64.max : timestampNanos + 1
            return [
                InputSample(timestampNanos: timestampNanos, kind: .key, keyCode: keyCode, modifiers: modifiers, isDown: true),
                InputSample(timestampNanos: releaseTimestamp, kind: .key, keyCode: keyCode, modifiers: modifiers, isDown: false)
            ]
        }
        if PhysicalInputSnapshot.modifierKeyCodes.contains(keyCode) {
            return [InputSample(
                timestampNanos: timestampNanos,
                kind: .flags,
                keyCode: keyCode,
                modifiers: modifiers,
                isDown: isPhysicallyDown
            )]
        }
        // Fn/Globe and any future flag-only non-lock key still belong to the
        // normal keyboard action space and retain their physical edges.
        return [InputSample(
            timestampNanos: timestampNanos,
            kind: .key,
            keyCode: keyCode,
            modifiers: modifiers,
            isDown: isPhysicallyDown
        )]
    }
}

/// Deterministic logical reducer shared by the immediate HUD state and the
/// independently filtered event-file state. Keeping this pure makes all key,
/// modifier, button, movement, scroll, stale-order, and shutdown behavior
/// directly testable without installing an event tap.
struct InputStateTracker: Sendable {
    private(set) var state = InputState.empty
    private(set) var pointer = CGPoint.zero
    private(set) var lastTimestamp: UInt64 = 0

    @discardableResult
    mutating func consume(_ sample: InputSample) -> Bool {
        guard sample.timestampNanos >= lastTimestamp else { return false }
        lastTimestamp = sample.timestampNanos
        state.mouseDelta = .zero
        state.scrollDelta = .zero
        switch sample.kind {
        case .key:
            if sample.isDown { state.keys.insert(sample.keyCode) }
            else { state.keys.remove(sample.keyCode) }
        case .mouseButton:
            pointer = CGPoint(x: sample.x, y: sample.y)
            if sample.isDown { state.buttons.insert(sample.button) }
            else { state.buttons.remove(sample.button) }
        case .mouseMove:
            pointer = CGPoint(x: sample.x, y: sample.y)
            state.mouseDelta = CGSize(width: sample.deltaX, height: sample.deltaY)
        case .scroll:
            pointer = CGPoint(x: sample.x, y: sample.y)
            state.scrollDelta = CGSize(width: sample.scrollX, height: sample.scrollY)
        case .flags:
            let mappings: [([UInt16], UInt64)] = [
                ([56, 60], CGEventFlags.maskShift.rawValue),
                ([59, 62], CGEventFlags.maskControl.rawValue),
                ([58, 61], CGEventFlags.maskAlternate.rawValue),
                ([55, 54], CGEventFlags.maskCommand.rawValue)
            ]
            for (keys, mask) in mappings {
                guard sample.modifiers & mask != 0 else {
                    state.keys.subtract(keys)
                    continue
                }
                if keys.contains(sample.keyCode) {
                    if sample.isDown { state.keys.insert(sample.keyCode) }
                    else { state.keys.remove(sample.keyCode) }
                }
                if state.keys.isDisjoint(with: keys) { state.keys.insert(keys[0]) }
            }
        }
        state.modifiers = sample.modifiers
        return true
    }

    @discardableResult
    mutating func clearTransientDeltas() -> Bool {
        guard state.mouseDelta != .zero || state.scrollDelta != .zero else { return false }
        state.mouseDelta = .zero
        state.scrollDelta = .zero
        return true
    }

    func terminalReleaseSamples(at requestedTimestamp: UInt64) -> [InputSample] {
        let modifierKeys: Set<UInt16> = [54, 55, 56, 58, 59, 60, 61, 62]
        let hasHeldModifierKey = !state.keys.isDisjoint(with: modifierKeys)
        guard !state.keys.isEmpty || !state.buttons.isEmpty ||
                state.modifiers & HotkeyBinding.cgModifierMask != 0 else { return [] }
        let nextTimestamp = lastTimestamp == .max ? UInt64.max : lastTimestamp + 1
        let timestamp = max(requestedTimestamp, nextTimestamp)
        let finalModifiers = state.modifiers & ~HotkeyBinding.cgModifierMask
        var samples = state.keys.subtracting(modifierKeys).sorted().map {
            InputSample(timestampNanos: timestamp, kind: .key, keyCode: $0, modifiers: state.modifiers, isDown: false)
        }
        samples += state.buttons.sorted().map {
            InputSample(timestampNanos: timestamp, kind: .mouseButton, x: pointer.x, y: pointer.y, button: $0, modifiers: state.modifiers, isDown: false)
        }
        if hasHeldModifierKey || state.modifiers & HotkeyBinding.cgModifierMask != 0 {
            samples.append(InputSample(timestampNanos: timestamp, kind: .flags, modifiers: finalModifiers))
        }
        return samples
    }
}

struct PhysicalInputSnapshot {
    static let modifierKeyCodes: Set<UInt16> = [54, 55, 56, 58, 59, 60, 61, 62]
    static let toggleKeyCodes: Set<UInt16> = [57]
    var keys: Set<UInt16>
    var buttons: Set<UInt8>
    var modifiers: UInt64
    var pointer: CGPoint

    static func current(keyCodes: Set<UInt16>? = nil, buttonNumbers: Set<UInt8>? = nil) -> Self {
        let candidateKeys = keyCodes ?? Set((0..<128).map(UInt16.init))
        let keys = Set(candidateKeys.compactMap { code -> UInt16? in
            guard !modifierKeyCodes.contains(code), !toggleKeyCodes.contains(code),
                  CGEventSource.keyState(.combinedSessionState, key: CGKeyCode(code)) else { return nil }
            return code
        })
        // NSEvent exposes a 32-bit pressed-button mask and gaming mice commonly
        // number side buttons beyond the first eight. Full scans are only used
        // for startup/tap recovery; normal reconciliation polls tracked buttons.
        let candidateButtons = buttonNumbers ?? Set((0..<32).map(UInt8.init))
        let buttons = Set(candidateButtons.compactMap { raw in
            CGEventSource.buttonState(.combinedSessionState, button: CGMouseButton(rawValue: UInt32(raw)) ?? .left) ? UInt8(raw) : nil
        })
        return Self(
            keys: keys,
            buttons: buttons,
            modifiers: CGEventSource.flagsState(.combinedSessionState).rawValue,
            pointer: CGEvent(source: nil)?.location ?? .zero
        )
    }
}

/// Tracks the physical HID truth independently of recording filters. Polling
/// reconciles only transitions missed by the event tap, so normal high-rate
/// mouse and keyboard capture remains event-driven.
struct PhysicalInputState: Sendable {
    private(set) var keys: Set<UInt16> = []
    private(set) var buttons: Set<UInt8> = []
    private(set) var modifiers: UInt64 = 0
    private var lastTimestamp: UInt64 = 0

    mutating func consume(_ sample: InputSample) {
        guard sample.timestampNanos >= lastTimestamp else { return }
        lastTimestamp = max(lastTimestamp, sample.timestampNanos)
        switch sample.kind {
        case .key:
            if sample.isDown { keys.insert(sample.keyCode) } else { keys.remove(sample.keyCode) }
        case .mouseButton:
            if sample.isDown { buttons.insert(sample.button) } else { buttons.remove(sample.button) }
        case .flags:
            modifiers = sample.modifiers
            synchronizeModifierKeys()
        case .mouseMove, .scroll:
            break
        }
    }

    mutating func seed(from snapshot: PhysicalInputSnapshot, timestampNanos: UInt64) {
        guard timestampNanos >= lastTimestamp else { return }
        keys = snapshot.keys
        buttons = snapshot.buttons
        modifiers = snapshot.modifiers
        lastTimestamp = timestampNanos
        synchronizeModifierKeys()
    }

    mutating func reconcile(to snapshot: PhysicalInputSnapshot, timestampNanos: UInt64) -> [InputSample] {
        let nextTimestamp = lastTimestamp == .max ? UInt64.max : lastTimestamp + 1
        let timestamp = max(timestampNanos, nextTimestamp)
        var samples: [InputSample] = []
        for key in keys.subtracting(snapshot.keys).subtracting(PhysicalInputSnapshot.modifierKeyCodes).sorted() {
            samples.append(InputSample(timestampNanos: timestamp, kind: .key, keyCode: key, modifiers: snapshot.modifiers, isDown: false))
        }
        for button in buttons.subtracting(snapshot.buttons).sorted() {
            samples.append(InputSample(timestampNanos: timestamp, kind: .mouseButton, x: snapshot.pointer.x, y: snapshot.pointer.y, button: button, modifiers: snapshot.modifiers, isDown: false))
        }
        if (modifiers ^ snapshot.modifiers) & HotkeyBinding.cgModifierMask != 0 {
            samples.append(InputSample(timestampNanos: timestamp, kind: .flags, modifiers: snapshot.modifiers))
        }
        for key in snapshot.keys.subtracting(keys).subtracting(PhysicalInputSnapshot.modifierKeyCodes).sorted() {
            samples.append(InputSample(timestampNanos: timestamp, kind: .key, keyCode: key, modifiers: snapshot.modifiers, isDown: true))
        }
        for button in snapshot.buttons.subtracting(buttons).sorted() {
            samples.append(InputSample(timestampNanos: timestamp, kind: .mouseButton, x: snapshot.pointer.x, y: snapshot.pointer.y, button: button, modifiers: snapshot.modifiers, isDown: true))
        }
        keys = snapshot.keys
        buttons = snapshot.buttons
        modifiers = snapshot.modifiers
        synchronizeModifierKeys()
        if !samples.isEmpty { lastTimestamp = timestamp }
        return samples
    }

    private mutating func synchronizeModifierKeys() {
        let mappings: [([UInt16], UInt64)] = [
            ([56, 60], CGEventFlags.maskShift.rawValue),
            ([59, 62], CGEventFlags.maskControl.rawValue),
            ([58, 61], CGEventFlags.maskAlternate.rawValue),
            ([55, 54], CGEventFlags.maskCommand.rawValue)
        ]
        for (codes, mask) in mappings {
            keys.subtract(codes)
            if modifiers & mask != 0 { keys.insert(codes[0]) }
        }
    }
}

/// Removes user-selected keys from recording data. Modifier flags are also
/// sanitized when either physical side of that modifier is excluded, so an
/// excluded Command/Shift/Option/Control key cannot leak through other samples.
struct RecordingKeyFilter: Sendable {
    var excludedKeyCodes: Set<UInt16> = [] {
        didSet { excludedModifierMask = Self.modifierMask(for: excludedKeyCodes) }
    }
    private var excludedModifierMask: UInt64 = 0

    init(excludedKeyCodes: Set<UInt16> = []) {
        self.excludedKeyCodes = excludedKeyCodes
        excludedModifierMask = Self.modifierMask(for: excludedKeyCodes)
    }

    mutating func process(_ input: InputSample) -> InputSample? {
        if input.kind == .key, excludedKeyCodes.contains(input.keyCode) { return nil }
        var sample = input
        sample.modifiers &= ~excludedModifierMask
        return sample
    }

    mutating func reset() {}

    private static func modifierMask(for keys: Set<UInt16>) -> UInt64 {
        var mask: UInt64 = 0
        if !keys.isDisjoint(with: [56, 60]) { mask |= CGEventFlags.maskShift.rawValue }
        if !keys.isDisjoint(with: [59, 62]) { mask |= CGEventFlags.maskControl.rawValue }
        if !keys.isDisjoint(with: [58, 61]) { mask |= CGEventFlags.maskAlternate.rawValue }
        if !keys.isDisjoint(with: [55, 54]) { mask |= CGEventFlags.maskCommand.rawValue }
        return mask
    }
}

/// Buffers modifier transitions until the following event reveals whether they
/// belong to an AgentTrainer shortcut. Normal modifier chords retain their
/// original timestamps and ordering; configured shortcuts are removed entirely.
struct HotkeyInputFilter: Sendable {
    var bindings: [HotkeyBinding] = []
    private var pendingFlags: [InputSample] = []
    private var suppressedKeyCodes: Set<UInt16> = []
    private var suppressedMouseButtons: Set<UInt8> = []
    private var suppressedModifierMask: UInt64 = 0
    private var emittedModifiers: UInt64 = 0

    init(bindings: [HotkeyBinding] = []) { self.bindings = bindings }

    mutating func primePersistedModifiers(_ modifiers: UInt64) {
        pendingFlags.removeAll(keepingCapacity: true)
        emittedModifiers = modifiers & HotkeyBinding.cgModifierMask
    }

    mutating func process(_ sample: InputSample) -> [InputSample] {
        switch sample.kind {
        case .flags:
            if suppressedModifierMask != 0 || !suppressedKeyCodes.isEmpty || !suppressedMouseButtons.isEmpty {
                let activeMask = suppressedModifierMask
                if sample.modifiers & activeMask == 0 {
                    suppressedKeyCodes.removeAll()
                    suppressedMouseButtons.removeAll()
                    suppressedModifierMask = 0
                }
                return emitModifierTransition(sample, removing: activeMask)
            }
            pendingFlags.append(sample)
            if sample.modifiers & HotkeyBinding.cgModifierMask == 0 { return drainPendingFlags() }
            return []

        case .key:
            if let binding = bindings.first(where: { $0.matches(sample) }) {
                let preserved = drainPendingFlags(removing: binding.cgEventModifiers)
                suppressedModifierMask |= binding.cgEventModifiers
                if sample.isDown { suppressedKeyCodes.insert(sample.keyCode) } else { suppressedKeyCodes.remove(sample.keyCode) }
                return preserved
            }
            if suppressedKeyCodes.contains(sample.keyCode) {
                if !sample.isDown { suppressedKeyCodes.remove(sample.keyCode) }
                return []
            }
            return drainPendingFlags() + [sample]

        case .mouseButton:
            if let binding = bindings.first(where: { $0.matches(sample) }) {
                let preserved = drainPendingFlags(removing: binding.cgEventModifiers)
                suppressedModifierMask |= binding.cgEventModifiers
                if sample.isDown { suppressedMouseButtons.insert(sample.button) }
                else { suppressedMouseButtons.remove(sample.button) }
                return preserved
            }
            if suppressedMouseButtons.contains(sample.button) {
                if !sample.isDown { suppressedMouseButtons.remove(sample.button) }
                return []
            }
            return drainPendingFlags() + [sample]

        default:
            return drainPendingFlags() + [sample]
        }
    }

    /// A native global shortcut may activate before the new recording/safety
    /// event tap observes its trailing events. Preserve any already-pending
    /// foreign modifier (for example Shift held while stopping with ⌃⌥⌘R), while
    /// removing only the shortcut-owned trigger and modifier lifecycle.
    mutating func processGloballySuppressed(_ sample: InputSample, shortcut: HotkeyBinding) -> [InputSample] {
        let preserved = drainPendingFlags(removing: shortcut.cgEventModifiers)
        suppressedModifierMask |= shortcut.cgEventModifiers
        switch sample.kind {
        case .key:
            if sample.isDown { suppressedKeyCodes.insert(sample.keyCode) }
            else { suppressedKeyCodes.remove(sample.keyCode) }
        case .mouseButton:
            if sample.isDown { suppressedMouseButtons.insert(sample.button) }
            else { suppressedMouseButtons.remove(sample.button) }
        case .flags:
            let activeMask = suppressedModifierMask
            if sample.modifiers & activeMask == 0 {
                suppressedKeyCodes.removeAll()
                suppressedMouseButtons.removeAll()
                suppressedModifierMask = 0
            }
            return preserved + emitModifierTransition(sample, removing: activeMask)
        case .mouseMove, .scroll:
            break
        }
        return preserved
    }

    mutating func reset() {
        pendingFlags.removeAll(keepingCapacity: false)
        suppressedKeyCodes.removeAll(keepingCapacity: false)
        suppressedMouseButtons.removeAll(keepingCapacity: false)
        suppressedModifierMask = 0
        emittedModifiers = 0
    }

    private mutating func drainPendingFlags(removing modifierMask: UInt64 = 0) -> [InputSample] {
        var result: [InputSample] = []
        result.reserveCapacity(pendingFlags.count)
        for sample in pendingFlags {
            result += emitModifierTransition(sample, removing: modifierMask)
        }
        pendingFlags.removeAll(keepingCapacity: true)
        return result
    }

    private mutating func emitModifierTransition(_ input: InputSample, removing modifierMask: UInt64) -> [InputSample] {
        var sample = input
        sample.modifiers &= ~modifierMask
        let semanticModifiers = sample.modifiers & HotkeyBinding.cgModifierMask
        guard semanticModifiers != emittedModifiers else { return [] }
        emittedModifiers = semanticModifiers
        return [sample]
    }
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T { lock(); defer { unlock() }; return try body() }
}
