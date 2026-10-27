import AppKit
import CoreGraphics
import Foundation

final class InputInjector: @unchecked Sendable {
    private let cursorLocation: @Sendable () -> CGPoint
    private var lastButtonPosition = CGPoint.zero
    var onState: (@Sendable (InputState) -> Void)?

    private let eventSink: @Sendable (CGEvent) -> Void
    private let cursorWarp: @Sendable (CGPoint) -> Void
    private let associateCursor: @Sendable () -> Void
    private let monotonicClock: @Sendable () -> UInt64
    private let hidEventSource = CGEventSource(stateID: .hidSystemState)
    private let lock = NSLock()
    private var heldKeys: Set<UInt16> = []
    private var heldButtons: Set<UInt8> = []
    private var modifiers: UInt64 = 0
    private var cursor = CGEvent(source: nil)?.location ?? .zero
    private var enabled = false
    private var outputPermissions = RuntimeOutputPermissions()
    private var usedGameCamera = false
    private var lastStateReportTime = 0.0
    private var lastReportedState = InputState.empty
    private var fullMacRect = CGRect.zero
    private var firstExecutionPost: UInt64?
    private var executionReceipt: UInt64 = 0
    var lastExecutionTimestamp: UInt64 { lock.withLock { executionReceipt } }
    private var decisionTimeline = CausalControlTimeline(origin: DispatchTime.now().uptimeNanoseconds)
    private var perceptionTimeline = CausalControlTimeline(origin: DispatchTime.now().uptimeNanoseconds)
    private var lastPerceptionQuery: UInt64?
    private var perceptionInterval: Double?

    struct TemporalSnapshot: Sendable {
        let timestamp: UInt64
        let features: [Float]
        let controls: [Float]
    }

    /// Queries are serialized with posting and permission-driven releases.
    /// Separate interval integrators keep action and perception rates independent.
    func temporalSnapshot(frameTime: UInt64, rect: CGRect, perception: Bool = false) -> TemporalSnapshot {
        lock.lock()
        defer { lock.unlock() }
        let now = monotonicClock()
        if perception {
            perceptionInterval = lastPerceptionQuery.map { TemporalInputContract.seconds(now, since: $0) }
            lastPerceptionQuery = now
            let features = perceptionTimeline.features(at: now, frameTime: frameTime,
                perceptionInterval: perceptionInterval, rect: rect)
            let controls = perceptionTimeline.controlValues(rect: rect)
            perceptionTimeline.endInterval(at: now)
            return TemporalSnapshot(timestamp: now, features: features, controls: controls)
        }
        let features = decisionTimeline.features(at: now, frameTime: frameTime,
            perceptionInterval: perceptionInterval, rect: rect)
        let controls = decisionTimeline.controlValues(rect: rect)
        decisionTimeline.endInterval(at: now)
        return TemporalSnapshot(timestamp: now, features: features, controls: controls)
    }

    init(
        eventSink: @escaping @Sendable (CGEvent) -> Void = { $0.post(tap: .cghidEventTap) },
        cursorWarp: @escaping @Sendable (CGPoint) -> Void = { point in _ = CGWarpMouseCursorPosition(point) },
        monotonicClock: @escaping @Sendable () -> UInt64 = { DispatchTime.now().uptimeNanoseconds },
        associateCursor: @escaping @Sendable () -> Void = { _ = CGAssociateMouseAndMouseCursorPosition(1) },
        cursorLocation: @escaping @Sendable () -> CGPoint = { CGEvent(source: nil)?.location ?? .zero }
    ) {
        self.eventSink = eventSink
        self.cursorWarp = cursorWarp
        self.monotonicClock = monotonicClock
        self.associateCursor = associateCursor
        self.cursorLocation = cursorLocation
        let origin = monotonicClock()
        decisionTimeline = CausalControlTimeline(origin: origin, censorInitialOnsets: false)
        perceptionTimeline = CausalControlTimeline(origin: origin, censorInitialOnsets: false)
    }

    func enable(outputPermissions: RuntimeOutputPermissions = RuntimeOutputPermissions()) {
        lock.lock()
        // A defensive re-enable must not forget controls that are still down.
        // Keep release events ordered with `execute` and the new session.
        let previousKeys = heldKeys
        let previousButtons = heldButtons
        let previousPosition = cursor
        let shouldResetGameCamera = usedGameCamera
        enabled = false
        for button in previousButtons { postButton(button, down: false, at: previousPosition) }
        for key in previousKeys { postKey(key, down: false, flags: []) }
        if shouldResetGameCamera { postNeutralGameCameraMove(at: previousPosition) }
        heldKeys.removeAll(); heldButtons.removeAll(); modifiers = 0
        self.outputPermissions = outputPermissions
        cursor = cursorLocation()
        // Display discovery crosses into WindowServer. Cache it once per run
        // instead of repeating the query at the action rate.
        fullMacRect = Self.combinedScreenRect()
        let origin = monotonicClock()
        decisionTimeline = CausalControlTimeline(origin: origin, censorInitialOnsets: false)
        perceptionTimeline = CausalControlTimeline(origin: origin, censorInitialOnsets: false)
        lastPerceptionQuery = nil; perceptionInterval = nil
        let seed = InputSample(timestampNanos: origin, kind: .mouseMove, x: cursor.x, y: cursor.y)
        decisionTimeline.consume(seed); perceptionTimeline.consume(seed)
        enabled = true; usedGameCamera = false; lastStateReportTime = 0; lastReportedState = .empty; lock.unlock()
        onState?(.empty)
    }

    /// Changes the run-only output firewall under the same lock used by
    /// `execute`, so a late action cannot re-press a key after keyboard output
    /// has been disabled. Any held keys/modifiers are released immediately.
    func updateOutputPermissions(_ permissions: RuntimeOutputPermissions) {
        lock.lock()
        let wasEnabled = enabled
        let keysToRelease = permissions.keyboard ? Set<UInt16>() : heldKeys
        let resetGameCamera = outputPermissions.cursorMovement && !permissions.cursorMovement && usedGameCamera
        let position = cursor
        outputPermissions = permissions
        if !permissions.keyboard {
            heldKeys.removeAll()
            modifiers = 0
        }
        if resetGameCamera { usedGameCamera = false }
        let state = InputState(keys: heldKeys, buttons: heldButtons, modifiers: modifiers)
        if wasEnabled {
            lastStateReportTime = CACurrentMediaTime()
            lastReportedState = state
            // Release while the execution lock is still held. Otherwise a fast
            // off/on toggle can post a new key-down before this old key-up.
            for key in keysToRelease { postKey(key, down: false, flags: []) }
            if resetGameCamera { postNeutralGameCameraMove(at: position) }
        }
        lock.unlock()

        guard wasEnabled else { return }
        onState?(state)
    }

    @discardableResult
    func execute(_ prediction: [Float], profile: AIProfile, allowedKeyCodes: Set<UInt16>, mouseMode: MouseControlMode, captureRect: CGRect, safety: AgentSafetyPolicy, gameCamera: GameCameraSettings = GameCameraSettings(), predictionIsFresh: Bool = true, shiftUsesKeyboardChannel: Bool = false) -> Bool {
        guard prediction.count >= ActionLayout.count,
              prediction.prefix(ActionLayout.count).allSatisfy(\.isFinite) else { return false }
        lock.lock()
        guard enabled else { lock.unlock(); return false }
        firstExecutionPost = nil
        let channels = profile.channels
        let restrictions = profile.effectiveRestrictions
        let requested = safety.allowFullMac ? fullMacRect : (safety.controlRegion?.cgRect ?? captureRect)
        let allowed = requested.isNull || requested.isEmpty || !requested.hasFiniteComponents ? captureRect : requested
        var mouseDelta = CGSize.zero
        var scrollDelta = CGSize.zero
        if !outputPermissions.cursorMovement || !channels.mouseMovement { cursor = cursorLocation() }

        if predictionIsFresh, outputPermissions.cursorMovement, channels.mouseMovement, mouseMode == .relative {
            let dx = GameCameraContract.runtimeDelta(forPrediction: prediction[2], sensitivity: gameCamera.sensitivity)
            let dy = GameCameraContract.runtimeDelta(forPrediction: prediction[3], sensitivity: gameCamera.sensitivity)
            let postedDX = Int64(dx.rounded())
            let postedDY = Int64(dy.rounded())
            // Game-camera mode keeps the system cursor at a stable anchor while
            // emitting raw deltas, so movement never stalls against a screen edge.
            cursor = CGPoint(x: captureRect.midX, y: captureRect.midY).clamped(to: allowed)
            if postedDX != 0 || postedDY != 0 {
                mouseDelta = CGSize(width: CGFloat(postedDX), height: CGFloat(postedDY))
                postRelativeMove(to: cursor, dx: postedDX, dy: postedDY, recenter: gameCamera.recenterCursor)
                usedGameCamera = true
            }
        } else if predictionIsFresh, outputPermissions.cursorMovement, channels.mouseMovement {
            let previousCursor = cursor
            let normalizedX = min(1, max(0, prediction[0]))
            let normalizedY = min(1, max(0, prediction[1]))
            cursor = CGPoint(x: captureRect.minX + CGFloat(normalizedX) * captureRect.width, y: captureRect.minY + CGFloat(normalizedY) * captureRect.height).clamped(to: allowed)
            postMove(to: cursor, from: previousCursor)
        }

        if channels.buttons {
            let inRegion = allowed.contains(cursor)
            let desired = Set((0..<8).compactMap { inRegion && prediction[4 + $0] >= 0.5 && restrictions.allowsButton(UInt8($0)) ? UInt8($0) : nil })
            for button in heldButtons.subtracting(desired) { postButton(button, down: false, at: inRegion ? cursor : lastButtonPosition) }
            for button in desired.subtracting(heldButtons) { postButton(button, down: true, at: cursor) }
            if !desired.isEmpty { lastButtonPosition = cursor }
            heldButtons = desired
        }

        if predictionIsFresh, channels.scroll, allowed.contains(cursor) {
            let sx = CGFloat(min(1, max(-1, prediction[12]))) * 20
            let sy = CGFloat(min(1, max(-1, prediction[13]))) * 20
            let postedX = Int32(sx.rounded())
            let postedY = Int32(sy.rounded())
            if postedX != 0 || postedY != 0 {
                scrollDelta = CGSize(width: CGFloat(postedX), height: CGFloat(postedY))
                postScroll(dx: postedX, dy: postedY)
            }
        }

        var desiredModifiers: UInt64 = 0
        let modifierMasks: [CGEventFlags] = [.maskShift, .maskControl, .maskAlternate, .maskCommand]
        let modifierKeys: [UInt16] = [56, 59, 58, 55]
        if outputPermissions.keyboard {
            for i in 0..<4 where (i == 0 && shiftUsesKeyboardChannel ? channels.keyboard : channels.modifiers)
                && prediction[142 + i] >= 0.5
                && restrictions.allowsModifier(i)
                && allowsDemonstratedModifier(i, keys: allowedKeyCodes) {
                desiredModifiers |= modifierMasks[i].rawValue
            }
        }
        let desiredKeys = outputPermissions.keyboard && channels.keyboard ? Set<UInt16>((0..<128).compactMap {
            let code = UInt16($0)
            return prediction[14 + $0] >= 0.5
                && !ActionLayout.allModifierKeyCodes.contains(code)
                && allowedKeyCodes.contains(code)
                && restrictions.allowsKey(code) ? code : nil
        }) : []
        if desiredKeys.contains(63) { desiredModifiers |= CGEventFlags.maskSecondaryFn.rawValue }
        var desiredWithModifiers = desiredKeys
        for i in 0..<4 where desiredModifiers & modifierMasks[i].rawValue != 0 { desiredWithModifiers.insert(modifierKeys[i]) }
        for key in heldKeys.subtracting(desiredWithModifiers) { postKey(key, down: false, flags: CGEventFlags(rawValue: desiredModifiers)) }
        for key in desiredWithModifiers.subtracting(heldKeys) { postKey(key, down: true, flags: CGEventFlags(rawValue: desiredModifiers)) }
        heldKeys = desiredWithModifiers
        modifiers = desiredModifiers
        let state = InputState(keys: heldKeys, buttons: heldButtons, modifiers: modifiers, mouseDelta: mouseDelta, scrollDelta: scrollDelta)
        let now = CACurrentMediaTime()
        let controlsChanged = state.keys != lastReportedState.keys || state.buttons != lastReportedState.buttons || state.modifiers != lastReportedState.modifiers
        let shouldReport = controlsChanged || now - lastStateReportTime >= 1.0 / 30.0
        if shouldReport { lastStateReportTime = now; lastReportedState = state }
        executionReceipt = firstExecutionPost ?? monotonicClock()
        lock.unlock()
        if shouldReport { onState?(state) }
        return true
    }

    func tapControllerKey(_ code: UInt16) {
        lock.lock()
        // Controller keys are never sampled from the policy. Serialize their
        // complete lifecycle with release and panic operations.
        let flags: CGEventFlags = switch code {
        case 56, 60: .maskShift
        case 59, 62: .maskControl
        case 58, 61: .maskAlternate
        case 55, 54: .maskCommand
        case 57: .maskAlphaShift
        case 63: .maskSecondaryFn
        default: []
        }
        postKey(code, down: true, flags: flags)
        heldKeys.insert(code)
        Thread.sleep(forTimeInterval: 0.04)
        postKey(code, down: false, flags: [])
        heldKeys.remove(code)
        lock.unlock()
    }

    func disableAndReleaseAll() {
        lock.lock()
        let keys = heldKeys, buttons = heldButtons, position = cursor, shouldResetGameCamera = usedGameCamera
        enabled = false
        // Do not allow a new session to start until every release has actually
        // been posted. This also closes the stop/restart counterpart of the
        // live-permission race handled above.
        for button in buttons { postButton(button, down: false, at: position) }
        for key in keys { postKey(key, down: false, flags: []) }
        // Return the HID cursor path to a neutral, associated state. Some games
        // retain the final relative delta unless they receive a zero-delta move.
        if shouldResetGameCamera { postNeutralGameCameraMove(at: position) }
        heldKeys.removeAll(); heldButtons.removeAll(); modifiers = 0; lastReportedState = .empty
        usedGameCamera = false
        lock.unlock()
        onState?(.empty)
    }

    func releaseAll() { disableAndReleaseAll() }

    private func postMove(to point: CGPoint, from previous: CGPoint) {
        let movement = movementEvent()
        guard let event = CGEvent(mouseEventSource: hidEventSource, mouseType: movement.type, mouseCursorPosition: point, mouseButton: movement.button) else { return }
        event.setIntegerValueField(.mouseEventDeltaX, value: Int64((point.x - previous.x).rounded()))
        event.setIntegerValueField(.mouseEventDeltaY, value: Int64((point.y - previous.y).rounded()))
        post(event)
    }

    private func postRelativeMove(to point: CGPoint, dx: Int64, dy: Int64, recenter: Bool) {
        guard dx != 0 || dy != 0 else { return }
        if recenter { cursorWarp(point) }
        // Locked-camera games consume raw HID movement, not UI drag semantics.
        // Keep this as mouseMoved even while a predicted button is held; button
        // down/up events are posted independently below.
        guard let event = CGEvent(mouseEventSource: hidEventSource, mouseType: .mouseMoved, mouseCursorPosition: point, mouseButton: .left) else { return }
        event.setIntegerValueField(.mouseEventDeltaX, value: dx)
        event.setIntegerValueField(.mouseEventDeltaY, value: dy)
        post(event)
        if recenter { cursorWarp(point) }
    }

    private func postButton(_ raw: UInt8, down: Bool, at point: CGPoint) {
        let button = CGMouseButton(rawValue: UInt32(raw)) ?? .left
        let type: CGEventType = switch (raw, down) {
        case (0, true): .leftMouseDown
        case (0, false): .leftMouseUp
        case (1, true): .rightMouseDown
        case (1, false): .rightMouseUp
        case (_, true): .otherMouseDown
        default: .otherMouseUp
        }
        guard let event = CGEvent(mouseEventSource: hidEventSource, mouseType: type, mouseCursorPosition: point, mouseButton: button) else { return }
        event.setIntegerValueField(.mouseEventButtonNumber, value: Int64(raw))
        post(event)
    }

    private func postScroll(dx: Int32, dy: Int32) {
        guard let event = CGEvent(scrollWheelEvent2Source: hidEventSource, units: .pixel, wheelCount: 2, wheel1: dy, wheel2: dx, wheel3: 0) else { return }
        event.location = cursor
        post(event)
    }

    private func postNeutralGameCameraMove(at position: CGPoint) {
        associateCursor()
        let neutralPoint = CGEvent(source: hidEventSource)?.location ?? position
        guard let neutral = CGEvent(mouseEventSource: hidEventSource, mouseType: .mouseMoved, mouseCursorPosition: neutralPoint, mouseButton: .left) else { return }
        neutral.setIntegerValueField(.mouseEventDeltaX, value: 0)
        neutral.setIntegerValueField(.mouseEventDeltaY, value: 0)
        post(neutral)
    }

    private func postKey(_ code: UInt16, down: Bool, flags: CGEventFlags) {
        guard let event = CGEvent(keyboardEventSource: hidEventSource, virtualKey: code, keyDown: down) else { return }
        event.flags = flags
        post(event, logicalSample: InputSample(timestampNanos: 0,
            kind: ActionLayout.allModifierKeyCodes.contains(code) ? .flags : .key,
            keyCode: code, modifiers: flags.rawValue, isDown: down))
    }

    private func post(_ event: CGEvent, logicalSample: InputSample? = nil) {
        event.setIntegerValueField(.eventSourceUserData, value: agentTrainerSyntheticTag)
        // Core Graphics creates flagsChanged for modifier keys. Preserve the
        // explicit flags passed by postKey for those events as well.
        let isKey = logicalSample != nil || event.type == .keyDown || event.type == .keyUp || event.type == .flagsChanged
        if !isKey { event.flags = CGEventFlags(rawValue: modifiers) }
        let time = monotonicClock()
        if firstExecutionPost == nil { firstExecutionPost = time }
        event.timestamp = time
        eventSink(event)
        if var sample = logicalSample {
            sample.timestampNanos = time
            sample.modifiers = event.flags.rawValue
            decisionTimeline.consume(sample); perceptionTimeline.consume(sample)
            return
        }
        let location = event.location
        var sample = InputSample(timestampNanos: time, kind: .mouseMove,
            x: location.x, y: location.y, modifiers: event.flags.rawValue)
        switch event.type {
        case .keyDown, .keyUp:
            sample.kind = .key
            sample.keyCode = UInt16(clamping: event.getIntegerValueField(.keyboardEventKeycode))
            sample.isDown = event.type == .keyDown
            if ActionLayout.allModifierKeyCodes.contains(sample.keyCode) { sample.kind = .flags }
        case .flagsChanged:
            sample.kind = .flags
            sample.keyCode = UInt16(clamping: event.getIntegerValueField(.keyboardEventKeycode))
        case .leftMouseDown, .leftMouseUp, .rightMouseDown, .rightMouseUp, .otherMouseDown, .otherMouseUp:
            sample.kind = .mouseButton
            sample.button = UInt8(clamping: event.getIntegerValueField(.mouseEventButtonNumber))
            sample.isDown = [.leftMouseDown, .rightMouseDown, .otherMouseDown].contains(event.type)
        case .scrollWheel:
            sample.kind = .scroll
            sample.scrollX = Double(event.getIntegerValueField(.scrollWheelEventPointDeltaAxis2))
            sample.scrollY = Double(event.getIntegerValueField(.scrollWheelEventPointDeltaAxis1))
        case .mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged:
            sample.deltaX = Double(event.getIntegerValueField(.mouseEventDeltaX))
            sample.deltaY = Double(event.getIntegerValueField(.mouseEventDeltaY))
        default: return
        }
        decisionTimeline.consume(sample); perceptionTimeline.consume(sample)
    }

    private func movementEvent() -> (type: CGEventType, button: CGMouseButton) {
        if heldButtons.contains(0) { return (.leftMouseDragged, .left) }
        if heldButtons.contains(1) { return (.rightMouseDragged, .right) }
        if let other = heldButtons.sorted().first { return (.otherMouseDragged, CGMouseButton(rawValue: UInt32(other)) ?? .center) }
        return (.mouseMoved, .left)
    }

    private static func combinedScreenRect() -> CGRect {
        var count: UInt32 = 0
        CGGetActiveDisplayList(0, nil, &count)
        var displays = [CGDirectDisplayID](repeating: 0, count: Int(count))
        CGGetActiveDisplayList(count, &displays, &count)
        return displays.prefix(Int(count)).map(CGDisplayBounds).reduce(.null) { $0.union($1) }
    }

    private func allowsDemonstratedModifier(_ index: Int, keys: Set<UInt16>) -> Bool {
        let equivalents: [[UInt16]] = [[56, 60], [59, 62], [58, 61], [55, 54]]
        guard equivalents.indices.contains(index) else { return false }
        return !keys.isDisjoint(with: equivalents[index])
    }
}

private extension CGPoint {
    func clamped(to rect: CGRect) -> CGPoint {
        CGPoint(x: min(rect.maxX, max(rect.minX, x)), y: min(rect.maxY, max(rect.minY, y)))
    }
}

private extension CGRect {
    var hasFiniteComponents: Bool {
        origin.x.isFinite && origin.y.isFinite && width.isFinite && height.isFinite
    }
}
