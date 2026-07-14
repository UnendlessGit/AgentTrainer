import AppKit
import CoreGraphics
import Foundation

final class InputInjector: @unchecked Sendable {
    var onState: (@Sendable (InputState) -> Void)?

    private let eventSink: @Sendable (CGEvent) -> Void
    private let cursorWarp: @Sendable (CGPoint) -> Void
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

    init(
        eventSink: @escaping @Sendable (CGEvent) -> Void = { $0.post(tap: .cghidEventTap) },
        cursorWarp: @escaping @Sendable (CGPoint) -> Void = { point in _ = CGWarpMouseCursorPosition(point) }
    ) {
        self.eventSink = eventSink
        self.cursorWarp = cursorWarp
    }

    func enable(outputPermissions: RuntimeOutputPermissions = RuntimeOutputPermissions()) {
        lock.lock(); heldKeys.removeAll(); heldButtons.removeAll(); modifiers = 0
        self.outputPermissions = outputPermissions
        cursor = CGEvent(source: hidEventSource)?.location ?? cursor
        // Display discovery crosses into WindowServer. Cache it once per run
        // instead of repeating the query at the action rate.
        fullMacRect = Self.combinedScreenRect()
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
        }
        lock.unlock()

        guard wasEnabled else { return }
        for key in keysToRelease { postKey(key, down: false, flags: []) }
        if resetGameCamera { postNeutralGameCameraMove(at: position) }
        onState?(state)
    }

    func execute(_ prediction: [Float], profile: AIProfile, allowedKeyCodes: Set<UInt16>, mouseMode: MouseControlMode, captureRect: CGRect, safety: AgentSafetyPolicy, gameCamera: GameCameraSettings = GameCameraSettings(), predictionIsFresh: Bool = true) {
        guard prediction.count >= ActionLayout.count else { return }
        lock.lock()
        defer { lock.unlock() }
        guard enabled else { return }
        let channels = profile.channels
        let restrictions = profile.effectiveRestrictions
        let allowed = safety.allowFullMac ? fullMacRect : (safety.controlRegion?.cgRect ?? captureRect)
        var mouseDelta = CGSize.zero
        var scrollDelta = CGSize.zero

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
            cursor = CGPoint(x: captureRect.minX + CGFloat(prediction[0]) * captureRect.width, y: captureRect.minY + CGFloat(prediction[1]) * captureRect.height).clamped(to: allowed)
            postMove(to: cursor)
        }

        if channels.buttons {
            let desired = Set((0..<8).compactMap { prediction[4 + $0] >= 0.5 && restrictions.allowsButton(UInt8($0)) ? UInt8($0) : nil })
            for button in heldButtons.subtracting(desired) { postButton(button, down: false, at: cursor) }
            for button in desired.subtracting(heldButtons) { postButton(button, down: true, at: cursor) }
            heldButtons = desired
        }

        if predictionIsFresh, channels.scroll {
            let sx = CGFloat(prediction[12]) * 20
            let sy = CGFloat(prediction[13]) * 20
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
        if outputPermissions.keyboard, channels.modifiers {
            for i in 0..<4 where prediction[142 + i] >= 0.5 && restrictions.allowsModifier(i) && allowsDemonstratedModifier(i, keys: allowedKeyCodes) {
                desiredModifiers |= modifierMasks[i].rawValue
            }
        }
        let desiredKeys = outputPermissions.keyboard && channels.keyboard ? Set<UInt16>((0..<128).compactMap {
            let code = UInt16($0)
            return prediction[14 + $0] >= 0.5 && allowedKeyCodes.contains(code) && restrictions.allowsKey(code) ? code : nil
        }) : []
        var desiredWithModifiers = desiredKeys
        for i in 0..<4 where desiredModifiers & modifierMasks[i].rawValue != 0 { desiredWithModifiers.insert(modifierKeys[i]) }
        for key in heldKeys.subtracting(desiredWithModifiers) { postKey(key, down: false, flags: CGEventFlags(rawValue: desiredModifiers)) }
        for key in desiredWithModifiers.subtracting(heldKeys) { postKey(key, down: true, flags: CGEventFlags(rawValue: desiredModifiers)) }
        heldKeys = desiredWithModifiers
        modifiers = desiredModifiers
        let state = InputState(keys: heldKeys, buttons: heldButtons, modifiers: modifiers, mouseDelta: mouseDelta, scrollDelta: scrollDelta)
        let now = CACurrentMediaTime()
        let controlsChanged = state.keys != lastReportedState.keys || state.buttons != lastReportedState.buttons || state.modifiers != lastReportedState.modifiers
        if controlsChanged || now - lastStateReportTime >= 1.0 / 30.0 {
            lastStateReportTime = now; lastReportedState = state; onState?(state)
        }
    }

    func disableAndReleaseAll() {
        lock.lock()
        let keys = heldKeys, buttons = heldButtons, position = cursor, shouldResetGameCamera = usedGameCamera
        enabled = false; heldKeys.removeAll(); heldButtons.removeAll(); modifiers = 0; lastReportedState = .empty
        usedGameCamera = false
        lock.unlock()
        for button in buttons { postButton(button, down: false, at: position) }
        for key in keys { postKey(key, down: false, flags: []) }
        // Return the HID cursor path to a neutral, associated state. Some games
        // retain the final relative delta unless they receive a zero-delta move.
        if shouldResetGameCamera { postNeutralGameCameraMove(at: position) }
        onState?(.empty)
    }

    func releaseAll() { disableAndReleaseAll() }

    private func postMove(to point: CGPoint) {
        let movement = movementEvent()
        guard let event = CGEvent(mouseEventSource: hidEventSource, mouseType: movement.type, mouseCursorPosition: point, mouseButton: movement.button) else { return }
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
        post(event)
    }

    private func postNeutralGameCameraMove(at position: CGPoint) {
        CGAssociateMouseAndMouseCursorPosition(1)
        let neutralPoint = CGEvent(source: hidEventSource)?.location ?? position
        guard let neutral = CGEvent(mouseEventSource: hidEventSource, mouseType: .mouseMoved, mouseCursorPosition: neutralPoint, mouseButton: .left) else { return }
        neutral.setIntegerValueField(.mouseEventDeltaX, value: 0)
        neutral.setIntegerValueField(.mouseEventDeltaY, value: 0)
        post(neutral)
    }

    private func postKey(_ code: UInt16, down: Bool, flags: CGEventFlags) {
        guard let event = CGEvent(keyboardEventSource: hidEventSource, virtualKey: code, keyDown: down) else { return }
        event.flags = flags
        post(event)
    }

    private func post(_ event: CGEvent) {
        event.setIntegerValueField(.eventSourceUserData, value: agentTrainerSyntheticTag)
        eventSink(event)
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
