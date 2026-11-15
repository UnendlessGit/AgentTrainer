import ATCapture
import ATCore
import CoreGraphics
import Foundation

/// What a running model is allowed to do. Independent of the model's training: blocking an
/// action here never changes weights or configuration, it just isn't executed.
public struct ActionPermissions: Codable, Hashable, Sendable {
    public var keyboard = true
    /// Letters, digits and symbols (typing). Movement/modifier keys are governed by `keyboard`.
    public var typingKeys = true
    public var mouseMovement = true
    public var mouseButtons = true
    public var scroll = true
    /// Keys never pressed (Command is blocked by default so the model can't quit or switch apps).
    public var blockedKeys: Set<String> = ["command", "rcommand"]
    public init() {}

    public func allows(key id: String) -> Bool {
        guard keyboard, !blockedKeys.contains(id), let info = KeyCatalog.info(id) else { return false }
        if [.letter, .digit, .symbol].contains(info.category) { return typingKeys }
        return true
    }
}

public enum MouseOutputMode: String, Codable, CaseIterable, Hashable, Sendable {
    /// Moves the pointer and reports the delta (apps, games with a visible cursor).
    case pointer
    /// Only reports deltas at the current location (games that lock the cursor for camera control).
    case camera
    public var label: String { self == .pointer ? "Move the pointer" : "Camera (locked cursor)" }
}

/// Turns model actions into real keyboard / mouse events (needs Accessibility permission).
public final class ActionExecutor: @unchecked Sendable {
    public private(set) var heldKeys = Set<String>()
    public private(set) var heldButtons = Set<Int>()
    private let source = CGEventSource(stateID: .hidSystemState)
    public var mouseMode = MouseOutputMode.pointer
    public var sensitivity = 1.0
    private var fractional = CGVector.zero

    public init() {}

    private func post(_ e: CGEvent?) {
        guard let e else { return }
        e.setIntegerValueField(.eventSourceUserData, value: SyntheticInput.userData)
        e.post(tap: .cghidEventTap)
    }

    private var modifierFlags: CGEventFlags {
        var f = CGEventFlags()
        for k in heldKeys {
            switch k {
            case "shift", "rshift": f.insert(.maskShift)
            case "control", "rcontrol": f.insert(.maskControl)
            case "option", "roption": f.insert(.maskAlternate)
            case "command", "rcommand": f.insert(.maskCommand)
            case "fn": f.insert(.maskSecondaryFn)
            default: break
            }
        }
        return f
    }

    static let modifierKeys: Set<String> = ["shift", "rshift", "control", "rcontrol", "option", "roption", "command", "rcommand", "fn", "capslock"]

    /// Executes one timestep. `area` is the controlled region in global points (for absolute pointer mode).
    public func apply(_ a: TimestepAction, absolute: Bool, area: CGRect, permissions p: ActionPermissions) {
        // Keys: diff against what's held.
        let wanted = Set(a.keys.filter { p.allows(key: $0) })
        for k in heldKeys.subtracting(wanted).sorted(by: { !Self.modifierKeys.contains($0) && Self.modifierKeys.contains($1) }) { key(k, down: false) }
        for k in wanted.subtracting(heldKeys).sorted(by: { Self.modifierKeys.contains($0) && !Self.modifierKeys.contains($1) }) { key(k, down: true) }

        // Pointer position.
        let current = CGEvent(source: nil)?.location ?? .zero
        var target = current
        var delta = CGVector.zero
        if p.mouseMovement {
            if absolute {
                target = CGPoint(x: area.minX + a.absX * area.width, y: area.minY + a.absY * area.height)
                delta = CGVector(dx: target.x - current.x, dy: target.y - current.y)
            } else {
                let dx = a.dx * sensitivity + fractional.dx, dy = a.dy * sensitivity + fractional.dy
                let ix = dx.rounded(.towardZero), iy = dy.rounded(.towardZero)
                fractional = CGVector(dx: dx - ix, dy: dy - iy)
                delta = CGVector(dx: ix, dy: iy)
                if mouseMode == .pointer { target = CGPoint(x: current.x + ix, y: current.y + iy) }
            }
        }
        if delta.dx != 0 || delta.dy != 0 {
            let type: CGEventType = heldButtons.contains(0) ? .leftMouseDragged : heldButtons.contains(1) ? .rightMouseDragged : heldButtons.isEmpty ? .mouseMoved : .otherMouseDragged
            let e = CGEvent(mouseEventSource: source, mouseType: type, mouseCursorPosition: target, mouseButton: .left)
            e?.setIntegerValueField(.mouseEventDeltaX, value: Int64(delta.dx))
            e?.setIntegerValueField(.mouseEventDeltaY, value: Int64(delta.dy))
            e?.setDoubleValueField(.mouseEventDeltaX, value: Double(delta.dx))
            e?.setDoubleValueField(.mouseEventDeltaY, value: Double(delta.dy))
            e?.flags = modifierFlags
            post(e)
        }

        // Buttons and scroll.
        let buttons = p.mouseButtons ? Set(a.pointer.compactMap(\.buttonNumber)) : []
        for b in heldButtons.subtracting(buttons) { button(b, down: false, at: target) }
        for b in buttons.subtracting(heldButtons) { button(b, down: true, at: target) }
        if p.scroll {
            for e in a.pointer where e.isScroll {
                let (x, y) = e.scrollLines
                post(CGEvent(scrollWheelEvent2Source: source, units: .line, wheelCount: 2, wheel1: Int32(y), wheel2: Int32(x), wheel3: 0))
            }
        }
    }

    private func key(_ id: String, down: Bool) {
        guard let info = KeyCatalog.info(id) else { return }
        if down { heldKeys.insert(id) } else { heldKeys.remove(id) }
        let e = CGEvent(keyboardEventSource: source, virtualKey: info.keyCode, keyDown: down)
        if Self.modifierKeys.contains(id) { e?.type = .flagsChanged }
        e?.flags = modifierFlags
        post(e)
    }

    private func button(_ b: Int, down: Bool, at p: CGPoint) {
        if down { heldButtons.insert(b) } else { heldButtons.remove(b) }
        let type: CGEventType = switch (b, down) {
        case (0, true): .leftMouseDown
        case (0, false): .leftMouseUp
        case (1, true): .rightMouseDown
        case (1, false): .rightMouseUp
        case (_, true): .otherMouseDown
        case (_, false): .otherMouseUp
        }
        let e = CGEvent(mouseEventSource: source, mouseType: type, mouseCursorPosition: p, mouseButton: CGMouseButton(rawValue: UInt32(b)) ?? .left)
        e?.setIntegerValueField(.mouseEventButtonNumber, value: Int64(b))
        e?.flags = modifierFlags
        post(e)
    }

    /// Releases everything the model is holding. Called on stop, pause, takeover and errors.
    public func releaseAll() {
        let at = CGEvent(source: nil)?.location ?? .zero
        for b in heldButtons { button(b, down: false, at: at) }
        for k in heldKeys.sorted(by: { !Self.modifierKeys.contains($0) && Self.modifierKeys.contains($1) }) { key(k, down: false) }
        heldKeys = []; heldButtons = []
        fractional = .zero
    }
}
