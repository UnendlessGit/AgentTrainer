import AppKit
import ATCore
import CoreGraphics
import Foundation

/// Marker AgentTrainer writes into events it posts, so recordings and the takeover detector can
/// tell model input from human input.
public enum SyntheticInput {
    public static let userData: Int64 = 0x4147_5452 // "AGTR"
}

/// Listen-only global event tap (needs Input Monitoring). Converts every event to an
/// `InputEventRecord` stamped on the host clock (absolute ns in `timeNs`).
public final class InputTap: @unchecked Sendable {
    public var onEvent: ((InputEventRecord) -> Void)?
    /// When set, events are dropped unless this process is frontmost.
    public var focusProcessID: pid_t?
    public var options = InputCaptureOptions()
    /// Key codes to ignore (AgentTrainer's own hotkeys while their modifiers are held).
    public var ignoredHotkeys: [(code: UInt16, modifiers: CGEventFlags)] = []
    public private(set) var clockDomain = "unknown"

    private var tap: CFMachPort?
    private var thread: Thread?
    private var runLoop: CFRunLoop?
    private var frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier ?? 0
    private var observer: NSObjectProtocol?
    private var timestampIsTicks: Bool?
    private var ignoredDown = Set<UInt16>()

    public init() {}

    public enum TapError: Error, LocalizedError {
        case permissionDenied
        public var errorDescription: String? { "AgentTrainer needs Input Monitoring permission to record keyboard and mouse input." }
    }

    public func start() throws {
        let types: [CGEventType] = [.keyDown, .keyUp, .flagsChanged, .mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged,
                                    .leftMouseDown, .leftMouseUp, .rightMouseDown, .rightMouseUp, .otherMouseDown, .otherMouseUp, .scrollWheel]
        let mask = types.reduce(CGEventMask(0)) { $0 | (1 << CGEventMask($1.rawValue)) }
        let refcon = Unmanaged.passUnretained(self).toOpaque()
        guard let tap = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap, options: .listenOnly,
                                          eventsOfInterest: mask, callback: inputTapCallback, userInfo: refcon) else {
            throw TapError.permissionDenied
        }
        self.tap = tap
        observer = NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: nil) { [weak self] n in
            if let app = n.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication { self?.frontmostPID = app.processIdentifier }
        }
        let source = CFMachPortCreateRunLoopSource(nil, tap, 0)
        let t = Thread { [weak self] in
            self?.runLoop = CFRunLoopGetCurrent()
            CFRunLoopAddSource(CFRunLoopGetCurrent(), source, .commonModes)
            CGEvent.tapEnable(tap: tap, enable: true)
            CFRunLoopRun()
        }
        t.qualityOfService = .userInteractive
        t.name = "agenttrainer.input-tap"
        thread = t
        t.start()
    }

    public func stop() {
        if let tap { CGEvent.tapEnable(tap: tap, enable: false); CFMachPortInvalidate(tap) }
        if let rl = runLoop { CFRunLoopStop(rl) }
        if let o = observer { NSWorkspace.shared.notificationCenter.removeObserver(o) }
        tap = nil; runLoop = nil; thread = nil; observer = nil
    }

    fileprivate func reenable() { if let tap { CGEvent.tapEnable(tap: tap, enable: true) } }

    /// CGEvent timestamps are documented as nanoseconds but some systems report mach ticks;
    /// decide once by comparing with "now" in both units.
    private func hostNanoseconds(_ ts: UInt64) -> UInt64 {
        if timestampIsTicks == nil {
            let ticks = mach_absolute_time(), ns = HostClock.ticksToNanoseconds(ticks)
            let dNs = ts > ns ? ts - ns : ns - ts
            let dTicks = ts > ticks ? ts - ticks : ticks - ts
            timestampIsTicks = dTicks < dNs
            clockDomain = timestampIsTicks! ? "mach-ticks" : "uptime-ns"
        }
        return timestampIsTicks! ? HostClock.ticksToNanoseconds(ts) : ts
    }

    static let modifierMasks: [UInt16: UInt64] = [
        0x38: 0x02, 0x3C: 0x04, 0x3B: 0x01, 0x3E: 0x2000, 0x3A: 0x20, 0x3D: 0x40, 0x37: 0x08, 0x36: 0x10,
        0x3F: CGEventFlags.maskSecondaryFn.rawValue, 0x39: CGEventFlags.maskAlphaShift.rawValue,
    ]

    fileprivate func handle(_ type: CGEventType, _ e: CGEvent) {
        if let f = focusProcessID, frontmostPID != f { return }
        let synthetic = e.getIntegerValueField(.eventSourceUserData) == SyntheticInput.userData
        let t = Int64(hostNanoseconds(e.timestamp))
        let loc = e.location
        var r: InputEventRecord?
        let flags: UInt8 = synthetic ? 1 : 0
        switch type {
        case .keyDown, .keyUp:
            guard options.keyboard else { return }
            let code = UInt16(e.getIntegerValueField(.keyboardEventKeycode))
            if type == .keyDown, ignoredHotkeys.contains(where: { $0.code == code && e.flags.contains($0.modifiers) }) { ignoredDown.insert(code); return }
            if type == .keyUp, ignoredDown.remove(code) != nil { return }
            if type == .keyDown, e.getIntegerValueField(.keyboardEventAutorepeat) != 0 { return }
            r = InputEventRecord(timeNs: t, kind: type == .keyDown ? .keyDown : .keyUp, flags: flags, code: code, x: Float(loc.x), y: Float(loc.y))
        case .flagsChanged:
            guard options.keyboard else { return }
            let code = UInt16(e.getIntegerValueField(.keyboardEventKeycode))
            guard let m = Self.modifierMasks[code] else { return }
            let down = e.flags.rawValue & m != 0
            r = InputEventRecord(timeNs: t, kind: .flagsChanged, flags: flags | (down ? 2 : 0), code: code, x: Float(loc.x), y: Float(loc.y))
        case .mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged:
            guard options.mouseMovement else { return }
            r = InputEventRecord(timeNs: t, kind: .mouseMove, flags: flags, x: Float(loc.x), y: Float(loc.y),
                                 dx: Float(e.getDoubleValueField(.mouseEventDeltaX)), dy: Float(e.getDoubleValueField(.mouseEventDeltaY)))
        case .leftMouseDown, .rightMouseDown, .otherMouseDown, .leftMouseUp, .rightMouseUp, .otherMouseUp:
            guard options.mouseButtons else { return }
            let down = [.leftMouseDown, .rightMouseDown, .otherMouseDown].contains(type)
            r = InputEventRecord(timeNs: t, kind: down ? .buttonDown : .buttonUp, flags: flags,
                                 code: UInt16(e.getIntegerValueField(.mouseEventButtonNumber)), x: Float(loc.x), y: Float(loc.y))
        case .scrollWheel:
            guard options.scroll else { return }
            // Fixed-point deltas are in lines (fractional for trackpads); y positive = up.
            r = InputEventRecord(timeNs: t, kind: .scroll, flags: flags, x: Float(loc.x), y: Float(loc.y),
                                 dx: Float(e.getDoubleValueField(.scrollWheelEventFixedPtDeltaAxis2)),
                                 dy: Float(e.getDoubleValueField(.scrollWheelEventFixedPtDeltaAxis1)))
        default:
            return
        }
        if let r { onEvent?(r) }
    }
}

private func inputTapCallback(proxy: CGEventTapProxy, type: CGEventType, event: CGEvent, refcon: UnsafeMutableRawPointer?) -> Unmanaged<CGEvent>? {
    guard let refcon else { return Unmanaged.passUnretained(event) }
    let tap = Unmanaged<InputTap>.fromOpaque(refcon).takeUnretainedValue()
    if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
        tap.reenable()
    } else {
        tap.handle(type, event)
    }
    return Unmanaged.passUnretained(event)
}
