import Carbon
import CoreGraphics
import Foundation

/// Coordinates Carbon hotkeys with the low-level event taps. Carbon invokes the
/// action on key-down, while key-up and modifier-release events arrive slightly
/// later. Those trailing events must not look like training input or a human
/// interruption of an agent that was just started by the same shortcut.
final class HotkeySuppression: @unchecked Sendable {
    static let shared = HotkeySuppression()

    private let lock = NSLock()
    private var activeBinding: HotkeyBinding?
    private var expiresAt: UInt64 = 0

    func activate(_ binding: HotkeyBinding, duration: TimeInterval = 1) {
        lock.lock()
        activeBinding = binding
        expiresAt = DispatchTime.now().uptimeNanoseconds &+ UInt64(max(0, duration) * 1_000_000_000)
        lock.unlock()
    }

    func suppresses(_ sample: InputSample) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard DispatchTime.now().uptimeNanoseconds <= expiresAt, let binding = activeBinding else {
            activeBinding = nil
            return false
        }
        switch sample.kind {
        case .key:
            return sample.keyCode == UInt16(binding.keyCode)
        case .flags:
            let relevant = sample.modifiers & HotkeyBinding.cgModifierMask
            return relevant & ~binding.cgEventModifiers == 0
        default:
            return false
        }
    }
}

final class GlobalHotkeyMonitor: @unchecked Sendable {
    private var reference: EventHotKeyRef?
    private var handler: EventHandlerRef?
    private let action: @Sendable () -> Void
    private let identifier: UInt32
    private var binding: HotkeyBinding
    private(set) var registrationStatus: OSStatus = noErr

    init(identifier: UInt32, binding: HotkeyBinding, action: @escaping @Sendable () -> Void) {
        self.identifier = identifier; self.binding = binding; self.action = action
    }

    func start() {
        guard reference == nil else { return }
        var eventType = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        let pointer = Unmanaged.passUnretained(self).toOpaque()
        registrationStatus = InstallEventHandler(GetApplicationEventTarget(), { _, event, userData in
            guard let userData, let event else { return OSStatus(eventNotHandledErr) }
            var id = EventHotKeyID()
            GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID), nil, MemoryLayout<EventHotKeyID>.size, nil, &id)
            let monitor = Unmanaged<GlobalHotkeyMonitor>.fromOpaque(userData).takeUnretainedValue()
            guard id.signature == fourCC("ATPN"), id.id == monitor.identifier else { return OSStatus(eventNotHandledErr) }
            HotkeySuppression.shared.activate(monitor.binding)
            monitor.action()
            return noErr
        }, 1, &eventType, pointer, &handler)
        guard registrationStatus == noErr else { handler = nil; return }
        let id = EventHotKeyID(signature: fourCC("ATPN"), id: identifier)
        registrationStatus = RegisterEventHotKey(binding.keyCode, binding.carbonModifiers, id, GetApplicationEventTarget(), 0, &reference)
        if registrationStatus != noErr {
            if let handler { RemoveEventHandler(handler) }
            handler = nil; reference = nil
        }
    }

    func update(_ binding: HotkeyBinding) { stop(); self.binding = binding; start() }

    func stop() {
        if let reference { UnregisterEventHotKey(reference) }
        if let handler { RemoveEventHandler(handler) }
        reference = nil; handler = nil
    }

    deinit { stop() }
}

extension HotkeyBinding {
    static let cgModifierMask = CGEventFlags.maskShift.rawValue | CGEventFlags.maskControl.rawValue | CGEventFlags.maskAlternate.rawValue | CGEventFlags.maskCommand.rawValue

    var cgEventModifiers: UInt64 {
        var flags: UInt64 = 0
        if carbonModifiers & UInt32(1 << 9) != 0 { flags |= CGEventFlags.maskShift.rawValue }
        if carbonModifiers & UInt32(1 << 12) != 0 { flags |= CGEventFlags.maskControl.rawValue }
        if carbonModifiers & UInt32(1 << 11) != 0 { flags |= CGEventFlags.maskAlternate.rawValue }
        if carbonModifiers & UInt32(1 << 8) != 0 { flags |= CGEventFlags.maskCommand.rawValue }
        return flags
    }

    func matches(_ sample: InputSample) -> Bool {
        sample.kind == .key && sample.keyCode == UInt16(keyCode) && (sample.modifiers & Self.cgModifierMask) == cgEventModifiers
    }
}

private func fourCC(_ string: String) -> OSType {
    string.utf8.prefix(4).reduce(0) { ($0 << 8) | OSType($1) }
}
