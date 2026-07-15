import AppKit
#if canImport(Carbon)
import Carbon
#endif
import CoreGraphics
import Foundation

/// Coordinates the native hotkey registration (or its future AppKit fallback)
/// with the low-level event taps.
/// Key-up and modifier-release events arrive after the shortcut action; those
/// trailing events must not look like training input or a human interruption of
/// an agent that was just started by the same shortcut.
final class HotkeySuppression: @unchecked Sendable {
    static let shared = HotkeySuppression()

    private let lock = NSLock()
    private var activeBinding: HotkeyBinding?
    private var expiresAt: UInt64 = 0

    func activate(_ binding: HotkeyBinding, duration: TimeInterval = 1) {
        lock.lock()
        activeBinding = binding
        let seconds = duration.isFinite ? min(60, max(0, duration)) : 1
        expiresAt = DispatchTime.now().uptimeNanoseconds &+ UInt64(seconds * 1_000_000_000)
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
            return UInt16(exactly: binding.keyCode) == sample.keyCode
        case .flags:
            let relevant = sample.modifiers & HotkeyBinding.cgModifierMask
            return relevant & ~binding.cgEventModifiers == 0
        default:
            return false
        }
    }
}

final class GlobalHotkeyMonitor: @unchecked Sendable {
    static let successStatus: Int32 = 0
    private static let unavailableStatus: Int32 = -1

    #if canImport(Carbon)
    private var reference: EventHotKeyRef?
    private var handler: EventHandlerRef?
    private var nativeBindingPressed = false
    private let identifier: UInt32
    #else
    private var globalMonitor: Any?
    private var localMonitor: Any?
    #endif
    private let action: @Sendable () -> Void
    private var binding: HotkeyBinding
    private(set) var registrationStatus: Int32 = successStatus

    init(identifier: UInt32, binding: HotkeyBinding, action: @escaping @Sendable () -> Void) {
        #if canImport(Carbon)
        self.identifier = identifier
        #else
        _ = identifier
        #endif
        self.binding = binding; self.action = action
    }

    func start() {
        #if canImport(Carbon)
        guard reference == nil, handler == nil else { return }
        nativeBindingPressed = false
        let eventTypes = [
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed)),
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased))
        ]
        let pointer = Unmanaged.passUnretained(self).toOpaque()
        registrationStatus = eventTypes.withUnsafeBufferPointer { events in
            InstallEventHandler(GetApplicationEventTarget(), { _, event, userData in
                guard let userData, let event else { return OSStatus(eventNotHandledErr) }
                var id = EventHotKeyID()
                GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID), nil, MemoryLayout<EventHotKeyID>.size, nil, &id)
                let monitor = Unmanaged<GlobalHotkeyMonitor>.fromOpaque(userData).takeUnretainedValue()
                guard id.signature == fourCC("ATPN"), id.id == monitor.identifier else { return OSStatus(eventNotHandledErr) }
                if GetEventKind(event) == UInt32(kEventHotKeyReleased) {
                    monitor.nativeBindingPressed = false
                    return noErr
                }
                guard !monitor.nativeBindingPressed else { return noErr }
                monitor.nativeBindingPressed = true
                HotkeySuppression.shared.activate(monitor.binding)
                monitor.action()
                return noErr
            }, events.count, events.baseAddress, pointer, &handler)
        }
        guard registrationStatus == Self.successStatus else { handler = nil; return }
        let id = EventHotKeyID(signature: fourCC("ATPN"), id: identifier)
        registrationStatus = RegisterEventHotKey(binding.keyCode, binding.carbonModifiers, id, GetApplicationEventTarget(), 0, &reference)
        if registrationStatus != Self.successStatus {
            if let handler { RemoveEventHandler(handler) }
            handler = nil; reference = nil
        }
        #else
        guard globalMonitor == nil, localMonitor == nil else { return }
        globalMonitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { [weak self] event in
            self?.handle(event)
        }
        localMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            self?.handle(event)
            return event
        }
        registrationStatus = globalMonitor != nil && localMonitor != nil ? Self.successStatus : Self.unavailableStatus
        if registrationStatus != Self.successStatus { stop() }
        #endif
    }

    func update(_ binding: HotkeyBinding) { stop(); self.binding = binding; start() }

    func stop() {
        #if canImport(Carbon)
        if let reference { UnregisterEventHotKey(reference) }
        if let handler { RemoveEventHandler(handler) }
        reference = nil; handler = nil; nativeBindingPressed = false
        #else
        if let globalMonitor { NSEvent.removeMonitor(globalMonitor) }
        if let localMonitor { NSEvent.removeMonitor(localMonitor) }
        globalMonitor = nil; localMonitor = nil
        #endif
    }

    deinit { stop() }

    #if !canImport(Carbon)
    private func handle(_ event: NSEvent) {
        guard !event.isARepeat,
              let keyCode = UInt16(exactly: binding.keyCode), event.keyCode == keyCode,
              event.modifierFlags.intersection(.deviceIndependentFlagsMask).intersection([.shift, .control, .option, .command]) == binding.nsEventModifiers else { return }
        HotkeySuppression.shared.activate(binding)
        action()
    }
    #endif
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

    var nsEventModifiers: NSEvent.ModifierFlags {
        var flags: NSEvent.ModifierFlags = []
        if carbonModifiers & UInt32(1 << 9) != 0 { flags.insert(.shift) }
        if carbonModifiers & UInt32(1 << 12) != 0 { flags.insert(.control) }
        if carbonModifiers & UInt32(1 << 11) != 0 { flags.insert(.option) }
        if carbonModifiers & UInt32(1 << 8) != 0 { flags.insert(.command) }
        return flags
    }

    func matches(_ sample: InputSample) -> Bool {
        sample.kind == .key && UInt16(exactly: keyCode) == sample.keyCode && (sample.modifiers & Self.cgModifierMask) == cgEventModifiers
    }
}

#if canImport(Carbon)
private func fourCC(_ string: String) -> OSType {
    string.utf8.prefix(4).reduce(0) { ($0 << 8) | OSType($1) }
}
#endif
