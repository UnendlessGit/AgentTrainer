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

    enum Decision: Equatable {
        case pass(InputSample)
        case suppress(HotkeyBinding)
    }

    private struct Active {
        var binding: HotkeyBinding
        var expiresAt: UInt64
        var triggerReleased = false
        var modifiersReleased: Bool
    }

    private let lock = NSLock()
    private var active: Active?

    func activate(_ binding: HotkeyBinding, duration: TimeInterval = 1) {
        lock.lock()
        let seconds = duration.isFinite ? min(60, max(0, duration)) : 1
        active = Active(
            binding: binding,
            expiresAt: DispatchTime.now().uptimeNanoseconds &+ UInt64(seconds * 1_000_000_000),
            modifiersReleased: binding.cgEventModifiers == 0
        )
        lock.unlock()
    }

    /// Suppresses only the physical lifecycle that invoked a global shortcut.
    /// A fixed time window alone is insufficient: after the shortcut is fully
    /// released it can swallow an unrelated Shift/Control release that happens
    /// to occur during that window. Foreign modifiers are preserved with only
    /// the shortcut-owned modifier bits removed.
    func process(_ sample: InputSample) -> Decision {
        lock.lock()
        defer { lock.unlock() }
        guard var active, DispatchTime.now().uptimeNanoseconds <= active.expiresAt else {
            self.active = nil
            return .pass(sample)
        }

        let triggerMatches = switch sample.kind {
        case .key:
            active.binding.mouseButton == nil && UInt16(exactly: active.binding.keyCode) == sample.keyCode
        case .mouseButton:
            active.binding.mouseButton == sample.button
        case .flags, .mouseMove, .scroll:
            false
        }
        if triggerMatches {
            if !sample.isDown { active.triggerReleased = true }
            self.active = active.triggerReleased && active.modifiersReleased ? nil : active
            return .suppress(active.binding)
        }

        if sample.kind == .flags, !active.modifiersReleased {
            let relevant = sample.modifiers & HotkeyBinding.cgModifierMask
            let shortcutModifiers = active.binding.cgEventModifiers
            let foreignModifiers = relevant & ~shortcutModifiers
            if relevant & shortcutModifiers == 0 { active.modifiersReleased = true }
            self.active = active.triggerReleased && active.modifiersReleased ? nil : active
            if foreignModifiers == 0 { return .suppress(active.binding) }
            var sanitized = sample
            sanitized.modifiers &= ~shortcutModifiers
            return .pass(sanitized)
        }

        self.active = active
        if !active.modifiersReleased, active.binding.cgEventModifiers != 0 {
            var sanitized = sample
            sanitized.modifiers &= ~active.binding.cgEventModifiers
            return .pass(sanitized)
        }
        return .pass(sample)
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
    private var globalKeyboardMonitor: Any?
    private var localKeyboardMonitor: Any?
    #endif
    private var globalMouseMonitor: Any?
    private var localMouseMonitor: Any?
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
        if binding.mouseButton != nil {
            startMouseMonitoring()
            return
        }
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
        guard globalKeyboardMonitor == nil, localKeyboardMonitor == nil else { return }
        globalKeyboardMonitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { [weak self] event in
            self?.handle(event)
        }
        localKeyboardMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            self?.handle(event)
            return event
        }
        registrationStatus = globalKeyboardMonitor != nil && localKeyboardMonitor != nil ? Self.successStatus : Self.unavailableStatus
        if registrationStatus != Self.successStatus { stop() }
        #endif
    }

    func update(_ binding: HotkeyBinding) { stop(); self.binding = binding; start() }

    func stop() {
        if let globalMouseMonitor { NSEvent.removeMonitor(globalMouseMonitor) }
        if let localMouseMonitor { NSEvent.removeMonitor(localMouseMonitor) }
        globalMouseMonitor = nil; localMouseMonitor = nil
        #if canImport(Carbon)
        if let reference { UnregisterEventHotKey(reference) }
        if let handler { RemoveEventHandler(handler) }
        reference = nil; handler = nil; nativeBindingPressed = false
        #else
        if let globalKeyboardMonitor { NSEvent.removeMonitor(globalKeyboardMonitor) }
        if let localKeyboardMonitor { NSEvent.removeMonitor(localKeyboardMonitor) }
        globalKeyboardMonitor = nil; localKeyboardMonitor = nil
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

    private func startMouseMonitoring() {
        guard globalMouseMonitor == nil, localMouseMonitor == nil else { return }
        let mask: NSEvent.EventTypeMask = [.leftMouseDown, .rightMouseDown, .otherMouseDown]
        globalMouseMonitor = NSEvent.addGlobalMonitorForEvents(matching: mask) { [weak self] event in
            _ = self?.handleMouse(event)
        }
        localMouseMonitor = NSEvent.addLocalMonitorForEvents(matching: mask) { [weak self] event in
            self?.handleMouse(event) == true ? nil : event
        }
        registrationStatus = globalMouseMonitor != nil && localMouseMonitor != nil ? Self.successStatus : Self.unavailableStatus
        if registrationStatus != Self.successStatus { stop() }
    }

    @discardableResult
    private func handleMouse(_ event: NSEvent) -> Bool {
        guard let button = binding.mouseButton,
              event.buttonNumber == Int(button),
              event.modifierFlags.intersection(.deviceIndependentFlagsMask).intersection([.shift, .control, .option, .command]) == binding.nsEventModifiers else { return false }
        HotkeySuppression.shared.activate(binding)
        action()
        return true
    }
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
        let inputMatches: Bool
        if let mouseButton {
            inputMatches = sample.kind == .mouseButton && sample.button == mouseButton
        } else {
            inputMatches = sample.kind == .key && UInt16(exactly: keyCode) == sample.keyCode
        }
        return inputMatches && (sample.modifiers & Self.cgModifierMask) == cgEventModifiers
    }

    var displayName: String {
        if let mouseButton {
            return switch mouseButton {
            case 0: "Left Mouse"
            case 1: "Right Mouse"
            case 2: "Middle Mouse"
            default: "Mouse \(Int(mouseButton) + 1)"
            }
        }
        return KeyNames.name(for: UInt16(clamping: keyCode))
    }
}

#if canImport(Carbon)
private func fourCC(_ string: String) -> OSType {
    string.utf8.prefix(4).reduce(0) { ($0 << 8) | OSType($1) }
}
#endif
