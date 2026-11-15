import AppKit
import Carbon.HIToolbox

/// System-wide shortcuts via Carbon hot keys (work in full-screen games, need no permission).
@MainActor
final class HotkeyCenter {
    static let shared = HotkeyCenter()
    private var refs: [UInt32: EventHotKeyRef] = [:]
    private var actions: [UInt32: () -> Void] = [:]
    private var handler: EventHandlerRef?

    private init() {
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(GetApplicationEventTarget(), { _, event, _ in
            var id = EventHotKeyID()
            GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID), nil,
                              MemoryLayout<EventHotKeyID>.size, nil, &id)
            let key = id.id
            DispatchQueue.main.async { MainActor.assumeIsolated { HotkeyCenter.shared.actions[key]?() } }
            return noErr
        }, 1, &spec, nil, &handler)
    }

    func register(_ id: UInt32, _ hotkey: Hotkey, action: @escaping () -> Void) {
        unregister(id)
        var ref: EventHotKeyRef?
        let hkID = EventHotKeyID(signature: OSType(0x4147_5452), id: id)
        if RegisterEventHotKey(UInt32(hotkey.keyCode), hotkey.carbonModifiers, hkID, GetApplicationEventTarget(), 0, &ref) == noErr, let ref {
            refs[id] = ref
            actions[id] = action
        }
    }

    func unregister(_ id: UInt32) {
        if let r = refs.removeValue(forKey: id) { UnregisterEventHotKey(r) }
        actions[id] = nil
    }
}
