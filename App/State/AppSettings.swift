import ATCore
import Carbon.HIToolbox
import Foundation
import Observation

/// A global keyboard shortcut.
struct Hotkey: Codable, Hashable {
    var keyCode: UInt16
    var command = false, shift = false, option = false, control = false

    var display: String {
        var s = ""
        if control { s += "⌃" }
        if option { s += "⌥" }
        if shift { s += "⇧" }
        if command { s += "⌘" }
        return s + (KeyCatalog.byKeyCode[keyCode]?.label.uppercased() ?? "#\(keyCode)")
    }

    var carbonModifiers: UInt32 {
        (command ? UInt32(cmdKey) : 0) | (shift ? UInt32(shiftKey) : 0) | (option ? UInt32(optionKey) : 0) | (control ? UInt32(controlKey) : 0)
    }

    var cgFlags: CGEventFlags {
        var f = CGEventFlags()
        if command { f.insert(.maskCommand) }
        if shift { f.insert(.maskShift) }
        if option { f.insert(.maskAlternate) }
        if control { f.insert(.maskControl) }
        return f
    }
}

enum HUDCorner: String, Codable, CaseIterable, Identifiable {
    case bottomRight, bottomLeft, topRight, topLeft
    var id: String { rawValue }
    var label: String {
        switch self {
        case .bottomRight: "Bottom right"
        case .bottomLeft: "Bottom left"
        case .topRight: "Top right"
        case .topLeft: "Top left"
        }
    }
}

struct HUDOptions: Codable, Hashable {
    var enabled = true
    var keyboard = true
    var mouse = true
    var vision = true
    var stats = true
    var scale = 1.0
    var opacity = 0.92
    var corner = HUDCorner.bottomRight
}

/// How Auto Label works, remembered between runs.
struct LabelOptions: Codable, Hashable {
    var detail = LabelDetail.normal
    var variations = 5
    var doubleCheck = false
    var video = LabelVideoDetail.standard
    var replaceAutomatic = true
}

struct StoredSettings: Codable {
    var libraryRoot: String?
    var modelsRoot: String?
    var cacheRoot: String?
    var recordHotkey = Hotkey(keyCode: 0x0F, command: true, shift: true, option: true) // ⌥⇧⌘R
    var cancelRecordingHotkey = Hotkey(keyCode: 0x07, command: true, shift: true, option: true) // ⌥⇧⌘X
    var runHotkey = Hotkey(keyCode: 0x31, command: true, shift: true, option: true) // ⌥⇧⌘Space
    var pauseHotkey = Hotkey(keyCode: 0x23, command: true, shift: true, option: true) // ⌥⇧⌘P
    var emergencyHotkey = Hotkey(keyCode: 0x35, command: true, shift: true) // ⇧⌘Esc
    /// Starts / stops replaying the recording armed in the Library.
    var replayHotkey = Hotkey(keyCode: 0x16, command: true) // ⌘6
    /// While the AI runs: take over (the AI freezes) / hand back (the AI carries on).
    var correctionHotkey = Hotkey(keyCode: 0x1A, command: true) // ⌘7
    var recordHUD = HUDOptions()
    var runHUD = HUDOptions()
    var captureFPS = 60
    var captureMaxHeight = 720
    var videoQuality = 0.7
    var showsCursor = true
    var input = InputCaptureOptions()
    var countdownSeconds = 3
    var mlxMemoryLimitGB = 24.0
    var inferenceQuantized = false
    var pauseOnHumanInput = true
    var keepGPUWarm = true
    var reduceRecordingWhileTraining = true
    /// Injected mouse units per Roblox mouse pixel, measured by Elkborn's CALIBRATE MOUSE (1 = not calibrated).
    var elkbornUnitsPerPixel = 1.0
    var elkbornCalibrated = false
    /// The labelling model Auto Label uses (`LanguageModels.labelers` id); nil = the best downloaded one.
    var labelerModelID: String?
    var labelOptions = LabelOptions()

    /// Every AgentTrainer shortcut: never recorded as input, never replayed, never a takeover.
    var appHotkeys: [Hotkey] { [recordHotkey, cancelRecordingHotkey, runHotkey, pauseHotkey, emergencyHotkey, replayHotkey, correctionHotkey] }

    /// Decodes saved settings on top of the defaults, so settings added in later versions (and any
    /// that fail to decode) fall back to their defaults instead of resetting everything.
    static func decodeMerging(_ data: Data) -> StoredSettings? {
        guard let saved = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let defaultsData = try? JSONEncoder().encode(StoredSettings()),
              var merged = try? JSONSerialization.jsonObject(with: defaultsData) as? [String: Any] else { return nil }
        for (k, v) in saved { merged[k] = v }
        if let d = try? JSONSerialization.data(withJSONObject: merged), let s = try? JSONDecoder().decode(StoredSettings.self, from: d) { return s }
        // A single bad value: keep the defaults for the keys that don't decode.
        var result = merged
        for (k, v) in saved {
            var trial = merged
            trial[k] = v
            if let d = try? JSONSerialization.data(withJSONObject: trial), (try? JSONDecoder().decode(StoredSettings.self, from: d)) != nil { result[k] = v } else { result[k] = merged[k] }
        }
        guard let d = try? JSONSerialization.data(withJSONObject: result) else { return nil }
        return try? JSONDecoder().decode(StoredSettings.self, from: d)
    }
}

/// App-wide preferences, persisted in UserDefaults.
@MainActor @Observable
final class AppSettings {
    var stored: StoredSettings { didSet { save() } }

    init() {
        if let d = UserDefaults.standard.data(forKey: "settings"), let s = StoredSettings.decodeMerging(d) {
            stored = s
        } else {
            stored = StoredSettings()
        }
    }

    private func save() {
        if let d = try? JSONEncoder().encode(stored) { UserDefaults.standard.set(d, forKey: "settings") }
    }

    var paths: AppPaths {
        let std = AppPaths.standard
        return AppPaths(libraryRoot: stored.libraryRoot.map { URL(fileURLWithPath: $0, isDirectory: true) } ?? std.libraryRoot,
                        modelsRoot: stored.modelsRoot.map { URL(fileURLWithPath: $0, isDirectory: true) } ?? std.modelsRoot,
                        cacheRoot: stored.cacheRoot.map { URL(fileURLWithPath: $0, isDirectory: true) } ?? std.cacheRoot,
                        pretrainedRoot: std.pretrainedRoot)
    }
}
