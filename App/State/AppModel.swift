import ATCore
import Foundation
import Observation

enum AppTab: String, CaseIterable, Identifiable {
    case recordings, library, models, training, run, arena, settings
    var id: String { rawValue }
    var title: String {
        switch self {
        case .recordings: "Recordings"
        case .library: "Library"
        case .models: "AI Models"
        case .training: "Training"
        case .run: "Run"
        case .arena: "Test Arena"
        case .settings: "Settings"
        }
    }
    var symbol: String {
        switch self {
        case .recordings: "record.circle"
        case .library: "film.stack"
        case .models: "brain"
        case .training: "chart.line.uptrend.xyaxis"
        case .run: "play.circle"
        case .arena: "gamecontroller"
        case .settings: "gearshape"
        }
    }
}

/// Root state shared by every tab.
@MainActor @Observable
final class AppModel {
    let settings: AppSettings
    let library: LibraryStore
    let models: ModelStore
    let recorder: RecorderController
    let training: TrainingController
    let runner: RunController
    let downloads: DownloadCenter
    var tab: AppTab = .recordings
    var selectedModelID: UUID?

    init() {
        let s = AppSettings()
        settings = s
        try? s.paths.ensureExists()
        library = LibraryStore(root: s.paths.libraryRoot)
        models = ModelStore(root: s.paths.modelsRoot)
        downloads = DownloadCenter(paths: s.paths)
        recorder = RecorderController(settings: s, library: library)
        training = TrainingController(settings: s, models: models, library: library)
        runner = RunController(settings: s, models: models)
        selectedModelID = models.models.first?.id
        registerHotkeys()
    }

    func registerHotkeys() {
        let st = settings.stored
        HotkeyCenter.shared.register(1, st.recordHotkey) { [weak self] in Task { await self?.recorder.toggle() } }
        HotkeyCenter.shared.register(2, st.runHotkey) { [weak self] in Task { await self?.runner.toggle() } }
        HotkeyCenter.shared.register(3, st.pauseHotkey) { [weak self] in self?.runner.togglePause() }
        HotkeyCenter.shared.register(4, st.emergencyHotkey) { [weak self] in self?.runner.emergencyStop() }
    }

    func applyStorageRoots() {
        let p = settings.paths
        try? p.ensureExists()
        library.setRoot(p.libraryRoot)
        models.setRoot(p.modelsRoot)
        recorder.destination = p.libraryRoot
    }
}
