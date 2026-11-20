import ATCore
import Foundation
import Observation

enum AppTab: String, CaseIterable, Identifiable {
    case recordings, library, models, training, run, arena, settings
    var id: String { rawValue }
    var title: String {
        switch self {
        case .recordings: "Record"
        case .library: "Library"
        case .models: "Models"
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
    static let shared = AppModel()
    let settings: AppSettings
    let library: LibraryStore
    let models: ModelStore
    let recorder: RecorderController
    let training: TrainingController
    let runner: RunController
    let replay: ReplayController
    let downloads: DownloadCenter
    let hub: HubDownloads
    let labels: LabelController
    var tab: AppTab = .recordings
    var selectedModelID: UUID?
    /// Opens the model editor on this section next time it appears (e.g. Training → Choose folders).
    var pendingModelSection: ModelSection?
    /// Library tab state kept here so it survives switching tabs.
    var libraryUI = LibraryUIState()

    init() {
        let s = AppSettings()
        settings = s
        try? s.paths.ensureExists()
        library = LibraryStore(root: s.paths.libraryRoot)
        models = ModelStore(root: s.paths.modelsRoot)
        downloads = DownloadCenter(paths: s.paths)
        hub = HubDownloads(pretrainedRoot: s.paths.pretrainedRoot)
        labels = LabelController(settings: s)
        recorder = RecorderController(settings: s, library: library)
        training = TrainingController(settings: s, models: models, library: library)
        runner = RunController(settings: s, models: models)
        replay = ReplayController(settings: s, runner: runner)
        selectedModelID = models.models.first?.id
        runner.library = library
        runner.correctionDestination = { [weak recorder, weak library] in recorder?.destination ?? library?.fs.root ?? s.paths.libraryRoot }
        labels.onFinished = { [weak library] in library?.refresh() }
        registerHotkeys()
    }

    func registerHotkeys() {
        // Snapshot and self-test copies of the app must never react to the user's shortcuts.
        if CommandLine.arguments.contains(where: { $0 == "--ui-snapshot" || $0.hasPrefix("--selftest") }) { return }
        let st = settings.stored
        HotkeyCenter.shared.register(1, st.recordHotkey) { [weak self] in Task { await self?.recorder.toggle() } }
        HotkeyCenter.shared.register(5, st.cancelRecordingHotkey) { [weak self] in Task { await self?.recorder.cancel() } }
        HotkeyCenter.shared.register(2, st.runHotkey) { [weak self] in Task { await self?.runner.toggle() } }
        HotkeyCenter.shared.register(3, st.pauseHotkey) { [weak self] in self?.runner.togglePause() }
        HotkeyCenter.shared.register(4, st.emergencyHotkey) { [weak self] in self?.runner.emergencyStop(); self?.replay.stop() }
        HotkeyCenter.shared.register(6, st.replayHotkey) { [weak self] in self?.replay.toggle() }
        HotkeyCenter.shared.register(7, st.correctionHotkey) { [weak self] in self?.runner.toggleCorrection() }
    }

    func applyStorageRoots() {
        let p = settings.paths
        try? p.ensureExists()
        library.setRoot(p.libraryRoot)
        models.setRoot(p.modelsRoot)
        recorder.destination = p.libraryRoot
    }
}

/// What the Library tab is showing: kept on the app model so leaving the tab and coming back restores it.
struct LibraryUIState {
    var selection = Set<URL>()
    var anchor: URL?
    var expanded = Set<URL>()
    var search = ""
}
