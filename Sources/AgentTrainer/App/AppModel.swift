import AppKit
import AVFoundation
import CoreGraphics
import Foundation
import MLX

@MainActor
final class AppModel: ObservableObject {
    @Published var selection: AppSection = .home
    @Published var captureKind: CaptureKind = .display
    @Published var captureSources: [CaptureSourceOption] = []
    @Published var selectedSourceID: UInt32?
    @Published var captureFPS = 60.0
    @Published var showsCursor = false
    @Published var regionX = 0.0
    @Published var regionY = 0.0
    @Published var regionWidth = 1280.0
    @Published var regionHeight = 720.0
    @Published var recordings: [RecordingItem] = []
    @Published var recordingFolders: [RecordingFolder] = []
    @Published var recordingDestinationFolderID: UUID?
    @Published var selectedRecordingID: UUID?
    @Published var profiles: [AIProfile] = []
    @Published var selectedProfileID: UUID?
    @Published var versions: [ModelVersionManifest] = []
    @Published private(set) var versionsLoadedForProfileID: UUID?
    @Published private(set) var isLoadingVersions = false
    @Published var isRecording = false
    @Published private(set) var isStartingRecording = false
    @Published var isTraining = false
    @Published var isRunning = false
    @Published var isReplaying = false
    /// UI-only visibility signal. Training/runtime keep working in the
    /// background, while decorative animation and chart refresh can pause.
    @Published var isAppActive = NSApplication.shared.isActive
    @Published var trainingMetrics = TrainingMetrics()
    @Published var runtimeMetrics = RuntimeMetrics()
    @Published var activityStatus = "Ready"
    @Published var trainingStatus = "Idle"
    @Published var runtimeStatus = "Idle"
    @Published var errorMessage: String?
    @Published private(set) var appUpdateProgress: AppUpdateProgress?
    @Published var storageBytes: Int64 = 0
    @Published private(set) var storageUsage = WorkspaceStorageUsage(totalBytes: 0, trainingDataBytes: 0, modelBytes: 0)
    @Published private(set) var storageLocations = WorkspaceLocations(
        supportRoot: WorkspaceStore.defaultRoot,
        trainingDataRoot: WorkspaceStore.defaultRoot,
        modelsRoot: WorkspaceStore.defaultRoot
    )
    @Published private(set) var isChangingStorageLocation = false
    @Published var screenPermission = false
    @Published var accessibilityPermission = false
    @Published var inputPermission = false
    @Published var frameMode: FrameMode = .newest
    @Published var safety = AgentSafetyPolicy()
    @Published var showVisionPreview = true
    @Published var visionPreviewFPS = 10.0
    @Published var visionPreviewMatchesPerception = false
    @Published var cnnVisualizationSettings = CNNVisualizationSettings() {
        didSet {
            persistWorkflowSettings()
            agent?.updateVisualizationSettings(cnnVisualizationSettings)
            hudModel.configureCNNVisualization(cnnVisualizationSettings)
        }
    }
    @Published var hotkeys = HotkeySettings()
    @Published var recordingExcludedKeyCodes: Set<UInt16> = [] { didSet { persistWorkflowSettings() } }
    @Published var recordingTrimStart = 0.5 { didSet { persistWorkflowSettings() } }
    @Published var recordingTrimEnd = 0.5 { didSet { persistWorkflowSettings() } }
    @Published var trainingRunSettings = TrainingRunSettings() { didSet { persistWorkflowSettings() } }
    @Published var runMouseMode: MouseControlMode = .automatic { didSet { persistWorkflowSettings() } }
    @Published var gameCamera = GameCameraSettings() { didSet { persistWorkflowSettings() } }
    @Published var runtimeOutputPermissions = RuntimeOutputPermissions() {
        didSet {
            persistWorkflowSettings()
            agent?.updateOutputPermissions(runtimeOutputPermissions)
        }
    }
    @Published private(set) var trainingProfileID: UUID?
    @Published private(set) var runningProfileID: UUID?

    let hudModel = InputHUDModel()
    private let capture = CaptureService()
    private let input = InputCaptureService()
    private let training = TrainingEngine()
    private let reenactor = InputReenactor()
    private let regionSelector = RegionSelector()
    private lazy var panicHotkey = GlobalHotkeyMonitor(identifier: 1, binding: hotkeys.panic) { [weak self] in Task { @MainActor in self?.panic() } }
    private lazy var recordHotkey = GlobalHotkeyMonitor(identifier: 2, binding: hotkeys.record) { [weak self] in Task { @MainActor in guard let self else { return }; self.recordingIsActiveOrStarting ? await self.stopRecording() : await self.startRecording() } }
    private lazy var runHotkey = GlobalHotkeyMonitor(identifier: 3, binding: hotkeys.run) { [weak self] in Task { @MainActor in guard let self else { return }; self.isRunning ? await self.stopAgent() : await self.startAgent() } }
    private var agent: AgentRuntime?
    private var eventWriter: InputEventWriter?
    private var recordingDirectory: URL?
    private var recordingID: UUID?
    private var recordingHostStart: UInt64 = 0
    private var recordingClock = RecordingClock()
    private var activeRecordingSpec: CaptureSpec?
    private var activeRecordingRect: CGRect?
    private var activeRecordingFolderID: UUID?
    private var activeRecordingExcludedKeyCodes: Set<UInt16> = []
    private var recordingLaunchRevision: UInt64 = 0
    private var lastEventClock = RecordingClock()
    private var profileAutosaveTask: Task<Void, Never>?
    private var isRestoringWorkflowSettings = true
    private var updateCheckStarted = false

    var selectedSource: CaptureSourceOption? { captureSources.first { $0.id == selectedSourceID && sourceMatchesKind($0) } }
    var selectedRecording: RecordingItem? { recordings.first { $0.id == selectedRecordingID } }
    var selectedProfile: AIProfile? { profiles.first { $0.id == selectedProfileID } }
    var recordingIsActiveOrStarting: Bool { isRecording || isStartingRecording }
    var canChangeStorageLocations: Bool {
        !isChangingStorageLocation && !recordingIsActiveOrStarting && !isTraining && !isRunning && !isReplaying
    }

    init() {
        var migratedMouseMode = false
        if let data = UserDefaults.standard.data(forKey: "AgentTrainer.WorkflowSettings"), let saved = try? JSONDecoder().decode(PersistentWorkflowSettings.self, from: data) {
            recordingExcludedKeyCodes = saved.recordingExcludedKeyCodes
            recordingTrimStart = saved.recordingTrimStart
            recordingTrimEnd = saved.recordingTrimEnd
            trainingRunSettings = saved.trainingRunSettings
            runMouseMode = saved.runMouseMode
            gameCamera = saved.gameCamera ?? GameCameraSettings()
            runtimeOutputPermissions = saved.runtimeOutputPermissions ?? RuntimeOutputPermissions()
            cnnVisualizationSettings = (saved.cnnVisualizationSettings ?? CNNVisualizationSettings()).sanitized()
        }
        // Absolute Cursor was the old default, even for locked-camera data.
        // Move existing installs once to the safer recording-aware mode; the
        // user's choice is persisted normally after this migration.
        if !UserDefaults.standard.bool(forKey: "AgentTrainer.MouseModeAutoMigration.v1") {
            if runMouseMode == .absolute { runMouseMode = .automatic; migratedMouseMode = true }
            UserDefaults.standard.set(true, forKey: "AgentTrainer.MouseModeAutoMigration.v1")
        }
        isRestoringWorkflowSettings = false
        if migratedMouseMode { persistWorkflowSettings() }
        if let data = UserDefaults.standard.data(forKey: "AgentTrainer.Hotkeys"), let saved = try? JSONDecoder().decode(HotkeySettings.self, from: data) { hotkeys = saved }
        if Set([hotkeys.panic, hotkeys.record, hotkeys.run]).count != 3 { hotkeys = HotkeySettings() }
        input.ignoredHotkeys = [hotkeys.panic, hotkeys.record, hotkeys.run]
        reenactor.onFinish = { [weak self] reason in Task { @MainActor in self?.isReplaying = false; self?.activityStatus = reason ?? "Reenactment complete" } }
        AppLog.write(category: "Lifecycle", "Application model initialized")
        panicHotkey.start(); recordHotkey.start(); runHotkey.start(); Task { await bootstrap() }
    }

    func bootstrap() async {
        storageLocations = await WorkspaceStore.shared.locations()
        do {
            try await WorkspaceStore.shared.prepare()
            let repairedRecordings = try await WorkspaceStore.shared.repairInvalidRecordingManifests()
            if repairedRecordings > 0 {
                AppLog.write(.warning, category: "Migration", "Recovered legacy recording manifests", details: "\(repairedRecordings) recording manifests repaired; video and input files were unchanged")
            }
            let removed = try await WorkspaceStore.shared.removeObsoleteModelArtifacts(currentSchema: ModelContract.schemaVersion)
            if removed > 0 { AppLog.write(.warning, category: "Migration", "Removed incompatible model artifacts", details: "\(removed) model/checkpoint directories; recordings and profiles were preserved") }
            let obsoleteCaches = try await WorkspaceStore.shared.removeObsoleteCaches(currentSchema: TrainingDataContract.schemaVersion)
            if obsoleteCaches > 0 { AppLog.write(category: "Migration", "Removed obsolete dataset caches", details: "\(obsoleteCaches) cache directories; recordings and runnable brains were preserved") }
            let prunedAutosaves = try await WorkspaceStore.shared.pruneAllAutosaves(keeping: 10)
            if prunedAutosaves > 0 { AppLog.write(category: "Storage", "Pruned old autosaves", details: "Removed \(prunedAutosaves); kept the newest 10 per AI and preserved Crystal V4") }
            await refreshLibrary(); await repairLegacyRecordingClocks(); await refreshSources(); refreshPermissions()
            await ensureRecordingThumbnails()
            if profiles.isEmpty { createProfile(name: "My First Agent") }
            AppLog.write(category: "Lifecycle", "Workspace ready", details: "\(recordings.count) recordings, \(profiles.count) profiles")
        } catch {
            await refreshStorageState()
            selection = .settings
            present(error)
        }
    }

    func checkForUpdatesAtLaunch() async {
        guard !updateCheckStarted else { return }
        updateCheckStarted = true
        guard let configuration = GitHubUpdateConfiguration() else {
            AppLog.write(.warning, category: "Updates", "GitHub update configuration is missing or invalid")
            return
        }
        let updater = GitHubReleaseUpdater(configuration: configuration)
        let release: GitHubRelease
        do {
            release = try await updater.latestRelease()
        } catch {
            // Offline launches, rate limits, and repositories with no releases
            // should never interrupt local recording, training, or runtime work.
            AppLog.write(.warning, category: "Updates", "GitHub update check unavailable", details: error.localizedDescription)
            return
        }
        guard updater.updateIsAvailable(release), let version = release.version else {
            AppLog.write(category: "Updates", "AgentTrainer is up to date", details: "Installed \(configuration.currentVersion), latest \(release.tagName)")
            return
        }
        AppLog.write(category: "Updates", "GitHub update available", details: "Installed \(configuration.currentVersion), available \(version)")
        let hasVerifiedInstaller = release.installerAsset != nil && release.checksumAsset != nil
        let alert = NSAlert()
        alert.alertStyle = .informational
        alert.messageText = "AgentTrainer \(version) is available"
        let notes = release.body?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let shortenedNotes = notes.count > 700 ? String(notes.prefix(700)) + "…" : notes
        alert.informativeText = "You have version \(configuration.currentVersion)." +
            (shortenedNotes.isEmpty ? "" : "\n\n\(shortenedNotes)") +
            (hasVerifiedInstaller ? "\n\nAgentTrainer will download, verify, install, and restart automatically." : "\n\nThis release has no verified disk image, so its GitHub page will open instead.")
        alert.addButton(withTitle: hasVerifiedInstaller ? "Update Now" : "View Release")
        alert.addButton(withTitle: "Not Now")
        guard alert.runModal() == .alertFirstButtonReturn else { return }

        do {
            if hasVerifiedInstaller {
                activityStatus = "Downloading AgentTrainer \(version)…"
                appUpdateProgress = AppUpdateProgress(detail: "Preparing secure download…", fraction: 0.01)
                let publishProgress: @Sendable (Double, String) -> Void = { [weak self] fraction, detail in
                    Task { @MainActor [weak self] in
                        self?.appUpdateProgress = AppUpdateProgress(detail: detail, fraction: fraction)
                    }
                }
                let prepared = try await updater.prepareUpdate(
                    for: release,
                    targetApplicationURL: Bundle.main.bundleURL.standardizedFileURL,
                    progress: publishProgress
                )
                appUpdateProgress = AppUpdateProgress(detail: "Preparing transactional replacement…", fraction: 0.96)
                let installer = try await SelfUpdateInstaller.prepare(prepared)
                appUpdateProgress = AppUpdateProgress(detail: "Restarting with AgentTrainer \(version)…", fraction: 0.99)
                try await installer.launch()
                appUpdateProgress = AppUpdateProgress(detail: "Update ready — restarting now…", fraction: 1)
                activityStatus = "Installing AgentTrainer \(version) and restarting"
                AppLog.write(category: "Updates", "Verified update ready to install", details: "\(configuration.currentVersion) → \(version)")
                try? await Task.sleep(for: .milliseconds(350))
                NSApplication.shared.terminate(nil)
            } else {
                NSWorkspace.shared.open(release.htmlURL)
            }
        } catch {
            appUpdateProgress = nil
            activityStatus = "Update failed; the current app was left unchanged"
            AppLog.write(.error, category: "Updates", "Automatic update failed", details: error.localizedDescription)
            present(error)
        }
    }

    func refreshSources() async {
        do {
            captureSources = try await CaptureService.availableSources()
            if selectedSource == nil { selectedSourceID = captureSources.first(where: sourceMatchesKind)?.id }
        } catch {
            captureSources = []
            activityStatus = "Grant Screen Recording permission in Settings"
            AppLog.write(.warning, category: "Capture", "Capture sources unavailable", details: error.localizedDescription)
        }
    }

    func refreshLibrary() async {
        let defaultFolderID = try? await WorkspaceStore.shared.normalizeRecordingFolders()
        recordings = await WorkspaceStore.shared.listRecordings()
        recordingFolders = await WorkspaceStore.shared.listRecordingFolders()
        if recordingDestinationFolderID == nil || !recordingFolders.contains(where: { $0.id == recordingDestinationFolderID }) {
            recordingDestinationFolderID = defaultFolderID ?? recordingFolders.first?.id
        }
        profiles = await WorkspaceStore.shared.listProfiles()
        await refreshStorageState()
        if selectedRecordingID == nil || !recordings.contains(where: { $0.id == selectedRecordingID }) { selectedRecordingID = recordings.first?.id }
        if selectedProfileID == nil || !profiles.contains(where: { $0.id == selectedProfileID }) { selectedProfileID = profiles.first?.id }
        if let loaded = versionsLoadedForProfileID {
            if loaded == selectedProfileID { await refreshVersions() }
            else { unloadVersions() }
        }
    }

    func refreshVersions() async {
        if let selectedProfileID {
            isLoadingVersions = true
            versions = await WorkspaceStore.shared.listVersions(profileID: selectedProfileID)
            versionsLoadedForProfileID = selectedProfileID
            isLoadingVersions = false
        } else {
            unloadVersions()
        }
    }

    func unloadVersions() {
        versions = []
        versionsLoadedForProfileID = nil
        isLoadingVersions = false
    }

    func refreshPermissions() {
        screenPermission = CGPreflightScreenCaptureAccess()
        accessibilityPermission = AXIsProcessTrusted()
        inputPermission = InputCaptureServicePermissionProbe.canCreateEventTap()
    }

    func startRecording() async {
        guard !isRecording, !isStartingRecording, !isRunning else { return }
        guard let source = selectedSource else { present(AgentTrainerError.invalidConfiguration("Select a capture source.")); return }
        guard let destinationFolderID = recordingDestinationFolderID else { present(AgentTrainerError.invalidConfiguration("Create or select a recording folder.")); return }
        isStartingRecording = true
        recordingLaunchRevision &+= 1
        let launchToken = recordingLaunchRevision
        defer { isStartingRecording = false }
        do {
            let spec = captureSpec(source: source), captureRect = effectiveCaptureRect(source)
            let excludedKeys = recordingExcludedKeyCodes
            let recordingShortcut = hotkeys.record
            let id = UUID()
            let directory = try await WorkspaceStore.shared.createRecordingDirectory(id: id)
            guard recordingLaunchRevision == launchToken else { throw CancellationError() }
            let writer = try InputEventWriter(url: directory.appendingPathComponent("events.atrevents"))
            recordingID = id; recordingDirectory = directory; eventWriter = writer; activeRecordingSpec = spec; activeRecordingRect = captureRect; activeRecordingFolderID = destinationFolderID; activeRecordingExcludedKeyCodes = excludedKeys
            let clock = RecordingClock()
            recordingClock = clock; recordingHostStart = 0
            let eventClock = RecordingClock(); lastEventClock = eventClock
            input.excludedKeyCodes = excludedKeys
            input.onSample = { [weak writer] sample in
                guard let hostStart = clock.value, sample.timestampNanos >= hostStart else { return }
                writer?.append(sample)
                eventClock.update(sample.timestampNanos)
            }
            input.onState = { [weak self] state in Task { @MainActor in self?.hudModel.update(state: state, source: .human) } }
            try input.start()
            guard recordingLaunchRevision == launchToken else { throw CancellationError() }
            try await capture.start(spec: spec, recordingURL: directory.appendingPathComponent("capture.mov"), onFirstFrame: { [weak self, weak writer] nanos in
                let pointer = CGEvent(source: nil)?.location ?? CGPoint(x: captureRect.midX, y: captureRect.midY)
                // The global Record chord is commonly still being released on
                // the first captured frame. Do not seed those shortcut
                // modifiers as demonstrated controls; later real input events
                // restore any modifier the user intentionally keeps held.
                let initialModifiers = CGEventSource.flagsState(.combinedSessionState).rawValue & ~recordingShortcut.cgEventModifiers
                var initialFilter = RecordingKeyFilter(excludedKeyCodes: excludedKeys)
                let initialSample = initialFilter.process(InputSample(
                    timestampNanos: nanos,
                    kind: .mouseMove,
                    x: pointer.x,
                    y: pointer.y,
                    modifiers: initialModifiers
                ))
                if let initialSample { writer?.append(initialSample); eventClock.update(nanos) }
                clock.set(nanos)
                Task { @MainActor [weak self] in self?.recordingHostStart = nanos }
            })
            guard recordingLaunchRevision == launchToken else { throw CancellationError() }
            hudModel.show(source: .human, vision: false)
            isRecording = true; activityStatus = "Recording — live keyboard is capture-excluded"
            AppLog.write(category: "Recording", "Recording started", details: "\(spec.kind.rawValue), \(Int(captureRect.width))×\(Int(captureRect.height)) at \(captureFPS.formatted()) FPS")
        } catch {
            _ = try? await capture.stop()
            input.stop(); input.onSample = nil; input.onState = nil; input.excludedKeyCodes = []; hudModel.hide(); _ = try? eventWriter?.finish(); eventWriter = nil
            if let recordingDirectory { try? FileManager.default.removeItem(at: recordingDirectory) }
            self.recordingDirectory = nil; recordingID = nil; activeRecordingSpec = nil; activeRecordingRect = nil; activeRecordingFolderID = nil; activeRecordingExcludedKeyCodes = []
            if error is CancellationError { activityStatus = "Recording start cancelled" }
            else { present(error) }
        }
    }

    func stopRecording() async {
        if isStartingRecording, !isRecording {
            recordingLaunchRevision &+= 1
            activityStatus = "Cancelling recording start…"
            return
        }
        guard isRecording, let directory = recordingDirectory, let id = recordingID, let captureSpec = activeRecordingSpec, let captureRect = activeRecordingRect, let destinationFolderID = activeRecordingFolderID else { return }
        input.stop(); input.onSample = nil; input.onState = nil; input.excludedKeyCodes = []; hudModel.hide()
        let eventResult = Result { try eventWriter?.finish() ?? 0 }
        eventWriter = nil
        do {
            let result = try await capture.stop()
            let eventCount = try eventResult.get()
            let hostStart = recordingClock.value ?? (result.firstFrameHostNanos > 0 ? result.firstFrameHostNanos : recordingHostStart)
            guard hostStart > 0 else { throw AgentTrainerError.capture("No complete screen frame was received; the recording was not saved.") }
            let inputDuration = lastEventClock.value.flatMap { $0 >= hostStart ? Double($0 - hostStart) / 1e9 : nil } ?? 0
            let duration = max(result.duration, inputDuration)
            var trimStart = min(duration, max(0, recordingTrimStart))
            var trimEnd = max(trimStart, duration - max(0, recordingTrimEnd))
            if trimEnd <= trimStart { trimStart = 0; trimEnd = duration }
            let manifest = RecordingManifest(id: id, name: "Recording \(Date().formatted(date: .abbreviated, time: .shortened))", createdAt: Date(), hostStartNanos: hostStart, duration: duration, capture: captureSpec, globalRect: CodableRect(captureRect), pixelWidth: result.width, pixelHeight: result.height, deliveredFPS: result.deliveredFPS, eventCount: eventCount, trimStart: trimStart, trimEnd: trimEnd, folderID: destinationFolderID, thumbnailFile: "thumbnail.jpg", excludedKeyCodes: activeRecordingExcludedKeyCodes)
            try await WorkspaceStore.shared.writeRecording(manifest, to: directory)
            await createThumbnail(for: directory.appendingPathComponent("capture.mov"), at: max(0, min(duration * 0.25, 2)), destination: directory.appendingPathComponent("thumbnail.jpg"))
            activityStatus = "Recording saved"
            AppLog.write(category: "Recording", "Recording saved", details: "\(eventCount) input events, \(result.width)×\(result.height), \(duration.formatted(.number.precision(.fractionLength(2)))) seconds")
        } catch { try? FileManager.default.removeItem(at: directory); present(error) }
        isRecording = false; recordingDirectory = nil; recordingID = nil; activeRecordingSpec = nil; activeRecordingRect = nil; activeRecordingFolderID = nil; activeRecordingExcludedKeyCodes = []; recordingClock = RecordingClock(); lastEventClock = RecordingClock(); await refreshLibrary()
    }

    func deleteRecording(_ item: RecordingItem) async {
        do {
            let nextSelection = recordings.first(where: { $0.id != item.id })?.id
            try await WorkspaceStore.shared.deleteRecording(item)
            if selectedRecordingID == item.id { selectedRecordingID = nextSelection }
            await refreshLibrary()
            activityStatus = "Recording deleted"
        } catch { present(error) }
    }
    func renameRecording(_ item: RecordingItem, name: String) async { do { try await WorkspaceStore.shared.renameRecording(item, to: name); await refreshLibrary() } catch { present(error) } }
    func moveRecording(_ item: RecordingItem, to folderID: UUID?) async { do { try await WorkspaceStore.shared.assignRecording(item, to: folderID); await refreshLibrary() } catch { present(error) } }
    func createRecordingFolder(name: String = "New Folder") async {
        do { let folder = RecordingFolder(id: UUID(), name: name, createdAt: Date()); try await WorkspaceStore.shared.saveRecordingFolder(folder); recordingDestinationFolderID = folder.id; await refreshLibrary() } catch { present(error) }
    }
    func renameRecordingFolder(_ folder: RecordingFolder, name: String) async { do { var changed = folder; changed.name = name; try await WorkspaceStore.shared.saveRecordingFolder(changed); await refreshLibrary() } catch { present(error) } }
    func deleteRecordingFolder(_ folder: RecordingFolder, includingRecordings: Bool = true) async { do { try await WorkspaceStore.shared.deleteRecordingFolder(folder, includingRecordings: includingRecordings); if recordingDestinationFolderID == folder.id { recordingDestinationFolderID = nil }; await refreshLibrary() } catch { present(error) } }

    func createProfile(name: String) {
        Task {
            do { let profile = AIProfile.fresh(name: name); try await WorkspaceStore.shared.saveProfile(profile); unloadVersions(); await refreshLibrary(); selectedProfileID = profile.id }
            catch { present(error) }
        }
    }

    func saveProfile(_ profile: AIProfile) {
        do { try validateProfile(profile) } catch { present(error); return }
        Task { do { try await WorkspaceStore.shared.saveProfile(profile); await refreshLibrary(); selectedProfileID = profile.id } catch { present(error) } }
    }
    func scheduleProfileAutosave(_ profile: AIProfile) {
        profileAutosaveTask?.cancel()
        if let index = profiles.firstIndex(where: { $0.id == profile.id }) { profiles[index] = profile }
        selectedProfileID = profile.id
        profileAutosaveTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(350))
            guard !Task.isCancelled, let self else { return }
            do {
                try self.validateProfile(profile)
                try await WorkspaceStore.shared.saveProfile(profile)
                self.activityStatus = "Model settings autosaved"
            } catch { /* transient invalid text-field edits are not committed */ }
        }
    }
    func duplicateProfile(_ profile: AIProfile) async {
        profileAutosaveTask?.cancel()
        do {
            let copy = try await WorkspaceStore.shared.duplicateProfile(profile)
            unloadVersions()
            await refreshLibrary()
            selectedProfileID = copy.id
            activityStatus = "Duplicated \(profile.name) with its active brain, training progress, and resumable checkpoint"
            AppLog.write(category: "Models", "AI duplicated with learned brain", details: "\(profile.name) → \(copy.name), step \(copy.trainingProgress?.globalStep ?? 0), epoch \(copy.trainingProgress?.epoch ?? 0)")
        } catch { present(error) }
    }

    @discardableResult
    func resetLearningAndSave(_ profile: AIProfile) async -> AIProfile? {
        profileAutosaveTask?.cancel()
        do {
            try validateProfile(profile)
            let reset = try await WorkspaceStore.shared.resetLearning(for: profile)
            unloadVersions()
            await refreshLibrary()
            selectedProfileID = reset.id
            activityStatus = "Architecture changed — previous training was cleared and this AI is ready for a fresh brain"
            AppLog.write(.warning, category: "Models", "Learned brain reset after confirmed architecture change", details: reset.name)
            return profiles.first(where: { $0.id == reset.id }) ?? reset
        } catch {
            present(error)
            return nil
        }
    }
    func deleteProfile(_ profile: AIProfile) async { profileAutosaveTask?.cancel(); do { try await WorkspaceStore.shared.deleteProfile(profile); selectedProfileID = nil; unloadVersions(); await refreshLibrary() } catch { present(error) } }

    func activateVersion(_ version: ModelVersionManifest) async {
        guard var profile = selectedProfile else { return }
        profile.activeVersionID = version.id
        do {
            let resumable = try await WorkspaceStore.shared.restoreVersionAsCheckpoint(profileID: profile.id, version: version)
            let count = await WorkspaceStore.shared.listVersions(profileID: profile.id).count
            profile.trainingProgress = TrainingProgressSummary(
                globalStep: version.globalStep,
                epoch: version.epoch ?? 0,
                updatedAt: version.createdAt,
                savedBrainCount: count,
                trainingDurationSeconds: version.trainingDurationSeconds,
                experienceDurationSeconds: version.experienceDurationSeconds
            )
            try await WorkspaceStore.shared.saveProfile(profile); await refreshLibrary(); selectedProfileID = profile.id
            activityStatus = resumable ? "Brain reverted to \(version.name); training will resume there" : "Activated \(version.name) for running"
        } catch { present(error) }
    }

    func deleteVersion(_ version: ModelVersionManifest) async {
        guard var profile = selectedProfile else { return }
        do {
            try await WorkspaceStore.shared.deleteVersion(profile: profile, versionID: version.id)
            if profile.activeVersionID == version.id { profile.activeVersionID = nil; try await WorkspaceStore.shared.saveProfile(profile) }
            if var progress = profile.trainingProgress {
                progress.savedBrainCount = await WorkspaceStore.shared.listVersions(profileID: profile.id).count
                profile.trainingProgress = progress
                try await WorkspaceStore.shared.saveProfile(profile)
            }
            await refreshLibrary(); activityStatus = "Model version deleted"
        } catch { present(error) }
    }

    func startTraining() {
        guard let profile = selectedProfile, !isTraining else { return }
        guard runningProfileID != profile.id else { present(AgentTrainerError.model("This AI is currently running. Select a different AI to train in the background.")); return }
        guard trainingRunSettings.maximumSteps >= 0,
              (1...100_000_000).contains(trainingRunSettings.autosaveSteps) else {
            present(AgentTrainerError.invalidConfiguration("Maximum Steps must be zero or greater, and Autosave Steps must be from 1 through 100,000,000."))
            return
        }
        do { try validateProfile(profile) } catch { present(error); return }
        let folderIDs = Set(profile.effectiveFolderIDs)
        let selected = recordings.filter { profile.recordingIDs.contains($0.id) || $0.manifest.folderID.map(folderIDs.contains) == true }
        guard !selected.isEmpty else { present(AgentTrainerError.noData); return }
        isTraining = true; trainingProfileID = profile.id; trainingStatus = "Preparing packed dataset cache for \(profile.name)"; activityStatus = trainingStatus
        AppLog.write(category: "Training", "Training started", details: "\(profile.name), \(selected.count) recordings, batch \(profile.training.batchSize), \(profile.preprocessing.width)×\(profile.preprocessing.height)")
        training.start(profile: profile, recordings: selected, runSettings: trainingRunSettings) { [weak self] value, status in
            Task { @MainActor in self?.trainingMetrics = value; self?.trainingStatus = status; self?.activityStatus = self?.isRunning == true ? "AI running • \(status)" : status }
        } completion: { [weak self] result in
            Task { @MainActor in
                guard let self else { return }; self.isTraining = false; self.trainingProfileID = nil
                switch result {
                case .success(let completion): self.trainingStatus = completion.completed ? "Training complete — runnable brain saved" : "Paused — current brain is ready to run and resume"; self.activityStatus = self.trainingStatus; AppLog.write(category: "Training", self.trainingStatus, details: "step \(completion.version.globalStep), loss \(completion.version.trainingLoss)"); await self.refreshLibrary()
                case .failure(let error): self.trainingStatus = error.localizedDescription; self.activityStatus = error.localizedDescription; if !error.localizedDescription.localizedCaseInsensitiveContains("paused") && !error.localizedDescription.localizedCaseInsensitiveContains("stopped") { self.present(error) }
                }
            }
        }
    }

    func pauseTraining() { training.pauseAndSave(); trainingStatus = "Saving exact training state…"; activityStatus = trainingStatus }
    func stopTraining() { training.stop(); trainingStatus = "Stopping training safely…"; activityStatus = trainingStatus }

    func startAgent() async {
        guard let profile = selectedProfile, let versionID = profile.activeVersionID,
              let source = selectedSource, !isRecording, !isRunning, agent == nil else { present(AgentTrainerError.model("Select a trained AI and a live capture source, and stop any current AI first.")); return }
        guard trainingProfileID != profile.id else { present(AgentTrainerError.model("This AI is currently training. Select another trained AI to run at the same time.")); return }
        var attemptedRuntime: AgentRuntime?
        do {
            guard let version = await WorkspaceStore.shared.version(profileID: profile.id, versionID: versionID) else {
                throw AgentTrainerError.model("The active runnable brain is missing. Load Saved Brains in AI Models and choose another version.")
            }
            let runtime = try AgentRuntime()
            attemptedRuntime = runtime
            runtime.onState = { [weak self] state in Task { @MainActor in self?.hudModel.update(state: state, source: .agent) } }
            runtime.onMetrics = { [weak self] value in Task { @MainActor in self?.runtimeMetrics = value } }
            runtime.onPreview = { [weak self] frame in Task { @MainActor in self?.hudModel.updateVision(frame) } }
            runtime.onVisualization = { [weak self] frame in Task { @MainActor in self?.hudModel.updateCNNVisualization(frame) } }
            runtime.onStop = { [weak self, weak runtime] reason in Task { @MainActor in
                guard let self, let runtime, self.agent === runtime else { return }
                self.agent = nil; self.isRunning = false; self.runningProfileID = nil; self.hudModel.hide(); self.runtimeStatus = reason ?? "Agent stopped"; self.activityStatus = self.runtimeStatus
                AppLog.write(reason == nil ? .info : .warning, category: "Runtime", self.runtimeStatus)
            } }
            agent = runtime; isRunning = true; runningProfileID = profile.id; hudModel.show(source: .agent, vision: showVisionPreview, cnnVisualization: cnnVisualizationSettings)
            runtimeStatus = "Agent starting at exactly \(version.preprocessing.width) × \(version.preprocessing.height)"; activityStatus = runtimeStatus
            var runtimeSafety = safety
            runtimeSafety.panicKeyCode = UInt16(clamping: hotkeys.panic.keyCode)
            runtimeSafety.panicModifiers = hotkeys.panic.cgEventModifiers
            let previewRate = visionPreviewMatchesPerception ? version.training.perceptionFPS : visionPreviewFPS
            let resolvedMouseMode: MouseControlMode
            if runMouseMode != .automatic {
                resolvedMouseMode = runMouseMode
            } else if let recommended = version.recommendedMouseMode {
                resolvedMouseMode = recommended
            } else {
                resolvedMouseMode = await self.resolvedMouseMode(for: profile)
            }
            let allowedKeyCodes: Set<UInt16>
            if let persisted = version.demonstratedKeyCodes { allowedKeyCodes = persisted }
            else { allowedKeyCodes = await demonstratedKeyCodes(for: profile) }
            try await runtime.start(profile: profile, version: version, allowedKeyCodes: allowedKeyCodes, captureSpec: captureSpec(source: source), captureRect: effectiveCaptureRect(source), mode: frameMode, mouseMode: resolvedMouseMode, gameCamera: gameCamera, outputPermissions: runtimeOutputPermissions, safety: runtimeSafety, previewFPS: showVisionPreview ? previewRate : 0, visualizationSettings: cnnVisualizationSettings, ignoredHotkeys: [hotkeys.panic, hotkeys.record, hotkeys.run])
            if isRunning, agent === runtime { runtimeStatus = "Agent running locally • \(resolvedMouseMode.rawValue)"; activityStatus = isTraining ? "AI running • \(trainingStatus)" : runtimeStatus; AppLog.write(category: "Runtime", "Agent started", details: "\(profile.name), \(resolvedMouseMode.rawValue), \(allowedKeyCodes.count) allowed keys, cursor \(runtimeOutputPermissions.cursorMovement ? "enabled" : "disabled"), keyboard \(runtimeOutputPermissions.keyboard ? "enabled" : "disabled"), CNN diagnostics \(cnnVisualizationSettings.enabled ? cnnVisualizationSettings.mode.rawValue : "disabled")") }
        } catch is CancellationError {
            guard attemptedRuntime == nil || agent === attemptedRuntime || agent == nil else { return }
            isRunning = false; runningProfileID = nil; hudModel.hide(); if agent === attemptedRuntime { agent = nil }; runtimeStatus = "Agent start cancelled"; activityStatus = runtimeStatus
        }
        catch {
            guard attemptedRuntime == nil || agent === attemptedRuntime || agent == nil else { return }
            isRunning = false; runningProfileID = nil; hudModel.hide(); if agent === attemptedRuntime { agent = nil }; runtimeStatus = error.localizedDescription; present(error)
        }
    }

    func stopAgent() async { let runtime = agent; agent = nil; await runtime?.stop(reason: "Agent stopped"); isRunning = false; runningProfileID = nil; runtimeMetrics = RuntimeMetrics(); hudModel.hide(); runtimeStatus = "Agent stopped; all hooks disabled and held inputs released"; activityStatus = isTraining ? trainingStatus : runtimeStatus; if runtime != nil { AppLog.write(category: "Runtime", "Agent stopped and input hooks released") } }
    func startReenactment() {
        guard let recording = selectedRecording else { return }
        do { try reenactor.start(recording: recording); isReplaying = true; activityStatus = "Guarded reenactment running"; AppLog.write(category: "Replay", "Reenactment started", details: recording.manifest.name) } catch { present(error) }
    }
    func stopReenactment() { reenactor.stop(); if isReplaying { AppLog.write(category: "Replay", "Reenactment stopped") }; isReplaying = false; activityStatus = "Reenactment stopped" }
    func panic() { AppLog.write(.warning, category: "Safety", "Panic stop requested"); Task { stopReenactment(); await stopAgent(); if recordingIsActiveOrStarting { await stopRecording() }; if isTraining { stopTraining() } } }
    func clearCaches() async { do { try await WorkspaceStore.shared.clearCaches(); await refreshStorageState(); activityStatus = "Dataset caches cleared"; AppLog.write(category: "Storage", "Dataset caches cleared") } catch { present(error) } }

    func chooseStorageLocation(_ kind: WorkspaceDataKind) async {
        guard canChangeStorageLocations else {
            present(AgentTrainerError.storage("Stop recording, training, running, and replay before changing storage locations."))
            return
        }
        let current = kind == .trainingData ? storageLocations.trainingDataRoot : storageLocations.modelsRoot
        let panel = NSOpenPanel()
        panel.title = "Choose \(kind.rawValue) Location"
        panel.message = kind == .trainingData
            ? "Choose the folder that will contain Recordings, Caches, and the recording-folder index. Existing data is safely copied and verified before the app switches."
            : "Choose the folder that will contain Profiles, checkpoints, and saved runnable brains. Existing models are safely copied and verified before the app switches."
        panel.prompt = "Choose Location"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        if FileManager.default.fileExists(atPath: current.path) { panel.directoryURL = current }
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        await changeStorageLocation(kind, to: destination)
    }

    func restoreDefaultStorageLocation(_ kind: WorkspaceDataKind) async {
        guard canChangeStorageLocations else {
            present(AgentTrainerError.storage("Stop recording, training, running, and replay before changing storage locations."))
            return
        }
        await changeStorageLocation(kind, to: storageLocations.supportRoot)
    }

    func revealStorageLocation(_ kind: WorkspaceDataKind) {
        let location = kind == .trainingData ? storageLocations.trainingDataRoot : storageLocations.modelsRoot
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: location.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            present(AgentTrainerError.storage("The saved \(kind.rawValue.lowercased()) location is unavailable. Reconnect its disk or choose another folder."))
            return
        }
        NSWorkspace.shared.activateFileViewerSelecting([location])
    }

    private func changeStorageLocation(_ kind: WorkspaceDataKind, to destination: URL) async {
        do {
            let inspection = try await WorkspaceStore.shared.inspectDestination(destination, for: kind)
            if inspection.isCurrentLocation {
                activityStatus = "\(kind.rawValue) already uses that location"
                return
            }
            var useExisting = false
            if inspection.containsManagedData {
                let alert = NSAlert()
                alert.alertStyle = .informational
                alert.messageText = "Use the existing \(kind.rawValue.lowercased())?"
                alert.informativeText = "This folder already contains an AgentTrainer \(kind == .trainingData ? "recording library" : "model library"). AgentTrainer can switch to it without merging or deleting the library at the current location."
                alert.addButton(withTitle: "Use Existing Library")
                alert.addButton(withTitle: "Cancel")
                guard alert.runModal() == .alertFirstButtonReturn else { return }
                useExisting = true
            }

            profileAutosaveTask?.cancel()
            isChangingStorageLocation = true
            activityStatus = useExisting ? "Switching \(kind.rawValue.lowercased())…" : "Copying and verifying \(kind.rawValue.lowercased())…"
            let result = try await WorkspaceStore.shared.relocate(kind, to: inspection.url, useExisting: useExisting)
            if kind == .trainingData {
                recordingDestinationFolderID = nil
                selectedRecordingID = nil
                _ = try await WorkspaceStore.shared.repairInvalidRecordingManifests()
                _ = try await WorkspaceStore.shared.removeObsoleteCaches(currentSchema: TrainingDataContract.schemaVersion)
                _ = try await WorkspaceStore.shared.normalizeRecordingFolders()
            } else {
                selectedProfileID = nil
                unloadVersions()
                _ = try await WorkspaceStore.shared.removeObsoleteModelArtifacts(currentSchema: ModelContract.schemaVersion)
                _ = try await WorkspaceStore.shared.pruneAllAutosaves(keeping: 10)
            }
            await refreshLibrary()
            let action = result.movedExistingData ? "moved" : "switched"
            activityStatus = "\(kind.rawValue) \(action) to \(result.destination.path)"
            if result.sourceCleanupComplete {
                AppLog.write(category: "Storage", "\(kind.rawValue) location changed", details: "\(action) to \(result.destination.path)")
            } else {
                AppLog.write(.warning, category: "Storage", "\(kind.rawValue) location changed with an old copy retained", details: result.destination.path)
                errorMessage = "\(kind.rawValue) is now using the new location, but an old copy could not be removed. Your data is safe; you may delete the old copy manually after verifying the new library."
            }
            isChangingStorageLocation = false
        } catch {
            isChangingStorageLocation = false
            await refreshStorageState()
            present(error)
        }
    }

    private func refreshStorageState() async {
        storageLocations = await WorkspaceStore.shared.locations()
        storageUsage = await WorkspaceStore.shared.storageUsage()
        storageBytes = storageUsage.totalBytes
    }

    func saveHotkeys(_ value: HotkeySettings) {
        guard Set([value.panic, value.record, value.run]).count == 3 else { resumeGlobalHotkeys(); present(AgentTrainerError.invalidConfiguration("Panic, recording, and agent shortcuts must be different.")); return }
        let previous = hotkeys
        hotkeys = value
        panicHotkey.update(value.panic); recordHotkey.update(value.record); runHotkey.update(value.run)
        let statuses = [panicHotkey.registrationStatus, recordHotkey.registrationStatus, runHotkey.registrationStatus]
        if let failure = statuses.first(where: { $0 != GlobalHotkeyMonitor.successStatus }) {
            hotkeys = previous
            panicHotkey.update(previous.panic); recordHotkey.update(previous.record); runHotkey.update(previous.run)
            present(AgentTrainerError.invalidConfiguration("macOS could not register that global shortcut (error \(failure)). The previous shortcuts were restored."))
        } else {
            if let data = try? JSONEncoder().encode(value) { UserDefaults.standard.set(data, forKey: "AgentTrainer.Hotkeys") }
            input.ignoredHotkeys = [value.panic, value.record, value.run]
            activityStatus = "Global shortcuts updated"
        }
    }

    func suspendGlobalHotkeys() { panicHotkey.stop(); recordHotkey.stop(); runHotkey.stop() }
    func resumeGlobalHotkeys() { panicHotkey.start(); recordHotkey.start(); runHotkey.start() }

    func selectScreenRegion() {
        guard captureKind == .screenRegion, let source = selectedSource else { return }
        regionSelector.select(on: source.frame) { [weak self] rect in
            guard let self, let rect else { return }
            self.regionX = rect.minX; self.regionY = rect.minY; self.regionWidth = rect.width; self.regionHeight = rect.height
        }
    }

    func openPrivacyPane(_ anchor: String) { if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?\(anchor)") { NSWorkspace.shared.open(url) } }

    private func captureSpec(source: CaptureSourceOption) -> CaptureSpec {
        var spec = CaptureSpec(kind: captureKind, requestedFPS: captureFPS, showsCursor: showsCursor)
        if source.kind == .display { spec.displayID = source.id } else { spec.windowID = source.id }
        if captureKind == .screenRegion || captureKind == .windowRegion { spec.region = CodableRect(CGRect(x: regionX, y: regionY, width: regionWidth, height: regionHeight)) }
        return spec
    }

    private func effectiveCaptureRect(_ source: CaptureSourceOption) -> CGRect {
        if captureKind == .screenRegion { return CGRect(x: regionX, y: regionY, width: regionWidth, height: regionHeight).intersection(source.frame) }
        if captureKind == .windowRegion { return CGRect(x: source.frame.minX + regionX, y: source.frame.minY + regionY, width: regionWidth, height: regionHeight).intersection(source.frame) }
        return source.frame
    }

    private func sourceMatchesKind(_ source: CaptureSourceOption) -> Bool {
        switch captureKind { case .display, .screenRegion: source.kind == .display; case .window, .windowRegion: source.kind == .window }
    }

    private func present(_ error: Error) { errorMessage = error.localizedDescription; activityStatus = error.localizedDescription; AppLog.write(.error, category: "Application", error.localizedDescription, details: String(reflecting: error)) }

    private func createThumbnail(for video: URL, at seconds: Double, destination: URL) async {
        await Task.detached(priority: .utility) {
            let generator = AVAssetImageGenerator(asset: AVURLAsset(url: video))
            generator.appliesPreferredTrackTransform = true
            generator.maximumSize = CGSize(width: 420, height: 240)
            let image: CGImage? = await withCheckedContinuation { continuation in
                generator.generateCGImageAsynchronously(for: CMTime(seconds: seconds, preferredTimescale: 600)) { image, _, _ in continuation.resume(returning: image) }
            }
            guard let image, let data = NSBitmapImageRep(cgImage: image).representation(using: .jpeg, properties: [.compressionFactor: 0.78]) else { return }
            try? data.write(to: destination, options: .atomic)
        }.value
    }

    private func ensureRecordingThumbnails() async {
        var changed = false
        for item in recordings {
            let destination = item.directory.appendingPathComponent("thumbnail.jpg")
            guard !FileManager.default.fileExists(atPath: destination.path) else { continue }
            await createThumbnail(for: item.directory.appendingPathComponent(item.manifest.videoFile), at: max(0, min(item.manifest.duration * 0.25, 2)), destination: destination)
            guard FileManager.default.fileExists(atPath: destination.path) else { continue }
            var manifest = item.manifest; manifest.thumbnailFile = "thumbnail.jpg"
            try? await WorkspaceStore.shared.writeRecording(manifest, to: item.directory)
            changed = true
        }
        if changed { recordings = await WorkspaceStore.shared.listRecordings() }
    }

    private func demonstratedKeyCodes(for profile: AIProfile) async -> Set<UInt16> {
        let folderIDs = Set(profile.effectiveFolderIDs)
        let selected = recordings.filter { profile.recordingIDs.contains($0.id) || $0.manifest.folderID.map(folderIDs.contains) == true }
        return await Task.detached(priority: .userInitiated) {
            selected.reduce(into: Set<UInt16>()) { result, item in
                let url = item.directory.appendingPathComponent(item.manifest.eventFile)
                if let keys = try? InputEventReader.demonstratedKeyCodes(url: url) { result.formUnion(keys) }
            }
        }.value
    }

    private func resolvedMouseMode(for profile: AIProfile) async -> MouseControlMode {
        guard runMouseMode == .automatic else { return runMouseMode }
        let folderIDs = Set(profile.effectiveFolderIDs)
        let selected = recordings.filter { profile.recordingIDs.contains($0.id) || $0.manifest.folderID.map(folderIDs.contains) == true }
        return await Task.detached(priority: .userInitiated) {
            var cameraRecordings = 0
            var cursorRecordings = 0
            for item in selected {
                let url = item.directory.appendingPathComponent(item.manifest.eventFile)
                guard let summary = try? InputEventReader.summarize(url: url, previewLimit: 0, globalRect: item.manifest.globalRect.cgRect), summary.mouse.moveEventCount > 0 else { continue }
                if summary.mouse.isGameCamera { cameraRecordings += 1 } else { cursorRecordings += 1 }
            }
            return cameraRecordings > cursorRecordings ? .relative : .absolute
        }.value
    }

    private func repairLegacyRecordingClocks() async {
        var changed = false
        for item in recordings where item.manifest.hostStartNanos == 0 || item.manifest.duration > 24 * 60 * 60 {
            let asset = AVURLAsset(url: item.directory.appendingPathComponent(item.manifest.videoFile))
            guard let time = try? await asset.load(.duration) else { continue }
            let videoDuration = CMTimeGetSeconds(time)
            guard videoDuration.isFinite, videoDuration > 0 else { continue }
            var manifest = item.manifest
            let events = (try? InputEventReader.read(url: item.directory.appendingPathComponent(item.manifest.eventFile))) ?? []
            if manifest.hostStartNanos == 0, let first = events.first {
                manifest.hostStartNanos = first.timestampNanos
                manifest.trimStart = 0
                manifest.trimEnd = events.last.map { min(videoDuration, Double($0.timestampNanos - first.timestampNanos) / 1e9 + 0.25) }
            }
            manifest.duration = videoDuration
            if let trimEnd = manifest.trimEnd { manifest.trimEnd = min(videoDuration, trimEnd) }
            try? await WorkspaceStore.shared.writeRecording(manifest, to: item.directory)
            changed = true
        }
        if changed {
            try? await WorkspaceStore.shared.clearCaches()
            recordings = await WorkspaceStore.shared.listRecordings()
            await refreshStorageState()
            activityStatus = "Repaired legacy recording clocks and cleared inflated caches"
        }
    }

    private func validateProfile(_ profile: AIProfile) throws {
        _ = try profile.preprocessing.validated()
        let training = profile.training
        guard training.learningRate.isFinite, training.weightDecay.isFinite,
              training.perceptionFPS.isFinite, training.actionFPS.isFinite,
              training.validationSplit.isFinite,
              (1...1_000_000).contains(training.epochs),
              (1...4_096).contains(training.batchSize),
              (0...256).contains(training.historyLength),
              training.learningRate >= 0.000_000_1, training.learningRate <= 0.003,
              training.weightDecay >= 0, training.weightDecay <= 1,
              training.perceptionFPS > 0, training.actionFPS > 0,
              training.perceptionFPS <= training.actionFPS,
              training.perceptionFPS <= 240, training.actionFPS <= 240,
              training.validationSplit >= 0, training.validationSplit < 1 else {
            throw AgentTrainerError.invalidConfiguration("Use bounded finite training values: learning rate 0.0000001–0.003, weight decay 0–1, history 0–256, Perception FPS no higher than Action FPS (both at most 240), and validation from 0 up to but not including 1.")
        }
        let architecture = training.architecture
        guard architecture.dropout.isFinite,
              (1...8).contains(architecture.convolutionChannels.count),
              architecture.convolutionChannels.allSatisfy({ (1...16_384).contains($0) }),
              architecture.kernelSizes.count == architecture.convolutionChannels.count,
              architecture.kernelSizes.allSatisfy({ (1...31).contains($0) }),
              architecture.strides.count == architecture.convolutionChannels.count,
              architecture.strides.allSatisfy({ (1...16).contains($0) }),
              (1...16_384).contains(architecture.visualEmbedding),
              (1...16_384).contains(architecture.recurrentWidth),
              architecture.fusionWidths.count <= 16,
              architecture.fusionWidths.allSatisfy({ (1...16_384).contains($0) }),
              architecture.dropout >= 0, architecture.dropout <= 0.5 else {
            throw AgentTrainerError.invalidConfiguration("Use 1–8 convolution stages with one kernel and stride per stage, positive bounded widths, and dropout from 0 through 0.5.")
        }
        let channels = profile.channels
        guard channels.mouseMovement || channels.buttons || channels.scroll || channels.keyboard || channels.modifiers else { throw AgentTrainerError.invalidConfiguration("Enable at least one control channel before training.") }
        let physicalMemory = Int64(ProcessInfo.processInfo.physicalMemory)
        let parameterCount = ModelSizing.parameterCount(profile)
        let estimatedWorkingSet = ModelSizing.estimatedTrainingWorkingSet(profile)
        guard parameterCount < Int64.max,
              estimatedWorkingSet < Int64.max,
              estimatedWorkingSet <= (physicalMemory / 3) * 2 else {
            throw AgentTrainerError.invalidConfiguration("This vision, batch, and architecture combination cannot fit safely in this Mac's unified memory. Reduce batch size, model vision, or network widths.")
        }
    }

    func isProfileBusy(_ id: UUID) -> Bool { trainingProfileID == id || runningProfileID == id }

    private func persistWorkflowSettings() {
        guard !isRestoringWorkflowSettings else { return }
        let value = PersistentWorkflowSettings(recordingExcludedKeyCodes: recordingExcludedKeyCodes, recordingTrimStart: recordingTrimStart, recordingTrimEnd: recordingTrimEnd, trainingRunSettings: trainingRunSettings, runMouseMode: runMouseMode, gameCamera: gameCamera, runtimeOutputPermissions: runtimeOutputPermissions, cnnVisualizationSettings: cnnVisualizationSettings)
        if let data = try? JSONEncoder().encode(value) { UserDefaults.standard.set(data, forKey: "AgentTrainer.WorkflowSettings") }
    }
}

private struct PersistentWorkflowSettings: Codable {
    var recordingExcludedKeyCodes: Set<UInt16>
    var recordingTrimStart: Double
    var recordingTrimEnd: Double
    var trainingRunSettings: TrainingRunSettings
    var runMouseMode: MouseControlMode
    var gameCamera: GameCameraSettings?
    var runtimeOutputPermissions: RuntimeOutputPermissions?
    var cnnVisualizationSettings: CNNVisualizationSettings?
}


enum InputCaptureServicePermissionProbe {
    static func canCreateEventTap() -> Bool {
        let mask = CGEventMask(1 << CGEventType.keyDown.rawValue)
        guard let tap = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap, options: .listenOnly, eventsOfInterest: mask, callback: { _, _, event, _ in Unmanaged.passUnretained(event) }, userInfo: nil) else { return false }
        CFMachPortInvalidate(tap); return true
    }
}

private final class RecordingClock: @unchecked Sendable {
    private let lock = NSLock()
    private var storedValue: UInt64?

    var value: UInt64? { lock.lock(); defer { lock.unlock() }; return storedValue }
    func set(_ value: UInt64) { lock.lock(); if storedValue == nil { storedValue = value }; lock.unlock() }
    func update(_ value: UInt64) { lock.lock(); storedValue = value; lock.unlock() }
}
