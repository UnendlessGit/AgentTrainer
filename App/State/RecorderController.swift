import AppKit
import ATCapture
import ATCore
import Foundation
import Observation
import ScreenCaptureKit

@MainActor @Observable
final class RecorderController {
    enum Phase: Equatable { case idle, countdown(Int), recording, saving }

    private let settings: AppSettings
    private let library: LibraryStore
    var targets: [CaptureTarget] = []
    var thumbnails: [String: NSImage] = [:]
    var selectedTargetID: String?
    var loadingTargets = false
    var task = ""
    var name = ""
    var destination: URL
    var phase = Phase.idle
    var stats = RecordingStats()
    var errorMessage: String?
    var lastSaved: URL?
    let hud = HUDPanelController()
    private var ownApps: [SCRunningApplication] = []
    private var session: RecordingSession?
    private var statsTimer: Timer?
    private var countdownTask: Task<Void, Never>?

    init(settings: AppSettings, library: LibraryStore) {
        self.settings = settings
        self.library = library
        destination = library.fs.root
    }

    var selectedTarget: CaptureTarget? { targets.first { $0.id == selectedTargetID } }
    /// A large, regularly refreshed still of the selected target (the Record tab's preview).
    var preview: NSImage?

    /// Refreshes `preview` about once a second while `active` returns true.
    func runPreview(active: @escaping () -> Bool) async {
        while !Task.isCancelled && active() {
            if let t = selectedTarget, let img = await CaptureTargets.thumbnail(t, own: ownApps, maxWidth: 1600) {
                preview = NSImage(cgImage: img, size: NSSize(width: img.width, height: img.height))
            }
            try? await Task.sleep(for: .seconds(1))
        }
    }
    var isBusy: Bool { phase != .idle || elkbornMode != .off }

    func refreshTargets() async {
        loadingTargets = true
        defer { loadingTargets = false }
        do {
            let r = try await CaptureTargets.load()
            targets = r.targets
            ownApps = r.ownApps
            if selectedTargetID == nil || selectedTarget == nil { selectedTargetID = targets.first?.id }
            errorMessage = nil
            for t in targets.prefix(24) {
                if let img = await CaptureTargets.thumbnail(t, own: ownApps, maxWidth: 360) {
                    thumbnails[t.id] = NSImage(cgImage: img, size: NSSize(width: img.width, height: img.height))
                }
            }
        } catch {
            errorMessage = "Couldn't list windows and displays. Check Screen Recording permission in Settings. (\(error.localizedDescription))"
        }
    }

    /// Resolution and frame rate the recording will use.
    var plannedSettings: CaptureSettings? {
        selectedTarget.map { CaptureSettings.fitting($0, maxHeight: settings.stored.captureMaxHeight, fps: settings.stored.captureFPS, showsCursor: settings.stored.showsCursor) }
    }

    func toggle() async {
        switch phase {
        case .idle: await start()
        case .countdown: countdownTask?.cancel(); phase = .idle
        case .recording: await stop()
        case .saving: break
        }
    }

    func start() async {
        guard phase == .idle, elkbornMode == .off else { return }
        if targets.isEmpty { await refreshTargets() }
        guard let target = selectedTarget, let capture = plannedSettings else {
            errorMessage = "Choose a window or display to record."
            return
        }
        let seconds = settings.stored.countdownSeconds
        if seconds > 0 {
            countdownTask = Task {
                for s in stride(from: seconds, to: 0, by: -1) {
                    phase = .countdown(s)
                    try? await Task.sleep(for: .seconds(1))
                    if Task.isCancelled { return }
                }
            }
            await countdownTask?.value
            if case .idle = phase { return }
        }
        let stamp = Date().formatted(.dateTime.year().month(.twoDigits).day(.twoDigits).hour().minute().second())
        var cfg = RecordingConfig(target: target, settings: capture, input: settings.stored.input,
                                  name: name.isEmpty ? "\(target.info.appName ?? target.name) \(stamp)" : name,
                                  task: task, destination: destination,
                                  appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev")
        cfg.videoQuality = settings.stored.videoQuality
        let st = settings.stored
        cfg.ignoredHotkeys = st.appHotkeys.map { ($0.keyCode, $0.cgFlags) }
        do {
            let s = try RecordingSession(config: cfg)
            s.onError = { [weak self] e in Task { @MainActor in self?.errorMessage = e.localizedDescription; await self?.stop() } }
            try await s.start(excluding: ownApps)
            session = s
            phase = .recording
            errorMessage = nil
            hud.show(mode: .record, options: settings.stored.recordHUD, on: target.frame) { [weak self, weak s] in
                guard let s else { return }
                self?.hud.model.push(s.live.snapshot)
                let st = s.stats
                self?.hud.model.statusLine = String(format: "%@ · %.0f fps", Duration.seconds(st.elapsed).formatted(.time(pattern: .minuteSecond)), st.averageFPS)
            }
            statsTimer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
                MainActor.assumeIsolated { if let s = self?.session { self?.stats = s.stats } }
            }
        } catch {
            phase = .idle
            errorMessage = error.localizedDescription
        }
    }

    /// Ends the recording (or countdown) without saving anything.
    func cancel() async {
        switch phase {
        case .countdown: countdownTask?.cancel(); phase = .idle
        case .recording:
            guard let s = session else { return }
            phase = .saving
            statsTimer?.invalidate()
            hud.hide()
            await s.cancel()
            session = nil
            phase = .idle
            library.refresh()
        case .idle, .saving: break
        }
    }

    // MARK: Elkborn replays

    enum ElkbornMode: Equatable { case off, listening, calibrating }
    var elkbornMode = ElkbornMode.off
    var elkbornPhase = ElkbornReplayRecorder.Phase.idle
    var elkbornStatus = ""
    var elkbornSaved = 0
    private var elkborn: ElkbornReplayRecorder?

    /// Captures the selected Roblox window and turns every Elkborn Training Data replay shown in it
    /// into a recording in the "Save to" folder (or, with `calibrate`, measures the mouse once).
    func startElkborn(calibrate: Bool) async {
        guard phase == .idle, elkbornMode == .off else { return }
        if targets.isEmpty { await refreshTargets() }
        guard let target = selectedTarget, let capture = plannedSettings else {
            errorMessage = "Choose the Roblox window first."
            return
        }
        // the folder chosen under "Save to", like ordinary recordings
        let folder = destination
        var cfg = ElkbornReplayRecorder.Config(target: target, settings: CaptureSettings(width: capture.width, height: capture.height, fps: capture.fps, showsCursor: false),
                                               destination: folder, appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev")
        cfg.videoQuality = settings.stored.videoQuality
        cfg.unitsPerPixel = settings.stored.elkbornUnitsPerPixel
        do {
            let r = try ElkbornReplayRecorder(config: cfg)
            r.onPhase = { [weak self] phase, text in
                Task { @MainActor in
                    self?.elkbornPhase = phase
                    if !text.isEmpty { self?.elkbornStatus = text }
                }
            }
            r.onSaved = { [weak self] _ in
                Task { @MainActor in
                    self?.elkbornSaved += 1
                    self?.library.refresh()
                }
            }
            r.onError = { [weak self] text in Task { @MainActor in self?.errorMessage = text } }
            r.onCalibrated = { [weak self] unitsPerPixel, _ in
                Task { @MainActor in
                    guard let self else { return }
                    self.settings.stored.elkbornUnitsPerPixel = unitsPerPixel
                    self.settings.stored.elkbornCalibrated = true
                    await self.stopElkborn()
                    self.elkbornStatus = String(format: "Calibrated: %.3f mouse units per Roblox pixel.", unitsPerPixel)
                }
            }
            try await r.start(excluding: ownApps, calibrate: calibrate)
            elkborn = r
            elkbornMode = calibrate ? .calibrating : .listening
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func stopElkborn() async {
        guard let r = elkborn else { return }
        elkborn = nil
        await r.stop()
        elkbornMode = .off
        elkbornPhase = .idle
        library.refresh()
    }

    func stop() async {
        guard phase == .recording, let s = session else { return }
        phase = .saving
        statsTimer?.invalidate()
        hud.hide()
        do {
            lastSaved = try await s.stop()
            stats = s.stats
        } catch {
            errorMessage = "Saving failed: \(error.localizedDescription)"
        }
        session = nil
        phase = .idle
        library.refresh()
    }
}
