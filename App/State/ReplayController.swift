import AppKit
import ATCapture
import ATCore
import ATRuntime
import Foundation
import Observation

/// Plays a recording's keyboard and mouse back into the Run tab's window, to check that a
/// recording (an Elkborn replay, say) reproduces the run when played on the same spot.
///
/// Arm a recording in the Library (it stays armed until another one is armed), go to the game and
/// press the replay shortcut. The same shortcut, or the emergency stop, stops it. The Run tab's
/// mouse output, allowed actions and "only act while the target is in front" apply.
@MainActor @Observable
final class ReplayController {
    enum Phase: Equatable {
        case idle
        case preparing(String)
        case playing
        case waitingForFocus
    }

    private(set) var armed: URL?
    private(set) var armedName = ""
    private var armedID: UUID?
    var phase = Phase.idle
    private(set) var elapsed = 0.0
    private(set) var duration = 0.0
    /// The window being played into (while active).
    private(set) var targetName = ""
    /// The outcome of the last replay, or why one couldn't start.
    private(set) var message: String?
    private(set) var messageIsError = false

    var isActive: Bool { phase != .idle }

    private let settings: AppSettings
    private let runner: RunController
    private var replayer: InputReplayer?
    private var skipped = 0
    private var startTask: Task<Void, Never>?
    private var timer: Timer?
    private var activationObserver: NSObjectProtocol?
    private var terminateObserver: NSObjectProtocol?
    private let front = FrontmostApp()
    let hud = HUDPanelController()

    private static let armedKey = "replayArmed"
    /// Snapshot copies of the app share the user's defaults: they never change what is armed there.
    private let persists = !CommandLine.arguments.contains("--ui-snapshot")

    init(settings: AppSettings, runner: RunController) {
        self.settings = settings
        self.runner = runner
        if let path = UserDefaults.standard.string(forKey: Self.armedKey) {
            let url = URL(fileURLWithPath: path)
            if let meta = try? RecordingIO.readMeta(url) { armed = url; armedName = meta.name; armedID = meta.id }
        }
        // Quitting mid-replay must not leave a key held down in the game.
        terminateObserver = NotificationCenter.default.addObserver(forName: NSApplication.willTerminateNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.replayer?.stop(waitingUpTo: 0.5) }
        }
    }

    // MARK: Arming

    func isArmed(_ url: URL) -> Bool { armed?.standardizedFileURL == url.standardizedFileURL }

    func arm(_ r: LibraryRecording) {
        guard !isActive else { return }
        armed = r.url
        armedName = r.meta.name
        armedID = r.meta.id
        message = nil
        if persists { UserDefaults.standard.set(r.url.path, forKey: Self.armedKey) }
    }

    func disarm() {
        guard !isActive else { return }
        armed = nil
        armedName = ""
        armedID = nil
        message = nil
        if persists { UserDefaults.standard.removeObject(forKey: Self.armedKey) }
    }

    /// Keeps the armed recording when the Library renames or moves it (found again by its id).
    func relocate(in recordings: [LibraryRecording]) {
        guard let id = armedID, let r = recordings.first(where: { $0.meta.id == id }) else { return }
        if !isArmed(r.url) || armedName != r.meta.name {
            armed = r.url
            armedName = r.meta.name
            if persists { UserDefaults.standard.set(r.url.path, forKey: Self.armedKey) }
        }
    }

    // MARK: Playing

    func toggle() { if isActive { stop() } else { start() } }

    /// `bringToFront`: started from AgentTrainer's own window, so the target is brought forward first.
    func start(bringToFront: Bool = false) {
        guard !isActive else { return }
        guard let url = armed, FileManager.default.fileExists(atPath: url.appendingPathComponent(RecordingPackage.meta).path) else {
            fail("Arm a recording first: select it in the Library and press Replay.")
            return
        }
        guard !runner.isRunning else { fail("Stop the AI before replaying a recording."); return }
        message = nil
        phase = .preparing("Getting ready…")
        startTask = Task { await prepareAndPlay(url, bringToFront: bringToFront) }
    }

    func stop() {
        if let r = replayer {
            r.stop() // onEnd finishes up once everything is let go
        } else if case .preparing = phase {
            startTask?.cancel()
            startTask = nil
            phase = .idle
            message = "Replay cancelled."
            messageIsError = false
        }
    }

    private func prepareAndPlay(_ url: URL, bringToFront: Bool) async {
        // The Run tab's window, with its frame as it is now.
        guard let id = runner.selectedTargetID else {
            return fail("Choose the window to replay into in the Run tab.")
        }
        let target: CaptureTarget
        do {
            let found = try await CaptureTargets.load().targets.first { $0.id == id }
            guard let found else { return fail("The Run tab's window is gone. Choose it again in the Run tab.") }
            target = found
        } catch {
            return fail("Couldn't find the window: \(error.localizedDescription)")
        }
        if Task.isCancelled { return }

        var options = ReplayPlan.Options()
        options.mouseMode = runner.prefs.mouseMode
        options.permissions = runner.prefs.permissions
        options.hotkeys = settings.stored.appHotkeys.map { ($0.keyCode, $0.cgFlags) }
        options.targetFrame = target.frame
        let plan: ReplayPlan
        do {
            plan = try await Task.detached(priority: .userInitiated) { try ReplayPlan.load(url, options: options) }.value
        } catch {
            return fail("Couldn't read the recording: \(error.localizedDescription)")
        }
        if Task.isCancelled { return }

        if bringToFront, let pid = target.processID {
            NSRunningApplication(processIdentifier: pid)?.activate()
            try? await Task.sleep(for: .milliseconds(700))
        }
        // Started with the shortcut: wait until its keys are let go, so the replay's first keys
        // don't arrive with ⌘ or ⌥ still held.
        phase = .preparing("Let go of the shortcut…")
        let modifiers: CGEventFlags = [.maskCommand, .maskAlternate, .maskControl, .maskShift]
        let deadline = Date().addingTimeInterval(4)
        while !CGEventSource.flagsState(.combinedSessionState).intersection(modifiers).isEmpty, Date() < deadline {
            try? await Task.sleep(for: .milliseconds(20))
            if Task.isCancelled { return }
        }
        try? await Task.sleep(for: .milliseconds(120))
        if Task.isCancelled || runner.isRunning { if runner.isRunning { fail("Stop the AI before replaying a recording.") }; return }

        play(plan, into: target)
    }

    private func play(_ plan: ReplayPlan, into target: CaptureTarget) {
        let r = InputReplayer(plan: plan)
        let guarded = runner.prefs.focusGuard && !target.isDisplay
        if guarded, let pid = target.processID {
            front.set(NSWorkspace.shared.frontmostApplication?.processIdentifier ?? 0)
            activationObserver = NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: nil) { [front] n in
                if let app = n.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication { front.set(app.processIdentifier) }
            }
            r.isTargetFrontmost = { [front] in front.pid == pid }
        }
        r.onEnd = { [weak self] ending in
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.finished(ending) } }
        }
        replayer = r
        skipped = plan.skippedPresses
        duration = plan.duration
        elapsed = 0
        targetName = target.name
        startTask = nil
        phase = .playing
        r.start()

        hud.show(mode: .replay, options: runner.prefs.hud, on: target.frame) { [weak self] in
            guard let self, let r = self.replayer else { return }
            let s = r.status
            self.hud.model.push(r.live.snapshot)
            self.hud.model.paused = s.waitingForFocus
            self.hud.model.statusLine = s.waitingForFocus ? "waiting for \(self.targetName)" : String(format: "%.1f / %.1f s", s.elapsed, s.duration)
        }
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, let r = self.replayer else { return }
                let s = r.status
                self.elapsed = s.elapsed
                if s.running { self.phase = s.waitingForFocus ? .waitingForFocus : .playing }
                // The AI and a replay would fight over the same keys.
                if self.runner.isRunning { self.message = "Stopped: the AI started."; r.stop() }
            }
        }
    }

    private func finished(_ ending: InputReplayer.Ending) {
        let at = replayer?.status.elapsed ?? elapsed
        timer?.invalidate(); timer = nil
        if let o = activationObserver { NSWorkspace.shared.notificationCenter.removeObserver(o); activationObserver = nil }
        hud.hide()
        replayer = nil
        phase = .idle
        messageIsError = false
        switch ending {
        case .finished:
            elapsed = duration
            var m = "Replayed \(duration.durationText) into \(targetName)."
            if skipped > 0 { m += " \(skipped) key or button press\(skipped == 1 ? " was" : "es were") left out (blocked keys or shortcuts)." }
            message = m
        case .stopped:
            if message == nil || message?.hasPrefix("Stopped") == false { message = "Stopped at \(at.durationText) of \(duration.durationText)." }
        }
    }

    private func fail(_ text: String) {
        startTask = nil
        phase = .idle
        message = text
        messageIsError = true
    }
}

/// The frontmost app's process, kept current from workspace notifications (read on the replay thread).
private final class FrontmostApp: @unchecked Sendable {
    private let lock = NSLock()
    private var value: pid_t = 0
    var pid: pid_t { lock.withLock { value } }
    func set(_ p: pid_t) { lock.withLock { value = p } }
}
