import AppKit
import ATCapture
import ATCore
import ATLanguage
import ATModel
import ATRuntime
import Foundation
import Observation
import ScreenCaptureKit

struct RunPreferences: Codable {
    var permissions = ActionPermissions()
    var temperature = 1.0
    var sensitivity = 1.0
    var mouseMode = MouseOutputMode.pointer
    var dryRun = false
    var focusGuard = true
    /// The correction shortcut saves what you did (plus the 10 s before it) as a recording.
    var saveCorrections = true
    /// Your input during a run (correcting or not) goes into the AI's action history as if it had
    /// done it, limited to what the AI may press. Only matters with `useActionHistory`.
    var correctionsAsHistory = true
    /// Off by default: with little training data, models fed their own past actions tend to repeat
    /// themselves (copycat). Pixel2Play feeds history; turn it on for large datasets.
    var useActionHistory = false
    /// Black out the bottom-left corner where Elkborn's replays draw their timecode strip, for
    /// models trained on Elkborn replays (those recordings have that corner blacked out).
    var maskElkbornCorner = false
    var hud = HUDOptions()

    init() {}
    init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        let x = RunPreferences()
        permissions = (try? c.decode(ActionPermissions.self, forKey: .permissions)) ?? x.permissions
        temperature = (try? c.decode(Double.self, forKey: .temperature)) ?? x.temperature
        sensitivity = (try? c.decode(Double.self, forKey: .sensitivity)) ?? x.sensitivity
        mouseMode = (try? c.decode(MouseOutputMode.self, forKey: .mouseMode)) ?? x.mouseMode
        dryRun = (try? c.decode(Bool.self, forKey: .dryRun)) ?? x.dryRun
        focusGuard = (try? c.decode(Bool.self, forKey: .focusGuard)) ?? x.focusGuard
        saveCorrections = (try? c.decode(Bool.self, forKey: .saveCorrections)) ?? x.saveCorrections
        correctionsAsHistory = (try? c.decode(Bool.self, forKey: .correctionsAsHistory)) ?? x.correctionsAsHistory
        useActionHistory = (try? c.decode(Bool.self, forKey: .useActionHistory)) ?? x.useActionHistory
        maskElkbornCorner = (try? c.decode(Bool.self, forKey: .maskElkbornCorner)) ?? x.maskElkbornCorner
        hud = (try? c.decode(HUDOptions.self, forKey: .hud)) ?? x.hud
    }
}

@MainActor @Observable
final class RunController {
    private let settings: AppSettings
    private let models: ModelStore
    var selectedModelID: UUID?
    var targets: [CaptureTarget] = []
    var thumbnails: [String: NSImage] = [:]
    var selectedTargetID: String?
    var prefs: RunPreferences { didSet { if let d = try? JSONEncoder().encode(prefs) { UserDefaults.standard.set(d, forKey: "runPrefs") } } }
    var instruction = ""
    var appliedInstruction: String?
    var isRunning = false
    var isStarting = false
    var stats = RuntimeStats()
    var errorMessage: String?
    var loadedCheckpoint = ""
    var footerLine = ""
    var encoderStatus = ""
    let hud = HUDPanelController()
    private var runtime: AgentRuntime?
    private var ownApps: [SCRunningApplication] = []
    private var timer: Timer?
    private var corrections: CorrectionRecorder?
    /// The Record tab's "Save to" folder: where corrections are saved.
    var correctionDestination: (() -> URL)?
    /// Corrections saved during this run.
    var savedCorrections = 0
    var isCorrecting: Bool { stats.correcting }
    private var encoder: InstructionEncoder?
    private var visionImage: CGImage?
    weak var library: LibraryStore?

    init(settings: AppSettings, models: ModelStore) {
        self.settings = settings
        self.models = models
        if let d = UserDefaults.standard.data(forKey: "runPrefs"), let p = try? JSONDecoder().decode(RunPreferences.self, from: d) { prefs = p } else { prefs = RunPreferences() }
        selectedModelID = models.models.first(where: { !$0.checkpoints.isEmpty })?.id
    }

    var selectedTarget: CaptureTarget? { targets.first { $0.id == selectedTargetID } }
    var selectedModel: ModelEntry? { models.entry(selectedModelID) }
    var isPaused: Bool { stats.paused }

    func refreshTargets() async {
        do {
            let r = try await CaptureTargets.load()
            targets = r.targets; ownApps = r.ownApps
            if selectedTarget == nil { selectedTargetID = targets.first(where: { !$0.isDisplay })?.id ?? targets.first?.id }
            for t in targets.prefix(24) {
                if let img = await CaptureTargets.thumbnail(t, own: ownApps, maxWidth: 360) { thumbnails[t.id] = NSImage(cgImage: img, size: .init(width: img.width, height: img.height)) }
            }
        } catch { errorMessage = "Couldn't list windows: \(error.localizedDescription)" }
    }

    func toggle() async { if isRunning { stop() } else { await start() } }

    func start() async {
        guard !isRunning, !isStarting else { return }
        guard let entry = selectedModel else { errorMessage = "Choose a trained model."; return }
        if targets.isEmpty { await refreshTargets() }
        guard let target = selectedTarget else { errorMessage = "Choose the window or display to control."; return }
        isStarting = true
        defer { isStarting = false }
        errorMessage = nil
        do {
            let url = entry.url, q = settings.stored.inferenceQuantized
            let (model, def, ckpt) = try await Task.detached(priority: .userInitiated) { try ModelLoader.load(url, quantized: q) }.value
            var o = RuntimeOptions()
            o.permissions = prefs.permissions
            o.temperature = Float(prefs.temperature)
            o.mouseSensitivity = prefs.sensitivity
            o.mouseMode = prefs.mouseMode
            o.dryRun = prefs.dryRun
            o.focusGuard = prefs.focusGuard
            o.useActionHistory = prefs.useActionHistory
            o.humanInputAsHistory = prefs.correctionsAsHistory
            o.pauseOnHumanInput = settings.stored.pauseOnHumanInput
            o.captureMaxHeight = settings.stored.captureMaxHeight
            o.keepGPUWarm = settings.stored.keepGPUWarm
            if prefs.maskElkbornCorner { o.masks = [ElkbornCorner.mask(window: target.frame)] }
            let st = settings.stored
            o.hotkeys = st.appHotkeys.map { ($0.keyCode, $0.cgFlags) }
            let rt = try AgentRuntime(model: model, definition: def, target: target, options: o)
            rt.onVision = { [weak self] img in DispatchQueue.main.async { MainActor.assumeIsolated { self?.visionImage = img } } }
            rt.onStop = { [weak self] reason in MainActor.assumeIsolated { if let reason { self?.errorMessage = reason }; self?.stop() } }
            if def.usesInstructions || def.architecture.usesDecoderMemory { prepareEncoder() }
            rt.encoder = encoder
            if def.usesInstructions {
                rt.setInstruction(embedding(for: instruction), text: instruction.isEmpty ? nil : instruction)
                appliedInstruction = instruction.isEmpty ? nil : instruction
            }
            try await rt.start(excluding: ownApps)
            runtime = rt
            loadedCheckpoint = ckpt
            isRunning = true
            savedCorrections = 0
            if prefs.saveCorrections { await startCorrections(target: target, model: def) }
            hud.show(mode: .run, options: prefs.hud, on: target.frame) { [weak self] in
                guard let self, let rt = self.runtime else { return }
                self.hud.model.push(rt.output.snapshot)
                self.hud.model.vision = self.visionImage
                let a = def.architecture
                self.hud.model.visionCaption = "\(a.frameSize)×\(a.frameSize) \(a.grayscale ? "gray" : "RGB")"
                self.hud.model.paused = rt.isPaused
                self.hud.model.correcting = rt.isCorrecting
                self.hud.model.instruction = self.appliedInstruction
                if rt.isCorrecting {
                    self.hud.model.statusLine = "You're in control · \(self.settings.stored.correctionHotkey.display) hands back"
                } else {
                    self.hud.model.statusLine = String(format: "%.1f / %d decisions/s · %.0f ms", self.stats.fps, a.stepRateHz, self.stats.endToEndMs)
                }
            }
            timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
                MainActor.assumeIsolated {
                    guard let self, let rt = self.runtime else { return }
                    self.stats = rt.currentStats
                    self.footerLine = rt.isCorrecting ? "You're correcting the AI" : rt.isPaused ? "AI paused" : String(format: "AI running · %.0f fps", self.stats.fps)
                }
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func stop() {
        guard isRunning else { return }
        runtime?.stop()
        runtime?.waitUntilStopped()
        timer?.invalidate()
        hud.hide()
        isRunning = false
        runtime = nil
        stats = RuntimeStats()
        if let c = corrections {
            corrections = nil
            c.end()   // a correction still going when the AI stops is kept
            Task.detached { await c.stop() }
        }
    }

    func togglePause() {
        guard let rt = runtime else { return }
        if rt.isCorrecting { toggleCorrection(); return }
        rt.setPaused(!rt.isPaused, reason: rt.isPaused ? "" : "Paused")
        stats = rt.currentStats
    }

    /// First press: the AI freezes and lets go of every key, and only you play. Second press: the AI
    /// carries on from where you left it, and what you did (with the 10 s before it) is saved.
    func toggleCorrection() {
        guard let rt = runtime else { return }
        if rt.isCorrecting {
            rt.setCorrecting(false)
            corrections?.end()
        } else {
            rt.setCorrecting(true)
            ActionExecutor().releaseAll()
            corrections?.begin()
        }
        stats = rt.currentStats
    }

    /// Stops immediately and releases every key and button.
    func emergencyStop() {
        stop()
        ActionExecutor().releaseAll()
    }

    // MARK: Instructions

    func prepareEncoder() {
        if encoder == nil { encoder = LanguageModels.instructionEncoder(pretrainedRoot: settings.paths.pretrainedRoot) }
        encoderStatus = encoder == nil ? "Download the instruction model (EmbeddingGemma) in Settings → Downloads to use instructions." : ""
    }

    func embedding(for text: String) -> [Float]? {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return nil }
        prepareEncoder()
        return try? encoder?.embed([t]).first
    }

    func applyInstruction() {
        guard let rt = runtime else { return }
        rt.setInstruction(embedding(for: instruction), text: instruction.isEmpty ? nil : instruction)
        appliedInstruction = instruction.isEmpty ? nil : instruction
    }

    // MARK: Corrections (DAgger-style)

    /// Keeps the last seconds of the run on disk so each correction can be saved with the moments
    /// that led to it: those are kept as context for training (the AI's part), and your part is what it learns.
    private func startCorrections(target: CaptureTarget, model: ModelDefinition) async {
        let st = settings.stored
        let folder = correctionDestination?() ?? library?.fs.root ?? settings.paths.libraryRoot
        let scratch = settings.paths.cacheRoot.appendingPathComponent("correction-buffer", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        var cfg = CorrectionRecorder.Config(target: target,
                                            settings: CaptureSettings.fitting(target, maxHeight: st.captureMaxHeight, fps: st.captureFPS, showsCursor: st.showsCursor),
                                            destination: folder, scratch: scratch,
                                            appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev",
                                            modelName: model.name)
        cfg.input = st.input
        cfg.input.onlyWhenTargetFocused = false
        cfg.videoQuality = st.videoQuality
        cfg.task = appliedInstruction ?? ""
        cfg.ignoredHotkeys = st.appHotkeys.map { (code: $0.keyCode, modifiers: $0.cgFlags) }
        do {
            let c = try CorrectionRecorder(config: cfg)
            c.onSaved = { [weak self] url in
                DispatchQueue.main.async { MainActor.assumeIsolated {
                    guard let self else { return }
                    self.savedCorrections += 1
                    self.library?.refresh()
                    self.hud.flash("Correction saved · \(url.deletingPathExtension().lastPathComponent)")
                } }
            }
            c.onError = { [weak self] msg in
                DispatchQueue.main.async { MainActor.assumeIsolated { self?.hud.flash(msg) } }
            }
            try await c.start(excluding: ownApps)
            corrections = c
        } catch { errorMessage = "Couldn't keep corrections: \(error.localizedDescription)" }
    }

    // MARK: Mouse injection test

    /// Brings the target to the front and moves the mouse in a small square so you can see
    /// whether the game reacts to synthetic input. Only runs when you press the button.
    func mouseTest() async {
        guard let t = selectedTarget else { return }
        if let pid = t.processID { NSRunningApplication(processIdentifier: pid)?.activate() }
        try? await Task.sleep(for: .seconds(1))
        let ex = ActionExecutor()
        ex.mouseMode = prefs.mouseMode
        var perms = ActionPermissions(); perms.keyboard = false; perms.mouseButtons = false; perms.scroll = false
        for (dx, dy) in [(12.0, 0.0), (0, 12), (-12, 0), (0, -12)] {
            for _ in 0 ..< 15 {
                var a = TimestepAction(); a.dx = dx; a.dy = dy
                ex.apply(a, absolute: false, area: t.frame, permissions: perms)
                try? await Task.sleep(for: .milliseconds(33))
            }
        }
        ex.releaseAll()
    }
}
