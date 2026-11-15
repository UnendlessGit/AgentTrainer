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
    var recordCorrections = false
    var useActionHistory = true
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
        recordCorrections = (try? c.decode(Bool.self, forKey: .recordCorrections)) ?? x.recordCorrections
        useActionHistory = (try? c.decode(Bool.self, forKey: .useActionHistory)) ?? x.useActionHistory
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
    private var correction: RecordingSession?
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
            o.pauseOnHumanInput = settings.stored.pauseOnHumanInput
            o.captureMaxHeight = settings.stored.captureMaxHeight
            let st = settings.stored
            o.hotkeys = [st.recordHotkey, st.runHotkey, st.pauseHotkey, st.emergencyHotkey].map { ($0.keyCode, $0.cgFlags) }
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
            if prefs.recordCorrections { await startCorrectionRecording(target: target, model: def) }
            hud.show(mode: .run, options: prefs.hud, on: target.frame) { [weak self] in
                guard let self, let rt = self.runtime else { return }
                self.hud.model.push(rt.output.snapshot)
                self.hud.model.vision = self.visionImage
                let a = def.architecture
                self.hud.model.visionCaption = "\(a.frameSize)×\(a.frameSize) \(a.grayscale ? "gray" : "RGB")"
                self.hud.model.paused = rt.isPaused
                self.hud.model.instruction = self.appliedInstruction
                self.hud.model.statusLine = String(format: "%.0f fps · %.0f ms", self.stats.fps, self.stats.endToEndMs)
            }
            timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
                MainActor.assumeIsolated {
                    guard let self, let rt = self.runtime else { return }
                    self.stats = rt.currentStats
                    self.footerLine = rt.isPaused ? "AI paused" : String(format: "AI running · %.0f fps", self.stats.fps)
                }
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func stop() {
        guard isRunning else { return }
        runtime?.stop()
        timer?.invalidate()
        hud.hide()
        isRunning = false
        let ranges = runtime?.controlRanges ?? []
        runtime = nil
        if let c = correction {
            correction = nil
            Task { await finishCorrectionRecording(c, ranges: ranges) }
        }
    }

    func togglePause() {
        guard let rt = runtime else { return }
        rt.setPaused(!rt.isPaused, reason: rt.isPaused ? "" : "Paused")
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

    private func startCorrectionRecording(target: CaptureTarget, model: ModelDefinition) async {
        guard let lib = library else { return }
        let folder = lib.fs.root.appendingPathComponent("Corrections", isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var cfg = RecordingConfig(target: target, settings: CaptureSettings.fitting(target, maxHeight: settings.stored.captureMaxHeight, fps: settings.stored.captureFPS, showsCursor: settings.stored.showsCursor),
                                  input: settings.stored.input, name: "\(model.name) corrections \(Date().formatted(.dateTime.month().day().hour().minute()))",
                                  task: appliedInstruction ?? "", destination: folder, appVersion: "run")
        cfg.input.onlyWhenTargetFocused = false
        do {
            let s = try RecordingSession(config: cfg)
            try await s.start(excluding: ownApps)
            correction = s
        } catch { errorMessage = "Couldn't record corrections: \(error.localizedDescription)" }
    }

    private func finishCorrectionRecording(_ s: RecordingSession, ranges: [(UInt64, UInt64)]) async {
        guard let url = try? await s.stop(), let t0 = s.firstFrameHostNs else { return }
        var ann = RecordingIO.readAnnotations(url)
        ann.modelControlled = ranges.map { TimeRange(start: Double(Int64($0.0) - Int64(t0)) / 1e9, end: Double(Int64($0.1) - Int64(t0)) / 1e9) }
        try? RecordingIO.writeAnnotations(ann, to: url)
        if var m = try? RecordingIO.readMeta(url) { m.isCorrectionRun = true; m.modelName = selectedModel?.definition.name; try? RecordingIO.writeMeta(m, to: url) }
        library?.refresh()
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
