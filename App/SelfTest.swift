import AppKit
import ATCapture
import ATCore
import ATModel
import MLX
import ATRuntime
import ScreenCaptureKit

/// `--selftest-capture <dir>`: records AgentTrainer's own Sync Check window with the real
/// capture pipeline while pressing a harmless key into it, then analyzes sync. Synthetic keys are
/// only posted while that window is the key window of the active app, so nothing reaches other apps.
@MainActor
enum CaptureSelfTest {
    static func runIfRequested() async {
        guard let i = CommandLine.arguments.firstIndex(of: "--selftest-capture"), i + 1 < CommandLine.arguments.count else { return }
        let out = URL(fileURLWithPath: CommandLine.arguments[i + 1])
        try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        var report: [String] = []
        defer {
            try? report.joined(separator: "\n").write(to: out.appendingPathComponent("report.txt"), atomically: true, encoding: .utf8)
            NSApp.terminate(nil)
        }
        let view = SyncCheckView()
        ArenaWindows.show(view, title: "AgentTrainer Sync Check")
        NSApp.activate(ignoringOtherApps: true)
        try? await Task.sleep(for: .seconds(1.5))
        guard let window = ArenaWindows.open.last else { report.append("FAIL: no window"); return }
        do {
            let (targets, own) = try await CaptureTargets.load()
            guard let target = targets.first(where: { $0.info.title == "AgentTrainer Sync Check" }) else {
                report.append("FAIL: Sync Check window not offered as a capture target (\(targets.count) targets)"); return
            }
            var cfg = RecordingConfig(target: target, settings: CaptureSettings.fitting(target, maxHeight: 480, fps: 60, showsCursor: false),
                                      input: InputCaptureOptions(), name: "selftest-\(Int(Date().timeIntervalSince1970))", task: "", destination: out, appVersion: "selftest")
            cfg.input.onlyWhenTargetFocused = false
            let session = try RecordingSession(config: cfg)
            try await session.start(excluding: own)
            report.append("recording started: \(cfg.settings.width)x\(cfg.settings.height) @ \(cfg.settings.fps)")
            try? await Task.sleep(for: .seconds(1))
            var posted = 0
            for _ in 0 ..< 8 {
                guard NSApp.isActive, window.isKeyWindow else { report.append("skipped a key: test window not in front"); try? await Task.sleep(for: .milliseconds(700)); continue }
                for down in [true, false] {
                    let e = CGEvent(keyboardEventSource: nil, virtualKey: 0x28 /* k */, keyDown: down)
                    e?.setIntegerValueField(.eventSourceUserData, value: SyntheticInput.userData)
                    e?.post(tap: .cghidEventTap)
                    try? await Task.sleep(for: .milliseconds(60))
                }
                posted += 1
                try? await Task.sleep(for: .milliseconds(640))
            }
            try? await Task.sleep(for: .seconds(0.5))
            let url = try await session.stop()
            let meta = try RecordingIO.readMeta(url)
            report.append("posted \(posted) key presses")
            report.append("recording: \(meta.frameCount) frames, \(meta.eventCount) events, \(String(format: "%.1f", meta.duration)) s, clock \(meta.eventClock), encoder drops \(meta.droppedFrames)")
            report.append("sync: " + (try SyncCheck.analyze(recording: url)).summary)
            report.append("package: \(url.path)")
        } catch {
            report.append("FAIL: \(error.localizedDescription)")
        }
    }
}


/// `--selftest-run <model.atmodel> <dir>`: a trained model plays AgentTrainer's own arena window live
/// (real capture → model → synthetic keys). Keys are only sent while that window is in front.
@MainActor
enum RunSelfTest {
    static func runIfRequested() async {
        guard let i = CommandLine.arguments.firstIndex(of: "--selftest-run"), i + 2 < CommandLine.arguments.count else { return }
        let modelURL = URL(fileURLWithPath: CommandLine.arguments[i + 1]), out = URL(fileURLWithPath: CommandLine.arguments[i + 2])
        try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        var report: [String] = []
        func mark(_ s: String) { report.append(s); try? report.joined(separator: "\n").write(to: out.appendingPathComponent("run-report.txt"), atomically: true, encoding: .utf8) }
        defer {
            try? report.joined(separator: "\n").write(to: out.appendingPathComponent("run-report.txt"), atomically: true, encoding: .utf8)
            NSApp.terminate(nil)
        }
        mark("stage: start")
        let view = ArenaGameView(mode: .chase)
        ArenaWindows.show(view, title: "AgentTrainer Test Arena — selftest")
        NSApp.activate(ignoringOtherApps: true)
        try? await Task.sleep(for: .seconds(1.5))
        do {
            mark("stage: loading model")
            let (model, def, ckpt) = try await Task.detached { try ModelLoader.load(modelURL) }.value
            let bench: Double = await Task.detached {
                let st = PolicyStepper(model: model)
                let frames = PolicyModel.randomFrames(4, size: def.architecture.frameSize)
                for i in 0 ..< 40 { _ = st.step(frame: frames[i % 4], text: nil) }
                var total = 0.0
                for i in 0 ..< 120 {
                    let t0 = Date()
                    _ = st.step(frame: frames[i % 4], text: nil)
                    total += Date().timeIntervalSince(t0)
                    Thread.sleep(forTimeInterval: 0.018)
                }
                return total / 120 * 1000
            }.value
            mark(String(format: "in-app stepper only (paced): %.1f ms/step", bench))
            mark("stage: listing targets")
            let (targets, own) = try await CaptureTargets.load()
            guard let target = targets.first(where: { ($0.info.title ?? "").hasPrefix("AgentTrainer Test Arena") }) else { report.append("FAIL: arena window not found"); return }
            var o = RuntimeOptions()
            o.permissions.mouseMovement = false; o.permissions.mouseButtons = false; o.permissions.scroll = false
            o.pauseOnHumanInput = false
            o.focusGuard = true
            o.useActionHistory = !CommandLine.arguments.contains("--no-history")
            if CommandLine.arguments.contains("--dry") { o.dryRun = true }
            if let j = CommandLine.arguments.firstIndex(of: "--capture-fps"), let f = Int(CommandLine.arguments[j + 1]) { o.captureFPS = f }
            if let j = CommandLine.arguments.firstIndex(of: "--capture-height"), let h = Int(CommandLine.arguments[j + 1]) { o.captureMaxHeight = h }
            if CommandLine.arguments.contains("--no-warm") { o.keepGPUWarm = false }
            let rt = try AgentRuntime(model: model, definition: def, target: target, options: o)
            mark("stage: starting runtime")
            try await rt.start(excluding: own)
            mark("stage: running")
            let start = view.game.score
            try? await Task.sleep(for: .seconds(30))
            let s = rt.currentStats
            rt.stop()
            rt.waitUntilStopped()
            report.append("model \(def.name) \(ckpt), \(def.architecture.stepRateHz) Hz, history \(o.useActionHistory), GPU warm \(o.keepGPUWarm)")
            report.append(String(format: "live: %.1f decisions/s, model step %.1f ms, preprocess %.1f ms, frame age %.1f ms, screen→action %.1f ms, missed ticks %d, steps %d",
                                 s.fps, s.modelMs, s.preprocessMs, s.frameAgeMs, s.endToEndMs, s.droppedTicks, s.steps))
            report.append(String(format: "step time: %.1f ms (steps 50-150), %.1f ms (last 100)", s.earlyStepMs, s.lateStepMs))
            report.append(String(format: "MLX memory: active %.0f MB, cache %.0f MB, peak %.0f MB", Double(MLX.GPU.activeMemory) / 1e6, Double(MLX.GPU.cacheMemory) / 1e6, Double(MLX.GPU.peakMemory) / 1e6))
            report.append("score in 30 s: \(view.game.score - start)")
            if !s.pauseReason.isEmpty { report.append("note: \(s.pauseReason)") }
        } catch {
            report.append("FAIL: \(error.localizedDescription)")
        }
    }
}

/// `--selftest-train <data dir> <dir>`: submits a short training job to the XPC engine, like the Training tab.
@MainActor
enum EngineTrainSelfTest {
    static func runIfRequested(app: AppModel) async {
        guard let i = CommandLine.arguments.firstIndex(of: "--selftest-train"), i + 2 < CommandLine.arguments.count else { return }
        let data = URL(fileURLWithPath: CommandLine.arguments[i + 1]), out = URL(fileURLWithPath: CommandLine.arguments[i + 2])
        try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        var report: [String] = []
        defer {
            try? report.joined(separator: "\n").write(to: out.appendingPathComponent("train-report.txt"), atomically: true, encoding: .utf8)
            NSApp.terminate(nil)
        }
        var def = ModelDefinition.make(.compact, name: "selftest")
        def.training.totalSteps = 20
        def.training.validateEverySteps = 20
        def.architecture.memorySeconds = 4
        // Optional: train a copy of an existing model's definition (speed comparisons).
        if i + 3 < CommandLine.arguments.count, let d = try? ModelPackage.read(URL(fileURLWithPath: CommandLine.arguments[i + 3])) {
            def = d; def.id = UUID(); def.name = "selftest"; def.architectureLocked = false; def.trainedSteps = 0; def.activeCheckpoint = nil
            def.training.totalSteps = 25; def.training.validateEverySteps = 1000
            report.append("using the definition of \(d.name)")
        }
        let modelURL = out.appendingPathComponent("selftest.atmodel")
        try? FileManager.default.removeItem(at: modelURL)
        do { try ModelPackage.write(def, to: modelURL) } catch { report.append("FAIL: \(error)"); return }
        let recs = LibraryFileSystem(root: data).recordings(in: data, recursive: true).map(\.url)
        let entry = ModelEntry(url: modelURL, definition: def, checkpoints: [])
        app.training.start(model: entry, recordings: Array(recs.prefix(3)))
        let t0 = Date()
        var pausedOnce = false
        while Date().timeIntervalSince(t0) < 600 {
            try? await Task.sleep(for: .seconds(1))
            if let p = app.training.phase, [.finished, .failed, .stopped].contains(p) { break }
            // Exercise pause/resume over XPC once training is under way.
            if !pausedOnce, app.training.phase == .training, (app.training.progress.last?.step ?? 0) >= 3 {
                pausedOnce = true
                let before = app.training.progress.last?.step ?? 0
                app.training.control("pause")
                var sawPaused = false
                for _ in 0 ..< 30 { try? await Task.sleep(for: .seconds(1)); if app.training.phase == .paused { sawPaused = true; break } }
                let during = app.training.progress.last?.step ?? 0
                try? await Task.sleep(for: .seconds(4))
                let after = app.training.progress.last?.step ?? 0
                report.append("pause: phase paused \(sawPaused), steps \(before)→\(during)→\(after) while paused (should not advance)")
                app.training.control("resume")
            }
        }
        report.append("phase: \(app.training.phase?.rawValue ?? "none") · \(app.training.message)")
        if let e = app.training.errorMessage { report.append("error: \(e)") }
        if let p = app.training.progress.last { report.append(String(format: "steps %d, loss %.3f, %.0f frames/s, %.2f s/step", p.step, p.loss, p.framesPerSecond, p.secondsPerStep)) }
        let late = app.training.progress.suffix(10)
        if !late.isEmpty { report.append(String(format: "mean of last %d steps: %.2f s/step", late.count, late.map(\.secondsPerStep).reduce(0, +) / Double(late.count))) }
        if let v = app.training.validation.last { report.append(String(format: "validation key F1 %.2f", v.keyF1)) }
        report.append("checkpoints: \(ModelPackage.checkpointNames(modelURL))")
    }
}

/// `--selftest-label <recording.atrec> <dir>`: Auto Label end to end through the engine on a copy of
/// a recording (the original is never touched): one chat message about the footage, then labelling of
/// its first 25 seconds with 3 variations. Writes report.txt and the labelled copy into <dir>.
@MainActor
enum LabelSelfTest {
    static func runIfRequested(app: AppModel) async {
        guard let i = CommandLine.arguments.firstIndex(of: "--selftest-label"), i + 2 < CommandLine.arguments.count else { return }
        let source = URL(fileURLWithPath: CommandLine.arguments[i + 1])
        let out = URL(fileURLWithPath: CommandLine.arguments[i + 2])
        var report: [String] = []
        defer {
            try? report.joined(separator: "\n").write(to: out.appendingPathComponent("report.txt"), atomically: true, encoding: .utf8)
            NSApp.terminate(nil)
        }
        let fm = FileManager.default
        try? fm.createDirectory(at: out, withIntermediateDirectories: true)
        let copy = out.appendingPathComponent("labelled.atrec")
        try? fm.removeItem(at: copy)
        do { try fm.copyItem(at: source, to: copy) } catch { report.append("FAIL: copy: \(error)"); return }
        var ann = RecordingIO.readAnnotations(copy)
        let duration = (try? RecordingIO.readMeta(copy).duration) ?? 0
        ann.segments.removeAll { $0.source == .auto }
        if duration > 25 { ann.excluded.append(TimeRange(start: 25, end: duration + 1)) }
        try? RecordingIO.writeAnnotations(ann, to: copy)

        let labels = app.labels
        guard labels.modelReady else { report.append("FAIL: labelling model \(labels.model.title) not downloaded"); return }
        report.append("model: \(labels.model.title)")

        // Chat
        let t0 = Date()
        labels.send("What do you see in this footage? Answer in two sentences.", about: [copy])
        while labels.chatBusy, Date().timeIntervalSince(t0) < 300 { try? await Task.sleep(for: .milliseconds(200)) }
        if let e = labels.chatError { report.append("FAIL: chat: \(e)"); return }
        guard let reply = labels.chat.last, reply.role == .assistant, !reply.text.isEmpty else { report.append("FAIL: no chat reply"); return }
        report.append(String(format: "chat reply in %.1f s: %@", Date().timeIntervalSince(t0), reply.text))
        // A follow-up in the same conversation (the model should still have the overview).
        let t1 = Date()
        labels.send("Write instructions about where the car is heading.", about: [copy])
        while labels.chatBusy, Date().timeIntervalSince(t1) < 300 { try? await Task.sleep(for: .milliseconds(200)) }
        report.append(String(format: "follow-up reply in %.1f s: %@", Date().timeIntervalSince(t1), labels.chat.last?.text ?? "none"))

        // Labelling
        var o = LabelOptions()
        o.variations = 3
        let t2 = Date()
        labels.label([copy], options: o)
        var revisions = 0, sawProgress = false
        while labels.running, Date().timeIntervalSince(t2) < 900 {
            try? await Task.sleep(for: .milliseconds(250))
            revisions = labels.revision(copy)
            if labels.progress.fraction > 0 { sawProgress = true }
        }
        if labels.running { report.append("FAIL: labelling still running after 15 min"); labels.stop(); return }
        if let e = labels.lastError { report.append("FAIL: labelling: \(e)"); return }
        let result = RecordingIO.readAnnotations(copy).segments.filter { $0.source == .auto }
        report.append(String(format: "labelled in %.1f s: %d instructions, %d live updates, progress seen: %@", Date().timeIntervalSince(t2), result.count, revisions, sawProgress ? "yes" : "no"))
        for s in result {
            report.append(String(format: "  %5.1f–%5.1f  %@  (%d variations)", s.start, s.end, s.text, s.variants?.count ?? 0))
            if let ob = s.observation { report.append("      seen: \(ob)") }
            for v in s.variants ?? [] { report.append("      · \(v)") }
        }
        let ok = !result.isEmpty && result.allSatisfy { $0.end <= 25.01 } && result.contains { ($0.variants?.count ?? 0) > 0 } && revisions >= result.count
        report.append(ok ? "PASS" : "FAIL: expected instructions with variations inside the first 25 s, each reported live")
    }
}
