import ATCapture
import ATCore
import ATLanguage
import ATModel
import ATRuntime
import ATTraining
import Foundation
import MLX

enum ArenaCommands {
    /// atctl arena-gen --mode chase --minutes 10 --out <folder> [--episodes 4]
    static func generate(_ args: [String]) throws {
        let o = Options(args)
        let mode = ArenaGame.Mode(rawValue: o.value("--mode") ?? "chase")!
        let out = URL(fileURLWithPath: o.value("--out") ?? "arena-data")
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let episodes = o.int("--episodes", 4)
        let seconds = o.double("--minutes", 10) * 60 / Double(episodes)
        let t0 = Date()
        for e in 0 ..< episodes {
            let url = try ArenaRecorder.generate(mode: mode, seconds: seconds, seed: UInt64(1000 + e), destination: out, name: "\(mode.rawValue)-\(e)")
            print("wrote \(url.lastPathComponent)")
        }
        print(String(format: "generated %.0f s of %@ demonstrations in %.1f s", seconds * Double(episodes), mode.rawValue, Date().timeIntervalSince(t0)))
    }

    /// atctl train --model <x.atmodel> --data <folder> [--create gameP2P150|compact|appControl] [--steps N] [--val 0.25]
    static func train(_ args: [String]) throws {
        let o = Options(args)
        let modelURL = URL(fileURLWithPath: o.value("--model") ?? "model.atmodel")
        if let preset = o.value("--create").flatMap(ModelDefinition.Preset.init(rawValue:)) {
            var def = ModelDefinition.make(preset, name: modelURL.deletingPathExtension().lastPathComponent)
            def.training.totalSteps = o.int("--steps", 300)
            def.training.validateEverySteps = o.int("--validate-every", 100)
            def.training.checkpointEveryMinutes = 60
            if let lr = o.value("--lr").flatMap(Double.init) { def.training.learningRate = lr }
            if let hd = o.value("--history-dropout").flatMap(Double.init) { def.training.historyDropout = hd }
            if o.flag("--no-text") { def.usesInstructions = false }
            try? FileManager.default.removeItem(at: modelURL)
            try ModelPackage.write(def, to: modelURL)
        } else if let steps = o.value("--steps").flatMap(Int.init) {
            var def = try ModelPackage.read(modelURL)
            def.training.totalSteps = steps
            try ModelPackage.write(def, to: modelURL)
        }
        let data = URL(fileURLWithPath: o.value("--data") ?? "arena-data")
        let recs = LibraryFileSystem(root: data).recordings(in: data, recursive: true).map(\.url)
        let split = (recs.count > 1 ? recs.dropFirst() : recs[...], recs.count > 1 ? [recs[0]] : [])
        let paths = AppPaths.standard
        let req = TrainingRequest(modelURL: modelURL, trainRecordings: Array(split.0), validationRecordings: split.1,
                                  cacheRoot: paths.cacheRoot, pretrainedRoot: paths.pretrainedRoot, memoryLimitGB: o.double("--memory", 24))
        let trainer = Trainer(request: req, encoderFactory: { LanguageModels.instructionEncoder(pretrainedRoot: paths.pretrainedRoot) }) { e in
            switch e {
            case .phase(let p, let m): print("[\(p.rawValue)] \(m)")
            case .dataset(let d): print("dataset: \(d.trainRecordings) train / \(d.validationRecordings) val recordings, \(Int(d.trainSeconds)) s, \(d.trainClips) clips/epoch, batch \(d.batchSize)×\(d.accumulation), mouse ×\(String(format: "%.2f", d.mouseUnitScale)), instructions \(Int(d.instructionCoverage * 100))%")
            case .progress(let p):
                if p.step % 10 == 0 || p.step == p.totalSteps {
                    print(String(format: "step %d/%d loss %.4f [%.3f %.3f %.3f %.3f] lr %.1e |g| %.2f %.2fs/step %.0f fps mem %.1f/%.1f GB",
                                 p.step, p.totalSteps, p.loss, p.headLosses[0], p.headLosses[1], p.headLosses[2], p.headLosses[3],
                                 p.learningRate, p.gradNorm, p.secondsPerStep, p.framesPerSecond, p.gpuMemoryGB, p.peakMemoryGB))
                }
            case .validation(let v):
                print(String(format: "VALID step %d: nll [%.3f %.3f %.3f %.3f] ppl %.3f rce [%.0f %.0f %.0f %.0f]%% keyF1 %.2f clickF1 %.2f mouseDir %.2f corr %.2f visual %.3f lang %@",
                             v.step, v.nll[0], v.nll[1], v.nll[2], v.nll[3], v.perplexity, v.rce[0], v.rce[1], v.rce[2], v.rce[3],
                             v.keyF1, v.clickF1, v.mouseDirectionAccuracy, v.mouseCorrelation, v.visualReliance,
                             v.languageReliance.map { String(format: "%.3f", $0) } ?? "–") + String(format: " history %.3f", v.historyReliance ?? 0))
            case .checkpoint(let n, _): print("checkpoint \(n)")
            case .log(let s): print("· \(s)")
            case .failed(let m): print("FAILED: \(m)")
            case .labelled, .labelProgress, .chatDelta, .chatDone: break
            }
        }
        trainer.run()
    }

    /// atctl arena-correct --model m.atmodel --mode chase --minutes 10 --out <folder>
    static func corrections(_ args: [String]) throws {
        let o = Options(args)
        let (model, _, ckpt) = try ModelLoader.load(URL(fileURLWithPath: o.value("--model") ?? "model.atmodel"))
        let mode = ArenaGame.Mode(rawValue: o.value("--mode") ?? "chase")!
        let out = URL(fileURLWithPath: o.value("--out") ?? "arena-corrections")
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let enc = LanguageModels.instructionEncoder(pretrainedRoot: AppPaths.standard.pretrainedRoot)
        let episodes = o.int("--episodes", 4)
        let seconds = o.double("--minutes", 10) * 60 / Double(episodes)
        for e in 0 ..< episodes {
            let r = try ArenaEvaluator.correctionRecording(model: model, mode: mode, seconds: seconds, seed: UInt64(20_000 + e + o.int("--seed", 0)),
                                                           destination: out, name: "\(mode.rawValue)-corrections-\(ckpt)-\(e)", encoder: enc)
            print(String(format: "wrote %@ (expert in control %.0f%% of the time)", r.url.lastPathComponent, r.expertShare * 100))
        }
    }

    /// atctl arena-eval --model <x.atmodel> --mode chase [--seconds 60] [--temperature 1]
    static func evaluate(_ args: [String]) throws {
        let o = Options(args)
        let (model, _, ckpt) = try ModelLoader.load(URL(fileURLWithPath: o.value("--model") ?? "model.atmodel"))
        let mode = ArenaGame.Mode(rawValue: o.value("--mode") ?? "chase")!
        let enc = LanguageModels.instructionEncoder(pretrainedRoot: AppPaths.standard.pretrainedRoot)
        if o.flag("--diagnose") {
            let d = ArenaEvaluator.diagnose(model: model, mode: mode, seconds: o.double("--seconds", 30), seed: 777)
            print(String(format: "open-loop key agreement with expert: %.0f%% with true history, %.0f%% with blank history (expert presses keys %.0f%% of steps)", d.withHistory * 100, d.blankHistory * 100, d.expertKeyRate * 100))
            return
        }
        var results: [ArenaEvaluator.Result] = []
        for s in 0 ..< o.int("--episodes", 3) {
            let r = ArenaEvaluator.evaluate(model: model, mode: mode, seconds: o.double("--seconds", 60), seed: UInt64(50_000 + s),
                                            temperature: Float(o.double("--temperature", 1)), useHistory: !o.flag("--no-history"),
                                            embed: { t in try? enc?.embed([t]).first }, encoder: enc,
                                            trace: o.flag("--trace") && s == 0 ? { i, a, st in
                                                if (i < 120 && i % 3 == 0) || i % 60 == 0 { print("  t=\(i) keys=\(a.keys.sorted()) \(st)") }
                                            } : nil)
            print("episode \(s) [\(ckpt)]: \(r)")
            results.append(r)
        }
        let rel = results.map(\.relative).reduce(0, +) / Double(results.count)
        print(String(format: "closed-loop: %.0f%% of expert performance (0%% = random, 100%% = expert)", rel * 100))
    }
}
