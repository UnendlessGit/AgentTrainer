import AppKit
import CryptoKit
import Foundation
import MLX
import MLXNN

struct RLAuditCell: Codable {
    var task: String
    var seed: UInt64
    var seconds: Double = 90
    var actionFPS: Double = 15
    var perceptionFPS: Double = 5
    var width: Int = 96
    var height: Int = 64
    var color = false
    var history: Int = 0
    var learningRate: Double? = nil
    var explorationScale: Double? = nil
}

struct RLAuditManifest: Codable {
    var operation: String? = nil
    var evaluationSeed: UInt64? = nil
    var implementation: String
    var root: String
    var cells: [RLAuditCell]
    var evaluationSeconds: Double = 20
}

/// Opt-in local diagnostic. All policy input remains on the production
/// ScreenCaptureKit -> encoder -> HID -> Vision OCR path.
@MainActor
enum RLLiveAudit {
    private static let environment = ProcessInfo.processInfo.environment
    private static var hasRun = false
    private static var activeEvaluation: AgentRuntime?
    private static var activeReader: RewardReader?
    static var hasActiveEvaluation: Bool { activeEvaluation != nil }
    static func suspendActiveEvaluation() { activeEvaluation?.suspendRLActions() }
    static func stopActiveEvaluation(reason: String) async {
        let runtime = activeEvaluation, reader = activeReader
        runtime?.suspendRLActions()
        await runtime?.stop(reason: reason)
        await reader?.stop()
        if activeEvaluation === runtime { activeEvaluation = nil; activeReader = nil }
    }
    static func run(model: AppModel) async {
        guard !hasRun, environment["AGENTTRAINER_RL_VALIDATION"] == "1",
              let path = environment["AGENTTRAINER_RL_AUDIT_MANIFEST"],
              let workspace = environment["AGENTTRAINER_WORKSPACE_ROOT"], workspace.contains("rl-validation") else { return }
        hasRun = true
        let report = environment["AGENTTRAINER_RL_VALIDATION_REPORT"].map { URL(fileURLWithPath: $0) }
        let saved = UserDefaults.standard.data(forKey: "AgentTrainer.WorkflowSettings")
        defer {
            if let saved { UserDefaults.standard.set(saved, forKey: "AgentTrainer.WorkflowSettings") }
            else { UserDefaults.standard.removeObject(forKey: "AgentTrainer.WorkflowSettings") }
            DispatchQueue.main.async { NSApp.terminate(nil) }
        }
        var results: [[String: Any]] = []
        do {
            let manifest = try JSONDecoder().decode(RLAuditManifest.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
            let root = URL(fileURLWithPath: manifest.root).standardizedFileURL.resolvingSymlinksInPath()
            let isolated = URL(fileURLWithPath: workspace).standardizedFileURL.resolvingSymlinksInPath()
            guard root.pathComponents.contains("rl-validation"), isolated.path.hasPrefix(root.path + "/"),
                  !manifest.implementation.isEmpty,
                  manifest.implementation.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-" || $0 == "_") }),
                  !manifest.cells.isEmpty,
                  manifest.cells.allSatisfy({ ["keyboard", "pointer", "relative"].contains($0.task) && $0.seconds > 0 && $0.seconds <= 90
                      && (1...240).contains($0.actionFPS) && $0.perceptionFPS > 0 && $0.perceptionFPS <= $0.actionFPS
                      && (16...1920).contains($0.width) && (16...1080).contains($0.height) && (0...8).contains($0.history) }),
                  manifest.evaluationSeconds >= 0, manifest.evaluationSeconds <= 20 else {
                throw AgentTrainerError.invalidConfiguration("Invalid bounded audit manifest.")
            }
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            await model.bootstrap()
            if manifest.operation == "smoke" || manifest.implementation == "smoke" {
                let result = try await smoke(model: model, manifest: manifest, root: root)
                if let report { try write(result, to: report) }
                return
            }
            if manifest.operation == "evaluate" {
                let result = try await evaluateSaved(model: model, manifest: manifest, root: root)
                if let report { try write(result, to: report) }
                return
            }
            for (index, cell) in manifest.cells.enumerated() {
                let sources = try await CaptureService.availableSources()
                guard let source = sources.first(where: { $0.kind == .window && $0.name == "AgentTrainer RL Validation Game" }) else {
                    throw AgentTrainerError.capture("The separate audit game is unavailable.")
                }
                let label = "\(manifest.implementation)-\(index)-\(cell.task)-\(cell.seed)"
                let directory = root.appendingPathComponent(label)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                var profile = AIProfile.fresh(name: label)
                profile.preprocessing = PreprocessingSpec(width: cell.width, height: cell.height, colorMode: cell.color ? .color : .grayscale)
                profile.training.architecture = .small; profile.training.precision = .float32
                profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: cell.history)
                profile.training.actionFPS = cell.actionFPS; profile.training.perceptionFPS = cell.perceptionFPS
                profile.channels.mouseMovement = cell.task != "keyboard"
                profile.channels.keyboard = cell.task == "keyboard"
                profile.channels.modifiers = false; profile.channels.buttons = cell.task == "pointer"
                profile.channels.scroll = cell.task == "relative"
                var configuration = RLConfiguration()
                configuration.allowedKeys = cell.task == "keyboard" ? [0, 2, 49, 1] : []
                configuration.allowedButtons = cell.task == "pointer" ? [0, 1] : []
                configuration.mouseMovement = cell.task != "keyboard"; configuration.scroll = cell.task == "relative"
                configuration.endKey = 14; configuration.controlsInitialized = true; configuration.seed = cell.seed
                configuration.rolloutSize = 64; configuration.batchSize = 16; configuration.epochs = 2
                configuration.learningRate = cell.learningRate ?? 0.0003
                configuration.explorationScale = cell.explorationScale ?? 1
                configuration.autosaveSeconds = 86400
                configuration.rewardRegion = RLRewardRegion(sourceID: source.id, isWindow: true,
                    normalized: CodableRect(CGRect(x: 0.01, y: 0.71, width: 0.98, height: 0.28)))
                profile.rlTraining = configuration
                let mode: MouseControlMode = cell.task == "relative" ? .relative : .absolute
                let initialDirectory = directory.appendingPathComponent("initial")
                let initialEngine = RLTrainingEngine()
                let initial = try await initialEngine.prepare(profile: profile, contextSchema: .empty, mouseMode: mode,
                    activeWeights: nil, checkpoint: directory.appendingPathComponent("absent"))
                _ = try await initialEngine.save(to: initialDirectory, versionID: UUID(), actorRandom: initial.1.actorRandom,
                    history: [], episodes: 0, elapsed: 0, pending: [])
                var row: [String: Any] = ["label": label, "cell": try JSONSerialization.jsonObject(with: JSONEncoder().encode(cell)),
                    "initialHash": try hash(initialDirectory.appendingPathComponent("weights.safetensors"))]
                if manifest.evaluationSeconds > 0 {
                    row["initialEvaluation"] = try await evaluate(profile: profile, snapshot: initial.0, weights: initialDirectory,
                        source: source, mode: mode, root: root, label: label + "-initial", seconds: manifest.evaluationSeconds,
                        seedBase: manifest.evaluationSeed ?? 900001)
                }
                try control(root: root, task: cell.task, seed: 1000 + cell.seed, label: label + "-train")
                try await WorkspaceStore.shared.saveProfile(profile)
                await model.refreshLibrary(); model.selectedProfileID = profile.id
                model.captureSources = sources; model.captureKind = .window; model.selectedSourceID = source.id
                model.runMouseMode = mode; model.selection = .run; model.runUsesRL = true
                model.showVisionPreview = false; model.cnnVisualizationSettings.enabled = false
                model.runtimeOutputPermissions = RuntimeOutputPermissions()
                model.safety.stopOnHumanInput = true; model.safety.stopOnFocusLoss = true
                model.safety.allowFullMac = false
                model.safety.controlRegion = CodableRect(CGRect(x: source.frame.minX, y: source.frame.minY + 30,
                    width: source.frame.width, height: source.frame.height * 0.62))
                await model.startRL()
                guard model.isRLTraining else { throw AgentTrainerError.model("Audit startup failed: \(model.errorMessage ?? model.runtimeStatus)") }
                // AppModel publishes the initial brain before enabling actions.
                let active = model.selectedProfile!
                let startVersion = await WorkspaceStore.shared.version(profileID: active.id, versionID: active.activeVersionID!)!
                let profileRoot = await WorkspaceStore.shared.profileDirectory(active.id)
                let startWeights = profileRoot.appendingPathComponent("Versions/\(startVersion.id.uuidString)/weights.safetensors")
                row["trainingInitialDelta"] = try difference(initialDirectory.appendingPathComponent("weights.safetensors"), startWeights)
                let began = CACurrentMediaTime()
                var samples: [[String: Any]] = []
                while model.isRLTraining, CACurrentMediaTime() - began < cell.seconds {
                    try await Task.sleep(for: .seconds(1))
                    let m = model.rlMetrics, r = model.runtimeMetrics
                    samples.append(["seconds": CACurrentMediaTime() - began, "episode": m.episode,
                        "reward": m.episodeReward, "steps": m.optimizerSteps, "updates": m.updates,
                        "actionFPS": r.actionFPS, "perceptionFPS": r.perceptionFPS, "rewardFPS": m.rewardFPS,
                        "headMS": r.actionInferenceMilliseconds, "visionMS": r.latencyMilliseconds,
                        "headP95MS": r.actionP95Milliseconds, "visionP95MS": r.encodingP95Milliseconds,
                        "observationAgeMS": r.observationAgeMilliseconds,
                        "backlog": m.backlogActions, "discarded": m.discardedActions,
                        "policyLoss": m.policyLoss, "valueLoss": m.valueLoss, "kl": m.kl,
                        "activeBytes": Memory.snapshot().activeMemory, "cacheBytes": Memory.snapshot().cacheMemory,
                        "detectedReward": model.rewardPreview?.reading?.reward as Any? ?? NSNull(),
                        "detectedLength": model.rewardPreview?.reading?.episodeLength as Any? ?? NSNull(),
                        "detection": model.rewardPreview?.status ?? "none",
                        "status": m.status])
                }
                row["trainingSeconds"] = CACurrentMediaTime() - began
                let endedEarly = !model.isRLTraining
                if endedEarly, let preview = model.rewardPreview,
                   let bytes = NSBitmapImageRep(cgImage: preview.image).representation(using: .png, properties: [:]) {
                    try bytes.write(to: directory.appendingPathComponent("failed-reward.png"))
                }
                await model.stopRL()
                row["endedEarly"] = endedEarly; row["status"] = model.runtimeStatus
                row["samples"] = samples; row["optimizerSteps"] = model.rlMetrics.optimizerSteps
                row["episodes"] = model.rlMetrics.history.map { ["reward": $0.reward, "interrupted": $0.interrupted, "duration": $0.duration] as [String: Any] }
                let finalProfile = model.profiles.first { $0.id == profile.id }!
                let checkpoint = await WorkspaceStore.shared.rlCheckpointDirectory(profile: finalProfile)
                row["policyDelta"] = try difference(initialDirectory.appendingPathComponent("weights.safetensors"), checkpoint.appendingPathComponent("weights.safetensors"))
                row["headsDelta"] = try difference(initialDirectory.appendingPathComponent("heads.safetensors"), checkpoint.appendingPathComponent("heads.safetensors"))
                row["finalHash"] = try hash(checkpoint.appendingPathComponent("weights.safetensors"))
                if manifest.evaluationSeconds > 0 {
                    let policy = AgentPolicy(profile: profile)
                    try policy.loadWeights(from: checkpoint.appendingPathComponent("weights.safetensors"))
                    let heads = RLHeads(width: policy.fusionWidth)
                    try heads.update(parameters: ModuleParameters.unflattened(MLX.loadArrays(url: checkpoint.appendingPathComponent("heads.safetensors"))), verify: .all)
                    let snapshot = RLWeightSnapshot(policy: RLWeightTransfer.copy(policy), heads: RLWeightTransfer.copy(heads), revision: model.rlMetrics.updates)
                    row["finalEvaluation"] = try await evaluate(profile: profile, snapshot: snapshot, weights: checkpoint,
                        source: source, mode: mode, root: root, label: label + "-final", seconds: manifest.evaluationSeconds,
                        seedBase: manifest.evaluationSeed ?? 900001)
                }
                results.append(row)
                if let report { try write(["implementation": manifest.implementation, "results": results], to: report) }
                print("RL_AUDIT_FINISHED \(label) steps=\(model.rlMetrics.optimizerSteps)")
                MLXMemoryLifecycle.reclaimCaches(after: "audit cell")
            }
        } catch {
            await model.stopRL()
            await model.stopAgent()
            if let report { try? write(["error": error.localizedDescription, "results": results], to: report) }
            print("RL_AUDIT_ERROR \(error.localizedDescription)")
        }
    }

    private static func evaluateSaved(model: AppModel, manifest: RLAuditManifest, root: URL) async throws -> [String: Any] {
        var results: [[String: Any]] = []
        let workspace = URL(fileURLWithPath: environment["AGENTTRAINER_WORKSPACE_ROOT"]!).lastPathComponent
        guard workspace.hasSuffix("-workspace") else { throw AgentTrainerError.model("Invalid evaluation workspace.") }
        let sourceName = String(workspace.dropLast("-workspace".count))
        guard let report = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent(sourceName + ".json"))) as? [String: Any],
              let recorded = report["results"] as? [[String: Any]] else { throw AgentTrainerError.model("Invalid training report.") }
        for cell in manifest.cells {
            let suffix = "-\(cell.task)-\(cell.seed)"
            guard let expected = recorded.first(where: { ($0["label"] as? String)?.hasSuffix(suffix) == true })?["finalHash"] as? String else {
                throw AgentTrainerError.model("The original training report is required for frozen evaluation.")
            }
            var selected: AIProfile?
            for candidate in model.profiles where candidate.name.hasSuffix(suffix) {
                guard let id = candidate.activeVersionID else { continue }
                let directory = await WorkspaceStore.shared.versionDirectory(profileID: candidate.id, versionID: id)
                if (try? hash(directory.appendingPathComponent("weights.safetensors"))) == expected { selected = candidate; break }
            }
            guard var profile = selected else {
                throw AgentTrainerError.model("Select an existing isolated profile for frozen evaluation.")
            }
            let sources = try await CaptureService.availableSources()
            guard let source = sources.first(where: { $0.kind == .window && $0.name == "AgentTrainer RL Validation Game" }) else {
                throw AgentTrainerError.capture("The audit game is unavailable.")
            }
            guard let old = profile.effectiveRL.rewardRegion else { throw AgentTrainerError.model("The saved profile has no reward region.") }
            var settings = profile.effectiveRL
            settings.rewardRegion = RLRewardRegion(sourceID: source.id, isWindow: true, normalized: old.normalized)
            profile.rlTraining = settings
            let initial = root.appendingPathComponent(profile.name).appendingPathComponent("initial")
            let final = await WorkspaceStore.shared.rlCheckpointDirectory(profile: profile)
            var row: [String: Any] = ["task": cell.task, "seed": cell.seed, "profile": profile.name]
            for (name, weights) in [("initial", initial), ("final", final)] {
                let policy = AgentPolicy(profile: profile)
                let heads = RLHeads(width: policy.fusionWidth)
                try policy.loadWeights(from: weights.appendingPathComponent("weights.safetensors"))
                try heads.update(parameters: ModuleParameters.unflattened(MLX.loadArrays(url: weights.appendingPathComponent("heads.safetensors"))), verify: .all)
                let snapshot = RLWeightSnapshot(policy: RLWeightTransfer.copy(policy), heads: RLWeightTransfer.copy(heads), revision: 0)
                row[name] = try await evaluate(profile: profile, snapshot: snapshot, weights: weights, source: source,
                    mode: cell.task == "relative" ? .relative : .absolute, root: root,
                    label: manifest.implementation + "-\(cell.task)-\(cell.seed)-\(name)", seconds: manifest.evaluationSeconds,
                    seedBase: manifest.evaluationSeed ?? 900001)
            }
            results.append(row)
        }
        return ["results": results]
    }

    private static func smoke(model: AppModel, manifest: RLAuditManifest, root: URL) async throws -> [String: Any] {
        var results: [[String: Any]] = []
        for cell in manifest.cells {
            guard let profile = model.profiles.first(where: { $0.name.hasSuffix("-\(cell.task)-\(cell.seed)") }),
                  let id = profile.activeVersionID,
                  let fixture = NSRunningApplication.runningApplications(withBundleIdentifier: "local.agenttrainer.rlfixture").first else {
                throw AgentTrainerError.model("Smoke tests require an existing isolated trained profile and the audit fixture.")
            }
            let sources = try await CaptureService.availableSources()
            guard let source = sources.first(where: { $0.kind == .window && $0.name == "AgentTrainer RL Validation Game" }) else {
                throw AgentTrainerError.capture("The audit fixture is unavailable.")
            }
            let directory = await WorkspaceStore.shared.versionDirectory(profileID: profile.id, versionID: id)
            let before = try hash(directory.appendingPathComponent("weights.safetensors"))
            try control(root: root, task: cell.task, seed: 900003, label: "smoke-\(cell.task)-\(cell.seed)")
            model.captureSources = sources; model.captureKind = .window; model.selectedSourceID = source.id
            model.selectedProfileID = profile.id; model.runUsesRL = false
            model.runtimeOutputPermissions = RuntimeOutputPermissions()
            model.runMouseMode = cell.task == "relative" ? .relative : .absolute
            model.showVisionPreview = false; model.cnnVisualizationSettings.enabled = false
            model.safety.stopOnFocusLoss = true; model.safety.stopOnHumanInput = true; model.safety.allowFullMac = false
            model.safety.controlRegion = CodableRect(CGRect(x: source.frame.minX, y: source.frame.minY + 30,
                width: source.frame.width, height: source.frame.height * 0.62))
            _ = fixture.activate(options: [.activateAllWindows])
            try await Task.sleep(for: .milliseconds(100))
            guard NSWorkspace.shared.frontmostApplication?.processIdentifier == fixture.processIdentifier else {
                throw AgentTrainerError.model("The audit fixture must be focused.")
            }
            let controller = InputInjector()
            controller.tapControllerKey(15)
            await model.startAgent()
            guard model.isRunning else { throw AgentTrainerError.model(model.runtimeStatus) }
            try await Task.sleep(for: .seconds(10))
            let ran = model.isRunning, fps = model.runtimeMetrics.actionFPS
            await model.stopAgent()
            if NSWorkspace.shared.frontmostApplication?.processIdentifier == fixture.processIdentifier { controller.tapControllerKey(14) }
            let after = try hash(directory.appendingPathComponent("weights.safetensors"))
            guard ran, before == after else { throw AgentTrainerError.model("Ordinary Run stopped early or changed the saved policy.") }
            var row: [String: Any] = ["task": cell.task, "ordinaryRun": true, "actionFPS": fps,
                "hashBefore": before, "hashAfter": after]
            if cell.task == "keyboard" {
                let checkpoint = await WorkspaceStore.shared.rlCheckpointDirectory(profile: profile)
                let previous = try JSONDecoder().decode(RLLearningState.self, from: Data(contentsOf: checkpoint.appendingPathComponent("state.json")))
                try control(root: root, task: cell.task, seed: 700007, label: "smoke-resume")
                await model.startRL()
                guard model.isRLTraining else { throw AgentTrainerError.model(model.runtimeStatus) }
                row["stepsBeforeResume"] = previous.optimizerSteps
                row["stepsAtResume"] = model.rlMetrics.optimizerSteps
                try await Task.sleep(for: .seconds(8))
                await model.stopRL()
                row["stepsAfterResume"] = model.rlMetrics.optimizerSteps
                guard model.rlMetrics.optimizerSteps > previous.optimizerSteps else {
                    throw AgentTrainerError.model("Resume did not apply another optimizer step.")
                }
            }
            results.append(row)
        }
        return ["results": results]
    }

    private static func evaluate(profile: AIProfile, snapshot: RLWeightSnapshot, weights: URL, source: CaptureSourceOption,
                                 mode: MouseControlMode, root: URL, label: String, seconds: Double, seedBase: UInt64 = 900001) async throws -> [String: Any] {
        let before = try hash(weights.appendingPathComponent("weights.safetensors"))
        var episodes: [[String: Any]] = []
        // Independent held-out schedules; both implementations use these exact
        // seeds. Each evaluation mode is reported separately.
        for deterministic in [true, false] {
            let seed: UInt64 = seedBase + (deterministic ? 0 : 1)
            try control(root: root, task: label.contains("keyboard") ? "keyboard" : (mode == .relative ? "relative" : "pointer"),
                seed: seed, label: label + (deterministic ? "-deterministic" : "-stochastic"))
            let runtime = try AgentRuntime(), reader = RewardReader()
            activeEvaluation = runtime; activeReader = reader
            defer {
                if activeEvaluation === runtime { activeEvaluation = nil; activeReader = nil }
            }
            let state = AuditEvaluationState()
            runtime.onWillStop = { reason in
                if let reason { Task { @MainActor in state.failure = reason } }
            }
            reader.onReading = { reading in Task { @MainActor in state.latest = reading } }
            reader.onFailure = { message in Task { @MainActor in state.failure = message } }
            let c = profile.effectiveRL, caps = c.capabilities(profile: profile, mouseMode: mode)
            let version = ModelVersionManifest(id: UUID(), name: "Frozen audit", createdAt: Date(), globalStep: 0,
                trainingLoss: 0, preprocessing: profile.preprocessing, channels: profile.channels, training: profile.training,
                isAutosave: false, demonstratedKeyCodes: caps.allowedKeyCodes, relativeMouseScale: GameCameraContract.deltaScale,
                trainingDataSchema: TrainingDataContract.schemaVersion, recommendedMouseMode: mode, contextSchema: .empty, outputCapabilities: caps)
            // Runtime opens immutable policy files before installing its owned
            // snapshot. This temporary, unlisted copy is never a publication.
            let runtimeDirectory = await WorkspaceStore.shared.versionDirectory(profileID: profile.id, versionID: version.id)
            try FileManager.default.createDirectory(at: runtimeDirectory, withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: weights.appendingPathComponent("weights.safetensors"),
                to: runtimeDirectory.appendingPathComponent("weights.safetensors"))
            defer { try? FileManager.default.removeItem(at: runtimeDirectory) }
            var safety = AgentSafetyPolicy(); safety.stopOnFocusLoss = true; safety.stopOnHumanInput = true
            safety.controlRegion = CodableRect(CGRect(x: source.frame.minX, y: source.frame.minY + 30,
                width: source.frame.width, height: source.frame.height * 0.62))
            let options = RLRuntimeOptions(initial: snapshot, random: RLRandom(state: seed),
                distribution: RLActionDistribution(profile: profile, configuration: c, mouseMode: mode),
                mask: c.rewardRegion!.normalized.cgRect, deterministic: deterministic,
                onDecision: { _ in state.receipt() })
            do {
                try await runtime.start(profile: profile, version: version, allowedKeyCodes: caps.allowedKeyCodes,
                    captureSpec: CaptureSpec(kind: .window, windowID: source.id, requestedFPS: profile.training.perceptionFPS),
                    captureRect: source.frame, mode: .newest, mouseMode: mode, safety: safety, rlOptions: options)
                try await reader.start(source: source, region: c.rewardRegion!, fps: profile.training.actionFPS, confidence: c.minimumConfidence)
                try await runtime.awaitInitialRLFocus()
                try await runtime.episodeKey(c.startKey)
                var reducer = RLEpisodeReducer(); reducer.reset(at: DispatchTime.now().uptimeNanoseconds)
                let began = CACurrentMediaTime()
                while CACurrentMediaTime() - began < seconds / 2 {
                    try await Task.sleep(for: .milliseconds(10))
                    if let failure = state.failure { throw AgentTrainerError.capture(failure) }
                    if let reading = state.latest, let accepted = reducer.accept(reading, mode: .cumulative) {
                        if accepted.ready { await runtime.beginRLEpisode() }
                        if accepted.terminal { break }
                    }
                }
                if let failure = state.failure { throw AgentTrainerError.model(failure) }
                runtime.suspendRLActions(); try await runtime.episodeKey(c.endKey)
                try await Task.sleep(for: .milliseconds(100))
                await reader.stop(); await runtime.stop(reason: nil)
                episodes.append(["deterministic": deterministic, "seed": seed, "reward": reducer.reward,
                    "actions": state.count, "ready": reducer.startedAt > 0, "seconds": CACurrentMediaTime() - began])
            } catch {
                runtime.suspendRLActions(); await reader.stop(); await runtime.stop(reason: nil)
                if let failure = state.failure { throw AgentTrainerError.model(failure) }
                throw error
            }
        }
        let after = try hash(weights.appendingPathComponent("weights.safetensors"))
        guard before == after else { throw AgentTrainerError.model("Frozen evaluation changed weights.") }
        return ["episodes": episodes, "hashBefore": before, "hashAfter": after, "optimizerInvocations": 0]
    }

    private static func control(root: URL, task: String, seed: UInt64, label: String) throws {
        try write(["task": task, "seed": seed, "label": label, "episodeSeconds": 10], to: root.appendingPathComponent("fixture-control.json"))
    }
    private static func write(_ object: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
    }
    private static func hash(_ url: URL) throws -> String { SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined() }
    private static func difference(_ first: URL, _ second: URL) throws -> [String: Any] {
        let a = try MLX.loadArrays(url: first), b = try MLX.loadArrays(url: second)
        guard a.keys == b.keys || Set(a.keys) == Set(b.keys) else { throw AgentTrainerError.model("Audit tensor names differ.") }
        var rows: [[String: Any]] = [], total = 0, sum = 0.0, largest = 0.0
        for name in a.keys.sorted() {
            let x = a[name]!, y = b[name]!
            guard x.shape == y.shape else { throw AgentTrainerError.model("Audit tensor shapes differ.") }
            let delta = y.asType(.float32) - x.asType(.float32); MLX.eval(delta)
            let values = delta.asArray(Float.self)
            guard values.allSatisfy(\.isFinite) else { throw AgentTrainerError.model("Nonfinite audit weights.") }
            let changed = values.filter { $0 != 0 }.count, square = values.reduce(0.0) { $0 + Double($1) * Double($1) }
            let maximum = Double(values.map(abs).max() ?? 0)
            rows.append(["name": name, "changed": changed, "l2": sqrt(square), "max": maximum])
            total += changed; sum += square; largest = max(largest, maximum)
        }
        return ["changed": total, "l2": sqrt(sum), "max": largest, "tensors": rows]
    }
}

private final class AuditEvaluationState: @unchecked Sendable {
    @MainActor var latest: RLRewardReading?
    @MainActor var failure: String?
    private let lock = NSLock()
    private var receipts = 0
    func receipt() { lock.withLock { receipts += 1 } }
    var count: Int { lock.withLock { receipts } }
}
