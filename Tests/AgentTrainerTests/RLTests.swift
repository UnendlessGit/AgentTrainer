import XCTest
import MLX
import MLXNN
import CoreVideo
@testable import AgentTrainer

final class RLTests: XCTestCase {
    private func profile() -> AIProfile {
        var p = AIProfile.fresh(name: "RL Test")
        p.preprocessing = PreprocessingSpec(width: 16, height: 16, colorMode: .grayscale)
        p.training.precision = .float32; p.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 0)
        p.training.architecture.convolutionChannels = [4]; p.training.architecture.kernelSizes = [3]
        p.training.architecture.strides = [2]; p.training.architecture.visualEmbedding = 8
        p.training.architecture.recurrentWidth = 8; p.training.architecture.fusionWidths = [16]
        p.training.architecture.dropout = 0; p.training.architecture.attentionHeads = 2
        p.channels = ActionChannels(absoluteMouse: false, relativeMouse: false, buttons: false, scroll: false, keyboard: true, modifiers: false)
        var c = RLConfiguration(); c.allowedKeys = [0, 2]; c.rolloutSize = 32; c.batchSize = 8; c.epochs = 2
        c.normalizeRewards = false; c.learningRate = 0.001; c.targetKL = 0.2
        p.rlTraining = c
        return p
    }

    private func observation(_ p: AIProfile, cue: Bool = false) -> RLObservation {
        RLObservation(current: Data(repeating: cue ? 220 : 30, count: p.preprocessing.sampleByteCount), past: [],
            controls: [], timing: [Float](repeating: 0, count: TemporalInputContract.valueCount), pastTiming: [], context: [])
    }

    private func reading(_ reward: Double, _ length: Double, _ time: UInt64) -> RLRewardReading {
        RLRewardReading(reward: reward, episodeLength: length, capturedAt: time, recognizedAt: time + 100, confidence: 1)
    }

    func testStrictNumericParsingAndAtomicPairs() {
        for (text, value) in [("-125.5", -125.5), ("+0.25", 0.25), ("−2", -2), ("0", 0)] { XCTAssertEqual(RLTextParser.number(text), value) }
        for bad in ["NaN", "inf", "_125.5", "1,000", "1 2", "12..3", "Reward 5"] { XCTAssertNil(RLTextParser.number(bad)) }
        XCTAssertEqual(RLTextParser.pair(["Reward: -50.5", "Episode Length: 0"])?.0, -50.5)
        XCTAssertEqual(RLTextParser.pair(["Reward: -50.5   Episode Length: 60"])?.1, 60)
        XCTAssertNil(RLTextParser.pair(["Reward: 5"]))
        XCTAssertNil(RLTextParser.pair(["Reward: 5", "Episode Length: -1"]))
        XCTAssertNil(RLTextParser.pair(["Reward: 1", "Reward: 2", "Episode Length: 60"]))
    }

    func testCumulativeTerminalRewardIsCountedOnceAndResetIsAcknowledged() {
        var r = RLEpisodeReducer(); r.reset(at: 100)
        XCTAssertNil(r.accept(reading(20, 60, 101), mode: .cumulative))
        XCTAssertTrue(r.accept(reading(0, 60, 102), mode: .cumulative)!.ready)
        XCTAssertEqual(r.accept(reading(125.5, 60, 103), mode: .cumulative)?.reward, 125.5)
        XCTAssertEqual(r.accept(reading(125.5, 30, 104), mode: .cumulative)?.reward, 0)
        let end = r.accept(reading(-50, 0, 105), mode: .cumulative)
        XCTAssertEqual(end?.reward, -175.5); XCTAssertEqual(r.reward, -50); XCTAssertTrue(end!.terminal)
        XCTAssertNil(r.accept(reading(-50, 0, 106), mode: .cumulative))
        r.reset(at: 200)
        XCTAssertNil(r.accept(reading(0, 60, 150), mode: .cumulative))
        XCTAssertTrue(r.accept(reading(0, 60, 201), mode: .cumulative)!.ready)
        XCTAssertEqual(r.reward, 0)
    }

    func testImmediateRewardsRepeatAndEpisodeLengthIsNotACountdown() {
        var r = RLEpisodeReducer(); r.reset(at: 1)
        _ = r.accept(reading(0, 60, 2), mode: .immediate)
        _ = r.accept(reading(1, 60, 1_000_000_002), mode: .immediate)
        _ = r.accept(reading(1, 30, 2_000_000_002), mode: .immediate)
        XCTAssertEqual(r.reward, 2); XCTAssertEqual(r.remaining(at: 5_000_000_002), 25, accuracy: 1e-8)
        XCTAssertNil(r.accept(reading(100, 60, 1_000_000_002), mode: .immediate))
    }

    func testReturnsRespectTerminalTruncationAndElapsedTime() {
        var c = RLConfiguration(); c.discount = 0.9; c.gaeLambda = 1
        let rows = [RLTransition(decisions: [], reward: 1, duration: 1, terminal: false),
                    RLTransition(decisions: [], reward: 2, duration: 1, terminal: true)]
        let targets = RLReturns.advantages(rows, values: [0, 0], configuration: c, actionFPS: 1)
        XCTAssertEqual(targets.returns[0], 2.8, accuracy: 1e-5)
        XCTAssertEqual(targets.returns[1], 2, accuracy: 1e-5)
        let cut = [RLTransition(decisions: [], reward: 1, duration: 2, terminal: false, boundary: true, bootstrapValue: 10)]
        XCTAssertEqual(RLReturns.advantages(cut, values: [0], configuration: c, actionFPS: 1).returns[0], 9.1, accuracy: 1e-4)
    }

    func testStoredLatentLikelihoodMatchesSamplingAndMasksReservedKeys() throws {
        var p = profile(); p.rlTraining!.allowedKeys.insert(15)
        let d = RLActionDistribution(profile: p, configuration: p.effectiveRL, mouseMode: .relative)
        XCTAssertFalse(d.binaryIndices.contains(29))
        var random = RLRandom(state: 100)
        let logits = [Float](repeating: 0.2, count: PolicyLogitsLayout.count)
        let std = [Float](repeating: -2, count: 8)
        let sample = d.sample(logits: logits, logStd: std, random: &random)
        let decision = RLDecision(observation: observation(p), latent: sample.latent, binary: sample.binary,
            logProbability: sample.logProbability, value: 0, timestamp: 1, revision: 0)
        let stats = d.statistics(logits: MLXArray(logits, [1, logits.count]), logStd: MLXArray(std), decision: decision)
        XCTAssertEqual(stats.0.item(Float.self), sample.logProbability, accuracy: 1e-5)
        XCTAssertEqual(sample.prediction[29], 0)
        p.channels.mouseMovement = true; p.rlTraining!.mouseMovement = true
        let mouse = RLActionDistribution(profile: p, configuration: p.effectiveRL, mouseMode: .relative)
        let action = mouse.sample(logits: logits, logStd: std, random: &random)
        var mouseDecision = decision; mouseDecision.latent = action.latent; mouseDecision.binary = action.binary
        XCTAssertEqual(mouse.statistics(logits: MLXArray(logits, [1, logits.count]), logStd: MLXArray(std), decision: mouseDecision).0.item(Float.self), action.logProbability, accuracy: 1e-5)
    }

    func testActualPPOUpdateChangesPolicyAndCheckpointResumes() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        var p = profile()
        let engine = RLTrainingEngine()
        let first = try await engine.prepare(profile: p, contextSchema: .empty, mouseMode: .relative,
            activeWeights: nil, checkpoint: root.appendingPathComponent("missing"))
        let actor = RLTrainablePolicy(policy: AgentPolicy(profile: p))
        try RLWeightTransfer.install(first.0.policy, in: actor.policy); try RLWeightTransfer.install(first.0.heads, in: actor.heads)
        let distribution = RLActionDistribution(profile: p, configuration: p.effectiveRL, mouseMode: .relative)
        var random = RLRandom(state: 5), rows: [RLTransition] = []
        for i in 0..<16 {
            let obs = observation(p, cue: i.isMultiple(of: 2)), output = actor.forward(obs)
            MLX.eval(output.0, output.1)
            let sampled = distribution.sample(logits: output.0.asArray(Float.self), logStd: actor.heads.logStandardDeviation.asArray(Float.self), random: &random)
            let decision = RLDecision(observation: obs, latent: sampled.latent, binary: sampled.binary,
                logProbability: sampled.logProbability, value: output.1.item(Float.self), timestamp: UInt64(i + 1), revision: 0)
            rows.append(RLTransition(decisions: [decision], reward: sampled.binary[14] > 0 ? 1 : -1, duration: 1 / 60, terminal: true))
        }
        let update = try await engine.update(rows)
        XCTAssertGreaterThan(update.steps, 0); XCTAssertTrue(update.policyLoss.isFinite)
        XCTAssertTrue(first.0.policy.contains { update.snapshot.policy[$0.key]?.values != $0.value.values })
        let id = UUID()
        _ = try await engine.save(to: root, versionID: id, actorRandom: random, history: [], episodes: 0, elapsed: 2, pending: [])
        p.activeVersionID = id
        let resumed = RLTrainingEngine()
        let restored = try await resumed.prepare(profile: p, contextSchema: .empty, mouseMode: .relative,
            activeWeights: root.appendingPathComponent("weights.safetensors"), checkpoint: root)
        XCTAssertEqual(restored.1.optimizerSteps, update.steps)
        for (name, tensor) in update.snapshot.policy { XCTAssertEqual(restored.0.policy[name]?.values, tensor.values) }
        let a = try await engine.update(rows), b = try await resumed.update(rows)
        for (name, tensor) in a.snapshot.policy { XCTAssertEqual(b.snapshot.policy[name]?.values, tensor.values) }
    }
}

extension RLTests {
    func testVisualKeyboardAndMouseLearningAcrossThreeSeeds() async throws {
        guard ProcessInfo.processInfo.environment["AGENTTRAINER_RUN_RL_LEARNING_TESTS"] == "1" else {
            throw XCTSkip("Set AGENTTRAINER_RUN_RL_LEARNING_TESTS=1 for the three-seed learning gate.")
        }
        var report: [[String: Any]] = []
        for mouse in [false, true] {
            for seed in [UInt64(7), 42, 101] {
                var p = profile(), c = p.effectiveRL
                c.seed = seed; c.batchSize = 16; c.epochs = 3; c.rolloutSize = 64
                c.entropyCoefficient = 0.001; c.learningRate = 0.001; c.targetKL = 0.15
                if mouse {
                    p.channels.keyboard = false; p.channels.mouseMovement = true
                    c.allowedKeys = []; c.mouseMovement = true; c.explorationScale = 2
                }
                p.rlTraining = c
                let mode: MouseControlMode = mouse ? .absolute : .relative
                let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
                defer { try? FileManager.default.removeItem(at: root) }
                let engine = RLTrainingEngine()
                var snapshot = try await engine.prepare(profile: p, contextSchema: .empty, mouseMode: mode,
                    activeWeights: nil, checkpoint: root).0
                let actor = RLTrainablePolicy(policy: AgentPolicy(profile: p))
                let distribution = RLActionDistribution(profile: p, configuration: c, mouseMode: mode)
                var random = RLRandom(state: seed ^ 77)
                func evaluate() throws -> Double {
                    try RLWeightTransfer.install(snapshot.policy, in: actor.policy)
                    try RLWeightTransfer.install(snapshot.heads, in: actor.heads)
                    var score = 0.0
                    for cue in [false, true] {
                        let logits = actor.forward(observation(p, cue: cue)).0
                        let prediction = actor.policy.activatedPredictions(logits: logits).asArray(Float.self)
                        if mouse { score += 1 - abs(Double(prediction[0]) - (cue ? 0.8 : 0.2)) }
                        else { score += (prediction[cue ? 16 : 14] >= 0.5 && prediction[cue ? 14 : 16] < 0.5) ? 1 : 0 }
                    }
                    return score / 2
                }
                let baseline = try evaluate()
                var last = baseline
                for rollout in 0..<40 {
                    try RLWeightTransfer.install(snapshot.policy, in: actor.policy)
                    try RLWeightTransfer.install(snapshot.heads, in: actor.heads)
                    var rows: [RLTransition] = []
                    for i in 0..<64 {
                        let cue = i.isMultiple(of: 2)
                        let obs = observation(p, cue: cue), forward = actor.forward(obs)
                        MLX.eval(forward.0, forward.1)
                        let sampled = distribution.sample(logits: forward.0.asArray(Float.self),
                            logStd: actor.heads.logStandardDeviation.asArray(Float.self), random: &random)
                        let reward: Float
                        if mouse {
                            let error = sampled.prediction[0] - (cue ? 0.8 : 0.2)
                            reward = 1 - 4 * error * error - pow(sampled.prediction[1] - 0.5, 2)
                        } else {
                            reward = sampled.binary[cue ? 16 : 14] > 0 && sampled.binary[cue ? 14 : 16] == 0 ? 1 : -1
                        }
                        let decision = RLDecision(observation: obs, latent: sampled.latent, binary: sampled.binary,
                            logProbability: sampled.logProbability, value: forward.1.item(Float.self),
                            timestamp: UInt64(rollout * 64 + i + 1), revision: snapshot.revision)
                        rows.append(RLTransition(decisions: [decision], reward: reward, duration: 1 / 60, terminal: true))
                    }
                    let result = try await engine.update(rows); snapshot = result.snapshot
                    last = try evaluate()
                    if rollout.isMultiple(of: 5) { print("RL_LEARNING task=\(mouse ? "mouse" : "keyboard") seed=\(seed) rollout=\(rollout) baseline=\(baseline) score=\(last)") }
                }
                print("RL_LEARNING_FINAL task=\(mouse ? "mouse" : "keyboard") seed=\(seed) baseline=\(baseline) score=\(last)")
                XCTAssertGreaterThan(last, mouse ? 0.88 : 0.85)
                XCTAssertGreaterThan(last, baseline + (mouse ? 0.1 : 0.2))
                report.append(["task": mouse ? "mouse" : "keyboard", "seed": seed, "baseline": baseline, "score": last])
            }
        }
        if let output = ProcessInfo.processInfo.environment["AGENTTRAINER_RL_LEARNING_REPORT"] {
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: URL(fileURLWithPath: output), options: .atomic)
        }
    }
}

extension RLTests {
    func testContextWideningKeepsPretrainedPredictions() throws {
        let p = profile(), base = AgentPolicy(profile: p)
        base.train(false)
        let old = ModelContextSchema.empty
        let context = ModelContextSchema(fields: [ContextFieldDefinition(id: UUID(), name: "Task", values: [ContextValueDefinition(id: UUID(), name: "One")])])
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: root) }
        try base.saveWeights(to: root)
        let widened = AgentPolicy(profile: p, contextSchema: context); widened.train(false)
        try widened.loadCompatibleWeights(from: root, sourceContext: old)
        let obs = observation(p)
        let images = VisionPreprocessor.mlxTensor(obs.current, batch: 1, spec: p.preprocessing)
        let timing = MLXArray(obs.timing, [1, TemporalInputContract.valueCount])
        let a = base.currentOnlyPredictions(currentImages: images, currentTiming: timing)
        let b = widened.currentOnlyPredictions(currentImages: images, currentTiming: timing)
        XCTAssertLessThan((a - b).abs().max().item(Float.self), 1e-6)
    }

    func testMaskErasesRewardPixelsBeforeEveryResizePolicy() throws {
        let width = 32, height = 16
        func buffer(white: Bool) throws -> CVPixelBuffer {
            var result: CVPixelBuffer?
            let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                [kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary, &result)
            XCTAssertEqual(status, kCVReturnSuccess)
            let image = try XCTUnwrap(result)
            CVPixelBufferLockBaseAddress(image, [])
            let pointer = CVPixelBufferGetBaseAddress(image)!.assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRow(image)
            for y in 0..<height { for x in 0..<width {
                let inside = x >= 8 && x < 24 && y >= 4 && y < 12
                for channel in 0..<3 { pointer[y * stride + x * 4 + channel] = white && inside ? 255 : 0 }
                pointer[y * stride + x * 4 + 3] = 255
            } }
            CVPixelBufferUnlockBaseAddress(image, []); return image
        }
        let text = try buffer(white: true), blank = try buffer(white: false), preprocessor = try VisionPreprocessor()
        for policy in ResizePolicy.allCases {
            for mode in ColorMode.allCases {
                let spec = PreprocessingSpec(width: 24, height: 24, colorMode: mode, resizePolicy: policy)
                let expected = try preprocessor.process(blank, spec: spec)
                let masked = try preprocessor.submit(text, specs: [spec], mask: CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5))[0].withPackedBytes { Data($0) }
                XCTAssertEqual(expected, masked, "\(policy), \(mode)")
                XCTAssertNotEqual(expected, try preprocessor.process(text, spec: spec))
            }
        }
    }

    func testRewardRegionUsesSourceLocalCoordinatesAndTracksMovement() {
        let region = RLRewardRegion(sourceID: 7, isWindow: true, normalized: CodableRect(CGRect(x: 0.5, y: 0.25, width: 0.25, height: 0.1)))
        let a = region.rect(in: CGRect(x: -1200, y: 80, width: 1000, height: 800))
        let b = region.rect(in: CGRect(x: 100, y: -300, width: 1000, height: 800))
        XCTAssertEqual(a.width, 250); XCTAssertEqual(a.height, 80)
        XCTAssertEqual(b.minX - a.minX, 1300); XCTAssertEqual(b.minY - a.minY, -380)
        XCTAssertTrue(region.isValid)
    }
}

extension RLTests {
    func testCompiledRuntimeGraphCapturesActorAndExplorationState() throws {
        let p = profile(), policy = AgentPolicy(profile: p), heads = RLHeads(width: policy.fusionWidth)
        policy.train(false)
        let inputs = [MLXArray.zeros([1, p.training.architecture.visualEmbedding]),
                      MLXArray.zeros([1, TemporalInputContract.valueCount])]
        let rl = RuntimePolicyGraph.make(model: policy, heads: heads)
        let normal = RuntimePolicyGraph.make(model: policy, heads: nil)
        let a = rl(inputs), b = normal(inputs)
        MLX.eval(a, b)
        XCTAssertEqual(a.count, 4); XCTAssertEqual(a[0].asArray(Float.self), b[0].asArray(Float.self))
        XCTAssertEqual(a[3].asArray(Float.self), heads.logStandardDeviation.asArray(Float.self))
        heads.update(parameters: ModuleParameters.unflattened(["logStandardDeviation": MLXArray.full([8], values: MLXArray(Float(-1)))]))
        XCTAssertEqual(rl(inputs)[3].asArray(Float.self), [Float](repeating: -1, count: 8))
    }
}

extension RLTests {
    func testPastEmbeddingRefreshUsesReducedCoordinateGridAndDeduplicates() {
        var p = profile()
        p.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 2, downsampleFactor: 2)
        let model = AgentPolicy(profile: p); model.train(false)
        let spec = p.training.effectiveTemporalVision.pastFrameSpec(from: p.preprocessing)
        let bytes = Data(repeating: 80, count: spec.sampleByteCount)
        let result = RLPastEncoder.encode([bytes, bytes], model: model)
        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0].shape, [1, p.training.architecture.visualEmbedding])
        XCTAssertEqual(result[0].asArray(Float.self), result[1].asArray(Float.self))
    }
}

extension RLTests {
    func testLearningContractIgnoresPresentationAndCapturePreferences() {
        let original = RLConfiguration()
        var edited = original
        edited.advanced = true; edited.episodeDelay = 2; edited.autosaveSeconds = 10
        edited.minimumConfidence = 0.95; edited.trainingComputeFraction = 0.2
        XCTAssertTrue(edited.hasSameLearningContract(as: original))
        edited.learningRate *= 2
        XCTAssertFalse(edited.hasSameLearningContract(as: original))
    }

    func testSettingsAutosaveCannotRestoreAnOlderBrainOrEraseProgress() {
        let currentID = UUID()
        var current = profile(); current.activeVersionID = currentID
        current.rlProgress = RLSavedSummary(updates: 12, episodes: 8, trainingSeconds: 90, experienceSeconds: 60)
        var stale = current; stale.activeVersionID = UUID(); stale.rlProgress = nil
        stale.training.learningRate = 0.0002
        let merged = stale.preservingLearning(from: current)
        XCTAssertEqual(merged.activeVersionID, currentID); XCTAssertEqual(merged.rlProgress, current.rlProgress)
        XCTAssertEqual(merged.training.learningRate, 0.0002)
        XCTAssertEqual(merged.trainingDurationSummary(recordings: []).trainingSeconds, 90)
    }

    func testReloadingAPureRLProfileDoesNotCountRLTimeAsDemonstrationTime() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root); try await store.prepare()
        var p = profile(); let id = UUID()
        p.activeVersionID = id; p.rlProgress = RLSavedSummary(updates: 3, episodes: 2, trainingSeconds: 90, experienceSeconds: 60)
        try await store.saveProfile(p)
        let version = ModelVersionManifest(id: id, name: "Pure RL", createdAt: Date(), globalStep: 4, trainingLoss: 0,
            preprocessing: p.preprocessing, channels: p.channels, training: p.training,
            trainingDurationSeconds: 90, experienceDurationSeconds: 60, rlSummary: p.rlProgress)
        try await store.saveVersionManifest(version, profileID: p.id)
        let profiles = await store.listProfiles()
        let loaded = try XCTUnwrap(profiles.first)
        XCTAssertNil(loaded.trainingProgress)
        XCTAssertEqual(loaded.trainingDurationSummary(recordings: []).trainingSeconds, 90)
        XCTAssertEqual(loaded.trainingDurationSummary(recordings: []).experienceSeconds, 60)
    }

    func testRLVersionActivationRestoresItsCheckpointAndFailedPublicationPreservesActiveBrain() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root); try await store.prepare()
        var p = profile(); try await store.saveProfile(p)
        let engine = RLTrainingEngine()
        _ = try await engine.prepare(profile: p, contextSchema: .empty, mouseMode: .relative, activeWeights: nil, checkpoint: root.appendingPathComponent("missing"))
        let firstID = UUID(), stage = root.appendingPathComponent("stage")
        _ = try await engine.save(to: stage, versionID: firstID, actorRandom: RLRandom(state: 3), history: [], episodes: 2, elapsed: 10, pending: [])
        let first = ModelVersionManifest(id: firstID, name: "RL First", createdAt: Date(), globalStep: 0,
            trainingLoss: 0, preprocessing: p.preprocessing, channels: p.channels, training: p.training,
            relativeMouseScale: GameCameraContract.deltaScale, outputCapabilities: p.effectiveRL.capabilities(profile: p, mouseMode: .relative),
            rlSummary: RLSavedSummary(updates: 0, episodes: 2))
        p = try await store.publishRL(profileID: p.id, version: first, staged: stage, keeping: 10)
        let secondID = UUID(), secondStage = root.appendingPathComponent("second")
        _ = try await engine.save(to: secondStage, versionID: secondID, actorRandom: RLRandom(state: 5), history: [], episodes: 3, elapsed: 20, pending: [])
        var second = first; second.id = secondID; second.name = "RL Second"
        p = try await store.publishRL(profileID: p.id, version: second, staged: secondStage, keeping: 10)
        let activated = try await store.activateVersion(profile: p, versionID: firstID)
        XCTAssertTrue(activated.checkpointIsResumable); XCTAssertEqual(activated.profile.activeVersionID, firstID)
        let checkpoint = await store.profileDirectory(p.id).appendingPathComponent("RLCheckpoint/state.json")
        let restored = try JSONDecoder().decode(RLLearningState.self, from: Data(contentsOf: checkpoint))
        XCTAssertEqual(restored.activeVersionID, firstID); XCTAssertEqual(restored.actorRandom.state, 3)
        // Emulate a crash after swapping in another checkpoint but before
        // committing its profile pointer. Recovery must use the active version.
        try JSONEncoder().encode(secondID).write(to: checkpoint.deletingLastPathComponent().appendingPathComponent("publication.json"), options: .atomic)
        let recovery = await store.rlCheckpointDirectory(profile: activated.profile)
        let committed = await store.versionDirectory(profileID: p.id, versionID: firstID)
        XCTAssertEqual(recovery, committed)
        let broken = root.appendingPathComponent("broken")
        try FileManager.default.createDirectory(at: broken, withIntermediateDirectories: true)
        do { _ = try await store.publishRL(profileID: p.id, version: second, staged: broken, keeping: 10); XCTFail("Incomplete publication accepted") }
        catch { }
        let active = await store.listProfiles().first { $0.id == p.id }
        XCTAssertEqual(active?.activeVersionID, firstID)
        let failingID = UUID(), failingStage = root.appendingPathComponent("commit-failure")
        _ = try await engine.save(to: failingStage, versionID: failingID, actorRandom: RLRandom(state: 9), history: [], episodes: 5, elapsed: 30, pending: [])
        var failing = first; failing.id = failingID
        let profileFile = await store.profileDirectory(p.id).appendingPathComponent("profile.json")
        // A readable but immutable destination fails at the final metadata
        // commit, after the new version/checkpoint have already been installed.
        try FileManager.default.setAttributes([.immutable: true], ofItemAtPath: profileFile.path)
        defer { try? FileManager.default.setAttributes([.immutable: false], ofItemAtPath: profileFile.path) }
        do { _ = try await store.publishRL(profileID: p.id, version: failing, staged: failingStage, keeping: 10); XCTFail("Immutable profile commit unexpectedly succeeded") }
        catch { }
        try FileManager.default.setAttributes([.immutable: false], ofItemAtPath: profileFile.path)
        let afterFailure = await store.listProfiles().first { $0.id == p.id }
        XCTAssertEqual(afterFailure?.activeVersionID, firstID)
        let rolledBack = try JSONDecoder().decode(RLLearningState.self, from: Data(contentsOf: checkpoint))
        XCTAssertEqual(rolledBack.activeVersionID, firstID); XCTAssertEqual(rolledBack.actorRandom.state, 3)
        let failedVersion = await store.versionDirectory(profileID: p.id, versionID: failingID)
        XCTAssertFalse(FileManager.default.fileExists(atPath: failedVersion.path))
    }
}

extension RLTests {
    func testFixedShapePPOCoefficientMatchesAutomaticDifferentiation() {
        for advantage: Float in [-2, -0.25, 0, 0.5, 3] {
            for ratio: Float in [-0.7, -0.1, 0.05, 0.8] {
                let reference = grad { x in
                    let r = exp(clip(x, min: -20, max: 20))
                    return -minimum(r * advantage, clip(r, min: 0.8, max: 1.2) * advantage)
                }(MLXArray(ratio)).item(Float.self)
                XCTAssertEqual(RLPPOObjective.evaluate(logRatio: ratio, advantage: advantage, clip: 0.2).coefficient, reference, accuracy: 1e-5)
            }
        }
    }
}

extension RLTests {
    @MainActor
    func testCancelledAutosaveAndOverlappingStopsCompleteWithoutLosingTheCheckpoint() async throws {
        let store = WorkspaceStore.shared
        let p = profile(); try await store.saveProfile(p)
        defer { Task { try? await store.deleteProfile(p) } }
        let session = RLSession(profile: p, runtime: try AgentRuntime(), mouseMode: .relative, context: .empty)
        _ = try await session.prepare()
        let manual = Task { await session.save() }
        await Task.yield()
        let autosave = Task { await session.save(autosave: true) }; autosave.cancel()
        await manual.value; await autosave.value
        var notifications = 0
        session.onStopped = { _ in notifications += 1 }
        async let first: Void = session.stop()
        async let second: Void = session.stop()
        _ = await (first, second)
        XCTAssertEqual(notifications, 1); XCTAssertNil(session.saveError)
        let checkpoint = await store.rlCheckpointDirectory(profile: session.currentProfile)
        XCTAssertTrue(FileManager.default.fileExists(atPath: checkpoint.appendingPathComponent("optimizer.safetensors").path))
    }

    func testLongRunningHistoryKeepsTheLifetimeBestRewardWithinABoundedBudget() {
        let history = [RLEpisodeRecord(id: 1, reward: 100, duration: 1)]
            + (2...10002).map { RLEpisodeRecord(id: $0, reward: 1, duration: 1) }
        let bounded = RLEpisodeRecord.boundedHistory(history)
        XCTAssertEqual(bounded.count, 10000); XCTAssertEqual(bounded.first?.reward, 100)
        XCTAssertEqual(bounded.last?.id, 10002)
    }

    func testBufferPressurePreservesTerminalRewardAndRefusesToDropTheCurrentTail() {
        let p = profile()
        let decision = RLDecision(observation: observation(p), latent: Array(repeating: 0, count: 8),
            binary: Array(repeating: 0, count: ActionLayout.count), logProbability: 0, value: 2, timestamp: 1, revision: 0)
        var rows = [RLTransition(decisions: [decision], reward: -50, duration: 1, terminal: true),
                    RLTransition(decisions: [decision], reward: 1, duration: 1, terminal: false),
                    RLTransition(decisions: [decision], reward: 2, duration: 1, terminal: false)]
        let result = RLExperienceBuffer.trim(&rows, maximumActions: 2, maximumBytes: 1_000_000)
        XCTAssertEqual(result.discarded, 1); XCTAssertFalse(result.overflow)
        XCTAssertEqual(rows.first?.reward, -50); XCTAssertTrue(rows.first!.terminal)
        let refused = RLExperienceBuffer.trim(&rows, maximumActions: 1, maximumBytes: 1_000_000)
        XCTAssertTrue(refused.overflow); XCTAssertEqual(rows.count, 2)
    }

    func testDecisionInboxUsesCaptureTimeInsteadOfRecognitionCompletionTime() {
        let p = profile(), inbox = RLDecisionInbox()
        for t: UInt64 in [10, 20, 30] {
            inbox.append(RLDecision(observation: observation(p), latent: Array(repeating: 0, count: 8),
                binary: Array(repeating: 0, count: ActionLayout.count), logProbability: 0, value: Float(t), timestamp: t, revision: 0))
        }
        let delayed = inbox.take(before: 25)
        XCTAssertEqual(delayed.records.map(\.timestamp), [10, 20]); XCTAssertEqual(delayed.nextValue, 30)
        XCTAssertTrue(inbox.take(before: 25).records.isEmpty)
        XCTAssertEqual(inbox.take(before: 31).records.map(\.timestamp), [30])
    }

    func testNewRLShortcutPreservesPreviouslyCustomizedShortcuts() {
        var settings = HotkeySettings()
        settings.run = settings.rlSessionToggle
        let migrated = settings.preservingExistingShortcuts()
        XCTAssertEqual(migrated.run, settings.run); XCTAssertEqual(migrated.panic, settings.panic)
        XCTAssertEqual(migrated.record, settings.record)
        XCTAssertEqual(Set([migrated.panic, migrated.record, migrated.run, migrated.rlSessionToggle]).count, 4)
    }

    func testCustomRLShortcutSurvivesSavingAndRelaunching() throws {
        var settings = HotkeySettings()
        settings.rlSessionToggle = HotkeyBinding(keyCode: 17, carbonModifiers: UInt32(1 << 12 | 1 << 11))
        XCTAssertEqual(try JSONDecoder().decode(HotkeySettings.self, from: JSONEncoder().encode(settings)), settings)
    }

    func testPolicyPrecisionChangePreservesWeightsByCastingInsteadOfReinitializing() throws {
        var p = profile()
        let old = AgentPolicy(profile: p)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: file) }
        try old.saveWeights(to: file)
        p.training.precision = .bfloat16
        let current = AgentPolicy(profile: p)
        try current.loadWeights(from: file)
        let before = Dictionary(uniqueKeysWithValues: old.parameters().flattened())
        for (name, parameter) in current.parameters().flattened() {
            XCTAssertEqual(parameter.dtype, .bfloat16)
            XCTAssertEqual(parameter.asArray(Float.self), before[name]!.asType(.bfloat16).asArray(Float.self))
        }
    }
}
