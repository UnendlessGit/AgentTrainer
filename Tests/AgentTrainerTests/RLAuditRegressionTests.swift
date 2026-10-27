import XCTest
import AppKit
import MLX
import MLXNN
@testable import AgentTrainer

final class RLAuditRegressionTests: XCTestCase {
    private func decision(_ value: Float = 0) -> RLDecision {
        RLDecision(observation: RLObservation(current: Data([0]), past: [], controls: [], timing: [0], pastTiming: [], context: []),
            latent: Array(repeating: 0, count: 8), binary: Array(repeating: 0, count: ActionLayout.count),
            logProbability: 0, value: value, timestamp: 1, revision: 0)
    }

    func testRetainedMultiActionTailSuppliesImmediateSuccessorValue() {
        var transitions = [RLTransition(decisions: [decision(1)], reward: 2, duration: 0.1, terminal: false),
            RLTransition(decisions: [decision(10), decision(40)], reward: 3, duration: 0.2, terminal: false)]
        let prefix = RLExperienceBuffer.takeReady(&transitions, minimumActions: 1, force: false, fallbackValue: 90)!
        XCTAssertEqual(prefix.count, 1); XCTAssertEqual(transitions.count, 1)
        XCTAssertEqual(prefix[0].bootstrapValue, 10)
        var c = RLConfiguration(); c.discount = 0.9; c.gaeLambda = 1
        XCTAssertEqual(RLReturns.advantages(prefix, values: [1], configuration: c, actionFPS: 10).returns[0], 11, accuracy: 1e-5)
        transitions[0].terminal = true
        let terminal = RLExperienceBuffer.takeReady(&transitions, minimumActions: 999, force: false, fallbackValue: 90)!
        XCTAssertEqual(terminal[0].bootstrapValue, 0); XCTAssertFalse(terminal[0].boundary)
    }

    func testEndKeyIsConsumedOnceForTimeoutThenTerminalAcknowledgment() {
        var keys = RLEpisodeKeyState()
        XCTAssertEqual(keys.takeEnd(15), 15) // timeout; start and end both R
        XCTAssertNil(keys.takeEnd(15)) // terminal acknowledgment
        keys.reset(); XCTAssertEqual(keys.takeEnd(15), 15)
        keys.reset(); XCTAssertNil(keys.takeEnd(nil)); XCTAssertTrue(keys.endSent)
    }

    func testEffectiveDistributionExcludesPermissionsAndEquivalentModifiers() {
        var profile = AIProfile.fresh()
        profile.rlTraining = RLConfiguration()
        profile.rlTraining!.allowedKeys = [0, 56, 60, 58]
        profile.rlTraining!.allowedButtons = [0]
        profile.rlTraining!.mouseMovement = true; profile.rlTraining!.scroll = true
        profile.restrictions = ActionRestrictions(blockedKeyCodes: [56])
        let distribution = RLActionDistribution(profile: profile, configuration: profile.effectiveRL, mouseMode: .absolute)
        XCTAssertFalse(distribution.binaryIndices.contains(142))
        XCTAssertTrue(distribution.binaryIndices.contains(144))
        let restricted = RLControlContract.profile(profile, permissions: RuntimeOutputPermissions(cursorMovement: false, keyboard: false))
        let limited = RLActionDistribution(profile: restricted, configuration: restricted.effectiveRL, mouseMode: .absolute)
        XCTAssertEqual(limited.binaryIndices, [4]); XCTAssertEqual(limited.continuousIndices, [4, 5])
    }

    func testButtonAndScrollOnlyActionsRespectRegionWithoutMovingTheCursor() {
        for inside in [false, true] {
            let events = AuditEvents()
            let point = inside ? CGPoint(x: 50, y: 50) : CGPoint(x: 500, y: 500)
            let injector = InputInjector(eventSink: { events.append($0) }, cursorLocation: { point })
            var profile = AIProfile.fresh(); profile.channels.mouseMovement = false; profile.channels.keyboard = false
            var prediction = Array(repeating: Float(0), count: ActionLayout.count)
            prediction[4] = 1; prediction[13] = 1
            injector.enable(outputPermissions: RuntimeOutputPermissions(cursorMovement: false, keyboard: false))
            injector.execute(prediction, profile: profile, allowedKeyCodes: [], mouseMode: .absolute,
                captureRect: CGRect(x: 0, y: 0, width: 100, height: 100), safety: AgentSafetyPolicy())
            injector.releaseAll()
            XCTAssertEqual(events.rows.contains { $0.type == .leftMouseDown }, inside)
            XCTAssertEqual(events.rows.contains { $0.type == .leftMouseUp }, inside)
            XCTAssertEqual(events.rows.contains { $0.type == .scrollWheel }, inside)
            XCTAssertFalse(events.rows.contains { $0.type == .mouseMoved })
            if inside { XCTAssertTrue(events.rows.allSatisfy { $0.location == point }) }
        }
    }

    func testMoveThenScrollTargetsTheNewPositionAndAllButtonsRelease() {
        let events = AuditEvents()
        let injector = InputInjector(eventSink: { events.append($0) }, cursorLocation: { CGPoint(x: 500, y: 500) })
        var profile = AIProfile.fresh(); profile.channels.keyboard = false
        var prediction = Array(repeating: Float(0), count: ActionLayout.count)
        prediction[0] = 0.25; prediction[1] = 0.75; prediction[12] = 1
        for button in 0..<8 { prediction[4 + button] = 1 }
        injector.enable()
        injector.execute(prediction, profile: profile, allowedKeyCodes: [], mouseMode: .absolute,
            captureRect: CGRect(x: 0, y: 0, width: 100, height: 100), safety: AgentSafetyPolicy())
        injector.releaseAll()
        XCTAssertEqual(events.rows.first { $0.type == .scrollWheel }?.location, CGPoint(x: 25, y: 75))
        for button in 0..<8 {
            XCTAssertTrue(events.rows.contains { $0.button == button && [.leftMouseDown, .rightMouseDown, .otherMouseDown].contains($0.type) })
            XCTAssertTrue(events.rows.contains { $0.button == button && [.leftMouseUp, .rightMouseUp, .otherMouseUp].contains($0.type) })
        }
    }

    func testExperienceBudgetIncludesTimingControlsAndActionStorage() {
        var d = decision()
        let pixels = d.observation.current.count
        d.observation.pastTiming = Array(repeating: 0, count: 8 * TemporalInputContract.valueCount)
        d.observation.controls = Array(repeating: 0, count: 8 * ActionLayout.count)
        XCTAssertGreaterThan(RLExperienceBuffer.bytes(d), pixels + 70_000)
        let inbox = RLDecisionInbox(); inbox.append(d)
        XCTAssertEqual(inbox.discardAll(), 1); XCTAssertEqual(inbox.discardAll(), 0)
    }

    func testRetinaIntegerZeroIsRecognizedWithoutGuessingOrReducingConfidence() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/RLTerminalRetina.png")
        let image = try XCTUnwrap(NSImage(contentsOf: url)?.cgImage(forProposedRect: nil, context: nil, hints: nil))
        let result = try XCTUnwrap(RewardReader.validatedPair(in: image, minimumConfidence: 0.8))
        XCTAssertEqual(result.0, 0.62, accuracy: 1e-6); XCTAssertEqual(result.1, 0)
        XCTAssertNil(try RewardReader.validatedPair(in: image, minimumConfidence: 1.01))
    }

    func testConstantRLScheduleDoesNotDecayOrRestart() {
        let optimizer = ResumableAdamW(learningRate: 0.0001, weightDecay: 0, warmupSteps: 1, schedule: .constant)
        for step in [1, 500, 500_000, 1_000_000, 2_000_001] {
            XCTAssertEqual(optimizer.effectiveLearningRate(at: step), 0.0001, accuracy: 1e-10)
        }
        XCTAssertEqual(TrainingConfiguration().effectiveLearningRateSchedule, .adaptiveCosine)
    }

    func testCorruptOptimizerMomentsAreRejectedBeforeTheyCanPoisonTheBrain() throws {
        let model = Linear(2, 1)
        let optimizer = ResumableAdamW(learningRate: 0.001, weightDecay: 0)
        optimizer.initialize(model: model)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let valid = root.appendingPathComponent("valid.safetensors")
        try optimizer.save(to: valid)
        let saved = try MLX.loadArraysAndMetadata(url: valid)
        for badVariance in [Float.nan, Float(-1)] {
            var arrays = saved.0
            let name = try XCTUnwrap(arrays.keys.first { $0.hasPrefix("v.") })
            arrays[name] = MLXArray.full(arrays[name]!.shape, values: MLXArray(badVariance))
            let corrupt = root.appendingPathComponent("corrupt.safetensors")
            try MLX.save(arrays: arrays, metadata: saved.1, url: corrupt)
            let restored = ResumableAdamW(learningRate: 0.001, weightDecay: 0)
            restored.initialize(model: model)
            XCTAssertThrowsError(try restored.load(from: corrupt))
        }
    }

    func testKLRejectionDoesNotAdvanceWeightsStepsOrPolicyAge() async throws {
        var p = AIProfile.fresh()
        p.preprocessing = PreprocessingSpec(width: 16, height: 16, colorMode: .grayscale)
        p.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 0)
        p.training.precision = .float32
        p.training.architecture.convolutionChannels = [4]; p.training.architecture.kernelSizes = [3]
        p.training.architecture.strides = [2]; p.training.architecture.visualEmbedding = 8
        p.training.architecture.recurrentWidth = 8; p.training.architecture.fusionWidths = [16]
        p.training.architecture.dropout = 0; p.training.architecture.attentionHeads = 2
        var c = RLConfiguration(); c.allowedKeys = [0]; c.normalizeRewards = false; p.rlTraining = c
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let engine = RLTrainingEngine()
        let initial = try await engine.prepare(profile: p, contextSchema: .empty, mouseMode: .absolute,
            activeWeights: nil, checkpoint: root)
        var d = decision()
        d.observation.current = Data(repeating: 0, count: p.preprocessing.sampleByteCount)
        d.observation.timing = Array(repeating: 0, count: TemporalInputContract.valueCount)
        d.logProbability = 1000 // grossly inconsistent behavior density
        let rejected = try await engine.update([RLTransition(decisions: [d], reward: 1, duration: 0.1, terminal: true)])
        XCTAssertEqual(rejected.steps, 0); XCTAssertEqual(rejected.updates, 0)
        XCTAssertEqual(rejected.snapshot.revision, initial.0.revision)
        XCTAssertEqual(rejected.attemptedUpdates, 1); XCTAssertEqual(rejected.rejectedUpdates, 1)
        XCTAssertEqual(rejected.klRejectedActions, 1)
        for (name, tensor) in initial.0.policy { XCTAssertEqual(tensor.values, rejected.snapshot.policy[name]?.values) }
        d.value = .nan
        do {
            _ = try await engine.update([RLTransition(decisions: [d], reward: 1, duration: 0.1, terminal: true)])
            XCTFail("Nonfinite experience must fail before any optimizer work")
        } catch {}
        let state = try await engine.save(to: root, versionID: UUID(), actorRandom: RLRandom(state: 1),
            history: [], episodes: 0, elapsed: 0, pending: [])
        XCTAssertEqual(state.optimizerSteps, 0); XCTAssertTrue(state.pending.isEmpty)
    }

    func testElapsedReturnsAndJointLikelihoodKeepTheirDocumentedUnits() {
        var c = RLConfiguration(); c.discount = 0.99; c.gaeLambda = 1
        let row = RLTransition(decisions: [decision()], reward: 1, duration: 0.2, terminal: false, boundary: true, bootstrapValue: 3)
        let at15 = RLReturns.advantages([row], values: [0], configuration: c, actionFPS: 15).returns[0]
        c.discount = pow(0.99, 15 / 60.0)
        XCTAssertEqual(RLReturns.advantages([row], values: [0], configuration: c, actionFPS: 60).returns[0], at15, accuracy: 1e-6)
        for count in [1, 2, 4, 8] {
            let ratio = Float(count) * 0.1
            let objective = RLPPOObjective.evaluate(logRatio: ratio, advantage: 1, clip: 0.2)
            XCTAssertEqual(objective.kl, exp(ratio) - 1 - ratio, accuracy: 1e-6)
            XCTAssertTrue(objective.loss.isFinite)
        }
    }

    func testCachedRuntimeAndLearnerReplayAgreeWithAndWithoutHistory() throws {
        for history in [0, 2] {
            var p = AIProfile.fresh()
            p.preprocessing = PreprocessingSpec(width: 16, height: 16, colorMode: history == 0 ? .grayscale : .color)
            p.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: history)
            p.training.precision = .float32
            p.training.architecture.convolutionChannels = [4]; p.training.architecture.kernelSizes = [3]
            p.training.architecture.strides = [2]; p.training.architecture.visualEmbedding = 8
            p.training.architecture.recurrentWidth = 8; p.training.architecture.fusionWidths = [16]
            p.training.architecture.dropout = 0; p.training.architecture.attentionHeads = 2
            let learner = RLTrainablePolicy(policy: AgentPolicy(profile: p))
            let policy = learner.policy
            let pastSpec = p.training.effectiveTemporalVision.pastFrameSpec(from: p.preprocessing)
            let obs = RLObservation(current: Data(repeating: 137, count: p.preprocessing.sampleByteCount),
                past: Array(repeating: Data(repeating: 63, count: pastSpec.sampleByteCount), count: history),
                controls: Array(repeating: 0.2, count: history * ActionLayout.count),
                timing: Array(repeating: 0.1, count: TemporalInputContract.valueCount),
                pastTiming: Array(repeating: 0.3, count: history * TemporalInputContract.valueCount), context: [])
            let image = VisionPreprocessor.mlxTensor(obs.current, batch: 1, spec: p.preprocessing)
            let current = policy.visualEmbedding(visualFeatures: policy.visualActivations(images: image).last!)
            var inputs = [current]
            if history > 0 {
                inputs += [stacked(RLPastEncoder.encode(obs.past, model: policy), axis: 1),
                    MLXArray(obs.controls, [1, history, ActionLayout.count])]
            }
            inputs.append(MLXArray(obs.timing, [1, TemporalInputContract.valueCount]))
            if history > 0 { inputs.append(MLXArray(obs.pastTiming, [1, history, TemporalInputContract.valueCount])) }
            let runtime = RuntimePolicyGraph.make(model: policy, heads: learner.heads)
            let actual = runtime(inputs), replay = learner.forward(obs)
            MLX.eval(actual); MLX.eval(replay.0, replay.1)
            XCTAssertLessThan((abs(actual[1] - replay.0)).max().item(Float.self), 1e-5)
            XCTAssertLessThan((abs(actual[2] - replay.1)).max().item(Float.self), 1e-5)
        }
    }

    func testInterruptedMinibatchResumesExactlyAndLegacyStateKeepsActor() async throws {
        var p = AIProfile.fresh()
        p.preprocessing = PreprocessingSpec(width: 16, height: 16, colorMode: .grayscale)
        p.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 0); p.training.precision = .float32
        p.training.architecture.convolutionChannels = [4]; p.training.architecture.kernelSizes = [3]
        p.training.architecture.strides = [2]; p.training.architecture.visualEmbedding = 8
        p.training.architecture.recurrentWidth = 8; p.training.architecture.fusionWidths = [16]
        p.training.architecture.dropout = 0; p.training.architecture.attentionHeads = 2
        var c = RLConfiguration(); c.allowedKeys = [0, 2]; c.batchSize = 4; c.epochs = 2
        c.normalizeRewards = false; c.targetKL = 0.5; c.learningRate = 0.00001; p.rlTraining = c
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let stop = StepInterruption()
        let interrupted = RLTrainingEngine(didApplyStep: { step in if step == 1 { stop.apply() } })
        stop.engine = interrupted
        let initial = try await interrupted.prepare(profile: p, contextSchema: .empty, mouseMode: .absolute,
            activeWeights: nil, checkpoint: root.appendingPathComponent("missing"))
        let reference = RLTrainingEngine()
        _ = try await reference.prepare(profile: p, contextSchema: .empty, mouseMode: .absolute,
            activeWeights: nil, checkpoint: root.appendingPathComponent("missing"))
        let actor = RLTrainablePolicy(policy: AgentPolicy(profile: p))
        try RLWeightTransfer.install(initial.0.policy, in: actor.policy); try RLWeightTransfer.install(initial.0.heads, in: actor.heads)
        let distribution = RLActionDistribution(profile: p, configuration: c, mouseMode: .absolute)
        var random = RLRandom(state: 55), rows: [RLTransition] = []
        for i in 0..<16 {
            var d = decision()
            d.observation.current = Data(repeating: i.isMultiple(of: 2) ? 30 : 200, count: p.preprocessing.sampleByteCount)
            d.observation.timing = Array(repeating: 0, count: TemporalInputContract.valueCount)
            let forward = actor.forward(d.observation); MLX.eval(forward.0, forward.1)
            let sample = distribution.sample(logits: forward.0.asArray(Float.self), logStd: actor.heads.logStandardDeviation.asArray(Float.self), random: &random)
            d.latent = sample.latent; d.binary = sample.binary; d.logProbability = sample.logProbability
            d.value = forward.1.item(Float.self)
            rows.append(RLTransition(decisions: [d], reward: d.binary[14] == 1 ? 1 : -1, duration: 1 / 15, terminal: true))
        }
        let partial = try await interrupted.update(rows)
        XCTAssertEqual(partial.steps, 1); XCTAssertEqual(partial.updates, 0)
        let id = UUID()
        let saved = try await interrupted.save(to: root, versionID: id, actorRandom: random, history: [], episodes: 1, elapsed: 1, pending: [])
        XCTAssertFalse(saved.pending.isEmpty); XCTAssertEqual(saved.offset, 4)
        p.activeVersionID = id
        let resumed = RLTrainingEngine()
        _ = try await resumed.prepare(profile: p, contextSchema: .empty, mouseMode: .absolute,
            activeWeights: root.appendingPathComponent("weights.safetensors"), checkpoint: root)
        let actual = try await resumed.update([]), expected = try await reference.update(rows)
        XCTAssertGreaterThan(actual.steps, 1); XCTAssertEqual(actual.steps, expected.steps)
        XCTAssertEqual(actual.updates, expected.updates)
        for (name, tensor) in expected.snapshot.policy { XCTAssertEqual(actual.snapshot.policy[name]?.values, tensor.values) }
        for (name, tensor) in expected.snapshot.heads { XCTAssertEqual(actual.snapshot.heads[name]?.values, tensor.values) }
        let next = root.appendingPathComponent("next")
        let sa = try await resumed.save(to: next, versionID: id, actorRandom: random, history: [], episodes: 1, elapsed: 1, pending: [])
        let ref = root.appendingPathComponent("reference")
        let sb = try await reference.save(to: ref, versionID: id, actorRandom: random, history: [], episodes: 1, elapsed: 1, pending: [])
        XCTAssertEqual(sa.random.state, sb.random.state)
        let oa = try MLX.loadArrays(url: next.appendingPathComponent("optimizer.safetensors"))
        let ob = try MLX.loadArrays(url: ref.appendingPathComponent("optimizer.safetensors"))
        for (name, tensor) in oa { XCTAssertEqual(tensor.asArray(Float.self), ob[name]?.asArray(Float.self)) }
        // Legacy pending work must not be replayed under corrected semantics.
        var legacy = try JSONSerialization.jsonObject(with: Data(contentsOf: next.appendingPathComponent("state.json"))) as! [String: Any]
        legacy.removeValue(forKey: "learningSemanticsVersion")
        try JSONSerialization.data(withJSONObject: legacy).write(to: next.appendingPathComponent("state.json"))
        let migrated = RLTrainingEngine()
        let restored = try await migrated.prepare(profile: p, contextSchema: .empty, mouseMode: .absolute,
            activeWeights: next.appendingPathComponent("weights.safetensors"), checkpoint: next)
        XCTAssertTrue(restored.1.pending.isEmpty); XCTAssertNotNil(restored.1.resumeNotice)
        XCTAssertEqual(restored.1.legacyUpdateAttempts, sa.updates)
        for (name, tensor) in actual.snapshot.policy { XCTAssertEqual(restored.0.policy[name]?.values, tensor.values) }
    }
}

private final class StepInterruption: @unchecked Sendable {
    weak var engine: RLTrainingEngine?
    func apply() { engine?.interrupt() }
}

private final class AuditEvents: @unchecked Sendable {
    struct Row { var type: CGEventType; var location: CGPoint; var button: Int }
    private let lock = NSLock()
    private var storage: [Row] = []
    var rows: [Row] { lock.withLock { storage } }
    func append(_ event: CGEvent) {
        lock.withLock { storage.append(Row(type: event.type, location: event.location,
            button: Int(event.getIntegerValueField(.mouseEventButtonNumber)))) }
    }
}
