import CoreGraphics
import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import AgentTrainer

final class TemporalInputTests: XCTestCase {
    private let origin: UInt64 = 9_000_000_000_000_000_000
    private let rect = CGRect(x: 0, y: 0, width: 100, height: 100)

    private func block(_ index: Int) -> Int {
        TemporalInputContract.clockCount + TemporalInputContract.controlLookup[index]! * TemporalInputContract.controlWidth
    }

    private func sample(_ millis: UInt64, key: UInt16 = 13, down: Bool) -> InputSample {
        InputSample(timestampNanos: origin + millis * 1_000_000, kind: .key, keyCode: key, isDown: down)
    }

    func testSubTickTapKeepsEdgesDurationAndDutyWithoutPretendingItIsHeld() {
        var timeline = CausalControlTimeline(origin: origin)
        timeline.consume(sample(5, down: true)); timeline.consume(sample(8, down: false))
        let values = timeline.features(at: origin + 16_000_000, frameTime: origin,
            perceptionInterval: nil, rect: rect)
        let base = block(ActionLayout.keyboard.lowerBound + 13)
        XCTAssertEqual(values[base], 0)
        XCTAssertEqual(values[base + 3], 0)
        XCTAssertEqual(values[base + 4], 0.011 / 0.25, accuracy: 1e-6)
        XCTAssertEqual(values[base + 7], 0.008 / 0.25, accuracy: 1e-6)
        XCTAssertEqual(values[base + 10], 0.003 / 0.25, accuracy: 1e-6)
        XCTAssertEqual(values[base + 12], 1)
        XCTAssertEqual(values[base + 13], Float(Foundation.log(2.0)), accuracy: 1e-6)
        XCTAssertEqual(values[base + 14], Float(Foundation.log(2.0)), accuracy: 1e-6)
        XCTAssertEqual(values[base + 15], 3.0 / 16, accuracy: 1e-6)
    }

    func testRepeatedTapsAndIdleWindowsKeepCompletedDurationAndRecency() {
        var timeline = CausalControlTimeline(origin: origin)
        for (time, down) in [(UInt64(1), true), (2, false), (3, true), (4, false)] {
            timeline.consume(sample(time, down: down))
        }
        let base = block(ActionLayout.keyboard.lowerBound + 13)
        let first = timeline.features(at: origin + 5_000_000, frameTime: origin,
            perceptionInterval: 0.005, rect: rect)
        XCTAssertEqual(first[base + 13], Float(Foundation.log(3.0)), accuracy: 1e-6)
        XCTAssertEqual(first[base + 15], 0.4, accuracy: 1e-6)
        timeline.endInterval(at: origin + 5_000_000)
        let later = timeline.features(at: origin + 20_000_000, frameTime: origin,
            perceptionInterval: 0.015, rect: rect)
        XCTAssertEqual(later[base + 13], 0)
        XCTAssertEqual(later[base + 15], 0)
        XCTAssertEqual(later[base + 10], first[base + 10])
        XCTAssertGreaterThan(later[base + 7], first[base + 7])
    }

    func testStartupCensoringAndPreTrimHoldsDoNotInventOnsets() {
        var timeline = CausalControlTimeline(origin: origin)
        timeline.consume(sample(0, down: true))
        let base = block(ActionLayout.keyboard.lowerBound + 13)
        var values = timeline.features(at: origin + 1_000_000_000, frameTime: origin,
            perceptionInterval: nil, rect: rect)
        XCTAssertEqual(values[base], 1)
        XCTAssertEqual(values[base + 3], 0)
        timeline.consume(sample(1_100, down: false))
        timeline.consume(sample(1_200, down: true))
        timeline.endInterval(at: origin + 2_000_000_000)
        values = timeline.features(at: origin + 8_000_000_000, frameTime: origin,
            perceptionInterval: 6, rect: rect)
        XCTAssertEqual(values[base + 3], 1)
        XCTAssertEqual(values[base + 2], Float(log1p(6.8) / Foundation.log(61.0)), accuracy: 1e-6)
        XCTAssertEqual(values[base + 15], 1)
        XCTAssertEqual(values[base + 12], 0)
    }

    func testMotionAndScrollUseObservedSpansAndRealIntervals() {
        var timeline = CausalControlTimeline(origin: origin)
        timeline.consume(InputSample(timestampNanos: origin + 2_000_000, kind: .mouseMove,
            x: 20, y: 30, deltaX: 8, deltaY: -4))
        timeline.consume(InputSample(timestampNanos: origin + 9_000_000, kind: .mouseMove,
            x: 30, y: 35, deltaX: 16, deltaY: 4))
        timeline.consume(InputSample(timestampNanos: origin + 10_000_000, kind: .scroll,
            x: 30, y: 35, scrollY: 2))
        let values = timeline.features(at: origin + 20_000_000, frameTime: origin + 5_000_000,
            perceptionInterval: 0.033, rect: rect)
        let base = TemporalInputContract.motionOffset
        XCTAssertEqual(values[base], 0.3, accuracy: 1e-6)
        XCTAssertEqual(values[base + 2], Float(log1p(0.3 / 0.02)), accuracy: 1e-6)
        XCTAssertEqual(values[base + 4], Float(Foundation.log(3.0)), accuracy: 1e-6)
        XCTAssertEqual(values[base + 7], 0.007 / 0.25, accuracy: 1e-6)
        XCTAssertEqual(values[base + 9], 1)
        XCTAssertEqual(values[base + TemporalInputContract.motionWidth + 7], 0)
        XCTAssertEqual(values[base + TemporalInputContract.motionWidth + 9], 1,
            "One event has a valid zero observed span, distinct from no events.")
        XCTAssertEqual(values[7], 0.015 / 0.25, accuracy: 1e-6)
        XCTAssertEqual(values[16], 0.3, accuracy: 1e-6)
    }

    func testButtonsModifiersAndExactQueryBoundariesRemainDistinct() {
        var timeline = CausalControlTimeline(origin: origin)
        timeline.consume(InputSample(timestampNanos: origin + 1_000_000, kind: .mouseButton,
            button: 2, modifiers: CGEventFlags.maskShift.rawValue, isDown: true))
        let before = timeline.features(at: origin + 2_000_000, frameTime: origin,
            perceptionInterval: nil, rect: rect)
        XCTAssertEqual(before[block(6)], 1)
        XCTAssertEqual(before[block(142)], 1)
        // The builder queries first, then admits events at the query boundary
        // into the following interval. Earlier snapshots are immutable.
        timeline.consume(InputSample(timestampNanos: origin + 2_000_000, kind: .mouseButton,
            button: 2, isDown: false))
        let after = timeline.features(at: origin + 3_000_000, frameTime: origin,
            perceptionInterval: nil, rect: rect)
        XCTAssertEqual(before[block(6)], 1)
        XCTAssertEqual(after[block(6)], 0)
        XCTAssertEqual(after[block(142)], 0)
    }

    func testTimingFirewallErasesEveryDerivativeOfBlockedControls() {
        var values = [Float](repeating: 1, count: TemporalInputContract.valueCount)
        var channels = ActionChannels.all
        channels.mouseMovement = false; channels.scroll = false; channels.modifiers = false
        let restrictions = ActionRestrictions(blockedKeyCodes: [13], blockedMouseButtons: [2])
        values.withUnsafeMutableBufferPointer {
            TemporalInputContract.sanitize($0, rows: 1, channels: channels, restrictions: restrictions)
        }
        for index in [6, ActionLayout.keyboard.lowerBound + 13, 143, 144, 145] {
            XCTAssertTrue(values[block(index)..<(block(index) + TemporalInputContract.controlWidth)].allSatisfy { $0 == 0 })
        }
        XCTAssertEqual(values[0], 1)
        XCTAssertEqual(values[block(142)], 1)
        XCTAssertTrue(values[TemporalInputContract.motionOffset...].allSatisfy { $0 == 0 })
    }

    func testInjectorHistoryReportsPostedEdgesAndPermissionReleases() {
        let point = CGPoint(x: rect.midX, y: rect.midY)
        let injector = InputInjector(eventSink: { _ in }, cursorWarp: { _ in }, cursorLocation: { point })
        injector.enable()
        defer { injector.disableAndReleaseAll() }
        var profile = AIProfile.fresh()
        profile.channels.mouseMovement = false
        let first = injector.temporalSnapshot(frameTime: DispatchTime.now().uptimeNanoseconds, rect: rect)
        var prediction = [Float](repeating: 0, count: ActionLayout.count)
        prediction[ActionLayout.keyboard.lowerBound + 13] = 1
        prediction[4] = 1
        injector.execute(prediction, profile: profile, allowedKeyCodes: [13], mouseMode: .absolute,
            captureRect: rect, safety: AgentSafetyPolicy(), shiftUsesKeyboardChannel: true)
        let held = injector.temporalSnapshot(frameTime: first.timestamp, rect: rect)
        XCTAssertEqual(held.features[block(ActionLayout.keyboard.lowerBound + 13)], 1)
        XCTAssertEqual(held.features[block(4)], 1)
        injector.updateOutputPermissions(RuntimeOutputPermissions(cursorMovement: false, keyboard: false))
        let released = injector.temporalSnapshot(frameTime: first.timestamp, rect: rect)
        XCTAssertEqual(released.features[block(ActionLayout.keyboard.lowerBound + 13)], 0)
        XCTAssertEqual(released.features[block(ActionLayout.keyboard.lowerBound + 13) + 9], 1)
        XCTAssertEqual(released.features[block(4)], 1, "Keyboard permission must not invent a mouse-button release.")
        XCTAssertGreaterThan(released.timestamp, first.timestamp)
    }

    func testCachedVisualEmbeddingsMatchFullForwardAcrossArchitecturesAndPrecisions() {
        MLXMemoryLifecycle.configure()
        for kind in [RecurrentKind.gru, .lstm] {
            for precision in [TrainingPrecision.float16, .bfloat16, .float32] {
                var profile = compactProfile()
                profile.training.architecture.recurrentKind = kind
                profile.training.precision = precision
                profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 2, frameSpacing: 1, downsampleFactor: 1)
                let model = AgentPolicy(profile: profile); model.train(false)
                let current = MLXArray.full([2, 8, 8, 1], values: MLXArray(Float(0.3)))
                let past = MLXArray.full([2, 2, 8, 8, 1], values: MLXArray(Float(0.2)))
                let controls = MLXArray.zeros([2, 2, ActionLayout.count])
                let timing = MLXArray.full([2, TemporalInputContract.valueCount], values: MLXArray(Float(0.01)))
                let pastTiming = MLXArray.full([2, 2, TemporalInputContract.valueCount], values: MLXArray(Float(0.02)))
                let full = model(currentImages: current, pastImages: past, pastControls: controls,
                    currentTiming: timing, pastTiming: pastTiming)
                let currentEmbedding = model.visualEmbedding(visualFeatures: model.visualActivations(images: current).last!)
                let pastEmbedding = model.pastVisualEmbedding(visualFeatures: model.pastVisualActivations(images: past.reshaped([4, 8, 8, 1])).last!).reshaped([2, 2, 8])
                let cached = model.logits(temporalFeatures: model.encodedTemporalFeatures(current: currentEmbedding,
                    past: pastEmbedding, controls: controls, currentTiming: timing, pastTiming: pastTiming))
                MLX.eval(full, cached)
                XCTAssertLessThan(abs(full - cached).max().item(Float.self), 0.002)
            }
        }
    }

    func testTimingLearnsReleaseDurationWhenImagesAndHeldStateAreIdentical() {
        MLXMemoryLifecycle.configure()
        let count = 64
        let profile = compactProfile()
        let images = MLXArray.zeros([count, 8, 8, 1])
        let labels = MLXArray((0..<count).map { Float($0 < count / 2 ? 1 : 0) })
        let base = block(ActionLayout.keyboard.lowerBound + 13)
        var values = [Float](repeating: 0, count: count * TemporalInputContract.valueCount)
        for row in 0..<count {
            var features = [Float](repeating: 0, count: TemporalInputContract.valueCount)
            features[0] = 1; features[base] = 1
            TemporalInputContract.putTime(Double(row + 1) / 320, in: &features, at: base + 1)
            for i in features.indices { values[row * features.count + i] = features[i] }
        }
        func train(ablated: Bool) -> (loss: Float, accuracy: Float) {
            MLXRandom.seed(42)
            let model = AgentPolicy(profile: profile)
            let optimizer = ResumableAdamW(learningRate: 0.003, weightDecay: 0,
                warmupSteps: 1, cycleSteps: 10_000)
            optimizer.initialize(model: model)
            var inputs = values
            if ablated { inputs.withUnsafeMutableBufferPointer { TemporalInputContract.ablate($0, rows: count) } }
            let timing = MLXArray(inputs, [count, TemporalInputContract.valueCount])
            model.train(true)
            let update = compile(inputs: [model, optimizer, MLXRandom.globalState],
                outputs: [model, optimizer, MLXRandom.globalState]) { arrays in
                let result = valueAndGrad(model: model) { policy, inputs in
                    let features = policy.currentOnlyTemporalFeatures(currentImages: inputs[0], currentTiming: inputs[1])
                    let logits = policy.logits(temporalFeatures: features)[0..., ActionLayout.keyboard.lowerBound + 13]
                    return [binaryCrossEntropy(logits: logits, targets: inputs[2], reduction: .mean)]
                }(model, arrays)
                optimizer.update(model: model, gradients: result.1, targetType: model.dtype)
                return result.0
            }
            for _ in 0..<192 {
                autoreleasepool { MLX.eval(update([images, timing, labels]), model.parameters(), optimizer.stateArrays()) }
            }
            model.train(false)
            let logits = model.logits(temporalFeatures: model.currentOnlyTemporalFeatures(currentImages: images, currentTiming: timing))[0..., ActionLayout.keyboard.lowerBound + 13]
            let loss = binaryCrossEntropy(logits: logits, targets: labels, reduction: .mean)
            let accuracy = ((sigmoid(logits) .>= 0.5) .== (labels .>= 0.5)).asType(.float32).mean()
            MLX.eval(loss, accuracy)
            return (loss.item(Float.self), accuracy.item(Float.self))
        }
        let baseline = train(ablated: true)
        let timed = train(ablated: false)
        XCTAssertLessThan(timed.loss, baseline.loss * 0.75)
        XCTAssertGreaterThan(timed.accuracy, 0.9)
        print("TEMPORAL_CONTROLLED_TASK baselineLoss=\(baseline.loss) timedLoss=\(timed.loss) timedAccuracy=\(timed.accuracy)")
    }

    private func compactProfile() -> AIProfile {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 8, height: 8, colorMode: .grayscale)
        profile.training.precision = .float32
        profile.training.architecture = ArchitectureSpec(convolutionChannels: [4], kernelSizes: [3], strides: [2], visualEmbedding: 8,
            recurrentWidth: 8, fusionWidths: [16], dropout: 0, visualPooling: .attention, attentionHeads: 2)
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 0, frameSpacing: 1, downsampleFactor: 1)
        profile.training.binaryFocalGamma = 0
        return profile
    }

    func testSharedVisionDoesNotCollapseIndependentActionTimingRows() {
        let profile = compactProfile()
        let model = AgentPolicy(profile: profile); model.train(false)
        let current = MLXArray.zeros([1, 8, 8, 1])
        let expanded = MLXArray.zeros([2, 8, 8, 1])
        let base = block(ActionLayout.keyboard.lowerBound + 13)
        var rows: [Float] = []
        for age in [0.01, 0.8] {
            var values = [Float](repeating: 0, count: TemporalInputContract.valueCount)
            values[0] = 1; values[base] = 1
            TemporalInputContract.putTime(age, in: &values, at: base + 1)
            rows += values
        }
        let timing = MLXArray(rows, [2, TemporalInputContract.valueCount])
        let shared = model.logits(temporalFeatures: model.currentOnlyTemporalFeatures(currentImages: current,
            sampleToVision: MLXArray([Int32(0), 0]), currentTiming: timing))
        let canonical = model.logits(temporalFeatures: model.currentOnlyTemporalFeatures(currentImages: expanded, currentTiming: timing))
        MLX.eval(shared, canonical)
        XCTAssertLessThan(abs(shared - canonical).max().item(Float.self), 1e-5)
        XCTAssertGreaterThan(abs(shared[0] - shared[1]).max().item(Float.self), 1e-5)
    }

    func testHistoricalTimingRemainsIndependentAfterGatheringSharedVisualRows() {
        var profile = compactProfile()
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 2, frameSpacing: 1, downsampleFactor: 1)
        let model = AgentPolicy(profile: profile); model.train(false)
        let current = MLXArray.zeros([1, 8])
        let past = MLXArray.zeros([1, 2, 8])
        let controls = MLXArray.zeros([1, 2, ActionLayout.count])
        var values = [Float](repeating: 0, count: 4 * TemporalInputContract.valueCount)
        for row in 0..<4 {
            values[row * TemporalInputContract.valueCount] = 1
            values[row * TemporalInputContract.valueCount + 10] = Float(row + 1) / 10
            values[row * TemporalInputContract.valueCount + 12] = 1
        }
        let timing = MLXArray(values, [2, 2, TemporalInputContract.valueCount])
        let currentTiming = MLXArray.zeros([2, TemporalInputContract.valueCount])
        let shared = model.logits(temporalFeatures: model.encodedTemporalFeatures(current: current,
            past: past, controls: controls, currentTiming: currentTiming, pastTiming: timing,
            sampleToVision: MLXArray([Int32(0), 0])))
        let canonical = model.logits(temporalFeatures: model.encodedTemporalFeatures(
            current: concatenated([current, current]), past: concatenated([past, past]),
            controls: concatenated([controls, controls]), currentTiming: currentTiming, pastTiming: timing))
        MLX.eval(shared, canonical)
        XCTAssertLessThan(abs(shared - canonical).max().item(Float.self), 1e-5)
        XCTAssertGreaterThan(abs(shared[0] - shared[1]).max().item(Float.self), 1e-5)
    }

    func testControlMaskAlsoMasksDurationsButKeepsObservationClocks() {
        let model = AgentPolicy(profile: compactProfile())
        model.train(false)
        let base = block(ActionLayout.keyboard.lowerBound + 13)
        var a = [Float](repeating: 0, count: TemporalInputContract.valueCount)
        a[0] = 1; a[base] = 1; a[base + 1] = 0.1
        var b = a; b[base + 1] = 0.9; b[16] = 0.7
        let maskedA = model.timingEmbedding(MLXArray(a, [1, a.count]), batch: 1, keep: MLXArray.zeros([1, 1]))
        let maskedB = model.timingEmbedding(MLXArray(b, [1, b.count]), batch: 1, keep: MLXArray.zeros([1, 1]))
        b[7] = 0.8; b[9] = 1
        let clock = model.timingEmbedding(MLXArray(b, [1, b.count]), batch: 1, keep: MLXArray.zeros([1, 1]))
        MLX.eval(maskedA, maskedB, clock)
        XCTAssertEqual(maskedA.asArray(Float.self), maskedB.asArray(Float.self))
        XCTAssertGreaterThan(abs(maskedA - clock).max().item(Float.self), 1e-5)
    }

    func testEpochStepGoalCountsFreshAndResumedEpochsExactlyOnce() {
        XCTAssertEqual(TrainingContinuationPlan.remainingSteps(completedEpoch: 0, batchOffset: 0,
            targetEpoch: 20, samplesPerEpoch: 65, batchSize: 32), 60)
        XCTAssertEqual(TrainingContinuationPlan.remainingSteps(completedEpoch: 3, batchOffset: 32,
            targetEpoch: 20, samplesPerEpoch: 65, batchSize: 32), 50)
        XCTAssertEqual(TrainingContinuationPlan.remainingSteps(completedEpoch: 20, batchOffset: 0,
            targetEpoch: 20, samplesPerEpoch: 65, batchSize: 32), 0)
    }

    private final class TestClock: @unchecked Sendable {
        private let lock = NSLock()
        private var time: UInt64 = 1_000_000_000
        func set(milliseconds: UInt64) { lock.lock(); time = 1_000_000_000 + milliseconds * 1_000_000; lock.unlock() }
        func read() -> UInt64 { lock.lock(); defer { lock.unlock() }; return time }
    }

    func testActionAndPerceptionIntervalsAreIndependentAndMatchRecordedEvents() {
        let clock = TestClock()
        let injector = InputInjector(eventSink: { _ in }, cursorWarp: { _ in }, monotonicClock: { clock.read() })
        injector.enable()
        defer { injector.disableAndReleaseAll() }
        let first = injector.temporalSnapshot(frameTime: clock.read(), rect: rect, perception: true)
        var recorded = CausalControlTimeline(origin: first.timestamp, censorInitialOnsets: false)
        var profile = compactProfile(); profile.channels.mouseMovement = false
        var prediction = [Float](repeating: 0, count: ActionLayout.count)
        let key = ActionLayout.keyboard.lowerBound + 13
        let base = block(key)
        clock.set(milliseconds: 5)
        prediction[key] = 1
        injector.execute(prediction, profile: profile, allowedKeyCodes: [13], mouseMode: .absolute,
            captureRect: rect, safety: AgentSafetyPolicy(), shiftUsesKeyboardChannel: true)
        recorded.consume(InputSample(timestampNanos: clock.read(), kind: .key, keyCode: 13, isDown: true))
        clock.set(milliseconds: 16)
        let action1 = injector.temporalSnapshot(frameTime: first.timestamp, rect: rect)
        let expected = recorded.features(at: clock.read(), frameTime: first.timestamp, perceptionInterval: nil, rect: rect)
        XCTAssertEqual(Array(action1.features[base..<(base + 16)]), Array(expected[base..<(base + 16)]))
        XCTAssertEqual(action1.features[base + 15], 11.0 / 16, accuracy: 1e-6)
        clock.set(milliseconds: 32)
        let action2 = injector.temporalSnapshot(frameTime: first.timestamp, rect: rect)
        XCTAssertEqual(action2.features[base + 15], 1)
        clock.set(milliseconds: 33)
        let perception = injector.temporalSnapshot(frameTime: first.timestamp, rect: rect, perception: true)
        XCTAssertEqual(perception.features[base + 15], 28.0 / 33, accuracy: 1e-6,
            "Action queries must not reset the perception interval's hold integral.")
        XCTAssertEqual(perception.features[base + 13], Float(Foundation.log(2.0)), accuracy: 1e-6)
        clock.set(milliseconds: 40)
        prediction[key] = 0
        injector.execute(prediction, profile: profile, allowedKeyCodes: [13], mouseMode: .absolute,
            captureRect: rect, safety: AgentSafetyPolicy(), shiftUsesKeyboardChannel: true)
        clock.set(milliseconds: 48)
        let release = injector.temporalSnapshot(frameTime: perception.timestamp, rect: rect)
        XCTAssertEqual(release.features[base], 0)
        XCTAssertEqual(release.features[base + 10], 0.035 / 0.25, accuracy: 1e-6)
        XCTAssertEqual(release.features[base + 15], 0.5, accuracy: 1e-6)
        clock.set(milliseconds: 80)
        let delayed = injector.temporalSnapshot(frameTime: perception.timestamp, rect: rect)
        XCTAssertEqual(delayed.features[1], 0.032 / 0.25, accuracy: 1e-6)
        XCTAssertEqual(delayed.features[7], 0.047 / 0.25, accuracy: 1e-6)
    }

    func testV6MigrationArchivesBrainsWithoutResettingProfileConfiguration() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("v7-migration-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()
        var profile = compactProfile()
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 7, frameSpacing: 3, downsampleFactor: 4)
        var version = ModelVersionManifest(id: UUID(), name: "Original v6", createdAt: Date(), globalStep: 20,
            trainingLoss: 0.1, preprocessing: profile.preprocessing, channels: profile.channels, training: profile.training)
        version.schemaVersion = 6
        profile.activeVersionID = version.id
        try await store.saveProfile(profile)
        try await store.saveVersionManifest(version, profileID: profile.id)
        let versionRoot = await store.versionDirectory(profileID: profile.id, versionID: version.id)
        try Data([1, 2, 3]).write(to: versionRoot.appendingPathComponent(version.weightsFile))
        try JSONEncoder().encode(6).write(to: root.appendingPathComponent("model-contract.json"))
        let count = try await store.removeObsoleteModelArtifacts(currentSchema: ModelContract.schemaVersion)
        let migratedProfiles = await store.listProfiles()
        let migrated = try XCTUnwrap(migratedProfiles.first)
        XCTAssertEqual(count, 1)
        XCTAssertNil(migrated.activeVersionID)
        XCTAssertEqual(migrated.training.architecture, profile.training.architecture)
        XCTAssertEqual(migrated.training.effectiveTemporalVision, profile.training.effectiveTemporalVision)
        let archived = await store.profileDirectory(profile.id)
            .appendingPathComponent("Archived Model Artifacts/Model Contract 6/Versions/\(version.id.uuidString)/\(version.weightsFile)")
        XCTAssertEqual(try Data(contentsOf: archived), Data([1, 2, 3]))
    }

    func testPostedModifierFlagsRetainTheirTimingAndHaveOneOutputOwner() {
        let clock = TestClock()
        let injector = InputInjector(eventSink: { _ in }, cursorWarp: { _ in }, monotonicClock: { clock.read() })
        injector.enable()
        defer { injector.disableAndReleaseAll() }
        var profile = compactProfile(); profile.channels.mouseMovement = false
        var prediction = [Float](repeating: 0, count: ActionLayout.count)
        prediction[ActionLayout.keyboard.lowerBound + 56] = 1
        clock.set(milliseconds: 5)
        injector.execute(prediction, profile: profile, allowedKeyCodes: [56], mouseMode: .absolute,
            captureRect: rect, safety: AgentSafetyPolicy(), shiftUsesKeyboardChannel: true)
        XCTAssertEqual(injector.temporalSnapshot(frameTime: clock.read(), rect: rect).features[block(142)], 0,
            "An untrained duplicate ordinary-key output must not press Shift.")
        prediction[142] = 1
        clock.set(milliseconds: 10)
        injector.execute(prediction, profile: profile, allowedKeyCodes: [56], mouseMode: .absolute,
            captureRect: rect, safety: AgentSafetyPolicy(), shiftUsesKeyboardChannel: true)
        clock.set(milliseconds: 30)
        let held = injector.temporalSnapshot(frameTime: clock.read(), rect: rect)
        XCTAssertEqual(held.features[block(142)], 1)
        XCTAssertEqual(held.features[block(142) + 1], 0.020 / 0.25, accuracy: 1e-6)
        XCTAssertEqual(held.features[block(ActionLayout.keyboard.lowerBound + 56)], 0)
        clock.set(milliseconds: 40)
        injector.updateOutputPermissions(RuntimeOutputPermissions(cursorMovement: false, keyboard: false))
        clock.set(milliseconds: 45)
        let released = injector.temporalSnapshot(frameTime: clock.read(), rect: rect)
        XCTAssertEqual(released.features[block(142)], 0)
        XCTAssertEqual(released.features[block(142) + 10], 0.030 / 0.25, accuracy: 1e-6)
    }

    func testImmutableCapabilitiesExcludeUntargetedButtonsAndBlockedKeys() {
        var weights = [Int](repeating: 0, count: ActionLayout.count)
        weights[ActionLayout.keyboard.lowerBound + 13] = 2
        weights[142] = 2
        let capabilities = ModelOutputCapabilities.trained(positiveCounts: weights,
            channels: .all, restrictions: ActionRestrictions())
        XCTAssertEqual(capabilities.allowedKeyCodes, [13, 56])
        XCTAssertTrue(capabilities.allowedMouseButtons.isEmpty)
        XCTAssertFalse(capabilities.restricting(.all, mouseMode: .relative).buttons)
        weights[4] = 2
        let blocked = ModelOutputCapabilities.trained(positiveCounts: weights, channels: .all,
            restrictions: ActionRestrictions(blockedKeyCodes: [13, 60], blockedMouseButtons: [0]))
        XCTAssertTrue(blocked.allowedKeyCodes.isEmpty)
        XCTAssertTrue(blocked.allowedMouseButtons.isEmpty)
    }

    func testStaleWatchdogCanReadPredictionAgeWithoutConsumingIt() {
        var latch = RuntimePredictionLatch()
        XCTAssertNil(latch.age(at: 10))
        latch.publish([Float](repeating: 0, count: ActionLayout.count), at: 10)
        XCTAssertEqual(latch.age(at: 10.4) ?? 0, 0.4, accuracy: 1e-6)
        XCTAssertTrue(latch.consume()?.isFresh == true)
        XCTAssertEqual(latch.age(at: 12), 2)
        XCTAssertFalse(latch.consume()?.isFresh ?? true)
        latch.reset()
        XCTAssertNil(latch.age(at: 15))
    }

    func testFlagOnlyOrdinaryKeysKeepTheirPostedDownAndUpEdges() {
        let clock = TestClock()
        let injector = InputInjector(eventSink: { _ in }, cursorWarp: { _ in }, monotonicClock: { clock.read() })
        injector.enable()
        defer { injector.disableAndReleaseAll() }
        var profile = compactProfile(); profile.channels.mouseMovement = false
        var prediction = [Float](repeating: 0, count: ActionLayout.count)
        prediction[ActionLayout.keyboard.lowerBound + 63] = 1
        clock.set(milliseconds: 5)
        injector.execute(prediction, profile: profile, allowedKeyCodes: [63], mouseMode: .absolute,
            captureRect: rect, safety: AgentSafetyPolicy(), shiftUsesKeyboardChannel: true)
        clock.set(milliseconds: 20)
        XCTAssertEqual(injector.temporalSnapshot(frameTime: clock.read(), rect: rect).features[block(ActionLayout.keyboard.lowerBound + 63)], 1)
        prediction[ActionLayout.keyboard.lowerBound + 63] = 0
        clock.set(milliseconds: 35)
        injector.execute(prediction, profile: profile, allowedKeyCodes: [63], mouseMode: .absolute,
            captureRect: rect, safety: AgentSafetyPolicy(), shiftUsesKeyboardChannel: true)
        let released = injector.temporalSnapshot(frameTime: clock.read(), rect: rect)
        XCTAssertEqual(released.features[block(ActionLayout.keyboard.lowerBound + 63)], 0)
        XCTAssertEqual(released.features[block(ActionLayout.keyboard.lowerBound + 63) + 10], 0.030 / 0.25, accuracy: 1e-6)
    }
}
