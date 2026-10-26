import Foundation
import MLX
import MLXNN
import XCTest
@testable import AgentTrainer

final class SingleRecordingTests: XCTestCase {
    func testRealRecordingMemorizationAndExecutedFeedback() async throws {
        guard ProcessInfo.processInfo.environment["AGENTTRAINER_SINGLE_RECORDING"] == "1" else {
            throw XCTSkip("Opt-in single real-recording memorization diagnosis.")
        }
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601
        let source = WorkspaceStore.defaultRoot
        let store = WorkspaceStore.shared
        let workspace = await store.root
        guard workspace != source else { throw AgentTrainerError.storage("Use an isolated memorization workspace.") }
        try await store.prepare()
        let referenceRoot = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("work/mouse-comparison-v8")
        let reference = try decoder.decode(AIProfile.self, from: Data(contentsOf: referenceRoot.appendingPathComponent("Reference-v7/profile.json")))
        let folders = try decoder.decode([RecordingFolder].self, from: Data(contentsOf: source.appendingPathComponent("recording-folders.json")))
        let folderID = try XCTUnwrap(folders.first { $0.name.lowercased() == "reborn camera check" }?.id)
        let recordings = try FileManager.default.contentsOfDirectory(at: source.appendingPathComponent("Recordings"), includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "atrrecord" }.compactMap { directory -> RecordingItem? in
                let manifest = try decoder.decode(RecordingManifest.self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
                return manifest.folderID == folderID ? RecordingItem(manifest: manifest, directory: directory) : nil
            }.sorted { $0.id.uuidString < $1.id.uuidString }
        XCTAssertEqual(recordings.count, 6)
        var evaluationProfile = reference
        evaluationProfile.training.movementHistoryMode = .bounded
        let fresh = try await DatasetCacheBuilder.shared.cache(for: evaluationProfile, recordings: recordings) { _, _ in }
        var reports: [String: MouseFeedbackMetrics] = [:]
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        // Evaluate every fresh recording before using any of it for fitting.
        for suffix in ["2510", "2511"] {
            let directory = referenceRoot.appendingPathComponent("Profiles/00000000-0000-0000-0000-00000000\(suffix)")
            let profile = try decoder.decode(AIProfile.self, from: Data(contentsOf: directory.appendingPathComponent("profile.json")))
            let versionRoot = directory.appendingPathComponent("Versions/\(try XCTUnwrap(profile.activeVersionID).uuidString)")
            let version = try decoder.decode(ModelVersionManifest.self, from: Data(contentsOf: versionRoot.appendingPathComponent("manifest.json")))
            let model = AgentPolicy(profile: profile, contextSchema: version.effectiveContextSchema, legacyControlFeedback: version.schemaVersion == 8)
            try model.loadWeights(from: versionRoot.appendingPathComponent(version.weightsFile)); model.train(false)
            reports["unseen-before-training-\(suffix)"] = try MouseFeedbackEvaluation.evaluate(model, dataset: fresh,
                rows: Array(0..<fresh.count), profile: profile, recordings: recordings,
                feedbackMode: .allControls, capabilities: version.outputCapabilities)
            try encoder.encode(reports).write(to: workspace.appendingPathComponent("single-recording-feedback.json"), options: .atomic)
        }
        let recording: RecordingItem
        if let requested = ProcessInfo.processInfo.environment["AGENTTRAINER_MEMORY_RECORDING_ID"] {
            guard let id = UUID(uuidString: requested), let selected = recordings.first(where: { $0.id == id }) else {
                throw AgentTrainerError.storage("The explicitly selected real memorization recording is unavailable.")
            }
            recording = selected
        } else {
            recording = try XCTUnwrap(recordings.max { $0.manifest.duration < $1.manifest.duration })
        }
        let arm = ProcessInfo.processInfo.environment["AGENTTRAINER_MEMORY_ARM"] ?? "baseline"
        let fit = ["fit", "phase", "dense", "grounded", "grounded-diverse", "grounded-visual"].contains(arm)
        let idSuffix = arm == "grounded-visual" ? "2527" : arm == "grounded-diverse" ? "2526" : arm == "grounded" ? "2525" : arm == "dense" ? "2524" : arm == "phase" ? "2523" : fit ? "2522" : arm == "master" ? "2521" : "2520"
        var profile = AIProfile.fresh(name: fit ? "Reborn • deterministic fitting diagnostic" : arm == "master" ? "Reborn • Float32 accumulation diagnostic" : "Reborn • single recording diagnostic")
        profile.id = UUID(uuidString: "00000000-0000-0000-0000-00000000\(idSuffix)")!
        profile.preprocessing = reference.preprocessing; profile.channels = reference.channels
        profile.restrictions = reference.restrictions; profile.training = reference.training
        profile.training.movementHistoryMode = arm == "grounded-visual" ? .visualOnly : .bounded
        profile.training.epochs = 20; profile.training.maximumSteps = 0
        profile.training.perceptionFPS = 60; profile.training.actionFPS = 60
        profile.training.binaryFocalGamma = 0; profile.training.validationSplit = 0
        // This deliberately tests fitting capacity, not regularized generalization.
        profile.training.architecture.dropout = 0; profile.training.weightDecay = 0
        if fit {
            profile.training.precision = .float32
            profile.training.learningRate = 0.001
            profile.training.cosineCycleEpochs = 20
            if ["dense", "grounded", "grounded-diverse", "grounded-visual"].contains(arm) { profile.training.batchSize = 4 }
        }
        var augmentation = profile.training.effectiveVisualAugmentation; augmentation.enabled = false
        profile.training.visualAugmentation = augmentation
        var futureAction = profile.training.effectiveFutureActionPrediction; futureAction.enabled = false
        profile.training.futureActionPrediction = futureAction
        var futureState = profile.training.effectiveFutureStatePrediction; futureState.enabled = false
        profile.training.futureStatePrediction = futureState
        profile.recordingIDs = [recording.id]
        if let existing = await store.listProfiles().first(where: { $0.id == profile.id }) { profile = existing }
        XCTAssertEqual(profile.recordingIDs, [recording.id])
        try await store.saveProfile(profile)
        let checkpoint = await store.checkpointDirectory(profileID: profile.id)
        let stateURL = checkpoint.appendingPathComponent("state.json")
        let state = (try? Data(contentsOf: stateURL)).flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        let completedEpochs = state?["epoch"] as? Int ?? 0
        guard completedEpochs <= 20 else { throw AgentTrainerError.model("Memorization run exceeds 20 epochs.") }
        if completedEpochs < 20 {
            let engine = TrainingEngine(diagnosticDeterministicFit: fit, diagnosticDiverseSampling: ["grounded-diverse", "grounded-visual"].contains(arm))
            let progress = SingleRecordingProgress()
            let completion: TrainingCompletion = try await withCheckedThrowingContinuation { continuation in
                engine.start(profile: profile, recordings: [recording],
                    runSettings: TrainingRunSettings(maximumSteps: 0, autosaveSteps: 10_000),
                    metrics: { progress.consume($0, $1) }, completion: { continuation.resume(with: $0) })
            }
            XCTAssertTrue(completion.completed); profile = completion.profile
        }
        let stateObject = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(contentsOf: stateURL)) as? [String: Any])
        XCTAssertEqual(stateObject["epoch"] as? Int, 20); XCTAssertEqual(stateObject["batchOffset"] as? Int, 0)
        let selected = await store.version(profileID: profile.id, versionID: try XCTUnwrap(profile.activeVersionID))
        let version = try XCTUnwrap(selected)
        let directory = await store.versionDirectory(profileID: profile.id, versionID: version.id)
        let model = AgentPolicy(profile: profile, contextSchema: version.effectiveContextSchema, legacyControlFeedback: version.schemaVersion == 8)
        try model.loadWeights(from: directory.appendingPathComponent(version.weightsFile)); model.train(false)
        let dataset = try await DatasetCacheBuilder.shared.cache(for: profile, recordings: [recording]) { _, _ in }
        let teacher = teacherMetrics(model, dataset: dataset, profile: profile)
        try encoder.encode(teacher).write(to: workspace.appendingPathComponent("memorized-teacher.json"), options: .atomic)
        for diagnostic in ["production", "activity-bypass", "control-masked", "recorded-history"] {
            reports["memorized-\(diagnostic)"] = try MouseFeedbackEvaluation.evaluate(model, dataset: dataset,
                rows: Array(0..<dataset.count), profile: profile, recordings: [recording], feedbackMode: diagnostic == "recorded-history" ? .recordedControls : .allControls,
                capabilities: version.outputCapabilities,
                bypassMovementActivityForDiagnosis: diagnostic == "activity-bypass",
                maskControlHistoryForDiagnosis: diagnostic == "control-masked")
        }
        try encoder.encode(reports).write(to: workspace.appendingPathComponent("single-recording-feedback.json"), options: .atomic)
        if ["grounded-diverse", "grounded-visual"].contains(arm) {
            let posted = try XCTUnwrap(reports["memorized-production"])
            XCTAssertGreaterThanOrEqual(posted.movementRecall, 0.8)
            XCTAssertLessThanOrEqual(posted.idlePostRate, 0.1)
            for code: UInt16 in [1, 2, 13, 49] {
                let score = try XCTUnwrap(posted.keyScores[code])
                XCTAssertLessThanOrEqual(score.maximumPostedHoldTicks, score.maximumTargetHoldTicks + 3)
                XCTAssertLessThanOrEqual(score.falseNegatives, 5)
                XCTAssertLessThanOrEqual(score.falsePositives, 15)
            }
        }
        print("MEMORIZATION_RESULT recording=\(recording.id) rows=\(dataset.count) teacher=\(String(data: try encoder.encode(teacher), encoding: .utf8)!)")
    }

    private func teacherMetrics(_ model: AgentPolicy, dataset: CachedDataset, profile: AIProfile) -> ValidationReport {
        let rows = Array(0..<dataset.count)
        let counts = dataset.binaryPositiveCounts(at: rows)
        let indices = Set(ActionLayout.learnableBinaryIndices(channels: profile.channels, restrictions: profile.effectiveRestrictions).filter { counts[$0] > 0 })
        var accumulator = ValidationAccumulator(profile: profile, activeBinaryIndices: indices)
        let frames = profile.training.effectiveTemporalVision.pastFrameCount
        let pastSpec = profile.training.effectiveTemporalVision.pastFrameSpec(from: profile.preprocessing)
        for start in stride(from: 0, to: rows.count, by: 32) {
            autoreleasepool {
                let batch = Array(rows[start..<min(rows.count, start + 32)]); let b = batch.count
                let timing = dataset.timingBatch(at: batch)
                let current = VisionPreprocessor.mlxTensor(MLXArray(dataset.packedObservations(at: batch),
                    [b, profile.preprocessing.sampleByteCount], dtype: .uint8), spec: profile.preprocessing)
                let past = VisionPreprocessor.mlxPastFrameTensor(MLXArray(dataset.pastPackedObservations(at: batch),
                    [b, frames, pastSpec.sampleByteCount], dtype: .uint8), spec: pastSpec)
                let prediction = model.predictions(currentImages: current, pastImages: past,
                    pastControls: MLXArray(dataset.pastControlBatch(at: batch), [b, frames, ActionLayout.count], type: Float.self),
                    currentTiming: MLXArray(timing.current, [b, TemporalInputContract.valueCount], type: Float.self),
                    pastTiming: MLXArray(timing.past, [b, frames, TemporalInputContract.valueCount], type: Float.self))
                MLX.eval(prediction)
                accumulator.consume(predictions: prediction.asArray(Float.self),
                    targets: batch.flatMap { dataset.action(at: $0) }, rowCount: b)
            }
        }
        return accumulator.finalize(sampleCount: rows.count)
    }
}

private final class SingleRecordingProgress: @unchecked Sendable {
    private let lock = NSLock()
    private var last = Date.distantPast
    func consume(_ metrics: TrainingMetrics, _ status: String) {
        lock.withLock {
            guard Date().timeIntervalSince(last) >= 5 || metrics.globalStep == 1 else { return }
            last = Date()
            print("MEMORIZATION_TRAIN epoch=\(metrics.epoch) step=\(metrics.globalStep) loss=\(metrics.trainingLoss) \(status)")
            fflush(stdout)
        }
    }
}
