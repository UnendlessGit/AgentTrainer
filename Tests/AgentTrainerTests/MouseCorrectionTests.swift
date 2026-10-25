import Foundation
import CoreGraphics
import MLX
import MLXNN
import XCTest
@testable import AgentTrainer

/// Uses only real Reborn demonstrations. Reports in an isolated workspace;
/// no synthetic trajectories are used for training and no global HID is posted.
final class MouseCorrectionTests: XCTestCase {
    func testRealRebornTrainingAndFeedback() async throws {
        guard ProcessInfo.processInfo.environment["AGENTTRAINER_MOUSE_CORRECTION"] == "1" else {
            throw XCTSkip("Run ./benchmark-mouse.sh for the isolated real-recording experiment.")
        }
        executionTimeAllowance = 2 * 60 * 60
        let source = WorkspaceStore.defaultRoot
        let store = WorkspaceStore.shared
        let workspace = await store.root
        guard workspace != source else { throw AgentTrainerError.storage("Mouse evaluation requires a separate workspace.") }
        try await store.prepare()
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601
        let referenceRoot = workspace.appendingPathComponent("Reference-v7")
        let savedProfile: AIProfile
        let reference: ModelVersionManifest
        if FileManager.default.fileExists(atPath: referenceRoot.appendingPathComponent("profile.json").path) {
            savedProfile = try decoder.decode(AIProfile.self, from: Data(contentsOf: referenceRoot.appendingPathComponent("profile.json")))
            reference = try decoder.decode(ModelVersionManifest.self, from: Data(contentsOf: referenceRoot.appendingPathComponent("manifest.json")))
        } else {
            let directories = try FileManager.default.contentsOfDirectory(at: source.appendingPathComponent("Profiles"), includingPropertiesForKeys: nil)
            savedProfile = try XCTUnwrap(directories.compactMap { url -> AIProfile? in
                guard let data = try? Data(contentsOf: url.appendingPathComponent("profile.json")) else { return nil }
                return try decoder.decode(AIProfile.self, from: data)
            }.first { $0.name == "Reborn" })
            let versionID = try XCTUnwrap(savedProfile.activeVersionID)
            let versionRoot = source.appendingPathComponent("Profiles/\(savedProfile.id.uuidString)/Versions/\(versionID.uuidString)")
            reference = try decoder.decode(ModelVersionManifest.self, from: Data(contentsOf: versionRoot.appendingPathComponent("manifest.json")))
            guard reference.schemaVersion == 7 else { throw AgentTrainerError.model("Freeze a v7 reference before the migration.") }
            try FileManager.default.createDirectory(at: referenceRoot, withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: versionRoot.appendingPathComponent(reference.weightsFile), to: referenceRoot.appendingPathComponent(reference.weightsFile))
            try FileManager.default.copyItem(at: versionRoot.appendingPathComponent("manifest.json"), to: referenceRoot.appendingPathComponent("manifest.json"))
            try Data(contentsOf: source.appendingPathComponent("Profiles/\(savedProfile.id.uuidString)/profile.json"))
                .write(to: referenceRoot.appendingPathComponent("profile.json"), options: .atomic)
        }
        let folders = try decoder.decode([RecordingFolder].self, from: Data(contentsOf: source.appendingPathComponent("recording-folders.json")))
        let folderID = try XCTUnwrap(folders.first { $0.name == "Reborn" }?.id)
        let recordings = try FileManager.default.contentsOfDirectory(at: source.appendingPathComponent("Recordings"), includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "atrrecord" }.compactMap { url -> RecordingItem? in
                let m = try decoder.decode(RecordingManifest.self, from: Data(contentsOf: url.appendingPathComponent("manifest.json")))
                return m.folderID == folderID ? RecordingItem(manifest: m, directory: url) : nil
            }.sorted { $0.id.uuidString < $1.id.uuidString }
        XCTAssertEqual(recordings.count, 39)
        let mode: MovementHistoryMode = ProcessInfo.processInfo.environment["AGENTTRAINER_MOUSE_HISTORY_MODE"] == "visual" ? .visualOnly : .bounded
        var profile = AIProfile.fresh(name: mode == .bounded ? "Reborn • bounded mouse history" : "Reborn • visual movement")
        profile.id = UUID(uuidString: mode == .bounded ? "00000000-0000-0000-0000-000000002511" : "00000000-0000-0000-0000-000000002510")!
        profile.preprocessing = reference.preprocessing
        profile.channels = reference.channels
        profile.restrictions = savedProfile.restrictions
        profile.training = reference.training
        profile.training.movementHistoryMode = mode
        profile.training.epochs = 20; profile.training.maximumSteps = 0
        profile.training.perceptionFPS = 60; profile.training.actionFPS = 60
        profile.training.binaryFocalGamma = 0
        profile.recordingIDs = recordings.map(\.id)
        if let existing = await store.listProfiles().first(where: { $0.id == profile.id }) { profile = existing }
        try await store.saveProfile(profile)
        let checkpoint = await store.checkpointDirectory(profileID: profile.id)
        let stateURL = checkpoint.appendingPathComponent("state.json")
        let state = (try? Data(contentsOf: stateURL)).flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        let epoch = state?["epoch"] as? Int ?? 0
        guard epoch <= 20 else { throw AgentTrainerError.model("Mouse experiment exceeds the 20-epoch budget.") }
        if epoch < 20 {
            let engine = TrainingEngine()
            let progress = MouseTrainingProgress()
            let completion: TrainingCompletion = try await withCheckedThrowingContinuation { continuation in
                engine.start(profile: profile, recordings: recordings,
                    runSettings: TrainingRunSettings(maximumSteps: 0, autosaveSteps: 10_000),
                    metrics: { metrics, status in progress.consume(metrics, status) },
                    completion: { continuation.resume(with: $0) })
            }
            XCTAssertTrue(completion.completed); profile = completion.profile
        }
        let finalState = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(contentsOf: stateURL)) as? [String: Any])
        XCTAssertEqual(finalState["epoch"] as? Int, 20); XCTAssertEqual(finalState["batchOffset"] as? Int, 0)
        let selectedVersion = await store.version(profileID: profile.id, versionID: try XCTUnwrap(profile.activeVersionID))
        let version = try XCTUnwrap(selectedVersion)
        let weightsRoot = await store.versionDirectory(profileID: profile.id, versionID: version.id)
        let model = AgentPolicy(profile: profile, contextSchema: version.effectiveContextSchema)
        try model.loadWeights(from: weightsRoot.appendingPathComponent(version.weightsFile)); model.train(false)
        let legacy = AgentPolicy(profile: profile, contextSchema: version.effectiveContextSchema, legacyMovementFeedback: true)
        try legacy.update(parameters: ModuleParameters.unflattened(MLX.loadArrays(url: referenceRoot.appendingPathComponent(reference.weightsFile))), verify: .noUnusedKeys)
        legacy.train(false)
        let dataset = try await DatasetCacheBuilder.shared.cache(for: profile, recordings: recordings) { _, _ in }
        let split = TrainingEngine().splitIndices(dataset: dataset, fraction: profile.training.validationSplit,
            seed: profile.training.seed, channels: profile.channels, restrictions: profile.effectiveRestrictions)
        // Each recording is tested from its own beginning. Target video responds
        // only to the demonstrator; this is feedback robustness, not gameplay.
        let selected = split.validation
        var reports: [String: MouseFeedbackMetrics] = [:]
        for cadence in [1, 2, 4] {
            let corrected = try MouseFeedbackEvaluation.evaluate(model, dataset: dataset, rows: selected,
                profile: profile, recordings: recordings, perceptionStride: cadence)
            let original = try MouseFeedbackEvaluation.evaluate(legacy, dataset: dataset, rows: selected,
                profile: profile, recordings: recordings, perceptionStride: cadence)
            reports["corrected-stride-\(cadence)"] = corrected
            reports["v7-reference-stride-\(cadence)"] = original
        }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(reports)
        try data.write(to: workspace.appendingPathComponent(mode == .bounded ? "mouse-results-bounded.json" : "mouse-results-visual.json"), options: .atomic)
        print("MOUSE_CORRECTION_RESULT mode=\(mode.rawValue) \(String(data: data, encoding: .utf8)!)")
        let corrected = reports["corrected-stride-1"]!, original = reports["v7-reference-stride-1"]!
        XCTAssertLessThan(corrected.feedbackRawMAE, original.feedbackRawMAE * 0.8,
            "Corrected posted feedback must improve mouse error beyond the frozen reference.")
        XCTAssertLessThan(corrected.idlePostRate, original.idlePostRate,
            "The correction must reduce actual post-rounding movement during demonstrated stops.")
        XCTAssertGreaterThanOrEqual(corrected.movementRecall, 0.7,
            "Suppressing useful movement is not a valid fix for drift.")

    }

}

private final class MouseTrainingProgress: @unchecked Sendable {
    private let lock = NSLock()
    private var last = Date.distantPast
    func consume(_ metrics: TrainingMetrics, _ status: String) {
        lock.lock(); defer { lock.unlock() }
        guard Date().timeIntervalSince(last) >= 15 || metrics.globalStep == 1 else { return }
        last = Date()
        print("MOUSE_TRAIN epoch=\(metrics.epoch) step=\(metrics.globalStep) loss=\(metrics.trainingLoss) rows/s=\(metrics.samplesPerSecond) \(status)")
        fflush(stdout)
    }
}
