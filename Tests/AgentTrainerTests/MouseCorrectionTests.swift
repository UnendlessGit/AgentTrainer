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
        let settingsURL = workspace.appendingPathComponent("settings-reference.json")
        let savedProfile: AIProfile
        if FileManager.default.fileExists(atPath: settingsURL.path) {
            savedProfile = try decoder.decode(AIProfile.self, from: Data(contentsOf: settingsURL))
        } else {
            let frozen = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("work/mouse-comparison-v8/Reference-v7/profile.json")
            if FileManager.default.fileExists(atPath: frozen.path) {
                savedProfile = try decoder.decode(AIProfile.self, from: Data(contentsOf: frozen))
            } else {
                let directories = try FileManager.default.contentsOfDirectory(at: source.appendingPathComponent("Profiles"), includingPropertiesForKeys: nil)
                savedProfile = try XCTUnwrap(directories.compactMap { url -> AIProfile? in
                    guard let data = try? Data(contentsOf: url.appendingPathComponent("profile.json")) else { return nil }
                    return try decoder.decode(AIProfile.self, from: data)
                }.first { $0.name == "Reborn" })
            }
            let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(savedProfile).write(to: settingsURL, options: .atomic)
        }
        let folders = try decoder.decode([RecordingFolder].self, from: Data(contentsOf: source.appendingPathComponent("recording-folders.json")))
        let folderIDs = Set(folders.filter { ["reborn", "reborn camera check"].contains($0.name.lowercased()) }.map(\.id))
        let available = try FileManager.default.contentsOfDirectory(at: source.appendingPathComponent("Recordings"), includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "atrrecord" }.compactMap { url -> RecordingItem? in
                let m = try decoder.decode(RecordingManifest.self, from: Data(contentsOf: url.appendingPathComponent("manifest.json")))
                return m.folderID.map(folderIDs.contains) == true ? RecordingItem(manifest: m, directory: url) : nil
            }.sorted { $0.id.uuidString < $1.id.uuidString }
        let selectionURL = workspace.appendingPathComponent("recording-selection.json")
        let selectedIDs: [UUID]
        if FileManager.default.fileExists(atPath: selectionURL.path) {
            selectedIDs = try decoder.decode([UUID].self, from: Data(contentsOf: selectionURL))
        } else {
            selectedIDs = available.map(\.id)
            try JSONEncoder().encode(selectedIDs).write(to: selectionURL, options: .atomic)
        }
        let byID = Dictionary(uniqueKeysWithValues: available.map { ($0.id, $0) })
        let recordings = try selectedIDs.map { try XCTUnwrap(byID[$0], "An experiment recording was removed.") }
        XCTAssertEqual(recordings.count, 46)
        let mode: MovementHistoryMode = ProcessInfo.processInfo.environment["AGENTTRAINER_MOUSE_HISTORY_MODE"] == "visual" ? .visualOnly : .bounded
        var profile = AIProfile.fresh(name: mode == .bounded ? "Reborn • grounded mouse history" : "Reborn • grounded visual movement")
        profile.id = UUID(uuidString: mode == .bounded ? "00000000-0000-0000-0000-000000002531" : "00000000-0000-0000-0000-000000002530")!
        profile.preprocessing = savedProfile.preprocessing
        profile.channels = savedProfile.channels
        profile.restrictions = savedProfile.restrictions
        profile.training = savedProfile.training
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
        let dataset = try await DatasetCacheBuilder.shared.cache(for: profile, recordings: recordings) { _, _ in }
        let split = TrainingEngine().splitIndices(dataset: dataset, fraction: profile.training.validationSplit,
            seed: profile.training.seed, channels: profile.channels, restrictions: profile.effectiveRestrictions)
        // Each recording is tested from its own beginning. Target video responds
        // only to the demonstrator; this is feedback robustness, not gameplay.
        let selected = split.validation
        var reports: [String: MouseFeedbackMetrics] = [:]
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let outputURL = workspace.appendingPathComponent(mode == .bounded ? "mouse-results-bounded.json" : "mouse-results-visual.json")
        for point in ["best", "final"] {
            if point == "final" { try model.loadWeights(from: checkpoint.appendingPathComponent("weights.safetensors")) }
            for cadence in [1, 2, 4] {
                let corrected = try MouseFeedbackEvaluation.evaluate(model, dataset: dataset, rows: selected,
                    profile: profile, recordings: recordings, perceptionStride: cadence,
                    feedbackMode: .allControls, capabilities: version.outputCapabilities)
                reports["\(point)-stride-\(cadence)"] = corrected
                try encoder.encode(reports).write(to: outputURL, options: .atomic)
                print("MOUSE_CORRECTION_RESULT mode=\(mode.rawValue) point=\(point) cadence=\(cadence) MAE=\(corrected.feedbackRawMAE) recall=\(corrected.movementRecall) idle=\(corrected.idlePostRate)")
                fflush(stdout)
            }
        }
        let corrected = reports["best-stride-1"]!
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
