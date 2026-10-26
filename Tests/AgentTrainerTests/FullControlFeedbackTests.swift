import Foundation
import MLX
import MLXNN
import XCTest
@testable import AgentTrainer

/// Diagnostic only: closes every posted control against real video. The game
/// cannot react to generated commands, so live testing is still required.
final class FullControlFeedbackTests: XCTestCase {
    func testRealRebornFullControlFeedback() async throws {
        guard ProcessInfo.processInfo.environment["AGENTTRAINER_FULL_CONTROL_PROBE"] == "1" else {
            throw XCTSkip("Opt-in full executed-control feedback diagnosis.")
        }
        MLXMemoryLifecycle.configure()
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("work/mouse-comparison-v8")
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601
        let store = WorkspaceStore(root: root); try await store.prepare()
        var reports: [String: MouseFeedbackMetrics] = [:]
        for suffix in ["2510", "2511"] {
            let directory = root.appendingPathComponent("Profiles/00000000-0000-0000-0000-00000000\(suffix)")
            var profile = try decoder.decode(AIProfile.self, from: Data(contentsOf: directory.appendingPathComponent("profile.json")))
            let versionRoot = directory.appendingPathComponent("Versions/\(try XCTUnwrap(profile.activeVersionID).uuidString)")
            let version = try decoder.decode(ModelVersionManifest.self, from: Data(contentsOf: versionRoot.appendingPathComponent("manifest.json")))
            profile.training = version.training; profile.preprocessing = version.preprocessing; profile.channels = version.channels
            let recordings = try profile.recordingIDs.map { id in
                let directory = WorkspaceStore.defaultRoot.appendingPathComponent("Recordings/\(id.uuidString).atrrecord")
                return try RecordingItem(manifest: decoder.decode(RecordingManifest.self,
                    from: Data(contentsOf: directory.appendingPathComponent("manifest.json"))), directory: directory)
            }
            let dataset = try await DatasetCacheBuilder(workspace: store).cache(for: profile, recordings: recordings) { _, _ in }
            let split = TrainingEngine().splitIndices(dataset: dataset, fraction: profile.training.validationSplit,
                seed: profile.training.seed, channels: profile.channels, restrictions: profile.effectiveRestrictions)
            let rows = split.validation
            for point in ["best", "final"] {
                let model = AgentPolicy(profile: profile, contextSchema: version.effectiveContextSchema, legacyControlFeedback: version.schemaVersion == 8)
                let weights = point == "best" ? versionRoot.appendingPathComponent(version.weightsFile)
                    : directory.appendingPathComponent("Checkpoint/weights.safetensors")
                try model.loadWeights(from: weights); model.train(false)
                for diagnostic in ["production", "activity-bypass", "control-masked"] {
                    let label = "\(suffix)-\(point)-\(diagnostic)"
                    let metrics = try MouseFeedbackEvaluation.evaluate(model, dataset: dataset, rows: rows,
                        profile: profile, recordings: recordings, feedbackMode: .allControls,
                        capabilities: version.outputCapabilities,
                        bypassMovementActivityForDiagnosis: diagnostic == "activity-bypass",
                        maskControlHistoryForDiagnosis: diagnostic == "control-masked")
                    reports[label] = metrics
                    print("FULL_CONTROL_PROBE \(label) mouseMAE=\(metrics.feedbackRawMAE) mouseRecall=\(metrics.movementRecall)")
                    fflush(stdout)
                }
                Memory.clearCache()
            }
        }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        try encoder.encode(reports).write(to: root.appendingPathComponent("full-control-feedback.json"), options: .atomic)
    }
}
