import CoreGraphics
import Foundation
import MLX
import MLXNN
import XCTest
@testable import AgentTrainer

/// Algebra, decoding, and event-posting tests; these fixtures are never used
/// as training data. Learning experiments use the real recordings only.
final class MovementControlTests: XCTestCase {
    private func profile(_ mode: MovementHistoryMode, past: Int = 2) -> AIProfile {
        var p = AIProfile.fresh()
        p.preprocessing = PreprocessingSpec(width: 8, height: 8, colorMode: .grayscale)
        p.training.precision = .float32
        p.training.architecture = ArchitectureSpec(convolutionChannels: [4], kernelSizes: [3], strides: [2],
            visualEmbedding: 8, recurrentWidth: 8, fusionWidths: [16], dropout: 0, visualPooling: .attention, attentionHeads: 2)
        p.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: past, frameSpacing: 1, downsampleFactor: 1)
        p.training.movementHistoryMode = mode
        p.training.actionFPS = 60; p.training.perceptionFPS = 60
        return p
    }

    private func timing(delta: Float, interval: Double) -> [Float] {
        var values = [Float](repeating: 0, count: TemporalInputContract.valueCount)
        TemporalInputContract.putClock(in: &values, interval: interval, perceptionInterval: interval, frameAge: 0)
        let offset = TemporalInputContract.motionOffset
        values[offset] = min(1, max(-1, delta))
        values[offset + 1] = -values[offset]
        let rate = Float(log1p(abs(Double(delta)) / interval)) * (delta < 0 ? -1 : 1)
        values[offset + 2] = rate; values[offset + 3] = -rate
        values[offset + 4] = delta == 0 ? 0 : Float(log(2.0))
        values[offset + 13] = 1
        return values
    }

    private func logits(_ model: AgentPolicy, profile: AIProfile, delta: Float, interval: Double) -> MLXArray {
        let count = profile.training.effectiveTemporalVision.pastFrameCount
        let values = timing(delta: delta, interval: interval)
        let current = MLXArray.zeros([1, 8])
        var controls = [Float](repeating: 0, count: count * ActionLayout.count)
        for frame in 0..<count {
            controls[frame * ActionLayout.count] = delta
            controls[frame * ActionLayout.count + 2] = delta
            controls[frame * ActionLayout.count + 3] = -delta
        }
        let features = model.encodedTemporalFeatures(current: current,
            past: count > 0 ? MLXArray.zeros([1, count, 8]) : nil,
            controls: count > 0 ? MLXArray(controls, [1, count, ActionLayout.count]) : nil,
            currentTiming: MLXArray(values, [1, values.count]),
            pastTiming: count > 0 ? MLXArray(Array(repeating: values, count: count).flatMap { $0 }, [1, count, values.count]) : nil)
        return model.logits(temporalFeatures: features)
    }

    func testVisionOnlyMovementCannotReadItsOwnCurrentOrPastDeltas() {
        for kind in [RecurrentKind.gru, .lstm] {
            for count in [0, 2] {
                var p = profile(.visualOnly, past: count); p.training.architecture.recurrentKind = kind
                let model = AgentPolicy(profile: p); model.train(false)
                let a = logits(model, profile: p, delta: -0.8, interval: 1.0 / 60)
                let b = logits(model, profile: p, delta: 0.8, interval: 1.0 / 60)
                MLX.eval(a, b)
                for index in [0, 1, 2, 3, 12, 13, 146, 147] {
                    XCTAssertEqual(a[0, index].item(Float.self), b[0, index].item(Float.self), accuracy: 1e-6)
                }
                XCTAssertGreaterThan(abs(a[0, 14..<146] - b[0, 14..<146]).max().item(Float.self), 1e-6,
                    "Full movement evidence must remain available to the discrete decision stream.")
            }
        }
    }

    func testRetainedHistoryHasBoundedGainAcrossRatesAndSensitivity() {
        for count in [0, 2] {
            for sensitivity in [0.25, 1.0, 4.0, 100.0] {
                let p = profile(.bounded, past: count)
                let model = AgentPolicy(profile: p, movementSensitivity: sensitivity); model.train(false)
                for interval in [1.0 / 240, 1.0 / 60, 1.0 / 15] {
                    let a = logits(model, profile: p, delta: 0, interval: interval)
                    let b = logits(model, profile: p, delta: 0.5, interval: interval)
                    MLX.eval(a, b)
                    let difference = abs(tanh(a[0, 2..<4]) - tanh(b[0, 2..<4])).max().item(Float.self) * Float(sensitivity)
                    let limit = AgentPolicy.maximumHistoryGain * 0.5 * Float(min(1, (1.0 / 60) / interval))
                    XCTAssertLessThanOrEqual(difference, limit + 0.0001)
                    XCTAssertGreaterThan(difference, 1e-7, "Bounded mode must actually retain movement history.")
                    for index in [0, 1, 12, 13, 146, 147] {
                        XCTAssertEqual(a[0, index].item(Float.self), b[0, index].item(Float.self), accuracy: 1e-6,
                            "Movement history cannot influence stopping confidence or the independent motor base.")
                    }
                }
            }
        }
    }

    func testConfidenceDecodingKeeps146PhysicalActionsAndCanStopEachAxis() {
        let p = profile(.bounded)
        let model = AgentPolicy(profile: p)
        var values = [Float](repeating: 0, count: PolicyLogitsLayout.count)
        values[2] = 1; values[3] = -1
        values[146] = -1; values[147] = 1
        let prediction = model.activatedPredictions(logits: MLXArray(values, [1, values.count]))
        MLX.eval(prediction)
        XCTAssertEqual(prediction.shape, [1, 146])
        XCTAssertEqual(prediction[0, 2].item(Float.self), 0)
        XCTAssertLessThan(prediction[0, 3].item(Float.self), -0.5)
    }

    func testUncertainActivityDoesNotCreateAnOnOffCameraDiscontinuity() {
        let model = AgentPolicy(profile: profile(.bounded))
        var a = [Float](repeating: 0, count: PolicyLogitsLayout.count)
        a[2] = 0.2; a[146] = -0.001
        var b = a; b[146] = 0.001
        let before = model.activatedPredictions(logits: MLXArray(a, [1, a.count]))
        let after = model.activatedPredictions(logits: MLXArray(b, [1, b.count]))
        MLX.eval(before, after)
        XCTAssertGreaterThan(before[0, 2].item(Float.self), 0)
        XCTAssertLessThan(abs(after[0, 2] - before[0, 2]).item(Float.self), 0.001)
        a[146] = -2
        let stopped = model.activatedPredictions(logits: MLXArray(a, [1, a.count]))
        XCTAssertEqual(stopped[0, 2].item(Float.self), 0)
    }

    func testIdleValidationCountsTheSameHalfPixelBoundaryAsExecution() {
        var p = profile(.bounded)
        p.channels.buttons = false; p.channels.keyboard = false; p.channels.modifiers = false; p.channels.scroll = false
        let target = [Float](repeating: 0, count: ActionLayout.count)
        var prediction = target
        prediction[2] = 0.5 / GameCameraContract.deltaScale
        var accumulator = ValidationAccumulator(profile: p, activeBinaryIndices: [])
        accumulator.consume(predictions: prediction, targets: target, rowCount: 1)
        XCTAssertEqual(accumulator.finalize(sampleCount: 1).idleContinuousFalseActionRate, 0.5)
        prediction[2] = 0.49 / GameCameraContract.deltaScale
        var below = ValidationAccumulator(profile: p, activeBinaryIndices: [])
        below.consume(predictions: prediction, targets: target, rowCount: 1)
        XCTAssertEqual(below.finalize(sampleCount: 1).idleContinuousFalseActionRate, 0)
    }
    func testV7MigrationKeepsSchema15PixelsAndPreservesOldBrain() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("movement-migration-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root); try await store.prepare()
        var p = profile(.bounded)
        let versionID = UUID(); p.activeVersionID = versionID
        let version = ModelVersionManifest(id: versionID, name: "v7", createdAt: Date(), globalStep: 1,
            trainingLoss: 0.1, validationLoss: nil, preprocessing: p.preprocessing, channels: p.channels, training: p.training,
            weightsFile: "weights.safetensors")
        var old = version; old.schemaVersion = 7
        try await store.saveProfile(p); try await store.saveVersionManifest(old, profileID: p.id)
        let directory = await store.versionDirectory(profileID: p.id, versionID: versionID)
        try Data([1, 2, 3]).write(to: directory.appendingPathComponent("weights.safetensors"))
        let cache = await store.cacheDirectory()
        let sentinel = cache.appendingPathComponent("kept.atrcache")
        try FileManager.default.createDirectory(at: sentinel, withIntermediateDirectories: true)
        try Data([9]).write(to: sentinel.appendingPathComponent("current-observations.bin"))
        try JSONEncoder().encode(7).write(to: root.appendingPathComponent("model-contract.json"))
        let archivedCount = try await store.removeObsoleteModelArtifacts(currentSchema: ModelContract.schemaVersion)
        XCTAssertEqual(archivedCount, 1)
        XCTAssertTrue(FileManager.default.fileExists(atPath: sentinel.appendingPathComponent("current-observations.bin").path))
        let archived = await store.profileDirectory(p.id)
            .appendingPathComponent("Archived Model Artifacts/Model Contract 7/Versions/\(versionID.uuidString)/weights.safetensors")
        XCTAssertEqual(try Data(contentsOf: archived), Data([1, 2, 3]))
    }

}
