import Foundation
import MLX
import MLXNN
import XCTest
@testable import AgentTrainer

/// Input/architecture invariants only. These fixtures never become training
/// examples; all fitting and feedback experiments use real recorded actions.
final class ControlGroundingTests: XCTestCase {
    private func profile(kind: RecurrentKind, precision: TrainingPrecision, past: Int) -> AIProfile {
        var p = AIProfile.fresh()
        p.preprocessing = PreprocessingSpec(width: 8, height: 8, colorMode: .grayscale)
        p.training.precision = precision
        p.training.architecture = ArchitectureSpec(convolutionChannels: [4], kernelSizes: [3], strides: [2],
            visualEmbedding: 8, recurrentWidth: 8, fusionWidths: [16], dropout: 0,
            visualPooling: .attention, attentionHeads: 2)
        p.training.architecture.recurrentKind = kind
        p.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: past, frameSpacing: 1, downsampleFactor: 1)
        return p
    }

    private func features(_ model: AgentPolicy, held: Bool) -> MLXArray {
        let n = model.profile.training.effectiveTemporalVision.pastFrameCount
        var timing = [Float](repeating: 0, count: TemporalInputContract.valueCount)
        TemporalInputContract.putClock(in: &timing, interval: 1.0 / 60, perceptionInterval: 1.0 / 60, frameAge: 0.01)
        for slot in TemporalInputContract.binaryIndices.indices {
            let offset = TemporalInputContract.clockCount + slot * TemporalInputContract.controlWidth
            timing[offset] = held ? 1 : 0
            TemporalInputContract.putTime(held ? 10 : 0.01, in: &timing, at: offset + 1)
        }
        return model.encodedTemporalFeatures(current: MLXArray.zeros([1, 8], dtype: model.dtype),
            past: n == 0 ? nil : MLXArray.zeros([1, n, 8], dtype: model.dtype),
            controls: n == 0 ? nil : (held ? MLXArray.ones([1, n, 146], dtype: model.dtype) : MLXArray.zeros([1, n, 146], dtype: model.dtype)),
            currentTiming: MLXArray(timing, [1, timing.count]),
            pastTiming: n == 0 ? nil : MLXArray(Array(repeating: timing, count: n).flatMap { $0 }, [1, n, timing.count]))
    }

    func testControlHistoryCannotChangeVisualBaseOrCameraConfidence() {
        for kind in [RecurrentKind.gru, .lstm] {
            for precision in TrainingPrecision.allCases {
                for n in [0, 2] {
                    let model = AgentPolicy(profile: profile(kind: kind, precision: precision, past: n)); model.train(false)
                    let a = model.fusedFeatures(temporalFeatures: features(model, held: false))
                    let b = model.fusedFeatures(temporalFeatures: features(model, held: true))
                    let baseA = model.visualActionLogits(fusedFeatures: a)
                    let baseB = model.visualActionLogits(fusedFeatures: b)
                    let actualA = model.logits(fusedFeatures: a)
                    let actualB = model.logits(fusedFeatures: b)
                    MLX.eval(baseA, baseB, actualA, actualB)
                    XCTAssertEqual(abs(baseA.asType(.float32) - baseB.asType(.float32)).max().item(Float.self), 0, accuracy: 1e-6)
                    for index in [0, 1, 2, 3, 12, 13, 146, 147] {
                        XCTAssertEqual(actualA[0, index].item(Float.self), actualB[0, index].item(Float.self), accuracy: 1e-6)
                    }
                    let indices = MLXArray(ActionLayout.binary)
                    let correction = abs(actualB[.ellipsis, indices].asType(.float32) - baseB[.ellipsis, indices].asType(.float32))
                    XCTAssertLessThanOrEqual(correction.max().item(Float.self), AgentPolicy.maximumControlLogitCorrection + 0.01)
                }
            }
        }
    }

    func testSameShapedOlderWeightsCannotSilentlyChangePolicySemantics() throws {
        let p = profile(kind: .gru, precision: .float32, past: 0)
        let model = AgentPolicy(profile: p)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("policy-format-\(UUID()).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let arrays = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        for format in ["AgentTrainer.Policy.v8", "unknown"] {
            try MLX.save(arrays: arrays, metadata: ["format": format], url: url)
            XCTAssertThrowsError(try model.loadWeights(from: url))
        }
        try MLX.save(arrays: arrays, url: url)
        XCTAssertThrowsError(try model.loadWeights(from: url))
        try model.saveWeights(to: url)
        XCTAssertNoThrow(try model.loadWeights(from: url))
        let diagnostic = AgentPolicy(profile: p, legacyControlFeedback: true)
        XCTAssertThrowsError(try diagnostic.saveWeights(to: url))
    }

    func testHistoryCannotOverrideConfidentPressOrRelease() {
        let model = AgentPolicy(profile: profile(kind: .gru, precision: .float32, past: 0))
        let base = MLXArray([-3.0, 3.0, 0.0, 0.0] as [Float], [1, 4])
        let context = MLXArray([1000.0, -1000.0, 2.0, -2.0] as [Float], [1, 4])
        let logits = model.boundedControlLogits(base: base, context: context)
        MLX.eval(logits)
        XCTAssertLessThan(logits[0, 0].item(Float.self), 0)
        XCTAssertGreaterThan(logits[0, 1].item(Float.self), 0)
        XCTAssertGreaterThan(logits[0, 2].item(Float.self), 0)
        XCTAssertLessThan(logits[0, 3].item(Float.self), 0)
        XCTAssertLessThanOrEqual(abs(logits - base).max().item(Float.self), AgentPolicy.maximumControlLogitCorrection)
    }
}
