import Foundation
import MLX
import MLXNN
import XCTest
@testable import AgentTrainer

/// Optimizer arithmetic fixtures, never mouse trajectories or training labels.
final class OptimizerPrecisionTests: XCTestCase {
    private func model(_ value: Float = 1) throws -> Linear {
        let model = Linear(1, 1, bias: false)
        try model.update(parameters: ModuleParameters.unflattened([
            ("weight", MLXArray([value], [1, 1]).asType(.bfloat16))
        ]), verify: .all)
        return model
    }
    private func step(_ model: Linear, _ optimizer: ResumableAdamW) {
        optimizer.update(model: model, gradients: model.mapParameters { MLXArray.ones(like: $0) }, targetType: .bfloat16)
        MLX.eval(model.parameters(), optimizer.stateArrays())
    }
    private func optimizer(master: Bool = true, learningRate: Float = 0.0003) -> ResumableAdamW {
        ResumableAdamW(learningRate: learningRate, weightDecay: 0, warmupSteps: 1,
            cycleSteps: 100_000, usesMasterWeights: master)
    }

    func testGlobalNormAccumulatesLowPrecisionGradientsInFloat32() {
        let gradients = ModuleParameters.unflattened([
            ("large", MLXArray([Float(256), 256], [2]).asType(.float16)),
            ("small", MLXArray([Float(0.0001)], [1]).asType(.float16))
        ])
        let norm = ResumableAdamW.globalGradientNorm(gradients)
        MLX.eval(norm)
        XCTAssertEqual(norm.dtype, .float32)
        XCTAssertEqual(norm.item(Float.self), Float(256) * sqrt(2), accuracy: 0.001)
    }

    func testSmallBFloat16UpdatesAccumulateInsteadOfDisappearing() throws {
        let accurate = try model(), rounded = try model()
        let a = optimizer(), b = optimizer(master: false)
        for _ in 0..<16 { step(accurate, a); step(rounded, b) }
        XCTAssertEqual(rounded.weight.item(Float.self), 1)
        XCTAssertLessThan(accurate.weight.item(Float.self), 0.999)
        XCTAssertEqual(accurate.weight.item(Float.self), 1 - 16 * 0.0003, accuracy: 0.002)
    }

    func testMasterResidualSurvivesExactCheckpointResume() throws {
        let a = try model(), oa = optimizer()
        for _ in 0..<6 { step(a, oa) }
        XCTAssertEqual(a.weight.item(Float.self), 1, "Six sub-ULP updates remain below this stored weight's rounding boundary.")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("master-resume-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let weights = root.appendingPathComponent("weights.safetensors"), state = root.appendingPathComponent("optimizer.safetensors")
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: a.parameters().flattened()), url: weights)
        try oa.save(to: state)
        step(a, oa)
        let b = try model(), ob = optimizer()
        try b.update(parameters: ModuleParameters.unflattened(MLX.loadArrays(url: weights)), verify: .all)
        try ob.load(from: state)
        XCTAssertTrue(ob.usesMasterWeights)
        step(b, ob)
        XCTAssertLessThan(a.weight.item(Float.self), 1)
        XCTAssertEqual(a.weight.item(Float.self), b.weight.item(Float.self))
        let sa = oa.stateArrays(), sb = ob.stateArrays()
        XCTAssertEqual(sa.count, sb.count)
        for (a, b) in zip(sa, sb) { XCTAssertEqual(abs(a - b).max().item(Float.self), 0) }
    }

    func testFreshWarmStartSeedsMastersFromSelectedWeights() throws {
        let model = try model(), optimizer = optimizer(learningRate: 0.1)
        optimizer.initialize(model: model)
        try model.update(parameters: ModuleParameters.unflattened([
            ("weight", MLXArray([Float(2)], [1, 1]).asType(.bfloat16))
        ]), verify: .all)
        optimizer.synchronizeInitialParameters(model: model)
        step(model, optimizer)
        XCTAssertEqual(model.weight.item(Float.self), 1.9, accuracy: 0.01)
    }

    func testLegacyCheckpointsRetainTheirOriginalRoundedAlgorithm() throws {
        let a = try model(), oa = optimizer(master: false)
        for _ in 0..<6 { step(a, oa) }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("legacy-master-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let state = root.appendingPathComponent("optimizer.safetensors")
        try oa.save(to: state)
        var (arrays, metadata) = try MLX.loadArraysAndMetadata(url: state)
        metadata.removeValue(forKey: "optimizerFormat"); metadata.removeValue(forKey: "usesMasterWeights")
        try MLX.save(arrays: arrays, metadata: metadata, url: state)
        let b = try model(), ob = optimizer()
        try ob.load(from: state); XCTAssertFalse(ob.usesMasterWeights)
        step(a, oa); step(b, ob)
        XCTAssertEqual(a.weight.item(Float.self), b.weight.item(Float.self))
        XCTAssertEqual(b.weight.item(Float.self), 1)
        // Explicit master checkpoints must never silently discard the accumulator.
        metadata["usesMasterWeights"] = "true"; metadata["optimizerFormat"] = "AgentTrainer.AdamW.v2"
        arrays.removeValue(forKey: "master.weight")
        try MLX.save(arrays: arrays, metadata: metadata, url: state)
        XCTAssertThrowsError(try optimizer().load(from: state))
    }
}
