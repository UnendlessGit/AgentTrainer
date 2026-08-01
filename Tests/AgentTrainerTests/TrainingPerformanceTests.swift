import Foundation
import MLX
import MLXNN
import MLXOptimizers
import XCTest
@testable import AgentTrainer

/// Opt-in hardware benchmark. Normal CI remains deterministic and quick; use
/// `./benchmark.sh` to build the release test bundle with MLX's Metal resource.
final class TrainingPerformanceTests: XCTestCase {
    func testCompiledMetalDataPathPreservesOrderedUpdatesAndMeasuresThroughput() throws {
        guard ProcessInfo.processInfo.environment["AGENTTRAINER_RUN_PERFORMANCE_TESTS"] == "1" else {
            throw XCTSkip("Set AGENTTRAINER_RUN_PERFORMANCE_TESTS=1 to run the local Metal benchmark.")
        }

        try Device.withDefaultDevice(.gpu) {
            var profile = AIProfile.fresh()
            let usesDefaultProfile = ProcessInfo.processInfo.environment["AGENTTRAINER_BENCHMARK_DEFAULT_PROFILE"] == "1"
            if !usesDefaultProfile {
                profile.preprocessing = PreprocessingSpec(
                    width: 128,
                    height: 72,
                    colorMode: .color,
                    bitDepth: 8,
                    chroma: .yuv420,
                    resizePolicy: .fit
                )
                profile.training.batchSize = 16
                profile.training.architecture = .small
            }
            if let rawBatchSize = ProcessInfo.processInfo.environment["AGENTTRAINER_BENCHMARK_BATCH_SIZE"],
               let batchSize = Int(rawBatchSize), batchSize > 0 {
                profile.training.batchSize = batchSize
            }
            // No recurrent random mask: this benchmark isolates graph/data-path
            // ordering. Exact random-state resume is covered by DomainTests.
            profile.training.historyLength = 0
            profile.training.architecture.dropout = 0
            profile.training.precision = .bfloat16

            MLXRandom.seed(91_337)
            let baselineModel = AgentPolicy(profile: profile)
            let fusedModel = AgentPolicy(profile: profile)
            let reusedVisionModel = AgentPolicy(profile: profile)
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("training-performance-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let initialWeights = directory.appendingPathComponent("initial.safetensors")
            try baselineModel.saveWeights(to: initialWeights)
            try fusedModel.loadWeights(from: initialWeights)
            try reusedVisionModel.loadWeights(from: initialWeights)

            let baselineOptimizer = ResumableAdamW(learningRate: 0.0003, weightDecay: 0.01)
            let fusedOptimizer = ResumableAdamW(learningRate: 0.0003, weightDecay: 0.01)
            let reusedVisionOptimizer = ResumableAdamW(learningRate: 0.0003, weightDecay: 0.01)
            for (model, optimizer) in [
                (baselineModel, baselineOptimizer),
                (fusedModel, fusedOptimizer),
                (reusedVisionModel, reusedVisionOptimizer)
            ] {
                optimizer.initialize(model: model)
            }

            let batchSize = profile.training.batchSize
            let sampleBytes = profile.preprocessing.sampleByteCount
            let uniqueVisionCount = (batchSize + 1) / 2
            var currentValues = [UInt8](repeating: 0, count: batchSize * sampleBytes)
            var previousValues = [UInt8](repeating: 0, count: batchSize * sampleBytes)
            var pairValues = [UInt8](repeating: 0, count: batchSize * 2 * sampleBytes)
            var uniquePairValues = [UInt8](repeating: 0, count: uniqueVisionCount * 2 * sampleBytes)
            for row in 0..<batchSize {
                let visionRow = row / 2
                for byte in 0..<sampleBytes {
                    let current = UInt8(truncatingIfNeeded: visionRow &* 29 &+ byte &* 17)
                    let previous = UInt8(truncatingIfNeeded: visionRow &* 31 &+ byte &* 13 &+ 7)
                    currentValues[row * sampleBytes + byte] = current
                    previousValues[row * sampleBytes + byte] = previous
                    pairValues[(row * 2) * sampleBytes + byte] = current
                    pairValues[(row * 2 + 1) * sampleBytes + byte] = previous
                    uniquePairValues[(visionRow * 2) * sampleBytes + byte] = current
                    uniquePairValues[(visionRow * 2 + 1) * sampleBytes + byte] = previous
                }
            }
            let currentBytes = Data(currentValues)
            let previousBytes = Data(previousValues)
            let pairBytes = Data(pairValues)
            let uniquePairBytes = Data(uniquePairValues)
            let visionMapping = (0..<batchSize).map { Int32($0 / 2) }
            let visionMappingBytes = visionMapping.withUnsafeBytes { Data($0) }

            let historyLength = 1
            let historyValues = [Float](repeating: 0, count: batchSize * ActionLayout.count)
            var targetValues = [Float](repeating: 0, count: batchSize * ActionLayout.count)
            var previousTargetValues = [Float](repeating: 0, count: batchSize * ActionLayout.count)
            var actionRows = [Float](repeating: 0, count: batchSize * (historyLength + 2) * ActionLayout.count)
            for row in 0..<batchSize {
                let targetBase = row * ActionLayout.count
                targetValues[targetBase] = Float(row) / Float(max(1, batchSize - 1))
                targetValues[targetBase + 1] = 1 - targetValues[targetBase]
                targetValues[targetBase + ActionLayout.keyboard.lowerBound + row] = 1
                if row.isMultiple(of: 2) {
                    previousTargetValues[targetBase + ActionLayout.keyboard.lowerBound + row] = 1
                }
                let fusedBase = row * (historyLength + 2) * ActionLayout.count
                actionRows.replaceSubrange(
                    fusedBase..<(fusedBase + ActionLayout.count),
                    with: historyValues[targetBase..<(targetBase + ActionLayout.count)]
                )
                actionRows.replaceSubrange(
                    (fusedBase + ActionLayout.count)..<(fusedBase + 2 * ActionLayout.count),
                    with: targetValues[targetBase..<(targetBase + ActionLayout.count)]
                )
                actionRows.replaceSubrange(
                    (fusedBase + 2 * ActionLayout.count)..<(fusedBase + 3 * ActionLayout.count),
                    with: previousTargetValues[targetBase..<(targetBase + ActionLayout.count)]
                )
            }
            let historyBytes = historyValues.withUnsafeBytes { Data($0) }
            let targetBytes = targetValues.withUnsafeBytes { Data($0) }
            let previousTargetBytes = previousTargetValues.withUnsafeBytes { Data($0) }
            let actionBytes = actionRows.withUnsafeBytes { Data($0) }
            let positiveWeights = [Float](repeating: 1, count: ActionLayout.count)
            let fusedInputBuffers = try MetalArrayBufferPool(maximumCachedBytes: 256 * 1_024 * 1_024)

            // Reproduce the cache gather that precedes MLX materialization.
            // `Data(source)` can share storage, so force the same allocation and
            // byte copy performed by the old per-field dataset helpers.
            func gatheredCopy(_ source: Data) -> Data {
                var destination = Data(count: source.count)
                destination.withUnsafeMutableBytes { output in
                    source.withUnsafeBytes { output.copyMemory(from: $0) }
                }
                return destination
            }

            func legacyInputs() -> [MLXArray] {
                let current = gatheredCopy(currentBytes)
                let previous = gatheredCopy(previousBytes)
                let gatheredHistory = gatheredCopy(historyBytes)
                let targets = gatheredCopy(targetBytes)
                let previousTargets = gatheredCopy(previousTargetBytes)
                return [
                    VisionPreprocessor.mlxTemporalTensor(
                        current: current,
                        previous: previous,
                        batch: batchSize,
                        spec: profile.preprocessing
                    ),
                    MLXArray(gatheredHistory, [batchSize, historyLength, ActionLayout.count], type: Float.self),
                    MLXArray(targets, [batchSize, ActionLayout.count], type: Float.self),
                    MLXArray(previousTargets, [batchSize, ActionLayout.count], type: Float.self)
                ]
            }

            func fusedInputs() -> [MLXArray] {
                return try! fusedInputBuffers.makeArrays([
                    .init([batchSize, 2, sampleBytes], dtype: .uint8),
                    .init([batchSize, historyLength + 2, ActionLayout.count], dtype: .float32)
                ]) { destinations in
                    pairBytes.withUnsafeBytes { destinations[0].copyMemory(from: $0) }
                    actionBytes.withUnsafeBytes { destinations[1].copyMemory(from: $0) }
                }
            }

            func reusedVisionInputs() -> [MLXArray] {
                return try! fusedInputBuffers.makeArrays([
                    .init([uniqueVisionCount, 2, sampleBytes], dtype: .uint8),
                    .init([batchSize], dtype: .int32),
                    .init([batchSize, historyLength + 2, ActionLayout.count], dtype: .float32)
                ]) { destinations in
                    uniquePairBytes.withUnsafeBytes { destinations[0].copyMemory(from: $0) }
                    visionMappingBytes.withUnsafeBytes { destinations[1].copyMemory(from: $0) }
                    actionBytes.withUnsafeBytes { destinations[2].copyMemory(from: $0) }
                }
            }

            func compiledLegacyStep(
                model: AgentPolicy,
                optimizer: ResumableAdamW
            ) -> @Sendable ([MLXArray]) -> [MLXArray] {
                compile(inputs: [model, optimizer], outputs: [model, optimizer]) { arrays in
                    let weights = MLXArray(positiveWeights, [ActionLayout.count])
                    let result = valueAndGrad(model: model) { model, inputs in
                        let visualFeatures = model.visualActivations(
                            images: inputs[0],
                            acceleratedOperators: false
                        ).last!
                        let logits = model.logits(
                            visualFeatures: visualFeatures,
                            history: inputs[1]
                        )
                        return [model.loss(
                            logits: logits,
                            history: inputs[1],
                            targets: inputs[2],
                            positiveWeights: weights,
                            previousTargets: inputs[3]
                        )]
                    }(model, arrays)
                    optimizer.update(
                        model: model,
                        gradients: clipGradNorm(gradients: result.1, maxNorm: 1).0,
                        targetType: model.dtype
                    )
                    return [result.0[0]]
                }
            }

            func compiledFusedStep(
                model: AgentPolicy,
                optimizer: ResumableAdamW
            ) -> @Sendable ([MLXArray]) -> [MLXArray] {
                compile(inputs: [model, optimizer], outputs: [model, optimizer]) { arrays in
                    let actions = arrays[1]
                    let inputs = [
                        VisionPreprocessor.mlxTemporalTensor(arrays[0], spec: profile.preprocessing),
                        actions[0..., 0..<historyLength, 0...],
                        actions[0..., historyLength, 0...],
                        actions[0..., historyLength + 1, 0...]
                    ]
                    let weights = MLXArray(positiveWeights, [ActionLayout.count])
                    let result = valueAndGrad(model: model) { model, inputs in
                        let visualFeatures = model.visualActivations(
                            images: inputs[0]
                        ).last!
                        let logits = model.logits(
                            visualFeatures: visualFeatures,
                            history: inputs[1]
                        )
                        return [model.loss(
                            logits: logits,
                            history: inputs[1],
                            targets: inputs[2],
                            positiveWeights: weights,
                            previousTargets: inputs[3]
                        )]
                    }(model, inputs)
                    optimizer.update(
                        model: model,
                        gradients: result.1,
                        targetType: model.dtype,
                        gradientNorm: ResumableAdamW.globalGradientNorm(result.1),
                        maxGradientNorm: 1
                    )
                    return [result.0[0]]
                }
            }

            func compiledReusedVisionStep(
                model: AgentPolicy,
                optimizer: ResumableAdamW
            ) -> @Sendable ([MLXArray]) -> [MLXArray] {
                compile(inputs: [model, optimizer], outputs: [model, optimizer]) { arrays in
                    let actions = arrays[2]
                    let inputs = [
                        VisionPreprocessor.mlxTemporalTensor(arrays[0], spec: profile.preprocessing),
                        actions[0..., 0..<historyLength, 0...],
                        actions[0..., historyLength, 0...],
                        actions[0..., historyLength + 1, 0...]
                    ]
                    let weights = MLXArray(positiveWeights, [ActionLayout.count])
                    let result = valueAndGrad(model: model) { model, inputs in
                        let uniqueVisualFeatures = model.visualActivations(images: inputs[0]).last!
                        let uniqueVisualEmbedding = model.visualEmbedding(
                            visualFeatures: uniqueVisualFeatures
                        )
                        let visualEmbedding = uniqueVisualEmbedding.take(arrays[1], axis: 0)
                        let logits = model.logits(
                            visualEmbedding: visualEmbedding,
                            history: inputs[1]
                        )
                        return [model.loss(
                            logits: logits,
                            history: inputs[1],
                            targets: inputs[2],
                            positiveWeights: weights,
                            previousTargets: inputs[3]
                        )]
                    }(model, inputs)
                    optimizer.update(
                        model: model,
                        gradients: result.1,
                        targetType: model.dtype,
                        gradientNorm: ResumableAdamW.globalGradientNorm(result.1),
                        maxGradientNorm: 1
                    )
                    return [result.0[0]]
                }
            }

            let baselineStep = compiledLegacyStep(model: baselineModel, optimizer: baselineOptimizer)
            let fusedStep = compiledFusedStep(model: fusedModel, optimizer: fusedOptimizer)
            let reusedVisionStep = compiledReusedVisionStep(
                model: reusedVisionModel,
                optimizer: reusedVisionOptimizer
            )

            func warmUp(
                _ step: @Sendable ([MLXArray]) -> [MLXArray],
                inputs: () -> [MLXArray],
                model: AgentPolicy,
                optimizer: ResumableAdamW
            ) {
                let loss = step(inputs())[0]
                MLX.eval(loss, model.parameters(), optimizer.stateArrays())
            }
            for _ in 0..<3 {
                warmUp(baselineStep, inputs: legacyInputs, model: baselineModel, optimizer: baselineOptimizer)
                warmUp(fusedStep, inputs: fusedInputs, model: fusedModel, optimizer: fusedOptimizer)
                warmUp(
                    reusedVisionStep,
                    inputs: reusedVisionInputs,
                    model: reusedVisionModel,
                    optimizer: reusedVisionOptimizer
                )
            }

            let iterations = usesDefaultProfile ? 12 : 32
            func runOne(
                _ step: @Sendable ([MLXArray]) -> [MLXArray],
                inputs: () -> [MLXArray],
                model: AgentPolicy,
                optimizer: ResumableAdamW
            ) -> (seconds: Double, loss: Float) {
                let start = ContinuousClock.now
                let loss = step(inputs())[0]
                MLX.eval(loss, model.parameters(), optimizer.stateArrays())
                return (start.duration(to: .now).benchmarkSeconds, loss.item(Float.self))
            }
            var baselineSeconds = 0.0, fusedSeconds = 0.0
            var reusedVisionSeconds = 0.0
            var baselineLosses: [Float] = [], fusedLosses: [Float] = []
            var reusedVisionLosses: [Float] = []
            baselineLosses.reserveCapacity(iterations)
            fusedLosses.reserveCapacity(iterations)
            reusedVisionLosses.reserveCapacity(iterations)
            func runBaseline() {
                let result = runOne(
                    baselineStep,
                    inputs: legacyInputs,
                    model: baselineModel,
                    optimizer: baselineOptimizer
                )
                baselineSeconds += result.seconds
                baselineLosses.append(result.loss)
            }
            func runFused() {
                let result = runOne(
                    fusedStep,
                    inputs: fusedInputs,
                    model: fusedModel,
                    optimizer: fusedOptimizer
                )
                fusedSeconds += result.seconds
                fusedLosses.append(result.loss)
            }
            func runReusedVision() {
                let result = runOne(
                    reusedVisionStep,
                    inputs: reusedVisionInputs,
                    model: reusedVisionModel,
                    optimizer: reusedVisionOptimizer
                )
                reusedVisionSeconds += result.seconds
                reusedVisionLosses.append(result.loss)
            }
            for iteration in 0..<iterations {
                if iteration.isMultiple(of: 2) {
                    runBaseline()
                    runFused()
                    runReusedVision()
                } else {
                    runReusedVision()
                    runFused()
                    runBaseline()
                }
            }

            let baselineToFusedLossDelta = maximumAbsoluteDifference(baselineLosses, fusedLosses)
            let baselineToFusedParameterDelta = maximumAbsoluteDifference(
                modelArrays(baselineModel),
                modelArrays(fusedModel)
            )
            XCTAssertLessThanOrEqual(baselineToFusedLossDelta, 0.02)
            let reusedVisionLossDelta = maximumAbsoluteDifference(fusedLosses, reusedVisionLosses)
            let reusedVisionParameterDelta = maximumAbsoluteDifference(
                modelArrays(fusedModel),
                modelArrays(reusedVisionModel)
            )
            XCTAssertLessThanOrEqual(reusedVisionLossDelta, 0.02)

            let fusedSpeedup = baselineSeconds / max(0.000_001, fusedSeconds)
            print(
                "TRAINING_BENCHMARK profile=\(usesDefaultProfile ? "default" : "compact") "
                    + "baseline_seconds=\(baselineSeconds) optimized_seconds=\(fusedSeconds) "
                    + "speedup=\(fusedSpeedup) "
                    + "optimized_samples_per_second=\(Double(iterations * batchSize) / fusedSeconds) "
                    + "max_loss_delta=\(baselineToFusedLossDelta) "
                    + "max_parameter_delta=\(baselineToFusedParameterDelta)"
            )
            print(
                "REUSED_VISION_BENCHMARK profile=\(usesDefaultProfile ? "default" : "compact") "
                    + "baseline_seconds=\(fusedSeconds) optimized_seconds=\(reusedVisionSeconds) "
                    + "speedup=\(fusedSeconds / max(0.000_001, reusedVisionSeconds)) "
                    + "optimized_samples_per_second=\(Double(iterations * batchSize) / reusedVisionSeconds) "
                    + "unique_vision_fraction=\(Double(uniqueVisionCount) / Double(batchSize)) "
                    + "max_loss_delta=\(reusedVisionLossDelta) "
                    + "max_parameter_delta=\(reusedVisionParameterDelta)"
            )
            print(
                "TOTAL_TRAINING_BENCHMARK profile=\(usesDefaultProfile ? "default" : "compact") "
                    + "original_seconds=\(baselineSeconds) optimized_seconds=\(reusedVisionSeconds) "
                    + "speedup=\(baselineSeconds / max(0.000_001, reusedVisionSeconds)) "
                    + "optimized_samples_per_second=\(Double(iterations * batchSize) / reusedVisionSeconds)"
            )

            fusedModel.train(false)
            reusedVisionModel.train(false)
            let qualityInputs = legacyInputs()
            let fusedQualityLoss = fusedModel.loss(
                images: qualityInputs[0],
                history: qualityInputs[1],
                targets: qualityInputs[2],
                positiveWeights: MLXArray(positiveWeights, [ActionLayout.count]),
                previousTargets: qualityInputs[3]
            )
            let reusedVisionQualityLoss = reusedVisionModel.loss(
                images: qualityInputs[0],
                history: qualityInputs[1],
                targets: qualityInputs[2],
                positiveWeights: MLXArray(positiveWeights, [ActionLayout.count]),
                previousTargets: qualityInputs[3]
            )
            MLX.eval(fusedQualityLoss, reusedVisionQualityLoss)
            let fusedQualityValue = fusedQualityLoss.item(Float.self)
            let reusedVisionQualityValue = reusedVisionQualityLoss.item(Float.self)
            let relativeQualityDelta = abs(fusedQualityValue - reusedVisionQualityValue)
                / max(0.000_001, abs(fusedQualityValue))
            XCTAssertLessThanOrEqual(relativeQualityDelta, 0.01)
            print(
                "LEARNING_QUALITY profile=\(usesDefaultProfile ? "default" : "compact") "
                    + "baseline_training_loss=\(fusedLosses.last ?? .nan) "
                    + "optimized_training_loss=\(reusedVisionLosses.last ?? .nan) "
                    + "baseline_validation_loss=\(fusedQualityValue) "
                    + "optimized_validation_loss=\(reusedVisionQualityValue) "
                    + "relative_validation_delta=\(relativeQualityDelta)"
            )
            let reusedValidationStep = compile(inputs: [reusedVisionModel]) { arrays in
                let actions = arrays[2]
                let images = VisionPreprocessor.mlxTemporalTensor(arrays[0], spec: profile.preprocessing)
                let history = actions[0..., 0..<historyLength, 0...]
                let targets = actions[0..., historyLength, 0...]
                let previousTargets = actions[0..., historyLength + 1, 0...]
                let uniqueVisualFeatures = reusedVisionModel.visualActivations(images: images).last!
                let uniqueVisualEmbedding = reusedVisionModel.visualEmbedding(
                    visualFeatures: uniqueVisualFeatures
                )
                let visualEmbedding = uniqueVisualEmbedding.take(arrays[1], axis: 0)
                let logits = reusedVisionModel.logits(
                    visualEmbedding: visualEmbedding,
                    history: history
                )
                let weights = MLXArray(positiveWeights, [ActionLayout.count])
                return [
                    reusedVisionModel.loss(
                        logits: logits,
                        history: history,
                        targets: targets,
                        positiveWeights: weights,
                        previousTargets: previousTargets
                    ),
                    fusedModel.activatedPredictions(logits: logits),
                    targets
                ]
            }
            func legacyValidation() -> [MLXArray] {
                let arrays = legacyInputs()
                return [
                    reusedVisionModel.loss(
                        images: arrays[0],
                        history: arrays[1],
                        targets: arrays[2],
                        positiveWeights: MLXArray(positiveWeights, [ActionLayout.count]),
                        previousTargets: arrays[3]
                    ),
                    reusedVisionModel.predictions(images: arrays[0], history: arrays[1]),
                    arrays[2]
                ]
            }
            func optimizedValidation() -> [MLXArray] {
                reusedValidationStep(reusedVisionInputs())
            }
            for _ in 0..<3 {
                MLX.eval(legacyValidation())
                MLX.eval(optimizedValidation())
            }
            let validationIterations = usesDefaultProfile ? 12 : 64
            var legacyValidationResult: [MLXArray] = []
            var optimizedValidationResult: [MLXArray] = []
            var legacyValidationSeconds = 0.0, optimizedValidationSeconds = 0.0
            func runLegacyValidation() {
                let start = ContinuousClock.now
                legacyValidationResult = legacyValidation()
                MLX.eval(legacyValidationResult)
                legacyValidationSeconds += start.duration(to: .now).benchmarkSeconds
            }
            func runOptimizedValidation() {
                let start = ContinuousClock.now
                optimizedValidationResult = optimizedValidation()
                MLX.eval(optimizedValidationResult)
                optimizedValidationSeconds += start.duration(to: .now).benchmarkSeconds
            }
            for iteration in 0..<validationIterations {
                if iteration.isMultiple(of: 2) {
                    runLegacyValidation()
                    runOptimizedValidation()
                } else {
                    runOptimizedValidation()
                    runLegacyValidation()
                }
            }
            let validationLossDelta = abs(
                legacyValidationResult[0].item(Float.self)
                    - optimizedValidationResult[0].item(Float.self)
            )
            let validationPredictionDelta = maximumAbsoluteDifference(
                legacyValidationResult[1].asArray(Float.self),
                optimizedValidationResult[1].asArray(Float.self)
            )
            XCTAssertLessThanOrEqual(validationLossDelta, 0.000_01)
            XCTAssertLessThanOrEqual(validationPredictionDelta, 0.000_01)
            print(
                "VALIDATION_BENCHMARK profile=\(usesDefaultProfile ? "default" : "compact") "
                    + "baseline_seconds=\(legacyValidationSeconds) "
                    + "optimized_seconds=\(optimizedValidationSeconds) "
                    + "speedup=\(legacyValidationSeconds / max(0.000_001, optimizedValidationSeconds)) "
                    + "max_loss_delta=\(validationLossDelta) "
                    + "max_prediction_delta=\(validationPredictionDelta)"
            )

            func compiledInference(accelerated: Bool) -> @Sendable ([MLXArray]) -> [MLXArray] {
                compile(inputs: [fusedModel]) { arrays in
                    let actions = arrays[1]
                    let images = VisionPreprocessor.mlxTemporalTensor(
                        arrays[0],
                        spec: profile.preprocessing
                    )
                    let history = actions[0..., 0..<historyLength, 0...]
                    let visual = accelerated
                        ? fusedModel.visualActivations(images: images).last!
                        : fusedModel.visualActivations(images: images, acceleratedOperators: false).last!
                    let logits = fusedModel.logits(
                        visualFeatures: visual,
                        history: history
                    )
                    return [fusedModel.activatedPredictions(logits: logits)]
                }
            }
            let legacyInference = compiledInference(accelerated: false)
            let acceleratedInference = compiledInference(accelerated: true)
            for _ in 0..<3 {
                MLX.eval(legacyInference(fusedInputs()))
                MLX.eval(acceleratedInference(fusedInputs()))
            }
            let inferenceIterations = usesDefaultProfile ? 24 : 96
            var legacyInferenceSeconds = 0.0, acceleratedInferenceSeconds = 0.0
            var legacyInferenceResult = MLXArray(0), acceleratedInferenceResult = MLXArray(0)
            func runLegacyInference() {
                let start = ContinuousClock.now
                legacyInferenceResult = legacyInference(fusedInputs())[0]
                MLX.eval(legacyInferenceResult)
                legacyInferenceSeconds += start.duration(to: .now).benchmarkSeconds
            }
            func runAcceleratedInference() {
                let start = ContinuousClock.now
                acceleratedInferenceResult = acceleratedInference(fusedInputs())[0]
                MLX.eval(acceleratedInferenceResult)
                acceleratedInferenceSeconds += start.duration(to: .now).benchmarkSeconds
            }
            for iteration in 0..<inferenceIterations {
                if iteration.isMultiple(of: 2) {
                    runLegacyInference()
                    runAcceleratedInference()
                } else {
                    runAcceleratedInference()
                    runLegacyInference()
                }
            }
            let inferencePredictionDelta = maximumAbsoluteDifference(
                legacyInferenceResult.asArray(Float.self),
                acceleratedInferenceResult.asArray(Float.self)
            )
            XCTAssertLessThanOrEqual(inferencePredictionDelta, 0.02)
            print(
                "INFERENCE_BENCHMARK profile=\(usesDefaultProfile ? "default" : "compact") "
                    + "baseline_seconds=\(legacyInferenceSeconds) "
                    + "optimized_seconds=\(acceleratedInferenceSeconds) "
                    + "speedup=\(legacyInferenceSeconds / max(0.000_001, acceleratedInferenceSeconds)) "
                    + "max_prediction_delta=\(inferencePredictionDelta)"
            )
        }
    }

    private func modelArrays(_ model: AgentPolicy) -> [MLXArray] {
        model.parameters().flattened().sorted { $0.0 < $1.0 }.map(\.1)
    }

    private func maximumAbsoluteDifference(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(0) { max($0, abs($1.0 - $1.1)) }
    }

    private func maximumAbsoluteDifference(_ lhs: [MLXArray], _ rhs: [MLXArray]) -> Float {
        guard lhs.count == rhs.count else { return .infinity }
        MLX.eval(lhs, rhs)
        return zip(lhs, rhs).reduce(0) { maximum, pair in
            max(maximum, maximumAbsoluteDifference(pair.0.asArray(Float.self), pair.1.asArray(Float.self)))
        }
    }
}

private extension Duration {
    var benchmarkSeconds: Double {
        let parts = components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
