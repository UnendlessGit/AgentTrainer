import Foundation
import MLX
import MLXNN
import XCTest
@testable import AgentTrainer

/// Opt-in hardware benchmark. Normal CI remains deterministic and quick; use
/// `./benchmark.sh` to exercise the complete packed temporal-vision data path.
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
                profile.training.temporalVision = TemporalVisionConfiguration(
                    pastFrameCount: 4,
                    frameSpacing: 2,
                    downsampleFactor: 2
                )
            }
            if let rawBatchSize = ProcessInfo.processInfo.environment["AGENTTRAINER_BENCHMARK_BATCH_SIZE"],
               let batchSize = Int(rawBatchSize), batchSize > 0 {
                profile.training.batchSize = batchSize
            }
            profile.training.architecture.dropout = 0
            profile.training.precision = .bfloat16

            MLXRandom.seed(91_337)
            let baselineModel = AgentPolicy(profile: profile)
            let fusedModel = AgentPolicy(profile: profile)
            let reusedTemporalModel = AgentPolicy(profile: profile)
            for model in [baselineModel, fusedModel, reusedTemporalModel] {
                // The benchmark measures deterministic optimizer and data-path
                // work. Dropout/control masking quality is covered separately.
                model.train(false)
            }

            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("training-performance-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let initialWeights = directory.appendingPathComponent("initial.safetensors")
            try baselineModel.saveWeights(to: initialWeights)
            try fusedModel.loadWeights(from: initialWeights)
            try reusedTemporalModel.loadWeights(from: initialWeights)

            let baselineOptimizer = ResumableAdamW(learningRate: 0.0003, weightDecay: 0.01)
            let fusedOptimizer = ResumableAdamW(learningRate: 0.0003, weightDecay: 0.01)
            let reusedTemporalOptimizer = ResumableAdamW(learningRate: 0.0003, weightDecay: 0.01)
            for (model, optimizer) in [
                (baselineModel, baselineOptimizer),
                (fusedModel, fusedOptimizer),
                (reusedTemporalModel, reusedTemporalOptimizer)
            ] {
                optimizer.initialize(model: model)
            }

            let batchSize = profile.training.batchSize
            let temporal = profile.training.effectiveTemporalVision
            let pastSpec = temporal.pastFrameSpec(from: profile.preprocessing)
            let currentBytesPerSample = profile.preprocessing.sampleByteCount
            let pastBytesPerFrame = pastSpec.sampleByteCount
            let frameCount = temporal.pastFrameCount
            let uniqueSequenceCount = (batchSize + 1) / 2
            let controlsPerSequence = frameCount * ActionLayout.count

            var uniqueCurrent = [UInt8](repeating: 0, count: uniqueSequenceCount * currentBytesPerSample)
            var uniquePast = [UInt8](repeating: 0, count: uniqueSequenceCount * frameCount * pastBytesPerFrame)
            var uniqueControls = [Float](repeating: 0, count: uniqueSequenceCount * controlsPerSequence)
            for sequence in 0..<uniqueSequenceCount {
                for byte in 0..<currentBytesPerSample {
                    uniqueCurrent[sequence * currentBytesPerSample + byte] = UInt8(
                        truncatingIfNeeded: sequence &* 29 &+ byte &* 17
                    )
                }
                for frame in 0..<frameCount {
                    for byte in 0..<pastBytesPerFrame {
                        uniquePast[(sequence * frameCount + frame) * pastBytesPerFrame + byte] = UInt8(
                            truncatingIfNeeded: sequence &* 31 &+ frame &* 23 &+ byte &* 13 &+ 7
                        )
                    }
                    let controlBase = (sequence * frameCount + frame) * ActionLayout.count
                    uniqueControls[controlBase] = Float(sequence) / Float(max(1, uniqueSequenceCount - 1))
                    uniqueControls[controlBase + ActionLayout.keyboard.lowerBound + ((sequence + frame) % 32)] = 1
                }
            }

            let mapping = (0..<batchSize).map { Int32($0 / 2) }
            var fullCurrent = [UInt8](repeating: 0, count: batchSize * currentBytesPerSample)
            var fullPast = [UInt8](repeating: 0, count: batchSize * frameCount * pastBytesPerFrame)
            var fullControls = [Float](repeating: 0, count: batchSize * controlsPerSequence)
            for row in 0..<batchSize {
                let sequence = Int(mapping[row])
                fullCurrent.replaceSubrange(
                    row * currentBytesPerSample..<(row + 1) * currentBytesPerSample,
                    with: uniqueCurrent[sequence * currentBytesPerSample..<(sequence + 1) * currentBytesPerSample]
                )
                fullPast.replaceSubrange(
                    row * frameCount * pastBytesPerFrame..<(row + 1) * frameCount * pastBytesPerFrame,
                    with: uniquePast[sequence * frameCount * pastBytesPerFrame..<(sequence + 1) * frameCount * pastBytesPerFrame]
                )
                fullControls.replaceSubrange(
                    row * controlsPerSequence..<(row + 1) * controlsPerSequence,
                    with: uniqueControls[sequence * controlsPerSequence..<(sequence + 1) * controlsPerSequence]
                )
            }

            var targetValues = [Float](repeating: 0, count: batchSize * ActionLayout.count)
            var previousTargetValues = [Float](repeating: 0, count: batchSize * ActionLayout.count)
            for row in 0..<batchSize {
                let base = row * ActionLayout.count
                targetValues[base] = Float(row) / Float(max(1, batchSize - 1))
                targetValues[base + 1] = 1 - targetValues[base]
                targetValues[base + ActionLayout.keyboard.lowerBound + (row % 32)] = 1
                if row.isMultiple(of: 2) {
                    previousTargetValues[base + ActionLayout.keyboard.lowerBound + (row % 32)] = 1
                }
            }
            var actionRows = [Float](repeating: 0, count: batchSize * 2 * ActionLayout.count)
            for row in 0..<batchSize {
                let source = row * ActionLayout.count
                let destination = row * 2 * ActionLayout.count
                actionRows.replaceSubrange(
                    destination..<(destination + ActionLayout.count),
                    with: targetValues[source..<(source + ActionLayout.count)]
                )
                actionRows.replaceSubrange(
                    (destination + ActionLayout.count)..<(destination + 2 * ActionLayout.count),
                    with: previousTargetValues[source..<(source + ActionLayout.count)]
                )
            }

            let fullCurrentData = Data(fullCurrent)
            let fullPastData = Data(fullPast)
            let fullControlData = fullControls.withUnsafeBytes { Data($0) }
            let targetData = targetValues.withUnsafeBytes { Data($0) }
            let previousTargetData = previousTargetValues.withUnsafeBytes { Data($0) }
            let uniqueCurrentData = Data(uniqueCurrent)
            let uniquePastData = Data(uniquePast)
            let uniqueControlData = uniqueControls.withUnsafeBytes { Data($0) }
            let mappingData = mapping.withUnsafeBytes { Data($0) }
            let actionData = actionRows.withUnsafeBytes { Data($0) }
            let positiveWeights = MLXArray.ones([ActionLayout.count])
            let bufferPool = try MetalArrayBufferPool(maximumCachedBytes: 512 * 1_024 * 1_024)

            func copied(_ source: Data) -> Data {
                var destination = Data(count: source.count)
                destination.withUnsafeMutableBytes { output in
                    source.withUnsafeBytes { output.copyMemory(from: $0) }
                }
                return destination
            }

            func baselineInputs() -> [MLXArray] {
                let current = VisionPreprocessor.mlxTensor(
                    copied(fullCurrentData), batch: batchSize, spec: profile.preprocessing
                )
                let past = VisionPreprocessor.mlxPastFrameTensor(
                    MLXArray(copied(fullPastData), [batchSize, frameCount, pastBytesPerFrame], dtype: .uint8),
                    spec: pastSpec
                )
                return [
                    current,
                    past,
                    MLXArray(copied(fullControlData), [batchSize, frameCount, ActionLayout.count], type: Float.self),
                    MLXArray(copied(targetData), [batchSize, ActionLayout.count], type: Float.self),
                    MLXArray(copied(previousTargetData), [batchSize, ActionLayout.count], type: Float.self)
                ]
            }

            func fusedInputs() -> [MLXArray] {
                try! bufferPool.makeArrays([
                    .init([batchSize, currentBytesPerSample], dtype: .uint8),
                    .init([batchSize, frameCount, pastBytesPerFrame], dtype: .uint8),
                    .init([batchSize, frameCount, ActionLayout.count], dtype: .float32),
                    .init([batchSize, 2, ActionLayout.count], dtype: .float32)
                ]) { destinations in
                    fullCurrentData.withUnsafeBytes { destinations[0].copyMemory(from: $0) }
                    fullPastData.withUnsafeBytes { destinations[1].copyMemory(from: $0) }
                    fullControlData.withUnsafeBytes { destinations[2].copyMemory(from: $0) }
                    actionData.withUnsafeBytes { destinations[3].copyMemory(from: $0) }
                }
            }

            func reusedInputs() -> [MLXArray] {
                try! bufferPool.makeArrays([
                    .init([uniqueSequenceCount, currentBytesPerSample], dtype: .uint8),
                    .init([uniqueSequenceCount, frameCount, pastBytesPerFrame], dtype: .uint8),
                    .init([uniqueSequenceCount, frameCount, ActionLayout.count], dtype: .float32),
                    .init([batchSize], dtype: .int32),
                    .init([batchSize, 2, ActionLayout.count], dtype: .float32)
                ]) { destinations in
                    uniqueCurrentData.withUnsafeBytes { destinations[0].copyMemory(from: $0) }
                    uniquePastData.withUnsafeBytes { destinations[1].copyMemory(from: $0) }
                    uniqueControlData.withUnsafeBytes { destinations[2].copyMemory(from: $0) }
                    mappingData.withUnsafeBytes { destinations[3].copyMemory(from: $0) }
                    actionData.withUnsafeBytes { destinations[4].copyMemory(from: $0) }
                }
            }

            func compiledBaselineStep(
                model: AgentPolicy,
                optimizer: ResumableAdamW
            ) -> @Sendable ([MLXArray]) -> [MLXArray] {
                compile(inputs: [model, optimizer], outputs: [model, optimizer]) { arrays in
                    let result = valueAndGrad(model: model) { model, inputs in
                        [model.loss(
                            currentImages: inputs[0],
                            pastImages: inputs[1],
                            pastControls: inputs[2],
                            targets: inputs[3],
                            positiveWeights: positiveWeights,
                            previousTargets: inputs[4]
                        )]
                    }(model, arrays)
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

            func compiledFusedStep(
                model: AgentPolicy,
                optimizer: ResumableAdamW
            ) -> @Sendable ([MLXArray]) -> [MLXArray] {
                compile(inputs: [model, optimizer], outputs: [model, optimizer]) { arrays in
                    let actions = arrays[3]
                    let inputs = [
                        VisionPreprocessor.mlxTensor(arrays[0], spec: profile.preprocessing),
                        VisionPreprocessor.mlxPastFrameTensor(arrays[1], spec: pastSpec),
                        arrays[2],
                        actions[0..., 0, 0...],
                        actions[0..., 1, 0...]
                    ]
                    let result = valueAndGrad(model: model) { model, values in
                        [model.loss(
                            currentImages: values[0],
                            pastImages: values[1],
                            pastControls: values[2],
                            targets: values[3],
                            positiveWeights: positiveWeights,
                            previousTargets: values[4]
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

            func compiledReusedStep(
                model: AgentPolicy,
                optimizer: ResumableAdamW
            ) -> @Sendable ([MLXArray]) -> [MLXArray] {
                compile(inputs: [model, optimizer], outputs: [model, optimizer]) { arrays in
                    let actions = arrays[4]
                    let uniqueInputs = [
                        VisionPreprocessor.mlxTensor(arrays[0], spec: profile.preprocessing),
                        VisionPreprocessor.mlxPastFrameTensor(arrays[1], spec: pastSpec),
                        arrays[2]
                    ]
                    let targets = actions[0..., 0, 0...]
                    let previousTargets = actions[0..., 1, 0...]
                    let result = valueAndGrad(model: model) { model, values in
                        let uniqueTemporal = model.temporalFeatures(
                            currentImages: values[0],
                            pastImages: values[1],
                            pastControls: values[2]
                        )
                        let logits = model.logits(temporalFeatures: uniqueTemporal.take(arrays[3], axis: 0))
                        return [model.loss(
                            logits: logits,
                            targets: targets,
                            positiveWeights: positiveWeights,
                            previousTargets: previousTargets
                        )]
                    }(model, uniqueInputs)
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

            let baselineStep = compiledBaselineStep(model: baselineModel, optimizer: baselineOptimizer)
            let fusedStep = compiledFusedStep(model: fusedModel, optimizer: fusedOptimizer)
            let reusedStep = compiledReusedStep(model: reusedTemporalModel, optimizer: reusedTemporalOptimizer)

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

            for _ in 0..<3 {
                _ = runOne(baselineStep, inputs: baselineInputs, model: baselineModel, optimizer: baselineOptimizer)
                _ = runOne(fusedStep, inputs: fusedInputs, model: fusedModel, optimizer: fusedOptimizer)
                _ = runOne(reusedStep, inputs: reusedInputs, model: reusedTemporalModel, optimizer: reusedTemporalOptimizer)
            }

            let iterations = usesDefaultProfile ? 12 : 32
            var baselineSeconds = 0.0, fusedSeconds = 0.0, reusedSeconds = 0.0
            var baselineLosses: [Float] = [], fusedLosses: [Float] = [], reusedLosses: [Float] = []
            for iteration in 0..<iterations {
                let order = iteration.isMultiple(of: 2) ? [0, 1, 2] : [2, 1, 0]
                for benchmark in order {
                    switch benchmark {
                    case 0:
                        let result = runOne(baselineStep, inputs: baselineInputs, model: baselineModel, optimizer: baselineOptimizer)
                        baselineSeconds += result.seconds; baselineLosses.append(result.loss)
                    case 1:
                        let result = runOne(fusedStep, inputs: fusedInputs, model: fusedModel, optimizer: fusedOptimizer)
                        fusedSeconds += result.seconds; fusedLosses.append(result.loss)
                    default:
                        let result = runOne(reusedStep, inputs: reusedInputs, model: reusedTemporalModel, optimizer: reusedTemporalOptimizer)
                        reusedSeconds += result.seconds; reusedLosses.append(result.loss)
                    }
                }
            }

            let fusedLossDelta = maximumAbsoluteDifference(baselineLosses, fusedLosses)
            let fusedParameterDelta = maximumAbsoluteDifference(modelArrays(baselineModel), modelArrays(fusedModel))
            let reusedLossDelta = maximumAbsoluteDifference(fusedLosses, reusedLosses)
            let reusedParameterDelta = maximumAbsoluteDifference(modelArrays(fusedModel), modelArrays(reusedTemporalModel))
            XCTAssertLessThanOrEqual(fusedLossDelta, 0.02)
            XCTAssertLessThanOrEqual(reusedLossDelta, 0.02)

            let profileName = usesDefaultProfile ? "default" : "compact"
            print(
                "TRAINING_BENCHMARK profile=\(profileName) baseline_seconds=\(baselineSeconds) "
                    + "optimized_seconds=\(fusedSeconds) speedup=\(baselineSeconds / max(0.000_001, fusedSeconds)) "
                    + "optimized_samples_per_second=\(Double(iterations * batchSize) / fusedSeconds) "
                    + "max_loss_delta=\(fusedLossDelta) max_parameter_delta=\(fusedParameterDelta)"
            )
            print(
                "REUSED_VISION_BENCHMARK profile=\(profileName) baseline_seconds=\(fusedSeconds) "
                    + "optimized_seconds=\(reusedSeconds) speedup=\(fusedSeconds / max(0.000_001, reusedSeconds)) "
                    + "optimized_samples_per_second=\(Double(iterations * batchSize) / reusedSeconds) "
                    + "unique_vision_fraction=\(Double(uniqueSequenceCount) / Double(batchSize)) "
                    + "max_loss_delta=\(reusedLossDelta) max_parameter_delta=\(reusedParameterDelta)"
            )
            print(
                "TOTAL_TRAINING_BENCHMARK profile=\(profileName) original_seconds=\(baselineSeconds) "
                    + "optimized_seconds=\(reusedSeconds) speedup=\(baselineSeconds / max(0.000_001, reusedSeconds)) "
                    + "optimized_samples_per_second=\(Double(iterations * batchSize) / reusedSeconds)"
            )

            let validationModel = reusedTemporalModel
            validationModel.train(false)
            func legacyValidation() -> [MLXArray] {
                let inputs = baselineInputs()
                let logits = validationModel.callAsFunction(
                    currentImages: inputs[0], pastImages: inputs[1], pastControls: inputs[2]
                )
                return [
                    validationModel.loss(
                        logits: logits,
                        targets: inputs[3],
                        positiveWeights: positiveWeights,
                        previousTargets: inputs[4]
                    ),
                    validationModel.activatedPredictions(logits: logits),
                    inputs[3]
                ]
            }
            let reusedValidation = compile(inputs: [validationModel]) { arrays in
                let actions = arrays[4]
                let current = VisionPreprocessor.mlxTensor(arrays[0], spec: profile.preprocessing)
                let past = VisionPreprocessor.mlxPastFrameTensor(arrays[1], spec: pastSpec)
                let temporalFeatures = validationModel.temporalFeatures(
                    currentImages: current,
                    pastImages: past,
                    pastControls: arrays[2]
                ).take(arrays[3], axis: 0)
                let logits = validationModel.logits(temporalFeatures: temporalFeatures)
                let targets = actions[0..., 0, 0...]
                return [
                    validationModel.loss(
                        logits: logits,
                        targets: targets,
                        positiveWeights: positiveWeights,
                        previousTargets: actions[0..., 1, 0...]
                    ),
                    validationModel.activatedPredictions(logits: logits),
                    targets
                ]
            }
            func optimizedValidation() -> [MLXArray] { reusedValidation(reusedInputs()) }
            for _ in 0..<3 { MLX.eval(legacyValidation()); MLX.eval(optimizedValidation()) }
            let validationIterations = usesDefaultProfile ? 12 : 64
            var legacySeconds = 0.0, optimizedSeconds = 0.0
            var legacyResult: [MLXArray] = [], optimizedResult: [MLXArray] = []
            for iteration in 0..<validationIterations {
                let firstLegacy = iteration.isMultiple(of: 2)
                for legacy in [firstLegacy, !firstLegacy] {
                    let start = ContinuousClock.now
                    if legacy { legacyResult = legacyValidation(); MLX.eval(legacyResult) }
                    else { optimizedResult = optimizedValidation(); MLX.eval(optimizedResult) }
                    let seconds = start.duration(to: .now).benchmarkSeconds
                    if legacy { legacySeconds += seconds } else { optimizedSeconds += seconds }
                }
            }
            let validationLossDelta = abs(legacyResult[0].item(Float.self) - optimizedResult[0].item(Float.self))
            let validationPredictionDelta = maximumAbsoluteDifference(
                legacyResult[1].asArray(Float.self), optimizedResult[1].asArray(Float.self)
            )
            XCTAssertLessThanOrEqual(validationLossDelta, 0.000_01)
            XCTAssertLessThanOrEqual(validationPredictionDelta, 0.000_01)
            print(
                "VALIDATION_BENCHMARK profile=\(profileName) baseline_seconds=\(legacySeconds) "
                    + "optimized_seconds=\(optimizedSeconds) speedup=\(legacySeconds / max(0.000_001, optimizedSeconds)) "
                    + "max_loss_delta=\(validationLossDelta) max_prediction_delta=\(validationPredictionDelta)"
            )

            func compiledInference(accelerated: Bool) -> @Sendable ([MLXArray]) -> [MLXArray] {
                compile(inputs: [validationModel]) { arrays in
                    let current = VisionPreprocessor.mlxTensor(arrays[0], spec: profile.preprocessing)
                    let past = VisionPreprocessor.mlxPastFrameTensor(arrays[1], spec: pastSpec)
                    let currentFeatures = validationModel.visualActivations(
                        images: current,
                        acceleratedOperators: accelerated
                    ).last!
                    let logits = validationModel.logits(
                        currentVisualFeatures: currentFeatures,
                        pastImages: past,
                        pastControls: arrays[2]
                    )
                    return [validationModel.activatedPredictions(logits: logits)]
                }
            }
            let legacyInference = compiledInference(accelerated: false)
            let acceleratedInference = compiledInference(accelerated: true)
            for _ in 0..<3 { MLX.eval(legacyInference(fusedInputs())); MLX.eval(acceleratedInference(fusedInputs())) }
            let inferenceIterations = usesDefaultProfile ? 24 : 96
            var legacyInferenceSeconds = 0.0, acceleratedInferenceSeconds = 0.0
            var legacyInferenceResult = MLXArray(0), acceleratedInferenceResult = MLXArray(0)
            for iteration in 0..<inferenceIterations {
                let firstLegacy = iteration.isMultiple(of: 2)
                for legacy in [firstLegacy, !firstLegacy] {
                    let start = ContinuousClock.now
                    if legacy {
                        legacyInferenceResult = legacyInference(fusedInputs())[0]
                        MLX.eval(legacyInferenceResult)
                    } else {
                        acceleratedInferenceResult = acceleratedInference(fusedInputs())[0]
                        MLX.eval(acceleratedInferenceResult)
                    }
                    let seconds = start.duration(to: .now).benchmarkSeconds
                    if legacy { legacyInferenceSeconds += seconds } else { acceleratedInferenceSeconds += seconds }
                }
            }
            let inferencePredictionDelta = maximumAbsoluteDifference(
                legacyInferenceResult.asArray(Float.self), acceleratedInferenceResult.asArray(Float.self)
            )
            XCTAssertLessThanOrEqual(inferencePredictionDelta, 0.02)
            print(
                "INFERENCE_BENCHMARK profile=\(profileName) baseline_seconds=\(legacyInferenceSeconds) "
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
