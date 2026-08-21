import AppKit
import Foundation
import MLX
import MLXNN
import XCTest
@testable import AgentTrainer

/// Opt-in diagnostics for reproducing training behavior against a real local
/// profile without copying source recordings into the test bundle. The normal
/// suite skips this class; release investigations can point it at a profile and
/// retain the same deterministic split, packed cache, and model contract used
/// by production training.
final class RealDatasetDiagnosticsTests: XCTestCase {
    func testSavedBrainBinaryScoreCalibration() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let profileIDText = environment["AGENTTRAINER_REAL_PROFILE_ID"],
              let profileID = UUID(uuidString: profileIDText) else {
            throw XCTSkip("Set AGENTTRAINER_REAL_PROFILE_ID to inspect a saved local brain.")
        }
        let root = environment["AGENTTRAINER_REAL_DATA_ROOT"].map {
            URL(fileURLWithPath: $0, isDirectory: true)
        } ?? WorkspaceStore.defaultRoot
        let workspace = WorkspaceStore(root: root)
        try await workspace.prepare()

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let profileDirectory = root
            .appendingPathComponent("Profiles", isDirectory: true)
            .appendingPathComponent(profileID.uuidString, isDirectory: true)
        let profile = try decoder.decode(
            AIProfile.self,
            from: Data(contentsOf: profileDirectory.appendingPathComponent("profile.json"))
        )
        let versionID = environment["AGENTTRAINER_REAL_VERSION_ID"]
            .flatMap(UUID.init(uuidString:))
            ?? profile.activeVersionID
        guard let versionID else {
            XCTFail("The selected profile has no active saved brain.")
            return
        }
        let versionDirectory = profileDirectory
            .appendingPathComponent("Versions", isDirectory: true)
            .appendingPathComponent(versionID.uuidString, isDirectory: true)
        let manifest = try decoder.decode(
            ModelVersionManifest.self,
            from: Data(contentsOf: versionDirectory.appendingPathComponent("manifest.json"))
        )

        let selectedFolderIDs = Set(profile.effectiveFolderIDs)
        let recordings = await workspace.listRecordings().filter {
            profile.recordingIDs.contains($0.id)
                || $0.manifest.folderID.map(selectedFolderIDs.contains) == true
        }
        XCTAssertFalse(recordings.isEmpty)

        let dataset: CachedDataset
        if let cachePath = environment["AGENTTRAINER_REAL_CACHE_PATH"] {
            dataset = try CachedDataset(directory: URL(fileURLWithPath: cachePath, isDirectory: true))
        } else {
            let builder = DatasetCacheBuilder(workspace: workspace)
            let progressLock = NSLock()
            nonisolated(unsafe) var lastProgressBucket = -1
            dataset = try await builder.cache(for: profile, recordings: recordings) { progress, status in
                let bucket = Int((progress * 20).rounded(.down))
                let shouldPrint = progressLock.withLock { () -> Bool in
                    guard bucket != lastProgressBucket else { return false }
                    lastProgressBucket = bucket
                    return true
                }
                if shouldPrint { print("REAL DATA \(Int(progress * 100))% \(status)") }
            }
        }
        let engine = TrainingEngine()
        let split = engine.splitIndices(
            dataset: dataset,
            fraction: profile.training.validationSplit,
            seed: profile.training.seed,
            channels: profile.channels,
            restrictions: profile.effectiveRestrictions
        )
        let validation = dataset.representativeValidationIndices(
            from: split.validation,
            limit: min(8_192, split.validation.count),
            channels: profile.channels,
            restrictions: profile.effectiveRestrictions
        )
        XCTAssertFalse(validation.isEmpty)

        // Match production exactly: model initialization and every training
        // perturbation advance one profile-seeded MLX random-state stream.
        let trainingRandomState = MLXRandom.RandomState(
            seed: profile.training.seed
        )
        let model = withRandomState(trainingRandomState) {
            AgentPolicy(profile: profile)
        }
        let startsFresh = environment["AGENTTRAINER_REAL_START_FRESH"] == "1"
        let suppliedWeights = environment["AGENTTRAINER_REAL_WEIGHTS_PATH"]
            .map { URL(fileURLWithPath: $0) }
        let simulatesTrainingRestart = !startsFresh
            && environment["AGENTTRAINER_REAL_SIMULATE_TRAINING_RESTART"] == "1"
            && manifest.requiresFreshSupervisedTrainingRestart
        if let suppliedWeights {
            try model.loadWeights(from: suppliedWeights)
        } else if !startsFresh {
            if !simulatesTrainingRestart {
                try model.loadWeights(
                    from: versionDirectory.appendingPathComponent(
                        manifest.weightsFile
                    )
                )
            }
        }
        if let requestedSteps = environment["AGENTTRAINER_REAL_VISUAL_GROUNDING_STEPS"].flatMap(Int.init),
           requestedSteps > 0 {
            model.resetHistoricalControlShortcut()
            try Self.fineTuneVisualGrounding(
                model: model,
                dataset: dataset,
                profile: profile,
                trainingIndices: split.train,
                steps: requestedSteps,
                randomState: trainingRandomState
            )
            if let outputPath = environment["AGENTTRAINER_REAL_SAVE_WEIGHTS_PATH"] {
                try model.saveWeights(to: URL(fileURLWithPath: outputPath))
            }
        }
        model.train(false)
        let temporal = profile.training.effectiveTemporalVision
        let pastSpec = temporal.pastFrameSpec(from: profile.preprocessing)
        let keys = dataset.trainingActionStatistics(at: split.train).demonstratedKeyCodes
            .filter { $0 < 128 && !ActionLayout.commandOptionControlKeyCodeSet.contains($0) }
            .sorted()
        var zeroScores = Dictionary(uniqueKeysWithValues: keys.map { ($0, [Float]()) })
        var teacherScores = Dictionary(uniqueKeysWithValues: keys.map { ($0, [Float]()) })
        var labels = Dictionary(uniqueKeysWithValues: keys.map { ($0, [Bool]()) })
        var didPrintVisualDiagnostics = false

        let batchSize = max(1, min(128, profile.training.batchSize))
        for start in stride(from: 0, to: validation.count, by: batchSize) {
            let end = min(validation.count, start + batchSize)
            let rows = Array(validation[start..<end])
            let batch = dataset.trainingBatch(at: rows)
            let currentPacked = MLXArray(
                batch.packedCurrentObservations,
                [batch.count, profile.preprocessing.sampleByteCount],
                dtype: .uint8
            )
            let currentImages = VisionPreprocessor.mlxTensor(
                currentPacked,
                spec: profile.preprocessing,
                dtype: model.dtype
            )
            let pastImages: MLXArray
            let teacherControls: MLXArray
            if temporal.pastFrameCount > 0 {
                let pastPacked = MLXArray(
                    batch.packedPastObservations,
                    [batch.count, temporal.pastFrameCount, pastSpec.sampleByteCount],
                    dtype: .uint8
                )
                pastImages = VisionPreprocessor.mlxPastFrameTensor(
                    pastPacked,
                    spec: pastSpec,
                    dtype: model.dtype
                )
                teacherControls = MLXArray(
                    batch.pastControlRows,
                    [batch.count, temporal.pastFrameCount, ActionLayout.count],
                    type: Float.self
                )
            } else {
                pastImages = MLXArray.zeros(
                    [batch.count, 1, 1, 1, profile.preprocessing.channelCount],
                    dtype: .float32
                )
                teacherControls = MLXArray.zeros(
                    [batch.count, 1, ActionLayout.count],
                    dtype: .float32
                )
            }
            let actionRows = MLXArray(
                batch.actionRows,
                [batch.count, 2, ActionLayout.count],
                type: Float.self
            )
            let targets = actionRows[0..., 0, 0...]
            let zeroControls = MLXArray.zeros(like: teacherControls)
            let zeroPredictions = model.predictions(
                currentImages: currentImages,
                pastImages: pastImages,
                pastControls: zeroControls
            )
            let teacherPredictions = model.predictions(
                currentImages: currentImages,
                pastImages: pastImages,
                pastControls: teacherControls
            )
            if !didPrintVisualDiagnostics {
                didPrintVisualDiagnostics = true
                let stages = model.visualActivations(images: currentImages)
                let embedding = model.visualEmbedding(visualFeatures: stages.last!)
                let centered = embedding - embedding.mean(axis: 0, keepDims: true)
                let groundingPredictions = model.activatedPredictions(logits: model.logits(
                    temporalFeatures: model.visualGroundingFeatures(currentImages: currentImages)
                ))
                let diagnosticArrays = stages + [embedding, groundingPredictions]
                MLX.eval(diagnosticArrays)
                let stageVariances = stages.map { stage -> Float in
                    let centered = stage - stage.mean(axis: 0, keepDims: true)
                    let variance = square(centered).mean()
                    MLX.eval(variance)
                    return variance.item(Float.self)
                }
                let embeddingVariance = square(centered).mean()
                MLX.eval(embeddingVariance)
                print("REAL DATA visual stage batch variances \(stageVariances) embedding \(embeddingVariance.item(Float.self))")
                func printParameterSummary(_ name: String, _ value: MLXArray?) {
                    guard let value else { return }
                    let meanMagnitude = abs(value).mean()
                    let rootMeanSquare = sqrt(square(value).mean())
                    let minimum = value.min()
                    let maximum = value.max()
                    MLX.eval(meanMagnitude, rootMeanSquare, minimum, maximum)
                    print(
                        "REAL DATA parameter \(name) meanAbs=\(meanMagnitude.item(Float.self)) rms=\(rootMeanSquare.item(Float.self)) min=\(minimum.item(Float.self)) max=\(maximum.item(Float.self))"
                    )
                }
                printParameterSummary("visualProjection.weight", model.visualProjection.weight)
                printParameterSummary("visualProjection.bias", model.visualProjection.bias)
                printParameterSummary("visualNormalization.weight", model.visualNormalization.weight)
                printParameterSummary("visualNormalization.bias", model.visualNormalization.bias)
                printParameterSummary("fusion.0.weight", model.fusion.first?.weight)
                printParameterSummary("keyboardHead.weight", model.keyboardHead.weight)
                if let outputPath = environment["AGENTTRAINER_REAL_PREVIEW_DIR"] {
                    let outputDirectory = URL(fileURLWithPath: outputPath, isDirectory: true)
                    try FileManager.default.createDirectory(
                        at: outputDirectory,
                        withIntermediateDirectories: true
                    )
                    for row in 0..<min(12, rows.count) {
                        let packed = dataset.packedObservation(at: rows[row])
                        guard let image = VisionPreprocessor.previewImage(
                            packed,
                            spec: profile.preprocessing
                        ), let tiff = image.tiffRepresentation,
                              let bitmap = NSBitmapImageRep(data: tiff),
                              let png = bitmap.representation(using: .png, properties: [:]) else {
                            continue
                        }
                        try png.write(
                            to: outputDirectory.appendingPathComponent(
                                "row-\(rows[row])-\(row).png"
                            ),
                            options: .atomic
                        )
                    }
                }
            }
            MLX.eval(zeroPredictions, teacherPredictions, targets)
            let zeroValues = zeroPredictions.asArray(Float.self)
            let teacherValues = teacherPredictions.asArray(Float.self)
            let targetValues = targets.asArray(Float.self)
            for row in 0..<batch.count {
                for key in keys {
                    let output = ActionLayout.keyboard.lowerBound + Int(key)
                    let offset = row * ActionLayout.count + output
                    zeroScores[key, default: []].append(zeroValues[offset])
                    teacherScores[key, default: []].append(teacherValues[offset])
                    labels[key, default: []].append(targetValues[offset] >= 0.5)
                }
            }
        }

        print("REAL DATA profile=\(profile.name) rows=\(dataset.count) train=\(split.train.count) validation=\(validation.count) objective=\(manifest.trainingObjectiveSchema ?? 0) fresh=\(startsFresh) suppliedWeights=\(suppliedWeights != nil) simulatedTrainingRestart=\(simulatesTrainingRestart)")
        print("key positives zeroMeanNegative zeroMeanPositive zeroMax threshold MCC bestThreshold bestMCC defaultTP defaultFN bestTP bestFN bestFP bestTN teacherTP teacherFN")
        for key in keys {
            let keyLabels = labels[key] ?? []
            let zero = zeroScores[key] ?? []
            let teacher = teacherScores[key] ?? []
            let positives = zip(zero, keyLabels).compactMap { $0.1 ? $0.0 : nil }
            let negatives = zip(zero, keyLabels).compactMap { !$0.1 ? $0.0 : nil }
            let calibrated = BinaryDecisionCalibration.calibrate(
                zip(zero, keyLabels).map {
                    BinaryDecisionObservation(score: $0.0, target: $0.1)
                }
            )
            let zeroDefault = Self.confusion(scores: zero, labels: keyLabels, threshold: 0.5)
            let zeroBest = Self.confusion(
                scores: zero,
                labels: keyLabels,
                threshold: calibrated.bestObservedThreshold
            )
            let teacherDefault = Self.confusion(scores: teacher, labels: keyLabels, threshold: 0.5)
            print(String(
                format: "%d %d %.8f %.8f %.8f %.8f %.5f %.8f %.5f %d %d %d %d %d %d %d %d",
                Int(key), positives.count,
                Self.mean(negatives), Self.mean(positives), zero.max() ?? 0,
                calibrated.threshold, calibrated.matthewsCorrelation,
                calibrated.bestObservedThreshold,
                calibrated.bestObservedMatthewsCorrelation,
                zeroDefault.tp, zeroDefault.fn,
                zeroBest.tp, zeroBest.fn, zeroBest.fp, zeroBest.tn,
                teacherDefault.tp, teacherDefault.fn
            ))
        }
    }

    private struct Confusion {
        var tp = 0
        var fn = 0
        var fp = 0
        var tn = 0
    }

    private static func confusion(scores: [Float], labels: [Bool], threshold: Float) -> Confusion {
        zip(scores, labels).reduce(into: Confusion()) { result, pair in
            switch (pair.0 >= threshold, pair.1) {
            case (true, true): result.tp += 1
            case (false, true): result.fn += 1
            case (true, false): result.fp += 1
            case (false, false): result.tn += 1
            }
        }
    }

    private static func mean(_ values: [Float]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0) { $0 + Double($1) } / Double(values.count)
    }

    private static func fineTuneVisualGrounding(
        model: AgentPolicy,
        dataset: CachedDataset,
        profile: AIProfile,
        trainingIndices: [Int],
        steps: Int,
        randomState: MLXRandom.RandomState
    ) throws {
        let batchSize = max(1, min(128, profile.training.batchSize))
        guard trainingIndices.count >= batchSize else { return }
        let engine = TrainingEngine()
        let order = engine.trainingOrder(
            dataset: dataset,
            indices: trainingIndices,
            batchSize: batchSize,
            seed: profile.training.seed,
            profile: profile
        )
        let statistics = dataset.trainingActionStatistics(at: trainingIndices)
        let positiveWeightValues = dataset.positiveClassWeights(
            statistics: statistics,
            restrictions: profile.effectiveRestrictions
        )
        let headLossWeights = TrainingEngine.objectiveHeadLossWeights(
            coverage: statistics.coverage,
            channels: profile.channels
        )
        print("REAL DATA head weights \(headLossWeights) coverage \(statistics.coverage)")
        let optimizer = ResumableAdamW(
            learningRate: Float(profile.training.learningRate),
            weightDecay: Float(profile.training.weightDecay),
            warmupSteps: TrainingEngine.recommendedWarmupSteps(
                stepsPerEpoch: Int(ceil(Double(trainingIndices.count) / Double(batchSize)))
            ),
            schedule: profile.training.effectiveLearningRateSchedule,
            cycleSteps: TrainingEngine.recommendedCycleSteps(
                stepsPerEpoch: Int(ceil(Double(trainingIndices.count) / Double(batchSize))),
                cycleEpochs: profile.training.effectiveCosineCycleEpochs
            ),
            minimumLearningRateRatio: Float(
                profile.training.effectiveMinimumLearningRateRatio
            )
        )
        optimizer.initialize(model: model)
        let temporal = profile.training.effectiveTemporalVision
        let pastSpec = temporal.pastFrameSpec(from: profile.preprocessing)
        model.train(false)
        let groundingLearningRateMultiplier = TrainingEngine
            .visualGroundingLearningRateMultiplier(
                configuredLearningRate: profile.training.learningRate
            )
        let visuallyRankedBinaryIndices = ActionLayout.learnableBinaryIndices(
            channels: profile.channels,
            restrictions: profile.effectiveRestrictions
        ).filter { positiveWeightValues[$0] > 0 }
        let trainingStep = withRandomState(randomState) {
            compile(
                inputs: [model, optimizer, randomState],
                outputs: [model, optimizer, randomState]
            ) { arrays in
                let classWeights = MLXArray(
                    positiveWeightValues,
                    [ActionLayout.count]
                )
                let result = valueAndGrad(model: model) { candidate, inputs in
                    let embedding = candidate.currentVisualEmbedding(
                        currentImages: inputs[0]
                    )
                    let temporalFeatures = candidate.visualGroundingFeatures(
                        currentVisualEmbedding: embedding
                    )
                    let logits = candidate.logits(
                        temporalFeatures: temporalFeatures
                    )
                    return [candidate.supervisedTrainingObjective(
                        primaryLogits: logits,
                        visualLogits: nil,
                        currentVisualEmbedding: embedding,
                        targets: inputs[3],
                        positiveWeights: classWeights,
                        previousTargets: inputs[4],
                        headWeights: headLossWeights,
                        supportedBinaryIndices: visuallyRankedBinaryIndices
                    )]
                }(model, arrays)
                optimizer.update(
                    model: model,
                    gradients: result.1,
                    targetType: model.dtype,
                    maxGradientNorm: 1,
                    learningRateMultiplier: groundingLearningRateMultiplier
                )
                return [result.0[0]]
            }
        }
        let started = ContinuousClock.now
        for step in 0..<steps {
            let start = (step * batchSize) % (order.count - batchSize + 1)
            let batch = dataset.trainingBatch(
                at: Array(order[start..<(start + batchSize)])
            )
            let current = VisionPreprocessor.mlxTensor(
                MLXArray(
                    batch.packedCurrentObservations,
                    [batch.count, profile.preprocessing.sampleByteCount],
                    dtype: .uint8
                ),
                spec: profile.preprocessing,
                dtype: model.dtype
            )
            let past: MLXArray
            let controls: MLXArray
            if temporal.pastFrameCount > 0 {
                past = VisionPreprocessor.mlxPastFrameTensor(
                    MLXArray(
                        batch.packedPastObservations,
                        [batch.count, temporal.pastFrameCount, pastSpec.sampleByteCount],
                        dtype: .uint8
                    ),
                    spec: pastSpec,
                    dtype: model.dtype
                )
                controls = MLXArray(
                    batch.pastControlRows,
                    [batch.count, temporal.pastFrameCount, ActionLayout.count],
                    type: Float.self
                )
            } else {
                past = MLXArray.zeros(
                    [batch.count, 1, 1, 1, profile.preprocessing.channelCount],
                    dtype: model.dtype
                )
                controls = MLXArray.zeros(
                    [batch.count, 1, ActionLayout.count],
                    dtype: model.dtype
                )
            }
            let actionRows = MLXArray(
                batch.actionRows,
                [batch.count, 2, ActionLayout.count],
                type: Float.self
            )
            let result = withRandomState(randomState) {
                trainingStep([
                    current,
                    past,
                    controls,
                    actionRows[0..., 0, 0...],
                    actionRows[0..., 1, 0...]
                ])
            }
            MLX.eval(
                result,
                model.parameters(),
                optimizer.stateArrays(),
                randomState.innerState()
            )
            if step == 0 || (step + 1).isMultiple(of: 100) || step + 1 == steps {
                print("REAL DATA grounding step \(step + 1)/\(steps) loss \(result[0].item(Float.self))")
            }
        }
        print("REAL DATA grounding elapsed \(started.duration(to: .now))")
    }
}
