import CryptoKit
import Foundation
import MLX
import MLXNN
import QuartzCore

struct TrainingCompletion: Sendable {
    let profile: AIProfile
    let version: ModelVersionManifest
    let completed: Bool
}

enum TrainingRandomState {
    static func save(_ randomState: MLXRandom.RandomState = MLXRandom.globalState, to url: URL) throws {
        guard let state = randomState.innerState().first else {
            throw AgentTrainerError.model("MLX random state is unavailable, so an exact checkpoint cannot be saved.")
        }
        MLX.eval(state); try MLX.save(arrays: ["state": state], url: url)
    }

    static func load(_ randomState: MLXRandom.RandomState = MLXRandom.globalState, from url: URL) throws {
        guard let restored = try MLX.loadArrays(url: url)["state"] else {
            throw AgentTrainerError.model("The checkpoint does not contain its saved MLX random state.")
        }
        guard let current = randomState.innerState().first else {
            throw AgentTrainerError.model("MLX random state is unavailable, so the checkpoint cannot resume exactly.")
        }
        current._updateInternal(restored)
        MLX.eval(current)
    }
}

private struct TrainingPaused: Error {
    let completion: TrainingCompletion
}

private enum TrainingSamplingContract {
    /// Versioned independently from the dataset bytes so an older checkpoint
    /// paused midway through an epoch can finish with its original exact order.
    static let salienceBalanced = 1
    /// Keeps every label exactly once while colocating rows that share a causal
    /// perception pair. The visual encoder evaluates each unique pair once.
    static let groupedVision = 2
    /// Keeps nearby causal windows together, then randomizes the order of those
    /// local batches. Overlapping windows can share individual past-frame CNN
    /// embeddings and mapped dataset reads retain useful page locality.
    static let localityGroupedVision = 3
}

private enum TrainingValidationContract {
    /// Version 5 retains the zero-history stress test and selects a supported
    /// per-output execution boundary from its raw scores. Validation, saved
    /// brain publication, recurrent feedback, and physical execution therefore
    /// measure the same policy instead of disagreeing at a universal 0.50.
    static let capabilityAware = 5
}

final class TrainingEngine: @unchecked Sendable {
    private static let minimumVisionReuseRatio = 1.20
    private static let metricsPublishInterval = 0.5
    private static let publishedLossHistoryLimit = 2_048
    private static let repeatedPerformanceWarningInterval = 5 * 60.0
    typealias MetricsHandler = @Sendable (TrainingMetrics, String) -> Void
    typealias CompletionHandler = @Sendable (Result<TrainingCompletion, Error>) -> Void

    private let lock = NSLock()
    private var task: Task<Void, Never>?
    private var pauseRequested = false
    private var stopRequested = false

    var isRunning: Bool { lock.withLock { task != nil } }

    func start(profile: AIProfile, recordings: [RecordingItem], runSettings: TrainingRunSettings, metrics: @escaping MetricsHandler, completion: @escaping CompletionHandler) {
        lock.withLock {
            guard task == nil else { return }
            pauseRequested = false
            stopRequested = false
            task = Task.detached(priority: .userInitiated) { [weak self] in
                guard let self else { return }
                let outcome: Result<TrainingCompletion, Error>
                do {
                    let randomState = MLXRandom.RandomState(seed: profile.training.seed)
                    let result = try await Device.withDefaultDevice(.gpu) {
                        // A dedicated GPU stream lets Metal schedule this long-running
                        // training graph independently from live inference and UI-side
                        // MLX work while preserving the exact same operations.
                        try await Stream.withNewDefaultStream(device: .gpu) {
                            try await withRandomState(randomState) {
                                try await self.train(profile: profile, recordings: recordings, runSettings: runSettings, randomState: randomState, metrics: metrics)
                            }
                        }
                    }
                    outcome = .success(result)
                } catch let paused as TrainingPaused {
                    outcome = .success(paused.completion)
                } catch is CancellationError {
                    outcome = .failure(AgentTrainerError.model("Training stopped."))
                } catch {
                    outcome = .failure(error)
                }
                // The optimizer/model stack is out of scope here. Drop unused
                // Metal buffers before publishing Idle so repeated training
                // sessions cannot accumulate allocator cache until app quit.
                MLXMemoryLifecycle.reclaimCaches(after: "training")
                self.lock.withLock { self.task = nil }
                completion(outcome)
            }
        }
    }

    func pauseAndSave() { lock.withLock { pauseRequested = true } }
    func stop() { lock.withLock { stopRequested = true; task?.cancel() } }

    private func train(profile: AIProfile, recordings: [RecordingItem], runSettings: TrainingRunSettings, randomState: MLXRandom.RandomState, metrics: @escaping MetricsHandler) async throws -> TrainingCompletion {
        MLXMemoryLifecycle.configure()

        let dataset = try await DatasetCacheBuilder.shared.cache(for: profile, recordings: recordings) { progress, status in
            var value = TrainingMetrics(); value.totalEpochs = profile.training.epochs
            // Dataset event counts are not optimizer steps. Keep the step total
            // unknown until the train/validation split and batch count exist.
            value.totalSteps = 0
            metrics(value, "\(status) • \(Int((progress * 100).rounded()))%")
        }
        guard dataset.count > 0 else { throw AgentTrainerError.noData }
        let packedRecordingIDs = Set(dataset.manifest.segments.map(\.recordingID))
        let trainingRecordings = recordings.filter {
            packedRecordingIDs.contains($0.id)
        }
        guard !trainingRecordings.isEmpty else { throw AgentTrainerError.noData }
        if trainingRecordings.count != recordings.count {
            AppLog.write(
                .warning,
                category: "Training",
                "Training set excludes unusable recordings",
                details: "Using \(trainingRecordings.count) of \(recordings.count) selected recordings. Check the earlier packing diagnostic for exact package names."
            )
        }
        let inputBufferPool = try MetalArrayBufferPool()
        let split = splitIndices(
            dataset: dataset,
            fraction: profile.training.validationSplit,
            seed: profile.training.seed,
            channels: profile.channels,
            restrictions: profile.effectiveRestrictions
        )
        guard !split.train.isEmpty else { throw AgentTrainerError.noData }
        let batchSize = max(1, profile.training.batchSize)
        let stepsPerEpoch = Int(ceil(Double(split.train.count) / Double(batchSize)))
        let (observationReuseRatio, groupedTrainingObservations): (Double, [[Int]]) = {
            let groups = dataset.observationGroups(at: split.train)
            let probeRows = Array(localityGroupedVisionTrainingOrder(
                groups: groups,
                batchSize: batchSize,
                seed: profile.training.seed,
                salientIndices: []
            ).prefix(batchSize))
            let probe = dataset.visionBatchPlan(at: probeRows)
            let pastSpec = profile.training.effectiveTemporalVision.pastFrameSpec(
                from: profile.preprocessing
            )
            let currentPixels = profile.preprocessing.width * profile.preprocessing.height
            let pastPixels = pastSpec.width * pastSpec.height
            let encoderReuse = probe.encoderWorkReuseRatio(
                currentPixels: currentPixels,
                pastPixels: pastPixels
            )
            // Integer gathers have a small fixed cost. Require enough measured
            // CNN work reuse in a locality-shaped batch to recover it even on a
            // compact network.
            guard encoderReuse >= Self.minimumVisionReuseRatio else {
                return (encoderReuse, [])
            }
            return (encoderReuse, groups)
        }()
        let reusesVisionFeatures = !groupedTrainingObservations.isEmpty
        let desiredSamplingStrategy = reusesVisionFeatures
            ? TrainingSamplingContract.localityGroupedVision
            : TrainingSamplingContract.salienceBalanced
        // Salience depends only on immutable cached targets and profile output
        // policy. Scan it once per run rather than repeating the CPU pass at
        // every epoch boundary.
        let salientTrainingIndices: Set<Int> = {
            let detected = dataset.salientTrainingIndices(
                at: split.train,
                channels: profile.channels,
                restrictions: profile.effectiveRestrictions
            )
            // When every row is active there is nothing to rebalance; release
            // the potentially large membership set and use the normal shuffle.
            return detected.count == split.train.count ? [] : detected
        }()
        let trainingActionStatistics = dataset.trainingActionStatistics(at: split.train)
        let trainingDataCoverage = trainingActionStatistics.coverage
        let objectiveHeadLossWeights = Self.objectiveHeadLossWeights(
            coverage: trainingDataCoverage,
            channels: profile.channels
        )
        let visionWarnings = profile.preprocessing.trainingQualityWarnings
        if !visionWarnings.isEmpty {
            AppLog.write(
                .warning,
                category: "Training",
                "Model vision may be too lossy for reliable control timing",
                details: visionWarnings.joined(separator: " ")
            )
        }
        let coverageWarnings = trainingDataCoverage.warnings(for: profile.channels)
        if !coverageWarnings.isEmpty {
            AppLog.write(
                .warning,
                category: "Training",
                "Some enabled controls have insufficient demonstrations",
                details: coverageWarnings.joined(separator: " ")
            )
        }
        let positiveClassWeightValues = dataset.positiveClassWeights(
            statistics: trainingActionStatistics,
            restrictions: profile.effectiveRestrictions
        )
        let visuallyRankedBinaryIndices = ActionLayout.learnableBinaryIndices(
            channels: profile.channels,
            restrictions: profile.effectiveRestrictions
        ).filter { positiveClassWeightValues[$0] > 0 }
        let validationSampleLimit = Self.recommendedValidationSampleLimit(
            total: split.validation.count,
            batchSize: profile.training.batchSize,
            segmentCount: dataset.segmentCount(at: split.validation)
        )
        let representativeValidationIndices = dataset.representativeValidationIndices(
            from: split.validation,
            limit: validationSampleLimit,
            channels: profile.channels,
            restrictions: profile.effectiveRestrictions
        )
        let (reusesValidationVisionFeatures, validationEvaluationIndices): (Bool, [Int]) = {
            let plan = dataset.visionBatchPlan(at: representativeValidationIndices)
            guard plan.reuseRatio >= Self.minimumVisionReuseRatio else {
                return (false, representativeValidationIndices)
            }
            return (
                true,
                dataset.observationGroups(
                    at: representativeValidationIndices,
                    using: plan
                ).flatMap { $0 }
            )
        }()

        let model = AgentPolicy(profile: profile)
        model.train(true)
        let optimizer = ResumableAdamW(
            learningRate: Float(profile.training.learningRate),
            weightDecay: Float(profile.training.weightDecay),
            warmupSteps: Self.recommendedWarmupSteps(stepsPerEpoch: stepsPerEpoch),
            schedule: profile.training.effectiveLearningRateSchedule,
            cycleSteps: Self.recommendedCycleSteps(
                stepsPerEpoch: stepsPerEpoch,
                cycleEpochs: profile.training.effectiveCosineCycleEpochs
            ),
            minimumLearningRateRatio: Float(profile.training.effectiveMinimumLearningRateRatio)
        )
        optimizer.initialize(model: model)
        let signature = try profileSignature(
            profile,
            recordings: trainingRecordings
        )
        let inputSummaries = try trainingRecordings.map { recording in
            let url = recording.directory.appendingPathComponent(recording.manifest.eventFile)
            return (recording, try InputEventReader.summarize(url: url, previewLimit: 0, globalRect: recording.manifest.globalRect.cgRect))
        }
        let demonstratedKeys = trainingActionStatistics.demonstratedKeyCodes
        let mouseDurations = inputSummaries.reduce(into: (camera: 0.0, cursor: 0.0)) { result, value in
            guard value.1.mouse.moveEventCount > 0 else { return }
            let recording = value.0.manifest
            let duration = max(0, min(recording.duration, recording.trimEnd ?? recording.duration) - max(0, recording.trimStart))
            if value.1.mouse.isGameCamera { result.camera += duration } else { result.cursor += duration }
        }
        let recommendedMouseMode: MouseControlMode = mouseDurations.camera > mouseDurations.cursor ? .relative : .absolute
        let recordingOrder = trainingRecordings.map(\.id)
        var state = CheckpointState(
            profileSignature: signature,
            epoch: 0,
            batchOffset: 0,
            globalStep: 0,
            elapsed: 0,
            lossHistory: [],
            validationHistory: [],
            demonstratedKeyCodes: demonstratedKeys,
            experienceSeconds: 0,
            recordingOrder: recordingOrder,
            samplingStrategy: desiredSamplingStrategy,
            validationStrategy: TrainingValidationContract.capabilityAware
        )
        let restore = try await restoreCheckpointIfPresent(
            profile: profile,
            expectedSignature: signature,
            expectedRecordingOrder: recordingOrder,
            legacyValidationNeedsRefresh: dataset.manifest.segments.count == 1,
            model: model,
            optimizer: optimizer,
            randomState: randomState,
            state: &state
        )
        if state.visualGroundingComplete == nil {
            if profile.training.effectiveTemporalVision.pastFrameCount > 0 {
                // This is an objective-v4 start, not an exact mid-epoch resume.
                // Remove every established or random recurrent contribution
                // before the visual-only curriculum.
                model.resetHistoricalControlShortcut()
            }
            state.visualGroundingComplete = false
        }
        let validationStep = compile(inputs: [model]) { (arrays: [MLXArray]) -> [MLXArray] in
            // Keep packed-frame expansion, the shared forward pass, loss, and activation
            // inside one Metal graph. The class weights are immutable for this
            // run and become a trace-time constant.
            let classWeights = MLXArray(positiveClassWeightValues, [ActionLayout.count])
            let batch = Self.expandBatch(
                arrays,
                profile: profile,
                dtype: model.dtype,
                reusesVisionFeatures: reusesValidationVisionFeatures
            )
            let temporalFeatures = model.temporalFeatures(
                currentImages: batch.currentImages,
                pastImages: batch.pastImages,
                // Runtime begins with no predicted actions and can remain at
                // that fixed point indefinitely. Stress best-brain candidates
                // without teacher-forced action history so weakness is visible.
                pastControls: Self.runtimeStartupControls(like: batch.pastControls),
                visionToPast: batch.visionToPast,
                pastFrameMask: batch.pastFrameMask
            )
            let uniqueLogits = model.logits(temporalFeatures: temporalFeatures)
            let logits: MLXArray
            if let sampleToVision = batch.sampleToVision,
               sampleToVision.dim(0) != uniqueLogits.dim(0) {
                logits = uniqueLogits.take(sampleToVision, axis: 0)
            } else {
                logits = uniqueLogits
            }
            return [
                model.loss(
                    logits: logits,
                    targets: batch.targets,
                    positiveWeights: classWeights,
                    previousTargets: batch.previousTargets,
                    headWeights: objectiveHeadLossWeights
                ),
                model.activatedPredictions(logits: logits),
                batch.targets
            ]
        }
        // The raw event stream is authoritative, including taps shorter than one
        // action interval. Refresh after restoring an older checkpoint.
        state.demonstratedKeyCodes = demonstratedKeys
        state.trainingDataCoverage = trainingDataCoverage
        // Legacy checkpoints did not persist sample-to-recording order. Preserve
        // their current epoch order, then make every subsequent save enforce it.
        if state.recordingOrder == nil { state.recordingOrder = recordingOrder }
        if state.batchOffset == 0,
           state.samplingStrategy != desiredSamplingStrategy {
            state.samplingStrategy = desiredSamplingStrategy
        }
        state.recommendedMouseMode = recommendedMouseMode
        let cursorDurations = trainingRecordings.reduce(into: (shown: 0.0, total: 0.0)) { result, recording in
            let start = max(0, min(recording.manifest.duration, recording.manifest.trimStart))
            let end = max(start, min(recording.manifest.duration, recording.manifest.trimEnd ?? recording.manifest.duration))
            let duration = end - start
            result.total += duration
            if recording.manifest.capture.showsCursor { result.shown += duration }
        }
        state.trainingShowsCursor = cursorDurations.total > 0
            ? cursorDurations.shown >= cursorDurations.total / 2
            : trainingRecordings.filter { $0.manifest.capture.showsCursor }.count * 2 >= trainingRecordings.count

        if split.validation.isEmpty {
            // A checkpoint from an older split may contain a score and best
            // weights even when the current disjoint split cannot hold out any
            // honest samples. Never publish that stale score as if it applied.
            state.validationHistory.removeAll(keepingCapacity: false)
            state.bestValidationLoss = nil
            state.bestGlobalStep = nil
            state.bestEpoch = nil
            state.bestTrainingLoss = nil
            state.bestElapsed = nil
            state.bestExperienceSeconds = nil
            state.currentValidationReport = nil
            state.currentValidationGlobalStep = nil
            state.bestValidationReport = nil
            state.currentBinaryDecisionThresholds = nil
            state.bestBinaryDecisionThresholds = nil
            state.schedulerBestMetric = nil
            state.schedulerPlateauEpochs = 0
        }

        // A warm-started brain's saved validation number may belong to a
        // different recording set, split, target contract, or loss definition.
        // Re-score those exact selected weights on this run's held-out examples
        // before comparing any fine-tuned epoch against them.
        if restore.captureValidationBaseline, !split.validation.isEmpty {
            let baseline = try evaluate(
                model: model,
                dataset: dataset,
                indices: validationEvaluationIndices,
                profile: profile,
                positiveClassWeightValues: positiveClassWeightValues,
                inputBufferPool: inputBufferPool,
                validationStep: validationStep,
                reusesVisionFeatures: reusesValidationVisionFeatures
            )
            let baselineValidationLoss = baseline.loss
            guard baselineValidationLoss.isFinite else {
                throw AgentTrainerError.model("The selected brain produced an invalid validation baseline on the current recordings.")
            }
            state.validationHistory = [baselineValidationLoss]
            state.currentValidationReport = baseline.report
            state.currentValidationGlobalStep = state.globalStep
            state.currentBinaryDecisionThresholds = baseline.binaryDecisionThresholds
            state.schedulerBestMetric = baselineValidationLoss
            state.schedulerPlateauEpochs = 0
            state.bestValidationLoss = baselineValidationLoss
            state.bestGlobalStep = state.globalStep
            state.bestEpoch = state.batchOffset > 0 ? state.epoch + 1 : state.epoch
            state.bestTrainingLoss = state.lossHistory.last
            state.bestElapsed = state.elapsed
            state.bestExperienceSeconds = state.experienceSeconds
            state.bestValidationReport = baseline.report
            state.bestBinaryDecisionThresholds = baseline.binaryDecisionThresholds
        }

        let visualGroundingLearningRateMultiplier = Self
            .visualGroundingLearningRateMultiplier(
                configuredLearningRate: profile.training.learningRate
            )

        func compiledTrainingStep(
            forceVisualGrounding: Bool
        ) -> @Sendable ([MLXArray]) -> [MLXArray] {
            compile(
                inputs: [model, optimizer, randomState],
                outputs: [model, optimizer, randomState]
            ) { (arrays: [MLXArray]) -> [MLXArray] in
                // Capture Sendable Swift values and materialize constants while
                // MLX traces the graph. MLXArray itself is non-Sendable.
                let classWeights = MLXArray(
                    positiveClassWeightValues,
                    [ActionLayout.count]
                )
                let batch = Self.expandBatch(
                    arrays,
                    profile: profile,
                    dtype: model.dtype,
                    reusesVisionFeatures: reusesVisionFeatures
                )
                let gradientInputs = [
                    batch.currentImages,
                    batch.pastImages,
                    batch.pastControls,
                    batch.targets,
                    batch.previousTargets
                ]
                // The row maps are integer metadata, not differentiable inputs.
                let sampleToVision = batch.sampleToVision
                let visionToPast = batch.visionToPast
                let pastFrameMask = batch.pastFrameMask
                let result = valueAndGrad(model: model) { model, arrays in
                    let currentVisualEmbedding = model.currentVisualEmbedding(
                        currentImages: arrays[0]
                    )
                    let temporalFeatures = forceVisualGrounding
                        ? model.visualGroundingFeatures(
                            currentVisualEmbedding: currentVisualEmbedding,
                            sampleToVision: sampleToVision
                        )
                        : model.temporalFeatures(
                            currentVisualEmbedding: currentVisualEmbedding,
                            pastImages: arrays[1],
                            pastControls: arrays[2],
                            visionToPast: visionToPast,
                            sampleToVision: sampleToVision,
                            pastFrameMask: pastFrameMask
                        )
                    let logits = model.logits(temporalFeatures: temporalFeatures)
                    let visualLogits: MLXArray?
                    if forceVisualGrounding
                        || profile.training.effectiveTemporalVision.pastFrameCount == 0 {
                        visualLogits = nil
                    } else {
                        visualLogits = model.logits(
                            temporalFeatures: model.visualGroundingFeatures(
                                currentVisualEmbedding: currentVisualEmbedding,
                                sampleToVision: sampleToVision
                            )
                        )
                    }
                    return [model.supervisedTrainingObjective(
                        primaryLogits: logits,
                        visualLogits: visualLogits,
                        currentVisualEmbedding: currentVisualEmbedding,
                        targets: arrays[3],
                        positiveWeights: classWeights,
                        previousTargets: arrays[4],
                        headWeights: objectiveHeadLossWeights,
                        supportedBinaryIndices: visuallyRankedBinaryIndices
                    )]
                }(model, gradientInputs)
                optimizer.update(
                    model: model,
                    gradients: result.1,
                    targetType: model.dtype,
                    maxGradientNorm: 1,
                    learningRateMultiplier: forceVisualGrounding
                        ? visualGroundingLearningRateMultiplier
                        : 1
                )
                return [result.0[0]]
            }
        }
        let trainingStep = compiledTrainingStep(forceVisualGrounding: false)
        var visualGroundingStep: (@Sendable ([MLXArray]) -> [MLXArray])? = state.visualGroundingComplete == true
            ? nil
            : compiledTrainingStep(forceVisualGrounding: true)
        if visualGroundingStep != nil {
            // The first invocation traces a deterministic graph: no pixel
            // augmentation, feature dropout, or label smoothing. Once a real
            // visual policy exists, later epochs restore every configured
            // regularizer around the ordinary temporal objective.
            model.train(false)
        }
        let started = ContinuousClock.now
        let baseElapsed = state.elapsed
        if state.experienceSeconds == nil {
            // Old checkpoints did not persist the exact final-batch sizes. Step
            // count is the most stable approximation because it remains valid
            // if the user later changes recording/folder selection.
            let completedSamples = Double(max(0, state.globalStep)) * Double(batchSize)
            state.experienceSeconds = completedSamples / max(0.0001, dataset.manifest.actionFPS)
        }
        let targetEpoch = TrainingContinuationPlan.targetEpoch(
            completedEpoch: state.epoch,
            batchOffset: state.batchOffset,
            savedTarget: state.epochGoal,
            configuredIncrement: profile.training.epochs
        )
        state.epochGoal = targetEpoch
        let remainingEpochSteps = TrainingContinuationPlan.remainingSteps(
            completedEpoch: state.epoch,
            batchOffset: state.batchOffset,
            targetEpoch: targetEpoch,
            samplesPerEpoch: split.train.count,
            batchSize: batchSize
        )
        let configuredMaximum = max(0, runSettings.maximumSteps)
        let epochStepTarget = state.globalStep.addingReportingOverflow(remainingEpochSteps).overflow ? Int.max : state.globalStep + remainingEpochSteps
        let runStepTarget: Int
        if configuredMaximum > 0 {
            let addition = state.globalStep.addingReportingOverflow(configuredMaximum)
            runStepTarget = addition.overflow ? Int.max : addition.partialValue
        } else {
            runStepTarget = Int.max
        }
        let totalSteps = min(epochStepTarget, runStepTarget)
        var latestSnapshot: TrainingCompletion?
        let autosaveInterval = max(1, runSettings.autosaveSteps)
        var nextAutosaveStep = saturatingAdd(state.globalStep, autosaveInterval)
        var autosavesPublished = 0

        // Activating a weights-only best brain intentionally discards an
        // unrelated newer optimizer checkpoint. Preserve that selected brain
        // as the validation baseline before fine-tuning, while initializing a
        // fresh exact-resume checkpoint from it. A first worse epoch can no
        // longer replace the brain the user explicitly chose.
        if restore.captureValidationBaseline,
           !split.validation.isEmpty,
           state.bestGlobalStep != nil {
            try await saveCheckpoint(profile: profile, model: model, optimizer: optimizer, randomState: randomState, state: state, captureBest: true)
        }

        let visualGroundingStatus: String
        if visualGroundingStep == nil {
            visualGroundingStatus = ""
        } else if profile.training.effectiveTemporalVision.pastFrameCount > 0 {
            visualGroundingStatus = " • deterministic current-pixel grounding at a capped rate with exact zero recurrent state"
        } else {
            visualGroundingStatus = " • deterministic current-pixel grounding at a capped rate"
        }
        let initialMemory = Memory.snapshot()
        metrics(TrainingMetrics(
            epoch: min(targetEpoch, state.epoch + (state.batchOffset > 0 ? 1 : 0)),
            totalEpochs: targetEpoch,
            batch: state.batchOffset / batchSize,
            totalBatches: stepsPerEpoch,
            globalStep: state.globalStep,
            totalSteps: totalSteps,
            nextAutosaveStep: nextAutosaveStep,
            autosavesPublished: autosavesPublished,
            trainingLoss: state.lossHistory.last ?? 0,
            epochTrainingLoss: state.epochLossHistory?.last,
            validationLoss: state.validationHistory.last,
            validationReport: state.currentValidationReport,
            trainingDataCoverage: trainingDataCoverage,
            effectiveLearningRate: Double(
                optimizer.effectiveLearningRate()
                    * (visualGroundingStep == nil
                        ? 1
                        : visualGroundingLearningRateMultiplier)
            ),
            learningRateScale: Double(optimizer.learningRateScale),
            samplesPerSecond: 0,
            elapsed: state.elapsed,
            experienceElapsed: state.experienceSeconds ?? 0,
            lossHistory: Array(state.lossHistory.suffix(Self.publishedLossHistoryLimit)),
            epochLossHistory: Array((state.epochLossHistory ?? []).suffix(1_024)),
            validationHistory: Array(state.validationHistory.suffix(1_024)),
            learningRateHistory: Array((state.learningRateHistory ?? []).suffix(1_024)),
            mlxActiveMemory: initialMemory.activeMemory,
            mlxCacheMemory: initialMemory.cacheMemory,
            mlxPeakMemory: initialMemory.peakMemory
        ), "\(restore.status)\(visualGroundingStatus) • continuing to epoch \(targetEpoch)")

        var lastMetricsPublish = CACurrentMediaTime() - 1
        var performance = TrainingPerformanceWindow()
        var lastLoggedPerformanceLimit: String?
        var lastPerformanceWarningTime = -Double.greatestFiniteMagnitude
        let optimizerPipelineDepth = Self.recommendedOptimizerPipelineDepth(
            profile: profile,
            physicalMemory: ProcessInfo.processInfo.physicalMemory
        )

        trainingLoop: for epoch in state.epoch..<targetEpoch {
            let epochSeed = profile.training.seed &+ UInt64(epoch) &* 0x9E3779B97F4A7C15
            let order: [Int]
            if state.samplingStrategy == TrainingSamplingContract.localityGroupedVision,
               reusesVisionFeatures {
                order = localityGroupedVisionTrainingOrder(
                    groups: groupedTrainingObservations,
                    batchSize: batchSize,
                    seed: epochSeed,
                    salientIndices: salientTrainingIndices
                )
            } else if state.samplingStrategy == TrainingSamplingContract.groupedVision,
               reusesVisionFeatures {
                order = groupedVisionTrainingOrder(
                    groups: groupedTrainingObservations,
                    batchSize: batchSize,
                    seed: epochSeed,
                    salientIndices: salientTrainingIndices
                )
            } else if state.samplingStrategy == TrainingSamplingContract.salienceBalanced
                        || state.samplingStrategy == TrainingSamplingContract.groupedVision
                        || state.samplingStrategy == TrainingSamplingContract.localityGroupedVision {
                order = trainingOrder(
                    dataset: dataset,
                    indices: split.train,
                    batchSize: batchSize,
                    seed: epochSeed,
                    profile: profile,
                    precomputedSalientIndices: salientTrainingIndices
                )
            } else {
                order = shuffled(split.train, seed: epochSeed)
            }
            var offset = epoch == state.epoch ? state.batchOffset : 0
            var epochWeightedLoss = offset > 0 ? state.currentEpochWeightedLoss ?? 0 : 0
            var epochSampleCount = offset > 0 ? state.currentEpochSampleCount ?? 0 : 0
            while offset < order.count {
                try Task.checkCancellation()
                if lock.withLock({ stopRequested }) { throw CancellationError() }
                let stepsUntilAutosave = max(1, nextAutosaveStep - state.globalStep)
                let stepsUntilRunLimit = configuredMaximum > 0
                    ? max(1, runStepTarget - state.globalStep)
                    : Int.max
                let queueLimit = min(
                    optimizerPipelineDepth,
                    stepsUntilAutosave,
                    stepsUntilRunLimit
                )
                var scheduled: [ScheduledTrainingStep] = []
                scheduled.reserveCapacity(queueLimit)
                var schedulingOffset = offset
                let queueStarted = CACurrentMediaTime()
                while schedulingOffset < order.count, scheduled.count < queueLimit {
                    try Task.checkCancellation()
                    if lock.withLock({ stopRequested }) { throw CancellationError() }
                    let end = min(order.count, schedulingOffset + batchSize)
                    let batch = Array(order[schedulingOffset..<end])
                    let preparationStarted = CACurrentMediaTime()
                    let arrays = try makeBatch(
                        dataset: dataset,
                        indices: batch,
                        profile: profile,
                        inputBufferPool: inputBufferPool,
                        reusesVisionFeatures: reusesVisionFeatures
                    )
                    let preparationSeconds = CACurrentMediaTime() - preparationStarted
                    let lossArray = (visualGroundingStep ?? trainingStep)(arrays)[0]
                    // Each compiled invocation observes the lazy state produced
                    // by the preceding invocation. Start it immediately so CPU
                    // mapped-data gathering for the next batch overlaps Metal,
                    // then synchronize the bounded ordered queue once.
                    MLX.asyncEval(
                        lossArray,
                        model.parameters(),
                        optimizer.stateArrays(),
                        randomState.innerState()
                    )
                    scheduled.append(ScheduledTrainingStep(
                        loss: lossArray,
                        sampleCount: batch.count,
                        endOffset: end,
                        preparationSeconds: preparationSeconds
                    ))
                    schedulingOffset = end
                    if lock.withLock({ pauseRequested }) { break }
                }
                MLX.eval(
                    scheduled.map(\.loss),
                    model.parameters(),
                    optimizer.stateArrays(),
                    randomState.innerState()
                )
                let losses = scheduled.map { Double($0.loss.item(Float.self)) }
                guard losses.allSatisfy(\.isFinite) else {
                    throw AgentTrainerError.model("Training became numerically unstable before this step could be saved. Lower the learning rate or reset this brain's learning state.")
                }
                let secondsPerStep = max(
                    0.000_001,
                    (CACurrentMediaTime() - queueStarted) / Double(max(1, scheduled.count))
                )
                for (scheduledStep, loss) in zip(scheduled, losses) {
                    performance.record(
                        samples: scheduledStep.sampleCount,
                        stepSeconds: secondsPerStep,
                        preparationSeconds: scheduledStep.preparationSeconds
                    )
                    guard state.globalStep < Int.max else {
                        throw AgentTrainerError.model("The restored optimizer step counter is invalid and cannot be advanced safely.")
                    }
                    state.globalStep += 1
                    state.experienceSeconds = (state.experienceSeconds ?? 0)
                        + Double(scheduledStep.sampleCount) / max(0.0001, dataset.manifest.actionFPS)
                    offset = scheduledStep.endOffset
                    state.epoch = epoch
                    state.batchOffset = offset
                    state.lossHistory.append(loss)
                    if state.lossHistory.count > 8_192 { state.lossHistory.removeFirst(4_096) }
                    epochWeightedLoss += loss * Double(scheduledStep.sampleCount)
                    epochSampleCount += scheduledStep.sampleCount
                }
                state.currentEpochWeightedLoss = epochWeightedLoss
                state.currentEpochSampleCount = epochSampleCount
                let loss = losses.last ?? 0

                let now = CACurrentMediaTime()
                if now - lastMetricsPublish >= Self.metricsPublishInterval || offset == order.count {
                    lastMetricsPublish = now
                    let elapsed = baseElapsed + started.duration(to: .now).seconds
                    let memory = Memory.snapshot()
                    let thermalState = TrainingThermalState.current
                    // Publish detached suffixes so Swift array copy-on-write does
                    // not force the optimizer loop to copy its full checkpoint
                    // history on the next append while SwiftUI still retains it.
                    let report = TrainingMetrics(
                        epoch: epoch + 1,
                        totalEpochs: targetEpoch,
                        batch: Int(ceil(Double(offset) / Double(batchSize))),
                        totalBatches: stepsPerEpoch,
                        globalStep: state.globalStep,
                        totalSteps: totalSteps,
                        nextAutosaveStep: nextAutosaveStep,
                        autosavesPublished: autosavesPublished,
                        trainingLoss: loss,
                        epochTrainingLoss: epochSampleCount > 0 ? epochWeightedLoss / Double(epochSampleCount) : nil,
                        validationLoss: state.validationHistory.last,
                        validationReport: state.currentValidationReport,
                        trainingDataCoverage: trainingDataCoverage,
                        effectiveLearningRate: Double(
                            optimizer.effectiveLearningRate()
                                * (visualGroundingStep == nil
                                    ? 1
                                    : visualGroundingLearningRateMultiplier)
                        ),
                        learningRateScale: Double(optimizer.learningRateScale),
                        samplesPerSecond: performance.samplesPerSecond,
                        trainingStepMilliseconds: performance.stepMilliseconds,
                        batchPreparationMilliseconds: performance.preparationMilliseconds,
                        throughputRetention: performance.throughputRetention,
                        thermalState: thermalState,
                        elapsed: elapsed,
                        experienceElapsed: state.experienceSeconds ?? 0,
                        lossHistory: Array(state.lossHistory.suffix(Self.publishedLossHistoryLimit)),
                        epochLossHistory: Array((state.epochLossHistory ?? []).suffix(1_024)),
                        validationHistory: Array(state.validationHistory.suffix(1_024)),
                        learningRateHistory: Array((state.learningRateHistory ?? []).suffix(1_024)),
                        mlxActiveMemory: memory.activeMemory,
                        mlxCacheMemory: memory.cacheMemory,
                        mlxPeakMemory: memory.peakMemory
                    )
                    let visionStatus = reusesVisionFeatures
                        ? " • shared temporal work \(observationReuseRatio.formatted(.number.precision(.fractionLength(2))))×"
                        : ""
                    let performanceLimit = performance.limitingStatus(
                        memory: memory,
                        thermalState: thermalState
                    )
                    if let performanceLimit,
                       (
                           performanceLimit != lastLoggedPerformanceLimit
                               || now - lastPerformanceWarningTime
                                   >= Self.repeatedPerformanceWarningInterval
                       ) {
                        lastLoggedPerformanceLimit = performanceLimit
                        lastPerformanceWarningTime = now
                        AppLog.write(
                            .warning,
                            category: "Training",
                            performanceLimit,
                            details: "throughput retention \((100 * performance.throughputRetention).formatted(.number.precision(.fractionLength(1))))%; "
                                + "step \(performance.stepMilliseconds.formatted(.number.precision(.fractionLength(1)))) ms; "
                                + "input \(performance.preparationMilliseconds.formatted(.number.precision(.fractionLength(1)))) ms; "
                                + "thermal \(thermalState.rawValue)"
                        )
                    }
                    metrics(
                        report,
                        (performanceLimit
                            ?? "Ordered \(optimizerPipelineDepth)-step MLX pipeline on Apple-silicon GPU\(visionStatus)")
                            + (visualGroundingStep == nil
                                ? ""
                                : " • visual-grounding epoch")
                    )
                }

                let shouldCheckpoint = state.globalStep >= nextAutosaveStep
                let shouldPause = lock.withLock { pauseRequested }
                if shouldCheckpoint || shouldPause {
                    state.elapsed = baseElapsed + started.duration(to: .now).seconds
                    try await saveCheckpoint(profile: profile, model: model, optimizer: optimizer, randomState: randomState, state: state)
                    latestSnapshot = try await publishRunnableSnapshot(
                        profile: profile,
                        state: state,
                        completed: false,
                        preferBest: !split.validation.isEmpty
                    )
                    autosavesPublished += 1
                    if shouldCheckpoint {
                        nextAutosaveStep = saturatingAdd(state.globalStep, autosaveInterval)
                    }
                    if shouldPause {
                        lock.withLock { pauseRequested = false }
                        guard let latestSnapshot else { throw AgentTrainerError.model("The paused brain snapshot could not be published.") }
                        throw TrainingPaused(completion: latestSnapshot)
                    }
                }
                if configuredMaximum > 0, state.globalStep >= runStepTarget { break trainingLoop }
            }
            state.batchOffset = 0
            state.epoch = epoch + 1
            // A checkpoint paused inside this epoch keeps its versioned order.
            // Adopt the best current strategy only after that epoch finishes.
            if state.samplingStrategy != desiredSamplingStrategy {
                state.samplingStrategy = desiredSamplingStrategy
            }
            let epochTrainingLoss = epochWeightedLoss / Double(max(1, epochSampleCount))
            state.currentEpochWeightedLoss = nil
            state.currentEpochSampleCount = nil
            var epochLossHistory = state.epochLossHistory ?? []
            epochLossHistory.append(epochTrainingLoss)
            if epochLossHistory.count > 2_048 { epochLossHistory.removeFirst(1_024) }
            state.epochLossHistory = epochLossHistory
            var capturedBest = false
            var validationSelectionStatus: String?
            var monitorMetric = epochTrainingLoss
            if !split.validation.isEmpty {
                let validation = try evaluate(
                    model: model,
                    dataset: dataset,
                    indices: validationEvaluationIndices,
                    profile: profile,
                    positiveClassWeightValues: positiveClassWeightValues,
                    inputBufferPool: inputBufferPool,
                    validationStep: validationStep,
                    reusesVisionFeatures: reusesValidationVisionFeatures
                )
                let validationLoss = validation.loss
                guard validationLoss.isFinite else {
                    throw AgentTrainerError.model("Validation became numerically unstable, so the current epoch was not published. Lower the learning rate or reset this brain's learning state.")
                }
                monitorMetric = validationLoss
                state.currentValidationReport = validation.report
                state.currentValidationGlobalStep = state.globalStep
                state.currentBinaryDecisionThresholds = validation.binaryDecisionThresholds
                state.validationHistory.append(validationLoss)
                if state.validationHistory.count > 2_048 { state.validationHistory.removeFirst(1_024) }
                let improvesAggregate = validationLoss < (state.bestValidationLoss ?? .infinity)
                let regressesSparseHead = state.bestValidationReport.map {
                    validation.report.hasSevereBinaryRegression(comparedTo: $0)
                } ?? false
                let hasActionCollapse = validation.report.hasExecutableBinaryCollapse
                let previousBestHadCollapse = state.bestValidationReport?.hasExecutableBinaryCollapse
                if Self.shouldSelectValidationCandidate(
                    loss: validationLoss,
                    report: validation.report,
                    bestLoss: state.bestValidationLoss,
                    bestReport: state.bestValidationReport
                ) {
                    state.bestValidationLoss = validationLoss
                    state.bestGlobalStep = state.globalStep
                    state.bestEpoch = state.epoch
                    state.bestTrainingLoss = epochTrainingLoss
                    state.bestElapsed = baseElapsed + started.duration(to: .now).seconds
                    state.bestExperienceSeconds = state.experienceSeconds
                    state.bestValidationReport = validation.report
                    state.bestBinaryDecisionThresholds = validation.binaryDecisionThresholds
                    capturedBest = true
                    if hasActionCollapse {
                        validationSelectionStatus = "Saved the best available brain • one or more control heads remain below the execution threshold"
                    } else if previousBestHadCollapse == true {
                        validationSelectionStatus = "Selected a fully responsive brain over the previous partial-control checkpoint"
                    }
                } else if improvesAggregate, hasActionCollapse,
                          previousBestHadCollapse == false {
                    validationSelectionStatus = "Zero-history loss improved, but the fully responsive best brain was retained"
                } else if improvesAggregate, regressesSparseHead {
                    validationSelectionStatus = "Zero-history loss improved, but the best brain was retained because a sparse control head regressed sharply"
                }
            }
            if state.visualGroundingComplete != true {
                state.visualGroundingComplete = true
                visualGroundingStep = nil
                model.train(true)
                validationSelectionStatus = validationSelectionStatus
                    ?? "Completed the visual-grounding epoch; later epochs can relearn bounded self-history"
            }
            let schedulerStatus = updateAdaptiveLearningRate(
                optimizer: optimizer,
                metric: monitorMetric,
                profile: profile,
                state: &state
            )
            var learningRateHistory = state.learningRateHistory ?? []
            learningRateHistory.append(Double(optimizer.effectiveLearningRate()))
            if learningRateHistory.count > 2_048 { learningRateHistory.removeFirst(1_024) }
            state.learningRateHistory = learningRateHistory
            state.elapsed = baseElapsed + started.duration(to: .now).seconds
            try await saveCheckpoint(profile: profile, model: model, optimizer: optimizer, randomState: randomState, state: state, captureBest: capturedBest)
            if capturedBest {
                // Make an epoch's newly validated runner active immediately.
                // The exact checkpoint still keeps the newest optimizer state,
                // while a later unstable step cannot strand the profile on the
                // preceding unvalidated periodic autosave.
                latestSnapshot = try await publishRunnableSnapshot(
                    profile: profile,
                    state: state,
                    completed: false,
                    preferBest: true
                )
                autosavesPublished += 1
            }
            let memory = Memory.snapshot()
            metrics(TrainingMetrics(
                epoch: state.epoch,
                totalEpochs: targetEpoch,
                batch: stepsPerEpoch,
                totalBatches: stepsPerEpoch,
                globalStep: state.globalStep,
                totalSteps: totalSteps,
                nextAutosaveStep: nextAutosaveStep,
                autosavesPublished: autosavesPublished,
                trainingLoss: epochTrainingLoss,
                epochTrainingLoss: epochTrainingLoss,
                validationLoss: state.validationHistory.last,
                validationReport: state.currentValidationReport,
                trainingDataCoverage: trainingDataCoverage,
                effectiveLearningRate: Double(optimizer.effectiveLearningRate()),
                learningRateScale: Double(optimizer.learningRateScale),
                samplesPerSecond: performance.samplesPerSecond,
                trainingStepMilliseconds: performance.stepMilliseconds,
                batchPreparationMilliseconds: performance.preparationMilliseconds,
                throughputRetention: performance.throughputRetention,
                thermalState: TrainingThermalState.current,
                elapsed: state.elapsed,
                experienceElapsed: state.experienceSeconds ?? 0,
                lossHistory: Array(state.lossHistory.suffix(Self.publishedLossHistoryLimit)),
                epochLossHistory: Array(epochLossHistory.suffix(1_024)),
                validationHistory: Array(state.validationHistory.suffix(1_024)),
                learningRateHistory: Array(learningRateHistory.suffix(1_024)),
                mlxActiveMemory: memory.activeMemory,
                mlxCacheMemory: memory.cacheMemory,
                mlxPeakMemory: memory.peakMemory
            ), schedulerStatus ?? validationSelectionStatus ?? (!split.validation.isEmpty
                ? "Validated epoch \(state.epoch) on \(validationEvaluationIndices.count) representative held-out rows"
                : "Completed epoch \(state.epoch) • adaptive schedule monitoring epoch-average loss"))
        }

        state.elapsed = baseElapsed + started.duration(to: .now).seconds
        if latestSnapshot?.version.globalStep != state.globalStep {
            try await saveCheckpoint(profile: profile, model: model, optimizer: optimizer, randomState: randomState, state: state)
        }
        let final = try await publishRunnableSnapshot(profile: profile, state: state, completed: true, preferBest: !split.validation.isEmpty)
        return final
    }

    private func makeBatch(
        dataset: CachedDataset,
        indices: [Int],
        profile: AIProfile,
        inputBufferPool: MetalArrayBufferPool,
        reusesVisionFeatures: Bool
    ) throws -> [MLXArray] {
        let temporal = profile.training.effectiveTemporalVision
        let count = indices.count
        let visionPlan = reusesVisionFeatures ? dataset.visionBatchPlan(at: indices) : nil
        let visionCount = visionPlan?.uniqueSequences.count ?? count
        if temporal.pastFrameCount == 0 {
            var descriptors: [MetalArrayBufferPool.Descriptor] = [
                .init([visionCount, profile.preprocessing.sampleByteCount], dtype: .uint8)
            ]
            if reusesVisionFeatures { descriptors.append(.init([count], dtype: .int32)) }
            descriptors.append(.init([count, 2, ActionLayout.count], dtype: .float32))
            return try inputBufferPool.makeArrays(descriptors) { destinations in
                let actionDestination = destinations[reusesVisionFeatures ? 2 : 1]
                dataset.populateCurrentOnlyTrainingBatch(
                    at: indices,
                    observationSequences: visionPlan?.uniqueSequences,
                    packedCurrentObservations: destinations[0],
                    actionRows: actionDestination
                )
                if let visionPlan {
                    visionPlan.sampleToVision.withUnsafeBytes {
                        destinations[1].copyMemory(from: $0)
                    }
                }
                ActionLayout.sanitizeTrainingRows(
                    actionDestination.bindMemory(to: Float.self),
                    rowCount: count * 2,
                    channels: profile.channels,
                    restrictions: profile.effectiveRestrictions
                )
            }
        }
        let pastSpec = temporal.pastFrameSpec(from: profile.preprocessing)
        let pastVisionCount = visionPlan?.uniquePastObservations.count
            ?? visionCount * temporal.pastFrameCount
        var descriptors: [MetalArrayBufferPool.Descriptor] = [
            .init([visionCount, profile.preprocessing.sampleByteCount], dtype: .uint8),
            reusesVisionFeatures
                ? .init([pastVisionCount, pastSpec.sampleByteCount], dtype: .uint8)
                : .init([visionCount, temporal.pastFrameCount, pastSpec.sampleByteCount], dtype: .uint8),
            .init([visionCount, temporal.pastFrameCount, ActionLayout.count], dtype: .float32),
            .init([visionCount, temporal.pastFrameCount, 1], dtype: .float32)
        ]
        if reusesVisionFeatures {
            descriptors.append(.init([visionCount, temporal.pastFrameCount], dtype: .int32))
            descriptors.append(.init([count], dtype: .int32))
        }
        descriptors.append(
            .init([count, 2, ActionLayout.count], dtype: .float32)
        )
        return try inputBufferPool.makeArrays(descriptors) { destinations in
            let actionDestination = destinations[reusesVisionFeatures ? 6 : 4]
            if let visionPlan {
                dataset.populateTrainingBatch(
                    at: indices,
                    visionPlan: visionPlan,
                    packedCurrentObservations: destinations[0],
                    packedPastObservations: destinations[1],
                    pastControlRows: destinations[2],
                    pastFrameValidity: destinations[3],
                    actionRows: actionDestination
                )
                visionPlan.visionToPast.withUnsafeBytes {
                    destinations[4].copyMemory(from: $0)
                }
                visionPlan.sampleToVision.withUnsafeBytes {
                    destinations[5].copyMemory(from: $0)
                }
            } else {
                dataset.populateTrainingBatch(
                    at: indices,
                    packedCurrentObservations: destinations[0],
                    packedPastObservations: destinations[1],
                    pastControlRows: destinations[2],
                    pastFrameValidity: destinations[3],
                    actionRows: actionDestination
                )
            }
            ActionLayout.sanitizeTrainingRows(
                destinations[2].bindMemory(to: Float.self),
                rowCount: visionCount * temporal.pastFrameCount,
                channels: profile.channels,
                restrictions: profile.effectiveRestrictions
            )
            ActionLayout.sanitizeTrainingRows(
                actionDestination.bindMemory(to: Float.self),
                rowCount: count * 2,
                channels: profile.channels,
                restrictions: profile.effectiveRestrictions
            )
        }
    }

    private static func expandBatch(
        _ arrays: [MLXArray],
        profile: AIProfile,
        dtype: DType,
        reusesVisionFeatures: Bool
    ) -> ExpandedTrainingBatch {
        let temporal = profile.training.effectiveTemporalVision
        if temporal.pastFrameCount == 0 {
            let actions = arrays[reusesVisionFeatures ? 2 : 1]
            let visionCount = arrays[0].dim(0)
            return ExpandedTrainingBatch(
                currentImages: VisionPreprocessor.mlxTensor(
                    arrays[0], spec: profile.preprocessing, dtype: dtype
                ),
                pastImages: MLXArray.zeros(
                    [visionCount, 1, 1, 1, profile.preprocessing.channelCount],
                    dtype: .float32
                ),
                pastControls: MLXArray.zeros([visionCount, 1, ActionLayout.count], dtype: .float32),
                visionToPast: nil,
                sampleToVision: reusesVisionFeatures ? arrays[1] : nil,
                pastFrameMask: nil,
                targets: actions[0..., 0, 0...],
                previousTargets: actions[0..., 1, 0...]
            )
        }
        let pastSpec = temporal.pastFrameSpec(from: profile.preprocessing)
        let actions = arrays[reusesVisionFeatures ? 6 : 4]
        return ExpandedTrainingBatch(
            currentImages: VisionPreprocessor.mlxTensor(
                arrays[0], spec: profile.preprocessing, dtype: dtype
            ),
            pastImages: reusesVisionFeatures
                ? VisionPreprocessor.mlxTensor(arrays[1], spec: pastSpec, dtype: dtype)
                : VisionPreprocessor.mlxPastFrameTensor(arrays[1], spec: pastSpec, dtype: dtype),
            pastControls: arrays[2],
            visionToPast: reusesVisionFeatures ? arrays[4] : nil,
            sampleToVision: reusesVisionFeatures ? arrays[5] : nil,
            pastFrameMask: arrays[3],
            targets: actions[0..., 0, 0...],
            previousTargets: actions[0..., 1, 0...]
        )
    }

    /// Validation deliberately evaluates the runtime-start condition instead
    /// of teacher forcing exact demonstrated actions. Kept as a named helper so
    /// the contract remains directly testable when batch plumbing changes.
    static func runtimeStartupControls(like controls: MLXArray) -> MLXArray {
        MLXArray.zeros(like: controls)
    }

    private func evaluate(
        model: AgentPolicy,
        dataset: CachedDataset,
        indices: [Int],
        profile: AIProfile,
        positiveClassWeightValues: [Float],
        inputBufferPool: MetalArrayBufferPool,
        validationStep: @Sendable ([MLXArray]) -> [MLXArray],
        reusesVisionFeatures: Bool
    ) throws -> ValidationEvaluation {
        model.train(false)
        defer { model.train(true) }
        var weightedLoss = 0.0
        var evaluated = 0
        var report = ValidationAccumulator(
            profile: profile,
            activeBinaryIndices: Set(ActionLayout.learnableBinaryIndices(
                channels: profile.channels,
                restrictions: profile.effectiveRestrictions
            ).filter { positiveClassWeightValues.indices.contains($0) && positiveClassWeightValues[$0] > 0 })
        )
        let batchSize = max(1, profile.training.batchSize)
        var prefetchedArrays: [MLXArray]?
        for start in Swift.stride(from: 0, to: indices.count, by: batchSize) {
            let end = min(indices.count, start + batchSize)
            let batch = Array(indices[start..<end])
            let arrays = try prefetchedArrays ?? makeBatch(
                dataset: dataset,
                indices: batch,
                profile: profile,
                inputBufferPool: inputBufferPool,
                reusesVisionFeatures: reusesVisionFeatures
            )
            prefetchedArrays = nil
            let result = validationStep(arrays)
            let loss = result[0]
            let predictions = result[1]
            let targets = result[2]
            MLX.asyncEval(loss, predictions, targets)
            if end < indices.count {
                let nextEnd = min(indices.count, end + batchSize)
                prefetchedArrays = try makeBatch(
                    dataset: dataset,
                    indices: Array(indices[end..<nextEnd]),
                    profile: profile,
                    inputBufferPool: inputBufferPool,
                    reusesVisionFeatures: reusesVisionFeatures
                )
            }
            MLX.eval(loss, predictions, targets)
            weightedLoss += Double(loss.item(Float.self)) * Double(batch.count)
            evaluated += batch.count
            report.consume(
                predictions: predictions.asArray(Float.self),
                targets: targets.asArray(Float.self),
                rowCount: batch.count
            )
        }
        let finalized = report.finalize(sampleCount: evaluated)
        return ValidationEvaluation(
            loss: weightedLoss / Double(max(1, evaluated)),
            report: finalized.report,
            binaryDecisionThresholds: finalized.binaryDecisionThresholds
        )
    }

    /// One epoch of warmup is enough to stabilize small datasets, while the cap
    /// preserves the established schedule for large runs. A ten-step floor keeps
    /// the first update controlled without spending hundreds of steps at a tiny
    /// learning rate when an entire epoch contains only a handful of batches.
    static func recommendedWarmupSteps(stepsPerEpoch: Int) -> Int {
        min(500, max(10, stepsPerEpoch))
    }

    static func recommendedCycleSteps(stepsPerEpoch: Int, cycleEpochs: Int) -> Int {
        let product = max(1, stepsPerEpoch).multipliedReportingOverflow(by: max(1, cycleEpochs))
        return min(1_000_000, max(10, product.overflow ? 1_000_000 : product.partialValue))
    }

    /// Scales statistical coverage with held-out size while keeping epoch-end
    /// evaluation bounded. The previous fixed 16-batch budget examined only 512
    /// rows for common profiles regardless of whether validation contained ten
    /// thousand rows and many recordings.
    static func recommendedValidationSampleLimit(total: Int, batchSize: Int, segmentCount: Int) -> Int {
        guard total > 0 else { return 0 }
        let baseline = min(8_192, max(1, min(256, batchSize)) * 32)
        let diversity = min(8_192, max(1, min(128, segmentCount)) * 64)
        let statistical = min(8_192, Int(Double(total).squareRoot() * 32))
        return min(total, max(baseline, diversity, statistical))
    }

    /// Underrepresented enabled heads retain a bounded share of any applicable
    /// safety/learning loss, but must not consume the same shared-encoder
    /// gradient budget as a well-demonstrated control family. Completely unseen
    /// keyboard/modifier dimensions remain excluded by the capability firewall.
    static func objectiveHeadLossWeights(
        coverage: TrainingDataCoverage,
        channels: ActionChannels
    ) -> ActionHeadLossWeights {
        // Sixteen rows are enough to avoid a meaningless warning in a tiny
        // fixture, but they are not enough to give a head equal influence over
        // a shared encoder trained on hundreds of thousands of labels. A
        // square-root support target grows conservatively with the library and
        // prevents a few accidental modifier/mouse rows from competing with a
        // genuinely demonstrated keyboard head.
        let reliableRows = coverage.reliableActiveRowTarget
        func weight(enabled: Bool, activeRows: Int) -> Float {
            guard enabled else { return 0 }
            return min(
                1,
                max(
                    0.1,
                    Float(max(0, activeRows))
                        / Float(reliableRows)
                )
            )
        }
        return ActionHeadLossWeights(
            mouseMovement: weight(
                enabled: channels.mouseMovement,
                activeRows: coverage.mouseMovementRows
            ),
            buttons: weight(
                enabled: channels.buttons,
                activeRows: coverage.mouseButtonRows
            ),
            scroll: weight(
                enabled: channels.scroll,
                activeRows: coverage.scrollRows
            ),
            keyboard: weight(
                enabled: channels.keyboard,
                activeRows: coverage.keyboardRows
            ),
            modifiers: weight(
                enabled: channels.modifiers,
                activeRows: coverage.modifierRows
            )
        )
    }

    static func visualGroundingLearningRateMultiplier(
        configuredLearningRate: Double
    ) -> Float {
        let configured = configuredLearningRate.isFinite
            ? max(0.000_000_1, configuredLearningRate)
            : TrainingObjectiveContract.maximumVisualGroundingLearningRate
        return Float(min(
            1,
            TrainingObjectiveContract.maximumVisualGroundingLearningRate
                / configured
        ))
    }

    /// A finite first candidate is always worth preserving. Once a best brain
    /// exists, complete discrete-head responsiveness outranks aggregate loss;
    /// among equally responsive candidates, lower loss wins unless a supported
    /// sparse head regressed sharply on the same held-out rows.
    static func shouldSelectValidationCandidate(
        loss: Double,
        report: ValidationReport,
        bestLoss: Double?,
        bestReport: ValidationReport?
    ) -> Bool {
        guard loss.isFinite else { return false }
        guard let bestLoss, bestLoss.isFinite else { return true }
        if let bestReport {
            guard !report.hasSevereBinaryRegression(comparedTo: bestReport) else {
                return false
            }
            if let candidateResponsive = report.responsiveBinaryOutputCount,
               let bestResponsive = bestReport.responsiveBinaryOutputCount,
               candidateResponsive != bestResponsive {
                return candidateResponsive > bestResponsive
            }
            let candidateHasCollapse = report.hasExecutableBinaryCollapse
            let bestHasCollapse = bestReport.hasExecutableBinaryCollapse
            if candidateHasCollapse != bestHasCollapse {
                return bestHasCollapse && !candidateHasCollapse
            }
        }
        return loss < bestLoss
    }

    /// Queues a few strictly ordered compiled updates before synchronizing the
    /// unified GPU. This removes a host fence per batch without accumulating an
    /// unbounded lazy graph. The conservative working-set estimate includes
    /// parameters, gradients, Adam moments, inputs, activations, and compiler
    /// temporaries; large Float32 stress profiles therefore choose less depth.
    static func recommendedOptimizerPipelineDepth(
        profile: AIProfile,
        physicalMemory: UInt64
    ) -> Int {
        let physical = Int64(min(physicalMemory, UInt64(Int64.max)))
        let workingSet = max(1, ModelSizing.estimatedTrainingWorkingSet(profile))
        guard physical > 0, workingSet < Int64.max else { return 1 }
        if workingSet <= physical / 5 { return 4 }
        if workingSet <= physical / 3 { return 3 }
        if workingSet <= physical / 2 { return 2 }
        return 1
    }

    static func adaptivePlateauUpdate(
        metric: Double,
        best: Double?,
        plateauEpochs: Int,
        patience: Int,
        minimumRelativeImprovement: Double = 0.002
    ) -> (best: Double, plateauEpochs: Int, shouldReduce: Bool) {
        guard let best else { return (metric, 0, false) }
        let meaningfulDelta = max(1e-8, abs(best) * max(0, minimumRelativeImprovement))
        if metric < best - meaningfulDelta { return (metric, 0, false) }
        let nextEpochs = max(0, plateauEpochs) + 1
        return nextEpochs >= max(1, patience)
            ? (best, 0, true)
            : (best, nextEpochs, false)
    }

    private func updateAdaptiveLearningRate(
        optimizer: ResumableAdamW,
        metric: Double,
        profile: AIProfile,
        state: inout CheckpointState
    ) -> String? {
        guard optimizer.schedule == .adaptiveCosine, metric.isFinite else { return nil }
        let update = Self.adaptivePlateauUpdate(
            metric: metric,
            best: state.schedulerBestMetric,
            plateauEpochs: state.schedulerPlateauEpochs ?? 0,
            patience: profile.training.effectivePlateauPatience
        )
        state.schedulerBestMetric = update.best
        state.schedulerPlateauEpochs = update.plateauEpochs
        guard update.shouldReduce else { return nil }
        let previousScale = optimizer.learningRateScale
        let nextScale = optimizer.reduceLearningRate()
        if nextScale < previousScale - 0.000_001 {
            state.schedulerReductions = (state.schedulerReductions ?? 0) + 1
            return "Plateau detected • learning-rate envelope reduced to \(nextScale.formatted(.number.precision(.fractionLength(4))))×; cosine exploration continues"
        }
        return "Plateau detected • learning-rate floor reached; cosine restarts remain active"
    }

    /// Distributes high-signal rows across fixed-size batches without duplicating
    /// or dropping any sample. Compared with a uniform shuffle, rare edges and
    /// additive controls reach the optimizer throughout the epoch instead of
    /// clustering in a few high-variance updates.
    func trainingOrder(
        dataset: CachedDataset,
        indices: [Int],
        batchSize rawBatchSize: Int,
        seed: UInt64,
        profile: AIProfile,
        precomputedSalientIndices: Set<Int>? = nil
    ) -> [Int] {
        let randomized = shuffled(indices, seed: seed)
        let batchSize = max(1, rawBatchSize)
        guard randomized.count > batchSize else { return randomized }
        let salient = precomputedSalientIndices ?? dataset.salientTrainingIndices(
            at: indices,
            channels: profile.channels,
            restrictions: profile.effectiveRestrictions
        )
        return salienceBalancedOrder(
            randomized: randomized,
            batchSize: batchSize,
            seed: seed,
            salient: salient
        )
    }

    func groupedVisionTrainingOrder(
        groups: [[Int]],
        batchSize rawBatchSize: Int,
        seed: UInt64,
        salientIndices: Set<Int>
    ) -> [Int] {
        guard !groups.isEmpty else { return [] }
        let groupIndices = Array(groups.indices)
        let randomized = shuffled(groupIndices, seed: seed)
        let salientGroups = Set(groupIndices.filter { group in
            groups[group].contains(where: salientIndices.contains)
        })
        let averageGroupSize = Double(groups.reduce(0) { $0 + $1.count })
            / Double(max(1, groups.count))
        let groupsPerBatch = max(
            1,
            Int((Double(max(1, rawBatchSize)) / max(1, averageGroupSize)).rounded())
        )
        let orderedGroups = salienceBalancedOrder(
            randomized: randomized,
            batchSize: groupsPerBatch,
            seed: seed,
            salient: salientGroups
        )
        return orderedGroups.flatMap { groups[$0] }
    }

    /// Builds temporally local batches, then shuffles the batches and their
    /// internal row groups deterministically. The optimizer still sees every
    /// row exactly once in a randomized epoch, while adjacent causal windows
    /// share mapped pages and individual past-frame CNN embeddings.
    func localityGroupedVisionTrainingOrder(
        groups: [[Int]],
        batchSize rawBatchSize: Int,
        seed: UInt64,
        salientIndices: Set<Int>
    ) -> [Int] {
        guard !groups.isEmpty else { return [] }
        let batchSize = max(1, rawBatchSize)
        let averageGroupSize = Double(groups.reduce(0) { $0 + $1.count })
            / Double(max(1, groups.count))
        let groupsPerBatch = max(
            1,
            Int((Double(batchSize) / max(1, averageGroupSize)).rounded())
        )
        let laneCount = groupsPerBatch >= 64 ? 4 : groupsPerBatch >= 16 ? 2 : 1
        let groupsPerLane = max(1, Int(ceil(Double(groupsPerBatch) / Double(laneCount))))
        let localityRuns: [[Int]] = stride(
            from: 0,
            to: groups.count,
            by: groupsPerLane
        ).map { start in
            Array(start..<min(groups.count, start + groupsPerLane))
        }
        // Combine a few independently selected local runs in each optimizer
        // batch. This keeps gradient diversity close to a global shuffle while
        // retaining long enough contiguous lanes for frame and page reuse.
        let randomizedRuns = shuffled(Array(localityRuns.indices), seed: seed)
        var localityBatches: [[Int]] = stride(
            from: 0,
            to: randomizedRuns.count,
            by: laneCount
        ).map { start in
            randomizedRuns[start..<min(randomizedRuns.count, start + laneCount)]
                .flatMap { localityRuns[$0] }
        }
        let salientGroups = Set(groups.indices.filter { group in
            groups[group].contains(where: salientIndices.contains)
        })

        // Preserve the existing high-signal coverage contract with minimal
        // damage to locality: only batches without a salient group receive one,
        // and it is swapped from a batch that has more than one.
        if salientGroups.count >= localityBatches.count {
            func salientCount(in batch: [Int]) -> Int {
                batch.count(where: salientGroups.contains)
            }
            for receiver in localityBatches.indices
            where salientCount(in: localityBatches[receiver]) == 0 {
                guard let donor = localityBatches.indices.first(where: {
                    $0 != receiver && salientCount(in: localityBatches[$0]) > 1
                }),
                let donorPosition = localityBatches[donor].firstIndex(where: salientGroups.contains),
                let receiverPosition = localityBatches[receiver].firstIndex(where: {
                    !salientGroups.contains($0)
                }) else { continue }
                let salientGroup = localityBatches[donor][donorPosition]
                localityBatches[donor][donorPosition] = localityBatches[receiver][receiverPosition]
                localityBatches[receiver][receiverPosition] = salientGroup
            }
        }

        let randomizedBatches = shuffled(Array(localityBatches.indices), seed: seed)
        return randomizedBatches.flatMap { batchIndex in
            let groupOrder = shuffled(
                localityBatches[batchIndex],
                seed: seed ^ (UInt64(batchIndex) &* 0xD1B54A32D192ED03)
            )
            return groupOrder.flatMap { groups[$0] }
        }
    }

    private func salienceBalancedOrder(
        randomized: [Int],
        batchSize: Int,
        seed: UInt64,
        salient: Set<Int>
    ) -> [Int] {
        guard !salient.isEmpty, salient.count < randomized.count else { return randomized }

        let priority = randomized.filter(salient.contains)
        let ordinary = randomized.filter { !salient.contains($0) }
        let batchCount = Int(ceil(Double(randomized.count) / Double(batchSize)))
        let targetSizes = (0..<batchCount).map { batch in
            min(batchSize, randomized.count - batch * batchSize)
        }
        var batches = Array(repeating: [Int](), count: batchCount)
        var placementRandom = SplitMix64(state: seed ^ 0xD1B54A32D192ED03)
        var cursor = Int(placementRandom.next() % UInt64(batchCount))
        for index in priority {
            while batches[cursor].count >= targetSizes[cursor] {
                cursor = (cursor + 1) % batchCount
            }
            batches[cursor].append(index)
            cursor = (cursor + 1) % batchCount
        }
        var ordinaryOffset = 0
        for batch in batches.indices {
            while batches[batch].count < targetSizes[batch] {
                batches[batch].append(ordinary[ordinaryOffset])
                ordinaryOffset += 1
            }
        }
        return batches.flatMap { $0 }
    }

    func splitIndices(
        dataset: CachedDataset,
        fraction: Double,
        seed: UInt64,
        channels: ActionChannels = .all,
        restrictions: ActionRestrictions = ActionRestrictions()
    ) -> (train: [Int], validation: [Int]) {
        let fraction = min(0.9, max(0, fraction))
        let learnableBinaryOutputs = ActionLayout.learnableBinaryIndices(
            channels: channels,
            restrictions: restrictions
        )
        if dataset.manifest.segments.count > 1 {
            let segments = dataset.manifest.segments
            let shuffledSegments = shuffled(Array(segments.indices), seed: seed)
            let validationCount = min(
                max(0, segments.count - 1),
                max(fraction > 0 ? 1 : 0, Int(Double(segments.count) * fraction))
            )
            let segmentCounts = segments.map { segment in
                dataset.binaryPositiveCounts(in: segment.start..<(segment.start + segment.count))
            }
            var remainingCounts = segmentCounts.reduce([Int](repeating: 0, count: ActionLayout.count)) { partial, counts in
                zip(partial, counts).map(+)
            }
            var validationSegments: Set<Int> = []
            var validationPositiveCounts = [Int](repeating: 0, count: ActionLayout.count)
            var validationRows = 0
            let targetRows = Int((Double(dataset.count) * fraction).rounded())
            let seededRank = Dictionary(uniqueKeysWithValues: shuffledSegments.enumerated().map { ($0.element, $0.offset) })

            // Choose the requested number of whole recordings, but aim each
            // slot at its share of the target row count. The previous random
            // recording-count split could hold out a huge recording and turn a
            // requested 10% validation fraction into most of the dataset.
            while validationSegments.count < validationCount {
                let remainingSlots = validationCount - validationSegments.count
                let desiredRows = max(1, (max(0, targetRows - validationRows) + remainingSlots - 1) / remainingSlots)
                let candidates = segments.indices.filter { segmentIndex in
                    guard !validationSegments.contains(segmentIndex) else { return false }
                    let counts = segmentCounts[segmentIndex]
                    return !learnableBinaryOutputs.contains { output in
                        counts[output] > 0 && remainingCounts[output] - counts[output] <= 0
                    }
                }
                guard let selected = candidates.min(by: { lhs, rhs in
                    func score(_ index: Int) -> Double {
                        let distance = Double(abs(segments[index].count - desiredRows)) / Double(desiredRows)
                        let newCoverage = learnableBinaryOutputs.count { output in
                            validationPositiveCounts[output] == 0 && segmentCounts[index][output] > 0
                        }
                        return distance - min(0.5, Double(newCoverage) * 0.05)
                    }
                    let left = score(lhs), right = score(rhs)
                    if abs(left - right) > 1e-12 { return left < right }
                    return (seededRank[lhs] ?? 0) < (seededRank[rhs] ?? 0)
                }) else { break }
                validationSegments.insert(selected)
                validationRows += segments[selected].count
                for output in learnableBinaryOutputs {
                    remainingCounts[output] -= segmentCounts[selected][output]
                    validationPositiveCounts[output] += segmentCounts[selected][output]
                }
            }
            var train: [Int] = [], validation: [Int] = []
            for (i, segment) in segments.enumerated() {
                let values = Array(segment.start..<(segment.start + segment.count))
                if validationSegments.contains(i) { validation += values } else { train += values }
            }
            return (train, validation)
        }
        let validationCount = Int(Double(dataset.count) * fraction)
        let proposedValidationStart = max(0, dataset.count - validationCount)
        guard validationCount > 0, proposedValidationStart > 0 else {
            return (Array(0..<dataset.count), [])
        }
        var trainingEnd = proposedValidationStart
        let totalCounts = dataset.binaryPositiveCounts(in: 0..<dataset.count)
        let trainCounts = dataset.binaryPositiveCounts(in: 0..<trainingEnd)
        var missing = Set(learnableBinaryOutputs.filter { totalCounts[$0] > 0 && trainCounts[$0] == 0 })
        if !missing.isEmpty {
            // Keep a single recording temporally contiguous. Moving isolated
            // validation rows into training leaks their neighboring temporal
            // frames; extend the boundary through the needed example.
            for index in trainingEnd..<dataset.count where !missing.isEmpty {
                let action = dataset.action(at: index)
                let covered = missing.filter { action[$0] >= 0.5 }
                if !covered.isEmpty {
                    trainingEnd = index + 1
                    missing.subtract(covered)
                }
            }
        }
        guard missing.isEmpty,
              let validationStart = dataset.firstDisjointValidationIndex(
                trainingEnd: trainingEnd,
                proposedStart: max(proposedValidationStart, trainingEnd)
              ),
              validationStart < dataset.count else {
            // A false held-out score is worse than no held-out score. When the
            // sole recording cannot supply disjoint context, train on all rows.
            return (Array(0..<dataset.count), [])
        }
        return (Array(0..<trainingEnd), Array(validationStart..<dataset.count))
    }

    private func shuffled(_ input: [Int], seed: UInt64) -> [Int] {
        var result = input
        var rng = SplitMix64(state: seed)
        guard result.count > 1 else { return result }
        for i in stride(from: result.count - 1, through: 1, by: -1) {
            result.swapAt(i, Int(rng.next() % UInt64(i + 1)))
        }
        return result
    }

    private func saturatingAdd(_ value: Int, _ increment: Int) -> Int {
        let result = value.addingReportingOverflow(max(0, increment))
        return result.overflow ? Int.max : result.partialValue
    }

    private func profileSignature(_ profile: AIProfile, recordings: [RecordingItem]) throws -> String {
        struct TrainingIdentity: Encodable {
            let trainingDataSchema: Int
            let trainingObjectiveSchema: Int
            let preprocessing: PreprocessingSpec
            let channels: ActionChannels
            let training: TrainingConfiguration
            let recordings: [RecordingTrainingIdentity]
            let folderIDs: [UUID]
            let restrictions: ActionRestrictions
        }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        var resumeCompatibleTraining = profile.training
        // These values control how long/often to run, not the meaning of a saved brain.
        // Users may extend an existing run without invalidating its exact checkpoint.
        resumeCompatibleTraining.epochs = 0
        resumeCompatibleTraining.maximumSteps = 0
        resumeCompatibleTraining.checkpointInterval = 0
        var normalizedChannels = profile.channels
        normalizedChannels.absoluteMouse = profile.channels.mouseMovement
        normalizedChannels.relativeMouse = profile.channels.mouseMovement
        let recordingIdentities = try recordings
            .map { try RecordingTrainingIdentity(recording: $0) }
            .sorted { $0.id.uuidString < $1.id.uuidString }
        let identity = TrainingIdentity(
            trainingDataSchema: TrainingDataContract.schemaVersion,
            trainingObjectiveSchema: TrainingObjectiveContract.schemaVersion,
            preprocessing: profile.preprocessing,
            channels: normalizedChannels,
            training: resumeCompatibleTraining,
            recordings: recordingIdentities,
            folderIDs: profile.effectiveFolderIDs.sorted { $0.uuidString < $1.uuidString },
            restrictions: profile.effectiveRestrictions
        )
        return SHA256.hash(data: try encoder.encode(identity)).map { String(format: "%02x", $0) }.joined()
    }

    private func saveCheckpoint(profile: AIProfile, model: AgentPolicy, optimizer: ResumableAdamW, randomState: MLXRandom.RandomState, state: CheckpointState, captureBest: Bool = false) async throws {
        let destination = await WorkspaceStore.shared.checkpointDirectory(profileID: profile.id)
        let temporary = destination.deletingLastPathComponent().appendingPathComponent(".Checkpoint.\(UUID().uuidString).tmp")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        do {
            let currentWeights = temporary.appendingPathComponent("weights.safetensors")
            try model.saveWeights(to: currentWeights)
            try optimizer.save(to: temporary.appendingPathComponent("optimizer.safetensors"))
            try TrainingRandomState.save(randomState, to: temporary.appendingPathComponent("random.safetensors"))
            let bestWeights = temporary.appendingPathComponent("best.weights.safetensors")
            if captureBest {
                try FileManager.default.copyItem(at: currentWeights, to: bestWeights)
            } else {
                let existingBest = destination.appendingPathComponent("best.weights.safetensors")
                if FileManager.default.fileExists(atPath: existingBest.path) {
                    try FileManager.default.copyItem(at: existingBest, to: bestWeights)
                }
            }
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(state).write(to: temporary.appendingPathComponent("state.json"), options: .atomic)
            try encoder.encode(ModelContract.schemaVersion).write(to: temporary.appendingPathComponent("model-schema.json"), options: .atomic)
            if FileManager.default.fileExists(atPath: destination.path) {
                let backupName = ".Checkpoint.backup.\(UUID().uuidString)"
                let backup = destination.deletingLastPathComponent().appendingPathComponent(backupName)
                _ = try FileManager.default.replaceItemAt(destination, withItemAt: temporary, backupItemName: backupName, options: .usingNewMetadataOnly)
                try? FileManager.default.removeItem(at: backup)
            } else {
                try FileManager.default.moveItem(at: temporary, to: destination)
            }
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
    }

    private func publishRunnableSnapshot(profile: AIProfile, state: CheckpointState, completed: Bool, preferBest: Bool = false) async throws -> TrainingCompletion {
        let checkpoint = await WorkspaceStore.shared.checkpointDirectory(profileID: profile.id)
        let bestWeights = checkpoint.appendingPathComponent("best.weights.safetensors")
        let usesBest = preferBest
            && state.bestGlobalStep != nil
            && FileManager.default.fileExists(atPath: bestWeights.path)
        let currentEpoch = state.batchOffset > 0 ? state.epoch + 1 : state.epoch
        let displayedEpoch = usesBest ? state.bestEpoch ?? currentEpoch : currentEpoch
        let displayedStep = usesBest ? state.bestGlobalStep ?? state.globalStep : state.globalStep
        let displayedLoss = usesBest
            ? state.bestTrainingLoss ?? state.epochLossHistory?.last ?? state.lossHistory.last ?? 0
            : state.epochLossHistory?.last ?? state.lossHistory.last ?? 0
        let displayedValidationLoss = usesBest ? state.bestValidationLoss : state.validationHistory.last
        let displayedValidationReport = usesBest ? state.bestValidationReport : state.currentValidationReport
        let displayedValidationStep = usesBest ? state.bestGlobalStep : state.currentValidationGlobalStep
        let displayedBinaryDecisionThresholds = usesBest
            ? state.bestBinaryDecisionThresholds
            : state.currentBinaryDecisionThresholds
        let version = ModelVersionManifest(
            id: UUID(),
            name: usesBest
                ? "Best \(completed ? "Brain" : "Autosave") • Epoch \(displayedEpoch) • Step \(displayedStep)"
                : completed
                    ? "Brain • Epoch \(displayedEpoch) • Step \(displayedStep)"
                    : "Autosave • Epoch \(displayedEpoch) • Step \(displayedStep)",
            createdAt: Date(),
            globalStep: displayedStep,
            trainingLoss: displayedLoss,
            validationLoss: displayedValidationLoss,
            preprocessing: profile.preprocessing,
            channels: profile.channels,
            training: profile.training,
            optimizerFile: usesBest ? nil : "optimizer.safetensors",
            trainingStateFile: usesBest ? nil : "state.json",
            randomStateFile: usesBest ? nil : "random.safetensors",
            epoch: displayedEpoch,
            isAutosave: !completed,
            demonstratedKeyCodes: state.demonstratedKeyCodes ?? [],
            relativeMouseScale: GameCameraContract.deltaScale,
            trainingDataSchema: TrainingDataContract.schemaVersion,
            trainingObjectiveSchema: TrainingObjectiveContract.schemaVersion,
            visualGroundingComplete: state.visualGroundingComplete,
            trainingDurationSeconds: usesBest ? state.bestElapsed : state.elapsed,
            experienceDurationSeconds: usesBest ? state.bestExperienceSeconds : state.experienceSeconds ?? 0,
            trainingShowsCursor: state.trainingShowsCursor,
            recommendedMouseMode: state.recommendedMouseMode,
            validationReport: displayedValidationReport,
            binaryDecisionSchema: displayedBinaryDecisionThresholds == nil
                ? nil
                : BinaryDecisionContract.schemaVersion,
            binaryDecisionThresholds: displayedBinaryDecisionThresholds.map {
                BinaryDecisionCalibration.normalized($0)
            },
            validationGlobalStep: displayedValidationStep,
            trainingDataCoverage: state.trainingDataCoverage
        )
        let destination = await WorkspaceStore.shared.versionDirectory(profileID: profile.id, versionID: version.id)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        do {
            let sourceWeights = usesBest ? bestWeights : checkpoint.appendingPathComponent("weights.safetensors")
            try FileManager.default.copyItem(at: sourceWeights, to: destination.appendingPathComponent(version.weightsFile))
            if !usesBest {
                try FileManager.default.copyItem(at: checkpoint.appendingPathComponent("optimizer.safetensors"), to: destination.appendingPathComponent("optimizer.safetensors"))
                try FileManager.default.copyItem(at: checkpoint.appendingPathComponent("state.json"), to: destination.appendingPathComponent("state.json"))
                try FileManager.default.copyItem(at: checkpoint.appendingPathComponent("random.safetensors"), to: destination.appendingPathComponent("random.safetensors"))
            }
            try await WorkspaceStore.shared.saveVersionManifest(version, profileID: profile.id)
        } catch {
            try? FileManager.default.removeItem(at: destination)
            throw error
        }
        var updated = profile
        updated.activeVersionID = version.id
        // Save the active reference before pruning so cleanup can never remove
        // the version that was just published.
        try await WorkspaceStore.shared.saveProfile(updated)
        let removed = try await WorkspaceStore.shared.pruneAutosaveVersions(profile: updated, keeping: 10)
        let savedBrainCount = await WorkspaceStore.shared.listVersions(profileID: profile.id).count
        updated.trainingProgress = TrainingProgressSummary(
            globalStep: state.globalStep,
            epoch: displayedEpoch,
            updatedAt: version.createdAt,
            savedBrainCount: savedBrainCount,
            trainingDurationSeconds: state.elapsed,
            experienceDurationSeconds: state.experienceSeconds ?? 0
        )
        try await WorkspaceStore.shared.saveProfile(updated)
        if removed > 1 {
            AppLog.write(category: "Training", "Pruned old autosaves", details: "\(profile.name): removed \(removed), kept the newest 10")
        }
        return TrainingCompletion(profile: updated, version: version, completed: completed)
    }

    private func restoreCheckpointIfPresent(
        profile: AIProfile,
        expectedSignature: String,
        expectedRecordingOrder: [UUID],
        legacyValidationNeedsRefresh: Bool,
        model: AgentPolicy,
        optimizer: ResumableAdamW,
        randomState: MLXRandom.RandomState,
        state: inout CheckpointState
    ) async throws -> CheckpointRestore {
        let directory = await WorkspaceStore.shared.checkpointDirectory(profileID: profile.id)
        let stateURL = directory.appendingPathComponent("state.json")
        if FileManager.default.fileExists(atPath: stateURL.path) {
            let restored = try JSONDecoder().decode(CheckpointState.self, from: Data(contentsOf: stateURL))
            let recordingOrderMatches = restored.recordingOrder.map { $0 == expectedRecordingOrder } ?? true
            let samplingStrategyIsKnown = restored.samplingStrategy == nil
                || restored.samplingStrategy == TrainingSamplingContract.salienceBalanced
                || restored.samplingStrategy == TrainingSamplingContract.groupedVision
                || restored.samplingStrategy == TrainingSamplingContract.localityGroupedVision
            if restored.profileSignature == expectedSignature,
               recordingOrderMatches,
               samplingStrategyIsKnown {
                let validationNeedsRefresh = restored.validationStrategy.map {
                    $0 != TrainingValidationContract.capabilityAware
                } ?? legacyValidationNeedsRefresh
                try model.loadWeights(from: directory.appendingPathComponent("weights.safetensors"))
                try optimizer.load(from: directory.appendingPathComponent("optimizer.safetensors"))
                let randomStateURL = directory.appendingPathComponent("random.safetensors")
                if FileManager.default.fileExists(atPath: randomStateURL.path) { try TrainingRandomState.load(randomState, from: randomStateURL) }
                state = restored
                state.validationStrategy = TrainingValidationContract.capabilityAware
                return CheckpointRestore(
                    status: validationNeedsRefresh
                        ? "Restored exact optimizer state; recalibrated validation for the current split"
                        : "Restored exact checkpoint; compiling resumed MLX graph",
                    captureValidationBaseline: validationNeedsRefresh
                )
            }
        }

        // A data-contract upgrade should never make a trained AI silently fall
        // back to random weights. When shapes still match, keep the active
        // runnable brain and begin a fresh optimizer/batch sequence on the new
        // targets. The old version itself remains immutable and runnable.
        if let versionID = profile.activeVersionID,
           let version = await WorkspaceStore.shared.version(profileID: profile.id, versionID: versionID),
           version.schemaVersion == ModelContract.schemaVersion,
           version.preprocessing == profile.preprocessing,
           version.training.architecture.weightLayout == profile.training.architecture.weightLayout {
            if version.requiresFreshSupervisedTrainingRestart {
                // Real record-disjoint diagnostics showed that pre-v4 policies
                // could collapse the visual projection as well as poison the
                // convolutional and recurrent readouts. The old brain remains
                // runnable, but objective v4 must start with a coherent fresh
                // model rather than inherit an all-no-op fixed point.
                return CheckpointRestore(
                    status: "Preserved the selected legacy brain; starting clean objective-v4 visual grounding because its old weights can encode a no-action fixed point",
                    captureValidationBaseline: false
                )
            }
            let versionDirectory = await WorkspaceStore.shared.versionDirectory(profileID: profile.id, versionID: versionID)
            try model.loadWeights(from: versionDirectory.appendingPathComponent(version.weightsFile))
            state.epoch = max(0, version.epoch ?? 0)
            state.batchOffset = 0
            state.globalStep = max(0, version.globalStep)
            state.elapsed = max(0, version.trainingDurationSeconds ?? profile.trainingProgress?.trainingDurationSeconds ?? 0)
            state.experienceSeconds = version.experienceDurationSeconds ?? profile.trainingProgress?.experienceDurationSeconds
            state.lossHistory = [version.trainingLoss]
            state.epochLossHistory = [version.trainingLoss]
            state.validationHistory = version.validationLoss.map { [$0] } ?? []
            state.currentValidationReport = version.validationReport
            state.currentValidationGlobalStep = version.validationGlobalStep
            state.visualGroundingComplete = version.visualGroundingComplete
            state.currentBinaryDecisionThresholds = BinaryDecisionCalibration.normalized(
                version.hasCurrentBinaryDecisionContract
                    ? version.binaryDecisionThresholds
                    : nil
            )
            if let validationLoss = version.validationLoss, validationLoss.isFinite {
                state.bestValidationLoss = validationLoss
                state.bestGlobalStep = state.globalStep
                state.bestEpoch = state.epoch
                state.bestTrainingLoss = version.trainingLoss
                state.bestElapsed = state.elapsed
                state.bestExperienceSeconds = state.experienceSeconds
                state.bestValidationReport = version.validationReport
                state.bestBinaryDecisionThresholds = state.currentBinaryDecisionThresholds
                return CheckpointRestore(
                    status: "Loaded the selected brain's compatible visual and action weights; optimizer restarted safely",
                    captureValidationBaseline: true
                )
            }
            return CheckpointRestore(
                status: "Loaded the active brain's compatible visual and action weights; optimizer restarted safely",
                captureValidationBaseline: true
            )
        }
        return CheckpointRestore(
            status: "Compiling fused MLX training graph on Apple GPU",
            captureValidationBaseline: false
        )
    }

}

private struct CheckpointRestore {
    var status: String
    var captureValidationBaseline: Bool
}

private struct TrainingPerformanceWindow {
    private static let capacity = 32
    private static let minimumStableSamples = 8

    private var measurements: [(samples: Int, seconds: Double, preparation: Double)] = []
    private var bestStableSamplesPerSecond = 0.0
    private(set) var samplesPerSecond = 0.0
    private(set) var stepMilliseconds = 0.0
    private(set) var preparationMilliseconds = 0.0
    private(set) var throughputRetention = 1.0

    init() {
        measurements.reserveCapacity(Self.capacity)
    }

    mutating func record(samples: Int, stepSeconds: Double, preparationSeconds: Double) {
        measurements.append((
            max(0, samples),
            max(0.000_001, stepSeconds),
            max(0, preparationSeconds)
        ))
        if measurements.count > Self.capacity {
            measurements.removeFirst(measurements.count - Self.capacity)
        }
        let sampleCount = measurements.reduce(0) { $0 + $1.samples }
        let seconds = measurements.reduce(0.0) { $0 + $1.seconds }
        samplesPerSecond = Double(sampleCount) / max(0.000_001, seconds)
        stepMilliseconds = 1_000 * seconds / Double(max(1, measurements.count))
        preparationMilliseconds = 1_000
            * measurements.reduce(0.0) { $0 + $1.preparation }
            / Double(max(1, measurements.count))
        if measurements.count >= Self.minimumStableSamples {
            bestStableSamplesPerSecond = max(bestStableSamplesPerSecond, samplesPerSecond)
        }
        throughputRetention = bestStableSamplesPerSecond > 0
            ? min(1, samplesPerSecond / bestStableSamplesPerSecond)
            : 1
    }

    func limitingStatus(
        memory: Memory.Snapshot,
        thermalState: TrainingThermalState,
        physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> String? {
        guard measurements.count == Self.capacity,
              throughputRetention < 0.80 else { return nil }
        if thermalState == .serious || thermalState == .critical {
            return "Thermal pressure is throttling Metal"
        }
        if preparationMilliseconds >= stepMilliseconds * 0.65 {
            return "Mapped-data paging is limiting the GPU input pipeline"
        }
        let usedUnifiedMemory = Double(memory.activeMemory) + Double(memory.cacheMemory)
        if usedUnifiedMemory > Double(physicalMemory) * 0.70 {
            return "Unified-memory pressure is limiting Metal"
        }
        return "Metal step time is below its warm-run peak"
    }
}

private struct ExpandedTrainingBatch {
    let currentImages: MLXArray
    let pastImages: MLXArray
    let pastControls: MLXArray
    let visionToPast: MLXArray?
    let sampleToVision: MLXArray?
    let pastFrameMask: MLXArray?
    let targets: MLXArray
    let previousTargets: MLXArray
}

private struct ScheduledTrainingStep {
    let loss: MLXArray
    let sampleCount: Int
    let endOffset: Int
    let preparationSeconds: Double
}

private struct ValidationEvaluation {
    var loss: Double
    var report: ValidationReport
    var binaryDecisionThresholds: [Float]
}

private struct BinaryValidationAccumulator {
    var truePositives = 0
    var falsePositives = 0
    var falseNegatives = 0
    var trueNegatives = 0

    mutating func consume(predicted: Bool, target: Bool) {
        switch (predicted, target) {
        case (true, true): truePositives += 1
        case (true, false): falsePositives += 1
        case (false, true): falseNegatives += 1
        case (false, false): trueNegatives += 1
        }
    }

    var total: Int { truePositives + falsePositives + falseNegatives + trueNegatives }
    var metrics: BinaryValidationMetrics? {
        guard total > 0 else { return nil }
        return BinaryValidationMetrics(
            truePositives: truePositives,
            falsePositives: falsePositives,
            falseNegatives: falseNegatives,
            trueNegatives: trueNegatives
        )
    }
}

private struct ValidationAccumulator {
    let profile: AIProfile
    let activeBinaryIndices: Set<Int>
    private var binaryObservations: [Int: [BinaryDecisionObservation]]
    private var absoluteError = 0.0
    private var absoluteCount = 0
    private var relativeError = 0.0
    private var relativeCount = 0
    private var scrollError = 0.0
    private var scrollCount = 0
    private var idleContinuousFalseActions = 0
    private var idleContinuousCount = 0

    init(profile: AIProfile, activeBinaryIndices: Set<Int>) {
        self.profile = profile
        self.activeBinaryIndices = activeBinaryIndices
        binaryObservations = Dictionary(
            uniqueKeysWithValues: activeBinaryIndices.map { ($0, []) }
        )
    }

    mutating func consume(predictions: [Float], targets: [Float], rowCount: Int) {
        guard predictions.count >= rowCount * ActionLayout.count,
              targets.count >= rowCount * ActionLayout.count else { return }
        for row in 0..<rowCount {
            let base = row * ActionLayout.count
            for index in activeBinaryIndices {
                let target = targets[base + index] >= 0.5
                binaryObservations[index, default: []].append(
                    BinaryDecisionObservation(
                        score: predictions[base + index],
                        target: target
                    )
                )
            }
            if profile.channels.mouseMovement {
                for index in ActionLayout.absoluteMouse {
                    absoluteError += Double(abs(predictions[base + index] - targets[base + index]))
                    absoluteCount += 1
                }
                consumeContinuous(
                    range: ActionLayout.relativeMouse,
                    predictions: predictions,
                    targets: targets,
                    base: base,
                    isRelativeMouse: true
                )
            }
            if profile.channels.scroll {
                consumeContinuous(
                    range: ActionLayout.scroll,
                    predictions: predictions,
                    targets: targets,
                    base: base,
                    isRelativeMouse: false
                )
            }
        }
    }

    private mutating func consumeContinuous(
        range: Range<Int>,
        predictions: [Float],
        targets: [Float],
        base: Int,
        isRelativeMouse: Bool
    ) {
        for index in range {
            let prediction = predictions[base + index]
            let target = targets[base + index]
            if abs(target) > 0.0001 {
                if isRelativeMouse {
                    relativeError += Double(abs(prediction - target))
                    relativeCount += 1
                } else {
                    scrollError += Double(abs(prediction - target))
                    scrollCount += 1
                }
            } else {
                idleContinuousCount += 1
                if abs(prediction) > 0.05 { idleContinuousFalseActions += 1 }
            }
        }
    }

    func finalize(sampleCount: Int) -> (
        report: ValidationReport,
        binaryDecisionThresholds: [Float]
    ) {
        var thresholds = BinaryDecisionCalibration.defaultThresholds
        var binary = BinaryValidationAccumulator()
        var buttons = BinaryValidationAccumulator()
        var keyboard = BinaryValidationAccumulator()
        var modifiers = BinaryValidationAccumulator()
        var calibratedOutputCount = 0
        var safetyDisabledOutputCount = 0
        var supportedOutputCount = 0
        var responsiveOutputCount = 0
        for index in activeBinaryIndices.sorted() {
            let observations = binaryObservations[index] ?? []
            let calibration = BinaryDecisionCalibration.calibrate(observations)
            thresholds[index] = calibration.threshold
            if calibration.disabledForSafety,
               calibration.positiveSupport > 0 {
                safetyDisabledOutputCount += 1
            } else if calibration.usedCalibratedThreshold {
                calibratedOutputCount += 1
            }
            let isSupported = calibration.positiveSupport
                >= BinaryDecisionContract.minimumPositiveSupport
                && calibration.negativeSupport
                    >= BinaryDecisionContract.minimumNegativeSupport
            if isSupported { supportedOutputCount += 1 }
            var outputTruePositives = 0
            var outputFalsePositives = 0
            var outputFalseNegatives = 0
            var outputTrueNegatives = 0
            for observation in observations {
                let predicted = BinaryDecisionCalibration.isActive(
                    score: observation.score,
                    threshold: calibration.threshold
                )
                switch (predicted, observation.target) {
                case (true, true): outputTruePositives += 1
                case (true, false): outputFalsePositives += 1
                case (false, true): outputFalseNegatives += 1
                case (false, false): outputTrueNegatives += 1
                }
                binary.consume(predicted: predicted, target: observation.target)
                switch index {
                case ActionLayout.buttons:
                    buttons.consume(predicted: predicted, target: observation.target)
                case ActionLayout.keyboardAndShift:
                    keyboard.consume(predicted: predicted, target: observation.target)
                case ActionLayout.commandOptionControl:
                    modifiers.consume(predicted: predicted, target: observation.target)
                default:
                    break
                }
            }
            let outputRecall = Double(outputTruePositives)
                / Double(max(1, outputTruePositives + outputFalseNegatives))
            let outputPrecision = Double(outputTruePositives)
                / Double(max(1, outputTruePositives + outputFalsePositives))
            let outputFalsePositiveRate = Double(outputFalsePositives)
                / Double(max(1, outputFalsePositives + outputTrueNegatives))
            let outputIsSafe = calibration.matthewsCorrelation
                    >= BinaryDecisionContract.minimumUsefulMatthewsCorrelation
                && outputRecall >= BinaryDecisionContract.minimumUsefulRecall
                && outputPrecision
                    >= BinaryDecisionContract.minimumUsefulPrecision
                && outputFalsePositiveRate
                    <= BinaryDecisionContract.maximumUsefulFalsePositiveRate
            if isSupported && outputIsSafe {
                responsiveOutputCount += 1
            }
        }
        let report = ValidationReport(
            sampleCount: sampleCount,
            binary: binary.metrics,
            buttons: buttons.metrics,
            keyboard: keyboard.metrics,
            modifiers: modifiers.metrics,
            absoluteMouseMAE: absoluteCount > 0 ? absoluteError / Double(absoluteCount) : nil,
            activeRelativeMouseMAE: relativeCount > 0 ? relativeError / Double(relativeCount) : nil,
            activeScrollMAE: scrollCount > 0 ? scrollError / Double(scrollCount) : nil,
            idleContinuousFalseActionRate: idleContinuousCount > 0
                ? Double(idleContinuousFalseActions) / Double(idleContinuousCount)
                : nil,
            calibratedBinaryOutputCount: calibratedOutputCount,
            safetyDisabledBinaryOutputCount: safetyDisabledOutputCount,
            supportedBinaryOutputCount: supportedOutputCount,
            responsiveBinaryOutputCount: responsiveOutputCount
        )
        return (report, thresholds)
    }
}

private struct CheckpointState: Codable {
    var profileSignature: String
    var epoch: Int
    var batchOffset: Int
    var globalStep: Int
    var elapsed: Double
    var lossHistory: [Double]
    var validationHistory: [Double]
    /// Epoch averages are substantially less noisy than the last shuffled batch
    /// and drive the scheduler when honest validation is unavailable.
    var epochLossHistory: [Double]? = nil
    var learningRateHistory: [Double]? = nil
    var currentEpochWeightedLoss: Double? = nil
    var currentEpochSampleCount: Int? = nil
    var currentValidationReport: ValidationReport? = nil
    var currentValidationGlobalStep: Int? = nil
    var bestValidationReport: ValidationReport? = nil
    var currentBinaryDecisionThresholds: [Float]? = nil
    var bestBinaryDecisionThresholds: [Float]? = nil
    var schedulerBestMetric: Double? = nil
    var schedulerPlateauEpochs: Int? = nil
    var schedulerReductions: Int? = nil
    var demonstratedKeyCodes: Set<UInt16>?
    var trainingDataCoverage: TrainingDataCoverage? = nil
    /// Persisted goal for the current epoch block. Optional keeps checkpoints
    /// from earlier releases decodable.
    var epochGoal: Int? = nil
    /// Optional keeps checkpoints from builds before experience counters
    /// decodable. Restore backfills a stable step/batch estimate once.
    var experienceSeconds: Double? = nil
    /// The lowest held-out loss and its exact runnable weights are tracked
    /// independently from the latest optimizer checkpoint. Long training runs
    /// can therefore resume from the latest step without publishing a brain
    /// that has regressed after its best epoch.
    var bestValidationLoss: Double? = nil
    var bestGlobalStep: Int? = nil
    var bestEpoch: Int? = nil
    var bestTrainingLoss: Double? = nil
    var bestElapsed: Double? = nil
    var bestExperienceSeconds: Double? = nil
    /// Cursor visibility is frozen from the duration-weighted recording mix.
    var trainingShowsCursor: Bool? = nil
    var recommendedMouseMode: MouseControlMode? = nil
    /// Dataset row indices depend on segment order. The profile signature treats
    /// the selected recordings as a set, so exact resume separately enforces the
    /// order that produced the saved batch offset.
    var recordingOrder: [UUID]? = nil
    /// Optional keeps pre-curriculum checkpoints decodable. A legacy checkpoint
    /// paused midway through an epoch finishes that epoch with uniform shuffling.
    var samplingStrategy: Int? = nil
    /// Changes to held-out context or comparison semantics refresh the baseline
    /// without discarding otherwise compatible model and optimizer state.
    var validationStrategy: Int? = nil
    /// The first complete objective-v4 epoch deterministically learns through
    /// current pixels (with an exact zero recurrent state when temporal).
    /// Persisting this flag makes a mid-epoch pause resume that curriculum.
    var visualGroundingComplete: Bool? = nil
}

private struct SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

private extension Duration {
    var seconds: Double {
        let parts = components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock(); defer { unlock() }; return try body()
    }
}
