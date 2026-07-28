import CryptoKit
import Foundation
import MLX
import MLXNN
import MLXOptimizers
import QuartzCore

struct TrainingCompletion: Sendable {
    let profile: AIProfile
    let version: ModelVersionManifest
    let completed: Bool
}

enum TrainingRandomState {
    static func save(_ randomState: MLXRandom.RandomState = MLXRandom.globalState, to url: URL) throws {
        guard let state = randomState.innerState().first else { return }
        MLX.eval(state); try MLX.save(arrays: ["state": state], url: url)
    }

    static func load(_ randomState: MLXRandom.RandomState = MLXRandom.globalState, from url: URL) throws {
        guard let restored = try MLX.loadArrays(url: url)["state"], let current = randomState.innerState().first else { return }
        current._updateInternal(restored)
        MLX.eval(current)
    }
}

private struct TrainingPaused: Error {
    let completion: TrainingCompletion
}

enum SnapshotPublicationReason {
    case autosave
    case pause
    case completion

    var completed: Bool { self == .completion }
    var isAutosave: Bool { self == .autosave }
    var currentName: String {
        switch self {
        case .autosave: "Autosave"
        case .pause: "Paused Brain"
        case .completion: "Brain"
        }
    }
}

enum RunnableSnapshotSelection {
    /// Completed training recommends a brain by demonstrated execution quality.
    /// This ranking never blocks publication or Run; loss remains the
    /// deterministic tie-breaker and the independent optimizer-scheduler metric.
    static func improvesValidatedQuality(
        candidateReport: ValidationReport,
        candidateLoss: Double,
        bestReport: ValidationReport?,
        bestLoss: Double?
    ) -> Bool {
        // In-sample execution calibration is allowed to tune runtime thresholds,
        // but it must never become evidence for model selection.
        guard candidateReport.effectiveEvaluationScope.isHeldOut else { return false }
        guard let bestReport, bestReport.effectiveEvaluationScope.isHeldOut else {
            return true
        }
        switch (
            candidateReport.deploymentQualityScore,
            bestReport.deploymentQualityScore
        ) {
        case let (candidateScore?, bestScore?):
            let meaningfulDelta = 0.002
            if candidateScore > bestScore + meaningfulDelta { return true }
            if candidateScore < bestScore - meaningfulDelta { return false }
            return candidateLoss < (bestLoss ?? .infinity)
        case (_?, nil):
            return true
        case (nil, _?):
            return false
        case (nil, nil):
            return candidateLoss < (bestLoss ?? .infinity)
        }
    }

    static func usesValidatedBest(
        preferBest: Bool,
        bestGlobalStep: Int?,
        bestWeightsExist: Bool
    ) -> Bool {
        preferBest && bestGlobalStep != nil && bestWeightsExist
    }

    static func currentSnapshotHasMatchingEvaluation(
        evaluationGlobalStep: Int?,
        currentGlobalStep: Int,
        hasEvaluation: Bool
    ) -> Bool {
        hasEvaluation && evaluationGlobalStep == currentGlobalStep
    }
}

private enum TrainingSamplingContract {
    /// Versioned independently from the dataset bytes so an older checkpoint
    /// paused midway through an epoch can finish with its original exact order.
    static let salienceBalanced = 1
}

private enum TrainingValidationContract {
    /// Version 8 keeps per-control execution diagnostics and adaptive camera
    /// calibration, but makes every quality threshold advisory. Pause publishes
    /// the exact current brain and completed training ranks every finite
    /// candidate without a minimum-quality eligibility gate. A zero held-out
    /// split now receives a clearly labelled in-sample execution calibration so
    /// runtime cannot silently fall back to a larger camera deadzone.
    static let disjointContext = 8
    /// Calibration runs at every runnable zero-validation snapshot. Keep this
    /// statistically useful but cheap relative to the default 1,000 optimizer
    /// steps between autosaves.
    static let maximumTrainingCalibrationRows = 2_048
}

final class TrainingEngine: @unchecked Sendable {
    typealias MetricsHandler = @Sendable (TrainingMetrics, String) -> Void
    typealias CompletionHandler = @Sendable (Result<TrainingCompletion, Error>) -> Void
    private typealias ValidationFunction = @Sendable ([MLXArray]) -> [MLXArray]

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
        let split = splitIndices(
            dataset: dataset,
            fraction: profile.training.validationSplit,
            seed: profile.training.seed,
            historyLength: PolicyInputContract.actionHistoryLength,
            visualMemoryMaximumLag: profile.training.visualMemoryMaximumLag,
            channels: profile.channels,
            restrictions: profile.effectiveRestrictions
        )
        guard !split.train.isEmpty else { throw AgentTrainerError.noData }
        let batchSize = max(1, profile.training.batchSize)
        let stepsPerEpoch = Int(ceil(Double(split.train.count) / Double(batchSize)))
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
        var balancePlan = dataset.binaryBalancePlan(
            at: split.train,
            channels: profile.channels,
            restrictions: profile.effectiveRestrictions
        )
        let continuousBalancePlan = dataset.continuousBalancePlan(
            at: split.train,
            channels: profile.channels
        )
        balancePlan.report.continuousOutputs = continuousBalancePlan.outputs
        var actionLossWeightValues = balancePlan.positiveWeights
        for output in continuousBalancePlan.outputs {
            actionLossWeightValues[output.outputIndex] = continuousBalancePlan.activeWeights[output.outputIndex]
        }
        let validationSampleLimit = Self.recommendedValidationSampleLimit(
            total: split.validation.count,
            batchSize: profile.training.batchSize,
            segmentCount: dataset.segmentCount(at: split.validation)
        )
        let validationEvaluationIndices = dataset.representativeValidationIndices(
            from: split.validation,
            limit: validationSampleLimit,
            channels: profile.channels,
            restrictions: profile.effectiveRestrictions
        )
        // Building representative indices scans target rows. Avoid that work
        // entirely when a real held-out split already provides the runtime
        // calibration attached to runnable snapshots.
        let trainingCalibrationEvaluationIndices: [Int] = {
            guard split.validation.isEmpty else { return [] }
            let sampleLimit = min(
                TrainingValidationContract.maximumTrainingCalibrationRows,
                Self.recommendedValidationSampleLimit(
                    total: split.train.count,
                    batchSize: profile.training.batchSize,
                    segmentCount: dataset.segmentCount(at: split.train)
                )
            )
            return dataset.representativeValidationIndices(
                from: split.train,
                limit: sampleLimit,
                channels: profile.channels,
                restrictions: profile.effectiveRestrictions
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
        let signature = try profileSignature(profile, recordings: recordings)
        let inputSummaries = try recordings.map { recording in
            let url = recording.directory.appendingPathComponent(recording.manifest.eventFile)
            return try InputEventReader.summarize(
                url: url,
                previewLimit: 0,
                globalRect: recording.manifest.globalRect.cgRect
            )
        }
        let demonstratedKeys = balancePlan.supportedKeyCodes
        if !balancePlan.report.ignoredOutputs.isEmpty {
            let details = balancePlan.report.ignoredOutputs.map {
                let duration = $0.activeDurationSeconds ?? 0
                return "\(ActionLayout.diagnosticName(for: $0.outputIndex)): \($0.positiveSamples) positive frames, \($0.pressEpisodes)/\(BinaryBalanceContract.minimumPressEpisodes) presses, \(duration.formatted(.number.precision(.fractionLength(2))))/\(BinaryBalanceContract.minimumHeldDurationSeconds.formatted(.number.precision(.fractionLength(1)))) seconds held"
            }.joined(separator: "; ")
            AppLog.write(
                .warning,
                category: "Training",
                "Ignored under-demonstrated binary controls",
                details: details
            )
        }
        let mouseEvidence = inputSummaries.reduce(into: InputEventReader.MouseModeEvidence()) { result, summary in
            result.include(summary.mouse)
        }
        let recommendedMouseMode = mouseEvidence.recommendedMode
        let recordingOrder = recordings.map(\.id)
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
            samplingStrategy: TrainingSamplingContract.salienceBalanced,
            validationStrategy: TrainingValidationContract.disjointContext
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
        // The raw event stream is authoritative, including taps shorter than one
        // action interval. Refresh after restoring an older checkpoint.
        state.demonstratedKeyCodes = demonstratedKeys
        // Legacy checkpoints did not persist sample-to-recording order. Preserve
        // their current epoch order, then make every subsequent save enforce it.
        if state.recordingOrder == nil { state.recordingOrder = recordingOrder }
        if state.samplingStrategy == nil, state.batchOffset == 0 {
            state.samplingStrategy = TrainingSamplingContract.salienceBalanced
        }
        state.recommendedMouseMode = recommendedMouseMode
        let cursorDurations = recordings.reduce(into: (shown: 0.0, total: 0.0)) { result, recording in
            let start = max(0, min(recording.manifest.duration, recording.manifest.trimStart))
            let end = max(start, min(recording.manifest.duration, recording.manifest.trimEnd ?? recording.manifest.duration))
            let duration = end - start
            result.total += duration
            if recording.manifest.capture.showsCursor { result.shown += duration }
        }
        state.trainingShowsCursor = cursorDurations.total > 0
            ? cursorDurations.shown >= cursorDurations.total / 2
            : recordings.filter { $0.manifest.capture.showsCursor }.count * 2 >= recordings.count

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
            state.bestValidationReport = nil
            let calibrationRemainsExact =
                !restore.captureValidationBaseline
                && state.currentValidationReport?.effectiveEvaluationScope
                    == .trainingCalibration
                && state.currentEvaluationGlobalStep == state.globalStep
            if !calibrationRemainsExact {
                state.currentValidationReport = nil
                state.currentEvaluationGlobalStep = nil
            }
            state.schedulerBestMetric = nil
            state.schedulerPlateauEpochs = 0
        }

        let validationFunction: ValidationFunction = compile(inputs: [model]) { (arrays: [MLXArray]) -> [MLXArray] in
            let classWeights = MLXArray(actionLossWeightValues, [ActionLayout.count])
            let logits = model(images: arrays[0], history: arrays[1])
            let losses = model.lossComponents(
                logits: logits,
                history: arrays[1],
                targets: arrays[2],
                positiveWeights: classWeights,
                previousTargets: arrays[3]
            )
            let unavailable = MLXArray(Float.nan, dtype: model.dtype)
            return [
                losses.total,
                model.activatedPredictions(logits: logits),
                losses.mouse ?? unavailable,
                losses.buttons ?? unavailable,
                losses.scroll ?? unavailable,
                losses.keyboard ?? unavailable,
                losses.modifiers ?? unavailable,
            ]
        }

        /// Validation zero means "train on every row," not "run without a
        /// deployment contract." Score a bounded representative training subset
        /// only to calibrate the exact saved tensors' live thresholds. Its scope
        /// prevents this resubstitution report from becoming a held-out score,
        /// scheduler metric, or best-brain selection signal.
        func evaluateTrainingCalibration() throws -> ValidationEvaluation {
            let calibration = evaluate(
                model: model,
                dataset: dataset,
                indices: trainingCalibrationEvaluationIndices,
                profile: profile,
                actionLossWeightValues: actionLossWeightValues,
                balanceReport: balancePlan.report,
                validationFunction: validationFunction,
                scope: .trainingCalibration
            )
            guard calibration.loss.isFinite else {
                throw AgentTrainerError.model(
                    "Execution calibration became numerically unstable, so the current brain was not published. Lower the learning rate or reset this brain's learning state."
                )
            }
            return calibration
        }

        // A warm-started brain's saved validation number may belong to a
        // different recording set, split, target contract, or loss definition.
        // Re-score those exact selected weights on this run's held-out examples
        // before comparing any fine-tuned epoch against them.
        if restore.captureValidationBaseline, !split.validation.isEmpty {
            let baseline = evaluate(
                model: model,
                dataset: dataset,
                indices: validationEvaluationIndices,
                profile: profile,
                actionLossWeightValues: actionLossWeightValues,
                balanceReport: balancePlan.report,
                validationFunction: validationFunction
            )
            let baselineValidationLoss = baseline.loss
            guard baselineValidationLoss.isFinite else {
                throw AgentTrainerError.model("The selected brain produced an invalid validation baseline on the current recordings.")
            }
            let baselineTrainingLoss: Double
            if let saved = state.lossHistory.last, saved.isFinite {
                baselineTrainingLoss = saved
            } else {
                // Objective upgrades deliberately discard incomparable loss
                // history. Score one training batch so a best baseline that is
                // published before the next full epoch still carries a value
                // produced by its own exact weights under the new objective.
                let baselineTraining = evaluate(
                    model: model,
                    dataset: dataset,
                    indices: Array(split.train.prefix(batchSize)),
                    profile: profile,
                    actionLossWeightValues: actionLossWeightValues,
                    balanceReport: balancePlan.report,
                    validationFunction: validationFunction
                ).loss
                guard baselineTraining.isFinite else {
                    throw AgentTrainerError.model("The selected brain produced an invalid training-objective baseline on the current recordings.")
                }
                baselineTrainingLoss = baselineTraining
            }
            state.validationHistory = [baselineValidationLoss]
            state.currentValidationReport = baseline.report
            state.currentEvaluationGlobalStep = state.globalStep
            // Quality thresholds are advisory. A finite, structurally valid
            // selected brain is always eligible to remain the user's active
            // recommendation, even when its report contains warnings.
            state.bestValidationLoss = baselineValidationLoss
            state.bestGlobalStep = state.globalStep
            state.bestEpoch = state.batchOffset > 0 ? state.epoch + 1 : state.epoch
            state.bestTrainingLoss = baselineTrainingLoss
            state.bestElapsed = state.elapsed
            state.bestExperienceSeconds = state.experienceSeconds
            state.bestValidationReport = baseline.report
            state.schedulerBestMetric = baselineValidationLoss
            state.schedulerPlateauEpochs = 0
        }

        let trainingStep = compile(inputs: [model, optimizer, randomState], outputs: [model, optimizer, randomState]) { (arrays: [MLXArray]) -> [MLXArray] in
            // Capture the Sendable Swift values and materialize the constant while
            // MLX traces the graph. MLXArray itself is intentionally non-Sendable.
            let classWeights = MLXArray(actionLossWeightValues, [ActionLayout.count])
            let result = valueAndGrad(model: model) { model, arrays in
                [model.loss(
                    images: arrays[0],
                    history: arrays[1],
                    targets: arrays[2],
                    positiveWeights: classWeights,
                    previousTargets: arrays[3]
                )]
            }(model, arrays)
            let clipped = clipGradNorm(gradients: result.1, maxNorm: 1).0
            optimizer.update(model: model, gradients: clipped, targetType: model.dtype)
            return [result.0[0]]
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
        if restore.captureValidationBaseline, !split.validation.isEmpty {
            try await saveCheckpoint(profile: profile, model: model, optimizer: optimizer, randomState: randomState, state: state, captureBest: true)
        }

        let initialMemory = Memory.snapshot()
        let verifiedKeyboardControls = balancePlan.report.outputs.count {
            $0.isSupported && (ActionLayout.keyboardAndShift.contains($0.outputIndex)
                || ActionLayout.commandOptionControl.contains($0.outputIndex))
        }
        let ignoredKeyboardControls = balancePlan.report.ignoredOutputs.count {
            ActionLayout.keyboardAndShift.contains($0.outputIndex)
                || ActionLayout.commandOptionControl.contains($0.outputIndex)
        }
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
            balanceReport: balancePlan.report,
            effectiveLearningRate: Double(optimizer.effectiveLearningRate()),
            learningRateScale: Double(optimizer.learningRateScale),
            samplesPerSecond: 0,
            elapsed: state.elapsed,
            experienceElapsed: state.experienceSeconds ?? 0,
            lossHistory: Array(state.lossHistory.suffix(4_096)),
            epochLossHistory: Array((state.epochLossHistory ?? []).suffix(1_024)),
            validationHistory: Array(state.validationHistory.suffix(1_024)),
            learningRateHistory: Array((state.learningRateHistory ?? []).suffix(1_024)),
            mlxActiveMemory: initialMemory.activeMemory,
            mlxCacheMemory: initialMemory.cacheMemory,
            mlxPeakMemory: initialMemory.peakMemory
        ), "\(restore.status) • \(verifiedKeyboardControls) evidence-verified keys/modifiers\(ignoredKeyboardControls > 0 ? " • \(ignoredKeyboardControls) awaiting more taps or held time" : "") • continuing to epoch \(targetEpoch)")

        var lastMetricsPublish = CACurrentMediaTime() - 1
        var lastRateTime = CACurrentMediaTime()
        var samplesSinceRate = 0

        trainingLoop: for epoch in state.epoch..<targetEpoch {
            let epochSeed = profile.training.seed &+ UInt64(epoch) &* 0x9E3779B97F4A7C15
            let order = state.samplingStrategy == TrainingSamplingContract.salienceBalanced
                ? trainingOrder(
                    dataset: dataset,
                    indices: split.train,
                    batchSize: batchSize,
                    seed: epochSeed,
                    profile: profile,
                    precomputedSalientIndices: salientTrainingIndices
                )
                : shuffled(split.train, seed: epochSeed)
            var offset = epoch == state.epoch ? state.batchOffset : 0
            var epochWeightedLoss = offset > 0 ? state.currentEpochWeightedLoss ?? 0 : 0
            var epochSampleCount = offset > 0 ? state.currentEpochSampleCount ?? 0 : 0
            var prefetchedBatch: PreparedBatch?
            while offset < order.count {
                try Task.checkCancellation()
                if lock.withLock({ stopRequested }) { throw CancellationError() }
                let end = min(order.count, offset + batchSize)
                let batch = Array(order[offset..<end])
                let prepared = prefetchedBatch ?? prepareBatch(dataset: dataset, indices: batch, profile: profile)
                prefetchedBatch = nil
                let result: (loss: Double, next: PreparedBatch?) = autoreleasepool {
                    let arrays = materializeBatch(prepared, profile: profile)
                    let lossArray = trainingStep(arrays)[0]
                    // Start Metal immediately, then gather the next mapped batch
                    // while the current optimizer graph is executing. The final
                    // eval still waits for every model and optimizer output, so
                    // numerical order and exact-resume semantics are unchanged.
                    MLX.asyncEval(lossArray, model.parameters(), optimizer.stateArrays())
                    let next: PreparedBatch?
                    if end < order.count {
                        let nextEnd = min(order.count, end + batchSize)
                        next = prepareBatch(dataset: dataset, indices: Array(order[end..<nextEnd]), profile: profile)
                    } else {
                        next = nil
                    }
                    MLX.eval(lossArray, model.parameters(), optimizer.stateArrays())
                    return (Double(lossArray.item(Float.self)), next)
                }
                let loss = result.loss
                guard loss.isFinite else {
                    throw AgentTrainerError.model("Training became numerically unstable before this step could be saved. Lower the learning rate or reset this brain's learning state.")
                }
                prefetchedBatch = result.next
                guard state.globalStep < Int.max else {
                    throw AgentTrainerError.model("The restored optimizer step counter is invalid and cannot be advanced safely.")
                }
                state.globalStep += 1
                samplesSinceRate += batch.count
                state.experienceSeconds = (state.experienceSeconds ?? 0) + Double(batch.count) / max(0.0001, dataset.manifest.actionFPS)
                offset = end
                state.epoch = epoch
                state.batchOffset = offset
                state.lossHistory.append(loss)
                if state.lossHistory.count > 8_192 { state.lossHistory.removeFirst(4_096) }
                epochWeightedLoss += loss * Double(batch.count)
                epochSampleCount += batch.count
                state.currentEpochWeightedLoss = epochWeightedLoss
                state.currentEpochSampleCount = epochSampleCount

                let now = CACurrentMediaTime()
                if now - lastMetricsPublish >= 0.25 || offset == order.count {
                    lastMetricsPublish = now
                    let elapsed = baseElapsed + started.duration(to: .now).seconds
                    let recentSeconds = max(0.001, now - lastRateTime)
                    lastRateTime = now
                    let memory = Memory.snapshot()
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
                        balanceReport: balancePlan.report,
                        effectiveLearningRate: Double(optimizer.effectiveLearningRate()),
                        learningRateScale: Double(optimizer.learningRateScale),
                        samplesPerSecond: Double(samplesSinceRate) / recentSeconds,
                        elapsed: elapsed,
                        experienceElapsed: state.experienceSeconds ?? 0,
                        lossHistory: Array(state.lossHistory.suffix(4_096)),
                        epochLossHistory: Array((state.epochLossHistory ?? []).suffix(1_024)),
                        validationHistory: Array(state.validationHistory.suffix(1_024)),
                        learningRateHistory: Array((state.learningRateHistory ?? []).suffix(1_024)),
                        mlxActiveMemory: memory.activeMemory,
                        mlxCacheMemory: memory.cacheMemory,
                        mlxPeakMemory: memory.peakMemory
                    )
                    samplesSinceRate = 0
                    metrics(report, "Pipelined training on Apple-silicon GPU")
                }

                let shouldCheckpoint = state.globalStep >= nextAutosaveStep
                let shouldPause = lock.withLock { pauseRequested }
                if shouldCheckpoint || shouldPause {
                    if split.validation.isEmpty {
                        let calibration = try evaluateTrainingCalibration()
                        state.currentValidationReport = calibration.report
                        state.currentEvaluationGlobalStep = state.globalStep
                    }
                    state.elapsed = baseElapsed + started.duration(to: .now).seconds
                    try await saveCheckpoint(profile: profile, model: model, optimizer: optimizer, randomState: randomState, state: state)
                    latestSnapshot = try await publishSnapshot(
                        profile: profile,
                        state: state,
                        reason: shouldPause ? .pause : .autosave,
                        preferBest: false
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
            // A legacy checkpoint paused inside this epoch used the historical
            // uniform order above. Upgrade only after that exact epoch finishes.
            if state.samplingStrategy == nil {
                state.samplingStrategy = TrainingSamplingContract.salienceBalanced
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
                let validation = evaluate(
                    model: model,
                    dataset: dataset,
                    indices: validationEvaluationIndices,
                    profile: profile,
                    actionLossWeightValues: actionLossWeightValues,
                    balanceReport: balancePlan.report,
                    validationFunction: validationFunction
                )
                let validationLoss = validation.loss
                guard validationLoss.isFinite else {
                    throw AgentTrainerError.model("Validation became numerically unstable, so the current epoch was not published. Lower the learning rate or reset this brain's learning state.")
                }
                monitorMetric = validationLoss
                state.currentValidationReport = validation.report
                state.currentEvaluationGlobalStep = state.globalStep
                state.validationHistory.append(validationLoss)
                if state.validationHistory.count > 2_048 { state.validationHistory.removeFirst(1_024) }
                let improvesRunnableQuality = RunnableSnapshotSelection.improvesValidatedQuality(
                    candidateReport: validation.report,
                    candidateLoss: validationLoss,
                    bestReport: state.bestValidationReport,
                    bestLoss: state.bestValidationLoss
                )
                if improvesRunnableQuality {
                    state.bestValidationLoss = validationLoss
                    state.bestGlobalStep = state.globalStep
                    state.bestEpoch = state.epoch
                    state.bestTrainingLoss = epochTrainingLoss
                    state.bestElapsed = baseElapsed + started.duration(to: .now).seconds
                    state.bestExperienceSeconds = state.experienceSeconds
                    state.bestValidationReport = validation.report
                    capturedBest = true
                    if validation.report.hasBinaryRecallCollapse
                        || validation.report.hasContinuousExecutionFailure {
                        validationSelectionStatus = "Saved the current recommended brain with advisory execution-quality warnings"
                    }
                } else {
                    validationSelectionStatus = "Validated epoch \(state.epoch); retained the brain with stronger demonstrated execution quality"
                }
            } else if state.currentEvaluationGlobalStep != state.globalStep {
                let calibration = try evaluateTrainingCalibration()
                state.currentValidationReport = calibration.report
                state.currentEvaluationGlobalStep = state.globalStep
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
                balanceReport: balancePlan.report,
                effectiveLearningRate: Double(optimizer.effectiveLearningRate()),
                learningRateScale: Double(optimizer.learningRateScale),
                samplesPerSecond: 0,
                elapsed: state.elapsed,
                experienceElapsed: state.experienceSeconds ?? 0,
                lossHistory: Array(state.lossHistory.suffix(4_096)),
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
        if split.validation.isEmpty,
           state.currentEvaluationGlobalStep != state.globalStep {
            let calibration = try evaluateTrainingCalibration()
            state.currentValidationReport = calibration.report
            state.currentEvaluationGlobalStep = state.globalStep
        }
        if latestSnapshot?.version.globalStep != state.globalStep {
            try await saveCheckpoint(profile: profile, model: model, optimizer: optimizer, randomState: randomState, state: state)
        }
        let final = try await publishSnapshot(
            profile: profile,
            state: state,
            reason: .completion,
            preferBest: !split.validation.isEmpty
        )
        return final
    }

    private func prepareBatch(dataset: CachedDataset, indices: [Int], profile: AIProfile) -> PreparedBatch {
        let b = indices.count
        let gathered = dataset.trainingBatch(
            at: indices,
            historyLength: PolicyInputContract.actionHistoryLength,
            visualMemoryLags: profile.training.visualMemoryLags
        )
        var targetData = gathered.targets
        var previousTargetData = gathered.previousTargets
        let restrictions = profile.effectiveRestrictions
        targetData.withUnsafeMutableBytes { raw in
            let values = raw.bindMemory(to: Float.self)
            ActionLayout.sanitizeTrainingRows(values, rowCount: b, channels: profile.channels, restrictions: restrictions)
        }
        var historyData = gathered.history
        historyData.withUnsafeMutableBytes { raw in
            let values = raw.bindMemory(to: Float.self)
            ActionLayout.sanitizeTrainingRows(
                values,
                rowCount: b * PolicyInputContract.placeholderHistoryRows,
                channels: profile.channels,
                restrictions: restrictions
            )
        }
        previousTargetData.withUnsafeMutableBytes { raw in
            let values = raw.bindMemory(to: Float.self)
            ActionLayout.sanitizeTrainingRows(values, rowCount: b, channels: profile.channels, restrictions: restrictions)
        }
        return PreparedBatch(
            count: b,
            packedObservations: gathered.currentObservations,
            visualMemory: PackedVisualMemoryContext(
                packedFrames: gathered.visualMemoryObservations,
                availability: gathered.visualMemoryAvailability
            ),
            history: historyData,
            targets: targetData,
            previousTargets: previousTargetData
        )
    }

    private func materializeBatch(_ batch: PreparedBatch, profile: AIProfile) -> [MLXArray] {
        return [
            VisionPreprocessor.mlxVisualMemoryTensor(
                current: batch.packedObservations,
                memory: batch.visualMemory,
                batch: batch.count,
                frameCount: profile.training.effectiveVisualMemoryFrames,
                spec: profile.preprocessing
            ),
            MLXArray(batch.history, [batch.count, PolicyInputContract.placeholderHistoryRows, ActionLayout.count], type: Float.self),
            MLXArray(batch.targets, [batch.count, ActionLayout.count], type: Float.self),
            MLXArray(batch.previousTargets, [batch.count, ActionLayout.count], type: Float.self)
        ]
    }

    private func makeBatch(dataset: CachedDataset, indices: [Int], profile: AIProfile) -> [MLXArray] {
        materializeBatch(prepareBatch(dataset: dataset, indices: indices, profile: profile), profile: profile)
    }

    private func evaluate(
        model: AgentPolicy,
        dataset: CachedDataset,
        indices: [Int],
        profile: AIProfile,
        actionLossWeightValues: [Float],
        balanceReport: TrainingBalanceReport,
        validationFunction: ValidationFunction,
        scope: ModelEvaluationScope = .heldOut
    ) -> ValidationEvaluation {
        model.train(false)
        defer { model.train(true) }
        var weightedLoss = 0.0
        var evaluated = 0
        var componentTotals = [Double](repeating: 0, count: 5)
        var componentSamples = [Int](repeating: 0, count: 5)
        var report = ValidationAccumulator(
            profile: profile,
            activeBinaryIndices: Set(ActionLayout.learnableBinaryIndices(
                channels: profile.channels,
                restrictions: profile.effectiveRestrictions
            ).filter { actionLossWeightValues.indices.contains($0) && actionLossWeightValues[$0] > 0 })
        )
        let batchSize = max(1, profile.training.batchSize)
        for start in Swift.stride(from: 0, to: indices.count, by: batchSize) {
            let end = min(indices.count, start + batchSize)
            let batch = Array(indices[start..<end])
            let arrays = makeBatch(dataset: dataset, indices: batch, profile: profile)
            let validation = validationFunction(arrays)
            guard validation.count >= 7 else { continue }
            let loss = validation[0]
            let predictions = validation[1]
            // Materialize every compiled output behind one barrier. Reading
            // five scalar components one-by-one would add avoidable CPU/GPU
            // synchronization even though they share the same forward graph.
            MLX.eval(validation + [arrays[2]])
            weightedLoss += Double(loss.item(Float.self)) * Double(batch.count)
            evaluated += batch.count
            for component in 0..<5 {
                let value = Double(validation[component + 2].item(Float.self))
                if value.isFinite {
                    componentTotals[component] += value * Double(batch.count)
                    componentSamples[component] += batch.count
                }
            }
            report.consume(
                predictions: predictions.asArray(Float.self),
                targets: arrays[2].asArray(Float.self),
                rowCount: batch.count
            )
        }
        func component(_ index: Int) -> Double? {
            componentSamples[index] > 0
                ? componentTotals[index] / Double(componentSamples[index])
                : nil
        }
        let breakdown = ValidationLossBreakdown(
            mouse: component(0),
            buttons: component(1),
            scroll: component(2),
            keyboard: component(3),
            modifiers: component(4)
        )
        return ValidationEvaluation(
            loss: weightedLoss / Double(max(1, evaluated)),
            report: report.finalize(
                sampleCount: evaluated,
                lossBreakdown: breakdown,
                trainingBalance: balanceReport,
                scope: scope
            )
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
        historyLength: Int? = nil,
        visualMemoryMaximumLag: Int,
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
            let segmentStatistics = segments.map { segment in
                dataset.binaryTargetStatistics(
                    at: segment.start..<(segment.start + segment.count)
                )
            }
            var remainingStatistics = BinaryTargetStatistics.zero
            for statistics in segmentStatistics {
                remainingStatistics.add(statistics)
            }
            let totalStatistics = remainingStatistics
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
                    let candidate = segmentStatistics[segmentIndex]
                    return !learnableBinaryOutputs.contains { output in
                        let totalPositives = totalStatistics.positives[output]
                        guard totalPositives > 0 else { return false }
                        let remainingPositives = remainingStatistics.positives[output]
                            - candidate.positives[output]
                        return !BinaryBalanceContract.preservesTrainingEvidence(
                            outputIndex: output,
                            totalPositiveSamples: totalPositives,
                            totalPressEpisodes: totalStatistics.pressEpisodes[output],
                            trainingPositiveSamples: remainingPositives,
                            trainingPressEpisodes: remainingStatistics.pressEpisodes[output]
                                - candidate.pressEpisodes[output],
                            actionFPS: dataset.manifest.actionFPS
                        )
                    }
                }
                guard let selected = candidates.min(by: { lhs, rhs in
                    func score(_ index: Int) -> Double {
                        let distance = Double(abs(segments[index].count - desiredRows)) / Double(desiredRows)
                        let newCoverage = learnableBinaryOutputs.count { output in
                            validationPositiveCounts[output] == 0
                                && segmentStatistics[index].positives[output] > 0
                        }
                        return distance - min(0.5, Double(newCoverage) * 0.05)
                    }
                    let left = score(lhs), right = score(rhs)
                    if abs(left - right) > 1e-12 { return left < right }
                    return (seededRank[lhs] ?? 0) < (seededRank[rhs] ?? 0)
                }) else { break }
                validationSegments.insert(selected)
                validationRows += segments[selected].count
                remainingStatistics.subtract(segmentStatistics[selected])
                for output in learnableBinaryOutputs {
                    validationPositiveCounts[output] += segmentStatistics[selected].positives[output]
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
        guard let trainingEnd = dataset.minimumTrainingEndPreservingBinaryEvidence(
                  proposedEnd: proposedValidationStart,
                  outputs: learnableBinaryOutputs
              ),
              let validationStart = dataset.firstDisjointValidationIndex(
                trainingEnd: trainingEnd,
                proposedStart: max(proposedValidationStart, trainingEnd),
                historyLength: historyLength,
                visualMemoryMaximumLag: visualMemoryMaximumLag
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
            let recordings: [RecordingManifest]
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
        resumeCompatibleTraining.historyLength = PolicyInputContract.actionHistoryLength
        // With zero/one remembered frame there are no spaced or "older"
        // slots, so stride and structured memory dropout cannot affect the
        // training graph. Normalize them out of exact-resume identity.
        if resumeCompatibleTraining.effectiveVisualMemoryFrames <= 1 {
            resumeCompatibleTraining.visualMemoryStride = 1
            resumeCompatibleTraining.visualMemoryDropout = 0
        }
        var normalizedChannels = profile.channels
        normalizedChannels.absoluteMouse = profile.channels.mouseMovement
        normalizedChannels.relativeMouse = profile.channels.mouseMovement
        let manifests = recordings.map(\.manifest).sorted { $0.id.uuidString < $1.id.uuidString }
        let identity = TrainingIdentity(
            trainingDataSchema: TrainingDataContract.schemaVersion,
            trainingObjectiveSchema: TrainingObjectiveContract.schemaVersion,
            preprocessing: profile.preprocessing,
            channels: normalizedChannels,
            training: resumeCompatibleTraining,
            recordings: manifests,
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

    private func publishSnapshot(
        profile: AIProfile,
        state: CheckpointState,
        reason: SnapshotPublicationReason,
        preferBest: Bool = false
    ) async throws -> TrainingCompletion {
        let checkpoint = await WorkspaceStore.shared.checkpointDirectory(profileID: profile.id)
        let bestWeights = checkpoint.appendingPathComponent("best.weights.safetensors")
        // Pause and periodic autosave deliberately publish the exact current
        // checkpoint. A report is attached only when it was computed from those
        // exact tensors: held-out at an epoch boundary or bounded in-sample
        // calibration for a zero-validation runnable snapshot.
        let usesBest = RunnableSnapshotSelection.usesValidatedBest(
            preferBest: preferBest,
            bestGlobalStep: state.bestGlobalStep,
            bestWeightsExist: FileManager.default.fileExists(atPath: bestWeights.path)
        )
        let currentHasMatchingEvaluation =
            RunnableSnapshotSelection.currentSnapshotHasMatchingEvaluation(
                evaluationGlobalStep: state.currentEvaluationGlobalStep,
                currentGlobalStep: state.globalStep,
                hasEvaluation: state.currentValidationReport != nil
            )
        let currentEpoch = state.batchOffset > 0 ? state.epoch + 1 : state.epoch
        let displayedEpoch = usesBest ? state.bestEpoch ?? currentEpoch : currentEpoch
        let displayedStep = usesBest ? state.bestGlobalStep ?? state.globalStep : state.globalStep
        let displayedLoss = usesBest
            ? state.bestTrainingLoss ?? state.epochLossHistory?.last ?? state.lossHistory.last ?? 0
            : state.epochLossHistory?.last ?? state.lossHistory.last ?? 0
        let displayedValidationLoss = usesBest
            ? state.bestValidationLoss
            : currentHasMatchingEvaluation ? state.validationHistory.last : nil
        let displayedValidationReport = usesBest
            ? state.bestValidationReport
            : currentHasMatchingEvaluation ? state.currentValidationReport : nil
        let version = ModelVersionManifest(
            id: UUID(),
            name: usesBest
                ? "Best Brain • Epoch \(displayedEpoch) • Step \(displayedStep)"
                : "\(reason.currentName) • Epoch \(displayedEpoch) • Step \(displayedStep)",
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
            isAutosave: reason.isAutosave,
            demonstratedKeyCodes: state.demonstratedKeyCodes ?? [],
            relativeMouseScale: GameCameraContract.deltaScale,
            trainingDataSchema: TrainingDataContract.schemaVersion,
            trainingObjectiveSchema: TrainingObjectiveContract.schemaVersion,
            trainingDurationSeconds: usesBest ? state.bestElapsed : state.elapsed,
            experienceDurationSeconds: usesBest ? state.bestExperienceSeconds : state.experienceSeconds ?? 0,
            trainingShowsCursor: state.trainingShowsCursor,
            recommendedMouseMode: state.recommendedMouseMode,
            validationReport: displayedValidationReport
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
        if removed > 0 {
            AppLog.write(category: "Training", "Pruned old autosaves", details: "\(profile.name): removed \(removed), kept the newest 10")
        }
        return TrainingCompletion(
            profile: updated,
            version: version,
            completed: reason.completed
        )
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
            if restored.profileSignature == expectedSignature,
               recordingOrderMatches,
               samplingStrategyIsKnown {
                let validationNeedsRefresh = restored.validationStrategy.map {
                    $0 != TrainingValidationContract.disjointContext
                } ?? legacyValidationNeedsRefresh
                try model.loadWeights(from: directory.appendingPathComponent("weights.safetensors"))
                try optimizer.load(from: directory.appendingPathComponent("optimizer.safetensors"))
                let randomStateURL = directory.appendingPathComponent("random.safetensors")
                if FileManager.default.fileExists(atPath: randomStateURL.path) { try TrainingRandomState.load(randomState, from: randomStateURL) }
                state = restored
                state.validationStrategy = TrainingValidationContract.disjointContext
                return CheckpointRestore(
                    status: validationNeedsRefresh
                        ? "Restored exact optimizer state; evaluation will be refreshed for the current split"
                        : "Restored exact checkpoint; compiling resumed MLX graph",
                    captureValidationBaseline: validationNeedsRefresh
                )
            }
        }

        // A data- or objective-contract upgrade should never make a trained AI
        // silently fall back to random weights. When shapes still match, keep
        // the active runnable brain and begin a fresh optimizer/batch sequence
        // on the new targets/loss. The old version remains immutable and
        // runnable.
        if let versionID = profile.activeVersionID,
           let version = await WorkspaceStore.shared.version(profileID: profile.id, versionID: versionID),
           version.schemaVersion == ModelContract.schemaVersion,
           LearnedBrainContract(
               preprocessing: version.preprocessing,
               visualMemoryFrames: version.training.effectiveVisualMemoryFrames,
               visualMemoryStride: version.training.learnedVisualMemoryStride,
               architecture: LearnedBrainArchitectureContract(version.training.architecture)
           ) == profile.learnedBrainContract {
            let versionDirectory = await WorkspaceStore.shared.versionDirectory(profileID: profile.id, versionID: versionID)
            try model.loadWeights(from: versionDirectory.appendingPathComponent(version.weightsFile))
            state.epoch = max(0, version.epoch ?? 0)
            state.batchOffset = 0
            state.globalStep = max(0, version.globalStep)
            state.elapsed = max(0, version.trainingDurationSeconds ?? profile.trainingProgress?.trainingDurationSeconds ?? 0)
            state.experienceSeconds = version.experienceDurationSeconds ?? profile.trainingProgress?.experienceDurationSeconds
            let usesCurrentObjective = version.trainingObjectiveSchema == TrainingObjectiveContract.schemaVersion
            state.lossHistory = usesCurrentObjective ? [version.trainingLoss] : []
            state.epochLossHistory = usesCurrentObjective ? [version.trainingLoss] : []
            state.validationHistory = usesCurrentObjective ? (version.validationLoss.map { [$0] } ?? []) : []
            state.currentValidationReport = usesCurrentObjective ? version.validationReport : nil
            state.currentEvaluationGlobalStep =
                usesCurrentObjective && version.validationReport != nil
                ? state.globalStep
                : nil
            let restartReason = usesCurrentObjective
                ? "optimizer restarted safely"
                : "objective v\(TrainingObjectiveContract.schemaVersion) recalibration; optimizer restarted safely"
            if usesCurrentObjective, let validationLoss = version.validationLoss, validationLoss.isFinite {
                state.bestValidationLoss = validationLoss
                state.bestGlobalStep = state.globalStep
                state.bestEpoch = state.epoch
                state.bestTrainingLoss = version.trainingLoss
                state.bestElapsed = state.elapsed
                state.bestExperienceSeconds = state.experienceSeconds
                state.bestValidationReport = version.validationReport
                return CheckpointRestore(status: "Loaded the selected best brain; \(restartReason)", captureValidationBaseline: true)
            }
            return CheckpointRestore(status: "Loaded the active brain for fine-tuning; \(restartReason)", captureValidationBaseline: true)
        }
        return CheckpointRestore(status: "Compiling fused MLX training graph on Apple GPU", captureValidationBaseline: false)
    }

}

private struct CheckpointRestore {
    var status: String
    var captureValidationBaseline: Bool
}

private struct PreparedBatch {
    let count: Int
    let packedObservations: Data
    let visualMemory: PackedVisualMemoryContext
    let history: Data
    let targets: Data
    let previousTargets: Data
}

private struct ValidationEvaluation {
    var loss: Double
    var report: ValidationReport
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

    mutating func add(_ metrics: BinaryValidationMetrics) {
        truePositives += metrics.truePositives
        falsePositives += metrics.falsePositives
        falseNegatives += metrics.falseNegatives
        trueNegatives += metrics.trueNegatives
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

/// A fixed-resolution score histogram makes per-control threshold calibration
/// O(rows x outputs), not O(rows x outputs x thresholds), and keeps validation
/// memory constant even for very large held-out sets.
struct BinaryScoreHistogram: Sendable {
    static let resolution = 100
    private(set) var positiveBins = [Int](repeating: 0, count: resolution + 1)
    private(set) var negativeBins = [Int](repeating: 0, count: resolution + 1)

    mutating func consume(probability: Float, target: Bool) {
        guard probability.isFinite else { return }
        let clamped = min(1, max(0, probability))
        let bin = min(Self.resolution, max(0, Int(floor(clamped * Float(Self.resolution)))))
        if target {
            positiveBins[bin] += 1
        } else {
            negativeBins[bin] += 1
        }
    }

    var positiveSupport: Int { positiveBins.reduce(0, +) }
    var negativeSupport: Int { negativeBins.reduce(0, +) }
    var total: Int { positiveSupport + negativeSupport }

    func metrics(thresholdPercent: Int) -> BinaryValidationMetrics? {
        guard total > 0 else { return nil }
        let threshold = min(Self.resolution, max(0, thresholdPercent))
        let truePositives = positiveBins[threshold...].reduce(0, +)
        let falsePositives = negativeBins[threshold...].reduce(0, +)
        return BinaryValidationMetrics(
            truePositives: truePositives,
            falsePositives: falsePositives,
            falseNegatives: positiveSupport - truePositives,
            trueNegatives: negativeSupport - falsePositives
        )
    }
}

/// Selects a conservative live threshold using the fixed held-out rows. The
/// conventional 0.5 behavior is always a candidate, sparse validation evidence
/// leaves it unchanged, and support-based shrinkage prevents a handful of rows
/// from causing a large runtime behavior change.
enum BinaryThresholdCalibration {
    static let minimumPositiveSupport = 8
    static let minimumNegativeSupport = 32
    static let maximumThresholdPercent = 95

    static func calibrate(
        outputIndex: Int,
        histogram: BinaryScoreHistogram
    ) -> BinaryOutputValidation? {
        guard let defaultMetrics = histogram.metrics(thresholdPercent: 50) else { return nil }
        guard histogram.positiveSupport >= minimumPositiveSupport,
              histogram.negativeSupport >= minimumNegativeSupport else {
            return BinaryOutputValidation(
                outputIndex: outputIndex,
                decisionThreshold: 0.5,
                defaultMetrics: defaultMetrics,
                calibratedMetrics: defaultMetrics
            )
        }

        var bestPercent = 50
        var bestMetrics = defaultMetrics
        for percent in 51...maximumThresholdPercent {
            guard let candidate = histogram.metrics(thresholdPercent: percent) else { continue }
            let f1Gain = candidate.f1 - bestMetrics.f1
            let effectivelyTied = abs(f1Gain) < 0.000_001
            if f1Gain > 0.000_001
                || (effectivelyTied && candidate.falsePositives < bestMetrics.falsePositives) {
                bestPercent = percent
                bestMetrics = candidate
            }
        }

        let positiveConfidence = min(1, Double(histogram.positiveSupport) / 64)
        let negativeConfidence = min(1, Double(histogram.negativeSupport) / 256)
        let confidence = positiveConfidence * negativeConfidence
        var selectedPercent = Int((50 + Double(bestPercent - 50) * confidence).rounded())
        selectedPercent = min(maximumThresholdPercent, max(50, selectedPercent))
        var calibratedMetrics = histogram.metrics(thresholdPercent: selectedPercent) ?? defaultMetrics
        // Shrinking can land between two local F1 peaks. Never deploy a
        // threshold that is materially worse than the safe baseline.
        if calibratedMetrics.f1 + 0.005 < defaultMetrics.f1 {
            selectedPercent = 50
            calibratedMetrics = defaultMetrics
        }
        return BinaryOutputValidation(
            outputIndex: outputIndex,
            decisionThreshold: Double(selectedPercent) / 100,
            defaultMetrics: defaultMetrics,
            calibratedMetrics: calibratedMetrics
        )
    }

    static func calibrate(
        outputIndex: Int,
        probabilities: [Float],
        targets: [Bool]
    ) -> BinaryOutputValidation? {
        guard probabilities.count == targets.count else { return nil }
        var histogram = BinaryScoreHistogram()
        for (probability, target) in zip(probabilities, targets) {
            histogram.consume(probability: probability, target: target)
        }
        return calibrate(outputIndex: outputIndex, histogram: histogram)
    }
}

private struct ValidationAccumulator {
    let profile: AIProfile
    let activeBinaryIndices: Set<Int>
    private var binaryHistograms: [BinaryScoreHistogram]
    private var absoluteError = 0.0
    private var absoluteCount = 0
    private var relativeError = 0.0
    private var relativeCount = 0
    private var relativeExecutableCounts = [Int](
        repeating: 0,
        count: GameCameraContract.calibrationMagnitudes.count
    )
    private var relativeIdleFalseCounts = [Int](
        repeating: 0,
        count: GameCameraContract.calibrationMagnitudes.count
    )
    private var relativeIdleCount = 0
    private var scrollError = 0.0
    private var scrollCount = 0
    private var scrollExecutableCount = 0
    private var scrollIdleFalseActions = 0
    private var scrollIdleCount = 0

    init(profile: AIProfile, activeBinaryIndices: Set<Int>) {
        self.profile = profile
        self.activeBinaryIndices = activeBinaryIndices
        self.binaryHistograms = [BinaryScoreHistogram](
            repeating: BinaryScoreHistogram(),
            count: ActionLayout.count
        )
    }

    mutating func consume(predictions: [Float], targets: [Float], rowCount: Int) {
        guard predictions.count >= rowCount * ActionLayout.count,
              targets.count >= rowCount * ActionLayout.count else { return }
        for row in 0..<rowCount {
            let base = row * ActionLayout.count
            for index in activeBinaryIndices {
                let target = targets[base + index] >= 0.5
                binaryHistograms[index].consume(
                    probability: predictions[base + index],
                    target: target
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
                    for (candidateIndex, magnitude) in
                        GameCameraContract.calibrationMagnitudes.enumerated() {
                        let runtimeDelta = GameCameraContract.postedDelta(
                            forPrediction: prediction,
                            sensitivity: 1,
                            minimumMagnitude: magnitude
                        )
                        if runtimeDelta != 0, prediction * target > 0 {
                            relativeExecutableCounts[candidateIndex] += 1
                        }
                    }
                } else {
                    scrollError += Double(abs(prediction - target))
                    scrollCount += 1
                    if abs(prediction * 20) >= 0.5, prediction * target > 0 {
                        scrollExecutableCount += 1
                    }
                }
            } else {
                if isRelativeMouse {
                    relativeIdleCount += 1
                    for (candidateIndex, magnitude) in
                        GameCameraContract.calibrationMagnitudes.enumerated()
                    where GameCameraContract.postedDelta(
                        forPrediction: prediction,
                        sensitivity: 1,
                        minimumMagnitude: magnitude
                    ) != 0 {
                        relativeIdleFalseCounts[candidateIndex] += 1
                    }
                } else {
                    scrollIdleCount += 1
                    if abs(prediction * 20) >= 0.5 {
                        scrollIdleFalseActions += 1
                    }
                }
            }
        }
    }

    func finalize(
        sampleCount: Int,
        lossBreakdown: ValidationLossBreakdown,
        trainingBalance: TrainingBalanceReport,
        scope: ModelEvaluationScope
    ) -> ValidationReport {
        var binary = BinaryValidationAccumulator()
        var buttons = BinaryValidationAccumulator()
        var keyboard = BinaryValidationAccumulator()
        var modifiers = BinaryValidationAccumulator()
        var outputReports: [BinaryOutputValidation] = []
        outputReports.reserveCapacity(activeBinaryIndices.count)
        for index in activeBinaryIndices.sorted() {
            guard let output = BinaryThresholdCalibration.calibrate(
                outputIndex: index,
                histogram: binaryHistograms[index]
            ) else { continue }
            outputReports.append(output)
            binary.add(output.calibratedMetrics)
            switch index {
            case ActionLayout.buttons: buttons.add(output.calibratedMetrics)
            case ActionLayout.keyboardAndShift: keyboard.add(output.calibratedMetrics)
            case ActionLayout.commandOptionControl: modifiers.add(output.calibratedMetrics)
            default: break
            }
        }
        let relativeDeadzone = GameCameraContract.calibratedMinimumPostedMagnitude(
            activeCorrectCounts: relativeExecutableCounts,
            activeCount: relativeCount,
            idleFalseCounts: relativeIdleFalseCounts,
            idleCount: relativeIdleCount
        )
        let relativeDeadzoneIndex = GameCameraContract.calibrationMagnitudes
            .firstIndex(of: relativeDeadzone)
            ?? GameCameraContract.calibrationMagnitudes.firstIndex(
                of: GameCameraContract.defaultMinimumPostedMagnitude
            )
            ?? 0
        let relativeExecutableCount = relativeExecutableCounts.indices
            .contains(relativeDeadzoneIndex)
            ? relativeExecutableCounts[relativeDeadzoneIndex]
            : 0
        let relativeIdleFalseActions = relativeIdleFalseCounts.indices
            .contains(relativeDeadzoneIndex)
            ? relativeIdleFalseCounts[relativeDeadzoneIndex]
            : 0
        let idleContinuousCount = relativeIdleCount + scrollIdleCount
        let idleContinuousFalseActions =
            relativeIdleFalseActions + scrollIdleFalseActions
        return ValidationReport(
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
            relativeMouseExecutionDeadzone:
                profile.channels.mouseMovement ? Double(relativeDeadzone) : nil,
            activeRelativeMouseExecutionRecall: relativeCount > 0
                ? Double(relativeExecutableCount) / Double(relativeCount)
                : nil,
            activeScrollExecutionRecall: scrollCount > 0
                ? Double(scrollExecutableCount) / Double(scrollCount)
                : nil,
            lossBreakdown: lossBreakdown,
            binaryOutputs: outputReports,
            trainingBalance: trainingBalance,
            evaluationScope: scope
        )
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
    /// Associates the report with exact tensors. Mid-epoch snapshots must not
    /// borrow thresholds or diagnostics from weights evaluated at an older step.
    var currentEvaluationGlobalStep: Int? = nil
    var bestValidationReport: ValidationReport? = nil
    var schedulerBestMetric: Double? = nil
    var schedulerPlateauEpochs: Int? = nil
    var schedulerReductions: Int? = nil
    var demonstratedKeyCodes: Set<UInt16>?
    /// Persisted goal for the current epoch block. Optional keeps checkpoints
    /// from earlier releases decodable.
    var epochGoal: Int? = nil
    /// Optional keeps checkpoints from builds before experience counters
    /// decodable. Restore backfills a stable step/batch estimate once.
    var experienceSeconds: Double? = nil
    /// The strongest held-out deployment score and its corresponding loss and
    /// exact runnable weights are tracked independently from the latest optimizer
    /// checkpoint. Long runs can resume from the latest step without publishing
    /// a brain that regressed after its best demonstrated execution behavior.
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
