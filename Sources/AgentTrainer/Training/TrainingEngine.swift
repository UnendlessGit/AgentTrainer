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

final class TrainingEngine: @unchecked Sendable {
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
                self.lock.withLock { self.task = nil }
                completion(outcome)
            }
        }
    }

    func pauseAndSave() { lock.withLock { pauseRequested = true } }
    func stop() { lock.withLock { stopRequested = true; task?.cancel() } }

    private func train(profile: AIProfile, recordings: [RecordingItem], runSettings: TrainingRunSettings, randomState: MLXRandom.RandomState, metrics: @escaping MetricsHandler) async throws -> TrainingCompletion {
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        // Leave a small OS reserve while allowing MLX to use nearly all unified
        // memory available to the Apple GPU. Batch size remains user-controlled.
        Memory.memoryLimit = max(1 << 30, Int(Double(physical) * 0.94))
        // Fixed-shape compiled batches reuse a small set of buffers. A bounded
        // cache prevents allocator growth during multi-hour sessions without
        // changing model math or precision.
        Memory.cacheLimit = max(512 << 20, min(2 << 30, Int(Double(physical) * 0.06)))

        let dataset = try await DatasetCacheBuilder.shared.cache(for: profile, recordings: recordings) { progress, status in
            var value = TrainingMetrics(); value.totalEpochs = profile.training.epochs
            // Dataset event counts are not optimizer steps. Keep the step total
            // unknown until the train/validation split and batch count exist.
            value.totalSteps = 0
            metrics(value, "\(status) • \(Int((progress * 100).rounded()))%")
        }
        guard dataset.count > 0 else { throw AgentTrainerError.noData }
        let split = splitIndices(dataset: dataset, fraction: profile.training.validationSplit, seed: profile.training.seed)
        guard !split.train.isEmpty else { throw AgentTrainerError.noData }

        let model = AgentPolicy(profile: profile)
        model.train(true)
        let optimizer = ResumableAdamW(learningRate: Float(profile.training.learningRate), weightDecay: Float(profile.training.weightDecay))
        optimizer.initialize(model: model)
        let signature = try profileSignature(profile, recordings: recordings)
        let demonstratedKeys = try recordings.reduce(into: Set<UInt16>()) { result, recording in
            let url = recording.directory.appendingPathComponent(recording.manifest.eventFile)
            result.formUnion(try InputEventReader.demonstratedKeyCodes(url: url))
        }
        var state = CheckpointState(profileSignature: signature, epoch: 0, batchOffset: 0, globalStep: 0, elapsed: 0, lossHistory: [], validationHistory: [], demonstratedKeyCodes: demonstratedKeys, experienceSeconds: 0)
        let restoreStatus = try await restoreCheckpointIfPresent(profile: profile, expectedSignature: signature, model: model, optimizer: optimizer, randomState: randomState, state: &state)
        // The raw event stream is authoritative, including taps shorter than one
        // action interval. Refresh after restoring an older checkpoint.
        state.demonstratedKeyCodes = demonstratedKeys

        let trainingStep = compile(inputs: [model, optimizer, randomState], outputs: [model, optimizer, randomState]) { images, history, targets in
            let result = valueAndGrad(model: model) { model, arrays in
                [model.loss(images: arrays[0], history: arrays[1], targets: arrays[2])]
            }(model, [images, history, targets])
            optimizer.update(model: model, gradients: result.1, targetType: model.dtype)
            return result.0[0]
        }
        let started = ContinuousClock.now
        let baseElapsed = state.elapsed
        let batchSize = max(1, profile.training.batchSize)
        let stepsPerEpoch = Int(ceil(Double(split.train.count) / Double(batchSize)))
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
        var nextAutosaveStep = state.globalStep + autosaveInterval
        var autosavesPublished = 0

        let initialMemory = Memory.snapshot()
        metrics(TrainingMetrics(epoch: min(targetEpoch, state.epoch + (state.batchOffset > 0 ? 1 : 0)), totalEpochs: targetEpoch, batch: state.batchOffset / batchSize, totalBatches: stepsPerEpoch, globalStep: state.globalStep, totalSteps: totalSteps, nextAutosaveStep: nextAutosaveStep, autosavesPublished: autosavesPublished, trainingLoss: state.lossHistory.last ?? 0, validationLoss: state.validationHistory.last, samplesPerSecond: 0, elapsed: state.elapsed, experienceElapsed: state.experienceSeconds ?? 0, lossHistory: Array(state.lossHistory.suffix(4_096)), validationHistory: Array(state.validationHistory.suffix(1_024)), mlxActiveMemory: initialMemory.activeMemory, mlxCacheMemory: initialMemory.cacheMemory, mlxPeakMemory: initialMemory.peakMemory), "\(restoreStatus) • continuing to epoch \(targetEpoch)")

        var lastMetricsPublish = CACurrentMediaTime() - 1
        var lastRateTime = CACurrentMediaTime()
        var samplesSinceRate = 0

        trainingLoop: for epoch in state.epoch..<targetEpoch {
            let order = shuffled(split.train, seed: profile.training.seed &+ UInt64(epoch) &* 0x9E3779B97F4A7C15)
            var offset = epoch == state.epoch ? state.batchOffset : 0
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
                    let lossArray = trainingStep(arrays[0], arrays[1], arrays[2])
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
                prefetchedBatch = result.next
                state.globalStep += 1
                samplesSinceRate += batch.count
                state.experienceSeconds = (state.experienceSeconds ?? 0) + Double(batch.count) / max(0.0001, dataset.manifest.actionFPS)
                offset = end
                state.epoch = epoch
                state.batchOffset = offset
                state.lossHistory.append(loss)
                if state.lossHistory.count > 8_192 { state.lossHistory.removeFirst(4_096) }

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
                    let report = TrainingMetrics(epoch: epoch + 1, totalEpochs: targetEpoch, batch: Int(ceil(Double(offset) / Double(batchSize))), totalBatches: stepsPerEpoch, globalStep: state.globalStep, totalSteps: totalSteps, nextAutosaveStep: nextAutosaveStep, autosavesPublished: autosavesPublished, trainingLoss: loss, validationLoss: state.validationHistory.last, samplesPerSecond: Double(samplesSinceRate) / recentSeconds, elapsed: elapsed, experienceElapsed: state.experienceSeconds ?? 0, lossHistory: Array(state.lossHistory.suffix(4_096)), validationHistory: Array(state.validationHistory.suffix(1_024)), mlxActiveMemory: memory.activeMemory, mlxCacheMemory: memory.cacheMemory, mlxPeakMemory: memory.peakMemory)
                    samplesSinceRate = 0
                    metrics(report, "Pipelined training on Apple-silicon GPU")
                }

                let shouldCheckpoint = state.globalStep >= nextAutosaveStep
                let shouldPause = lock.withLock { pauseRequested }
                if shouldCheckpoint || shouldPause {
                    state.elapsed = baseElapsed + started.duration(to: .now).seconds
                    try await saveCheckpoint(profile: profile, model: model, optimizer: optimizer, randomState: randomState, state: state)
                    latestSnapshot = try await publishRunnableSnapshot(profile: profile, state: state, completed: false)
                    autosavesPublished += 1
                    if shouldCheckpoint {
                        repeat { nextAutosaveStep += autosaveInterval } while nextAutosaveStep <= state.globalStep
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
            if !split.validation.isEmpty {
                let validationLoss = evaluate(model: model, dataset: dataset, indices: split.validation, profile: profile)
                state.validationHistory.append(validationLoss)
                if state.validationHistory.count > 2_048 { state.validationHistory.removeFirst(1_024) }
            }
            state.elapsed = baseElapsed + started.duration(to: .now).seconds
            try await saveCheckpoint(profile: profile, model: model, optimizer: optimizer, randomState: randomState, state: state)
        }

        state.elapsed = baseElapsed + started.duration(to: .now).seconds
        let finalLoss = state.lossHistory.last ?? 0
        if latestSnapshot?.version.globalStep != state.globalStep {
            try await saveCheckpoint(profile: profile, model: model, optimizer: optimizer, randomState: randomState, state: state)
        }
        let final = try await publishRunnableSnapshot(profile: profile, state: state, completed: true, lossOverride: finalLoss)
        return final
    }

    private func prepareBatch(dataset: CachedDataset, indices: [Int], profile: AIProfile) -> PreparedBatch {
        let b = indices.count
        var targetData = dataset.actionBatch(at: indices)
        let restrictions = profile.effectiveRestrictions
        targetData.withUnsafeMutableBytes { raw in
            let values = raw.bindMemory(to: Float.self)
            for row in 0..<b {
                let base = row * ActionLayout.count
                for key in restrictions.blockedKeyCodes where key < 128 {
                    values[base + 14 + Int(key)] = 0
                }
                for modifier in 0..<4 where !restrictions.allowsModifier(modifier) { values[base + 142 + modifier] = 0 }
                for button in restrictions.blockedMouseButtons where button < 8 { values[base + 4 + Int(button)] = 0 }
            }
        }
        let historyData = dataset.historyBatch(at: indices)
        return PreparedBatch(
            count: b,
            packedObservations: dataset.packedObservations(at: indices),
            history: historyData,
            targets: targetData
        )
    }

    private func materializeBatch(_ batch: PreparedBatch, profile: AIProfile) -> [MLXArray] {
        return [
            VisionPreprocessor.mlxTensor(batch.packedObservations, batch: batch.count, spec: profile.preprocessing),
            MLXArray(batch.history, [batch.count, max(1, profile.training.historyLength), ActionLayout.count], type: Float.self),
            MLXArray(batch.targets, [batch.count, ActionLayout.count], type: Float.self)
        ]
    }

    private func makeBatch(dataset: CachedDataset, indices: [Int], profile: AIProfile) -> [MLXArray] {
        materializeBatch(prepareBatch(dataset: dataset, indices: indices, profile: profile), profile: profile)
    }

    private func evaluate(model: AgentPolicy, dataset: CachedDataset, indices: [Int], profile: AIProfile) -> Double {
        let limited = Array(indices.prefix(max(1, min(indices.count, profile.training.batchSize * 4))))
        let arrays = makeBatch(dataset: dataset, indices: limited, profile: profile)
        model.train(false)
        defer { model.train(true) }
        let loss = model.loss(images: arrays[0], history: arrays[1], targets: arrays[2])
        MLX.eval(loss)
        return Double(loss.item(Float.self))
    }

    private func splitIndices(dataset: CachedDataset, fraction: Double, seed: UInt64) -> (train: [Int], validation: [Int]) {
        let fraction = min(0.9, max(0, fraction))
        if dataset.manifest.segments.count > 1 {
            let shuffledSegments = shuffled(Array(dataset.manifest.segments.indices), seed: seed)
            let validationCount = max(fraction > 0 ? 1 : 0, Int(Double(shuffledSegments.count) * fraction))
            let validationSegments = Set(shuffledSegments.suffix(validationCount))
            var train: [Int] = [], validation: [Int] = []
            for (i, segment) in dataset.manifest.segments.enumerated() {
                let values = Array(segment.start..<(segment.start + segment.count))
                if validationSegments.contains(i) { validation += values } else { train += values }
            }
            return (train, validation)
        }
        let validationCount = Int(Double(dataset.count) * fraction)
        return (Array(0..<max(0, dataset.count - validationCount)), Array(max(0, dataset.count - validationCount)..<dataset.count))
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

    private func profileSignature(_ profile: AIProfile, recordings: [RecordingItem]) throws -> String {
        struct TrainingIdentity: Encodable {
            let trainingDataSchema: Int
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
        var normalizedChannels = profile.channels
        normalizedChannels.absoluteMouse = profile.channels.mouseMovement
        normalizedChannels.relativeMouse = profile.channels.mouseMovement
        let manifests = recordings.map(\.manifest).sorted { $0.id.uuidString < $1.id.uuidString }
        let identity = TrainingIdentity(trainingDataSchema: TrainingDataContract.schemaVersion, preprocessing: profile.preprocessing, channels: normalizedChannels, training: resumeCompatibleTraining, recordings: manifests, folderIDs: profile.effectiveFolderIDs.sorted { $0.uuidString < $1.uuidString }, restrictions: profile.effectiveRestrictions)
        return SHA256.hash(data: try encoder.encode(identity)).map { String(format: "%02x", $0) }.joined()
    }

    private func saveCheckpoint(profile: AIProfile, model: AgentPolicy, optimizer: ResumableAdamW, randomState: MLXRandom.RandomState, state: CheckpointState) async throws {
        let destination = await WorkspaceStore.shared.checkpointDirectory(profileID: profile.id)
        let temporary = destination.deletingLastPathComponent().appendingPathComponent(".Checkpoint.\(UUID().uuidString).tmp")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        try model.saveWeights(to: temporary.appendingPathComponent("weights.safetensors"))
        try optimizer.save(to: temporary.appendingPathComponent("optimizer.safetensors"))
        try TrainingRandomState.save(randomState, to: temporary.appendingPathComponent("random.safetensors"))
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(state).write(to: temporary.appendingPathComponent("state.json"), options: .atomic)
        if FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.removeItem(at: destination) }
        try FileManager.default.moveItem(at: temporary, to: destination)
    }

    private func publishRunnableSnapshot(profile: AIProfile, state: CheckpointState, completed: Bool, lossOverride: Double? = nil) async throws -> TrainingCompletion {
        let checkpoint = await WorkspaceStore.shared.checkpointDirectory(profileID: profile.id)
        let displayedEpoch = state.batchOffset > 0 ? state.epoch + 1 : state.epoch
        let version = ModelVersionManifest(
            id: UUID(),
            name: completed ? "Brain • Epoch \(displayedEpoch) • Step \(state.globalStep)" : "Autosave • Epoch \(displayedEpoch) • Step \(state.globalStep)",
            createdAt: Date(),
            globalStep: state.globalStep,
            trainingLoss: lossOverride ?? state.lossHistory.last ?? 0,
            validationLoss: state.validationHistory.last,
            preprocessing: profile.preprocessing,
            channels: profile.channels,
            training: profile.training,
            optimizerFile: "optimizer.safetensors",
            trainingStateFile: "state.json",
            randomStateFile: "random.safetensors",
            epoch: displayedEpoch,
            isAutosave: !completed,
            demonstratedKeyCodes: state.demonstratedKeyCodes ?? [],
            relativeMouseScale: GameCameraContract.deltaScale,
            trainingDataSchema: TrainingDataContract.schemaVersion,
            trainingDurationSeconds: state.elapsed,
            experienceDurationSeconds: state.experienceSeconds ?? 0
        )
        let destination = await WorkspaceStore.shared.versionDirectory(profileID: profile.id, versionID: version.id)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        do {
            try FileManager.default.copyItem(at: checkpoint.appendingPathComponent("weights.safetensors"), to: destination.appendingPathComponent(version.weightsFile))
            try FileManager.default.copyItem(at: checkpoint.appendingPathComponent("optimizer.safetensors"), to: destination.appendingPathComponent("optimizer.safetensors"))
            try FileManager.default.copyItem(at: checkpoint.appendingPathComponent("state.json"), to: destination.appendingPathComponent("state.json"))
            try FileManager.default.copyItem(at: checkpoint.appendingPathComponent("random.safetensors"), to: destination.appendingPathComponent("random.safetensors"))
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
        return TrainingCompletion(profile: updated, version: version, completed: completed)
    }

    private func restoreCheckpointIfPresent(profile: AIProfile, expectedSignature: String, model: AgentPolicy, optimizer: ResumableAdamW, randomState: MLXRandom.RandomState, state: inout CheckpointState) async throws -> String {
        let directory = await WorkspaceStore.shared.checkpointDirectory(profileID: profile.id)
        let stateURL = directory.appendingPathComponent("state.json")
        if FileManager.default.fileExists(atPath: stateURL.path) {
            let restored = try JSONDecoder().decode(CheckpointState.self, from: Data(contentsOf: stateURL))
            if restored.profileSignature == expectedSignature {
                try model.loadWeights(from: directory.appendingPathComponent("weights.safetensors"))
                try optimizer.load(from: directory.appendingPathComponent("optimizer.safetensors"))
                let randomStateURL = directory.appendingPathComponent("random.safetensors")
                if FileManager.default.fileExists(atPath: randomStateURL.path) { try TrainingRandomState.load(randomState, from: randomStateURL) }
                state = restored
                return "Restored exact checkpoint; compiling resumed MLX graph"
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
           version.training.architecture == profile.training.architecture {
            let versionDirectory = await WorkspaceStore.shared.versionDirectory(profileID: profile.id, versionID: versionID)
            try model.loadWeights(from: versionDirectory.appendingPathComponent(version.weightsFile))
            state.epoch = max(0, version.epoch ?? 0)
            state.batchOffset = 0
            state.globalStep = max(0, version.globalStep)
            state.elapsed = max(0, version.trainingDurationSeconds ?? profile.trainingProgress?.trainingDurationSeconds ?? 0)
            state.experienceSeconds = version.experienceDurationSeconds ?? profile.trainingProgress?.experienceDurationSeconds
            state.lossHistory = [version.trainingLoss]
            state.validationHistory = version.validationLoss.map { [$0] } ?? []
            return "Loaded the active brain for the corrected training-data contract; optimizer restarted safely"
        }
        return "Compiling fused MLX training graph on Apple GPU"
    }

}

private struct PreparedBatch {
    let count: Int
    let packedObservations: Data
    let history: Data
    let targets: Data
}

private struct CheckpointState: Codable {
    var profileSignature: String
    var epoch: Int
    var batchOffset: Int
    var globalStep: Int
    var elapsed: Double
    var lossHistory: [Double]
    var validationHistory: [Double]
    var demonstratedKeyCodes: Set<UInt16>?
    /// Persisted goal for the current epoch block. Optional keeps checkpoints
    /// from earlier releases decodable.
    var epochGoal: Int? = nil
    /// Optional keeps checkpoints from builds before experience counters
    /// decodable. Restore backfills a stable step/batch estimate once.
    var experienceSeconds: Double? = nil
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
