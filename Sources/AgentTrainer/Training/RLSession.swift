import AppKit
import Foundation

@MainActor
final class RLSession {
    let runtime: AgentRuntime
    let budget: RLComputeBudget
    let learner: any RLLearningAlgorithm
    let reader: any RewardReading = RewardReader()
    var onMetrics: ((RLMetrics) -> Void)?
    var onRewardPreview: ((RewardPreview) -> Void)?
    var onPublished: ((AIProfile) -> Void)?
    var onStopped: ((String) -> Void)?
    var currentProfile: AIProfile { profile }
    private var prepared = false
    private var initialRandom = RLRandom(state: 42)
    private var runtimeStarted = false
    private var profile: AIProfile
    private let configuration: RLConfiguration
    private let movementSensitivity: Double
    private let mouseMode: MouseControlMode
    private let context: ModelContextSchema
    private var reducer = RLEpisodeReducer()
    private var metrics = RLMetrics()
    nonisolated private let inbox = RLDecisionInbox()
    private var transitions: [RLTransition] = []
    private var updateTask: Task<Void, Never>?
    private var timerTask: Task<Void, Never>?
    private var resetTask: Task<Void, Never>?
    private var stopping = false
    private var stopFinished = false
    private var stopWaiters: [CheckedContinuation<Void, Never>] = []
    private var saving = false
    private var saveWaiters: [CheckedContinuation<Void, Never>] = []
    private(set) var saveError: String?
    private var geometryTask: Task<Void, Never>?
    private var lastSaveAt = DispatchTime.now().uptimeNanoseconds
    private var beganAt = DispatchTime.now().uptimeNanoseconds
    private var priorElapsed = 0.0
    private var priorExperience = 0.0
    private var sessionExperience = 0.0
    private var lastValidAt: UInt64 = 0
    private var readyNotBefore: UInt64 = 0
    private var generation: UInt64 = 0
    private var episodeCount = 0
    private var lastDecisionValue: Float = 0
    private var episodeKeys = RLEpisodeKeyState()

    init(profile: AIProfile, runtime: AgentRuntime, mouseMode: MouseControlMode, context: ModelContextSchema, movementSensitivity: Double = 1) {
        self.profile = profile; self.runtime = runtime; self.mouseMode = mouseMode
        budget = RLComputeBudget(fraction: profile.effectiveRL.trainingComputeFraction)
        learner = RLAlgorithmFactory.make(profile.effectiveRL.algorithm, budget: budget)
        self.movementSensitivity = movementSensitivity
        self.context = context; configuration = profile.effectiveRL
        let distribution = RLActionDistribution(profile: profile, configuration: profile.effectiveRL, mouseMode: mouseMode)
        metrics.controls = "\(mouseMode.rawValue) • \(distribution.binaryIndices.count) binary / \(distribution.continuousIndices.count) analog controls"
    }

    func prepare() async throws -> RLWeightSnapshot {
        let root = await WorkspaceStore.shared.profileDirectory(profile.id)
        let checkpoint = await WorkspaceStore.shared.rlCheckpointDirectory(profile: profile)
        let version = profile.activeVersionID == nil ? nil : await WorkspaceStore.shared.version(profileID: profile.id, versionID: profile.activeVersionID!)
        let weights = version.map { root.appendingPathComponent("Versions/\($0.id.uuidString)/\($0.weightsFile)") }
        let prepared = try await learner.prepare(profile: profile, contextSchema: context, mouseMode: mouseMode,
            activeWeights: weights, checkpoint: checkpoint, movementSensitivity: movementSensitivity)
        guard !stopping else { throw CancellationError() }
        self.prepared = true; initialRandom = prepared.1.actorRandom
        if let data = try? Data(contentsOf: checkpoint.appendingPathComponent("collected.plist")),
           let collected = try? PropertyListDecoder().decode([RLTransition].self, from: data),
           prepared.1.activeVersionID == profile.activeVersionID, prepared.1.configuration.hasSameLearningContract(as: configuration) {
            var validation = prepared.1; validation.pending = collected; validation.order = Array(collected.indices)
            validation.advantages = Array(repeating: 0, count: collected.count); validation.returns = validation.advantages
            validation.epoch = 0; validation.offset = 0; try validation.validateResume(profile: profile)
            transitions = collected
        }
        episodeCount = max(prepared.1.episodeCount, prepared.1.history.map(\.id).max() ?? 0)
        metrics.totalEpisodes = episodeCount; metrics.history = prepared.1.history
        metrics.updates = prepared.1.updates; metrics.optimizerSteps = prepared.1.optimizerSteps
        metrics.attemptedUpdates = prepared.1.attemptedUpdates ?? prepared.1.updates
        metrics.rejectedUpdates = prepared.1.rejectedUpdates ?? 0
        metrics.legacyUpdateAttempts = prepared.1.legacyUpdateAttempts ?? 0
        metrics.resumeNotice = prepared.1.resumeNotice ?? ""
        priorElapsed = prepared.1.trainingSeconds; priorExperience = prepared.1.experienceSeconds ?? 0
        metrics.trainingSeconds = priorElapsed
        metrics.experienceSeconds = priorExperience
        // A resumed minibatch continues before collecting a fresh episode.
        if !prepared.1.pending.isEmpty {
            let result = try await learner.update([])
            metrics.updates = result.updates; metrics.optimizerSteps = result.steps
            return result.snapshot
        }
        return prepared.0
    }

    func actorRandom() async -> RLRandom {
        initialRandom
    }

    func start(source: CaptureSourceOption) async throws {
        guard !stopping else { throw CancellationError() }
        runtimeStarted = true
        guard let region = configuration.rewardRegion else { throw AgentTrainerError.invalidConfiguration("Select a reward region first.") }
        reader.onReading = { [weak self] reading in Task { @MainActor in await self?.receive(reading) } }
        reader.onPreview = { [weak self] preview in Task { @MainActor in
            guard let self, !self.stopping else { return }
            self.metrics.detectionStatus = preview.status
            self.metrics.rewardFPS = preview.framesPerSecond
            self.onRewardPreview?(preview)
            if let reading = preview.reading { self.metrics.recognitionLatency = Double(reading.recognizedAt - reading.capturedAt) / 1e6 }
        } }
        reader.onFailure = { [weak self] reason in Task { @MainActor in await self?.stop(reason: "Reward capture interrupted: \(reason)") } }
        try await reader.start(source: source, region: region, fps: profile.training.actionFPS, confidence: configuration.minimumConfidence)
        guard !stopping else { await reader.stop(); throw CancellationError() }
        beganAt = DispatchTime.now().uptimeNanoseconds; generation &+= 1
        try await runtime.awaitInitialRLFocus()
        guard !stopping else { throw CancellationError() }
        await startEpisode()
        let originalFrame = source.frame
        geometryTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(500))
                guard let self, !stopping else { return }
                do {
                    let sources = try await CaptureService.availableSources()
                    guard let current = sources.first(where: { $0.id == source.id && $0.kind == source.kind }) else {
                        await stop(reason: "The capture source is unavailable. RL progress saved."); return
                    }
                    guard current.pointPixelScale == source.pointPixelScale else {
                        await stop(reason: "Display scaling changed. Test reward detection before resuming."); return
                    }
                    guard current.frame.size == originalFrame.size else {
                        await stop(reason: "The source was resized. Reselect and test the reward region before resuming."); return
                    }
                    runtime.moveRLCapture(by: CGSize(width: current.frame.minX - originalFrame.minX,
                        height: current.frame.minY - originalFrame.minY))
                } catch { await stop(reason: "Capture geometry could not be verified: \(error.localizedDescription)"); return }
            }
        }
        timerTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(100))
                guard let self, !stopping else { return }
                await tick()
            }
        }
    }

    nonisolated func record(_ decision: RLDecision) { inbox.append(decision) }

    private func startEpisode() async {
        guard !stopping else { return }
        runtime.suspendRLActions(); metrics.status = "Resetting environment"
        do {
            episodeKeys.reset()
            try await runtime.episodeKey(configuration.startKey)
            guard !stopping else { return }
            reducer.reset(at: DispatchTime.now().uptimeNanoseconds)
            readyNotBefore = reducer.resetAt + UInt64(configuration.resetSettlingSeconds * 1e9)
            lastValidAt = reducer.resetAt; lastDecisionValue = 0
            metrics.unobservedActions += inbox.discardAll()
            metrics.episode = episodeCount + 1; metrics.episodeReward = 0
            metrics.status = "Waiting for Reward: 0 and positive Episode Length"
        } catch { await stop(reason: error.localizedDescription) }
    }

    private func receive(_ reading: RLRewardReading) async {
        guard !stopping else { return }
        let now = DispatchTime.now().uptimeNanoseconds
        guard now >= reading.capturedAt, Double(now - reading.capturedAt) / 1e9 <= configuration.detectionTimeout else { return }
        if reducer.phase == .waiting && reading.capturedAt < readyNotBefore { return }
        guard let accepted = reducer.accept(reading, mode: configuration.rewardMode) else { return }
        guard Float(accepted.reward * configuration.rewardScale).isFinite, reducer.reward.isFinite else {
            await stop(reason: "The reward exceeds the supported training range. Reduce Reward Scale before resuming."); return
        }
        lastValidAt = reading.capturedAt
        if accepted.ready {
            metrics.status = "Learning live • PPO"
            await runtime.beginRLEpisode()
            return
        }
        let batch = inbox.take(before: reading.capturedAt)
        if batch.overflow { await stop(reason: "RL experience buffer overflowed; reduce vision or rollout size."); return }
        let records = batch.records.filter { $0.timestamp >= reducer.startedAt }
        lastDecisionValue = batch.nextValue ?? records.last?.value ?? lastDecisionValue
        let gap = configuration.rewardMode == .immediate && (records.count > 1 || accepted.duration > 1.75 / profile.training.actionFPS)
        if !records.isEmpty {
            if gap {
                // A valid terminal immediate reward still belongs to its final
                // observed action; unknown earlier intervals remain excluded.
                metrics.skippedIntervals += max(0, records.count - 1)
                if !transitions.isEmpty { transitions[transitions.count - 1].boundary = true; transitions[transitions.count - 1].bootstrapValue = records[0].value }
            }
            let included = gap ? [records.last!] : records
            transitions.append(RLTransition(decisions: included, reward: Float(accepted.reward * configuration.rewardScale),
                duration: gap ? 1 / profile.training.actionFPS : accepted.duration, terminal: accepted.terminal,
                bootstrapValue: lastDecisionValue))
        } else if configuration.rewardMode == .immediate && !accepted.terminal {
            // OCR and action clocks can straddle each other. A second reading
            // without a newly executed interval must not pay the same action twice.
            reducer.reward -= accepted.reward
        } else if let last = transitions.last, !last.terminal,
                  let timestamp = last.decisions.last?.timestamp, timestamp >= reducer.startedAt {
            transitions[transitions.count - 1].reward += Float(accepted.reward * configuration.rewardScale)
            transitions[transitions.count - 1].terminal = accepted.terminal
            transitions[transitions.count - 1].duration += accepted.duration
        }
        let trimmed = RLExperienceBuffer.trim(&transitions, maximumActions: configuration.rolloutSize * 2, maximumBytes: 384 * 1024 * 1024)
        metrics.discardedActions += trimmed.discarded
        metrics.backlogActions = transitions.reduce(0) { $0 + $1.decisions.count }
        metrics.episodeReward = reducer.reward
        if accepted.terminal {
            runtime.suspendRLActions()
            await finishEpisode(sendEnd: true)
        } else if !trimmed.overflow { maybeUpdate() }
        if trimmed.overflow { await stop(reason: "The learner cannot keep up without discarding terminal rewards. Reduce vision or rollout size before resuming.") }
    }

    private func tick() async {
        let now = DispatchTime.now().uptimeNanoseconds
        guard AXIsProcessTrusted(), CGPreflightScreenCaptureAccess(), CGPreflightListenEventAccess() else {
            await stop(reason: "A required macOS capture or input permission was lost. Inputs released and progress saved."); return
        }
        metrics.trainingSeconds = priorElapsed + Double(DispatchTime.now().uptimeNanoseconds - beganAt) / 1e9
        metrics.experienceSeconds = priorExperience + sessionExperience + (reducer.phase == .active ? Double(now - reducer.startedAt) / 1e9 : 0)
        metrics.remaining = reducer.remaining(at: now)
        if reducer.phase == .waiting, Double(now - reducer.resetAt) / 1e9 > configuration.resetTimeout {
            await stop(reason: "Reset was not confirmed. Display Reward: 0 with a positive Episode Length after the start key."); return
        }
        if reducer.phase == .active, Double(now - lastValidAt) / 1e9 > configuration.detectionTimeout {
            await stop(reason: "Reward detection was lost. Inputs released and RL progress saved."); return
        }
        if reducer.phase == .active, metrics.remaining <= 0 {
            reducer.phase = .finishing; runtime.suspendRLActions(); metrics.status = "Reading final reward"
            do { try await runtime.episodeKey(episodeKeys.takeEnd(configuration.endKey)) } catch { await stop(reason: error.localizedDescription); return }
            let token = generation
            let finalDrainSeconds = max(0.5, configuration.detectionTimeout)
            resetTask = Task { [weak self] in
                try? await Task.sleep(for: .seconds(finalDrainSeconds))
                guard let self, !Task.isCancelled, !stopping, generation == token, reducer.phase == .finishing else { return }
                if Double(DispatchTime.now().uptimeNanoseconds - lastValidAt) / 1e9 > configuration.detectionTimeout {
                    await stop(reason: "The final reward could not be confirmed. Test detection before resuming; progress saved."); return
                }
                if !transitions.isEmpty { transitions[transitions.count - 1].terminal = true }
                reducer.phase = .paused
                await finishEpisode(sendEnd: false)
            }
        }
        if configuration.automaticCheckpoints, Double(now - lastSaveAt) / 1e9 >= configuration.autosaveSeconds, !saving { await save(autosave: true) }
        onMetrics?(metrics)
    }

    private func finishEpisode(sendEnd: Bool) async {
        guard !stopping else { return }
        episodeCount += 1; metrics.totalEpisodes = episodeCount
        let elapsed = Double(reducer.lastReadingAt >= reducer.startedAt ? reducer.lastReadingAt - reducer.startedAt : 0) / 1e9
        sessionExperience += elapsed
        metrics.experienceSeconds = priorExperience + sessionExperience
        metrics.history.append(RLEpisodeRecord(id: episodeCount, reward: reducer.reward, duration: elapsed))
        metrics.history = RLEpisodeRecord.boundedHistory(metrics.history)
        if sendEnd {
            do { try await runtime.episodeKey(episodeKeys.takeEnd(configuration.endKey)) } catch { await stop(reason: error.localizedDescription); return }
        }
        maybeUpdate(force: true)
        metrics.status = "Between episodes"; onMetrics?(metrics)
        resetTask?.cancel()
        resetTask = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(for: .seconds(configuration.episodeDelay))
            guard !Task.isCancelled, !stopping else { return }
            await startEpisode()
        }
    }

    private func maybeUpdate(force: Bool = false) {
        guard updateTask == nil, !stopping, !transitions.isEmpty else { return }
        // Keep one reward interval uncommitted until the next reading. A final
        // reward after releasing the last action can then extend this tail even
        // when the preceding rollout is already being optimized.
        guard let batch = RLExperienceBuffer.takeReady(&transitions, minimumActions: configuration.rolloutSize,
            force: force, fallbackValue: lastDecisionValue) else { return }
        updateTask = Task { [weak self] in
            guard let self else { return }
            do {
                let update = try await learner.update(batch)
                if !stopping { try await runtime.installRLSnapshot(update.snapshot) }
                metrics.updates = update.updates; metrics.optimizerSteps = update.steps
                metrics.policyLoss = update.policyLoss; metrics.valueLoss = update.valueLoss
                metrics.entropy = update.entropy; metrics.kl = update.kl
                metrics.discardedActions += update.rejectedActions
                metrics.attemptedUpdates = update.attemptedUpdates; metrics.rejectedUpdates = update.rejectedUpdates
                metrics.klRejectedActions += update.klRejectedActions
            } catch {
                updateTask = nil
                if !stopping { await self.stop(reason: "RL update stopped: \(error.localizedDescription)") }
                return
            }
            updateTask = nil
            if !stopping { maybeUpdate() }
        }
    }

    func save(autosave: Bool = false) async {
        guard prepared else { return }
        while saving { await withCheckedContinuation { saveWaiters.append($0) } }
        guard !autosave || (!stopping && !Task.isCancelled) else { return }
        saving = true; saveError = nil
        defer {
            saving = false; let waiters = saveWaiters; saveWaiters = []
            waiters.forEach { $0.resume() }
        }
        do {
            let root = await WorkspaceStore.shared.profileDirectory(profile.id)
            let id = UUID(), stage = root.appendingPathComponent(".RLPublication.\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: stage) }
            let random = runtimeStarted ? await runtime.rlRandomState() : initialRandom
            var collected = transitions
            if !collected.isEmpty { collected[collected.count - 1].boundary = !collected[collected.count - 1].terminal; collected[collected.count - 1].bootstrapValue = lastDecisionValue }
            let saved = try await learner.save(to: stage, versionID: id, actorRandom: random,
                history: metrics.history, episodes: episodeCount, elapsed: metrics.trainingSeconds, experience: metrics.experienceSeconds, pending: collected)
            let caps = configuration.capabilities(profile: profile, mouseMode: mouseMode)
            let version = ModelVersionManifest(id: id, name: "\(autosave ? "RL Autosave" : "RL Brain") • Episode \(episodeCount) • Update \(saved.updates)",
                createdAt: Date(), globalStep: saved.optimizerSteps, trainingLoss: metrics.policyLoss + metrics.valueLoss,
                preprocessing: profile.preprocessing, channels: profile.channels, training: profile.training,
                isAutosave: autosave, demonstratedKeyCodes: caps.allowedKeyCodes, relativeMouseScale: GameCameraContract.deltaScale,
                trainingDataSchema: TrainingDataContract.schemaVersion, trainingDurationSeconds: metrics.trainingSeconds,
                experienceDurationSeconds: metrics.experienceSeconds, recommendedMouseMode: mouseMode,
                contextSchema: context, outputCapabilities: caps,
                rlSummary: RLSavedSummary(updates: saved.updates, episodes: episodeCount, trainingSeconds: metrics.trainingSeconds, experienceSeconds: metrics.experienceSeconds))
            profile = try await WorkspaceStore.shared.publishRL(profileID: profile.id, version: version, staged: stage,
                keeping: configuration.retainedAutosaves)
            lastSaveAt = DispatchTime.now().uptimeNanoseconds; onPublished?(profile)
        } catch {
            saveError = error.localizedDescription
            metrics.status = "Save failed: \(error.localizedDescription)"; onMetrics?(metrics)
            AppLog.write(.error, category: "RL", "Checkpoint save failed", details: error.localizedDescription)
        }
    }

    func stop(reason: String = "RL training paused and saved") async {
        if stopping {
            if !stopFinished { await withCheckedContinuation { stopWaiters.append($0) } }
            return
        }
        stopping = true; generation &+= 1
        AppLog.write(category: "RL", "Stopping live training", details: reason)
        defer {
            stopFinished = true; let waiters = stopWaiters; stopWaiters = []
            waiters.forEach { $0.resume() }
        }
        let stoppedAt = DispatchTime.now().uptimeNanoseconds
        runtime.suspendRLActions(); learner.interrupt()
        timerTask?.cancel(); resetTask?.cancel(); geometryTask?.cancel()
        let interrupted = reducer.phase == .active || reducer.phase == .finishing
        reducer.phase = .paused
        await reader.stop()
        metrics.unobservedActions += inbox.discardAll()
        await updateTask?.value
        if interrupted {
            let elapsed = max(0, Double(stoppedAt - reducer.startedAt) / 1e9)
            sessionExperience += elapsed; episodeCount += 1; metrics.totalEpisodes = episodeCount
            metrics.history.append(RLEpisodeRecord(id: episodeCount, reward: reducer.reward, duration: elapsed, interrupted: true))
            metrics.history = RLEpisodeRecord.boundedHistory(metrics.history)
        }
        if !transitions.isEmpty { transitions[transitions.count - 1].boundary = true; transitions[transitions.count - 1].bootstrapValue = lastDecisionValue }
        metrics.trainingSeconds = priorElapsed + Double(DispatchTime.now().uptimeNanoseconds - beganAt) / 1e9
        metrics.experienceSeconds = priorExperience + sessionExperience
        await save()
        await runtime.stop(reason: nil)
        let finalReason = saveError.map { "RL stopped. Unsaved progress is retained in memory; retry saving: " + $0 } ?? reason
        metrics.status = finalReason; onMetrics?(metrics); onStopped?(finalReason)
    }
}
