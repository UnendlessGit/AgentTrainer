@preconcurrency import AppKit
@preconcurrency import CoreMedia
@preconcurrency import CoreVideo
import CryptoKit
import Foundation
import MLX
import QuartzCore
@preconcurrency import ScreenCaptureKit

struct ReinforcementLearningCompletion: Sendable {
    var profile: AIProfile
    var version: ModelVersionManifest?
    var reason: String
}

private struct ReinforcementRewardStatistics: Codable, Sendable {
    var count: Int = 0
    var mean = 0.0
    var squaredDeviation = 0.0

    mutating func normalize(_ value: Double, epsilon: Double) -> Double {
        count += 1
        let delta = value - mean
        mean += delta / Double(count)
        let nextDelta = value - mean
        squaredDeviation += delta * nextDelta
        guard count > 1 else { return value }
        let variance = max(0, squaredDeviation / Double(count - 1))
        return (value - mean) / Foundation.sqrt(variance + epsilon)
    }
}

private struct ReinforcementCheckpointState: Codable, Sendable {
    var signature: String
    var actorVersionID: UUID?
    var completedEpisodes: Int
    var updates: Int
    var environmentSteps: Int
    var globalStep: Int
    var displayedEpoch: Int
    var elapsed: Double
    var samplingRandomState: UInt64
    var rewardStatistics: ReinforcementRewardStatistics
    var episodeReturns: [Double]
    var cumulativeEpisodeReturn: Double
    var lastPolicyLoss: Double
    var lastValueLoss: Double
    var lastEntropy: Double
    var lastApproximateKL: Double
    var lastClipFraction: Double
    var lastExplainedVariance: Double
}

private struct PreparedReinforcementSnapshot: Sendable {
    var manifest: ModelVersionManifest
    var profile: AIProfile
    var progress: TrainingProgressSummary
}

/// Online RL is intentionally separate from `TrainingEngine` and
/// `AgentRuntime`. It reuses the same vision, Policy-v6 actor, input injector,
/// Run safety settings, and immutable version format without adding any reward
/// dependency to ordinary inference.
final class ReinforcementLearningEngine: @unchecked Sendable {
    typealias MetricsHandler = @Sendable (ReinforcementLearningMetrics, String) -> Void
    typealias CompletionHandler = @Sendable (Result<ReinforcementLearningCompletion, Error>) -> Void

    var onState: (@Sendable (InputState) -> Void)?
    var onPreview: (@Sendable (VisionPreviewFrame) -> Void)?
    var onRuntimeMetrics: (@Sendable (RuntimeMetrics) -> Void)?
    var onVersionPublished: (@Sendable (AIProfile, ModelVersionManifest) -> Void)?

    private let capture = CaptureService()
    private let preprocessor: VisionPreprocessor
    private let injector = InputInjector()
    private let safetyMonitor = InputCaptureService()
    private let rewardAccumulator = ReinforcementRewardAccumulator()
    private lazy var rewardServer = RewardWebSocketServer(
        accumulator: rewardAccumulator
    ) { [weak self] state, detail in
        self?.handleConnectionState(state, detail: detail)
    }
    private let inferenceQueue = DispatchQueue(
        label: "AgentTrainer.RL.InferenceAndUpdates",
        qos: .userInitiated
    )
    private let actionQueue = DispatchQueue(
        label: "AgentTrainer.RL.Actions",
        qos: .userInteractive
    )
    private let lock = NSLock()

    private var policyOptimizer: ReinforcementPolicyOptimizer?
    private var profile: AIProfile?
    private var actorProfile: AIProfile?
    private var executionProfile: AIProfile?
    private var configuration = ReinforcementLearningConfiguration()
    private var webSocketSettings = RewardWebSocketServerSettings()
    private var contextSchema = ModelContextSchema.empty
    private var contextVector: [Float] = []
    private var actionPlan: ReinforcementActionPlan?
    private var checkpointState: ReinforcementCheckpointState?
    private var checkpointDirectory: URL?
    private var profileRoot: URL?
    private var captureRect = CGRect.zero
    private var frameMode = FrameMode.newest
    private var mouseMode = MouseControlMode.absolute
    private var gameCamera = GameCameraSettings()
    private var outputPermissions = RuntimeOutputPermissions()
    private var safety = AgentSafetyPolicy()
    private var targetPID: pid_t?
    private var trainingShowsCursor = false
    private var metricsHandler: MetricsHandler?
    private var completionHandler: CompletionHandler?

    private var stopped = true
    private var starting = false
    private var teardownInProgress = false
    private var connectionState = ReinforcementConnectionState.disconnected
    private var rewardIsStale = false
    private var rewardStaleStartedAt: Double?
    private var connectionPauseStartedAt: Double?
    private var rewardFreshnessBaseline = 0.0
    private var updateInProgress = false
    private var episodeBoundaryInProgress = false
    private var episodeIsOpen = false
    private var latestFrame: CVPixelBuffer?
    private var latestFrameHostNanos: UInt64 = 0
    private var lastUsableFrame: CVPixelBuffer?
    private var processing = false
    private var nextPerceptionTime = 0.0
    private var actionTimer: DispatchSourceTimer?
    private var predictionLatch = RuntimePredictionLatch()

    private struct TemporalFrame: Sendable {
        var packed: Data
        var controls: [Float]
    }

    private struct TemporalFrameRing: Sendable {
        private var storage: [TemporalFrame?] = []
        private var oldest = 0
        private(set) var count = 0

        mutating func reset() {
            storage.removeAll(keepingCapacity: false)
            oldest = 0
            count = 0
        }

        mutating func append(_ frame: TemporalFrame, capacity rawCapacity: Int) {
            let capacity = max(1, rawCapacity)
            if storage.count != capacity {
                storage = [TemporalFrame?](repeating: nil, count: capacity)
                oldest = 0
                count = 0
            }
            if count < capacity {
                storage[(oldest + count) % capacity] = frame
                count += 1
            } else {
                storage[oldest] = frame
                oldest = (oldest + 1) % capacity
            }
        }

        func frame(distanceBack: Int) -> TemporalFrame? {
            guard distanceBack > 0, distanceBack <= count,
                  !storage.isEmpty else { return nil }
            return storage[(oldest + count - distanceBack) % storage.count]
        }
    }

    private var temporalFrames = TemporalFrameRing()
    private var rollout: [ReinforcementTransition] = []
    private var pendingTransition: ReinforcementTransition?
    private var samplingRandom = ReinforcementRandomGenerator(seed: 7_431)
    private var actionFilter = ReinforcementActionFilter()
    private var rewardStatistics = ReinforcementRewardStatistics()
    private var currentEpisode = 0
    private var sessionStartingCompletedEpisodes = 0
    private var sessionStartingUpdates = 0
    private var episodeStep = 0
    private var episodeReturn = 0.0
    private var episodeStartedAt = 0.0
    private var sessionStartedAt = 0.0
    private var lastPublishedUpdate = 0
    private var latestPublishedVersion: ModelVersionManifest?
    private var lastPreviewTime = 0.0
    private var previewFPS = 0.0
    private var runtimeMetrics = RuntimeMetrics()
    private var reinforcementMetrics = ReinforcementLearningMetrics()
    private var lastMetricsPublishTime = 0.0
    private var lastRewardMessageCount = 0
    private var lastRewardRateTime = 0.0
    private var lastFocusCheckTime = 0.0
    private var disconnectSafetyRevision: UInt64 = 0
    private var disconnectSafetyTask: Task<Void, Never>?

    var isRunning: Bool {
        lock.withLock { !stopped || starting || teardownInProgress }
    }

    init() throws {
        MLXMemoryLifecycle.configure()
        preprocessor = try VisionPreprocessor()
        injector.onState = { [weak self] state in self?.onState?(state) }
    }

    func start(
        profile requestedProfile: AIProfile,
        version: ModelVersionManifest?,
        webSocketSettings requestedWebSocketSettings: RewardWebSocketServerSettings,
        captureSpec requestedCaptureSpec: CaptureSpec,
        captureRect: CGRect,
        frameMode: FrameMode,
        mouseMode: MouseControlMode,
        gameCamera: GameCameraSettings,
        outputPermissions: RuntimeOutputPermissions,
        contextSelection: [UUID: UUID],
        safety: AgentSafetyPolicy,
        previewFPS: Double,
        ignoredHotkeys: [HotkeyBinding],
        metrics: @escaping MetricsHandler,
        completion: @escaping CompletionHandler
    ) async throws {
        guard AXIsProcessTrusted() else {
            let options = [
                kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true
            ] as CFDictionary
            AXIsProcessTrustedWithOptions(options)
            throw AgentTrainerError.permission(
                "Accessibility permission is required before RL training can control keys or the mouse."
            )
        }
        let accepted = lock.withLock { () -> Bool in
            guard stopped, !starting, !teardownInProgress else { return false }
            starting = true
            stopped = false
            return true
        }
        guard accepted else {
            throw AgentTrainerError.model("RL training is already starting or running.")
        }
        metricsHandler = metrics
        completionHandler = completion

        do {
            let configuration = try requestedProfile.effectiveReinforcementLearning.validated()
            let webSocketSettings = try requestedWebSocketSettings.validated()
            var actorProfile = requestedProfile
            let contextSchema: ModelContextSchema
            if let version {
                guard version.schemaVersion == ModelContract.schemaVersion,
                      version.relativeMouseScale == GameCameraContract.deltaScale else {
                    throw AgentTrainerError.model(
                        "The selected brain does not use the current Policy-v6 runtime contract."
                    )
                }
                actorProfile.preprocessing = version.preprocessing
                actorProfile.training = version.training
                actorProfile.channels = RuntimeActionSemantics.effectiveChannels(
                    saved: version.channels,
                    current: requestedProfile.channels
                )
                contextSchema = try version.effectiveContextSchema.validated()
            } else {
                contextSchema = .empty
            }
            _ = try actorProfile.preprocessing.validated()
            _ = try actorProfile.training.effectiveTemporalVision.validated(
                current: actorProfile.preprocessing
            )
            let actionPlan = try ReinforcementActionPlan.make(
                profile: actorProfile,
                configuration: configuration,
                mouseMode: mouseMode,
                outputPermissions: outputPermissions
            )
            let rolloutBytes = configuration.estimatedRolloutBytes(
                profile: actorProfile,
                contextSchema: contextSchema
            )
            let physicalMemory = Int64(ProcessInfo.processInfo.physicalMemory)
            let modelWorkingSet = ModelSizing.estimatedTrainingWorkingSet(
                actorProfile,
                contextSchema: contextSchema
            )
            let combinedWorkingSet = modelWorkingSet.addingReportingOverflow(
                Int64(rolloutBytes)
            )
            guard rolloutBytes <= ReinforcementLearningConfiguration.maximumRolloutBytes,
                  Int64(rolloutBytes) <= physicalMemory / 4,
                  !combinedWorkingSet.overflow,
                  combinedWorkingSet.partialValue <= physicalMemory * 3 / 4 else {
                throw AgentTrainerError.invalidConfiguration(
                    "This RL rollout would retain \(ByteCountFormatter.string(fromByteCount: Int64(rolloutBytes), countStyle: .memory)) of observations. Reduce Rollout Steps, temporal frames, or model vision so online training remains bounded."
                )
            }
            var executionProfile = actorProfile
            executionProfile.channels = configuration.actionSpace.effectiveChannels(
                base: actorProfile.channels,
                mouseMode: mouseMode
            )
            executionProfile.restrictions = configuration.actionSpace.applying(
                to: requestedProfile.effectiveRestrictions
            )
            // Publish enough immutable startup context before the first actor
            // await that a keybind cancellation can complete cleanly without
            // inventing a missing-profile failure.
            lock.withLock {
                self.profile = requestedProfile
                self.actorProfile = actorProfile
                self.executionProfile = executionProfile
                self.configuration = configuration
                self.webSocketSettings = webSocketSettings
                self.contextSchema = contextSchema
                self.contextVector = contextSchema.vector(
                    selection: contextSelection
                )
                self.actionPlan = actionPlan
                self.captureRect = captureRect
                self.frameMode = frameMode
                self.mouseMode = mouseMode
                self.gameCamera = gameCamera
                self.outputPermissions = outputPermissions
                self.safety = safety
            }

            let profileRoot = await WorkspaceStore.shared.profileDirectory(requestedProfile.id)
            let checkpointDirectory = await WorkspaceStore.shared.reinforcementCheckpointDirectory(
                profileID: requestedProfile.id
            )
            let signature = try Self.checkpointSignature(
                profile: actorProfile,
                configuration: configuration,
                contextSchema: contextSchema,
                mouseMode: mouseMode
            )
            let restoredState = Self.loadCheckpointState(
                at: checkpointDirectory,
                expectedSignature: signature,
                expectedActorVersionID: requestedProfile.activeVersionID
            )
            let checkpointFilesAreComplete = restoredState != nil
                && ["rl.weights.safetensors", "optimizer.safetensors", "random.safetensors"]
                    .allSatisfy {
                        FileManager.default.fileExists(
                            atPath: checkpointDirectory.appendingPathComponent($0).path
                        )
                    }
            let activeWeightsURL: URL? = if let version {
                await WorkspaceStore.shared.versionDirectory(
                    profileID: requestedProfile.id,
                    versionID: version.id
                ).appendingPathComponent(version.weightsFile)
            } else {
                nil
            }
            let policyOptimizer = try Device.withDefaultDevice(.gpu) {
                try ReinforcementPolicyOptimizer(
                    profile: actorProfile,
                    configuration: configuration,
                    contextSchema: contextSchema,
                    actionPlan: actionPlan,
                    actorWeightsURL: activeWeightsURL,
                    reinforcementWeightsURL: checkpointFilesAreComplete
                        ? checkpointDirectory.appendingPathComponent("rl.weights.safetensors") : nil,
                    optimizerURL: checkpointFilesAreComplete
                        ? checkpointDirectory.appendingPathComponent("optimizer.safetensors") : nil,
                    randomStateURL: checkpointFilesAreComplete
                        ? checkpointDirectory.appendingPathComponent("random.safetensors") : nil
                )
            }
            let state = restoredState ?? Self.initialCheckpointState(
                signature: signature,
                profile: requestedProfile,
                version: version,
                seed: configuration.seed
            )

            guard lock.withLock({ !stopped }) else { throw CancellationError() }
            try await rewardServer.start(settings: webSocketSettings)
            try await rewardServer.waitForRewarder(
                timeout: webSocketSettings.connectionWaitTimeoutSeconds
            )
            guard lock.withLock({ !stopped }) else { throw CancellationError() }

            let now = CACurrentMediaTime()
            let initialCursor = Self.normalizedCursor(in: captureRect)
            let installed = lock.withLock { () -> Bool in
                guard !stopped else { return false }
                self.profile = requestedProfile
                self.actorProfile = actorProfile
                self.executionProfile = executionProfile
                self.configuration = configuration
                self.contextSchema = contextSchema
                self.contextVector = contextSchema.vector(selection: contextSelection)
                self.actionPlan = actionPlan
                self.policyOptimizer = policyOptimizer
                self.checkpointState = state
                self.checkpointDirectory = checkpointDirectory
                self.profileRoot = profileRoot
                self.captureRect = captureRect
                self.frameMode = frameMode
                self.mouseMode = mouseMode
                self.gameCamera = gameCamera
                self.outputPermissions = outputPermissions
                self.safety = safety
                self.previewFPS = max(0, previewFPS)
                self.trainingShowsCursor = version?.trainingShowsCursor
                    ?? requestedCaptureSpec.showsCursor
                samplingRandom = ReinforcementRandomGenerator(
                    seed: state.samplingRandomState
                )
                actionFilter.reset(absoluteMousePosition: initialCursor)
                rewardStatistics = state.rewardStatistics
                rollout.removeAll(keepingCapacity: false)
                rollout.reserveCapacity(configuration.rolloutSteps)
                pendingTransition = nil
                temporalFrames.reset()
                predictionLatch.reset()
                latestFrame = nil
                latestFrameHostNanos = 0
                lastUsableFrame = nil
                processing = false
                nextPerceptionTime = 0
                rewardIsStale = false
                rewardStaleStartedAt = nil
                connectionPauseStartedAt = nil
                rewardFreshnessBaseline = now
                updateInProgress = false
                episodeBoundaryInProgress = false
                episodeIsOpen = true
                currentEpisode = state.completedEpisodes + 1
                sessionStartingCompletedEpisodes = state.completedEpisodes
                sessionStartingUpdates = state.updates
                episodeStep = 0
                episodeReturn = 0
                episodeStartedAt = now
                sessionStartedAt = now
                lastPublishedUpdate = version?.reinforcementLearningSummary?.update ?? 0
                latestPublishedVersion = nil
                lastPreviewTime = 0
                runtimeMetrics = RuntimeMetrics()
                reinforcementMetrics = ReinforcementLearningMetrics(
                    connectionState: connectionState,
                    episode: currentEpisode,
                    sessionEpisode: 1,
                    update: state.updates,
                    environmentSteps: state.environmentSteps,
                    averageEpisodeReturn: Self.average(
                        cumulativeReturn: state.cumulativeEpisodeReturn,
                        episodeCount: state.completedEpisodes
                    ),
                    elapsed: state.elapsed
                )
                lastMetricsPublishTime = 0
                let rewardStats = rewardAccumulator.statistics()
                lastRewardMessageCount = rewardStats.received
                lastRewardRateTime = now
                lastFocusCheckTime = 0
                if connectionState == .connected {
                    injector.enable(outputPermissions: outputPermissions)
                }
                return true
            }
            guard installed else { throw CancellationError() }

            guard lock.withLock({ !stopped && connectionState == .connected }) else {
                throw RewardWebSocketError.notConnected
            }

            await sendEpisodeMessage(
                event: .episodeStarting,
                episode: currentEpisode
            )
            if configuration.interEpisodeStartDelaySeconds > 0 {
                try await Task.sleep(
                    for: .seconds(configuration.interEpisodeStartDelaySeconds)
                )
            }
            guard lock.withLock({ !stopped }) else { throw CancellationError() }

            safetyMonitor.ignoredHotkeys = ignoredHotkeys
            safetyMonitor.onSample = { [weak self] _ in
                guard let self, self.lock.withLock({ !self.stopped }),
                      self.lock.withLock({ self.safety.stopOnHumanInput }) else { return }
                Task { await self.stop(reason: "RL stopped on human input") }
            }
            try safetyMonitor.start()
            let targetPID = await focusTargetIfNeeded(requestedCaptureSpec)
            guard lock.withLock({ !stopped }) else { throw CancellationError() }
            lock.withLock { self.targetPID = targetPID }

            var liveCaptureSpec = requestedCaptureSpec
            liveCaptureSpec.requestedFPS = actorProfile.training.perceptionFPS
            liveCaptureSpec.showsCursor = trainingShowsCursor
            try await capture.start(
                spec: liveCaptureSpec,
                queueDepth: frameMode == .newest ? 3 : 8,
                onFrame: { [weak self] buffer, pts in
                    self?.schedule(
                        buffer,
                        timestampNanos: Self.hostTimeNanos(pts),
                        countThrottledAsDropped: true
                    )
                },
                onIdle: { [weak self] pts in
                    self?.receiveIdle(timestampNanos: Self.hostTimeNanos(pts))
                },
                onUnexpectedStop: { [weak self] error in
                    Task {
                        await self?.stop(
                            reason: "RL capture stopped: \(error.localizedDescription)"
                        )
                    }
                }
            )
            guard lock.withLock({ !stopped }) else { throw CancellationError() }
            let activeStart = CACurrentMediaTime()
            lock.withLock {
                episodeStartedAt = activeStart
                sessionStartedAt = activeStart
                rewardFreshnessBaseline = activeStart
            }
            startActionTimer(fps: actorProfile.training.actionFPS)
            lock.withLock { starting = false }
            rewardAccumulator.resetEpisode(clearLastMessageTime: true)
            await sendEpisodeMessage(event: .episodeStarted, episode: currentEpisode)
            let initialStatus = checkpointFilesAreComplete
                ? "RL connected • exact PPO checkpoint restored • episode \(currentEpisode)"
                : version == nil
                    ? "RL connected • new Policy-v6 actor • episode \(currentEpisode)"
                    : "RL connected • active brain warm-started • episode \(currentEpisode)"
            inferenceQueue.async { [weak self] in
                self?.publishMetrics(status: initialStatus)
            }
        } catch {
            lock.withLock { starting = false }
            await teardownWithoutSnapshot()
            throw error
        }
    }

    func stop(reason: String = "RL training stopped by keybind") async {
        let transition = lock.withLock { () -> (Bool, DispatchSourceTimer?) in
            guard !teardownInProgress, (!stopped || starting) else {
                return (false, nil)
            }
            stopped = true
            starting = false
            teardownInProgress = true
            disconnectSafetyRevision &+= 1
            disconnectSafetyTask?.cancel()
            disconnectSafetyTask = nil
            let timer = actionTimer
            actionTimer = nil
            latestFrame = nil
            latestFrameHostNanos = 0
            predictionLatch.reset()
            return (true, timer)
        }
        guard transition.0 else { return }
        let timer = transition.1
        timer?.setEventHandler {}
        timer?.cancel()
        await drain(queue: actionQueue)
        injector.disableAndReleaseAll()
        safetyMonitor.stop()
        safetyMonitor.onSample = nil
        _ = try? await capture.stop()
        await drain(queue: inferenceQueue)

        let finalization = ReinforcementFinalizationBox()
        await withCheckedContinuation { continuation in
            inferenceQueue.async { [weak self] in
                guard let self else {
                    finalization.set(.failure(CancellationError()))
                    continuation.resume()
                    return
                }
                do {
                    try self.finalizePendingTransition(terminal: true)
                    if ReinforcementRolloutPolicy.canUpdateFinalBatch(
                        transitionCount: self.rollout.count
                    ) {
                        _ = try self.performPolicyUpdate(bootstrapValue: 0)
                    }
                    let hasUnpublishedUpdate = (self.checkpointState?.updates ?? 0)
                        > self.lock.withLock { self.lastPublishedUpdate }
                    let prepared: PreparedReinforcementSnapshot?
                    if hasUnpublishedUpdate {
                        // Snapshot preparation saves the exact checkpoint and
                        // the standalone runtime actor in one ordered operation.
                        prepared = try self.prepareSnapshot(isAutosave: false)
                    } else {
                        // A stop caused by quit, panic, capture loss, reward
                        // timeout, or disconnect still refreshes the recoverable
                        // actor-critic checkpoint between normal save cadences.
                        try self.saveCheckpoint()
                        prepared = nil
                    }
                    finalization.set(.success(prepared))
                } catch {
                    finalization.set(.failure(error))
                }
                continuation.resume()
            }
        }
        if let episode = lock.withLock({ () -> Int? in
            guard episodeIsOpen, currentEpisode > 0 else { return nil }
            episodeIsOpen = false
            return currentEpisode
        }) {
            await sendEpisodeMessage(
                event: .episodeFinished,
                episode: episode,
                reason: Self.protocolStopReason(reason),
                episodeReturn: episodeReturn,
                steps: episodeStep
            )
        }
        let completionResult: Result<ReinforcementLearningCompletion, Error>
        switch finalization.value {
        case .failure(let error):
            completionResult = .failure(error)
        case .success(let prepared?) :
            do {
                let updated = try await WorkspaceStore.shared.commitReinforcementVersion(
                    prepared.manifest,
                    profile: prepared.profile,
                    progress: prepared.progress
                )
                onVersionPublished?(updated, prepared.manifest)
                completionResult = .success(
                    ReinforcementLearningCompletion(
                        profile: updated,
                        version: prepared.manifest,
                        reason: reason
                    )
                )
            } catch {
                completionResult = .failure(error)
            }
        case .success(nil):
            if let profile = lock.withLock({ self.profile }) {
                let publishedVersion = lock.withLock {
                    self.latestPublishedVersion
                }
                completionResult = .success(
                    ReinforcementLearningCompletion(
                        profile: profile,
                        version: publishedVersion,
                        reason: reason
                    )
                )
            } else {
                completionResult = .failure(
                    AgentTrainerError.model("RL stopped before its profile was prepared.")
                )
            }
        case nil:
            completionResult = .failure(CancellationError())
        }
        await rewardServer.stop()
        finishTeardown()
        completionHandler?(completionResult)
    }

    func updateOutputPermissions(_ permissions: RuntimeOutputPermissions) {
        let invalidatesActionSpace = lock.withLock { () -> Bool in
            let movementIsConfigured = actorProfile?.channels.mouseMovement == true
                && (mouseMode == .absolute
                    ? configuration.actionSpace.allowsAbsoluteMouseMovement
                    : configuration.actionSpace.allowsRelativeMouseMovement)
            let keyboardIsConfigured = !configuration.actionSpace
                .allowedKeyCodes.isEmpty
                && (actorProfile?.channels.keyboard == true
                    || actorProfile?.channels.modifiers == true)
            let changesMovement = outputPermissions.cursorMovement
                != permissions.cursorMovement && movementIsConfigured
            let changesKeyboard = outputPermissions.keyboard
                != permissions.keyboard && keyboardIsConfigured
            outputPermissions = permissions
            return !stopped && (changesMovement || changesKeyboard)
        }
        actionQueue.async { [weak self] in
            guard let self else { return }
            self.injector.updateOutputPermissions(permissions)
        }
        if invalidatesActionSpace {
            Task {
                await stop(
                    reason: "RL saved and stopped because live output permissions changed its active action space"
                )
            }
        }
    }

    func updateContextSelection(_ selection: [UUID: UUID]) {
        let updated = lock.withLock { contextSchema.vector(selection: selection) }
        inferenceQueue.async { [weak self] in
            guard let self else { return }
            let changed = self.lock.withLock { () -> Bool in
                guard updated != self.contextVector else { return false }
                self.contextVector = updated
                return !self.stopped && !self.episodeBoundaryInProgress
            }
            guard changed else { return }
            do { try self.finalizePendingTransition(terminal: true) }
            catch {
                Task { await self.stop(reason: error.localizedDescription) }
                return
            }
            self.beginEpisodeBoundary(reason: "context_changed")
        }
    }

    private func teardownWithoutSnapshot() async {
        lock.withLock {
            stopped = true
            starting = false
            actionTimer?.setEventHandler {}
            actionTimer?.cancel()
            actionTimer = nil
        }
        safetyMonitor.stop()
        safetyMonitor.onSample = nil
        _ = try? await capture.stop()
        await drain(queue: actionQueue)
        injector.disableAndReleaseAll()
        await drain(queue: inferenceQueue)
        await rewardServer.stop()
        finishTeardown()
    }

    private func finishTeardown() {
        lock.withLock {
            policyOptimizer = nil
            profile = nil
            actorProfile = nil
            executionProfile = nil
            actionPlan = nil
            actionFilter.reset()
            checkpointState = nil
            checkpointDirectory = nil
            profileRoot = nil
            contextSchema = .empty
            contextVector.removeAll(keepingCapacity: false)
            lastPublishedUpdate = 0
            latestPublishedVersion = nil
            rollout.removeAll(keepingCapacity: false)
            pendingTransition = nil
            temporalFrames.reset()
            predictionLatch.reset()
            latestFrame = nil
            lastUsableFrame = nil
            processing = false
            targetPID = nil
            connectionState = .disconnected
            updateInProgress = false
            episodeBoundaryInProgress = false
            episodeIsOpen = false
            disconnectSafetyRevision &+= 1
            disconnectSafetyTask?.cancel()
            disconnectSafetyTask = nil
            teardownInProgress = false
        }
        MLXMemoryLifecycle.reclaimCaches(after: "reinforcement learning")
    }

    private func schedule(
        _ buffer: CVPixelBuffer,
        timestampNanos: UInt64,
        countThrottledAsDropped: Bool
    ) {
        let now = CACurrentMediaTime()
        lock.lock()
        guard !stopped, !episodeBoundaryInProgress, !updateInProgress,
              let actorProfile else { lock.unlock(); return }
        lastUsableFrame = buffer
        let interval = 1 / max(0.0001, actorProfile.training.perceptionFPS)
        guard now >= nextPerceptionTime else {
            if countThrottledAsDropped { runtimeMetrics.droppedFrames += 1 }
            lock.unlock()
            return
        }
        nextPerceptionTime = now + interval
        if frameMode == .ordered {
            let retained = ReinforcementPixelBuffer(buffer)
            lock.unlock()
            inferenceQueue.sync { [weak self] in
                self?.infer(retained.value, timestampNanos: timestampNanos)
            }
            return
        }
        latestFrame = buffer
        latestFrameHostNanos = timestampNanos
        if processing { lock.unlock(); return }
        processing = true
        lock.unlock()
        inferenceQueue.async { [weak self] in self?.drainNewest() }
    }

    private func receiveIdle(timestampNanos: UInt64) {
        let frame = lock.withLock { stopped ? nil : lastUsableFrame }
        if let frame {
            schedule(
                frame,
                timestampNanos: timestampNanos,
                countThrottledAsDropped: false
            )
        }
    }

    private func drainNewest() {
        while true {
            lock.lock()
            guard let frame = latestFrame, !stopped else {
                processing = false
                lock.unlock()
                return
            }
            let timestamp = latestFrameHostNanos
            latestFrame = nil
            latestFrameHostNanos = 0
            lock.unlock()
            infer(frame, timestampNanos: timestamp)
            lock.lock()
            if latestFrame == nil {
                processing = false
                lock.unlock()
                return
            }
            lock.unlock()
        }
    }

    private func infer(_ buffer: CVPixelBuffer, timestampNanos _: UInt64) {
        let staleBaseline = lock.withLock {
            rewardIsStale ? rewardFreshnessBaseline : nil
        }
        if let staleBaseline {
            let lastMessage = rewardAccumulator.statistics().lastMessageTime ?? 0
            if lastMessage > staleBaseline {
                let resumedSignal = rewardAccumulator.drain(
                    aggregation: configuration.rewardAggregation
                )
                resumeAfterFreshReward()
                if resumedSignal.episodeFinished {
                    beginEpisodeBoundary(reason: "remote_done")
                } else {
                    publishMetrics(status: "RL rewards resumed • waiting for a fresh perception")
                }
            }
            return
        }
        let session = lock.withLock { () -> (
            ReinforcementPolicyOptimizer,
            AIProfile,
            ReinforcementActionPlan,
            [Float]
        )? in
            guard !stopped, !episodeBoundaryInProgress, !updateInProgress,
                  connectionState == .connected,
                  let policyOptimizer, let actorProfile, let actionPlan else {
                return nil
            }
            return (policyOptimizer, actorProfile, actionPlan, contextVector)
        }
        guard let session else { return }
        let began = CACurrentMediaTime()
        do {
            let temporal = session.1.training.effectiveTemporalVision
            let pastSpec = temporal.pastFrameSpec(from: session.1.preprocessing)
            let selectedPast = lock.withLock {
                (0..<temporal.pastFrameCount).map { frame in
                    temporalFrames.frame(
                        distanceBack: temporal.frameSpacing
                            * (temporal.pastFrameCount - frame)
                    )
                }
            }
            let sharesRepresentation = temporal.pastFrameCount > 0
                && pastSpec == session.1.preprocessing
            let jobs = try preprocessor.submit(
                buffer,
                specs: temporal.pastFrameCount > 0 && !sharesRepresentation
                    ? [session.1.preprocessing, pastSpec]
                    : [session.1.preprocessing]
            )
            guard let currentJob = jobs.first else {
                throw AgentTrainerError.model("The RL vision pipeline returned no current frame.")
            }
            let currentPacked = try currentJob.withPackedBytes { Data($0) }
            let currentPastPacked: Data
            if temporal.pastFrameCount == 0 {
                currentPastPacked = Data()
            } else if sharesRepresentation {
                currentPastPacked = currentPacked
            } else if jobs.indices.contains(1) {
                currentPastPacked = try jobs[1].withPackedBytes { Data($0) }
            } else {
                throw AgentTrainerError.model("The RL vision pipeline returned no temporal frame.")
            }
            let zeroControls = [Float](repeating: 0, count: ActionLayout.count)
            let resolvedPast = selectedPast.map { frame in
                frame ?? TemporalFrame(
                    packed: currentPastPacked,
                    controls: zeroControls
                )
            }
            var packedPast = Data(
                capacity: temporal.pastFrameCount * pastSpec.sampleByteCount
            )
            resolvedPast.forEach { packedPast.append($0.packed) }
            let observation = ReinforcementObservation(
                currentPacked: currentPacked,
                pastPacked: packedPast,
                pastControls: resolvedPast.flatMap(\.controls),
                contextValues: session.3
            )
            let prediction = try Device.withDefaultDevice(.gpu) {
                try session.0.prediction(for: observation)
            }
            let rewardDrain = rewardAccumulator.drain(
                aggregation: configuration.rewardAggregation
            )
            let now = CACurrentMediaTime()
            let episodeStart = lock.withLock { episodeStartedAt }
            let freshnessBaseline = lock.withLock { rewardFreshnessBaseline }
            let rewardClock = max(
                rewardDrain.lastMessageTime ?? 0,
                max(episodeStart, freshnessBaseline)
            )
            let stale = webSocketSettings.rewardTimeoutSeconds > 0
                && now - rewardClock > webSocketSettings.rewardTimeoutSeconds
            if stale {
                try finalizePendingTransition(terminal: true, rewardDrain: rewardDrain)
                pauseForStaleReward()
                if webSocketSettings.stopAndSaveOnRewardTimeout {
                    Task {
                        await stop(
                            reason: "RL saved and stopped because the reward stream became stale"
                        )
                    }
                } else {
                    publishMetrics(
                        status: "RL paused safely • waiting for a reward or heartbeat message"
                    )
                }
                return
            }
            if rewardIsStale {
                resumeAfterFreshReward()
                // The first post-stall frame seeds a fresh causal timeline.
                return
            }
            try finalizePendingTransition(
                terminal: rewardDrain.episodeFinished,
                rewardDrain: rewardDrain
            )

            let durationFinished = now - episodeStart
                >= configuration.episodeDurationSeconds
            let stepFinished = configuration.maximumEpisodeSteps > 0
                && episodeStep >= configuration.maximumEpisodeSteps
            if rewardDrain.episodeFinished || durationFinished || stepFinished {
                let reason = rewardDrain.episodeFinished
                    ? "remote_done" : stepFinished ? "step_limit" : "time_limit"
                let bootstrap = !rewardDrain.episodeFinished
                    && configuration.bootstrapTimeLimits ? prediction.value : 0
                beginEpisodeBoundary(reason: reason, bootstrapValue: bootstrap)
                return
            }
            if rollout.count >= configuration.rolloutSteps {
                let remoteDone = try performPolicyUpdate(
                    bootstrapValue: prediction.value
                )
                if remoteDone {
                    beginEpisodeBoundary(reason: "remote_done_during_update")
                }
                return
            }

            var random = samplingRandom
            let behavior = configuration.effectiveActionBehavior
            let sample = try session.2.sample(
                logits: prediction.logits,
                behavior: behavior,
                random: &random
            )
            samplingRandom = random
            let stabilizedActions = actionFilter.apply(
                to: sample.executedActions,
                plan: session.2,
                behavior: behavior
            )
            let frameControls = RuntimeActionSemantics.temporalControlValues(
                stabilizedActions,
                channels: executionProfile?.channels,
                restrictions: executionProfile?.effectiveRestrictions
                    ?? ActionRestrictions(),
                allowedKeyCodes: session.2.allowedKeyCodes,
                outputPermissions: outputPermissions,
                shiftUsesKeyboardChannel: true
            )
            pendingTransition = ReinforcementTransition(
                observation: observation,
                distributionActions: sample.distributionActions,
                executedActions: frameControls,
                oldLogProbability: sample.logProbability,
                oldValue: prediction.value,
                reward: 0,
                terminal: false
            )
            if temporal.pastFrameCount > 0 {
                lock.withLock {
                    temporalFrames.append(
                        TemporalFrame(
                            packed: currentPastPacked,
                            controls: frameControls
                        ),
                        capacity: temporal.pastFrameCount * temporal.frameSpacing + 1
                    )
                }
            }
            lock.withLock {
                predictionLatch.publish(frameControls)
                runtimeMetrics.frameCount += 1
                let elapsed = max(0.001, now - sessionStartedAt)
                runtimeMetrics.perceptionFPS = Double(runtimeMetrics.frameCount) / elapsed
                runtimeMetrics.latencyMilliseconds = (CACurrentMediaTime() - began) * 1_000
            }
            episodeStep += 1
            if var state = checkpointState {
                state.environmentSteps += 1
                state.samplingRandomState = samplingRandom.state
                checkpointState = state
            }
            publishPreviewIfNeeded(
                currentPacked,
                spec: session.1.preprocessing,
                now: now
            )
            if webSocketSettings.stopAndSaveOnInvalidMessageFlood,
               rewardAccumulator.statistics().invalid
                    >= webSocketSettings.maximumInvalidMessages {
                Task {
                    await stop(
                        reason: "RL saved and stopped after too many invalid reward messages"
                    )
                }
                return
            }
            publishMetrics(status: "RL training • episode \(currentEpisode) • rollout \(rollout.count)/\(configuration.rolloutSteps)")
        } catch {
            Task { await stop(reason: error.localizedDescription) }
        }
    }

    private func finalizePendingTransition(
        terminal: Bool,
        rewardDrain providedDrain: ReinforcementRewardDrain? = nil
    ) throws {
        guard var transition = pendingTransition else {
            if providedDrain == nil {
                _ = rewardAccumulator.drain(
                    aggregation: configuration.rewardAggregation
                )
            }
            return
        }
        let drain = providedDrain ?? rewardAccumulator.drain(
            aggregation: configuration.rewardAggregation
        )
        var scaledReward = drain.reward * configuration.rewardScale
        if configuration.rewardClipMagnitude > 0 {
            scaledReward = min(
                configuration.rewardClipMagnitude,
                max(-configuration.rewardClipMagnitude, scaledReward)
            )
        }
        guard scaledReward.isFinite else {
            throw AgentTrainerError.model("The conditioned RL reward became non-finite.")
        }
        episodeReturn += scaledReward
        let learningReward = configuration.normalizeRewards
            ? rewardStatistics.normalize(
                scaledReward,
                epsilon: configuration.rewardNormalizationEpsilon
            ) : scaledReward
        transition.reward = Float(learningReward)
        transition.terminal = terminal
        rollout.append(transition)
        pendingTransition = nil
        reinforcementMetrics.lastReward = scaledReward
        reinforcementMetrics.episodeReturn = episodeReturn
    }

    private func performPolicyUpdate(bootstrapValue: Float) throws -> Bool {
        guard let policyOptimizer, !rollout.isEmpty else { return false }
        lock.withLock {
            updateInProgress = true
            predictionLatch.reset()
            temporalFrames.reset()
        }
        resetActionFilterToCurrentCursor()
        actionQueue.sync { injector.disableAndReleaseAll() }
        let updateEpisode = currentEpisode
        sendEpisodeMessageSynchronously(
            event: .trainingUpdateStarted,
            episode: updateEpisode
        )
        let began = CACurrentMediaTime()
        let update = try Device.withDefaultDevice(.gpu) {
            try policyOptimizer.update(
                transitions: rollout,
                bootstrapValue: bootstrapValue,
                shuffleSeed: configuration.seed
                    &+ UInt64(checkpointState?.updates ?? 0)
                    &* 0x9E37_79B9_7F4A_7C15
            )
        }
        rollout.removeAll(keepingCapacity: true)
        let pausedSignal = rewardAccumulator.drain(
            aggregation: configuration.rewardAggregation
        )
        reinforcementMetrics.rewardMessagesIgnoredDuringUpdates +=
            pausedSignal.messageCount
        let updateDuration = CACurrentMediaTime() - began
        lock.withLock { episodeStartedAt += updateDuration }
        if var state = checkpointState {
            state.updates += 1
            state.globalStep += update.optimizerSteps
            state.rewardStatistics = rewardStatistics
            state.samplingRandomState = samplingRandom.state
            state.lastPolicyLoss = update.policyLoss
            state.lastValueLoss = update.valueLoss
            state.lastEntropy = update.entropy
            state.lastApproximateKL = update.approximateKL
            state.lastClipFraction = update.clipFraction
            state.lastExplainedVariance = update.explainedVariance
            checkpointState = state
            let updatesThisSession = max(0, state.updates - sessionStartingUpdates)
            if updatesThisSession > 0,
               updatesThisSession.isMultiple(of: configuration.checkpointEveryUpdates) {
                try saveCheckpoint()
            }
        }
        reinforcementMetrics.update = checkpointState?.updates ?? 0
        reinforcementMetrics.policyLoss = update.policyLoss
        reinforcementMetrics.valueLoss = update.valueLoss
        reinforcementMetrics.entropy = update.entropy
        reinforcementMetrics.approximateKL = update.approximateKL
        reinforcementMetrics.clipFraction = update.clipFraction
        reinforcementMetrics.explainedVariance = update.explainedVariance
        let memory = Memory.snapshot()
        reinforcementMetrics.mlxActiveMemory = memory.activeMemory
        reinforcementMetrics.mlxCacheMemory = memory.cacheMemory
        reinforcementMetrics.mlxPeakMemory = memory.peakMemory
        sendEpisodeMessageSynchronously(
            event: .trainingUpdateFinished,
            episode: updateEpisode
        )
        lock.withLock {
            updateInProgress = false
            nextPerceptionTime = 0
            rewardFreshnessBaseline = CACurrentMediaTime()
            if !stopped, connectionState == .connected, !rewardIsStale,
               !pausedSignal.episodeFinished {
                injector.enable(outputPermissions: outputPermissions)
            }
        }
        return pausedSignal.episodeFinished
    }

    private func beginEpisodeBoundary(reason: String, bootstrapValue: Float = 0) {
        let accepted = lock.withLock { () -> Bool in
            guard !episodeBoundaryInProgress, !stopped else { return false }
            episodeBoundaryInProgress = true
            episodeIsOpen = false
            predictionLatch.reset()
            temporalFrames.reset()
            return true
        }
        guard accepted else { return }
        resetActionFilterToCurrentCursor()
        let finishedEpisode = currentEpisode
        let finishedReturn = episodeReturn
        let finishedSteps = episodeStep
        let anticipatedLifetimeCompleted = (checkpointState?.completedEpisodes ?? 0) + 1
        let anticipatedMaximum = ReinforcementSessionLimits.reachedMaximumEpisodes(
            maximumEpisodes: configuration.maximumEpisodes,
            lifetimeCompleted: anticipatedLifetimeCompleted,
            lifetimeAtSessionStart: sessionStartingCompletedEpisodes
        )
        actionQueue.sync { injector.disableAndReleaseAll() }
        sendEpisodeMessageSynchronously(
            event: .episodeFinished,
            episode: finishedEpisode,
            reason: reason,
            episodeReturn: finishedReturn,
            steps: finishedSteps,
            nextEpisode: anticipatedMaximum ? nil : finishedEpisode + 1
        )
        let canUpdateAtBoundary = ReinforcementRolloutPolicy.canUpdate(
            transitionCount: rollout.count
        ) || (anticipatedMaximum
            && ReinforcementRolloutPolicy.canUpdateFinalBatch(
                transitionCount: rollout.count
            ))
        if !canUpdateAtBoundary, !rollout.isEmpty {
            // A retained short rollout must not propagate GAE through an
            // environment reset. Immediate larger updates can bootstrap time
            // limits from the current critic; deferred tiny segments use a
            // terminal separator instead of joining unrelated episodes.
            rollout[rollout.count - 1].terminal = true
        }
        if canUpdateAtBoundary {
            do { _ = try performPolicyUpdate(bootstrapValue: bootstrapValue) }
            catch {
                failEpisodeBoundary(
                    error,
                    episode: finishedEpisode,
                    episodeReturn: finishedReturn,
                    steps: finishedSteps
                )
                return
            }
        }
        if var state = checkpointState {
            state.completedEpisodes += 1
            state.episodeReturns.append(finishedReturn)
            state.cumulativeEpisodeReturn += finishedReturn
            if state.episodeReturns.count > 1_024 {
                state.episodeReturns.removeFirst(state.episodeReturns.count - 1_024)
            }
            state.rewardStatistics = rewardStatistics
            state.samplingRandomState = samplingRandom.state
            checkpointState = state
        }
        reinforcementMetrics.lastEpisodeReturn = finishedReturn
        reinforcementMetrics.averageEpisodeReturn = Self.average(
            cumulativeReturn: checkpointState?.cumulativeEpisodeReturn ?? 0,
            episodeCount: checkpointState?.completedEpisodes ?? 0
        )
        let completedThisSession = ReinforcementSessionLimits.completedEpisodes(
            lifetimeCompleted: checkpointState?.completedEpisodes ?? 0,
            lifetimeAtSessionStart: sessionStartingCompletedEpisodes
        )
        let reachedMaximum = ReinforcementSessionLimits.reachedMaximumEpisodes(
            maximumEpisodes: configuration.maximumEpisodes,
            lifetimeCompleted: checkpointState?.completedEpisodes ?? 0,
            lifetimeAtSessionStart: sessionStartingCompletedEpisodes
        )
        let shouldPublish = completedThisSession > 0
            && completedThisSession.isMultiple(
                of: configuration.publishEveryEpisodes
            )
        var prepared: PreparedReinforcementSnapshot?
        if (shouldPublish || reachedMaximum),
           (checkpointState?.updates ?? 0) > lock.withLock({ lastPublishedUpdate }) {
            do { prepared = try prepareSnapshot(isAutosave: !reachedMaximum) }
            catch {
                failEpisodeBoundary(
                    error,
                    episode: finishedEpisode,
                    episodeReturn: finishedReturn,
                    steps: finishedSteps
                )
                return
            }
        }
        Task { [weak self] in
            guard let self else { return }
            if let prepared {
                do {
                    let updated = try await WorkspaceStore.shared.commitReinforcementVersion(
                        prepared.manifest,
                        profile: prepared.profile,
                        progress: prepared.progress
                    )
                    self.onVersionPublished?(updated, prepared.manifest)
                    self.lock.withLock {
                        self.profile = updated
                        self.lastPublishedUpdate = prepared.manifest
                            .reinforcementLearningSummary?.update
                            ?? self.lastPublishedUpdate
                        self.latestPublishedVersion = prepared.manifest
                    }
                } catch {
                    await self.stop(reason: "RL snapshot publication failed: \(error.localizedDescription)")
                    return
                }
            }
            if reachedMaximum {
                await self.stop(reason: "RL completed its configured episode limit")
                return
            }
            await self.sendEpisodeMessage(
                event: .episodeStarting,
                episode: finishedEpisode + 1
            )
            let delay = self.configuration.interEpisodeDelaySeconds
                + self.configuration.interEpisodeStartDelaySeconds
            if delay > 0 { try? await Task.sleep(for: .seconds(delay)) }
            self.inferenceQueue.async { [weak self] in
                self?.finishEpisodeBoundary(nextEpisode: finishedEpisode + 1)
            }
        }
    }

    private func failEpisodeBoundary(
        _ error: Error,
        episode _: Int,
        episodeReturn _: Double,
        steps _: Int
    ) {
        Task { [weak self] in
            guard let self else { return }
            await self.stop(reason: error.localizedDescription)
        }
    }

    private func finishEpisodeBoundary(nextEpisode: Int) {
        guard !stopped else { return }
        currentEpisode = nextEpisode
        episodeStep = 0
        episodeReturn = 0
        rewardAccumulator.resetEpisode(clearLastMessageTime: true)
        pendingTransition = nil
        // Keep a terminal-marked short rollout across episode boundaries. GAE
        // already stops at each terminal, and discarding fewer than four steps
        // here made short externally-ended episodes incapable of learning.
        resetActionFilterToCurrentCursor()
        lock.withLock {
            episodeBoundaryInProgress = false
            episodeIsOpen = true
            updateInProgress = false
            rewardIsStale = false
            rewardStaleStartedAt = nil
            episodeStartedAt = CACurrentMediaTime()
            rewardFreshnessBaseline = episodeStartedAt
            nextPerceptionTime = 0
            temporalFrames.reset()
            predictionLatch.reset()
            if connectionState == .connected {
                injector.enable(outputPermissions: outputPermissions)
            }
        }
        reinforcementMetrics.episode = nextEpisode
        reinforcementMetrics.sessionEpisode = max(
            1,
            nextEpisode - sessionStartingCompletedEpisodes
        )
        reinforcementMetrics.episodeStep = 0
        reinforcementMetrics.episodeReturn = 0
        Task {
            await sendEpisodeMessage(event: .episodeStarted, episode: nextEpisode)
        }
        publishMetrics(status: "RL episode \(nextEpisode) started")
    }

    private func pauseForStaleReward() {
        let paused = lock.withLock { () -> Bool in
            guard !rewardIsStale else { return false }
            rewardIsStale = true
            let now = CACurrentMediaTime()
            rewardStaleStartedAt = now
            rewardFreshnessBaseline = now
            predictionLatch.reset()
            temporalFrames.reset()
            return true
        }
        guard paused else { return }
        resetActionFilterToCurrentCursor()
        actionQueue.async { [weak self] in self?.injector.disableAndReleaseAll() }
    }

    private func resumeAfterFreshReward() {
        rewardAccumulator.resetEpisode(clearLastMessageTime: false)
        resetActionFilterToCurrentCursor()
        lock.withLock {
            if let rewardStaleStartedAt {
                episodeStartedAt += max(0, CACurrentMediaTime() - rewardStaleStartedAt)
            }
            rewardIsStale = false
            rewardStaleStartedAt = nil
            rewardFreshnessBaseline = CACurrentMediaTime()
            temporalFrames.reset()
            predictionLatch.reset()
            nextPerceptionTime = 0
        }
        actionQueue.async { [weak self] in
            guard let self else { return }
            self.lock.withLock {
                guard !self.stopped, self.connectionState == .connected else { return }
                self.injector.enable(outputPermissions: self.outputPermissions)
            }
        }
    }

    private func handleConnectionState(
        _ state: ReinforcementConnectionState,
        detail: String?
    ) {
        let transition = lock.withLock { () -> (
            beganPause: Bool,
            scheduleSafetyStop: Bool,
            revision: UInt64,
            grace: Double
        ) in
            connectionState = state
            disconnectSafetyRevision &+= 1
            let revision = disconnectSafetyRevision
            disconnectSafetyTask?.cancel()
            disconnectSafetyTask = nil
            var beganPause = false
            if state != .connected {
                predictionLatch.reset()
                temporalFrames.reset()
                if profile != nil, connectionPauseStartedAt == nil,
                   !rewardIsStale {
                    connectionPauseStartedAt = CACurrentMediaTime()
                    beganPause = true
                }
            } else if let connectionPauseStartedAt {
                episodeStartedAt += max(
                    0,
                    CACurrentMediaTime() - connectionPauseStartedAt
                )
                self.connectionPauseStartedAt = nil
                rewardFreshnessBaseline = CACurrentMediaTime()
                temporalFrames.reset()
                predictionLatch.reset()
            }
            return (
                beganPause,
                beganPause && webSocketSettings.stopAndSaveOnDisconnect,
                revision,
                webSocketSettings.disconnectGracePeriodSeconds
            )
        }
        if state != .connected {
            actionQueue.async { [weak self] in self?.injector.disableAndReleaseAll() }
            if transition.beganPause {
                inferenceQueue.async { [weak self] in
                    guard let self, !self.lock.withLock({ self.stopped }) else { return }
                    self.resetActionFilterToCurrentCursor()
                    do { try self.finalizePendingTransition(terminal: true) }
                    catch {
                        Task { await self.stop(reason: error.localizedDescription) }
                        return
                    }
                    self.rewardAccumulator.resetEpisode(clearLastMessageTime: true)
                }
            }
            if transition.scheduleSafetyStop {
                let task = Task { [weak self] in
                    if transition.grace > 0 {
                        try? await Task.sleep(for: .seconds(transition.grace))
                    }
                    guard !Task.isCancelled, let self,
                          self.lock.withLock({
                              !self.stopped
                                  && self.connectionState != .connected
                                  && self.disconnectSafetyRevision
                                      == transition.revision
                          }) else { return }
                    await self.stop(
                        reason: "RL saved and stopped because the rewarder disconnected"
                    )
                }
                lock.withLock {
                    if disconnectSafetyRevision == transition.revision {
                        disconnectSafetyTask = task
                    } else {
                        task.cancel()
                    }
                }
            }
        } else {
            rewardAccumulator.resetEpisode(clearLastMessageTime: true)
            inferenceQueue.async { [weak self] in
                self?.resetActionFilterToCurrentCursor()
            }
            actionQueue.async { [weak self] in
                guard let self else { return }
                self.lock.withLock {
                    guard !self.stopped, !self.updateInProgress,
                          !self.episodeBoundaryInProgress,
                          !self.rewardIsStale else { return }
                    self.injector.enable(outputPermissions: self.outputPermissions)
                }
            }
        }
        let suffix = detail.map { " • \($0)" } ?? ""
        inferenceQueue.async { [weak self] in
            self?.publishMetrics(
                status: "Reward server: \(state.rawValue)\(suffix)"
            )
        }
    }

    private func startActionTimer(fps: Double) {
        let timer = DispatchSource.makeTimerSource(queue: actionQueue)
        timer.schedule(
            deadline: .now(),
            repeating: 1 / max(0.0001, fps),
            leeway: .microseconds(250)
        )
        timer.setEventHandler { [weak self] in self?.actionTick() }
        let accepted = lock.withLock { () -> Bool in
            guard !stopped else { return false }
            actionTimer = timer
            return true
        }
        timer.resume()
        if !accepted { timer.setEventHandler {}; timer.cancel() }
    }

    private func actionTick() {
        let session = lock.withLock { () -> (
            RuntimePredictionLatch.Snapshot,
            AIProfile,
            ReinforcementActionPlan,
            AgentSafetyPolicy,
            CGRect,
            MouseControlMode,
            GameCameraSettings,
            pid_t?
        )? in
            guard !stopped, !updateInProgress, !episodeBoundaryInProgress,
                  !rewardIsStale, connectionState == .connected,
                  let latched = predictionLatch.consume(),
                  let executionProfile, let actionPlan else { return nil }
            return (
                latched, executionProfile, actionPlan, safety, captureRect,
                mouseMode, gameCamera, targetPID
            )
        }
        guard let session else { return }
        let now = CACurrentMediaTime()
        let maximumAge = max(
            0.35,
            3 / max(0.0001, session.1.training.perceptionFPS)
        )
        guard now - session.0.publishedAt <= maximumAge else {
            Task { await stop(reason: "RL stopped because inference became stale") }
            return
        }
        if session.3.stopOnFocusLoss, now - sessionStartedAt > 0.75,
           let targetPID = session.7, now - lastFocusCheckTime >= 0.1 {
            lastFocusCheckTime = now
            guard NSWorkspace.shared.frontmostApplication?.processIdentifier == targetPID else {
                Task { await stop(reason: "RL stopped because the target window lost focus") }
                return
            }
        }
        injector.execute(
            session.0.values,
            profile: session.1,
            allowedKeyCodes: session.2.allowedKeyCodes,
            mouseMode: session.5,
            captureRect: session.4,
            safety: session.3,
            gameCamera: session.6,
            predictionIsFresh: session.0.isFresh,
            shiftUsesKeyboardChannel: true
        )
        lock.withLock {
            runtimeMetrics.actionCount += 1
            runtimeMetrics.actionFPS = Double(runtimeMetrics.actionCount)
                / max(0.001, now - sessionStartedAt)
        }
    }

    private func publishPreviewIfNeeded(
        _ packed: Data,
        spec: PreprocessingSpec,
        now: Double
    ) {
        guard previewFPS > 0, now - lastPreviewTime >= 1 / max(0.1, previewFPS) else { return }
        lastPreviewTime = now
        onPreview?(VisionPreviewFrame(packed: packed, spec: spec, timestamp: now))
    }

    private func publishMetrics(status: String) {
        let now = CACurrentMediaTime()
        guard now - lastMetricsPublishTime >= 0.1 || status.contains("connected")
                || status.contains("paused") || status.contains("started") else { return }
        lastMetricsPublishTime = now
        let rewardCounts = rewardAccumulator.statistics()
        let rewardInterval = max(0.001, now - lastRewardRateTime)
        if rewardInterval >= 0.5 {
            reinforcementMetrics.rewardMessagesPerSecond = Double(
                rewardCounts.received - lastRewardMessageCount
            ) / rewardInterval
            lastRewardMessageCount = rewardCounts.received
            lastRewardRateTime = now
        }
        reinforcementMetrics.connectionState = lock.withLock { connectionState }
        reinforcementMetrics.invalidRewardMessages = rewardCounts.invalid
        reinforcementMetrics.episode = currentEpisode
        reinforcementMetrics.sessionEpisode = max(
            1,
            currentEpisode - sessionStartingCompletedEpisodes
        )
        reinforcementMetrics.environmentSteps = checkpointState?.environmentSteps ?? 0
        reinforcementMetrics.episodeStep = episodeStep
        reinforcementMetrics.rolloutCount = rollout.count + (pendingTransition == nil ? 0 : 1)
        reinforcementMetrics.episodeReturn = episodeReturn
        reinforcementMetrics.elapsed = (checkpointState?.elapsed ?? 0)
            + max(0, now - sessionStartedAt)
        let runtime = lock.withLock { runtimeMetrics }
        onRuntimeMetrics?(runtime)
        metricsHandler?(reinforcementMetrics, status)
    }

    private func prepareSnapshot(isAutosave: Bool) throws -> PreparedReinforcementSnapshot {
        guard let policyOptimizer, var state = checkpointState,
              let profile, let actorProfile, let profileRoot,
              let actionPlan else {
            throw AgentTrainerError.model("The RL session cannot snapshot before its actor is ready.")
        }
        state.elapsed += max(0, CACurrentMediaTime() - sessionStartedAt)
        sessionStartedAt = CACurrentMediaTime()
        state.rewardStatistics = rewardStatistics
        state.samplingRandomState = samplingRandom.state
        let versionID = UUID()
        state.actorVersionID = versionID
        checkpointState = state
        let versionDirectory = profileRoot
            .appendingPathComponent("Versions", isDirectory: true)
            .appendingPathComponent(versionID.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: versionDirectory,
            withIntermediateDirectories: true
        )
        do {
            try policyOptimizer.saveActor(
                to: versionDirectory.appendingPathComponent("weights.safetensors")
            )
            try saveCheckpoint()
        } catch {
            try? FileManager.default.removeItem(at: versionDirectory)
            throw error
        }
        let averageReturn = Self.average(
            cumulativeReturn: state.cumulativeEpisodeReturn,
            episodeCount: state.completedEpisodes
        )
        let combinedLoss = state.lastPolicyLoss
            + configuration.valueLossCoefficient * state.lastValueLoss
            - configuration.entropyCoefficient * state.lastEntropy
        let summary = ReinforcementLearningSummary(
            episode: state.completedEpisodes,
            update: state.updates,
            environmentSteps: state.environmentSteps,
            lastEpisodeReturn: state.episodeReturns.last ?? episodeReturn,
            averageEpisodeReturn: averageReturn
        )
        let manifest = ModelVersionManifest(
            id: versionID,
            name: isAutosave
                ? "RL Autosave • Episode \(state.completedEpisodes) • Update \(state.updates)"
                : "RL Brain • Episode \(state.completedEpisodes) • Update \(state.updates)",
            createdAt: Date(),
            globalStep: state.globalStep,
            trainingLoss: combinedLoss.isFinite ? combinedLoss : 0,
            validationLoss: nil,
            preprocessing: actorProfile.preprocessing,
            channels: actorProfile.channels,
            training: actorProfile.training,
            epoch: state.displayedEpoch,
            isAutosave: isAutosave,
            demonstratedKeyCodes: actionPlan.allowedKeyCodes,
            relativeMouseScale: GameCameraContract.deltaScale,
            trainingDataSchema: TrainingDataContract.schemaVersion,
            trainingDurationSeconds: state.elapsed,
            experienceDurationSeconds: Double(state.environmentSteps)
                / max(0.0001, actorProfile.training.actionFPS),
            trainingShowsCursor: trainingShowsCursor,
            recommendedMouseMode: mouseMode,
            contextSchema: contextSchema.isEmpty ? nil : contextSchema,
            trainingOrigin: .reinforcementLearning,
            reinforcementLearningSummary: summary,
            reinforcementActionSpace: configuration.actionSpace,
            reinforcementActionBehavior: configuration.effectiveActionBehavior
        )
        var progress = TrainingProgressSummary(
            globalStep: state.globalStep,
            epoch: state.displayedEpoch,
            updatedAt: manifest.createdAt,
            savedBrainCount: max(1, (profile.trainingProgress?.savedBrainCount ?? 0) + 1),
            trainingDurationSeconds: state.elapsed,
            experienceDurationSeconds: manifest.experienceDurationSeconds,
            reinforcementEpisodes: state.completedEpisodes,
            reinforcementUpdates: state.updates,
            reinforcementAverageReturn: averageReturn
        )
        if progress.trainingDurationSeconds?.isFinite != true {
            progress.trainingDurationSeconds = 0
        }
        return PreparedReinforcementSnapshot(
            manifest: manifest,
            profile: profile,
            progress: progress
        )
    }

    private func saveCheckpoint() throws {
        guard let policyOptimizer, let state = checkpointState,
              let destination = checkpointDirectory else { return }
        var persistedState = state
        persistedState.elapsed += max(0, CACurrentMediaTime() - sessionStartedAt)
        let temporary = destination.deletingLastPathComponent()
            .appendingPathComponent(".RLCheckpoint.\(UUID().uuidString).tmp")
        try FileManager.default.createDirectory(
            at: temporary,
            withIntermediateDirectories: true
        )
        do {
            try policyOptimizer.saveReinforcementWeights(
                to: temporary.appendingPathComponent("rl.weights.safetensors")
            )
            try policyOptimizer.saveOptimizer(
                to: temporary.appendingPathComponent("optimizer.safetensors")
            )
            try policyOptimizer.saveRandomState(
                to: temporary.appendingPathComponent("random.safetensors")
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(persistedState).write(
                to: temporary.appendingPathComponent("state.json"),
                options: .atomic
            )
            try encoder.encode(ModelContract.schemaVersion).write(
                to: temporary.appendingPathComponent("model-schema.json"),
                options: .atomic
            )
            if FileManager.default.fileExists(atPath: destination.path) {
                let backupName = ".RLCheckpoint.backup.\(UUID().uuidString)"
                let backup = destination.deletingLastPathComponent()
                    .appendingPathComponent(backupName)
                _ = try FileManager.default.replaceItemAt(
                    destination,
                    withItemAt: temporary,
                    backupItemName: backupName,
                    options: .usingNewMetadataOnly
                )
                try? FileManager.default.removeItem(at: backup)
            } else {
                try FileManager.default.moveItem(at: temporary, to: destination)
            }
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
    }

    private func sendEpisodeMessage(
        event: RewardWebSocketProtocol.OutgoingEvent,
        episode: Int,
        reason: String? = nil,
        episodeReturn: Double? = nil,
        steps: Int? = nil,
        nextEpisode: Int? = nil
    ) async {
        guard let data = try? RewardWebSocketProtocol.episodeMessage(
            event: event,
            names: webSocketSettings.messageNames,
            episode: episode,
            reason: reason,
            episodeReturn: episodeReturn,
            steps: steps,
            nextEpisode: nextEpisode
        ) else { return }
        await rewardServer.send(data)
    }

    private func sendEpisodeMessageSynchronously(
        event: RewardWebSocketProtocol.OutgoingEvent,
        episode: Int,
        reason: String? = nil,
        episodeReturn: Double? = nil,
        steps: Int? = nil,
        nextEpisode: Int? = nil
    ) {
        guard let data = try? RewardWebSocketProtocol.episodeMessage(
            event: event,
            names: webSocketSettings.messageNames,
            episode: episode,
            reason: reason,
            episodeReturn: episodeReturn,
            steps: steps,
            nextEpisode: nextEpisode
        ) else { return }
        let delivered = DispatchSemaphore(value: 0)
        Task { [rewardServer] in
            await rewardServer.send(data)
            delivered.signal()
        }
        _ = delivered.wait(timeout: .now() + 1)
    }

    private static func loadCheckpointState(
        at directory: URL,
        expectedSignature: String,
        expectedActorVersionID: UUID?
    ) -> ReinforcementCheckpointState? {
        let stateURL = directory.appendingPathComponent("state.json")
        let markerURL = directory.appendingPathComponent("model-schema.json")
        guard let markerData = try? Data(contentsOf: markerURL),
              (try? JSONDecoder().decode(Int.self, from: markerData))
                == ModelContract.schemaVersion,
              let data = try? Data(contentsOf: stateURL),
              let state = try? JSONDecoder().decode(
                ReinforcementCheckpointState.self,
                from: data
              ),
              state.signature == expectedSignature,
              state.actorVersionID == expectedActorVersionID,
              state.completedEpisodes >= 0, state.updates >= 0,
              state.environmentSteps >= 0, state.globalStep >= 0,
              state.completedEpisodes <= 1_000_000_000,
              state.updates <= 1_000_000_000,
              state.environmentSteps <= 1_000_000_000_000,
              state.elapsed.isFinite, state.elapsed >= 0,
              state.episodeReturns.allSatisfy(\.isFinite),
              state.cumulativeEpisodeReturn.isFinite else { return nil }
        return state
    }

    private static func initialCheckpointState(
        signature: String,
        profile: AIProfile,
        version: ModelVersionManifest?,
        seed: UInt64
    ) -> ReinforcementCheckpointState {
        let summary = version?.reinforcementLearningSummary
        return ReinforcementCheckpointState(
            signature: signature,
            actorVersionID: profile.activeVersionID,
            completedEpisodes: min(1_000_000_000, max(
                0,
                summary?.episode
                    ?? profile.trainingProgress?.reinforcementEpisodes ?? 0
            )),
            updates: min(1_000_000_000, max(
                0,
                summary?.update
                    ?? profile.trainingProgress?.reinforcementUpdates ?? 0
            )),
            environmentSteps: min(1_000_000_000_000, max(
                0,
                summary?.environmentSteps ?? 0
            )),
            globalStep: max(0, version?.globalStep
                ?? profile.trainingProgress?.globalStep ?? 0),
            displayedEpoch: max(0, version?.epoch
                ?? profile.trainingProgress?.epoch ?? 0),
            elapsed: max(0, version?.trainingDurationSeconds
                ?? profile.trainingProgress?.trainingDurationSeconds ?? 0),
            samplingRandomState: seed,
            rewardStatistics: ReinforcementRewardStatistics(),
            episodeReturns: summary.map { [$0.lastEpisodeReturn] } ?? [],
            cumulativeEpisodeReturn: summary.map {
                $0.averageEpisodeReturn * Double($0.episode)
            } ?? 0,
            lastPolicyLoss: 0,
            lastValueLoss: 0,
            lastEntropy: 0,
            lastApproximateKL: 0,
            lastClipFraction: 0,
            lastExplainedVariance: 0
        )
    }

    private static func checkpointSignature(
        profile: AIProfile,
        configuration: ReinforcementLearningConfiguration,
        contextSchema: ModelContextSchema,
        mouseMode: MouseControlMode
    ) throws -> String {
        struct Identity: Encodable {
            var modelSchema: Int
            var preprocessing: PreprocessingSpec
            var temporalVision: TemporalVisionConfiguration
            var architecture: ArchitectureSpec
            var precision: TrainingPrecision
            var channels: ActionChannels
            var restrictions: ActionRestrictions
            var contextSchema: ModelContextSchema
            var mouseMode: MouseControlMode
            var algorithm: ReinforcementLearningAlgorithm
            var actionSpace: ReinforcementActionSpace
            var rolloutSteps: Int
            var updateEpochs: Int
            var miniBatchSize: Int
            var learningRate: Double
            var weightDecay: Double
            var discountFactor: Double
            var lambda: Double
            var clipRange: Double
            var valueCoefficient: Double
            var entropyCoefficient: Double
            var maxGradientNorm: Double
            var targetKL: Double
            var normalizeAdvantages: Bool
            var clipValueLoss: Bool
            var bootstrapTimeLimits: Bool
            var criticHiddenWidth: Int
            var criticLayerCount: Int
            var actionBehavior: ReinforcementActionBehavior
            var rewardAggregation: ReinforcementRewardAggregation
            var rewardScale: Double
            var rewardClipMagnitude: Double
            var normalizeRewards: Bool
            var rewardNormalizationEpsilon: Double
            var seed: UInt64
        }
        let identity = Identity(
            modelSchema: ModelContract.schemaVersion,
            preprocessing: profile.preprocessing,
            temporalVision: profile.training.effectiveTemporalVision,
            architecture: profile.training.architecture.weightLayout,
            precision: profile.training.precision,
            channels: profile.channels,
            restrictions: profile.effectiveRestrictions,
            contextSchema: contextSchema,
            mouseMode: mouseMode,
            algorithm: configuration.algorithm,
            actionSpace: configuration.actionSpace,
            rolloutSteps: configuration.rolloutSteps,
            updateEpochs: configuration.updateEpochs,
            miniBatchSize: configuration.miniBatchSize,
            learningRate: configuration.learningRate,
            weightDecay: configuration.weightDecay,
            discountFactor: configuration.discountFactor,
            lambda: configuration.generalizedAdvantageLambda,
            clipRange: configuration.policyClipRange,
            valueCoefficient: configuration.valueLossCoefficient,
            entropyCoefficient: configuration.entropyCoefficient,
            maxGradientNorm: configuration.maximumGradientNorm,
            targetKL: configuration.targetKLDivergence,
            normalizeAdvantages: configuration.normalizeAdvantages,
            clipValueLoss: configuration.clipValueLoss,
            bootstrapTimeLimits: configuration.bootstrapTimeLimits,
            criticHiddenWidth: configuration.criticHiddenWidth,
            criticLayerCount: configuration.criticLayerCount,
            actionBehavior: configuration.effectiveActionBehavior,
            rewardAggregation: configuration.rewardAggregation,
            rewardScale: configuration.rewardScale,
            rewardClipMagnitude: configuration.rewardClipMagnitude,
            normalizeRewards: configuration.normalizeRewards,
            rewardNormalizationEpsilon: configuration.rewardNormalizationEpsilon,
            seed: configuration.seed
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return SHA256.hash(data: try encoder.encode(identity)).map {
            String(format: "%02x", $0)
        }.joined()
    }

    private func focusTargetIfNeeded(_ spec: CaptureSpec) async -> pid_t? {
        guard let id = spec.windowID,
              let content = try? await SCShareableContent.excludingDesktopWindows(
                false,
                onScreenWindowsOnly: false
              ),
              let pid = content.windows.first(where: { $0.windowID == id })?
                .owningApplication?.processID,
              let app = NSRunningApplication(processIdentifier: pid) else {
            return nil
        }
        app.activate(options: [.activateAllWindows])
        return pid
    }

    private func drain(queue: DispatchQueue) async {
        await withCheckedContinuation { continuation in
            queue.async { continuation.resume() }
        }
    }

    private static func hostTimeNanos(
        _ time: CMTime = CMClockGetTime(CMClockGetHostTimeClock())
    ) -> UInt64 {
        let scaled = CMTimeConvertScale(
            time,
            timescale: 1_000_000_000,
            method: .default
        )
        guard scaled.isNumeric, scaled.value > 0 else {
            return DispatchTime.now().uptimeNanoseconds
        }
        return UInt64(scaled.value)
    }

    private static func normalizedCursor(
        in rect: CGRect
    ) -> (x: Float, y: Float)? {
        guard rect.width.isFinite, rect.height.isFinite,
              rect.width > 0, rect.height > 0,
              let point = CGEvent(source: nil)?.location else { return nil }
        return (
            Float(min(1, max(0, (point.x - rect.minX) / rect.width))),
            Float(min(1, max(0, (point.y - rect.minY) / rect.height)))
        )
    }

    private func resetActionFilterToCurrentCursor() {
        actionFilter.reset(
            absoluteMousePosition: Self.normalizedCursor(in: captureRect)
        )
    }

    private static func average(
        cumulativeReturn: Double,
        episodeCount: Int
    ) -> Double {
        guard episodeCount > 0 else { return 0 }
        return cumulativeReturn / Double(episodeCount)
    }

    private static func protocolStopReason(_ reason: String) -> String {
        let value = reason.lowercased()
        if value.contains("rewarder") && value.contains("disconnect") {
            return "rewarder_disconnected"
        }
        if value.contains("reward") && value.contains("stale") {
            return "reward_timeout"
        }
        if value.contains("focus") { return "focus_lost" }
        if value.contains("human input") { return "human_input" }
        if value.contains("capture") { return "capture_stopped" }
        if value.contains("invalid reward") { return "invalid_reward_messages" }
        if value.contains("inference") { return "inference_stale" }
        if value.contains("episode limit") { return "episode_limit" }
        if value.contains("panic") { return "panic" }
        return "training_stopped"
    }

    deinit {
        actionTimer?.setEventHandler {}
        actionTimer?.cancel()
        safetyMonitor.stop()
        injector.disableAndReleaseAll()
    }
}

private final class ReinforcementPixelBuffer: @unchecked Sendable {
    let value: CVPixelBuffer
    init(_ value: CVPixelBuffer) { self.value = value }
}

private final class ReinforcementFinalizationBox: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Result<PreparedReinforcementSnapshot?, Error>?

    var value: Result<PreparedReinforcementSnapshot?, Error>? {
        lock.withLock { stored }
    }

    func set(_ value: Result<PreparedReinforcementSnapshot?, Error>) {
        lock.withLock { stored = value }
    }
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}
