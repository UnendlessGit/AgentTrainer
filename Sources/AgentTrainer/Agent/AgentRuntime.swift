@preconcurrency import CoreMedia
@preconcurrency import CoreVideo
import AppKit
import Foundation
import MLX
import MLXNN
@preconcurrency import ScreenCaptureKit

final class AgentRuntime: @unchecked Sendable {
    private typealias VisualizationFunction = @Sendable ([MLXArray]) -> [MLXArray]

    private enum StopAction {
        case perform(DispatchSourceTimer?)
        case waitForTeardown
        case waitForStartup
        case finished
    }

    var onState: (@Sendable (InputState) -> Void)?
    var onMetrics: (@Sendable (RuntimeMetrics) -> Void)?
    var onStop: (@Sendable (String?) -> Void)?
    var onPreview: (@Sendable (VisionPreviewFrame) -> Void)?
    var onVisualization: (@Sendable (CNNVisualizationFrame) -> Void)?
    var onReinforcementSignal: (@Sendable (ReinforcementSignal) -> Void)?
    var onReinforcementMetrics: (@Sendable (ReinforcementMetrics) -> Void)?
    var onReinforcementError: (@Sendable (String) -> Void)?
    var onReinforcementSnapshot: (@Sendable (ReinforcementSnapshot) async throws -> Void)?

    private let capture = CaptureService()
    private let preprocessor: VisionPreprocessor
    private let inferenceInputBuffers: MetalArrayBufferPool
    private let injector = InputInjector()
    private let safetyMonitor = InputCaptureService()
    private let inferenceQueue = DispatchQueue(label: "AgentTrainer.Inference", qos: .userInteractive)
    private let actionQueue = DispatchQueue(label: "AgentTrainer.Actions", qos: .userInteractive)
    private let lock = NSLock()
    private var model: AgentPolicy?
    /// Accessed only on `inferenceQueue` after startup. The runtime lock owns
    /// publication/removal of the reference and the pending-signal handoff.
    private var reinforcementTrainer: ReinforcementTrainer?
    private var pendingReinforcementSignals: [ReinforcementSignal] = []
    private var reinforcementScroll = ReinforcementScrollAccumulator()
    private var predictionFunction: VisualizationFunction?
    private var activationVisualizationFunctions: [VisualizationFunction] = []
    private var channelVisualizationFunction: (@Sendable ([MLXArray]) -> [MLXArray])?
    private var saliencyVisualizationFunction: (@Sendable ([MLXArray]) -> [MLXArray])?
    private var saliencyGradientFunction: (([MLXArray]) -> MLXArray)?
    private var profile: AIProfile?
    private var safety = AgentSafetyPolicy()
    private var captureRect = CGRect.zero
    private var mode: FrameMode = .newest
    private var mouseMode: MouseControlMode = .absolute
    private var gameCamera = GameCameraSettings()
    private var outputPermissions = RuntimeOutputPermissions()
    private var outputPermissionsRevision = 0
    private var allowedKeyCodes: Set<UInt16> = []
    private var shiftUsesKeyboardChannel = false
    private var latestFrame: CVPixelBuffer?
    private var lastUsableCaptureFrame: CVPixelBuffer?
    private var processing = false
    private var predictionLatch = RuntimePredictionLatch()
    private struct TemporalFrame: Sendable {
        /// Compact reduced-frame embedding computed when this perception was
        /// current. Reusing it avoids re-running the visual encoder over the
        /// entire causal history on every inference tick.
        var embedding: [Float]
        var controls: [Float]
    }
    /// Fixed-capacity circular storage keeps sampling and appending O(1), even
    /// at the largest supported frame count and spacing. Slot zero relative to
    /// the end is the most recently completed perception.
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
            guard distanceBack > 0, distanceBack <= count, !storage.isEmpty else { return nil }
            return storage[(oldest + count - distanceBack) % storage.count]
        }
    }
    private var temporalFrames = TemporalFrameRing()
    private var actionTimer: DispatchSourceTimer?
    private var nextPerceptionTime = 0.0
    private var metrics = RuntimeMetrics()
    private var startedAt = 0.0
    private var stopped = true
    private var targetPID: pid_t?
    private var previewFPS = 0.0
    private var lastPreviewTime = 0.0
    private var visualizationSettings = CNNVisualizationSettings()
    private var visualizationSettingsRevision = 0
    private var lastVisualizationTime = 0.0
    private var lastMetricsReportTime = 0.0
    private var lastFocusCheckTime = 0.0
    private var launchRevision: UInt64 = 0
    private var starting = false
    private var teardownInProgress = false
    private var startupWaiters: [CheckedContinuation<Void, Never>] = []
    private var teardownWaiters: [CheckedContinuation<Void, Never>] = []

    init() throws {
        MLXMemoryLifecycle.configure()
        preprocessor = try VisionPreprocessor()
        inferenceInputBuffers = try MetalArrayBufferPool(maximumCachedBytes: 32 * 1_024 * 1_024)
        injector.onState = { [weak self] state in self?.onState?(state) }
    }

    func start(
        profile: AIProfile,
        version: ModelVersionManifest?,
        allowedKeyCodes: Set<UInt16>,
        captureSpec: CaptureSpec,
        captureRect: CGRect,
        mode: FrameMode,
        mouseMode: MouseControlMode,
        gameCamera: GameCameraSettings = GameCameraSettings(),
        outputPermissions: RuntimeOutputPermissions = RuntimeOutputPermissions(),
        safety: AgentSafetyPolicy,
        previewFPS: Double = 0,
        visualizationSettings: CNNVisualizationSettings = CNNVisualizationSettings(),
        ignoredHotkeys: [HotkeyBinding] = [],
        reinforcementConfiguration: ReinforcementConfiguration? = nil,
        reinforcementSnapshotRoot: URL? = nil
    ) async throws {
        guard AXIsProcessTrusted() else {
            let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
            AXIsProcessTrustedWithOptions(options)
            throw AgentTrainerError.permission("Accessibility permission is required before AgentTrainer can press keys or control the mouse.")
        }
        if let version {
            guard version.schemaVersion == ModelContract.schemaVersion,
                  version.relativeMouseScale == GameCameraContract.deltaScale else {
                throw AgentTrainerError.model("This brain predates the current visual and Game Camera contracts. Retrain it from the original recordings.")
            }
        } else if reinforcementConfiguration?.enabled != true {
            throw AgentTrainerError.model("This AI has no runnable brain. Enable Reinforcement Learning to start it safely from a new neutral policy.")
        }
        let launchToken: UInt64? = lock.withLock {
            guard stopped, !starting, !teardownInProgress else { return nil }
            starting = true
            launchRevision &+= 1
            return launchRevision
        }
        guard let launchToken else { throw AgentTrainerError.model("This AI is already starting or running.") }
        defer {
            let waiters = lock.withLock { () -> [CheckedContinuation<Void, Never>] in
                starting = false
                let waiters = startupWaiters
                startupWaiters.removeAll(keepingCapacity: false)
                return waiters
            }
            waiters.forEach { $0.resume() }
        }

        // A runnable brain is immutable. Timing, temporal vision, enabled heads,
        // precision, and architecture must come from the saved version rather
        // than mutable editor fields that may have changed after training.
        var runtimeProfile = profile
        if let version {
            runtimeProfile.preprocessing = version.preprocessing
            runtimeProfile.channels = RuntimeActionSemantics.effectiveChannels(saved: version.channels, current: profile.channels)
            runtimeProfile.training = version.training
        }
        _ = try runtimeProfile.preprocessing.validated()
        _ = try runtimeProfile.training.effectiveTemporalVision.validated(
            current: runtimeProfile.preprocessing,
            cachedEmbeddingWidth: runtimeProfile.training.architecture.visualEmbedding
        )
        let startRevisions = lock.withLock { (outputPermissionsRevision, visualizationSettingsRevision) }
        let model = AgentPolicy(profile: runtimeProfile)
        let versionDirectory: URL?
        if let version {
            let directory = await WorkspaceStore.shared.versionDirectory(profileID: profile.id, versionID: version.id)
            try model.loadWeights(from: directory.appendingPathComponent(version.weightsFile))
            versionDirectory = directory
        } else {
            model.initializeForSafeExploration()
            versionDirectory = nil
        }
        model.train(false)
        let reinforcementTrainer: ReinforcementTrainer?
        if let reinforcementConfiguration, reinforcementConfiguration.enabled {
            guard let reinforcementSnapshotRoot else {
                throw AgentTrainerError.storage("The managed model folder is unavailable for online-learning snapshots.")
            }
            reinforcementTrainer = try ReinforcementTrainer(
                model: model,
                profile: runtimeProfile,
                configuration: reinforcementConfiguration,
                baseVersion: version,
                baseVersionDirectory: versionDirectory,
                snapshotRoot: reinforcementSnapshotRoot,
                demonstratedKeyCodes: allowedKeyCodes,
                trainingShowsCursor: version?.trainingShowsCursor ?? captureSpec.showsCursor,
                recommendedMouseMode: mouseMode
            )
        } else {
            reinforcementTrainer = nil
        }
        let reinforcementIsActive = reinforcementTrainer != nil
        let temporal = runtimeProfile.training.effectiveTemporalVision
        let hasTemporalMemory = temporal.pastFrameCount > 0
        let pastSpec = temporal.pastFrameSpec(from: runtimeProfile.preprocessing)
        let predictionFunction: VisualizationFunction = compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
            let currentImages = VisionPreprocessor.mlxTensor(
                inputs[0], spec: runtimeProfile.preprocessing, dtype: model.dtype
            )
            if hasTemporalMemory {
                let currentEmbedding = model.visualEmbedding(
                    visualFeatures: model.visualActivations(images: currentImages).last!
                )
                let reducedEmbedding = model.pastVisualEmbedding(
                    visualFeatures: model.pastVisualActivations(
                        images: VisionPreprocessor.mlxTensor(inputs[1], spec: pastSpec, dtype: model.dtype)
                    ).last!
                )
                let temporalFeatures = model.temporalFeatures(
                    currentVisualEmbedding: currentEmbedding,
                    pastVisualEmbeddings: inputs[2],
                    pastControls: inputs[3]
                )
                let logits = model.logits(temporalFeatures: temporalFeatures)
                return [
                    reinforcementIsActive ? logits : model.activatedPredictions(logits: logits),
                    reducedEmbedding.asType(.float32)
                ]
            }
            let logits = model.logits(
                temporalFeatures: model.currentOnlyTemporalFeatures(currentImages: currentImages)
            )
            return [reinforcementIsActive ? logits : model.activatedPredictions(logits: logits)]
        }
        // Diagnostic graphs are lazy: creating these closures does not execute
        // or materialize an extra tensor. The selected graph first compiles only
        // when its view is enabled and reaches its independently capped rate.
        let activationVisualizationFunctions: [VisualizationFunction] = (0..<max(1, model.convolutions.count)).map { selectedLayer in
            compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
                let currentImages = VisionPreprocessor.mlxTensor(
                    inputs[0], spec: runtimeProfile.preprocessing, dtype: model.dtype
                )
                let layers = model.visualActivations(images: currentImages)
                let reducedEmbedding = hasTemporalMemory ? model.pastVisualEmbedding(
                    visualFeatures: model.pastVisualActivations(
                        images: VisionPreprocessor.mlxTensor(inputs[1], spec: pastSpec, dtype: model.dtype)
                    ).last!
                ) : nil
                let temporalFeatures = hasTemporalMemory
                    ? model.temporalFeatures(
                        currentVisualEmbedding: model.visualEmbedding(visualFeatures: layers.last!),
                        pastVisualEmbeddings: inputs[2],
                        pastControls: inputs[3]
                    )
                    : model.currentOnlyTemporalFeatures(currentVisualFeatures: layers.last!)
                let logits = model.logits(temporalFeatures: temporalFeatures)
                let map = model.sampledForVisualization(layers[selectedLayer]).mean(axis: -1, keepDims: true)
                return [reinforcementIsActive ? logits : model.activatedPredictions(logits: logits)]
                    + (reducedEmbedding.map { [$0.asType(.float32)] } ?? [])
                    + [map]
            }
        }
        let channelVisualizationFunction = compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
            let currentImages = VisionPreprocessor.mlxTensor(
                inputs[0], spec: runtimeProfile.preprocessing, dtype: model.dtype
            )
            let layers = model.visualActivations(images: currentImages)
            let reducedEmbedding = hasTemporalMemory ? model.pastVisualEmbedding(
                visualFeatures: model.pastVisualActivations(
                    images: VisionPreprocessor.mlxTensor(inputs[1], spec: pastSpec, dtype: model.dtype)
                ).last!
            ) : nil
            let temporalFeatures = hasTemporalMemory
                ? model.temporalFeatures(
                    currentVisualEmbedding: model.visualEmbedding(visualFeatures: layers.last!),
                    pastVisualEmbeddings: inputs[2],
                    pastControls: inputs[3]
                )
                : model.currentOnlyTemporalFeatures(currentVisualFeatures: layers.last!)
            let logits = model.logits(temporalFeatures: temporalFeatures)
            return [reinforcementIsActive ? logits : model.activatedPredictions(logits: logits)]
                + (reducedEmbedding.map { [$0.asType(.float32)] } ?? [])
                + [model.strongestChannelsForVisualization(layers.last!)]
        }
        let saliencyVisualizationFunction = compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
            let currentImages = VisionPreprocessor.mlxTensor(
                inputs[0], spec: runtimeProfile.preprocessing, dtype: model.dtype
            )
            let layers = model.visualActivations(images: currentImages)
            let reducedEmbedding = hasTemporalMemory ? model.pastVisualEmbedding(
                visualFeatures: model.pastVisualActivations(
                    images: VisionPreprocessor.mlxTensor(inputs[1], spec: pastSpec, dtype: model.dtype)
                ).last!
            ) : nil
            let temporalFeatures = hasTemporalMemory
                ? model.temporalFeatures(
                    currentVisualEmbedding: model.visualEmbedding(visualFeatures: layers.last!),
                    pastVisualEmbeddings: inputs[2],
                    pastControls: inputs[3]
                )
                : model.currentOnlyTemporalFeatures(currentVisualFeatures: layers.last!)
            let logits = model.logits(temporalFeatures: temporalFeatures)
            // Keep the exact final tensor on GPU for the post-CNN gradient.
            // This graph intentionally omits the channel ranking used by the
            // separate feature-grid view.
            return [reinforcementIsActive ? logits : model.activatedPredictions(logits: logits)]
                + (reducedEmbedding.map { [$0.asType(.float32)] } ?? [])
                + [layers.last!]
        }
        let saliencyGradient = grad({ (inputs: [MLXArray]) -> MLXArray in
            if hasTemporalMemory {
                let logits = model.logits(
                    temporalFeatures: model.temporalFeatures(
                        currentVisualEmbedding: model.visualEmbedding(visualFeatures: inputs[0]),
                        pastVisualEmbeddings: inputs[1],
                        pastControls: inputs[2]
                    )
                )
                return (logits * inputs[3].asType(model.dtype)).sum()
            }
            let logits = model.logits(
                temporalFeatures: model.currentOnlyTemporalFeatures(currentVisualFeatures: inputs[0])
            )
            return (logits * inputs[1].asType(model.dtype)).sum()
        }, argumentNumbers: [0])
        let accepted = lock.withLock { () -> Bool in
            guard launchRevision == launchToken, starting, stopped else { return false }
            self.model = model
            self.reinforcementTrainer = reinforcementTrainer
            self.pendingReinforcementSignals.removeAll(keepingCapacity: false)
            self.reinforcementScroll = ReinforcementScrollAccumulator()
            self.predictionFunction = predictionFunction
            self.activationVisualizationFunctions = activationVisualizationFunctions
            self.channelVisualizationFunction = channelVisualizationFunction
            self.saliencyVisualizationFunction = saliencyVisualizationFunction
            self.saliencyGradientFunction = saliencyGradient
            self.profile = runtimeProfile
            self.allowedKeyCodes = allowedKeyCodes
            self.shiftUsesKeyboardChannel = (version?.trainingDataSchema ?? TrainingDataContract.schemaVersion) >= 7
            self.safety = safety
            self.captureRect = captureRect
            self.mode = mode
            self.mouseMode = mouseMode
            self.gameCamera = gameCamera
            if outputPermissionsRevision == startRevisions.0 { self.outputPermissions = outputPermissions }
            if visualizationSettingsRevision == startRevisions.1 {
                self.visualizationSettings = visualizationSettings.sanitized(layerCount: runtimeProfile.training.architecture.convolutionChannels.count)
            } else {
                self.visualizationSettings = self.visualizationSettings.sanitized(layerCount: runtimeProfile.training.architecture.convolutionChannels.count)
            }
            self.previewFPS = max(0, previewFPS)
            self.lastPreviewTime = 0
            self.lastVisualizationTime = 0
            latestFrame = nil; lastUsableCaptureFrame = nil; predictionLatch.reset(); temporalFrames.reset()
            metrics = RuntimeMetrics(); startedAt = CACurrentMediaTime(); nextPerceptionTime = 0; lastMetricsReportTime = 0; lastFocusCheckTime = 0; stopped = false
            // Keep session enablement ordered with live permission changes. A
            // toggle made while model weights are loading must not be replaced
            // by the older settings captured when `start` was first called.
            injector.enable(outputPermissions: self.outputPermissions)
            stopped = false
            starting = false
            return true
        }
        guard accepted else { throw CancellationError() }
        if let reinforcementTrainer {
            onReinforcementMetrics?(reinforcementTrainer.metrics)
        }
        do {
            guard lock.withLock({ !stopped }) else { throw CancellationError() }
            safetyMonitor.ignoredHotkeys = ignoredHotkeys
            safetyMonitor.onSample = { [weak self] sample in
                guard let self else { return }
                if self.processReinforcementInput(sample) { return }
                let policy = self.lock.withLock { self.safety }
                let panic = sample.kind == .key && sample.isDown && sample.keyCode == policy.panicKeyCode && (sample.modifiers & policy.panicModifiers) == policy.panicModifiers
                guard panic || policy.stopOnHumanInput else { return }
                Task { await self.stop(reason: panic ? "Panic stop" : "Stopped on human input") }
            }
            try safetyMonitor.start()
            let targetPID = await focusTargetIfNeeded(captureSpec)
            guard lock.withLock({ !stopped }) else { throw CancellationError() }
            lock.withLock { self.targetPID = targetPID }
            var liveCaptureSpec = captureSpec
            liveCaptureSpec.requestedFPS = runtimeProfile.training.perceptionFPS
            liveCaptureSpec.showsCursor = version?.trainingShowsCursor ?? captureSpec.showsCursor
            try await capture.start(spec: liveCaptureSpec, queueDepth: mode == .newest ? 3 : 8, onFrame: { [weak self] buffer, pts in
                self?.receive(buffer, timestamp: pts)
            }, onIdle: { [weak self] pts in
                self?.receiveIdle(timestamp: pts)
            }, onUnexpectedStop: { [weak self] error in
                Task { await self?.stop(reason: "Capture stopped: \(error.localizedDescription)") }
            })
            guard lock.withLock({ !stopped }) else { _ = try? await capture.stop(); throw CancellationError() }
            startActionTimer(fps: runtimeProfile.training.actionFPS)
            guard lock.withLock({ !stopped }) else { throw CancellationError() }
        } catch {
            await stop(reason: nil)
            // A concurrent stop can finish between a cancellation check and a
            // subsequently-started monitor/stream. Clean those late resources
            // again after joining teardown so a cancelled launch can never
            // leave an input tap or capture stream behind.
            safetyMonitor.stop()
            safetyMonitor.onSample = nil
            _ = try? await capture.stop()
            injector.disableAndReleaseAll()
            throw error
        }
    }

    /// Applies run-only output changes immediately. InputInjector serializes
    /// this with action execution and releases held keyboard state when needed.
    func updateOutputPermissions(_ permissions: RuntimeOutputPermissions) {
        lock.withLock {
            outputPermissionsRevision &+= 1
            outputPermissions = permissions
            injector.updateOutputPermissions(permissions)
        }
    }

    /// View controls are presentation-only and may change during a run. The
    /// runtime lock orders them with frame scheduling; disabling the view makes
    /// the next and every later perception use the standard prediction graph.
    func updateVisualizationSettings(_ settings: CNNVisualizationSettings) {
        lock.withLock {
            visualizationSettingsRevision &+= 1
            let layerCount = profile?.training.architecture.convolutionChannels.count
            visualizationSettings = settings.sanitized(layerCount: layerCount)
            lastVisualizationTime = 0
        }
    }

    /// Every provider—manual today and screen-aware automation later—enters
    /// through this timestamped boundary. Signals are bounded immediately and
    /// optimized on the inference queue, never concurrently with prediction.
    func submitReinforcementSignal(_ rawSignal: ReinforcementSignal) {
        guard rawSignal.timestamp.isFinite,
              rawSignal.value.isFinite,
              rawSignal.value != 0 else { return }
        var signal = rawSignal
        signal.value = min(
            ReinforcementLearningContract.maximumSignalMagnitude,
            max(-ReinforcementLearningContract.maximumSignalMagnitude, signal.value)
        )
        let accepted = lock.withLock { () -> Bool in
            guard !stopped, reinforcementTrainer != nil else { return false }
            if pendingReinforcementSignals.count >= 256,
               let last = pendingReinforcementSignals.indices.last {
                let combined = pendingReinforcementSignals[last].value + signal.value
                pendingReinforcementSignals[last].value = min(
                    ReinforcementLearningContract.maximumSignalMagnitude,
                    max(
                        -ReinforcementLearningContract.maximumSignalMagnitude,
                        combined
                    )
                )
                pendingReinforcementSignals[last].timestamp = max(
                    pendingReinforcementSignals[last].timestamp,
                    signal.timestamp
                )
                pendingReinforcementSignals[last].sourceIdentifier = "feedback.coalesced"
                pendingReinforcementSignals[last].detail = "High-rate feedback coalesced at the bounded input queue"
                if pendingReinforcementSignals[last].value == 0 {
                    pendingReinforcementSignals.removeLast()
                }
            } else {
                pendingReinforcementSignals.append(signal)
            }
            return true
        }
        guard accepted else { return }
        onReinforcementSignal?(signal)
        inferenceQueue.async { [weak self] in
            self?.applyPendingReinforcementSignals(publishAutosaves: true)
        }
    }

    func submitReinforcement(
        value: Double,
        sourceIdentifier: String,
        detail: String? = nil,
        timestamp: Double = CACurrentMediaTime()
    ) {
        submitReinforcementSignal(ReinforcementSignal(
            timestamp: timestamp,
            value: value,
            sourceIdentifier: sourceIdentifier,
            detail: detail
        ))
    }

    private func processReinforcementInput(_ sample: InputSample) -> Bool {
        let result = lock.withLock { () -> (Bool, Double?) in
            guard let reinforcementTrainer else { return (false, nil) }
            let result = reinforcementScroll.process(
                sample,
                configuration: reinforcementTrainer.configuration
            )
            return (result.handled, result.signal)
        }
        if let value = result.1 {
            submitReinforcement(
                value: value,
                sourceIdentifier: "manual.scroll",
                detail: "Modifier + feedback wheel"
            )
        }
        return result.0
    }

    /// Must run on `inferenceQueue`. The queue is also the sole owner of model
    /// inference and transition history, so an AdamW update can never race a
    /// prediction or publish half-updated parameters.
    private func applyPendingReinforcementSignals(publishAutosaves: Bool) {
        let pending = lock.withLock { () -> ([ReinforcementSignal], ReinforcementTrainer?) in
            let pending = pendingReinforcementSignals
            pendingReinforcementSignals.removeAll(keepingCapacity: true)
            return (pending, reinforcementTrainer)
        }
        guard let trainer = pending.1, !pending.0.isEmpty else { return }
        do {
            var shouldAutosave = false
            for signal in pending.0 {
                let result = try trainer.apply(signal)
                shouldAutosave = shouldAutosave || result.shouldAutosave
                onReinforcementMetrics?(result.metrics)
            }
            if publishAutosaves, shouldAutosave,
               let snapshot = try trainer.makeSnapshot(isAutosave: true) {
                onReinforcementMetrics?(trainer.metrics)
                publishReinforcementSnapshotInBackground(snapshot)
            }
        } catch {
            let message = error.localizedDescription
            onReinforcementError?(message)
            Task { await stop(reason: "Online learning stopped safely: \(message)") }
        }
    }

    private func publishReinforcementSnapshotInBackground(_ snapshot: ReinforcementSnapshot) {
        guard let publisher = onReinforcementSnapshot else {
            try? FileManager.default.removeItem(at: snapshot.stagingDirectory)
            return
        }
        Task { [onReinforcementError] in
            do {
                try await publisher(snapshot)
            } catch {
                try? FileManager.default.removeItem(at: snapshot.stagingDirectory)
                onReinforcementError?(error.localizedDescription)
            }
        }
    }

    func stop(reason: String? = nil) async {
        let action = lock.withLock { () -> StopAction in
            launchRevision &+= 1
            if teardownInProgress { return .waitForTeardown }
            guard !stopped else { return starting ? .waitForStartup : .finished }
            stopped = true
            teardownInProgress = true
            let timer = actionTimer; actionTimer = nil
            latestFrame = nil; lastUsableCaptureFrame = nil; temporalFrames.reset(); predictionLatch.reset(); targetPID = nil
            return .perform(timer)
        }
        let timer: DispatchSourceTimer?
        switch action {
        case .finished:
            return
        case .waitForStartup:
            await waitForStartupCompletion()
            return
        case .waitForTeardown:
            await waitForTeardownCompletion()
            return
        case .perform(let value):
            timer = value
        }
        timer?.setEventHandler {}
        timer?.cancel()
        // Once no action block can still be executing, release physical input
        // immediately. ScreenCaptureKit or an in-flight MLX eval may take
        // seconds to drain, but they are no longer allowed to hold a key/button
        // or leave relative mouse state associated during that wait.
        await drain(queue: actionQueue)
        injector.disableAndReleaseAll()
        safetyMonitor.stop()
        safetyMonitor.onSample = nil
        _ = try? await capture.stop()
        await drain(queue: inferenceQueue)
        let finalReinforcementSnapshot: ReinforcementSnapshot? = await withCheckedContinuation { continuation in
            inferenceQueue.async { [weak self] in
                guard let self else { continuation.resume(returning: nil); return }
                self.applyPendingReinforcementSignals(publishAutosaves: false)
                do {
                    continuation.resume(returning: try self.reinforcementTrainer?.makeSnapshot(isAutosave: false))
                } catch {
                    self.onReinforcementError?(error.localizedDescription)
                    continuation.resume(returning: nil)
                }
            }
        }
        if let finalReinforcementSnapshot {
            if let publisher = onReinforcementSnapshot {
                do { try await publisher(finalReinforcementSnapshot) }
                catch {
                    try? FileManager.default.removeItem(at: finalReinforcementSnapshot.stagingDirectory)
                    onReinforcementError?(error.localizedDescription)
                }
            } else {
                try? FileManager.default.removeItem(at: finalReinforcementSnapshot.stagingDirectory)
            }
        }
        lock.withLock {
            predictionFunction = nil; activationVisualizationFunctions.removeAll(keepingCapacity: false); channelVisualizationFunction = nil; saliencyVisualizationFunction = nil; saliencyGradientFunction = nil; reinforcementTrainer = nil; pendingReinforcementSignals.removeAll(keepingCapacity: false); reinforcementScroll = ReinforcementScrollAccumulator(); model = nil; profile = nil; allowedKeyCodes.removeAll(keepingCapacity: false); shiftUsesKeyboardChannel = false
            latestFrame = nil; lastUsableCaptureFrame = nil; temporalFrames.reset(); predictionLatch.reset(); processing = false
            visualizationSettings = CNNVisualizationSettings(); lastVisualizationTime = 0
        }
        onReinforcementMetrics?(ReinforcementMetrics())
        MLXMemoryLifecycle.reclaimCaches(after: "agent runtime")
        onStop?(reason)
        let waiters = lock.withLock { () -> [CheckedContinuation<Void, Never>] in
            teardownInProgress = false
            let waiters = teardownWaiters
            teardownWaiters.removeAll(keepingCapacity: false)
            return waiters
        }
        waiters.forEach { $0.resume() }
    }

    private func receive(_ buffer: CVPixelBuffer, timestamp: CMTime) {
        schedule(buffer, countThrottledAsDropped: true)
    }

    private func receiveIdle(timestamp: CMTime) {
        let frame = lock.withLock { stopped ? nil : lastUsableCaptureFrame }
        if let frame { schedule(frame, countThrottledAsDropped: false) }
    }

    private func schedule(_ buffer: CVPixelBuffer, countThrottledAsDropped: Bool) {
        let now = CACurrentMediaTime()
        lock.lock()
        guard !stopped, let profile else { lock.unlock(); return }
        if countThrottledAsDropped { lastUsableCaptureFrame = buffer }
        let interval = 1 / max(0.0001, profile.training.perceptionFPS)
        guard now >= nextPerceptionTime else {
            if countThrottledAsDropped { metrics.droppedFrames += 1 }
            lock.unlock()
            return
        }
        nextPerceptionTime = now + interval
        if mode == .ordered {
            let frame = SendablePixelBuffer(buffer)
            lock.unlock()
            // Backpressure the capture callback instead of building an unbounded
            // queue during long ordered-frame runs.
            inferenceQueue.sync { [weak self] in self?.infer(frame.value) }
            return
        }
        latestFrame = buffer
        if processing { lock.unlock(); return }
        processing = true
        lock.unlock()
        inferenceQueue.async { [weak self] in self?.drainNewest() }
    }

    private func drainNewest() {
        while true {
            lock.lock()
            guard let frame = latestFrame, !stopped else { processing = false; lock.unlock(); return }
            latestFrame = nil
            lock.unlock()
            infer(frame)
            lock.lock()
            if latestFrame == nil { processing = false; lock.unlock(); return }
            lock.unlock()
        }
    }

    private func infer(_ buffer: CVPixelBuffer) {
        lock.lock()
        guard !stopped, let predictionFunction, let model, let profile else { lock.unlock(); return }
        let reinforcementTrainer = self.reinforcementTrainer
        let samplingOutputPermissions = outputPermissions
        let temporal = profile.training.effectiveTemporalVision
        let selectedPriorFrames: [TemporalFrame?] = (0..<temporal.pastFrameCount).map { frame in
            let distance = temporal.frameSpacing * (temporal.pastFrameCount - frame)
            return temporalFrames.frame(distanceBack: distance)
        }
        let now = CACurrentMediaTime()
        let settings = visualizationSettings
        let visualizationDue = settings.enabled && now - lastVisualizationTime >= 1 / settings.framesPerSecond
        if visualizationDue { lastVisualizationTime = now }
        let activationVisualizationFunctions = self.activationVisualizationFunctions
        let channelVisualizationFunction = self.channelVisualizationFunction
        let saliencyVisualizationFunction = self.saliencyVisualizationFunction
        let saliencyGradientFunction = self.saliencyGradientFunction
        let mouseMode = self.mouseMode
        lock.unlock()
        let began = CACurrentMediaTime()
        do {
            let pastSpec = temporal.pastFrameSpec(from: profile.preprocessing)
            let hasTemporalMemory = temporal.pastFrameCount > 0
            let sharesPackedRepresentation = hasTemporalMemory && pastSpec == profile.preprocessing
            let requestedSpecs = hasTemporalMemory && !sharesPackedRepresentation
                ? [profile.preprocessing, pastSpec]
                : [profile.preprocessing]
            // One source mapping and command buffer produces every resolution
            // needed by this perception.
            let jobs = try preprocessor.submit(buffer, specs: requestedSpecs)
            guard let currentJob = jobs.first else {
                throw AgentTrainerError.model("The live vision pipeline returned no current frame.")
            }
            let packed = try currentJob.withPackedBytes { Data($0) }
            let packedPastFrame: Data
            if !hasTemporalMemory {
                packedPastFrame = Data()
            } else if sharesPackedRepresentation {
                packedPastFrame = packed
            } else if jobs.indices.contains(1) {
                packedPastFrame = try jobs[1].withPackedBytes { Data($0) }
            } else {
                throw AgentTrainerError.model("The live vision pipeline returned no temporal frame.")
            }
            let zeroControls = [Float](repeating: 0, count: ActionLayout.count)
            let zeroEmbedding = [Float](
                repeating: 0,
                count: profile.training.architecture.visualEmbedding
            )
            let selectedFrames: [TemporalFrame] = selectedPriorFrames.map { frame in
                frame ?? TemporalFrame(embedding: zeroEmbedding, controls: zeroControls)
            }
            var descriptors: [MetalArrayBufferPool.Descriptor] = [
                .init([1, profile.preprocessing.sampleByteCount], dtype: .uint8)
            ]
            if hasTemporalMemory {
                descriptors.append(.init([1, pastSpec.sampleByteCount], dtype: .uint8))
                descriptors.append(.init([
                    1,
                    temporal.pastFrameCount,
                    profile.training.architecture.visualEmbedding
                ], dtype: .float32))
                descriptors.append(.init([1, temporal.pastFrameCount, ActionLayout.count], dtype: .float32))
            }
            let inferenceInputs = try inferenceInputBuffers.makeArrays(descriptors) { destinations in
                packed.withUnsafeBytes { source in
                    destinations[0].copyMemory(from: source)
                }
                guard hasTemporalMemory else { return }
                packedPastFrame.withUnsafeBytes { source in
                    destinations[1].copyMemory(from: source)
                }
                let embeddingRowBytes = profile.training.architecture.visualEmbedding
                    * MemoryLayout<Float>.size
                let controlRowBytes = ActionLayout.count * MemoryLayout<Float>.size
                for (frame, selected) in selectedFrames.enumerated() {
                    selected.embedding.withUnsafeBytes { source in
                        UnsafeMutableRawBufferPointer(
                            rebasing: destinations[2][frame * embeddingRowBytes..<(frame + 1) * embeddingRowBytes]
                        ).copyMemory(from: source)
                    }
                    selected.controls.withUnsafeBytes { source in
                        UnsafeMutableRawBufferPointer(
                            rebasing: destinations[3][frame * controlRowBytes..<(frame + 1) * controlRowBytes]
                        ).copyMemory(from: source)
                    }
                }
            }
            let result: [MLXArray] = Device.withDefaultDevice(.gpu) {
                guard visualizationDue else { return predictionFunction(inferenceInputs) }
                switch settings.mode {
                case .activationOverlay:
                    let selectedLayer = max(0, settings.convolutionLayer)
                    guard activationVisualizationFunctions.indices.contains(selectedLayer) else { return predictionFunction(inferenceInputs) }
                    return activationVisualizationFunctions[selectedLayer](inferenceInputs)
                case .featureChannels:
                    guard let forward = channelVisualizationFunction?(inferenceInputs), forward.count >= 2 else { return predictionFunction(inferenceInputs) }
                    return forward
                case .actionSaliency:
                    let selector = Self.actionSelector(focus: settings.actionFocus, mouseMode: mouseMode)
                    guard let saliencyVisualizationFunction, let saliencyGradientFunction else { return predictionFunction(inferenceInputs) }
                    let forward = saliencyVisualizationFunction(inferenceInputs)
                    let featureIndex = hasTemporalMemory ? 2 : 1
                    guard forward.indices.contains(featureIndex) else { return predictionFunction(inferenceInputs) }
                    let gradientInputs = hasTemporalMemory
                        ? [forward[featureIndex], inferenceInputs[2], inferenceInputs[3], selector]
                        : [forward[featureIndex], selector]
                    let gradients = saliencyGradientFunction(gradientInputs)
                    let weights = gradients.mean(axes: [1, 2], keepDims: true)
                    let saliency = relu((forward[featureIndex] * weights).sum(axis: -1, keepDims: true))
                    return hasTemporalMemory
                        ? [forward[0], forward[1], model.sampledForVisualization(saliency)]
                        : [forward[0], model.sampledForVisualization(saliency)]
                }
            }
            guard let output = result.first else { throw AgentTrainerError.model("The inference graph returned no prediction.") }
            let visualization = visualizationDue ? Self.visualizationFrame(
                profile: profile,
                settings: settings,
                outputs: result,
                hasTemporalMemory: hasTemporalMemory
            ) : nil
            MLX.eval(result)
            let policyValues = output.asArray(Float.self)
            guard policyValues.count >= ActionLayout.count, policyValues.prefix(ActionLayout.count).allSatisfy(\.isFinite) else {
                throw AgentTrainerError.model("The brain produced an invalid prediction, so all outputs were stopped safely.")
            }
            let values: [Float]
            if let reinforcementTrainer {
                let allowedMask = ReinforcementActionPolicy.allowedMask(
                    profile: profile,
                    allowedKeyCodes: allowedKeyCodes,
                    mouseMode: mouseMode,
                    outputPermissions: samplingOutputPermissions
                )
                values = reinforcementTrainer.sample(
                    logits: Array(policyValues.prefix(ActionLayout.count)),
                    allowedMask: allowedMask
                )
            } else {
                values = policyValues
            }
            if let visualization {
                let frame = CNNVisualizationFrame(
                    packed: packed,
                    spec: profile.preprocessing,
                    settings: visualization.settings,
                    tensors: visualization.arrays.enumerated().map { offset, array in
                        let layer = visualization.layers[offset]
                        let geometry = CNNGeometry.layer(layer, architecture: profile.training.architecture)
                        return CNNFeatureTensor(width: array.dim(2), height: array.dim(1), channels: array.dim(3), values: array.asArray(Float.self), convolutionLayer: layer, kernelSize: geometry.kernelSize, effectiveStride: geometry.effectiveStride, receptiveField: geometry.receptiveField)
                    },
                    timestamp: CACurrentMediaTime()
                )
                onVisualization?(frame)
            }
            let preview: VisionPreviewFrame? = lock.withLock {
                guard !stopped else { return nil }
                let now = CACurrentMediaTime()
                guard previewFPS > 0, now - lastPreviewTime >= 1 / max(0.1, previewFPS) else { return nil }
                lastPreviewTime = now
                return VisionPreviewFrame(packed: packed, spec: profile.preprocessing, timestamp: now)
            }
            if let preview { onPreview?(preview) }
            let reducedEmbedding: [Float]
            if hasTemporalMemory {
                guard result.indices.contains(1) else {
                    throw AgentTrainerError.model("The temporal inference graph returned no reusable visual embedding.")
                }
                reducedEmbedding = result[1].asArray(Float.self)
                guard reducedEmbedding.count == profile.training.architecture.visualEmbedding,
                      reducedEmbedding.allSatisfy(\.isFinite) else {
                    throw AgentTrainerError.model("The temporal visual cache produced invalid values.")
                }
            } else {
                reducedEmbedding = []
            }
            lock.lock()
            guard !stopped else { lock.unlock(); return }
            let transitionOutputPermissions = outputPermissions
            let frameControls = RuntimeActionSemantics.temporalControlValues(
                values,
                channels: profile.channels,
                restrictions: profile.effectiveRestrictions,
                allowedKeyCodes: allowedKeyCodes,
                outputPermissions: transitionOutputPermissions,
                shiftUsesKeyboardChannel: shiftUsesKeyboardChannel
            )
            if hasTemporalMemory {
                let retainedFrameCount = temporal.pastFrameCount * temporal.frameSpacing
                temporalFrames.append(
                    TemporalFrame(embedding: reducedEmbedding, controls: frameControls),
                    capacity: retainedFrameCount
                )
            }
            predictionLatch.publish(values)
            metrics.frameCount += 1
            let elapsed = max(0.001, CACurrentMediaTime() - startedAt)
            metrics.perceptionFPS = Double(metrics.frameCount) / elapsed
            metrics.latencyMilliseconds = (CACurrentMediaTime() - began) * 1_000
            let now = CACurrentMediaTime()
            let snapshot: RuntimeMetrics? = now - lastMetricsReportTime >= 0.1 ? metrics : nil
            if snapshot != nil { lastMetricsReportTime = now }
            lock.unlock()
            if let reinforcementTrainer {
                reinforcementTrainer.record(
                    timestamp: now,
                    currentPacked: packed,
                    pastEmbeddings: selectedFrames.flatMap(\.embedding),
                    pastControls: selectedFrames.flatMap(\.controls),
                    behaviorLogits: Array(policyValues.prefix(ActionLayout.count)),
                    action: frameControls,
                    allowedMask: ReinforcementActionPolicy.allowedMask(
                        profile: profile,
                        allowedKeyCodes: allowedKeyCodes,
                        mouseMode: mouseMode,
                        outputPermissions: transitionOutputPermissions
                    )
                )
                applyPendingReinforcementSignals(publishAutosaves: true)
            }
            if let snapshot { onMetrics?(snapshot) }
        } catch {
            Task { await stop(reason: error.localizedDescription) }
        }
    }

    private struct VisualizationArrays {
        var settings: CNNVisualizationSettings
        var arrays: [MLXArray]
        var layers: [Int]
    }

    private static func visualizationFrame(
        profile: AIProfile,
        settings rawSettings: CNNVisualizationSettings,
        outputs: [MLXArray],
        hasTemporalMemory: Bool = false
    ) -> VisualizationArrays? {
        let modelLayerCount = max(1, profile.training.architecture.convolutionChannels.count)
        let settings = rawSettings.sanitized(layerCount: profile.training.architecture.convolutionChannels.count)
        let hasConvolutions = !profile.training.architecture.convolutionChannels.isEmpty
        let finalLayer = hasConvolutions ? modelLayerCount - 1 : -1
        let visualizationIndex = hasTemporalMemory ? 2 : 1
        let selected: [(MLXArray, Int)]
        switch settings.mode {
        case .activationOverlay:
            guard outputs.indices.contains(visualizationIndex) else { return nil }
            selected = [(outputs[visualizationIndex], hasConvolutions ? settings.convolutionLayer : -1)]
        case .featureChannels, .actionSaliency:
            guard outputs.indices.contains(visualizationIndex) else { return nil }
            selected = [(outputs[visualizationIndex], finalLayer)]
        }
        let valid = selected.filter { $0.0.ndim == 4 && $0.0.dim(0) == 1 && $0.0.dim(1) > 0 && $0.0.dim(2) > 0 && $0.0.dim(3) > 0 }
        guard !valid.isEmpty else { return nil }
        return VisualizationArrays(settings: settings, arrays: valid.map { $0.0.asType(.float32) }, layers: valid.map(\.1))
    }

    private static func actionSelector(focus: CNNActionFocus, mouseMode: MouseControlMode) -> MLXArray {
        let indices: [Int] = switch focus {
        case .movement: Array(mouseMode == .relative ? ActionLayout.relativeMouse : ActionLayout.absoluteMouse)
        case .mouseButtons: Array(ActionLayout.buttons)
        case .scroll: Array(ActionLayout.scroll)
        case .keyboard: ActionLayout.keyboardAndShiftIndices
        case .modifiers: Array(ActionLayout.commandOptionControl)
        }
        var values = [Float](repeating: 0, count: ActionLayout.count)
        let weight = 1 / Float(max(1, indices.count))
        for index in indices { values[index] = weight }
        return MLXArray(values, [1, ActionLayout.count])
    }

    private func startActionTimer(fps: Double) {
        let timer = DispatchSource.makeTimerSource(queue: actionQueue)
        let interval = 1 / max(0.0001, fps)
        timer.schedule(deadline: .now(), repeating: interval, leeway: .microseconds(250))
        timer.setEventHandler { [weak self] in self?.actionTick() }
        let accepted = lock.withLock { () -> Bool in
            guard !stopped else { return false }
            actionTimer = timer; return true
        }
        timer.resume()
        if !accepted { timer.setEventHandler {}; timer.cancel() }
    }

    private func actionTick() {
        lock.lock()
        guard !stopped, let latched = predictionLatch.consume(), let profile else { lock.unlock(); return }
        let prediction = latched.values
        let safety = self.safety, rect = captureRect, targetPID = self.targetPID, mouseMode = self.mouseMode, gameCamera = self.gameCamera, allowedKeyCodes = self.allowedKeyCodes, shiftUsesKeyboardChannel = self.shiftUsesKeyboardChannel
        let now = CACurrentMediaTime()
        let maximumPredictionAge = max(0.35, 3 / max(0.0001, profile.training.perceptionFPS))
        if now - latched.publishedAt > maximumPredictionAge {
            lock.unlock()
            Task { await stop(reason: "Stopped because live inference stopped producing fresh predictions") }
            return
        }
        // Frontmost-app lookup crosses into AppKit/WindowServer. Ten checks per
        // second preserves a near-immediate safety stop without doing that work
        // at a 60–240 Hz action rate.
        if safety.stopOnFocusLoss, now - startedAt > 0.75, let targetPID, now - lastFocusCheckTime >= 0.1 {
            lastFocusCheckTime = now
            if NSWorkspace.shared.frontmostApplication?.processIdentifier != targetPID {
                lock.unlock()
                Task { await stop(reason: "Stopped because the target window lost focus") }
                return
            }
        }
        metrics.actionCount += 1
        let elapsed = max(0.001, CACurrentMediaTime() - startedAt)
        metrics.actionFPS = Double(metrics.actionCount) / elapsed
        let snapshot: RuntimeMetrics? = now - lastMetricsReportTime >= 0.1 ? metrics : nil
        if snapshot != nil { lastMetricsReportTime = now }
        lock.unlock()
        injector.execute(
            prediction,
            profile: profile,
            allowedKeyCodes: allowedKeyCodes,
            mouseMode: mouseMode,
            captureRect: rect,
            safety: safety,
            gameCamera: gameCamera,
            predictionIsFresh: latched.isFresh,
            shiftUsesKeyboardChannel: shiftUsesKeyboardChannel
        )
        if let snapshot { onMetrics?(snapshot) }
    }

    private func focusTargetIfNeeded(_ spec: CaptureSpec) async -> pid_t? {
        guard let id = spec.windowID,
              let content = try? await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false),
              let pid = content.windows.first(where: { $0.windowID == id })?.owningApplication?.processID,
              let app = NSRunningApplication(processIdentifier: pid) else { return nil }
        app.activate(options: [.activateAllWindows])
        return pid
    }

    private func drain(queue: DispatchQueue) async {
        await withCheckedContinuation { continuation in queue.async { continuation.resume() } }
    }

    private func waitForStartupCompletion() async {
        await withCheckedContinuation { continuation in
            let resumeImmediately = lock.withLock { () -> Bool in
                guard starting else { return true }
                startupWaiters.append(continuation)
                return false
            }
            if resumeImmediately { continuation.resume() }
        }
    }

    private func waitForTeardownCompletion() async {
        await withCheckedContinuation { continuation in
            let resumeImmediately = lock.withLock { () -> Bool in
                guard teardownInProgress else { return true }
                teardownWaiters.append(continuation)
                return false
            }
            if resumeImmediately { continuation.resume() }
        }
    }

    deinit {
        actionTimer?.setEventHandler {}
        actionTimer?.cancel()
        safetyMonitor.stop()
        injector.disableAndReleaseAll()
    }
}

private final class SendablePixelBuffer: @unchecked Sendable {
    let value: CVPixelBuffer
    init(_ value: CVPixelBuffer) { self.value = value }
}

/// A policy prediction contains both held state (buttons/keys) and transient
/// deltas (Game Camera/scroll). Held state may be inspected on every action
/// tick, but each published prediction's transient values must be consumed only
/// once or a slow inference frame moves the camera repeatedly.
struct RuntimePredictionLatch: Sendable {
    struct Snapshot: Sendable {
        var values: [Float]
        var isFresh: Bool
        var publishedAt: Double
    }

    private var latest: [Float]?
    private var revision: UInt64 = 0
    private var consumedRevision: UInt64 = 0
    private var publishedAt = 0.0

    mutating func publish(_ values: [Float], at time: Double = CACurrentMediaTime()) {
        latest = values
        publishedAt = time
        revision &+= 1
    }

    mutating func consume() -> Snapshot? {
        guard let latest else { return nil }
        let isFresh = revision != consumedRevision
        if isFresh { consumedRevision = revision }
        return Snapshot(values: latest, isFresh: isFresh, publishedAt: publishedAt)
    }

    mutating func reset() {
        latest = nil
        revision = 0
        consumedRevision = 0
        publishedAt = 0
    }
}

enum RuntimeActionSemantics {
    /// Saved channels define what a brain learned, but the current Modifiers
    /// off switch is a safety override for every brain generation. Intersection
    /// semantics prevent a later editor change from enabling an untrained head.
    static func effectiveChannels(saved: ActionChannels, current: ActionChannels) -> ActionChannels {
        var result = saved
        result.modifiers = saved.modifiers && current.modifiers
        return result
    }

    /// Produces the complete executable controls paired with one perceived
    /// frame. Every row represents a fresh prediction, so relative movement and
    /// scroll are retained once alongside held keys, modifiers, and buttons.
    static func temporalControlValues(
        _ prediction: [Float],
        channels: ActionChannels? = nil,
        restrictions: ActionRestrictions = ActionRestrictions(),
        allowedKeyCodes: Set<UInt16>? = nil,
        outputPermissions: RuntimeOutputPermissions = RuntimeOutputPermissions(),
        shiftUsesKeyboardChannel: Bool = true
    ) -> [Float] {
        guard prediction.count >= ActionLayout.count else { return prediction }
        var values = prediction
        let mouseMovementEnabled = channels?.mouseMovement ?? true
        let buttonsEnabled = channels?.buttons ?? true
        let scrollEnabled = channels?.scroll ?? true
        let keyboardEnabled = channels?.keyboard ?? true
        let modifiersEnabled = channels?.modifiers ?? true

        for index in ActionLayout.absoluteMouse {
            values[index] = outputPermissions.cursorMovement && mouseMovementEnabled
                ? min(1, max(0, values[index])) : 0
        }
        for index in ActionLayout.relativeMouse {
            values[index] = outputPermissions.cursorMovement && mouseMovementEnabled
                ? min(1, max(-1, values[index])) : 0
        }
        for button in 0..<8 {
            let allowed = buttonsEnabled && restrictions.allowsButton(UInt8(button))
            values[ActionLayout.buttons.lowerBound + button] = allowed && values[ActionLayout.buttons.lowerBound + button] >= 0.5 ? 1 : 0
        }
        for index in ActionLayout.scroll {
            values[index] = scrollEnabled ? min(1, max(-1, values[index])) : 0
        }

        let keyboardOutputEnabled = outputPermissions.keyboard && keyboardEnabled
        for key in 0..<128 {
            let code = UInt16(key)
            let capabilityAllows = allowedKeyCodes?.contains(code) ?? true
            let allowed = keyboardOutputEnabled
                && capabilityAllows
                && restrictions.allowsKey(code)
                && !ActionLayout.commandOptionControlKeyCodeSet.contains(code)
            let index = ActionLayout.keyboard.lowerBound + key
            values[index] = allowed && values[index] >= 0.5 ? 1 : 0
        }

        let modifierEquivalents: [[UInt16]] = [[56, 60], [59, 62], [58, 61], [55, 54]]
        for modifier in 0..<4 {
            let channelEnabled = modifier == 0 && shiftUsesKeyboardChannel
                ? keyboardEnabled : modifiersEnabled
            let capabilityAllows = allowedKeyCodes.map {
                !$0.isDisjoint(with: modifierEquivalents[modifier])
            } ?? true
            let allowed = outputPermissions.keyboard
                && channelEnabled
                && capabilityAllows
                && restrictions.allowsModifier(modifier)
            let index = ActionLayout.modifiers.lowerBound + modifier
            values[index] = allowed && values[index] >= 0.5 ? 1 : 0
        }
        return values
    }
}
