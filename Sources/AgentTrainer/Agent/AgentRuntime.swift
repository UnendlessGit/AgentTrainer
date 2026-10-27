@preconcurrency import CoreMedia
@preconcurrency import CoreVideo
@preconcurrency import AppKit
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
    private enum DecisionAdmission { case idle, infer, stale }

    var onState: (@Sendable (InputState) -> Void)?
    var onMetrics: (@Sendable (RuntimeMetrics) -> Void)?
    var onWillStop: (@Sendable (String?) -> Void)?
    var onStop: (@Sendable (String?) -> Void)?
    var onPreview: (@Sendable (VisionPreviewFrame) -> Void)?
    var onVisualization: (@Sendable (CNNVisualizationFrame) -> Void)?

    private let capture = CaptureService()
    private let preprocessor: VisionPreprocessor
    private let inferenceInputBuffers: MetalArrayBufferPool
    private let injector = InputInjector()
    private let safetyMonitor = InputCaptureService()
    private let inferenceQueue = DispatchQueue(label: "AgentTrainer.Inference", qos: .userInteractive)
    private let actionQueue = DispatchQueue(label: "AgentTrainer.Actions", qos: .userInteractive)
    private let lock = NSLock()
    private var rlOptions: RLRuntimeOptions?
    private var visionMask: CGRect?
    private var rlHeads: RLHeads?
    private var rlRandom = RLRandom(state: 42)
    private var rlRevision = 0
    private var rlActionsEnabled = false
    private var model: AgentPolicy?
    private var visualEncodingFunction: VisualizationFunction?
    private var predictionFunction: VisualizationFunction?
    private var activationVisualizationFunctions: [VisualizationFunction] = []
    private var channelVisualizationFunction: (@Sendable ([MLXArray]) -> [MLXArray])?
    private var saliencyVisualizationFunction: (@Sendable ([MLXArray]) -> [MLXArray])?
    private var saliencyGradientFunction: (([MLXArray]) -> MLXArray)?
    private var profile: AIProfile?
    private var contextSchema = ModelContextSchema.empty
    private var contextVector: [Float] = []
    private var contextRevision: UInt64 = 0
    private var contextTransitionPending = false
    private var safety = AgentSafetyPolicy()
    private var captureRect = CGRect.zero
    private var originalRLCaptureRect = CGRect.zero
    private var mode: FrameMode = .newest
    private var mouseMode: MouseControlMode = .absolute
    private var gameCamera = GameCameraSettings()
    private var outputPermissions = RuntimeOutputPermissions()
    private var outputPermissionsRevision = 0
    private var allowedKeyCodes: Set<UInt16> = []
    private var shiftUsesKeyboardChannel = false
    private var latestFrame: CVPixelBuffer?
    private var latestFrameHostNanos: UInt64 = 0
    private var lastUsableCaptureFrame: CVPixelBuffer?
    private var lastUsableFrameSourceNanos: UInt64 = 0
    private var actionInferencePending = false
    private var cachedVision: CachedVision?
    private var processing = false
    private var predictionLatch = RuntimePredictionLatch()
    private struct TemporalFrame: @unchecked Sendable {
        var packed: Data
        var controls: [Float]
        var timestampNanos: UInt64
        var timing: [Float]
        var embedding: MLXArray
        var revision: Int = 0
    }
    private struct CachedVision: @unchecked Sendable {
        var current: MLXArray
        var packed: Data
        var past: [TemporalFrame]
        var sourceTime: UInt64
        var publishedAt: Double
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

        mutating func replace(_ frame: TemporalFrame) {
            if let index = storage.firstIndex(where: { $0?.timestampNanos == frame.timestampNanos }) { storage[index] = frame }
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
    private var headLatencies: [Double] = []
    private var encodingLatencies: [Double] = []
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

    func start(profile: AIProfile, version: ModelVersionManifest, allowedKeyCodes: Set<UInt16>, captureSpec: CaptureSpec, captureRect: CGRect, mode: FrameMode, mouseMode: MouseControlMode, gameCamera: GameCameraSettings = GameCameraSettings(), outputPermissions: RuntimeOutputPermissions = RuntimeOutputPermissions(), contextSelection: [UUID: UUID] = [:], safety: AgentSafetyPolicy, previewFPS: Double = 0, visualizationSettings: CNNVisualizationSettings = CNNVisualizationSettings(), ignoredHotkeys: [HotkeyBinding] = [], rlOptions: RLRuntimeOptions? = nil, visionMask: CGRect? = nil) async throws {
        guard AXIsProcessTrusted() else {
            let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
            AXIsProcessTrustedWithOptions(options)
            throw AgentTrainerError.permission("Accessibility permission is required before AgentTrainer can press keys or control the mouse.")
        }
        guard version.schemaVersion == ModelContract.schemaVersion,
              version.temporalInputVersion == TemporalInputContract.version,
              version.relativeMouseScale == GameCameraContract.deltaScale else {
            throw AgentTrainerError.model("This brain predates the current visual and Game Camera contracts. Retrain it from the original recordings.")
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
        runtimeProfile.preprocessing = version.preprocessing
        runtimeProfile.channels = RuntimeActionSemantics.effectiveChannels(saved: version.channels, current: profile.channels)
        runtimeProfile.training = version.training
        var runtimeAllowedKeyCodes = allowedKeyCodes
        if let capabilities = version.outputCapabilities {
            runtimeProfile.channels = capabilities.restricting(runtimeProfile.channels, mouseMode: mouseMode)
            runtimeProfile.restrictions = capabilities.restricting(profile.effectiveRestrictions)
            runtimeAllowedKeyCodes.formIntersection(capabilities.allowedKeyCodes)
        } else {
            // Private or hand-authored manifests lacking a button vocabulary
            // cannot establish that a button's binary head was ever trained.
            runtimeProfile.channels.buttons = false
        }
        _ = try runtimeProfile.preprocessing.validated()
        _ = try runtimeProfile.training.effectiveTemporalVision.validated(
            current: runtimeProfile.preprocessing, visualEmbedding: runtimeProfile.training.architecture.visualEmbedding
        )
        let contextSchema = try version.effectiveContextSchema.validated()
        let initialContextVector = contextSchema.vector(selection: contextSelection)
        let startRevisions = lock.withLock { (outputPermissionsRevision, visualizationSettingsRevision) }
        let model = AgentPolicy(profile: runtimeProfile, contextSchema: contextSchema, movementSensitivity: mouseMode == .relative ? gameCamera.sensitivity : 1)
        let versionDirectory = await WorkspaceStore.shared.versionDirectory(profileID: profile.id, versionID: version.id)
        try model.loadWeights(from: versionDirectory.appendingPathComponent(version.weightsFile))
        model.train(false)
        let rlHeads: RLHeads? = rlOptions.map { options in
            let heads = RLHeads(width: model.fusionWidth)
            return heads
        }
        if let rlHeads, let rlOptions {
            try RLWeightTransfer.install(rlOptions.initial.heads, in: rlHeads)
            try RLWeightTransfer.install(rlOptions.initial.policy, in: model)
        }
        let temporal = runtimeProfile.training.effectiveTemporalVision
        let hasTemporalMemory = temporal.pastFrameCount > 0
        let hasContext = !contextSchema.isEmpty
        let currentTimingIndex = hasTemporalMemory ? 3 : 1
        let contextInputIndex = hasTemporalMemory ? 5 : 2
        let pastSpec = temporal.pastFrameSpec(from: runtimeProfile.preprocessing)
        let visualEncodingFunction: VisualizationFunction = compile(inputs: [model]) { inputs in
            let current = model.visualEmbedding(visualFeatures: model.visualActivations(
                images: VisionPreprocessor.mlxTensor(inputs[0], spec: runtimeProfile.preprocessing)).last!)
            if hasTemporalMemory {
                let past = model.pastVisualEmbedding(visualFeatures: model.pastVisualActivations(
                    images: VisionPreprocessor.mlxTensor(inputs[1], spec: pastSpec)).last!)
                return [current, past]
            }
            return [current]
        }
        let predictionFunction = RuntimePolicyGraph.make(model: model, heads: rlHeads)
        // Diagnostic graphs are lazy: creating these closures does not execute
        // or materialize an extra tensor. The selected graph first compiles only
        // when its view is enabled and reaches its independently capped rate.
        let activationVisualizationFunctions: [VisualizationFunction] = (0..<max(1, model.convolutions.count)).map { selectedLayer in
            compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
                let currentImages = VisionPreprocessor.mlxTensor(inputs[0], spec: runtimeProfile.preprocessing)
                let layers = model.visualActivations(images: currentImages)
                let contextValues = hasContext ? inputs[contextInputIndex] : nil
                let temporalFeatures = hasTemporalMemory
                    ? model.temporalFeatures(
                        currentVisualFeatures: layers.last!,
                        pastImages: VisionPreprocessor.mlxPastFrameTensor(inputs[1], spec: pastSpec),
                        pastControls: inputs[2],
                    currentTiming: inputs[currentTimingIndex], pastTiming: inputs[4]
                    )
                    : model.currentOnlyTemporalFeatures(currentVisualFeatures: layers.last!, currentTiming: inputs[currentTimingIndex])
                let logits = model.logits(
                    temporalFeatures: temporalFeatures,
                    contextValues: contextValues
                )
                let map = model.sampledForVisualization(layers[selectedLayer]).mean(axis: -1, keepDims: true)
                return [model.activatedPredictions(logits: logits), map]
            }
        }
        let channelVisualizationFunction = compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
            let currentImages = VisionPreprocessor.mlxTensor(inputs[0], spec: runtimeProfile.preprocessing)
            let layers = model.visualActivations(images: currentImages)
            let contextValues = hasContext ? inputs[contextInputIndex] : nil
            let temporalFeatures = hasTemporalMemory
                ? model.temporalFeatures(
                    currentVisualFeatures: layers.last!,
                    pastImages: VisionPreprocessor.mlxPastFrameTensor(inputs[1], spec: pastSpec),
                    pastControls: inputs[2],
                    currentTiming: inputs[currentTimingIndex], pastTiming: inputs[4]
                )
                : model.currentOnlyTemporalFeatures(currentVisualFeatures: layers.last!, currentTiming: inputs[currentTimingIndex])
            let logits = model.logits(
                temporalFeatures: temporalFeatures,
                contextValues: contextValues
            )
            return [model.activatedPredictions(logits: logits), model.strongestChannelsForVisualization(layers.last!)]
        }
        let saliencyVisualizationFunction = compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
            let currentImages = VisionPreprocessor.mlxTensor(inputs[0], spec: runtimeProfile.preprocessing)
            let layers = model.visualActivations(images: currentImages)
            let contextValues = hasContext ? inputs[contextInputIndex] : nil
            let temporalFeatures = hasTemporalMemory
                ? model.temporalFeatures(
                    currentVisualFeatures: layers.last!,
                    pastImages: VisionPreprocessor.mlxPastFrameTensor(inputs[1], spec: pastSpec),
                    pastControls: inputs[2],
                    currentTiming: inputs[currentTimingIndex], pastTiming: inputs[4]
                )
                : model.currentOnlyTemporalFeatures(currentVisualFeatures: layers.last!, currentTiming: inputs[currentTimingIndex])
            let logits = model.logits(
                temporalFeatures: temporalFeatures,
                contextValues: contextValues
            )
            // Keep the exact final tensor on GPU for the post-CNN gradient.
            // This graph intentionally omits the channel ranking used by the
            // separate feature-grid view.
            return [model.activatedPredictions(logits: logits), layers.last!]
        }
        let saliencyGradient = grad({ (inputs: [MLXArray]) -> MLXArray in
            if hasTemporalMemory {
                let pastImages = VisionPreprocessor.mlxPastFrameTensor(inputs[1], spec: pastSpec)
                let contextValues = hasContext ? inputs[5] : nil
                let selector = inputs[hasContext ? 6 : 5]
                let logits = model.logits(
                    currentVisualFeatures: inputs[0],
                    pastImages: pastImages,
                    pastControls: inputs[2],
                    contextValues: contextValues,
                    currentTiming: inputs[3], pastTiming: inputs[4]
                )
                return (logits * selector.asType(model.dtype)).sum()
            }
            let contextValues = hasContext ? inputs[2] : nil
            let selector = inputs[hasContext ? 3 : 2]
            let logits = model.logits(
                temporalFeatures: model.currentOnlyTemporalFeatures(currentVisualFeatures: inputs[0], currentTiming: inputs[1]),
                contextValues: contextValues
            )
            return (logits * selector.asType(model.dtype)).sum()
        }, argumentNumbers: [0])
        let accepted = lock.withLock { () -> Bool in
            guard launchRevision == launchToken, starting, stopped else { return false }
            self.model = model
            self.rlOptions = rlOptions; self.rlHeads = rlHeads; self.visionMask = rlOptions?.mask ?? visionMask
            self.rlRandom = rlOptions?.random ?? RLRandom(state: 42)
            self.rlRevision = rlOptions?.initial.revision ?? 0; self.rlActionsEnabled = rlOptions == nil
            self.visualEncodingFunction = visualEncodingFunction
            self.predictionFunction = predictionFunction
            self.activationVisualizationFunctions = activationVisualizationFunctions
            self.channelVisualizationFunction = channelVisualizationFunction
            self.saliencyVisualizationFunction = saliencyVisualizationFunction
            self.saliencyGradientFunction = saliencyGradient
            self.profile = runtimeProfile
            self.contextSchema = contextSchema
            self.contextVector = initialContextVector
            self.contextRevision &+= 1
            self.contextTransitionPending = false
            self.allowedKeyCodes = runtimeAllowedKeyCodes
            self.shiftUsesKeyboardChannel = (version.trainingDataSchema ?? 0) >= 7
            self.safety = safety
            self.captureRect = captureRect; self.originalRLCaptureRect = captureRect
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
            latestFrame = nil; latestFrameHostNanos = 0; lastUsableCaptureFrame = nil
            predictionLatch.reset(); temporalFrames.reset(); cachedVision = nil; actionInferencePending = false; lastUsableFrameSourceNanos = 0
            metrics = RuntimeMetrics(); startedAt = CACurrentMediaTime(); nextPerceptionTime = 0; lastMetricsReportTime = 0; lastFocusCheckTime = 0; stopped = false
            headLatencies = []; encodingLatencies = []
            // Keep session enablement ordered with live permission changes. A
            // toggle made while model weights are loading must not be replaced
            // by the older settings captured when `start` was first called.
            injector.enable(outputPermissions: self.outputPermissions)
            stopped = false
            starting = false
            return true
        }
        guard accepted else { throw CancellationError() }
        do {
            guard lock.withLock({ !stopped }) else { throw CancellationError() }
            safetyMonitor.ignoredHotkeys = ignoredHotkeys
            safetyMonitor.onSample = { [weak self] sample in
                guard let self else { return }
                let panic = sample.kind == .key && sample.isDown && sample.keyCode == self.safety.panicKeyCode && (sample.modifiers & self.safety.panicModifiers) == self.safety.panicModifiers
                guard panic || self.safety.stopOnHumanInput else { return }
                self.suspendRLActions()
                Task { await self.stop(reason: panic ? "Panic stop" : "Stopped on human input") }
            }
            try safetyMonitor.start()
            let rewardTarget = rlOptions.map { CGPoint(x: captureRect.minX + $0.mask.midX * captureRect.width,
                y: captureRect.minY + $0.mask.midY * captureRect.height) }
            let targetPID = await focusTargetIfNeeded(captureSpec, rewardTarget: rewardTarget)
            if rlOptions != nil && targetPID == nil { throw AgentTrainerError.capture("The reward region must belong to an available game or application window.") }
            guard lock.withLock({ !stopped }) else { throw CancellationError() }
            lock.withLock { self.targetPID = targetPID }
            var liveCaptureSpec = captureSpec
            liveCaptureSpec.requestedFPS = runtimeProfile.training.perceptionFPS
            liveCaptureSpec.showsCursor = version.trainingShowsCursor ?? captureSpec.showsCursor
            try await capture.start(spec: liveCaptureSpec, queueDepth: mode == .newest ? 3 : 8, nativeWindowGeometry: rlOptions != nil || visionMask != nil, onFrame: { [weak self] buffer, pts in
                self?.schedule(buffer, timestampNanos: Self.hostTimeNanos(pts), countThrottledAsDropped: true)
            }, onIdle: { [weak self] pts in
                self?.receiveIdle(timestamp: pts)
            }, onUnexpectedStop: { [weak self] error in
                self?.suspendRLActions()
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
            safetyMonitor.onState = nil
            _ = try? await capture.stop()
            injector.disableAndReleaseAll()
            throw error
        }
    }

    func moveRLCapture(by offset: CGSize) {
        lock.withLock {
            let rect = originalRLCaptureRect.offsetBy(dx: offset.width, dy: offset.height)
            if rect != captureRect { captureRect = rect; contextRevision &+= 1; predictionLatch.reset() }
        }
    }

    func suspendRLActions() {
        lock.withLock { rlActionsEnabled = false; contextRevision &+= 1; predictionLatch.reset() }
        injector.disableAndReleaseAll()
    }

    func beginRLEpisode() async {
        let revision = lock.withLock { contextRevision }
        await drain(queue: inferenceQueue)
        await drain(queue: actionQueue)
        lock.withLock {
            guard !stopped, rlOptions != nil, contextRevision == revision else { return }
            temporalFrames.reset(); cachedVision = nil; predictionLatch.reset()
            contextRevision &+= 1
            injector.enable(outputPermissions: outputPermissions)
            rlActionsEnabled = true
        }
    }

    func awaitInitialRLFocus() async throws {
        guard lock.withLock({ !stopped }) else { throw CancellationError() }
        guard let pid = lock.withLock({ targetPID }), let app = NSRunningApplication(processIdentifier: pid) else {
            throw AgentTrainerError.capture("The training application is unavailable.")
        }
        await MainActor.run { _ = app.activate(options: [.activateAllWindows]) }
        for _ in 0..<100 {
            guard lock.withLock({ !stopped }) else { throw CancellationError() }
            if NSWorkspace.shared.frontmostApplication?.processIdentifier == pid { return }
            try await Task.sleep(for: .milliseconds(20))
        }
        throw AgentTrainerError.capture("Activate the training application and start RL using its global shortcut.")
    }

    func episodeKey(_ key: UInt16?) async throws {
        guard let key else { return }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            actionQueue.async { [self] in
                do {
                    try lock.withLock {
                        guard !stopped, !rlActionsEnabled, outputPermissions.keyboard, AXIsProcessTrusted() else {
                            throw AgentTrainerError.permission("Episode keys require keyboard output and Accessibility permission.")
                        }
                        if let targetPID, NSWorkspace.shared.frontmostApplication?.processIdentifier != targetPID {
                            throw AgentTrainerError.model("The target application lost focus before an episode key could be sent.")
                        }
                        injector.tapControllerKey(key)
                    }
                    continuation.resume()
                } catch { continuation.resume(throwing: error) }
            }
        }
    }

    func rlRandomState() async -> RLRandom {
        await withCheckedContinuation { continuation in
            inferenceQueue.async { [self] in continuation.resume(returning: rlRandom) }
        }
    }

    func installRLSnapshot(_ snapshot: RLWeightSnapshot) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            inferenceQueue.async { [self] in
                do {
                    guard let model = lock.withLock({ !stopped ? self.model : nil }), let rlHeads else {
                        throw CancellationError()
                    }
                    try RLWeightTransfer.install(snapshot.policy, in: model)
                    try RLWeightTransfer.install(snapshot.heads, in: rlHeads)
                    var vision = lock.withLock { cachedVision }
                    if vision != nil {
                        let images = VisionPreprocessor.mlxTensor(vision!.packed, batch: 1, spec: model.profile.preprocessing)
                        let current = model.visualEmbedding(visualFeatures: model.visualActivations(images: images).last!)
                        MLX.eval(current); vision!.current = current
                        let refreshed = RLPastEncoder.encode(vision!.past.map(\.packed), model: model)
                        for i in vision!.past.indices { vision!.past[i].embedding = refreshed[i]; vision!.past[i].revision = snapshot.revision }
                    }
                    lock.withLock {
                        if !stopped { cachedVision = vision; rlRevision = snapshot.revision }
                    }
                    continuation.resume()
                } catch { continuation.resume(throwing: error) }
            }
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

    /// Context is an ordinary live model input, not mutable model metadata.
    /// Updating a picker therefore takes effect on the next admitted perception
    /// without recompiling the graph; old-context held controls are released.
    func updateContextSelection(_ selection: [UUID: UUID]) {
        let transitionRevision = lock.withLock { () -> UInt64? in
            let updated = contextSchema.vector(selection: selection)
            guard updated != contextVector else { return nil }
            contextVector = updated
            contextRevision &+= 1
            temporalFrames.reset()
            latestFrame = nil; latestFrameHostNanos = 0
            cachedVision = nil
            predictionLatch.reset()
            guard !stopped else { return nil }
            contextTransitionPending = true
            return contextRevision
        }
        guard let transitionRevision else { return }
        actionQueue.async { [weak self] in
            guard let self else { return }
            self.lock.withLock {
                guard !self.stopped else { return }
                // A context switch is a semantic session boundary. Release any
                // state held by the previous prediction before admitting the
                // first action conditioned on the new values.
                self.injector.disableAndReleaseAll()
                self.injector.enable(outputPermissions: self.outputPermissions)
                if self.contextRevision == transitionRevision {
                    self.contextTransitionPending = false
                }
            }
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

    func stop(reason: String? = nil) async {
        let action = lock.withLock { () -> StopAction in
            launchRevision &+= 1
            if teardownInProgress { return .waitForTeardown }
            guard !stopped else { return starting ? .waitForStartup : .finished }
            stopped = true
            teardownInProgress = true
            let timer = actionTimer; actionTimer = nil
            latestFrame = nil; latestFrameHostNanos = 0; lastUsableCaptureFrame = nil
            temporalFrames.reset(); cachedVision = nil; predictionLatch.reset(); targetPID = nil
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
        if let reason { AppLog.write(category: "Runtime", "Stopped execution", details: reason) }
        onWillStop?(reason)
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
        safetyMonitor.onState = nil
        safetyMonitor.excludedKeyCodes = []
        _ = try? await capture.stop()
        await drain(queue: inferenceQueue)
        // An inference that published just before stop may enqueue its guarded
        // execution after the first action drain. Join that final callback too,
        // before a new session can own the pending-decision flag.
        await drain(queue: actionQueue)
        lock.withLock {
            visualEncodingFunction = nil; cachedVision = nil; actionInferencePending = false; predictionFunction = nil; activationVisualizationFunctions.removeAll(keepingCapacity: false); channelVisualizationFunction = nil; saliencyVisualizationFunction = nil; saliencyGradientFunction = nil; model = nil; rlHeads = nil; rlOptions = nil; profile = nil; contextSchema = .empty; contextVector.removeAll(keepingCapacity: false); contextRevision &+= 1; contextTransitionPending = false; allowedKeyCodes.removeAll(keepingCapacity: false); shiftUsesKeyboardChannel = false
            latestFrame = nil; latestFrameHostNanos = 0; lastUsableCaptureFrame = nil; temporalFrames.reset(); predictionLatch.reset(); processing = false
            visualizationSettings = CNNVisualizationSettings(); lastVisualizationTime = 0
        }
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

    private func receiveIdle(timestamp: CMTime) {
        let frame = lock.withLock { stopped ? nil : lastUsableCaptureFrame.map { ($0, lastUsableFrameSourceNanos) } }
        if let frame {
            schedule(frame.0, timestampNanos: frame.1, countThrottledAsDropped: false)
        }
    }

    private func schedule(
        _ buffer: CVPixelBuffer,
        timestampNanos: UInt64,
        countThrottledAsDropped: Bool
    ) {
        let now = CACurrentMediaTime()
        lock.lock()
        guard !stopped, let profile else { lock.unlock(); return }
        lastUsableCaptureFrame = buffer
        if countThrottledAsDropped { lastUsableFrameSourceNanos = timestampNanos }
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
            inferenceQueue.sync { [weak self] in
                self?.infer(frame.value, timestampNanos: timestampNanos)
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

    private func drainNewest() {
        let next = lock.withLock { () -> (CVPixelBuffer, UInt64)? in
            guard !stopped, let frame = latestFrame else { processing = false; return nil }
            latestFrame = nil
            return (frame, latestFrameHostNanos)
        }
        guard let next else { return }
        infer(next.0, timestampNanos: next.1)
        let again = lock.withLock { () -> Bool in
            if latestFrame != nil && !stopped { return true }
            processing = false; return false
        }
        // Yield between encodes so a pending action head cannot be starved by
        // a continuously arriving newest-frame stream.
        if again { inferenceQueue.async { [weak self] in self?.drainNewest() } }
    }

    private func infer(_ buffer: CVPixelBuffer, timestampNanos: UInt64) {
        lock.lock()
        guard !stopped, let visualEncodingFunction, let model, let profile else { lock.unlock(); return }
        let temporal = profile.training.effectiveTemporalVision
        var selectedPriorFrames: [TemporalFrame?] = (0..<temporal.pastFrameCount).map { frame in
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
        let contextVector = self.contextVector
        let inferenceContextRevision = contextRevision
        let inferencePolicyRevision = rlRevision
        let inferencePermissionsRevision = outputPermissionsRevision
        let rect = captureRect
        lock.unlock()
        let began = CACurrentMediaTime()
        do {
            let stale = selectedPriorFrames.indices.filter { selectedPriorFrames[$0].map { $0.revision != inferencePolicyRevision } ?? false }
            let refreshed = RLPastEncoder.encode(stale.map { selectedPriorFrames[$0]!.packed }, model: model)
            for (i, slot) in stale.enumerated() { selectedPriorFrames[slot]!.embedding = refreshed[i]; selectedPriorFrames[slot]!.revision = inferencePolicyRevision }
            let pastSpec = temporal.pastFrameSpec(from: profile.preprocessing)
            let hasTemporalMemory = temporal.pastFrameCount > 0
            let sharesPackedRepresentation = hasTemporalMemory && pastSpec == profile.preprocessing
            let requestedSpecs = hasTemporalMemory && !sharesPackedRepresentation
                ? [profile.preprocessing, pastSpec]
                : [profile.preprocessing]
            // One source mapping and command buffer produces every resolution
            // needed by this perception.
            let jobs = try preprocessor.submit(buffer, specs: requestedSpecs, mask: visionMask)
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
            let frameTimestamp = timestampNanos > 0 ? timestampNanos : Self.hostTimeNanos()
            let snapshot = injector.temporalSnapshot(frameTime: frameTimestamp, rect: rect, perception: true)
            let zeroControls = [Float](repeating: 0, count: ActionLayout.count)
            let zeroTiming = [Float](repeating: 0, count: TemporalInputContract.valueCount)
            var descriptors: [MetalArrayBufferPool.Descriptor] = [
                .init([1, profile.preprocessing.sampleByteCount], dtype: .uint8)
            ]
            if hasTemporalMemory {
                descriptors.append(.init([1, temporal.pastFrameCount, pastSpec.sampleByteCount], dtype: .uint8))
                descriptors.append(.init([1, temporal.pastFrameCount, ActionLayout.count], dtype: .float32))
            }
            let currentTimingIndex = descriptors.count
            descriptors.append(.init([1, TemporalInputContract.valueCount], dtype: .float32))
            let pastTimingIndex: Int? = hasTemporalMemory ? descriptors.count : nil
            if hasTemporalMemory { descriptors.append(.init([1, temporal.pastFrameCount, TemporalInputContract.valueCount], dtype: .float32)) }
            let contextInputIndex: Int? = contextVector.isEmpty ? nil : descriptors.count
            if !contextVector.isEmpty {
                descriptors.append(.init([1, contextVector.count], dtype: .float32))
            }
            let reducedCurrentIndex: Int? = hasTemporalMemory ? descriptors.count : nil
            if hasTemporalMemory { descriptors.append(.init([1, pastSpec.sampleByteCount], dtype: .uint8)) }
            let inferenceInputs = try inferenceInputBuffers.makeArrays(descriptors) { destinations in
                packed.withUnsafeBytes { source in
                    destinations[0].copyMemory(from: source)
                }
                if hasTemporalMemory {
                    let controlRowBytes = ActionLayout.count * MemoryLayout<Float>.size
                    for (frame, selected) in selectedPriorFrames.enumerated() {
                        (selected?.packed ?? packedPastFrame).withUnsafeBytes { source in
                            UnsafeMutableRawBufferPointer(
                                rebasing: destinations[1][frame * pastSpec.sampleByteCount..<(frame + 1) * pastSpec.sampleByteCount]
                            ).copyMemory(from: source)
                        }
                        (selected?.controls ?? zeroControls).withUnsafeBytes { source in
                            UnsafeMutableRawBufferPointer(
                                rebasing: destinations[2][frame * controlRowBytes..<(frame + 1) * controlRowBytes]
                            ).copyMemory(from: source)
                        }
                    }
                }
                snapshot.features.withUnsafeBytes { destinations[currentTimingIndex].copyMemory(from: $0) }
                if let pastTimingIndex {
                    for (frame, prior) in selectedPriorFrames.enumerated() {
                        let values = Self.historicalTiming(prior, at: snapshot.timestamp,
                            previous: frame > 0 ? selectedPriorFrames[frame - 1]?.timestampNanos : nil)
                        values.withUnsafeBytes { source in
                            UnsafeMutableRawBufferPointer(rebasing: destinations[pastTimingIndex][frame * TemporalInputContract.rowBytes..<(frame + 1) * TemporalInputContract.rowBytes]).copyMemory(from: source)
                        }
                    }
                }
                if let reducedCurrentIndex { packedPastFrame.withUnsafeBytes { destinations[reducedCurrentIndex].copyMemory(from: $0) } }
                if let contextInputIndex {
                    contextVector.withUnsafeBytes {
                        destinations[contextInputIndex].copyMemory(from: $0)
                    }
                }
            }
            let encoderInputs = reducedCurrentIndex.map { [inferenceInputs[0], inferenceInputs[$0]] } ?? [inferenceInputs[0]]
            let encoded = Device.withDefaultDevice(.gpu) { visualEncodingFunction(encoderInputs) }
            MLX.eval(encoded)
            let fallback: [MLXArray] = []
            let result: [MLXArray] = Device.withDefaultDevice(.gpu) {
                guard visualizationDue else { return fallback }
                switch settings.mode {
                case .activationOverlay:
                    let selectedLayer = max(0, settings.convolutionLayer)
                    guard activationVisualizationFunctions.indices.contains(selectedLayer) else { return fallback }
                    return activationVisualizationFunctions[selectedLayer](inferenceInputs)
                case .featureChannels:
                    guard let forward = channelVisualizationFunction?(inferenceInputs), forward.count >= 2 else { return fallback }
                    return forward
                case .actionSaliency:
                    let selector = Self.actionSelector(focus: settings.actionFocus, mouseMode: mouseMode)
                    guard let saliencyVisualizationFunction, let saliencyGradientFunction else { return fallback }
                    let forward = saliencyVisualizationFunction(inferenceInputs)
                    guard forward.count >= 2 else { return fallback }
                    var gradientInputs = hasTemporalMemory
                        ? [forward[1], inferenceInputs[1], inferenceInputs[2], inferenceInputs[currentTimingIndex], inferenceInputs[pastTimingIndex!]]
                        : [forward[1], inferenceInputs[currentTimingIndex]]
                    if let contextInputIndex {
                        gradientInputs.append(inferenceInputs[contextInputIndex])
                    }
                    gradientInputs.append(selector)
                    let gradients = saliencyGradientFunction(gradientInputs)
                    let weights = gradients.mean(axes: [1, 2], keepDims: true)
                    let saliency = relu((forward[1] * weights).sum(axis: -1, keepDims: true))
                    return [forward[0], model.sampledForVisualization(saliency)]
                }
            }
            let visualization = visualizationDue ? Self.visualizationFrame(profile: profile, settings: settings, outputs: result) : nil
            if let visualization { MLX.eval(visualization.arrays) }
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
            lock.lock()
            guard !stopped, !contextTransitionPending,
                  contextRevision == inferenceContextRevision,
                  outputPermissionsRevision == inferencePermissionsRevision else {
                lock.unlock()
                return
            }
            let fallbackFrame = hasTemporalMemory ? TemporalFrame(packed: packedPastFrame,
                controls: zeroControls, timestampNanos: snapshot.timestamp,
                timing: zeroTiming, embedding: encoded[1], revision: inferencePolicyRevision) : nil
            for frame in selectedPriorFrames.compactMap({ $0 }) { temporalFrames.replace(frame) }
            let selectedFrames = selectedPriorFrames.compactMap { $0 ?? fallbackFrame }
            cachedVision = CachedVision(current: encoded[0], packed: packed, past: selectedFrames,
                sourceTime: frameTimestamp, publishedAt: CACurrentMediaTime())
            if hasTemporalMemory {
                temporalFrames.append(TemporalFrame(packed: packedPastFrame,
                    controls: snapshot.controls, timestampNanos: snapshot.timestamp,
                    timing: snapshot.features, embedding: encoded[1], revision: inferencePolicyRevision),
                    capacity: temporal.pastFrameCount * temporal.frameSpacing + 1)
            }
            metrics.frameCount += 1
            let elapsed = max(0.001, CACurrentMediaTime() - startedAt)
            metrics.perceptionFPS = Double(metrics.frameCount) / elapsed
            metrics.latencyMilliseconds = (CACurrentMediaTime() - began) * 1_000
            encodingLatencies.append(metrics.latencyMilliseconds)
            if encodingLatencies.count > 512 { encodingLatencies.removeFirst() }
            let now = CACurrentMediaTime()
            if now - lastMetricsReportTime >= 0.1 {
                metrics.encodingP95Milliseconds = Self.p95(encodingLatencies); metrics.actionP95Milliseconds = Self.p95(headLatencies)
            }
            let metricsSnapshot: RuntimeMetrics? = now - lastMetricsReportTime >= 0.1 ? metrics : nil
            if metricsSnapshot != nil { lastMetricsReportTime = now }
            lock.unlock()
            if let metricsSnapshot { onMetrics?(metricsSnapshot) }
        } catch {
            Task { await stop(reason: error.localizedDescription) }
        }
    }

    private struct VisualizationArrays {
        var settings: CNNVisualizationSettings
        var arrays: [MLXArray]
        var layers: [Int]
    }

    private static func visualizationFrame(profile: AIProfile, settings rawSettings: CNNVisualizationSettings, outputs: [MLXArray]) -> VisualizationArrays? {
        let modelLayerCount = max(1, profile.training.architecture.convolutionChannels.count)
        let settings = rawSettings.sanitized(layerCount: profile.training.architecture.convolutionChannels.count)
        let hasConvolutions = !profile.training.architecture.convolutionChannels.isEmpty
        let finalLayer = hasConvolutions ? modelLayerCount - 1 : -1
        let selected: [(MLXArray, Int)]
        switch settings.mode {
        case .activationOverlay:
            guard outputs.indices.contains(1) else { return nil }
            selected = [(outputs[1], hasConvolutions ? settings.convolutionLayer : -1)]
        case .featureChannels, .actionSaliency:
            guard outputs.indices.contains(1) else { return nil }
            selected = [(outputs[1], finalLayer)]
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
        var values = [Float](repeating: 0, count: PolicyLogitsLayout.count)
        let weight = 1 / Float(max(1, indices.count))
        for index in indices { values[index] = weight }
        return MLXArray(values, [1, PolicyLogitsLayout.count])
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

    private static func historicalTiming(_ frame: TemporalFrame?, at now: UInt64, previous: UInt64?) -> [Float] {
        guard let frame, frame.timing[0] > 0 else { return [Float](repeating: 0, count: TemporalInputContract.valueCount) }
        var values = frame.timing
        for i in 10..<16 { values[i] = 0 }
        TemporalInputContract.putTime(TemporalInputContract.seconds(now, since: frame.timestampNanos), in: &values, at: 10)
        if let previous {
            TemporalInputContract.putTime(TemporalInputContract.seconds(frame.timestampNanos, since: previous), in: &values, at: 13)
        }
        return values
    }

    private func actionTick() {
        let admission = lock.withLock { () -> DecisionAdmission in
            guard !stopped, let profile, rlOptions == nil || rlActionsEnabled else { return .idle }
            let maximumAge = max(0.35, 3 / max(0.0001, profile.training.perceptionFPS))
            // This check runs on the action timer, independently of MLX. A
            // queued or stalled head must not keep an old physical hold alive.
            if let age = predictionLatch.age(), age > maximumAge { return .stale }
            guard !contextTransitionPending, cachedVision != nil, !actionInferencePending else { return .idle }
            actionInferencePending = true
            return .infer
        }
        switch admission {
        case .idle: break
        case .stale:
            suspendRLActions()
            Task { await stop(reason: "Stopped because action inference stopped producing fresh predictions") }
        case .infer:
            inferenceQueue.async { [weak self] in self?.inferAction() }
        }
    }

    /// Only this lightweight graph runs at Action FPS. All MLX work remains on
    /// one serial queue; at most one decision and one newest image are pending.
    private func inferAction() {
        let state = lock.withLock { () -> (VisualizationFunction, AIProfile, CachedVision, [Float], UInt64, Int, CGRect)? in
            guard !stopped, rlOptions == nil || rlActionsEnabled, !contextTransitionPending, let predictionFunction, let profile, let cachedVision else { return nil }
            return (predictionFunction, profile, cachedVision, contextVector, contextRevision, outputPermissionsRevision, captureRect)
        }
        guard let (predict, profile, vision, context, revision, permissionsRevision, rect) = state else {
            lock.withLock { actionInferencePending = false }; return
        }
        let maximumAge = max(0.35, 3 / max(0.0001, profile.training.perceptionFPS))
        guard CACurrentMediaTime() - vision.publishedAt <= maximumAge else {
            lock.withLock { actionInferencePending = false }
            Task { await stop(reason: "Stopped because live vision stopped producing fresh observations") }
            return
        }
        do {
            let began = CACurrentMediaTime()
            let snapshot = injector.temporalSnapshot(frameTime: vision.sourceTime, rect: rect)
            var inputs = [vision.current]
            if !vision.past.isEmpty {
                inputs.append(stacked(vision.past.map { $0.embedding[0] }, axis: 0).expandedDimensions(axis: 0))
                inputs.append(MLXArray(vision.past.flatMap(\.controls), [1, vision.past.count, ActionLayout.count]))
            }
            var timing = snapshot.features
            timing.withUnsafeMutableBufferPointer {
                TemporalInputContract.sanitize($0, rows: 1, channels: profile.channels, restrictions: profile.effectiveRestrictions)
            }
            inputs.append(MLXArray(timing, [1, TemporalInputContract.valueCount]))
            if !vision.past.isEmpty {
                var pastTiming: [Float] = []
                for (slot, frame) in vision.past.enumerated() {
                    pastTiming += Self.historicalTiming(frame, at: snapshot.timestamp,
                        previous: slot > 0 && vision.past[slot - 1].timing[0] > 0
                            ? vision.past[slot - 1].timestampNanos : nil)
                }
                pastTiming.withUnsafeMutableBufferPointer {
                    TemporalInputContract.sanitize($0, rows: vision.past.count, channels: profile.channels, restrictions: profile.effectiveRestrictions)
                }
                inputs.append(MLXArray(pastTiming, [1, vision.past.count, TemporalInputContract.valueCount]))
            }
            if !context.isEmpty { inputs.append(MLXArray(context, [1, context.count])) }
            let outputs = Device.withDefaultDevice(.gpu) { predict(inputs) }
            MLX.eval(outputs)
            rlOptions?.budget?.noteInference(milliseconds: (CACurrentMediaTime() - began) * 1000, actionFPS: profile.training.actionFPS)
            var values = outputs[0].asArray(Float.self)
            var decision: RLDecision?
            if let options = rlOptions, outputs.count == 4 {
                let sampled = options.distribution.sample(logits: outputs[1].asArray(Float.self),
                    logStd: outputs[3].asArray(Float.self), random: &rlRandom)
                guard sampled.logProbability.isFinite, sampled.latent.allSatisfy(\.isFinite), outputs[2].item(Float.self).isFinite else {
                    throw AgentTrainerError.model("The RL policy produced a non-finite action or value; inputs were stopped safely.")
                }
                values = options.deterministic ? outputs[0].asArray(Float.self) : sampled.prediction
                let currentTiming = inputs[vision.past.isEmpty ? 1 : 3].asArray(Float.self)
                let pastTiming = vision.past.isEmpty ? [] : inputs[4].asArray(Float.self)
                decision = RLDecision(observation: RLObservation(current: vision.packed, past: vision.past.map(\.packed),
                    controls: vision.past.flatMap(\.controls), timing: currentTiming, pastTiming: pastTiming, context: context),
                    latent: sampled.latent, binary: sampled.binary, logProbability: sampled.logProbability,
                    value: outputs[2].item(Float.self), timestamp: 0, revision: rlRevision)
            }
            guard values.count == ActionLayout.count, values.allSatisfy(\.isFinite) else {
                throw AgentTrainerError.model("The brain produced an invalid prediction, so all outputs were stopped safely.")
            }
            let accepted = lock.withLock { () -> Bool in
                guard !stopped, rlOptions == nil || rlActionsEnabled, !contextTransitionPending, contextRevision == revision,
                      outputPermissionsRevision == permissionsRevision else { return false }
                predictionLatch.publish(values)
                metrics.actionInferenceMilliseconds = (CACurrentMediaTime() - began) * 1000
                headLatencies.append(metrics.actionInferenceMilliseconds)
                if headLatencies.count > 512 { headLatencies.removeFirst() }
                if CACurrentMediaTime() - lastMetricsReportTime >= 0.1 {
                    metrics.actionP95Milliseconds = Self.p95(headLatencies); metrics.encodingP95Milliseconds = Self.p95(encodingLatencies)
                }
                metrics.observationAgeMilliseconds = TemporalInputContract.seconds(snapshot.timestamp, since: vision.sourceTime) * 1000
                return true
            }
            guard accepted else { lock.withLock { actionInferencePending = false }; return }
            let executedDecision = decision
            actionQueue.async { [weak self] in
                guard let self else { return }
                defer { self.lock.withLock { self.actionInferencePending = false } }
                self.executePrediction(context: revision, permissions: permissionsRevision, rlDecision: executedDecision)
            }
        } catch {
            lock.withLock { actionInferencePending = false }
            Task { await stop(reason: error.localizedDescription) }
        }
    }

    private func executePrediction(context expectedContext: UInt64, permissions expectedPermissions: Int, rlDecision: RLDecision? = nil) {
        lock.lock()
        guard !stopped, rlOptions == nil || rlActionsEnabled, !contextTransitionPending,
              contextRevision == expectedContext, outputPermissionsRevision == expectedPermissions,
              let latched = predictionLatch.consume(), let profile else { lock.unlock(); return }
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
                suspendRLActions()
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
        let executed = injector.execute(
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
        if executed, var decision = rlDecision {
            decision.timestamp = injector.lastExecutionTimestamp
            rlOptions?.onDecision(decision)
        }
        if let snapshot { onMetrics?(snapshot) }
    }

    private static func p95(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.sorted()[min(values.count - 1, Int(Double(values.count) * 0.95))]
    }

    private func focusTargetIfNeeded(_ spec: CaptureSpec, rewardTarget: CGPoint? = nil) async -> pid_t? {
        var pid: pid_t?
        if let windowID = spec.windowID,
           let windows = try? await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false).windows {
            pid = windows.first(where: { $0.windowID == windowID })?.owningApplication?.processID
        } else if let point = rewardTarget,
                  let windows = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] {
            for window in windows {
                guard let bounds = window[kCGWindowBounds as String] as? NSDictionary,
                      let rect = CGRect(dictionaryRepresentation: bounds), rect.contains(point),
                      let owner = window[kCGWindowOwnerPID as String] as? Int32,
                      owner != ProcessInfo.processInfo.processIdentifier,
                      (window[kCGWindowLayer as String] as? Int ?? 1) == 0 else { continue }
                pid = owner; break
            }
        }
        if let pid, let app = NSRunningApplication(processIdentifier: pid) {
            await MainActor.run { _ = app.activate(options: [.activateAllWindows]) }
        }
        return pid
    }

    private static func hostTimeNanos(_ time: CMTime = CMClockGetTime(CMClockGetHostTimeClock())) -> UInt64 {
        let scaled = CMTimeConvertScale(time, timescale: 1_000_000_000, method: .default)
        guard scaled.isNumeric, scaled.value > 0 else {
            return DispatchTime.now().uptimeNanoseconds
        }
        return UInt64(scaled.value)
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

    func age(at time: Double = CACurrentMediaTime()) -> Double? {
        latest.map { _ in max(0, time - publishedAt) }
    }

    mutating func reset() {
        latest = nil
        revision = 0
        consumedRevision = 0
        publishedAt = 0
    }
}

enum RuntimeActionSemantics {
    /// Saved channels define what a brain learned; every current channel switch
    /// is also a live safety disable. Intersection semantics prevent both
    /// re-enabling an untrained head and executing a head the user turned off.
    static func effectiveChannels(saved: ActionChannels, current: ActionChannels) -> ActionChannels {
        var result = saved
        result.mouseMovement = saved.mouseMovement && current.mouseMovement
        result.buttons = saved.buttons && current.buttons
        result.scroll = saved.scroll && current.scroll
        result.keyboard = saved.keyboard && current.keyboard
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
                && !ActionLayout.allModifierKeyCodes.contains(code)
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
