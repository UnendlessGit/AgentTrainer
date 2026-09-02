@preconcurrency import CoreMedia
@preconcurrency import CoreVideo
@preconcurrency import AppKit
import Foundation
import MLX
import MLXNN
@preconcurrency import ScreenCaptureKit

struct DAggerStartupBuffer: Sendable {
    let maximumSamples: Int
    let maximumFrameBytes: Int
    private(set) var samples: [InputSample] = []
    private(set) var frames: [DAggerFrame] = []
    private(set) var frameBytes = 0
    private(set) var overflowed = false

    init(
        maximumSamples: Int = 131_072,
        maximumFrameBytes: Int = DAggerConfiguration.maximumBufferedContextBytes
    ) {
        self.maximumSamples = max(1, maximumSamples)
        self.maximumFrameBytes = max(1, maximumFrameBytes)
    }

    mutating func append(_ sample: InputSample) {
        guard samples.count < maximumSamples else {
            overflowed = true
            return
        }
        samples.append(sample)
    }

    mutating func append(_ frame: DAggerFrame) {
        let bytes = frame.packedVision.count
            + frame.executedControls.count * MemoryLayout<Float>.size
            + 64
        guard bytes <= maximumFrameBytes - frameBytes else {
            overflowed = true
            return
        }
        frames.append(frame)
        frameBytes += bytes
    }

    mutating func markOverflow() {
        overflowed = true
    }

    mutating func reset() {
        samples.removeAll(keepingCapacity: true)
        frames.removeAll(keepingCapacity: true)
        frameBytes = 0
        overflowed = false
    }
}

final class AgentRuntime: @unchecked Sendable {
    private typealias VisualizationFunction = @Sendable ([MLXArray]) -> [MLXArray]
    private enum DAggerInstallResult: Equatable {
        case installed
        case stopped
        case overflow
    }

    private enum StopAction {
        case perform(DispatchSourceTimer?)
        case waitForTeardown
        case waitForStartup
        case finished
    }

    var onState: (@Sendable (InputState) -> Void)?
    var onHumanState: (@Sendable (InputState) -> Void)?
    var onMetrics: (@Sendable (RuntimeMetrics) -> Void)?
    var onStop: (@Sendable (String?) -> Void)?
    var onPreview: (@Sendable (VisionPreviewFrame) -> Void)?
    var onVisualization: (@Sendable (CNNVisualizationFrame) -> Void)?
    var onDAggerRecordingChanged: (@Sendable (Bool) -> Void)?
    var onDAggerFinished: (@Sendable (Result<DAggerCaptureResult, Error>) -> Void)?

    private let capture = CaptureService()
    private let daggerCapture = CaptureService()
    private let preprocessor: VisionPreprocessor
    private let daggerPreprocessor: VisionPreprocessor
    private let inferenceInputBuffers: MetalArrayBufferPool
    private let injector = InputInjector()
    private let safetyMonitor = InputCaptureService()
    private let inferenceQueue = DispatchQueue(label: "AgentTrainer.Inference", qos: .userInteractive)
    private let daggerFrameQueue = DispatchQueue(label: "AgentTrainer.DAgger.Frames", qos: .userInitiated)
    private let actionQueue = DispatchQueue(label: "AgentTrainer.Actions", qos: .userInteractive)
    private let lock = NSLock()
    private var model: AgentPolicy?
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
    private var mode: FrameMode = .newest
    private var mouseMode: MouseControlMode = .absolute
    private var gameCamera = GameCameraSettings()
    private var outputPermissions = RuntimeOutputPermissions()
    private var outputPermissionsRevision = 0
    private var allowedKeyCodes: Set<UInt16> = []
    private var sourceVersionID: UUID?
    private var shiftUsesKeyboardChannel = false
    private var reinforcementActionPlan: ReinforcementActionPlan?
    private var reinforcementActionBehavior: ReinforcementActionBehavior?
    private var reinforcementRandom = ReinforcementRandomGenerator(seed: 7_431)
    private var reinforcementActionFilter = ReinforcementActionFilter()
    private var latestFrame: CVPixelBuffer?
    private var latestFrameHostNanos: UInt64 = 0
    private var lastUsableCaptureFrame: CVPixelBuffer?
    private var processing = false
    private var predictionLatch = RuntimePredictionLatch()
    private struct TemporalFrame: Sendable {
        var packed: Data
        var controls: [Float]
        var timestampNanos: UInt64
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
    private struct DAggerFrameRing: Sendable {
        private var storage: [DAggerFrame?] = []
        private var oldest = 0
        private(set) var count = 0

        mutating func reset() {
            storage.removeAll(keepingCapacity: false)
            oldest = 0
            count = 0
        }

        mutating func append(_ frame: DAggerFrame, capacity rawCapacity: Int) {
            let capacity = max(1, rawCapacity)
            if storage.count != capacity {
                let retained = frames().suffix(capacity)
                storage = [DAggerFrame?](repeating: nil, count: capacity)
                oldest = 0
                count = 0
                for frame in retained { append(frame, capacity: capacity) }
            }
            if count < capacity {
                storage[(oldest + count) % capacity] = frame
                count += 1
            } else {
                storage[oldest] = frame
                oldest = (oldest + 1) % capacity
            }
        }

        func frames() -> [DAggerFrame] {
            guard !storage.isEmpty, count > 0 else { return [] }
            return (0..<count).compactMap { storage[(oldest + $0) % storage.count] }
        }
    }
    private var daggerFrames = DAggerFrameRing()
    private var daggerRecorder: DAggerRecorder?
    private var daggerOutputPaused = false
    private var daggerInferencePaused = false
    private var daggerStarting = false
    private var daggerStartupBuffer = DAggerStartupBuffer()
    private var daggerRuntimeConfiguration: DAggerRuntimeCaptureConfiguration?
    private var daggerColorQuantization: RecordingColorQuantizationTables?
    private var recentDAggerDropTimestamps: [UInt64] = []
    private var latestDAggerSourceFrame: CVPixelBuffer?
    private var nextDAggerFrameHostNanos: UInt64 = 0
    private var pendingDAggerFrameJobs = 0
    private var daggerFrameCapturePaused = false
    private var latestContextControls = [Float](repeating: 0, count: ActionLayout.count)
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

    var isDAggerRecording: Bool { lock.withLock { daggerRecorder != nil } }

    init() throws {
        MLXMemoryLifecycle.configure()
        preprocessor = try VisionPreprocessor()
        daggerPreprocessor = try VisionPreprocessor()
        inferenceInputBuffers = try MetalArrayBufferPool(maximumCachedBytes: 32 * 1_024 * 1_024)
        injector.onState = { [weak self] state in self?.onState?(state) }
    }

    func start(profile: AIProfile, version: ModelVersionManifest, allowedKeyCodes: Set<UInt16>, captureSpec: CaptureSpec, captureRect: CGRect, mode: FrameMode, mouseMode: MouseControlMode, gameCamera: GameCameraSettings = GameCameraSettings(), outputPermissions: RuntimeOutputPermissions = RuntimeOutputPermissions(), contextSelection: [UUID: UUID] = [:], safety: AgentSafetyPolicy, daggerCaptureConfiguration: DAggerRuntimeCaptureConfiguration, previewFPS: Double = 0, visualizationSettings: CNNVisualizationSettings = CNNVisualizationSettings(), ignoredHotkeys: [HotkeyBinding] = []) async throws {
        guard AXIsProcessTrusted() else {
            let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
            AXIsProcessTrustedWithOptions(options)
            throw AgentTrainerError.permission("Accessibility permission is required before AgentTrainer can press keys or control the mouse.")
        }
        guard version.schemaVersion == ModelContract.schemaVersion,
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
        _ = try profile.training.effectiveDAgger.validated()
        let daggerCaptureConfiguration = try daggerCaptureConfiguration.validated()
        var runtimeProfile = profile
        runtimeProfile.preprocessing = version.preprocessing
        runtimeProfile.channels = RuntimeActionSemantics.effectiveChannels(saved: version.channels, current: profile.channels)
        runtimeProfile.training = version.training
        var runtimeAllowedKeyCodes = allowedKeyCodes
        var reinforcementPlan: ReinforcementActionPlan?
        var reinforcementBehavior: ReinforcementActionBehavior?
        if let reinforcementActionSpace = version.reinforcementActionSpace {
            runtimeProfile.channels = reinforcementActionSpace.effectiveChannels(
                base: runtimeProfile.channels,
                mouseMode: mouseMode
            )
            runtimeProfile.restrictions = reinforcementActionSpace.applying(
                to: profile.effectiveRestrictions
            )
            runtimeAllowedKeyCodes.formIntersection(
                reinforcementActionSpace.allowedKeyCodes
            )
            var runtimeConfiguration = ReinforcementLearningConfiguration()
            runtimeConfiguration.actionSpace = reinforcementActionSpace
            let behavior = try (
                version.reinforcementActionBehavior
                    ?? ReinforcementActionBehavior()
            ).validated()
            runtimeConfiguration.actionBehavior = behavior
            reinforcementPlan = try ReinforcementActionPlan.make(
                profile: runtimeProfile,
                configuration: runtimeConfiguration,
                mouseMode: mouseMode
            )
            reinforcementBehavior = behavior
        }
        _ = try runtimeProfile.preprocessing.validated()
        _ = try runtimeProfile.training.effectiveTemporalVision.validated(
            current: runtimeProfile.preprocessing
        )
        let contextSchema = try version.effectiveContextSchema.validated()
        let initialContextVector = contextSchema.vector(selection: contextSelection)
        let startRevisions = lock.withLock { (outputPermissionsRevision, visualizationSettingsRevision) }
        let model = AgentPolicy(profile: runtimeProfile, contextSchema: contextSchema)
        let versionDirectory = await WorkspaceStore.shared.versionDirectory(profileID: profile.id, versionID: version.id)
        try model.loadWeights(from: versionDirectory.appendingPathComponent(version.weightsFile))
        model.train(false)
        let temporal = runtimeProfile.training.effectiveTemporalVision
        let hasTemporalMemory = temporal.pastFrameCount > 0
        let hasContext = !contextSchema.isEmpty
        let usesReinforcementSemantics = reinforcementPlan != nil
        let contextInputIndex = hasTemporalMemory ? 3 : 1
        let pastSpec = temporal.pastFrameSpec(from: runtimeProfile.preprocessing)
        let predictionFunction: VisualizationFunction = compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
            let currentImages = VisionPreprocessor.mlxTensor(inputs[0], spec: runtimeProfile.preprocessing)
            let contextValues = hasContext ? inputs[contextInputIndex] : nil
            if hasTemporalMemory {
                let logits = model(
                    currentImages: currentImages,
                    pastImages: VisionPreprocessor.mlxPastFrameTensor(inputs[1], spec: pastSpec),
                    pastControls: inputs[2],
                    contextValues: contextValues
                )
                return [usesReinforcementSemantics
                    ? logits : model.activatedPredictions(logits: logits)]
            }
            let logits = model.logits(
                temporalFeatures: model.currentOnlyTemporalFeatures(
                    currentImages: currentImages
                ),
                contextValues: contextValues
            )
            return [usesReinforcementSemantics
                ? logits : model.activatedPredictions(logits: logits)]
        }
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
                        pastControls: inputs[2]
                    )
                    : model.currentOnlyTemporalFeatures(currentVisualFeatures: layers.last!)
                let logits = model.logits(
                    temporalFeatures: temporalFeatures,
                    contextValues: contextValues
                )
                let map = model.sampledForVisualization(layers[selectedLayer]).mean(axis: -1, keepDims: true)
                return [usesReinforcementSemantics
                    ? logits : model.activatedPredictions(logits: logits), map]
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
                    pastControls: inputs[2]
                )
                : model.currentOnlyTemporalFeatures(currentVisualFeatures: layers.last!)
            let logits = model.logits(
                temporalFeatures: temporalFeatures,
                contextValues: contextValues
            )
            return [usesReinforcementSemantics
                ? logits : model.activatedPredictions(logits: logits), model.strongestChannelsForVisualization(layers.last!)]
        }
        let saliencyVisualizationFunction = compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
            let currentImages = VisionPreprocessor.mlxTensor(inputs[0], spec: runtimeProfile.preprocessing)
            let layers = model.visualActivations(images: currentImages)
            let contextValues = hasContext ? inputs[contextInputIndex] : nil
            let temporalFeatures = hasTemporalMemory
                ? model.temporalFeatures(
                    currentVisualFeatures: layers.last!,
                    pastImages: VisionPreprocessor.mlxPastFrameTensor(inputs[1], spec: pastSpec),
                    pastControls: inputs[2]
                )
                : model.currentOnlyTemporalFeatures(currentVisualFeatures: layers.last!)
            let logits = model.logits(
                temporalFeatures: temporalFeatures,
                contextValues: contextValues
            )
            // Keep the exact final tensor on GPU for the post-CNN gradient.
            // This graph intentionally omits the channel ranking used by the
            // separate feature-grid view.
            return [usesReinforcementSemantics
                ? logits : model.activatedPredictions(logits: logits), layers.last!]
        }
        let saliencyGradient = grad({ (inputs: [MLXArray]) -> MLXArray in
            if hasTemporalMemory {
                let pastImages = VisionPreprocessor.mlxPastFrameTensor(inputs[1], spec: pastSpec)
                let contextValues = hasContext ? inputs[3] : nil
                let selector = inputs[hasContext ? 4 : 3]
                let logits = model.logits(
                    currentVisualFeatures: inputs[0],
                    pastImages: pastImages,
                    pastControls: inputs[2],
                    contextValues: contextValues
                )
                return (logits * selector.asType(model.dtype)).sum()
            }
            let contextValues = hasContext ? inputs[1] : nil
            let selector = inputs[hasContext ? 2 : 1]
            let logits = model.logits(
                temporalFeatures: model.currentOnlyTemporalFeatures(currentVisualFeatures: inputs[0]),
                contextValues: contextValues
            )
            return (logits * selector.asType(model.dtype)).sum()
        }, argumentNumbers: [0])
        let initialReinforcementCursor = Self.normalizedCursor(in: captureRect)
        let accepted = lock.withLock { () -> Bool in
            guard launchRevision == launchToken, starting, stopped else { return false }
            self.model = model
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
            self.sourceVersionID = version.id
            self.daggerRuntimeConfiguration = daggerCaptureConfiguration
            self.daggerColorQuantization = RecordingColorQuantizationTables(
                bits: daggerCaptureConfiguration.preprocessing.bitDepth
            )
            self.allowedKeyCodes = runtimeAllowedKeyCodes
            self.shiftUsesKeyboardChannel = (version.trainingDataSchema ?? 0) >= 7
            self.reinforcementActionPlan = reinforcementPlan
            self.reinforcementActionBehavior = reinforcementBehavior
            self.reinforcementRandom = ReinforcementRandomGenerator(
                seed: Self.reinforcementRuntimeSeed(version.id)
                    ^ DispatchTime.now().uptimeNanoseconds
            )
            self.reinforcementActionFilter.reset(
                absoluteMousePosition: initialReinforcementCursor
            )
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
            latestFrame = nil; latestFrameHostNanos = 0; lastUsableCaptureFrame = nil
            predictionLatch.reset(); temporalFrames.reset(); daggerFrames.reset()
            daggerRecorder = nil; daggerOutputPaused = false; daggerInferencePaused = false
            daggerStarting = false; daggerStartupBuffer.reset()
            recentDAggerDropTimestamps.removeAll(keepingCapacity: false)
            latestDAggerSourceFrame = nil; nextDAggerFrameHostNanos = 0; pendingDAggerFrameJobs = 0; daggerFrameCapturePaused = false
            latestContextControls = [Float](repeating: 0, count: ActionLayout.count)
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
        do {
            guard lock.withLock({ !stopped }) else { throw CancellationError() }
            safetyMonitor.ignoredHotkeys = ignoredHotkeys
            safetyMonitor.onSample = { [weak self] sample in
                guard let self else { return }
                let panic = sample.kind == .key && sample.isDown && sample.keyCode == self.safety.panicKeyCode && (sample.modifiers & self.safety.panicModifiers) == self.safety.panicModifiers
                if !panic {
                    let admittedToCorrection = self.lock.withLock { () -> Bool in
                        if self.daggerStarting {
                            self.daggerStartupBuffer.append(sample)
                            return true
                        }
                        guard let recorder = self.daggerRecorder else { return false }
                        // Input callback admission shares the runtime lock with
                        // recorder detachment. stopDAggerRecording therefore
                        // cannot close the recorder between observing it here
                        // and appending this already-admitted expert event.
                        recorder.append(sample)
                        return true
                    }
                    if admittedToCorrection { return }
                }
                guard panic || self.safety.stopOnHumanInput else { return }
                Task { await self.stop(reason: panic ? "Panic stop" : "Stopped on human input") }
            }
            safetyMonitor.onState = { [weak self] state, _ in
                guard let self else { return }
                let isCorrection = self.lock.withLock {
                    self.daggerStarting || self.daggerRecorder != nil
                }
                if isCorrection { self.onHumanState?(state) }
            }
            try safetyMonitor.start()
            let targetPID = await focusTargetIfNeeded(captureSpec)
            guard lock.withLock({ !stopped }) else { throw CancellationError() }
            lock.withLock { self.targetPID = targetPID }
            var liveCaptureSpec = captureSpec
            let needsDedicatedDAggerCapture = daggerCaptureConfiguration.capture.showsCursor
                != (version.trainingShowsCursor ?? captureSpec.showsCursor)
            liveCaptureSpec.requestedFPS = needsDedicatedDAggerCapture
                ? runtimeProfile.training.perceptionFPS
                : max(runtimeProfile.training.perceptionFPS, daggerCaptureConfiguration.capture.requestedFPS)
            liveCaptureSpec.showsCursor = version.trainingShowsCursor ?? captureSpec.showsCursor
            let reportsInferenceThrottle = liveCaptureSpec.requestedFPS
                <= runtimeProfile.training.perceptionFPS + 0.001
            try await capture.start(spec: liveCaptureSpec, queueDepth: mode == .newest ? 3 : 8, onFrame: { [weak self] buffer, pts in
                self?.receive(
                    buffer,
                    timestamp: pts,
                    suppliesDAggerFrames: !needsDedicatedDAggerCapture,
                    reportsInferenceThrottle: reportsInferenceThrottle
                )
            }, onIdle: { [weak self] pts in
                self?.receiveIdle(timestamp: pts, suppliesDAggerFrames: !needsDedicatedDAggerCapture)
            }, onUnexpectedStop: { [weak self] error in
                Task { await self?.stop(reason: "Capture stopped: \(error.localizedDescription)") }
            })
            if needsDedicatedDAggerCapture {
                var recordingCapture = daggerCaptureConfiguration.capture
                recordingCapture.requestedFPS = daggerCaptureConfiguration.capture.requestedFPS
                try await daggerCapture.start(
                    spec: recordingCapture,
                    queueDepth: 5,
                    onFrame: { [weak self] buffer, pts in
                        self?.receiveDAggerSource(buffer, timestampNanos: Self.hostTimeNanos(pts))
                    },
                    onIdle: { [weak self] pts in
                        self?.receiveDAggerIdle(timestampNanos: Self.hostTimeNanos(pts))
                    },
                    onUnexpectedStop: { [weak self] error in
                        Task { await self?.stop(reason: "DAgger capture stopped: \(error.localizedDescription)") }
                    }
                )
            }
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
            _ = try? await daggerCapture.stop()
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
            predictionLatch.reset()
            reinforcementActionFilter.reset(
                absoluteMousePosition: Self.normalizedCursor(in: captureRect)
            )
            latestContextControls = [Float](repeating: 0, count: ActionLayout.count)
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

    func startDAggerRecording(
        request: DAggerCaptureRequest,
        recordingShortcut: HotkeyBinding
    ) async throws {
        let now = Self.hostTimeNanos()
        let context = try lock.withLock { () throws -> (frames: [DAggerFrame], drops: Int) in
            guard !stopped, daggerRecorder == nil, !daggerStarting, let profile,
                  let recording = daggerRuntimeConfiguration,
                  sourceVersionID == request.versionID,
                  profile.id == request.profileID,
                  recording.preprocessing == request.preprocessing,
                  recording.capture.requestedFPS == request.perceptionFPS,
                  recording.leadUpSeconds == request.leadUpSeconds else {
                throw AgentTrainerError.model("DAgger can start only from this Run's frozen Record settings, and only one correction can be recorded at a time.")
            }
            let sizing = DAggerContextSizing.recordingFrameCapacity(
                preprocessing: recording.preprocessing,
                framesPerSecond: recording.capture.requestedFPS,
                leadUpSeconds: recording.leadUpSeconds
            )
            guard sizing.maximum >= sizing.requested else {
                throw AgentTrainerError.invalidConfiguration("This DAgger lead-up exceeds the 512 MB runtime buffer. Reduce Record lead-up, FPS, or resolution.")
            }
            let nanos = UInt64((recording.leadUpSeconds * 1e9).rounded())
            let cutoff = now > nanos ? now - nanos : 0
            let retained = daggerFrames.frames().filter { $0.timestampNanos >= cutoff }
            let dropped = recentDAggerDropTimestamps.count { $0 >= cutoff }
            guard !retained.isEmpty else {
                throw AgentTrainerError.model("Wait for the first complete Record-quality lead-up frame before starting a DAgger correction.")
            }
            let retainedBytes = retained.reduce(into: 0) { total, frame in
                let bytes = frame.packedVision.count
                    + frame.executedControls.count * MemoryLayout<Float>.size
                    + 64
                total = total.addingReportingOverflow(bytes).overflow
                    ? DAggerConfiguration.maximumBufferedContextBytes
                    : total + bytes
            }
            // This is the explicit trigger boundary. Stop admitting AI action
            // ticks before constructing encoders or copying pre-roll so the
            // first expert input can never be mistaken for ordinary takeover.
            daggerStarting = true
            daggerOutputPaused = true
            daggerInferencePaused = true
            latestFrame = nil
            latestFrameHostNanos = 0
            // Snapshot and recorder closures now own the lead-up payload. Stop
            // retaining the same timeline in the rolling ring while correction
            // frames arrive, and bound startup additions by the remaining part
            // of the one shared 512 MB visual-payload budget.
            daggerFrames.reset()
            recentDAggerDropTimestamps.removeAll(keepingCapacity: true)
            daggerStartupBuffer = DAggerStartupBuffer(
                maximumFrameBytes: max(
                    1,
                    DAggerConfiguration.maximumBufferedContextBytes - retainedBytes
                )
            )
            return (retained, dropped)
        }
        safetyMonitor.excludedKeyCodes = request.excludedKeyCodes
        await drain(queue: actionQueue)
        await drain(queue: inferenceQueue)
        injector.disableAndReleaseAll()
        do {
            let pointer = CGEvent(source: nil)?.location
                ?? CGPoint(x: request.captureRect.midX, y: request.captureRect.midY)
            let seed = safetyMonitor.recordingSeedSamples(
                timestampNanos: now,
                pointer: pointer,
                excluding: recordingShortcut
            )
            let recorder = try DAggerRecorder(
                request: request,
                correctionStartHostNanos: now,
                preRoll: context.frames,
                initialDroppedFrameCount: context.drops
            )
            for sample in seed {
                recorder.append(sample)
            }
            let installResult = lock.withLock { () -> DAggerInstallResult in
                guard !stopped, daggerStarting, daggerRecorder == nil else { return .stopped }
                guard !daggerStartupBuffer.overflowed else { return .overflow }
                for sample in daggerStartupBuffer.samples {
                    recorder.append(sample)
                }
                for frame in daggerStartupBuffer.frames { recorder.append(frame) }
                daggerStartupBuffer.reset()
                daggerRecorder = recorder
                daggerStarting = false
                return .installed
            }
            guard installResult == .installed else {
                _ = try? await recorder.finish(endHostNanos: Self.hostTimeNanos())
                if installResult == .overflow {
                    throw AgentTrainerError.capture("DAgger startup could not keep up with the initial correction stream inside its bounded buffer. The partial package was discarded; try again with a shorter Record lead-up, lower Record FPS, or smaller recording resolution.")
                }
                throw CancellationError()
            }
            onDAggerRecordingChanged?(true)
        } catch {
            lock.withLock {
                daggerStarting = false
                daggerStartupBuffer.reset()
                guard !stopped, daggerRecorder == nil else { return }
                daggerOutputPaused = false
                daggerInferencePaused = false
                temporalFrames.reset()
                predictionLatch.reset()
                reinforcementActionFilter.reset(
                    absoluteMousePosition: Self.normalizedCursor(in: captureRect)
                )
                injector.enable(outputPermissions: outputPermissions)
            }
            safetyMonitor.excludedKeyCodes = []
            throw error
        }
    }

    func stopDAggerRecording() async {
        let shouldStop = lock.withLock { () -> Bool in
            guard daggerRecorder != nil else { return false }
            daggerFrameCapturePaused = true
            return true
        }
        guard shouldStop else { return }
        await drain(queue: daggerFrameQueue)
        let recorder = lock.withLock { () -> DAggerRecorder? in
            let recorder = daggerRecorder
            daggerRecorder = nil
            guard !stopped else { return recorder }
            daggerOutputPaused = false
            daggerInferencePaused = false
            daggerFrameCapturePaused = false
            temporalFrames.reset()
            predictionLatch.reset()
            reinforcementActionFilter.reset(
                absoluteMousePosition: Self.normalizedCursor(in: captureRect)
            )
            daggerFrames.reset()
            recentDAggerDropTimestamps.removeAll(keepingCapacity: true)
            nextDAggerFrameHostNanos = 0
            injector.enable(outputPermissions: outputPermissions)
            return recorder
        }
        guard let recorder else { return }
        safetyMonitor.excludedKeyCodes = []
        onDAggerRecordingChanged?(false)
        let result: Result<DAggerCaptureResult, Error>
        do {
            result = .success(try await recorder.finish(endHostNanos: Self.hostTimeNanos()))
        } catch {
            result = .failure(error)
        }
        onDAggerFinished?(result)
    }

    func stop(reason: String? = nil) async {
        let transition = lock.withLock { () -> (StopAction, DAggerRecorder?) in
            launchRevision &+= 1
            if teardownInProgress { return (.waitForTeardown, nil) }
            guard !stopped else { return (starting ? .waitForStartup : .finished, nil) }
            stopped = true
            teardownInProgress = true
            let timer = actionTimer; actionTimer = nil
            let recorder = daggerRecorder
            daggerRecorder = nil; daggerOutputPaused = false; daggerInferencePaused = false; daggerFrameCapturePaused = true
            daggerStarting = false; daggerStartupBuffer.reset()
            latestFrame = nil; latestFrameHostNanos = 0; lastUsableCaptureFrame = nil
            temporalFrames.reset(); predictionLatch.reset(); reinforcementActionFilter.reset(); targetPID = nil
            return (.perform(timer), recorder)
        }
        let action = transition.0
        let stoppedDAggerRecorder = transition.1
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
        safetyMonitor.onState = nil
        safetyMonitor.excludedKeyCodes = []
        _ = try? await capture.stop()
        _ = try? await daggerCapture.stop()
        await drain(queue: inferenceQueue)
        await drain(queue: daggerFrameQueue)
        if let stoppedDAggerRecorder {
            do {
                let result = try await stoppedDAggerRecorder.finish(endHostNanos: Self.hostTimeNanos())
                onDAggerFinished?(.success(result))
            } catch {
                onDAggerFinished?(.failure(error))
            }
        }
        lock.withLock {
            predictionFunction = nil; activationVisualizationFunctions.removeAll(keepingCapacity: false); channelVisualizationFunction = nil; saliencyVisualizationFunction = nil; saliencyGradientFunction = nil; model = nil; profile = nil; contextSchema = .empty; contextVector.removeAll(keepingCapacity: false); contextRevision &+= 1; contextTransitionPending = false; sourceVersionID = nil; daggerRuntimeConfiguration = nil; daggerColorQuantization = nil; recentDAggerDropTimestamps.removeAll(keepingCapacity: false); allowedKeyCodes.removeAll(keepingCapacity: false); shiftUsesKeyboardChannel = false; reinforcementActionPlan = nil; reinforcementActionBehavior = nil
            latestFrame = nil; latestFrameHostNanos = 0; lastUsableCaptureFrame = nil; temporalFrames.reset(); daggerFrames.reset(); predictionLatch.reset(); reinforcementActionFilter.reset(); processing = false
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

    private func receive(
        _ buffer: CVPixelBuffer,
        timestamp: CMTime,
        suppliesDAggerFrames: Bool,
        reportsInferenceThrottle: Bool
    ) {
        let timestampNanos = Self.hostTimeNanos(timestamp)
        if suppliesDAggerFrames {
            receiveDAggerSource(buffer, timestampNanos: timestampNanos)
        }
        schedule(
            buffer,
            timestampNanos: timestampNanos,
            countThrottledAsDropped: reportsInferenceThrottle
        )
    }

    private func receiveIdle(timestamp: CMTime, suppliesDAggerFrames: Bool) {
        let timestampNanos = Self.hostTimeNanos(timestamp)
        let frame = lock.withLock { stopped ? nil : lastUsableCaptureFrame }
        if let frame {
            if suppliesDAggerFrames {
                receiveDAggerSource(frame, timestampNanos: timestampNanos)
            }
            schedule(frame, timestampNanos: timestampNanos, countThrottledAsDropped: false)
        }
    }

    private func receiveDAggerIdle(timestampNanos: UInt64) {
        let frame = lock.withLock { stopped ? nil : latestDAggerSourceFrame }
        if let frame { receiveDAggerSource(frame, timestampNanos: timestampNanos) }
    }

    private func receiveDAggerSource(
        _ buffer: CVPixelBuffer,
        timestampNanos: UInt64
    ) {
        let retained = SendablePixelBuffer(buffer)
        lock.lock()
        guard !stopped, !daggerFrameCapturePaused,
              let configuration = daggerRuntimeConfiguration,
              let quantization = daggerColorQuantization else {
            lock.unlock()
            return
        }
        latestDAggerSourceFrame = buffer
        let interval = UInt64((1e9 / max(0.0001, configuration.capture.requestedFPS)).rounded())
        guard timestampNanos >= nextDAggerFrameHostNanos else {
            lock.unlock()
            return
        }
        nextDAggerFrameHostNanos = timestampNanos &+ interval
        let capacity = DAggerContextSizing.recordingFrameCapacity(
            preprocessing: configuration.preprocessing,
            framesPerSecond: configuration.capture.requestedFPS,
            leadUpSeconds: configuration.leadUpSeconds
        )
        // Bounded preprocessing protects inference latency if an extreme
        // Record resolution is slower than its requested frame rate. Drops at
        // the trigger boundary fail startup; active drops enter quality gates;
        // recent pre-trigger drops remain attached to the eventual package.
        guard pendingDAggerFrameJobs < 4 else {
            if daggerStarting {
                daggerStartupBuffer.markOverflow()
            } else if let daggerRecorder {
                daggerRecorder.recordDroppedVisualFrame()
            } else {
                recentDAggerDropTimestamps.append(timestampNanos)
                if recentDAggerDropTimestamps.count > max(1, capacity.requested) {
                    recentDAggerDropTimestamps.removeFirst(
                        recentDAggerDropTimestamps.count - max(1, capacity.requested)
                    )
                }
            }
            lock.unlock()
            return
        }
        pendingDAggerFrameJobs += 1
        let work = (
            target: configuration.preprocessing,
            capacity: max(1, min(capacity.requested, capacity.maximum)),
            quantization: quantization
        )
        // Queue insertion is ordered by the same lock that admits work and that
        // stopDAggerRecording uses before inserting its drain marker. A marker
        // can therefore never overtake a frame already counted as pending.
        daggerFrameQueue.async { [weak self] in
            guard let self else { return }
            do {
                var sourceSpec = work.target
                sourceSpec.bitDepth = 8
                var packed = try self.daggerPreprocessor.process(
                    retained.value,
                    spec: sourceSpec
                )
                if work.target.bitDepth < 8 {
                    try work.quantization.quantizePacked(&packed, spec: work.target)
                }
                let delivery = self.lock.withLock { () -> (DAggerRecorder?, DAggerFrame)? in
                    self.pendingDAggerFrameJobs = max(0, self.pendingDAggerFrameJobs - 1)
                    guard !self.stopped else { return nil }
                    let frame = DAggerFrame(
                        timestampNanos: timestampNanos,
                        packedVision: packed,
                        preprocessing: work.target,
                        executedControls: self.latestContextControls
                    )
                    if self.daggerStarting {
                        self.daggerStartupBuffer.append(frame)
                    } else if self.daggerRecorder == nil {
                        self.daggerFrames.append(frame, capacity: work.capacity)
                    }
                    return (self.daggerRecorder, frame)
                }
                if let delivery { delivery.0?.append(delivery.1) }
            } catch {
                self.lock.withLock {
                    self.pendingDAggerFrameJobs = max(0, self.pendingDAggerFrameJobs - 1)
                }
                Task { await self.stop(reason: "DAgger recording vision failed: \(error.localizedDescription)") }
            }
        }
        lock.unlock()
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
        guard !daggerInferencePaused else { lock.unlock(); return }
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
        while true {
            lock.lock()
            guard let frame = latestFrame, !stopped else { processing = false; lock.unlock(); return }
            let timestampNanos = latestFrameHostNanos
            latestFrame = nil
            latestFrameHostNanos = 0
            lock.unlock()
            infer(frame, timestampNanos: timestampNanos)
            lock.lock()
            if latestFrame == nil { processing = false; lock.unlock(); return }
            lock.unlock()
        }
    }

    private func infer(_ buffer: CVPixelBuffer, timestampNanos: UInt64) {
        lock.lock()
        guard !stopped, let predictionFunction, let model, let profile else { lock.unlock(); return }
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
        let contextVector = self.contextVector
        let inferenceContextRevision = contextRevision
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
            let selectedFrames: [TemporalFrame] = selectedPriorFrames.map { frame in
                frame ?? TemporalFrame(
                    packed: packedPastFrame,
                    controls: zeroControls,
                    timestampNanos: timestampNanos
                )
            }
            var descriptors: [MetalArrayBufferPool.Descriptor] = [
                .init([1, profile.preprocessing.sampleByteCount], dtype: .uint8)
            ]
            if hasTemporalMemory {
                descriptors.append(.init([1, temporal.pastFrameCount, pastSpec.sampleByteCount], dtype: .uint8))
                descriptors.append(.init([1, temporal.pastFrameCount, ActionLayout.count], dtype: .float32))
            }
            let contextInputIndex: Int? = contextVector.isEmpty ? nil : descriptors.count
            if !contextVector.isEmpty {
                descriptors.append(.init([1, contextVector.count], dtype: .float32))
            }
            let inferenceInputs = try inferenceInputBuffers.makeArrays(descriptors) { destinations in
                packed.withUnsafeBytes { source in
                    destinations[0].copyMemory(from: source)
                }
                if hasTemporalMemory {
                    let controlRowBytes = ActionLayout.count * MemoryLayout<Float>.size
                    for (frame, selected) in selectedFrames.enumerated() {
                        selected.packed.withUnsafeBytes { source in
                            UnsafeMutableRawBufferPointer(
                                rebasing: destinations[1][frame * pastSpec.sampleByteCount..<(frame + 1) * pastSpec.sampleByteCount]
                            ).copyMemory(from: source)
                        }
                        selected.controls.withUnsafeBytes { source in
                            UnsafeMutableRawBufferPointer(
                                rebasing: destinations[2][frame * controlRowBytes..<(frame + 1) * controlRowBytes]
                            ).copyMemory(from: source)
                        }
                    }
                }
                if let contextInputIndex {
                    contextVector.withUnsafeBytes {
                        destinations[contextInputIndex].copyMemory(from: $0)
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
                    guard forward.count >= 2 else { return predictionFunction(inferenceInputs) }
                    var gradientInputs = hasTemporalMemory
                        ? [forward[1], inferenceInputs[1], inferenceInputs[2]]
                        : [forward[1]]
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
            guard let output = result.first else { throw AgentTrainerError.model("The inference graph returned no prediction.") }
            let visualization = visualizationDue ? Self.visualizationFrame(profile: profile, settings: settings, outputs: result) : nil
            if let visualization {
                MLX.eval([output] + visualization.arrays)
            } else {
                MLX.eval(output)
            }
            let values = output.asArray(Float.self)
            guard values.count >= ActionLayout.count, values.prefix(ActionLayout.count).allSatisfy(\.isFinite) else {
                throw AgentTrainerError.model("The brain produced an invalid prediction, so all outputs were stopped safely.")
            }
            let sampledValues: [Float]
            if let reinforcementActionPlan, let reinforcementActionBehavior {
                var random = reinforcementRandom
                sampledValues = try reinforcementActionPlan.sample(
                    logits: values,
                    behavior: reinforcementActionBehavior,
                    includesContinuousExploration: false,
                    random: &random
                ).executedActions
                reinforcementRandom = random
            } else {
                sampledValues = values
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
            lock.lock()
            guard !stopped, !daggerInferencePaused, !contextTransitionPending,
                  contextRevision == inferenceContextRevision else {
                lock.unlock()
                return
            }
            let executablePrediction: [Float]
            if let reinforcementActionPlan, let reinforcementActionBehavior {
                executablePrediction = reinforcementActionFilter.apply(
                    to: sampledValues,
                    plan: reinforcementActionPlan,
                    behavior: reinforcementActionBehavior
                )
            } else {
                executablePrediction = sampledValues
            }
            let frameControls = RuntimeActionSemantics.temporalControlValues(
                executablePrediction,
                channels: profile.channels,
                restrictions: profile.effectiveRestrictions,
                allowedKeyCodes: allowedKeyCodes,
                outputPermissions: outputPermissions,
                shiftUsesKeyboardChannel: shiftUsesKeyboardChannel
            )
            latestContextControls = frameControls
            let frameTimestamp = timestampNanos > 0 ? timestampNanos : Self.hostTimeNanos()
            if hasTemporalMemory {
                let retainedFrameCount = temporal.pastFrameCount * temporal.frameSpacing + 1
                temporalFrames.append(
                    TemporalFrame(
                        packed: packedPastFrame,
                        controls: frameControls,
                        timestampNanos: frameTimestamp
                    ),
                    capacity: retainedFrameCount
                )
            }
            predictionLatch.publish(
                reinforcementActionPlan == nil
                    ? executablePrediction : frameControls
            )
            metrics.frameCount += 1
            let elapsed = max(0.001, CACurrentMediaTime() - startedAt)
            metrics.perceptionFPS = Double(metrics.frameCount) / elapsed
            metrics.latencyMilliseconds = (CACurrentMediaTime() - began) * 1_000
            let now = CACurrentMediaTime()
            let snapshot: RuntimeMetrics? = now - lastMetricsReportTime >= 0.1 ? metrics : nil
            if snapshot != nil { lastMetricsReportTime = now }
            lock.unlock()
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
        guard !stopped, !daggerOutputPaused,
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

    private static func hostTimeNanos(_ time: CMTime = CMClockGetTime(CMClockGetHostTimeClock())) -> UInt64 {
        let scaled = CMTimeConvertScale(time, timescale: 1_000_000_000, method: .default)
        guard scaled.isNumeric, scaled.value > 0 else {
            return DispatchTime.now().uptimeNanoseconds
        }
        return UInt64(scaled.value)
    }

    private static func reinforcementRuntimeSeed(_ versionID: UUID) -> UInt64 {
        versionID.uuidString.utf8.reduce(0xcbf2_9ce4_8422_2325) {
            ($0 ^ UInt64($1)) &* 0x0000_0100_0000_01b3
        }
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
