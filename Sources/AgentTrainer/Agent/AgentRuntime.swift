@preconcurrency import CoreMedia
@preconcurrency import CoreVideo
import AppKit
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
    var onMetrics: (@Sendable (RuntimeMetrics) -> Void)?
    var onStop: (@Sendable (String?) -> Void)?
    var onPreview: (@Sendable (VisionPreviewFrame) -> Void)?
    var onVisualization: (@Sendable (CNNVisualizationFrame) -> Void)?
    var onDAggerRecordingChanged: (@Sendable (Bool) -> Void)?
    var onDAggerFinished: (@Sendable (Result<DAggerCaptureResult, Error>) -> Void)?

    private let capture = CaptureService()
    private let preprocessor: VisionPreprocessor
    private let inferenceInputBuffers: MetalArrayBufferPool
    private let injector = InputInjector()
    private let safetyMonitor = InputCaptureService()
    private let inferenceQueue = DispatchQueue(label: "AgentTrainer.Inference", qos: .userInteractive)
    private let actionQueue = DispatchQueue(label: "AgentTrainer.Actions", qos: .userInteractive)
    private let lock = NSLock()
    private var model: AgentPolicy?
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
    private var sourceVersionID: UUID?
    private var shiftUsesKeyboardChannel = false
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
    private var daggerHumanActions: ActionAccumulator?
    private var daggerOutputPaused = false
    private var daggerStarting = false
    private var daggerStartupBuffer = DAggerStartupBuffer()
    private var daggerConfiguration = DAggerConfiguration()
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
        inferenceInputBuffers = try MetalArrayBufferPool(maximumCachedBytes: 32 * 1_024 * 1_024)
        injector.onState = { [weak self] state in self?.onState?(state) }
    }

    func start(profile: AIProfile, version: ModelVersionManifest, allowedKeyCodes: Set<UInt16>, captureSpec: CaptureSpec, captureRect: CGRect, mode: FrameMode, mouseMode: MouseControlMode, gameCamera: GameCameraSettings = GameCameraSettings(), outputPermissions: RuntimeOutputPermissions = RuntimeOutputPermissions(), safety: AgentSafetyPolicy, previewFPS: Double = 0, visualizationSettings: CNNVisualizationSettings = CNNVisualizationSettings(), ignoredHotkeys: [HotkeyBinding] = []) async throws {
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
        let liveDAggerConfiguration = try profile.training.effectiveDAgger.validated()
        var runtimeProfile = profile
        runtimeProfile.preprocessing = version.preprocessing
        runtimeProfile.channels = RuntimeActionSemantics.effectiveChannels(saved: version.channels, current: profile.channels)
        runtimeProfile.training = version.training
        _ = try runtimeProfile.preprocessing.validated()
        _ = try runtimeProfile.training.effectiveTemporalVision.validated(
            current: runtimeProfile.preprocessing
        )
        let startRevisions = lock.withLock { (outputPermissionsRevision, visualizationSettingsRevision) }
        let model = AgentPolicy(profile: runtimeProfile)
        let versionDirectory = await WorkspaceStore.shared.versionDirectory(profileID: profile.id, versionID: version.id)
        try model.loadWeights(from: versionDirectory.appendingPathComponent(version.weightsFile))
        model.train(false)
        let temporal = runtimeProfile.training.effectiveTemporalVision
        let hasTemporalMemory = temporal.pastFrameCount > 0
        let pastSpec = temporal.pastFrameSpec(from: runtimeProfile.preprocessing)
        let predictionFunction: VisualizationFunction = compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
            let currentImages = VisionPreprocessor.mlxTensor(inputs[0], spec: runtimeProfile.preprocessing)
            if hasTemporalMemory {
                return [model.predictions(
                    currentImages: currentImages,
                    pastImages: VisionPreprocessor.mlxPastFrameTensor(inputs[1], spec: pastSpec),
                    pastControls: inputs[2]
                )]
            }
            return [model.currentOnlyPredictions(currentImages: currentImages)]
        }
        // Diagnostic graphs are lazy: creating these closures does not execute
        // or materialize an extra tensor. The selected graph first compiles only
        // when its view is enabled and reaches its independently capped rate.
        let activationVisualizationFunctions: [VisualizationFunction] = (0..<max(1, model.convolutions.count)).map { selectedLayer in
            compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
                let currentImages = VisionPreprocessor.mlxTensor(inputs[0], spec: runtimeProfile.preprocessing)
                let layers = model.visualActivations(images: currentImages)
                let temporalFeatures = hasTemporalMemory
                    ? model.temporalFeatures(
                        currentVisualFeatures: layers.last!,
                        pastImages: VisionPreprocessor.mlxPastFrameTensor(inputs[1], spec: pastSpec),
                        pastControls: inputs[2]
                    )
                    : model.currentOnlyTemporalFeatures(currentVisualFeatures: layers.last!)
                let logits = model.logits(temporalFeatures: temporalFeatures)
                let map = model.sampledForVisualization(layers[selectedLayer]).mean(axis: -1, keepDims: true)
                return [model.activatedPredictions(logits: logits), map]
            }
        }
        let channelVisualizationFunction = compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
            let currentImages = VisionPreprocessor.mlxTensor(inputs[0], spec: runtimeProfile.preprocessing)
            let layers = model.visualActivations(images: currentImages)
            let temporalFeatures = hasTemporalMemory
                ? model.temporalFeatures(
                    currentVisualFeatures: layers.last!,
                    pastImages: VisionPreprocessor.mlxPastFrameTensor(inputs[1], spec: pastSpec),
                    pastControls: inputs[2]
                )
                : model.currentOnlyTemporalFeatures(currentVisualFeatures: layers.last!)
            let logits = model.logits(temporalFeatures: temporalFeatures)
            return [model.activatedPredictions(logits: logits), model.strongestChannelsForVisualization(layers.last!)]
        }
        let saliencyVisualizationFunction = compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
            let currentImages = VisionPreprocessor.mlxTensor(inputs[0], spec: runtimeProfile.preprocessing)
            let layers = model.visualActivations(images: currentImages)
            let temporalFeatures = hasTemporalMemory
                ? model.temporalFeatures(
                    currentVisualFeatures: layers.last!,
                    pastImages: VisionPreprocessor.mlxPastFrameTensor(inputs[1], spec: pastSpec),
                    pastControls: inputs[2]
                )
                : model.currentOnlyTemporalFeatures(currentVisualFeatures: layers.last!)
            let logits = model.logits(temporalFeatures: temporalFeatures)
            // Keep the exact final tensor on GPU for the post-CNN gradient.
            // This graph intentionally omits the channel ranking used by the
            // separate feature-grid view.
            return [model.activatedPredictions(logits: logits), layers.last!]
        }
        let saliencyGradient = grad({ (inputs: [MLXArray]) -> MLXArray in
            if hasTemporalMemory {
                let pastImages = VisionPreprocessor.mlxPastFrameTensor(inputs[1], spec: pastSpec)
                let logits = model.logits(
                    currentVisualFeatures: inputs[0],
                    pastImages: pastImages,
                    pastControls: inputs[2]
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
            self.predictionFunction = predictionFunction
            self.activationVisualizationFunctions = activationVisualizationFunctions
            self.channelVisualizationFunction = channelVisualizationFunction
            self.saliencyVisualizationFunction = saliencyVisualizationFunction
            self.saliencyGradientFunction = saliencyGradient
            self.profile = runtimeProfile
            self.sourceVersionID = version.id
            self.daggerConfiguration = liveDAggerConfiguration
            self.allowedKeyCodes = allowedKeyCodes
            self.shiftUsesKeyboardChannel = (version.trainingDataSchema ?? 0) >= 7
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
            daggerRecorder = nil; daggerHumanActions = nil; daggerOutputPaused = false
            daggerStarting = false; daggerStartupBuffer.reset()
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
                    let correctionState = self.lock.withLock { () -> (DAggerRecorder?, Bool) in
                        if self.daggerStarting {
                            self.daggerStartupBuffer.append(sample)
                            return (nil, true)
                        }
                        return (self.daggerRecorder, self.daggerRecorder != nil)
                    }
                    if correctionState.1 {
                        correctionState.0?.append(sample)
                        self.lock.withLock { self.daggerHumanActions?.consume(sample) }
                        return
                    }
                }
                guard panic || self.safety.stopOnHumanInput else { return }
                Task { await self.stop(reason: panic ? "Panic stop" : "Stopped on human input") }
            }
            try safetyMonitor.start()
            let targetPID = await focusTargetIfNeeded(captureSpec)
            guard lock.withLock({ !stopped }) else { throw CancellationError() }
            lock.withLock { self.targetPID = targetPID }
            var liveCaptureSpec = captureSpec
            liveCaptureSpec.requestedFPS = runtimeProfile.training.perceptionFPS
            liveCaptureSpec.showsCursor = version.trainingShowsCursor ?? captureSpec.showsCursor
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

    func startDAggerRecording(
        request: DAggerCaptureRequest,
        recordingShortcut: HotkeyBinding
    ) async throws {
        let now = Self.hostTimeNanos()
        let snapshot = try lock.withLock { () throws -> [DAggerFrame] in
            guard !stopped, daggerRecorder == nil, !daggerStarting, let profile,
                  sourceVersionID == request.versionID,
                  profile.id == request.profileID,
                  profile.preprocessing == request.preprocessing else {
                throw AgentTrainerError.model("DAgger can start only after this AI has produced a live perception, and only one correction can be recorded at a time.")
            }
            let configured = daggerConfiguration
            let sizing = DAggerContextSizing.recentFrameCapacity(
                preprocessing: profile.preprocessing,
                temporal: profile.training.effectiveTemporalVision,
                perceptionFPS: profile.training.perceptionFPS,
                configuration: configured
            )
            guard sizing.maximum >= sizing.requested else {
                throw AgentTrainerError.invalidConfiguration("This DAgger context exceeds the 512 MB runtime buffer. Reduce lead-up, temporal context, or vision size.")
            }
            let nanos = UInt64((configured.preRollSeconds * 1e9).rounded())
            let cutoff = now > nanos ? now - nanos : 0
            let recent = daggerFrames.frames().filter { $0.timestampNanos >= cutoff }
            let temporal = profile.training.effectiveTemporalVision
            let pastSpec = temporal.pastFrameSpec(from: profile.preprocessing)
            let exactTemporalContext: [DAggerFrame] = (0..<temporal.pastFrameCount).compactMap { frame in
                // The ring already contains the current frame at trigger time;
                // add one to the normal inference distance to recover the exact
                // causal frames the policy would select for this current image.
                let distance = temporal.frameSpacing * (temporal.pastFrameCount - frame) + 1
                guard let stored = temporalFrames.frame(distanceBack: distance) else { return nil }
                return DAggerFrame(
                    timestampNanos: stored.timestampNanos,
                    packedVision: stored.packed,
                    preprocessing: pastSpec,
                    executedControls: stored.controls
                )
            }
            var byTimestamp: [UInt64: DAggerFrame] = [:]
            for frame in exactTemporalContext { byTimestamp[frame.timestampNanos] = frame }
            // Prefer full current-resolution recent frames when the two sources
            // overlap; older sparse entries exist only to cover long spacing.
            for frame in recent { byTimestamp[frame.timestampNanos] = frame }
            let retained = byTimestamp.values.sorted { $0.timestampNanos < $1.timestampNanos }
            guard !retained.isEmpty else {
                throw AgentTrainerError.model("Wait for the AI's first complete perception before starting a DAgger correction.")
            }
            // This is the explicit trigger boundary. Stop admitting AI action
            // ticks before constructing encoders or copying pre-roll so the
            // first expert input can never be mistaken for ordinary takeover.
            daggerStarting = true
            daggerOutputPaused = true
            daggerStartupBuffer.reset()
            return retained
        }
        safetyMonitor.excludedKeyCodes = request.excludedKeyCodes
        await drain(queue: actionQueue)
        injector.disableAndReleaseAll()
        do {
            let provisional = RecordingManifest(
                id: UUID(), name: "DAgger", createdAt: Date(), hostStartNanos: now,
                duration: 1, capture: request.capture, globalRect: CodableRect(request.captureRect),
                pixelWidth: request.preprocessing.width,
                pixelHeight: request.preprocessing.height,
                deliveredFPS: request.perceptionFPS, eventCount: 0
            )
            var humanActions = ActionAccumulator(manifest: provisional)
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
                preRoll: snapshot
            )
            for sample in seed {
                recorder.append(sample)
                humanActions.consume(sample)
            }
            let installResult = lock.withLock { () -> DAggerInstallResult in
                guard !stopped, daggerStarting, daggerRecorder == nil else { return .stopped }
                guard !daggerStartupBuffer.overflowed else { return .overflow }
                for sample in daggerStartupBuffer.samples {
                    recorder.append(sample)
                    humanActions.consume(sample)
                }
                for frame in daggerStartupBuffer.frames { recorder.append(frame) }
                daggerStartupBuffer.reset()
                daggerRecorder = recorder
                daggerHumanActions = humanActions
                daggerStarting = false
                return .installed
            }
            guard installResult == .installed else {
                _ = try? await recorder.finish(endHostNanos: Self.hostTimeNanos())
                if installResult == .overflow {
                    throw AgentTrainerError.capture("DAgger startup could not keep up with the initial correction stream inside its bounded buffer. The partial package was discarded; try again with a shorter lead-up, lower Perception FPS, or smaller vision size.")
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
                injector.enable(outputPermissions: outputPermissions)
            }
            safetyMonitor.excludedKeyCodes = []
            throw error
        }
    }

    func stopDAggerRecording() async {
        let recorder = lock.withLock { () -> DAggerRecorder? in
            let recorder = daggerRecorder
            daggerRecorder = nil
            daggerHumanActions = nil
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
        lock.withLock {
            guard !stopped else { return }
            daggerOutputPaused = false
            injector.enable(outputPermissions: outputPermissions)
        }
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
            daggerRecorder = nil; daggerHumanActions = nil; daggerOutputPaused = false
            daggerStarting = false; daggerStartupBuffer.reset()
            latestFrame = nil; latestFrameHostNanos = 0; lastUsableCaptureFrame = nil
            temporalFrames.reset(); predictionLatch.reset(); targetPID = nil
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
        if stoppedDAggerRecorder != nil { onDAggerRecordingChanged?(false) }
        safetyMonitor.stop()
        safetyMonitor.onSample = nil
        safetyMonitor.excludedKeyCodes = []
        _ = try? await capture.stop()
        await drain(queue: inferenceQueue)
        if let stoppedDAggerRecorder {
            do {
                let result = try await stoppedDAggerRecorder.finish(endHostNanos: Self.hostTimeNanos())
                onDAggerFinished?(.success(result))
            } catch {
                onDAggerFinished?(.failure(error))
            }
        }
        lock.withLock {
            predictionFunction = nil; activationVisualizationFunctions.removeAll(keepingCapacity: false); channelVisualizationFunction = nil; saliencyVisualizationFunction = nil; saliencyGradientFunction = nil; model = nil; profile = nil; sourceVersionID = nil; daggerConfiguration = DAggerConfiguration(); allowedKeyCodes.removeAll(keepingCapacity: false); shiftUsesKeyboardChannel = false
            latestFrame = nil; latestFrameHostNanos = 0; lastUsableCaptureFrame = nil; temporalFrames.reset(); daggerFrames.reset(); predictionLatch.reset(); processing = false
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

    private func receive(_ buffer: CVPixelBuffer, timestamp: CMTime) {
        schedule(
            buffer,
            timestampNanos: Self.hostTimeNanos(timestamp),
            countThrottledAsDropped: true
        )
    }

    private func receiveIdle(timestamp: CMTime) {
        let frame = lock.withLock { stopped ? nil : lastUsableCaptureFrame }
        if let frame {
            schedule(
                frame,
                timestampNanos: Self.hostTimeNanos(timestamp),
                countThrottledAsDropped: false
            )
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
            let inferenceInputs = try inferenceInputBuffers.makeArrays(descriptors) { destinations in
                packed.withUnsafeBytes { source in
                    destinations[0].copyMemory(from: source)
                }
                guard hasTemporalMemory else { return }
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
                    let gradientInputs = hasTemporalMemory
                        ? [forward[1], inferenceInputs[1], inferenceInputs[2], selector]
                        : [forward[1], selector]
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
            guard !stopped else { lock.unlock(); return }
            let frameControls = RuntimeActionSemantics.temporalControlValues(
                values,
                channels: profile.channels,
                restrictions: profile.effectiveRestrictions,
                allowedKeyCodes: allowedKeyCodes,
                outputPermissions: outputPermissions,
                shiftUsesKeyboardChannel: shiftUsesKeyboardChannel
            )
            var contextualControls = frameControls
            if var humanActions = daggerHumanActions {
                contextualControls = humanActions.actionValues()
                humanActions.endTick()
                daggerHumanActions = humanActions
            }
            let frameTimestamp = timestampNanos > 0 ? timestampNanos : Self.hostTimeNanos()
            if hasTemporalMemory {
                let retainedFrameCount = temporal.pastFrameCount * temporal.frameSpacing + 1
                temporalFrames.append(
                    TemporalFrame(
                        packed: packedPastFrame,
                        controls: contextualControls,
                        timestampNanos: frameTimestamp
                    ),
                    capacity: retainedFrameCount
                )
            }
            let daggerFrame = DAggerFrame(
                timestampNanos: frameTimestamp,
                packedVision: packed,
                preprocessing: profile.preprocessing,
                executedControls: contextualControls
            )
            let daggerSizing = DAggerContextSizing.recentFrameCapacity(
                preprocessing: profile.preprocessing,
                temporal: temporal,
                perceptionFPS: profile.training.perceptionFPS,
                configuration: daggerConfiguration
            )
            daggerFrames.append(
                daggerFrame,
                capacity: max(1, min(daggerSizing.requested, daggerSizing.maximum))
            )
            if daggerStarting {
                daggerStartupBuffer.append(daggerFrame)
            }
            let activeDAggerRecorder = daggerRecorder
            predictionLatch.publish(values)
            metrics.frameCount += 1
            let elapsed = max(0.001, CACurrentMediaTime() - startedAt)
            metrics.perceptionFPS = Double(metrics.frameCount) / elapsed
            metrics.latencyMilliseconds = (CACurrentMediaTime() - began) * 1_000
            let now = CACurrentMediaTime()
            let snapshot: RuntimeMetrics? = now - lastMetricsReportTime >= 0.1 ? metrics : nil
            if snapshot != nil { lastMetricsReportTime = now }
            lock.unlock()
            activeDAggerRecorder?.append(daggerFrame)
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
