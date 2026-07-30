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

    private let capture = CaptureService()
    private let preprocessor: VisionPreprocessor
    private let injector = InputInjector()
    private let safetyMonitor = InputCaptureService()
    private let inferenceQueue = DispatchQueue(label: "AgentTrainer.Inference", qos: .userInteractive)
    private let actionQueue = DispatchQueue(label: "AgentTrainer.Actions", qos: .userInteractive)
    private let lock = NSLock()
    private var model: AgentPolicy?
    private var predictionFunction: (@Sendable (MLXArray, MLXArray) -> MLXArray)?
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
    private var history: [[Float]] = []
    private var historyWriteIndex = 0
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
    private var previousPackedVision: Data?
    private var launchRevision: UInt64 = 0
    private var starting = false
    private var teardownInProgress = false
    private var startupWaiters: [CheckedContinuation<Void, Never>] = []
    private var teardownWaiters: [CheckedContinuation<Void, Never>] = []

    init() throws {
        MLXMemoryLifecycle.configure()
        preprocessor = try VisionPreprocessor()
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

        // A runnable brain is immutable. Timing, history length, enabled heads,
        // precision, and architecture must come from the saved version rather
        // than mutable editor fields that may have changed after training.
        var runtimeProfile = profile
        runtimeProfile.preprocessing = version.preprocessing
        runtimeProfile.channels = RuntimeActionSemantics.effectiveChannels(saved: version.channels, current: profile.channels)
        runtimeProfile.training = version.training
        let startRevisions = lock.withLock { (outputPermissionsRevision, visualizationSettingsRevision) }
        let model = AgentPolicy(profile: runtimeProfile)
        let versionDirectory = await WorkspaceStore.shared.versionDirectory(profileID: profile.id, versionID: version.id)
        try model.loadWeights(from: versionDirectory.appendingPathComponent(version.weightsFile))
        model.train(false)
        let predictionFunction = compile(inputs: [model]) { images, history in model.predictions(images: images, history: history) }
        // Diagnostic graphs are lazy: creating these closures does not execute
        // or materialize an extra tensor. The selected graph first compiles only
        // when its view is enabled and reaches its independently capped rate.
        let activationVisualizationFunctions: [VisualizationFunction] = (0..<max(1, model.convolutions.count)).map { selectedLayer in
            compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
                let layers = model.visualActivations(images: inputs[0])
                let logits = model.logits(visualFeatures: layers.last!, history: inputs[1])
                let map = model.sampledForVisualization(layers[selectedLayer]).mean(axis: -1, keepDims: true)
                return [model.activatedPredictions(logits: logits), map]
            }
        }
        let channelVisualizationFunction = compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
            let layers = model.visualActivations(images: inputs[0])
            let logits = model.logits(visualFeatures: layers.last!, history: inputs[1])
            return [model.activatedPredictions(logits: logits), model.strongestChannelsForVisualization(layers.last!)]
        }
        let saliencyVisualizationFunction = compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
            let layers = model.visualActivations(images: inputs[0])
            let logits = model.logits(visualFeatures: layers.last!, history: inputs[1])
            // Keep the exact final tensor on GPU for the post-CNN gradient.
            // This graph intentionally omits the channel ranking used by the
            // separate feature-grid view.
            return [model.activatedPredictions(logits: logits), layers.last!]
        }
        let saliencyGradient = grad({ (inputs: [MLXArray]) -> MLXArray in
            let logits = model.logits(visualFeatures: inputs[0], history: inputs[1])
            return (logits * inputs[2].asType(model.dtype)).sum()
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
            latestFrame = nil; lastUsableCaptureFrame = nil; previousPackedVision = nil; predictionLatch.reset(); history = Array(repeating: [Float](repeating: 0, count: ActionLayout.count), count: max(1, runtimeProfile.training.historyLength))
            historyWriteIndex = 0; metrics = RuntimeMetrics(); startedAt = CACurrentMediaTime(); nextPerceptionTime = 0; lastMetricsReportTime = 0; lastFocusCheckTime = 0; stopped = false
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

    func stop(reason: String? = nil) async {
        let action = lock.withLock { () -> StopAction in
            launchRevision &+= 1
            if teardownInProgress { return .waitForTeardown }
            guard !stopped else { return starting ? .waitForStartup : .finished }
            stopped = true
            teardownInProgress = true
            let timer = actionTimer; actionTimer = nil
            latestFrame = nil; lastUsableCaptureFrame = nil; previousPackedVision = nil; predictionLatch.reset(); targetPID = nil
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
        lock.withLock {
            predictionFunction = nil; activationVisualizationFunctions.removeAll(keepingCapacity: false); channelVisualizationFunction = nil; saliencyVisualizationFunction = nil; saliencyGradientFunction = nil; model = nil; profile = nil; allowedKeyCodes.removeAll(keepingCapacity: false); shiftUsesKeyboardChannel = false
            latestFrame = nil; lastUsableCaptureFrame = nil; previousPackedVision = nil; predictionLatch.reset(); history.removeAll(keepingCapacity: false); historyWriteIndex = 0; processing = false
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
        let history: [Float] = (0..<self.history.count).flatMap { offset in
            self.history[(historyWriteIndex + offset) % self.history.count]
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
            let packed = try preprocessor.process(buffer, spec: profile.preprocessing)
            let previousPacked = lock.withLock { () -> Data? in
                let previous = previousPackedVision
                previousPackedVision = packed
                return previous
            }
            let image = VisionPreprocessor.mlxTemporalTensor(current: packed, previous: previousPacked, batch: 1, spec: profile.preprocessing)
            let historyArray = MLXArray(history, [1, max(1, profile.training.historyLength), ActionLayout.count])
            let result: [MLXArray] = Device.withDefaultDevice(.gpu) {
                guard visualizationDue else { return [predictionFunction(image, historyArray)] }
                switch settings.mode {
                case .activationOverlay:
                    let selectedLayer = max(0, settings.convolutionLayer)
                    guard activationVisualizationFunctions.indices.contains(selectedLayer) else { return [predictionFunction(image, historyArray)] }
                    return activationVisualizationFunctions[selectedLayer]([image, historyArray])
                case .featureChannels:
                    guard let forward = channelVisualizationFunction?([image, historyArray]), forward.count >= 2 else { return [predictionFunction(image, historyArray)] }
                    return forward
                case .actionSaliency:
                    let selector = Self.actionSelector(focus: settings.actionFocus, mouseMode: mouseMode)
                    guard let saliencyVisualizationFunction, let saliencyGradientFunction else { return [predictionFunction(image, historyArray)] }
                    let forward = saliencyVisualizationFunction([image, historyArray])
                    guard forward.count >= 2 else { return [predictionFunction(image, historyArray)] }
                    let gradients = saliencyGradientFunction([forward[1], historyArray, selector])
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
            predictionLatch.publish(values)
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
        guard !stopped, let latched = predictionLatch.consume(), let profile else { lock.unlock(); return }
        let prediction = latched.values
        let safety = self.safety, rect = captureRect, targetPID = self.targetPID, mouseMode = self.mouseMode, gameCamera = self.gameCamera, allowedKeyCodes = self.allowedKeyCodes, shiftUsesKeyboardChannel = self.shiftUsesKeyboardChannel, outputPermissions = self.outputPermissions
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
        // A configured history of zero still uses one placeholder tensor row,
        // matching training, but that row must remain zero rather than becoming
        // an accidental one-step runtime history.
        if profile.training.historyLength > 0, !history.isEmpty {
            history[historyWriteIndex] = RuntimeActionSemantics.historyValues(
                prediction,
                predictionIsFresh: latched.isFresh,
                channels: profile.channels,
                restrictions: profile.effectiveRestrictions,
                allowedKeyCodes: allowedKeyCodes,
                outputPermissions: outputPermissions,
                shiftUsesKeyboardChannel: shiftUsesKeyboardChannel
            )
            historyWriteIndex = (historyWriteIndex + 1) % history.count
        }
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

    /// Training history is one row per action tick. Reused policy state remains
    /// useful for held buttons/keys, while additive channels are zero on ticks
    /// where no new prediction was executed.
    static func historyValues(
        _ prediction: [Float],
        predictionIsFresh: Bool,
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
        if !predictionIsFresh {
            for index in ActionLayout.relativeMouse { values[index] = 0 }
            for index in ActionLayout.scroll { values[index] = 0 }
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
