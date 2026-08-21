import Foundation
import MLX
import MLXNN
import QuartzCore

private struct ReinforcementOptimizerIdentity: Codable, Hashable {
    var learningRate: Double
    var policyClip: Double
    var behaviorAnchor: Double
    var entropyBonus: Double
    var binaryExploration: Double
    var continuousExplorationStandardDeviation: Double
    var maximumGradientNorm: Double

    init(_ configuration: ReinforcementConfiguration) {
        learningRate = configuration.learningRate
        policyClip = configuration.policyClip
        behaviorAnchor = configuration.behaviorAnchor
        entropyBonus = configuration.entropyBonus
        binaryExploration = configuration.binaryExploration
        continuousExplorationStandardDeviation = configuration.continuousExplorationStandardDeviation
        maximumGradientNorm = configuration.maximumGradientNorm
    }
}

private struct ReinforcementCheckpointState: Codable {
    var schemaVersion = ReinforcementLearningContract.schemaVersion
    var optimizerIdentity: ReinforcementOptimizerIdentity
    var randomState: UInt64
    var feedbackCount: Int
    var updateCount: Int
    var rewardTotal: Double
    var punishmentTotal: Double
    var trainingSeconds: Double

    var isValid: Bool {
        feedbackCount >= 0
            && updateCount >= 0
            && rewardTotal.isFinite
            && rewardTotal >= 0
            && punishmentTotal.isFinite
            && punishmentTotal >= 0
            && trainingSeconds.isFinite
            && trainingSeconds >= 0
    }
}

/// A tiny deterministic generator is used only for action exploration. Its
/// state is persisted beside AdamW so a resumed online-learning brain does not
/// silently restart its exploration sequence.
struct ReinforcementRandomGenerator: Sendable {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        return value ^ (value >> 31)
    }

    mutating func uniform() -> Float {
        let upper = next() >> 11
        return Float(Double(upper) / Double(UInt64(1) << 53))
    }

    mutating func normal() -> Float {
        let first = max(1e-7, uniform())
        let second = uniform()
        return sqrt(-2 * Foundation.log(first)) * Foundation.cos(2 * .pi * second)
    }
}

enum ReinforcementActionPolicy {
    static func allowedMask(
        profile: AIProfile,
        allowedKeyCodes: Set<UInt16>,
        mouseMode: MouseControlMode,
        outputPermissions: RuntimeOutputPermissions
    ) -> [Float] {
        var mask = [Float](repeating: 0, count: ActionLayout.count)
        let channels = profile.channels
        let restrictions = profile.effectiveRestrictions

        if channels.mouseMovement, outputPermissions.cursorMovement {
            let range = mouseMode == .relative ? ActionLayout.relativeMouse : ActionLayout.absoluteMouse
            for index in range { mask[index] = 1 }
        }
        if channels.buttons {
            for button in 0..<8 where restrictions.allowsButton(UInt8(button)) {
                mask[ActionLayout.buttons.lowerBound + button] = 1
            }
        }
        if channels.scroll {
            for index in ActionLayout.scroll { mask[index] = 1 }
        }
        if channels.keyboard, outputPermissions.keyboard {
            for code in allowedKeyCodes where code < 128
                && restrictions.allowsKey(code)
                && !ActionLayout.commandOptionControlKeyCodeSet.contains(code) {
                mask[ActionLayout.keyboard.lowerBound + Int(code)] = 1
            }
            if !allowedKeyCodes.isDisjoint(with: [56, 60]), restrictions.allowsModifier(0) {
                mask[ActionLayout.shift.lowerBound] = 1
            }
        }
        if channels.modifiers, outputPermissions.keyboard {
            let equivalents: [[UInt16]] = [[59, 62], [58, 61], [55, 54]]
            for offset in equivalents.indices
                where !allowedKeyCodes.isDisjoint(with: equivalents[offset])
                    && restrictions.allowsModifier(offset + 1) {
                mask[ActionLayout.commandOptionControl.lowerBound + offset] = 1
            }
        }
        return mask
    }

    static func activated(logits: [Float]) -> [Float] {
        guard logits.count >= ActionLayout.count else { return logits }
        var values = logits
        for index in ActionLayout.absoluteMouse { values[index] = sigmoid(logits[index]) }
        for index in ActionLayout.relativeMouse { values[index] = Foundation.tanh(logits[index]) }
        for index in ActionLayout.buttons { values[index] = sigmoid(logits[index]) }
        for index in ActionLayout.scroll { values[index] = Foundation.tanh(logits[index]) }
        for index in ActionLayout.keyboard { values[index] = sigmoid(logits[index]) }
        for index in ActionLayout.modifiers { values[index] = sigmoid(logits[index]) }
        return values
    }

    static func sample(
        logits: [Float],
        allowedMask: [Float],
        configuration: ReinforcementConfiguration,
        random: inout ReinforcementRandomGenerator
    ) -> [Float] {
        var actions = activated(logits: logits)
        guard actions.count >= ActionLayout.count, allowedMask.count >= ActionLayout.count else { return actions }
        let continuousStandardDeviation = Float(configuration.continuousExplorationStandardDeviation)
        let binaryExploration = Float(configuration.binaryExploration)

        for index in ActionLayout.absoluteMouse where allowedMask[index] > 0 {
            actions[index] = min(1, max(0, actions[index] + random.normal() * continuousStandardDeviation))
        }
        for index in Array(ActionLayout.relativeMouse) + Array(ActionLayout.scroll) where allowedMask[index] > 0 {
            actions[index] = min(1, max(-1, actions[index] + random.normal() * continuousStandardDeviation))
        }
        for index in ActionLayout.binary where allowedMask[index] > 0 {
            let probability = (1 - binaryExploration) * actions[index] + binaryExploration * 0.5
            actions[index] = random.uniform() < probability ? 1 : 0
        }
        return actions
    }

    private static func sigmoid(_ value: Float) -> Float {
        if value >= 0 { return 1 / (1 + Foundation.exp(-value)) }
        let exponential = Foundation.exp(value)
        return exponential / (1 + exponential)
    }
}

/// Converts a modifier+wheel gesture into exact fixed feedback increments. Raw
/// macOS wheel events use roughly ten point-delta units per physical detent;
/// high-resolution trackpad events are accumulated until the same boundary.
struct ReinforcementScrollAccumulator: Sendable {
    static let pointsPerStep = 10.0
    private(set) var residual = 0.0

    mutating func process(
        _ sample: InputSample,
        configuration: ReinforcementConfiguration
    ) -> (handled: Bool, signal: Double?) {
        guard configuration.scrollFeedbackEnabled else { return (false, nil) }
        if sample.kind == .flags {
            return (Self.modifierKeyCodes(configuration.scrollCarbonModifiers).contains(sample.keyCode), nil)
        }
        guard sample.kind == .scroll else { return (false, nil) }
        let expected = HotkeyBinding(
            keyCode: 0,
            carbonModifiers: configuration.scrollCarbonModifiers
        ).cgEventModifiers
        guard sample.modifiers & HotkeyBinding.cgModifierMask == expected else { return (false, nil) }
        let delta = sample.scrollY
        guard delta.isFinite, delta != 0 else { return (true, nil) }
        if residual != 0, residual.sign != delta.sign { residual = 0 }
        residual += delta
        let stepCount = Int(abs(residual) / Self.pointsPerStep)
        guard stepCount > 0 else { return (true, nil) }
        let direction = residual.sign == .plus ? 1.0 : -1.0
        residual -= direction * Double(stepCount) * Self.pointsPerStep
        let rewardDirection = configuration.scrollUpRewards ? direction : -direction
        let value = rewardDirection * Double(stepCount) * configuration.scrollStep
        return (
            true,
            min(
                ReinforcementLearningContract.maximumSignalMagnitude,
                max(-ReinforcementLearningContract.maximumSignalMagnitude, value)
            )
        )
    }

    static func modifierKeyCodes(_ carbonModifiers: UInt32) -> Set<UInt16> {
        var result: Set<UInt16> = []
        if carbonModifiers & UInt32(1 << 9) != 0 { result.formUnion([56, 60]) }
        if carbonModifiers & UInt32(1 << 12) != 0 { result.formUnion([59, 62]) }
        if carbonModifiers & UInt32(1 << 11) != 0 { result.formUnion([58, 61]) }
        if carbonModifiers & UInt32(1 << 8) != 0 { result.formUnion([55, 54]) }
        return result
    }
}

enum ReinforcementPolicyObjective {
    static func loss(
        logits: MLXArray,
        behaviorLogits: MLXArray,
        actions: MLXArray,
        actionMask: MLXArray,
        advantages: MLXArray,
        configuration: ReinforcementConfiguration,
        dtype: DType
    ) -> MLXArray {
        var binaryValues = [Float](repeating: 0, count: ActionLayout.count)
        for index in ActionLayout.binary { binaryValues[index] = 1 }
        let binaryConstant = MLXArray(binaryValues, [1, ActionLayout.count]).asType(dtype)
        let actions = actions.asType(dtype)
        let mask = actionMask.asType(dtype)
        let binaryMask = mask * binaryConstant
        let continuousMask = mask * (1 - binaryConstant)
        let current = activated(logits, dtype: dtype)
        let behavior = stopGradient(activated(behaviorLogits, dtype: dtype))

        let exploration = Float(configuration.binaryExploration)
        let currentProbability = clip(
            (1 - exploration) * current + exploration * 0.5,
            min: 1e-5,
            max: 1 - 1e-5
        )
        let behaviorProbability = clip(
            (1 - exploration) * behavior + exploration * 0.5,
            min: 1e-5,
            max: 1 - 1e-5
        )
        let binaryLogProbability = actions * log(currentProbability)
            + (1 - actions) * log(1 - currentProbability)
        let behaviorBinaryLogProbability = actions * log(behaviorProbability)
            + (1 - actions) * log(1 - behaviorProbability)

        let standardDeviation = Float(configuration.continuousExplorationStandardDeviation)
        let inverseVariance = 1 / max(1e-8, standardDeviation * standardDeviation)
        let continuousLogProbability = -0.5 * square(actions - current) * inverseVariance
        let behaviorContinuousLogProbability = -0.5 * square(actions - behavior) * inverseVariance
        let denominator = mask.sum(axis: -1) + 1e-6
        let currentLogProbability = (
            binaryLogProbability * binaryMask
                + continuousLogProbability * continuousMask
        ).sum(axis: -1) / denominator
        let oldLogProbability = stopGradient((
            behaviorBinaryLogProbability * binaryMask
                + behaviorContinuousLogProbability * continuousMask
        ).sum(axis: -1) / denominator)

        let ratio = exp(clip(currentLogProbability - oldLogProbability, min: -5, max: 5))
        let clipAmount = Float(configuration.policyClip)
        let clippedRatio = clip(ratio, min: 1 - clipAmount, max: 1 + clipAmount)
        let advantage = advantages.asType(dtype)
        let activeRows = (abs(advantage) .> 0).asType(dtype)
        let surrogate = minimum(ratio * advantage, clippedRatio * advantage)
        let rowCount = activeRows.sum() + 1e-6
        let policyLoss = -(surrogate * activeRows).sum() / rowCount

        let binaryDivergence = behaviorProbability * log(behaviorProbability / currentProbability)
            + (1 - behaviorProbability) * log((1 - behaviorProbability) / (1 - currentProbability))
        let continuousDivergence = 0.5 * square(current - behavior) * inverseVariance
        let divergence = (
            binaryDivergence * binaryMask + continuousDivergence * continuousMask
        ).sum(axis: -1) / denominator
        let anchor = Float(configuration.behaviorAnchor) * (divergence * activeRows).sum() / rowCount

        let entropy = -(
            currentProbability * log(currentProbability)
                + (1 - currentProbability) * log(1 - currentProbability)
        )
        let binaryDenominator = binaryMask.sum(axis: -1) + 1e-6
        let rowEntropy = (entropy * binaryMask).sum(axis: -1) / binaryDenominator
        let entropyBonus = Float(configuration.entropyBonus) * (rowEntropy * activeRows).sum() / rowCount
        return policyLoss + anchor - entropyBonus
    }

    private static func activated(_ logits: MLXArray, dtype: DType) -> MLXArray {
        concatenated([
            sigmoid(logits[.ellipsis, ActionLayout.absoluteMouse]),
            tanh(logits[.ellipsis, ActionLayout.relativeMouse]),
            sigmoid(logits[.ellipsis, ActionLayout.buttons]),
            tanh(logits[.ellipsis, ActionLayout.scroll]),
            sigmoid(logits[.ellipsis, ActionLayout.keyboard]),
            sigmoid(logits[.ellipsis, ActionLayout.modifiers])
        ], axis: -1).asType(dtype)
    }
}

final class ReinforcementTrainer: @unchecked Sendable {
    private struct Transition: Sendable {
        var timestamp: Double
        var currentPacked: Data
        var pastEmbeddings: [Float]
        var pastControls: [Float]
        var behaviorLogits: [Float]
        var action: [Float]
        var allowedMask: [Float]
    }

    private struct CreditedTransition {
        var transition: Transition
        var advantage: Float
        var mask: [Float]
    }

    private typealias TrainingFunction = @Sendable ([MLXArray]) -> [MLXArray]

    let configuration: ReinforcementConfiguration
    let batchCapacity: Int
    private let profile: AIProfile
    private let baseVersion: ModelVersionManifest?
    private let model: AgentPolicy
    private let optimizer: ResumableAdamW
    private let inputBuffers: MetalArrayBufferPool
    private let snapshotRoot: URL
    private let demonstratedKeyCodes: Set<UInt16>
    private let trainingShowsCursor: Bool
    private let recommendedMouseMode: MouseControlMode
    private let sessionID = UUID()
    private let sessionStartedAt = Date()
    private let trainingStep: TrainingFunction
    private var transitions: [Transition] = []
    private var random: ReinforcementRandomGenerator
    private var sequence = 0
    private var lastSnapshotUpdateCount = 0
    private var sessionFeedbackCount = 0
    private var sessionRewardCount = 0
    private var sessionPunishmentCount = 0
    private var sessionRewardTotal = 0.0
    private var sessionPunishmentTotal = 0.0
    private var sessionUpdateCount = 0
    private var cumulativeFeedbackCount: Int
    private var cumulativeUpdateCount: Int
    private var cumulativeRewardTotal: Double
    private var cumulativePunishmentTotal: Double
    private var cumulativeTrainingSeconds: Double
    private var lastPolicyLoss: Double?
    private var lastSignal: ReinforcementSignal?
    private var lastCreditedFrames = 0
    private var lastUpdateMilliseconds = 0.0
    private var autosavesPublished = 0

    init(
        model: AgentPolicy,
        profile: AIProfile,
        configuration rawConfiguration: ReinforcementConfiguration,
        baseVersion: ModelVersionManifest?,
        baseVersionDirectory: URL?,
        snapshotRoot: URL,
        demonstratedKeyCodes: Set<UInt16>,
        trainingShowsCursor: Bool,
        recommendedMouseMode: MouseControlMode
    ) throws {
        let configuration = try rawConfiguration.validated()
        self.configuration = configuration
        self.profile = profile
        self.baseVersion = baseVersion
        self.model = model
        self.snapshotRoot = snapshotRoot
        self.demonstratedKeyCodes = demonstratedKeyCodes
        self.trainingShowsCursor = trainingShowsCursor
        self.recommendedMouseMode = recommendedMouseMode
        inputBuffers = try MetalArrayBufferPool(maximumCachedBytes: 128 * 1_024 * 1_024)

        let temporal = profile.training.effectiveTemporalVision
        let transitionBytes = max(1, profile.preprocessing.sampleByteCount)
            + temporal.pastFrameCount
                * (profile.training.architecture.visualEmbedding + ActionLayout.count)
                * MemoryLayout<Float>.size
            + ActionLayout.count * MemoryLayout<Float>.size * 3
        let memoryCapacity = max(1, ReinforcementLearningContract.maximumTransitionBytes / max(1, transitionBytes))
        let configuredFPS = profile.training.perceptionFPS
        let safeFPS = configuredFPS.isFinite && configuredFPS > 0 ? configuredFPS : 60
        let timeCapacity = max(1, Int(ceil(configuration.creditWindowSeconds * safeFPS)))
        batchCapacity = min(configuration.maximumCreditFrames, min(timeCapacity, memoryCapacity))

        optimizer = ResumableAdamW(
            learningRate: Float(configuration.learningRate),
            weightDecay: 0,
            warmupSteps: 1,
            schedule: .adaptiveCosine,
            cycleSteps: 1_000_000,
            minimumLearningRateRatio: 0.5
        )
        optimizer.initialize(model: model)

        var restoredState: ReinforcementCheckpointState?
        if let baseVersionDirectory,
           let optimizerFile = baseVersion?.reinforcementOptimizerFile,
           let stateFile = baseVersion?.reinforcementStateFile,
           let state = try? JSONDecoder().decode(
               ReinforcementCheckpointState.self,
               from: Data(contentsOf: baseVersionDirectory.appendingPathComponent(stateFile))
           ),
           state.schemaVersion == ReinforcementLearningContract.schemaVersion,
           state.isValid {
            restoredState = state
            if state.optimizerIdentity == ReinforcementOptimizerIdentity(configuration) {
                try optimizer.load(from: baseVersionDirectory.appendingPathComponent(optimizerFile))
            }
        }

        cumulativeFeedbackCount = restoredState?.feedbackCount
            ?? baseVersion?.reinforcementFeedbackCount ?? 0
        cumulativeUpdateCount = restoredState?.updateCount
            ?? baseVersion?.reinforcementUpdateCount ?? 0
        let netReward = baseVersion?.reinforcementNetReward ?? 0
        cumulativeRewardTotal = restoredState?.rewardTotal ?? max(0, netReward)
        cumulativePunishmentTotal = restoredState?.punishmentTotal ?? max(0, -netReward)
        cumulativeTrainingSeconds = restoredState?.trainingSeconds
            ?? baseVersion?.reinforcementTrainingSeconds ?? 0
        random = ReinforcementRandomGenerator(
            state: restoredState?.randomState
                ?? (profile.training.seed ^ UInt64(max(0, baseVersion?.globalStep ?? 0)) ^ 0xA0761D6478BD642F)
        )

        let hasTemporalMemory = temporal.pastFrameCount > 0
        let trainingConfiguration = configuration
        let trainingProfile = profile
        let trainingModel = model
        let trainingOptimizer = optimizer
        trainingStep = compile(
            inputs: [trainingModel, trainingOptimizer],
            outputs: [trainingModel, trainingOptimizer]
        ) { (arrays: [MLXArray]) -> [MLXArray] in
            let currentImages = VisionPreprocessor.mlxTensor(
                arrays[0], spec: trainingProfile.preprocessing, dtype: trainingModel.dtype
            )
            let actionIndex = hasTemporalMemory ? 3 : 1
            let behaviorIndex = actionIndex + 1
            let maskIndex = actionIndex + 2
            let advantageIndex = actionIndex + 3
            let gradientInputs = arrays
            let result = valueAndGrad(model: trainingModel) { candidate, values in
                let temporalFeatures: MLXArray
                if hasTemporalMemory {
                    let currentEmbedding = candidate.visualEmbedding(
                        visualFeatures: candidate.visualActivations(images: currentImages).last!
                    )
                    temporalFeatures = candidate.temporalFeatures(
                        currentVisualEmbedding: currentEmbedding,
                        pastVisualEmbeddings: values[1],
                        pastControls: values[2]
                    )
                } else {
                    temporalFeatures = candidate.currentOnlyTemporalFeatures(currentImages: currentImages)
                }
                let logits = candidate.logits(temporalFeatures: temporalFeatures)
                return [ReinforcementPolicyObjective.loss(
                    logits: logits,
                    behaviorLogits: values[behaviorIndex],
                    actions: values[actionIndex],
                    actionMask: values[maskIndex],
                    advantages: values[advantageIndex],
                    configuration: trainingConfiguration,
                    dtype: candidate.dtype
                )]
            }(trainingModel, gradientInputs)
            trainingOptimizer.update(
                model: trainingModel,
                gradients: result.1,
                targetType: trainingModel.dtype,
                maxGradientNorm: Float(trainingConfiguration.maximumGradientNorm)
            )
            return result.0
        }
        model.train(false)
    }

    func sample(logits: [Float], allowedMask: [Float]) -> [Float] {
        ReinforcementActionPolicy.sample(
            logits: logits,
            allowedMask: allowedMask,
            configuration: configuration,
            random: &random
        )
    }

    func record(
        timestamp: Double,
        currentPacked: Data,
        pastEmbeddings: [Float],
        pastControls: [Float],
        behaviorLogits: [Float],
        action: [Float],
        allowedMask: [Float]
    ) {
        let temporal = profile.training.effectiveTemporalVision
        let expectedEmbeddings = temporal.pastFrameCount
            * profile.training.architecture.visualEmbedding
        let expectedControls = temporal.pastFrameCount * ActionLayout.count
        guard timestamp.isFinite,
              currentPacked.count == profile.preprocessing.sampleByteCount,
              pastEmbeddings.count == expectedEmbeddings,
              pastControls.count == expectedControls,
              behaviorLogits.count == ActionLayout.count,
              action.count == ActionLayout.count,
              allowedMask.count == ActionLayout.count else { return }
        transitions.append(Transition(
            timestamp: timestamp,
            currentPacked: currentPacked,
            pastEmbeddings: pastEmbeddings,
            pastControls: pastControls,
            behaviorLogits: behaviorLogits,
            action: action,
            allowedMask: allowedMask
        ))
        if transitions.count > batchCapacity {
            transitions.removeFirst(transitions.count - batchCapacity)
        }
    }

    func apply(_ rawSignal: ReinforcementSignal) throws -> (metrics: ReinforcementMetrics, shouldAutosave: Bool) {
        let clampedValue = min(
            ReinforcementLearningContract.maximumSignalMagnitude,
            max(-ReinforcementLearningContract.maximumSignalMagnitude, rawSignal.value)
        )
        guard rawSignal.timestamp.isFinite,
              clampedValue.isFinite,
              clampedValue != 0 else { return (metrics, false) }
        var signal = rawSignal
        signal.value = clampedValue
        lastSignal = signal
        sessionFeedbackCount += 1
        cumulativeFeedbackCount += 1
        if clampedValue > 0 {
            sessionRewardCount += 1
            sessionRewardTotal += clampedValue
            cumulativeRewardTotal += clampedValue
        } else {
            sessionPunishmentCount += 1
            sessionPunishmentTotal += -clampedValue
            cumulativePunishmentTotal += -clampedValue
        }

        let earliest = signal.timestamp - configuration.creditWindowSeconds
        let eligible = transitions.filter {
            $0.timestamp <= signal.timestamp && $0.timestamp >= earliest
        }.suffix(batchCapacity)
        var credited: [CreditedTransition] = []
        credited.reserveCapacity(eligible.count)
        for (offset, transition) in eligible.enumerated() {
            let distanceFromNewest = eligible.count - offset - 1
            let credit = Float(clampedValue * Foundation.pow(configuration.creditDecay, Double(distanceFromNewest)))
            var mask = transition.allowedMask
            for index in ActionLayout.binary
                where transition.action[index] < 0.5 && !configuration.learnFromInaction {
                mask[index] = 0
            }
            for index in Array(ActionLayout.relativeMouse) + Array(ActionLayout.scroll)
                where abs(transition.action[index]) < 0.005 && !configuration.learnFromInaction {
                mask[index] = 0
            }
            guard mask.contains(where: { $0 > 0 }) else { continue }
            credited.append(CreditedTransition(transition: transition, advantage: credit, mask: mask))
        }
        lastCreditedFrames = credited.count
        guard !credited.isEmpty else { return (metrics, false) }

        let arrays = try makeTrainingArrays(credited)
        let began = CACurrentMediaTime()
        let output = trainingStep(arrays)
        // A compiled MLX call is lazy. Force the objective, updated weights,
        // and Adam state together so metrics and snapshots can never observe a
        // partially evaluated online step.
        MLX.eval(output, model.parameters(), optimizer.stateArrays())
        let elapsed = CACurrentMediaTime() - began
        let loss = output[0].item(Float.self)
        guard loss.isFinite else {
            throw AgentTrainerError.model("The online RL update produced a non-finite objective, so the brain was not published.")
        }
        lastPolicyLoss = Double(loss)
        lastUpdateMilliseconds = elapsed * 1_000
        cumulativeTrainingSeconds += elapsed
        sessionUpdateCount += 1
        cumulativeUpdateCount += 1
        let shouldAutosave = sessionFeedbackCount.isMultiple(of: configuration.autosaveFeedbackCount)
            && sessionUpdateCount > lastSnapshotUpdateCount
        return (metrics, shouldAutosave)
    }

    var metrics: ReinforcementMetrics {
        ReinforcementMetrics(
            isActive: true,
            feedbackCount: sessionFeedbackCount,
            rewardCount: sessionRewardCount,
            punishmentCount: sessionPunishmentCount,
            rewardTotal: sessionRewardTotal,
            punishmentTotal: sessionPunishmentTotal,
            updateCount: sessionUpdateCount,
            optimizerStep: optimizer.step,
            lastSignal: lastSignal?.value,
            lastSignalSource: lastSignal?.sourceIdentifier ?? "",
            creditedFrames: lastCreditedFrames,
            lastPolicyLoss: lastPolicyLoss,
            lastUpdateMilliseconds: lastUpdateMilliseconds,
            autosavesPublished: autosavesPublished,
            pendingFeedback: 0
        )
    }

    var hasUpdates: Bool { sessionUpdateCount > 0 }

    func makeSnapshot(isAutosave: Bool) throws -> ReinforcementSnapshot? {
        guard hasUpdates else { return nil }
        sequence += 1
        let versionID = UUID()
        let staging = snapshotRoot.appendingPathComponent(
            ".ReinforcementSnapshot.\(sessionID.uuidString).\(sequence).tmp",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        do {
            let weightsFile = "weights.safetensors"
            let optimizerFile = "reinforcement-optimizer.safetensors"
            let stateFile = "reinforcement-state.json"
            try model.saveWeights(to: staging.appendingPathComponent(weightsFile))
            try optimizer.save(to: staging.appendingPathComponent(optimizerFile))
            let state = ReinforcementCheckpointState(
                optimizerIdentity: ReinforcementOptimizerIdentity(configuration),
                randomState: random.state,
                feedbackCount: cumulativeFeedbackCount,
                updateCount: cumulativeUpdateCount,
                rewardTotal: cumulativeRewardTotal,
                punishmentTotal: cumulativePunishmentTotal,
                trainingSeconds: cumulativeTrainingSeconds
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(state).write(
                to: staging.appendingPathComponent(stateFile),
                options: .atomic
            )

            let startingGlobalStep = max(
                0,
                baseVersion?.globalStep ?? profile.trainingProgress?.globalStep ?? 0
            )
            let previousTrainingSeconds = max(
                0,
                baseVersion?.trainingDurationSeconds
                    ?? profile.trainingProgress?.trainingDurationSeconds ?? 0
            )
            let version = ModelVersionManifest(
                id: versionID,
                name: isAutosave
                    ? "RL Autosave • Feedback \(cumulativeFeedbackCount) • Step \(startingGlobalStep + sessionUpdateCount)"
                    : "RL Brain • Feedback \(cumulativeFeedbackCount) • Step \(startingGlobalStep + sessionUpdateCount)",
                createdAt: Date(),
                globalStep: startingGlobalStep + sessionUpdateCount,
                trainingLoss: lastPolicyLoss ?? baseVersion?.trainingLoss ?? 0,
                validationLoss: nil,
                preprocessing: profile.preprocessing,
                channels: profile.channels,
                training: profile.training,
                epoch: baseVersion?.epoch ?? profile.trainingProgress?.epoch ?? 0,
                isAutosave: isAutosave,
                demonstratedKeyCodes: demonstratedKeyCodes,
                relativeMouseScale: GameCameraContract.deltaScale,
                // Online updates do not reinterpret the demonstration target
                // layout that produced their starting brain. Preserve that
                // contract across RL descendants; a brand-new neutral policy
                // adopts the current layout.
                trainingDataSchema: baseVersion?.trainingDataSchema
                    ?? TrainingDataContract.schemaVersion,
                trainingObjectiveSchema: baseVersion?.trainingObjectiveSchema,
                visualGroundingComplete: baseVersion?.visualGroundingComplete,
                trainingDurationSeconds: previousTrainingSeconds + cumulativeTrainingSeconds
                    - (baseVersion?.reinforcementTrainingSeconds ?? 0),
                experienceDurationSeconds: baseVersion?.experienceDurationSeconds
                    ?? profile.trainingProgress?.experienceDurationSeconds,
                trainingShowsCursor: trainingShowsCursor,
                recommendedMouseMode: recommendedMouseMode,
                validationReport: nil,
                binaryDecisionSchema: BinaryDecisionContract.schemaVersion,
                binaryDecisionThresholds: BinaryDecisionCalibration.normalized(
                    baseVersion?.hasCurrentBinaryDecisionContract == true
                        ? baseVersion?.binaryDecisionThresholds
                        : nil
                ),
                trainingDataCoverage: baseVersion?.trainingDataCoverage,
                reinforcementOptimizerFile: optimizerFile,
                reinforcementStateFile: stateFile,
                reinforcementSessionID: sessionID,
                reinforcementSessionStartedAt: sessionStartedAt,
                reinforcementSequence: sequence,
                reinforcementFeedbackCount: cumulativeFeedbackCount,
                reinforcementUpdateCount: cumulativeUpdateCount,
                reinforcementNetReward: cumulativeRewardTotal - cumulativePunishmentTotal,
                reinforcementTrainingSeconds: cumulativeTrainingSeconds
            )
            lastSnapshotUpdateCount = sessionUpdateCount
            if isAutosave { autosavesPublished += 1 }
            return ReinforcementSnapshot(
                profileID: profile.id,
                stagingDirectory: staging,
                manifest: version
            )
        } catch {
            try? FileManager.default.removeItem(at: staging)
            throw error
        }
    }

    private func makeTrainingArrays(_ credited: [CreditedTransition]) throws -> [MLXArray] {
        let temporal = profile.training.effectiveTemporalVision
        let hasTemporalMemory = temporal.pastFrameCount > 0
        let currentBytes = profile.preprocessing.sampleByteCount
        let embeddingValues = temporal.pastFrameCount * profile.training.architecture.visualEmbedding
        let controlValues = temporal.pastFrameCount * ActionLayout.count
        var descriptors: [MetalArrayBufferPool.Descriptor] = [
            .init([batchCapacity, currentBytes], dtype: .uint8)
        ]
        if hasTemporalMemory {
            descriptors.append(.init([
                batchCapacity,
                temporal.pastFrameCount,
                profile.training.architecture.visualEmbedding
            ], dtype: .float32))
            descriptors.append(.init([
                batchCapacity,
                temporal.pastFrameCount,
                ActionLayout.count
            ], dtype: .float32))
        }
        descriptors += [
            .init([batchCapacity, ActionLayout.count], dtype: .float32),
            .init([batchCapacity, ActionLayout.count], dtype: .float32),
            .init([batchCapacity, ActionLayout.count], dtype: .float32),
            .init([batchCapacity], dtype: .float32)
        ]

        return try inputBuffers.makeArrays(descriptors) { destinations in
            for destination in destinations {
                destination.initializeMemory(as: UInt8.self, repeating: 0)
            }
            let actionIndex = hasTemporalMemory ? 3 : 1
            for row in 0..<min(batchCapacity, credited.count) {
                let item = credited[row]
                item.transition.currentPacked.withUnsafeBytes { source in
                    UnsafeMutableRawBufferPointer(
                        rebasing: destinations[0][row * currentBytes..<(row + 1) * currentBytes]
                    ).copyMemory(from: source)
                }
                if hasTemporalMemory {
                    copy(
                        item.transition.pastEmbeddings,
                        valuesPerRow: embeddingValues,
                        row: row,
                        to: destinations[1]
                    )
                    copy(
                        item.transition.pastControls,
                        valuesPerRow: controlValues,
                        row: row,
                        to: destinations[2]
                    )
                }
                copy(item.transition.action, valuesPerRow: ActionLayout.count, row: row, to: destinations[actionIndex])
                copy(item.transition.behaviorLogits, valuesPerRow: ActionLayout.count, row: row, to: destinations[actionIndex + 1])
                copy(item.mask, valuesPerRow: ActionLayout.count, row: row, to: destinations[actionIndex + 2])
                destinations[actionIndex + 3].bindMemory(to: Float.self)[row] = item.advantage
            }
        }
    }

    private func copy(
        _ values: [Float],
        valuesPerRow: Int,
        row: Int,
        to destination: UnsafeMutableRawBufferPointer
    ) {
        guard valuesPerRow > 0, values.count >= valuesPerRow else { return }
        let byteCount = valuesPerRow * MemoryLayout<Float>.size
        values.withUnsafeBytes { source in
            UnsafeMutableRawBufferPointer(
                rebasing: destination[row * byteCount..<(row + 1) * byteCount]
            ).copyMemory(from: UnsafeRawBufferPointer(rebasing: source[0..<byteCount]))
        }
    }
}
