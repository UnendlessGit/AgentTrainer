import Foundation
import MLX
import MLXNN
import MLXRandom

struct ReinforcementObservation: Sendable {
    var currentPacked: Data
    var pastPacked: Data
    var pastControls: [Float]
    var contextValues: [Float]
}

struct ReinforcementTransition: Sendable {
    var observation: ReinforcementObservation
    /// Binary samples are 0/1. Continuous samples retain the unbounded latent
    /// Gaussian value so PPO can recompute the exact transformed action density.
    var distributionActions: [Float]
    var executedActions: [Float]
    var oldLogProbability: Float
    var oldValue: Float
    var reward: Float
    var terminal: Bool
}

struct ReinforcementPolicySample: Sendable {
    var distributionActions: [Float]
    var executedActions: [Float]
    var logProbability: Float
}

struct ReinforcementUpdateMetrics: Sendable {
    var optimizerSteps = 0
    var policyLoss = 0.0
    var valueLoss = 0.0
    var entropy = 0.0
    var approximateKL = 0.0
    var clipFraction = 0.0
    var explainedVariance = 0.0
}

struct ReinforcementActionPlan: Hashable, Sendable {
    var binaryIndices: [Int]
    var continuousIndices: [Int]
    var absoluteContinuousIndices: Set<Int>
    var allowedKeyCodes: Set<UInt16>
    var allowedMouseButtons: Set<UInt8>

    var actionCount: Int { binaryIndices.count + continuousIndices.count }

    static func make(
        profile: AIProfile,
        configuration: ReinforcementLearningConfiguration,
        mouseMode: MouseControlMode
    ) throws -> Self {
        let actionSpace = try configuration.actionSpace.validated()
        let restrictions = profile.effectiveRestrictions
        var binary: [Int] = []
        let buttons = actionSpace.executableMouseButtons(restrictions: restrictions)
        if profile.channels.buttons {
            binary += buttons.sorted().map { ActionLayout.buttons.lowerBound + Int($0) }
        }

        let keys = actionSpace.executableKeyCodes(restrictions: restrictions)
        if profile.channels.keyboard {
            binary += keys.sorted().compactMap { code in
                guard code <= ReinforcementActionSpace.maximumKeyCode,
                      !ActionLayout.allModifierKeyCodes.contains(code),
                      !ActionLayout.commandOptionControlKeyCodeSet.contains(code) else { return nil }
                return ActionLayout.keyboard.lowerBound + Int(code)
            }
            if !keys.isDisjoint(with: ActionLayout.shiftKeyCodes),
               restrictions.allowsModifier(0) {
                binary.append(ActionLayout.shift.lowerBound)
            }
        }
        let modifierEquivalents: [Set<UInt16>] = [
            ActionLayout.shiftKeyCodes, [59, 62], [58, 61], [55, 54]
        ]
        if profile.channels.modifiers {
            for modifier in 1..<4 where restrictions.allowsModifier(modifier)
                && !keys.isDisjoint(with: modifierEquivalents[modifier]) {
                binary.append(ActionLayout.modifiers.lowerBound + modifier)
            }
        }

        var continuous: [Int] = []
        var absolute: Set<Int> = []
        if profile.channels.mouseMovement {
            switch mouseMode {
            case .absolute where actionSpace.allowsAbsoluteMouseMovement:
                continuous += ActionLayout.absoluteMouse
                absolute.formUnion(ActionLayout.absoluteMouse)
            case .relative where actionSpace.allowsRelativeMouseMovement:
                continuous += ActionLayout.relativeMouse
            default:
                break
            }
        }
        if profile.channels.scroll, actionSpace.allowsScroll {
            continuous += ActionLayout.scroll
        }
        binary = Array(Set(binary)).sorted()
        continuous = Array(Set(continuous)).sorted()
        guard !binary.isEmpty || !continuous.isEmpty else {
            throw AgentTrainerError.invalidConfiguration(
                "The RL allowlist has no actions left after the AI's channel switches and blocked-output restrictions are applied."
            )
        }
        return Self(
            binaryIndices: binary,
            continuousIndices: continuous,
            absoluteContinuousIndices: absolute,
            allowedKeyCodes: keys,
            allowedMouseButtons: buttons
        )
    }

    func sample(
        logits: [Float],
        standardDeviation: Float,
        binaryTemperature: Float,
        random: inout ReinforcementRandomGenerator
    ) throws -> ReinforcementPolicySample {
        guard logits.count >= ActionLayout.count,
              logits.prefix(ActionLayout.count).allSatisfy(\.isFinite) else {
            throw AgentTrainerError.model("The RL actor produced an invalid action distribution.")
        }
        let std = max(0.001, standardDeviation)
        let temperature = max(0.1, binaryTemperature)
        var distribution = [Float](repeating: 0, count: ActionLayout.count)
        var executed = [Float](repeating: 0, count: ActionLayout.count)
        var logProbability = 0.0
        for index in binaryIndices {
            let scaled = Double(logits[index] / temperature)
            let probability = Self.sigmoid(scaled)
            let action: Float = random.uniformUnit() < probability ? 1 : 0
            distribution[index] = action
            executed[index] = action
            logProbability += action > 0.5
                ? Self.logSigmoid(scaled) : Self.logSigmoid(-scaled)
        }
        let logNormalizer = Foundation.log(Double(std))
            + 0.5 * Foundation.log(2 * Double.pi)
        for index in continuousIndices {
            let mean = Double(logits[index])
            let latent = mean + Double(std) * random.standardNormal()
            let standardized = (latent - mean) / Double(std)
            logProbability += -0.5 * standardized * standardized - logNormalizer
            distribution[index] = Float(latent)
            if absoluteContinuousIndices.contains(index) {
                let bounded = Self.sigmoid(latent)
                executed[index] = Float(bounded)
                logProbability -= Foundation.log(max(1e-7, bounded * (1 - bounded)))
            } else {
                let bounded = Foundation.tanh(latent)
                executed[index] = Float(bounded)
                logProbability -= Foundation.log(max(1e-7, 1 - bounded * bounded))
            }
        }
        guard logProbability.isFinite else {
            throw AgentTrainerError.model("The RL actor produced a non-finite action probability.")
        }
        return ReinforcementPolicySample(
            distributionActions: distribution,
            executedActions: executed,
            logProbability: Float(logProbability)
        )
    }

    var binaryMask: [Float] {
        mask(indices: binaryIndices)
    }

    var continuousMask: [Float] {
        mask(indices: continuousIndices)
    }

    var absoluteContinuousMask: [Float] {
        mask(indices: Array(absoluteContinuousIndices))
    }

    private func mask(indices: [Int]) -> [Float] {
        var result = [Float](repeating: 0, count: ActionLayout.count)
        for index in indices where result.indices.contains(index) { result[index] = 1 }
        return result
    }

    private static func sigmoid(_ value: Double) -> Double {
        if value >= 0 { return 1 / (1 + Foundation.exp(-value)) }
        let exponential = Foundation.exp(value)
        return exponential / (1 + exponential)
    }

    private static func logSigmoid(_ value: Double) -> Double {
        value >= 0
            ? -Foundation.log1p(Foundation.exp(-value))
            : value - Foundation.log1p(Foundation.exp(value))
    }
}

struct ReinforcementRandomGenerator: RandomNumberGenerator, Sendable {
    var state: UInt64

    init(seed: UInt64) {
        state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
    }

    mutating func next() -> UInt64 {
        // xoshiro-style scrambling is ample for action sampling and the state
        // is compact enough to checkpoint exactly.
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }

    mutating func uniformUnit() -> Double {
        Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }

    mutating func standardNormal() -> Double {
        let u1 = max(Double.leastNonzeroMagnitude, uniformUnit())
        let u2 = uniformUnit()
        return Foundation.sqrt(-2 * Foundation.log(u1))
            * Foundation.cos(2 * Double.pi * u2)
    }
}

enum ReinforcementAdvantageMath {
    static func advantagesAndReturns(
        rewards: [Float],
        values: [Float],
        terminals: [Bool],
        bootstrapValue: Float,
        discount: Float,
        lambda: Float,
        normalize: Bool,
        epsilon: Float = 1e-5
    ) -> (advantages: [Float], returns: [Float]) {
        guard rewards.count == values.count,
              rewards.count == terminals.count,
              !rewards.isEmpty else { return ([], []) }
        var advantages = [Float](repeating: 0, count: rewards.count)
        var nextValue = bootstrapValue
        var generalizedAdvantage: Float = 0
        for index in rewards.indices.reversed() {
            let continuation: Float = terminals[index] ? 0 : 1
            let delta = rewards[index]
                + discount * nextValue * continuation - values[index]
            generalizedAdvantage = delta
                + discount * lambda * continuation * generalizedAdvantage
            advantages[index] = generalizedAdvantage
            nextValue = values[index]
        }
        let returns = zip(advantages, values).map(+)
        guard normalize, advantages.count > 1 else { return (advantages, returns) }
        let mean = advantages.reduce(0, +) / Float(advantages.count)
        let variance = advantages.reduce(Float(0)) {
            let centered = $1 - mean
            return $0 + centered * centered
        } / Float(advantages.count)
        let scale = Foundation.sqrt(max(0, variance) + epsilon)
        return (advantages.map { ($0 - mean) / scale }, returns)
    }

    static func explainedVariance(predictions: [Float], targets: [Float]) -> Double {
        guard predictions.count == targets.count, predictions.count > 1 else { return 0 }
        let targetMean = targets.reduce(0, +) / Float(targets.count)
        let residuals = zip(targets, predictions).map(-)
        let residualMean = residuals.reduce(0, +) / Float(residuals.count)
        let targetVariance = targets.reduce(Double(0)) {
            let value = Double($1 - targetMean)
            return $0 + value * value
        } / Double(targets.count)
        guard targetVariance > 1e-12 else { return 0 }
        let residualVariance = residuals.reduce(Double(0)) {
            let value = Double($1 - residualMean)
            return $0 + value * value
        } / Double(residuals.count)
        return 1 - residualVariance / targetVariance
    }
}

final class ReinforcementActorCritic: Module, @unchecked Sendable {
    let dtype: DType
    @ModuleInfo var actor: AgentPolicy
    @ModuleInfo var valueLayers: [Linear]
    @ModuleInfo var valueNormalizations: [LayerNorm]
    @ModuleInfo var valueHead: Linear

    init(
        actor: AgentPolicy,
        profile: AIProfile,
        configuration: ReinforcementLearningConfiguration,
        contextSchema: ModelContextSchema
    ) {
        self.actor = actor
        dtype = actor.dtype
        let architecture = profile.training.architecture
        var inputWidth = architecture.fusionWidths.last
            ?? (architecture.visualEmbedding + architecture.recurrentWidth
                + contextSchema.inputValueCount)
        var layers: [Linear] = []
        var normalizations: [LayerNorm] = []
        for _ in 0..<configuration.criticLayerCount {
            layers.append(Linear(max(1, inputWidth), configuration.criticHiddenWidth))
            normalizations.append(LayerNorm(dimensions: configuration.criticHiddenWidth))
            inputWidth = configuration.criticHiddenWidth
        }
        valueLayers = layers
        valueNormalizations = normalizations
        valueHead = Linear(max(1, inputWidth), 1)
        super.init()
        if dtype != .float32 {
            update(parameters: mapParameters { $0.asType(self.dtype) })
        }
    }

    func value(fusedFeatures: MLXArray) -> MLXArray {
        var hidden = fusedFeatures
        for (index, layer) in valueLayers.enumerated() {
            hidden = layer(hidden)
            if valueNormalizations.indices.contains(index) {
                hidden = valueNormalizations[index](hidden)
            }
            hidden = silu(hidden)
        }
        return valueHead(hidden).reshaped([-1]).asType(.float32)
    }

    func saveWeights(to url: URL) throws {
        try MLX.save(
            arrays: Dictionary(uniqueKeysWithValues: parameters().flattened()),
            metadata: ["format": "AgentTrainer.PPO.ActorCritic.v1"],
            url: url
        )
    }

    func loadWeights(from url: URL) throws {
        let arrays = try MLX.loadArrays(url: url)
        try update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
    }
}

/// Serial session-owned PPO graph. Capture/injection remains responsive while
/// updates run because the live runtime pauses action admission and performs
/// this work on its dedicated MLX queue.
final class ReinforcementPolicyOptimizer: @unchecked Sendable {
    private typealias InferenceFunction = @Sendable ([MLXArray]) -> [MLXArray]

    let profile: AIProfile
    let configuration: ReinforcementLearningConfiguration
    let contextSchema: ModelContextSchema
    let actionPlan: ReinforcementActionPlan
    let model: ReinforcementActorCritic
    let optimizer: ResumableAdamW
    let randomState: MLXRandom.RandomState
    private let inferenceFunction: InferenceFunction
    private let trainingFunction: InferenceFunction

    init(
        profile: AIProfile,
        configuration: ReinforcementLearningConfiguration,
        contextSchema: ModelContextSchema,
        actionPlan: ReinforcementActionPlan,
        actorWeightsURL: URL?,
        reinforcementWeightsURL: URL? = nil,
        optimizerURL: URL? = nil,
        randomStateURL: URL? = nil
    ) throws {
        self.profile = profile
        self.configuration = configuration
        self.contextSchema = contextSchema
        self.actionPlan = actionPlan
        let actor = AgentPolicy(profile: profile, contextSchema: contextSchema)
        if let actorWeightsURL { try actor.loadWeights(from: actorWeightsURL) }
        model = ReinforcementActorCritic(
            actor: actor,
            profile: profile,
            configuration: configuration,
            contextSchema: contextSchema
        )
        if let reinforcementWeightsURL {
            try model.loadWeights(from: reinforcementWeightsURL)
        }
        optimizer = ResumableAdamW(
            learningRate: Float(configuration.learningRate),
            weightDecay: Float(configuration.weightDecay),
            warmupSteps: 1,
            schedule: .adaptiveCosine,
            cycleSteps: 10_000,
            minimumLearningRateRatio: 0.5
        )
        optimizer.initialize(model: model)
        if let optimizerURL { try optimizer.load(from: optimizerURL) }
        randomState = MLXRandom.RandomState(seed: configuration.seed)
        if let randomStateURL {
            try TrainingRandomState.load(randomState, from: randomStateURL)
        }

        let temporal = profile.training.effectiveTemporalVision
        let pastSpec = temporal.pastFrameSpec(from: profile.preprocessing)
        let hasTemporal = temporal.pastFrameCount > 0
        let hasContext = !contextSchema.isEmpty
        let actorCritic = model
        let ppoOptimizer = optimizer
        inferenceFunction = compile(inputs: [actorCritic]) { arrays in
            let current = VisionPreprocessor.mlxTensor(arrays[0], spec: profile.preprocessing)
            let temporalFeatures: MLXArray
            var cursor = 1
            if hasTemporal {
                temporalFeatures = actorCritic.actor.temporalFeatures(
                    currentImages: current,
                    pastImages: VisionPreprocessor.mlxPastFrameTensor(arrays[cursor], spec: pastSpec),
                    pastControls: arrays[cursor + 1]
                )
                cursor += 2
            } else {
                temporalFeatures = actorCritic.actor.currentOnlyTemporalFeatures(currentImages: current)
            }
            let fused = actorCritic.actor.fusedFeatures(
                temporalFeatures: temporalFeatures,
                contextValues: hasContext ? arrays[cursor] : nil
            )
            return [actorCritic.actor.logits(fusedFeatures: fused), actorCritic.value(fusedFeatures: fused)]
        }

        let binaryMask = MLXArray(actionPlan.binaryMask, [1, ActionLayout.count])
        let continuousMask = MLXArray(actionPlan.continuousMask, [1, ActionLayout.count])
        let absoluteContinuousMask = MLXArray(
            actionPlan.absoluteContinuousMask,
            [1, ActionLayout.count]
        )
        let signedContinuousMask = continuousMask - absoluteContinuousMask
        let activeActionCount = Float(max(1, actionPlan.actionCount))
        trainingFunction = compile(
            inputs: [actorCritic, ppoOptimizer, randomState],
            outputs: [actorCritic, ppoOptimizer, randomState]
        ) { arrays in
            var cursor = 1
            if hasTemporal {
                cursor += 2
            }
            let contextIndex: Int? = hasContext ? cursor : nil
            if hasContext { cursor += 1 }
            let actionIndex = cursor
            let oldLogProbabilityIndex = cursor + 1
            let returnIndex = cursor + 2
            let advantageIndex = cursor + 3
            let oldValueIndex = cursor + 4
            let result = valueAndGrad(model: actorCritic) { model, differentiableInputs in
                let current = VisionPreprocessor.mlxTensor(
                    differentiableInputs[0],
                    spec: profile.preprocessing
                )
                let temporalFeatures: MLXArray
                if hasTemporal {
                    temporalFeatures = model.actor.temporalFeatures(
                        currentImages: current,
                        pastImages: VisionPreprocessor.mlxPastFrameTensor(
                            differentiableInputs[1],
                            spec: pastSpec
                        ),
                        pastControls: differentiableInputs[2]
                    )
                } else {
                    temporalFeatures = model.actor.currentOnlyTemporalFeatures(
                        currentImages: current
                    )
                }
                let fused = model.actor.fusedFeatures(
                    temporalFeatures: temporalFeatures,
                    contextValues: contextIndex.map { differentiableInputs[$0] }
                )
                let logits = model.actor.logits(fusedFeatures: fused).asType(.float32)
                let values = model.value(fusedFeatures: fused)
                let actions = differentiableInputs[actionIndex].asType(.float32)
                let scaledBinaryLogits = logits / Float(configuration.binaryActionTemperature)
                let binaryLogProbability = -binaryCrossEntropy(
                    logits: scaledBinaryLogits,
                    targets: actions * binaryMask,
                    reduction: .none
                )
                let standardDeviation = Float(configuration.continuousActionStandardDeviation)
                let normalized = (actions - logits) / standardDeviation
                let normalLogProbability = -0.5 * square(normalized)
                    - Foundation.log(Double(standardDeviation))
                    - 0.5 * Foundation.log(2 * Double.pi)
                let absoluteActions = sigmoid(actions)
                let signedActions = tanh(actions)
                let transformLogJacobian = log(
                    absoluteActions * (1 - absoluteActions) + 1e-7
                ) * absoluteContinuousMask + log(
                    1 - square(signedActions) + 1e-7
                ) * signedContinuousMask
                let newLogProbability = (
                    binaryLogProbability * binaryMask
                        + (normalLogProbability - transformLogJacobian)
                            * continuousMask
                ).sum(axis: -1)
                let oldLogProbability = differentiableInputs[oldLogProbabilityIndex].asType(.float32)
                let advantages = differentiableInputs[advantageIndex].asType(.float32)
                let logRatio = clip(
                    newLogProbability - oldLogProbability,
                    min: -20,
                    max: 20
                )
                let ratio = exp(logRatio)
                let clippedRatio = clip(
                    ratio,
                    min: 1 - Float(configuration.policyClipRange),
                    max: 1 + Float(configuration.policyClipRange)
                )
                let policyLoss = -minimum(ratio * advantages, clippedRatio * advantages).mean()

                let returns = differentiableInputs[returnIndex].asType(.float32)
                let oldValues = differentiableInputs[oldValueIndex].asType(.float32)
                let rawValueLoss = square(values - returns)
                let valueLoss: MLXArray
                if configuration.clipValueLoss {
                    let clippedValues = oldValues + clip(
                        values - oldValues,
                        min: -Float(configuration.policyClipRange),
                        max: Float(configuration.policyClipRange)
                    )
                    valueLoss = 0.5 * maximum(
                        rawValueLoss,
                        square(clippedValues - returns)
                    ).mean()
                } else {
                    valueLoss = 0.5 * rawValueLoss.mean()
                }

                let probabilities = sigmoid(scaledBinaryLogits)
                let binaryEntropy = -(
                    probabilities * log(probabilities + 1e-7)
                        + (1 - probabilities) * log(1 - probabilities + 1e-7)
                )
                let normalEntropy = MLXArray.full(
                    logits.shape,
                    values: MLXArray(
                        Float(0.5 * Foundation.log(2 * Double.pi * Double(standardDeviation * standardDeviation)) + 0.5),
                        dtype: .float32
                    )
                )
                let entropy = ((binaryEntropy * binaryMask + normalEntropy * continuousMask)
                    .sum(axis: -1) / activeActionCount).mean()
                let total = policyLoss
                    + Float(configuration.valueLossCoefficient) * valueLoss
                    - Float(configuration.entropyCoefficient) * entropy
                let approximateKL = (exp(logRatio) - 1 - logRatio).mean()
                let clipFraction = (abs(ratio - 1) .> Float(configuration.policyClipRange))
                    .asType(.float32).mean()
                return [total, policyLoss, valueLoss, entropy, approximateKL, clipFraction]
            }(actorCritic, arrays)
            ppoOptimizer.update(
                model: actorCritic,
                gradients: result.1,
                targetType: actorCritic.dtype,
                gradientNorm: ResumableAdamW.globalGradientNorm(result.1),
                maxGradientNorm: Float(configuration.maximumGradientNorm)
            )
            return result.0
        }
    }

    func prediction(for observation: ReinforcementObservation) throws -> (logits: [Float], value: Float) {
        model.train(false)
        let arrays = try observationArrays([observation])
        let outputs = inferenceFunction(arrays)
        MLX.eval(outputs)
        let logits = outputs[0].asArray(Float.self)
        guard logits.count >= ActionLayout.count,
              let value = outputs[1].asArray(Float.self).first,
              value.isFinite else {
            throw AgentTrainerError.model("The RL actor-critic produced an invalid prediction.")
        }
        return (Array(logits.prefix(ActionLayout.count)), value)
    }

    func update(
        transitions: [ReinforcementTransition],
        bootstrapValue: Float,
        shuffleSeed: UInt64
    ) throws -> ReinforcementUpdateMetrics {
        guard transitions.count >= 4 else { return ReinforcementUpdateMetrics() }
        let rewards = transitions.map(\.reward)
        let oldValues = transitions.map(\.oldValue)
        let (advantages, returns) = ReinforcementAdvantageMath.advantagesAndReturns(
            rewards: rewards,
            values: oldValues,
            terminals: transitions.map(\.terminal),
            bootstrapValue: bootstrapValue,
            discount: Float(configuration.discountFactor),
            lambda: Float(configuration.generalizedAdvantageLambda),
            normalize: configuration.normalizeAdvantages,
            epsilon: Float(configuration.rewardNormalizationEpsilon)
        )
        var random = ReinforcementRandomGenerator(seed: shuffleSeed)
        var aggregate = ReinforcementUpdateMetrics()
        var weightedCount = 0
        // PPO must evaluate the exact policy that produced each stored log
        // probability. Supervised dropout remains unchanged, but is disabled
        // here so stochastic masks cannot masquerade as policy divergence.
        model.train(false)
        updateLoop: for _ in 0..<configuration.updateEpochs {
            var order = Array(transitions.indices)
            if order.count > 1 {
                for index in stride(from: order.count - 1, through: 1, by: -1) {
                    order.swapAt(index, Int(random.next() % UInt64(index + 1)))
                }
            }
            for start in stride(from: 0, to: order.count, by: configuration.miniBatchSize) {
                let indices = Array(order[start..<min(order.count, start + configuration.miniBatchSize)])
                let observations = indices.map { transitions[$0].observation }
                var arrays = try observationArrays(observations)
                arrays.append(MLXArray(floatData(indices.flatMap { transitions[$0].distributionActions }), [indices.count, ActionLayout.count], type: Float.self))
                arrays.append(MLXArray(indices.map { transitions[$0].oldLogProbability }))
                arrays.append(MLXArray(indices.map { returns[$0] }))
                arrays.append(MLXArray(indices.map { advantages[$0] }))
                arrays.append(MLXArray(indices.map { transitions[$0].oldValue }))
                let outputs = withRandomState(randomState) { trainingFunction(arrays) }
                MLX.eval(outputs, model.parameters(), optimizer.stateArrays())
                let values = outputs.map { Double($0.item(Float.self)) }
                guard values.count >= 6, values.allSatisfy(\.isFinite) else {
                    throw AgentTrainerError.model(
                        "RL training became numerically unstable. Lower the learning rate, reward scale, entropy, or action standard deviation."
                    )
                }
                let count = indices.count
                aggregate.optimizerSteps += 1
                aggregate.policyLoss += values[1] * Double(count)
                aggregate.valueLoss += values[2] * Double(count)
                aggregate.entropy += values[3] * Double(count)
                aggregate.approximateKL += values[4] * Double(count)
                aggregate.clipFraction += values[5] * Double(count)
                weightedCount += count
                if configuration.targetKLDivergence > 0,
                   values[4] > configuration.targetKLDivergence * 1.5 {
                    break updateLoop
                }
            }
        }
        model.train(false)
        if weightedCount > 0 {
            let divisor = Double(weightedCount)
            aggregate.policyLoss /= divisor
            aggregate.valueLoss /= divisor
            aggregate.entropy /= divisor
            aggregate.approximateKL /= divisor
            aggregate.clipFraction /= divisor
        }
        aggregate.explainedVariance = ReinforcementAdvantageMath.explainedVariance(
            predictions: oldValues,
            targets: returns
        )
        return aggregate
    }

    func saveActor(to url: URL) throws { try model.actor.saveWeights(to: url) }
    func saveReinforcementWeights(to url: URL) throws { try model.saveWeights(to: url) }
    func saveOptimizer(to url: URL) throws { try optimizer.save(to: url) }
    func saveRandomState(to url: URL) throws { try TrainingRandomState.save(randomState, to: url) }

    private func observationArrays(_ observations: [ReinforcementObservation]) throws -> [MLXArray] {
        guard !observations.isEmpty else { throw AgentTrainerError.model("RL received an empty observation batch.") }
        let count = observations.count
        let currentBytes = profile.preprocessing.sampleByteCount
        let temporal = profile.training.effectiveTemporalVision
        let pastSpec = temporal.pastFrameSpec(from: profile.preprocessing)
        let expectedPastBytes = temporal.pastFrameCount * pastSpec.sampleByteCount
        let expectedPastControls = temporal.pastFrameCount * ActionLayout.count
        let contextCount = contextSchema.inputValueCount
        guard observations.allSatisfy({
            $0.currentPacked.count == currentBytes
                && $0.pastPacked.count == expectedPastBytes
                && $0.pastControls.count == expectedPastControls
                && $0.contextValues.count == contextCount
        }) else {
            throw AgentTrainerError.model("An RL rollout observation does not match this model's immutable vision/context contract.")
        }
        var current = Data(capacity: count * currentBytes)
        observations.forEach { current.append($0.currentPacked) }
        var result = [MLXArray(current, [count, currentBytes], dtype: .uint8)]
        if temporal.pastFrameCount > 0 {
            var past = Data(capacity: count * expectedPastBytes)
            observations.forEach { past.append($0.pastPacked) }
            result.append(MLXArray(
                past,
                [count, temporal.pastFrameCount, pastSpec.sampleByteCount],
                dtype: .uint8
            ))
            result.append(MLXArray(
                floatData(observations.flatMap(\.pastControls)),
                [count, temporal.pastFrameCount, ActionLayout.count],
                type: Float.self
            ))
        }
        if contextCount > 0 {
            result.append(MLXArray(
                floatData(observations.flatMap(\.contextValues)),
                [count, contextCount],
                type: Float.self
            ))
        }
        return result
    }

    private func floatData(_ values: [Float]) -> Data {
        values.withUnsafeBytes { Data($0) }
    }
}
