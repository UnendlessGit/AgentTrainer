import Foundation
import MLX
import MLXNN

final class RLHeads: Module, @unchecked Sendable {
    @ModuleInfo var critic: Linear
    @ParameterInfo var logStandardDeviation: MLXArray
    init(width: Int) {
        critic = Linear(width, 1)
        logStandardDeviation = MLXArray([Float](arrayLiteral: -2.3, -2.3, -2.5, -2.5, -3.2, -3.2, -1.05, -1.05))
        super.init()
        critic.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray.zeros([1, width]), "bias": MLXArray.zeros([1])]))
    }
    func value(_ features: MLXArray) -> MLXArray { critic(features.asType(.float32)).squeezed(axis: -1) }
}

/// The wrapper is used only by RL. Ordinary saved policy names stay unchanged.
final class RLTrainablePolicy: Module, @unchecked Sendable {
    @ModuleInfo var policy: AgentPolicy
    @ModuleInfo var heads: RLHeads
    init(policy: AgentPolicy) {
        self.policy = policy; heads = RLHeads(width: policy.fusionWidth)
        super.init(); train(false)
    }

    func forward(_ observation: RLObservation) -> (MLXArray, MLXArray) { forward(inputs: inputs([observation])) }

    func inputs(_ observations: [RLObservation]) -> [MLXArray] {
        let n = observations.count, p = policy.profile
        let count = p.training.effectiveTemporalVision.pastFrameCount
        let pastSpec = p.training.effectiveTemporalVision.pastFrameSpec(from: p.preprocessing)
        let current = observations.reduce(into: Data()) { $0.append($1.current) }
        let past = observations.reduce(into: Data()) { bytes, obs in for frame in obs.past { bytes.append(frame) } }
        return [MLXArray(current, [n, p.preprocessing.sampleByteCount], type: UInt8.self),
            MLXArray(observations.flatMap(\.timing), [n, TemporalInputContract.valueCount]),
            MLXArray(past, [n, count, pastSpec.sampleByteCount], type: UInt8.self),
            MLXArray(observations.flatMap(\.controls), [n, count, ActionLayout.count]),
            MLXArray(observations.flatMap(\.pastTiming), [n, count, TemporalInputContract.valueCount]),
            MLXArray(observations.flatMap(\.context), [n, policy.contextSchema.inputValueCount])]
    }

    func forward(inputs: [MLXArray]) -> (MLXArray, MLXArray) {
        let p = policy.profile, count = p.training.effectiveTemporalVision.pastFrameCount
        let pastSpec = p.training.effectiveTemporalVision.pastFrameSpec(from: p.preprocessing)
        let image = VisionPreprocessor.mlxTensor(inputs[0], spec: p.preprocessing)
        let temporal: MLXArray
        if count == 0 { temporal = policy.currentOnlyTemporalFeatures(currentImages: image, currentTiming: inputs[1]) }
        else {
            let past = VisionPreprocessor.mlxPastFrameTensor(inputs[2], spec: pastSpec)
            temporal = policy.temporalFeatures(currentImages: image, pastImages: past,
                pastControls: inputs[3], currentTiming: inputs[1], pastTiming: inputs[4])
        }
        let fused = policy.fusedFeatures(temporalFeatures: temporal,
            contextValues: policy.contextSchema.isEmpty ? nil : inputs[5])
        return (policy.logits(fusedFeatures: fused).asType(.float32), heads.value(policy.decisionFeatures(fused)))
    }

}

struct RLOwnedTensor: Codable, Sendable {
    var shape: [Int]
    var values: [Float]
}

struct RLWeightSnapshot: Sendable {
    var policy: [String: RLOwnedTensor]
    var heads: [String: RLOwnedTensor]
    var revision: Int
}

enum RLWeightTransfer {
    static func copy(_ module: Module) -> [String: RLOwnedTensor] {
        MLX.eval(module)
        return Dictionary(uniqueKeysWithValues: module.parameters().flattened().map {
            ($0.0, RLOwnedTensor(shape: $0.1.shape, values: $0.1.asArray(Float.self)))
        })
    }
    static func install(_ tensors: [String: RLOwnedTensor], in module: Module) throws {
        let old = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
        guard tensors.count == old.count else { throw AgentTrainerError.model("Incomplete RL weight snapshot.") }
        var arrays: [String: MLXArray] = [:]
        for (key, tensor) in tensors {
            guard let parameter = old[key], tensor.shape == parameter.shape,
                  tensor.values.count == parameter.size, tensor.values.allSatisfy(\.isFinite) else {
                throw AgentTrainerError.model("An RL weight snapshot is incompatible or non-finite.")
            }
            arrays[key] = MLXArray(tensor.values, tensor.shape).asType(parameter.dtype)
        }
        try module.update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
        MLX.eval(module)
    }
}

struct RLActionDistribution: Sendable {
    var binaryIndices: [Int]
    var continuousIndices: [Int]
    var configuration: RLConfiguration
    static let latentIndices = [0, 1, 2, 3, 12, 13, 146, 147]

    init(profile: AIProfile, configuration: RLConfiguration, mouseMode: MouseControlMode) {
        self.configuration = configuration
        let caps = configuration.capabilities(profile: profile, mouseMode: mouseMode)
        binaryIndices = []
        if profile.channels.keyboard {
            binaryIndices += caps.allowedKeyCodes.filter { !ActionLayout.allModifierKeyCodes.contains($0) }.map { 14 + Int($0) }
            if !caps.allowedKeyCodes.isDisjoint(with: [56, 60]) { binaryIndices.append(142) }
        }
        if profile.channels.modifiers {
            for (i, keys) in [[59, 62], [58, 61], [55, 54]].enumerated()
                where !caps.allowedKeyCodes.isDisjoint(with: keys.map(UInt16.init)) { binaryIndices.append(143 + i) }
        }
        if profile.channels.buttons { binaryIndices += caps.allowedMouseButtons.map { 4 + Int($0) } }
        binaryIndices.sort()
        continuousIndices = []
        if caps.allowsAbsoluteMouseMovement { continuousIndices += [0, 1] }
        if caps.allowsRelativeMouseMovement { continuousIndices += [2, 3, 6, 7] }
        if caps.allowsScroll { continuousIndices += [4, 5] }
    }

    var isEmpty: Bool { binaryIndices.isEmpty && continuousIndices.isEmpty }

    func sample(logits: [Float], logStd: [Float], random: inout RLRandom) -> (prediction: [Float], latent: [Float], binary: [Float], logProbability: Float) {
        var latent = [Float](repeating: 0, count: 8), binary = [Float](repeating: 0, count: ActionLayout.count)
        var raw = logits, logP: Float = 0
        for i in binaryIndices {
            let p = probability(logits[i])
            binary[i] = random.uniform() < p ? 1 : 0
            logP += log(binary[i] > 0 ? p : 1 - p)
        }
        for slot in continuousIndices {
            let index = Self.latentIndices[slot], mean = logits[index]
            let scale = exp(min(0.5, max(-5, logStd[slot]))) * Float(configuration.explorationScale)
            let noise = random.normal()
            latent[slot] = mean + noise * scale; raw[index] = latent[slot]
            logP += -0.5 * noise * noise - log(scale) - 0.9189385332
        }
        var prediction = binary
        if continuousIndices.contains(0) { prediction[0] = sigmoid(raw[0]); prediction[1] = sigmoid(raw[1]) }
        if continuousIndices.contains(2) {
            for axis in 0..<2 {
                let confidence = min(1, max(0, 0.5 + 0.5 * raw[146 + axis]))
                prediction[2 + axis] = tanh(raw[2 + axis]) * confidence * confidence * (3 - 2 * confidence)
            }
        }
        if continuousIndices.contains(4) { prediction[12] = tanh(raw[12]); prediction[13] = tanh(raw[13]) }
        return (prediction, latent, binary, logP)
    }

    private func sigmoid(_ x: Float) -> Float { x >= 0 ? 1 / (1 + exp(-x)) : exp(x) / (1 + exp(x)) }
    func probability(_ logit: Float) -> Float {
        let e = Float(configuration.binaryExploration)
        return min(1 - 1e-6, max(1e-6, (1 - e) * sigmoid(logit) + e * 0.5))
    }

    /// Densities are evaluated in the stored latent action space. The fixed
    /// execution decoder (including zero-motion gates and clipping) is part of
    /// the environment map, so no invalid inverse/Jacobian is needed.
    func statistics(logits: MLXArray, logStd: MLXArray, decision: RLDecision) -> (MLXArray, MLXArray) {
        statistics(logits: logits, logStd: logStd,
            latent: MLXArray(decision.latent, [1, 8]), binary: MLXArray(decision.binary, [1, ActionLayout.count]))
    }

    func statistics(logits: MLXArray, logStd: MLXArray, latent: MLXArray, binary: MLXArray) -> (MLXArray, MLXArray) {
        let rows = perRowStatistics(logits: logits, logStd: logStd, latent: latent, binary: binary)
        return (rows.0.sum(), rows.1.mean())
    }

    func perRowStatistics(logits: MLXArray, logStd: MLXArray, latent: MLXArray, binary: MLXArray) -> (MLXArray, MLXArray) {
        var logP = MLXArray.zeros([logits.dim(0)]), entropy = MLXArray.zeros([logits.dim(0)])
        if !binaryIndices.isEmpty {
            let indices = MLXArray(binaryIndices.map(Int32.init))
            let p = clip((1 - Float(configuration.binaryExploration)) * MLXNN.sigmoid(logits.take(indices, axis: -1))
                + Float(configuration.binaryExploration) * 0.5, min: 1e-6, max: 1 - 1e-6)
            let actions = binary.take(indices, axis: -1)
            logP = (actions * log(p) + (1 - actions) * log(1 - p)).sum(axis: -1)
            entropy = -(p * log(p) + (1 - p) * log(1 - p)).sum(axis: -1)
        }
        if !continuousIndices.isEmpty {
            let mean = logits.take(MLXArray(continuousIndices.map { Int32(Self.latentIndices[$0]) }), axis: -1)
            let logScale = clip(logStd.take(MLXArray(continuousIndices.map(Int32.init)), axis: -1), min: -5, max: 0.5)
                + log(Float(configuration.explorationScale))
            let actions = latent.take(MLXArray(continuousIndices.map(Int32.init)), axis: -1)
            logP = logP + (-0.5 * square((actions - mean) / exp(logScale)) - logScale - 0.9189385332).sum(axis: -1)
            entropy = entropy + (logScale + 1.4189385332).sum()
        }
        return (logP, entropy / Float(max(1, binaryIndices.count + continuousIndices.count)))
    }
}
