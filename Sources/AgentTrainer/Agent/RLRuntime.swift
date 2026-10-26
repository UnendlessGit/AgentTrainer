import Foundation
import AppKit
import MLX

struct RLRuntimeOptions: Sendable {
    var initial: RLWeightSnapshot
    var random: RLRandom
    var distribution: RLActionDistribution
    var budget: RLComputeBudget? = nil
    var mask: CGRect
    var onDecision: @Sendable (RLDecision) -> Void
}

/// Shared by production inference and the compiled-graph regression test.
enum RuntimePolicyGraph {
    static func make(model: AgentPolicy, heads: RLHeads?) -> @Sendable ([MLXArray]) -> [MLXArray] {
        let hasHistory = model.profile.training.effectiveTemporalVision.pastFrameCount > 0
        let hasContext = !model.contextSchema.isEmpty
        let timingIndex = hasHistory ? 3 : 1
        let contextIndex = hasHistory ? 5 : 2
        let state: [any Updatable] = [model] + (heads.map { [$0 as any Updatable] } ?? [])
        return compile(inputs: state) { inputs in
            let temporal = model.encodedTemporalFeatures(current: inputs[0], past: hasHistory ? inputs[1] : nil,
                controls: hasHistory ? inputs[2] : nil, currentTiming: inputs[timingIndex],
                pastTiming: hasHistory ? inputs[4] : nil)
            let fused = model.fusedFeatures(temporalFeatures: temporal,
                contextValues: hasContext ? inputs[contextIndex] : nil)
            let logits = model.logits(fusedFeatures: fused)
            if let heads {
                // Returning a parameter object directly aliases the compiler's
                // state tracer, which is restored after tracing. Materialize an
                // expression so the output retains its captured dependency.
                return [model.activatedPredictions(logits: logits), logits.asType(.float32),
                    heads.value(model.decisionFeatures(fused)), heads.logStandardDeviation + Float(0)]
            }
            return [model.activatedPredictions(logits: logits)]
        }
    }
}

/// Re-encode only selected historical images after a policy revision. Distant
/// ring slots retain raw pixels and are refreshed lazily when selected later.
enum RLPastEncoder {
    static func encode(_ frames: [Data], model: AgentPolicy) -> [MLXArray] {
        guard !frames.isEmpty else { return [] }
        var unique: [Data] = [], indices: [Int] = [], lookup: [Data: Int] = [:]
        for frame in frames {
            if let index = lookup[frame] { indices.append(index) }
            else { lookup[frame] = unique.count; indices.append(unique.count); unique.append(frame) }
        }
        let spec = model.profile.training.effectiveTemporalVision.pastFrameSpec(from: model.profile.preprocessing)
        let packed = unique.reduce(into: Data()) { $0.append($1) }
        let images = VisionPreprocessor.mlxTensor(packed, batch: unique.count, spec: spec)
        let encoded = model.pastVisualEmbedding(visualFeatures: model.pastVisualActivations(images: images).last!)
        MLX.eval(encoded)
        return indices.map { encoded[$0..<($0 + 1)] }
    }
}
