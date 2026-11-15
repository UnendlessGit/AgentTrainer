import ATCore
import ATModel
import ATTraining
import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// Measures training-step time and peak memory on random clips.
enum TrainBenchCommand {
    static func run(_ args: [String]) throws {
        let o = Options(args)
        var arch = PolicyArchitecture.p2p150M
        arch.layers = o.int("--layers", 10)
        arch.memorySeconds = o.double("--seconds", 10)
        let T = arch.contextSteps, B = o.int("--batch", 1)
        let actions = ActionSpaceConfig.gameDefault
        let model = PolicyModel(architecture: arch, actions: actions)
        if o.flag("--bf16") { model.convert(to: .bfloat16) }
        model.setTraining()
        model.gradientCheckpointing = !o.flag("--no-ckpt")
        let opt = AdamW(learningRate: 1e-4, betas: (0.9, 0.999), weightDecay: 1e-4)
        let counts = actions.slotClassCounts
        func batch() -> TrainingBatch {
            let frames = MLXRandom.randInt(low: 0, high: 256, [B, T, 192, 192, 3]).asType(.uint8)
            let acts = stacked((0 ..< counts.count).map { MLXRandom.randInt(low: 0, high: counts[$0], [B, T]).asType(.int32) }, axis: -1)
            return TrainingBatch(frames: frames, actions: acts, lossMask: MLXArray.ones([B, T], dtype: .bool),
                                 text: MLXArray.zeros([B, T, 768]), textPresent: MLXArray.zeros([B, T], dtype: .bool))
        }
        func lossFn(_ m: PolicyModel, _ b: TrainingBatch) -> MLXArray {
            let logits = m(frames: b.frames, actionsIn: b.actions, text: b.text, textPresent: b.textPresent)
            return PolicyLoss.compute(logits, actions: b.actions, mask: b.lossMask, space: actions, zWeight: 1e-4).total
        }
        let vg = valueAndGrad(model: model) { (m: PolicyModel, b: [TrainingBatch]) -> [MLXArray] in [lossFn(m, b[0])] }
        print("training \(arch.layers)L × \(arch.width), clip \(T) steps (\(T * model.layout.tokensPerStep) tokens), batch \(B), dtype \(model.computeDType)")
        GPU.resetPeakMemory()
        var times: [Double] = []
        for i in 0 ..< o.int("--steps", 4) {
            let b = batch()
            eval(b.frames, b.actions)
            let t0 = Date()
            let (loss, grads) = vg(model, [b])
            opt.update(model: model, gradients: grads)
            eval(model, opt, loss)
            let dt = Date().timeIntervalSince(t0)
            if i > 0 { times.append(dt) }
            print(String(format: "step %d: loss %.4f  %.2f s  peak %.1f GB", i, loss[0].item(Float.self), dt, Double(GPU.peakMemory) / 1e9))
        }
        let mean = times.reduce(0, +) / Double(max(times.count, 1))
        print(String(format: "mean %.2f s/step → %.0f frames/s (%.1f s of gameplay per second at %d Hz)", mean, Double(B * T) / mean, Double(B * T) / mean / Double(arch.stepRateHz), arch.stepRateHz))
    }
}
