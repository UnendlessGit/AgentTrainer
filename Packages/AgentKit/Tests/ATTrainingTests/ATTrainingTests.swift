import ATCore
import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import Testing
@testable import ATTraining

@Suite(.serialized) struct ATTrainingTests {
    @Test func fusedAdamWMatchesMLXAdamW() {
        let a = Linear(16, 8), b = Linear(16, 8), copy = Linear(16, 8)
        b.update(parameters: a.parameters())
        let ref = AdamW(learningRate: 1e-3, betas: (0.9, 0.999), eps: 1e-8, weightDecay: 1e-4)
        let mine = FusedAdamW(learningRate: 1e-3, betas: (0.9, 0.999), eps: 1e-8, weightDecay: 1e-4, computeType: .float32)
        for i in 0 ..< 4 {
            let g = ModuleParameters.unflattened(a.trainableParameters().flattened().map { ($0.0, MLXRandom.normal($0.1.shape) * Float(i + 1)) })
            ref.learningRate = 1e-3 * Float(i + 1); mine.learningRate = ref.learningRate
            ref.update(model: a, gradients: clipGradNorm(gradients: g, maxNorm: 1).0)
            _ = mine.update(master: b, compute: copy, gradients: g, clip: 1)
            eval(a, b, copy, ref)
        }
        for (x, y) in zip(a.parameters().flattened(), b.parameters().flattened()) {
            #expect(abs(x.1 - y.1).max().item(Float.self) < 1e-6)
        }
        // Checkpoints written by MLX's AdamW load into the fused one.
        let loaded = FusedAdamW(learningRate: 1e-3, betas: (0.9, 0.999), eps: 1e-8, weightDecay: 1e-4, computeType: .float32)
        loaded.update(parameters: ref.state())
        #expect(loaded.stepCount == 4)
        #expect(loaded.state().flattened().count == ref.state().flattened().count)
    }

    @Test func metalAugmentationsMatchReference() {
        let frames = MLXRandom.randInt(low: 0, high: 256, [4, 32, 32, 3]).asType(.uint8)
        let aug = Augmenter([.spatial], strength: 1)
        let ref = clip(aug.warp(frames.asType(.float32), angle: 0.04, scale: 1.03, tx: 1.5, ty: -0.7), min: 0, max: 255).round()
        let mine = Augmenter.warpUInt8(frames, angle: 0.04, scale: 1.03, tx: 1.5, ty: -0.7, final: true, dtype: .float32)
        let diff = abs(ref - mine)
        #expect(diff.max().item(Float.self) <= 1)
        #expect((diff .> 0).asType(.float32).mean().item(Float.self) < 0.002)
        let f = frames.asType(.float32)
        let cref = aug.colorJitter(f, brightness: 1.1, contrast: 0.9, saturation: 1.2, hue: 0.03) * MLXArray([Float(1.05), 0.97, 0.9])
        let cm = Augmenter.color(f, brightness: 1.1, contrast: 0.9, saturation: 1.2, hue: 0.03, gains: [1.05, 0.97, 0.9], jitter: true)
        #expect(abs(cref - cm).max().item(Float.self) < 0.01)
        #expect(abs(aug.blur(f, sigma: 1.3) - Augmenter.fastBlur(f, sigma: 1.3)).max().item(Float.self) < 0.01)
    }
}
