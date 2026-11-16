import ATCore
import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import ATModel

/// The training-speed rewrites must compute the same thing as the straightforward versions.
@Suite(.serialized) struct TrainingSpeedTests {
    @Test func depthwiseBackwardMatchesMLX() {
        for (k, st, h, c) in [(3, 1, 11, 4), (3, 2, 11, 4), (5, 2, 12, 3), (5, 1, 10, 3)] {
            let x = MLXRandom.normal([2, h, h + 1, c]), w = MLXRandom.normal([c, k, k, 1])
            let p = (k - 1) / 2
            let r = MLXRandom.normal(conv2d(x, w, stride: .init(st), padding: .init(p), groups: c).shape)
            func loss(_ a: [MLXArray]) -> MLXArray { (DepthwiseConv.apply(a[0], a[1], stride: st, padding: p) * r).sum() }
            DepthwiseConv.customBackward = false
            let ref = grad(loss, argumentNumbers: [0, 1])([x, w])
            DepthwiseConv.customBackward = true
            let mine = grad(loss, argumentNumbers: [0, 1])([x, w])
            for i in 0 ..< 2 {
                let rel = abs(ref[i] - mine[i]).max().item(Float.self) / max(abs(ref[i]).max().item(Float.self), 1e-6)
                #expect(rel < 1e-5, "k\(k) s\(st) gradient \(i): \(rel)")
            }
        }
    }

    @Test func fusedStemMatchesPlainStem() {
        let tok = ConvTokenizer(frameSize: 64, width: 32, stochasticDepth: 0)
        tok.update(parameters: ModuleParameters.unflattened(tok.parameters().flattened().map { k, v in
            k.hasSuffix("running_var") ? (k, MLXRandom.uniform(0.5 ..< 2, v.shape)) : k.hasSuffix("running_mean") ? (k, MLXRandom.normal(v.shape) * 0.1) :
            k.hasSuffix("bn.weight") ? (k, MLXRandom.uniform(0.5 ..< 1.5, v.shape)) : (k, v) }))
        tok.train(false)
        let x = MLXRandom.uniform(0 ..< 1, [3, 64, 64, 3])
        ConvBNAct.fused = false
        let plain = tok(x)
        ConvBNAct.fused = true
        let fused = tok(x)
        #expect(abs(plain - fused).max().item(Float.self) < 1e-4)
        // Inference cache gives the same result.
        tok.prepareInference(dtype: .float32)
        #expect(abs(plain - tok(x)).max().item(Float.self) < 1e-4)
    }

    @Test func frozenBatchNormKeepsBatchNormParameterNames() {
        let names = Set(FrozenBatchNorm(4).parameters().flattened().map(\.0))
        #expect(names == ["weight", "bias", "running_mean", "running_var"])
        #expect(Set(FrozenBatchNorm(4).trainableParameters().flattened().map(\.0)) == ["weight", "bias"])
    }

    @Test func chunkedAttentionEqualsFullAttention() {
        let layout = PolicyLayout(thinkingTokens: 1, slots: 8)
        let T = 10, N = T * layout.tokensPerStep
        let layer = TransformerLayer(width: 64, heads: 2, kvHeads: 2, dropout: 0, sinks: 0)
        let x = MLXRandom.normal([2, N, 64])
        for history in [T - 1, 3] {
            let mask = layout.fullMask(steps: T, history: history)
            let full = layer(x, mask: mask)
            let additive = which(mask, MLXArray(Float(0)), MLXArray(-Float.infinity))
            let chunked = layer(x, mask: additive, chunks: layout.attentionChunks(steps: T, history: history, count: 3))
            #expect(abs(full - chunked).max().item(Float.self) < 1e-5, "history \(history)")
        }
    }

    @Test func singlePassStemSamplingMatchesTwoPass() {
        var a = tinyArchitecture(); a.stochasticDepth = 0; a.dropout = 0
        let m = PolicyModel(architecture: a, actions: .gameDefault)
        m.setTraining(); m.freezeBatchNormStatistics(); m.visionGradientFrames = 3
        let frames = MLXRandom.randInt(low: 0, high: 256, [8, 32, 32, 3]).asType(.uint8)
        func tokensAndGrad(legacy: Bool) -> (MLXArray, MLXArray) {
            PolicyModel.legacyStemSampling = legacy
            defer { PolicyModel.legacyStemSampling = false }
            MLXRandom.seed(3)
            let t = m.imageTokens(frames)
            MLXRandom.seed(3)
            let g = valueAndGrad(model: m) { (mm: PolicyModel, f: [MLXArray]) -> [MLXArray] in [mm.imageTokens(f[0]).square().sum()] }(m, [frames]).1
            return (t, Dictionary(uniqueKeysWithValues: g.flattened())["vision.proj.weight"]!)
        }
        let (t1, g1) = tokensAndGrad(legacy: true), (t2, g2) = tokensAndGrad(legacy: false)
        #expect(abs(t1 - t2).max().item(Float.self) < 1e-4)
        #expect(abs(g1 - g2).max().item(Float.self) / abs(g1).max().item(Float.self) < 1e-4)
    }

    @Test func smallPresetsHaveValidShapes() {
        for a in [PolicyArchitecture.tiny5M, .small10M, .medium22M] {
            #expect(a.width % a.heads == 0 && a.headDim == 64)
            #expect(a.width % a.decoderHeads == 0)
            let n = a.parameterCount(actions: .gameDefault)
            #expect(n > 3_000_000 && n < 25_000_000)
        }
    }
}
