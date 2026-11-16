import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// AdamW with MLX's exact update rule (decoupled weight decay, no bias correction), where each
/// parameter's whole update - gradient clipping scale, weight decay, both moments and the
/// low-precision copy for the compute model - is one fused kernel instead of ~10 separate ones.
/// It saves and loads the same state layout as `MLXOptimizers.AdamW` (`<param>.m/.v/.step`), so
/// existing checkpoints resume unchanged.
final class FusedAdamW {
    var learningRate: Float
    private(set) var stepCount = 0
    private var m: [String: MLXArray] = [:]
    private var v: [String: MLXArray] = [:]
    private let kernel: @Sendable ([MLXArray]) -> [MLXArray]
    private let kernelFP32: @Sendable ([MLXArray]) -> [MLXArray]
    /// Parameters whose compute copy stays fp32.
    var keepsFP32: (String) -> Bool = { _ in false }
    private var normFn: (@Sendable ([MLXArray]) -> [MLXArray])?

    init(learningRate: Float, betas: (Float, Float), eps: Float, weightDecay: Float, computeType: DType) {
        self.learningRate = learningRate
        let (b1, b2) = betas
        func make(_ copyType: DType) -> @Sendable ([MLXArray]) -> [MLXArray] {
            compile(shapeless: true) { a in
                // a: parameter (fp32), gradient, m, v, learning rate, clip scale
                let g = a[1].asType(.float32) * a[5]
                let lr = a[4]
                let p = a[0] * (1 - lr * weightDecay)
                let m = b1 * a[2] + (1 - b1) * g
                let v = b2 * a[3] + (1 - b2) * square(g)
                let next = p - lr * m / (sqrt(v) + eps)
                return [next, m, v, next.asType(copyType)]
            }
        }
        kernel = make(computeType)
        kernelFP32 = make(.float32)
    }

    /// Arrays to evaluate after an update.
    var arrays: [MLXArray] { Array(m.values) + Array(v.values) }

    /// Clips the global gradient norm to `clip` (when > 0, exactly like `clipGradNorm`), applies the
    /// update to `master` and writes the low-precision copy into `compute`. Returns the norm.
    func update(master: Module, compute: Module, gradients: ModuleParameters, clip: Float) -> MLXArray {
        let grads = gradients.flattened()
        var total = MLXArray(Float(0)), scale = MLXArray(Float(1))
        if clip > 0 {
            if normFn == nil {
                normFn = compile { gs in [sqrt(gs.reduce(MLXArray(Float(0))) { $0 + $1.asType(.float32).square().sum() })] }
            }
            total = normFn!(grads.map(\.1))[0]
            scale = which(total .< clip, MLXArray(Float(1)), clip / (total + 1e-6))
        }
        let params = Dictionary(uniqueKeysWithValues: master.parameters().flattened())
        let lr = MLXArray(learningRate)
        var newMaster: [(String, MLXArray)] = [], newCompute: [(String, MLXArray)] = []
        newMaster.reserveCapacity(grads.count); newCompute.reserveCapacity(grads.count)
        for (k, g) in grads {
            guard let p = params[k] else { continue }
            let out = (keepsFP32(k) ? kernelFP32 : kernel)([p, g, m[k] ?? MLXArray.zeros(like: p), v[k] ?? MLXArray.zeros(like: p), lr, scale])
            newMaster.append((k, out[0])); m[k] = out[1]; v[k] = out[2]; newCompute.append((k, out[3]))
        }
        master.update(parameters: ModuleParameters.unflattened(newMaster))
        compute.update(parameters: ModuleParameters.unflattened(newCompute))
        stepCount += 1
        return total
    }

    /// Same tree as `AdamW.state()`.
    func state() -> ModuleParameters {
        let step = MLXArray(Int32(stepCount))
        var flat: [(String, MLXArray)] = []
        for (k, a) in m { flat.append(("\(k).m", a)) }
        for (k, a) in v { flat.append(("\(k).v", a)) }
        for k in m.keys { flat.append(("\(k).step", step)) }
        return ModuleParameters.unflattened(flat)
    }

    /// Loads a tree written by `state()` or by `AdamW.state()`.
    func update(parameters: ModuleParameters) {
        for (key, a) in parameters.flattened() {
            guard let dot = key.lastIndex(of: ".") else { continue }
            let k = String(key[..<dot]), suffix = key[key.index(after: dot)...]
            switch suffix {
            case "m": m[k] = a
            case "v": v[k] = a
            case "step": stepCount = max(stepCount, Int(a.item(Int32.self)))
            default: continue
            }
        }
    }
}

/// Developer check: FusedAdamW against MLXOptimizers.AdamW + clipGradNorm, and their speed.
public enum OptimizerCheck {
    public static func run() {
        let a = Linear(64, 32), b = Linear(64, 32)
        b.update(parameters: a.parameters())
        let c = Linear(64, 32)
        let ref = MLXOptimizers.AdamW(learningRate: 1e-3, betas: (0.9, 0.999), eps: 1e-8, weightDecay: 1e-4)
        let mine = FusedAdamW(learningRate: 1e-3, betas: (0.9, 0.999), eps: 1e-8, weightDecay: 1e-4, computeType: .float32)
        for i in 0 ..< 5 {
            let g = ModuleParameters.unflattened(a.trainableParameters().flattened().map { ($0.0, MLXRandom.normal($0.1.shape) * Float(i + 1)) })
            let lr = Float(1e-3) * Float(i + 1)
            ref.learningRate = lr; mine.learningRate = lr
            let (clipped, _) = clipGradNorm(gradients: g, maxNorm: 1)
            ref.update(model: a, gradients: clipped)
            _ = mine.update(master: b, compute: c, gradients: g, clip: 1)
            eval(a, b, c, ref)
        }
        let d = zip(a.parameters().flattened(), b.parameters().flattened()).map { abs($0.1 - $1.1).max().item(Float.self) }.max() ?? 0
        print(String(format: "FusedAdamW vs AdamW after 5 clipped steps: max |Δparam| %.2e", d))
        // State round trip through the AdamW layout.
        let restored = FusedAdamW(learningRate: 1e-3, betas: (0.9, 0.999), eps: 1e-8, weightDecay: 1e-4, computeType: .float32)
        restored.update(parameters: ref.state())
        print("loaded AdamW state: step \(restored.stepCount), keys \(restored.state().flattened().count) vs \(ref.state().flattened().count)")
    }
}
