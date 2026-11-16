import Foundation
import MLX
import MLXNN
import MLXRandom

/// Fused elementwise kernels (one pass over the activation instead of several).
enum StemKernels {
    static let biasSiLU: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { a in
        let z = a[0] + a[1]
        return [z * sigmoid(z)]
    }
    /// `(y + bias) * keep + x` (residual with row-wise stochastic depth).
    static let biasDropResidual: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { a in [(a[0] + a[1]) * a[2] + a[3]] }
    static let biasResidual: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { a in [a[0] + a[1] + a[2]] }
}

/// BatchNorm with frozen statistics, as AgentTrainer always trains it (the affine weight and bias
/// still learn). Same parameter names as `MLXNN.BatchNorm`, so checkpoints and the P2P importer
/// are unchanged. Because the statistics are constants, the whole normalisation is a per-channel
/// scale and shift that can be folded into the preceding convolution.
final class FrozenBatchNorm: Module, UnaryLayer {
    @ParameterInfo var weight: MLXArray
    @ParameterInfo var bias: MLXArray
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray
    let eps: Float

    init(_ n: Int, eps: Float = 1e-5) {
        _weight.wrappedValue = MLXArray.ones([n])
        _bias.wrappedValue = MLXArray.zeros([n])
        _runningMean.wrappedValue = MLXArray.zeros([n])
        _runningVar.wrappedValue = MLXArray.ones([n])
        self.eps = eps
        super.init()
        freeze(recursive: false, keys: ["running_mean", "running_var"])
    }

    /// Per-channel (scale, shift) in fp32: `bn(x) = x * scale + shift`.
    func affine() -> (MLXArray, MLXArray) {
        let s = weight.asType(.float32) * rsqrt(runningVar.asType(.float32) + eps)
        return (s, bias.asType(.float32) - runningMean.asType(.float32) * s)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (s, b) = affine()
        return x * s.asType(x.dtype) + b.asType(x.dtype)
    }
}

/// Conv → BatchNorm → optional SiLU (torchvision `Conv2dNormActivation`), NHWC.
final class ConvBNAct: Module, UnaryLayer {
    @ModuleInfo var conv: Conv2d
    @ModuleInfo var bn: FrozenBatchNorm
    let activation: Bool
    /// A/B switch for benchmarks: false = the plain conv → BN → SiLU sequence.
    nonisolated(unsafe) static var fused = true

    init(_ cin: Int, _ cout: Int, kernel: Int, stride: Int = 1, groups: Int = 1, activation: Bool = true) {
        _conv.wrappedValue = Conv2d(inputChannels: cin, outputChannels: cout, kernelSize: .init(kernel),
                                    stride: .init(stride), padding: .init((kernel - 1) / 2), groups: groups, bias: false)
        _bn.wrappedValue = FrozenBatchNorm(cout, eps: 1e-5)
        self.activation = activation
    }

    /// Folded weights cached for inference (`prepareInference`); never used while training.
    final class FoldCache { var dtype: DType?; var w: MLXArray?; var b: MLXArray? }
    let foldCache = FoldCache()

    /// Conv weights with the BatchNorm scale folded in, and the remaining per-channel shift.
    func foldedWeights(_ dtype: DType) -> (MLXArray, MLXArray) {
        if !training, foldCache.dtype == dtype, let w = foldCache.w, let b = foldCache.b { return (w, b) }
        let (s, b) = bn.affine()
        return ((conv.weight.asType(.float32) * s.reshaped(-1, 1, 1, 1)).asType(dtype), b.asType(dtype))
    }

    /// Computes the folded weights once for a model that won't change (live inference). Call
    /// outside `compile`; switching to training ignores the cache.
    func prepareInference(dtype: DType) {
        foldCache.dtype = nil
        let (w, b) = foldedWeights(dtype)
        eval(w, b)
        foldCache.w = w; foldCache.b = b; foldCache.dtype = dtype
    }

    /// Convolution with the BatchNorm scale folded into its weights; returns (pre-shift output, shift).
    func foldedConv(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let (w, b) = foldedWeights(x.dtype)
        // The custom depthwise backward only matters when gradients flow (training).
        let y = training && conv.groups > 1 && conv.groups == x.dim(-1)
            ? DepthwiseConv.apply(x, w, stride: conv.stride.0, padding: conv.padding.0)
            : conv2d(x, w, stride: .init(conv.stride), padding: .init(conv.padding), dilation: .init(conv.dilation), groups: conv.groups)
        return (y, b)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard Self.fused else {
            let y = bn(conv(x))
            return activation ? silu(y) : y
        }
        let (y, b) = foldedConv(x)
        return activation ? StemKernels.biasSiLU([y, b])[0] : y + b
    }
}

/// Squeeze-and-excitation with SiLU / sigmoid (torchvision `SqueezeExcitation`).
final class SqueezeExcite: Module, UnaryLayer {
    @ModuleInfo var fc1: Conv2d
    @ModuleInfo var fc2: Conv2d

    init(channels: Int, squeeze: Int) {
        _fc1.wrappedValue = Conv2d(inputChannels: channels, outputChannels: squeeze, kernelSize: 1, bias: true)
        _fc2.wrappedValue = Conv2d(inputChannels: squeeze, outputChannels: channels, kernelSize: 1, bias: true)
    }

    /// Per-sample channel gates `[N, 1, 1, C]`.
    func gates(_ x: MLXArray) -> MLXArray { sigmoid(fc2(silu(fc1(x.mean(axes: [1, 2], keepDims: true))))) }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x * gates(x) }
}

/// Mobile inverted bottleneck block (torchvision `MBConv`) with row-wise stochastic depth.
final class MBConv: Module, UnaryLayer {
    @ModuleInfo var expand: ConvBNAct?
    @ModuleInfo var depthwise: ConvBNAct
    @ModuleInfo var se: SqueezeExcite
    @ModuleInfo var project: ConvBNAct
    let residual: Bool
    let dropProbability: Float

    init(cin: Int, cout: Int, expandRatio: Int, kernel: Int, stride: Int, dropProbability: Float) {
        let hidden = cin * expandRatio
        if expandRatio != 1 { _expand.wrappedValue = ConvBNAct(cin, hidden, kernel: 1) }
        _depthwise.wrappedValue = ConvBNAct(hidden, hidden, kernel: kernel, stride: stride, groups: hidden)
        _se.wrappedValue = SqueezeExcite(channels: hidden, squeeze: max(1, cin / 4))
        _project.wrappedValue = ConvBNAct(hidden, cout, kernel: 1, activation: false)
        residual = stride == 1 && cin == cout
        self.dropProbability = dropProbability
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        if let expand { y = expand(y) }
        guard ConvBNAct.fused else {
            y = project(se(depthwise(y)))
            guard residual else { return y }
            if training && dropProbability > 0 {
                let keep = 1 - dropProbability
                let mask = MLXRandom.bernoulli(MLXArray(keep), [y.dim(0), 1, 1, 1]).asType(y.dtype) / keep
                y = y * mask
            }
            return y + x
        }
        y = depthwise(y)
        // Squeeze-excite gates and the projection's BatchNorm are folded into per-sample 1×1
        // projection weights, so the gated activation is never written out.
        let g = se.gates(y)
        let (wFold, bias) = project.foldedWeights(y.dtype)
        let N = y.dim(0), H = y.dim(1), W = y.dim(2), C = y.dim(3), O = wFold.dim(0)
        let wT = wFold.reshaped(O, C).transposed()
        let out = matmul(y.reshaped(N, H * W, C), g.reshaped(N, C, 1) * wT).reshaped(N, H, W, O)
        guard residual else { return out + bias }
        if training && dropProbability > 0 {
            let keep = 1 - dropProbability
            let mask = MLXRandom.bernoulli(MLXArray(keep), [N, 1, 1, 1]).asType(out.dtype) / keep
            return StemKernels.biasDropResidual([out, bias, mask, x])[0]
        }
        return StemKernels.biasResidual([out, bias, x])[0]
    }
}

/// EfficientNet-B0 `features[0:6]` (stem + MBConv stages 1-5), stride 16, 112 output channels.
/// Pixel2Play initialises it from ImageNet and trains it end to end.
final class EfficientNetStem: Module, UnaryLayer {
    @ModuleInfo var stem: ConvBNAct
    @ModuleInfo var blocks: [MBConv]

    static let outputChannels = 112
    static let stride = 16

    init(stochasticDepth: Float = 0.2) {
        _stem.wrappedValue = ConvBNAct(3, 32, kernel: 3, stride: 2)
        // (expand, kernel, stride, in, out, repeats) for torchvision B0 stages 1-5.
        let stages: [(Int, Int, Int, Int, Int, Int)] = [
            (1, 3, 1, 32, 16, 1), (6, 3, 2, 16, 24, 2), (6, 5, 2, 24, 40, 2), (6, 3, 2, 40, 80, 3), (6, 5, 1, 80, 112, 3),
        ]
        let totalBlocks = 16 // all of B0's blocks: stochastic depth ramps over the full network
        var blocks: [MBConv] = []
        var id = 0
        for (e, k, s, cin, cout, n) in stages {
            for i in 0 ..< n {
                blocks.append(MBConv(cin: i == 0 ? cin : cout, cout: cout, expandRatio: e, kernel: k,
                                     stride: i == 0 ? s : 1,
                                     dropProbability: stochasticDepth * Float(id) / Float(totalBlocks)))
                id += 1
            }
        }
        _blocks.wrappedValue = blocks
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = stem(x)
        for b in blocks { y = b(y) }
        return y
    }
}

/// Pixel2Play `ConvTokenizer`: EfficientNet stem → flatten → Linear → LayerNorm = one token per frame.
/// Features are flattened in HWC order; the P2P importer permutes the projection from CHW.
final class ConvTokenizer: Module {
    @ModuleInfo var stem: EfficientNetStem
    @ModuleInfo var proj: Linear
    @ModuleInfo var norm: LayerNorm
    let frameSize: Int

    init(frameSize: Int, width: Int, stochasticDepth: Float) {
        precondition(frameSize % EfficientNetStem.stride == 0, "frame size must be a multiple of 16")
        self.frameSize = frameSize
        let s = frameSize / EfficientNetStem.stride
        _stem.wrappedValue = EfficientNetStem(stochasticDepth: stochasticDepth)
        _proj.wrappedValue = Linear(EfficientNetStem.outputChannels * s * s, width)
        _norm.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
    }

    func clearInferenceCache() {
        for (_, m) in namedModules() { if let c = m as? ConvBNAct { c.foldCache.dtype = nil; c.foldCache.w = nil; c.foldCache.b = nil } }
    }

    /// Folds every BatchNorm into its conv once (for a model used only for inference).
    func prepareInference(dtype: DType) {
        for (_, m) in namedModules() { (m as? ConvBNAct)?.prepareInference(dtype: dtype) }
    }

    /// `frames`: `[N, H, W, 3]` floats in [0, 1]. Returns `[N, width]`.
    func callAsFunction(_ frames: MLXArray) -> MLXArray {
        let f = stem(frames)
        return norm(proj(f.reshaped(f.dim(0), -1)))
    }
}
