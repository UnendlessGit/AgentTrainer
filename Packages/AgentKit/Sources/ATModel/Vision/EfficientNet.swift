import Foundation
import MLX
import MLXNN
import MLXRandom

/// Conv → BatchNorm → optional SiLU (torchvision `Conv2dNormActivation`), NHWC.
final class ConvBNAct: Module, UnaryLayer {
    @ModuleInfo var conv: Conv2d
    @ModuleInfo var bn: BatchNorm
    let activation: Bool

    init(_ cin: Int, _ cout: Int, kernel: Int, stride: Int = 1, groups: Int = 1, activation: Bool = true) {
        _conv.wrappedValue = Conv2d(inputChannels: cin, outputChannels: cout, kernelSize: .init(kernel),
                                    stride: .init(stride), padding: .init((kernel - 1) / 2), groups: groups, bias: false)
        _bn.wrappedValue = BatchNorm(featureCount: cout, eps: 1e-5, momentum: 0.1)
        self.activation = activation
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = bn(conv(x))
        return activation ? silu(y) : y
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

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = sigmoid(fc2(silu(fc1(x.mean(axes: [1, 2], keepDims: true)))))
        return x * s
    }
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
        y = project(se(depthwise(y)))
        guard residual else { return y }
        if training && dropProbability > 0 {
            let keep = 1 - dropProbability
            let mask = MLXRandom.bernoulli(MLXArray(keep), [y.dim(0), 1, 1, 1]).asType(y.dtype) / keep
            y = y * mask
        }
        return y + x
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

    /// `frames`: `[N, H, W, 3]` floats in [0, 1]. Returns `[N, width]`.
    func callAsFunction(_ frames: MLXArray) -> MLXArray {
        let f = stem(frames)
        return norm(proj(f.reshaped(f.dim(0), -1)))
    }
}
