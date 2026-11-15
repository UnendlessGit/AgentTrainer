import Foundation
import MLX
import MLXNN
import MLXRandom

extension PolicyModel {
    /// Casts every parameter (and BatchNorm statistic) to `dtype`, e.g. bf16 for inference.
    public func convert(to dtype: DType) {
        let p = parameters().flattened().map { ($0.0, $0.1.dtype.isFloatingPoint ? $0.1.asType(dtype) : $0.1) }
        update(parameters: ModuleParameters.unflattened(p))
        eval(self)
    }

    /// Replaces every Linear in the backbone and decoder with an 8-bit quantized version (inference only).
    public func quantizeForInference(bits: Int = 8, groupSize: Int = 64) {
        quantize(model: self, groupSize: groupSize, bits: bits) { path, module in
            guard let linear = module as? Linear, linear.weight.dim(1) % groupSize == 0 else { return false }
            return path.hasPrefix("backbone") || path.hasPrefix("decoder")
        }
        eval(self)
    }

    public static func randomFrames(_ n: Int, size: Int) -> MLXArray {
        let low = MLXRandom.randInt(low: 0, high: 256, [n, size / 16, size / 16, 3]).asType(.float32)
        let up = repeated(repeated(low, count: 16, axis: 1), count: 16, axis: 2)
        return up.asType(.uint8)
    }
}
