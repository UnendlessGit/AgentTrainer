import Foundation
import MLX
import MLXNN

public enum WeightIOError: Error, LocalizedError {
    case mismatch(String)
    case notFinite(String)
    public var errorDescription: String? {
        switch self {
        case .mismatch(let m): "Saved weights don't match this model: \(m)"
        case .notFinite(let k): "This checkpoint is broken: its weights contain NaN or infinite values (\(k)), left by a training run that went wrong. Choose another checkpoint or train again."
        }
    }
}

extension PolicyModel {
    /// Writes all parameters (and BatchNorm statistics) as safetensors.
    public func saveWeights(to url: URL, metadata: [String: String] = [:]) throws {
        let flat = Dictionary(uniqueKeysWithValues: parameters().flattened())
        try save(arrays: flat, metadata: metadata, url: url)
    }

    /// Loads weights written by `saveWeights`, verifying every name and shape.
    public func loadWeights(from url: URL) throws {
        let loaded = try loadArrays(url: url)
        let current = Dictionary(uniqueKeysWithValues: parameters().flattened())
        // Parameters added after a checkpoint was written keep their (neutral) initial values.
        let optional: Set<String> = ["action_unknown"]
        for (k, v) in current {
            guard let w = loaded[k] else {
                if optional.contains(k) { continue }
                throw WeightIOError.mismatch("missing \(k)")
            }
            guard w.shape == v.shape else { throw WeightIOError.mismatch("\(k) is \(w.shape), expected \(v.shape)") }
        }
        // One reduction over all weights: NaN/inf anywhere makes the sum non-finite.
        let probes = current.keys.sorted().compactMap { k in loaded[k].map { (k, $0.asType(.float32).abs().sum()) } }
        eval(probes.map(\.1))
        if let bad = probes.first(where: { !$0.1.item(Float.self).isFinite }) { throw WeightIOError.notFinite(bad.0) }
        update(parameters: ModuleParameters.unflattened(current.keys.compactMap { k in loaded[k].map { (k, $0.asType(current[k]!.dtype)) } }))
        eval(self)
    }
}
