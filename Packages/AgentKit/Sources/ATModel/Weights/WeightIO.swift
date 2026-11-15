import Foundation
import MLX
import MLXNN

public enum WeightIOError: Error, LocalizedError {
    case mismatch(String)
    public var errorDescription: String? {
        if case .mismatch(let m) = self { return "Saved weights don't match this model: \(m)" }
        return nil
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
        for (k, v) in current {
            guard let w = loaded[k] else { throw WeightIOError.mismatch("missing \(k)") }
            guard w.shape == v.shape else { throw WeightIOError.mismatch("\(k) is \(w.shape), expected \(v.shape)") }
        }
        update(parameters: ModuleParameters.unflattened(current.keys.map { ($0, loaded[$0]!.asType(current[$0]!.dtype)) }))
        eval(self)
    }
}
