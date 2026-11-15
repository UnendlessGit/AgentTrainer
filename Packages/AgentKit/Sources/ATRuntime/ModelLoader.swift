import ATCore
import ATModel
import Foundation
import MLX

/// Loads a trained model package for inference.
public enum ModelLoader {
    public enum LoadError: Error, LocalizedError {
        case notTrained(String)
        public var errorDescription: String? {
            if case .notTrained(let n) = self { return "“\(n)” has no checkpoints yet. Train it first." }
            return nil
        }
    }

    /// Returns the model (eval mode, bf16 unless `fp32`) and its definition. Uses `checkpoint`,
    /// the model's active checkpoint, or the newest one.
    public static func load(_ url: URL, checkpoint: String? = nil, fp32: Bool = false, quantized: Bool = false) throws -> (PolicyModel, ModelDefinition, String) {
        let def = try ModelPackage.read(url)
        let names = ModelPackage.checkpointNames(url)
        guard let name = checkpoint ?? def.activeCheckpoint.flatMap({ names.contains($0) ? $0 : nil }) ?? names.last else {
            throw LoadError.notTrained(def.name)
        }
        let model = PolicyModel(architecture: def.architecture, actions: def.actions)
        try model.loadWeights(from: url.appendingPathComponent(ModelPackage.checkpoints).appendingPathComponent(name).appendingPathComponent(ModelPackage.weights))
        model.train(false)
        if !fp32 { model.convert(to: .bfloat16) }
        if quantized { model.quantizeForInference() }
        return (model, def, name)
    }
}
