import Foundation
import MLX

extension PolicyModel {
    /// Image tokens without the position embedding (P2P `image_tokenizer` output), for parity checks.
    public func rawImageTokens(_ frames: MLXArray) -> MLXArray {
        vision(frames.asType(computeDType) / 255)
    }
}

extension PolicyModel {
    /// Image tokens through the training path (checkpointed when enabled), for memory diagnostics.
    public func debugImageTokens(_ frames: MLXArray) -> MLXArray { imageTokens(frames) }
}
