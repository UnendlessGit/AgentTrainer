import ATCore
import ATModel
import Foundation
import MLX
import MLXNN

/// One batch of training clips.
public struct TrainingBatch {
    /// `[B, T, S, S, 3]` uint8 or float (0-255) frames.
    public var frames: MLXArray
    /// `[B, T, slots]` int32: the action taken after each frame (fed back as history and to the decoder).
    public var actions: MLXArray
    /// `[B, T]` bool: steps that contribute to the loss (human-controlled, not padding).
    public var lossMask: MLXArray
    /// `[B, T, 768]` instruction embeddings and `[B, T]` presence.
    public var text: MLXArray
    public var textPresent: MLXArray
    /// Optional `[B, T, D]` screen-text summary added to the thinking token.
    public var thinkingExtra: MLXArray?

    public init(frames: MLXArray, actions: MLXArray, lossMask: MLXArray, text: MLXArray, textPresent: MLXArray, thinkingExtra: MLXArray? = nil) {
        self.frames = frames; self.actions = actions; self.lossMask = lossMask; self.text = text; self.textPresent = textPresent; self.thinkingExtra = thinkingExtra
    }
}

/// Pixel2Play's objective: per-head cross-entropy divided by log(#classes), summed, plus z-loss.
public enum PolicyLoss {
    public struct Parts {
        public let total: MLXArray
        public let perHead: [MLXArray] // keys, pointer, x, y (raw CE, masked mean)
        public let z: MLXArray
    }

    static func maskedCE(_ logits: MLXArray, _ labels: MLXArray, _ mask: MLXArray) -> (ce: MLXArray, z: MLXArray) {
        // logits [B,T,s,C], labels [B,T,s], mask [B,T]
        let lf = logits.asType(.float32)
        let lse = logSumExp(lf, axis: -1)
        let picked = takeAlong(lf, labels.expandedDimensions(axis: -1).asType(.int32), axis: -1).squeezed(axis: -1)
        let ce = lse - picked
        let m = broadcast(mask.expandedDimensions(axis: -1), to: ce.shape).asType(.float32)
        let n = maximum(m.sum(), MLXArray(Float(1)))
        return ((ce * m).sum() / n, ((lse * lse) * m).sum() / n)
    }

    public static func compute(_ logits: PolicyLogits, actions: MLXArray, mask: MLXArray, space: ActionSpaceConfig, zWeight: Float) -> Parts {
        let ks = space.keySlots, ps = space.pointerSlots
        let groups: [(MLXArray, MLXArray, Int)] = [
            (logits.keys, actions[.ellipsis, 0 ..< ks], space.keyClasses),
            (logits.pointer, actions[.ellipsis, ks ..< ks + ps], space.pointerClasses),
            (logits.x, actions[.ellipsis, (ks + ps) ..< (ks + ps + 1)], space.xClasses),
            (logits.y, actions[.ellipsis, (ks + ps + 1) ..< (ks + ps + 2)], space.yClasses),
        ]
        var total = MLXArray(Float(0)), z = MLXArray(Float(0))
        var heads: [MLXArray] = []
        for (lg, lb, classes) in groups {
            let (ce, zl) = maskedCE(lg, lb, mask)
            heads.append(ce)
            total = total + ce * Float(ActionSpaceConfig.lossScale(classes: classes))
            z = z + zl
        }
        return Parts(total: total + z * zWeight, perHead: heads, z: z)
    }
}
