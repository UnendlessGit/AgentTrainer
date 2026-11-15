import Foundation
import MLX

/// Pixel2Play's per-timestep attention pattern. Each step is
/// `[text, image, thinking×n, a_out, action×slots]`:
/// - earlier steps: everything except `a_out` is visible (within the history window);
/// - same step: image-type tokens see each other; `a_out` sees image-type tokens and itself;
///   action tokens see image-type tokens and all of this step's actions; nothing sees `a_out`.
public struct PolicyLayout: Hashable, Sendable {
    public let thinkingTokens: Int
    public let slots: Int
    public var imageTypeCount: Int { 2 + thinkingTokens }
    public var aOutIndex: Int { imageTypeCount }
    public var tokensPerStep: Int { imageTypeCount + 1 + slots }
    public var textIndex: Int { 0 }
    public var imageIndex: Int { 1 }

    public init(thinkingTokens: Int, slots: Int) {
        self.thinkingTokens = thinkingTokens
        self.slots = slots
    }

    /// Whether a token (relative index `qr` in step `qs`) may attend to `(ks, kr)`.
    public func allowed(qs: Int, qr: Int, ks: Int, kr: Int, history: Int) -> Bool {
        if ks > qs || qs - ks > history { return false }
        let kIsAOut = kr == aOutIndex
        if ks < qs { return !kIsAOut }
        let qImg = qr < imageTypeCount, kImg = kr < imageTypeCount
        if qImg { return kImg }
        if qr == aOutIndex { return kImg || kIsAOut }
        return !kIsAOut
    }

    /// Boolean `[N, N]` mask for a full sequence of `steps` timesteps (true = may attend).
    public func fullMask(steps: Int, history: Int) -> MLXArray {
        let n = steps * tokensPerStep
        let idx = MLXArray(Int32(0) ..< Int32(n))
        let tps = Int32(tokensPerStep)
        let s = floorDivide(idx, MLXArray(tps)), r = idx % tps
        let qs = s.expandedDimensions(axis: 1), ks = s.expandedDimensions(axis: 0)
        let qr = r.expandedDimensions(axis: 1), kr = r.expandedDimensions(axis: 0)
        return Self.maskOps(qs: qs, qr: qr, ks: ks, kr: kr, imageTypes: Int32(imageTypeCount), aOut: Int32(aOutIndex), history: Int32(history))
    }

    /// The same rule expressed with MLX ops so the streaming stepper can build it for arbitrary positions.
    static func maskOps(qs: MLXArray, qr: MLXArray, ks: MLXArray, kr: MLXArray, imageTypes: Int32, aOut: Int32, history: Int32) -> MLXArray {
        let kIsAOut = kr .== aOut
        let qImg = qr .< imageTypes, kImg = kr .< imageTypes
        let qIsAOut = qr .== aOut
        let earlier = (ks .< qs) .&& (logicalNot(kIsAOut))
        let sameImg = qImg .&& kImg
        let sameAOut = qIsAOut .&& (kImg .|| kIsAOut)
        let sameAct = logicalNot(qImg) .&& logicalNot(qIsAOut) .&& logicalNot(kIsAOut)
        let same = (ks .== qs) .&& (sameImg .|| sameAOut .|| sameAct)
        let inWindow = (qs - ks) .<= history
        return (earlier .|| same) .&& inWindow
    }
}
