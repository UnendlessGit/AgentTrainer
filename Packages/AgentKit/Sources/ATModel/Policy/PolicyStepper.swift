import ATCore
import Foundation
import MLX
import MLXFast
import MLXRandom

/// Output of one real-time step.
public struct StepResult: Sendable {
    /// Sampled class per action slot.
    public let classes: [Int32]
    /// Per-slot logits (only when requested; used for parity checks and diagnostics).
    public let logits: [[Float]]?
}

/// Streams a `PolicyModel` one timestep at a time with a fixed-size KV cache that holds the last
/// `contextSteps` timesteps (the model's temporal memory) - nothing is ever re-processed.
///
/// Each call runs **one** backbone pass of `tokensPerStep` tokens: the previous step's action
/// tokens followed by this frame's `[text, image, thinking, a_out]`. This is exactly equivalent
/// to Pixel2Play's two passes per frame because no token can see same-step actions except other
/// action tokens. Keys are cached with RoPE applied; positions are periodically re-based so they
/// never grow large enough to lose precision.
public final class PolicyStepper {
    public let model: PolicyModel
    public let contextSteps: Int
    public var temperature: Float = 1
    /// When false, past actions are hidden from the model (it decides from what it sees only).
    public var useActionHistory = true
    /// RoPE positions are rebased once they exceed this (kept small enough for fp32 angle precision).
    public var rebaseThreshold = 1 << 20

    private let layout: PolicyLayout
    private let tps: Int
    private let capacityBlocks: Int
    private let capacity: Int
    private var keys: [MLXArray]
    private var values: [MLXArray]
    private var slotPos: [Int32]
    private var slotValid: [Bool]
    private var stepIndex = 0
    private var positionShift = 0
    private var previous: MLXArray
    private var textCacheKey: [Float]?
    private var textToken: MLXArray?
    /// Sampling loop compiled once (weights are captured as constants; recreate the stepper after reloading weights).
    private lazy var compiledDecode: ([MLXArray]) -> [MLXArray] = {
        let m = model
        let S = m.actions.slotCount, D = m.width
        let counts = m.actions.slotClassCounts
        return compile { (inputs: [MLXArray]) -> [MLXArray] in
            let aOut = inputs[0], u = inputs[1], temp = inputs[2]
            var embs: [MLXArray] = []
            var out: [MLXArray] = []
            for s in 0 ..< S {
                let h = m.decoder.run(aOut: aOut, inputs: embs.isEmpty ? nil : concatenated(embs, axis: 1))
                let logits = m.headLogits(slot: s, h[0..., h.dim(1) - 1, 0...]).asType(.float32)
                let g = -log(-log(u[s, 0 ..< counts[s]]))
                let cls = argMax(logits / temp + g, axis: -1).asType(.int32)
                out.append(cls)
                embs.append(m.embedSlot(s, cls).reshaped(1, 1, D))
            }
            return [concatenated(out, axis: 0)]
        }
    }()
    /// Backbone output at `a_out` from the last step (diagnostics).
    public private(set) var lastAOutState: MLXArray?

    public init(model: PolicyModel, contextSteps: Int? = nil) {
        self.model = model
        self.contextSteps = contextSteps ?? model.architecture.contextSteps
        layout = model.layout
        tps = layout.tokensPerStep
        let blocks = self.contextSteps + 2
        let cap = blocks * model.layout.tokensPerStep
        capacityBlocks = blocks
        capacity = cap
        let a = model.architecture
        let dt = model.computeDType
        keys = (0 ..< a.layers).map { _ in MLXArray.zeros([1, a.kvHeads, cap, a.headDim], dtype: dt) }
        values = (0 ..< a.layers).map { _ in MLXArray.zeros([1, a.kvHeads, cap, a.headDim], dtype: dt) }
        slotPos = Array(repeating: 0, count: cap)
        slotValid = Array(repeating: false, count: slotPos.count)
        previous = MLXArray.zeros([1, model.actions.slotCount], dtype: .int32)
    }

    /// Clears temporal memory (new episode).
    public func reset() {
        for i in keys.indices { keys[i] = MLXArray.zeros(like: keys[i]); values[i] = MLXArray.zeros(like: values[i]) }
        slotValid = Array(repeating: false, count: slotValid.count)
        stepIndex = 0
        positionShift = 0
        previous = MLXArray.zeros([1, model.actions.slotCount], dtype: .int32)
    }

    public var stepsTaken: Int { stepIndex }

    /// Runs one timestep.
    /// - frame: uint8 `[H, W, 3]` at the model's resolution.
    /// - text: 768-d instruction embedding, or nil for "no instruction".
    /// - previousAction: what actually happened during the last step (defaults to what the model sampled).
    /// - forced: teacher-forced classes for this step's decoder (parity tests / replay evaluation).
    public func step(frame: MLXArray, text: [Float]?, previousAction: [Int32]? = nil, forced: [Int32]? = nil,
                     thinkingExtra: MLXArray? = nil, returnLogits: Bool = false) -> StepResult {
        rebaseIfNeeded()
        let m = model
        let D = m.width, S = m.actions.slotCount
        if let p = previousAction { previous = MLXArray(p).reshaped(1, S) }

        // Tokens: actions(t-1) then [text, image, thinking, a_out](t).
        let act = useActionHistory ? m.actionEmbeddings(previous) + m.actionPos : (m.actionUnknown + m.actionPos).expandedDimensions(axis: 0)
        let txt = textTokenFor(text)
        let img = m.imageTokens(frame.reshaped(1, frame.dim(0), frame.dim(1), frame.dim(2))).reshaped(1, 1, D)
        var think = m.thinking.expandedDimensions(axis: 0)
        if let extra = thinkingExtra {
            let first = think[0..., 0 ..< 1, 0...] + extra.reshaped(1, 1, D).asType(think.dtype)
            think = m.architecture.thinkingTokens > 1 ? concatenated([first, think[0..., 1..., 0...]], axis: 1) : first
        }
        let aOut = m.actionOut.expandedDimensions(axis: 0)
        var x = concatenated([act.reshaped(1, S, D), txt.reshaped(1, 1, D), img, think, aOut], axis: 1)

        // Positions (absolute, shifted): token (step s, slot r) sits at tps*(s+1) + r - shift.
        let blockStart = tps * stepIndex + layout.aOutIndex + 1 - positionShift
        let block = stepIndex % capacityBlocks
        let first = block * tps
        for j in 0 ..< tps {
            slotPos[first + j] = Int32(blockStart + j)
            // At t = 0 there is no previous step: its action tokens are placeholders nobody may read.
            slotValid[first + j] = !(stepIndex == 0 && j < S)
        }
        let qPos = MLXArray(Int32(blockStart) ..< Int32(blockStart + tps))
        let mask = streamingMask(qPos: qPos)

        for (l, layer) in m.backbone.enumerated() {
            let (q, k, v) = layer.attention.projectQKV(layer.attentionNorm(x), ropeOffset: blockStart)
            keys[l][0..., 0..., first ..< first + tps, 0...] = k
            values[l][0..., 0..., first ..< first + tps, 0...] = v
            let h = x + layer.attention.attend(q, keys[l], values[l], mask: mask)
            x = h + layer.ffn(layer.ffnNorm(h))
        }
        let aOutState = x[0..., tps - 1, 0...] // a_out is the last token of the pass
        lastAOutState = aOutState

        // Autoregressive decoding: one decoder pass per slot (recomputing the ≤9-token prefix is
        // as cheap as caching because the decoder is weight-bandwidth bound).
        if forced == nil && !returnLogits {
            let maxC = m.actions.slotClassCounts.max()!
            let u = MLXRandom.uniform(low: Float(1e-9), high: 1, [S, maxC])
            let classes = compiledDecode([aOutState, u, MLXArray(max(temperature, 1e-4))])[0]
            previous = classes.reshaped(1, S)
            eval(classes, keys, values)
            stepIndex += 1
            return StepResult(classes: classes.asArray(Int32.self), logits: nil)
        }
        var inputs: [MLXArray] = []
        var sampled: [MLXArray] = []
        var allLogits: [MLXArray] = []
        for s in 0 ..< S {
            let h = m.decoder.run(aOut: aOutState, inputs: inputs.isEmpty ? nil : concatenated(inputs, axis: 1))
            let logits = m.headLogits(slot: s, h[0..., h.dim(1) - 1, 0...]).asType(.float32)
            let cls: MLXArray
            if let f = forced {
                cls = MLXArray([f[s]])
            } else if temperature <= 0 {
                cls = argMax(logits, axis: -1).asType(.int32)
            } else {
                let u = MLXRandom.uniform(low: Float(1e-9), high: 1, logits.shape)
                cls = argMax(logits / temperature - log(-log(u)), axis: -1).asType(.int32)
            }
            sampled.append(cls)
            if returnLogits { allLogits.append(logits) }
            inputs.append(m.embedSlot(s, cls).reshaped(1, 1, D))
        }
        let classes = concatenated(sampled, axis: 0)
        previous = classes.reshaped(1, S)
        if returnLogits { eval(classes, keys, values, allLogits) } else { eval(classes, keys, values) }
        stepIndex += 1
        return StepResult(classes: classes.asArray(Int32.self),
                          logits: returnLogits ? allLogits.map { $0.asArray(Float.self) } : nil)
    }

    private func textTokenFor(_ text: [Float]?) -> MLXArray {
        if let t = textToken, textCacheKey == text { return t }
        let m = model
        let token: MLXArray
        if let text {
            token = m.textTokens(MLXArray(text).reshaped(1, -1), present: MLXArray([true]))
        } else {
            token = m.textTokens(MLXArray.zeros([1, m.architecture.textEmbeddingDim]), present: MLXArray([false]))
        }
        eval(token)
        textCacheKey = text
        textToken = token
        return token
    }

    private func streamingMask(qPos: MLXArray) -> MLXArray {
        let tpsA = MLXArray(Int32(tps))
        let kPos = MLXArray(slotPos)
        let qs = floorDivide(qPos, tpsA).expandedDimensions(axis: 1), qr = (qPos % tpsA).expandedDimensions(axis: 1)
        let ks = floorDivide(kPos, tpsA).expandedDimensions(axis: 0), kr = (kPos % tpsA).expandedDimensions(axis: 0)
        let rule = PolicyLayout.maskOps(qs: qs, qr: qr, ks: ks, kr: kr, imageTypes: Int32(layout.imageTypeCount),
                                        aOut: Int32(layout.aOutIndex), history: Int32(contextSteps - 1))
        return rule .&& MLXArray(slotValid).expandedDimensions(axis: 0)
    }

    /// Keeps RoPE positions small: rotates every cached key back by a multiple of the step size.
    private func rebaseIfNeeded() {
        let next = tps * (stepIndex + 2) - positionShift
        guard next > rebaseThreshold else { return }
        let shift = (next - 4 * tps * (contextSteps + 2)) / tps * tps
        let a = model.architecture
        for l in keys.indices {
            let k = keys[l].reshaped(-1, 1, a.headDim)
            keys[l] = MLXFast.RoPE(k, dimensions: a.headDim, traditional: true, base: 10_000, scale: 1, offset: -shift)
                .reshaped(keys[l].shape)
        }
        for i in slotPos.indices { slotPos[i] -= Int32(shift) }
        positionShift += shift
        eval(keys)
    }
}
