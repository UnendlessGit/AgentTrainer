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
    private var language: LanguageInputs?
    private var textCacheKey: [Float]?
    private var textToken: MLXArray?
    // Per-step pieces compiled once with fixed shapes (weights captured as constants): far fewer
    // GPU dispatches than op-by-op execution, which matters when the GPU is shared with a game.
    private lazy var compiledVision: ([MLXArray]) -> [MLXArray] = {
        let m = model
        return compile { a in [m.imageTokens(a[0])] }
    }()

    private lazy var compiledMask: ([MLXArray]) -> [MLXArray] = {
        let tps = self.tps, img = Int32(layout.imageTypeCount), aOut = Int32(layout.aOutIndex), hist = Int32(contextSteps - 1)
        return compile { a in
            let t = MLXArray(Int32(tps))
            let qPos = a[0], kPos = a[1], valid = a[2]
            let qs = floorDivide(qPos, t).expandedDimensions(axis: 1), qr = (qPos % t).expandedDimensions(axis: 1)
            let ks = floorDivide(kPos, t).expandedDimensions(axis: 0), kr = (kPos % t).expandedDimensions(axis: 0)
            return [PolicyLayout.maskOps(qs: qs, qr: qr, ks: ks, kr: kr, imageTypes: img, aOut: aOut, history: hist) .&& valid.expandedDimensions(axis: 0)]
        }
    }()

    private lazy var compiledQKV: [([MLXArray]) -> [MLXArray]] = model.backbone.map { layer in
        compile { a in
            let (q, k, v) = layer.attention.projectQKV(layer.attentionNorm(a[0]), ropeOffset: a[1])
            return [q, k, v]
        }
    }

    private lazy var compiledAttendFFN: [([MLXArray]) -> [MLXArray]] = model.backbone.map { layer in
        compile { a in
            let h = a[0] + layer.attention.attend(a[1], a[2], a[3], mask: a[4])
            return [h + layer.ffn(layer.ffnNorm(h))]
        }
    }

    /// Sampling loop compiled once (weights are captured as constants; recreate the stepper after reloading weights).
    private lazy var compiledDecode: ([MLXArray]) -> [MLXArray] = makeDecoder(withMemory: false)
    private lazy var compiledDecodeWithMemory: ([MLXArray]) -> [MLXArray] = makeDecoder(withMemory: true)

    private func makeDecoder(withMemory: Bool) -> ([MLXArray]) -> [MLXArray] {
        let m = model
        let S = m.actions.slotCount, D = m.width
        let counts = m.actions.slotClassCounts
        return compile { (inputs: [MLXArray]) -> [MLXArray] in
            let aOut = inputs[0], u = inputs[1], temp = inputs[2]
            let memory = withMemory ? inputs[3] : nil, memoryMask = withMemory ? inputs[4] : nil
            var embs: [MLXArray] = []
            var out: [MLXArray] = []
            for s in 0 ..< S {
                let h = m.decoder.run(aOut: aOut, inputs: embs.isEmpty ? nil : concatenated(embs, axis: 1), memory: memory, memoryMask: memoryMask)
                let logits = m.headLogits(slot: s, h[0..., h.dim(1) - 1, 0...]).asType(.float32)
                let g = -log(-log(u[s, 0 ..< counts[s]]))
                let cls = argMax(logits / temp + g, axis: -1).asType(.int32)
                out.append(cls)
                embs.append(m.embedSlot(s, cls).reshaped(1, 1, D))
            }
            return [concatenated(out, axis: 0)]
        }
    }
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
        keys = (0 ..< a.layers).map { _ in MLXArray.zeros([1, a.resolvedKVHeads, cap, a.headDim], dtype: dt) }
        values = (0 ..< a.layers).map { _ in MLXArray.zeros([1, a.resolvedKVHeads, cap, a.headDim], dtype: dt) }
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

    /// Current screen text / typing inputs (shapes without the leading batch/time dims: `[K, 768]`,
    /// `[K, 5]`, `[K]`, `[L]`). Kept until replaced.
    public func setLanguage(_ l: LanguageInputs?) { language = l }

    /// Runs one timestep.
    /// - frame: uint8 `[H, W, 3]` at the model's resolution.
    /// - text: 768-d instruction embedding, or nil for "no instruction".
    /// - previousAction: what actually happened during the last step (defaults to what the model sampled).
    /// - forced: teacher-forced classes for this step's decoder (parity tests / replay evaluation).
    public func step(frame: MLXArray, text: [Float]?, previousAction: [Int32]? = nil, forced: [Int32]? = nil,
                     thinkingExtra: MLXArray? = nil, returnLogits: Bool = false, imageToken: MLXArray? = nil) -> StepResult {
        rebaseIfNeeded()
        let m = model
        let D = m.width, S = m.actions.slotCount
        var lang: LanguageInputs?
        if let l = language {
            lang = LanguageInputs(screenEmbeddings: l.screenEmbeddings?.expandedDimensions(axis: 0), screenBoxes: l.screenBoxes?.expandedDimensions(axis: 0),
                                  screenMask: l.screenMask?.expandedDimensions(axis: 0), typingBytes: l.typingBytes?.expandedDimensions(axis: 0))
        }
        let langTokens = m.languageTokens(lang, leading: [1])
        var thinkingExtra = thinkingExtra
        if let s = langTokens?.summary { thinkingExtra = thinkingExtra.map { $0.reshaped(1, -1) + s } ?? s }
        if let p = previousAction { previous = MLXArray(p).reshaped(1, S) }

        // Tokens: actions(t-1) then [text, image, thinking, a_out](t).
        let act = useActionHistory ? m.actionEmbeddings(previous) + m.actionPos : (m.actionUnknown + m.actionPos).expandedDimensions(axis: 0)
        let txt = textTokenFor(text)
        let img = (imageToken ?? compiledVision([frame.reshaped(1, frame.dim(0), frame.dim(1), frame.dim(2))])[0]).reshaped(1, 1, D)
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
        let mask = compiledMask([qPos, MLXArray(slotPos), MLXArray(slotValid)])[0]
        let offset = MLXArray(Int32(blockStart))

        for l in m.backbone.indices {
            let qkv = compiledQKV[l]([x, offset])
            keys[l][0..., 0..., first ..< first + tps, 0...] = qkv[1]
            values[l][0..., 0..., first ..< first + tps, 0...] = qkv[2]
            x = compiledAttendFFN[l]([x, qkv[0], keys[l], values[l], mask])[0]
        }
        let aOutState = x[0..., tps - 1, 0...] // a_out is the last token of the pass
        lastAOutState = aOutState

        // Autoregressive decoding: one decoder pass per slot (recomputing the ≤9-token prefix is
        // as cheap as caching because the decoder is weight-bandwidth bound).
        let memory = langTokens?.memory, memoryMask = langTokens?.mask
        if forced == nil && !returnLogits {
            let maxC = m.actions.slotClassCounts.max()!
            let u = MLXRandom.uniform(low: Float(1e-9), high: 1, [S, maxC])
            let classes = memory != nil
                ? compiledDecodeWithMemory([aOutState, u, MLXArray(max(temperature, 1e-4)), memory!, memoryMask!])[0]
                : compiledDecode([aOutState, u, MLXArray(max(temperature, 1e-4))])[0]
            eval(classes, keys, values)
            let c = canonical(classes.asArray(Int32.self))
            previous = MLXArray(c).reshaped(1, S)
            stepIndex += 1
            return StepResult(classes: c, logits: nil)
        }
        var inputs: [MLXArray] = []
        var sampled: [MLXArray] = []
        var allLogits: [MLXArray] = []
        for s in 0 ..< S {
            let h = m.decoder.run(aOut: aOutState, inputs: inputs.isEmpty ? nil : concatenated(inputs, axis: 1), memory: memory, memoryMask: memoryMask)
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
        if returnLogits { eval(classes, keys, values, allLogits) } else { eval(classes, keys, values) }
        let c = forced ?? canonical(classes.asArray(Int32.self))
        previous = MLXArray(c).reshaped(1, S)
        stepIndex += 1
        return StepResult(classes: c,
                          logits: returnLogits ? allLogits.map { $0.asArray(Float.self) } : nil)
    }

    /// Runs the (compiled) vision encoder on a frame and keeps the GPU clocked up between ticks.
    /// Returns the image token so a step can reuse it if no newer frame arrived.
    public func encode(frame: MLXArray) -> MLXArray {
        let t = compiledVision([frame.reshaped(1, frame.dim(0), frame.dim(1), frame.dim(2))])[0]
        eval(t)
        return t
    }

    /// The model samples each slot separately; training only ever shows sorted, de-duplicated,
    /// zero-padded key / pointer sets. Feed back (and return) the same canonical form.
    func canonical(_ c: [Int32]) -> [Int32] {
        let ks = model.actions.keySlots, ps = model.actions.pointerSlots
        func group(_ r: Range<Int>) -> [Int32] {
            let v = Array(Set(c[r].filter { $0 > 0 })).sorted()
            return Array((v + Array(repeating: 0, count: r.count)).prefix(r.count))
        }
        return group(0 ..< ks) + group(ks ..< ks + ps) + Array(c[(ks + ps)...])
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
