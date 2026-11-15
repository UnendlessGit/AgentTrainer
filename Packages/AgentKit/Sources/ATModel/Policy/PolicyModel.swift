import ATCore
import Foundation
import MLX
import MLXNN
import MLXRandom

/// Gated cross-attention from decoder tokens to a memory of on-screen text items and instruction
/// bytes. The gate starts at 0 (tanh), so a pretrained decoder behaves exactly as before until
/// training opens it (Flamingo-style).
final class MemoryCrossAttention: Module {
    @ModuleInfo var norm: RMSNormF32
    @ModuleInfo var query: Linear
    @ModuleInfo(key: "key_value") var keyValue: Linear
    @ModuleInfo var out: Linear
    @ParameterInfo var gate: MLXArray
    let heads: Int, headDim: Int

    init(width: Int, heads: Int) {
        self.heads = heads
        headDim = width / heads
        _norm.wrappedValue = RMSNormF32(dimensions: width)
        _query.wrappedValue = Linear(width, width, bias: false)
        _keyValue.wrappedValue = Linear(width, 2 * width, bias: false)
        _out.wrappedValue = Linear(width, width, bias: false)
        _gate.wrappedValue = MLXArray.zeros([1])
    }

    func callAsFunction(_ x: MLXArray, memory: MLXArray, mask: MLXArray) -> MLXArray {
        let N = x.dim(0), L = x.dim(1), M = memory.dim(1), D = heads * headDim
        let q = query(norm(x)).reshaped(N, L, heads, headDim).transposed(0, 2, 1, 3)
        let kv = keyValue(memory.asType(x.dtype))
        let k = kv[.ellipsis, 0 ..< D].reshaped(N, M, heads, headDim).transposed(0, 2, 1, 3)
        let v = kv[.ellipsis, D...].reshaped(N, M, heads, headDim).transposed(0, 2, 1, 3)
        let a = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / Float(headDim).squareRoot(),
                                                  mask: .array(mask.reshaped(N, 1, 1, M)))
        return x + tanh(gate).asType(x.dtype) * out(a.transposed(0, 2, 1, 3).reshaped(N, L, D))
    }
}

/// Pixel2Play's autoregressive action decoder: a small causal transformer that turns the
/// backbone's `a_out` token into one token per action slot.
final class ActionDecoder: Module {
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ParameterInfo(key: "pos_tokens") var posTokens: MLXArray
    @ModuleInfo(key: "transformer_layers") var layers: [TransformerLayer]
    @ModuleInfo(key: "memory_attention") var memoryAttention: [MemoryCrossAttention]
    let slots: Int

    init(width: Int, layers n: Int, heads: Int, sinks: Int, slots: Int, dropout: Float, memory: Bool = false) {
        self.slots = slots
        _inputProj.wrappedValue = Linear(width, width, bias: true)
        _posTokens.wrappedValue = MLXRandom.normal([slots + 1, width]) * 3.0
        _layers.wrappedValue = (0 ..< n).map { _ in
            TransformerLayer(width: width, heads: heads, kvHeads: heads, dropout: dropout, sinks: sinks)
        }
        _memoryAttention.wrappedValue = memory ? (0 ..< n).map { _ in MemoryCrossAttention(width: width, heads: heads) } : []
    }

    static func causalMask(_ n: Int) -> MLXArray {
        let i = MLXArray(Int32(0) ..< Int32(n))
        return i.expandedDimensions(axis: 1) .>= i.expandedDimensions(axis: 0)
    }

    /// Runs the decoder over `[proj(a_out), inputs...]` and returns hidden states `[N, L, D]`
    /// (L = 1 + inputs.count). Used both teacher-forced (training) and per sampling step.
    func run(aOut: MLXArray, inputs: MLXArray?, memory: MLXArray? = nil, memoryMask: MLXArray? = nil) -> MLXArray {
        var x = inputProj(aOut).expandedDimensions(axis: 1)
        if let inputs { x = concatenated([x, inputs], axis: 1) }
        let L = x.dim(1)
        x = x + posTokens[0 ..< L].asType(x.dtype)
        let mask = Self.causalMask(L)
        for (i, layer) in layers.enumerated() {
            x = layer(x, mask: mask)
            if let memory, let memoryMask, i < memoryAttention.count { x = memoryAttention[i](x, memory: memory, mask: memoryMask) }
        }
        return x
    }
}

/// Logits for one batch of timesteps, grouped by head.
public struct PolicyLogits {
    public let keys: MLXArray     // [B, T, keySlots, keyClasses]
    public let pointer: MLXArray  // [B, T, pointerSlots, pointerClasses]
    public let x: MLXArray        // [B, T, 1, xClasses]
    public let y: MLXArray        // [B, T, 1, yClasses]
}

/// The full Pixel2Play-style policy: EfficientNet image tokens + text token + thinking token +
/// `a_out` + previous-action tokens per timestep, a block-causal transformer backbone, and an
/// autoregressive action decoder with shared action embeddings.
public final class PolicyModel: Module {
    @ModuleInfo var vision: ConvTokenizer
    @ModuleInfo(key: "backbone") var backbone: [TransformerLayer]
    @ModuleInfo(key: "text_proj") var textProj: Linear
    @ParameterInfo(key: "img_pos") var imgPos: MLXArray
    @ParameterInfo(key: "text_pos") var textPos: MLXArray
    @ParameterInfo(key: "no_text") var noText: MLXArray
    @ParameterInfo(key: "thinking") var thinking: MLXArray
    @ParameterInfo(key: "action_out") var actionOut: MLXArray
    @ParameterInfo(key: "action_pos") var actionPos: MLXArray
    /// Stands in for past actions that are hidden (history dropout, or running without history).
    @ParameterInfo(key: "action_unknown") var actionUnknown: MLXArray
    @ModuleInfo var decoder: ActionDecoder
    @ModuleInfo(key: "emb_key") var embKey: Embedding
    @ModuleInfo(key: "emb_pointer") var embPointer: Embedding
    @ModuleInfo(key: "emb_x") var embX: Embedding
    @ModuleInfo(key: "emb_y") var embY: Embedding
    @ModuleInfo(key: "head_key") var headKey: Linear
    @ModuleInfo(key: "head_pointer") var headPointer: Linear
    @ModuleInfo(key: "head_x") var headX: Linear
    @ModuleInfo(key: "head_y") var headY: Linear
    // Optional language extensions (screen text, literal typing).
    @ModuleInfo(key: "screen_item_proj") var screenItemProj: Linear?
    @ModuleInfo(key: "screen_summary") var screenSummary: Linear?
    @ModuleInfo(key: "byte_emb") var byteEmb: Embedding?
    @ParameterInfo(key: "byte_pos") var bytePos: MLXArray?
    @ParameterInfo(key: "memory_null") var memoryNull: MLXArray?

    /// Recompute activations in backward (needed to fit 10 s clips in memory).
    public var gradientCheckpointing = false
    /// Frames per clip that get gradients through the EfficientNet stem (0 = all frames).
    public var visionGradientFrames = 32
    private var layerCheckpoints: [ModuleCheckpoint<TransformerLayer>] = []

    public let architecture: PolicyArchitecture
    public let actions: ActionSpaceConfig
    public let layout: PolicyLayout
    private var maskCache: [String: MLXArray] = [:]
    /// Backbone outputs at the `a_out` positions from the last forward (diagnostics).
    public internal(set) var lastAOutStates: MLXArray?
    /// Intermediate tensors from the last forward, for parity debugging.
    public var debugTrace: [String: MLXArray] = [:]

    public init(architecture a: PolicyArchitecture, actions: ActionSpaceConfig) {
        self.architecture = a
        self.actions = actions
        self.layout = PolicyLayout(thinkingTokens: a.thinkingTokens, slots: actions.slotCount)
        let d = a.width
        let dropout = Float(a.dropout)
        _vision.wrappedValue = ConvTokenizer(frameSize: a.frameSize, width: d, stochasticDepth: Float(a.stochasticDepth))
        _backbone.wrappedValue = (0 ..< a.layers).map { _ in
            TransformerLayer(width: d, heads: a.heads, kvHeads: a.resolvedKVHeads, dropout: dropout, sinks: 0)
        }
        _textProj.wrappedValue = Linear(a.textEmbeddingDim, d, bias: false)
        _imgPos.wrappedValue = MLXRandom.normal([1, d]) * 0.05
        _textPos.wrappedValue = MLXRandom.normal([1, d]) * 0.05
        _noText.wrappedValue = MLXRandom.normal([1, d]) * 0.05
        _thinking.wrappedValue = MLXRandom.normal([a.thinkingTokens, d]) * 0.05
        _actionOut.wrappedValue = MLXRandom.normal([1, d]) * 0.05
        _actionPos.wrappedValue = MLXRandom.normal([actions.slotCount, d]) * 0.05
        _actionUnknown.wrappedValue = MLXArray.zeros([actions.slotCount, d])
        _decoder.wrappedValue = ActionDecoder(width: d, layers: a.decoderLayers, heads: a.decoderHeads,
                                              sinks: a.decoderSinks, slots: actions.slotCount, dropout: dropout, memory: a.usesDecoderMemory)
        if a.screenText {
            _screenItemProj.wrappedValue = Linear(a.textEmbeddingDim + 5, d)
            let summary = Linear(d, d)
            summary.update(parameters: ModuleParameters.unflattened(["weight": MLXArray.zeros([d, d]), "bias": MLXArray.zeros([d])]))
            _screenSummary.wrappedValue = summary
        }
        if a.literalTyping {
            let e = Embedding(embeddingCount: 257, dimensions: d)
            e.update(parameters: ModuleParameters.unflattened(["weight": MLXRandom.normal([257, d]) * 0.02]))
            _byteEmb.wrappedValue = e
            _bytePos.wrappedValue = MLXRandom.normal([a.typingBytes, d]) * 0.02
        }
        if a.usesDecoderMemory { _memoryNull.wrappedValue = MLXRandom.normal([1, d]) * 0.02 }
        func emb(_ n: Int) -> Embedding {
            let e = Embedding(embeddingCount: n, dimensions: d)
            e.update(parameters: ModuleParameters.unflattened(["weight": MLXRandom.normal([n, d]) * 0.1]))
            return e
        }
        _embKey.wrappedValue = emb(actions.keyClasses)
        _embPointer.wrappedValue = emb(actions.pointerClasses)
        _embX.wrappedValue = emb(actions.xClasses)
        _embY.wrappedValue = emb(actions.yClasses)
        _headKey.wrappedValue = Linear(d, actions.keyClasses)
        _headPointer.wrappedValue = Linear(d, actions.pointerClasses)
        _headX.wrappedValue = Linear(d, actions.xClasses)
        _headY.wrappedValue = Linear(d, actions.yClasses)
        super.init()
    }

    public var width: Int { architecture.width }
    public var computeDType: DType { imgPos.dtype }

    // MARK: Pieces shared by training and streaming

    /// uint8 `[N, H, W, 3]` RGB → `[N, D]` image tokens (P2P: `/255`, no mean/std).
    func imageTokens(_ frames: MLXArray) -> MLXArray {
        let n = frames.dim(0)
        guard training, visionGradientFrames > 0, n > visionGradientFrames else { return vision(normalize(frames)) + imgPos }
        // Stem gradient sampling: every frame goes through the EfficientNet stem without gradients,
        // a random subset is recomputed with gradients (scaled by n/k so the estimate is unbiased).
        // BatchNorm statistics are frozen in training, so both passes give identical values.
        let k = visionGradientFrames
        let all = stopGradient(vision.stem(normalize(frames))).reshaped(n, -1)
        let order = argSort(MLXRandom.uniform(0 ..< 1, [n]))[0 ..< k]
        let sub = vision.stem(normalize(frames[order])).reshaped(k, -1)
        let mixed = all.at[order].add((sub - stopGradient(sub)) * (Float(n) / Float(k)))
        return vision.norm(vision.proj(mixed)) + imgPos
    }

    func normalize(_ frames: MLXArray) -> MLXArray {
        var f = frames.asType(computeDType) / 255
        if architecture.grayscale {
            let g = (f[.ellipsis, 0 ..< 1] * 0.299 + f[.ellipsis, 1 ..< 2] * 0.587 + f[.ellipsis, 2 ..< 3] * 0.114)
            f = concatenated([g, g, g], axis: -1)
        }
        return f
    }

    /// Training mode with frozen BatchNorm statistics in the vision stem (affine parameters still learn).
    public func setTraining() {
        train(true)
        for (_, m) in vision.namedModules() where m is BatchNorm { m.train(false) }
    }

    /// Excludes BatchNorm running statistics from the trainable parameters (no weight decay on them).
    public func freezeBatchNormStatistics() {
        for (_, m) in vision.namedModules() where m is BatchNorm {
            m.freeze(recursive: false, keys: ["running_mean", "running_var"])
        }
    }

    /// `[N, 768]` text embeddings + `[N]` presence → `[N, D]` text tokens.
    func textTokens(_ text: MLXArray, present: MLXArray) -> MLXArray {
        let projected = textProj(text.asType(computeDType))
        let chosen = which(present.expandedDimensions(axis: -1), projected, noText)
        return chosen + textPos
    }

    /// `[..., S]` action classes → `[..., S, D]` embeddings (no position tokens).
    func actionEmbeddings(_ a: MLXArray) -> MLXArray {
        let ks = actions.keySlots, ps = actions.pointerSlots
        let parts = [
            embKey(a[.ellipsis, 0 ..< ks]),
            embPointer(a[.ellipsis, ks ..< ks + ps]),
            embX(a[.ellipsis, (ks + ps) ..< (ks + ps + 1)]),
            embY(a[.ellipsis, (ks + ps + 1) ..< (ks + ps + 2)]),
        ]
        return concatenated(parts, axis: -2)
    }

    /// Logits for decoder outputs `[..., S, D]`.
    func logits(fromDecoder h: MLXArray) -> PolicyLogits {
        let ks = actions.keySlots, ps = actions.pointerSlots
        return PolicyLogits(
            keys: headKey(h[.ellipsis, 0 ..< ks, 0...]),
            pointer: headPointer(h[.ellipsis, ks ..< ks + ps, 0...]),
            x: headX(h[.ellipsis, (ks + ps) ..< (ks + ps + 1), 0...]),
            y: headY(h[.ellipsis, (ks + ps + 1) ..< (ks + ps + 2), 0...]))
    }

    /// The head for one slot (used while sampling).
    func headLogits(slot s: Int, _ h: MLXArray) -> MLXArray {
        switch actions.kind(ofSlot: s) {
        case .key: headKey(h)
        case .pointer: headPointer(h)
        case .x: headX(h)
        case .y: headY(h)
        }
    }

    func embedSlot(_ s: Int, _ classes: MLXArray) -> MLXArray {
        switch actions.kind(ofSlot: s) {
        case .key: embKey(classes)
        case .pointer: embPointer(classes)
        case .x: embX(classes)
        case .y: embY(classes)
        }
    }

    func history(forLayer i: Int, steps: Int) -> Int {
        if let h = architecture.historySteps, i < h.count { return max(0, min(h[i], steps) - 1) }
        return steps - 1
    }

    func mask(steps: Int, history: Int) -> MLXArray {
        let key = "\(steps)/\(history)"
        if let m = maskCache[key] { return m }
        let m = layout.fullMask(steps: steps, history: history)
        maskCache[key] = m
        return m
    }

    /// Screen-text item tokens, their summary (for the thinking token) and the decoder memory.
    func languageTokens(_ l: LanguageInputs?, leading: [Int]) -> (summary: MLXArray?, memory: MLXArray, mask: MLXArray)? {
        guard architecture.usesDecoderMemory, let l, let null = memoryNull else { return nil }
        let D = width
        var parts: [MLXArray] = [broadcast(null.asType(computeDType), to: leading + [1, D])]
        var masks: [MLXArray] = [MLXArray.ones(leading + [1], dtype: .bool)]
        var summary: MLXArray?
        if let proj = screenItemProj, let emb = l.screenEmbeddings, let box = l.screenBoxes, let m = l.screenMask {
            let items = proj(concatenated([emb.asType(computeDType), box.asType(computeDType)], axis: -1))
            let mf = m.asType(computeDType).expandedDimensions(axis: -1)
            let mean = (items * mf).sum(axis: -2) / maximum(mf.sum(axis: -2), MLXArray(Float(1)).asType(computeDType))
            summary = screenSummary.map { $0(mean) }
            parts.append(items); masks.append(m)
        }
        if let be = byteEmb, let pos = bytePos, let bytes = l.typingBytes {
            parts.append(be(bytes) + pos[0 ..< bytes.dim(-1)].asType(computeDType))
            masks.append(bytes .!= Int32(256))
        }
        return (summary, concatenated(parts, axis: -2), concatenated(masks, axis: -1))
    }

    // MARK: Training / reference forward

    /// Teacher-forced forward over whole clips.
    /// - frames: uint8 `[B, T, H, W, 3]`
    /// - actionsIn: int32 `[B, T, S]` actions taken after each frame (fed back as history and as decoder inputs)
    /// - text: `[B, T, 768]` instruction embeddings, `textPresent`: bool `[B, T]`
    /// - thinkingExtra: optional `[B, T, D]` added to the first thinking token (screen-text summary)
    public func callAsFunction(frames: MLXArray, actionsIn: MLXArray, text: MLXArray, textPresent: MLXArray,
                               thinkingExtra: MLXArray? = nil, historyKeep: MLXArray? = nil, language: LanguageInputs? = nil) -> PolicyLogits {
        let B = frames.dim(0), T = frames.dim(1), D = width, S = actions.slotCount
        let lang = languageTokens(language, leading: [B, T])
        var extras = [MLXArray]()
        if let t = thinkingExtra { extras.append(t) }
        if let s = lang?.summary { extras.append(s) }
        let thinkingExtra: MLXArray? = extras.isEmpty ? nil : extras.dropFirst().reduce(extras[0], +)
        let img = imageTokens(frames.reshaped(B * T, frames.dim(2), frames.dim(3), frames.dim(4))).reshaped(B, T, 1, D)
        let txt = textTokens(text.reshaped(B * T, -1), present: textPresent.reshaped(B * T)).reshaped(B, T, 1, D)
        var think = broadcast(thinking.expandedDimensions(axes: [0, 1]), to: [B, T, architecture.thinkingTokens, D])
        if let extra = thinkingExtra {
            let first = think[0..., 0..., 0 ..< 1, 0...] + extra.reshaped(B, T, 1, D).asType(think.dtype)
            think = architecture.thinkingTokens > 1 ? concatenated([first, think[0..., 0..., 1..., 0...]], axis: 2) : first
        }
        let aOut = broadcast(actionOut.expandedDimensions(axes: [0, 1]), to: [B, T, 1, D])
        let actEmb = actionEmbeddings(actionsIn)
        var act = actEmb + actionPos
        if let keep = historyKeep {
            act = which(keep.reshaped(B, T, 1, 1), act, broadcast(actionUnknown + actionPos, to: act.shape).asType(act.dtype))
        }
        var x = concatenated([txt, img, think, aOut, act], axis: 2).reshaped(B, T * layout.tokensPerStep, D)
        debugTrace["seq_in"] = x
        if gradientCheckpointing && training && layerCheckpoints.count != backbone.count {
            layerCheckpoints = backbone.map { l in ModuleCheckpoint(l) { m, a in [m(a[0], mask: a[1])] } }
        }
        for (i, layer) in backbone.enumerated() {
            let m = mask(steps: T, history: history(forLayer: i, steps: T))
            if gradientCheckpointing && training {
                x = layerCheckpoints[i]([x, m])[0]
            } else {
                x = layer(x, mask: m)
            }
            if i == 0 { debugTrace["layer0"] = x }
        }
        debugTrace["seq_out"] = x
        let aOutStates = x.reshaped(B, T, layout.tokensPerStep, D)[0..., 0..., layout.aOutIndex, 0...]
        lastAOutStates = aOutStates
        let h = decoder.run(aOut: aOutStates.reshaped(B * T, D), inputs: actEmb.reshaped(B * T, S, D),
                            memory: lang?.memory.reshaped(B * T, -1, D), memoryMask: lang?.mask.reshaped(B * T, -1))
        let out = h[0..., 0 ..< S, 0...].reshaped(B, T, S, D)
        return logits(fromDecoder: out)
    }
}

/// Optional per-timestep language inputs (all shaped with the same leading dims as the frames).
public struct LanguageInputs {
    public var screenEmbeddings: MLXArray?  // [..., K, 768]
    public var screenBoxes: MLXArray?       // [..., K, 5]  x, y, w, h, confidence
    public var screenMask: MLXArray?        // [..., K] bool
    public var typingBytes: MLXArray?       // [..., L] int32, 256 = padding
    public init(screenEmbeddings: MLXArray? = nil, screenBoxes: MLXArray? = nil, screenMask: MLXArray? = nil, typingBytes: MLXArray? = nil) {
        self.screenEmbeddings = screenEmbeddings; self.screenBoxes = screenBoxes; self.screenMask = screenMask; self.typingBytes = typingBytes
    }

    /// UTF-8 bytes of an instruction, padded/truncated to `length` (256 = padding).
    public static func bytes(_ s: String?, length: Int) -> [Int32] {
        let b = Array((s ?? "").utf8.prefix(length)).map(Int32.init)
        return b + Array(repeating: 256, count: length - b.count)
    }
}
