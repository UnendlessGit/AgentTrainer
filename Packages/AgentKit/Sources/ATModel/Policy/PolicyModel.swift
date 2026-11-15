import ATCore
import Foundation
import MLX
import MLXNN
import MLXRandom

/// Pixel2Play's autoregressive action decoder: a small causal transformer that turns the
/// backbone's `a_out` token into one token per action slot.
final class ActionDecoder: Module {
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ParameterInfo(key: "pos_tokens") var posTokens: MLXArray
    @ModuleInfo(key: "transformer_layers") var layers: [TransformerLayer]
    let slots: Int

    init(width: Int, layers n: Int, heads: Int, sinks: Int, slots: Int, dropout: Float) {
        self.slots = slots
        _inputProj.wrappedValue = Linear(width, width, bias: true)
        _posTokens.wrappedValue = MLXRandom.normal([slots + 1, width]) * 3.0
        _layers.wrappedValue = (0 ..< n).map { _ in
            TransformerLayer(width: width, heads: heads, kvHeads: heads, dropout: dropout, sinks: sinks)
        }
    }

    static func causalMask(_ n: Int) -> MLXArray {
        let i = MLXArray(Int32(0) ..< Int32(n))
        return i.expandedDimensions(axis: 1) .>= i.expandedDimensions(axis: 0)
    }

    /// Runs the decoder over `[proj(a_out), inputs...]` and returns hidden states `[N, L, D]`
    /// (L = 1 + inputs.count). Used both teacher-forced (training) and per sampling step.
    func run(aOut: MLXArray, inputs: MLXArray?) -> MLXArray {
        var x = inputProj(aOut).expandedDimensions(axis: 1)
        if let inputs { x = concatenated([x, inputs], axis: 1) }
        let L = x.dim(1)
        x = x + posTokens[0 ..< L].asType(x.dtype)
        let mask = Self.causalMask(L)
        for layer in layers { x = layer(x, mask: mask) }
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
            TransformerLayer(width: d, heads: a.heads, kvHeads: a.kvHeads, dropout: dropout, sinks: 0)
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
                                              sinks: a.decoderSinks, slots: actions.slotCount, dropout: dropout)
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

    // MARK: Training / reference forward

    /// Teacher-forced forward over whole clips.
    /// - frames: uint8 `[B, T, H, W, 3]`
    /// - actionsIn: int32 `[B, T, S]` actions taken after each frame (fed back as history and as decoder inputs)
    /// - text: `[B, T, 768]` instruction embeddings, `textPresent`: bool `[B, T]`
    /// - thinkingExtra: optional `[B, T, D]` added to the first thinking token (screen-text summary)
    public func callAsFunction(frames: MLXArray, actionsIn: MLXArray, text: MLXArray, textPresent: MLXArray,
                               thinkingExtra: MLXArray? = nil, historyKeep: MLXArray? = nil) -> PolicyLogits {
        let B = frames.dim(0), T = frames.dim(1), D = width, S = actions.slotCount
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
        let h = decoder.run(aOut: aOutStates.reshaped(B * T, D), inputs: actEmb.reshaped(B * T, S, D))
        let out = h[0..., 0 ..< S, 0...].reshaped(B, T, S, D)
        return logits(fromDecoder: out)
    }
}
