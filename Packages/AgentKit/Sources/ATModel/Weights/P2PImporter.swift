import ATCore
import Foundation
import MLX
import MLXNN

public struct ImportReport: Sendable, CustomStringConvertible {
    public var transferred = 0
    public var partiallyTransferred: [String] = []
    public var freshlyInitialized: [String] = []
    public var ignored: [String] = []
    public var description: String {
        "transferred \(transferred) tensors; partial: \(partiallyTransferred.count); fresh: \(freshlyInitialized.count); ignored: \(ignored.count)"
    }
}

public enum P2PImportError: Error, LocalizedError {
    case incompatible(String)
    public var errorDescription: String? {
        if case .incompatible(let m) = self { return "These P2P weights don't fit this model: \(m)" }
        return nil
    }
}

/// Loads Open Pixel2Play (elefantai/open-p2p) Lightning checkpoints into a `PolicyModel`.
public enum P2PImporter {
    /// Converts the checkpoint into this model's parameters. Rows for keys / pointer events the
    /// model adds beyond P2P's vocabulary keep fresh weights and start with near-zero probability.
    @discardableResult
    public static func load(checkpoint url: URL, into model: PolicyModel) throws -> ImportReport {
        let reader = try TorchCheckpointReader(url: url)
        let sd = try reader.stateDict()
        func t(_ k: String) throws -> MLXArray {
            guard let r = sd[k] else { throw TorchCheckpointError.missingEntry(k) }
            return try reader.load(r).asType(.float32)
        }
        let a = model.architecture
        let layerCount = sd.keys.filter { $0.hasPrefix("bc_transformer._transformer.transformer_layers.") && $0.hasSuffix(".ffn.w2.weight") }.count
        let width = try t("bc_transformer.text_embed_mlp.weight").dim(0)
        guard width == a.width, layerCount == a.layers, a.frameSize == 192 || a.frameSize == 256 else {
            throw P2PImportError.incompatible("checkpoint has \(layerCount) layers × \(width) wide; model has \(a.layers) × \(a.width) at \(a.frameSize)px")
        }
        let projW = try t("bc_transformer.image_tokenizer.mlp.0.weight")
        let s = a.frameSize / 16
        guard projW.dim(1) == 112 * s * s else {
            throw P2PImportError.incompatible("vision projection expects \(projW.dim(1) / 112) spatial cells; model has \(s * s)")
        }

        var out: [String: MLXArray] = [:]
        var report = ImportReport()

        // Vision: EfficientNet stem (conv weights OIHW → OHWI) and projection (CHW → HWC flatten).
        let stageOffsets = [0, 1, 3, 5, 8]
        for (k, _) in sd where k.hasPrefix("bc_transformer.image_tokenizer.efficientnet_preprocess.") {
            let rest = String(k.dropFirst("bc_transformer.image_tokenizer.efficientnet_preprocess.".count))
            guard !rest.hasSuffix("num_batches_tracked"), let target = mapEfficientNet(rest, stageOffsets: stageOffsets) else { continue }
            var v = try t(k)
            if v.ndim == 4 { v = v.transposed(0, 2, 3, 1) }
            out["vision.stem." + target] = v
        }
        out["vision.proj.weight"] = projW.reshaped(width, 112, s, s).transposed(0, 2, 3, 1).reshaped(width, -1)
        out["vision.proj.bias"] = try t("bc_transformer.image_tokenizer.mlp.0.bias")
        out["vision.norm.weight"] = try t("bc_transformer.image_tokenizer.mlp.1.weight")
        out["vision.norm.bias"] = try t("bc_transformer.image_tokenizer.mlp.1.bias")

        // Backbone and decoder transformer layers share P2P's naming.
        for (k, _) in sd {
            if k.hasPrefix("bc_transformer._transformer.transformer_layers.") {
                out["backbone." + k.dropFirst("bc_transformer._transformer.transformer_layers.".count)] = try t(k)
            } else if k.hasPrefix("bc_transformer.action_decoder.transformer_layers.") {
                out["decoder.transformer_layers." + k.dropFirst("bc_transformer.action_decoder.transformer_layers.".count)] = try t(k)
            }
        }
        out["decoder.input_proj.weight"] = try t("bc_transformer.action_decoder.input_proj.weight")
        out["decoder.input_proj.bias"] = try t("bc_transformer.action_decoder.input_proj.bias")
        out["text_proj.weight"] = try t("bc_transformer.text_embed_mlp.weight")
        out["img_pos"] = try t("bc_transformer.img_pos_tokens").reshaped(1, width)
        out["text_pos"] = try t("bc_transformer.text_pos_tokens").reshaped(1, width)
        out["no_text"] = try t("bc_transformer.text_embedding_for_no_text_input").reshaped(1, width)
        out["action_out"] = try t("bc_transformer.action_out_token").reshaped(1, width)
        let think = try t("bc_transformer.thinking_pos_tokens").reshaped(-1, width)
        if think.dim(0) == a.thinkingTokens { out["thinking"] = think } else { report.freshlyInitialized.append("thinking") }

        // Per-slot position tokens: P2P slots are keys[0..3], buttons[0..1], dx, dy.
        let acts = model.actions
        var slotMap: [Int: Int] = [:] // our slot → P2P slot
        for i in 0 ..< min(4, acts.keySlots) { slotMap[i] = i }
        for i in 0 ..< min(2, acts.pointerSlots) { slotMap[acts.keySlots + i] = 4 + i }
        slotMap[acts.keySlots + acts.pointerSlots] = 6
        slotMap[acts.keySlots + acts.pointerSlots + 1] = 7
        let p2pActionPos = try t("bc_transformer.action_pos_tokens").reshaped(-1, width)
        out["action_pos"] = remapRows(current: model.actionPos, source: p2pActionPos, map: slotMap)
        let p2pDecPos = try t("bc_transformer.action_decoder.pos_tokens")
        var decMap: [Int: Int] = [0: 0]
        for (ours, theirs) in slotMap { decMap[ours + 1] = theirs + 1 }
        out["decoder.pos_tokens"] = remapRows(current: model.decoder.posTokens, source: p2pDecPos, map: decMap)
        if slotMap.count < acts.slotCount { report.partiallyTransferred.append("action_pos") }

        // Vocabulary tables.
        var keyMap: [Int: Int] = [0: 0]
        let p2pKeyIndex = Dictionary(uniqueKeysWithValues: KeyCatalog.p2pKeys.enumerated().map { ($1, $0 + 1) })
        for (i, k) in acts.keys.enumerated() { if let j = p2pKeyIndex[k] { keyMap[i + 1] = j } }
        var ptrMap: [Int: Int] = [0: 0]
        for (i, e) in acts.pointerEvents.enumerated() {
            if let j = PointerEvent.p2pOrder.firstIndex(of: e) { ptrMap[i + 1] = j + 1 }
        }
        try transferTable(model: model, ours: "key", theirsEmbedding: "key_action_embedding.weight",
                          theirsHead: "keyboard_out_logits", map: keyMap, classes: acts.keyClasses, t: t, into: &out, report: &report)
        try transferTable(model: model, ours: "pointer", theirsEmbedding: "mouse_button_embedding.weight",
                          theirsHead: "mouse_button_out_logits", map: ptrMap, classes: acts.pointerClasses, t: t, into: &out, report: &report)
        if acts.mouseMode == .relative, acts.mouseXEdges == ActionSpaceConfig.p2pXEdges, acts.mouseYEdges == ActionSpaceConfig.p2pYEdges {
            let xm = Dictionary(uniqueKeysWithValues: (0 ..< acts.xClasses).map { ($0, $0) })
            let ym = Dictionary(uniqueKeysWithValues: (0 ..< acts.yClasses).map { ($0, $0) })
            try transferTable(model: model, ours: "x", theirsEmbedding: "mouse_delta_x_embedding.weight",
                              theirsHead: "mouse_delta_x_out_logits", map: xm, classes: acts.xClasses, t: t, into: &out, report: &report)
            try transferTable(model: model, ours: "y", theirsEmbedding: "mouse_delta_y_embedding.weight",
                              theirsHead: "mouse_delta_y_out_logits", map: ym, classes: acts.yClasses, t: t, into: &out, report: &report)
        } else {
            report.freshlyInitialized += ["emb_x", "emb_y", "head_x", "head_y"]
        }

        // Apply, checking every name exists in the model with the right shape.
        let current = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        for (k, v) in out {
            guard let c = current[k] else { report.ignored.append(k); out[k] = nil; continue }
            guard c.shape == v.shape else {
                throw P2PImportError.incompatible("\(k): checkpoint \(v.shape) vs model \(c.shape)")
            }
            out[k] = v.asType(c.dtype)
        }
        report.transferred = out.count
        for k in current.keys where out[k] == nil && !report.freshlyInitialized.contains(where: { k.hasPrefix($0) }) {
            report.freshlyInitialized.append(k)
        }
        model.update(parameters: ModuleParameters.unflattened(out))
        eval(model)
        return report
    }

    /// `rest` like "2.1.block.3.1.weight" → "blocks.4.project.bn.weight".
    static func mapEfficientNet(_ rest: String, stageOffsets: [Int]) -> String? {
        let p = rest.split(separator: ".").map(String.init)
        guard let stage = Int(p[0]) else { return nil }
        if stage == 0 {
            guard p.count == 3, let sub = Int(p[1]) else { return nil }
            return "stem.\(sub == 0 ? "conv" : "bn").\(p[2])"
        }
        guard stage <= 5, p.count >= 5, let i = Int(p[1]), p[2] == "block", let j = Int(p[3]) else { return nil }
        let block = stageOffsets[stage - 1] + i
        let parts = stage == 1 ? ["depthwise", "se", "project"] : ["expand", "depthwise", "se", "project"]
        guard j < parts.count else { return nil }
        let part = parts[j]
        if part == "se" { return "blocks.\(block).se.\(p[4]).\(p[5])" }
        guard let sub = Int(p[4]), p.count == 6 else { return nil }
        return "blocks.\(block).\(part).\(sub == 0 ? "conv" : "bn").\(p[5])"
    }

    static func remapRows(current: MLXArray, source: MLXArray, map: [Int: Int]) -> MLXArray {
        var rows = (0 ..< current.dim(0)).map { current[$0] }
        for (ours, theirs) in map where ours < rows.count && theirs < source.dim(0) { rows[ours] = source[theirs] }
        return stacked(rows).asType(.float32)
    }

    static func transferTable(model: PolicyModel, ours: String, theirsEmbedding: String, theirsHead: String, map: [Int: Int],
                              classes: Int, t: (String) throws -> MLXArray, into out: inout [String: MLXArray],
                              report: inout ImportReport) throws {
        let emb = try t(theirsEmbedding)
        let hw = try t(theirsHead + ".weight"), hb = try t(theirsHead + ".bias")
        let params = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        guard let curEmb = params["emb_\(ours).weight"], let curW = params["head_\(ours).weight"], let curB = params["head_\(ours).bias"] else { return }
        out["emb_\(ours).weight"] = remapRows(current: curEmb, source: emb, map: map)
        // New classes start silent: zero weights and a bias well below every pretrained class.
        let floor = hb.min().item(Float.self) - 3
        let silentW = MLXArray.zeros(like: curW), silentB = MLXArray.full(curB.shape, values: MLXArray(floor))
        out["head_\(ours).weight"] = remapRows(current: silentW, source: hw, map: map)
        out["head_\(ours).bias"] = remapRows(current: silentB.reshaped(-1, 1), source: hb.reshaped(-1, 1), map: map).reshaped(-1)
        if map.count < classes { report.partiallyTransferred.append("\(ours) vocabulary (\(map.count)/\(classes) classes from P2P)") }
    }
}

extension P2PImporter {
    /// Copies only P2P's EfficientNet stem (and its BatchNorm statistics) - a strong starting point
    /// for blank models of any width.
    public static func loadVisionStem(checkpoint url: URL, into model: PolicyModel) throws {
        let reader = try TorchCheckpointReader(url: url)
        let sd = try reader.stateDict()
        var out: [(String, MLXArray)] = []
        let current = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        for (k, ref) in sd where k.hasPrefix("bc_transformer.image_tokenizer.efficientnet_preprocess.") {
            let rest = String(k.dropFirst("bc_transformer.image_tokenizer.efficientnet_preprocess.".count))
            guard !rest.hasSuffix("num_batches_tracked"), let target = mapEfficientNet(rest, stageOffsets: [0, 1, 3, 5, 8]) else { continue }
            var v = try reader.load(ref).asType(.float32)
            if v.ndim == 4 { v = v.transposed(0, 2, 3, 1) }
            let name = "vision.stem." + target
            if let c = current[name], c.shape == v.shape { out.append((name, v.asType(c.dtype))) }
        }
        model.update(parameters: ModuleParameters.unflattened(out))
        eval(model)
    }
}
