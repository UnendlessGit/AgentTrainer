import ATCapture
import ATCore
import ATLanguage
import ATModel
import Foundation
import MLX

/// Builds a model's per-step language inputs (instruction bytes, on-screen text) and caches
/// text embeddings by string. Call from the thread that runs the model.
public final class LanguageFeed {
    let architecture: PolicyArchitecture
    let encoder: InstructionEncoder?
    private var cache: [String: [Float]] = [:]
    private var instruction: String?
    private var items: [ScreenTextItem] = []
    private var dirty = true

    public init(architecture: PolicyArchitecture, encoder: InstructionEncoder?) {
        self.architecture = architecture
        self.encoder = encoder
    }

    public var needed: Bool { architecture.usesDecoderMemory }

    public func setInstruction(_ s: String?) { if s != instruction { instruction = s; dirty = true } }
    public func setScreenText(_ i: [ScreenTextItem]) { if i != items { items = i; dirty = true } }

    /// Returns new inputs when something changed (embedding at most `budget` new strings now).
    public func update(budget: Int = 6) -> LanguageInputs?? {
        guard needed, dirty else { return .none }
        var l = LanguageInputs()
        if architecture.screenText {
            let K = architecture.screenTextItems
            let chosen = Array(items.prefix(K))
            let missing = chosen.map(\.text).filter { cache[$0] == nil }
            if !missing.isEmpty, let enc = encoder, let e = try? enc.embed(Array(missing.prefix(budget))) {
                for (t, v) in zip(missing.prefix(budget), e) { cache[t] = v }
            }
            var emb = [Float](), box = [Float](), mask = [Bool]()
            for k in 0 ..< K {
                if k < chosen.count, let v = cache[chosen[k].text] {
                    let it = chosen[k]
                    emb += v; box += [it.x, it.y, it.w, it.h, it.confidence]; mask.append(true)
                } else {
                    emb += [Float](repeating: 0, count: architecture.textEmbeddingDim); box += [0, 0, 0, 0, 0]; mask.append(false)
                }
            }
            l.screenEmbeddings = MLXArray(emb, [K, architecture.textEmbeddingDim])
            l.screenBoxes = MLXArray(box, [K, 5])
            l.screenMask = MLXArray(mask)
            if missing.count <= budget { dirty = false }
        } else {
            dirty = false
        }
        if architecture.literalTyping {
            l.typingBytes = MLXArray(LanguageInputs.bytes(instruction, length: architecture.typingBytes))
        }
        return .some(l)
    }
}
