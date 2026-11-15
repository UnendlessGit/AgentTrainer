import ATCore
import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
import Tokenizers

/// Bridges swift-transformers tokenizers to MLXLMCommon (what `#huggingFaceTokenizerLoader` expands to).
struct TransformersTokenizerLoader: MLXLMCommon.TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        TokenizerBridge(try await Tokenizers.AutoTokenizer.from(modelFolder: directory))
    }
}

struct TokenizerBridge: MLXLMCommon.Tokenizer {
    let upstream: any Tokenizers.Tokenizer
    init(_ u: any Tokenizers.Tokenizer) { upstream = u }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { upstream.encode(text: text, addSpecialTokens: addSpecialTokens) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens) }
    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }
    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}

/// Runs an async load synchronously (models are loaded on background threads).
func blocking<T>(_ body: @escaping @Sendable () async throws -> T) throws -> T {
    let sem = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var result: Result<T, Error>?
    Task.detached {
        do { result = .success(try await body()) } catch { result = .failure(error) }
        sem.signal()
    }
    sem.wait()
    return try result!.get()
}

/// EmbeddingGemma-300m exactly as Pixel2Play uses it: the transformer's last hidden state,
/// masked-mean pooled over real tokens (no Dense projection, no normalization), 768-d.
public final class GemmaInstructionEncoder: InstructionEncoder, @unchecked Sendable {
    public let dimension = 768
    let model: any EmbeddingModel
    let tokenizer: any MLXLMCommon.Tokenizer
    let maxTokens = 128
    private let lock = NSLock()

    public init(directory: URL) throws {
        let ctx = try blocking { try await EmbedderModelFactory.shared.load(from: directory, using: TransformersTokenizerLoader()) }
        model = ctx.model
        tokenizer = ctx.tokenizer
    }

    public func embed(_ texts: [String]) throws -> [[Float]] {
        lock.lock(); defer { lock.unlock() }
        guard !texts.isEmpty else { return [] }
        let ids = texts.map { Array(tokenizer.encode(text: $0, addSpecialTokens: true).prefix(maxTokens)) }
        let L = max(1, ids.map(\.count).max() ?? 1)
        var flat = [Int32](), mask = [Int32]()
        for row in ids {
            flat += row.map(Int32.init) + Array(repeating: 0, count: L - row.count)
            mask += Array(repeating: 1, count: row.count) + Array(repeating: 0, count: L - row.count)
        }
        let inputs = MLXArray(flat, [texts.count, L]), m = MLXArray(mask, [texts.count, L])
        let out = model(inputs, positionIds: nil, tokenTypeIds: nil, attentionMask: m)
        guard let hidden = out.hiddenStates else { return texts.map { _ in Array(repeating: 0, count: dimension) } }
        let mf = m.asType(.float32).expandedDimensions(axis: -1)
        let pooled = (hidden.asType(.float32) * mf).sum(axis: 1) / maximum(mf.sum(axis: 1), MLXArray(Float(1)))
        eval(pooled)
        let v = pooled.asArray(Float.self)
        return (0 ..< texts.count).map { Array(v[$0 * dimension ..< ($0 + 1) * dimension]) }
    }
}
