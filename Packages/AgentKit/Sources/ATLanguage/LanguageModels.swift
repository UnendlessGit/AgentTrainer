import Foundation

/// The on-device language models AgentTrainer can download, and where they live.
public enum LanguageModels {
    public struct HubModel: Sendable {
        public let id: String
        public let title: String
        public let repo: String
        public let license: String
        public let approximateBytes: Int64
        public let files: [String]
    }

    /// EmbeddingGemma-300m: the text encoder Pixel2Play's text token was trained with.
    public static let embeddingGemma = HubModel(
        id: "embeddinggemma", title: "Instruction model (EmbeddingGemma 300M)", repo: "mlx-community/embeddinggemma-300m-bf16",
        license: "Gemma Terms of Use", approximateBytes: 660_000_000,
        files: ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "modules.json"])

    /// Local vision-language model used to write instructions for recordings (auto-labelling).
    public static let labeler = HubModel(
        id: "labeler", title: "Labelling model (Qwen3-VL 4B, 4-bit)", repo: "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        license: "Apache 2.0", approximateBytes: 3_100_000_000,
        files: [])

    public static func directory(_ m: HubModel, pretrainedRoot: URL) -> URL {
        pretrainedRoot.appendingPathComponent("hub").appendingPathComponent(m.repo.replacingOccurrences(of: "/", with: "--"))
    }

    public static func isDownloaded(_ m: HubModel, pretrainedRoot: URL) -> Bool {
        let d = directory(m, pretrainedRoot: pretrainedRoot)
        return FileManager.default.fileExists(atPath: d.appendingPathComponent(".complete").path)
    }

    nonisolated(unsafe) private static var cachedEncoder: (URL, InstructionEncoder)?
    private static let lock = NSLock()

    /// The instruction encoder, or nil if EmbeddingGemma hasn't been downloaded.
    public static func instructionEncoder(pretrainedRoot: URL) -> InstructionEncoder? {
        lock.lock(); defer { lock.unlock() }
        let dir = directory(embeddingGemma, pretrainedRoot: pretrainedRoot)
        if let c = cachedEncoder, c.0 == dir { return c.1 }
        guard isDownloaded(embeddingGemma, pretrainedRoot: pretrainedRoot), let e = try? GemmaInstructionEncoder(directory: dir) else { return nil }
        cachedEncoder = (dir, e)
        return e
    }
}
