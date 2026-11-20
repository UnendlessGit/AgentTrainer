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
        /// One line on what it's good at (labelling models).
        public var summary = ""
        /// Unified memory it needs while working, in GB (labelling models).
        public var memoryGB = 0.0
    }

    /// EmbeddingGemma-300m: the text encoder Pixel2Play's text token was trained with.
    public static let embeddingGemma = HubModel(
        id: "embeddinggemma", title: "Instruction model (EmbeddingGemma 300M)", repo: "mlx-community/embeddinggemma-300m-bf16",
        license: "Gemma Terms of Use", approximateBytes: 660_000_000,
        files: ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "modules.json"])

    /// Vision-language models that can write instructions for recordings, best first. Both watch
    /// the footage as video (motion between frames), not single pictures.
    public static let labelers: [HubModel] = [labelerBest, labelerLight]

    /// Qwen3.8 27B: the newest Qwen, dense, natively trained on images and video. The same
    /// files as LM Studio's MLX build, so an LM Studio copy can be reused as is.
    public static let labelerBest = HubModel(
        id: "labeler-qwen38-27b", title: "Qwen3.8 27B", repo: "lmstudio-community/Qwen3.8-27B-MLX-4bit",
        license: "Apache 2.0", approximateBytes: 16_100_000_000, files: [],
        summary: "Understands scenes and motion far better. Recommended.", memoryGB: 22)

    /// Qwen3-VL 4B: the original small labeller.
    public static let labelerLight = HubModel(
        id: "labeler", title: "Qwen3-VL 4B", repo: "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        license: "Apache 2.0", approximateBytes: 3_100_000_000, files: [],
        summary: "Small and quick, but often misreads scenes.", memoryGB: 6)

    /// The labeller with this id, if it's one of `labelers`.
    public static func labeler(id: String?) -> HubModel? { labelers.first { $0.id == id } }

    /// The best labeller that is downloaded, if any.
    public static func bestDownloadedLabeler(pretrainedRoot: URL) -> HubModel? {
        labelers.first { isDownloaded($0, pretrainedRoot: pretrainedRoot) }
    }

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
