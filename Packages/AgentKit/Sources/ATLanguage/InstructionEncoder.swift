import Foundation

/// Turns instruction / screen text into the 768-d embeddings Pixel2Play's text token expects
/// (EmbeddingGemma-300m, masked mean of the last hidden state).
public protocol InstructionEncoder: AnyObject {
    var dimension: Int { get }
    func embed(_ texts: [String]) throws -> [[Float]]
}

/// Deduplicated text strings with cached embeddings.
public final class TextTable: @unchecked Sendable {
    public private(set) var strings: [String] = []
    private var index: [String: Int32] = [:]
    public private(set) var embeddings: [[Float]] = []
    private let lock = NSLock()

    public init() {}

    public func id(for s: String?) -> Int32 {
        guard let s, !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return -1 }
        lock.lock(); defer { lock.unlock() }
        if let i = index[s] { return i }
        let i = Int32(strings.count)
        strings.append(s)
        index[s] = i
        return i
    }

    public var count: Int { strings.count }

    /// Embeds every string not embedded yet.
    public func embedAll(with encoder: InstructionEncoder, batch: Int = 16) throws {
        while embeddings.count < strings.count {
            let next = Array(strings[embeddings.count ..< min(strings.count, embeddings.count + batch)])
            embeddings += try encoder.embed(next)
        }
    }
}
