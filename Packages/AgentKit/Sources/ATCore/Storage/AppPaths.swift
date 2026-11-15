import Foundation

/// Where AgentTrainer keeps data. Roots are user-configurable (Settings); defaults live in
/// `~/Library/Application Support/AgentTrainer`.
public struct AppPaths: Sendable {
    public var libraryRoot: URL
    public var modelsRoot: URL
    public var cacheRoot: URL
    public var pretrainedRoot: URL

    public static var defaultBase: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("AgentTrainer", isDirectory: true)
    }

    public static var standard: AppPaths {
        let b = defaultBase
        return AppPaths(libraryRoot: b.appendingPathComponent("Library", isDirectory: true),
                        modelsRoot: b.appendingPathComponent("Models", isDirectory: true),
                        cacheRoot: b.appendingPathComponent("Cache", isDirectory: true),
                        pretrainedRoot: b.appendingPathComponent("Pretrained", isDirectory: true))
    }

    public init(libraryRoot: URL, modelsRoot: URL, cacheRoot: URL, pretrainedRoot: URL) {
        self.libraryRoot = libraryRoot; self.modelsRoot = modelsRoot; self.cacheRoot = cacheRoot; self.pretrainedRoot = pretrainedRoot
    }

    public func ensureExists() throws {
        for u in [libraryRoot, modelsRoot, cacheRoot, pretrainedRoot] {
            try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        }
    }

    public func pretrainedCheckpoint(_ i: ModelInitialization) -> URL? {
        i.checkpointFolder.map { pretrainedRoot.appendingPathComponent("p2p-\($0)").appendingPathComponent("checkpoint.ckpt") }
    }

    public var hubCache: URL { pretrainedRoot.appendingPathComponent("hub", isDirectory: true) }
}
