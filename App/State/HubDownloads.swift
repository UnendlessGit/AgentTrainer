import ATCore
import ATLanguage
import Foundation
import Observation

/// Downloads a Hugging Face model folder (file by file, resumable per file) into Pretrained/hub.
@MainActor @Observable
final class HubDownloads {
    struct State: Equatable {
        var received: Int64 = 0
        var total: Int64 = 0
        var running = false
        var done = false
        var error: String?
        var fraction: Double { total > 0 ? Double(received) / Double(total) : 0 }
    }

    private(set) var states: [String: State] = [:]
    private var tasks: [String: Task<Void, Never>] = [:]
    let pretrainedRoot: URL

    init(pretrainedRoot: URL) {
        self.pretrainedRoot = pretrainedRoot
        for m in [LanguageModels.embeddingGemma, LanguageModels.labeler] {
            states[m.id] = State(done: LanguageModels.isDownloaded(m, pretrainedRoot: pretrainedRoot))
        }
    }

    func state(_ m: LanguageModels.HubModel) -> State { states[m.id] ?? State() }

    func start(_ m: LanguageModels.HubModel) {
        guard tasks[m.id] == nil else { return }
        states[m.id] = State(running: true)
        let root = pretrainedRoot
        tasks[m.id] = Task { [weak self] in
            do {
                let dir = LanguageModels.directory(m, pretrainedRoot: root)
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                let listing = try await Self.listing(m.repo)
                let files = m.files.isEmpty ? listing.filter { !$0.path.hasPrefix(".") && $0.path != "README.md" }
                    : listing.filter { m.files.contains($0.path) }
                let total = files.reduce(0) { $0 + $1.size }
                await MainActor.run { self?.states[m.id]?.total = total }
                var done: Int64 = 0
                for f in files {
                    try Task.checkCancellation()
                    let dest = dir.appendingPathComponent(f.path)
                    if let s = (try? FileManager.default.attributesOfItem(atPath: dest.path))?[.size] as? Int64, s == f.size { done += s; continue }
                    let url = URL(string: "https://huggingface.co/\(m.repo)/resolve/main/\(f.path)")!
                    let (tmp, _) = try await URLSession.shared.download(from: url, delegate: Progress { r in
                        Task { @MainActor in self?.states[m.id]?.received = done + r }
                    })
                    try? FileManager.default.removeItem(at: dest)
                    try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try FileManager.default.moveItem(at: tmp, to: dest)
                    done += f.size
                    await MainActor.run { self?.states[m.id]?.received = done }
                }
                FileManager.default.createFile(atPath: dir.appendingPathComponent(".complete").path, contents: nil)
                await MainActor.run { self?.states[m.id] = State(received: total, total: total, done: true); self?.tasks[m.id] = nil }
            } catch {
                await MainActor.run {
                    self?.states[m.id]?.running = false
                    self?.states[m.id]?.error = error is CancellationError ? nil : error.localizedDescription
                    self?.tasks[m.id] = nil
                }
            }
        }
    }

    func cancel(_ m: LanguageModels.HubModel) { tasks[m.id]?.cancel() }

    func delete(_ m: LanguageModels.HubModel) {
        cancel(m)
        try? FileManager.default.removeItem(at: LanguageModels.directory(m, pretrainedRoot: pretrainedRoot))
        states[m.id] = State()
    }

    struct FileEntry: Decodable { let path: String; let size: Int64; let type: String }

    static func listing(_ repo: String) async throws -> [FileEntry] {
        let (d, _) = try await URLSession.shared.data(from: URL(string: "https://huggingface.co/api/models/\(repo)/tree/main?recursive=true")!)
        return try JSONDecoder().decode([FileEntry].self, from: d).filter { $0.type == "file" }
    }

    /// URLSession per-task delegate reporting bytes written.
    final class Progress: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
        let onBytes: (Int64) -> Void
        init(_ f: @escaping (Int64) -> Void) { onBytes = f }
        func urlSession(_ s: URLSession, downloadTask: URLSessionDownloadTask, didWriteData _: Int64, totalBytesWritten w: Int64, totalBytesExpectedToWrite _: Int64) { onBytes(w) }
        func urlSession(_ s: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
    }
}
