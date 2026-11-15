import ATCore
import Foundation
import Observation

/// Downloads pretrained weights (resumable) into the Pretrained folder.
@MainActor @Observable
final class DownloadCenter: NSObject {
    struct Item: Identifiable {
        let id: String
        let title: String
        let url: URL
        let destination: URL
        let expectedBytes: Int64
        var received: Int64 = 0
        var state = State.notStarted
        enum State: Equatable { case notStarted, downloading, paused, done, failed(String) }
        var fraction: Double { expectedBytes > 0 ? Double(received) / Double(expectedBytes) : 0 }
    }

    private(set) var items: [String: Item] = [:]
    private var tasks: [String: URLSessionDownloadTask] = [:]
    private var resumeData: [String: Data] = [:]
    private var session: URLSession!
    let paths: AppPaths

    init(paths: AppPaths) {
        self.paths = paths
        super.init()
        session = URLSession(configuration: .default, delegate: self, delegateQueue: .main)
        for i in [ModelInitialization.p2p150M, .p2p300M] { register(i) }
    }

    static func id(_ i: ModelInitialization) -> String { "p2p-\(i.rawValue)" }

    private func register(_ i: ModelInitialization) {
        guard let folder = i.checkpointFolder, let dest = paths.pretrainedCheckpoint(i) else { return }
        let url = URL(string: "https://huggingface.co/elefantai/open-p2p/resolve/main/\(folder)/checkpoint-step%3D00500000.ckpt")!
        var item = Item(id: Self.id(i), title: "Pixel2Play \(folder) weights", url: url, destination: dest, expectedBytes: i.downloadBytes)
        if let size = (try? FileManager.default.attributesOfItem(atPath: dest.path))?[.size] as? Int64, size == i.downloadBytes {
            item.state = .done; item.received = size
        }
        items[item.id] = item
    }

    func item(for i: ModelInitialization) -> Item? { items[Self.id(i)] }
    func isReady(_ i: ModelInitialization) -> Bool { i == .blank || item(for: i)?.state == .done }

    func start(_ id: String) {
        guard var item = items[id], item.state != .done, item.state != .downloading else { return }
        let task = resumeData[id].map { session.downloadTask(withResumeData: $0) } ?? session.downloadTask(with: item.url)
        task.taskDescription = id
        tasks[id] = task
        item.state = .downloading
        items[id] = item
        task.resume()
    }

    func pause(_ id: String) {
        tasks[id]?.cancel { [weak self] data in
            Task { @MainActor in
                self?.resumeData[id] = data
                self?.items[id]?.state = .paused
            }
        }
    }

    func delete(_ id: String) {
        guard let item = items[id] else { return }
        tasks[id]?.cancel()
        try? FileManager.default.removeItem(at: item.destination)
        items[id]?.state = .notStarted
        items[id]?.received = 0
    }
}

extension DownloadCenter: URLSessionDownloadDelegate {
    nonisolated func urlSession(_ s: URLSession, downloadTask t: URLSessionDownloadTask, didWriteData _: Int64, totalBytesWritten w: Int64, totalBytesExpectedToWrite _: Int64) {
        let id = t.taskDescription ?? ""
        MainActor.assumeIsolated { items[id]?.received = w }
    }

    nonisolated func urlSession(_ s: URLSession, downloadTask t: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        let id = t.taskDescription ?? ""
        // The temporary file must be moved before this method returns.
        let staged = FileManager.default.temporaryDirectory.appendingPathComponent("at-\(UUID().uuidString)")
        try? FileManager.default.moveItem(at: location, to: staged)
        MainActor.assumeIsolated {
            guard let item = items[id] else { return }
            let size = ((try? FileManager.default.attributesOfItem(atPath: staged.path))?[.size] as? Int64) ?? 0
            guard size == item.expectedBytes else {
                items[id]?.state = .failed("Downloaded \(size) bytes, expected \(item.expectedBytes).")
                try? FileManager.default.removeItem(at: staged)
                return
            }
            try? FileManager.default.createDirectory(at: item.destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: item.destination)
            do {
                try FileManager.default.moveItem(at: staged, to: item.destination)
                items[id]?.state = .done
                items[id]?.received = size
            } catch {
                items[id]?.state = .failed(error.localizedDescription)
            }
        }
    }

    nonisolated func urlSession(_ s: URLSession, task t: URLSessionTask, didCompleteWithError error: Error?) {
        guard let error, (error as NSError).code != NSURLErrorCancelled else { return }
        let id = t.taskDescription ?? ""
        let data = (error as NSError).userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        MainActor.assumeIsolated {
            resumeData[id] = data
            items[id]?.state = .failed(error.localizedDescription)
        }
    }
}
