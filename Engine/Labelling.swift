import ATCore
import ATLanguage
import Foundation
import MLX

/// The labelling model, shared by every connection: loaded once, kept while you work (the chat and
/// the labelling use the same one) and released after a few idle minutes or when training starts.
final class LabellerHost: @unchecked Sendable {
    static let shared = LabellerHost()
    private let lock = NSLock()
    private var labeler: AutoLabeler?
    private var unload: DispatchWorkItem?
    private(set) var busy = false
    private var stopRequested = false
    static let idleSeconds = 300.0

    /// Runs `body` with the labeller for `directory`, loading it first if needed. Only one job at a time.
    func use<T>(_ directory: URL, loading: () -> Void = {}, _ body: (AutoLabeler) throws -> T) throws -> T {
        lock.lock()
        if busy { lock.unlock(); throw LabelError.busy }
        if TrainingGate.isTraining { lock.unlock(); throw LabelError.training }
        busy = true; stopRequested = false
        unload?.cancel(); unload = nil
        var current = labeler
        lock.unlock()
        defer {
            lock.lock(); busy = false; lock.unlock()
            scheduleUnload()
        }
        if current?.directory.standardizedFileURL != directory.standardizedFileURL {
            lock.lock(); labeler = nil; lock.unlock()
            current = nil
            MLX.GPU.clearCache()
            loading()
            let l = try AutoLabeler(directory: directory)
            lock.lock(); labeler = l; lock.unlock()
            current = l
        }
        return try body(current!)
    }

    func stop() { lock.lock(); stopRequested = true; lock.unlock() }
    var shouldStop: Bool { lock.lock(); defer { lock.unlock() }; return stopRequested }

    /// Frees the model's memory now (e.g. before training). Does nothing while it's working.
    func release() {
        lock.lock(); defer { lock.unlock() }
        guard !busy else { return }
        labeler = nil
        unload?.cancel(); unload = nil
        MLX.GPU.clearCache()
    }

    private func scheduleUnload() {
        let work = DispatchWorkItem { [weak self] in self?.release() }
        lock.lock(); unload?.cancel(); unload = work; lock.unlock()
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + Self.idleSeconds, execute: work)
    }
}

/// Whether a training job is running in this engine (labelling and training each need most of the GPU memory).
enum TrainingGate {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var count = 0
    static var isTraining: Bool { lock.lock(); defer { lock.unlock() }; return count > 0 }
    static func begin() { lock.lock(); count += 1; lock.unlock() }
    static func end() { lock.lock(); count = max(0, count - 1); lock.unlock() }
}

enum LabelError: LocalizedError {
    case busy, training
    var errorDescription: String? {
        switch self {
        case .busy: "The labelling model is busy with another job."
        case .training: "A model is training. Labelling needs the same GPU memory, so wait for training to finish or stop it."
        }
    }
}

/// Labels recordings one after another, writing each instruction into its recording as soon as it's made.
enum LabelJob {
    static func run(_ req: AutoLabelRequest, emit: @escaping (TrainingEvent) -> Void) {
        let host = LabellerHost.shared
        do {
            try host.use(req.modelDirectory, loading: { emit(.labelProgress(recording: 0, of: req.recordings.count, fraction: 0, secondsLeft: nil, status: "Loading the labelling model")) }) { labeler in
                // Footage seconds per wall-clock second so far, for the estimate of time left.
                let durations = req.recordings.map { (try? RecordingIO.readMeta($0).duration) ?? 0 }
                let footage = durations.reduce(0, +)
                var footageDone = 0.0
                let started = Date()
                for (i, url) in req.recordings.enumerated() {
                    if host.shouldStop { break }
                    let oldAuto = Set(RecordingIO.readAnnotations(url).segments.filter { $0.source == .auto }.map(\.id))
                    var processedTo = 0.0
                    func save(_ seg: TextSegment) {
                        // Read it again each time: you may be editing this recording while it's labelled.
                        var ann = RecordingIO.readAnnotations(url)
                        processedTo = max(processedTo, seg.end)
                        if req.replaceAutomatic { ann.segments.removeAll { oldAuto.contains($0.id) && $0.start < processedTo } }
                        if let k = ann.segments.firstIndex(where: { $0.id == seg.id }) { ann.segments[k] = seg } else { ann.segments.append(seg) }
                        ann.segments.sort { $0.start < $1.start }
                        try? RecordingIO.writeAnnotations(ann, to: url)
                        emit(.labelled(recording: url, segment: seg))
                    }
                    let base = footageDone
                    try labeler.label(recording: url, request: req, shouldStop: { host.shouldStop }, progress: { p in
                        let doneNow = base + p.fraction * durations[i]
                        let elapsed = Date().timeIntervalSince(started)
                        let left: Double? = doneNow > 1 && elapsed > 20 ? (footage - doneNow) * elapsed / doneNow : nil
                        emit(.labelProgress(recording: i, of: req.recordings.count, fraction: footage > 0 ? doneNow / footage : 0, secondsLeft: left, status: p.status))
                    }, onSegment: save)
                    // Finished the whole recording: no earlier automatic instruction is left over.
                    if !host.shouldStop, req.replaceAutomatic {
                        var ann = RecordingIO.readAnnotations(url)
                        ann.segments.removeAll { oldAuto.contains($0.id) }
                        try? RecordingIO.writeAnnotations(ann, to: url)
                    }
                    footageDone += durations[i]
                }
            }
            let stopped = host.shouldStop
            emit(.phase(stopped ? .stopped : .finished, stopped ? "Stopped" : "Labelled \(req.recordings.count) recording\(req.recordings.count == 1 ? "" : "s")"))
        } catch {
            emit(.failed(error.localizedDescription))
        }
    }
}

/// One reply in the chat about the footage, streamed as it's written.
enum LabelChatJob {
    static func run(_ req: LabelChatRequest, emit: @escaping (TrainingEvent) -> Void) {
        let host = LabellerHost.shared
        do {
            var lastSent = Date.distantPast
            let reply = try host.use(req.modelDirectory, loading: { emit(.chatDelta("")) }) { labeler in
                try labeler.chat(req, shouldStop: { host.shouldStop }) { text in
                    // About 20 updates a second is plenty for the screen.
                    if Date().timeIntervalSince(lastSent) > 0.05 { lastSent = Date(); emit(.chatDelta(text)) }
                }
            }
            emit(.chatDelta(reply.trimmingCharacters(in: .whitespacesAndNewlines)))
            emit(.chatDone(nil))
        } catch {
            emit(.chatDone(error.localizedDescription))
        }
    }
}
