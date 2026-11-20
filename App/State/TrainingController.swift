import ATCore
import Foundation
import IOKit
import Observation

/// Receives engine events on the XPC connection.
final class EngineEventReceiver: NSObject, EngineClientProtocol {
    let handler: (TrainingEvent) -> Void
    init(_ handler: @escaping (TrainingEvent) -> Void) { self.handler = handler }
    func receive(_ event: Data) { if let e = TrainingEvent.decode(event) { handler(e) } }
}

@MainActor @Observable
final class TrainingController {
    private let settings: AppSettings
    private let models: ModelStore
    private let library: LibraryStore
    private var connection: NSXPCConnection?
    private(set) var jobID: String?
    private(set) var jobModelID: UUID?
    var phase: TrainingPhase?
    var message = ""
    var dataset: DatasetSummary?
    var progress: [TrainingProgress] = []
    var validation: [ValidationMetrics] = []
    var logs: [String] = []
    var errorMessage: String?
    var gpuUtilization: Double?
    var thermal = ProcessInfo.processInfo.thermalState
    private var monitor: Timer?
    private var startedAt: Date?

    init(settings: AppSettings, models: ModelStore, library: LibraryStore) {
        self.settings = settings; self.models = models; self.library = library
        migrateRecordingSelection()
        monitor = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.gpuUtilization = GPUMonitor.utilization()
                self?.thermal = ProcessInfo.processInfo.thermalState
            }
        }
    }

    var isActive: Bool { [.preparing, .training, .validating, .paused, .saving].contains(phase) }
    var footerLine: String? {
        guard isActive, let p = progress.last else { return isActive ? message : nil }
        return phase == .paused ? "Training paused" : "Training \(Int(Double(p.step) / Double(max(p.totalSteps, 1)) * 100))%"
    }

    var eta: Double? {
        guard let p = progress.last, phase == .training || phase == .validating else { return nil }
        let recent = progress.suffix(20).map(\.secondsPerStep)
        return Double(p.totalSteps - p.step) * recent.reduce(0, +) / Double(recent.count)
    }

    /// The recordings a model trains on: every recording directly inside its chosen library folders.
    func recordings(for def: ModelDefinition) -> [LibraryRecording] {
        let fm = FileManager.default
        return Set(def.dataFolders ?? []).sorted().flatMap { key -> [LibraryRecording] in
            let url = library.folderURL(key)
            return fm.fileExists(atPath: url.path) ? library.fs.recordings(in: url, recursive: false) : []
        }
    }

    /// Training data used to be picked per recording in the Training tab; models now keep whole
    /// folders. Each model that had recordings picked gets the folders those recordings are in.
    private func migrateRecordingSelection() {
        let key = "trainingSelection"
        guard let d = UserDefaults.standard.data(forKey: key), let old = try? JSONDecoder().decode([UUID: [String]].self, from: d) else { return }
        for (id, paths) in old where !paths.isEmpty {
            guard let e = models.entry(id), e.definition.dataFolders == nil else { continue }
            let folders = Set(paths.compactMap { library.folderKey(URL(fileURLWithPath: $0).deletingLastPathComponent()) })
            guard !folders.isEmpty else { continue }
            var def = e.definition
            def.dataFolders = folders.sorted()
            models.save(def)
        }
        UserDefaults.standard.removeObject(forKey: key)
    }

    /// Loads past progress for a model from its training log.
    func loadHistory(for entry: ModelEntry) {
        guard !isActive || jobModelID != entry.id else { return }
        progress = []; validation = []; dataset = nil; logs = []; phase = nil; message = ""
        guard let text = try? String(contentsOf: entry.url.appendingPathComponent(ModelPackage.log), encoding: .utf8) else { return }
        let dec = TrainingJSON.decoder()
        for line in text.split(separator: "\n") {
            guard let l = try? dec.decode(TrainingLogLine.self, from: Data(line.utf8)) else { continue }
            if let p = l.progress { progress.append(p) }
            if let v = l.validation { validation.append(v) }
        }
        if progress.count > 2000 { progress = stride(from: 0, to: progress.count, by: progress.count / 1000).map { progress[$0] } }
    }

    private func engine() -> EngineServiceProtocol? {
        if connection == nil {
            let c = NSXPCConnection(serviceName: EngineXPC.serviceName)
            c.remoteObjectInterface = NSXPCInterface(with: EngineServiceProtocol.self)
            c.exportedInterface = NSXPCInterface(with: EngineClientProtocol.self)
            c.exportedObject = EngineEventReceiver { [weak self] e in DispatchQueue.main.async { MainActor.assumeIsolated { self?.handle(e) } } }
            c.setCodeSigningRequirement(EngineXPC.serviceRequirement)
            c.invalidationHandler = { [weak self] in DispatchQueue.main.async { MainActor.assumeIsolated { self?.engineLost() } } }
            c.interruptionHandler = { [weak self] in DispatchQueue.main.async { MainActor.assumeIsolated { self?.engineLost() } } }
            c.resume()
            connection = c
        }
        return connection?.remoteObjectProxyWithErrorHandler { [weak self] e in
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.errorMessage = e.localizedDescription } }
        } as? EngineServiceProtocol
    }

    private func engineLost() {
        connection = nil
        if isActive {
            phase = .failed
            errorMessage = "The training engine stopped unexpectedly. The last checkpoint is kept; you can resume from it."
        }
    }

    /// `recordings` overrides the model's folders (self-test).
    func start(model: ModelEntry, resumeFrom: String? = nil, recordings: [URL]? = nil) {
        let recs = recordings ?? self.recordings(for: model.definition).map(\.url)
        guard !recs.isEmpty else {
            errorMessage = (model.definition.dataFolders ?? []).isEmpty
                ? "Choose the folders to train on in Models → \(model.definition.name) → Data."
                : "The folders chosen for this model have no recordings yet."
            return
        }
        let split = Self.split(recs, fraction: model.definition.training.validationFraction, seed: model.definition.training.seed)
        let p = settings.paths
        let req = TrainingRequest(modelURL: model.url, trainRecordings: split.train, validationRecordings: split.validation,
                                  cacheRoot: p.cacheRoot, pretrainedRoot: p.pretrainedRoot, resumeFrom: resumeFrom,
                                  memoryLimitGB: settings.stored.mlxMemoryLimitGB)
        guard let data = try? JSONEncoder().encode(EngineJob.train(req)) else { return }
        errorMessage = nil
        progress = []; validation = []; logs = []; dataset = nil
        phase = .preparing; message = "Starting"
        jobModelID = model.id
        startedAt = Date()
        engine()?.submit(data) { [weak self] id, err in
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    if let err { self?.errorMessage = err; self?.phase = nil } else { self?.jobID = id.flatMap { String(data: $0, encoding: .utf8) } }
                }
            }
        }
    }

    func control(_ command: String) {
        guard let id = jobID else { return }
        engine()?.control(id, command: command) { _ in }
        if command == "pause" { message = "Pausing after this step…" }
        if command == "stop" { message = "Stopping and saving…" }
    }

    private func handle(_ e: TrainingEvent) {
        switch e {
        case .phase(let p, let m):
            phase = p; message = m
            if [.finished, .stopped, .failed].contains(p) { jobID = nil; models.refresh() }
        case .dataset(let d): dataset = d
        case .progress(let p): progress.append(p)
        case .validation(let v): validation.append(v)
        case .checkpoint: models.refresh()
        case .log(let s): logs.append(s)
        case .failed(let m): phase = .failed; errorMessage = m; jobID = nil; models.refresh()
        }
    }

    /// Same deterministic split as the engine's command-line path: by recording, never by frame.
    static func split(_ urls: [URL], fraction: Double, seed: Int) -> (train: [URL], validation: [URL]) {
        guard fraction > 0, urls.count >= 2 else { return (urls, []) }
        let sorted = urls.sorted { $0.lastPathComponent < $1.lastPathComponent }
        var h = Hasher()
        h.combine(seed)
        let n = max(1, Int((Double(urls.count) * fraction).rounded()))
        // Stable choice: rank by a hash of the file name.
        let ranked = sorted.sorted { stableHash($0.lastPathComponent, seed) < stableHash($1.lastPathComponent, seed) }
        let val = Set(ranked.prefix(n))
        return (sorted.filter { !val.contains($0) }, sorted.filter { val.contains($0) })
    }

    static func stableHash(_ s: String, _ seed: Int) -> UInt64 {
        var h: UInt64 = 1_469_598_103_934_665_603 &+ UInt64(bitPattern: Int64(seed))
        for b in s.utf8 { h = (h ^ UInt64(b)) &* 1_099_511_628_211 }
        return h
    }
}

enum GPUMonitor {
    /// "Device Utilization %" from the GPU's IOAccelerator performance statistics.
    static func utilization() -> Double? {
        var iterator: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &iterator) == KERN_SUCCESS else { return nil }
        defer { IOObjectRelease(iterator) }
        var service = IOIteratorNext(iterator)
        while service != 0 {
            defer { IOObjectRelease(service); service = IOIteratorNext(iterator) }
            var props: Unmanaged<CFMutableDictionary>?
            if IORegistryEntryCreateCFProperties(service, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS,
               let d = props?.takeRetainedValue() as? [String: Any], let perf = d["PerformanceStatistics"] as? [String: Any],
               let u = perf["Device Utilization %"] as? Int {
                return Double(u)
            }
        }
        return nil
    }
}
