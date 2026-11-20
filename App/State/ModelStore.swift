import ATCore
import Foundation
import Observation

struct ModelEntry: Identifiable, Hashable {
    var id: UUID { definition.id }
    var url: URL
    var definition: ModelDefinition
    var checkpoints: [String]
}

/// Models on disk (`<Models>/<name>.atmodel`).
@MainActor @Observable
final class ModelStore {
    private(set) var root: URL
    private(set) var models: [ModelEntry] = []
    var lastError: String?

    init(root: URL) {
        self.root = root
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        refresh()
    }

    func setRoot(_ url: URL) { root = url; try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true); refresh() }

    func refresh() {
        let items = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        models = items.filter { $0.pathExtension == ModelPackage.fileExtension }.compactMap { u in
            (try? ModelPackage.read(u)).map { ModelEntry(url: u, definition: $0, checkpoints: ModelPackage.checkpointNames(u)) }
        }.sorted { $0.definition.createdAt > $1.definition.createdAt }
    }

    func entry(_ id: UUID?) -> ModelEntry? { models.first { $0.id == id } }

    private func uniqueURL(_ name: String) -> URL {
        let safe = name.replacingOccurrences(of: "/", with: "-")
        var u = root.appendingPathComponent(safe).appendingPathExtension(ModelPackage.fileExtension), n = 2
        while FileManager.default.fileExists(atPath: u.path) { u = root.appendingPathComponent("\(safe) \(n)").appendingPathExtension(ModelPackage.fileExtension); n += 1 }
        return u
    }

    @discardableResult
    func create(_ def: ModelDefinition) -> ModelEntry? {
        do {
            let url = uniqueURL(def.name)
            try ModelPackage.write(def, to: url)
            refresh()
            return entry(def.id)
        } catch { lastError = error.localizedDescription; return nil }
    }

    func save(_ def: ModelDefinition) {
        guard let e = entry(def.id) else { return }
        do { try ModelPackage.write(def, to: e.url); refresh() } catch { lastError = error.localizedDescription }
    }

    func duplicate(_ e: ModelEntry) -> ModelEntry? {
        var d = e.definition
        d.id = UUID(); d.name += " copy"; d.createdAt = Date(); d.architectureLocked = false; d.trainedSteps = 0; d.activeCheckpoint = nil; d.lastTrainedAt = nil
        return create(d)
    }

    /// Forgets everything the model learned: its checkpoints, training log and metrics go to the
    /// Trash, and every setting that training froze can be changed again. Returns the new definition.
    func resetBrain(_ e: ModelEntry) -> ModelDefinition? {
        let fm = FileManager.default
        do {
            for name in [ModelPackage.checkpoints, ModelPackage.log, ModelPackage.metrics] {
                let u = e.url.appendingPathComponent(name)
                if fm.fileExists(atPath: u.path) { try fm.trashItem(at: u, resultingItemURL: nil) }
            }
            var d = e.definition
            d.architectureLocked = false; d.trainedSteps = 0; d.activeCheckpoint = nil; d.lastTrainedAt = nil
            try ModelPackage.write(d, to: e.url)
            refresh()
            lastError = nil
            return d
        } catch {
            lastError = "Couldn't reset \(e.definition.name): \(error.localizedDescription)"
            refresh()
            return nil
        }
    }

    func trash(_ e: ModelEntry) {
        do { try FileManager.default.trashItem(at: e.url, resultingItemURL: nil); refresh() } catch { lastError = error.localizedDescription }
    }
}
