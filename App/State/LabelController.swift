import ATCore
import ATLanguage
import Foundation
import Observation

/// Runs local auto-labelling of recordings in the engine.
@MainActor @Observable
final class LabelController {
    private let settings: AppSettings
    private var connection: NSXPCConnection?
    var running = false
    var message = ""
    var lastError: String?
    var onFinished: (() -> Void)?

    init(settings: AppSettings) { self.settings = settings }

    var modelReady: Bool { LanguageModels.isDownloaded(LanguageModels.labeler, pretrainedRoot: settings.paths.pretrainedRoot) }

    func label(_ recordings: [URL]) {
        guard !running, !recordings.isEmpty else { return }
        let req = AutoLabelRequest(recordings: recordings, modelDirectory: LanguageModels.directory(LanguageModels.labeler, pretrainedRoot: settings.paths.pretrainedRoot))
        guard let data = try? JSONEncoder().encode(EngineJob.autoLabel(req)) else { return }
        running = true; message = "Starting"; lastError = nil
        engine()?.submit(data) { [weak self] _, err in
            DispatchQueue.main.async { MainActor.assumeIsolated { if let err { self?.running = false; self?.lastError = err } } }
        }
    }

    func stop() { engine()?.control("", command: "stop-labelling") { _ in } }

    private func engine() -> EngineServiceProtocol? {
        if connection == nil {
            let c = NSXPCConnection(serviceName: EngineXPC.serviceName)
            c.remoteObjectInterface = NSXPCInterface(with: EngineServiceProtocol.self)
            c.exportedInterface = NSXPCInterface(with: EngineClientProtocol.self)
            c.exportedObject = EngineEventReceiver { [weak self] e in DispatchQueue.main.async { MainActor.assumeIsolated { self?.handle(e) } } }
            c.setCodeSigningRequirement(EngineXPC.serviceRequirement)
            c.invalidationHandler = { [weak self] in DispatchQueue.main.async { MainActor.assumeIsolated { self?.connection = nil; if self?.running == true { self?.running = false; self?.lastError = "The engine stopped unexpectedly." } } } }
            c.resume()
            connection = c
        }
        return connection?.remoteObjectProxy as? EngineServiceProtocol
    }

    private func handle(_ e: TrainingEvent) {
        switch e {
        case .phase(let p, let m):
            message = m
            if [.finished, .stopped].contains(p) { running = false; onFinished?() }
        case .failed(let m): running = false; lastError = m; onFinished?()
        case .log(let m): message = m
        default: break
        }
    }
}
