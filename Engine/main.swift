import ATCore
import ATLanguage
import ATModel
import ATTraining
import Foundation

/// The training engine runs in its own process so heavy GPU work can't take the app down and
/// all of its memory is returned to the system when the job ends.
final class EngineService: NSObject, EngineServiceProtocol {
    weak var connection: NSXPCConnection?
    private var trainer: Trainer?
    private var jobID: String?
    private let lock = NSLock()

    init(connection: NSXPCConnection) {
        self.connection = connection
    }

    private var client: EngineClientProtocol? {
        connection?.remoteObjectProxyWithErrorHandler { _ in } as? EngineClientProtocol
    }

    func ping(reply: @escaping (String) -> Void) {
        reply("engine pid \(getpid()): " + MLXSmokeTest.run())
    }

    func submit(_ request: Data, reply: @escaping (Data?, String?) -> Void) {
        guard let job = try? JSONDecoder().decode(EngineJob.self, from: request) else { reply(nil, "Invalid request"); return }
        switch job {
        case .train(let req): startTraining(req, reply: reply)
        case .autoLabel(let req): startLabelling(req, reply: reply)
        case .labelChat(let req): startChat(req, reply: reply)
        }
    }

    private func startLabelling(_ req: AutoLabelRequest, reply: @escaping (Data?, String?) -> Void) {
        if LabellerHost.shared.busy { reply(nil, LabelError.busy.localizedDescription); return }
        if TrainingGate.isTraining { reply(nil, LabelError.training.localizedDescription); return }
        let client = self.client
        let thread = Thread { LabelJob.run(req) { client?.receive($0.encoded()) } }
        thread.stackSize = 16 << 20
        thread.qualityOfService = .userInitiated
        thread.name = "agenttrainer.labelling"
        thread.start()
        reply(req.jobID.uuidString.data(using: .utf8), nil)
    }

    private func startChat(_ req: LabelChatRequest, reply: @escaping (Data?, String?) -> Void) {
        if LabellerHost.shared.busy { reply(nil, LabelError.busy.localizedDescription); return }
        let client = self.client
        let thread = Thread { LabelChatJob.run(req) { client?.receive($0.encoded()) } }
        thread.stackSize = 16 << 20
        thread.qualityOfService = .userInitiated
        thread.name = "agenttrainer.label-chat"
        thread.start()
        reply(req.jobID.uuidString.data(using: .utf8), nil)
    }

    private func startTraining(_ req: TrainingRequest, reply: @escaping (Data?, String?) -> Void) {
        lock.lock()
        if trainer != nil { lock.unlock(); reply(nil, "A training job is already running."); return }
        if LabellerHost.shared.busy { lock.unlock(); reply(nil, "Auto Label is running. Training needs the same GPU memory, so wait for it to finish or stop it."); return }
        let client = self.client
        let t = Trainer(request: req, encoderFactory: { LanguageModels.instructionEncoder(pretrainedRoot: req.pretrainedRoot) }) { event in
            client?.receive(event.encoded())
        }
        trainer = t
        jobID = req.jobID.uuidString
        lock.unlock()
        // Training needs the GPU memory the labelling model would hold.
        LabellerHost.shared.release()
        TrainingGate.begin()
        let thread = Thread { [weak self] in
            defer { TrainingGate.end() }
            t.run()
            self?.lock.lock()
            self?.trainer = nil
            self?.jobID = nil
            self?.lock.unlock()
        }
        thread.qualityOfService = .userInitiated
        thread.stackSize = 16 << 20
        thread.name = "agenttrainer.training"
        thread.start()
        reply(req.jobID.uuidString.data(using: .utf8), nil)
    }

    func control(_ jobID: String, command: String, reply: @escaping (Bool) -> Void) {
        lock.lock()
        let t = trainer
        if command == "stop-labelling" { lock.unlock(); LabellerHost.shared.stop(); reply(true); return }
        if command == "release-labeller" { lock.unlock(); LabellerHost.shared.release(); reply(true); return }
        lock.unlock()
        guard let t else { reply(false); return }
        switch command {
        case "pause": t.pause()
        case "resume": t.resume()
        case "stop": t.stop()
        case "checkpoint": t.checkpointNow()
        default: reply(false); return
        }
        reply(true)
    }

    /// The app went away: stop cleanly so the latest progress is checkpointed.
    func connectionLost() {
        lock.lock(); let t = trainer; lock.unlock()
        t?.stop()
    }
}

final class ServiceDelegate: NSObject, NSXPCListenerDelegate {
    func listener(_ listener: NSXPCListener, shouldAcceptNewConnection connection: NSXPCConnection) -> Bool {
        connection.setCodeSigningRequirement(EngineXPC.clientRequirement)
        connection.exportedInterface = NSXPCInterface(with: EngineServiceProtocol.self)
        connection.remoteObjectInterface = NSXPCInterface(with: EngineClientProtocol.self)
        let service = EngineService(connection: connection)
        connection.exportedObject = service
        connection.invalidationHandler = { service.connectionLost() }
        connection.interruptionHandler = { service.connectionLost() }
        connection.resume()
        return true
    }
}

let delegate = ServiceDelegate()
let listener = NSXPCListener.service()
listener.delegate = delegate
listener.resume()
