import Foundation

/// XPC interface exported by `AgentTrainerEngine.xpc` (training, dataset preparation, auto-labelling).
/// Payloads are JSON-encoded `EngineRequest` / `EngineEvent` values so the protocol stays stable as jobs evolve.
@objc public protocol EngineServiceProtocol {
    func ping(reply: @escaping (String) -> Void)
    func submit(_ request: Data, reply: @escaping (Data?, String?) -> Void)
    func control(_ jobID: String, command: String, reply: @escaping (Bool) -> Void)
}

/// Interface the app exports back to the engine so jobs can stream progress events.
@objc public protocol EngineClientProtocol {
    func receive(_ event: Data)
}

public enum EngineXPC {
    public static let serviceName = AgentTrainerInfo.engineBundleIdentifier
    /// Only the AgentTrainer app, signed by our team, may talk to the engine.
    public static let clientRequirement =
        #"identifier "com.unendless.AgentTrainer" and anchor apple generic and certificate 1[field.1.2.840.113635.100.6.2.1] exists and certificate leaf[subject.OU] = "V5S6J7KC33""#
    /// The app only talks to the engine bundled with it.
    public static let serviceRequirement =
        #"identifier "com.unendless.AgentTrainer.Engine" and anchor apple generic and certificate 1[field.1.2.840.113635.100.6.2.1] exists and certificate leaf[subject.OU] = "V5S6J7KC33""#
}
