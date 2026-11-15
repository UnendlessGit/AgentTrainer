import ATCore
import ATModel
import Foundation

final class EngineService: NSObject, EngineServiceProtocol {
    weak var connection: NSXPCConnection?

    init(connection: NSXPCConnection) {
        self.connection = connection
    }

    func ping(reply: @escaping (String) -> Void) {
        reply("engine pid \(getpid()): " + MLXSmokeTest.run())
    }

    func submit(_ request: Data, reply: @escaping (Data?, String?) -> Void) {
        reply(nil, "not implemented")
    }

    func control(_ jobID: String, command: String, reply: @escaping (Bool) -> Void) {
        reply(false)
    }
}

final class ServiceDelegate: NSObject, NSXPCListenerDelegate {
    func listener(_ listener: NSXPCListener, shouldAcceptNewConnection connection: NSXPCConnection) -> Bool {
        connection.setCodeSigningRequirement(EngineXPC.clientRequirement)
        connection.exportedInterface = NSXPCInterface(with: EngineServiceProtocol.self)
        connection.remoteObjectInterface = NSXPCInterface(with: EngineClientProtocol.self)
        connection.exportedObject = EngineService(connection: connection)
        connection.resume()
        return true
    }
}

let delegate = ServiceDelegate()
let listener = NSXPCListener.service()
listener.delegate = delegate
listener.resume()
