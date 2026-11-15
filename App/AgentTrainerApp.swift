import ATCore
import ATModel
import SwiftUI

@main
struct AgentTrainerMain {
    static func main() {
        if CommandLine.arguments.contains("--smoke-test") {
            SmokeTest.runAndExit()
        }
        AgentTrainerApp.main()
    }
}

struct AgentTrainerApp: App {
    var body: some Scene {
        WindowGroup {
            Text("AgentTrainer").padding()
        }
    }
}

/// Headless check used by scripts/verify.sh: MLX in-process and inside the XPC engine.
enum SmokeTest {
    static func runAndExit() -> Never {
        print("app: " + MLXSmokeTest.run())
        let c = NSXPCConnection(serviceName: EngineXPC.serviceName)
        c.remoteObjectInterface = NSXPCInterface(with: EngineServiceProtocol.self)
        c.setCodeSigningRequirement(EngineXPC.serviceRequirement)
        c.resume()
        let done = DispatchSemaphore(value: 0)
        var ok = false
        let proxy = c.remoteObjectProxyWithErrorHandler { e in
            print("engine error: \(e)")
            done.signal()
        } as? EngineServiceProtocol
        proxy?.ping { s in
            print("engine: " + s)
            ok = s.contains("MLX ok")
            done.signal()
        }
        _ = done.wait(timeout: .now() + 60)
        exit(ok ? 0 : 1)
    }
}
