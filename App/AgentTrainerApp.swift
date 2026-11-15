import ATCapture
import ATCore
import ATModel
import Combine
import SwiftUI

@main
struct AgentTrainerMain {
    static func main() {
        if CommandLine.arguments.contains("--smoke-test") {
            SmokeTest.runAndExit()
        }
        // Permission checks must run when launched by LaunchServices (`open`), otherwise macOS
        // attributes them to the parent terminal instead of AgentTrainer.
        if let i = CommandLine.arguments.firstIndex(of: "--write-permission-status"), i + 1 < CommandLine.arguments.count {
            let lines = PrivacyPermission.allCases.map { "\($0.title): \($0.isGranted ? "granted" : "not granted")" }
            try? lines.joined(separator: "\n").write(toFile: CommandLine.arguments[i + 1], atomically: true, encoding: .utf8)
            exit(0)
        }
        AgentTrainerApp.main()
    }
}

struct AgentTrainerApp: App {
    var body: some Scene {
        WindowGroup("AgentTrainer") {
            PermissionsView()
                .frame(minWidth: 560, minHeight: 420)
        }
        .windowResizability(.contentSize)
    }
}

/// First-run permission setup; later embedded in onboarding and Settings.
struct PermissionsView: View {
    @State private var granted: [PrivacyPermission: Bool] = [:]
    private let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Set up permissions").font(.largeTitle.bold())
                Text("AgentTrainer needs these to record and to run models. They stay granted across updates because every build is signed with the same identity.")
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ForEach(PrivacyPermission.allCases) { p in
                HStack(spacing: 14) {
                    Image(systemName: p.symbol).font(.title2).frame(width: 32)
                        .foregroundStyle(.tint)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(p.title).font(.headline)
                        Text(p.purpose).font(.callout).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if granted[p] == true {
                        Label("Granted", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                    } else {
                        Button("Open Settings") { p.request(); p.openSettings() }
                    }
                }
                .padding(14)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
            }
            Text("If a permission was just turned on, macOS may ask you to quit and reopen AgentTrainer.")
                .font(.footnote).foregroundStyle(.secondary)
        }
        .padding(28)
        .onAppear {
            refresh()
            for p in PrivacyPermission.allCases where !p.isGranted { p.request() }
        }
        .onReceive(timer) { _ in refresh() }
    }

    private func refresh() {
        for p in PrivacyPermission.allCases { granted[p] = p.isGranted }
    }
}

/// Headless check used by scripts: MLX in-process and inside the XPC engine.
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
