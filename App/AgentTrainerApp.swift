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

/// Runs headless self-tests and debug modes as soon as the app has launched (independent of windows).
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        guard CommandLine.arguments.contains(where: { $0.hasPrefix("--selftest") || $0 == "--ui-snapshot" }) else { return }
        Task { @MainActor in
            let app = AppModel.shared
            await UISnapshotter.runIfRequested(app: app)
            await CaptureSelfTest.runIfRequested()
            await RunSelfTest.runIfRequested()
            await EngineTrainSelfTest.runIfRequested(app: app)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    /// Stop a running model (releasing any held keys) before the process exits.
    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated { AppModel.shared.runner.stop() }
    }
}

struct AgentTrainerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var app = AppModel.shared
    @State private var showPermissions = !PrivacyPermission.allCases.allSatisfy(\.isGranted) && !CommandLine.arguments.contains("--ui-snapshot") && !CommandLine.arguments.contains(where: { $0.hasPrefix("--selftest") })

    var body: some Scene {
        Window("AgentTrainer", id: "main") {
            RootView(app: app)
                .sheet(isPresented: $showPermissions) {
                    PermissionsView()
                        .frame(width: 560)
                        .overlay(alignment: .topTrailing) { Button("Done") { showPermissions = false }.padding() }
                }
        }
        .defaultSize(width: 1280, height: 860)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("Session") {
                Button("Start / Stop Recording") { Task { await app.recorder.toggle() } }
                Button("Start / Stop AI") { Task { await app.runner.toggle() } }
                Button("Emergency Stop AI") { app.runner.emergencyStop() }.keyboardShortcut(".", modifiers: .command)
            }
        }
    }
}

/// First-run permission setup; also embedded in Settings.
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
                    Image(systemName: p.symbol).font(.title2).frame(width: 32).foregroundStyle(.tint)
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
        }
        .padding(28)
        .onAppear { refresh() }
        .onReceive(timer) { _ in refresh() }
    }

    private func refresh() { for p in PrivacyPermission.allCases { granted[p] = p.isGranted } }
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

/// Debug aid: `--ui-snapshot <dir>` renders the main window for every tab to PNGs, then quits.
/// Renders the app's own views only (no screen capture involved).
enum UISnapshotter {
    @MainActor static func runIfRequested(app: AppModel) async {
        guard let i = CommandLine.arguments.firstIndex(of: "--ui-snapshot"), i + 1 < CommandLine.arguments.count else { return }
        let dir = URL(fileURLWithPath: CommandLine.arguments[i + 1])
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try? await Task.sleep(for: .seconds(2))
        if !NSApp.windows.contains(where: { $0.isVisible && $0.title == "AgentTrainer" }) {
            NSApp.sendAction(Selector(("newWindowForTab:")), to: nil, from: nil)
        }
        let only = CommandLine.arguments.firstIndex(of: "--tab").map { CommandLine.arguments[$0 + 1] }
        if CommandLine.arguments.contains("--fullscreen"), let window = NSApp.windows.first(where: { $0.isVisible && $0.contentView != nil }) {
            // Full-screen check: the title bar lives in a separate window there, so capture the display.
            if let t = only.flatMap(AppTab.init(rawValue:)) { app.tab = t }
            window.toggleFullScreen(nil)
            try? await Task.sleep(for: .seconds(3))
            if let targets = try? await CaptureTargets.load().targets, let display = targets.first(where: \.isDisplay),
               let img = await CaptureTargets.thumbnail(display, own: [], maxWidth: 1728) {
                let rep = NSBitmapImageRep(cgImage: img)
                try? rep.representation(using: .png, properties: [:])?.write(to: dir.appendingPathComponent("fullscreen.png"))
            }
            window.toggleFullScreen(nil)
            try? await Task.sleep(for: .seconds(1.5))
            NSApp.terminate(nil)
            return
        }
        for tab in AppTab.allCases where only == nil || only == tab.rawValue {
            app.tab = tab
            try? await Task.sleep(for: .seconds(tab == .recordings ? 4 : 1.5))
            guard let window = NSApp.windows.first(where: { $0.isVisible && $0.contentView != nil }), let view = window.contentView?.superview ?? window.contentView else { continue }
            guard let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { continue }
            view.cacheDisplay(in: view.bounds, to: rep)
            try? rep.representation(using: .png, properties: [:])?.write(to: dir.appendingPathComponent("\(tab.rawValue).png"))
        }
        NSApp.terminate(nil)
    }
}
