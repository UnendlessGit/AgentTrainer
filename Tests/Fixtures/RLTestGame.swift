import AppKit
import Foundation

final class GameView: NSView {
    var reward = 0.0
    var length = 0.0
    var keys = Set<UInt16>()
    var started: Date?
    var episode = 0
    var right = false
    var ticks = 0
    let scenario = ProcessInfo.processInfo.environment["AGENTTRAINER_FIXTURE_SCENARIO"] ?? "death"
    var timer: Timer?
    let logURL = ProcessInfo.processInfo.environment["AGENTTRAINER_FIXTURE_LOG"].map { URL(fileURLWithPath: $0) }
    override var acceptsFirstResponder: Bool { true }
    override init(frame: NSRect) {
        super.init(frame: frame)
        timer = Timer.scheduledTimer(withTimeInterval: 1 / 30, repeats: true) { [weak self] _ in self?.tick() }
    }
    required init?(coder: NSCoder) { fatalError() }
    override func keyDown(with event: NSEvent) {
        guard !event.isARepeat else { return }
        if event.keyCode == 15 {
            episode += 1; reward = 0; length = scenario == "timeout" ? 4 : 10; started = Date(); ticks = 0; keys = []
        } else if event.keyCode == 14 {
            length = 0; started = nil; keys = []
            if let logURL, let data = try? JSONSerialization.data(withJSONObject: ["event": "controllerEnd", "episode": episode, "reward": reward]),
               let handle = try? FileHandle(forWritingTo: logURL) {
                _ = try? handle.seekToEnd(); try? handle.write(contentsOf: data + Data([10])); try? handle.close()
            }
        }
        else { keys.insert(event.keyCode) }
        needsDisplay = true
    }
    override func keyUp(with event: NSEvent) { keys.remove(event.keyCode) }
    func tick() {
        guard let started, length > 0 else { return }
        let elapsed = Date().timeIntervalSince(started) - 0.5
        if scenario == "dynamic" && elapsed >= 2 { length = 3 }
        if elapsed >= (scenario == "death" ? 6 : 12) {
            reward -= 3; length = 0; self.started = nil; keys = []
        } else if elapsed >= 0 {
            ticks += 1; right = ((ticks / 15) + episode) % 2 == 0
            let correct = right ? keys == [2] : keys == [0]
            reward += correct ? 1 : -0.25
        }
        if let logURL {
            let object: [String: Any] = ["episode": episode, "reward": reward, "length": length,
                "elapsed": elapsed, "right": right, "keys": keys.sorted(), "timestamp": DispatchTime.now().uptimeNanoseconds]
            if let data = try? JSONSerialization.data(withJSONObject: object), let handle = try? FileHandle(forWritingTo: logURL) {
                _ = try? handle.seekToEnd(); try? handle.write(contentsOf: data + Data([10])); try? handle.close()
            }
        }
        needsDisplay = true
    }
    override func draw(_ rect: NSRect) {
        NSColor(white: 0.12, alpha: 1).setFill(); bounds.fill()
        let square = CGRect(x: right ? 430 : 80, y: 180, width: 120, height: 120)
        NSColor.white.setFill(); square.fill()
        NSColor.black.setFill(); CGRect(x: 0, y: 0, width: bounds.width, height: 130).fill()
        let attributes: [NSAttributedString.Key: Any] = [.font: NSFont.monospacedSystemFont(ofSize: 30, weight: .regular), .foregroundColor: NSColor.white]
        if scenario != "missing" || started.map({ Date().timeIntervalSince($0) < 3 }) != false {
            String(format: "Reward: %.2f", reward).draw(at: CGPoint(x: 20, y: 74), withAttributes: attributes)
            String(format: "Episode Length: %.0f", length).draw(at: CGPoint(x: 20, y: 28), withAttributes: attributes)
        }
    }
}

final class Delegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    func applicationDidFinishLaunching(_ notification: Notification) {
        window = NSWindow(contentRect: NSRect(x: 80, y: 180, width: 640, height: 400),
            styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.title = "AgentTrainer RL Validation Game"
        let view = GameView(frame: CGRect(x: 0, y: 0, width: 640, height: 400))
        window.contentView = view; window.makeKeyAndOrderFront(nil); window.makeFirstResponder(view)
        NSApp.activate(ignoringOtherApps: true)
        if let log = view.logURL { FileManager.default.createFile(atPath: log.path, contents: nil) }
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
let app = NSApplication.shared
let delegate = Delegate(); app.delegate = delegate; app.setActivationPolicy(.regular); app.run()
