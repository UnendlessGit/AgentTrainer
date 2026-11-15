import AppKit
import ATCapture
import ATCore
import ScreenCaptureKit

/// `--selftest-capture <dir>`: records AgentTrainer's own Sync Check window with the real
/// capture pipeline while pressing a harmless key into it, then analyzes sync. Synthetic keys are
/// only posted while that window is the key window of the active app, so nothing reaches other apps.
@MainActor
enum CaptureSelfTest {
    static func runIfRequested() async {
        guard let i = CommandLine.arguments.firstIndex(of: "--selftest-capture"), i + 1 < CommandLine.arguments.count else { return }
        let out = URL(fileURLWithPath: CommandLine.arguments[i + 1])
        try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        var report: [String] = []
        defer {
            try? report.joined(separator: "\n").write(to: out.appendingPathComponent("report.txt"), atomically: true, encoding: .utf8)
            NSApp.terminate(nil)
        }
        let view = SyncCheckView()
        ArenaWindows.show(view, title: "AgentTrainer Sync Check")
        NSApp.activate(ignoringOtherApps: true)
        try? await Task.sleep(for: .seconds(1.5))
        guard let window = ArenaWindows.open.last else { report.append("FAIL: no window"); return }
        do {
            let (targets, own) = try await CaptureTargets.load()
            guard let target = targets.first(where: { $0.info.title == "AgentTrainer Sync Check" }) else {
                report.append("FAIL: Sync Check window not offered as a capture target (\(targets.count) targets)"); return
            }
            var cfg = RecordingConfig(target: target, settings: CaptureSettings.fitting(target, maxHeight: 480, fps: 60, showsCursor: false),
                                      input: InputCaptureOptions(), name: "selftest-\(Int(Date().timeIntervalSince1970))", task: "", destination: out, appVersion: "selftest")
            cfg.input.onlyWhenTargetFocused = false
            let session = try RecordingSession(config: cfg)
            try await session.start(excluding: own)
            report.append("recording started: \(cfg.settings.width)x\(cfg.settings.height) @ \(cfg.settings.fps)")
            try? await Task.sleep(for: .seconds(1))
            var posted = 0
            for _ in 0 ..< 8 {
                guard NSApp.isActive, window.isKeyWindow else { report.append("skipped a key: test window not in front"); try? await Task.sleep(for: .milliseconds(700)); continue }
                for down in [true, false] {
                    let e = CGEvent(keyboardEventSource: nil, virtualKey: 0x28 /* k */, keyDown: down)
                    e?.setIntegerValueField(.eventSourceUserData, value: SyntheticInput.userData)
                    e?.post(tap: .cghidEventTap)
                    try? await Task.sleep(for: .milliseconds(60))
                }
                posted += 1
                try? await Task.sleep(for: .milliseconds(640))
            }
            try? await Task.sleep(for: .seconds(0.5))
            let url = try await session.stop()
            let meta = try RecordingIO.readMeta(url)
            report.append("posted \(posted) key presses")
            report.append("recording: \(meta.frameCount) frames, \(meta.eventCount) events, \(String(format: "%.1f", meta.duration)) s, clock \(meta.eventClock), encoder drops \(meta.droppedFrames)")
            report.append("sync: " + (try SyncCheck.analyze(recording: url)).summary)
            report.append("package: \(url.path)")
        } catch {
            report.append("FAIL: \(error.localizedDescription)")
        }
    }
}
