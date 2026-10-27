import AppKit
import Foundation

// Standalone screen/HID environment. Its JSON log is an observer only; the
// trainer reads rewards exclusively through the two rendered OCR lines.
struct AuditSetup: Decodable {
    var task: String
    var seed: UInt64
    var label: String
    var episodeSeconds: Double = 10
}

final class AuditGame: NSView {
    let root = URL(fileURLWithPath: ProcessInfo.processInfo.environment["AGENTTRAINER_AUDIT_ROOT"]!)
    var setup = AuditSetup(task: "keyboard", seed: 7, label: "idle")
    var rng: UInt64 = 7
    var reward = 0.0, length = 0.0, started = 0.0, previous = 0.0
    var episode = 0, trial = -1, kind = 0
    var keys = Set<UInt16>(), buttons = Set<Int>()
    var pointer = CGPoint(x: 0.5, y: 0.5), target = CGPoint(x: 0.5, y: 0.5)
    var pawn = CGPoint(x: 0.5, y: 0.5)
    var best = 1.0, success = false, picked = false
    var counts: [String: Double] = [:]
    var timer: Timer?
    var tracking: NSTrackingArea?
    override var acceptsFirstResponder: Bool { true }
    override var isFlipped: Bool { true }
    override init(frame: NSRect) {
        super.init(frame: frame)
        timer = Timer.scheduledTimer(withTimeInterval: 1 / 30, repeats: true) { [weak self] _ in self?.tick() }
    }
    required init?(coder: NSCoder) { fatalError() }
    override func updateTrackingAreas() {
        if let tracking { removeTrackingArea(tracking) }
        tracking = NSTrackingArea(rect: bounds, options: [.activeAlways, .mouseMoved, .inVisibleRect], owner: self)
        addTrackingArea(tracking!)
        super.updateTrackingAreas()
    }
    func random() -> Double {
        rng &+= 0x9E3779B97F4A7C15
        var z = rng; z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return Double((z ^ (z >> 31)) >> 11) / 9007199254740992
    }
    func add(_ name: String, _ value: Double = 1) { counts[name, default: 0] += value }
    func log(_ event: String) {
        let row: [String: Any] = ["event": event, "label": setup.label, "task": setup.task,
            "seed": setup.seed, "episode": episode, "trial": trial, "kind": kind,
            "reward": reward, "length": length, "counts": counts,
            "timestamp": DispatchTime.now().uptimeNanoseconds]
        let url = root.appendingPathComponent("fixture.jsonl")
        if let data = try? JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]),
           let handle = try? FileHandle(forWritingTo: url) {
            _ = try? handle.seekToEnd(); try? handle.write(contentsOf: data + Data([10])); try? handle.close()
        }
    }
    func reset() {
        if length > 0 { log("interrupted") }
        if let data = try? Data(contentsOf: root.appendingPathComponent("fixture-control.json")),
           let next = try? JSONDecoder().decode(AuditSetup.self, from: data) {
            if next.label != setup.label { episode = 0 }
            setup = next
        }
        episode += 1; rng = setup.seed &+ UInt64(episode) &* 7919
        reward = 0; length = setup.episodeSeconds; started = CACurrentMediaTime(); previous = started
        keys = []; buttons = []; counts = [:]; trial = -1; pointer = CGPoint(x: 0.5, y: 0.4)
        let origin = window!.convertPoint(toScreen: convert(CGPoint(x: 320, y: 160), to: nil))
        let screenY = (NSScreen.screens.first?.frame.height ?? 0) - origin.y
        CGWarpMouseCursorPosition(CGPoint(x: origin.x, y: screenY))
        log("reset"); needsDisplay = true
    }
    func end() {
        guard length > 0 else { return }
        closeTrial(); length = 0; keys = []; buttons = []; log("terminal"); needsDisplay = true
    }
    func closeTrial() {
        guard trial >= 0 else { return }
        add("trials\(kind)"); if success { add("successes\(kind)") }
        log("trial")
    }
    func nextTrial(_ index: Int) {
        closeTrial(); trial = index
        kind = Int(random() * (setup.task == "keyboard" ? 4 : 3))
        target = CGPoint(x: random() < 0.5 ? 0.25 : 0.75, y: random() < 0.5 ? 0.22 : 0.54)
        pawn = CGPoint(x: 0.5, y: 0.38); success = false; picked = false
        best = distance(setup.task == "pointer" ? pointer : pawn)
    }
    func distance(_ point: CGPoint) -> Double { hypot(point.x - target.x, point.y - target.y) }
    func approach(_ point: CGPoint) {
        let d = distance(point)
        add("errorSum\(kind)", d); add("errorSamples\(kind)")
        if d < best { reward += (best - d) * 3; best = d }
    }
    func tick() {
        let now = CACurrentMediaTime()
        guard length > 0 else { return }
        let elapsed = now - started
        guard elapsed >= 0.6 else { needsDisplay = true; return }
        let dt = min(0.1, now - previous); previous = now
        if elapsed >= setup.episodeSeconds - 0.2 { end(); return }
        let index = Int((elapsed - 0.6) / 1.5)
        if index != trial { nextTrial(index) }
        if setup.task == "keyboard" {
            let required: UInt16 = [0, 2, 49, 1][kind]
            let correct = keys == [required]
            add("seconds\(kind)", dt)
            if correct { add("correctSeconds\(kind)", dt); reward += dt * 2; if kind != 2 { success = true } }
            else { reward -= dt * 0.5 }
        } else if setup.task == "pointer" { approach(pointer) }
        else { approach(pawn) }
        needsDisplay = true
    }
    override func keyDown(with event: NSEvent) {
        guard !event.isARepeat else { return }
        if event.keyCode == 15 { reset(); return }
        if event.keyCode == 14 { end(); return }
        guard length > 0 else { return }
        keys.insert(event.keyCode)
        guard trial >= 0 else { return }
        add("keyDown\(event.keyCode)")
        if setup.task == "keyboard", kind == 2, event.keyCode == 49, !success {
            reward += 1; add("spaceEdges"); success = true
        }
        log("input")
    }
    override func keyUp(with event: NSEvent) { keys.remove(event.keyCode) }
    func move(_ event: NSEvent) {
        let p = convert(event.locationInWindow, from: nil)
        pointer = CGPoint(x: p.x / 640, y: p.y / 400)
        guard length > 0, trial >= 0 else { return }
        add("moves")
        if setup.task == "relative", kind == 0 {
            pawn.x = min(0.95, max(0.05, pawn.x + event.deltaX / 160))
            pawn.y = min(0.62, max(0.08, pawn.y + event.deltaY / 160))
            if distance(pawn) < 0.09, !success { success = true; reward += 2 }
        }
        needsDisplay = true
    }
    override func mouseMoved(with event: NSEvent) { move(event) }
    override func mouseDragged(with event: NSEvent) { move(event) }
    override func rightMouseDragged(with event: NSEvent) { move(event) }
    override func otherMouseDragged(with event: NSEvent) { move(event) }
    func button(_ event: NSEvent, down: Bool) {
        move(event)
        guard length > 0, trial >= 0 else { return }
        let b = event.buttonNumber
        add("button\(b)\(down ? "Down" : "Up")")
        if down { buttons.insert(b) } else { buttons.remove(b) }
        if setup.task == "pointer" {
            if kind < 2, down, b == kind, distance(pointer) < 0.15, !success {
                success = true; reward += 3
            } else if kind == 2, b == 0 {
                if down, hypot(pointer.x - 0.5, pointer.y - 0.38) < 0.15 { picked = true; reward += 0.2 }
                if !down, picked, distance(pointer) < 0.15, !success { success = true; reward += 3; add("dragCompletions") }
            } else if down { reward -= 0.05 }
        }
        log("input")
    }
    override func mouseDown(with event: NSEvent) { button(event, down: true) }
    override func mouseUp(with event: NSEvent) { button(event, down: false) }
    override func rightMouseDown(with event: NSEvent) { button(event, down: true) }
    override func rightMouseUp(with event: NSEvent) { button(event, down: false) }
    override func otherMouseDown(with event: NSEvent) { button(event, down: true) }
    override func otherMouseUp(with event: NSEvent) { button(event, down: false) }
    override func scrollWheel(with event: NSEvent) {
        guard length > 0, trial >= 0 else { return }
        add("scrollX", abs(event.scrollingDeltaX)); add("scrollY", abs(event.scrollingDeltaY))
        if setup.task == "relative", kind > 0 {
            if kind == 1 { pawn.x = min(0.95, max(0.05, pawn.x + event.scrollingDeltaX / 80)); target.y = pawn.y }
            else { pawn.y = min(0.62, max(0.08, pawn.y - event.scrollingDeltaY / 80)); target.x = pawn.x }
            if distance(pawn) < 0.09, !success { success = true; reward += 2 }
        }
        log("input"); needsDisplay = true
    }
    override func draw(_ dirtyRect: NSRect) {
        NSColor(white: 0.12, alpha: 1).setFill(); bounds.fill()
        NSColor.white.setFill()
        if setup.task == "keyboard" {
            let rects = [CGRect(x: 70, y: 130, width: 130, height: 130), CGRect(x: 440, y: 130, width: 130, height: 130),
                         CGRect(x: 255, y: 45, width: 130, height: 130), CGRect(x: 220, y: 205, width: 200, height: 55)]
            rects[kind].fill()
        } else {
            let rect = CGRect(x: target.x * 640 - 38, y: target.y * 400 - 38, width: 76, height: 76)
            if kind == 0 { NSBezierPath(ovalIn: rect).fill() }
            else if kind == 1 { rect.fill() }
            else { NSBezierPath(rect: rect).stroke() }
            NSColor.gray.setFill()
            NSBezierPath(ovalIn: CGRect(x: 300, y: 132, width: 40, height: 40)).fill()
            let p = setup.task == "pointer" ? pointer : pawn
            NSColor.systemOrange.setFill()
            NSBezierPath(ovalIn: CGRect(x: p.x * 640 - 6, y: p.y * 400 - 6, width: 12, height: 12)).fill()
        }
        NSColor.black.setFill(); CGRect(x: 0, y: 280, width: 640, height: 120).fill()
        let a: [NSAttributedString.Key: Any] = [.font: NSFont.monospacedSystemFont(ofSize: 28, weight: .regular), .foregroundColor: NSColor.white]
        String(format: "Reward: %.2f", reward).draw(at: CGPoint(x: 16, y: 292), withAttributes: a)
        String(format: "Episode Length: %.1f", length).draw(at: CGPoint(x: 16, y: 344), withAttributes: a)
    }
}

final class AuditDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    func applicationDidFinishLaunching(_ notification: Notification) {
        window = NSWindow(contentRect: NSRect(x: 80, y: 180, width: 640, height: 400), styleMask: [.titled], backing: .buffered, defer: false)
        window.title = "AgentTrainer RL Validation Game"
        let view = AuditGame(frame: CGRect(x: 0, y: 0, width: 640, height: 400))
        window.contentView = view; window.acceptsMouseMovedEvents = true
        window.makeKeyAndOrderFront(nil); window.makeFirstResponder(view)
        NSApp.activate(ignoringOtherApps: true)
        let log = view.root.appendingPathComponent("fixture.jsonl")
        if !FileManager.default.fileExists(atPath: log.path) { FileManager.default.createFile(atPath: log.path, contents: nil) }
    }
}
let app = NSApplication.shared
let delegate = AuditDelegate(); app.delegate = delegate; app.setActivationPolicy(.regular); app.run()
