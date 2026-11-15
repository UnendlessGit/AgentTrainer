import AppKit
import ATCapture
import ATCore
import SwiftUI

/// A playable Test Arena window. You can record it like any game, and run models against it.
final class ArenaGameView: NSView {
    let game: ArenaGame
    private var keys = Set<String>()
    private var delta = CGVector.zero
    private var pointer = CGPoint(x: 0.5, y: 0.5)
    private var button = false
    private var timer: Timer?

    init(mode: ArenaGame.Mode) {
        game = ArenaGame(mode: mode, seed: UInt64(Date().timeIntervalSince1970))
        super.init(frame: NSRect(x: 0, y: 0, width: 480, height: 480))
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30, repeats: true) { [weak self] _ in self?.tick() }
        addTrackingArea(NSTrackingArea(rect: .zero, options: [.mouseMoved, .activeInKeyWindow, .inVisibleRect], owner: self))
    }

    required init?(coder: NSCoder) { fatalError() }
    deinit { timer?.invalidate() }

    override var acceptsFirstResponder: Bool { true }
    override var isFlipped: Bool { true }

    private func tick() {
        var a = TimestepAction()
        a.keys = keys
        a.dx = delta.dx; a.dy = delta.dy
        a.absX = pointer.x; a.absY = pointer.y
        if button { a.pointer = [.left] }
        delta = .zero
        game.step(a, dt: 1.0 / 30)
        needsDisplay = true
        window?.title = "AgentTrainer Test Arena — \(game.mode.title) — score \(game.score)"
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        ctx.saveGState()
        // The game draws with a top-left origin into a bottom-left context; undo the view flip.
        ctx.translateBy(x: 0, y: bounds.height)
        ctx.scaleBy(x: 1, y: -1)
        game.render(ctx, width: bounds.width, height: bounds.height)
        ctx.restoreGState()
        if let ins = game.instruction {
            let s = NSAttributedString(string: ins, attributes: [.font: NSFont.systemFont(ofSize: 13, weight: .semibold), .foregroundColor: NSColor.white.withAlphaComponent(0.85)])
            s.draw(at: CGPoint(x: 10, y: 8))
        }
    }

    private func id(_ e: NSEvent) -> String? { KeyCatalog.byKeyCode[e.keyCode]?.id }
    override func keyDown(with e: NSEvent) { if !e.isARepeat, let k = id(e) { keys.insert(k) } }
    override func keyUp(with e: NSEvent) { if let k = id(e) { keys.remove(k) } }
    override func mouseMoved(with e: NSEvent) { track(e) }
    override func mouseDragged(with e: NSEvent) { track(e) }
    override func mouseDown(with e: NSEvent) { button = true; track(e) }
    override func mouseUp(with e: NSEvent) { button = false }
    private func track(_ e: NSEvent) {
        delta.dx += e.deltaX; delta.dy += e.deltaY
        let p = convert(e.locationInWindow, from: nil)
        pointer = CGPoint(x: min(1, max(0, p.x / bounds.width)), y: min(1, max(0, p.y / bounds.height)))
    }
}

/// Black window that turns white for 120 ms on every key press (for measuring sync).
final class SyncCheckView: NSView {
    private var flashUntil = Date.distantPast
    override var acceptsFirstResponder: Bool { true }
    override func keyDown(with e: NSEvent) { flashUntil = Date().addingTimeInterval(0.12); needsDisplay = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.13) { [weak self] in self?.needsDisplay = true }
    }
    override func draw(_ r: NSRect) {
        (Date() < flashUntil ? NSColor.white : NSColor.black).setFill()
        bounds.fill()
        if Date() >= flashUntil {
            let s = NSAttributedString(string: "Record this window, then press keys a few times (about a second apart).",
                                       attributes: [.font: NSFont.systemFont(ofSize: 13), .foregroundColor: NSColor.gray])
            s.draw(at: CGPoint(x: 16, y: 16))
        }
    }
}

@MainActor
enum ArenaWindows {
    static var open: [NSWindow] = []

    static func show(_ view: NSView, title: String) {
        let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 480, height: 480), styleMask: [.titled, .closable, .miniaturizable], backing: .buffered, defer: false)
        w.title = title
        w.contentView = view
        w.isReleasedWhenClosed = false
        w.center()
        w.makeKeyAndOrderFront(nil)
        w.makeFirstResponder(view)
        open.append(w)
    }
}
