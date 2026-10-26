import AppKit
import Foundation

@MainActor
final class RegionSelector {
    private var panel: RegionPanel?

    func select(on sourceFrame: CGRect, confirmWithEnter: Bool = false, background: CGImage? = nil, completion: @escaping (CGRect?) -> Void) {
        panel?.close()
        let appKitFrame = CGRect(x: sourceFrame.minX, y: (NSScreen.screens.first?.frame.height ?? sourceFrame.maxY) - sourceFrame.maxY, width: sourceFrame.width, height: sourceFrame.height)
        let panel = RegionPanel(contentRect: appKitFrame, styleMask: [.borderless], backing: .buffered, defer: false)
        panel.level = .screenSaver
        panel.isOpaque = false
        panel.backgroundColor = NSColor.black.withAlphaComponent(0.18)
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        let selector = RegionSelectionView(frame: CGRect(origin: .zero, size: sourceFrame.size))
        selector.confirmWithEnter = confirmWithEnter; selector.background = background
        selector.onComplete = { [weak self, weak panel] local in
            panel?.orderOut(nil); panel?.close(); self?.panel = nil
            guard let local else { completion(nil); return }
            let topLeft = CGRect(x: sourceFrame.minX + local.minX, y: sourceFrame.minY + sourceFrame.height - local.maxY, width: local.width, height: local.height)
            completion(topLeft)
        }
        panel.contentView = selector
        self.panel = panel
        panel.makeKeyAndOrderFront(nil)
        panel.makeFirstResponder(selector)
        NSApp.activate(ignoringOtherApps: true)
    }
}

private final class RegionPanel: NSPanel {
    override var canBecomeKey: Bool { true }
}

private final class RegionSelectionView: NSView {
    var confirmWithEnter = false
    var background: CGImage?
    var onComplete: ((CGRect?) -> Void)?
    private var start: CGPoint?
    private var selection = CGRect.zero

    override var acceptsFirstResponder: Bool { true }
    override func mouseDown(with event: NSEvent) { let point = convert(event.locationInWindow, from: nil); start = point; selection = CGRect(origin: point, size: .zero); needsDisplay = true }
    override func mouseDragged(with event: NSEvent) { guard let start else { return }; selection = normalized(start, convert(event.locationInWindow, from: nil)); needsDisplay = true }
    override func mouseUp(with event: NSEvent) { guard let start else { onComplete?(nil); return }; selection = normalized(start, convert(event.locationInWindow, from: nil)); if !confirmWithEnter { onComplete?(selection.width >= 8 && selection.height >= 8 ? selection : nil) }; needsDisplay = true }
    override func keyDown(with event: NSEvent) { if event.keyCode == 53 { onComplete?(nil) } else if event.keyCode == 36 || event.keyCode == 76 { if selection.width >= 8 && selection.height >= 8 { onComplete?(selection) } } else { super.keyDown(with: event) } }

    override func draw(_ dirtyRect: NSRect) {
        if let background { NSImage(cgImage: background, size: bounds.size).draw(in: bounds) }
        NSColor.black.withAlphaComponent(0.25).setFill(); bounds.fill()
        guard selection.width > 0, selection.height > 0 else {
            let text = confirmWithEnter ? "Drag to select reward text • Enter to confirm • Esc to cancel" : "Drag to select the exact screen region • Esc to cancel"
            text.draw(at: CGPoint(x: bounds.midX - 170, y: bounds.midY), withAttributes: [.foregroundColor: NSColor.white, .font: NSFont.systemFont(ofSize: 15, weight: .semibold)])
            return
        }
        NSGraphicsContext.saveGraphicsState()
        let path = NSBezierPath(rect: bounds); path.appendRect(selection); path.windingRule = .evenOdd
        NSColor.black.withAlphaComponent(0.55).setFill(); path.fill()
        NSColor.systemCyan.setStroke(); let border = NSBezierPath(roundedRect: selection, xRadius: 8, yRadius: 8); border.lineWidth = 2; border.stroke()
        let label = "\(Int(selection.width)) × \(Int(selection.height))"
        label.draw(at: CGPoint(x: selection.minX + 10, y: selection.maxY + 8), withAttributes: [.foregroundColor: NSColor.systemCyan, .font: NSFont.monospacedDigitSystemFont(ofSize: 13, weight: .semibold)])
        NSGraphicsContext.restoreGraphicsState()
    }

    private func normalized(_ a: CGPoint, _ b: CGPoint) -> CGRect { CGRect(x: min(a.x, b.x), y: min(a.y, b.y), width: abs(a.x - b.x), height: abs(a.y - b.y)) }
}
