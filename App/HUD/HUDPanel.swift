import AppKit
import ATCapture
import SwiftUI

/// The floating HUD window: click-through, above full-screen apps, invisible to screen capture.
@MainActor
final class HUDPanelController {
    let model = HUDModel()
    private var panel: NSPanel?
    private var timer: Timer?
    private var screenFrame: CGRect?

    func show(mode: HUDModel.Mode, options: HUDOptions, on targetFrame: CGRect?, feed: @escaping @MainActor () -> Void) {
        model.mode = mode
        model.options = options
        guard options.enabled else { hide(); return }
        if panel == nil {
            let p = NSPanel(contentRect: .zero, styleMask: [.nonactivatingPanel, .borderless], backing: .buffered, defer: false)
            p.isOpaque = false
            p.backgroundColor = .clear
            p.hasShadow = false
            p.ignoresMouseEvents = true
            p.level = .screenSaver
            p.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]
            p.sharingType = .none // never appears in recordings or in the model's view
            p.contentView = NSHostingView(rootView: HUDView(model: model))
            panel = p
        }
        screenFrame = Self.screen(for: targetFrame)?.visibleFrame ?? NSScreen.main?.visibleFrame
        panel?.orderFrontRegardless()
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                feed()
                self?.layout()
            }
        }
        layout()
    }

    func hide() {
        timer?.invalidate(); timer = nil
        panel?.orderOut(nil)
    }

    private func layout() {
        guard let p = panel, let v = p.contentView, let s = screenFrame else { return }
        let size = v.fittingSize
        let m: CGFloat = 16
        let o: CGPoint
        switch model.options.corner {
        case .bottomRight: o = CGPoint(x: s.maxX - size.width - m, y: s.minY + m)
        case .bottomLeft: o = CGPoint(x: s.minX + m, y: s.minY + m)
        case .topRight: o = CGPoint(x: s.maxX - size.width - m, y: s.maxY - size.height - m)
        case .topLeft: o = CGPoint(x: s.minX + m, y: s.maxY - size.height - m)
        }
        let f = CGRect(origin: o, size: size)
        if p.frame != f { p.setFrame(f, display: true) }
    }

    /// The NSScreen containing a rect given in global top-left-origin points.
    static func screen(for rect: CGRect?) -> NSScreen? {
        guard let rect, let primary = NSScreen.screens.first else { return nil }
        let flipped = CGRect(x: rect.minX, y: primary.frame.height - rect.maxY, width: rect.width, height: rect.height)
        return NSScreen.screens.max { $0.frame.intersection(flipped).area < $1.frame.intersection(flipped).area }
    }
}

extension CGRect { var area: CGFloat { isNull ? 0 : width * height } }
