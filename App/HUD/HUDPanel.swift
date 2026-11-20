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
    private var noticeTask: Task<Void, Never>?

    func show(mode: HUDModel.Mode, options: HUDOptions, on targetFrame: CGRect?, feed: @escaping @MainActor () -> Void) {
        model.mode = mode
        model.options = options
        model.reset()
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

    /// Shows a short message on the HUD for a few seconds.
    func flash(_ text: String, seconds: Double = 3.5) {
        model.notice = text
        noticeTask?.cancel()
        noticeTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            self?.model.notice = nil
        }
    }

    func hide() {
        timer?.invalidate(); timer = nil
        noticeTask?.cancel(); model.notice = nil; model.correcting = false
        panel?.orderOut(nil)
    }

    private func layout() {
        guard let p = panel, let s = screenFrame else { return }
        // Fixed size: the HUD grows inside the panel towards its corner, so the window itself never jumps.
        let size = HUDView.panelSize(scale: model.options.scale)
        let m: CGFloat = 2
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
