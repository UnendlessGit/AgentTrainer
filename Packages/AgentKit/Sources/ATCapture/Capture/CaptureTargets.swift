import AppKit
import ATCore
import Foundation
import ScreenCaptureKit

/// A display or window that can be recorded or controlled.
public struct CaptureTarget: Identifiable, Hashable, @unchecked Sendable {
    public enum Source: Hashable, @unchecked Sendable { case display(SCDisplay), window(SCWindow) }
    public let source: Source
    public let info: CaptureTargetInfo
    /// Area in global screen points (top-left origin).
    public let frame: CGRect
    public let processID: pid_t?

    public var id: String {
        switch source {
        case .display(let d): "display-\(d.displayID)"
        case .window(let w): "window-\(w.windowID)"
        }
    }
    public var name: String { info.displayName }
    public var isDisplay: Bool { if case .display = source { true } else { false } }

    public static func == (a: CaptureTarget, b: CaptureTarget) -> Bool { a.id == b.id }
    public func hash(into h: inout Hasher) { h.combine(id) }

    /// Content filter that never includes AgentTrainer's own windows (HUD, panels).
    public func filter(excludingSelf apps: [SCRunningApplication]) -> SCContentFilter {
        switch source {
        case .display(let d): SCContentFilter(display: d, excludingApplications: apps, exceptingWindows: [])
        case .window(let w): SCContentFilter(desktopIndependentWindow: w)
        }
    }

    public var pointPixelScale: CGFloat {
        let screen = NSScreen.screens.first { $0.frame.intersects(flippedFrame) } ?? NSScreen.main
        return screen?.backingScaleFactor ?? 2
    }

    private var flippedFrame: CGRect {
        let h = NSScreen.screens.first?.frame.height ?? 0
        return CGRect(x: frame.minX, y: h - frame.maxY, width: frame.width, height: frame.height)
    }
}

public enum CaptureTargets {
    /// Lists displays and normal on-screen app windows (excluding AgentTrainer itself).
    public static func load() async throws -> (targets: [CaptureTarget], ownApps: [SCRunningApplication]) {
        let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true)
        let me = Bundle.main.bundleIdentifier
        let own = content.applications.filter { $0.bundleIdentifier == me }
        var out: [CaptureTarget] = []
        for (i, d) in content.displays.enumerated() {
            out.append(CaptureTarget(source: .display(d), info: CaptureTargetInfo(kind: .display, displayID: d.displayID,
                                     title: content.displays.count > 1 ? "Display \(i + 1) (\(d.width)×\(d.height))" : "Entire display (\(d.width)×\(d.height))"),
                                     frame: d.frame, processID: nil))
        }
        // AgentTrainer's own windows are never offered, except its Test Arena / Sync Check windows.
        let windows = content.windows.filter {
            $0.windowLayer == 0 && $0.isOnScreen && $0.frame.width >= 120 && $0.frame.height >= 90
                && ($0.owningApplication?.bundleIdentifier != me || ($0.title ?? "").hasPrefix("AgentTrainer Test Arena") || ($0.title ?? "").hasPrefix("AgentTrainer Sync Check"))
                && !($0.owningApplication?.applicationName.isEmpty ?? true)
        }
        for w in windows.sorted(by: { ($0.owningApplication?.applicationName ?? "") < ($1.owningApplication?.applicationName ?? "") }) {
            let app = w.owningApplication
            out.append(CaptureTarget(source: .window(w), info: CaptureTargetInfo(kind: .window, windowID: w.windowID,
                                     appBundleID: app?.bundleIdentifier, appName: app?.applicationName, title: w.title),
                                     frame: w.frame, processID: app?.processID))
        }
        return (out, own)
    }

    /// A still image of a target for pickers.
    public static func thumbnail(_ t: CaptureTarget, own: [SCRunningApplication], maxWidth: Int = 480) async -> CGImage? {
        let cfg = SCStreamConfiguration()
        let aspect = t.frame.height / max(t.frame.width, 1)
        cfg.width = maxWidth
        cfg.height = max(1, Int(Double(maxWidth) * aspect))
        cfg.showsCursor = false
        return try? await SCScreenshotManager.captureImage(contentFilter: t.filter(excludingSelf: own), configuration: cfg)
    }
}
