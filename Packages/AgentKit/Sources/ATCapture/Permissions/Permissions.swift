import AppKit
import ApplicationServices
import CoreGraphics
import Foundation

/// The three macOS privacy permissions AgentTrainer needs. They are granted to the app's
/// designated requirement (bundle id + team), so they persist across rebuilds and updates.
public enum PrivacyPermission: String, CaseIterable, Identifiable, Sendable {
    case screenRecording, inputMonitoring, accessibility

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .screenRecording: "Screen Recording"
        case .inputMonitoring: "Input Monitoring"
        case .accessibility: "Accessibility"
        }
    }

    public var purpose: String {
        switch self {
        case .screenRecording: "Capture the game or app you record, and let a running model see it."
        case .inputMonitoring: "Record your keyboard, mouse movement, clicks and scrolling."
        case .accessibility: "Let a running model press keys and move the mouse."
        }
    }

    public var symbol: String {
        switch self {
        case .screenRecording: "rectangle.dashed.badge.record"
        case .inputMonitoring: "keyboard"
        case .accessibility: "cursorarrow.motionlines"
        }
    }

    public var isGranted: Bool {
        switch self {
        case .screenRecording: CGPreflightScreenCaptureAccess()
        case .inputMonitoring: CGPreflightListenEventAccess()
        case .accessibility: AXIsProcessTrusted() && CGPreflightPostEventAccess()
        }
    }

    /// Shows the system prompt (first time) and registers the app in System Settings.
    @discardableResult
    public func request() -> Bool {
        switch self {
        case .screenRecording:
            return CGRequestScreenCaptureAccess()
        case .inputMonitoring:
            return CGRequestListenEventAccess()
        case .accessibility:
            let opts = ["AXTrustedCheckOptionPrompt": true] as CFDictionary
            let ax = AXIsProcessTrustedWithOptions(opts)
            _ = CGRequestPostEventAccess()
            return ax
        }
    }

    public var settingsURL: URL {
        let anchor = switch self {
        case .screenRecording: "Privacy_ScreenCapture"
        case .inputMonitoring: "Privacy_ListenEvent"
        case .accessibility: "Privacy_Accessibility"
        }
        return URL(string: "x-apple.systempreferences:com.apple.preference.security?\(anchor)")!
    }

    @MainActor public func openSettings() { NSWorkspace.shared.open(settingsURL) }
}
