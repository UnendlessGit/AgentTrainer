import AppKit
import Foundation
import SwiftUI

struct AppearanceColor: Hashable, Sendable {
    var red: Double
    var green: Double
    var blue: Double
    var opacity: Double = 1

    var color: Color { Color(red: red, green: green, blue: blue, opacity: opacity) }

    func separated(from canvas: AppearanceColor, by factor: Double) -> AppearanceColor {
        AppearanceColor(
            red: min(1, max(0, canvas.red + (red - canvas.red) * factor)),
            green: min(1, max(0, canvas.green + (green - canvas.green) * factor)),
            blue: min(1, max(0, canvas.blue + (blue - canvas.blue) * factor)),
            opacity: opacity
        )
    }
}

/// A small, intentional theme set replaces the former per-element appearance
/// editor. Every preset is hand-balanced as one palette, which keeps contrast,
/// hierarchy, and control states consistent in both light and dark modes.
enum AppTheme: String, CaseIterable, Identifiable, Sendable {
    case midnight
    case daylight
    case graphite
    case ember

    var id: String { rawValue }

    var name: String {
        switch self {
        case .midnight: "Midnight"
        case .daylight: "Daylight"
        case .graphite: "Graphite"
        case .ember: "Ember"
        }
    }

    var detail: String {
        switch self {
        case .midnight: "Deep blue-black with crisp cyan focus"
        case .daylight: "Soft daylight with high-contrast ink"
        case .graphite: "Neutral, low-distraction studio gray"
        case .ember: "Warm dark surfaces with copper accents"
        }
    }

    var symbol: String {
        switch self {
        case .midnight: "moon.stars.fill"
        case .daylight: "sun.max.fill"
        case .graphite: "circle.lefthalf.filled"
        case .ember: "flame.fill"
        }
    }

    var colorScheme: ColorScheme { self == .daylight ? .light : .dark }

    var configuration: UIConfiguration {
        switch self {
        case .midnight:
            UIConfiguration(
                canvas: .init(red: 0.008, green: 0.012, blue: 0.02),
                sidebar: .init(red: 0.012, green: 0.02, blue: 0.034),
                panel: .init(red: 0.025, green: 0.035, blue: 0.052),
                raised: .init(red: 0.045, green: 0.061, blue: 0.085),
                border: .init(red: 0.55, green: 0.72, blue: 0.9, opacity: 0.16),
                text: .init(red: 0.94, green: 0.97, blue: 1),
                cyan: .init(red: 0.13, green: 0.78, blue: 1),
                violet: .init(red: 0.66, green: 0.48, blue: 1),
                green: .init(red: 0.27, green: 0.88, blue: 0.48),
                amber: .init(red: 1, green: 0.7, blue: 0.2),
                coral: .init(red: 1, green: 0.34, blue: 0.38)
            )
        case .daylight:
            UIConfiguration(
                canvas: .init(red: 0.94, green: 0.96, blue: 0.985),
                sidebar: .init(red: 0.89, green: 0.925, blue: 0.965),
                panel: .init(red: 0.995, green: 0.998, blue: 1),
                raised: .init(red: 0.91, green: 0.94, blue: 0.975),
                border: .init(red: 0.08, green: 0.16, blue: 0.25, opacity: 0.15),
                text: .init(red: 0.07, green: 0.09, blue: 0.14),
                cyan: .init(red: 0, green: 0.48, blue: 0.72),
                violet: .init(red: 0.4, green: 0.25, blue: 0.78),
                green: .init(red: 0.04, green: 0.55, blue: 0.27),
                amber: .init(red: 0.76, green: 0.43, blue: 0.02),
                coral: .init(red: 0.82, green: 0.16, blue: 0.22),
                controlOpacity: 0.12,
                statusOpacity: 0.11
            )
        case .graphite:
            UIConfiguration(
                canvas: .init(red: 0.035, green: 0.038, blue: 0.045),
                sidebar: .init(red: 0.045, green: 0.049, blue: 0.058),
                panel: .init(red: 0.068, green: 0.073, blue: 0.084),
                raised: .init(red: 0.095, green: 0.102, blue: 0.116),
                border: .init(red: 0.75, green: 0.79, blue: 0.86, opacity: 0.14),
                text: .init(red: 0.94, green: 0.95, blue: 0.97),
                cyan: .init(red: 0.45, green: 0.8, blue: 1),
                violet: .init(red: 0.72, green: 0.62, blue: 1),
                green: .init(red: 0.42, green: 0.84, blue: 0.56),
                amber: .init(red: 0.96, green: 0.7, blue: 0.28),
                coral: .init(red: 0.96, green: 0.4, blue: 0.42),
                cornerRadius: 14
            )
        case .ember:
            UIConfiguration(
                canvas: .init(red: 0.032, green: 0.018, blue: 0.015),
                sidebar: .init(red: 0.045, green: 0.024, blue: 0.019),
                panel: .init(red: 0.073, green: 0.039, blue: 0.029),
                raised: .init(red: 0.105, green: 0.057, blue: 0.04),
                border: .init(red: 1, green: 0.65, blue: 0.42, opacity: 0.17),
                text: .init(red: 1, green: 0.96, blue: 0.92),
                cyan: .init(red: 1, green: 0.58, blue: 0.26),
                violet: .init(red: 0.94, green: 0.42, blue: 0.56),
                green: .init(red: 0.55, green: 0.86, blue: 0.42),
                amber: .init(red: 1, green: 0.72, blue: 0.25),
                coral: .init(red: 1, green: 0.34, blue: 0.27)
            )
        }
    }
}

struct UIConfiguration: Hashable, Sendable {
    var appName = "AgentTrainer"
    var appSubtitle = "Local AI studio"
    var canvas: AppearanceColor
    var sidebar: AppearanceColor
    var panel: AppearanceColor
    var raised: AppearanceColor
    var border: AppearanceColor
    var text: AppearanceColor
    var cyan: AppearanceColor
    var violet: AppearanceColor
    var green: AppearanceColor
    var amber: AppearanceColor
    var coral: AppearanceColor
    var fontScale = 1.0
    var cornerRadius = 16.0
    var sidebarWidth = 230.0
    var borderWidth = 1.0
    var controlOpacity = 0.16
    var statusOpacity = 0.1

    var effectiveSidebar: AppearanceColor { sidebar }
    var effectiveBorderWidth: Double { borderWidth }
    var effectiveControlOpacity: Double { controlOpacity }
    var effectiveStatusOpacity: Double { statusOpacity }

    func label(for section: AppSection) -> String { section.rawValue }
}

struct UIAppearanceTuning: Codable, Hashable, Sendable {
    var cornerRadius = 16.0
    var surfaceContrast = 1.0
    var accentIntensity = 1.0
    var sidebarWidth = 230.0

    static let standard = UIAppearanceTuning()

    var sanitized: UIAppearanceTuning {
        UIAppearanceTuning(
            cornerRadius: Self.clamp(cornerRadius, to: 4...28, fallback: 16),
            surfaceContrast: Self.clamp(surfaceContrast, to: 0.7...1.45, fallback: 1),
            accentIntensity: Self.clamp(accentIntensity, to: 0.65...1.5, fallback: 1),
            sidebarWidth: Self.clamp(sidebarWidth, to: 205...300, fallback: 230)
        )
    }

    func applying(to base: UIConfiguration) -> UIConfiguration {
        let tuning = sanitized
        var result = base
        result.sidebar = base.sidebar.separated(from: base.canvas, by: tuning.surfaceContrast)
        result.panel = base.panel.separated(from: base.canvas, by: tuning.surfaceContrast)
        result.raised = base.raised.separated(from: base.canvas, by: tuning.surfaceContrast)
        result.border.opacity = min(0.5, max(0.06, base.border.opacity * (0.75 + tuning.surfaceContrast * 0.25)))
        result.cornerRadius = tuning.cornerRadius
        result.sidebarWidth = tuning.sidebarWidth
        result.borderWidth = min(1.5, max(0.7, base.borderWidth * (0.85 + tuning.surfaceContrast * 0.15)))
        result.controlOpacity = min(0.42, max(0.06, base.controlOpacity * tuning.accentIntensity))
        result.statusOpacity = min(0.3, max(0.05, base.statusOpacity * tuning.accentIntensity))
        return result
    }

    private static func clamp(_ value: Double, to range: ClosedRange<Double>, fallback: Double) -> Double {
        guard value.isFinite else { return fallback }
        return min(range.upperBound, max(range.lowerBound, value))
    }
}

final class UIAppearanceStore: ObservableObject, @unchecked Sendable {
    static let shared = UIAppearanceStore()

    @Published var selectedTheme: AppTheme {
        didSet { UserDefaults.standard.set(selectedTheme.rawValue, forKey: Self.themeKey) }
    }
    @Published var motionEnabled: Bool {
        didSet { UserDefaults.standard.set(motionEnabled, forKey: Self.motionKey) }
    }
    @Published var tuning: UIAppearanceTuning {
        didSet {
            let value = tuning.sanitized
            if let data = try? JSONEncoder().encode(value) { UserDefaults.standard.set(data, forKey: Self.tuningKey) }
        }
    }

    var configuration: UIConfiguration { tuning.applying(to: selectedTheme.configuration) }
    var colorScheme: ColorScheme { selectedTheme.colorScheme }

    private static let themeKey = "AgentTrainer.AppTheme.v2"
    private static let motionKey = "AgentTrainer.InterfaceMotion.v1"
    private static let tuningKey = "AgentTrainer.InterfaceTuning.v1"

    private init() {
        selectedTheme = UserDefaults.standard.string(forKey: Self.themeKey).flatMap(AppTheme.init(rawValue:)) ?? .midnight
        motionEnabled = UserDefaults.standard.object(forKey: Self.motionKey) as? Bool ?? true
        tuning = UserDefaults.standard.data(forKey: Self.tuningKey)
            .flatMap { try? JSONDecoder().decode(UIAppearanceTuning.self, from: $0) }?.sanitized ?? .standard
    }

    func select(_ theme: AppTheme) { selectedTheme = theme }
    func resetTuning() { tuning = .standard }
}
