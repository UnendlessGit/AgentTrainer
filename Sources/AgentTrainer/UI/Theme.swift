import SwiftUI

private struct UIMotionEnabledKey: EnvironmentKey {
    static let defaultValue = true
}

extension EnvironmentValues {
    var uiMotionEnabled: Bool {
        get { self[UIMotionEnabledKey.self] }
        set { self[UIMotionEnabledKey.self] = newValue }
    }
}

enum UIMotion {
    static let quick = Animation.easeOut(duration: 0.12)
    static let chartUpdate = Animation.easeOut(duration: 0.16)
    static let standard = Animation.easeInOut(duration: 0.18)
    static let reveal = Animation.easeOut(duration: 0.32)
}

enum ATColor {
    private static var configuration: UIConfiguration { UIAppearanceStore.shared.configuration }
    static var canvas: Color { configuration.canvas.color }
    static var sidebar: Color { configuration.effectiveSidebar.color }
    static var panel: Color { configuration.panel.color }
    static var raised: Color { configuration.raised.color }
    static var border: Color { configuration.border.color }
    static var text: Color { configuration.text.color }
    static var cyan: Color { configuration.cyan.color }
    static var violet: Color { configuration.violet.color }
    static var green: Color { configuration.green.color }
    static var amber: Color { configuration.amber.color }
    static var coral: Color { configuration.coral.color }
}

enum ATCorner {
    static func scaled(_ reference: CGFloat) -> CGFloat {
        let scale = CGFloat(UIAppearanceStore.shared.configuration.cornerRadius / 16)
        return min(36, max(1.5, reference * scale))
    }
}

/// The fast surface used throughout the backup build: one fill, one border,
/// and no per-card blur, gradient, shadow, or compositing group.
struct OLEDCard<Content: View>: View {
    var padding: CGFloat = 18
    @ViewBuilder let content: Content
    @ObservedObject private var appearance = UIAppearanceStore.shared
    @Environment(\.uiMotionEnabled) private var motionEnabled
    @State private var hovering = false

    var body: some View {
        let configuration = appearance.configuration
        let shape = RoundedRectangle(cornerRadius: CGFloat(configuration.cornerRadius), style: .continuous)
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(shape.fill(ATColor.panel))
            .overlay(shape.strokeBorder(ATColor.border.opacity(hovering ? 1.45 : 1), lineWidth: configuration.effectiveBorderWidth))
            .scaleEffect(hovering && motionEnabled ? 1.0015 : 1)
            .animation(motionEnabled ? UIMotion.quick : nil, value: hovering)
            .onHover { hovering = motionEnabled ? $0 : false }
    }
}

struct StatusPill: View {
    let text: String
    let color: Color
    @ObservedObject private var appearance = UIAppearanceStore.shared

    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(text).font(.caption.weight(.semibold)).lineLimit(1)
        }
        .foregroundStyle(color)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Capsule().fill(color.opacity(appearance.configuration.effectiveStatusOpacity)))
        .overlay(Capsule().stroke(color.opacity(0.3), lineWidth: 0.8))
    }
}

struct MetricCard: View {
    let title: String
    let value: String
    let symbol: String
    let color: Color
    @ObservedObject private var appearance = UIAppearanceStore.shared

    var body: some View {
        OLEDCard(padding: 14) {
            VStack(alignment: .leading, spacing: 12) {
                Image(systemName: symbol).foregroundStyle(color).font(.title3)
                Text(value)
                    .font(.system(size: 22 * appearance.configuration.fontScale, weight: .semibold, design: .rounded))
                    .foregroundStyle(color)
                Text(title).font(.caption).foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

struct SectionTitle: View {
    let title: String
    let subtitle: String?
    @ObservedObject private var appearance = UIAppearanceStore.shared

    init(_ title: String, _ subtitle: String? = nil) {
        self.title = title
        self.subtitle = subtitle
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.system(size: 28 * appearance.configuration.fontScale, weight: .bold, design: .rounded))
            if let subtitle {
                Text(subtitle).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct FlatPrimaryButtonStyle: ButtonStyle {
    let color: Color
    @ObservedObject private var appearance = UIAppearanceStore.shared

    func makeBody(configuration button: Configuration) -> some View {
        FlatPrimaryButtonBody(button: button, color: color, appearance: appearance)
    }
}

private struct FlatPrimaryButtonBody: View {
    let button: ButtonStyleConfiguration
    let color: Color
    @ObservedObject var appearance: UIAppearanceStore
    @Environment(\.uiMotionEnabled) private var motionEnabled
    @State private var hovering = false

    var body: some View {
        let radius = CGFloat(min(18, max(3, appearance.configuration.cornerRadius * 0.62)))
        let shape = RoundedRectangle(cornerRadius: radius, style: .continuous)
        button.label
            .fontWeight(.semibold)
            .padding(.horizontal, 15)
            .padding(.vertical, 9)
            .frame(minHeight: 34)
            .contentShape(shape)
            .foregroundStyle(color)
            .background(shape.fill(color.opacity(min(0.34, appearance.configuration.effectiveControlOpacity * (button.isPressed ? 1.55 : hovering ? 1.05 : 0.82)))))
            .overlay(shape.stroke(color.opacity(button.isPressed ? 0.9 : hovering ? 0.7 : 0.5), lineWidth: 0.8))
            .scaleEffect(motionEnabled ? (button.isPressed ? 0.965 : hovering ? 1.018 : 1) : 1)
            .animation(motionEnabled ? UIMotion.quick : nil, value: button.isPressed)
            .animation(motionEnabled ? UIMotion.quick : nil, value: hovering)
            .onHover { hovering = motionEnabled ? $0 : false }
    }
}

private struct RaisedSurfaceModifier: ViewModifier {
    let cornerRadius: CGFloat
    let tint: Color?
    @ObservedObject private var appearance = UIAppearanceStore.shared

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: ATCorner.scaled(cornerRadius), style: .continuous)
        content
            .background(shape.fill(tint?.opacity(0.075) ?? ATColor.raised))
            .overlay(shape.strokeBorder(tint?.opacity(0.28) ?? ATColor.border, lineWidth: 0.7))
    }
}

private struct HoverResponseModifier: ViewModifier {
    let scale: CGFloat
    @Environment(\.uiMotionEnabled) private var motionEnabled
    @State private var hovering = false

    func body(content: Content) -> some View {
        content
            .scaleEffect(hovering && motionEnabled ? scale : 1)
            .opacity(hovering && motionEnabled ? 0.96 : 1)
            .animation(motionEnabled ? UIMotion.quick : nil, value: hovering)
            .onHover { hovering = motionEnabled ? $0 : false }
    }
}

extension View {
    func primaryButton(color: Color = ATColor.cyan) -> some View {
        buttonStyle(FlatPrimaryButtonStyle(color: color))
    }

    func raisedGlassSurface(cornerRadius: CGFloat = 12, tint: Color? = nil) -> some View {
        modifier(RaisedSurfaceModifier(cornerRadius: cornerRadius, tint: tint))
    }

    func uiHoverResponse(scale: CGFloat = 1.012) -> some View {
        modifier(HoverResponseModifier(scale: scale))
    }
}
