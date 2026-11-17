import SwiftUI

/// AgentTrainer's OLED design system: true black, flat surfaces with hairline edges, large
/// tight-tracked numerals, quiet controls and spring motion.
enum Theme {
    static let ground = Color.black
    static let surface = Color(hex: 0x0E0E10)
    static let raised = Color(hex: 0x1C1C1E)
    static let raisedHover = Color(hex: 0x242427)
    static let track = Color(hex: 0x2C2C2E)
    static let off = Color(hex: 0x3A3A3C)
    static let hairline = Color.white.opacity(0.08)
    static let hairlineStrong = Color.white.opacity(0.14)
    static let text = Color(hex: 0xF5F5F7)
    static let text2 = Color(hex: 0xC7C7CC)
    static let muted = Color(hex: 0x8E8E93)
    static let faint = Color(hex: 0x636366)
    static let accent = Color(hex: 0x4C6FFF)
    static let accentLine = Color(hex: 0x6E8BFF)
    static let live = Color(hex: 0x30D158)
    static let danger = Color(hex: 0xFF453A)
    static let dangerText = Color(hex: 0xFF6961)
    static let warning = Color(hex: 0xFFB340)

    static let spring = Animation.spring(response: 0.42, dampingFraction: 0.86)
    static let snappy = Animation.spring(response: 0.28, dampingFraction: 0.9)

    /// The interface face. One place to swap the family (e.g. for Geist).
    static func font(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight)
    }
}

extension Color {
    init(hex: UInt32, opacity: Double = 1) {
        self.init(.sRGB, red: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255, blue: Double(hex & 0xFF) / 255, opacity: opacity)
    }
}

// MARK: Text styles

extension View {
    /// Large titles and key numbers: semibold, tight tracking, tabular figures.
    func displayStyle(_ size: CGFloat) -> some View {
        font(Theme.font(size, .semibold)).tracking(-size * 0.025).monospacedDigit().foregroundStyle(Theme.text)
    }
}

/// Small uppercase section label.
struct SectionLabel: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text.uppercased()).font(Theme.font(11, .semibold)).tracking(1.3).foregroundStyle(Theme.muted)
    }
}

// MARK: Surfaces

struct Panel<Content: View>: View {
    var padding: CGFloat = 18
    var radius: CGFloat = 16
    @ViewBuilder var content: Content
    var body: some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.surface, in: RoundedRectangle(cornerRadius: radius, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: radius, style: .continuous).strokeBorder(Theme.hairline))
    }
}

/// A grouped list of rows separated by hairlines, inside a panel.
struct Rows<Content: View>: View {
    var title: String? = nil
    var footer: String? = nil
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let title {
                Text(title).font(Theme.font(14, .semibold)).foregroundStyle(Theme.text).padding(.horizontal, 18).padding(.top, 16).padding(.bottom, 4)
            }
            Group(subviews: content) { subviews in
                ForEach(Array(subviews.enumerated()), id: \.offset) { i, sub in
                    if i > 0 { Rectangle().fill(Color.white.opacity(0.06)).frame(height: 1).padding(.leading, 18) }
                    sub.padding(.horizontal, 18).padding(.vertical, 10).frame(minHeight: 44)
                }
            }
            if let footer {
                Text(footer).font(Theme.font(12)).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 18).padding(.bottom, 14).padding(.top, 2)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).strokeBorder(Theme.hairline))
    }
}

/// One row: a label (with optional detail) and a trailing control.
struct Row<Trailing: View>: View {
    let title: String
    var detail: String? = nil
    @ViewBuilder var trailing: Trailing
    var body: some View {
        HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(Theme.font(14)).foregroundStyle(Theme.text)
                if let detail { Text(detail).font(Theme.font(12)).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true) }
            }
            Spacer(minLength: 8)
            trailing
        }
    }
}

extension Row where Trailing == EmptyView {
    init(title: String, detail: String? = nil) { self.init(title: title, detail: detail) { EmptyView() } }
}

/// A big number with a caption under it.
struct BigStat: View {
    let value: String
    var unit: String? = nil
    let caption: String
    var size: CGFloat = 40
    var color: Color = Theme.text
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(value).displayStyle(size).foregroundStyle(color).contentTransition(.numericText()).lineLimit(1).minimumScaleFactor(0.5)
                if let unit { Text(unit).font(Theme.font(size * 0.5, .semibold)).foregroundStyle(Theme.faint) }
            }
            Text(caption).font(Theme.font(12)).foregroundStyle(Theme.muted).lineLimit(1)
        }
    }
}

/// A pulsing status dot.
struct LiveDot: View {
    var color: Color = Theme.live
    var size: CGFloat = 7
    var pulsing = true
    @State private var on = false
    var body: some View {
        Circle().fill(color).frame(width: size, height: size)
            .opacity(pulsing && on ? 0.3 : 1)
            .shadow(color: color.opacity(0.6), radius: pulsing ? 4 : 0)
            .onAppear { if pulsing { withAnimation(.easeInOut(duration: 0.7).repeatForever(autoreverses: true)) { on = true } } }
    }
}

struct Pill: View {
    let text: String
    var color: Color = Theme.live
    var body: some View {
        Text(text).font(Theme.font(12, .semibold)).foregroundStyle(color)
            .padding(.horizontal, 10).padding(.vertical, 4)
            .background(color.opacity(0.14), in: Capsule())
    }
}

/// A flat placeholder frame for thumbnails and previews.
struct Placeholder: View {
    var symbol: String = "film"
    var radius: CGFloat = 8
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: radius, style: .continuous).fill(Theme.raised)
            Image(systemName: symbol).font(.system(size: 13)).foregroundStyle(Theme.faint)
        }
        .overlay(RoundedRectangle(cornerRadius: radius, style: .continuous).strokeBorder(Color.white.opacity(0.06)))
    }
}

// MARK: Motion

private struct RiseIn: ViewModifier {
    let delay: Double
    @State private var shown = false
    func body(content: Content) -> some View {
        content
            .opacity(shown ? 1 : 0)
            .offset(y: shown ? 0 : 10)
            .onAppear { withAnimation(.spring(response: 0.6, dampingFraction: 0.9).delay(delay)) { shown = true } }
    }
}

extension View {
    /// Content rises 10 pt and fades in on appear (staggered with `delay`).
    func riseIn(_ delay: Double = 0) -> some View { modifier(RiseIn(delay: delay)) }
}

// MARK: Buttons

enum ATButtonKind { case primary, secondary, destructive, quiet }

struct ATButtonStyle: ButtonStyle {
    var kind: ATButtonKind = .secondary
    var height: CGFloat = 40
    func makeBody(configuration: Configuration) -> some View { StyledButton(configuration: configuration, kind: kind, height: height) }

    private struct StyledButton: View {
        let configuration: Configuration
        let kind: ATButtonKind
        let height: CGFloat
        @State private var hover = false
        @Environment(\.isEnabled) private var enabled
        var body: some View {
            configuration.label
                .font(Theme.font(height >= 40 ? 14 : 13, .semibold))
                .foregroundStyle(foreground)
                .padding(.horizontal, height >= 40 ? 18 : 12)
                .frame(height: height)
                .background(background, in: RoundedRectangle(cornerRadius: height >= 40 ? 12 : 9, style: .continuous))
                .opacity(enabled ? 1 : 0.4)
                .scaleEffect(configuration.isPressed ? 0.97 : 1)
                .animation(Theme.snappy, value: configuration.isPressed)
                .animation(.easeOut(duration: 0.15), value: hover)
                .onHover { hover = $0 }
                .contentShape(Rectangle())
        }
        var foreground: Color {
            switch kind {
            case .primary: .black
            case .destructive: Theme.dangerText
            case .secondary: Theme.text
            case .quiet: hover ? Theme.text : Theme.text2
            }
        }
        var background: Color {
            switch kind {
            case .primary: hover ? .white : Theme.text
            case .secondary, .destructive: hover ? Theme.raisedHover : Theme.raised
            case .quiet: hover ? Theme.raised : .clear
            }
        }
    }
}

extension ButtonStyle where Self == ATButtonStyle {
    static var atPrimary: ATButtonStyle { ATButtonStyle(kind: .primary) }
    static var atSecondary: ATButtonStyle { ATButtonStyle(kind: .secondary) }
    static var atDestructive: ATButtonStyle { ATButtonStyle(kind: .destructive) }
    static var atSmall: ATButtonStyle { ATButtonStyle(kind: .secondary, height: 30) }
    static var atSmallPrimary: ATButtonStyle { ATButtonStyle(kind: .primary, height: 30) }
    static var atSmallDestructive: ATButtonStyle { ATButtonStyle(kind: .destructive, height: 30) }
    static var atQuiet: ATButtonStyle { ATButtonStyle(kind: .quiet, height: 30) }
}

/// A plain list-row button with hover and selection backgrounds.
struct RowButtonStyle: ButtonStyle {
    var selected = false
    var radius: CGFloat = 10
    func makeBody(configuration: Configuration) -> some View { RowBody(configuration: configuration, selected: selected, radius: radius) }
    private struct RowBody: View {
        let configuration: Configuration
        let selected: Bool
        let radius: CGFloat
        @State private var hover = false
        var body: some View {
            configuration.label
                .foregroundStyle(Theme.text)
                .background(RoundedRectangle(cornerRadius: radius, style: .continuous).fill(selected ? Theme.raised : hover ? Color.white.opacity(0.04) : .clear))
                .overlay(RoundedRectangle(cornerRadius: radius, style: .continuous).strokeBorder(selected ? Color.white.opacity(0.12) : .clear))
                .scaleEffect(configuration.isPressed ? 0.99 : 1)
                .animation(.easeOut(duration: 0.15), value: hover)
                .animation(Theme.snappy, value: selected)
                .onHover { hover = $0 }
                .contentShape(Rectangle())
        }
    }
}

// MARK: Toggle

struct OLEDToggleStyle: ToggleStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 12) {
            configuration.label
            Spacer(minLength: 0)
            Switch(isOn: configuration.$isOn)
        }
    }

    struct Switch: View {
        @Binding var isOn: Bool
        @Environment(\.isEnabled) private var enabled
        var body: some View {
            Button { withAnimation(Theme.snappy) { isOn.toggle() } } label: {
                Capsule().fill(isOn ? Theme.live : Theme.off)
                    .frame(width: 40, height: 24)
                    .overlay(alignment: isOn ? .trailing : .leading) {
                        Circle().fill(.white).frame(width: 20, height: 20).padding(2).shadow(color: .black.opacity(0.25), radius: 2, y: 1)
                    }
            }
            .buttonStyle(.plain)
            .opacity(enabled ? 1 : 0.4)
            .accessibilityAddTraits(.isToggle)
            .accessibilityValue(isOn ? "On" : "Off")
        }
    }
}

extension ToggleStyle where Self == OLEDToggleStyle {
    static var oled: OLEDToggleStyle { OLEDToggleStyle() }
}

// MARK: Segmented control

/// A segmented picker with a sliding white selection.
struct Segmented<T: Hashable>: View {
    let options: [(T, String)]
    @Binding var selection: T
    var compact = false
    @Namespace private var ns
    var body: some View {
        HStack(spacing: 2) {
            ForEach(options, id: \.0) { value, label in
                let on = value == selection
                Button { withAnimation(Theme.spring) { selection = value } } label: {
                    Text(label)
                        .font(Theme.font(13, on ? .semibold : .medium))
                        .foregroundStyle(on ? Color.black : Theme.muted)
                        .padding(.horizontal, compact ? 12 : 16)
                        .frame(height: compact ? 28 : 34)
                        .background {
                            if on { RoundedRectangle(cornerRadius: compact ? 7 : 9, style: .continuous).fill(Theme.text).matchedGeometryEffect(id: "sel", in: ns) }
                        }
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(3)
        .background(compact ? Theme.raised : Theme.surface, in: RoundedRectangle(cornerRadius: compact ? 10 : 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: compact ? 10 : 12, style: .continuous).strokeBorder(compact ? .clear : Theme.hairline))
    }
}

// MARK: Field

struct DarkFieldStyle: TextFieldStyle {
    func _body(configuration: TextField<Self._Label>) -> some View {
        configuration
            .textFieldStyle(.plain)
            .font(Theme.font(14))
            .foregroundStyle(Theme.text)
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .frame(minHeight: 36)
            .background(Theme.raised, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}
