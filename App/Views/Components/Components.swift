import SwiftUI

/// Titled card used across tabs.
struct Card<Content: View>: View {
    let title: String
    var symbol: String? = nil
    var trailing: AnyView? = nil
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Text(title).font(Theme.font(14, .semibold)).foregroundStyle(Theme.text)
                Spacer()
                trailing
            }
            content
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).strokeBorder(Theme.hairline))
    }
}

struct StatTile: View {
    let title: String
    let value: String
    var symbol: String? = nil
    var tint: Color = Theme.text

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(value).font(Theme.font(20, .semibold)).monospacedDigit().foregroundStyle(tint == .accentColor ? Theme.text : tint)
                .lineLimit(1).minimumScaleFactor(0.6).contentTransition(.numericText())
            Text(title).font(Theme.font(12)).foregroundStyle(Theme.muted).lineLimit(1)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(Theme.hairline))
    }
}

struct PageHeader: View {
    let title: String
    let subtitle: String
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).displayStyle(44)
            Text(subtitle).font(Theme.font(14)).foregroundStyle(Theme.muted)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct InlineError: View {
    let message: String
    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
            Text(message).fixedSize(horizontal: false, vertical: true)
        }
        .font(Theme.font(13))
        .foregroundStyle(Theme.warning)
        .padding(.horizontal, 16).padding(.vertical, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.warning.opacity(0.1), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .transition(.opacity.combined(with: .move(edge: .top)))
    }
}

struct LabeledSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    var step: Double? = nil
    var format: (Double) -> String = { String(format: "%.2f", $0) }
    var body: some View {
        HStack(spacing: 14) {
            Text(title).font(Theme.font(14)).foregroundStyle(Theme.text)
            Spacer(minLength: 12)
            Group {
                if let step { Slider(value: $value, in: range, step: step) } else { Slider(value: $value, in: range) }
            }
            .tint(Theme.text)
            .frame(maxWidth: 220)
            Text(format(value)).font(Theme.font(13, .medium)).monospacedDigit().frame(width: 58, alignment: .trailing).foregroundStyle(Theme.text2)
        }
    }
}

extension Double {
    var durationText: String {
        let s = Int(self.rounded())
        return s >= 3600 ? String(format: "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60) : String(format: "%d:%02d", s / 60, s % 60)
    }
}

extension Int64 {
    var bytesText: String { ByteCountFormatter.string(fromByteCount: self, countStyle: .file) }
}
