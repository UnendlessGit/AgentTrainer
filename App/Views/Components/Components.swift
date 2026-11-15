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
                if let symbol { Image(systemName: symbol).foregroundStyle(.tint) }
                Text(title).font(.headline)
                Spacer()
                trailing
            }
            content
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(.separator.opacity(0.5)))
    }
}

struct StatTile: View {
    let title: String
    let value: String
    var symbol: String? = nil
    var tint: Color = .accentColor

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 5) {
                if let symbol { Image(systemName: symbol).font(.caption).foregroundStyle(tint) }
                Text(title).font(.caption).foregroundStyle(.secondary)
            }
            Text(value).font(.system(.title3, design: .rounded).weight(.semibold)).monospacedDigit().lineLimit(1).minimumScaleFactor(0.6)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.45), in: RoundedRectangle(cornerRadius: 10))
    }
}

struct PageHeader: View {
    let title: String
    let subtitle: String
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.largeTitle.weight(.bold))
            Text(subtitle).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct InlineError: View {
    let message: String
    var body: some View {
        Label(message, systemImage: "exclamationmark.triangle.fill")
            .font(.callout)
            .foregroundStyle(.orange)
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
    }
}

struct LabeledSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    var step: Double? = nil
    var format: (Double) -> String = { String(format: "%.2f", $0) }
    var body: some View {
        HStack {
            Text(title)
            if let step { Slider(value: $value, in: range, step: step) } else { Slider(value: $value, in: range) }
            Text(format(value)).monospacedDigit().frame(width: 64, alignment: .trailing).foregroundStyle(.secondary)
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
