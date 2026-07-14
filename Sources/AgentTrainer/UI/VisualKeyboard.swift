import SwiftUI

struct VisualKeyboard: View {
    let usedKeys: Set<UInt16>
    let activeKeys: Set<UInt16>
    var accent: Color = ATColor.cyan
    var compact = false
    @Environment(\.uiMotionEnabled) private var motionEnabled

    private struct Key: Hashable {
        let code: UInt16
        let label: String
        var units: CGFloat = 1
    }

    private let rows: [[Key]] = [
        [Key(code: 53, label: "esc", units: 1.35), Key(code: 122, label: "F1"), Key(code: 120, label: "F2"), Key(code: 99, label: "F3"), Key(code: 118, label: "F4"), Key(code: 96, label: "F5"), Key(code: 97, label: "F6"), Key(code: 98, label: "F7"), Key(code: 100, label: "F8"), Key(code: 101, label: "F9"), Key(code: 109, label: "F10"), Key(code: 103, label: "F11"), Key(code: 111, label: "F12")],
        [Key(code: 50, label: "`"), Key(code: 18, label: "1"), Key(code: 19, label: "2"), Key(code: 20, label: "3"), Key(code: 21, label: "4"), Key(code: 23, label: "5"), Key(code: 22, label: "6"), Key(code: 26, label: "7"), Key(code: 28, label: "8"), Key(code: 25, label: "9"), Key(code: 29, label: "0"), Key(code: 27, label: "–"), Key(code: 24, label: "="), Key(code: 51, label: "delete", units: 1.55)],
        [Key(code: 48, label: "tab", units: 1.4), Key(code: 12, label: "Q"), Key(code: 13, label: "W"), Key(code: 14, label: "E"), Key(code: 15, label: "R"), Key(code: 17, label: "T"), Key(code: 16, label: "Y"), Key(code: 32, label: "U"), Key(code: 34, label: "I"), Key(code: 31, label: "O"), Key(code: 35, label: "P"), Key(code: 33, label: "["), Key(code: 30, label: "]"), Key(code: 42, label: "\\", units: 1.3)],
        [Key(code: 57, label: "caps", units: 1.65), Key(code: 0, label: "A"), Key(code: 1, label: "S"), Key(code: 2, label: "D"), Key(code: 3, label: "F"), Key(code: 5, label: "G"), Key(code: 4, label: "H"), Key(code: 38, label: "J"), Key(code: 40, label: "K"), Key(code: 37, label: "L"), Key(code: 41, label: ";"), Key(code: 39, label: "'"), Key(code: 36, label: "return", units: 1.8)],
        [Key(code: 56, label: "shift", units: 2), Key(code: 6, label: "Z"), Key(code: 7, label: "X"), Key(code: 8, label: "C"), Key(code: 9, label: "V"), Key(code: 11, label: "B"), Key(code: 45, label: "N"), Key(code: 46, label: "M"), Key(code: 43, label: ","), Key(code: 47, label: "."), Key(code: 44, label: "/"), Key(code: 60, label: "shift", units: 1.7), Key(code: 126, label: "↑")],
        [Key(code: 59, label: "ctrl", units: 1.25), Key(code: 58, label: "opt", units: 1.25), Key(code: 55, label: "⌘", units: 1.4), Key(code: 49, label: "space", units: 5.2), Key(code: 54, label: "⌘", units: 1.4), Key(code: 61, label: "opt", units: 1.25), Key(code: 62, label: "ctrl", units: 1.25), Key(code: 123, label: "←"), Key(code: 125, label: "↓"), Key(code: 124, label: "→")]
    ]

    var body: some View {
        VStack(spacing: compact ? 3 : 5) {
            if visibleKeyCodes.isEmpty {
                Text("Keys appear here as they are pressed")
                    .font(.system(size: compact ? 8 : 10, weight: .medium))
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, minHeight: compact ? 16 : 22, alignment: .leading)
            }
            ForEach(Array(rows.enumerated()), id: \.offset) { rowIndex, row in
                let visible = row.filter { visibleKeyCodes.contains($0.code) }
                if !visible.isEmpty {
                    HStack(spacing: compact ? 3 : 5) {
                        ForEach(visible, id: \.self) { key in
                            keyView(key).frame(width: (compact ? 18 : 22) * key.units)
                        }
                        Spacer(minLength: 0)
                    }
                    .padding(.leading, CGFloat(rowIndex) * (compact ? 3 : 5))
                    .frame(height: compact ? 14 : 20)
                }
            }
            if !unmappedUsed.isEmpty {
                HStack(spacing: 4) {
                    Text("Other").foregroundStyle(.secondary)
                    ForEach(unmappedUsed, id: \.self) { code in
                        Text(KeyNames.name(for: code)).padding(.horizontal, 5).padding(.vertical, 2)
                            .background(Capsule().fill(accent.opacity(activeKeys.contains(code) ? 0.38 : 0.14)))
                    }
                    Spacer()
                }.font(.system(size: compact ? 7 : 9, weight: .semibold))
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Visual keyboard")
    }

    private func keyView(_ key: Key) -> some View {
        let active = activeKeys.contains(key.code)
        let used = usedKeys.contains(key.code) || active
        let accessibilityText = key.label + (active ? ", active" : (used ? ", used" : ""))
        let fillColor = active ? accent.opacity(0.42) : (used ? accent.opacity(0.12) : ATColor.raised)
        let strokeColor = active ? accent : (used ? accent.opacity(0.35) : ATColor.border)
        return Text(key.label)
            .font(.system(size: compact ? 6.5 : 8.5, weight: active ? .bold : .medium, design: .rounded))
            .foregroundStyle(active ? Color.white : used ? accent : Color.secondary.opacity(0.65))
            .lineLimit(1).minimumScaleFactor(0.55)
            .frame(maxWidth: .infinity, minHeight: compact ? 14 : 20)
            .background(RoundedRectangle(cornerRadius: ATCorner.scaled(compact ? 3 : 5), style: .continuous).fill(fillColor))
            .overlay(RoundedRectangle(cornerRadius: ATCorner.scaled(compact ? 3 : 5), style: .continuous).stroke(strokeColor, lineWidth: active ? 1.1 : 0.7))
            .scaleEffect(active && motionEnabled ? 1.045 : 1)
            .animation(motionEnabled ? UIMotion.quick : nil, value: active)
            .accessibilityLabel(accessibilityText)
    }

    private var unmappedUsed: [UInt16] {
        let mapped = Set(rows.flatMap { $0.map(\.code) })
        return usedKeys.subtracting(mapped).sorted()
    }

    private var visibleKeyCodes: Set<UInt16> { usedKeys.union(activeKeys) }
}
