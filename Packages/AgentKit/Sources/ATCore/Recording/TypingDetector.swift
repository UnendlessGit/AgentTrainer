import Foundation

/// Finds stretches of typing in a recording and turns them into `type "…"` instructions, which
/// teach models with literal typing to copy text from an instruction.
public enum TypingDetector {
    public static func segments(events: [InputEventRecord], minLength: Int = 3, maxGap: Double = 1.2) -> [TextSegment] {
        var out: [TextSegment] = []
        var text = "", start = 0.0, last = -10.0
        var shift = false
        func flush() {
            let trimmed = text.trimmingCharacters(in: .whitespaces)
            if trimmed.count >= minLength { out.append(TextSegment(start: max(0, start - 0.3), end: last + 0.4, text: "type \"\(trimmed)\"", source: .typing)) }
            text = ""
        }
        for e in events where !e.isSynthetic {
            let t = Double(e.timeNs) / 1e9
            if e.kind == .flagsChanged, e.code == 0x38 || e.code == 0x3C { shift = e.flags & 2 != 0; continue }
            guard e.kind == .keyDown, let info = KeyCatalog.byKeyCode[e.code] else { continue }
            guard let ch = shift ? info.shiftedCharacter : info.character, ch != "\t", ch != "\n" else {
                if info.id == "return" || info.id == "tab" { flush() }
                continue
            }
            if t - last > maxGap { flush(); start = t }
            if text.isEmpty { start = t }
            text.append(ch)
            last = t
        }
        flush()
        return out
    }
}
