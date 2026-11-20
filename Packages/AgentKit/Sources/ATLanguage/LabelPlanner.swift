import ATCore
import Foundation

/// A stretch of a recording that gets one instruction.
public struct LabelWindow: Hashable, Sendable {
    public var start: Double
    public var end: Double
    public init(start: Double, end: Double) { self.start = start; self.end = end }
    public var duration: Double { end - start }
}

/// Cuts a recording into the clips the labeller looks at, and describes the input in each one.
///
/// Clips end where what you're doing changes (keys held, mouse, idle), close to the chosen length,
/// instead of at fixed marks that split one action across two labels. Parts left out of training
/// are skipped.
public enum LabelPlanner {
    /// Steps per second the planner expects (`TimelineBuilder.build(..., rateHz: 10)`).
    public static let rate = 10.0

    public static func windows(steps: [TimelineStep], duration: Double, target: Double, skip: [TimeRange]) -> [LabelWindow] {
        guard duration > 0.5 else { return [] }
        // The parts that may be labelled: the whole recording minus the skipped ranges.
        var spans: [LabelWindow] = [LabelWindow(start: 0, end: duration)]
        for r in skip.sorted(by: { $0.start < $1.start }) {
            spans = spans.flatMap { s -> [LabelWindow] in
                guard r.end > s.start, r.start < s.end else { return [s] }
                return [LabelWindow(start: s.start, end: max(s.start, r.start)), LabelWindow(start: min(s.end, r.end), end: s.end)]
                    .filter { $0.duration >= 1.5 }
            }
        }
        let change = changeScores(steps)
        var out: [LabelWindow] = []
        for span in spans {
            var s = span.start
            while span.end - s > 0.01 {
                let left = span.end - s
                if left <= target * 1.5 { out.append(LabelWindow(start: s, end: span.end)); break }
                let cut = bestCut(from: s + target * 0.6, to: s + target * 1.4, ideal: s + target, change: change, limit: span.end - target * 0.5)
                out.append(LabelWindow(start: s, end: cut))
                s = cut
            }
        }
        return out
    }

    /// How much the input changes at each step: keys/buttons whose held share differs between the
    /// second before and the second after, plus starting or stopping to move the mouse.
    static func changeScores(_ steps: [TimelineStep]) -> [Double] {
        let n = steps.count, w = Int(rate)
        guard n > 2 * w else { return Array(repeating: 0, count: n) }
        func share(_ range: Range<Int>) -> (keys: [String: Double], mouse: Double) {
            var k: [String: Double] = [:], m = 0.0
            for i in range {
                let a = steps[i].action
                for key in a.keys { k[key, default: 0] += 1 }
                for p in a.pointer where p.isButton { k["p:" + p.rawValue, default: 0] += 1 }
                if abs(a.dx) + abs(a.dy) > 1 { m += 1 }
            }
            let c = Double(range.count)
            return (k.mapValues { $0 / c }, m / c)
        }
        var scores = Array(repeating: 0.0, count: n)
        for i in w ..< n - w {
            let a = share(i - w ..< i), b = share(i ..< i + w)
            var score = 0.0
            for key in Set(a.keys.keys).union(b.keys.keys) { score += abs((a.keys[key] ?? 0) - (b.keys[key] ?? 0)) }
            score += abs(a.mouse - b.mouse) * 0.5
            scores[i] = score
        }
        return scores
    }

    /// The step in [from, to] with the clearest change, preferring times near `ideal`.
    static func bestCut(from: Double, to: Double, ideal: Double, change: [Double], limit: Double) -> Double {
        let hi = min(to, limit)
        guard hi > from, !change.isEmpty else { return min(ideal, limit) }
        var best = min(ideal, limit), bestScore = -Double.infinity
        var i = Int((from * rate).rounded(.up))
        while Double(i) / rate <= hi, i < change.count {
            let t = Double(i) / rate
            let score = change[i] - abs(t - ideal) / max(ideal - from, 0.5) * 0.35
            if score > bestScore { bestScore = score; best = t }
            i += 1
        }
        // No real change anywhere: cut at the planned length.
        return bestScore < 0.2 ? min(ideal, limit) : best
    }

    // MARK: Input log

    /// What the person did during a clip, with times relative to the clip start, e.g.
    /// `0.0–2.4 s: holding W` / `1.2 s: tapped Space` / `0.5–3.0 s: camera turning right`.
    public static func inputLog(_ steps: [TimelineStep], from start: Double, aiControlled: [TimeRange] = []) -> String {
        guard !steps.isEmpty else { return "no input recorded" }
        var lines: [(Double, String)] = []
        func label(_ k: String) -> String { k.hasPrefix("p:") ? (PointerEvent(rawValue: String(k.dropFirst(2)))?.label ?? k) : (KeyCatalog.info(k)?.label.uppercased() ?? k) }
        // Held keys and buttons: continuous runs.
        var ids = Set<String>()
        for s in steps { ids.formUnion(s.action.keys); ids.formUnion(s.action.pointer.filter(\.isButton).map { "p:" + $0.rawValue }) }
        for id in ids.sorted() {
            var runStart: Int?
            for i in 0 ... steps.count {
                let on = i < steps.count && (steps[i].action.keys.contains(id) || steps[i].action.pointer.contains { "p:" + $0.rawValue == id })
                if on, runStart == nil { runStart = i }
                if !on, let r = runStart {
                    let t0 = steps[r].time - start, t1 = (i < steps.count ? steps[i].time : steps[steps.count - 1].time + 1 / rate) - start
                    lines.append((t0, t1 - t0 < 0.35 ? String(format: "%.1f s: tapped %@", t0, label(id))
                                                     : String(format: "%.1f–%.1f s: holding %@", t0, t1, label(id))))
                    runStart = nil
                }
            }
        }
        // Scrolling.
        for s in steps where s.action.pointer.contains(where: \.isScroll) {
            lines.append((s.time - start, String(format: "%.1f s: scrolled", s.time - start)))
        }
        // Mouse / camera motion, per half second.
        let chunk = Int(rate / 2)
        var motion: [(Double, Double, String)] = []
        var i = 0
        while i < steps.count {
            let part = steps[i ..< min(steps.count, i + chunk)]
            let dx = part.reduce(0) { $0 + $1.action.dx }, dy = part.reduce(0) { $0 + $1.action.dy }
            let t0 = part.first!.time - start, t1 = part.last!.time + 1 / rate - start
            var dir = ""
            if abs(dx) > 25 || abs(dy) > 25 {
                let h = abs(dx) > 25 ? (dx > 0 ? "right" : "left") : ""
                let v = abs(dy) > 25 && abs(dy) > abs(dx) * 0.4 ? (dy > 0 ? "down" : "up") : ""
                dir = [h, v].filter { !$0.isEmpty }.joined(separator: " and ")
                if max(abs(dx), abs(dy)) > 300 { dir = "quickly " + dir }
            }
            if let last = motion.last, last.2 == dir, abs(last.1 - t0) < 0.01 { motion[motion.count - 1].1 = t1 } else { motion.append((t0, t1, dir)) }
            i += chunk
        }
        for m in motion where !m.2.isEmpty {
            lines.append((m.0, String(format: "%.1f–%.1f s: mouse moving %@", m.0, m.1, m.2)))
        }
        // The AI's part of a correction recording.
        for r in aiControlled {
            let a = max(r.start, start) - start, b = min(r.end, (steps.last?.time ?? start) + 1 / rate) - start
            if b > a { lines.append((a, String(format: "%.1f–%.1f s: the AI was playing (not the person)", a, b))) }
        }
        guard !lines.isEmpty else { return "no input: the person waited or watched" }
        return lines.sorted { $0.0 < $1.0 }.map(\.1).joined(separator: "\n")
    }
}
