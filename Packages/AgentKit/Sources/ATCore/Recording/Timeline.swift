import CoreGraphics
import Foundation

/// One model timestep derived from a recording.
public struct TimelineStep: Hashable, Sendable {
    /// Index into the recording's frames (the latest frame displayed at the step's start).
    public var frameIndex: Int
    /// Step start time in seconds since the recording's time zero.
    public var time: Double
    /// Input during (start, start + 1/rate]: what the person did after seeing the frame.
    public var action: TimestepAction
    /// True when the model (not the person) was in control - kept as history, not trained on.
    public var modelControlled: Bool
    public var excluded: Bool
    public var instruction: String?
}

/// Resamples a recording to a fixed control rate using host-clock timestamps, so frames and
/// input line up exactly regardless of capture jitter.
///
/// Labels follow Pixel2Play: keys/buttons held at the end of the window, **plus** anything
/// pressed during it (P2P drops taps shorter than a window); pointer deltas and scroll are summed.
public enum TimelineBuilder {
    public static func build(frames: [FrameRecord], events: [InputEventRecord], meta: RecordingMeta,
                             annotations: RecordingAnnotations, rateHz: Int) -> [TimelineStep] {
        guard let first = frames.first, let last = frames.last else { return [] }
        let stepNs = Int64(1_000_000_000 / Double(rateHz))
        let t0 = first.timeNs
        let count = Int((last.timeNs - t0) / stepNs)
        guard count > 0 else { return [] }
        let src = meta.sourceRect.cgRect
        var steps: [TimelineStep] = []
        steps.reserveCapacity(count)

        var heldKeys = Set<UInt16>()
        var heldButtons = Set<UInt16>()
        var pointer = (x: Double(src.midX), y: Double(src.midY))
        var e = 0, f = 0
        // Apply events up to and including the start of the first step.
        while e < events.count, events[e].timeNs <= t0 { apply(events[e], &heldKeys, &heldButtons, &pointer); e += 1 }

        for k in 0 ..< count {
            let start = t0 + Int64(k) * stepNs, end = start + stepNs
            while f + 1 < frames.count, frames[f + 1].timeNs <= start { f += 1 }
            var pressedKeys = Set<UInt16>(), pressedButtons = Set<UInt16>()
            var dx = 0.0, dy = 0.0, sx = 0.0, sy = 0.0
            while e < events.count, events[e].timeNs <= end {
                let ev = events[e]
                switch ev.kind {
                case .keyDown: pressedKeys.insert(ev.code)
                case .flagsChanged where ev.flags & 2 != 0: pressedKeys.insert(ev.code)
                case .buttonDown: pressedButtons.insert(ev.code)
                case .mouseMove: dx += Double(ev.dx); dy += Double(ev.dy)
                case .scroll: sx += Double(ev.dx); sy += Double(ev.dy)
                default: break
                }
                apply(ev, &heldKeys, &heldButtons, &pointer)
                e += 1
            }
            var a = TimestepAction()
            for code in heldKeys.union(pressedKeys) { if let id = KeyCatalog.byKeyCode[code]?.id { a.keys.insert(id) } }
            for b in heldButtons.union(pressedButtons) {
                if let p = PointerEvent.allCases.first(where: { $0.buttonNumber == Int(b) }) { a.pointer.insert(p) }
            }
            if let s = PointerEvent.fromScroll(lines: sx, sy) { a.pointer.insert(s) }
            a.dx = dx; a.dy = dy
            a.absX = src.width > 0 ? min(1, max(0, (pointer.x - src.minX) / src.width)) : 0.5
            a.absY = src.height > 0 ? min(1, max(0, (pointer.y - src.minY) / src.height)) : 0.5
            let time = Double(start - t0) / 1e9
            steps.append(TimelineStep(
                frameIndex: f, time: time, action: a,
                modelControlled: annotations.modelControlled.contains { $0.contains(time) },
                excluded: annotations.excluded.contains { $0.contains(time) },
                instruction: annotations.instruction(at: time)))
        }
        return steps
    }

    private static func apply(_ ev: InputEventRecord, _ keys: inout Set<UInt16>, _ buttons: inout Set<UInt16>, _ pointer: inout (x: Double, y: Double)) {
        switch ev.kind {
        case .keyDown: keys.insert(ev.code)
        case .keyUp: keys.remove(ev.code)
        case .flagsChanged: if ev.flags & 2 != 0 { keys.insert(ev.code) } else { keys.remove(ev.code) }
        case .buttonDown: buttons.insert(ev.code)
        case .buttonUp: buttons.remove(ev.code)
        case .mouseMove: break
        case .scroll: break
        }
        if ev.kind == .mouseMove || ev.kind == .buttonDown || ev.kind == .buttonUp {
            pointer = (Double(ev.x), Double(ev.y))
        }
    }

    /// Standard deviation of summed pointer deltas per window at `rateHz` (non-idle windows), for mouse-unit calibration.
    public static func pointerDeltaStd(_ steps: [TimelineStep]) -> (x: Double, y: Double) {
        let moving = steps.filter { $0.action.dx != 0 || $0.action.dy != 0 }
        guard moving.count > 10 else { return (0, 0) }
        func std(_ v: [Double]) -> Double {
            let m = v.reduce(0, +) / Double(v.count)
            return (v.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(v.count)).squareRoot()
        }
        return (std(moving.map(\.action.dx)), std(moving.map(\.action.dy)))
    }
}
