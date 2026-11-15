import ATCore
import Foundation

/// Measures input → screen timing from a recording of the Sync Check window (which turns white
/// on every key press). Reports how long after each key event the change appeared in a captured
/// frame - a direct check that frames and input share one clock.
public enum SyncCheck {
    public struct Result: Sendable {
        public var samples: Int
        public var medianMs: Double
        public var p90Ms: Double
        public var jitterMs: Double
        public var summary: String {
            samples == 0 ? "No key presses found. Record the Sync Check window and press keys a few times."
                : String(format: "%d presses · key-to-frame %.0f ms median (90%% within %.0f ms) · jitter ±%.0f ms", samples, medianMs, p90Ms, jitterMs)
        }
    }

    public static func analyze(recording: URL) throws -> Result {
        let meta = try RecordingIO.readMeta(recording)
        let frames = try RecordingIO.readFrames(recording), events = try RecordingIO.readEvents(recording)
        let reader = try ProxyReader(url: recording.appendingPathComponent(RecordingPackage.proxy(size: meta.proxySizes.first ?? 192)))
        func brightness(_ i: Int) -> Double {
            guard let f = try? reader.frame(min(i, reader.frameCount - 1)) else { return 0 }
            return f.withUnsafeBytes { p in Double(p.bindMemory(to: UInt8.self).reduce(0) { $0 + Int($1) }) / Double(p.count) }
        }
        var lat: [Double] = []
        for e in events where e.kind == .keyDown {
            guard let first = frames.firstIndex(where: { $0.timeNs > e.timeNs }) else { continue }
            // Was the screen dark just before the press?
            if first > 0, brightness(first - 1) > 100 { continue }
            for j in first ..< min(frames.count, first + 30) where brightness(j) > 140 {
                lat.append(Double(frames[j].timeNs - e.timeNs) / 1e6)
                break
            }
        }
        guard !lat.isEmpty else { return Result(samples: 0, medianMs: 0, p90Ms: 0, jitterMs: 0) }
        lat.sort()
        let med = lat[lat.count / 2], p90 = lat[min(lat.count - 1, Int(Double(lat.count) * 0.9))]
        let mad = lat.map { abs($0 - med) }.sorted()[lat.count / 2]
        return Result(samples: lat.count, medianMs: med, p90Ms: p90, jitterMs: mad)
    }
}
