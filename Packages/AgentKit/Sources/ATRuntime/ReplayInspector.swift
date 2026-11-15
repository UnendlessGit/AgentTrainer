import ATCapture
import ATCore
import ATModel
import Foundation
import MLX

/// Runs a model over a stretch of a recording (with the person's real action history) and returns
/// what it would have done at each step, for side-by-side comparison in the Library.
public enum ReplayInspector {
    public struct Result: Sendable {
        public var start: Double
        public var rateHz: Int
        public var human: [TimestepAction]
        public var model: [TimestepAction]
        public var keyAgreement: Double
    }

    public static func compare(modelURL: URL, recording: URL, from start: Double, seconds: Double) throws -> Result {
        let (model, def, _) = try ModelLoader.load(modelURL)
        let meta = try RecordingIO.readMeta(recording)
        let frames = try RecordingIO.readFrames(recording), events = try RecordingIO.readEvents(recording)
        let steps = TimelineBuilder.build(frames: frames, events: events, meta: meta, annotations: RecordingIO.readAnnotations(recording), rateHz: def.architecture.stepRateHz)
        let size = def.architecture.frameSize
        let proxyURL = recording.appendingPathComponent(RecordingPackage.proxy(size: size))
        let reader = try ProxyReader(url: FileManager.default.fileExists(atPath: proxyURL.path) ? proxyURL : recording.appendingPathComponent(RecordingPackage.proxy(size: 192)))
        let stepper = PolicyStepper(model: model)
        stepper.temperature = 0
        let rate = Double(def.architecture.stepRateHz)
        // Warm the memory with the seconds before the window, like a running model would have.
        let first = max(0, Int((start - def.architecture.memorySeconds) * rate)), from = Int(start * rate)
        let to = min(steps.count, from + Int(seconds * rate))
        guard from < to else { return Result(start: start, rateHz: Int(rate), human: [], model: [], keyAgreement: 0) }
        var human: [TimestepAction] = [], predicted: [TimestepAction] = []
        var prev: [Int32]?
        var agree = 0
        for i in first ..< to {
            var frame = try reader.frame(min(Int(steps[i].frameIndex), reader.frameCount - 1))
            if def.actions.mouseMode == .absolute {
                let c = i > 0 ? (steps[i - 1].action.absX, steps[i - 1].action.absY) : (steps[i].action.absX, steps[i].action.absY)
                frame.withUnsafeMutableBytes { FrameResizer.drawCursorMarker($0, size: reader.width, at: CGPoint(x: c.0, y: c.1)) }
            }
            let r = stepper.step(frame: MLXArray(frame, [reader.height, reader.width, 3], dtype: .uint8), text: nil, previousAction: prev)
            prev = def.actions.encode(steps[i].action)
            guard i >= from else { continue }
            let m = def.actions.decode(r.classes) { lo, hi, _ in (lo + hi) / 2 }
            human.append(steps[i].action)
            predicted.append(m)
            let allowed = Set(def.actions.keys)
            if m.keys == steps[i].action.keys.intersection(allowed) { agree += 1 }
        }
        return Result(start: start, rateHz: Int(rate), human: human, model: predicted, keyAgreement: Double(agree) / Double(max(predicted.count, 1)))
    }
}
