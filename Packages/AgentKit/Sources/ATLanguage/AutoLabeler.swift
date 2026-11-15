import ATCore
import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXVLM

/// Writes goal-level instructions for recordings with a local vision-language model, the way
/// Pixel2Play annotated its data with Gemini - but entirely on this Mac.
public final class AutoLabeler: @unchecked Sendable {
    let context: ModelContext

    public init(directory: URL) throws {
        context = try blocking { try await VLMModelFactory.shared.load(from: directory, using: TransformersTokenizerLoader()) }
    }

    /// Labels one recording; returns the new automatic segments.
    public func label(recording url: URL, request: AutoLabelRequest, progress: (Double, String) -> Void) throws -> [TextSegment] {
        let meta = try RecordingIO.readMeta(url)
        let frames = try RecordingIO.readFrames(url), events = try RecordingIO.readEvents(url)
        let steps = TimelineBuilder.build(frames: frames, events: events, meta: meta, annotations: RecordingAnnotations(), rateHz: 10)
        let asset = AVURLAsset(url: url.appendingPathComponent(RecordingPackage.video))
        let gen = AVAssetImageGenerator(asset: asset)
        gen.maximumSize = CGSize(width: 512, height: 512)
        gen.requestedTimeToleranceBefore = CMTime(value: 1, timescale: 10)
        gen.requestedTimeToleranceAfter = CMTime(value: 1, timescale: 10)
        gen.appliesPreferredTrackTransform = true

        let W = request.windowSeconds
        var windows: [(start: Double, end: Double, text: String)] = []
        var t = 0.0
        while t < meta.duration - 2 {
            let end = min(meta.duration, t + W)
            progress(t / max(meta.duration, 1), "\(meta.name): \(Int(t))–\(Int(end)) s")
            let times = (0 ..< request.framesPerWindow).map { t + (end - t) * (Double($0) + 0.5) / Double(request.framesPerWindow) }
            var images: [UserInput.Image] = []
            for s in times {
                if let img = try? blocking({ try await gen.image(at: CMTime(seconds: s, preferredTimescale: 600)).image }) {
                    images.append(.ciImage(CIImage(cgImage: img)))
                }
            }
            guard !images.isEmpty else { t = end; continue }
            let summary = Self.inputSummary(steps.filter { $0.time >= t && $0.time < end })
            let prompt = Self.prompt(seconds: end - t, task: meta.task, inputs: summary, frames: images.count)
            let text = try generate(prompt: prompt, images: images)
            if let clean = Self.clean(text) { windows.append((t, end, clean)) }
            t = end
        }
        // Merge neighbouring windows that describe the same goal.
        var segments: [TextSegment] = []
        for w in windows {
            if let last = segments.last, last.text.lowercased() == w.text.lowercased(), abs(last.end - w.start) < 0.01 {
                segments[segments.count - 1].end = w.end
            } else {
                segments.append(TextSegment(start: w.start, end: w.end, text: w.text, source: .auto))
            }
        }
        return segments
    }

    func generate(prompt: String, images: [UserInput.Image]) throws -> String {
        let ctx = context
        let input = UserInput(prompt: prompt, images: images)
        return try blocking {
            let lm = try await ctx.processor.prepare(input: input)
            let result = try MLXLMCommon.generate(input: lm, parameters: GenerateParameters(maxTokens: 40, temperature: 0.2), context: ctx) { tokens in
                tokens.count >= 40 ? .stop : .more
            }
            return result.output
        }
    }

    static func prompt(seconds: Double, task: String, inputs: String, frames: Int) -> String {
        """
        You are labelling a screen recording of a person playing a game or using an app. The labels teach an AI agent \
        to follow instructions. The \(frames) images are frames from a \(Int(seconds))-second clip, in time order.
        \(task.isEmpty ? "" : "The person described the session as: \"\(task)\".")
        Their input during the clip: \(inputs).
        Write ONE short instruction (3 to 12 words, imperative mood) that states the goal the person is pursuing in this clip, \
        for example "Chop down the tree in front of you" or "Open the settings menu". Describe the goal, not individual key presses. \
        Do not use the words "left" or "right". Reply with the instruction only.
        """
    }

    static func inputSummary(_ steps: [TimelineStep]) -> String {
        guard !steps.isEmpty else { return "none" }
        var keys: [String: Int] = [:], clicks = 0, scrolls = 0
        var moved = 0.0
        for s in steps {
            for k in s.action.keys { keys[k, default: 0] += 1 }
            if s.action.pointer.contains(where: \.isButton) { clicks += 1 }
            if s.action.pointer.contains(where: \.isScroll) { scrolls += 1 }
            moved += abs(s.action.dx) + abs(s.action.dy)
        }
        var parts = keys.sorted { $0.value > $1.value }.prefix(5).map { k, n in
            "\(KeyCatalog.info(k)?.label.uppercased() ?? k) held \(Int(Double(n) / Double(steps.count) * 100))% of the time"
        }
        if clicks > 0 { parts.append("clicking") }
        if scrolls > 0 { parts.append("scrolling") }
        if moved / Double(steps.count) > 3 { parts.append("moving the mouse a lot") }
        return parts.isEmpty ? "almost no input" : parts.joined(separator: ", ")
    }

    static func clean(_ s: String) -> String? {
        var t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if let line = t.split(separator: "\n").first { t = String(line) }
        t = t.trimmingCharacters(in: CharacterSet(charactersIn: "\"'“”. ")).trimmingCharacters(in: .whitespaces)
        guard t.count >= 3, t.split(separator: " ").count <= 16 else { return nil }
        return t.prefix(1).uppercased() + t.dropFirst()
    }
}
