import ATCore
import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXVLM

/// Writes goal-level instructions for recordings with a local vision-language model, the way
/// Pixel2Play annotated its data with Gemini, but entirely on this Mac.
///
/// For each clip the model watches the footage as video (with the motion between frames), reads an
/// exact log of what was pressed, sees the instructions it already wrote before that clip, and
/// follows what you told it about the footage. It writes what it sees first, then the instruction
/// and its variations; a second look can check the instruction against the clip.
public final class AutoLabeler: @unchecked Sendable {
    let context: ModelContext
    public let directory: URL
    static let debug = ProcessInfo.processInfo.environment["AT_LABEL_DEBUG"] != nil

    public init(directory: URL) throws {
        self.directory = directory
        context = try blocking { try await VLMModelFactory.shared.load(from: directory, using: TransformersTokenizerLoader()) }
    }

    /// How a label run is going, for progress and the estimate of time left.
    public struct Progress: Sendable {
        public var fraction: Double
        public var status: String
    }

    /// Labels one recording. `onSegment` receives each instruction as soon as it's settled (a clip
    /// that continues the previous goal extends that instruction instead, and is reported again with
    /// the same id). Returns all instructions written.
    @discardableResult
    public func label(recording url: URL, request: AutoLabelRequest,
                      shouldStop: @escaping () -> Bool = { false },
                      progress: (Progress) -> Void = { _ in },
                      onSegment: (TextSegment) -> Void = { _ in }) throws -> [TextSegment] {
        let meta = try RecordingIO.readMeta(url)
        let frames = try RecordingIO.readFrames(url), events = try RecordingIO.readEvents(url)
        let ann = RecordingIO.readAnnotations(url)
        let steps = TimelineBuilder.build(frames: frames, events: events, meta: meta, annotations: RecordingAnnotations(), rateHz: Int(LabelPlanner.rate))
        let windows = LabelPlanner.windows(steps: steps, duration: meta.duration, target: request.detail.seconds, skip: ann.excluded)
        let guidance = LabelPrompts.guidance(from: request.conversation)
        let asset = AVURLAsset(url: url.appendingPathComponent(RecordingPackage.video))
        let app = meta.target.appName

        // The person's own instructions stay; the labeller still sees them as context.
        var written: [TextSegment] = []
        let total = windows.reduce(0) { $0 + $1.duration }
        var done = 0.0
        for (n, w) in windows.enumerated() {
            if shouldStop() { break }
            progress(Progress(fraction: done / max(total, 0.01), status: "\(meta.name) · clip \(n + 1) of \(windows.count)"))
            let clipSteps = steps.filter { $0.time >= w.start && $0.time < w.end }
            let log = LabelPlanner.inputLog(clipSteps, from: w.start, aiControlled: ann.modelControlled)
            let video = try clip(asset, w, fps: request.framesPerSecond)
            guard !video.isEmpty else { done += w.duration; continue }
            let previous = Array((written + ann.segments.filter { $0.source == .user && $0.end <= w.start + 0.5 })
                .sorted { $0.start < $1.start }.suffix(4))
            let prompt = LabelPrompts.labelPrompt(clip: w, recordingName: meta.name, app: app, task: meta.task, inputLog: log,
                                                  previous: previous, guidance: guidance, variations: request.variations)
            // One conversation per clip: the double-check is a follow-up in the same conversation, so
            // the model keeps the clip in memory instead of reading the video again.
            let session = makeSession(system: LabelPrompts.system, maxPixels: request.maxPixels, fps: request.framesPerSecond)
            var answer: LabelAnswer?
            let text = try ask(session, prompt, videos: [.frames(video)], maxTokens: 220 + request.variations * 28, temperature: 0.15, shouldStop: shouldStop)
            answer = LabelPrompts.parseAnswer(text, variations: request.variations)
            if answer == nil, !shouldStop() {
                // An answer that doesn't parse gets one reminder.
                let retry = try ask(session, "That wasn't valid JSON in the requested shape. Reply again with only the JSON object.",
                                    maxTokens: 220 + request.variations * 28, temperature: 0.3, shouldStop: shouldStop)
                answer = LabelPrompts.parseAnswer(retry, variations: request.variations)
            }
            if shouldStop() { break }
            guard var a = answer else { done += w.duration; continue }

            if request.doubleCheck, !shouldStop() {
                let check = try ask(session, LabelPrompts.reviewPrompt(answer: a, variations: request.variations),
                                    maxTokens: 160 + request.variations * 28, temperature: 0.1, shouldStop: shouldStop)
                if let r = LabelPrompts.parseReview(check, variations: request.variations), !r.accurate, let fixed = r.instruction {
                    a.instruction = fixed
                    a.variations = r.variations
                    a.sameAsPrevious = false
                }
            }

            // Carry on the previous instruction when the goal hasn't changed.
            if let last = written.last, abs(last.end - w.start) < 0.05,
               a.sameAsPrevious || LabelPrompts.key(a.instruction) == LabelPrompts.key(last.text) {
                var merged = last
                merged.end = w.end
                var extra = merged.variants ?? []
                for v in [a.instruction] + a.variations where extra.count < request.variations
                    && !merged.wordings.map(LabelPrompts.key).contains(LabelPrompts.key(v)) && !extra.map(LabelPrompts.key).contains(LabelPrompts.key(v)) {
                    extra.append(v)
                }
                merged.variants = extra.isEmpty ? nil : extra
                written[written.count - 1] = merged
                onSegment(merged)
            } else {
                let seg = TextSegment(start: w.start, end: w.end, text: a.instruction, source: .auto,
                                      variants: a.variations.isEmpty ? nil : a.variations, observation: a.observation.isEmpty ? nil : a.observation)
                written.append(seg)
                onSegment(seg)
            }
            done += w.duration
            MLX.GPU.clearCache()
            if Self.debug {
                print(String(format: "      [memory] active %.1f GB, cache %.1f GB, peak %.1f GB", Double(MLX.GPU.activeMemory) / 1e9,
                             Double(MLX.GPU.cacheMemory) / 1e9, Double(MLX.GPU.peakMemory) / 1e9))
            }
        }
        progress(Progress(fraction: 1, status: meta.name))
        return written
    }

    // MARK: Chat

    /// Answers the person about their footage. The model watches an overview of the recordings
    /// (frames spread across each, up to three recordings). `onText` receives the reply as it grows.
    public func chat(_ request: LabelChatRequest, shouldStop: @escaping () -> Bool = { false }, onText: @escaping (String) -> Void) throws -> String {
        var overview: [UserInput.VideoFrame] = []
        let recs = Array(request.recordings.prefix(3))
        let perRecording = max(8, 24 / max(recs.count, 1))
        var offset = 0.0
        var names: [String] = []
        for url in recs {
            guard let meta = try? RecordingIO.readMeta(url) else { continue }
            names.append("“\(meta.name)” (\(Int(meta.duration)) s\(meta.task.isEmpty ? "" : ", described as “\(meta.task)”"))")
            let asset = AVURLAsset(url: url.appendingPathComponent(RecordingPackage.video))
            let times = (0 ..< perRecording).map { meta.duration * (Double($0) + 0.5) / Double(perRecording) }
            for (i, img) in images(asset, at: times, maxPixels: 640 * 360).enumerated() {
                overview.append(.init(image: .ciImage(CIImage(cgImage: img)), timeStamp: CMTime(seconds: offset + Double(i) * 0.5, preferredTimescale: 600)))
            }
            offset += Double(perRecording) * 0.5
        }
        let intro = "The video is an overview of \(names.count == 1 ? "the recording" : "\(names.count) recordings") to label: "
            + names.joined(separator: ", ")
            + (request.recordings.count > recs.count ? ", and \(request.recordings.count - recs.count) more like them" : "")
            + ". Frames are spread across the whole of each, so time jumps between them."

        // The same conversation as last time plus one new message carries on (the model still has
        // the overview in memory). Anything else starts over.
        let history = Array(request.messages.dropLast())
        guard let last = request.messages.last, last.role == .user else { return "" }
        if let s = chatSession, chatRecordings == request.recordings,
           chatHistory.map(\.text) == history.map(\.text), chatHistory.map(\.role) == history.map(\.role) {
            let reply = try ask(s, last.text, maxTokens: 400, temperature: 0.6, shouldStop: shouldStop, onText: onText)
            chatHistory = request.messages + [LabelChatMessage(role: .assistant, text: reply)]
            return reply
        }
        let s = makeSession(system: LabelPrompts.chatSystem, maxPixels: 640 * 360, fps: 2)
        var messages: [Chat.Message] = []
        for (i, m) in request.messages.enumerated() {
            let text = i == 0 ? intro + "\n\n" + m.text : m.text
            if i == 0, m.role == .user, !overview.isEmpty {
                messages.append(.user(text, videos: [.frames(overview)]))
            } else {
                messages.append(m.role == .user ? .user(text) : .assistant(text))
            }
        }
        let reply = try ask(s, messages: messages, maxTokens: 400, temperature: 0.6, shouldStop: shouldStop, onText: onText)
        chatSession = s
        chatRecordings = request.recordings
        chatHistory = request.messages + [LabelChatMessage(role: .assistant, text: reply)]
        return reply
    }

    /// Forgets the chat (the next message starts a new conversation).
    public func resetChat() { chatSession = nil; chatHistory = []; chatRecordings = [] }

    private var chatSession: ChatSession?
    private var chatHistory: [LabelChatMessage] = []
    private var chatRecordings: [URL] = []

    // MARK: Model

    func makeSession(system: String, maxPixels: Int, fps: Double) -> ChatSession {
        var p = UserInput.Processing(maxPixels: maxPixels)
        p.video = .init(targetFramesPerSecond: fps)
        return ChatSession(context, instructions: system,
                           generateParameters: GenerateParameters(maxTokens: 400, temperature: 0.2, topP: 0.9, repetitionPenalty: 1.05),
                           processing: p, additionalContext: ["enable_thinking": false])
    }

    func ask(_ s: ChatSession, _ prompt: String, videos: [UserInput.Video] = [], maxTokens: Int, temperature: Float,
             shouldStop: @escaping () -> Bool, onText: @escaping (String) -> Void = { _ in }) throws -> String {
        try ask(s, messages: [.user(prompt, videos: videos)], maxTokens: maxTokens, temperature: temperature, shouldStop: shouldStop, onText: onText)
    }

    func ask(_ s: ChatSession, messages: [Chat.Message], maxTokens: Int, temperature: Float,
             shouldStop: @escaping () -> Bool, onText: @escaping (String) -> Void = { _ in }) throws -> String {
        s.generateParameters.maxTokens = maxTokens
        s.generateParameters.temperature = temperature
        let box = TextBox()
        let session = UnsafeSendable(s), msgs = UnsafeSendable(messages)
        let stop = UnsafeSendable(shouldStop), emit = UnsafeSendable(onText)
        try blocking {
            for try await g in session.value.streamDetails(to: msgs.value) {
                if case .chunk(let c) = g {
                    box.append(c)
                    emit.value(box.text)
                }
                if case .info(let i) = g, Self.debug {
                    print(String(format: "      [model] prompt %d tokens in %.1f s, wrote %d tokens at %.1f/s",
                                 i.promptTokenCount, i.promptTime, i.generationTokenCount, i.tokensPerSecond))
                }
                if stop.value() { break }
            }
        }
        return box.text
    }

    // MARK: Video

    /// The clip as video frames at `fps`, with their real timestamps.
    func clip(_ asset: AVURLAsset, _ w: LabelWindow, fps: Double) throws -> [UserInput.VideoFrame] {
        let count = max(4, min(48, Int((w.duration * fps).rounded())))
        let times = (0 ..< count).map { w.start + w.duration * (Double($0) + 0.5) / Double(count) }
        return zip(times, images(asset, at: times, maxPixels: 960 * 540)).map { t, img in
            UserInput.VideoFrame(image: .ciImage(CIImage(cgImage: img)), timeStamp: CMTime(seconds: t - w.start, preferredTimescale: 600))
        }
    }

    func images(_ asset: AVURLAsset, at times: [Double], maxPixels: Int) -> [CGImage] {
        let gen = AVAssetImageGenerator(asset: asset)
        let side = (Double(maxPixels) * 16 / 9).squareRoot()
        gen.maximumSize = CGSize(width: side, height: side)
        gen.requestedTimeToleranceBefore = CMTime(value: 1, timescale: 20)
        gen.requestedTimeToleranceAfter = CMTime(value: 1, timescale: 20)
        gen.appliesPreferredTrackTransform = true
        return times.compactMap { t in try? blocking { try await gen.image(at: CMTime(seconds: t, preferredTimescale: 600)).image } }
    }
}

private final class TextBox: @unchecked Sendable {
    private let lock = NSLock()
    private var value = ""
    func append(_ s: String) { lock.lock(); value += s; lock.unlock() }
    var text: String { lock.lock(); defer { lock.unlock() }; return value }
}

private struct UnsafeSendable<T>: @unchecked Sendable {
    let value: T
    init(_ v: T) { value = v }
}
