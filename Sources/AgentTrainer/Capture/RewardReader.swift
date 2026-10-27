import AppKit
import CoreVideo
import Vision

struct RewardPreview: @unchecked Sendable {
    var image: CGImage
    var reading: RLRewardReading?
    var status: String
    var framesPerSecond: Double
}

protocol RewardReading: AnyObject, Sendable {
    var onPreview: (@Sendable (RewardPreview) -> Void)? { get set }
    var onReading: (@Sendable (RLRewardReading) -> Void)? { get set }
    var onFailure: (@Sendable (String) -> Void)? { get set }
    func start(source: CaptureSourceOption, region: RLRewardRegion, fps: Double, confidence: Double) async throws
    func stop() async
}

/// OCR never runs on the capture, inference, action, or main queues. Coalescing
/// bounds latency; cumulative accounting retains the skipped source-time span.
final class RewardReader: RewardReading, @unchecked Sendable {
    var onPreview: (@Sendable (RewardPreview) -> Void)?
    var onReading: (@Sendable (RLRewardReading) -> Void)?
    var onFailure: (@Sendable (String) -> Void)?
    private let capture = CaptureService()
    private let queue = DispatchQueue(label: "AgentTrainer.RewardOCR", qos: .userInitiated)
    private let lock = NSLock()
    private struct Frame: @unchecked Sendable { var image: CGImage; var data: Data; var time: UInt64 }
    private var newest: Frame?
    private var previousFrame: Frame?
    private var processing = false
    private var stopped = true
    private var generation: UInt64 = 0
    private var minimumConfidence: Float = 0.8
    private var cachedBytes: Data?
    private var cachedConfidence: Float = 0
    private var cachedPair: (Double, Double)?
    private var numericRects: [CGRect] = []
    private var lastCalibration: UInt64 = 0
    private var readings = 0
    private var startedAt = DispatchTime.now().uptimeNanoseconds

    func start(source: CaptureSourceOption, region: RLRewardRegion, fps: Double, confidence: Double) async throws {
        guard region.matches(source), region.isValid else { throw AgentTrainerError.capture("Select a reward region on the current capture source.") }
        let rect = region.rect(in: source.frame)
        var spec = CaptureSpec(kind: region.isWindow ? .windowRegion : .screenRegion, requestedFPS: fps, showsCursor: false)
        spec.displayID = region.isWindow ? nil : source.id; spec.windowID = region.isWindow ? source.id : nil
        spec.region = CodableRect(region.isWindow ? rect.offsetBy(dx: -source.frame.minX, dy: -source.frame.minY) : rect)
        await withCheckedContinuation { continuation in queue.async { [self] in
            cachedBytes = nil; cachedPair = nil; cachedConfidence = 0; numericRects = []
            readings = 0; lastCalibration = 0; startedAt = DispatchTime.now().uptimeNanoseconds
            continuation.resume()
        } }
        lock.withLock { stopped = false; generation &+= 1; minimumConfidence = Float(confidence) }
        try await capture.start(spec: spec, queueDepth: 3, nativeWindowGeometry: true, onFrame: { [weak self] buffer, pts in
            let time = CMTimeConvertScale(pts, timescale: 1_000_000_000, method: .default)
            self?.receive(buffer, timestamp: time.value > 0 ? UInt64(time.value) : DispatchTime.now().uptimeNanoseconds)
        }, onIdle: { [weak self] pts in
            let time = CMTimeConvertScale(pts, timescale: 1_000_000_000, method: .default)
            self?.idle(timestamp: time.value > 0 ? UInt64(time.value) : DispatchTime.now().uptimeNanoseconds)
        }, onUnexpectedStop: { [weak self] error in self?.onFailure?(error.localizedDescription) })
    }

    func stop() async {
        lock.withLock { stopped = true; generation &+= 1; newest = nil; previousFrame = nil }
        _ = try? await capture.stop()
        await withCheckedContinuation { continuation in queue.async { continuation.resume() } }
    }

    private func receive(_ buffer: CVPixelBuffer, timestamp: UInt64) {
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        guard width > 0, height > 0, width * height <= 4_194_304 else {
            onFailure?("The reward region is too large. Select a small region containing just the two text values."); return
        }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let pointer = CVPixelBufferGetBaseAddress(buffer) else { return }
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let bytes = Data(bytes: pointer, count: stride * height)
        guard let provider = CGDataProvider(data: bytes as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: stride, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue).union(.byteOrder32Little),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { return }
        submit(Frame(image: image, data: bytes, time: timestamp))
    }

    private func idle(timestamp: UInt64) {
        lock.lock(); var frame = previousFrame; lock.unlock()
        if frame != nil { frame!.time = timestamp; submit(frame!) }
    }

    private func submit(_ frame: Frame) {
        lock.lock()
        guard !stopped else { lock.unlock(); return }
        newest = frame; previousFrame = frame
        guard !processing else { lock.unlock(); return }
        processing = true; let token = generation; lock.unlock()
        queue.async { [weak self] in self?.drain(generation: token) }
    }

    private func drain(generation token: UInt64) {
        while true {
            lock.lock()
            guard !stopped, generation == token, let frame = newest else { processing = false; lock.unlock(); return }
            newest = nil; lock.unlock()
            autoreleasepool {
                let pair: (Double, Double)?
                var confidence: Float = 1
                do {
                    if frame.data == cachedBytes, let cachedPair { pair = cachedPair; confidence = cachedConfidence }
                    else if numericRects.count == 2, frame.time - lastCalibration < 1_000_000_000,
                            let fast = try numericPair(frame.image) { pair = fast.0; confidence = fast.1 }
                    else {
                        let recognized = try Self.accurateFields(in: frame.image, minimumConfidence: minimumConfidence)
                        pair = RLTextParser.pair(recognized.map(\.0))
                        confidence = recognized.map(\.1).min() ?? 0
                        numericRects = pair == nil ? [] : rectangles(recognized, image: frame.image)
                        lastCalibration = frame.time
                    }
                } catch { pair = nil }
                let valid = confidence >= minimumConfidence ? pair : nil
                let reading = valid.map { RLRewardReading(reward: $0.0, episodeLength: $0.1, capturedAt: frame.time,
                    recognizedAt: DispatchTime.now().uptimeNanoseconds, confidence: confidence) }
                if let valid { cachedConfidence = confidence; cachedPair = valid; cachedBytes = frame.data; readings += 1 }
                else { cachedBytes = nil; cachedPair = nil; numericRects = [] }
                lock.lock(); let current = !stopped && generation == token; lock.unlock()
                guard current else { return }
                if let reading { onReading?(reading) }
                onPreview?(RewardPreview(image: frame.image, reading: reading,
                    status: reading == nil ? "Cannot read both values reliably" : "Reading reward and episode length",
                    framesPerSecond: Double(readings) * 1e9 / Double(max(1, DispatchTime.now().uptimeNanoseconds - startedAt))))
            }
        }
    }

    private static func recognize(_ image: CGImage, accurate: Bool) throws -> [(String, Float, VNRecognizedText)] {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = accurate ? .accurate : .fast
        request.recognitionLanguages = ["en-US"]; request.usesLanguageCorrection = false
        try VNImageRequestHandler(cgImage: image).perform([request])
        return (request.results ?? []).compactMap { observation in
            guard let text = observation.topCandidates(1).first else { return nil }
            return (text.string, text.confidence, text)
        }
    }

    /// Vision can omit an isolated integer zero at Retina text sizes even
    /// though it reads the same pixels correctly at a smaller scale. Retry
    /// accurate OCR on the complete pair; never substitute guessed characters
    /// or lower the configured confidence requirement.
    static func accurateFields(in image: CGImage, minimumConfidence: Float) throws -> [(String, Float, VNRecognizedText)] {
        guard minimumConfidence.isFinite, (0...1).contains(minimumConfidence) else { return [] }
        let native = try recognize(image, accurate: true)
        func valid(_ fields: [(String, Float, VNRecognizedText)]) -> Bool {
            RLTextParser.pair(fields.map(\.0)) != nil && (fields.map(\.1).min() ?? 0) >= minimumConfidence
        }
        if valid(native) { return native }
        let text = native.map(\.0).joined(separator: "\n")
        for label in [#"(?i)\breward\s*:"#, #"(?i)\bepisode\s*length\s*:"#] {
            let regex = try NSRegularExpression(pattern: label)
            if regex.numberOfMatches(in: text, range: NSRange(text.startIndex..., in: text)) > 1 { return native }
        }
        for scale in [0.75, 0.5] {
            let width = max(1, Int(Double(image.width) * scale)), height = max(1, Int(Double(image.height) * scale))
            guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { continue }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            guard let smaller = context.makeImage() else { continue }
            let fields = try recognize(smaller, accurate: true)
            if valid(fields) { return fields }
        }
        return native
    }

    static func validatedPair(in image: CGImage, minimumConfidence: Float) throws -> (Double, Double)? {
        let fields = try accurateFields(in: image, minimumConfidence: minimumConfidence)
        guard (fields.map(\.1).min() ?? 0) >= minimumConfidence else { return nil }
        return RLTextParser.pair(fields.map(\.0))
    }

    private func rectangles(_ lines: [(String, Float, VNRecognizedText)], image: CGImage) -> [CGRect] {
        var result: [String: CGRect] = [:]
        guard let pattern = try? NSRegularExpression(pattern: #"(?i)\b(reward|episode\s*length)\s*:\s*([+\-−]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))"#) else { return [] }
        for (line, _, candidate) in lines {
            for match in pattern.matches(in: line, range: NSRange(line.startIndex..., in: line)) {
            guard let labelRange = Range(match.range(at: 1), in: line), let valueRange = Range(match.range(at: 2), in: line) else { continue }
            let label = String(line[labelRange]).lowercased().filter { !$0.isWhitespace }
            guard let observation = try? candidate.boundingBox(for: valueRange) else { continue }
            let r = observation.boundingBox
            let x = max(0, r.minX * Double(image.width) - 6)
            let y = max(0, (1 - r.maxY) * Double(image.height) - 5)
            let bottom = min(Double(image.height), (1 - r.minY) * Double(image.height) + 5)
            result[label] = CGRect(x: x, y: y, width: min(Double(image.width) - x, max(64, r.width * Double(image.width) * 1.5 + 40)), height: bottom - y)
            }
        }
        guard let reward = result["reward"], let length = result["episodelength"] else { return [] }
        return [reward, length]
    }

    private func numericPair(_ image: CGImage) throws -> ((Double, Double), Float)? {
        var values: [Double] = [], confidence: Float = 1
        for rect in numericRects {
            guard let crop = image.cropping(to: rect), let text = try Self.recognize(crop, accurate: false).first,
                  text.1 >= minimumConfidence, let number = RLTextParser.number(text.0.filter { !$0.isWhitespace }) else { return nil }
            let range = text.0.startIndex..<text.0.endIndex
            if let bounds = try? text.2.boundingBox(for: range), bounds.boundingBox.maxX > 0.96 { return nil }
            values.append(number); confidence = min(confidence, text.1)
        }
        guard values.count == 2, values[1] >= 0, values[1] <= 604800 else { return nil }
        return ((values[0], values[1]), confidence)
    }
}
