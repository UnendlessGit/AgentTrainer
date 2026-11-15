import ATCore
import AVFoundation
import CoreImage
import Foundation
import Vision

/// One piece of text found on screen. Box is normalized with a top-left origin.
public struct ScreenTextItem: Codable, Hashable, Sendable {
    public var text: String
    public var x: Float, y: Float, w: Float, h: Float
    public var confidence: Float
}

public struct ScreenTextSample: Codable, Hashable, Sendable {
    public var time: Double
    public var items: [ScreenTextItem]
}

/// On-device OCR (Apple Vision) with fixed settings, used identically for training data and
/// live runs so the model sees the same kind of text either way.
public enum ScreenTextReader {
    public static let file = "screen-text.json"
    public static let interval = 0.5

    public static func read(_ image: CGImage, maxItems: Int = 16) -> [ScreenTextItem] {
        let req = VNRecognizeTextRequest()
        req.recognitionLevel = .accurate
        req.usesLanguageCorrection = false
        req.minimumTextHeight = 0.012
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try? handler.perform([req])
        return rank(req.results ?? [], maxItems: maxItems)
    }

    public static func read(_ pb: CVPixelBuffer, maxItems: Int = 16) -> [ScreenTextItem] {
        let req = VNRecognizeTextRequest()
        req.recognitionLevel = .accurate
        req.usesLanguageCorrection = false
        req.minimumTextHeight = 0.012
        let handler = VNImageRequestHandler(cvPixelBuffer: pb, options: [:])
        try? handler.perform([req])
        return rank(req.results ?? [], maxItems: maxItems)
    }

    static func rank(_ obs: [VNRecognizedTextObservation], maxItems: Int) -> [ScreenTextItem] {
        let items: [ScreenTextItem] = obs.compactMap { o in
            guard let c = o.topCandidates(1).first, c.string.trimmingCharacters(in: .whitespaces).count >= 2 else { return nil }
            let b = o.boundingBox
            return ScreenTextItem(text: String(c.string.prefix(64)), x: Float(b.minX), y: Float(1 - b.maxY), w: Float(b.width), h: Float(b.height),
                                  confidence: c.confidence)
        }
        // Larger, more confident text first (titles, buttons, prompts).
        return Array(items.sorted { $0.w * $0.h * $0.confidence > $1.w * $1.h * $1.confidence }.prefix(maxItems))
    }

    /// Reads (and caches in the package) the on-screen text of a recording every `interval` seconds.
    public static func index(recording: URL) throws -> [ScreenTextSample] {
        let cache = recording.appendingPathComponent(file)
        if let d = try? Data(contentsOf: cache), let s = try? JSONDecoder().decode([ScreenTextSample].self, from: d) { return s }
        let asset = AVURLAsset(url: recording.appendingPathComponent(RecordingPackage.video))
        guard let track = asset.tracks(withMediaType: .video).first else { return [] }
        let reader = try AVAssetReader(asset: asset)
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        out.alwaysCopiesSampleData = false
        reader.add(out)
        reader.startReading()
        var samples: [ScreenTextSample] = []
        var next = 0.0
        while let sb = out.copyNextSampleBuffer() {
            let t = sb.presentationTimeStamp.seconds
            guard t >= next, let pb = sb.imageBuffer else { continue }
            samples.append(ScreenTextSample(time: t, items: read(pb)))
            next = t + interval
        }
        try? JSONEncoder().encode(samples).write(to: cache)
        return samples
    }
}
