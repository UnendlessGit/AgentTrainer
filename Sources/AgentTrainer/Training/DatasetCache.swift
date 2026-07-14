@preconcurrency import AVFoundation
import CoreGraphics
import CryptoKit
import Foundation

enum ActionLayout {
    static let absoluteMouse = 0..<2
    static let relativeMouse = 2..<4
    static let buttons = 4..<12
    static let scroll = 12..<14
    static let keyboard = 14..<142
    static let modifiers = 142..<146
    static let count = 146
}

struct CacheSegment: Codable, Hashable, Sendable {
    var recordingID: UUID
    var start: Int
    var count: Int
}

struct DatasetCacheManifest: Codable, Hashable, Sendable {
    var schemaVersion = TrainingDataContract.schemaVersion
    var key: String
    var createdAt: Date
    var preprocessing: PreprocessingSpec
    var actionFPS: Double
    var perceptionFPS: Double
    var historyLength: Int
    var sampleCount: Int
    var observationBytesPerSample: Int
    var actionValuesPerSample: Int
    var segments: [CacheSegment]
}

final class CachedDataset: @unchecked Sendable {
    let manifest: DatasetCacheManifest
    private let observations: Data
    private let actions: Data

    init(directory: URL) throws {
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601
        manifest = try decoder.decode(DatasetCacheManifest.self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
        guard manifest.schemaVersion == TrainingDataContract.schemaVersion else {
            throw AgentTrainerError.storage("This dataset cache uses an obsolete input contract and must be rebuilt.")
        }
        observations = try Data(contentsOf: directory.appendingPathComponent("observations.bin"), options: .mappedIfSafe)
        actions = try Data(contentsOf: directory.appendingPathComponent("actions.bin"), options: .mappedIfSafe)
        guard observations.count == manifest.sampleCount * manifest.observationBytesPerSample,
              actions.count == manifest.sampleCount * manifest.actionValuesPerSample * MemoryLayout<Float>.size else {
            throw AgentTrainerError.storage("The dataset cache is incomplete or corrupt.")
        }
    }

    var count: Int { manifest.sampleCount }

    /// Derives the exact keyboard capability from the cached training targets.
    /// This works for both new and already-existing caches without another pass
    /// over the source recordings.
    func demonstratedKeyCodes() -> Set<UInt16> {
        var result: Set<UInt16> = []
        actions.withUnsafeBytes { raw in
            guard let address = raw.baseAddress else { return }
            let values = address.assumingMemoryBound(to: UInt32.self)
            for row in 0..<manifest.sampleCount {
                let base = row * manifest.actionValuesPerSample
                for key in 0..<128 where Float(bitPattern: UInt32(littleEndian: values[base + 14 + key])) >= 0.5 {
                    result.insert(UInt16(key))
                }
                let modifierKeys: [UInt16] = [56, 59, 58, 55]
                for modifier in 0..<4 where Float(bitPattern: UInt32(littleEndian: values[base + 142 + modifier])) >= 0.5 {
                    result.insert(modifierKeys[modifier])
                }
            }
        }
        return result
    }

    func packedObservation(at index: Int) -> Data {
        let size = manifest.observationBytesPerSample
        return observations.subdata(in: index * size..<(index + 1) * size)
    }

    func packedObservations(at indices: [Int]) -> Data {
        let size = manifest.observationBytesPerSample
        var result = Data(count: indices.count * size)
        result.withUnsafeMutableBytes { destination in
            observations.withUnsafeBytes { source in
                guard let destinationBase = destination.baseAddress, let sourceBase = source.baseAddress else { return }
                for (row, index) in indices.enumerated() {
                    memcpy(destinationBase.advanced(by: row * size), sourceBase.advanced(by: index * size), size)
                }
            }
        }
        return result
    }

    func action(at index: Int) -> [Float] {
        let count = manifest.actionValuesPerSample
        let offset = index * count * MemoryLayout<Float>.size
        return actions.withUnsafeBytes { raw in
            guard let address = raw.baseAddress else { return [Float](repeating: 0, count: count) }
            let base = address.advanced(by: offset).assumingMemoryBound(to: UInt32.self)
            return (0..<count).map { Float(bitPattern: UInt32(littleEndian: base[$0])) }
        }
    }

    func actionBatch(at indices: [Int]) -> Data {
        let rowBytes = manifest.actionValuesPerSample * MemoryLayout<Float>.size
        var result = Data(count: indices.count * rowBytes)
        result.withUnsafeMutableBytes { destination in
            actions.withUnsafeBytes { source in
                guard let destinationBase = destination.baseAddress, let sourceBase = source.baseAddress else { return }
                for (row, index) in indices.enumerated() {
                    memcpy(destinationBase.advanced(by: row * rowBytes), sourceBase.advanced(by: index * rowBytes), rowBytes)
                }
            }
        }
        return result
    }

    func historyBatch(at indices: [Int]) -> Data {
        let historyLength = max(1, manifest.historyLength)
        let rowBytes = manifest.actionValuesPerSample * MemoryLayout<Float>.size
        var result = Data(count: indices.count * historyLength * rowBytes)
        guard manifest.historyLength > 0 else { return result }
        result.withUnsafeMutableBytes { destination in
            actions.withUnsafeBytes { source in
                guard let destinationBase = destination.baseAddress, let sourceBase = source.baseAddress else { return }
                for (batchRow, index) in indices.enumerated() {
                    let segmentStart = segmentStart(for: index)
                    for historyRow in 0..<manifest.historyLength {
                        let sourceIndex = index - manifest.historyLength + historyRow
                        guard sourceIndex >= segmentStart, sourceIndex < index else { continue }
                        let destinationOffset = (batchRow * historyLength + historyRow) * rowBytes
                        memcpy(destinationBase.advanced(by: destinationOffset), sourceBase.advanced(by: sourceIndex * rowBytes), rowBytes)
                    }
                }
            }
        }
        return result
    }

    func history(at index: Int) -> [Float] {
        let segmentStart = segmentStart(for: index)
        var values = [Float](repeating: 0, count: max(1, manifest.historyLength) * ActionLayout.count)
        guard manifest.historyLength > 0 else { return values }
        for h in 0..<manifest.historyLength {
            let source = index - manifest.historyLength + h
            guard source >= segmentStart, source < index else { continue }
            values.replaceSubrange(h * ActionLayout.count..<(h + 1) * ActionLayout.count, with: action(at: source))
        }
        return values
    }

    private func segmentStart(for index: Int) -> Int {
        var low = 0, high = manifest.segments.count
        while low < high {
            let mid = (low + high) / 2
            if manifest.segments[mid].start <= index { low = mid + 1 } else { high = mid }
        }
        return low > 0 ? manifest.segments[low - 1].start : 0
    }
}

actor DatasetCacheBuilder {
    static let shared = DatasetCacheBuilder()
    private var preprocessor: VisionPreprocessor?

    init() {}

    func cache(for profile: AIProfile, recordings: [RecordingItem], progress: @escaping @Sendable (Double, String) -> Void) async throws -> CachedDataset {
        guard !recordings.isEmpty else { throw AgentTrainerError.noData }
        if preprocessor == nil { preprocessor = try VisionPreprocessor() }
        try await WorkspaceStore.shared.prepare()
        let root = await WorkspaceStore.shared.cacheDirectory()
        let key = try cacheKey(profile: profile, recordings: recordings)
        let directory = root.appendingPathComponent("\(key).atrcache", isDirectory: true)
        if FileManager.default.fileExists(atPath: directory.appendingPathComponent("manifest.json").path), let cached = try? CachedDataset(directory: directory) {
            progress(1, "Reusing packed dataset cache")
            return cached
        }

        let temporary = root.appendingPathComponent(".\(key).\(UUID().uuidString).tmp", isDirectory: true)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: temporary.appendingPathComponent("observations.bin").path, contents: nil)
        FileManager.default.createFile(atPath: temporary.appendingPathComponent("actions.bin").path, contents: nil)
        let observations = try FileHandle(forWritingTo: temporary.appendingPathComponent("observations.bin"))
        let actions = try FileHandle(forWritingTo: temporary.appendingPathComponent("actions.bin"))
        var segments: [CacheSegment] = []
        var sampleCount = 0
        do {
            for (recordingIndex, recording) in recordings.enumerated() {
                try Task.checkCancellation()
                progress(Double(recordingIndex) / Double(recordings.count), "Packing \(recording.manifest.name)")
                let start = sampleCount
                sampleCount += try await appendRecording(recording, profile: profile, observations: observations, actions: actions)
                segments.append(CacheSegment(recordingID: recording.id, start: start, count: sampleCount - start))
            }
            try observations.close(); try actions.close()
            let manifest = DatasetCacheManifest(key: key, createdAt: Date(), preprocessing: profile.preprocessing, actionFPS: profile.training.actionFPS, perceptionFPS: profile.training.perceptionFPS, historyLength: profile.training.historyLength, sampleCount: sampleCount, observationBytesPerSample: profile.preprocessing.sampleByteCount, actionValuesPerSample: ActionLayout.count, segments: segments)
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]; encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(manifest).write(to: temporary.appendingPathComponent("manifest.json"), options: .atomic)
            if FileManager.default.fileExists(atPath: directory.path) { try FileManager.default.removeItem(at: directory) }
            try FileManager.default.moveItem(at: temporary, to: directory)
            progress(1, "Dataset cache ready")
            return try CachedDataset(directory: directory)
        } catch {
            try? observations.close(); try? actions.close(); try? FileManager.default.removeItem(at: temporary)
            throw error
        }
    }

    private func cacheKey(profile: AIProfile, recordings: [RecordingItem]) throws -> String {
        struct Identity: Encodable {
            let cacheSchema: Int
            let preprocessing: PreprocessingSpec
            let actionFPS: Double
            let perceptionFPS: Double
            let historyLength: Int
            let recordings: [RecordingManifest]
        }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]; encoder.dateEncodingStrategy = .iso8601
        let identity = Identity(cacheSchema: TrainingDataContract.schemaVersion, preprocessing: profile.preprocessing, actionFPS: profile.training.actionFPS, perceptionFPS: profile.training.perceptionFPS, historyLength: profile.training.historyLength, recordings: recordings.map(\.manifest))
        let digest = SHA256.hash(data: try encoder.encode(identity))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private func appendRecording(_ recording: RecordingItem, profile: AIProfile, observations: FileHandle, actions: FileHandle) async throws -> Int {
        guard let preprocessor else { throw AgentTrainerError.model("Metal preprocessing is unavailable.") }
        let events = try InputEventReader.read(url: recording.directory.appendingPathComponent(recording.manifest.eventFile))
        let asset = AVURLAsset(url: recording.directory.appendingPathComponent(recording.manifest.videoFile))
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { return 0 }
        let reader = try AVAssetReader(asset: asset)
        let trainingStart = max(0, min(recording.manifest.duration, recording.manifest.trimStart))
        let trainingEnd = max(trainingStart, min(recording.manifest.duration, recording.manifest.trimEnd ?? recording.manifest.duration))
        guard trainingEnd > trainingStart else { return 0 }
        reader.timeRange = CMTimeRange(start: CMTime(seconds: trainingStart, preferredTimescale: 1_000_000_000), duration: CMTime(seconds: trainingEnd - trainingStart, preferredTimescale: 1_000_000_000))
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw AgentTrainerError.storage("The recording video cannot be decoded for training.") }
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? AgentTrainerError.storage("The recording video could not be opened.") }

        let actionInterval = 1 / max(0.0001, profile.training.actionFPS)
        let perceptionInterval = 1 / max(0.0001, profile.training.perceptionFPS)
        var nextAction = trainingStart
        var nextPerception = trainingStart
        var firstPTS: CMTime?
        var latestPacked: Data?
        var eventIndex = 0
        var accumulator = ActionAccumulator(manifest: recording.manifest, events: events)
        var count = 0

        // Establish the held-control and pointer state at the trim boundary,
        // but discard movement/scroll that happened before the usable range.
        // Without this priming, the first target can contain the entire trimmed
        // lead-in as one large, directionally biased mouse action.
        let trainingStartNanos = recording.manifest.hostStartNanos + UInt64(max(0, trainingStart) * 1_000_000_000)
        while eventIndex < events.count, events[eventIndex].timestampNanos <= trainingStartNanos {
            accumulator.consume(events[eventIndex])
            eventIndex += 1
        }
        accumulator.endTick()

        while let sample = output.copyNextSampleBuffer(), let buffer = sample.imageBuffer {
            try Task.checkCancellation()
            let pts = sample.presentationTimeStamp
            if firstPTS == nil { firstPTS = pts }
            guard let firstPTS else { continue }
            let t = trainingStart + max(0, CMTimeGetSeconds(pts - firstPTS))
            if let previous = latestPacked {
                while nextAction < t - 0.000_001 && nextAction <= trainingEnd {
                    try writeTick(packed: previous, time: nextAction, actionInterval: actionInterval, trainingEnd: trainingEnd, recording: recording, events: events, eventIndex: &eventIndex, accumulator: &accumulator, observations: observations, actions: actions)
                    count += 1; nextAction += actionInterval
                }
            }
            if latestPacked == nil || t + 0.000_001 >= nextPerception {
                latestPacked = try preprocessor.process(buffer, spec: profile.preprocessing)
                while nextPerception <= t { nextPerception += perceptionInterval }
            }
            if let latestPacked {
                while nextAction <= t + 0.000_001 && nextAction <= trainingEnd {
                    try writeTick(packed: latestPacked, time: nextAction, actionInterval: actionInterval, trainingEnd: trainingEnd, recording: recording, events: events, eventIndex: &eventIndex, accumulator: &accumulator, observations: observations, actions: actions)
                    count += 1; nextAction += actionInterval
                }
            }
        }
        if reader.status == .failed { throw reader.error ?? AgentTrainerError.storage("Video decoding failed while building the cache.") }
        if let latestPacked {
            while nextAction <= trainingEnd {
                try writeTick(packed: latestPacked, time: nextAction, actionInterval: actionInterval, trainingEnd: trainingEnd, recording: recording, events: events, eventIndex: &eventIndex, accumulator: &accumulator, observations: observations, actions: actions)
                count += 1; nextAction += actionInterval
            }
        }
        return count
    }

    private func writeTick(packed: Data, time: Double, actionInterval: Double, trainingEnd: Double, recording: RecordingItem, events: [InputSample], eventIndex: inout Int, accumulator: inout ActionAccumulator, observations: FileHandle, actions: FileHandle) throws {
        // Pair the frame at `time` with the controls demonstrated immediately
        // after it. This causal interval is what live inference must predict.
        let targetEnd = min(trainingEnd, time + actionInterval)
        let absoluteNanos = recording.manifest.hostStartNanos + UInt64(max(0, targetEnd) * 1_000_000_000)
        while eventIndex < events.count, events[eventIndex].timestampNanos < absoluteNanos { accumulator.consume(events[eventIndex]); eventIndex += 1 }
        try observations.write(contentsOf: packed)
        try actions.write(contentsOf: accumulator.actionData())
        accumulator.endTick()
    }
}

struct ActionAccumulator {
    let manifest: RecordingManifest
    var x = 0.0, y = 0.0, dx = 0.0, dy = 0.0, sx = 0.0, sy = 0.0
    var buttons: Set<UInt8> = []
    var keys: Set<UInt16> = []
    var flags: UInt64 = 0

    init(manifest: RecordingManifest, events: [InputSample] = []) {
        self.manifest = manifest
        if let pointer = events.first(where: { $0.kind == .mouseMove || $0.kind == .mouseButton || $0.kind == .scroll }) {
            x = pointer.x
            y = pointer.y
        } else {
            let rect = manifest.globalRect.cgRect
            x = rect.midX
            y = rect.midY
        }
    }

    mutating func consume(_ event: InputSample) {
        flags = event.modifiers
        switch event.kind {
        case .mouseMove: x = event.x; y = event.y; dx += event.deltaX; dy += event.deltaY
        case .mouseButton:
            x = event.x; y = event.y
            if event.isDown { buttons.insert(event.button) } else { buttons.remove(event.button) }
        case .scroll: x = event.x; y = event.y; sx += event.scrollX; sy += event.scrollY
        case .key: if event.isDown { keys.insert(event.keyCode) } else { keys.remove(event.keyCode) }
        case .flags: break
        }
    }

    mutating func endTick() { dx = 0; dy = 0; sx = 0; sy = 0 }

    func actionData() -> Data {
        var values = [Float](repeating: 0, count: ActionLayout.count)
        let rect = manifest.globalRect.cgRect
        values[0] = Float((x - rect.minX) / max(1, rect.width)).clamped01
        values[1] = Float((y - rect.minY) / max(1, rect.height)).clamped01
        values[2] = GameCameraContract.trainingValue(forRawDelta: dx)
        values[3] = GameCameraContract.trainingValue(forRawDelta: dy)
        for button in buttons where button < 8 { values[4 + Int(button)] = 1 }
        values[12] = Float(sx / 20).clamped(-1, 1)
        values[13] = Float(sy / 20).clamped(-1, 1)
        for key in keys where key < 128 { values[14 + Int(key)] = 1 }
        let masks: [UInt64] = [CGEventFlags.maskShift.rawValue, CGEventFlags.maskControl.rawValue, CGEventFlags.maskAlternate.rawValue, CGEventFlags.maskCommand.rawValue]
        for i in 0..<4 { values[142 + i] = flags & masks[i] != 0 ? 1 : 0 }
        // AgentTrainer is arm64-only, whose native Float representation is the
        // little-endian cache contract. Copy all 146 values in one operation
        // instead of growing Data through 146 tiny appends per action sample.
        return values.withUnsafeBytes { Data($0) }
    }
}

private extension Float {
    var clamped01: Float { clamped(0, 1) }
    func clamped(_ lower: Float, _ upper: Float) -> Float { min(upper, max(lower, self)) }
}
