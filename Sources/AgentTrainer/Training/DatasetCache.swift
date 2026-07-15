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
    static let shift = 142..<143
    static let keyboardAndShift = 14..<143
    static let commandOptionControl = 143..<146
    static let modifiers = 142..<146
    static let count = 146
    static let binary = Array(buttons) + Array(keyboard) + Array(modifiers)
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
    var observationCount: Int
    var observationBytesPerSample: Int
    var actionValuesPerSample: Int
    var segments: [CacheSegment]
}

final class CachedDataset: @unchecked Sendable {
    let manifest: DatasetCacheManifest
    private let observations: Data
    private let observationIndices: Data
    private let actions: Data

    init(directory: URL) throws {
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(DatasetCacheManifest.self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
        guard decoded.schemaVersion == TrainingDataContract.schemaVersion else {
            throw AgentTrainerError.storage("This dataset cache uses an obsolete input contract and must be rebuilt.")
        }
        _ = try decoded.preprocessing.validated()
        guard decoded.sampleCount >= 0,
              decoded.observationCount >= 0,
              decoded.sampleCount == 0 || decoded.observationCount > 0,
              decoded.observationCount <= Int(UInt32.max),
              decoded.historyLength >= 0,
              decoded.actionFPS.isFinite, decoded.actionFPS > 0,
              decoded.perceptionFPS.isFinite, decoded.perceptionFPS > 0,
              decoded.observationBytesPerSample == decoded.preprocessing.sampleByteCount,
              decoded.observationBytesPerSample > 0,
              decoded.actionValuesPerSample == ActionLayout.count else {
            throw AgentTrainerError.storage("The dataset cache manifest is invalid.")
        }

        let observationSize = decoded.observationCount.multipliedReportingOverflow(by: decoded.observationBytesPerSample)
        let mappingValueCount = decoded.sampleCount.multipliedReportingOverflow(by: 2)
        let mappingSize = mappingValueCount.partialValue.multipliedReportingOverflow(by: MemoryLayout<UInt32>.size)
        let actionValueCount = decoded.sampleCount.multipliedReportingOverflow(by: decoded.actionValuesPerSample)
        let actionSize = actionValueCount.partialValue.multipliedReportingOverflow(by: MemoryLayout<Float>.size)
        guard !observationSize.overflow, !mappingValueCount.overflow, !mappingSize.overflow,
              !actionValueCount.overflow, !actionSize.overflow else {
            throw AgentTrainerError.storage("The dataset cache manifest exceeds this Mac's addressable memory.")
        }
        var segmentEnd = 0
        for segment in decoded.segments {
            let end = segmentEnd.addingReportingOverflow(segment.count)
            guard segment.start == segmentEnd, segment.count >= 0, !end.overflow else {
                throw AgentTrainerError.storage("The dataset cache segment index is invalid.")
            }
            segmentEnd = end.partialValue
        }
        guard segmentEnd == decoded.sampleCount else {
            throw AgentTrainerError.storage("The dataset cache segment index is incomplete.")
        }

        let loadedObservations = try Data(contentsOf: directory.appendingPathComponent("observations.bin"), options: .mappedIfSafe)
        let loadedObservationIndices = try Data(contentsOf: directory.appendingPathComponent("observation-indices.bin"), options: .mappedIfSafe)
        let loadedActions = try Data(contentsOf: directory.appendingPathComponent("actions.bin"), options: .mappedIfSafe)
        guard loadedObservations.count == observationSize.partialValue,
              loadedObservationIndices.count == mappingSize.partialValue,
              loadedActions.count == actionSize.partialValue else {
            throw AgentTrainerError.storage("The dataset cache is incomplete or corrupt.")
        }
        let mappingsAreValid = loadedObservationIndices.withUnsafeBytes { raw -> Bool in
            guard decoded.sampleCount == 0 || raw.baseAddress != nil else { return false }
            for sample in 0..<decoded.sampleCount {
                for slot in 0..<2 {
                    let offset = (sample * 2 + slot) * MemoryLayout<UInt32>.size
                    let value = raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian
                    if Int(value) >= decoded.observationCount { return false }
                }
            }
            return true
        }
        guard mappingsAreValid else { throw AgentTrainerError.storage("The dataset cache contains an invalid frame index.") }
        manifest = decoded
        observations = loadedObservations
        observationIndices = loadedObservationIndices
        actions = loadedActions
    }

    var count: Int { manifest.sampleCount }

    /// Derives the exact keyboard capability from the cached training targets.
    /// This works for both new and already-existing caches without another pass
    /// over the source recordings.
    func demonstratedKeyCodes(at indices: [Int]? = nil) -> Set<UInt16> {
        let counts = indices.map { binaryPositiveCounts(at: $0) }
            ?? binaryPositiveCounts(in: 0..<manifest.sampleCount)
        var result: Set<UInt16> = []
        for key in 0..<128 where counts[ActionLayout.keyboard.lowerBound + key] > 0 { result.insert(UInt16(key)) }
        let modifierKeys: [UInt16] = [56, 59, 58, 55]
        for modifier in 0..<4 where counts[ActionLayout.modifiers.lowerBound + modifier] > 0 { result.insert(modifierKeys[modifier]) }
        return result
    }

    func binaryPositiveCounts(in range: Range<Int>) -> [Int] {
        binaryPositiveCounts(at: range)
    }

    func binaryPositiveCounts<S: Sequence>(at indices: S) -> [Int] where S.Element == Int {
        var positiveCounts = [Int](repeating: 0, count: ActionLayout.count)
        actions.withUnsafeBytes { raw in
            guard let address = raw.baseAddress else { return }
            let values = address.assumingMemoryBound(to: UInt32.self)
            for row in indices {
                let base = row * manifest.actionValuesPerSample
                for index in ActionLayout.binary where Float(bitPattern: UInt32(littleEndian: values[base + index])) >= 0.5 {
                    positiveCounts[index] += 1
                }
            }
        }
        return positiveCounts
    }

    func packedObservation(at index: Int) -> Data {
        let size = manifest.observationBytesPerSample
        let observation = observationIndex(at: index, slot: 0)
        return observations.subdata(in: observation * size..<(observation + 1) * size)
    }

    func packedObservations(at indices: [Int]) -> Data {
        let size = manifest.observationBytesPerSample
        var result = Data(count: indices.count * size)
        result.withUnsafeMutableBytes { destination in
            observations.withUnsafeBytes { source in
                guard let destinationBase = destination.baseAddress, let sourceBase = source.baseAddress else { return }
                for (row, index) in indices.enumerated() {
                    let observation = observationIndex(at: index, slot: 0)
                    memcpy(destinationBase.advanced(by: row * size), sourceBase.advanced(by: observation * size), size)
                }
            }
        }
        return result
    }

    /// Returns the exact immediately preceding perception frame for every
    /// action sample. Compact frame indices avoid duplicating large packed
    /// images when Action FPS exceeds Perception FPS and remain exact even when
    /// the two rates are not integer multiples. Segment boundaries point to the
    /// current frame, yielding an intentional zero temporal difference.
    func precedingPackedObservations(at indices: [Int]) -> Data {
        let size = manifest.observationBytesPerSample
        var result = Data(count: indices.count * size)
        result.withUnsafeMutableBytes { destination in
            observations.withUnsafeBytes { source in
                guard let destinationBase = destination.baseAddress, let sourceBase = source.baseAddress else { return }
                for (row, index) in indices.enumerated() {
                    let observation = observationIndex(at: index, slot: 1)
                    memcpy(destinationBase.advanced(by: row * size), sourceBase.advanced(by: observation * size), size)
                }
            }
        }
        return result
    }

    /// Per-output positive weights for class-balanced binary control losses.
    /// A keyboard tensor has 128 mostly-zero values, so unweighted BCE rewards
    /// an inert policy. Weights are derived only from the training split and are
    /// bounded so a handful of noisy samples cannot dominate every batch. The
    /// ceiling remains high enough for brief but intentional controls to matter.
    func positiveClassWeights(at indices: [Int], restrictions: ActionRestrictions) -> [Float] {
        guard !indices.isEmpty else { return [Float](repeating: 1, count: ActionLayout.count) }
        let positiveCounts = binaryPositiveCounts(at: indices)

        var result = [Float](repeating: 1, count: ActionLayout.count)
        for index in ActionLayout.binary {
            let isBlocked: Bool
            switch index {
            case ActionLayout.buttons: isBlocked = restrictions.blockedMouseButtons.contains(UInt8(index - ActionLayout.buttons.lowerBound))
            case ActionLayout.keyboard: isBlocked = restrictions.blockedKeyCodes.contains(UInt16(index - ActionLayout.keyboard.lowerBound))
            case ActionLayout.modifiers: isBlocked = !restrictions.allowsModifier(index - ActionLayout.modifiers.lowerBound)
            default: isBlocked = false
            }
            if isBlocked {
                result[index] = 0
                continue
            }
            let positives = positiveCounts[index]
            if positives == 0 {
                // Keyboard and modifier outputs are protected by the runtime's
                // demonstrated-key firewall, so completely unseen dimensions
                // should not let thousands of easy zero labels dominate the
                // useful controls. Mouse buttons have no equivalent capability
                // firewall and remain trained toward off when unseen.
                if ActionLayout.keyboard.contains(index) || ActionLayout.modifiers.contains(index) { result[index] = 0 }
                continue
            }
            let negatives = max(1, indices.count - positives)
            result[index] = min(1_024, max(1, Float(negatives) / Float(positives)))
        }
        return result
    }

    /// Builds a fixed held-out subset once per run. At least one positive for
    /// every demonstrated binary output comes first so an inert policy cannot
    /// look good on a tiny validation budget. Press/release boundaries and
    /// active delta/scroll examples follow; remaining slots are distributed
    /// evenly across the entire held-out timeline.
    func representativeValidationIndices(from indices: [Int], limit rawLimit: Int) -> [Int] {
        let limit = min(indices.count, max(1, rawLimit))
        guard indices.count > limit else { return indices }
        var transitionRows = [Int?](repeating: nil, count: ActionLayout.count)
        var positiveRows = [Int?](repeating: nil, count: ActionLayout.count)
        let continuousOutputs = Array(ActionLayout.relativeMouse) + Array(ActionLayout.scroll)
        var continuousRows = [Int?](repeating: nil, count: continuousOutputs.count)
        actions.withUnsafeBytes { raw in
            guard let address = raw.baseAddress else { return }
            let values = address.assumingMemoryBound(to: UInt32.self)
            func value(row: Int, output: Int) -> Float {
                Float(bitPattern: UInt32(littleEndian: values[row * manifest.actionValuesPerSample + output]))
            }
            for row in indices {
                let segmentStart = segmentStart(for: row)
                for output in ActionLayout.binary {
                    let current = value(row: row, output: output) >= 0.5
                    if current, positiveRows[output] == nil { positiveRows[output] = row }
                    let previous = row > segmentStart ? value(row: row - 1, output: output) >= 0.5 : current
                    if current != previous, transitionRows[output] == nil { transitionRows[output] = row }
                }
                for (offset, output) in continuousOutputs.enumerated()
                where continuousRows[offset] == nil && abs(value(row: row, output: output)) > 0.0001 {
                    continuousRows[offset] = row
                }
            }
        }

        var selected: Set<Int> = []
        func include(_ rows: [Int?]) {
            for row in rows.compactMap({ $0 }) where selected.count < limit { selected.insert(row) }
        }
        include(positiveRows)
        include(transitionRows)
        include(continuousRows)
        let fillCount = limit - selected.count
        if fillCount > 0 {
            for slot in 0..<fillCount {
                let position = fillCount == 1 ? indices.count / 2 : slot * (indices.count - 1) / (fillCount - 1)
                selected.insert(indices[position])
            }
        }
        if selected.count < limit {
            let stride = max(1, indices.count / (limit - selected.count))
            for position in Swift.stride(from: 0, to: indices.count, by: stride) where selected.count < limit {
                selected.insert(indices[position])
            }
        }
        return selected.sorted()
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

    private func observationIndex(at sample: Int, slot: Int) -> Int {
        observationIndices.withUnsafeBytes { raw in
            let offset = (sample * 2 + slot) * MemoryLayout<UInt32>.size
            return Int(raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian)
        }
    }
}

actor DatasetCacheBuilder {
    static let shared = DatasetCacheBuilder()
    private var preprocessor: VisionPreprocessor?
    private let workspace: WorkspaceStore

    init(workspace: WorkspaceStore = .shared) { self.workspace = workspace }

    func cache(for profile: AIProfile, recordings: [RecordingItem], progress: @escaping @Sendable (Double, String) -> Void) async throws -> CachedDataset {
        guard !recordings.isEmpty else { throw AgentTrainerError.noData }
        if preprocessor == nil { preprocessor = try VisionPreprocessor() }
        try await workspace.prepare()
        let root = await workspace.cacheDirectory()
        let key = try cacheKey(profile: profile, recordings: recordings)
        let directory = root.appendingPathComponent("\(key).atrcache", isDirectory: true)
        if FileManager.default.fileExists(atPath: directory.appendingPathComponent("manifest.json").path), let cached = try? CachedDataset(directory: directory) {
            progress(1, "Reusing packed dataset cache")
            return cached
        }

        let temporary = root.appendingPathComponent(".\(key).\(UUID().uuidString).tmp", isDirectory: true)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        let observations = try BufferedFileWriter(url: temporary.appendingPathComponent("observations.bin"), capacity: 8 * 1_024 * 1_024)
        let observationIndices = try BufferedFileWriter(url: temporary.appendingPathComponent("observation-indices.bin"), capacity: 1 * 1_024 * 1_024)
        let actions = try BufferedFileWriter(url: temporary.appendingPathComponent("actions.bin"), capacity: 1 * 1_024 * 1_024)
        var segments: [CacheSegment] = []
        var sampleCount = 0
        var observationCount = 0
        let usableDurations = recordings.map { recording in
            let start = max(0, min(recording.manifest.duration, recording.manifest.trimStart))
            let end = max(start, min(recording.manifest.duration, recording.manifest.trimEnd ?? recording.manifest.duration))
            return end - start
        }
        let totalUsableDuration = max(0.000_001, usableDurations.reduce(0, +))
        var completedDuration = 0.0
        do {
            for (recordingIndex, recording) in recordings.enumerated() {
                try Task.checkCancellation()
                let recordingDuration = usableDurations[recordingIndex]
                progress(completedDuration / totalUsableDuration, "Packing \(recording.manifest.name) • \(Int((completedDuration / totalUsableDuration * 100).rounded()))%")
                let start = sampleCount
                let appended = try await appendRecording(
                    recording,
                    profile: profile,
                    observationBase: observationCount,
                    observations: observations,
                    observationIndices: observationIndices,
                    actions: actions,
                    progress: { recordingFraction in
                        let overall = min(0.999, (completedDuration + recordingDuration * recordingFraction) / totalUsableDuration)
                        progress(overall, "Packing \(recording.manifest.name) • \(Int((overall * 100).rounded()))%")
                    }
                )
                completedDuration += recordingDuration
                observationCount += appended.observations
                sampleCount += appended.samples
                if appended.samples > 0 {
                    segments.append(CacheSegment(recordingID: recording.id, start: start, count: appended.samples))
                }
            }
            try observations.finish(); try observationIndices.finish(); try actions.finish()
            let manifest = DatasetCacheManifest(key: key, createdAt: Date(), preprocessing: profile.preprocessing, actionFPS: profile.training.actionFPS, perceptionFPS: profile.training.perceptionFPS, historyLength: profile.training.historyLength, sampleCount: sampleCount, observationCount: observationCount, observationBytesPerSample: profile.preprocessing.sampleByteCount, actionValuesPerSample: ActionLayout.count, segments: segments)
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]; encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(manifest).write(to: temporary.appendingPathComponent("manifest.json"), options: .atomic)
            if FileManager.default.fileExists(atPath: directory.path) { try FileManager.default.removeItem(at: directory) }
            try FileManager.default.moveItem(at: temporary, to: directory)
            progress(1, "Dataset cache ready")
            return try CachedDataset(directory: directory)
        } catch {
            try? observations.finish(); try? observationIndices.finish(); try? actions.finish(); try? FileManager.default.removeItem(at: temporary)
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

    private func appendRecording(_ recording: RecordingItem, profile: AIProfile, observationBase: Int, observations: BufferedFileWriter, observationIndices: BufferedFileWriter, actions: BufferedFileWriter, progress: (Double) -> Void) async throws -> (samples: Int, observations: Int) {
        guard let preprocessor else { throw AgentTrainerError.model("Metal preprocessing is unavailable.") }
        guard recording.manifest.isStructurallyValid, recording.manifest.hostStartNanos > 0 else {
            throw AgentTrainerError.storage("\(recording.manifest.name) has an invalid recording timeline or manifest.")
        }
        let events = try InputEventReader.mapped(url: recording.directory.appendingPathComponent(recording.manifest.eventFile))
        let asset = AVURLAsset(url: recording.directory.appendingPathComponent(recording.manifest.videoFile))
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { return (0, 0) }
        let reader = try AVAssetReader(asset: asset)
        let trainingStart = max(0, min(recording.manifest.duration, recording.manifest.trimStart))
        let trainingEnd = max(trainingStart, min(recording.manifest.duration, recording.manifest.trimEnd ?? recording.manifest.duration))
        guard trainingEnd > trainingStart else { return (0, 0) }
        reader.timeRange = CMTimeRange(start: CMTime(seconds: trainingStart, preferredTimescale: 1_000_000_000), duration: CMTime(seconds: trainingEnd - trainingStart, preferredTimescale: 1_000_000_000))
        // H.264/HEVC are natively decoded by VideoToolbox into bi-planar YUV.
        // Requesting BGRA forced a full-resolution conversion (about 30 MB for
        // each 3456x2168 frame) before Metal immediately converted it back to
        // packed YUV. Keeping the native video-range surfaces removes that
        // bandwidth while the shared preprocessor performs the same matrix and
        // resize at the final training resolution.
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw AgentTrainerError.storage("The recording video cannot be decoded for training.") }
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? AgentTrainerError.storage("The recording video could not be opened.") }

        let actionInterval = 1 / max(0.0001, profile.training.actionFPS)
        let perceptionInterval = 1 / max(0.0001, profile.training.perceptionFPS)
        var nextAction = trainingStart
        var nextPerception = trainingStart
        var firstPTS: CMTime?
        var latestObservationIndex: Int?
        var precedingObservationIndex: Int?
        var localObservationCount = 0
        var eventIndex = 0
        let initialPointer = events.first { $0.kind == .mouseMove || $0.kind == .mouseButton || $0.kind == .scroll }
        var accumulator = ActionAccumulator(manifest: recording.manifest, initialPointer: initialPointer)
        var count = 0
        let usableDuration = trainingEnd - trainingStart
        let progressInterval = max(2, usableDuration / 100)
        var nextProgressTime = trainingStart

        // Establish the held-control and pointer state at the trim boundary,
        // but discard movement/scroll that happened before the usable range.
        // Without this priming, the first target can contain the entire trimmed
        // lead-in as one large, directionally biased mouse action.
        let trainingStartNanos = try absoluteHostNanos(recording.manifest, seconds: trainingStart)
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
            if t >= nextProgressTime {
                progress(min(1, max(0, (t - trainingStart) / usableDuration)))
                nextProgressTime = t + progressInterval
            }
            if let currentObservation = latestObservationIndex, let precedingObservation = precedingObservationIndex {
                while nextAction < t - 0.000_001 && nextAction <= trainingEnd {
                    try writeTick(currentObservation: currentObservation, precedingObservation: precedingObservation, time: nextAction, actionInterval: actionInterval, trainingEnd: trainingEnd, recording: recording, events: events, eventIndex: &eventIndex, accumulator: &accumulator, observationIndices: observationIndices, actions: actions)
                    count += 1; nextAction += actionInterval
                }
            }
            if latestObservationIndex == nil || t + 0.000_001 >= nextPerception {
                try preprocessor.withPackedBytes(buffer, spec: profile.preprocessing) { try observations.append($0) }
                let newIndex = observationBase + localObservationCount
                localObservationCount += 1
                precedingObservationIndex = latestObservationIndex ?? newIndex
                latestObservationIndex = newIndex
                while nextPerception <= t { nextPerception += perceptionInterval }
            }
            if let currentObservation = latestObservationIndex, let precedingObservation = precedingObservationIndex {
                while nextAction <= t + 0.000_001 && nextAction <= trainingEnd {
                    try writeTick(currentObservation: currentObservation, precedingObservation: precedingObservation, time: nextAction, actionInterval: actionInterval, trainingEnd: trainingEnd, recording: recording, events: events, eventIndex: &eventIndex, accumulator: &accumulator, observationIndices: observationIndices, actions: actions)
                    count += 1; nextAction += actionInterval
                }
            }
        }
        if reader.status == .failed { throw reader.error ?? AgentTrainerError.storage("Video decoding failed while building the cache.") }
        if let currentObservation = latestObservationIndex, let precedingObservation = precedingObservationIndex {
            while nextAction <= trainingEnd {
                try writeTick(currentObservation: currentObservation, precedingObservation: precedingObservation, time: nextAction, actionInterval: actionInterval, trainingEnd: trainingEnd, recording: recording, events: events, eventIndex: &eventIndex, accumulator: &accumulator, observationIndices: observationIndices, actions: actions)
                count += 1; nextAction += actionInterval
            }
        }
        progress(1)
        return (count, localObservationCount)
    }

    private func writeTick(currentObservation: Int, precedingObservation: Int, time: Double, actionInterval: Double, trainingEnd: Double, recording: RecordingItem, events: InputEventReader.MappedEvents, eventIndex: inout Int, accumulator: inout ActionAccumulator, observationIndices: BufferedFileWriter, actions: BufferedFileWriter) throws {
        // Pair the frame at `time` with the controls demonstrated immediately
        // after it. This causal interval is what live inference must predict.
        let targetEnd = min(trainingEnd, time + actionInterval)
        let absoluteNanos = try absoluteHostNanos(recording.manifest, seconds: targetEnd)
        while eventIndex < events.count, events[eventIndex].timestampNanos < absoluteNanos { accumulator.consume(events[eventIndex]); eventIndex += 1 }
        guard let current = UInt32(exactly: currentObservation), let preceding = UInt32(exactly: precedingObservation) else {
            throw AgentTrainerError.storage("The dataset contains too many perception frames for its index format.")
        }
        try observationIndices.appendLittleEndian(current)
        try observationIndices.appendLittleEndian(preceding)
        try accumulator.withActionBytes { try actions.append($0) }
        accumulator.endTick()
    }

    private func absoluteHostNanos(_ manifest: RecordingManifest, seconds: Double) throws -> UInt64 {
        let nanos = seconds * 1_000_000_000
        guard seconds.isFinite, seconds >= 0, nanos < Double(UInt64.max - manifest.hostStartNanos) else {
            throw AgentTrainerError.storage("\(manifest.name) has a recording timestamp outside the supported range.")
        }
        return manifest.hostStartNanos + UInt64(nanos)
    }
}

struct ActionAccumulator {
    let manifest: RecordingManifest
    var x = 0.0, y = 0.0, dx = 0.0, dy = 0.0, sx = 0.0, sy = 0.0
    var buttons: Set<UInt8> = []
    var keys: Set<UInt16> = []
    var pressedButtonsThisTick: Set<UInt8> = []
    var pressedKeysThisTick: Set<UInt16> = []
    var flags: UInt64 = 0
    var pressedModifierMasksThisTick: UInt64 = 0

    init(manifest: RecordingManifest, events: [InputSample] = []) {
        self.init(manifest: manifest, initialPointer: events.first(where: { $0.kind == .mouseMove || $0.kind == .mouseButton || $0.kind == .scroll }))
    }

    init(manifest: RecordingManifest, initialPointer: InputSample?) {
        self.manifest = manifest
        if let pointer = initialPointer {
            x = pointer.x
            y = pointer.y
        } else {
            let rect = manifest.globalRect.cgRect
            x = rect.midX
            y = rect.midY
        }
    }

    mutating func consume(_ event: InputSample) {
        pressedModifierMasksThisTick |= event.modifiers
        flags = event.modifiers
        switch event.kind {
        case .mouseMove: x = event.x; y = event.y; dx += event.deltaX; dy += event.deltaY
        case .mouseButton:
            x = event.x; y = event.y
            if event.isDown { buttons.insert(event.button); pressedButtonsThisTick.insert(event.button) }
            else { buttons.remove(event.button) }
        case .scroll: x = event.x; y = event.y; sx += event.scrollX; sy += event.scrollY
        case .key:
            if event.isDown { keys.insert(event.keyCode); pressedKeysThisTick.insert(event.keyCode) }
            else { keys.remove(event.keyCode) }
        case .flags: break
        }
    }

    mutating func endTick() {
        dx = 0; dy = 0; sx = 0; sy = 0
        pressedButtonsThisTick.removeAll(keepingCapacity: true)
        pressedKeysThisTick.removeAll(keepingCapacity: true)
        pressedModifierMasksThisTick = 0
    }

    func withActionBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        var values = [Float](repeating: 0, count: ActionLayout.count)
        let rect = manifest.globalRect.cgRect
        values[0] = Float((x - rect.minX) / max(1, rect.width)).clamped01
        values[1] = Float((y - rect.minY) / max(1, rect.height)).clamped01
        values[2] = GameCameraContract.trainingValue(forRawDelta: dx)
        values[3] = GameCameraContract.trainingValue(forRawDelta: dy)
        for button in buttons.union(pressedButtonsThisTick) where button < 8 { values[4 + Int(button)] = 1 }
        values[12] = Float(sx / 20).clamped(-1, 1)
        values[13] = Float(sy / 20).clamped(-1, 1)
        for key in keys.union(pressedKeysThisTick) where key < 128 { values[14 + Int(key)] = 1 }
        let masks: [UInt64] = [CGEventFlags.maskShift.rawValue, CGEventFlags.maskControl.rawValue, CGEventFlags.maskAlternate.rawValue, CGEventFlags.maskCommand.rawValue]
        let effectiveFlags = flags | pressedModifierMasksThisTick
        for i in 0..<4 { values[142 + i] = effectiveFlags & masks[i] != 0 ? 1 : 0 }
        // AgentTrainer is arm64-only, whose native Float representation is the
        // little-endian cache contract. Copy all 146 values in one operation
        // instead of growing Data through 146 tiny appends per action sample.
        return try values.withUnsafeBytes(body)
    }

    func actionData() -> Data {
        withActionBytes { Data($0) }
    }
}

/// FileHandle writes are comparatively expensive when issued once for every
/// observation and 584-byte action row. This bounded writer turns hundreds of
/// thousands of tiny syscalls into sequential multi-megabyte writes without
/// changing a single cache byte.
private final class BufferedFileWriter {
    private let handle: FileHandle
    private let capacity: Int
    private var buffer: Data
    private var closed = false

    init(url: URL, capacity: Int) throws {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
        self.capacity = max(64 * 1_024, capacity)
        buffer = Data()
        buffer.reserveCapacity(self.capacity)
    }

    func append(_ data: Data) throws {
        try data.withUnsafeBytes { try append($0) }
    }

    func append(_ bytes: UnsafeRawBufferPointer) throws {
        guard !closed else { throw AgentTrainerError.storage("The dataset cache writer was already closed.") }
        buffer.append(contentsOf: bytes)
        if buffer.count >= capacity { try flush() }
    }

    func appendLittleEndian<T: FixedWidthInteger>(_ value: T) throws {
        var little = value.littleEndian
        try Swift.withUnsafeBytes(of: &little) { try append($0) }
    }

    func finish() throws {
        guard !closed else { return }
        try flush()
        try handle.synchronize()
        try handle.close()
        closed = true
    }

    private func flush() throws {
        guard !buffer.isEmpty else { return }
        try handle.write(contentsOf: buffer)
        buffer.removeAll(keepingCapacity: true)
    }
}

private extension Float {
    var clamped01: Float { clamped(0, 1) }
    func clamped(_ lower: Float, _ upper: Float) -> Float { min(upper, max(lower, self)) }
}
