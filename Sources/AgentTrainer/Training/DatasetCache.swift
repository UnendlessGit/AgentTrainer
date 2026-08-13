@preconcurrency import AVFoundation
import CoreGraphics
import CryptoKit
import Foundation

/// Only fields and artifact metadata that can change sampled pixels, event
/// timing, or target normalization participate in cache/checkpoint identity.
/// Renaming or moving a recording in the Library therefore keeps expensive
/// packed caches and exact optimizer state reusable.
struct RecordingTrainingIdentity: Codable, Hashable, Sendable {
    struct ArtifactFingerprint: Codable, Hashable, Sendable {
        let size: Int64
        let modifiedAt: Date?
    }

    let id: UUID
    let hostStartNanos: UInt64
    let duration: Double
    let globalRect: CodableRect
    let trimStart: Double
    let trimEnd: Double?
    let pixelWidth: Int
    let pixelHeight: Int
    let eventCount: Int
    let videoFile: String
    let eventFile: String
    let video: ArtifactFingerprint
    let events: ArtifactFingerprint

    init(recording: RecordingItem) throws {
        let manifest = recording.manifest
        func fingerprint(_ url: URL) throws -> ArtifactFingerprint {
            let values = try url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
            guard let size = values.fileSize, size >= 0 else {
                throw AgentTrainerError.storage("A recording artifact is unavailable while preparing training data.")
            }
            return ArtifactFingerprint(size: Int64(size), modifiedAt: values.contentModificationDate)
        }
        id = manifest.id
        hostStartNanos = manifest.hostStartNanos
        duration = manifest.duration
        globalRect = manifest.globalRect
        trimStart = manifest.trimStart
        trimEnd = manifest.trimEnd
        pixelWidth = manifest.pixelWidth
        pixelHeight = manifest.pixelHeight
        eventCount = manifest.eventCount
        videoFile = manifest.videoFile
        eventFile = manifest.eventFile
        video = try fingerprint(recording.directory.appendingPathComponent(manifest.videoFile))
        events = try fingerprint(recording.directory.appendingPathComponent(manifest.eventFile))
    }
}

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
    /// Command, Option, and Control are also reported by macOS as ordinary
    /// key codes. Those duplicate keyboard outputs must never be learned or
    /// executed; their dedicated modifier outputs are the single source of
    /// truth. Shift intentionally remains part of Keyboard.
    static let commandOptionControlKeyCodes: [UInt16] = [54, 55, 58, 59, 61, 62]
    static let commandOptionControlKeyCodeSet = Set(commandOptionControlKeyCodes)
    static let commandOptionControlKeyboardIndices = commandOptionControlKeyCodes.map { keyboard.lowerBound + Int($0) }
    static let commandOptionControlKeyboardIndexSet = Set(commandOptionControlKeyboardIndices)
    static let keyboardAndShiftIndices = Array(keyboardAndShift).filter { !commandOptionControlKeyboardIndexSet.contains($0) }
    static let binary = Array(buttons) + keyboardAndShiftIndices + Array(commandOptionControl)

    static func learnableBinaryIndices(
        channels: ActionChannels,
        restrictions: ActionRestrictions
    ) -> [Int] {
        var result: [Int] = []
        if channels.buttons {
            result += buttons.filter {
                restrictions.allowsButton(UInt8($0 - buttons.lowerBound))
            }
        }
        if channels.keyboard {
            result += keyboardAndShiftIndices.filter { index in
                if index == shift.lowerBound { return restrictions.allowsModifier(0) }
                return restrictions.allowsKey(UInt16(index - keyboard.lowerBound))
            }
        }
        if channels.modifiers {
            result += commandOptionControl.filter {
                restrictions.allowsModifier($0 - modifiers.lowerBound)
            }
        }
        return result
    }

    /// Removes controls that do not belong to the selected training channels
    /// from targets and frame-aligned temporal controls. This prevents a
    /// disabled channel from becoming a hidden shortcut through the recurrent
    /// branch.
    static func sanitizeTrainingRows(
        _ values: UnsafeMutableBufferPointer<Float>,
        rowCount: Int,
        channels: ActionChannels,
        restrictions: ActionRestrictions
    ) {
        guard rowCount > 0, values.count >= rowCount * count else { return }
        for row in 0..<rowCount {
            let base = row * count
            if !channels.mouseMovement {
                for index in absoluteMouse { values[base + index] = 0 }
                for index in relativeMouse { values[base + index] = 0 }
            }
            if !channels.buttons {
                for index in buttons { values[base + index] = 0 }
            } else {
                for button in restrictions.blockedMouseButtons where button < 8 {
                    values[base + buttons.lowerBound + Int(button)] = 0
                }
            }
            if !channels.scroll {
                for index in scroll { values[base + index] = 0 }
            }
            if !channels.keyboard {
                for index in keyboardAndShift { values[base + index] = 0 }
            } else {
                for key in restrictions.blockedKeyCodes where key < 128 {
                    values[base + keyboard.lowerBound + Int(key)] = 0
                }
                if !restrictions.allowsModifier(0) { values[base + shift.lowerBound] = 0 }
            }
            // Close the duplicate-keyboard loophole even when both channels
            // are enabled. Only the dedicated modifier outputs may own these.
            for index in commandOptionControlKeyboardIndices { values[base + index] = 0 }
            if !channels.modifiers {
                for index in commandOptionControl { values[base + index] = 0 }
            } else {
                for modifier in 1..<4 where !restrictions.allowsModifier(modifier) {
                    values[base + modifiers.lowerBound + modifier] = 0
                }
            }
        }
    }
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
    var pastPreprocessing: PreprocessingSpec
    var temporalVision: TemporalVisionConfiguration
    var actionFPS: Double
    var perceptionFPS: Double
    var sampleCount: Int
    var observationCount: Int
    var currentObservationBytesPerSample: Int
    var pastObservationBytesPerSample: Int
    var actionValuesPerSample: Int
    var segments: [CacheSegment]

    var observationIndexValuesPerSample: Int { 1 + temporalVision.pastFrameCount }
}

/// Compact host batch copied from the memory-mapped cache. Current and past
/// vision remain independently packed UInt8, past controls stay paired with
/// their frames, and action rows contain target plus real previous target.
struct CachedTrainingBatch: Sendable {
    let count: Int
    let packedCurrentObservations: Data
    let packedPastObservations: Data
    let pastControlRows: Data
    let actionRows: Data
}

struct CachedObservationSequence: Hashable, Sendable {
    /// Slot zero is current. Remaining slots are causal past frames ordered
    /// oldest to newest. `UInt32.max` marks unavailable segment-leading slots.
    let indices: [UInt32]
}

/// Maps action rows onto the distinct temporal frame sequences used by a batch.
/// Action FPS commonly exceeds perception FPS, so several labels can share one
/// expensive temporal frame sequence while retaining independent targets.
struct CachedVisionBatchPlan: Sendable {
    let uniqueSequences: [CachedObservationSequence]
    /// Distinct reduced-resolution observations referenced by every past slot
    /// in `uniqueSequences`. Segment-leading padding resolves to that
    /// sequence's current observation, exactly as it does in the canonical
    /// batch path.
    let uniquePastObservations: [UInt32]
    /// Row-major [unique sequence, past slot] map into
    /// `uniquePastObservations`. Keeping this as Int32 lets MLX gather the
    /// already-encoded frame embeddings without another host conversion.
    let visionToPast: [Int32]
    let sampleToVision: [Int32]

    var reuseRatio: Double {
        Double(sampleToVision.count) / Double(max(1, uniqueSequences.count))
    }

    var pastFrameReuseRatio: Double {
        Double(visionToPast.count) / Double(max(1, uniquePastObservations.count))
    }

    func encoderWorkReuseRatio(currentPixels: Int, pastPixels: Int) -> Double {
        let pastFrameCount = uniqueSequences.first.map { max(0, $0.indices.count - 1) } ?? 0
        let reference = Double(sampleToVision.count)
            * Double(max(0, currentPixels) + pastFrameCount * max(0, pastPixels))
        let shared = Double(uniqueSequences.count) * Double(max(0, currentPixels))
            + Double(uniquePastObservations.count) * Double(max(0, pastPixels))
        return reference / max(1, shared)
    }
}

final class CachedDataset: @unchecked Sendable {
    let manifest: DatasetCacheManifest
    private let currentObservations: Data
    private let pastObservations: Data
    private let observationIndices: Data
    private let frameActions: Data
    private let actions: Data

    init(directory: URL) throws {
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(DatasetCacheManifest.self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
        guard decoded.schemaVersion == TrainingDataContract.schemaVersion else {
            throw AgentTrainerError.storage("This dataset cache uses an obsolete input contract and must be rebuilt.")
        }
        _ = try decoded.preprocessing.validated()
        _ = try decoded.temporalVision.validated(current: decoded.preprocessing)
        _ = try decoded.pastPreprocessing.validated()
        let expectedPastSpec = decoded.temporalVision.pastFrameSpec(from: decoded.preprocessing)
        let expectedPastBytes = decoded.temporalVision.pastFrameCount > 0
            ? decoded.pastPreprocessing.sampleByteCount
            : 0
        guard decoded.sampleCount >= 0,
              decoded.observationCount >= 0,
              decoded.sampleCount == 0 || decoded.observationCount > 0,
              decoded.observationCount <= Int(UInt32.max),
              decoded.actionFPS.isFinite, decoded.actionFPS > 0,
              decoded.perceptionFPS.isFinite, decoded.perceptionFPS > 0,
              decoded.pastPreprocessing == expectedPastSpec,
              decoded.currentObservationBytesPerSample == decoded.preprocessing.sampleByteCount,
              decoded.pastObservationBytesPerSample == expectedPastBytes,
              decoded.currentObservationBytesPerSample > 0,
              decoded.pastObservationBytesPerSample >= 0,
              decoded.actionValuesPerSample == ActionLayout.count else {
            throw AgentTrainerError.storage("The dataset cache manifest is invalid.")
        }

        let currentObservationSize = decoded.observationCount.multipliedReportingOverflow(by: decoded.currentObservationBytesPerSample)
        let pastObservationSize = decoded.observationCount.multipliedReportingOverflow(by: decoded.pastObservationBytesPerSample)
        let mappingValueCount = decoded.sampleCount.multipliedReportingOverflow(by: decoded.observationIndexValuesPerSample)
        let mappingSize = mappingValueCount.partialValue.multipliedReportingOverflow(by: MemoryLayout<UInt32>.size)
        let actionValueCount = decoded.sampleCount.multipliedReportingOverflow(by: decoded.actionValuesPerSample)
        let actionSize = actionValueCount.partialValue.multipliedReportingOverflow(by: MemoryLayout<Float>.size)
        let retainedFrameActionCount = decoded.temporalVision.pastFrameCount > 0
            ? decoded.observationCount
            : 0
        let frameActionValueCount = retainedFrameActionCount.multipliedReportingOverflow(by: decoded.actionValuesPerSample)
        let frameActionSize = frameActionValueCount.partialValue.multipliedReportingOverflow(by: MemoryLayout<Float>.size)
        guard !currentObservationSize.overflow, !pastObservationSize.overflow,
              !mappingValueCount.overflow, !mappingSize.overflow,
              !actionValueCount.overflow, !actionSize.overflow,
              !frameActionValueCount.overflow, !frameActionSize.overflow else {
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

        // Cache files can be many gigabytes. Requiring virtual mappings keeps
        // startup constant-memory and lets macOS page random shuffled batches
        // on demand instead of first copying the complete dataset into RAM.
        let loadedCurrentObservations = try Data(contentsOf: directory.appendingPathComponent("current-observations.bin"), options: .alwaysMapped)
        let loadedPastObservations = try Data(contentsOf: directory.appendingPathComponent("past-observations.bin"), options: .alwaysMapped)
        let loadedObservationIndices = try Data(contentsOf: directory.appendingPathComponent("observation-indices.bin"), options: .alwaysMapped)
        let loadedFrameActions = try Data(contentsOf: directory.appendingPathComponent("frame-actions.bin"), options: .alwaysMapped)
        let loadedActions = try Data(contentsOf: directory.appendingPathComponent("actions.bin"), options: .alwaysMapped)
        guard loadedCurrentObservations.count == currentObservationSize.partialValue,
              loadedPastObservations.count == pastObservationSize.partialValue,
              loadedObservationIndices.count == mappingSize.partialValue,
              loadedFrameActions.count == frameActionSize.partialValue,
              loadedActions.count == actionSize.partialValue else {
            throw AgentTrainerError.storage("The dataset cache is incomplete or corrupt.")
        }
        let mappingsAreValid = loadedObservationIndices.withUnsafeBytes { raw -> Bool in
            guard decoded.sampleCount == 0 || raw.baseAddress != nil else { return false }
            func value(sample: Int, slot: Int) -> UInt32 {
                let offset = (sample * decoded.observationIndexValuesPerSample + slot) * MemoryLayout<UInt32>.size
                return raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian
            }
            for segment in decoded.segments {
                guard segment.count > 0 else { continue }
                let observationBase = value(sample: segment.start, slot: 0)
                guard observationBase != UInt32.max, Int(observationBase) < decoded.observationCount else { return false }
                var previousCurrent = observationBase
                for sample in segment.start..<(segment.start + segment.count) {
                    let current = value(sample: sample, slot: 0)
                    if current == UInt32.max || Int(current) >= decoded.observationCount || current < previousCurrent {
                        return false
                    }
                    previousCurrent = current
                    let localCurrent = Int(current - observationBase)
                    for frame in 0..<decoded.temporalVision.pastFrameCount {
                        let mapped = value(sample: sample, slot: frame + 1)
                        let distance = decoded.temporalVision.frameSpacing
                            * (decoded.temporalVision.pastFrameCount - frame)
                        if localCurrent >= distance {
                            let expected = current - UInt32(distance)
                            if mapped != expected { return false }
                        } else if mapped != UInt32.max {
                            return false
                        }
                    }
                }
            }
            return true
        }
        guard mappingsAreValid else { throw AgentTrainerError.storage("The dataset cache contains an invalid frame index.") }
        Self.adviseAdaptiveAccess(loadedCurrentObservations)
        Self.adviseAdaptiveAccess(loadedPastObservations)
        Self.adviseAdaptiveAccess(loadedObservationIndices)
        Self.adviseAdaptiveAccess(loadedFrameActions)
        Self.adviseAdaptiveAccess(loadedActions)
        manifest = decoded
        currentObservations = loadedCurrentObservations
        pastObservations = loadedPastObservations
        observationIndices = loadedObservationIndices
        frameActions = loadedFrameActions
        actions = loadedActions
    }

    var count: Int { manifest.sampleCount }

    private static func adviseAdaptiveAccess(_ data: Data) {
        data.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress, !bytes.isEmpty else { return }
            // Training randomizes locality batches rather than individual
            // observations. Normal advice lets macOS detect and read ahead the
            // contiguous mapped pages inside each batch. MADV_RANDOM disabled
            // that useful behavior and made page-fault stalls more likely once
            // macOS began reclaiming otherwise warm file-backed pages.
            _ = madvise(
                UnsafeMutableRawPointer(mutating: baseAddress),
                bytes.count,
                MADV_NORMAL
            )
        }
    }

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
        let size = manifest.currentObservationBytesPerSample
        let observation = observationIndex(at: index, slot: 0)
        return currentObservations.subdata(in: observation * size..<(observation + 1) * size)
    }

    /// Gathers every tensor needed by one optimizer step in a single pass over
    /// each mapped file. The older per-field helpers remain useful for focused
    /// tests and runtime reads, while training avoids repeated allocations,
    /// mapping lookups, and segment binary searches.
    func trainingBatch(at indices: [Int]) -> CachedTrainingBatch {
        let batchCount = indices.count
        let pastFrameCount = manifest.temporalVision.pastFrameCount
        let currentBytes = manifest.currentObservationBytesPerSample
        let pastBytes = manifest.pastObservationBytesPerSample
        let actionRowBytes = manifest.actionValuesPerSample * MemoryLayout<Float>.size
        var packedCurrent = Data(count: batchCount * currentBytes)
        var packedPast = Data(count: batchCount * pastFrameCount * pastBytes)
        var pastControls = Data(count: batchCount * pastFrameCount * actionRowBytes)
        var actionRows = Data(count: batchCount * 2 * actionRowBytes)
        packedCurrent.withUnsafeMutableBytes { currentDestination in
            packedPast.withUnsafeMutableBytes { pastDestination in
                pastControls.withUnsafeMutableBytes { controlDestination in
                    actionRows.withUnsafeMutableBytes { actionDestination in
                        populateTrainingBatch(
                            at: indices,
                            packedCurrentObservations: currentDestination,
                            packedPastObservations: pastDestination,
                            pastControlRows: controlDestination,
                            actionRows: actionDestination
                        )
                    }
                }
            }
        }
        return CachedTrainingBatch(
            count: batchCount,
            packedCurrentObservations: packedCurrent,
            packedPastObservations: packedPast,
            pastControlRows: pastControls,
            actionRows: actionRows
        )
    }

    func visionBatchPlan(at indices: [Int]) -> CachedVisionBatchPlan {
        guard !indices.isEmpty else {
            return CachedVisionBatchPlan(
                uniqueSequences: [],
                uniquePastObservations: [],
                visionToPast: [],
                sampleToVision: []
            )
        }
        var uniqueSequences: [CachedObservationSequence] = []
        uniqueSequences.reserveCapacity(indices.count)
        // Observation indices are global across all cache segments, so the
        // current observation is a compact unique key for its complete causal
        // sequence. Hashing a freshly allocated [UInt32] for every action row
        // was measurable CPU work at high action rates, especially when two or
        // more labels share the same perception. Read the mapped index file in
        // one scope and materialize the full sequence only on first use.
        var currentToSequence: [UInt32: Int32] = [:]
        currentToSequence.reserveCapacity(indices.count)
        var sampleToVision: [Int32] = []
        sampleToVision.reserveCapacity(indices.count)
        observationIndices.withUnsafeBytes { raw in
            let valuesPerSample = manifest.observationIndexValuesPerSample
            for sampleIndex in indices {
                let byteOffset = sampleIndex * valuesPerSample * MemoryLayout<UInt32>.size
                let current = raw.loadUnaligned(
                    fromByteOffset: byteOffset,
                    as: UInt32.self
                ).littleEndian
                if let existing = currentToSequence[current] {
                    sampleToVision.append(existing)
                    continue
                }
                precondition(
                    uniqueSequences.count < Int(Int32.max),
                    "A vision batch has too many distinct frame sequences."
                )
                var sequenceValues: [UInt32] = []
                sequenceValues.reserveCapacity(valuesPerSample)
                for slot in 0..<valuesPerSample {
                    sequenceValues.append(raw.loadUnaligned(
                        fromByteOffset: byteOffset + slot * MemoryLayout<UInt32>.size,
                        as: UInt32.self
                    ).littleEndian)
                }
                let sequenceIndex = Int32(uniqueSequences.count)
                uniqueSequences.append(CachedObservationSequence(indices: sequenceValues))
                currentToSequence[current] = sequenceIndex
                sampleToVision.append(sequenceIndex)
            }
        }
        let pastFrameCount = manifest.temporalVision.pastFrameCount
        var uniquePastObservations: [UInt32] = []
        uniquePastObservations.reserveCapacity(uniqueSequences.count * pastFrameCount)
        var pastObservationToIndex: [UInt32: Int32] = [:]
        pastObservationToIndex.reserveCapacity(uniqueSequences.count * pastFrameCount)
        var visionToPast: [Int32] = []
        visionToPast.reserveCapacity(uniqueSequences.count * pastFrameCount)
        for sequence in uniqueSequences {
            precondition(sequence.indices.count == pastFrameCount + 1)
            let current = sequence.indices[0]
            for frame in 0..<pastFrameCount {
                let mapped = sequence.indices[frame + 1]
                let observation = mapped == UInt32.max ? current : mapped
                if let existing = pastObservationToIndex[observation] {
                    visionToPast.append(existing)
                } else {
                    precondition(
                        uniquePastObservations.count < Int(Int32.max),
                        "A vision batch has too many distinct past frames."
                    )
                    let index = Int32(uniquePastObservations.count)
                    uniquePastObservations.append(observation)
                    pastObservationToIndex[observation] = index
                    visionToPast.append(index)
                }
            }
        }
        return CachedVisionBatchPlan(
            uniqueSequences: uniqueSequences,
            uniquePastObservations: uniquePastObservations,
            visionToPast: visionToPast,
            sampleToVision: sampleToVision
        )
    }

    /// Groups rows by temporal observation without dropping or duplicating any
    /// labels. The trainer shuffles these groups each epoch so repeated visual
    /// inputs land in the same batch and can share one CNN evaluation.
    func observationGroups(
        at indices: [Int],
        using suppliedPlan: CachedVisionBatchPlan? = nil
    ) -> [[Int]] {
        if let plan = suppliedPlan {
            precondition(
                plan.sampleToVision.count == indices.count,
                "A vision plan must describe every supplied dataset row."
            )
            var groups = Array(repeating: [Int](), count: plan.uniqueSequences.count)
            for (row, sampleIndex) in indices.enumerated() {
                groups[Int(plan.sampleToVision[row])].append(sampleIndex)
            }
            return groups
        }

        // The cache contract assigns one global current-observation index to a
        // complete causal sequence. Grouping needs only that compact key; a
        // full visionBatchPlan would also materialize every temporal sequence,
        // deduplicate every past frame, and build two gather maps that startup
        // immediately discards.
        guard !indices.isEmpty else { return [] }
        var currentToGroup: [UInt32: Int] = [:]
        currentToGroup.reserveCapacity(indices.count)
        var groups: [[Int]] = []
        groups.reserveCapacity(indices.count)
        observationIndices.withUnsafeBytes { raw in
            let valuesPerSample = manifest.observationIndexValuesPerSample
            for sampleIndex in indices {
                let byteOffset = sampleIndex * valuesPerSample * MemoryLayout<UInt32>.size
                let current = raw.loadUnaligned(
                    fromByteOffset: byteOffset,
                    as: UInt32.self
                ).littleEndian
                if let group = currentToGroup[current] {
                    groups[group].append(sampleIndex)
                } else {
                    currentToGroup[current] = groups.count
                    groups.append([sampleIndex])
                }
            }
        }
        return groups
    }

    func observationReuseRatio(at indices: [Int]) -> Double {
        visionBatchPlan(at: indices).reuseRatio
    }

    /// Writes a fused batch into caller-owned storage. Training passes Metal
    /// shared buffers here, removing the otherwise unavoidable second copy from
    /// temporary `Data` into MLX input storage.
    func populateTrainingBatch(
        at indices: [Int],
        packedCurrentObservations currentDestination: UnsafeMutableRawBufferPointer,
        packedPastObservations pastDestination: UnsafeMutableRawBufferPointer,
        pastControlRows controlDestination: UnsafeMutableRawBufferPointer,
        actionRows actionDestination: UnsafeMutableRawBufferPointer
    ) {
        let sequences = observationSequences(at: indices)
        populateTrainingBatch(
            at: indices,
            observationSequences: sequences,
            pastObservationIndices: expandedPastObservationIndices(for: sequences),
            packedCurrentObservations: currentDestination,
            packedPastObservations: pastDestination,
            pastControlRows: controlDestination,
            actionRows: actionDestination
        )
    }

    func populateTrainingBatch(
        at indices: [Int],
        observationSequences: [CachedObservationSequence],
        packedCurrentObservations currentDestination: UnsafeMutableRawBufferPointer,
        packedPastObservations pastDestination: UnsafeMutableRawBufferPointer,
        pastControlRows controlDestination: UnsafeMutableRawBufferPointer,
        actionRows actionDestination: UnsafeMutableRawBufferPointer
    ) {
        populateTrainingBatch(
            at: indices,
            observationSequences: observationSequences,
            pastObservationIndices: expandedPastObservationIndices(for: observationSequences),
            packedCurrentObservations: currentDestination,
            packedPastObservations: pastDestination,
            pastControlRows: controlDestination,
            actionRows: actionDestination
        )
    }

    /// Writes a deduplicated visual batch. Current images remain one per unique
    /// temporal sequence, while each reduced-resolution past observation is
    /// copied and encoded only once even when overlapping causal windows refer
    /// to it repeatedly. Controls intentionally remain sequence-shaped because
    /// padding slots and real slots can share an image but never share meaning.
    func populateTrainingBatch(
        at indices: [Int],
        visionPlan: CachedVisionBatchPlan,
        packedCurrentObservations currentDestination: UnsafeMutableRawBufferPointer,
        packedPastObservations pastDestination: UnsafeMutableRawBufferPointer,
        pastControlRows controlDestination: UnsafeMutableRawBufferPointer,
        actionRows actionDestination: UnsafeMutableRawBufferPointer
    ) {
        precondition(visionPlan.sampleToVision.count == indices.count)
        precondition(
            visionPlan.visionToPast.count
                == visionPlan.uniqueSequences.count * manifest.temporalVision.pastFrameCount
        )
        populateTrainingBatch(
            at: indices,
            observationSequences: visionPlan.uniqueSequences,
            pastObservationIndices: visionPlan.uniquePastObservations,
            packedCurrentObservations: currentDestination,
            packedPastObservations: pastDestination,
            pastControlRows: controlDestination,
            actionRows: actionDestination
        )
    }

    private func expandedPastObservationIndices(
        for observationSequences: [CachedObservationSequence]
    ) -> [UInt32] {
        let pastFrameCount = manifest.temporalVision.pastFrameCount
        var result: [UInt32] = []
        result.reserveCapacity(observationSequences.count * pastFrameCount)
        for sequence in observationSequences {
            precondition(sequence.indices.count == pastFrameCount + 1)
            let current = sequence.indices[0]
            for frame in 0..<pastFrameCount {
                let mapped = sequence.indices[frame + 1]
                result.append(mapped == UInt32.max ? current : mapped)
            }
        }
        return result
    }

    private func populateTrainingBatch(
        at indices: [Int],
        observationSequences: [CachedObservationSequence],
        pastObservationIndices: [UInt32],
        packedCurrentObservations currentDestination: UnsafeMutableRawBufferPointer,
        packedPastObservations pastDestination: UnsafeMutableRawBufferPointer,
        pastControlRows controlDestination: UnsafeMutableRawBufferPointer,
        actionRows actionDestination: UnsafeMutableRawBufferPointer
    ) {
        let batchCount = indices.count
        let pastFrameCount = manifest.temporalVision.pastFrameCount
        let currentBytes = manifest.currentObservationBytesPerSample
        let pastBytes = manifest.pastObservationBytesPerSample
        let actionRowBytes = manifest.actionValuesPerSample * MemoryLayout<Float>.size
        precondition(currentDestination.count == observationSequences.count * currentBytes)
        precondition(pastDestination.count == pastObservationIndices.count * pastBytes)
        precondition(controlDestination.count == observationSequences.count * pastFrameCount * actionRowBytes)
        precondition(actionDestination.count == batchCount * 2 * actionRowBytes)
        if pastFrameCount == 0 {
            populateCurrentOnlyTrainingBatch(
                at: indices,
                observationSequences: observationSequences,
                packedCurrentObservations: currentDestination,
                actionRows: actionDestination
            )
            return
        }
        if let controlBase = controlDestination.baseAddress {
            memset(controlBase, 0, controlDestination.count)
        }
        if let actionBase = actionDestination.baseAddress {
            memset(actionBase, 0, actionDestination.count)
        }

        currentObservations.withUnsafeBytes { currentSource in
            pastObservations.withUnsafeBytes { pastSource in
                frameActions.withUnsafeBytes { frameActionSource in
                    guard let currentDestinationBase = currentDestination.baseAddress,
                          let pastDestinationBase = pastDestination.baseAddress,
                          let controlDestinationBase = controlDestination.baseAddress,
                          let currentSourceBase = currentSource.baseAddress,
                          let pastSourceBase = pastSource.baseAddress,
                          let frameActionSourceBase = frameActionSource.baseAddress else { return }
                    for (visionRow, sequence) in observationSequences.enumerated() {
                        precondition(sequence.indices.count == pastFrameCount + 1)
                        let current = Int(sequence.indices[0])
                        memcpy(
                            currentDestinationBase.advanced(by: visionRow * currentBytes),
                            currentSourceBase.advanced(by: current * currentBytes),
                            currentBytes
                        )
                        for frame in 0..<pastFrameCount {
                            let mapped = sequence.indices[frame + 1]
                            // Padding frames intentionally carry no controls, so
                            // duplicating the current low-resolution image at a
                            // segment boundary can never leak its target action.
                            if mapped != UInt32.max {
                                memcpy(
                                    controlDestinationBase.advanced(by: (visionRow * pastFrameCount + frame) * actionRowBytes),
                                    frameActionSourceBase.advanced(by: Int(mapped) * actionRowBytes),
                                    actionRowBytes
                                )
                            }
                        }
                    }
                    for (pastRow, observation) in pastObservationIndices.enumerated() {
                        memcpy(
                            pastDestinationBase.advanced(by: pastRow * pastBytes),
                            pastSourceBase.advanced(by: Int(observation) * pastBytes),
                            pastBytes
                        )
                    }
                }
            }
        }

        actions.withUnsafeBytes { source in
            guard let destinationBase = actionDestination.baseAddress,
                  let sourceBase = source.baseAddress else { return }
            for (batchRow, sampleIndex) in indices.enumerated() {
                memcpy(
                    destinationBase.advanced(by: batchRow * 2 * actionRowBytes),
                    sourceBase.advanced(by: sampleIndex * actionRowBytes),
                    actionRowBytes
                )
                if sampleIndex > segmentStart(for: sampleIndex) {
                    memcpy(
                        destinationBase.advanced(by: (batchRow * 2 + 1) * actionRowBytes),
                        sourceBase.advanced(by: (sampleIndex - 1) * actionRowBytes),
                        actionRowBytes
                    )
                }
            }
        }
    }

    /// Current-frame-only profiles avoid allocating, mapping, and copying
    /// reduced vision or frame-control rows. The action layout remains exactly
    /// the same, including the real immediately preceding target.
    func populateCurrentOnlyTrainingBatch(
        at indices: [Int],
        observationSequences suppliedSequences: [CachedObservationSequence]? = nil,
        packedCurrentObservations currentDestination: UnsafeMutableRawBufferPointer,
        actionRows actionDestination: UnsafeMutableRawBufferPointer
    ) {
        let sequences = suppliedSequences ?? indices.map(observationSequence(at:))
        let currentBytes = manifest.currentObservationBytesPerSample
        let actionRowBytes = manifest.actionValuesPerSample * MemoryLayout<Float>.size
        precondition(manifest.temporalVision.pastFrameCount == 0)
        precondition(sequences.count <= indices.count)
        precondition(currentDestination.count == sequences.count * currentBytes)
        precondition(actionDestination.count == indices.count * 2 * actionRowBytes)
        if let actionBase = actionDestination.baseAddress {
            memset(actionBase, 0, actionDestination.count)
        }
        currentObservations.withUnsafeBytes { source in
            guard let destinationBase = currentDestination.baseAddress,
                  let sourceBase = source.baseAddress else { return }
            for (row, sequence) in sequences.enumerated() {
                precondition(sequence.indices.count == 1)
                let current = Int(sequence.indices[0])
                memcpy(
                    destinationBase.advanced(by: row * currentBytes),
                    sourceBase.advanced(by: current * currentBytes),
                    currentBytes
                )
            }
        }
        actions.withUnsafeBytes { source in
            guard let destinationBase = actionDestination.baseAddress,
                  let sourceBase = source.baseAddress else { return }
            for (row, sampleIndex) in indices.enumerated() {
                memcpy(
                    destinationBase.advanced(by: row * 2 * actionRowBytes),
                    sourceBase.advanced(by: sampleIndex * actionRowBytes),
                    actionRowBytes
                )
                if sampleIndex > segmentStart(for: sampleIndex) {
                    memcpy(
                        destinationBase.advanced(by: (row * 2 + 1) * actionRowBytes),
                        sourceBase.advanced(by: (sampleIndex - 1) * actionRowBytes),
                        actionRowBytes
                    )
                }
            }
        }
    }

    func packedObservations(at indices: [Int]) -> Data {
        let size = manifest.currentObservationBytesPerSample
        var result = Data(count: indices.count * size)
        result.withUnsafeMutableBytes { destination in
            currentObservations.withUnsafeBytes { source in
                guard let destinationBase = destination.baseAddress, let sourceBase = source.baseAddress else { return }
                for (row, index) in indices.enumerated() {
                    let observation = observationIndex(at: index, slot: 0)
                    memcpy(destinationBase.advanced(by: row * size), sourceBase.advanced(by: observation * size), size)
                }
            }
        }
        return result
    }

    /// Returns every causal past frame at its native reduced resolution,
    /// ordered oldest to newest for each action sample.
    func pastPackedObservations(at indices: [Int]) -> Data {
        let frameCount = manifest.temporalVision.pastFrameCount
        let size = manifest.pastObservationBytesPerSample
        var result = Data(count: indices.count * frameCount * size)
        result.withUnsafeMutableBytes { destination in
            pastObservations.withUnsafeBytes { source in
                guard let destinationBase = destination.baseAddress, let sourceBase = source.baseAddress else { return }
                for (row, sample) in indices.enumerated() {
                    let sequence = observationSequence(at: sample)
                    let current = Int(sequence.indices[0])
                    for frame in 0..<frameCount {
                        let mapped = sequence.indices[frame + 1]
                        let observation = mapped == UInt32.max ? current : Int(mapped)
                        memcpy(
                            destinationBase.advanced(by: (row * frameCount + frame) * size),
                            sourceBase.advanced(by: observation * size),
                            size
                        )
                    }
                }
            }
        }
        return result
    }

    func pastControlBatch(at indices: [Int]) -> Data {
        let frameCount = manifest.temporalVision.pastFrameCount
        let rowBytes = manifest.actionValuesPerSample * MemoryLayout<Float>.size
        var result = Data(count: indices.count * frameCount * rowBytes)
        result.withUnsafeMutableBytes { destination in
            frameActions.withUnsafeBytes { source in
                guard let destinationBase = destination.baseAddress, let sourceBase = source.baseAddress else { return }
                for (row, sample) in indices.enumerated() {
                    let sequence = observationSequence(at: sample)
                    for frame in 0..<frameCount {
                        let mapped = sequence.indices[frame + 1]
                        guard mapped != UInt32.max else { continue }
                        memcpy(
                            destinationBase.advanced(by: (row * frameCount + frame) * rowBytes),
                            sourceBase.advanced(by: Int(mapped) * rowBytes),
                            rowBytes
                        )
                    }
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

    /// Finds action rows that carry substantially more learning signal than an
    /// ordinary held-state frame. The trainer still consumes every row exactly
    /// once per epoch; it only spreads these transitions and active additive
    /// controls across batches to reduce gradient variance.
    func salientTrainingIndices(
        at indices: [Int],
        channels: ActionChannels,
        restrictions: ActionRestrictions
    ) -> Set<Int> {
        guard !indices.isEmpty else { return [] }
        let binaryOutputs = ActionLayout.learnableBinaryIndices(
            channels: channels,
            restrictions: restrictions
        )
        let continuousOutputs =
            (channels.mouseMovement ? Array(ActionLayout.relativeMouse) : [])
            + (channels.scroll ? Array(ActionLayout.scroll) : [])

        var result: Set<Int> = []
        actions.withUnsafeBytes { raw in
            guard let address = raw.baseAddress else { return }
            let values = address.assumingMemoryBound(to: UInt32.self)
            func value(row: Int, output: Int) -> Float {
                Float(bitPattern: UInt32(littleEndian: values[row * manifest.actionValuesPerSample + output]))
            }
            for row in indices {
                if continuousOutputs.contains(where: { abs(value(row: row, output: $0)) > 0.0001 }) {
                    result.insert(row)
                    continue
                }
                let segmentStart = segmentStart(for: row)
                for output in binaryOutputs {
                    let current = value(row: row, output: output) >= 0.5
                    let previous = row > segmentStart ? value(row: row - 1, output: output) >= 0.5 : false
                    if current != previous {
                        result.insert(row)
                        break
                    }
                }
            }
        }
        return result
    }

    /// Returns the first single-recording validation row whose complete current
    /// and past-frame sequence is disjoint from training.
    /// Recorded frame delivery can be irregular, so an FPS-derived fixed gap is
    /// not sufficient: a static frame may back many action rows.
    func firstDisjointValidationIndex(trainingEnd: Int, proposedStart: Int) -> Int? {
        guard trainingEnd > 0, trainingEnd < count else { return nil }
        let lastTrainingSequence = observationSequence(at: trainingEnd - 1)
        let maximumTrainingObservation = lastTrainingSequence.indices
            .filter { $0 != UInt32.max }
            .max() ?? 0
        var candidate = max(proposedStart, trainingEnd)
        while candidate < count {
            let sequence = observationSequence(at: candidate)
            let frames = sequence.indices.filter { $0 != UInt32.max }
            if frames.count == manifest.observationIndexValuesPerSample,
               frames.allSatisfy({ $0 > maximumTrainingObservation }) {
                return candidate
            }
            candidate += 1
        }
        return nil
    }

    /// Builds a fixed held-out subset once per run. At least one positive for
    /// every demonstrated binary output comes first so an inert policy cannot
    /// look good on a tiny validation budget. Press/release boundaries and
    /// active delta/scroll examples follow; remaining slots are distributed
    /// evenly across the entire held-out timeline.
    func representativeValidationIndices(
        from indices: [Int],
        limit rawLimit: Int,
        channels: ActionChannels = .all,
        restrictions: ActionRestrictions = ActionRestrictions()
    ) -> [Int] {
        let limit = min(indices.count, max(1, rawLimit))
        guard indices.count > limit else { return indices }
        let binaryOutputs = ActionLayout.learnableBinaryIndices(
            channels: channels,
            restrictions: restrictions
        )
        var transitionRows = [Int?](repeating: nil, count: ActionLayout.count)
        var positiveRows = [Int?](repeating: nil, count: ActionLayout.count)
        let continuousOutputs =
            (channels.mouseMovement ? Array(ActionLayout.relativeMouse) : [])
            + (channels.scroll ? Array(ActionLayout.scroll) : [])
        var continuousRows = [Int?](repeating: nil, count: continuousOutputs.count)
        actions.withUnsafeBytes { raw in
            guard let address = raw.baseAddress else { return }
            let values = address.assumingMemoryBound(to: UInt32.self)
            func value(row: Int, output: Int) -> Float {
                Float(bitPattern: UInt32(littleEndian: values[row * manifest.actionValuesPerSample + output]))
            }
            for row in indices {
                let segmentStart = segmentStart(for: row)
                for output in binaryOutputs {
                    let current = value(row: row, output: output) >= 0.5
                    if current, positiveRows[output] == nil { positiveRows[output] = row }
                    let previous = row > segmentStart ? value(row: row - 1, output: output) >= 0.5 : false
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
        // Every held-out recording should influence the score when the budget
        // permits. Purely even sampling over one concatenated timeline can omit
        // short recordings and make a long easy recording dominate selection.
        var rowsBySegment: [Int: [Int]] = [:]
        for row in indices { rowsBySegment[segmentIndex(for: row), default: []].append(row) }
        let orderedSegments = rowsBySegment.keys.sorted()
        include(orderedSegments.map { segment in
            rowsBySegment[segment].map { $0[$0.count / 2] }
        })
        include(orderedSegments.flatMap { segment -> [Int?] in
            guard let rows = rowsBySegment[segment], let first = rows.first, let last = rows.last else { return [] }
            return [first, last]
        })
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

    /// Returns the real immediately preceding action row for transition loss.
    /// It is intentionally separate from the more widely spaced controls paired
    /// with past visual frames. Segment starts use zero state.
    func previousActionBatch(at indices: [Int]) -> Data {
        let rowBytes = manifest.actionValuesPerSample * MemoryLayout<Float>.size
        var result = Data(count: indices.count * rowBytes)
        result.withUnsafeMutableBytes { destination in
            actions.withUnsafeBytes { source in
                guard let destinationBase = destination.baseAddress, let sourceBase = source.baseAddress else { return }
                for (row, index) in indices.enumerated() where index > segmentStart(for: index) {
                    memcpy(
                        destinationBase.advanced(by: row * rowBytes),
                        sourceBase.advanced(by: (index - 1) * rowBytes),
                        rowBytes
                    )
                }
            }
        }
        return result
    }

    func segmentCount(at indices: [Int]) -> Int {
        var segments: Set<Int> = []
        for index in indices { segments.insert(segmentIndex(for: index)) }
        return segments.count
    }

    private func segmentStart(for index: Int) -> Int {
        let index = segmentIndex(for: index)
        return manifest.segments.indices.contains(index) ? manifest.segments[index].start : 0
    }

    private func segmentIndex(for index: Int) -> Int {
        var low = 0, high = manifest.segments.count
        while low < high {
            let mid = (low + high) / 2
            if manifest.segments[mid].start <= index { low = mid + 1 } else { high = mid }
        }
        return max(0, low - 1)
    }

    private func observationIndex(at sample: Int, slot: Int) -> Int {
        observationIndices.withUnsafeBytes { raw in
            let offset = (sample * manifest.observationIndexValuesPerSample + slot) * MemoryLayout<UInt32>.size
            return Int(raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian)
        }
    }

    private func observationSequence(at sample: Int) -> CachedObservationSequence {
        observationIndices.withUnsafeBytes { raw in
            observationSequence(at: sample, bytes: raw)
        }
    }

    private func observationSequences(at samples: [Int]) -> [CachedObservationSequence] {
        observationIndices.withUnsafeBytes { raw in
            samples.map { observationSequence(at: $0, bytes: raw) }
        }
    }

    private func observationSequence(
        at sample: Int,
        bytes raw: UnsafeRawBufferPointer
    ) -> CachedObservationSequence {
        let base = sample * manifest.observationIndexValuesPerSample * MemoryLayout<UInt32>.size
        return CachedObservationSequence(indices: (0..<manifest.observationIndexValuesPerSample).map { slot in
            raw.loadUnaligned(
                fromByteOffset: base + slot * MemoryLayout<UInt32>.size,
                as: UInt32.self
            ).littleEndian
        })
    }
}

private struct PackedRecordingSegment: Sendable {
    let index: Int
    let recordingID: UUID
    let directory: URL
    let samples: Int
    let observations: Int
}

struct RecordingReadinessFailure: Equatable, Sendable {
    let recordingID: UUID
    let recordingName: String
    let reason: String

    var diagnosticSummary: String {
        "\(recordingName) [\(recordingID.uuidString)]: \(reason)"
    }
}

struct TrainingRecordingReadiness: Sendable {
    let recordings: [RecordingItem]
    let failures: [RecordingReadinessFailure]
}

private struct IndexedRecordingReadiness: Sendable {
    let index: Int
    let failure: RecordingReadinessFailure?
}

private final class DatasetPackingProgress: @unchecked Sendable {
    private let lock = NSLock()
    private let names: [String]
    private let durations: [Double]
    private let totalDuration: Double
    private let progress: @Sendable (Double, String) -> Void
    private var fractions: [Double]
    private var completedDuration = 0.0
    private var lastOverall = 0.0

    init(
        recordings: [RecordingItem],
        usableDurations: [Double],
        totalUsableDuration: Double,
        progress: @escaping @Sendable (Double, String) -> Void
    ) {
        names = recordings.map { $0.manifest.name }
        durations = usableDurations
        totalDuration = totalUsableDuration
        self.progress = progress
        fractions = [Double](repeating: 0, count: recordings.count)
    }

    func update(index: Int, fraction rawFraction: Double) {
        lock.lock()
        defer { lock.unlock() }
        guard fractions.indices.contains(index) else { return }
        let fraction = rawFraction.isFinite ? min(1, max(0, rawFraction)) : 0
        let previous = fractions[index]
        guard fraction > previous else { return }
        fractions[index] = fraction
        completedDuration += (fraction - previous) * durations[index]
        let overall = max(lastOverall, min(0.999, completedDuration / totalDuration))
        lastOverall = overall
        progress(
            overall,
            "Packing \(names[index]) in parallel"
        )
    }
}

actor DatasetCacheBuilder {
    static let shared = DatasetCacheBuilder()
    private var preprocessor: VisionPreprocessor?
    private let workspace: WorkspaceStore

    private static let recordingPreflightFraction = 0.05
    private static let minimumFreeSpaceReserve = UInt64(256 * 1_024 * 1_024)

    init(workspace: WorkspaceStore = .shared) { self.workspace = workspace }

    func cache(for profile: AIProfile, recordings: [RecordingItem], progress: @escaping @Sendable (Double, String) -> Void) async throws -> CachedDataset {
        guard !recordings.isEmpty else { throw AgentTrainerError.noData }
        try await workspace.prepare()
        let root = await workspace.cacheDirectory()
        let selectedRecordingCount = recordings.count
        let readiness = try await Self.trainingReadyRecordings(recordings) { completed, total, name in
            let fraction = Double(completed) / Double(max(1, total))
            progress(
                Self.recordingPreflightFraction * fraction,
                "Checking recording \(completed) of \(total): \(name)"
            )
        }
        let recordings = readiness.recordings
        guard !recordings.isEmpty else {
            let firstFailure = readiness.failures.first?.diagnosticSummary
                ?? "No selected recording contained usable synchronized media."
            throw AgentTrainerError.storage(
                "None of the \(selectedRecordingCount) selected recordings can be opened for training. \(firstFailure)"
            )
        }
        if !readiness.failures.isEmpty {
            let shown = readiness.failures.prefix(12).map(\.diagnosticSummary).joined(separator: "\n")
            let omitted = readiness.failures.count - min(12, readiness.failures.count)
            let suffix = omitted > 0 ? "\n… and \(omitted) more" : ""
            AppLog.write(
                .warning,
                category: "Training",
                "Skipped unreadable recordings before packing",
                details: "Using \(recordings.count) of \(selectedRecordingCount). The source packages were left unchanged.\n\(shown)\(suffix)"
            )
            progress(
                Self.recordingPreflightFraction,
                "Skipping \(readiness.failures.count) unreadable recording\(readiness.failures.count == 1 ? "" : "s"); using \(recordings.count)"
            )
        }
        let key = try cacheKey(profile: profile, recordings: recordings)
        let directory = root.appendingPathComponent("\(key).atrcache", isDirectory: true)
        if FileManager.default.fileExists(atPath: directory.appendingPathComponent("manifest.json").path), let cached = try? CachedDataset(directory: directory) {
            progress(1, "Reusing packed dataset cache")
            return cached
        }

        let estimate = Self.estimatedCacheStorage(profile: profile, recordings: recordings)
        try Self.requireFreeSpace(for: estimate, at: root)
        let estimatedSize = ByteCountFormatter.string(
            fromByteCount: Int64(min(estimate.cacheBytes, UInt64(Int64.max))),
            countStyle: .file
        )
        progress(
            Self.recordingPreflightFraction,
            "Preparing about \(estimatedSize) of packed training data"
        )
        if preprocessor == nil { preprocessor = try VisionPreprocessor() }

        let temporary = root.appendingPathComponent(".\(key).\(UUID().uuidString).tmp", isDirectory: true)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        let currentObservations = try BufferedFileWriter(url: temporary.appendingPathComponent("current-observations.bin"), capacity: 8 * 1_024 * 1_024)
        let pastObservations = try BufferedFileWriter(url: temporary.appendingPathComponent("past-observations.bin"), capacity: 8 * 1_024 * 1_024)
        let observationIndices = try BufferedFileWriter(url: temporary.appendingPathComponent("observation-indices.bin"), capacity: 1 * 1_024 * 1_024)
        let frameActions = try BufferedFileWriter(url: temporary.appendingPathComponent("frame-actions.bin"), capacity: 1 * 1_024 * 1_024)
        let actions = try BufferedFileWriter(url: temporary.appendingPathComponent("actions.bin"), capacity: 1 * 1_024 * 1_024)
        var segments: [CacheSegment] = []
        var sampleCount = 0
        var observationCount = 0
        guard let packingPreprocessor = preprocessor else {
            throw AgentTrainerError.model("Metal preprocessing is unavailable.")
        }
        let usableDurations = recordings.map { recording in
            let start = max(0, min(recording.manifest.duration, recording.manifest.trimStart))
            let end = max(start, min(recording.manifest.duration, recording.manifest.trimEnd ?? recording.manifest.duration))
            return end - start
        }
        let totalUsableDuration = max(0.000_001, usableDurations.reduce(0, +))
        let packingProgress: @Sendable (Double, String) -> Void = { fraction, status in
            let bounded = fraction.isFinite ? min(1, max(0, fraction)) : 0
            progress(
                Self.recordingPreflightFraction
                    + (1 - Self.recordingPreflightFraction) * bounded,
                status
            )
        }
        do {
            if recordings.count == 1, let recording = recordings.first {
                packingProgress(0, "Packing \(recording.manifest.name)")
                let appended = try await Self.appendRecording(
                    recording,
                    profile: profile,
                    preprocessor: packingPreprocessor,
                    observationBase: 0,
                    currentObservations: currentObservations,
                    pastObservations: pastObservations,
                    observationIndices: observationIndices,
                    frameActions: frameActions,
                    actions: actions,
                    progress: { fraction in
                        let overall = min(0.999, max(0, fraction))
                        packingProgress(overall, "Packing \(recording.manifest.name)")
                    }
                )
                sampleCount = appended.samples
                observationCount = appended.observations
                if appended.samples > 0 {
                    segments.append(CacheSegment(
                        recordingID: recording.id,
                        start: 0,
                        count: appended.samples
                    ))
                }
            } else {
                let segmentRoot = temporary.appendingPathComponent("segments", isDirectory: true)
                try FileManager.default.createDirectory(
                    at: segmentRoot,
                    withIntermediateDirectories: true
                )
                let tracker = DatasetPackingProgress(
                    recordings: recordings,
                    usableDurations: usableDurations,
                    totalUsableDuration: totalUsableDuration,
                    progress: packingProgress
                )
                let concurrency = Self.recommendedPackingConcurrency(
                    recordings: recordings,
                    physicalMemory: ProcessInfo.processInfo.physicalMemory,
                    processorCount: ProcessInfo.processInfo.activeProcessorCount
                )
                try await withThrowingTaskGroup(of: PackedRecordingSegment.self) { group in
                    let initialCount = min(concurrency, recordings.count)
                    let maximumOutstanding = Self.recommendedPackingLookahead(
                        recordingCount: recordings.count,
                        concurrency: concurrency
                    )
                    for index in 0..<initialCount {
                        let recording = recordings[index]
                        group.addTask(priority: .userInitiated) {
                            try await Self.packRecordingSegment(
                                index: index,
                                recording: recording,
                                profile: profile,
                                preprocessor: packingPreprocessor,
                                root: segmentRoot,
                                progress: { tracker.update(index: index, fraction: $0) }
                            )
                        }
                    }
                    var nextSubmission = initialCount
                    var nextMerge = 0
                    var activeCount = initialCount
                    var pending: [Int: PackedRecordingSegment] = [:]
                    while activeCount > 0 {
                        guard let result = try await group.next() else {
                            throw AgentTrainerError.storage("Parallel recording packing ended before every submitted recording completed.")
                        }
                        activeCount -= 1
                        pending[result.index] = result
                        while let segment = pending.removeValue(forKey: nextMerge) {
                            let start = sampleCount
                            try Self.mergePackedRecordingSegment(
                                segment,
                                observationBase: observationCount,
                                currentObservations: currentObservations,
                                pastObservations: pastObservations,
                                observationIndices: observationIndices,
                                frameActions: frameActions,
                                actions: actions
                            )
                            observationCount += segment.observations
                            sampleCount += segment.samples
                            if segment.samples > 0 {
                                segments.append(CacheSegment(
                                    recordingID: segment.recordingID,
                                    start: start,
                                    count: segment.samples
                                ))
                            }
                            try FileManager.default.removeItem(at: segment.directory)
                            nextMerge += 1
                        }
                        while nextSubmission < recordings.count,
                              activeCount < concurrency,
                              nextSubmission - nextMerge < maximumOutstanding {
                            let index = nextSubmission
                            let recording = recordings[index]
                            group.addTask(priority: .userInitiated) {
                                try await Self.packRecordingSegment(
                                    index: index,
                                    recording: recording,
                                    profile: profile,
                                    preprocessor: packingPreprocessor,
                                    root: segmentRoot,
                                    progress: { tracker.update(index: index, fraction: $0) }
                                )
                            }
                            nextSubmission += 1
                            activeCount += 1
                        }
                    }
                    guard nextMerge == recordings.count, pending.isEmpty else {
                        throw AgentTrainerError.storage("Parallel recording packing did not produce a complete ordered dataset.")
                    }
                }
                try? FileManager.default.removeItem(at: segmentRoot)
            }
            try currentObservations.finish()
            try pastObservations.finish()
            try observationIndices.finish()
            try frameActions.finish()
            try actions.finish()
            let temporal = profile.training.effectiveTemporalVision
            let pastSpec = temporal.pastFrameSpec(from: profile.preprocessing)
            let manifest = DatasetCacheManifest(
                key: key,
                createdAt: Date(),
                preprocessing: profile.preprocessing,
                pastPreprocessing: pastSpec,
                temporalVision: temporal,
                actionFPS: profile.training.actionFPS,
                perceptionFPS: profile.training.perceptionFPS,
                sampleCount: sampleCount,
                observationCount: observationCount,
                currentObservationBytesPerSample: profile.preprocessing.sampleByteCount,
                pastObservationBytesPerSample: temporal.pastFrameCount > 0 ? pastSpec.sampleByteCount : 0,
                actionValuesPerSample: ActionLayout.count,
                segments: segments
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(manifest).write(
                to: temporary.appendingPathComponent("manifest.json"),
                options: .atomic
            )
            if FileManager.default.fileExists(atPath: directory.path) {
                try FileManager.default.removeItem(at: directory)
            }
            try FileManager.default.moveItem(at: temporary, to: directory)
            progress(1, "Dataset cache ready")
            return try CachedDataset(directory: directory)
        } catch {
            try? currentObservations.finish()
            try? pastObservations.finish()
            try? observationIndices.finish()
            try? frameActions.finish()
            try? actions.finish()
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
    }

    private func cacheKey(profile: AIProfile, recordings: [RecordingItem]) throws -> String {
        struct Identity: Encodable {
            let cacheSchema: Int
            let preprocessing: PreprocessingSpec
            let temporalVision: TemporalVisionConfiguration
            let actionFPS: Double
            let perceptionFPS: Double
            let recordings: [RecordingTrainingIdentity]
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let recordingIdentities = try recordings.map(RecordingTrainingIdentity.init(recording:))
        let identity = Identity(
            cacheSchema: TrainingDataContract.schemaVersion,
            preprocessing: profile.preprocessing,
            temporalVision: profile.training.effectiveTemporalVision,
            actionFPS: profile.training.actionFPS,
            perceptionFPS: profile.training.perceptionFPS,
            recordings: recordingIdentities
        )
        let digest = SHA256.hash(data: try encoder.encode(identity))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Checks every selected package before expensive frame packing starts.
    /// Older libraries can contain recordings from an interrupted writer or a
    /// manual copy even though current imports validate media transactionally.
    /// A small metadata-only worker window finds those packages quickly without
    /// opening hundreds of VideoToolbox decoder sessions at once.
    nonisolated static func trainingReadyRecordings(
        _ recordings: [RecordingItem],
        progress: @escaping @Sendable (Int, Int, String) -> Void
    ) async throws -> TrainingRecordingReadiness {
        guard !recordings.isEmpty else {
            return TrainingRecordingReadiness(recordings: [], failures: [])
        }
        let concurrency = min(
            recordings.count,
            max(1, min(8, ProcessInfo.processInfo.activeProcessorCount))
        )
        var failures = [RecordingReadinessFailure?](
            repeating: nil,
            count: recordings.count
        )
        var completed = 0
        try await withThrowingTaskGroup(of: IndexedRecordingReadiness.self) { group in
            for index in 0..<concurrency {
                let recording = recordings[index]
                group.addTask(priority: .userInitiated) {
                    try await validateTrainingRecording(recording, index: index)
                }
            }
            var nextSubmission = concurrency
            while let result = try await group.next() {
                failures[result.index] = result.failure
                completed += 1
                progress(
                    completed,
                    recordings.count,
                    recordings[result.index].manifest.name
                )
                if nextSubmission < recordings.count {
                    let index = nextSubmission
                    let recording = recordings[index]
                    group.addTask(priority: .userInitiated) {
                        try await validateTrainingRecording(recording, index: index)
                    }
                    nextSubmission += 1
                }
            }
        }
        let accepted = recordings.enumerated().compactMap { index, recording in
            failures[index] == nil ? recording : nil
        }
        return TrainingRecordingReadiness(
            recordings: accepted,
            failures: failures.compactMap { $0 }
        )
    }

    private nonisolated static func validateTrainingRecording(
        _ recording: RecordingItem,
        index: Int
    ) async throws -> IndexedRecordingReadiness {
        try Task.checkCancellation()
        let manifest = recording.manifest
        guard manifest.isStructurallyValid, manifest.hostStartNanos > 0 else {
            return IndexedRecordingReadiness(
                index: index,
                failure: RecordingReadinessFailure(
                    recordingID: recording.id,
                    recordingName: manifest.name,
                    reason: "its recording manifest or synchronized timeline is invalid"
                )
            )
        }
        let eventURL = recording.directory.appendingPathComponent(manifest.eventFile)
        do {
            let events = try InputEventReader.mapped(url: eventURL)
            guard events.count == manifest.eventCount else {
                throw AgentTrainerError.storage(
                    "the manifest declares \(manifest.eventCount) input events but the file contains \(events.count)"
                )
            }
        } catch {
            if Task.isCancelled { throw CancellationError() }
            return IndexedRecordingReadiness(
                index: index,
                failure: RecordingReadinessFailure(
                    recordingID: recording.id,
                    recordingName: manifest.name,
                    reason: "its synchronized input cannot be read (\(error.localizedDescription))"
                )
            )
        }

        let videoURL = recording.directory.appendingPathComponent(manifest.videoFile)
        do {
            let values = try videoURL.resourceValues(forKeys: [
                .isRegularFileKey,
                .fileSizeKey
            ])
            guard values.isRegularFile == true, (values.fileSize ?? 0) > 0 else {
                throw AgentTrainerError.storage("the video file is missing or empty")
            }
            let asset = AVURLAsset(url: videoURL)
            async let loadedDuration = asset.load(.duration)
            async let loadedTracks = asset.loadTracks(withMediaType: .video)
            let (durationTime, tracks) = try await (loadedDuration, loadedTracks)
            guard let track = tracks.first else {
                throw AgentTrainerError.storage("the media contains no video track")
            }
            let duration = CMTimeGetSeconds(durationTime)
            let naturalSize = try await track.load(.naturalSize)
            guard duration.isFinite, duration > 0,
                  naturalSize.width.isFinite, naturalSize.height.isFinite,
                  naturalSize.width > 0, naturalSize.height > 0 else {
                throw AgentTrainerError.storage("the video timeline or dimensions are invalid")
            }
            return IndexedRecordingReadiness(index: index, failure: nil)
        } catch {
            if Task.isCancelled { throw CancellationError() }
            return IndexedRecordingReadiness(
                index: index,
                failure: RecordingReadinessFailure(
                    recordingID: recording.id,
                    recordingName: manifest.name,
                    reason: "its video cannot be opened and may be incomplete or damaged (\(error.localizedDescription))"
                )
            )
        }
    }

    struct EstimatedCacheStorage: Equatable, Sendable {
        let cacheBytes: UInt64
        let peakWorkingBytes: UInt64
    }

    nonisolated static func estimatedCacheStorage(
        profile: AIProfile,
        recordings: [RecordingItem]
    ) -> EstimatedCacheStorage {
        let temporal = profile.training.effectiveTemporalVision
        let pastSpec = temporal.pastFrameSpec(from: profile.preprocessing)
        let observationBytes = saturatedAdd(
            UInt64(max(0, profile.preprocessing.sampleByteCount)),
            saturatedAdd(
                temporal.pastFrameCount > 0
                    ? UInt64(max(0, pastSpec.sampleByteCount))
                    : 0,
                temporal.pastFrameCount > 0
                    ? UInt64(ActionLayout.count * MemoryLayout<Float>.size)
                    : 0
            )
        )
        let sampleBytes = saturatedAdd(
            UInt64(ActionLayout.count * MemoryLayout<Float>.size),
            UInt64((1 + max(0, temporal.pastFrameCount)) * MemoryLayout<UInt32>.size)
        )
        var cacheBytes: UInt64 = 0
        var largestShard: UInt64 = 0
        for recording in recordings {
            let start = max(
                0,
                min(recording.manifest.duration, recording.manifest.trimStart)
            )
            let end = max(
                start,
                min(
                    recording.manifest.duration,
                    recording.manifest.trimEnd ?? recording.manifest.duration
                )
            )
            let duration = end - start
            let observations = estimatedTickCount(
                duration: duration,
                rate: profile.training.perceptionFPS
            )
            let samples = estimatedTickCount(
                duration: duration,
                rate: profile.training.actionFPS
            )
            let shardBytes = saturatedAdd(
                saturatedMultiply(observations, observationBytes),
                saturatedMultiply(samples, sampleBytes)
            )
            cacheBytes = saturatedAdd(cacheBytes, shardBytes)
            largestShard = max(largestShard, shardBytes)
        }
        return EstimatedCacheStorage(
            cacheBytes: cacheBytes,
            peakWorkingBytes: saturatedAdd(
                saturatedAdd(cacheBytes, largestShard),
                minimumFreeSpaceReserve
            )
        )
    }

    private nonisolated static func estimatedTickCount(
        duration: Double,
        rate: Double
    ) -> UInt64 {
        let value = max(0, duration) * max(0, rate)
        guard value.isFinite, value < Double(UInt64.max - 1) else {
            return UInt64.max
        }
        return UInt64(value.rounded(.down)) + 1
    }

    private nonisolated static func saturatedAdd(
        _ lhs: UInt64,
        _ rhs: UInt64
    ) -> UInt64 {
        let result = lhs.addingReportingOverflow(rhs)
        return result.overflow ? UInt64.max : result.partialValue
    }

    private nonisolated static func saturatedMultiply(
        _ lhs: UInt64,
        _ rhs: UInt64
    ) -> UInt64 {
        let result = lhs.multipliedReportingOverflow(by: rhs)
        return result.overflow ? UInt64.max : result.partialValue
    }

    private nonisolated static func requireFreeSpace(
        for estimate: EstimatedCacheStorage,
        at directory: URL
    ) throws {
        guard let availableValue = try? directory.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        ).volumeAvailableCapacityForImportantUsage,
        availableValue >= 0 else { return }
        let available = UInt64(availableValue)
        guard estimate.peakWorkingBytes <= available else {
            let requiredText = ByteCountFormatter.string(
                fromByteCount: Int64(min(estimate.peakWorkingBytes, UInt64(Int64.max))),
                countStyle: .file
            )
            let availableText = ByteCountFormatter.string(
                fromByteCount: availableValue,
                countStyle: .file
            )
            throw AgentTrainerError.storage(
                "Packing these recordings needs about \(requiredText) of free working space, but this training-data disk has \(availableText) available. Free space or move Training Data in Settings before starting."
            )
        }
    }

    nonisolated static func recommendedPackingLookahead(
        recordingCount: Int,
        concurrency: Int
    ) -> Int {
        guard recordingCount > 0 else { return 0 }
        let workers = min(recordingCount, max(1, concurrency))
        let doubled = workers.multipliedReportingOverflow(by: 2)
        return min(recordingCount, doubled.overflow ? Int.max : doubled.partialValue)
    }

    nonisolated static func recommendedPackingConcurrency(
        recordings: [RecordingItem],
        physicalMemory: UInt64,
        processorCount: Int
    ) -> Int {
        let largestDecodedFrame = recordings.reduce(UInt64(1)) { current, recording in
            let width = UInt64(max(1, recording.manifest.pixelWidth))
            let height = UInt64(max(1, recording.manifest.pixelHeight))
            let product = width.multipliedReportingOverflow(by: height)
            let pixels = product.overflow ? UInt64.max : product.partialValue
            // Native 4:2:0 is nominally 1.5 bytes/pixel. Two bytes/pixel is a
            // conservative allowance for row alignment and decoder surfaces.
            let bytes = pixels.multipliedReportingOverflow(by: 2)
            return max(current, bytes.overflow ? UInt64.max : bytes.partialValue)
        }
        return recommendedPackingConcurrency(
            recordingCount: recordings.count,
            largestDecodedFrameBytes: largestDecodedFrame,
            physicalMemory: physicalMemory,
            processorCount: processorCount
        )
    }

    nonisolated static func recommendedPackingConcurrency(
        recordingCount: Int,
        largestDecodedFrameBytes: UInt64,
        physicalMemory: UInt64,
        processorCount: Int
    ) -> Int {
        guard recordingCount > 1 else { return 1 }
        let mebibyte = UInt64(1 << 20)
        let decodedSurfaces = largestDecodedFrameBytes.multipliedReportingOverflow(by: 12)
        let surfaceAllowance = decodedSurfaces.overflow
            ? UInt64.max
            : decodedSurfaces.partialValue
        let base = max(192 * mebibyte, surfaceAllowance)
        let withOutputs = base.addingReportingOverflow(64 * mebibyte)
        let perWorker = withOutputs.overflow ? UInt64.max : withOutputs.partialValue
        let memoryBudget = max(perWorker, physicalMemory / 2)
        let memoryBound = max(1, Int(min(UInt64(Int.max), memoryBudget / max(1, perWorker))))
        let cpuBound = max(1, min(4, max(1, processorCount) / 2))
        return min(recordingCount, cpuBound, memoryBound)
    }

    private nonisolated static func packRecordingSegment(
        index: Int,
        recording: RecordingItem,
        profile: AIProfile,
        preprocessor: VisionPreprocessor,
        root: URL,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> PackedRecordingSegment {
        let directory = root.appendingPathComponent(
            String(format: "%06d-%@", index, UUID().uuidString),
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let currentObservations = try BufferedFileWriter(
            url: directory.appendingPathComponent("current-observations.bin"),
            capacity: 8 * 1_024 * 1_024
        )
        let pastObservations = try BufferedFileWriter(
            url: directory.appendingPathComponent("past-observations.bin"),
            capacity: 8 * 1_024 * 1_024
        )
        let observationIndices = try BufferedFileWriter(
            url: directory.appendingPathComponent("observation-indices.bin"),
            capacity: 1 * 1_024 * 1_024
        )
        let frameActions = try BufferedFileWriter(
            url: directory.appendingPathComponent("frame-actions.bin"),
            capacity: 1 * 1_024 * 1_024
        )
        let actions = try BufferedFileWriter(
            url: directory.appendingPathComponent("actions.bin"),
            capacity: 1 * 1_024 * 1_024
        )
        do {
            let result = try await appendRecording(
                recording,
                profile: profile,
                preprocessor: preprocessor,
                observationBase: 0,
                currentObservations: currentObservations,
                pastObservations: pastObservations,
                observationIndices: observationIndices,
                frameActions: frameActions,
                actions: actions,
                progress: progress
            )
            try currentObservations.finish(synchronize: false)
            try pastObservations.finish(synchronize: false)
            try observationIndices.finish(synchronize: false)
            try frameActions.finish(synchronize: false)
            try actions.finish(synchronize: false)
            return PackedRecordingSegment(
                index: index,
                recordingID: recording.id,
                directory: directory,
                samples: result.samples,
                observations: result.observations
            )
        } catch {
            try? currentObservations.finish(synchronize: false)
            try? pastObservations.finish(synchronize: false)
            try? observationIndices.finish(synchronize: false)
            try? frameActions.finish(synchronize: false)
            try? actions.finish(synchronize: false)
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    private nonisolated static func mergePackedRecordingSegment(
        _ segment: PackedRecordingSegment,
        observationBase: Int,
        currentObservations: BufferedFileWriter,
        pastObservations: BufferedFileWriter,
        observationIndices: BufferedFileWriter,
        frameActions: BufferedFileWriter,
        actions: BufferedFileWriter
    ) throws {
        func mapped(_ name: String) throws -> Data {
            try Data(
                contentsOf: segment.directory.appendingPathComponent(name),
                options: .alwaysMapped
            )
        }
        try currentObservations.append(mapped("current-observations.bin"))
        try pastObservations.append(mapped("past-observations.bin"))
        try observationIndices.appendRebasedObservationIndices(
            mapped("observation-indices.bin"),
            observationBase: observationBase
        )
        try frameActions.append(mapped("frame-actions.bin"))
        try actions.append(mapped("actions.bin"))
    }

    private nonisolated static func appendRecording(
        _ recording: RecordingItem,
        profile: AIProfile,
        preprocessor: VisionPreprocessor,
        observationBase: Int,
        currentObservations: BufferedFileWriter,
        pastObservations: BufferedFileWriter,
        observationIndices: BufferedFileWriter,
        frameActions: BufferedFileWriter,
        actions: BufferedFileWriter,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> (samples: Int, observations: Int) {
        guard recording.manifest.isStructurallyValid, recording.manifest.hostStartNanos > 0 else {
            throw AgentTrainerError.storage("\(recording.manifest.name) has an invalid recording timeline or manifest.")
        }
        let temporal = try profile.training.effectiveTemporalVision.validated(
            current: profile.preprocessing,
            cachedEmbeddingWidth: profile.training.architecture.visualEmbedding
        )
        let pastSpec = temporal.pastFrameSpec(from: profile.preprocessing)
        let events: InputEventReader.MappedEvents
        do {
            events = try InputEventReader.mapped(
                url: recording.directory.appendingPathComponent(
                    recording.manifest.eventFile
                )
            )
        } catch {
            throw AgentTrainerError.storage(
                "\(recording.manifest.name)'s synchronized input cannot be opened for training: \(error.localizedDescription)"
            )
        }
        let asset = AVURLAsset(
            url: recording.directory.appendingPathComponent(
                recording.manifest.videoFile
            )
        )
        let track: AVAssetTrack
        do {
            guard let loadedTrack = try await asset.loadTracks(withMediaType: .video).first else {
                throw AgentTrainerError.storage(
                    "\(recording.manifest.name) contains no readable video track."
                )
            }
            track = loadedTrack
        } catch let error as AgentTrainerError {
            throw error
        } catch {
            throw AgentTrainerError.storage(
                "\(recording.manifest.name)'s video cannot be opened for training and may be incomplete or damaged: \(error.localizedDescription)"
            )
        }
        let reader: AVAssetReader
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw AgentTrainerError.storage(
                "\(recording.manifest.name)'s video reader could not be created: \(error.localizedDescription)"
            )
        }
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
        guard reader.canAdd(output) else {
            throw AgentTrainerError.storage(
                "\(recording.manifest.name)'s video cannot be decoded for training."
            )
        }
        reader.add(output)
        guard reader.startReading() else {
            throw AgentTrainerError.storage(
                "\(recording.manifest.name)'s video could not be opened for training: \(reader.error?.localizedDescription ?? "unknown media error")"
            )
        }

        let actionInterval = 1 / max(0.0001, profile.training.actionFPS)
        let perceptionInterval = 1 / max(0.0001, profile.training.perceptionFPS)
        var nextPerception = trainingStart
        var firstPTS: CMTime?
        var observationTimes: [Double] = []
        observationTimes.reserveCapacity(max(1, Int((trainingEnd - trainingStart) * profile.training.perceptionFPS)))
        let usableDuration = trainingEnd - trainingStart
        let progressInterval = max(2, usableDuration / 100)
        var nextProgressTime = trainingStart
        let hasTemporalMemory = temporal.pastFrameCount > 0
        let sharesPackedRepresentation = hasTemporalMemory && pastSpec == profile.preprocessing
        let maximumPackedPipelineBytes = 64 * 1_024 * 1_024
        let bytesPerPendingObservation = profile.preprocessing.sampleByteCount
            + (hasTemporalMemory && !sharesPackedRepresentation ? pastSpec.sampleByteCount : 0)
        let packedPipelineDepth = max(
            1,
            min(8, maximumPackedPipelineBytes / max(1, bytesPerPendingObservation))
        )
        var pendingPackedFrames: [(
            current: VisionPreprocessor.PendingPackedFrame,
            past: VisionPreprocessor.PendingPackedFrame?,
            repetitions: Int
        )] = []
        pendingPackedFrames.reserveCapacity(packedPipelineDepth)
        func writeOldestPackedFrame() throws {
            let frame = pendingPackedFrames.removeFirst()
            try frame.current.withPackedBytes { bytes in
                try currentObservations.appendRepeated(bytes, count: frame.repetitions)
                if sharesPackedRepresentation {
                    try pastObservations.appendRepeated(bytes, count: frame.repetitions)
                }
            }
            if let past = frame.past {
                try past.withPackedBytes { bytes in
                    try pastObservations.appendRepeated(bytes, count: frame.repetitions)
                }
            }
        }

        func appendPackedObservations(_ buffer: CVPixelBuffer, repetitions: Int) throws {
            guard repetitions > 0 else { return }
            let requestedSpecs = hasTemporalMemory && !sharesPackedRepresentation
                ? [profile.preprocessing, pastSpec]
                : [profile.preprocessing]
            let jobs = try preprocessor.submit(buffer, specs: requestedSpecs)
            guard let current = jobs.first else {
                throw AgentTrainerError.model("The vision preprocessing pipeline returned no current frame.")
            }
            pendingPackedFrames.append((
                current: current,
                past: jobs.count > 1 ? jobs[1] : nil,
                repetitions: repetitions
            ))
            if pendingPackedFrames.count >= packedPipelineDepth {
                try writeOldestPackedFrame()
            }
        }

        var lastDecodedBuffer: CVPixelBuffer?
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
            if let previous = lastDecodedBuffer {
                // A compressed frame remains visually current until the next
                // decoded frame's PTS. Materialize every configured perception
                // interval in that gap from the held frame. This is essential for
                // static screens, where ScreenCaptureKit may emit only idle status.
                let heldStart = observationTimes.count
                while nextPerception < min(trainingEnd, t) - 0.000_001 {
                    observationTimes.append(min(trainingEnd, max(trainingStart, nextPerception)))
                    nextPerception += perceptionInterval
                }
                try appendPackedObservations(
                    previous,
                    repetitions: observationTimes.count - heldStart
                )
            }
            if nextPerception <= min(trainingEnd, t) + 0.000_001 {
                observationTimes.append(min(trainingEnd, max(trainingStart, nextPerception)))
                try appendPackedObservations(buffer, repetitions: 1)
                nextPerception += perceptionInterval
            }
            lastDecodedBuffer = buffer
        }
        if reader.status == .failed {
            throw AgentTrainerError.storage(
                "Decoding \(recording.manifest.name) failed while building the packed cache: \(reader.error?.localizedDescription ?? "unknown media error")"
            )
        }
        if let lastDecodedBuffer {
            let heldStart = observationTimes.count
            while nextPerception <= trainingEnd + 0.000_001 {
                observationTimes.append(min(trainingEnd, max(trainingStart, nextPerception)))
                nextPerception += perceptionInterval
            }
            try appendPackedObservations(
                lastDecodedBuffer,
                repetitions: observationTimes.count - heldStart
            )
        }
        while !pendingPackedFrames.isEmpty { try writeOldestPackedFrame() }
        guard !observationTimes.isEmpty else {
            progress(1)
            return (0, 0)
        }

        let initialPointer = events.first { $0.kind == .mouseMove || $0.kind == .mouseButton || $0.kind == .scroll }
        let trainingStartNanos = try absoluteHostNanos(recording.manifest, seconds: trainingStart)
        func makePrimedAccumulator() -> (ActionAccumulator, Int) {
            var accumulator = ActionAccumulator(manifest: recording.manifest, initialPointer: initialPointer)
            var eventIndex = 0
            while eventIndex < events.count, events[eventIndex].timestampNanos <= trainingStartNanos {
                accumulator.consume(events[eventIndex])
                eventIndex += 1
            }
            // Retain held state and pointer position, but discard all additive
            // movement, scroll, and tap edges from before the usable trim.
            accumulator.endTick()
            return (accumulator, eventIndex)
        }

        let primed = makePrimedAccumulator()
        if hasTemporalMemory {
            // Build one comprehensive control row for every perception
            // interval only when a temporal branch can consume it.
            var frameAccumulator = primed.0
            var frameEventIndex = primed.1
            for observation in observationTimes.indices {
                let intervalEnd = observation + 1 < observationTimes.count
                    ? min(trainingEnd, observationTimes[observation + 1])
                    : trainingEnd
                let endNanos = try absoluteHostNanos(recording.manifest, seconds: intervalEnd)
                while frameEventIndex < events.count, events[frameEventIndex].timestampNanos < endNanos {
                    frameAccumulator.consume(events[frameEventIndex])
                    frameEventIndex += 1
                }
                try frameAccumulator.withActionBytes { try frameActions.append($0) }
                frameAccumulator.endTick()
            }
        }

        // Action targets retain their independently configured rate. Every row
        // maps its current perception plus the exact requested causal frame
        // spacing; unavailable leading slots use the reserved sentinel.
        var targetAccumulator = primed.0
        var targetEventIndex = primed.1
        var nextAction = trainingStart
        var currentLocalObservation = 0
        var sampleCount = 0
        while nextAction <= trainingEnd + 0.000_001 {
            while currentLocalObservation + 1 < observationTimes.count,
                  observationTimes[currentLocalObservation + 1] <= nextAction + 0.000_001 {
                currentLocalObservation += 1
            }
            let targetEnd = min(trainingEnd, nextAction + actionInterval)
            let targetEndNanos = try absoluteHostNanos(recording.manifest, seconds: targetEnd)
            while targetEventIndex < events.count, events[targetEventIndex].timestampNanos < targetEndNanos {
                targetAccumulator.consume(events[targetEventIndex])
                targetEventIndex += 1
            }
            let currentGlobal = observationBase + currentLocalObservation
            guard let current = UInt32(exactly: currentGlobal), current != UInt32.max else {
                throw AgentTrainerError.storage("The dataset contains too many perception frames for its index format.")
            }
            try observationIndices.appendLittleEndian(current)
            for frame in 0..<temporal.pastFrameCount {
                let distance = temporal.frameSpacing * (temporal.pastFrameCount - frame)
                let localPast = currentLocalObservation - distance
                if localPast >= 0 {
                    guard let past = UInt32(exactly: observationBase + localPast), past != UInt32.max else {
                        throw AgentTrainerError.storage("The dataset contains too many perception frames for its index format.")
                    }
                    try observationIndices.appendLittleEndian(past)
                } else {
                    try observationIndices.appendLittleEndian(UInt32.max)
                }
            }
            try targetAccumulator.withActionBytes { try actions.append($0) }
            targetAccumulator.endTick()
            sampleCount += 1
            nextAction += actionInterval
        }
        progress(1)
        return (sampleCount, observationTimes.count)
    }

    private nonisolated static func absoluteHostNanos(_ manifest: RecordingManifest, seconds: Double) throws -> UInt64 {
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
final class BufferedFileWriter {
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
        guard !closed else { throw AgentTrainerError.storage("The dataset cache writer was already closed.") }
        if buffer.isEmpty, data.count >= capacity {
            try handle.write(contentsOf: data)
        } else {
            try data.withUnsafeBytes { try append($0) }
        }
    }

    func append(_ bytes: UnsafeRawBufferPointer) throws {
        guard !closed else { throw AgentTrainerError.storage("The dataset cache writer was already closed.") }
        buffer.append(contentsOf: bytes)
        if buffer.count >= capacity { try flush() }
    }

    func appendRepeated(_ bytes: UnsafeRawBufferPointer, count repetitions: Int) throws {
        guard !closed else { throw AgentTrainerError.storage("The dataset cache writer was already closed.") }
        guard repetitions > 0, !bytes.isEmpty else { return }
        var remaining = repetitions
        while remaining > 0 {
            if buffer.count + bytes.count > capacity, !buffer.isEmpty { try flush() }
            if bytes.count >= capacity {
                try handle.write(contentsOf: Data(bytes))
                remaining -= 1
                continue
            }
            let copies = min(remaining, max(1, (capacity - buffer.count) / bytes.count))
            let destinationOffset = buffer.count
            buffer.count += copies * bytes.count
            buffer.withUnsafeMutableBytes { destination in
                guard let destinationBase = destination.baseAddress,
                      let sourceBase = bytes.baseAddress else { return }
                for copy in 0..<copies {
                    memcpy(
                        destinationBase.advanced(by: destinationOffset + copy * bytes.count),
                        sourceBase,
                        bytes.count
                    )
                }
            }
            remaining -= copies
            if buffer.count >= capacity { try flush() }
        }
    }

    func appendRebasedObservationIndices(_ data: Data, observationBase: Int) throws {
        guard data.count.isMultiple(of: MemoryLayout<UInt32>.size),
              let base = UInt32(exactly: observationBase) else {
            throw AgentTrainerError.storage("A packed recording has an invalid observation index file.")
        }
        var adjusted = Data(data)
        try adjusted.withUnsafeMutableBytes { raw in
            for offset in stride(from: 0, to: raw.count, by: MemoryLayout<UInt32>.size) {
                let value = raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian
                guard value != UInt32.max else { continue }
                let rebased = value.addingReportingOverflow(base)
                guard !rebased.overflow, rebased.partialValue != UInt32.max else {
                    throw AgentTrainerError.storage("The dataset contains too many perception frames for its index format.")
                }
                raw.storeBytes(
                    of: rebased.partialValue.littleEndian,
                    toByteOffset: offset,
                    as: UInt32.self
                )
            }
        }
        try append(adjusted)
    }

    func appendLittleEndian<T: FixedWidthInteger>(_ value: T) throws {
        var little = value.littleEndian
        try Swift.withUnsafeBytes(of: &little) { try append($0) }
    }

    func finish(synchronize: Bool = true) throws {
        guard !closed else { return }
        try flush()
        // Per-recording shard files are immediately merged into the durable
        // cache and then removed. Avoiding five redundant fsyncs per shard
        // keeps parallel packing throughput high; the final cache writers are
        // still synchronized before the atomic directory move.
        if synchronize { try handle.synchronize() }
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
