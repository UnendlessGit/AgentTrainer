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
    let sampleToVision: [Int32]

    var reuseRatio: Double {
        Double(sampleToVision.count) / Double(max(1, uniqueSequences.count))
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
        guard decoded.sampleCount >= 0,
              decoded.observationCount >= 0,
              decoded.sampleCount == 0 || decoded.observationCount > 0,
              decoded.observationCount <= Int(UInt32.max),
              decoded.actionFPS.isFinite, decoded.actionFPS > 0,
              decoded.perceptionFPS.isFinite, decoded.perceptionFPS > 0,
              decoded.pastPreprocessing == expectedPastSpec,
              decoded.currentObservationBytesPerSample == decoded.preprocessing.sampleByteCount,
              decoded.pastObservationBytesPerSample == decoded.pastPreprocessing.sampleByteCount,
              decoded.currentObservationBytesPerSample > 0,
              decoded.pastObservationBytesPerSample > 0,
              decoded.actionValuesPerSample == ActionLayout.count else {
            throw AgentTrainerError.storage("The dataset cache manifest is invalid.")
        }

        let currentObservationSize = decoded.observationCount.multipliedReportingOverflow(by: decoded.currentObservationBytesPerSample)
        let pastObservationSize = decoded.observationCount.multipliedReportingOverflow(by: decoded.pastObservationBytesPerSample)
        let mappingValueCount = decoded.sampleCount.multipliedReportingOverflow(by: decoded.observationIndexValuesPerSample)
        let mappingSize = mappingValueCount.partialValue.multipliedReportingOverflow(by: MemoryLayout<UInt32>.size)
        let actionValueCount = decoded.sampleCount.multipliedReportingOverflow(by: decoded.actionValuesPerSample)
        let actionSize = actionValueCount.partialValue.multipliedReportingOverflow(by: MemoryLayout<Float>.size)
        let frameActionValueCount = decoded.observationCount.multipliedReportingOverflow(by: decoded.actionValuesPerSample)
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
        Self.adviseRandomAccess(loadedCurrentObservations)
        Self.adviseRandomAccess(loadedPastObservations)
        Self.adviseRandomAccess(loadedObservationIndices)
        Self.adviseRandomAccess(loadedFrameActions)
        Self.adviseRandomAccess(loadedActions)
        manifest = decoded
        currentObservations = loadedCurrentObservations
        pastObservations = loadedPastObservations
        observationIndices = loadedObservationIndices
        frameActions = loadedFrameActions
        actions = loadedActions
    }

    var count: Int { manifest.sampleCount }

    private static func adviseRandomAccess(_ data: Data) {
        data.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress, !bytes.isEmpty else { return }
            // Epoch orders are deliberately shuffled. Telling the VM subsystem
            // avoids spending I/O bandwidth on sequential read-ahead pages that
            // the next batch is unlikely to consume.
            _ = madvise(
                UnsafeMutableRawPointer(mutating: baseAddress),
                bytes.count,
                MADV_RANDOM
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
            return CachedVisionBatchPlan(uniqueSequences: [], sampleToVision: [])
        }
        var uniqueSequences: [CachedObservationSequence] = []
        uniqueSequences.reserveCapacity(indices.count)
        var sequenceToIndex: [CachedObservationSequence: Int32] = [:]
        sequenceToIndex.reserveCapacity(indices.count)
        var sampleToVision: [Int32] = []
        sampleToVision.reserveCapacity(indices.count)
        for sampleIndex in indices {
            let sequence = observationSequence(at: sampleIndex)
            if let existing = sequenceToIndex[sequence] {
                sampleToVision.append(existing)
            } else {
                precondition(uniqueSequences.count < Int(Int32.max), "A vision batch has too many distinct frame sequences.")
                let index = Int32(uniqueSequences.count)
                uniqueSequences.append(sequence)
                sequenceToIndex[sequence] = index
                sampleToVision.append(index)
            }
        }
        return CachedVisionBatchPlan(
            uniqueSequences: uniqueSequences,
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
        let plan = suppliedPlan ?? visionBatchPlan(at: indices)
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
        populateTrainingBatch(
            at: indices,
            observationSequences: indices.map(observationSequence(at:)),
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
        let batchCount = indices.count
        let pastFrameCount = manifest.temporalVision.pastFrameCount
        let currentBytes = manifest.currentObservationBytesPerSample
        let pastBytes = manifest.pastObservationBytesPerSample
        let actionRowBytes = manifest.actionValuesPerSample * MemoryLayout<Float>.size
        precondition(currentDestination.count == observationSequences.count * currentBytes)
        precondition(pastDestination.count == observationSequences.count * pastFrameCount * pastBytes)
        precondition(controlDestination.count == observationSequences.count * pastFrameCount * actionRowBytes)
        precondition(actionDestination.count == batchCount * 2 * actionRowBytes)
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
                            let sourceObservation = mapped == UInt32.max ? current : Int(mapped)
                            memcpy(
                                pastDestinationBase.advanced(by: (visionRow * pastFrameCount + frame) * pastBytes),
                                pastSourceBase.advanced(by: sourceObservation * pastBytes),
                                pastBytes
                            )
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
            let base = sample * manifest.observationIndexValuesPerSample * MemoryLayout<UInt32>.size
            return CachedObservationSequence(indices: (0..<manifest.observationIndexValuesPerSample).map { slot in
                raw.loadUnaligned(
                    fromByteOffset: base + slot * MemoryLayout<UInt32>.size,
                    as: UInt32.self
                ).littleEndian
            })
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
        let currentObservations = try BufferedFileWriter(url: temporary.appendingPathComponent("current-observations.bin"), capacity: 8 * 1_024 * 1_024)
        let pastObservations = try BufferedFileWriter(url: temporary.appendingPathComponent("past-observations.bin"), capacity: 8 * 1_024 * 1_024)
        let observationIndices = try BufferedFileWriter(url: temporary.appendingPathComponent("observation-indices.bin"), capacity: 1 * 1_024 * 1_024)
        let frameActions = try BufferedFileWriter(url: temporary.appendingPathComponent("frame-actions.bin"), capacity: 1 * 1_024 * 1_024)
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
                    currentObservations: currentObservations,
                    pastObservations: pastObservations,
                    observationIndices: observationIndices,
                    frameActions: frameActions,
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
            try currentObservations.finish(); try pastObservations.finish()
            try observationIndices.finish(); try frameActions.finish(); try actions.finish()
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
                pastObservationBytesPerSample: pastSpec.sampleByteCount,
                actionValuesPerSample: ActionLayout.count,
                segments: segments
            )
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]; encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(manifest).write(to: temporary.appendingPathComponent("manifest.json"), options: .atomic)
            if FileManager.default.fileExists(atPath: directory.path) { try FileManager.default.removeItem(at: directory) }
            try FileManager.default.moveItem(at: temporary, to: directory)
            progress(1, "Dataset cache ready")
            return try CachedDataset(directory: directory)
        } catch {
            try? currentObservations.finish(); try? pastObservations.finish()
            try? observationIndices.finish(); try? frameActions.finish(); try? actions.finish()
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
            let recordings: [RecordingManifest]
        }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]; encoder.dateEncodingStrategy = .iso8601
        let identity = Identity(cacheSchema: TrainingDataContract.schemaVersion, preprocessing: profile.preprocessing, temporalVision: profile.training.effectiveTemporalVision, actionFPS: profile.training.actionFPS, perceptionFPS: profile.training.perceptionFPS, recordings: recordings.map(\.manifest))
        let digest = SHA256.hash(data: try encoder.encode(identity))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private func appendRecording(
        _ recording: RecordingItem,
        profile: AIProfile,
        observationBase: Int,
        currentObservations: BufferedFileWriter,
        pastObservations: BufferedFileWriter,
        observationIndices: BufferedFileWriter,
        frameActions: BufferedFileWriter,
        actions: BufferedFileWriter,
        progress: (Double) -> Void
    ) async throws -> (samples: Int, observations: Int) {
        guard let preprocessor else { throw AgentTrainerError.model("Metal preprocessing is unavailable.") }
        guard recording.manifest.isStructurallyValid, recording.manifest.hostStartNanos > 0 else {
            throw AgentTrainerError.storage("\(recording.manifest.name) has an invalid recording timeline or manifest.")
        }
        let temporal = try profile.training.effectiveTemporalVision.validated(current: profile.preprocessing)
        let pastSpec = temporal.pastFrameSpec(from: profile.preprocessing)
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
        var nextPerception = trainingStart
        var firstPTS: CMTime?
        var observationTimes: [Double] = []
        observationTimes.reserveCapacity(max(1, Int((trainingEnd - trainingStart) * profile.training.perceptionFPS)))
        let usableDuration = trainingEnd - trainingStart
        let progressInterval = max(2, usableDuration / 100)
        var nextProgressTime = trainingStart
        let maximumPackedPipelineBytes = 64 * 1_024 * 1_024
        let bytesPerPendingObservation = profile.preprocessing.sampleByteCount + pastSpec.sampleByteCount
        let packedPipelineDepth = max(
            1,
            min(3, maximumPackedPipelineBytes / max(1, bytesPerPendingObservation))
        )
        var pendingPackedFrames: [(current: VisionPreprocessor.PendingPackedFrame, past: VisionPreprocessor.PendingPackedFrame)] = []
        pendingPackedFrames.reserveCapacity(packedPipelineDepth)
        func writeOldestPackedFrame() throws {
            let frame = pendingPackedFrames.removeFirst()
            try frame.current.withPackedBytes { try currentObservations.append($0) }
            try frame.past.withPackedBytes { try pastObservations.append($0) }
        }

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
            if observationTimes.isEmpty || t + 0.000_001 >= nextPerception {
                pendingPackedFrames.append((
                    current: try preprocessor.submit(buffer, spec: profile.preprocessing),
                    past: try preprocessor.submit(buffer, spec: pastSpec)
                ))
                if pendingPackedFrames.count >= packedPipelineDepth {
                    try writeOldestPackedFrame()
                }
                observationTimes.append(min(trainingEnd, max(trainingStart, t)))
                while nextPerception <= t { nextPerception += perceptionInterval }
            }
        }
        if reader.status == .failed { throw reader.error ?? AgentTrainerError.storage("Video decoding failed while building the cache.") }
        while !pendingPackedFrames.isEmpty { try writeOldestPackedFrame() }
        guard !observationTimes.isEmpty else { progress(1); return (0, 0) }

        let initialPointer = events.first { $0.kind == .mouseMove || $0.kind == .mouseButton || $0.kind == .scroll }
        let trainingStartNanos = try absoluteHostNanos(recording.manifest, seconds: trainingStart)
        func primedAccumulator() -> (ActionAccumulator, Int) {
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

        // Build one comprehensive control row for every perception interval.
        // These rows include held state, sub-frame taps, mouse deltas, buttons,
        // scroll, keys, and modifiers and are later paired with past frames.
        var (frameAccumulator, frameEventIndex) = primedAccumulator()
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

        // Action targets retain their independently configured rate. Every row
        // maps its current perception plus the exact requested causal frame
        // spacing; unavailable leading slots use the reserved sentinel.
        var (targetAccumulator, targetEventIndex) = primedAccumulator()
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
