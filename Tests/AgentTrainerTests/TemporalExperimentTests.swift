import CoreGraphics
import Foundation
import MLX
import MLXNN
import XCTest
@testable import AgentTrainer

/// Opt-in, local-only experiment. Source recording packages are read directly;
/// caches, checkpoints, and aggregate reports live in an isolated workspace.
final class TemporalExperimentTests: XCTestCase {
    func testBladeBallTemporalAblation() async throws {
        guard let path = ProcessInfo.processInfo.environment["AGENTTRAINER_TEMPORAL_DATA_ROOT"] else {
            throw XCTSkip("Run ./benchmark-temporal.sh to execute the two 20-epoch Blade Ball arms.")
        }
        executionTimeAllowance = 4 * 60 * 60
        let source = URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL
        let workspace = await WorkspaceStore.shared.root.standardizedFileURL
        guard workspace != WorkspaceStore.defaultRoot,
              !workspace.path.hasPrefix(source.path + "/"), !source.path.hasPrefix(workspace.path + "/") else {
            throw AgentTrainerError.storage("The timing experiment requires a separate AGENTTRAINER_WORKSPACE_ROOT.")
        }
        try await WorkspaceStore.shared.prepare()
        let folders = try JSONDecoder.experiment.decode([RecordingFolder].self,
            from: Data(contentsOf: source.appendingPathComponent("recording-folders.json")))
        let folder = try XCTUnwrap(folders.first { $0.name.caseInsensitiveCompare("Blade Ball") == .orderedSame })
        let packages = try FileManager.default.contentsOfDirectory(at: source.appendingPathComponent("Recordings"), includingPropertiesForKeys: nil)
        let recordings = try packages.filter { $0.pathExtension == "atrrecord" }.compactMap { package -> RecordingItem? in
            let manifest = try JSONDecoder.experiment.decode(RecordingManifest.self, from: Data(contentsOf: package.appendingPathComponent("manifest.json")))
            return manifest.folderID == folder.id ? RecordingItem(manifest: manifest, directory: package) : nil
        }.sorted { $0.id.uuidString < $1.id.uuidString }
        XCTAssertEqual(recordings.count, 145, "Freeze and review the experiment selection if the source folder changes.")
        var reports: [ExperimentReport] = []
        for (name, ablated, id) in [
            ("timing-ablated", true, "00000000-0000-0000-0000-000000002501"),
            ("timing-enabled", false, "00000000-0000-0000-0000-000000002502")
        ] {
            var profile = AIProfile.fresh(name: "Blade Ball • \(name)")
            profile.id = UUID(uuidString: id)!
            profile.preprocessing = PreprocessingSpec(width: 128, height: 128)
            profile.training.perceptionFPS = 60; profile.training.actionFPS = 60
            profile.training.binaryFocalGamma = 0; profile.training.epochs = 20
            profile.training.maximumSteps = 0
            profile.recordingIDs = recordings.map(\.id)
            if let existing = await WorkspaceStore.shared.listProfiles().first(where: { $0.id == profile.id }) {
                profile.activeVersionID = existing.activeVersionID
                profile.trainingProgress = existing.trainingProgress
            }
            try await WorkspaceStore.shared.saveProfile(profile)
            let stateURL = await WorkspaceStore.shared.checkpointDirectory(profileID: profile.id).appendingPathComponent("state.json")
            let state = (try? Data(contentsOf: stateURL)).flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
            let epoch = state?["epoch"] as? Int ?? 0
            guard epoch <= 20 else { throw AgentTrainerError.model("An experiment arm exceeds the approved 20 epochs.") }
            let progress = ExperimentProgress(name: name)
            if epoch < 20 {
                let engine = TrainingEngine(ablateActionTiming: ablated)
                let completion: TrainingCompletion = try await withCheckedThrowingContinuation { continuation in
                    engine.start(profile: profile, recordings: recordings,
                        runSettings: TrainingRunSettings(maximumSteps: 0, autosaveSteps: 10_000),
                        metrics: { metrics, status in progress.consume(metrics, status: status) },
                        completion: { continuation.resume(with: $0) })
                }
                profile = completion.profile
                XCTAssertTrue(completion.completed)
            }
            let finalState = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(contentsOf: stateURL)) as? [String: Any])
            XCTAssertEqual(finalState["epoch"] as? Int, 20)
            XCTAssertEqual(finalState["batchOffset"] as? Int, 0)
            let versionID = try XCTUnwrap(profile.activeVersionID)
            let loadedVersion = await WorkspaceStore.shared.version(profileID: profile.id, versionID: versionID)
            let version = try XCTUnwrap(loadedVersion)
            let report = try await evaluate(profile: profile, version: version, recordings: recordings, ablated: ablated, name: name)
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: workspace.appendingPathComponent("\(name).json"), options: .atomic)
            print("TEMPORAL_EXPERIMENT_RESULT \(String(data: try encoder.encode(report), encoding: .utf8)!)")
            reports.append(report)
        }
        XCTAssertLessThanOrEqual(reports[1].loss, reports[0].loss * 1.01,
            "Timing must not regress immediate held-out loss beyond the approved 1% gate.")
    }

    private func evaluate(profile: AIProfile, version: ModelVersionManifest, recordings: [RecordingItem], ablated: Bool, name: String) async throws -> ExperimentReport {
        let dataset = try await DatasetCacheBuilder.shared.cache(for: profile, recordings: recordings) { _, _ in }
        let engine = TrainingEngine()
        let split = engine.splitIndices(dataset: dataset, fraction: profile.training.validationSplit,
            seed: profile.training.seed, channels: profile.channels, restrictions: profile.effectiveRestrictions)
        let model = AgentPolicy(profile: profile, contextSchema: dataset.manifest.effectiveContextSchema)
        let weights = await WorkspaceStore.shared.versionDirectory(profileID: profile.id, versionID: version.id).appendingPathComponent(version.weightsFile)
        try model.loadWeights(from: weights); model.train(false)
        let positiveWeights = dataset.positiveClassWeights(at: split.train, restrictions: profile.effectiveRestrictions)
        let classWeights = MLXArray(positiveWeights)
        let frameCount = profile.training.effectiveTemporalVision.pastFrameCount
        let pastSpec = profile.training.effectiveTemporalVision.pastFrameSpec(from: profile.preprocessing)
        let evaluate = compile(inputs: [model]) { arrays in
            let temporal = model.temporalFeatures(
                currentImages: VisionPreprocessor.mlxTensor(arrays[0], spec: profile.preprocessing),
                pastImages: VisionPreprocessor.mlxPastFrameTensor(arrays[1], spec: pastSpec),
                pastControls: arrays[2], currentTiming: arrays[5], pastTiming: arrays[6])
            let logits = model.logits(temporalFeatures: temporal,
                contextValues: dataset.manifest.effectiveContextSchema.isEmpty ? nil : arrays[7])
            return [model.loss(logits: logits, targets: arrays[3], positiveWeights: classWeights,
                previousTargets: arrays[4]), model.activatedPredictions(logits: logits)]
        }
        let active = ActionLayout.learnableBinaryIndices(channels: profile.channels, restrictions: profile.effectiveRestrictions)
            .filter { positiveWeights[$0] > 0 }
        var truePositive = 0, falsePositive = 0, falseNegative = 0
        var lossSum = 0.0, relativeError = 0.0, scrollError = 0.0, continuousCount = 0
        var predictedEdges: [UUID: [Int: Edges]] = [:]
        var previous = [Int: Bool](); var previousRecording: UUID?
        let started = CACurrentMediaTime()
        for start in stride(from: 0, to: split.validation.count, by: profile.training.batchSize) {
            let rows = Array(split.validation[start..<min(split.validation.count, start + profile.training.batchSize)])
            let result: (Float, [Float]) = autoreleasepool {
                let batch = dataset.trainingBatch(at: rows)
                var timing = dataset.timingBatch(at: rows)
                if ablated {
                    timing.current.withUnsafeMutableBytes { TemporalInputContract.ablate($0.bindMemory(to: Float.self), rows: rows.count) }
                    timing.past.withUnsafeMutableBytes { TemporalInputContract.ablate($0.bindMemory(to: Float.self), rows: rows.count * frameCount) }
                }
                let actionRows = MLXArray(batch.actionRows, [rows.count, 2, ActionLayout.count], type: Float.self)
                let arrays = [
                    MLXArray(batch.packedCurrentObservations, [rows.count, profile.preprocessing.sampleByteCount], dtype: .uint8),
                    MLXArray(batch.packedPastObservations, [rows.count, frameCount, pastSpec.sampleByteCount], dtype: .uint8),
                    MLXArray(batch.pastControlRows, [rows.count, frameCount, ActionLayout.count], type: Float.self),
                    actionRows[0..., 0, 0...], actionRows[0..., 1, 0...],
                    MLXArray(timing.current, [rows.count, TemporalInputContract.valueCount], type: Float.self),
                    MLXArray(timing.past, [rows.count, frameCount, TemporalInputContract.valueCount], type: Float.self),
                    dataset.manifest.effectiveContextSchema.isEmpty ? MLXArray.zeros([rows.count, 1])
                        : MLXArray(dataset.contextBatch(at: rows), [rows.count, dataset.manifest.effectiveContextSchema.inputValueCount], type: Float.self)
                ]
                let output = evaluate(arrays); MLX.eval(output)
                return (output[0].item(Float.self), output[1].asArray(Float.self))
            }
            lossSum += Double(result.0) * Double(rows.count)
            for (row, index) in rows.enumerated() {
                let segment = dataset.manifest.segments.last { $0.start <= index }!
                if previousRecording != segment.recordingID { previous.removeAll(); previousRecording = segment.recordingID }
                let target = dataset.action(at: index)
                let time = Double(dataset.decisionTime(at: index).query) / 1e9
                for output in active {
                    let prediction = result.1[row * ActionLayout.count + output] >= 0.5
                    let truth = target[output] >= 0.5
                    if prediction && truth { truePositive += 1 }
                    if prediction && !truth { falsePositive += 1 }
                    if !prediction && truth { falseNegative += 1 }
                    if let old = previous[output], old != prediction {
                        if prediction { predictedEdges[segment.recordingID, default: [:]][output, default: Edges()].down.append(time) }
                        else { predictedEdges[segment.recordingID, default: [:]][output, default: Edges()].up.append(time) }
                    }
                    previous[output] = prediction
                }
                for output in ActionLayout.relativeMouse { relativeError += Double(abs(result.1[row * ActionLayout.count + output] - target[output])) }
                for output in ActionLayout.scroll { scrollError += Double(abs(result.1[row * ActionLayout.count + output] - target[output])) }
                continuousCount += 2
            }
        }
        var presses = EdgeScore(), releases = EdgeScore()
        var durationError = 0.0, durationCount = 0
        let validationRows = Set(split.validation)
        let validationRecordings = Set(dataset.manifest.segments.filter { validationRows.contains($0.start) }.map(\.recordingID))
        for recording in recordings where validationRecordings.contains(recording.id) {
            let manifest = recording.manifest
            let events = try InputEventReader.mapped(url: recording.directory.appendingPathComponent(manifest.eventFile))
            var timeline = CausalControlTimeline(origin: manifest.hostStartNanos)
            var held = Set<Int>(); var truth: [Int: Edges] = [:]
            for event in events {
                timeline.consume(event)
                let values = timeline.controlValues(rect: manifest.globalRect.cgRect)
                let next = Set(active.filter { values[$0] >= 0.5 })
                let time = TemporalInputContract.seconds(event.timestampNanos, since: manifest.hostStartNanos)
                if time >= manifest.effectiveSupervisionStart, time < (manifest.trimEnd ?? manifest.duration) {
                    for index in next.subtracting(held) { truth[index, default: Edges()].down.append(time) }
                    for index in held.subtracting(next) { truth[index, default: Edges()].up.append(time) }
                }
                held = next
                timeline.endInterval(at: event.timestampNanos)
            }
            for output in active {
                let predicted = predictedEdges[recording.id]?[output] ?? Edges()
                let expected = truth[output] ?? Edges()
                presses.consume(predicted: predicted.down, truth: expected.down)
                releases.consume(predicted: predicted.up, truth: expected.up)
                for (down, up) in expected.completeHolds {
                    guard let pair = predicted.completeHolds.min(by: { abs($0.0 - down) < abs($1.0 - down) }),
                          abs(pair.0 - down) <= 0.1, abs(pair.1 - up) <= 0.1 else { continue }
                    durationError += abs((pair.1 - pair.0) - (up - down)); durationCount += 1
                }
            }
        }
        let elapsed = CACurrentMediaTime() - started
        return ExperimentReport(name: name, recordings: recordings.count, epochs: 20,
            evaluatedRows: split.validation.count, trainingRows: split.train.count,
            selectedEpoch: version.epoch ?? 0, selectedStep: version.globalStep,
            loss: lossSum / Double(split.validation.count),
            binaryPrecision: Double(truePositive) / Double(max(1, truePositive + falsePositive)),
            binaryRecall: Double(truePositive) / Double(max(1, truePositive + falseNegative)),
            presses: presses, releases: releases,
            matchedHoldDurationMAE: durationCount > 0 ? durationError / Double(durationCount) : nil,
            matchedHoldCount: durationCount,
            relativeMouseMAE: relativeError / Double(max(1, continuousCount)),
            scrollMAE: scrollError / Double(max(1, continuousCount)),
            evaluationRowsPerSecond: Double(split.validation.count) / elapsed,
            peakMLXBytes: Memory.snapshot().peakMemory)
    }
}

private struct Edges {
    var down: [Double] = []
    var up: [Double] = []
    var completeHolds: [(Double, Double)] {
        down.compactMap { start in up.first(where: { $0 > start }).map { (start, $0) } }
    }
}

private struct EdgeScore: Encodable {
    var matches = 0, predictedCount = 0, truthCount = 0
    var absoluteError = 0.0
    var precision: Double { Double(matches) / Double(max(1, predictedCount)) }
    var recall: Double { Double(matches) / Double(max(1, truthCount)) }
    var meanAbsoluteError: Double? { matches > 0 ? absoluteError / Double(matches) : nil }
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(matches, forKey: .matches)
        try container.encode(predictedCount, forKey: .predictedCount)
        try container.encode(truthCount, forKey: .truthCount)
        try container.encode(precision, forKey: .precision)
        try container.encode(recall, forKey: .recall)
        try container.encodeIfPresent(meanAbsoluteError, forKey: .meanAbsoluteError)
    }
    private enum CodingKeys: String, CodingKey { case matches, predictedCount, truthCount, precision, recall, meanAbsoluteError }
    mutating func consume(predicted: [Double], truth: [Double]) {
        predictedCount += predicted.count; truthCount += truth.count
        var cursor = 0
        for time in truth {
            while cursor < predicted.count, predicted[cursor] < time - 0.1 { cursor += 1 }
            guard cursor < predicted.count, abs(predicted[cursor] - time) <= 0.1 else { continue }
            matches += 1; absoluteError += abs(predicted[cursor] - time); cursor += 1
        }
    }
}

private struct ExperimentReport: Encodable {
    let name: String
    let recordings: Int
    let epochs: Int
    let evaluatedRows: Int
    let trainingRows: Int
    let selectedEpoch: Int
    let selectedStep: Int
    let loss: Double
    let binaryPrecision: Double
    let binaryRecall: Double
    let presses: EdgeScore
    let releases: EdgeScore
    let matchedHoldDurationMAE: Double?
    let matchedHoldCount: Int
    let relativeMouseMAE: Double
    let scrollMAE: Double
    let evaluationRowsPerSecond: Double
    let peakMLXBytes: Int
}

private final class ExperimentProgress: @unchecked Sendable {
    let name: String
    private let lock = NSLock()
    private var lastStep = -1, lastEpoch = -1
    init(name: String) { self.name = name }
    func consume(_ metrics: TrainingMetrics, status: String) {
        lock.lock(); defer { lock.unlock() }
        guard metrics.globalStep / 500 != lastStep || metrics.epoch != lastEpoch else { return }
        lastStep = metrics.globalStep / 500; lastEpoch = metrics.epoch
        print("TEMPORAL_EXPERIMENT \(name) epoch=\(metrics.epoch)/20 step=\(metrics.globalStep)/\(metrics.totalSteps) loss=\(metrics.trainingLoss) val=\(metrics.validationLoss ?? -1) samples/s=\(metrics.samplesPerSecond) memory=\(metrics.mlxActiveMemory) thermal=\(metrics.thermalState.rawValue) \(status)")
        fflush(nil)
    }
}

private extension JSONDecoder {
    static var experiment: JSONDecoder { let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601; return decoder }
}
