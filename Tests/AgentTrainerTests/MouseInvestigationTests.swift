import Foundation
import CoreGraphics
import MLX
import MLXNN
import XCTest
@testable import AgentTrainer

/// Local, read-only diagnosis of a saved brain against demonstration inputs.
/// This does not post HID events or publish/modify any brain or recording.
final class MouseInvestigationTests: XCTestCase {
    func testSavedRebornBrainTimingSensitivity() async throws {
        guard ProcessInfo.processInfo.environment["AGENTTRAINER_MOUSE_PROBE"] == "1" else {
            throw XCTSkip("Opt-in local mouse feedback investigation.")
        }
        MLXMemoryLifecycle.configure()
        let source = WorkspaceStore.defaultRoot
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601
        let frozen = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("work/mouse-comparison-v8/Reference-v7")
        let saved: AIProfile
        let versionRoot: URL
        if FileManager.default.fileExists(atPath: frozen.appendingPathComponent("profile.json").path) {
            saved = try decoder.decode(AIProfile.self, from: Data(contentsOf: frozen.appendingPathComponent("profile.json")))
            versionRoot = frozen
        } else {
            let profileURLs = try FileManager.default.contentsOfDirectory(at: source.appendingPathComponent("Profiles"), includingPropertiesForKeys: nil)
            saved = try XCTUnwrap(profileURLs.compactMap { directory -> AIProfile? in
                guard let data = try? Data(contentsOf: directory.appendingPathComponent("profile.json")) else { return nil }
                return try decoder.decode(AIProfile.self, from: data)
            }.first { $0.name == "Reborn" })
            let versionID = try XCTUnwrap(saved.activeVersionID)
            versionRoot = source.appendingPathComponent("Profiles/\(saved.id.uuidString)/Versions/\(versionID.uuidString)")
        }
        let version = try decoder.decode(ModelVersionManifest.self, from: Data(contentsOf: versionRoot.appendingPathComponent("manifest.json")))
        guard version.schemaVersion == 7 else { throw XCTSkip("The diagnostic requires a preserved v7 reference.") }
        var profile = saved
        profile.preprocessing = version.preprocessing; profile.training = version.training; profile.channels = version.channels
        let folders = try decoder.decode([RecordingFolder].self, from: Data(contentsOf: source.appendingPathComponent("recording-folders.json")))
        let folderID = try XCTUnwrap(folders.first { $0.name == "Reborn" }?.id)
        let recordings = try FileManager.default.contentsOfDirectory(at: source.appendingPathComponent("Recordings"), includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "atrrecord" }.compactMap { directory -> RecordingItem? in
                let manifest = try decoder.decode(RecordingManifest.self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
                return manifest.folderID == folderID ? RecordingItem(manifest: manifest, directory: directory) : nil
            }.sorted { $0.id.uuidString < $1.id.uuidString }
        let workspace = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("work/mouse-investigation-v7")
        let store = WorkspaceStore(root: workspace); try await store.prepare()
        let dataset = try await DatasetCacheBuilder(workspace: store).cache(for: profile, recordings: recordings) { fraction, status in
            if fraction == 1 { print("MOUSE_PROBE_CACHE \(status)") }
        }
        let split = TrainingEngine().splitIndices(dataset: dataset, fraction: profile.training.validationSplit,
            seed: profile.training.seed, channels: profile.channels, restrictions: profile.effectiveRestrictions)
        let rows = stride(from: 0, to: split.validation.count, by: max(1, split.validation.count / 512)).map { split.validation[$0] }
        let model = AgentPolicy(profile: profile, contextSchema: version.effectiveContextSchema, legacyMovementFeedback: true)
        let referenceArrays = try MLX.loadArrays(url: versionRoot.appendingPathComponent(version.weightsFile))
        try model.update(parameters: ModuleParameters.unflattened(referenceArrays), verify: .noUnusedKeys)
        model.train(false)
        let count = profile.training.effectiveTemporalVision.pastFrameCount
        let pastSpec = profile.training.effectiveTemporalVision.pastFrameSpec(from: profile.preprocessing)
        let head = compile(inputs: [model]) { arrays in
            let features = model.encodedTemporalFeatures(current: arrays[0], past: arrays[1], controls: arrays[2],
                currentTiming: arrays[3], pastTiming: arrays[4])
            return [model.activatedPredictions(logits: model.logits(temporalFeatures: features))]
        }
        let variants = ["recorded", "age-16ms", "age-33ms", "age-66ms", "age-125ms", "posted-motion", "center-pointer", "motion-zero", "timing-zero"]
        var reports = Dictionary(uniqueKeysWithValues: variants.map { ($0, MouseScore()) })
        var feedback = [Float](repeating: 0, count: 2)
        for start in stride(from: 0, to: rows.count, by: 32) {
            let batchRows = Array(rows[start..<min(rows.count, start + 32)])
            autoreleasepool {
                let b = batchRows.count
                let batch = dataset.trainingBatch(at: batchRows), timing = dataset.timingBatch(at: batchRows)
                let images = VisionPreprocessor.mlxTensor(MLXArray(batch.packedCurrentObservations,
                    [b, profile.preprocessing.sampleByteCount], dtype: .uint8), spec: profile.preprocessing)
                let pastImages = VisionPreprocessor.mlxPastFrameTensor(MLXArray(batch.packedPastObservations,
                    [b, count, pastSpec.sampleByteCount], dtype: .uint8), spec: pastSpec)
                let current = model.visualEmbedding(visualFeatures: model.visualActivations(images: images).last!)
                let past = model.pastVisualEmbedding(visualFeatures: model.pastVisualActivations(
                    images: pastImages.reshaped([b * count, pastSpec.height, pastSpec.width, pastSpec.channelCount])).last!)
                    .reshaped([b, count, profile.training.architecture.visualEmbedding])
                let controls = MLXArray(batch.pastControlRows, [b, count, ActionLayout.count], type: Float.self)
                let pastTiming = MLXArray(timing.past, [b, count, TemporalInputContract.valueCount], type: Float.self)
                let original = MLXArray(timing.current, [b, TemporalInputContract.valueCount], type: Float.self).asArray(Float.self)
                MLX.eval(current, past)
                for variant in variants {
                    var values = original
                    for row in 0..<b {
                        let base = row * TemporalInputContract.valueCount
                        if variant.hasPrefix("age-") {
                            let age = Double(variant.dropFirst(4).dropLast(2))! / 1000
                            var triplet = [Float](repeating: 0, count: 3)
                            TemporalInputContract.putTime(age, in: &triplet, at: 0)
                            for i in 0..<3 { values[base + 7 + i] = triplet[i] }
                        }
                        if variant == "center-pointer" { values[base + 16] = 0.5; values[base + 17] = 0.5 }
                        if variant == "motion-zero" {
                            for i in 0..<TemporalInputContract.motionWidth { values[base + TemporalInputContract.motionOffset + i] = 0 }
                        }
                        if variant == "timing-zero" { for i in 0..<TemporalInputContract.valueCount { values[base + i] = 0 } }
                        if variant == "posted-motion" { Self.coarsenMotion(&values, base: base) }
                    }
                    let output = head([current, past, controls, MLXArray(values, [b, TemporalInputContract.valueCount]), pastTiming])[0]
                    MLX.eval(output)
                    let predictions = output.asArray(Float.self)
                    for (row, index) in batchRows.enumerated() {
                        let target = dataset.action(at: index)
                        reports[variant]!.consume(Array(predictions[row * ActionLayout.count..<(row + 1) * ActionLayout.count]), target)
                    }
                }
                // Freeze 32 visual contexts and repeatedly feed the predicted
                // raw movement back into the new current timing branch. This
                // isolates feedback sensitivity; it is not a game simulation.
                if start == 0 {
                    var values = original
                    for row in 0..<b {
                        var interval = [Float](repeating: 0, count: 3)
                        TemporalInputContract.putTime(1.0 / 60, in: &interval, at: 0)
                        for i in 0..<3 { values[row * TemporalInputContract.valueCount + 1 + i] = interval[i] }
                    }
                    var path: [[Float]] = []
                    for tick in 0..<60 {
                        let output = head([current, past, controls, MLXArray(values, [b, TemporalInputContract.valueCount]), pastTiming])[0]
                        MLX.eval(output)
                        let predictions = output.asArray(Float.self)
                        var magnitude: Float = 0
                        for row in 0..<b {
                            let base = row * TemporalInputContract.valueCount
                            let motion = base + TemporalInputContract.motionOffset
                            for axis in 0..<2 {
                                let delta = (predictions[row * ActionLayout.count + 2 + axis] * 80).rounded() / 80
                                values[motion + axis] = delta
                                magnitude += abs(delta) * 80 / Float(b)
                            }
                            values[base + 16] = 0.5; values[base + 17] = 0.5
                            Self.coarsenMotion(&values, base: base)
                        }
                        if [0, 1, 5, 15, 30, 59].contains(tick) { path.append([Float(tick), magnitude]) }
                    }
                    print("MOUSE_FEEDBACK_FIXED_VISION \(path)")
                    feedback = path.last ?? feedback
                }
            }
        }
        // Preserve the recorded visual sequence and non-mouse control context,
        // but close the new current-movement feedback path with posted deltas.
        // This isolates that path without pretending recorded video responds to
        // our counterfactual actions as a live game would.
        let rolloutRows = Array(split.validation.prefix(600))
        var rolloutRecorded = MouseScore(), rolloutFeedback = MouseScore()
        var timeline: CausalControlTimeline?
        var previousQuery: UInt64?
        let rect = CGRect(x: 0, y: 0, width: 100, height: 100)
        for start in stride(from: 0, to: rolloutRows.count, by: 32) {
            let selected = Array(rolloutRows[start..<min(rolloutRows.count, start + 32)])
            autoreleasepool {
                let b = selected.count
                let batch = dataset.trainingBatch(at: selected), timing = dataset.timingBatch(at: selected)
                let images = VisionPreprocessor.mlxTensor(MLXArray(batch.packedCurrentObservations,
                    [b, profile.preprocessing.sampleByteCount], dtype: .uint8), spec: profile.preprocessing)
                let pastImages = VisionPreprocessor.mlxPastFrameTensor(MLXArray(batch.packedPastObservations,
                    [b, count, pastSpec.sampleByteCount], dtype: .uint8), spec: pastSpec)
                let current = model.visualEmbedding(visualFeatures: model.visualActivations(images: images).last!)
                let past = model.pastVisualEmbedding(visualFeatures: model.pastVisualActivations(
                    images: pastImages.reshaped([b * count, pastSpec.height, pastSpec.width, pastSpec.channelCount])).last!)
                    .reshaped([b, count, profile.training.architecture.visualEmbedding])
                let controls = MLXArray(batch.pastControlRows, [b, count, ActionLayout.count], type: Float.self)
                let pastTiming = MLXArray(timing.past, [b, count, TemporalInputContract.valueCount], type: Float.self)
                let original = MLXArray(timing.current, [b, TemporalInputContract.valueCount], type: Float.self).asArray(Float.self)
                MLX.eval(current, past)
                for (row, index) in selected.enumerated() {
                    let query = UInt64(1_000_000_000) + dataset.decisionTime(at: index).query
                    if previousQuery == nil || query <= previousQuery! || query - previousQuery! > 100_000_000 {
                        timeline = CausalControlTimeline(origin: query, censorInitialOnsets: false)
                    }
                    let motion = timeline!.features(at: query, frameTime: nil, perceptionInterval: nil, rect: rect)
                    timeline!.endInterval(at: query)
                    var values = Array(original[row * TemporalInputContract.valueCount..<(row + 1) * TemporalInputContract.valueCount])
                    let recorded = head([current[row..<(row + 1)], past[row..<(row + 1)], controls[row..<(row + 1)],
                        MLXArray(values, [1, values.count]), pastTiming[row..<(row + 1)]])[0]
                    MLX.eval(recorded)
                    rolloutRecorded.consume(recorded.asArray(Float.self), dataset.action(at: index))
                    for i in 0..<TemporalInputContract.motionWidth {
                        values[TemporalInputContract.motionOffset + i] = motion[TemporalInputContract.motionOffset + i]
                    }
                    values[16] = 0.5; values[17] = 0.5
                    let output = head([current[row..<(row + 1)], past[row..<(row + 1)], controls[row..<(row + 1)],
                        MLXArray(values, [1, values.count]), pastTiming[row..<(row + 1)]])[0]
                    MLX.eval(output)
                    let prediction = output.asArray(Float.self)
                    rolloutFeedback.consume(prediction, dataset.action(at: index))
                    timeline!.consume(InputSample(timestampNanos: query + 1_000_000, kind: .mouseMove,
                        x: 50, y: 50, deltaX: Double(GameCameraContract.runtimeDelta(forPrediction: prediction[2], sensitivity: 1).rounded()),
                        deltaY: Double(GameCameraContract.runtimeDelta(forPrediction: prediction[3], sensitivity: 1).rounded())))
                    previousQuery = query
                }
            }
        }
        reports["sequence-recorded"] = rolloutRecorded
        reports["sequence-current-feedback"] = rolloutFeedback
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(reports)
        try data.write(to: workspace.appendingPathComponent("sensitivity.json"), options: .atomic)
        print("MOUSE_PROBE_RESULT \(String(data: data, encoding: .utf8)!) feedback=\(feedback)")
    }

    private static func coarsenMotion(_ values: inout [Float], base: Int) {
        let motion = base + TemporalInputContract.motionOffset
        let interval = max(1e-6, expm1(Double(values[base + 2]) * log(61)))
        let x = (values[motion] * 80).rounded() / 80, y = (values[motion + 1] * 80).rounded() / 80
        for i in 0..<TemporalInputContract.motionWidth { values[motion + i] = 0 }
        values[motion] = x; values[motion + 1] = y
        for (axis, value) in [x, y].enumerated() {
            values[motion + 2 + axis] = Float((value < 0 ? -1 : 1) * log1p(abs(Double(value)) / interval))
        }
        if x != 0 || y != 0 {
            values[motion + 4] = Float(log(2.0)); values[motion + 5] = 0.001; values[motion + 6] = 0.001
            values[motion + 9] = 1
            var clock = [Float](repeating: 0, count: 3)
            TemporalInputContract.putTime(max(0, interval - 0.001), in: &clock, at: 0)
            for i in 0..<3 { values[motion + 10 + i] = clock[i] }
        }
        values[motion + 13] = 1
    }
}

private struct MouseScore: Encodable {
    var rows = 0, activeRows = 0, idleRows = 0, idleMoved = 0
    private var rawError = 0.0, activeRawError = 0.0, absoluteError = 0.0, magnitude = 0.0, biasX = 0.0, biasY = 0.0
    private enum CodingKeys: String, CodingKey {
        case rows, activeRows, idleRows, idleMoved, rawMAE, activeRawMAE, absoluteMAE, rawMagnitude, rawBiasX, rawBiasY, idleMovementRate
    }
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(rows, forKey: .rows); try c.encode(activeRows, forKey: .activeRows)
        try c.encode(idleRows, forKey: .idleRows); try c.encode(idleMoved, forKey: .idleMoved)
        let n = Double(max(1, rows))
        try c.encode(rawError / n, forKey: .rawMAE)
        try c.encode(activeRawError / Double(max(1, activeRows)), forKey: .activeRawMAE)
        try c.encode(absoluteError / n, forKey: .absoluteMAE)
        try c.encode(magnitude / n, forKey: .rawMagnitude)
        try c.encode(biasX / n, forKey: .rawBiasX); try c.encode(biasY / n, forKey: .rawBiasY)
        try c.encode(Double(idleMoved) / Double(max(1, idleRows)), forKey: .idleMovementRate)
    }
    mutating func consume(_ prediction: [Float], _ target: [Float]) {
        rows += 1
        let error = Double(abs(prediction[2] - target[2]) + abs(prediction[3] - target[3])) * 40
        rawError += error
        absoluteError += Double(abs(prediction[0] - target[0]) + abs(prediction[1] - target[1])) / 2
        magnitude += Double(abs(prediction[2]) + abs(prediction[3])) * 80
        biasX += Double(prediction[2]) * 80; biasY += Double(prediction[3]) * 80
        if abs(target[2]) + abs(target[3]) > 0.0001 { activeRows += 1; activeRawError += error }
        else { idleRows += 1; if abs(prediction[2]) >= 0.5 / 80 || abs(prediction[3]) >= 0.5 / 80 { idleMoved += 1 } }
    }
}
