import CoreGraphics
import Foundation
import MLX
import MLXNN
@testable import AgentTrainer

/// Executes the policy and production injector against real recorded visual
/// observations. Only mouse feedback is closed; other controls come from the
/// actual demonstration. No generated actions become training targets.
enum MouseFeedbackEvaluation {
    private struct Frame {
        let current: MLXArray
        let reduced: MLXArray
        let controls: [Float]
        let timing: [Float]
        let query: UInt64
        let source: UInt64
    }

    static func evaluate(_ model: AgentPolicy, dataset: CachedDataset, rows: [Int], profile: AIProfile,
                         recordings: [RecordingItem], perceptionStride: Int = 1) throws -> MouseFeedbackMetrics {
        var score = MouseFeedbackMetrics()
        let count = profile.training.effectiveTemporalVision.pastFrameCount
        let spacing = profile.training.effectiveTemporalVision.frameSpacing
        let pastSpec = profile.training.effectiveTemporalVision.pastFrameSpec(from: profile.preprocessing)
        let byID = Dictionary(uniqueKeysWithValues: recordings.map { ($0.id, $0) })
        let head = compile(inputs: [model]) { arrays in
            let features = model.encodedTemporalFeatures(current: arrays[0], past: arrays[1], controls: arrays[2],
                currentTiming: arrays[3], pastTiming: arrays[4])
            return [model.activatedPredictions(logits: model.logits(temporalFeatures: features,
                contextValues: model.contextSchema.isEmpty ? nil : arrays[5]))]
        }
        let sink = MouseEventSink()
        let injector = InputInjector(eventSink: { sink.post($0) }, cursorWarp: { _ in },
            monotonicClock: { sink.now }, associateCursor: {})
        defer { injector.disableAndReleaseAll() }
        var execution = profile
        execution.channels.mouseMovement = true
        execution.channels.buttons = false; execution.channels.scroll = false
        execution.channels.keyboard = false; execution.channels.modifiers = false
        var camera = GameCameraSettings(); camera.recenterCursor = false
        let rect = CGRect(x: 0, y: 0, width: 100, height: 100)
        var recordingID: UUID?
        var events: InputEventReader.MappedEvents?
        var eventIndex = 0
        var teacherDecision: CausalControlTimeline?
        var teacherPerception: CausalControlTimeline?
        var nativeOffset: UInt64 = 0
        var sessionRow = 0
        var history: [Frame] = []
        var latest: (current: MLXArray, past: [Frame?], fallback: MLXArray, source: UInt64)?
        var lastPerception: UInt64?
        var latch = RuntimePredictionLatch()
        for start in stride(from: 0, to: rows.count, by: 32) {
            let indices = Array(rows[start..<min(rows.count, start + 32)])
            try autoreleasepool {
                let b = indices.count
                let full = VisionPreprocessor.mlxTensor(MLXArray(dataset.packedObservations(at: indices),
                    [b, profile.preprocessing.sampleByteCount], dtype: .uint8), spec: profile.preprocessing)
                let reduced = VisionPreprocessor.mlxTensor(MLXArray(dataset.reducedCurrentObservations(at: indices),
                    [b, pastSpec.sampleByteCount], dtype: .uint8), spec: pastSpec)
                let currentEmbeddings = model.visualEmbedding(visualFeatures: model.visualActivations(images: full).last!)
                let pastEmbeddings = model.pastVisualEmbedding(visualFeatures: model.pastVisualActivations(images: reduced).last!)
                MLX.eval(currentEmbeddings, pastEmbeddings)
                for (row, index) in indices.enumerated() {
                    let segment = dataset.manifest.segments.last { $0.start <= index }!
                    let item = byID[segment.recordingID]!
                    let query = dataset.decisionTime(at: index).query
                    let sequence = dataset.visionBatchPlan(at: [index]).uniqueSequences[0]
                    let source = dataset.observationTime(at: Int(sequence.indices[0])).source
                    let teacherTime = item.manifest.hostStartNanos + query
                    if recordingID != item.id {
                        injector.disableAndReleaseAll()
                        nativeOffset = sink.now + 1_000_000_000
                        sink.setTime(nativeOffset + query)
                        injector.enable(outputPermissions: RuntimeOutputPermissions(cursorMovement: true, keyboard: false))
                        recordingID = item.id
                        events = try InputEventReader.mapped(url: item.directory.appendingPathComponent(item.manifest.eventFile))
                        eventIndex = 0
                        teacherDecision = CausalControlTimeline(origin: item.manifest.hostStartNanos)
                        teacherPerception = CausalControlTimeline(origin: item.manifest.hostStartNanos)
                        while eventIndex < events!.count, events![eventIndex].timestampNanos < teacherTime {
                            teacherDecision!.consume(events![eventIndex]); teacherPerception!.consume(events![eventIndex]); eventIndex += 1
                        }
                        teacherDecision!.endInterval(at: teacherTime); teacherPerception!.endInterval(at: teacherTime)
                        history.removeAll(); latest = nil; lastPerception = nil; sessionRow = 0; latch.reset()
                        score.recordingBoundary()
                    }
                    sink.setTime(nativeOffset + query)
                    while eventIndex < events!.count, events![eventIndex].timestampNanos < teacherTime {
                        teacherDecision!.consume(events![eventIndex]); teacherPerception!.consume(events![eventIndex]); eventIndex += 1
                    }
                    if sessionRow % max(1, perceptionStride) == 0 || latest == nil {
                        let prior: [Frame?] = (0..<count).map { slot in
                            let distance = spacing * (count - slot)
                            return distance <= history.count ? history[history.count - distance] : nil
                        }
                        latest = (currentEmbeddings[row..<(row + 1)], prior, pastEmbeddings[row..<(row + 1)], source)
                        let native = injector.temporalSnapshot(frameTime: nativeOffset + source, rect: rect, perception: true)
                        var timing = teacherPerception!.features(at: teacherTime,
                            frameTime: item.manifest.hostStartNanos + source,
                            perceptionInterval: lastPerception.map { TemporalInputContract.seconds(query, since: $0) }, rect: item.manifest.globalRect.cgRect)
                        var controls = teacherPerception!.controlValues(rect: item.manifest.globalRect.cgRect)
                        replaceMouse(in: &timing, from: native.features)
                        for i in 0..<4 { controls[i] = native.controls[i] }
                        controls.withUnsafeMutableBufferPointer {
                            ActionLayout.sanitizeTrainingRows($0, rowCount: 1, channels: profile.channels, restrictions: profile.effectiveRestrictions)
                        }
                        sanitize(&timing, profile: profile)
                        history.append(Frame(current: latest!.current, reduced: latest!.fallback,
                            controls: controls, timing: timing, query: query, source: source))
                        if history.count > count * spacing + 1 { history.removeFirst() }
                        teacherPerception!.endInterval(at: teacherTime); lastPerception = query
                    }
                    let vision = latest!
                    let native = injector.temporalSnapshot(frameTime: nativeOffset + vision.source, rect: rect)
                    var timing = teacherDecision!.features(at: teacherTime,
                        frameTime: item.manifest.hostStartNanos + vision.source,
                        perceptionInterval: lastPerception.map { _ in 1 / profile.training.perceptionFPS * Double(perceptionStride) },
                        rect: item.manifest.globalRect.cgRect)
                    teacherDecision!.endInterval(at: teacherTime)
                    replaceMouse(in: &timing, from: native.features)
                    sanitize(&timing, profile: profile)
                    var pastTiming: [Float] = []
                    for (slot, frame) in vision.past.enumerated() {
                        var values = frame?.timing ?? [Float](repeating: 0, count: TemporalInputContract.valueCount)
                        if let frame {
                            for i in 10..<16 { values[i] = 0 }
                            TemporalInputContract.putTime(TemporalInputContract.seconds(query, since: frame.query), in: &values, at: 10)
                            if slot > 0, let previous = vision.past[slot - 1] {
                                TemporalInputContract.putTime(TemporalInputContract.seconds(frame.query, since: previous.query), in: &values, at: 13)
                            }
                        }
                        pastTiming += values
                    }
                    let past = stacked(vision.past.map { ($0?.reduced ?? vision.fallback)[0] }, axis: 0).expandedDimensions(axis: 0)
                    let controls = vision.past.flatMap { $0?.controls ?? [Float](repeating: 0, count: ActionLayout.count) }
                    var arrays = [vision.current, past, MLXArray(controls, [1, count, ActionLayout.count]),
                        MLXArray(timing, [1, timing.count]), MLXArray(pastTiming, [1, count, TemporalInputContract.valueCount])]
                    if !model.contextSchema.isEmpty {
                        arrays.append(MLXArray(dataset.contextBatch(at: [index]), [1, model.contextSchema.inputValueCount], type: Float.self))
                    }
                    let began = DispatchTime.now().uptimeNanoseconds
                    let output = head(arrays)[0]; MLX.eval(output)
                    let prediction = output.asArray(Float.self)
                    guard prediction.count == ActionLayout.count, prediction.allSatisfy(\.isFinite) else {
                        throw AgentTrainerError.model("Non-finite feedback-evaluation prediction.")
                    }
                    let latency = Double(DispatchTime.now().uptimeNanoseconds - began) / 1e6
                    latch.publish(prediction, at: Double(nativeOffset + query) / 1e9)
                    let action = latch.consume()!
                    sink.clearMovement(); sink.setTime(nativeOffset + query + 1_000_000)
                    injector.execute(action.values, profile: execution, allowedKeyCodes: [], mouseMode: .relative,
                        captureRect: rect, safety: AgentSafetyPolicy(), gameCamera: camera, predictionIsFresh: action.isFresh)
                    let posted = sink.movement
                    score.consume(posted: posted, target: dataset.action(at: index), latency: latency)
                    sessionRow += 1
                }
            }
        }
        return score
    }

    private static func replaceMouse(in values: inout [Float], from actual: [Float]) {
        values[16] = actual[16]; values[17] = actual[17]
        for i in 0..<TemporalInputContract.motionWidth {
            values[TemporalInputContract.motionOffset + i] = actual[TemporalInputContract.motionOffset + i]
        }
    }
    private static func sanitize(_ values: inout [Float], profile: AIProfile) {
        values.withUnsafeMutableBufferPointer {
            TemporalInputContract.sanitize($0, rows: 1, channels: profile.channels, restrictions: profile.effectiveRestrictions)
        }
    }
}

private final class MouseEventSink: @unchecked Sendable {
    private let lock = NSLock()
    private var time: UInt64 = 1_000_000_000
    private var x = 0.0, y = 0.0
    var now: UInt64 { lock.withLock { time } }
    var movement: (Double, Double) { lock.withLock { (x, y) } }
    func setTime(_ value: UInt64) { lock.withLock { time = value } }
    func clearMovement() { lock.withLock { x = 0; y = 0 } }
    func post(_ event: CGEvent) {
        guard event.type == .mouseMoved else { return }
        lock.withLock {
            x += Double(event.getIntegerValueField(.mouseEventDeltaX))
            y += Double(event.getIntegerValueField(.mouseEventDeltaY))
        }
    }
}

struct MouseFeedbackMetrics: Encodable {
    var rows = 0, idle = 0, idlePosts = 0, active = 0, activePosts = 0
    private var error = 0.0, magnitude = 0.0, signedX = 0.0, signedY = 0.0
    private var activeAxes = 0, correctDirection = 0
    private var lastTarget = [Double](repeating: 0, count: 2)
    private var stopAges: [Int?] = [nil, nil]
    private var stopLags: [Int] = [], missedStops = 0
    private var reverseAges: [Int?] = [nil, nil]
    private var reverseSigns = [Double](repeating: 0, count: 2)
    private var reverseLags: [Int] = [], missedReversals = 0
    private var latencies: [Double] = []
    var feedbackRawMAE: Double { error / Double(max(1, rows)) }
    var idlePostRate: Double { Double(idlePosts) / Double(max(1, idle)) }
    var movementRecall: Double { Double(activePosts) / Double(max(1, active)) }
    mutating func recordingBoundary() {
        lastTarget = [0, 0]; stopAges = [nil, nil]; reverseAges = [nil, nil]; reverseSigns = [0, 0]
    }
    mutating func consume(posted: (Double, Double), target: [Float], latency: Double) {
        let p = [posted.0, posted.1], t = [Double(target[2]) * 80, Double(target[3]) * 80]
        rows += 1; latencies.append(latency)
        error += (abs(p[0] - t[0]) + abs(p[1] - t[1])) / 2
        magnitude += abs(p[0]) + abs(p[1]); signedX += p[0]; signedY += p[1]
        let moved = p[0] != 0 || p[1] != 0
        if abs(t[0]) + abs(t[1]) < 0.0001 { idle += 1; if moved { idlePosts += 1 } }
        else { active += 1; if moved { activePosts += 1 } }
        for axis in 0..<2 {
            let targetSign = t[axis] == 0 ? 0 : (t[axis] < 0 ? -1.0 : 1.0)
            let postedSign = p[axis] == 0 ? 0 : (p[axis] < 0 ? -1.0 : 1.0)
            if targetSign != 0 { activeAxes += 1; if postedSign == targetSign { correctDirection += 1 } }
            if t[axis] == 0, lastTarget[axis] != 0 { stopAges[axis] = 0 }
            if t[axis] != 0, stopAges[axis] != nil { missedStops += 1; stopAges[axis] = nil }
            if let age = stopAges[axis] {
                if p[axis] == 0 { stopLags.append(age); stopAges[axis] = nil }
                else { stopAges[axis] = age + 1 }
            }
            if t[axis] * lastTarget[axis] < 0 {
                if reverseAges[axis] != nil { missedReversals += 1 }
                reverseAges[axis] = 0; reverseSigns[axis] = targetSign
            }
            if let age = reverseAges[axis] {
                if postedSign == reverseSigns[axis] { reverseLags.append(age); reverseAges[axis] = nil }
                else if targetSign == 0 { missedReversals += 1; reverseAges[axis] = nil }
                else { reverseAges[axis] = age + 1 }
            }
            lastTarget[axis] = t[axis]
        }
    }
    private enum CodingKeys: String, CodingKey {
        case rows, idle, idlePosts, active, activePosts, feedbackRawMAE, idlePostRate, movementRecall,
             directionAccuracy, magnitude, signedX, signedY, stopMatches, missedStops, stopP95Ticks,
             reversalMatches, missedReversals, reversalP95Ticks, headP50Milliseconds, headP95Milliseconds
    }
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(rows, forKey: .rows); try c.encode(idle, forKey: .idle); try c.encode(idlePosts, forKey: .idlePosts)
        try c.encode(active, forKey: .active); try c.encode(activePosts, forKey: .activePosts)
        try c.encode(feedbackRawMAE, forKey: .feedbackRawMAE); try c.encode(idlePostRate, forKey: .idlePostRate)
        try c.encode(movementRecall, forKey: .movementRecall)
        try c.encode(Double(correctDirection) / Double(max(1, activeAxes)), forKey: .directionAccuracy)
        let n = Double(max(1, rows))
        try c.encode(magnitude / n, forKey: .magnitude); try c.encode(signedX / n, forKey: .signedX); try c.encode(signedY / n, forKey: .signedY)
        try c.encode(stopLags.count, forKey: .stopMatches); try c.encode(missedStops, forKey: .missedStops)
        try c.encode(reverseLags.count, forKey: .reversalMatches); try c.encode(missedReversals, forKey: .missedReversals)
        if !stopLags.isEmpty { let a = stopLags.sorted(); try c.encode(a[Int(Double(a.count - 1) * 0.95)], forKey: .stopP95Ticks) }
        if !reverseLags.isEmpty { let a = reverseLags.sorted(); try c.encode(a[Int(Double(a.count - 1) * 0.95)], forKey: .reversalP95Ticks) }
        if !latencies.isEmpty {
            let a = latencies.sorted()
            try c.encode(a[a.count / 2], forKey: .headP50Milliseconds)
            try c.encode(a[Int(Double(a.count - 1) * 0.95)], forKey: .headP95Milliseconds)
        }
    }
}
