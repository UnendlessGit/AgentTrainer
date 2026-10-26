import CoreGraphics
import Foundation
import MLX
import MLXNN
@testable import AgentTrainer

/// Executes the policy and production injector against real recorded visual
/// observations. Can close mouse or all posted controls, or retain recorded
/// history for a pipeline parity check. No generated action is a training target.
enum MouseFeedbackEvaluation {
    enum FeedbackMode: Equatable { case mouseOnly, allControls, recordedControls }
    private struct Frame {
        let current: MLXArray
        let reduced: MLXArray
        let controls: [Float]
        let timing: [Float]
        let query: UInt64
        let source: UInt64
    }

    static func evaluate(_ model: AgentPolicy, dataset: CachedDataset, rows: [Int], profile: AIProfile,
                         recordings: [RecordingItem], perceptionStride: Int = 1,
                         feedbackMode: FeedbackMode = .mouseOnly,
                         capabilities: ModelOutputCapabilities? = nil,
                         bypassMovementActivityForDiagnosis: Bool = false,
                         maskControlHistoryForDiagnosis: Bool = false) throws -> MouseFeedbackMetrics {
        guard feedbackMode == .mouseOnly || capabilities != nil else {
            throw AgentTrainerError.model("Full-control evaluation requires immutable output capabilities.")
        }
        var score = MouseFeedbackMetrics()
        let count = profile.training.effectiveTemporalVision.pastFrameCount
        let spacing = profile.training.effectiveTemporalVision.frameSpacing
        let pastSpec = profile.training.effectiveTemporalVision.pastFrameSpec(from: profile.preprocessing)
        let byID = Dictionary(uniqueKeysWithValues: recordings.map { ($0.id, $0) })
        let head = compile(inputs: [model]) { arrays in
            let features = model.encodedTemporalFeatures(current: arrays[0], past: arrays[1], controls: arrays[2],
                currentTiming: arrays[3], pastTiming: arrays[4])
            let logits = model.logits(temporalFeatures: features,
                contextValues: model.contextSchema.isEmpty ? nil : arrays[5])
            let prediction = model.activatedPredictions(logits: logits)
            if bypassMovementActivityForDiagnosis {
                return [concatenated([prediction[.ellipsis, 0..<2], tanh(logits[.ellipsis, 2..<4]),
                    prediction[.ellipsis, 4...]], axis: -1)]
            }
            return [prediction]
        }
        let sink = MouseEventSink()
        let injector = InputInjector(eventSink: { sink.post($0) }, cursorWarp: { _ in },
            monotonicClock: { sink.now }, associateCursor: {})
        injector.onState = { sink.accept($0) }
        defer { injector.disableAndReleaseAll() }
        var execution = profile
        if feedbackMode != .mouseOnly, let capabilities {
            execution.channels = capabilities.restricting(execution.channels, mouseMode: .relative)
            execution.restrictions = capabilities.restricting(execution.effectiveRestrictions)
        } else {
            execution.channels.mouseMovement = true
            execution.channels.buttons = false; execution.channels.scroll = false
            execution.channels.keyboard = false; execution.channels.modifiers = false
        }
        let allowedKeys = feedbackMode != .mouseOnly ? capabilities?.allowedKeyCodes ?? [] : []
        let keyboardEnabled = feedbackMode != .mouseOnly
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
                        injector.enable(outputPermissions: RuntimeOutputPermissions(cursorMovement: true, keyboard: keyboardEnabled))
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
                        if feedbackMode == .allControls {
                            timing.replaceSubrange(16..., with: native.features[16...])
                            controls = native.controls
                        } else if feedbackMode == .mouseOnly {
                            replaceMouse(in: &timing, from: native.features)
                            for i in 0..<4 { controls[i] = native.controls[i] }
                        }
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
                    if feedbackMode == .allControls {
                        timing.replaceSubrange(16..., with: native.features[16...])
                    } else if feedbackMode == .mouseOnly { replaceMouse(in: &timing, from: native.features) }
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
                    var controls = vision.past.flatMap { $0?.controls ?? [Float](repeating: 0, count: ActionLayout.count) }
                    if maskControlHistoryForDiagnosis {
                        for index in 16..<timing.count { timing[index] = 0 }
                        for row in 0..<count {
                            for index in 16..<TemporalInputContract.valueCount {
                                pastTiming[row * TemporalInputContract.valueCount + index] = 0
                            }
                        }
                        controls = [Float](repeating: 0, count: controls.count)
                    }
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
                    injector.execute(action.values, profile: execution, allowedKeyCodes: allowedKeys, mouseMode: .relative,
                        captureRect: rect, safety: AgentSafetyPolicy(), gameCamera: camera,
                        predictionIsFresh: action.isFresh, shiftUsesKeyboardChannel: true)
                    let posted = sink.movement
                    score.consume(posted: posted, target: dataset.action(at: index), latency: latency)
                    if feedbackMode != .mouseOnly {
                        score.consumeKeys(state: sink.state, target: dataset.action(at: index), allowedKeys: allowedKeys)
                    }
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
    private var controlState = InputState.empty
    var state: InputState { lock.withLock { controlState } }
    func accept(_ state: InputState) { lock.withLock { controlState = state } }
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
    private(set) var keyScores: [UInt16: FeedbackKeyMetrics] = [:]
    mutating func consumeKeys(state: InputState, target: [Float], allowedKeys: Set<UInt16>) {
        for code in allowedKeys where !ActionLayout.allModifierKeyCodes.contains(code) {
            keyScores[code, default: FeedbackKeyMetrics()].consume(
                actual: state.keys.contains(code), target: target[14 + Int(code)] >= 0.5)
        }
        let families: [Set<UInt16>] = [[56, 60], [59, 62], [58, 61], [54, 55]]
        let primary: [UInt16] = [56, 59, 58, 55]
        let masks: [CGEventFlags] = [.maskShift, .maskControl, .maskAlternate, .maskCommand]
        for i in families.indices where !allowedKeys.isDisjoint(with: families[i]) {
            keyScores[primary[i], default: FeedbackKeyMetrics()].consume(
                actual: state.modifiers & masks[i].rawValue != 0, target: target[142 + i] >= 0.5)
        }
    }
    var feedbackRawMAE: Double { error / Double(max(1, rows)) }
    var idlePostRate: Double { Double(idlePosts) / Double(max(1, idle)) }
    var movementRecall: Double { Double(activePosts) / Double(max(1, active)) }
    mutating func recordingBoundary() {
        for code in keyScores.keys { keyScores[code]?.recordingBoundary() }
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
             reversalMatches, missedReversals, reversalP95Ticks, headP50Milliseconds, headP95Milliseconds, keyScores
    }
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(Dictionary(uniqueKeysWithValues: keyScores.map { (String($0.key), $0.value) }), forKey: .keyScores)
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


struct FeedbackKeyMetrics: Encodable {
    var truePositives = 0, falsePositives = 0, falseNegatives = 0, trueNegatives = 0
    var maximumPostedHoldTicks = 0, maximumTargetHoldTicks = 0
    private var postedHold = 0, targetHold = 0
    mutating func consume(actual: Bool, target: Bool) {
        if actual && target { truePositives += 1 }
        else if actual { falsePositives += 1 }
        else if target { falseNegatives += 1 }
        else { trueNegatives += 1 }
        postedHold = actual ? postedHold + 1 : 0
        targetHold = target ? targetHold + 1 : 0
        maximumPostedHoldTicks = max(maximumPostedHoldTicks, postedHold)
        maximumTargetHoldTicks = max(maximumTargetHoldTicks, targetHold)
    }
    mutating func recordingBoundary() { postedHold = 0; targetHold = 0 }
    private enum CodingKeys: String, CodingKey {
        case truePositives, falsePositives, falseNegatives, trueNegatives,
             maximumPostedHoldTicks, maximumTargetHoldTicks
    }
}
