import Foundation

enum RLAlgorithm: String, Codable, CaseIterable, Identifiable, Sendable {
    case ppo = "PPO"
    var id: String { rawValue }
}

enum RLRewardMode: String, Codable, CaseIterable, Identifiable, Sendable {
    case cumulative = "Cumulative"
    case immediate = "Immediate"
    var id: String { rawValue }
}

enum RLPreset: String, CaseIterable, Identifiable {
    case stable = "Stable Learning", balanced = "Balanced Learning", exploratory = "Exploratory Learning"
    var id: String { rawValue }
}

struct RLRewardRegion: Codable, Hashable, Sendable {
    var sourceID: UInt32
    var isWindow: Bool
    /// Top-left normalized coordinates within the entire selected source.
    var normalized: CodableRect

    func rect(in frame: CGRect) -> CGRect {
        let r = normalized.cgRect
        return CGRect(x: frame.minX + r.minX * frame.width, y: frame.minY + r.minY * frame.height,
                      width: r.width * frame.width, height: r.height * frame.height)
    }

    func matches(_ source: CaptureSourceOption) -> Bool {
        source.id == sourceID && (source.kind == .window) == isWindow
    }

    var isValid: Bool {
        let r = normalized.cgRect
        return [r.minX, r.minY, r.width, r.height].allSatisfy(\.isFinite)
            && r.minX >= 0 && r.minY >= 0 && r.width > 0 && r.height > 0
            && r.maxX <= 1.000001 && r.maxY <= 1.000001
    }
}

struct RLConfiguration: Codable, Hashable, Sendable {
    var algorithm: RLAlgorithm = .ppo
    var advanced = false
    var rewardMode: RLRewardMode = .cumulative
    var rewardRegion: RLRewardRegion?
    var allowedKeys: Set<UInt16> = []
    var allowedButtons: Set<UInt8> = []
    var mouseMovement = false
    var scroll = false
    var controlsInitialized = false
    var startKey: UInt16? = 15
    var endKey: UInt16?
    var episodeDelay = 0.5
    var resetTimeout = 10.0
    var resetSettlingSeconds = 0.1
    var detectionTimeout = 1.0
    var minimumConfidence = 0.8
    var learningRate = 0.0001
    var rolloutSize = 256
    var batchSize = 32
    var epochs = 4
    var discount = 0.999
    var gaeLambda = 0.95
    var clip = 0.2
    var valueCoefficient = 0.5
    var entropyCoefficient = 0.01
    var maximumGradientNorm = 0.5
    var targetKL = 0.02
    var binaryExploration = 0.03
    var explorationScale = 1.0
    var normalizeAdvantages = true
    var normalizeRewards = true
    var rewardScale = 1.0
    var weightDecay = 0.0
    var automaticCheckpoints = true
    var autosaveSeconds = 60.0
    var retainedAutosaves = 10
    var trainingComputeFraction = 0.35
    var seed: UInt64 = 42

    mutating func apply(_ preset: RLPreset) {
        switch preset {
        case .stable:
            learningRate = 0.00005; rolloutSize = 256; epochs = 3
            binaryExploration = 0.01; explorationScale = 0.5; entropyCoefficient = 0.003
        case .balanced:
            learningRate = 0.0001; rolloutSize = 256; epochs = 4
            binaryExploration = 0.03; explorationScale = 1; entropyCoefficient = 0.01
        case .exploratory:
            learningRate = 0.0002; rolloutSize = 128; epochs = 4
            binaryExploration = 0.1; explorationScale = 1.5; entropyCoefficient = 0.02
        }
    }

    func validated() throws -> Self {
        let finite = [episodeDelay, resetTimeout, resetSettlingSeconds, detectionTimeout, minimumConfidence, learningRate,
                      discount, gaeLambda, clip, valueCoefficient, entropyCoefficient, maximumGradientNorm,
                      targetKL, binaryExploration, explorationScale, rewardScale, weightDecay, autosaveSeconds, trainingComputeFraction]
        guard finite.allSatisfy(\.isFinite), (0...60).contains(episodeDelay), (1...120).contains(resetTimeout), (0...5).contains(resetSettlingSeconds),
              (0.25...30).contains(detectionTimeout), (0.1...1).contains(minimumConfidence),
              (0.0000001...0.003).contains(learningRate), (2...4096).contains(rolloutSize),
              (1...512).contains(batchSize), batchSize <= rolloutSize, (1...20).contains(epochs),
              (0.8...1).contains(discount), (0...1).contains(gaeLambda), (0.01...0.5).contains(clip),
              (0...10).contains(valueCoefficient), (0...1).contains(entropyCoefficient),
              (0.01...100).contains(maximumGradientNorm), (0.0001...0.5).contains(targetKL),
              (0...0.5).contains(binaryExploration), (0.05...5).contains(explorationScale),
              (0.000001...1000000).contains(rewardScale), (0...1).contains(weightDecay),
              (0.1...0.8).contains(trainingComputeFraction), (5...86400).contains(autosaveSeconds), (1...100).contains(retainedAutosaves),
              allowedKeys.allSatisfy({ $0 < 128 }), allowedButtons.allSatisfy({ $0 < 8 }),
              startKey.map({ $0 < 128 }) != false, endKey.map({ $0 < 128 }) != false,
              rewardRegion.map(\.isValid) != false else {
            throw AgentTrainerError.invalidConfiguration("RL settings contain an invalid value. Check the highlighted ranges and selected reward region.")
        }
        return self
    }

    func hasSameLearningContract(as other: Self) -> Bool {
        func normalized(_ input: Self) -> Self {
            var value = input
            value.advanced = false; value.rewardRegion = nil; value.controlsInitialized = false
            value.episodeDelay = 0; value.resetTimeout = 10; value.resetSettlingSeconds = 0.1; value.detectionTimeout = 1
            value.minimumConfidence = 0.8; value.autosaveSeconds = 60; value.automaticCheckpoints = true
            value.retainedAutosaves = 10; value.trainingComputeFraction = 0.35
            return value
        }
        return normalized(self) == normalized(other)
    }

    func capabilities(profile: AIProfile, mouseMode: MouseControlMode) -> ModelOutputCapabilities {
        var reserved = Set([startKey, endKey].compactMap { $0 })
        for pair in [[UInt16(56), 60], [59, 62], [58, 61], [55, 54]] where !reserved.isDisjoint(with: pair) { reserved.formUnion(pair) }
        let keys = allowedKeys.subtracting(reserved).filter { profile.effectiveRestrictions.allowsKey($0) }
        let buttons = allowedButtons.filter { profile.effectiveRestrictions.allowsButton($0) }
        return ModelOutputCapabilities(allowedKeyCodes: Set(keys), allowedMouseButtons: Set(buttons),
            allowsAbsoluteMouseMovement: mouseMovement && profile.channels.mouseMovement && mouseMode != .relative,
            allowsRelativeMouseMovement: mouseMovement && profile.channels.mouseMovement && mouseMode == .relative,
            allowsScroll: scroll && profile.channels.scroll)
    }
}

struct RLRewardReading: Sendable {
    var reward: Double
    var episodeLength: Double
    var capturedAt: UInt64
    var recognizedAt: UInt64
    var confidence: Float
}

enum RLTextParser {
    static func number(_ text: String) -> Double? {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "−", with: "-")
        guard value.range(of: #"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)$"#, options: .regularExpression) != nil,
              let result = Double(value), result.isFinite else { return nil }
        return result
    }

    static func pair(_ lines: [String]) -> (Double, Double)? {
        var reward: Double?, length: Double?
        let fields = lines.flatMap { $0.replacingOccurrences(of: #"(?i)(?=\b(?:Reward|Episode\s*Length)\s*:)"#,
            with: "\n", options: .regularExpression).components(separatedBy: .newlines) }
        for line in fields {
            let parts = line.split(separator: ":", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { continue }
            let label = parts[0].lowercased().filter { !$0.isWhitespace }
            if label == "reward" { guard reward == nil else { return nil }; reward = number(parts[1]) }
            if label == "episodelength" { guard length == nil else { return nil }; length = number(parts[1]) }
        }
        guard let reward, let length, length >= 0, length <= 604800 else { return nil }
        return (reward, length)
    }
}

/// The reducer owns accounting, not OCR completion order or nominal frame numbers.
struct RLEpisodeReducer: Sendable {
    enum Phase: String, Sendable { case waiting, active, finishing, paused }
    struct Accepted: Sendable {
        var reward: Double
        var duration: Double
        var terminal: Bool
        var ready: Bool = false
    }
    var phase: Phase = .paused
    var generation: UInt64 = 0
    var resetAt: UInt64 = 0
    var startedAt: UInt64 = 0
    var lastReadingAt: UInt64 = 0
    var previousReward = 0.0
    var reward = 0.0
    var maximumDuration = 0.0

    mutating func reset(at now: UInt64) {
        generation &+= 1; resetAt = now; startedAt = 0; lastReadingAt = 0
        previousReward = 0; reward = 0; maximumDuration = 0; phase = .waiting
    }

    mutating func accept(_ reading: RLRewardReading, mode: RLRewardMode) -> Accepted? {
        guard reading.reward.isFinite, reading.episodeLength.isFinite, reading.episodeLength >= 0,
              reading.capturedAt > resetAt, reading.capturedAt > lastReadingAt else { return nil }
        if phase == .waiting {
            guard reading.reward == 0, reading.episodeLength > 0 else { return nil }
            phase = .active; startedAt = reading.capturedAt; lastReadingAt = reading.capturedAt
            maximumDuration = reading.episodeLength; previousReward = 0
            return Accepted(reward: 0, duration: 0, terminal: false, ready: true)
        }
        guard phase == .active || phase == .finishing else { return nil }
        let delta = mode == .cumulative ? reading.reward - previousReward : reading.reward
        let duration = Double(reading.capturedAt - lastReadingAt) / 1e9
        previousReward = reading.reward; lastReadingAt = reading.capturedAt
        reward += delta; maximumDuration = reading.episodeLength
        let terminal = reading.episodeLength == 0
        if terminal { phase = .paused }
        return Accepted(reward: delta, duration: duration, terminal: terminal)
    }

    func remaining(at now: UInt64) -> Double {
        guard startedAt > 0 else { return 0 }
        return max(0, maximumDuration - Double(now >= startedAt ? now - startedAt : 0) / 1e9)
    }
}

struct RLObservation: Codable, Sendable {
    var current: Data
    var past: [Data]
    var controls: [Float]
    var timing: [Float]
    var pastTiming: [Float]
    var context: [Float]
}

struct RLDecision: Codable, Sendable {
    var observation: RLObservation
    var latent: [Float]
    var binary: [Float]
    var logProbability: Float
    var value: Float
    var timestamp: UInt64
    var revision: Int
}

struct RLTransition: Codable, Sendable {
    var decisions: [RLDecision]
    var reward: Float
    var duration: Double
    var terminal: Bool
    var boundary: Bool = false
    var bootstrapValue: Float = 0
}

enum RLExperienceBuffer {
    static func bytes(_ transition: RLTransition) -> Int {
        transition.decisions.reduce(0) { $0 + $1.observation.current.count + $1.observation.past.reduce(0) { $0 + $1.count } }
    }

    /// Keep terminal rewards and the current uncommitted tail. If those alone
    /// fill the budget, pause collection instead of losing an episode's result.
    static func trim(_ transitions: inout [RLTransition], maximumActions: Int, maximumBytes: Int) -> (discarded: Int, overflow: Bool) {
        var actions = transitions.reduce(0) { $0 + $1.decisions.count }
        var storage = transitions.reduce(0) { $0 + bytes($1) }
        var discarded = 0
        while actions > maximumActions || storage > maximumBytes {
            guard let index = transitions.indices.dropLast().first(where: { !transitions[$0].terminal }) else {
                return (discarded, true)
            }
            let removed = transitions.remove(at: index)
            if index > 0, !transitions[index - 1].terminal {
                transitions[index - 1].boundary = true
                transitions[index - 1].bootstrapValue = removed.decisions.first?.value ?? 0
            }
            actions -= removed.decisions.count; storage -= bytes(removed)
            discarded += removed.decisions.count
        }
        return (discarded, false)
    }
}

struct RLEpisodeRecord: Codable, Identifiable, Sendable {
    var id: Int
    var reward: Double
    var duration: Double
    var interrupted: Bool = false

    static func boundedHistory(_ records: [Self], limit: Int = 10000) -> [Self] {
        guard records.count > limit, limit > 1 else { return records }
        let recent = Array(records.suffix(limit))
        guard let best = records.filter({ !$0.interrupted }).max(by: { $0.reward < $1.reward }),
              !recent.contains(where: { $0.id == best.id }) else { return recent }
        return [best] + recent.suffix(limit - 1)
    }
}

struct RLMetrics: Sendable {
    var status = "Ready"
    var episode = 0
    var totalEpisodes = 0
    var episodeReward = 0.0
    var remaining = 0.0
    var trainingSeconds = 0.0
    var experienceSeconds = 0.0
    var updates = 0
    var optimizerSteps = 0
    var policyLoss = 0.0
    var valueLoss = 0.0
    var entropy = 0.0
    var kl = 0.0
    var rewardFPS = 0.0
    var detectionStatus = "Not tested"
    var recognitionLatency = 0.0
    var skippedIntervals = 0
    var discardedActions = 0
    var backlogActions = 0
    var history: [RLEpisodeRecord] = []
    var best: Double? { history.filter { !$0.interrupted }.map(\.reward).max() }
    var average: Double? {
        let values = history.filter { !$0.interrupted }.suffix(100)
        return values.isEmpty ? nil : values.reduce(0) { $0 + $1.reward } / Double(values.count)
    }
}

struct RLRandom: Codable, Sendable {
    var state: UInt64
    mutating func uniform() -> Float {
        state &+= 0x9E3779B97F4A7C15
        var z = state; z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return Float((z ^ (z >> 31)) >> 40) / 16777216
    }
    mutating func normal() -> Float {
        sqrt(-2 * log(max(0.0000001, uniform()))) * cos(2 * .pi * uniform())
    }
}

enum RLReturns {
    static func advantages(_ transitions: [RLTransition], values: [Float], configuration: RLConfiguration,
                           actionFPS: Double, rewardDivisor: Float = 1) -> (advantages: [Float], returns: [Float]) {
        precondition(values.count == transitions.count)
        var advantages = [Float](repeating: 0, count: values.count)
        var next: Float = 0
        for i in transitions.indices.reversed() {
            let step = transitions[i]
            let gamma = Float(pow(configuration.discount, max(0.001, step.duration * actionFPS)))
            let lambda = Float(pow(configuration.gaeLambda, max(0.001, step.duration * actionFPS)))
            let bootstrap = step.terminal ? 0 : (step.boundary || i + 1 == values.count
                ? step.bootstrapValue : values[i + 1])
            let delta = step.reward / max(0.000001, rewardDivisor) + gamma * bootstrap - values[i]
            next = delta + (step.terminal || step.boundary ? 0 : gamma * lambda * next)
            advantages[i] = next
        }
        return (advantages, zip(advantages, values).map(+))
    }
}

/// Executed actions enter synchronously on the HID queue; main-actor task
/// delivery order is never used to align them with captured reward frames.
final class RLDecisionInbox: @unchecked Sendable {
    private let lock = NSLock()
    private var decisions: [RLDecision] = []
    private var bytes = 0
    private var overflow = false
    func append(_ decision: RLDecision) {
        lock.withLock {
            let size = decision.observation.current.count + decision.observation.past.reduce(0) { $0 + $1.count }
            guard bytes + size <= 512 * 1024 * 1024, decisions.count < 4096 else { overflow = true; return }
            decisions.append(decision); bytes += size
        }
    }
    func take(before timestamp: UInt64) -> (records: [RLDecision], overflow: Bool, nextValue: Float?) {
        lock.withLock {
            let count = decisions.prefix { $0.timestamp < timestamp }.count
            let result = Array(decisions.prefix(count)); decisions.removeFirst(count)
            bytes -= result.reduce(0) { $0 + $1.observation.current.count + $1.observation.past.reduce(0) { $0 + $1.count } }
            return (result, overflow, decisions.first?.value)
        }
    }
    func clear() { lock.withLock { decisions = []; bytes = 0; overflow = false } }
}

/// Bounds learner duty cycle while the independent inference stream is active.
/// Individual Metal kernels are not preemptible, so measured overrun increases
/// the yield between microbatches rather than queueing more GPU work.
final class RLComputeBudget: @unchecked Sendable {
    private let lock = NSLock()
    private var late = false
    var fraction: Double
    init(fraction: Double = 0.35) { self.fraction = fraction }
    func noteInference(milliseconds: Double, actionFPS: Double) {
        lock.withLock { late = milliseconds > 1000 / actionFPS }
    }
    func yield(after seconds: Double, cancelled: () -> Bool) {
        let share = lock.withLock { min(0.8, max(0.1, fraction * (late ? 0.5 : 1))) }
        let duration = UInt64(min(1, max(0, seconds * (1 - share) / share)) * 1e9)
        let deadline = DispatchTime.now().uptimeNanoseconds + duration
        while !cancelled() {
            let now = DispatchTime.now().uptimeNanoseconds
            if now >= deadline { break }
            Thread.sleep(forTimeInterval: min(0.01, Double(deadline - now) / 1e9))
        }
    }
}

/// Exact derivative of the clipped PPO surrogate with respect to a joint
/// log-likelihood. Evaluating it first permits bounded, fixed-shape gradients.
enum RLPPOObjective {
    static func evaluate(logRatio: Float, advantage: Float, clip: Float) -> (loss: Float, coefficient: Float, kl: Float) {
        let ratio = exp(min(20, max(-20, logRatio)))
        let bounded = min(1 + clip, max(1 - clip, ratio))
        let loss = -min(ratio * advantage, bounded * advantage)
        let differentiable = abs(logRatio) < 20 && ratio * advantage <= bounded * advantage
        return (loss, differentiable ? -advantage * ratio : 0, ratio - 1 - logRatio)
    }
}
