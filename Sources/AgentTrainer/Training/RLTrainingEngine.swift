import Foundation
import MLX
import MLXNN
import MLXRandom

struct RLSavedSummary: Codable, Hashable, Sendable {
    var algorithm: RLAlgorithm = .ppo
    var updates: Int
    var episodes: Int
    var checkpointFormat = 1
    var trainingSeconds: Double? = nil
    var experienceSeconds: Double? = nil
}

struct RLLearningState: Codable, Sendable {
    var format = 1
    var learningSemanticsVersion: Int? = 2
    var attemptedUpdates: Int? = 0
    var rejectedUpdates: Int? = 0
    var legacyUpdateAttempts: Int? = 0
    var resumeNotice: String? = nil
    var rolloutStartStep: Int? = nil
    var activeVersionID: UUID?
    var revision = 0
    var updates = 0
    var optimizerSteps = 0
    var random = RLRandom(state: 42)
    var actorRandom = RLRandom(state: 42)
    var history: [RLEpisodeRecord] = []
    var episodeCount = 0
    var trainingSeconds = 0.0
    var rewardSquareSum = 0.0001
    var rewardCount = 0.0001
    var pending: [RLTransition] = []
    var order: [Int] = []
    var epoch = 0
    var offset = 0
    var advantages: [Float] = []
    var returns: [Float] = []
    var configuration: RLConfiguration
    var preprocessing: PreprocessingSpec
    var architecture: ArchitectureSpec
    var temporal: TemporalVisionConfiguration
    var contextSchema: ModelContextSchema
    var movementSensitivity: Double? = nil
    var channels: ActionChannels? = nil
    var mouseMode: MouseControlMode? = nil
    var actionFPS: Double? = nil
    var perceptionFPS: Double? = nil
    var restrictions: ActionRestrictions? = nil
    var precision: TrainingPrecision? = nil
    var experienceSeconds: Double? = nil
}

struct RLUpdateResult: Sendable {
    var snapshot: RLWeightSnapshot
    var updates: Int
    var steps: Int
    var policyLoss: Double
    var valueLoss: Double
    var entropy: Double
    var kl: Double
    var rejectedActions: Int = 0
    var attemptedUpdates: Int = 0
    var rejectedUpdates: Int = 0
    var klRejectedActions: Int = 0
}

protocol RLLearningAlgorithm: Sendable {
    func prepare(profile: AIProfile, contextSchema: ModelContextSchema, mouseMode: MouseControlMode,
                 activeWeights: URL?, checkpoint: URL, movementSensitivity: Double) async throws -> (RLWeightSnapshot, RLLearningState)
    func update(_ transitions: [RLTransition]) async throws -> RLUpdateResult
    func interrupt()
    func save(to directory: URL, versionID: UUID, actorRandom: RLRandom, history: [RLEpisodeRecord],
              episodes: Int, elapsed: Double, experience: Double, pending: [RLTransition]) async throws -> RLLearningState
}

enum RLAlgorithmFactory {
    static func make(_ algorithm: RLAlgorithm, budget: RLComputeBudget) -> any RLLearningAlgorithm {
        switch algorithm { case .ppo: RLTrainingEngine(budget: budget) }
    }
}

/// All MLX objects live on this serial owner. Inference receives owned CPU
/// buffers, never references to mutable modules or lazy training graphs.
final class RLTrainingEngine: RLLearningAlgorithm, @unchecked Sendable {
    private let queue = DispatchQueue(label: "AgentTrainer.RLLearner", qos: .utility)
    private let cancellation = NSLock()
    private let budget: RLComputeBudget?
    private let didApplyStep: (@Sendable (Int) -> Void)?
    init(budget: RLComputeBudget? = nil, didApplyStep: (@Sendable (Int) -> Void)? = nil) {
        self.budget = budget; self.didApplyStep = didApplyStep
    }
    private var interrupted = false
    private var model: RLTrainablePolicy!
    private var optimizer: ResumableAdamW!
    private var distribution: RLActionDistribution!
    private var state: RLLearningState!
    private var profile: AIProfile!
    private var forwardFunction: (([MLXArray]) -> [MLXArray])?
    private var gradientFunction: (([MLXArray]) -> [MLXArray])?
    private var gradientNames: [String] = []

    func interrupt() { cancellation.lock(); interrupted = true; cancellation.unlock() }
    private var shouldInterrupt: Bool { cancellation.lock(); defer { cancellation.unlock() }; return interrupted }

    private func perform<T: Sendable>(_ body: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do { continuation.resume(returning: try Device.withDefaultDevice(.gpu) {
                    try Stream.withNewDefaultStream(device: .gpu) { try body() }
                }) } catch { continuation.resume(throwing: error) }
            }
        }
    }

    func prepare(profile: AIProfile, contextSchema: ModelContextSchema, mouseMode: MouseControlMode,
                 activeWeights: URL?, checkpoint: URL, movementSensitivity: Double = 1) async throws -> (RLWeightSnapshot, RLLearningState) {
        try await perform { [self] in
            self.profile = profile
            let settings = try profile.effectiveRL.validated()
            distribution = RLActionDistribution(profile: profile, configuration: settings, mouseMode: mouseMode)
            guard !distribution.isEmpty else { throw AgentTrainerError.invalidConfiguration("Add allowed RL keys, mouse buttons, movement, or scrolling before training.") }
            let randomState = MLXRandom.RandomState(seed: settings.seed)
            model = withRandomState(randomState) { RLTrainablePolicy(policy: AgentPolicy(profile: profile, contextSchema: contextSchema,
                movementSensitivity: mouseMode == .relative ? movementSensitivity : 1)) }
            optimizer = ResumableAdamW(learningRate: Float(settings.learningRate), weightDecay: Float(settings.weightDecay),
                warmupSteps: 1, schedule: .constant)
            state = RLLearningState(random: RLRandom(state: settings.seed), actorRandom: RLRandom(state: settings.seed ^ 0x12345678),
                configuration: settings, preprocessing: profile.preprocessing, architecture: profile.training.architecture,
                temporal: profile.training.effectiveTemporalVision, contextSchema: contextSchema,
                movementSensitivity: movementSensitivity, channels: profile.channels, mouseMode: mouseMode,
                actionFPS: profile.training.actionFPS, perceptionFPS: profile.training.perceptionFPS, restrictions: profile.effectiveRestrictions, precision: profile.training.precision)
            let stateURL = checkpoint.appendingPathComponent("state.json")
            if FileManager.default.fileExists(atPath: stateURL.path) {
                let saved = try JSONDecoder().decode(RLLearningState.self, from: Data(contentsOf: stateURL))
                guard saved.format == 1 else { throw AgentTrainerError.model("Unsupported RL checkpoint format; the active policy was preserved.") }
                let sameLayout = saved.preprocessing == profile.preprocessing
                    && saved.architecture.weightLayout == profile.training.architecture.weightLayout
                    && saved.temporal == profile.training.effectiveTemporalVision && saved.contextSchema.hasSameLayout(as: contextSchema)
                if sameLayout, saved.learningSemanticsVersion == 2,
                   saved.activeVersionID == profile.activeVersionID, saved.configuration.hasSameLearningContract(as: settings),
                   saved.channels == profile.channels, saved.restrictions == profile.effectiveRestrictions, saved.precision == profile.training.precision, saved.mouseMode == mouseMode,
                   saved.movementSensitivity == movementSensitivity, saved.actionFPS == profile.training.actionFPS,
                   saved.perceptionFPS == profile.training.perceptionFPS {
                    try saved.validateResume(profile: profile)
                    let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: checkpoint.appendingPathComponent("trainer.safetensors"))
                    guard metadata["format"] == "AgentTrainer.RL.v1" else { throw AgentTrainerError.model("Invalid RL checkpoint weights.") }
                    try model.update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
                    try optimizer.load(from: checkpoint.appendingPathComponent("optimizer.safetensors"))
                    state = saved; state.configuration = settings
                } else {
                    if let activeWeights { try model.policy.loadWeights(from: activeWeights) }
                    // Keep compatible critic/exploration weights across BC switches.
                    if sameLayout, let arrays = try? MLX.loadArrays(url: checkpoint.appendingPathComponent("heads.safetensors")) {
                        try model.heads.update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
                    }
                    state.history = saved.history; state.episodeCount = saved.episodeCount
                    state.trainingSeconds = saved.trainingSeconds; state.updates = saved.updates; state.experienceSeconds = saved.experienceSeconds
                    state.revision = saved.revision; state.actorRandom = saved.actorRandom; state.optimizerSteps = saved.optimizerSteps
                    state.attemptedUpdates = saved.attemptedUpdates ?? saved.updates
                    state.rejectedUpdates = saved.rejectedUpdates ?? 0
                    state.legacyUpdateAttempts = saved.learningSemanticsVersion == nil ? saved.updates : saved.legacyUpdateAttempts
                    state.resumeNotice = "Retained active policy; started a fresh optimizer and excluded incompatible pending experience."
                    if (state.legacyUpdateAttempts ?? 0) > 0 {
                        state.resumeNotice! += " Historical rollout counts include legacy attempts; applied updates below count new semantics only."
                    }
                }
            } else if let activeWeights { try model.policy.loadWeights(from: activeWeights) }
            optimizer.initialize(model: model)
            if optimizer.step == 0 { optimizer.synchronizeInitialParameters(model: model) }
            model.train(false); MLX.eval(model!.parameters(), optimizer!.stateArrays())
            gradientNames = model.parameters().flattened().map(\.0).sorted()
            let trainable = model!, actionDistribution = distribution!, configuration = settings
            let names = gradientNames
            forwardFunction = compile(inputs: [trainable]) { arrays in
                let forward = trainable.forward(inputs: Array(arrays[0..<6]))
                let statistics = actionDistribution.perRowStatistics(logits: forward.0,
                    logStd: trainable.heads.logStandardDeviation, latent: arrays[6], binary: arrays[7])
                return [statistics.0, statistics.1, forward.1]
            }
            gradientFunction = compile(inputs: [trainable]) { inputs in
                let result = valueAndGrad(model: trainable) { (model: RLTrainablePolicy, arrays: [MLXArray]) -> [MLXArray] in
                    let forward = model.forward(inputs: Array(arrays[0..<6]))
                    let statistics = actionDistribution.perRowStatistics(logits: forward.0,
                        logStd: model.heads.logStandardDeviation, latent: arrays[6], binary: arrays[7])
                    let weights = arrays[8]
                    let policy = (weights[0..., 0] * statistics.0).sum()
                    let value = (weights[0..., 2] * square(forward.1 - weights[0..., 1])).sum() * Float(configuration.valueCoefficient * 0.5)
                    let entropy = (weights[0..., 3] * statistics.1).sum() * Float(configuration.entropyCoefficient)
                    return [policy + value - entropy]
                }(trainable, inputs)
                let gradients = Dictionary(uniqueKeysWithValues: result.1.flattened())
                return result.0 + names.map { gradients[$0]! }
            }
            if budget != nil {
                let count = profile.training.effectiveTemporalVision.pastFrameCount
                let pastSpec = profile.training.effectiveTemporalVision.pastFrameSpec(from: profile.preprocessing)
                let empty = RLObservation(current: Data(repeating: 0, count: profile.preprocessing.sampleByteCount),
                    past: Array(repeating: Data(repeating: 0, count: pastSpec.sampleByteCount), count: count),
                    controls: Array(repeating: 0, count: count * ActionLayout.count),
                    timing: Array(repeating: 0, count: TemporalInputContract.valueCount),
                    pastTiming: Array(repeating: 0, count: count * TemporalInputContract.valueCount),
                    context: Array(repeating: 0, count: contextSchema.inputValueCount))
                var inputs = model.inputs([empty, empty])
                inputs += [MLXArray.zeros([2, 8]), MLXArray.zeros([2, ActionLayout.count])]
                MLX.eval(forwardFunction!(inputs))
                inputs.append(MLXArray.zeros([2, 4]))
                MLX.eval(gradientFunction!(inputs))
            }

            cancellation.lock(); interrupted = false; cancellation.unlock()
            let initial = snapshot()
            guard initial.policy.values.allSatisfy({ $0.values.allSatisfy(\.isFinite) }), initial.heads.values.allSatisfy({ $0.values.allSatisfy(\.isFinite) }) else {
                throw AgentTrainerError.model("The selected RL weights contain non-finite values; no inputs were enabled.")
            }
            return (initial, state)
        }
    }

    private func snapshot() -> RLWeightSnapshot {
        RLWeightSnapshot(policy: RLWeightTransfer.copy(model.policy), heads: RLWeightTransfer.copy(model.heads), revision: state.revision)
    }

    func update(_ transitions: [RLTransition]) async throws -> RLUpdateResult {
        try await perform { [self] in
            let c = state.configuration
            var rejectedActions = 0
            guard state.pending.isEmpty || transitions.isEmpty else {
                throw AgentTrainerError.model("Finish pending PPO work before admitting a new rollout.")
            }
            guard transitions.allSatisfy({ transition in
                transition.reward.isFinite && transition.duration.isFinite && transition.duration > 0
                    && transition.bootstrapValue.isFinite && transition.decisions.allSatisfy {
                        $0.logProbability.isFinite && $0.value.isFinite
                            && $0.latent.count == 8 && $0.latent.allSatisfy(\.isFinite)
                            && $0.binary.count == ActionLayout.count && $0.binary.allSatisfy { $0 == 0 || $0 == 1 }
                    }
            }) else { throw AgentTrainerError.model("Invalid or non-finite PPO experience; no update was applied.") }
            if state.pending.isEmpty {
                var accepted: [RLTransition] = []
                for transition in transitions {
                    if !transition.decisions.isEmpty, transition.reward.isFinite,
                       transition.decisions.allSatisfy({ $0.revision >= state.revision - 1 && $0.revision <= state.revision }) {
                        accepted.append(transition)
                    } else {
                        rejectedActions += transition.decisions.count
                        if !accepted.isEmpty, !accepted[accepted.count - 1].terminal {
                            accepted[accepted.count - 1].boundary = true
                            accepted[accepted.count - 1].bootstrapValue = transition.decisions.first?.value ?? 0
                        }
                    }
                }
                guard !accepted.isEmpty else {
                    var empty = result(losses: [0, 0, 0, 0]); empty.rejectedActions = rejectedActions; return empty
                }
                state.pending = accepted; state.epoch = 0; state.offset = 0
                state.rolloutStartStep = state.optimizerSteps
                state.attemptedUpdates = (state.attemptedUpdates ?? 0) + 1
                let divisor: Float = c.normalizeRewards ? Float(sqrt(state.rewardSquareSum / state.rewardCount + 1e-8)) : 1
                let values = accepted.map { $0.decisions[0].value }
                let targets = RLReturns.advantages(accepted, values: values, configuration: c,
                    actionFPS: profile.training.actionFPS, rewardDivisor: divisor)
                state.advantages = targets.advantages; state.returns = targets.returns
                if c.normalizeAdvantages, state.advantages.count > 1 {
                    let mean = state.advantages.reduce(0, +) / Float(state.advantages.count)
                    let variance = state.advantages.reduce(0) { $0 + pow($1 - mean, 2) } / Float(state.advantages.count)
                    state.advantages = state.advantages.map { ($0 - mean) / sqrt(variance + 1e-8) }
                }
                state.order = Array(accepted.indices)
                shuffleOrder()
                for step in accepted {
                    state.rewardSquareSum += Double(step.reward) * Double(step.reward); state.rewardCount += 1
                }
            }
            var latest: [Double] = [0, 0, 0, 0]
            var stopForKL = false
            var klRejectedActions = 0
            let stepsBefore = state.optimizerSteps
            optimization: while state.epoch < c.epochs && !stopForKL && !shouldInterrupt {
                while state.offset < state.order.count && !shouldInterrupt {
                    let end = min(state.order.count, state.offset + c.batchSize)
                    let indices = Array(state.order[state.offset..<end])
                    var accumulated: [String: MLXArray] = [:]
                    var losses = [Double](repeating: 0, count: 4)
                    // First obtain the complete joint likelihood for every
                    // variable-length reward interval. Then apply its exact PPO
                    // derivative in fixed-size microbatches. This avoids tracing
                    // new graphs or retaining a long CNN graph during play.
                    let intervals = indices.map { state.pending[$0] }
                    let decisions = intervals.flatMap(\.decisions)
                    var owners: [Int] = [], firstRows: [Int] = []
                    for (i, interval) in intervals.enumerated() {
                        firstRows.append(owners.count)
                        owners += Array(repeating: i, count: interval.decisions.count)
                    }
                    let microSize = budget == nil ? 8 : 2
                    var rowLogP = [Float](repeating: 0, count: decisions.count)
                    var rowEntropy = rowLogP, rowValues = rowLogP
                    for offset in stride(from: 0, to: decisions.count, by: microSize) {
                        if shouldInterrupt { break optimization }
                        let workBegan = DispatchTime.now().uptimeNanoseconds
                        let end = min(decisions.count, offset + microSize)
                        var chunk = Array(decisions[offset..<end])
                        while chunk.count < microSize { chunk.append(chunk.last!) }
                        let inputs = decisionInputs(chunk)
                        let result = forwardFunction!(inputs); MLX.eval(result)
                        let logP = result[0].asArray(Float.self), entropy = result[1].asArray(Float.self), values = result[2].asArray(Float.self)
                        for row in offset..<end {
                            rowLogP[row] = logP[row - offset]; rowEntropy[row] = entropy[row - offset]; rowValues[row] = values[row - offset]
                        }
                        budget?.yield(after: Double(DispatchTime.now().uptimeNanoseconds - workBegan) / 1e9, cancelled: { self.shouldInterrupt })
                    }
                    var coefficients = [Float](repeating: 0, count: intervals.count)
                    for i in intervals.indices {
                        let rows = firstRows[i]..<(firstRows[i] + intervals[i].decisions.count)
                        let logP = rows.reduce(Float(0)) { $0 + rowLogP[$1] }
                        let oldLogP = intervals[i].decisions.reduce(Float(0)) { $0 + $1.logProbability }
                        let surrogate = RLPPOObjective.evaluate(logRatio: logP - oldLogP, advantage: state.advantages[indices[i]], clip: Float(c.clip))
                        coefficients[i] = surrogate.coefficient / Float(intervals.count)
                        losses[0] += Double(surrogate.loss) / Double(intervals.count)
                        losses[1] += Double(0.5 * pow(rowValues[firstRows[i]] - state.returns[indices[i]], 2)) / Double(intervals.count)
                        losses[2] += Double(rows.reduce(Float(0)) { $0 + rowEntropy[$1] }) / Double(rows.count * intervals.count)
                        losses[3] += Double(surrogate.kl) / Double(intervals.count)
                    }
                    guard losses.allSatisfy(\.isFinite) else { throw AgentTrainerError.model("Non-finite PPO objective; the last valid weights were preserved.") }
                    if losses[3] > c.targetKL * 1.5 {
                        latest = losses; stopForKL = true; klRejectedActions += decisions.count; break
                    }
                    for offset in stride(from: 0, to: decisions.count, by: microSize) {
                        if shouldInterrupt { break optimization }
                        let workBegan = DispatchTime.now().uptimeNanoseconds
                        let end = min(decisions.count, offset + microSize)
                        var chunk = Array(decisions[offset..<end]), weights: [Float] = []
                        for row in offset..<end {
                            let owner = owners[row]
                            weights += [coefficients[owner], state.returns[indices[owner]],
                                row == firstRows[owner] ? 1 / Float(intervals.count) : 0,
                                1 / Float(intervals.count * intervals[owner].decisions.count)]
                        }
                        while chunk.count < microSize { chunk.append(chunk.last!); weights += [0, 0, 0, 0] }
                        var inputs = decisionInputs(chunk); inputs.append(MLXArray(weights, [microSize, 4]))
                        let output = gradientFunction!(inputs); MLX.eval(output)
                        guard output[0].item(Float.self).isFinite else { throw AgentTrainerError.model("Non-finite PPO gradient objective; no update was applied.") }
                        for (name, gradient) in zip(gradientNames, output.dropFirst()) {
                            let g = gradient.asType(.float32)
                            accumulated[name] = accumulated[name].map { $0 + g } ?? g
                        }
                        MLX.eval(gradientNames.compactMap { accumulated[$0] })
                        budget?.yield(after: Double(DispatchTime.now().uptimeNanoseconds - workBegan) / 1e9, cancelled: { self.shouldInterrupt })
                    }
                    let gradients = ModuleParameters.unflattened(accumulated)
                    let norm = sqrt(gradientNames.compactMap { accumulated[$0] }.reduce(MLXArray(Float(0))) { $0 + $1.asType(.float32).square().sum() }); MLX.eval(norm)
                    guard norm.item(Float.self).isFinite, losses.allSatisfy(\.isFinite) else {
                        throw AgentTrainerError.model("Non-finite PPO gradients; the last valid checkpoint is intact.")
                    }
                    if losses[3] > c.targetKL * 1.5 { stopForKL = true; break }
                    optimizer.update(model: model, gradients: gradients, targetType: .float32,
                        gradientNorm: norm, maxGradientNorm: Float(c.maximumGradientNorm), preserveParameterTypes: true)
                    MLX.eval(model!.parameters(), optimizer!.stateArrays())
                    state.optimizerSteps += 1; state.offset = end; latest = losses
                    didApplyStep?(state.optimizerSteps)
                }
                if !shouldInterrupt {
                    state.epoch += 1; state.offset = 0
                    if state.epoch < c.epochs { shuffleOrder() }
                }
            }
            if !shouldInterrupt {
                if state.optimizerSteps > (state.rolloutStartStep ?? stepsBefore) { state.updates += 1 }
                else if stopForKL { state.rejectedUpdates = (state.rejectedUpdates ?? 0) + 1 }
                state.pending = []; state.order = []; state.advantages = []; state.returns = []
                state.epoch = 0; state.offset = 0
                state.rolloutStartStep = nil
            }
            // Inference must never age otherwise-current experience merely
            // because an unchanged policy rejected a rollout before any step.
            if state.optimizerSteps > stepsBefore { state.revision += 1 }
            var update = result(losses: latest)
            update.rejectedActions = rejectedActions; update.klRejectedActions = klRejectedActions
            return update
        }
    }

    private func decisionInputs(_ decisions: [RLDecision]) -> [MLXArray] {
        var inputs = model.inputs(decisions.map(\.observation))
        inputs.append(MLXArray(decisions.flatMap(\.latent), [decisions.count, 8]))
        inputs.append(MLXArray(decisions.flatMap(\.binary), [decisions.count, ActionLayout.count]))
        return inputs
    }

    private func shuffleOrder() {
        for i in state.order.indices.dropFirst().reversed() {
            let j = min(i, Int(state.random.uniform() * Float(i + 1)))
            state.order.swapAt(i, j)
        }
    }

    private func result(losses: [Double]) -> RLUpdateResult {
        RLUpdateResult(snapshot: snapshot(), updates: state.updates, steps: state.optimizerSteps,
            policyLoss: losses[0], valueLoss: losses[1], entropy: losses[2], kl: losses[3],
            attemptedUpdates: state.attemptedUpdates ?? state.updates, rejectedUpdates: state.rejectedUpdates ?? 0)
    }

    func save(to directory: URL, versionID: UUID, actorRandom: RLRandom, history: [RLEpisodeRecord],
              episodes: Int, elapsed: Double, experience: Double = 0, pending: [RLTransition]) async throws -> RLLearningState {
        try await perform { [self] in
            state.activeVersionID = versionID; state.actorRandom = actorRandom
            state.experienceSeconds = experience
            state.history = RLEpisodeRecord.boundedHistory(history); state.episodeCount = episodes; state.trainingSeconds = elapsed
            // Pending optimizer work takes precedence over the next collected rollout.
            // Store the latter separately so neither is silently replayed or lost.
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try model.policy.saveWeights(to: directory.appendingPathComponent("weights.safetensors"))
            try MLX.save(arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
                metadata: ["format": "AgentTrainer.RL.v1"], url: directory.appendingPathComponent("trainer.safetensors"))
            try MLX.save(arrays: Dictionary(uniqueKeysWithValues: model.heads.parameters().flattened()),
                metadata: ["format": "AgentTrainer.RLHeads.v1"], url: directory.appendingPathComponent("heads.safetensors"))
            try optimizer.save(to: directory.appendingPathComponent("optimizer.safetensors"))
            try JSONEncoder().encode(state).write(to: directory.appendingPathComponent("state.json"), options: .atomic)
            try JSONEncoder().encode(versionID).write(to: directory.appendingPathComponent("publication.json"), options: .atomic)
            let encoder = PropertyListEncoder(); encoder.outputFormat = .binary
            try encoder.encode(pending).write(to: directory.appendingPathComponent("collected.plist"), options: .atomic)
            return state
        }
    }
}

extension RLLearningState {
    func validateResume(profile: AIProfile) throws {
        guard revision >= 0, updates >= 0, optimizerSteps >= 0, episodeCount >= 0,
              trainingSeconds.isFinite, trainingSeconds >= 0, rewardSquareSum.isFinite, rewardSquareSum > 0,
              rewardCount.isFinite, rewardCount > 0, epoch >= 0, epoch < configuration.epochs,
              offset >= 0, offset <= order.count, pending.count <= 8192,
              order.count == pending.count, Set(order) == Set(pending.indices),
              advantages.count == pending.count, returns.count == pending.count,
              advantages.allSatisfy(\.isFinite), returns.allSatisfy(\.isFinite) else {
            throw AgentTrainerError.model("The RL checkpoint has invalid progress or optimizer-batch metadata.")
        }
        let count = profile.training.effectiveTemporalVision.pastFrameCount
        let past = profile.training.effectiveTemporalVision.pastFrameSpec(from: profile.preprocessing)
        for transition in pending {
            guard transition.reward.isFinite, transition.duration.isFinite, transition.duration > 0,
                  transition.bootstrapValue.isFinite, !transition.decisions.isEmpty,
                  transition.decisions.count <= 4096 else { throw AgentTrainerError.model("Invalid RL reward interval in the checkpoint.") }
            for decision in transition.decisions {
                let o = decision.observation
                guard decision.latent.count == 8, decision.binary.count == ActionLayout.count,
                      decision.latent.allSatisfy(\.isFinite), decision.binary.allSatisfy({ $0 == 0 || $0 == 1 }),
                      decision.logProbability.isFinite, decision.value.isFinite,
                      o.current.count == profile.preprocessing.sampleByteCount,
                      o.past.count == count, o.past.allSatisfy({ $0.count == past.sampleByteCount }),
                      o.controls.count == count * ActionLayout.count, o.controls.allSatisfy(\.isFinite),
                      o.timing.count == TemporalInputContract.valueCount, o.timing.allSatisfy(\.isFinite),
                      o.pastTiming.count == count * TemporalInputContract.valueCount, o.pastTiming.allSatisfy(\.isFinite),
                      o.context.count == contextSchema.inputValueCount, o.context.allSatisfy(\.isFinite) else {
                    throw AgentTrainerError.model("The RL checkpoint contains an invalid observation or action.")
                }
            }
        }
    }
}
