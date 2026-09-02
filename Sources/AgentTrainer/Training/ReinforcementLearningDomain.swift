import Foundation

enum ReinforcementLearningAlgorithm: String, Codable, CaseIterable, Identifiable, Sendable {
    case proximalPolicyOptimization = "Proximal Policy Optimization (PPO)"

    var id: String { rawValue }
}

enum ReinforcementRewardAggregation: String, Codable, CaseIterable, Identifiable, Sendable {
    case sum = "Sum"
    case mean = "Mean"
    case latest = "Latest"

    var id: String { rawValue }
}

enum ModelTrainingOrigin: String, Codable, Sendable {
    case demonstration
    case reinforcementLearning
}

/// Sampling and stabilization semantics shared by online collection and normal
/// Run for an RL-trained actor. This is persisted with every RL brain so later
/// profile edits cannot silently change how its logits become executable input.
struct ReinforcementActionBehavior: Codable, Hashable, Sendable {
    static let defaultAbsoluteMouseStandardDeviation = 0.035
    static let defaultRelativeMouseStandardDeviation = 0.08
    static let defaultScrollStandardDeviation = 0.06

    var binaryActionTemperature = 1.0
    var absoluteMouseStandardDeviation = Self.defaultAbsoluteMouseStandardDeviation
    var relativeMouseStandardDeviation = Self.defaultRelativeMouseStandardDeviation
    var scrollStandardDeviation = Self.defaultScrollStandardDeviation
    /// Previous-action weight. Zero is unfiltered; values near one are smoother.
    var absoluteMouseSmoothing = 0.65
    var relativeMouseSmoothing = 0.45
    var scrollSmoothing = 0.25
    /// Maximum normalized cursor distance admitted per perception. One permits
    /// a full control-region traversal in a single policy decision.
    var maximumAbsoluteMouseStep = 0.08
    /// Caps the signed policy value before the fixed 80-point Game Camera scale
    /// and user sensitivity are restored. This prevents a new actor from
    /// immediately producing extreme locked-camera turns.
    var maximumRelativeMouseMagnitude = 0.25
    /// Removes tiny signed Game Camera values that otherwise become visible HID
    /// jitter after the fixed runtime delta scale is restored.
    var relativeMouseDeadZone = 0.01

    func validated() throws -> Self {
        let values = [
            binaryActionTemperature,
            absoluteMouseStandardDeviation,
            relativeMouseStandardDeviation,
            scrollStandardDeviation,
            absoluteMouseSmoothing,
            relativeMouseSmoothing,
            scrollSmoothing,
            maximumAbsoluteMouseStep,
            maximumRelativeMouseMagnitude,
            relativeMouseDeadZone
        ]
        guard values.allSatisfy(\.isFinite),
              (0.1...5).contains(binaryActionTemperature),
              (0.001...0.5).contains(absoluteMouseStandardDeviation),
              (0.001...1).contains(relativeMouseStandardDeviation),
              (0.001...1).contains(scrollStandardDeviation),
              (0...0.98).contains(absoluteMouseSmoothing),
              (0...0.98).contains(relativeMouseSmoothing),
              (0...0.98).contains(scrollSmoothing),
              (0.005...1).contains(maximumAbsoluteMouseStep),
              (0.01...1).contains(maximumRelativeMouseMagnitude),
              (0...0.25).contains(relativeMouseDeadZone) else {
            throw AgentTrainerError.invalidConfiguration(
                "RL action behavior is outside its safe bounds. Check binary temperature, per-action exploration, smoothing, cursor step, maximum Game Camera output, and dead zone."
            )
        }
        return self
    }

    func standardDeviation(for actionIndex: Int) -> Float {
        if ActionLayout.absoluteMouse.contains(actionIndex) {
            return Float(absoluteMouseStandardDeviation)
        }
        if ActionLayout.relativeMouse.contains(actionIndex) {
            return Float(relativeMouseStandardDeviation)
        }
        if ActionLayout.scroll.contains(actionIndex) {
            return Float(scrollStandardDeviation)
        }
        return 1
    }
}

struct RewardWebSocketMessageNames: Codable, Hashable, Sendable {
    static let maximumNameLength = 96

    // Incoming JSON fields and type values.
    var typeField = "type"
    var rewardField = "reward"
    var doneField = "done"
    var incomingEpisodeFinished = "episode_finished"
    var incomingHeartbeat = "heartbeat"

    // Outgoing type values. Payload metadata keeps stable descriptive keys.
    var outgoingEpisodeFinished = "episode_finished"
    var outgoingEpisodeStarting = "episode_starting"
    var outgoingEpisodeStarted = "episode_started"
    var outgoingTrainingUpdateStarted = "training_update_started"
    var outgoingTrainingUpdateFinished = "training_update_finished"

    func validated() throws -> Self {
        var value = self
        value.typeField = try Self.validatedName(typeField, label: "Type field")
        value.rewardField = try Self.validatedName(rewardField, label: "Reward field")
        value.doneField = try Self.validatedName(doneField, label: "Done field")
        value.incomingEpisodeFinished = try Self.validatedName(
            incomingEpisodeFinished,
            label: "Incoming episode-finished message"
        )
        value.incomingHeartbeat = try Self.validatedName(
            incomingHeartbeat,
            label: "Incoming heartbeat message"
        )
        value.outgoingEpisodeFinished = try Self.validatedName(
            outgoingEpisodeFinished,
            label: "Outgoing episode-finished message"
        )
        value.outgoingEpisodeStarting = try Self.validatedName(
            outgoingEpisodeStarting,
            label: "Outgoing episode-starting message"
        )
        value.outgoingEpisodeStarted = try Self.validatedName(
            outgoingEpisodeStarted,
            label: "Outgoing episode-started message"
        )
        value.outgoingTrainingUpdateStarted = try Self.validatedName(
            outgoingTrainingUpdateStarted,
            label: "Outgoing update-started message"
        )
        value.outgoingTrainingUpdateFinished = try Self.validatedName(
            outgoingTrainingUpdateFinished,
            label: "Outgoing update-finished message"
        )
        guard Set([value.typeField, value.rewardField, value.doneField]).count == 3 else {
            throw AgentTrainerError.invalidConfiguration(
                "WebSocket type, reward, and done field names must be different."
            )
        }
        let fixedOutgoingFields: Set<String> = [
            "episode", "timestamp", "reason", "return", "steps",
            "next_episode"
        ]
        guard !fixedOutgoingFields.contains(value.typeField) else {
            throw AgentTrainerError.invalidConfiguration(
                "The WebSocket type field cannot reuse an outgoing metadata field name."
            )
        }
        guard value.incomingEpisodeFinished != value.incomingHeartbeat else {
            throw AgentTrainerError.invalidConfiguration(
                "Incoming episode-finished and heartbeat message names must be different."
            )
        }
        let outgoing = [
            value.outgoingEpisodeFinished,
            value.outgoingEpisodeStarting,
            value.outgoingEpisodeStarted,
            value.outgoingTrainingUpdateStarted,
            value.outgoingTrainingUpdateFinished
        ]
        guard Set(outgoing).count == outgoing.count else {
            throw AgentTrainerError.invalidConfiguration(
                "Every outgoing WebSocket message name must be unique."
            )
        }
        return value
    }

    private static func validatedName(_ raw: String, label: String) throws -> String {
        let name = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, name.count <= maximumNameLength,
              name.unicodeScalars.allSatisfy({
                  !CharacterSet.controlCharacters.contains($0)
              }) else {
            throw AgentTrainerError.invalidConfiguration(
                "\(label) must contain 1–\(maximumNameLength) visible characters."
            )
        }
        return name
    }
}

/// One app-wide reward endpoint. AgentTrainer is the WebSocket server; the
/// external rewarder connects to this URL whenever RL starts listening.
struct RewardWebSocketServerSettings: Codable, Hashable, Sendable {
    static let defaultURL = "ws://127.0.0.1:47831"
    static let maximumMessageBytes = 64 * 1_024

    var url = Self.defaultURL
    var allowRemoteConnections = false
    var connectionWaitTimeoutSeconds = 15.0
    var rewardTimeoutSeconds = 5.0
    var stopAndSaveOnDisconnect = true
    var disconnectGracePeriodSeconds = 1.0
    var stopAndSaveOnRewardTimeout = true
    var stopAndSaveOnInvalidMessageFlood = true
    var maximumInvalidMessages = 25
    var messageNames = RewardWebSocketMessageNames()

    var validatedURL: URL? {
        let trimmedURL = url.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let components = URLComponents(string: trimmedURL),
              components.scheme?.lowercased() == "ws",
              let host = components.host, !host.isEmpty,
              let port = components.port, (1_024...65_535).contains(port),
              components.user == nil, components.password == nil,
              components.query == nil, components.fragment == nil else { return nil }
        let normalizedHost = host.lowercased()
        let loopback = normalizedHost == "localhost"
            || normalizedHost == "127.0.0.1" || normalizedHost == "::1"
        guard allowRemoteConnections || loopback else { return nil }
        return components.url
    }

    func validated() throws -> Self {
        var value = self
        value.url = url.trimmingCharacters(in: .whitespacesAndNewlines)
        guard value.validatedURL != nil else {
            throw AgentTrainerError.invalidConfiguration(
                allowRemoteConnections
                    ? "The reward server URL must be an unencrypted ws:// URL with a host and explicit non-privileged port, without credentials, query, or fragment."
                    : "The reward server URL must use ws://localhost, ws://127.0.0.1, or ws://[::1] with an explicit port. Enable remote connections only when a trusted LAN rewarder needs access."
            )
        }
        let timing = [
            connectionWaitTimeoutSeconds, rewardTimeoutSeconds,
            disconnectGracePeriodSeconds
        ]
        guard timing.allSatisfy(\.isFinite),
              (0.5...300).contains(connectionWaitTimeoutSeconds),
              (0...600).contains(rewardTimeoutSeconds),
              (0...30).contains(disconnectGracePeriodSeconds),
              (1...100_000).contains(maximumInvalidMessages) else {
            throw AgentTrainerError.invalidConfiguration(
                "WebSocket safety timing or invalid-message limits are outside their supported bounds."
            )
        }
        value.messageNames = try messageNames.validated()
        return value
    }
}

/// Explicit allowlist used only while collecting reinforcement-learning
/// experience. An empty category is disabled; there is deliberately no
/// implicit "all keys" fallback.
struct ReinforcementActionSpace: Codable, Hashable, Sendable {
    static let maximumKeyCode: UInt16 = 127
    static let maximumMouseButton: UInt8 = 7

    var allowedKeyCodes: Set<UInt16> = []
    var allowedMouseButtons: Set<UInt8> = []
    var allowsAbsoluteMouseMovement = false
    var allowsRelativeMouseMovement = false
    var allowsScroll = false

    static let keyboardWASD = ReinforcementActionSpace(
        allowedKeyCodes: [0, 1, 2, 13]
    )
    static let keyboardArrows = ReinforcementActionSpace(
        allowedKeyCodes: [123, 124, 125, 126]
    )
    static let keyboardNumbers = ReinforcementActionSpace(
        allowedKeyCodes: [18, 19, 20, 21, 22, 23, 25, 26, 28, 29]
    )

    var hasAnySelection: Bool {
        !allowedKeyCodes.isEmpty || !allowedMouseButtons.isEmpty
            || allowsAbsoluteMouseMovement || allowsRelativeMouseMovement
            || allowsScroll
    }

    var modifierKeyCodes: Set<UInt16> {
        allowedKeyCodes.intersection(ActionLayout.allModifierKeyCodes)
    }

    func validated() throws -> Self {
        guard allowedKeyCodes.allSatisfy({ $0 <= Self.maximumKeyCode }) else {
            throw AgentTrainerError.invalidConfiguration(
                "RL key allowlists support macOS key codes 0 through \(Self.maximumKeyCode)."
            )
        }
        guard allowedMouseButtons.allSatisfy({ $0 <= Self.maximumMouseButton }) else {
            throw AgentTrainerError.invalidConfiguration(
                "RL mouse-button allowlists support buttons 1 through \(Int(Self.maximumMouseButton) + 1)."
            )
        }
        guard hasAnySelection else {
            throw AgentTrainerError.invalidConfiguration(
                "Allow at least one key, mouse button, mouse-movement mode, or scrolling action for RL training."
            )
        }
        return self
    }

    /// Converts the allowlist into the existing disable-only channel contract.
    /// This lets the mature Run input firewall remain the final authority.
    func effectiveChannels(base: ActionChannels, mouseMode: MouseControlMode) -> ActionChannels {
        var result = base
        result.absoluteMouse = base.mouseMovement
            && mouseMode == .absolute && allowsAbsoluteMouseMovement
        result.relativeMouse = base.mouseMovement
            && mouseMode == .relative && allowsRelativeMouseMovement
        result.buttons = base.buttons && !allowedMouseButtons.isEmpty
        result.scroll = base.scroll && allowsScroll
        result.keyboard = base.keyboard && allowedKeyCodes.contains { code in
            !ActionLayout.commandOptionControlKeyCodeSet.contains(code)
        }
        result.modifiers = base.modifiers && !modifierKeyCodes.isEmpty
        return result
    }

    /// Blocks every mouse button outside the explicit RL allowlist while
    /// preserving restrictions the user already placed on the profile.
    func applying(to restrictions: ActionRestrictions) -> ActionRestrictions {
        var result = restrictions
        result.blockedMouseButtons.formUnion(
            (UInt8(0)...Self.maximumMouseButton).filter {
                !allowedMouseButtons.contains($0)
            }
        )
        return result
    }

    func executableKeyCodes(restrictions: ActionRestrictions) -> Set<UInt16> {
        allowedKeyCodes.filter { restrictions.allowsKey($0) }
    }

    func executableMouseButtons(restrictions: ActionRestrictions) -> Set<UInt8> {
        allowedMouseButtons.filter { restrictions.allowsButton($0) }
    }
}

/// Profile-owned online-learning settings. Defaults are conservative and keep
/// the action space empty until the user makes an explicit safety choice.
struct ReinforcementLearningConfiguration: Codable, Hashable, Sendable {
    static let maximumRolloutBytes = 1_024 * 1_024 * 1_024

    var algorithm: ReinforcementLearningAlgorithm = .proximalPolicyOptimization
    var actionSpace = ReinforcementActionSpace()
    // Decode-only migration fields from 2.4.0. AppModel moves them into the
    // universal server settings once, clears them, and never uses them for RL.
    var webSocketURL: String? = nil
    var connectionTimeoutSeconds: Double? = nil
    var rewardTimeoutSeconds: Double? = nil
    var pauseActionsWhenRewardStale: Bool? = nil
    var reconnectInitialDelaySeconds: Double? = nil
    var reconnectMaximumDelaySeconds: Double? = nil

    // Episode and rollout controls.
    var episodeDurationSeconds = 60.0
    /// Zero leaves the episode bounded only by duration or a remote done event.
    var maximumEpisodeSteps = 0
    /// Zero keeps starting episodes until the RL keybind is pressed again.
    var maximumEpisodes = 0
    var rolloutSteps = 64
    var updateEpochs = 4
    var miniBatchSize = 32
    var interEpisodeDelaySeconds = 1.0

    // PPO optimization.
    var learningRate = 0.000_03
    var weightDecay = 0.0
    var discountFactor = 0.99
    var generalizedAdvantageLambda = 0.95
    var policyClipRange = 0.20
    var valueLossCoefficient = 0.50
    var entropyCoefficient = 0.01
    var maximumGradientNorm = 0.50
    /// Zero disables early stopping of an update on approximate KL.
    var targetKLDivergence = 0.03
    var normalizeAdvantages = true
    var clipValueLoss = true
    var bootstrapTimeLimits = true
    var criticHiddenWidth = 256
    var criticLayerCount = 2

    // Hybrid action distribution.
    /// Retained only so 2.4.0 profile JSON remains decodable. New sessions use
    /// `actionBehavior`; a missing behavior is migrated to conservative values.
    var continuousActionStandardDeviation = 0.18
    var binaryActionTemperature = 1.0
    var actionBehavior: ReinforcementActionBehavior? = ReinforcementActionBehavior()

    // Reward conditioning and transport resilience.
    var rewardAggregation: ReinforcementRewardAggregation = .sum
    var rewardScale = 1.0
    /// Zero disables reward clipping.
    var rewardClipMagnitude = 10.0
    var normalizeRewards = false
    var rewardNormalizationEpsilon = 0.000_01
    var interEpisodeStartDelaySeconds = 0.15

    // Persistence and determinism.
    var publishEveryEpisodes = 5
    var checkpointEveryUpdates = 5
    var seed: UInt64 = 7_431

    var effectiveActionBehavior: ReinforcementActionBehavior {
        if let actionBehavior { return actionBehavior }
        // Profiles saved by 2.4.0 used one large standard deviation for every
        // continuous head. Preserve their binary temperature while migrating
        // mouse/scroll exploration to category-appropriate safe values.
        return ReinforcementActionBehavior(
            binaryActionTemperature: binaryActionTemperature,
            absoluteMouseStandardDeviation: min(
                0.05,
                max(0.01, continuousActionStandardDeviation * 0.20)
            ),
            relativeMouseStandardDeviation: min(
                0.12,
                max(0.02, continuousActionStandardDeviation * 0.45)
            ),
            scrollStandardDeviation: min(
                0.10,
                max(0.02, continuousActionStandardDeviation * 0.35)
            )
        )
    }

    func estimatedRolloutBytes(
        profile: AIProfile,
        contextSchema: ModelContextSchema = .empty
    ) -> Int {
        let temporal = profile.training.effectiveTemporalVision
        let past = temporal.pastFrameSpec(from: profile.preprocessing)
        let currentBytes = profile.preprocessing.sampleByteCount
        let pastVision = saturatedMultiply(
            temporal.pastFrameCount,
            past.sampleByteCount
        )
        let pastControls = saturatedMultiply(
            temporal.pastFrameCount,
            ActionLayout.count * MemoryLayout<Float>.size
        )
        let context = saturatedMultiply(
            contextSchema.inputValueCount,
            MemoryLayout<Float>.size
        )
        // Actions, latent samples, values, rewards, and Swift container
        // overhead remain small beside vision but are included conservatively.
        let transitionMetadata = 2 * ActionLayout.count * MemoryLayout<Float>.size + 512
        let perStep = saturatedAdd(
            saturatedAdd(currentBytes, pastVision),
            saturatedAdd(pastControls, saturatedAdd(context, transitionMetadata))
        )
        return saturatedMultiply(rolloutSteps, perStep)
    }

    func validated(requiresActionSelection: Bool = true) throws -> Self {
        if requiresActionSelection { _ = try actionSpace.validated() }
        else {
            guard actionSpace.allowedKeyCodes.allSatisfy({ $0 <= ReinforcementActionSpace.maximumKeyCode }),
                  actionSpace.allowedMouseButtons.allSatisfy({ $0 <= ReinforcementActionSpace.maximumMouseButton }) else {
                throw AgentTrainerError.invalidConfiguration("The RL action allowlist contains an unsupported key or mouse button.")
            }
        }
        let finiteValues = [
            episodeDurationSeconds, interEpisodeDelaySeconds, learningRate,
            weightDecay, discountFactor, generalizedAdvantageLambda,
            policyClipRange, valueLossCoefficient, entropyCoefficient,
            maximumGradientNorm, targetKLDivergence,
            continuousActionStandardDeviation, binaryActionTemperature,
            rewardScale, rewardClipMagnitude, rewardNormalizationEpsilon,
            interEpisodeStartDelaySeconds
        ]
        guard finiteValues.allSatisfy(\.isFinite),
              (1...86_400).contains(episodeDurationSeconds),
              (0...10_000_000).contains(maximumEpisodeSteps),
              (0...10_000_000).contains(maximumEpisodes),
              (8...4_096).contains(rolloutSteps),
              (1...64).contains(updateEpochs),
              (4...rolloutSteps).contains(miniBatchSize),
              (0...60).contains(interEpisodeDelaySeconds),
              (0.000_000_01...0.003).contains(learningRate),
              (0...1).contains(weightDecay),
              (0.8...1).contains(discountFactor),
              (0...1).contains(generalizedAdvantageLambda),
              (0.01...0.5).contains(policyClipRange),
              (0...10).contains(valueLossCoefficient),
              (0...1).contains(entropyCoefficient),
              (0.01...100).contains(maximumGradientNorm),
              (0...10).contains(targetKLDivergence),
              (16...4_096).contains(criticHiddenWidth),
              (1...8).contains(criticLayerCount),
              (0.001...2).contains(continuousActionStandardDeviation),
              (0.1...5).contains(binaryActionTemperature),
              (0.000_001...1_000_000).contains(rewardScale),
              (0...1_000_000).contains(rewardClipMagnitude),
              (0.000_000_001...0.1).contains(rewardNormalizationEpsilon),
              (0...10).contains(interEpisodeStartDelaySeconds),
              (1...100_000).contains(publishEveryEpisodes),
              (1...100_000).contains(checkpointEveryUpdates) else {
            throw AgentTrainerError.invalidConfiguration(
                "RL settings are outside their safe bounds. Check episode/rollout sizes, PPO rates and coefficients, action sampling, reward conditioning, lifecycle timing, and save intervals."
            )
        }
        var value = self
        value.actionBehavior = try effectiveActionBehavior.validated()
        return value
    }

    private func saturatedMultiply(_ lhs: Int, _ rhs: Int) -> Int {
        let result = lhs.multipliedReportingOverflow(by: rhs)
        return result.overflow ? Int.max : result.partialValue
    }

    private func saturatedAdd(_ lhs: Int, _ rhs: Int) -> Int {
        let result = lhs.addingReportingOverflow(rhs)
        return result.overflow ? Int.max : result.partialValue
    }
}

struct ReinforcementLearningSummary: Codable, Hashable, Sendable {
    var episode: Int
    var update: Int
    var environmentSteps: Int
    var lastEpisodeReturn: Double
    var averageEpisodeReturn: Double
}

enum ReinforcementSessionLimits {
    static func completedEpisodes(
        lifetimeCompleted: Int,
        lifetimeAtSessionStart: Int
    ) -> Int {
        max(0, lifetimeCompleted - lifetimeAtSessionStart)
    }

    static func reachedMaximumEpisodes(
        maximumEpisodes: Int,
        lifetimeCompleted: Int,
        lifetimeAtSessionStart: Int
    ) -> Bool {
        maximumEpisodes > 0 && completedEpisodes(
            lifetimeCompleted: lifetimeCompleted,
            lifetimeAtSessionStart: lifetimeAtSessionStart
        ) >= maximumEpisodes
    }
}

enum ReinforcementRolloutPolicy {
    static let preferredMinimumUpdateTransitions = 4

    static func canUpdate(transitionCount: Int) -> Bool {
        transitionCount >= preferredMinimumUpdateTransitions
    }

    static func canUpdateFinalBatch(transitionCount: Int) -> Bool {
        transitionCount > 0
    }
}

enum ReinforcementConnectionState: String, Sendable {
    case disconnected = "Disconnected"
    case listening = "Listening"
    case connected = "Connected"
}

struct ReinforcementLearningMetrics: Sendable {
    var connectionState = ReinforcementConnectionState.disconnected
    var episode = 0
    var sessionEpisode = 0
    var update = 0
    var environmentSteps = 0
    var episodeStep = 0
    var rolloutCount = 0
    var episodeReturn = 0.0
    var lastEpisodeReturn: Double?
    var averageEpisodeReturn = 0.0
    var lastReward = 0.0
    var rewardMessagesPerSecond = 0.0
    var invalidRewardMessages = 0
    var rewardMessagesIgnoredDuringUpdates = 0
    var policyLoss = 0.0
    var valueLoss = 0.0
    var entropy = 0.0
    var approximateKL = 0.0
    var clipFraction = 0.0
    var explainedVariance = 0.0
    var elapsed = 0.0
    var mlxActiveMemory = 0
    var mlxCacheMemory = 0
    var mlxPeakMemory = 0
}
