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
    static let defaultWebSocketURL = "ws://127.0.0.1:47831"
    static let maximumRolloutBytes = 1_024 * 1_024 * 1_024

    var algorithm: ReinforcementLearningAlgorithm = .proximalPolicyOptimization
    var webSocketURL = Self.defaultWebSocketURL
    var actionSpace = ReinforcementActionSpace()

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
    var continuousActionStandardDeviation = 0.18
    var binaryActionTemperature = 1.0

    // Reward conditioning and transport resilience.
    var rewardAggregation: ReinforcementRewardAggregation = .sum
    var rewardScale = 1.0
    /// Zero disables reward clipping.
    var rewardClipMagnitude = 10.0
    var normalizeRewards = false
    var rewardNormalizationEpsilon = 0.000_01
    var connectionTimeoutSeconds = 5.0
    var rewardTimeoutSeconds = 3.0
    var pauseActionsWhenRewardStale = true
    var reconnectInitialDelaySeconds = 0.25
    var reconnectMaximumDelaySeconds = 5.0
    var interEpisodeStartDelaySeconds = 0.15

    // Persistence and determinism.
    var publishEveryEpisodes = 5
    var checkpointEveryUpdates = 5
    var seed: UInt64 = 7_431

    var validatedWebSocketURL: URL? {
        guard let components = URLComponents(string: webSocketURL),
              let scheme = components.scheme?.lowercased(),
              scheme == "ws" || scheme == "wss",
              components.host?.isEmpty == false,
              components.user == nil, components.password == nil,
              components.fragment == nil else { return nil }
        return components.url
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
        guard validatedWebSocketURL != nil else {
            throw AgentTrainerError.invalidConfiguration(
                "The RL reward URL must be a ws:// or wss:// WebSocket URL with a host and no embedded credentials or fragment."
            )
        }
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
            connectionTimeoutSeconds, rewardTimeoutSeconds,
            reconnectInitialDelaySeconds, reconnectMaximumDelaySeconds,
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
              (0.25...60).contains(connectionTimeoutSeconds),
              (0...600).contains(rewardTimeoutSeconds),
              (0.05...60).contains(reconnectInitialDelaySeconds),
              reconnectMaximumDelaySeconds >= reconnectInitialDelaySeconds,
              reconnectMaximumDelaySeconds <= 300,
              (0...10).contains(interEpisodeStartDelaySeconds),
              (1...100_000).contains(publishEveryEpisodes),
              (1...100_000).contains(checkpointEveryUpdates) else {
            throw AgentTrainerError.invalidConfiguration(
                "RL settings are outside their safe bounds. Check episode/rollout sizes, PPO rates and coefficients, action sampling, reward conditioning, reconnect timing, and save intervals."
            )
        }
        return self
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

enum ReinforcementConnectionState: String, Sendable {
    case disconnected = "Disconnected"
    case connecting = "Connecting"
    case connected = "Connected"
    case reconnecting = "Reconnecting"
}

struct ReinforcementLearningMetrics: Sendable {
    var connectionState = ReinforcementConnectionState.disconnected
    var episode = 0
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
