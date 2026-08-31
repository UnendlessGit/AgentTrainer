import Foundation
import MLX
import XCTest
@testable import AgentTrainer

final class ReinforcementLearningTests: XCTestCase {
    func testDefaultRLConfigurationUsesLocalWebSocketAndSafeEmptyAllowlist() throws {
        let configuration = ReinforcementLearningConfiguration()
        XCTAssertEqual(configuration.webSocketURL, "ws://127.0.0.1:47831")
        XCTAssertEqual(configuration.validatedWebSocketURL?.scheme, "ws")
        XCTAssertFalse(configuration.actionSpace.hasAnySelection)
        XCTAssertThrowsError(try configuration.validated())
        XCTAssertNoThrow(try configuration.validated(requiresActionSelection: false))
    }

    func testRLWebSocketURLRejectsUnsafeOrNonWebSocketForms() {
        for value in [
            "http://127.0.0.1:47831",
            "ws://user:password@127.0.0.1:47831",
            "ws://127.0.0.1:47831/#fragment",
            "ws://",
            "not a url"
        ] {
            var configuration = ReinforcementLearningConfiguration()
            configuration.webSocketURL = value
            XCTAssertNil(configuration.validatedWebSocketURL, value)
        }
        var secure = ReinforcementLearningConfiguration()
        secure.webSocketURL = "wss://rewards.example.test/v1"
        XCTAssertEqual(secure.validatedWebSocketURL?.scheme, "wss")
    }

    func testRLActionPresetsAndBoundsAreExact() throws {
        XCTAssertEqual(ReinforcementActionSpace.keyboardWASD.allowedKeyCodes, [0, 1, 2, 13])
        XCTAssertEqual(ReinforcementActionSpace.keyboardArrows.allowedKeyCodes, [123, 124, 125, 126])
        XCTAssertEqual(ReinforcementActionSpace.keyboardNumbers.allowedKeyCodes.count, 10)
        XCTAssertNoThrow(try ReinforcementActionSpace.keyboardWASD.validated())

        var invalidKey = ReinforcementActionSpace.keyboardWASD
        invalidKey.allowedKeyCodes.insert(128)
        XCTAssertThrowsError(try invalidKey.validated())
        let invalidButton = ReinforcementActionSpace(allowedMouseButtons: [8])
        XCTAssertThrowsError(try invalidButton.validated())
    }

    func testRLActionPlanIntersectsAllowlistChannelsAndRestrictions() throws {
        var profile = compactProfile()
        profile.channels = .all
        profile.restrictions = ActionRestrictions(
            blockedKeyCodes: [1, 56],
            blockedMouseButtons: [1]
        )
        var configuration = compactConfiguration()
        configuration.actionSpace = ReinforcementActionSpace(
            allowedKeyCodes: [0, 1, 13, 56, 59],
            allowedMouseButtons: [0, 1],
            allowsAbsoluteMouseMovement: false,
            allowsRelativeMouseMovement: true,
            allowsScroll: true
        )
        let plan = try ReinforcementActionPlan.make(
            profile: profile,
            configuration: configuration,
            mouseMode: .relative
        )
        XCTAssertEqual(plan.allowedKeyCodes, [0, 13, 59])
        XCTAssertEqual(plan.allowedMouseButtons, [0])
        XCTAssertTrue(plan.continuousIndices.contains(ActionLayout.relativeMouse.lowerBound))
        XCTAssertTrue(plan.continuousIndices.contains(ActionLayout.scroll.lowerBound))
        XCTAssertFalse(plan.binaryIndices.contains(ActionLayout.keyboard.lowerBound + 1))
        XCTAssertTrue(plan.binaryIndices.contains(ActionLayout.modifiers.lowerBound + 1))
        XCTAssertFalse(plan.binaryIndices.contains(ActionLayout.shift.lowerBound))
    }

    func testRLAllowlistRemainsADisableOnlyFirewallDuringNormalRun() {
        let actionSpace = ReinforcementActionSpace(
            allowedKeyCodes: [13],
            allowedMouseButtons: [0],
            allowsAbsoluteMouseMovement: false,
            allowsRelativeMouseMovement: false,
            allowsScroll: false
        )
        let channels = actionSpace.effectiveChannels(base: .all, mouseMode: .absolute)
        let restrictions = actionSpace.applying(to: ActionRestrictions())
        var prediction = [Float](repeating: 1, count: ActionLayout.count)
        prediction[ActionLayout.relativeMouse.lowerBound] = -1
        prediction[ActionLayout.scroll.lowerBound] = -1
        let executable = RuntimeActionSemantics.temporalControlValues(
            prediction,
            channels: channels,
            restrictions: restrictions,
            allowedKeyCodes: actionSpace.allowedKeyCodes,
            outputPermissions: RuntimeOutputPermissions(),
            shiftUsesKeyboardChannel: true
        )
        XCTAssertTrue(ActionLayout.absoluteMouse.allSatisfy { executable[$0] == 0 })
        XCTAssertTrue(ActionLayout.relativeMouse.allSatisfy { executable[$0] == 0 })
        XCTAssertTrue(ActionLayout.scroll.allSatisfy { executable[$0] == 0 })
        XCTAssertEqual(executable[ActionLayout.buttons.lowerBound], 1)
        XCTAssertTrue(ActionLayout.buttons.dropFirst().allSatisfy { executable[$0] == 0 })
        XCTAssertEqual(executable[ActionLayout.keyboard.lowerBound + 13], 1)
        XCTAssertEqual(executable[ActionLayout.keyboard.lowerBound], 0)
    }

    func testRLActionSamplingIsDeterministicBoundedAndNeverUsesInactiveOutputs() throws {
        var profile = compactProfile()
        profile.channels = .all
        var configuration = compactConfiguration()
        configuration.actionSpace = ReinforcementActionSpace(
            allowedKeyCodes: [13],
            allowedMouseButtons: [0],
            allowsAbsoluteMouseMovement: true,
            allowsRelativeMouseMovement: false,
            allowsScroll: false
        )
        let plan = try ReinforcementActionPlan.make(
            profile: profile,
            configuration: configuration,
            mouseMode: .absolute
        )
        var firstRandom = ReinforcementRandomGenerator(seed: 99)
        var secondRandom = ReinforcementRandomGenerator(seed: 99)
        let logits = [Float](repeating: 0, count: ActionLayout.count)
        let first = try plan.sample(
            logits: logits,
            standardDeviation: 0.2,
            binaryTemperature: 1,
            random: &firstRandom
        )
        let second = try plan.sample(
            logits: logits,
            standardDeviation: 0.2,
            binaryTemperature: 1,
            random: &secondRandom
        )
        XCTAssertEqual(first.executedActions, second.executedActions)
        XCTAssertEqual(first.distributionActions, second.distributionActions)
        XCTAssertEqual(first.logProbability, second.logProbability)
        XCTAssertTrue(first.executedActions[0...1].allSatisfy { (0...1).contains($0) })
        let active = Set(plan.binaryIndices + plan.continuousIndices)
        for index in 0..<ActionLayout.count where !active.contains(index) {
            XCTAssertEqual(first.executedActions[index], 0, accuracy: 0)
            XCTAssertEqual(first.distributionActions[index], 0, accuracy: 0)
        }
    }

    func testGeneralizedAdvantageHonorsTerminalsAndNormalizes() {
        let unnormalized = ReinforcementAdvantageMath.advantagesAndReturns(
            rewards: [1, 2],
            values: [0.5, 0.25],
            terminals: [false, true],
            bootstrapValue: 99,
            discount: 1,
            lambda: 1,
            normalize: false
        )
        XCTAssertEqual(unnormalized.advantages[1], 1.75, accuracy: 0.0001)
        XCTAssertEqual(unnormalized.advantages[0], 2.5, accuracy: 0.0001)
        XCTAssertEqual(unnormalized.returns, [3, 2])

        let normalized = ReinforcementAdvantageMath.advantagesAndReturns(
            rewards: [1, -1, 0.5, 2],
            values: [0, 0, 0, 0],
            terminals: [false, false, false, true],
            bootstrapValue: 0,
            discount: 0.99,
            lambda: 0.95,
            normalize: true
        )
        XCTAssertEqual(normalized.advantages.reduce(0, +), 0, accuracy: 0.0001)
        XCTAssertTrue(normalized.advantages.allSatisfy(\.isFinite))
    }

    func testRewardAccumulatorSustainsFrequentMessagesWithoutQueueGrowth() throws {
        let accumulator = ReinforcementRewardAccumulator()
        for index in 0..<3_600 {
            let reward = Double(index % 7) / 10
            XCTAssertTrue(accumulator.ingest(Data("{\"reward\":\(reward)}".utf8)))
        }
        XCTAssertTrue(accumulator.ingest(Data("{\"reward\":1.5,\"done\":true}".utf8)))
        let sum = accumulator.drain(aggregation: .sum)
        XCTAssertEqual(sum.messageCount, 3_601)
        XCTAssertTrue(sum.reward.isFinite)
        XCTAssertTrue(sum.episodeFinished)
        XCTAssertEqual(accumulator.drain(aggregation: .sum).messageCount, 0)

        XCTAssertTrue(accumulator.ingest(Data("{\"reward\":2}".utf8)))
        XCTAssertTrue(accumulator.ingest(Data("{\"reward\":4}".utf8)))
        XCTAssertEqual(accumulator.drain(aggregation: .mean).reward, 3, accuracy: 0.0001)
        XCTAssertTrue(accumulator.ingest(Data("{\"reward\":7}".utf8)))
        XCTAssertTrue(accumulator.ingest(Data("{\"reward\":9}".utf8)))
        XCTAssertEqual(accumulator.drain(aggregation: .latest).reward, 9, accuracy: 0.0001)
        let huge = Data("{\"reward\":1.7976931348623157e308}".utf8)
        XCTAssertTrue(accumulator.ingest(huge))
        XCTAssertTrue(accumulator.ingest(huge))
        XCTAssertTrue(accumulator.drain(aggregation: .sum).reward.isFinite)
    }

    func testRewardProtocolRejectsMalformedValuesAndBuildsEpisodeLifecycleJSON() throws {
        XCTAssertNil(RewardWebSocketProtocol.decode(Data("{}".utf8)))
        XCTAssertNil(RewardWebSocketProtocol.decode(Data("{\"reward\":\"1\"}".utf8)))
        XCTAssertNil(RewardWebSocketProtocol.decode(Data("{\"reward\":true}".utf8)))
        XCTAssertNil(RewardWebSocketProtocol.decode(Data("{\"reward\":1e999}".utf8)))
        XCTAssertEqual(
            RewardWebSocketProtocol.decode(Data("{\"type\":\"episode_finished\"}".utf8)),
            .init(reward: nil, episodeFinished: true)
        )
        let message = try RewardWebSocketProtocol.episodeMessage(
            type: "episode_finished",
            episode: 4,
            reason: "time_limit",
            episodeReturn: 12.5,
            steps: 90,
            nextEpisode: 5,
            timestamp: Date(timeIntervalSince1970: 0)
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: message) as? [String: Any]
        )
        XCTAssertEqual(object["type"] as? String, "episode_finished")
        XCTAssertEqual(object["episode"] as? Int, 4)
        XCTAssertEqual(object["next_episode"] as? Int, 5)
        XCTAssertEqual(object["steps"] as? Int, 90)
        XCTAssertEqual(object["return"] as? Double, 12.5)
    }

    func testRewardWebSocketHandlesHighRateRewardsAndOutboundEpisodeLifecycle() async throws {
        let probe = Process()
        probe.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        probe.arguments = ["python3", "-c", "import websockets"]
        probe.standardOutput = Pipe()
        probe.standardError = Pipe()
        try probe.run()
        probe.waitUntilExit()
        guard probe.terminationStatus == 0 else {
            throw XCTSkip("Python websockets is unavailable for the local integration test.")
        }

        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let portFile = directory.appendingPathComponent("port.txt")
        let messagesFile = directory.appendingPathComponent("messages.json")
        let script = #"""
import asyncio, json, sys, websockets

port_file, messages_file = sys.argv[1], sys.argv[2]

async def main():
    finished = asyncio.Event()
    async def handler(socket):
        for _ in range(120):
            await socket.send(json.dumps({"reward": 0.25}))
            await asyncio.sleep(1 / 120)
        received = []
        while len(received) < 3:
            message = json.loads(await asyncio.wait_for(socket.recv(), timeout=5))
            received.append(message.get("type"))
        with open(messages_file, "w", encoding="utf-8") as output:
            json.dump(received, output)
        finished.set()
    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        with open(port_file, "w", encoding="utf-8") as output:
            output.write(str(port))
        await asyncio.wait_for(finished.wait(), timeout=10)

asyncio.run(main())
"""#
        let server = Process()
        server.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        server.arguments = [
            "python3", "-c", script, portFile.path, messagesFile.path
        ]
        server.standardOutput = Pipe()
        server.standardError = Pipe()
        try server.run()
        defer { if server.isRunning { server.terminate() } }
        try await waitUntil(timeout: 5) {
            FileManager.default.fileExists(atPath: portFile.path)
        }
        let port = try XCTUnwrap(
            Int(String(contentsOf: portFile, encoding: .utf8))
        )
        let accumulator = ReinforcementRewardAccumulator()
        let client = RewardWebSocketClient(accumulator: accumulator) { _, _ in }
        try await client.start(
            url: try XCTUnwrap(URL(string: "ws://127.0.0.1:\(port)")),
            connectionTimeout: 3,
            reconnectInitialDelay: 0.05,
            reconnectMaximumDelay: 0.2
        )
        try await waitUntil(timeout: 5) {
            accumulator.statistics().received >= 120
        }
        for type in ["episode_finished", "episode_starting", "episode_started"] {
            await client.send(try RewardWebSocketProtocol.episodeMessage(
                type: type,
                episode: type == "episode_finished" ? 1 : 2
            ))
        }
        try await waitUntil(timeout: 5) {
            FileManager.default.fileExists(atPath: messagesFile.path)
        }
        await client.stop()
        server.waitUntilExit()
        XCTAssertEqual(server.terminationStatus, 0)
        let receivedTypes = try JSONDecoder().decode(
            [String].self,
            from: Data(contentsOf: messagesFile)
        )
        XCTAssertEqual(
            receivedTypes,
            ["episode_finished", "episode_starting", "episode_started"]
        )
        let reward = accumulator.drain(aggregation: .sum)
        XCTAssertEqual(reward.messageCount, 120)
        XCTAssertEqual(reward.reward, 30, accuracy: 0.0001)
    }

    func testRewardWebSocketReconnectsAfterEstablishedConnectionDrops() async throws {
        let probe = Process()
        probe.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        probe.arguments = ["python3", "-c", "import websockets"]
        probe.standardOutput = Pipe()
        probe.standardError = Pipe()
        try probe.run()
        probe.waitUntilExit()
        guard probe.terminationStatus == 0 else {
            throw XCTSkip("Python websockets is unavailable for the local integration test.")
        }

        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let portFile = directory.appendingPathComponent("port.txt")
        let messageFile = directory.appendingPathComponent("message.txt")
        let script = #"""
import asyncio, json, sys, websockets

port_file, message_file = sys.argv[1], sys.argv[2]

async def main():
    finished = asyncio.Event()
    connections = 0
    async def handler(socket):
        nonlocal connections
        connections += 1
        if connections == 1:
            await socket.send(json.dumps({"reward": 1}))
            await asyncio.sleep(0.35)
            await socket.close()
            return
        await socket.send(json.dumps({"reward": 2}))
        message = json.loads(await asyncio.wait_for(socket.recv(), timeout=5))
        with open(message_file, "w", encoding="utf-8") as output:
            output.write(message.get("type", ""))
        finished.set()
    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        with open(port_file, "w", encoding="utf-8") as output:
            output.write(str(server.sockets[0].getsockname()[1]))
        await asyncio.wait_for(finished.wait(), timeout=10)

asyncio.run(main())
"""#
        let server = Process()
        server.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        server.arguments = [
            "python3", "-c", script, portFile.path, messageFile.path
        ]
        server.standardOutput = Pipe()
        server.standardError = Pipe()
        try server.run()
        defer { if server.isRunning { server.terminate() } }
        try await waitUntil(timeout: 5) {
            FileManager.default.fileExists(atPath: portFile.path)
        }
        let port = try XCTUnwrap(
            Int(String(contentsOf: portFile, encoding: .utf8))
        )
        let accumulator = ReinforcementRewardAccumulator()
        let client = RewardWebSocketClient(accumulator: accumulator) { _, _ in }
        try await client.start(
            url: try XCTUnwrap(URL(string: "ws://127.0.0.1:\(port)")),
            connectionTimeout: 3,
            reconnectInitialDelay: 0.05,
            reconnectMaximumDelay: 0.2
        )
        try await waitUntil(timeout: 6) {
            accumulator.statistics().received >= 2
        }
        await client.send(try RewardWebSocketProtocol.episodeMessage(
            type: "episode_started",
            episode: 2
        ))
        try await waitUntil(timeout: 5) {
            FileManager.default.fileExists(atPath: messageFile.path)
        }
        await client.stop()
        server.waitUntilExit()
        XCTAssertEqual(server.terminationStatus, 0)
        XCTAssertEqual(
            try String(contentsOf: messageFile, encoding: .utf8),
            "episode_started"
        )
        let rewards = accumulator.drain(aggregation: .sum)
        XCTAssertEqual(rewards.messageCount, 2)
        XCTAssertEqual(rewards.reward, 3, accuracy: 0.0001)
    }

    func testRLConfigurationValidatesOptimizationAndRolloutBounds() throws {
        var configuration = compactConfiguration()
        XCTAssertNoThrow(try configuration.validated())
        configuration.miniBatchSize = configuration.rolloutSteps + 1
        XCTAssertThrowsError(try configuration.validated())
        configuration = compactConfiguration()
        configuration.discountFactor = 0.5
        XCTAssertThrowsError(try configuration.validated())

        let profile = compactProfile()
        let bytes = compactConfiguration().estimatedRolloutBytes(profile: profile)
        XCTAssertGreaterThan(bytes, profile.preprocessing.sampleByteCount * 8)
        XCTAssertLessThan(bytes, ReinforcementLearningConfiguration.maximumRolloutBytes)
    }

    func testPPOUpdateIsFiniteAndActorWeightsRoundTripIntoNormalPolicy() throws {
        let profile = compactProfile()
        let configuration = compactConfiguration()
        let plan = try ReinforcementActionPlan.make(
            profile: profile,
            configuration: configuration,
            mouseMode: .absolute
        )
        let optimizer = try ReinforcementPolicyOptimizer(
            profile: profile,
            configuration: configuration,
            contextSchema: .empty,
            actionPlan: plan,
            actorWeightsURL: nil
        )
        let observation = ReinforcementObservation(
            currentPacked: Data(repeating: 127, count: profile.preprocessing.sampleByteCount),
            pastPacked: Data(),
            pastControls: [],
            contextValues: []
        )
        let convolutionParameter = try XCTUnwrap(
            optimizer.model.actor.parameters().flattened().first {
                $0.0.contains("convolutions")
            }
        )
        MLX.eval(convolutionParameter.1)
        let convolutionBefore = convolutionParameter.1.asArray(Float.self)
        var random = ReinforcementRandomGenerator(seed: 123)
        var transitions: [ReinforcementTransition] = []
        for index in 0..<8 {
            let prediction = try optimizer.prediction(for: observation)
            let sample = try plan.sample(
                logits: prediction.logits,
                standardDeviation: Float(configuration.continuousActionStandardDeviation),
                binaryTemperature: Float(configuration.binaryActionTemperature),
                random: &random
            )
            transitions.append(ReinforcementTransition(
                observation: observation,
                distributionActions: sample.distributionActions,
                executedActions: sample.executedActions,
                oldLogProbability: sample.logProbability,
                oldValue: prediction.value,
                reward: index.isMultiple(of: 2) ? 1 : -0.25,
                terminal: index == 7
            ))
        }
        let update = try optimizer.update(
            transitions: transitions,
            bootstrapValue: 0,
            shuffleSeed: 77
        )
        XCTAssertGreaterThan(update.optimizerSteps, 0)
        XCTAssertTrue(update.policyLoss.isFinite)
        XCTAssertTrue(update.valueLoss.isFinite)
        XCTAssertTrue(update.entropy.isFinite)
        XCTAssertEqual(optimizer.optimizer.step, update.optimizerSteps)
        let convolutionAfterArray = try XCTUnwrap(
            optimizer.model.actor.parameters().flattened().first {
                $0.0 == convolutionParameter.0
            }
        ).1
        MLX.eval(convolutionAfterArray)
        XCTAssertNotEqual(convolutionBefore, convolutionAfterArray.asArray(Float.self))

        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let actorURL = directory.appendingPathComponent("actor.safetensors")
        let reinforcementURL = directory.appendingPathComponent("rl.safetensors")
        let optimizerURL = directory.appendingPathComponent("optimizer.safetensors")
        let randomURL = directory.appendingPathComponent("random.safetensors")
        try optimizer.saveActor(to: actorURL)
        try optimizer.saveReinforcementWeights(to: reinforcementURL)
        try optimizer.saveOptimizer(to: optimizerURL)
        try optimizer.saveRandomState(to: randomURL)
        let restored = try ReinforcementPolicyOptimizer(
            profile: profile,
            configuration: configuration,
            contextSchema: .empty,
            actionPlan: plan,
            actorWeightsURL: nil,
            reinforcementWeightsURL: reinforcementURL,
            optimizerURL: optimizerURL,
            randomStateURL: randomURL
        )
        let trainedPrediction = try optimizer.prediction(for: observation)
        let restoredPrediction = try restored.prediction(for: observation)
        XCTAssertEqual(restored.optimizer.step, optimizer.optimizer.step)
        XCTAssertEqual(restoredPrediction.logits, trainedPrediction.logits)
        XCTAssertEqual(restoredPrediction.value, trainedPrediction.value)
        let normalRunActor = AgentPolicy(profile: profile)
        XCTAssertNoThrow(try normalRunActor.loadWeights(from: actorURL))
        XCTAssertEqual(
            normalRunActor.parameters().flattened().map(\.0),
            optimizer.model.actor.parameters().flattened().map(\.0)
        )
        for (loaded, trained) in zip(
            normalRunActor.parameters().flattened().map(\.1),
            optimizer.model.actor.parameters().flattened().map(\.1)
        ) {
            MLX.eval(loaded, trained)
            XCTAssertEqual(loaded.asArray(Float.self), trained.asArray(Float.self))
        }
    }

    func testRLCanWarmStartFromDemonstrationActorWeights() throws {
        let profile = compactProfile()
        let configuration = compactConfiguration()
        let plan = try ReinforcementActionPlan.make(
            profile: profile,
            configuration: configuration,
            mouseMode: .absolute
        )
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let actorURL = directory.appendingPathComponent("demonstration.safetensors")
        let demonstrationActor = AgentPolicy(profile: profile)
        try demonstrationActor.saveWeights(to: actorURL)
        let optimizer = try ReinforcementPolicyOptimizer(
            profile: profile,
            configuration: configuration,
            contextSchema: .empty,
            actionPlan: plan,
            actorWeightsURL: actorURL
        )
        for (demonstration, reinforcement) in zip(
            demonstrationActor.parameters().flattened().map(\.1),
            optimizer.model.actor.parameters().flattened().map(\.1)
        ) {
            MLX.eval(demonstration, reinforcement)
            XCTAssertEqual(
                demonstration.asArray(Float.self),
                reinforcement.asArray(Float.self)
            )
        }
    }

    func testRLVersionMetadataRequiresSummaryAndImmutableAllowlist() {
        let profile = compactProfile()
        var valid = modelVersion(profile: profile)
        valid.trainingOrigin = .reinforcementLearning
        valid.reinforcementLearningSummary = ReinforcementLearningSummary(
            episode: 3,
            update: 2,
            environmentSteps: 120,
            lastEpisodeReturn: 4,
            averageEpisodeReturn: 3
        )
        valid.reinforcementActionSpace = .keyboardWASD
        XCTAssertTrue(valid.reinforcementMetadataIsValid)
        valid.reinforcementLearningSummary = nil
        XCTAssertFalse(valid.reinforcementMetadataIsValid)
        valid.trainingOrigin = nil
        valid.reinforcementActionSpace = nil
        XCTAssertTrue(valid.reinforcementMetadataIsValid)
    }

    func testRLVersionCommitAndLearningResetAreTransactional() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(root: root)
        try await workspace.prepare()
        var profile = compactProfile()
        profile.reinforcementLearning = compactConfiguration()
        try await workspace.saveProfile(profile)
        var version = modelVersion(profile: profile)
        version.trainingOrigin = .reinforcementLearning
        version.reinforcementLearningSummary = ReinforcementLearningSummary(
            episode: 1,
            update: 1,
            environmentSteps: 8,
            lastEpisodeReturn: 1,
            averageEpisodeReturn: 1
        )
        version.reinforcementActionSpace = compactConfiguration().actionSpace
        let versionDirectory = await workspace.versionDirectory(
            profileID: profile.id,
            versionID: version.id
        )
        try FileManager.default.createDirectory(at: versionDirectory, withIntermediateDirectories: true)
        try Data("weights".utf8).write(to: versionDirectory.appendingPathComponent(version.weightsFile))
        let checkpoint = await workspace.reinforcementCheckpointDirectory(profileID: profile.id)
        try FileManager.default.createDirectory(at: checkpoint, withIntermediateDirectories: true)
        try Data("checkpoint".utf8).write(to: checkpoint.appendingPathComponent("state.json"))
        let progress = TrainingProgressSummary(
            globalStep: 1,
            epoch: 0,
            updatedAt: Date(),
            savedBrainCount: 1,
            reinforcementEpisodes: 1,
            reinforcementUpdates: 1,
            reinforcementAverageReturn: 1
        )
        profile = try await workspace.commitReinforcementVersion(
            version,
            profile: profile,
            progress: progress
        )
        XCTAssertEqual(profile.activeVersionID, version.id)
        let versions = await workspace.listVersions(profileID: profile.id)
        XCTAssertEqual(versions.count, 1)
        let activation = try await workspace.activateVersion(
            profile: profile,
            versionID: version.id
        )
        XCTAssertEqual(activation.profile.trainingProgress?.reinforcementEpisodes, 1)
        XCTAssertEqual(activation.profile.trainingProgress?.reinforcementUpdates, 1)
        XCTAssertEqual(
            activation.profile.trainingProgress?.reinforcementAverageReturn,
            1
        )
        profile = activation.profile
        let reset = try await workspace.resetLearning(for: profile)
        XCTAssertNil(reset.activeVersionID)
        XCTAssertFalse(FileManager.default.fileExists(atPath: checkpoint.path))
    }

    func testHotkeySettingsDecodeAddsDedicatedRLDefault() throws {
        let legacy = """
        {
          "panic":{"keyCode":53,"carbonModifiers":0},
          "record":{"keyCode":15,"carbonModifiers":0},
          "run":{"keyCode":0,"carbonModifiers":0}
        }
        """
        let decoded = try JSONDecoder().decode(
            HotkeySettings.self,
            from: Data(legacy.utf8)
        )
        XCTAssertEqual(decoded.reinforcementLearning, .reinforcementLearning)
        XCTAssertEqual(Set([
            decoded.panic, decoded.record, decoded.run,
            decoded.reinforcementLearning
        ]).count, 4)

        let customRun = HotkeyBinding.reinforcementLearning
        let conflicting = HotkeySettings(
            panic: .panic,
            record: .record,
            run: customRun,
            reinforcementLearning: .reinforcementLearning
        ).resolvingReinforcementConflict()
        XCTAssertEqual(conflicting.run, customRun)
        XCTAssertEqual(conflicting.panic, .panic)
        XCTAssertEqual(conflicting.record, .record)
        XCTAssertNotEqual(conflicting.reinforcementLearning, customRun)
        XCTAssertEqual(Set([
            conflicting.panic, conflicting.record, conflicting.run,
            conflicting.reinforcementLearning
        ]).count, 4)
    }

    private func compactProfile() -> AIProfile {
        var profile = AIProfile.fresh(name: "RL Test")
        profile.preprocessing = PreprocessingSpec(
            width: 16,
            height: 16,
            colorMode: .grayscale,
            bitDepth: 8,
            chroma: .yuv420,
            resizePolicy: .stretch
        )
        profile.channels = ActionChannels(
            absoluteMouse: false,
            relativeMouse: false,
            buttons: false,
            scroll: false,
            keyboard: true,
            modifiers: false
        )
        profile.training.temporalVision = TemporalVisionConfiguration(
            pastFrameCount: 0,
            frameSpacing: 1,
            downsampleFactor: 1
        )
        profile.training.precision = .float32
        profile.training.architecture = ArchitectureSpec(
            convolutionChannels: [4],
            kernelSizes: [3],
            strides: [2],
            visualEmbedding: 8,
            recurrentKind: .gru,
            recurrentWidth: 8,
            fusionWidths: [12],
            dropout: 0,
            visualPooling: .attention,
            attentionHeads: 2
        )
        return profile
    }

    private func compactConfiguration() -> ReinforcementLearningConfiguration {
        var configuration = ReinforcementLearningConfiguration()
        configuration.actionSpace = ReinforcementActionSpace(
            allowedKeyCodes: [13]
        )
        configuration.rolloutSteps = 8
        configuration.miniBatchSize = 4
        configuration.updateEpochs = 1
        configuration.criticHiddenWidth = 16
        configuration.criticLayerCount = 1
        configuration.learningRate = 0.0003
        return configuration
    }

    private func modelVersion(profile: AIProfile) -> ModelVersionManifest {
        ModelVersionManifest(
            id: UUID(),
            name: "Test Brain",
            createdAt: Date(),
            globalStep: 0,
            trainingLoss: 0,
            validationLoss: nil,
            preprocessing: profile.preprocessing,
            channels: profile.channels,
            training: profile.training,
            demonstratedKeyCodes: [13],
            relativeMouseScale: GameCameraContract.deltaScale,
            trainingDataSchema: TrainingDataContract.schemaVersion
        )
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AgentTrainer-RLTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        return directory
    }

    private func waitUntil(
        timeout: TimeInterval,
        condition: @escaping @Sendable () -> Bool
    ) async throws {
        let deadline = ContinuousClock.now + .seconds(timeout)
        while !condition() {
            guard ContinuousClock.now < deadline else {
                XCTFail("Timed out waiting for local integration state.")
                throw CocoaError(.coderReadCorrupt)
            }
            try await Task.sleep(for: .milliseconds(20))
        }
    }
}
