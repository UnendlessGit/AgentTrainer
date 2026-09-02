import Foundation
import MLX
import XCTest
@testable import AgentTrainer

final class ReinforcementLearningTests: XCTestCase {
    func testDefaultRLConfigurationUsesSafeEmptyAllowlistAndGlobalServer() throws {
        let configuration = ReinforcementLearningConfiguration()
        XCTAssertFalse(configuration.actionSpace.hasAnySelection)
        XCTAssertThrowsError(try configuration.validated())
        XCTAssertNoThrow(try configuration.validated(requiresActionSelection: false))
        let server = RewardWebSocketServerSettings()
        XCTAssertEqual(server.url, "ws://127.0.0.1:47831")
        XCTAssertEqual(server.validatedURL?.scheme, "ws")
        XCTAssertFalse(server.allowRemoteConnections)
        XCTAssertTrue(server.stopAndSaveOnDisconnect)
        XCTAssertTrue(server.stopAndSaveOnRewardTimeout)
        XCTAssertNoThrow(try server.validated())
    }

    func testRewardServerURLRejectsUnsafeRemoteAndNonServerForms() {
        for value in [
            "http://127.0.0.1:47831",
            "ws://user:password@127.0.0.1:47831",
            "ws://127.0.0.1:47831/#fragment",
            "ws://127.0.0.1",
            "ws://192.168.1.5:47831",
            "ws://",
            "not a url"
        ] {
            var settings = RewardWebSocketServerSettings()
            settings.url = value
            XCTAssertNil(settings.validatedURL, value)
        }
        var remote = RewardWebSocketServerSettings()
        remote.allowRemoteConnections = true
        remote.url = "ws://192.168.1.5:47831"
        XCTAssertEqual(remote.validatedURL?.host, "192.168.1.5")
        remote.url = "wss://192.168.1.5:47831"
        XCTAssertNil(remote.validatedURL)
        var spaced = RewardWebSocketServerSettings()
        spaced.url = "  ws://127.0.0.1:47831  "
        XCTAssertEqual(try? spaced.validated().url, "ws://127.0.0.1:47831")
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
        XCTAssertEqual(KeyNames.name(for: 49), "Space")
        XCTAssertEqual(KeyNames.name(for: 56), "Left Shift")
        XCTAssertEqual(KeyNames.name(for: 59), "Left Control")
        XCTAssertEqual(KeyNames.name(for: 63), "Fn / Globe")
        XCTAssertEqual(KeyNames.name(for: 117), "Forward Delete")
    }

    func testRLRuntimeUsesBernoulliSamplingInsteadOfFixedThresholds() throws {
        var profile = compactProfile()
        profile.channels = .all
        var configuration = compactConfiguration()
        configuration.actionSpace = ReinforcementActionSpace(
            allowedKeyCodes: [13]
        )
        let plan = try ReinforcementActionPlan.make(
            profile: profile,
            configuration: configuration,
            mouseMode: .absolute
        )
        var logits = [Float](repeating: -20, count: ActionLayout.count)
        let keyIndex = ActionLayout.keyboard.lowerBound + 13
        logits[keyIndex] = -0.4 // Probability is below 0.5 but still meaningful.
        var random = ReinforcementRandomGenerator(seed: 44)
        var sampled: Set<Float> = []
        for _ in 0..<64 {
            sampled.insert(try plan.sample(
                logits: logits,
                behavior: configuration.effectiveActionBehavior,
                includesContinuousExploration: false,
                random: &random
            ).executedActions[keyIndex])
        }
        XCTAssertEqual(sampled, [0, 1])
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

    func testRLActionPlanRejectsActionsDisabledByLiveOutputPermissions() throws {
        var profile = compactProfile()
        profile.channels = .all
        var keyboard = compactConfiguration()
        keyboard.actionSpace = ReinforcementActionSpace(allowedKeyCodes: [13])
        XCTAssertThrowsError(try ReinforcementActionPlan.make(
            profile: profile,
            configuration: keyboard,
            mouseMode: .absolute,
            outputPermissions: RuntimeOutputPermissions(
                cursorMovement: true,
                keyboard: false
            )
        ))

        var cursor = compactConfiguration()
        cursor.actionSpace = ReinforcementActionSpace(
            allowsAbsoluteMouseMovement: true
        )
        XCTAssertThrowsError(try ReinforcementActionPlan.make(
            profile: profile,
            configuration: cursor,
            mouseMode: .absolute,
            outputPermissions: RuntimeOutputPermissions(
                cursorMovement: false,
                keyboard: true
            )
        ))
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
            behavior: configuration.effectiveActionBehavior,
            random: &firstRandom
        )
        let second = try plan.sample(
            logits: logits,
            behavior: configuration.effectiveActionBehavior,
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

    func testRLMouseFilterBoundsJitterAndUsesTheSameDeterministicContract() throws {
        var profile = compactProfile()
        profile.channels = .all
        var configuration = compactConfiguration()
        configuration.actionSpace = ReinforcementActionSpace(
            allowsAbsoluteMouseMovement: true,
            allowsRelativeMouseMovement: true,
            allowsScroll: true
        )
        var behavior = ReinforcementActionBehavior()
        behavior.absoluteMouseSmoothing = 0.5
        behavior.relativeMouseSmoothing = 0.5
        behavior.maximumAbsoluteMouseStep = 0.05
        let absolutePlan = try ReinforcementActionPlan.make(
            profile: profile,
            configuration: configuration,
            mouseMode: .absolute
        )
        var first = ReinforcementActionFilter()
        var second = ReinforcementActionFilter()
        var initial = [Float](repeating: 0, count: ActionLayout.count)
        initial[0] = 0.2
        initial[1] = 0.2
        XCTAssertEqual(
            first.apply(to: initial, plan: absolutePlan, behavior: behavior),
            second.apply(to: initial, plan: absolutePlan, behavior: behavior)
        )
        var jump = initial
        jump[0] = 1
        jump[1] = 1
        let filtered = first.apply(
            to: jump,
            plan: absolutePlan,
            behavior: behavior
        )
        let distance = hypot(filtered[0] - 0.2, filtered[1] - 0.2)
        XCTAssertLessThanOrEqual(distance, 0.050_01)

        let relativePlan = try ReinforcementActionPlan.make(
            profile: profile,
            configuration: configuration,
            mouseMode: .relative
        )
        var relativeFilter = ReinforcementActionFilter()
        var tiny = [Float](repeating: 0, count: ActionLayout.count)
        tiny[2] = 0.005
        tiny[3] = -0.005
        let deadZoned = relativeFilter.apply(
            to: tiny,
            plan: relativePlan,
            behavior: behavior
        )
        XCTAssertEqual(deadZoned[2], 0, accuracy: 0.000_001)
        XCTAssertEqual(deadZoned[3], 0, accuracy: 0.000_001)
        var extreme = tiny
        extreme[2] = 1
        extreme[3] = 1
        let bounded = relativeFilter.apply(
            to: extreme,
            plan: relativePlan,
            behavior: behavior
        )
        XCTAssertLessThanOrEqual(
            hypot(bounded[2], bounded[3]),
            Float(behavior.maximumRelativeMouseMagnitude) + 0.000_01
        )
    }

    func testMaximumEpisodesCountsOnlyTheCurrentTrainingRun() {
        XCTAssertEqual(
            ReinforcementSessionLimits.completedEpisodes(
                lifetimeCompleted: 104,
                lifetimeAtSessionStart: 100
            ),
            4
        )
        XCTAssertFalse(ReinforcementSessionLimits.reachedMaximumEpisodes(
            maximumEpisodes: 5,
            lifetimeCompleted: 104,
            lifetimeAtSessionStart: 100
        ))
        XCTAssertTrue(ReinforcementSessionLimits.reachedMaximumEpisodes(
            maximumEpisodes: 5,
            lifetimeCompleted: 105,
            lifetimeAtSessionStart: 100
        ))
        XCTAssertFalse(ReinforcementSessionLimits.reachedMaximumEpisodes(
            maximumEpisodes: 0,
            lifetimeCompleted: 10_000,
            lifetimeAtSessionStart: 0
        ))
        XCTAssertFalse(ReinforcementRolloutPolicy.canUpdate(transitionCount: 3))
        XCTAssertTrue(ReinforcementRolloutPolicy.canUpdate(transitionCount: 4))
        XCTAssertTrue(ReinforcementRolloutPolicy.canUpdateFinalBatch(transitionCount: 1))
        XCTAssertFalse(ReinforcementRolloutPolicy.canUpdateFinalBatch(transitionCount: 0))
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

    func testRewardProtocolCustomNamesCoverEveryIncomingAndOutgoingMessage() throws {
        let names = try RewardWebSocketMessageNames(
            typeField: "event",
            rewardField: "score",
            doneField: "reset",
            incomingEpisodeFinished: "round_over",
            incomingHeartbeat: "alive",
            outgoingEpisodeFinished: "round_finished",
            outgoingEpisodeStarting: "round_starting",
            outgoingEpisodeStarted: "round_started",
            outgoingTrainingUpdateStarted: "learning_started",
            outgoingTrainingUpdateFinished: "learning_finished"
        ).validated()
        XCTAssertEqual(
            RewardWebSocketProtocol.decode(
                Data(#"{"score":2.5,"event":"alive"}"#.utf8),
                names: names
            ),
            .init(reward: 2.5, episodeFinished: false, heartbeat: true)
        )
        XCTAssertEqual(
            RewardWebSocketProtocol.decode(
                Data(#"{"event":"round_over"}"#.utf8),
                names: names
            )?.episodeFinished,
            true
        )
        let outgoing = try RewardWebSocketProtocol.episodeMessage(
            event: .trainingUpdateFinished,
            names: names,
            episode: 8
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: outgoing) as? [String: Any]
        )
        XCTAssertEqual(object["event"] as? String, "learning_finished")
        XCTAssertNil(object["type"])
        var collision = names
        collision.typeField = "episode"
        XCTAssertThrowsError(try collision.validated())
    }

    func testRewardWebSocketHandlesHighRateRewardsAndOutboundEpisodeLifecycle() async throws {
        try requirePythonWebSockets()

        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let messagesFile = directory.appendingPathComponent("messages.json")
        let port = try unusedLoopbackPort()
        var settings = RewardWebSocketServerSettings()
        settings.url = "ws://127.0.0.1:\(port)"
        let accumulator = ReinforcementRewardAccumulator()
        let rewardServer = RewardWebSocketServer(
            accumulator: accumulator
        ) { _, _ in }
        try await rewardServer.start(settings: settings)
        defer { Task { await rewardServer.stop() } }

        let script = #"""
import asyncio, json, sys, websockets

url, messages_file = sys.argv[1], sys.argv[2]

async def main():
    async with websockets.connect(url) as socket:
        for _ in range(120):
            await socket.send(json.dumps({"reward": 0.25}))
            await asyncio.sleep(1 / 120)
        received = []
        while len(received) < 3:
            message = json.loads(await asyncio.wait_for(socket.recv(), timeout=5))
            received.append(message.get("type"))
        with open(messages_file, "w", encoding="utf-8") as output:
            json.dump(received, output)

asyncio.run(main())
"""#
        let rewarder = Process()
        rewarder.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        rewarder.arguments = [
            "python3", "-c", script, settings.url, messagesFile.path
        ]
        rewarder.standardOutput = Pipe()
        rewarder.standardError = Pipe()
        try rewarder.run()
        defer { if rewarder.isRunning { rewarder.terminate() } }
        try await rewardServer.waitForRewarder(timeout: 3)
        try await waitUntil(timeout: 5) {
            accumulator.statistics().received >= 120
        }
        for event in [
            RewardWebSocketProtocol.OutgoingEvent.episodeFinished,
            .episodeStarting,
            .episodeStarted
        ] {
            await rewardServer.send(try RewardWebSocketProtocol.episodeMessage(
                event: event,
                episode: event == .episodeFinished ? 1 : 2
            ))
        }
        try await waitUntil(timeout: 5) {
            FileManager.default.fileExists(atPath: messagesFile.path)
        }
        try await waitUntil(timeout: 5) { !rewarder.isRunning }
        XCTAssertEqual(rewarder.terminationStatus, 0)
        await rewardServer.stop()
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
        try requirePythonWebSockets()

        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let messageFile = directory.appendingPathComponent("message.txt")
        let port = try unusedLoopbackPort()
        var settings = RewardWebSocketServerSettings()
        settings.url = "ws://127.0.0.1:\(port)"
        let accumulator = ReinforcementRewardAccumulator()
        let rewardServer = RewardWebSocketServer(
            accumulator: accumulator
        ) { _, _ in }
        try await rewardServer.start(settings: settings)
        defer { Task { await rewardServer.stop() } }

        let script = #"""
import asyncio, json, sys, websockets

url, message_file = sys.argv[1], sys.argv[2]

async def main():
    async with websockets.connect(url) as socket:
            await socket.send(json.dumps({"reward": 1}))
            await asyncio.sleep(0.25)
            await socket.close()
    await asyncio.sleep(0.15)
    async with websockets.connect(url) as socket:
        await socket.send(json.dumps({"reward": 2}))
        message = json.loads(await asyncio.wait_for(socket.recv(), timeout=5))
        with open(message_file, "w", encoding="utf-8") as output:
            output.write(message.get("type", ""))

asyncio.run(main())
"""#
        let rewarder = Process()
        rewarder.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        rewarder.arguments = [
            "python3", "-c", script, settings.url, messageFile.path
        ]
        rewarder.standardOutput = Pipe()
        rewarder.standardError = Pipe()
        try rewarder.run()
        defer { if rewarder.isRunning { rewarder.terminate() } }
        try await rewardServer.waitForRewarder(timeout: 3)
        try await waitUntil(timeout: 6) {
            accumulator.statistics().received >= 2
        }
        await rewardServer.send(try RewardWebSocketProtocol.episodeMessage(
            event: .episodeStarted,
            episode: 2
        ))
        try await waitUntil(timeout: 5) {
            FileManager.default.fileExists(atPath: messageFile.path)
        }
        try await waitUntil(timeout: 5) { !rewarder.isRunning }
        XCTAssertEqual(rewarder.terminationStatus, 0)
        await rewardServer.stop()
        XCTAssertEqual(
            try String(contentsOf: messageFile, encoding: .utf8),
            "episode_started"
        )
        let rewards = accumulator.drain(aggregation: .sum)
        XCTAssertEqual(rewards.messageCount, 2)
        XCTAssertEqual(rewards.reward, 3, accuracy: 0.0001)
    }

    func testHostedRewardServerAppliesCustomMessageNamesEndToEnd() async throws {
        try requirePythonWebSockets()
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let responseFile = directory.appendingPathComponent("custom-message.txt")
        var settings = RewardWebSocketServerSettings()
        settings.url = "ws://127.0.0.1:\(try unusedLoopbackPort())"
        settings.messageNames.typeField = "event"
        settings.messageNames.rewardField = "score"
        settings.messageNames.doneField = "round_done"
        settings.messageNames.incomingHeartbeat = "still_alive"
        settings.messageNames.outgoingEpisodeStarted = "round_started"

        let accumulator = ReinforcementRewardAccumulator()
        let server = RewardWebSocketServer(accumulator: accumulator) { _, _ in }
        try await server.start(settings: settings)
        defer { Task { await server.stop() } }
        let script = #"""
import asyncio, json, sys, websockets

async def main():
    async with websockets.connect(sys.argv[1]) as socket:
        await socket.send(json.dumps({"score": 3.5, "event": "still_alive"}))
        message = json.loads(await asyncio.wait_for(socket.recv(), timeout=5))
        with open(sys.argv[2], "w", encoding="utf-8") as output:
            output.write(message.get("event", ""))

asyncio.run(main())
"""#
        let rewarder = Process()
        rewarder.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        rewarder.arguments = [
            "python3", "-c", script, settings.url, responseFile.path
        ]
        rewarder.standardOutput = Pipe()
        rewarder.standardError = Pipe()
        try rewarder.run()
        defer { if rewarder.isRunning { rewarder.terminate() } }
        try await server.waitForRewarder(timeout: 3)
        try await waitUntil(timeout: 3) {
            accumulator.statistics().received >= 1
        }
        await server.send(try RewardWebSocketProtocol.episodeMessage(
            event: .episodeStarted,
            names: settings.messageNames,
            episode: 1
        ))
        try await waitUntil(timeout: 5) {
            FileManager.default.fileExists(atPath: responseFile.path)
        }
        try await waitUntil(timeout: 5) { !rewarder.isRunning }
        XCTAssertEqual(rewarder.terminationStatus, 0)
        await server.stop()
        XCTAssertEqual(
            try String(contentsOf: responseFile, encoding: .utf8),
            "round_started"
        )
        let reward = accumulator.drain(aggregation: .sum)
        XCTAssertEqual(reward.reward, 3.5, accuracy: 0.000_1)
        XCTAssertEqual(reward.messageCount, 1)
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

    func testLegacyRLTransportAndExplorationSettingsRemainMigrationDecodable() throws {
        var legacy = compactConfiguration()
        legacy.actionBehavior = nil
        legacy.webSocketURL = "ws://127.0.0.1:49001"
        legacy.connectionTimeoutSeconds = 7
        legacy.rewardTimeoutSeconds = 4
        legacy.pauseActionsWhenRewardStale = true
        legacy.reconnectInitialDelaySeconds = 0.5
        legacy.reconnectMaximumDelaySeconds = 8
        legacy.continuousActionStandardDeviation = 0.18
        legacy.binaryActionTemperature = 1.25
        let decoded = try JSONDecoder().decode(
            ReinforcementLearningConfiguration.self,
            from: JSONEncoder().encode(legacy)
        )
        XCTAssertEqual(decoded.webSocketURL, legacy.webSocketURL)
        XCTAssertEqual(decoded.connectionTimeoutSeconds, 7)
        XCTAssertEqual(decoded.effectiveActionBehavior.binaryActionTemperature, 1.25)
        XCTAssertLessThanOrEqual(
            decoded.effectiveActionBehavior.absoluteMouseStandardDeviation,
            0.05
        )
        XCTAssertNoThrow(try decoded.validated())
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
                behavior: configuration.effectiveActionBehavior,
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

    func testPPOFinalSingleTransitionBatchIsFiniteInsteadOfDiscarded() throws {
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
            currentPacked: Data(
                repeating: 127,
                count: profile.preprocessing.sampleByteCount
            ),
            pastPacked: Data(),
            pastControls: [],
            contextValues: []
        )
        let prediction = try optimizer.prediction(for: observation)
        var random = ReinforcementRandomGenerator(seed: 91)
        let sample = try plan.sample(
            logits: prediction.logits,
            behavior: configuration.effectiveActionBehavior,
            random: &random
        )
        let update = try optimizer.update(
            transitions: [ReinforcementTransition(
                observation: observation,
                distributionActions: sample.distributionActions,
                executedActions: sample.executedActions,
                oldLogProbability: sample.logProbability,
                oldValue: prediction.value,
                reward: 1,
                terminal: true
            )],
            bootstrapValue: 0,
            shuffleSeed: 92
        )
        XCTAssertGreaterThan(update.optimizerSteps, 0)
        XCTAssertTrue(update.policyLoss.isFinite)
        XCTAssertTrue(update.valueLoss.isFinite)
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
        valid.reinforcementActionBehavior = ReinforcementActionBehavior()
        XCTAssertTrue(valid.reinforcementMetadataIsValid)
        valid.reinforcementActionBehavior?.maximumAbsoluteMouseStep = 2
        XCTAssertFalse(valid.reinforcementMetadataIsValid)
        valid.reinforcementActionBehavior = ReinforcementActionBehavior()
        valid.reinforcementLearningSummary = nil
        XCTAssertFalse(valid.reinforcementMetadataIsValid)
        valid.trainingOrigin = nil
        valid.reinforcementActionSpace = nil
        valid.reinforcementActionBehavior = nil
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

    private func requirePythonWebSockets() throws {
        let probe = Process()
        probe.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        probe.arguments = ["python3", "-c", "import websockets"]
        probe.standardOutput = Pipe()
        probe.standardError = Pipe()
        try probe.run()
        probe.waitUntilExit()
        guard probe.terminationStatus == 0 else {
            throw XCTSkip(
                "Python websockets is unavailable for the local integration test."
            )
        }
    }

    private func unusedLoopbackPort() throws -> Int {
        let probe = Process()
        let output = Pipe()
        probe.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        probe.arguments = [
            "python3", "-c",
            "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()"
        ]
        probe.standardOutput = output
        probe.standardError = Pipe()
        try probe.run()
        probe.waitUntilExit()
        guard probe.terminationStatus == 0,
              let port = Int(
                  String(
                      decoding: output.fileHandleForReading.readDataToEndOfFile(),
                      as: UTF8.self
                  ).trimmingCharacters(in: .whitespacesAndNewlines)
              ) else {
            throw CocoaError(.fileReadUnknown)
        }
        return port
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
