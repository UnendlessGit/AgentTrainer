import Foundation
import Network
import QuartzCore

struct ReinforcementRewardDrain: Equatable, Sendable {
    var reward: Double
    var messageCount: Int
    var episodeFinished: Bool
    var lastMessageTime: Double?
}

/// Lock-backed aggregation keeps the 30–60 Hz WebSocket receive path bounded:
/// no per-message task, array, or unbounded AsyncStream is created.
final class ReinforcementRewardAccumulator: @unchecked Sendable {
    private let lock = NSLock()
    private var sum = 0.0
    private var compensation = 0.0
    private var latest = 0.0
    private var count = 0
    private var episodeFinished = false
    private var lastMessageTime: Double?
    private var receivedTotal = 0
    private var invalidTotal = 0

    func ingest(
        _ data: Data,
        names: RewardWebSocketMessageNames = RewardWebSocketMessageNames(),
        receivedAt: Double = CACurrentMediaTime()
    ) -> Bool {
        guard data.count <= RewardWebSocketServerSettings.maximumMessageBytes,
              let signal = RewardWebSocketProtocol.decode(data, names: names) else {
            lock.withLock { invalidTotal += 1 }
            return false
        }
        lock.withLock {
            if let reward = signal.reward {
                // Kahan accumulation prevents long high-frequency episodes from
                // losing small rewards beside a much larger running return.
                let adjusted = reward - compensation
                let next = sum + adjusted
                if next.isFinite {
                    compensation = (next - sum) - adjusted
                    sum = next
                } else {
                    compensation = 0
                    sum = reward.sign == .minus
                        ? -Double.greatestFiniteMagnitude
                        : Double.greatestFiniteMagnitude
                }
                latest = reward
                count += 1
            }
            episodeFinished = episodeFinished || signal.episodeFinished
            lastMessageTime = receivedAt
            receivedTotal += 1
        }
        return true
    }

    func drain(aggregation: ReinforcementRewardAggregation) -> ReinforcementRewardDrain {
        lock.withLock {
            let reward: Double = switch aggregation {
            case .sum: sum
            case .mean: count > 0 ? sum / Double(count) : 0
            case .latest: count > 0 ? latest : 0
            }
            let result = ReinforcementRewardDrain(
                reward: reward,
                messageCount: count,
                episodeFinished: episodeFinished,
                lastMessageTime: lastMessageTime
            )
            sum = 0
            compensation = 0
            latest = 0
            count = 0
            episodeFinished = false
            return result
        }
    }

    func resetEpisode(clearLastMessageTime: Bool = false) {
        lock.withLock {
            sum = 0
            compensation = 0
            latest = 0
            count = 0
            episodeFinished = false
            if clearLastMessageTime { lastMessageTime = nil }
        }
    }

    func statistics() -> (received: Int, invalid: Int, lastMessageTime: Double?) {
        lock.withLock { (receivedTotal, invalidTotal, lastMessageTime) }
    }
}

enum RewardWebSocketProtocol {
    enum OutgoingEvent: Equatable, Sendable {
        case episodeFinished
        case episodeStarting
        case episodeStarted
        case trainingUpdateStarted
        case trainingUpdateFinished
    }

    struct Signal: Equatable, Sendable {
        var reward: Double?
        var episodeFinished: Bool
        var heartbeat = false
    }

    static func decode(
        _ data: Data,
        names: RewardWebSocketMessageNames = RewardWebSocketMessageNames()
    ) -> Signal? {
        guard let object = try? JSONSerialization.jsonObject(with: data),
              let dictionary = object as? [String: Any] else { return nil }
        let reward: Double?
        if let number = dictionary[names.rewardField] as? NSNumber {
            guard CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
            let value = number.doubleValue
            guard value.isFinite else { return nil }
            reward = value
        } else if dictionary[names.rewardField] != nil {
            return nil
        } else {
            reward = nil
        }
        if dictionary[names.typeField] != nil,
           dictionary[names.typeField] as? String == nil { return nil }
        if dictionary[names.doneField] != nil,
           dictionary[names.doneField] as? Bool == nil { return nil }
        let type = dictionary[names.typeField] as? String
        var done = (dictionary[names.doneField] as? Bool) == true
            || type == names.incomingEpisodeFinished
        // Preserve the permissive 2.4.0 protocol only while every relevant name
        // is still at its default. Custom naming switches to exact matching.
        if names == RewardWebSocketMessageNames() {
            done = done
                || (dictionary["episode_finished"] as? Bool) == true
                || (dictionary["terminated"] as? Bool) == true
                || type?.lowercased() == "episode_finished"
                || type?.lowercased() == "episode_end"
        }
        let heartbeat = type == names.incomingHeartbeat
        guard reward != nil || done || heartbeat else { return nil }
        return Signal(
            reward: reward,
            episodeFinished: done,
            heartbeat: heartbeat
        )
    }

    static func episodeMessage(
        event: OutgoingEvent,
        names: RewardWebSocketMessageNames = RewardWebSocketMessageNames(),
        episode: Int,
        reason: String? = nil,
        episodeReturn: Double? = nil,
        steps: Int? = nil,
        nextEpisode: Int? = nil,
        timestamp: Date = Date()
    ) throws -> Data {
        let type = switch event {
        case .episodeFinished: names.outgoingEpisodeFinished
        case .episodeStarting: names.outgoingEpisodeStarting
        case .episodeStarted: names.outgoingEpisodeStarted
        case .trainingUpdateStarted: names.outgoingTrainingUpdateStarted
        case .trainingUpdateFinished: names.outgoingTrainingUpdateFinished
        }
        var object: [String: Any] = [
            names.typeField: type,
            "episode": max(0, episode),
            "timestamp": ISO8601DateFormatter().string(from: timestamp)
        ]
        if let reason { object["reason"] = reason }
        if let episodeReturn, episodeReturn.isFinite { object["return"] = episodeReturn }
        if let steps { object["steps"] = max(0, steps) }
        if let nextEpisode { object["next_episode"] = max(0, nextEpisode) }
        return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    /// Source compatibility for helpers written against the 2.4.0 protocol.
    static func episodeMessage(
        type: String,
        episode: Int,
        reason: String? = nil,
        episodeReturn: Double? = nil,
        steps: Int? = nil,
        nextEpisode: Int? = nil,
        timestamp: Date = Date()
    ) throws -> Data {
        let event: OutgoingEvent = switch type {
        case "episode_starting": .episodeStarting
        case "episode_started": .episodeStarted
        case "training_update_started": .trainingUpdateStarted
        case "training_update_finished": .trainingUpdateFinished
        default: .episodeFinished
        }
        return try episodeMessage(
            event: event,
            episode: episode,
            reason: reason,
            episodeReturn: episodeReturn,
            steps: steps,
            nextEpisode: nextEpisode,
            timestamp: timestamp
        )
    }
}

enum RewardWebSocketError: LocalizedError {
    case listenerTimedOut
    case rewarderConnectionTimedOut
    case listenerFailed(String)
    case notConnected
    case unsupportedMessage

    var errorDescription: String? {
        switch self {
        case .listenerTimedOut:
            "AgentTrainer could not start the reward WebSocket server before the timeout. Check that its port is available."
        case .rewarderConnectionTimedOut:
            "AgentTrainer is hosting the reward WebSocket, but no rewarder connected before the startup timeout. Start the rewarder and try the RL shortcut again."
        case .listenerFailed(let detail):
            "The reward WebSocket server could not listen at the configured URL: \(detail)"
        case .notConnected: "No rewarder is connected to AgentTrainer's WebSocket server."
        case .unsupportedMessage:
            "The rewarder sent an unsupported WebSocket frame. Use complete UTF-8 text or binary JSON messages."
        }
    }
}

/// Single-client WebSocket listener owned by one RL session. AgentTrainer hosts
/// the endpoint and keeps listening after a rewarder disconnects so a client may
/// reconnect during the configured safety grace period. Normal Run never
/// constructs this type and therefore has no network dependency.
actor RewardWebSocketServer {
    typealias StateHandler = @Sendable (ReinforcementConnectionState, String?) -> Void

    private let accumulator: ReinforcementRewardAccumulator
    private let onState: StateHandler
    private let queue = DispatchQueue(
        label: "AgentTrainer.RewardWebSocketServer",
        qos: .userInitiated
    )
    private var listener: NWListener?
    private var connection: NWConnection?
    private var running = false
    private var generation: UInt64 = 0
    private var listenerReady = false
    private var clientConnected = false
    private var listenerFailure: String?
    private var settings = RewardWebSocketServerSettings()
    private var pendingOutbound: [Data] = []
    private var state = ReinforcementConnectionState.disconnected

    init(
        accumulator: ReinforcementRewardAccumulator,
        onState: @escaping StateHandler
    ) {
        self.accumulator = accumulator
        self.onState = onState
    }

    func start(settings requestedSettings: RewardWebSocketServerSettings) async throws {
        stop()
        let settings = try requestedSettings.validated()
        guard let portValue = settings.validatedURL?.port,
              let port = NWEndpoint.Port(rawValue: UInt16(portValue)) else {
            throw AgentTrainerError.invalidConfiguration(
                "The reward WebSocket server URL has no usable port."
            )
        }
        running = true
        generation &+= 1
        let launchGeneration = generation
        self.settings = settings
        listenerReady = false
        clientConnected = false
        listenerFailure = nil

        let webSocket = NWProtocolWebSocket.Options()
        webSocket.autoReplyPing = true
        webSocket.maximumMessageSize = Self.maximumMessageBytes
        webSocket.setClientRequestHandler(queue) { _, _ in
            NWProtocolWebSocket.Response(status: .accept, subprotocol: nil)
        }
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        parameters.acceptLocalOnly = !settings.allowRemoteConnections
        parameters.includePeerToPeer = settings.allowRemoteConnections
        parameters.defaultProtocolStack.applicationProtocols.insert(
            webSocket,
            at: 0
        )
        do {
            let listener = try NWListener(using: parameters, on: port)
            listener.stateUpdateHandler = { [weak self] state in
                Task { await self?.handleListenerState(
                    state,
                    generation: launchGeneration
                ) }
            }
            listener.newConnectionHandler = { [weak self] connection in
                Task { await self?.accept(
                    connection,
                    generation: launchGeneration
                ) }
            }
            self.listener = listener
            listener.start(queue: queue)
            try await waitForListener(
                timeout: min(10, settings.connectionWaitTimeoutSeconds)
            )
        } catch {
            stop()
            throw error
        }
    }

    func waitForRewarder(timeout: Double) async throws {
        let deadline = ContinuousClock.now + .seconds(timeout)
        while running, !clientConnected {
            if let listenerFailure {
                throw RewardWebSocketError.listenerFailed(listenerFailure)
            }
            guard ContinuousClock.now < deadline else {
                throw RewardWebSocketError.rewarderConnectionTimedOut
            }
            try Task.checkCancellation()
            try await Task.sleep(for: .milliseconds(20))
        }
        guard running, clientConnected else { throw CancellationError() }
    }

    func send(_ data: Data) async {
        guard data.count <= Self.maximumMessageBytes else { return }
        guard state == .connected, let connection else {
            enqueue(data)
            return
        }
        let expectedGeneration = generation
        let metadata = NWProtocolWebSocket.Metadata(opcode: .text)
        let context = NWConnection.ContentContext(
            identifier: "AgentTrainer.RewardMessage.Awaited",
            metadata: [metadata]
        )
        let error: NWError? = await withCheckedContinuation { continuation in
            connection.send(
                content: data,
                contentContext: context,
                isComplete: true,
                completion: .contentProcessed { continuation.resume(returning: $0) }
            )
        }
        guard let error, running, generation == expectedGeneration,
              self.connection === connection else { return }
        enqueue(data)
        disconnect(connection, detail: error.localizedDescription)
    }

    func stop() {
        running = false
        generation &+= 1
        connection?.stateUpdateHandler = nil
        connection?.cancel()
        listener?.stateUpdateHandler = nil
        listener?.newConnectionHandler = nil
        listener?.cancel()
        connection = nil
        listener = nil
        listenerReady = false
        clientConnected = false
        listenerFailure = nil
        pendingOutbound.removeAll(keepingCapacity: false)
        transition(to: .disconnected, detail: nil)
    }

    private func waitForListener(timeout: Double) async throws {
        let deadline = ContinuousClock.now + .seconds(timeout)
        while running, !listenerReady {
            if let listenerFailure {
                throw RewardWebSocketError.listenerFailed(listenerFailure)
            }
            guard ContinuousClock.now < deadline else {
                throw RewardWebSocketError.listenerTimedOut
            }
            try Task.checkCancellation()
            try await Task.sleep(for: .milliseconds(20))
        }
        guard running, listenerReady else { throw CancellationError() }
    }

    private func handleListenerState(
        _ newState: NWListener.State,
        generation expectedGeneration: UInt64
    ) {
        guard running, generation == expectedGeneration else { return }
        switch newState {
        case .ready:
            listenerReady = true
            listenerFailure = nil
            transition(to: .listening, detail: "Waiting for the rewarder to connect")
        case .waiting(let error):
            transition(to: .listening, detail: error.localizedDescription)
        case .failed(let error):
            listenerFailure = error.localizedDescription
            transition(to: .disconnected, detail: error.localizedDescription)
        case .cancelled:
            if running {
                listenerFailure = "The listener was cancelled unexpectedly."
                transition(to: .disconnected, detail: listenerFailure)
            }
        case .setup:
            break
        @unknown default:
            listenerFailure = "The listener entered an unknown state."
        }
    }

    private func accept(
        _ candidate: NWConnection,
        generation expectedGeneration: UInt64
    ) {
        guard running, generation == expectedGeneration,
              connection == nil else {
            candidate.cancel()
            return
        }
        connection = candidate
        candidate.stateUpdateHandler = { [weak self, weak candidate] state in
            guard let candidate else { return }
            Task { await self?.handleConnectionState(
                state,
                connection: candidate,
                generation: expectedGeneration
            ) }
        }
        candidate.start(queue: queue)
    }

    private func handleConnectionState(
        _ newState: NWConnection.State,
        connection candidate: NWConnection,
        generation expectedGeneration: UInt64
    ) {
        guard running, generation == expectedGeneration,
              connection === candidate else { return }
        switch newState {
        case .ready:
            clientConnected = true
            transition(to: .connected, detail: nil)
            receiveNext(over: candidate, generation: expectedGeneration)
            let queued = pendingOutbound
            pendingOutbound.removeAll(keepingCapacity: true)
            for data in queued {
                send(data, over: candidate, generation: expectedGeneration)
            }
        case .failed(let error):
            disconnect(candidate, detail: error.localizedDescription)
        case .cancelled:
            disconnect(candidate, detail: "The rewarder disconnected")
        case .waiting(let error):
            if clientConnected {
                disconnect(candidate, detail: error.localizedDescription)
            }
        case .setup, .preparing:
            break
        @unknown default:
            disconnect(candidate, detail: "The rewarder connection entered an unknown state")
        }
    }

    private func receiveNext(
        over candidate: NWConnection,
        generation expectedGeneration: UInt64
    ) {
        candidate.receiveMessage { [weak self, weak candidate] data, context, _, error in
            guard let self, let candidate else { return }
            Task { await self.receive(
                data,
                context: context,
                error: error,
                over: candidate,
                generation: expectedGeneration
            ) }
        }
    }

    private func receive(
        _ data: Data?,
        context: NWConnection.ContentContext?,
        error: NWError?,
        over candidate: NWConnection,
        generation expectedGeneration: UInt64
    ) {
        guard running, generation == expectedGeneration,
              connection === candidate else { return }
        if let error {
            disconnect(candidate, detail: error.localizedDescription)
            return
        }
        let metadata = context?.protocolMetadata(
            definition: NWProtocolWebSocket.definition
        ) as? NWProtocolWebSocket.Metadata
        switch metadata?.opcode {
        case .text, .binary:
            guard let data, data.count <= Self.maximumMessageBytes else {
                disconnect(candidate, detail: "A reward message exceeded the 64 KiB limit")
                return
            }
            _ = accumulator.ingest(data, names: settings.messageNames)
        case .close:
            disconnect(candidate, detail: "The rewarder closed the WebSocket")
            return
        case .ping, .pong:
            break
        default:
            disconnect(candidate, detail: RewardWebSocketError.unsupportedMessage.localizedDescription)
            return
        }
        receiveNext(over: candidate, generation: expectedGeneration)
    }

    private func send(
        _ data: Data,
        over candidate: NWConnection,
        generation expectedGeneration: UInt64
    ) {
        let metadata = NWProtocolWebSocket.Metadata(opcode: .text)
        let context = NWConnection.ContentContext(
            identifier: "AgentTrainer.RewardMessage",
            metadata: [metadata]
        )
        candidate.send(
            content: data,
            contentContext: context,
            isComplete: true,
            completion: .contentProcessed { [weak self, weak candidate] error in
                guard let self, let candidate, let error else { return }
                Task { await self.sendFailed(
                    data,
                    error: error,
                    over: candidate,
                    generation: expectedGeneration
                ) }
            }
        )
    }

    private func sendFailed(
        _ data: Data,
        error: NWError,
        over candidate: NWConnection,
        generation expectedGeneration: UInt64
    ) {
        guard running, generation == expectedGeneration,
              connection === candidate else { return }
        enqueue(data)
        disconnect(candidate, detail: error.localizedDescription)
    }

    private func disconnect(_ candidate: NWConnection, detail: String) {
        guard connection === candidate else { return }
        candidate.stateUpdateHandler = nil
        candidate.cancel()
        connection = nil
        clientConnected = false
        transition(to: .listening, detail: detail)
    }

    private func enqueue(_ data: Data) {
        pendingOutbound.append(data)
        if pendingOutbound.count > 32 {
            pendingOutbound.removeFirst(pendingOutbound.count - 32)
        }
    }

    private func transition(
        to newState: ReinforcementConnectionState,
        detail: String?
    ) {
        guard state != newState || detail != nil else { return }
        state = newState
        onState(newState, detail)
    }

    private static let maximumMessageBytes = RewardWebSocketServerSettings.maximumMessageBytes

    deinit {
        connection?.cancel()
        listener?.cancel()
    }
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}
