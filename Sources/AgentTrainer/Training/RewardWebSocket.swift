import Foundation
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

    func ingest(_ data: Data, receivedAt: Double = CACurrentMediaTime()) -> Bool {
        guard data.count <= RewardWebSocketProtocol.maximumMessageBytes,
              let signal = RewardWebSocketProtocol.decode(data) else {
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
    static let maximumMessageBytes = 64 * 1_024

    struct Signal: Equatable, Sendable {
        var reward: Double?
        var episodeFinished: Bool
    }

    static func decode(_ data: Data) -> Signal? {
        guard let object = try? JSONSerialization.jsonObject(with: data),
              let dictionary = object as? [String: Any] else { return nil }
        let reward: Double?
        if let number = dictionary["reward"] as? NSNumber {
            guard CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
            let value = number.doubleValue
            guard value.isFinite else { return nil }
            reward = value
        } else if dictionary["reward"] != nil {
            return nil
        } else {
            reward = nil
        }
        let type = (dictionary["type"] as? String)?.lowercased()
        let done = (dictionary["done"] as? Bool) == true
            || (dictionary["episode_finished"] as? Bool) == true
            || (dictionary["terminated"] as? Bool) == true
            || type == "episode_finished" || type == "episode_end"
        guard reward != nil || done else { return nil }
        return Signal(reward: reward, episodeFinished: done)
    }

    static func episodeMessage(
        type: String,
        episode: Int,
        reason: String? = nil,
        episodeReturn: Double? = nil,
        steps: Int? = nil,
        nextEpisode: Int? = nil,
        timestamp: Date = Date()
    ) throws -> Data {
        var object: [String: Any] = [
            "type": type,
            "episode": max(0, episode),
            "timestamp": ISO8601DateFormatter().string(from: timestamp)
        ]
        if let reason { object["reason"] = reason }
        if let episodeReturn, episodeReturn.isFinite { object["return"] = episodeReturn }
        if let steps { object["steps"] = max(0, steps) }
        if let nextEpisode { object["next_episode"] = max(0, nextEpisode) }
        return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }
}

enum RewardWebSocketError: LocalizedError {
    case connectionTimedOut
    case notConnected
    case unsupportedMessage

    var errorDescription: String? {
        switch self {
        case .connectionTimedOut: "The RL reward WebSocket did not complete its handshake before the connection timeout."
        case .notConnected: "The RL reward WebSocket is not connected."
        case .unsupportedMessage: "The RL reward WebSocket sent a message that was not UTF-8 text or binary data."
        }
    }
}

/// One connection owner with bounded outbound lifecycle buffering and automatic
/// reconnect after an established session drops. Normal Run never constructs
/// this type and therefore has no network dependency.
actor RewardWebSocketClient {
    typealias StateHandler = @Sendable (ReinforcementConnectionState, String?) -> Void

    private let accumulator: ReinforcementRewardAccumulator
    private let onState: StateHandler
    private var session: URLSession?
    private var socket: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?
    private var reconnectTask: Task<Void, Never>?
    private var running = false
    private var generation: UInt64 = 0
    private var url: URL?
    private var connectionTimeout = 5.0
    private var reconnectInitialDelay = 0.25
    private var reconnectMaximumDelay = 5.0
    private var reconnectDelay = 0.25
    private var pendingOutbound: [Data] = []
    private var state = ReinforcementConnectionState.disconnected

    init(
        accumulator: ReinforcementRewardAccumulator,
        onState: @escaping StateHandler
    ) {
        self.accumulator = accumulator
        self.onState = onState
    }

    func start(
        url: URL,
        connectionTimeout: Double,
        reconnectInitialDelay: Double,
        reconnectMaximumDelay: Double
    ) async throws {
        await stop()
        running = true
        generation &+= 1
        self.url = url
        self.connectionTimeout = connectionTimeout
        self.reconnectInitialDelay = reconnectInitialDelay
        self.reconnectMaximumDelay = reconnectMaximumDelay
        reconnectDelay = reconnectInitialDelay
        do {
            try await connect(generation: generation, reconnecting: false)
        } catch {
            running = false
            transition(to: .disconnected, detail: error.localizedDescription)
            throw error
        }
    }

    func send(_ data: Data) async {
        guard data.count <= RewardWebSocketProtocol.maximumMessageBytes else { return }
        guard state == .connected, let socket else {
            enqueue(data)
            return
        }
        do {
            try await socket.send(webSocketMessage(for: data))
        } catch {
            enqueue(data)
            connectionFailed(error, generation: generation)
        }
    }

    func stop() async {
        running = false
        generation &+= 1
        receiveTask?.cancel()
        reconnectTask?.cancel()
        receiveTask = nil
        reconnectTask = nil
        socket?.cancel(with: .goingAway, reason: nil)
        session?.invalidateAndCancel()
        socket = nil
        session = nil
        pendingOutbound.removeAll(keepingCapacity: false)
        transition(to: .disconnected, detail: nil)
    }

    private func connect(generation expectedGeneration: UInt64, reconnecting: Bool) async throws {
        guard running, generation == expectedGeneration, let url else {
            throw CancellationError()
        }
        transition(to: reconnecting ? .reconnecting : .connecting, detail: nil)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.waitsForConnectivity = false
        configuration.timeoutIntervalForRequest = connectionTimeout
        let session = URLSession(configuration: configuration)
        let socket = session.webSocketTask(with: url)
        socket.maximumMessageSize = RewardWebSocketProtocol.maximumMessageBytes
        self.session = session
        self.socket = socket
        socket.resume()
        // URLSession completes WebSocket control frames while a receive is in
        // flight. Start the bounded receive loop before the handshake ping so
        // a quiet reward server cannot deadlock startup waiting for a pong.
        receiveTask = Task { [weak self, weak socket] in
            guard let self, let socket else { return }
            await self.receiveLoop(
                socket: socket,
                generation: expectedGeneration
            )
        }
        try await ping(socket, timeout: connectionTimeout)
        guard running, generation == expectedGeneration, self.socket === socket else {
            socket.cancel(with: .goingAway, reason: nil)
            session.invalidateAndCancel()
            throw CancellationError()
        }
        reconnectDelay = reconnectInitialDelay
        transition(to: .connected, detail: nil)
        let queued = pendingOutbound
        pendingOutbound.removeAll(keepingCapacity: true)
        for data in queued {
            do { try await socket.send(webSocketMessage(for: data)) }
            catch {
                enqueue(data)
                connectionFailed(error, generation: expectedGeneration)
                return
            }
        }
    }

    private func receiveLoop(
        socket expectedSocket: URLSessionWebSocketTask,
        generation expectedGeneration: UInt64
    ) async {
        do {
            while running, generation == expectedGeneration,
                  socket === expectedSocket, !Task.isCancelled {
                let message = try await expectedSocket.receive()
                let data: Data = switch message {
                case .data(let data): data
                case .string(let text): Data(text.utf8)
                @unknown default: throw RewardWebSocketError.unsupportedMessage
                }
                _ = accumulator.ingest(data)
            }
        } catch is CancellationError {
            return
        } catch {
            connectionFailed(error, generation: expectedGeneration)
        }
    }

    private func connectionFailed(_ error: Error, generation expectedGeneration: UInt64) {
        guard running, generation == expectedGeneration else { return }
        socket?.cancel(with: .abnormalClosure, reason: nil)
        session?.invalidateAndCancel()
        socket = nil
        session = nil
        receiveTask?.cancel()
        receiveTask = nil
        transition(to: .reconnecting, detail: error.localizedDescription)
        guard reconnectTask == nil else { return }
        let delay = reconnectDelay
        reconnectDelay = min(reconnectMaximumDelay, max(reconnectInitialDelay, delay * 2))
        reconnectTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(delay))
            guard !Task.isCancelled, let self else { return }
            await self.retryConnection(generation: expectedGeneration)
        }
    }

    private func retryConnection(generation expectedGeneration: UInt64) async {
        reconnectTask = nil
        guard running, generation == expectedGeneration else { return }
        do {
            try await connect(generation: expectedGeneration, reconnecting: true)
        } catch is CancellationError {
            return
        } catch {
            connectionFailed(error, generation: expectedGeneration)
        }
    }

    private func enqueue(_ data: Data) {
        pendingOutbound.append(data)
        if pendingOutbound.count > 32 {
            pendingOutbound.removeFirst(pendingOutbound.count - 32)
        }
    }

    private func webSocketMessage(
        for data: Data
    ) -> URLSessionWebSocketTask.Message {
        String(data: data, encoding: .utf8).map {
            .string($0)
        } ?? .data(data)
    }

    private func transition(
        to newState: ReinforcementConnectionState,
        detail: String?
    ) {
        guard state != newState || detail != nil else { return }
        state = newState
        onState(newState, detail)
    }

    private func ping(
        _ socket: URLSessionWebSocketTask,
        timeout: Double
    ) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                try await withCheckedThrowingContinuation { continuation in
                    socket.sendPing { error in
                        if let error { continuation.resume(throwing: error) }
                        else { continuation.resume() }
                    }
                }
            }
            group.addTask {
                try await Task.sleep(for: .seconds(timeout))
                socket.cancel(with: .goingAway, reason: nil)
                throw RewardWebSocketError.connectionTimedOut
            }
            guard let result = try await group.next() else {
                throw RewardWebSocketError.connectionTimedOut
            }
            group.cancelAll()
            return result
        }
    }
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}
