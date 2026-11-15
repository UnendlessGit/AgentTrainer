import ATCore
import Foundation

/// Thread-safe snapshot of what's currently pressed / moving, for the HUD (human or model input).
public final class LiveInputState: @unchecked Sendable {
    public struct Snapshot: Equatable, Sendable {
        public var keys: Set<String> = []
        public var buttons: Set<Int> = []
        /// Recent pointer velocity (points per second), decayed.
        public var velocity: CGVector = .zero
        /// Scroll direction recently used (y > 0 up), decays to 0.
        public var scroll: CGVector = .zero
        public init(keys: Set<String> = [], buttons: Set<Int> = [], velocity: CGVector = .zero, scroll: CGVector = .zero) {
            self.keys = keys; self.buttons = buttons; self.velocity = velocity; self.scroll = scroll
        }
    }

    private let lock = NSLock()
    private var snap = Snapshot()
    private var heldCodes = Set<UInt16>()
    private var lastMove = HostClock.now()

    public init() {}

    public var snapshot: Snapshot {
        lock.lock(); defer { lock.unlock() }
        // Decay movement / scroll so the HUD settles when input stops.
        let age = Double(HostClock.now() - lastMove) / 1e9
        var s = snap
        if age > 0.12 { s.velocity = .zero }
        if age > 0.25 { s.scroll = .zero }
        return s
    }

    public func ingest(_ e: InputEventRecord) {
        lock.lock(); defer { lock.unlock() }
        switch e.kind {
        case .keyDown: heldCodes.insert(e.code)
        case .keyUp: heldCodes.remove(e.code)
        case .flagsChanged: if e.flags & 2 != 0 { heldCodes.insert(e.code) } else { heldCodes.remove(e.code) }
        case .buttonDown: snap.buttons.insert(Int(e.code))
        case .buttonUp: snap.buttons.remove(Int(e.code))
        case .mouseMove:
            let now = HostClock.now()
            let dt = max(Double(now - lastMove) / 1e9, 1.0 / 240)
            let v = CGVector(dx: Double(e.dx) / dt, dy: Double(e.dy) / dt)
            snap.velocity = CGVector(dx: snap.velocity.dx * 0.6 + v.dx * 0.4, dy: snap.velocity.dy * 0.6 + v.dy * 0.4)
            lastMove = now
        case .scroll:
            snap.scroll = CGVector(dx: Double(e.dx), dy: Double(e.dy))
            lastMove = HostClock.now()
        }
        snap.keys = Set(heldCodes.compactMap { KeyCatalog.byKeyCode[$0]?.id })
    }

    /// Sets the state directly (model output in Run mode).
    public func set(keys: Set<String>, buttons: Set<Int>, velocity: CGVector, scroll: CGVector) {
        lock.lock(); defer { lock.unlock() }
        snap = Snapshot(keys: keys, buttons: buttons, velocity: velocity, scroll: scroll)
        lastMove = HostClock.now()
    }

    public func clear() {
        lock.lock(); defer { lock.unlock() }
        snap = Snapshot(); heldCodes = []
    }
}
