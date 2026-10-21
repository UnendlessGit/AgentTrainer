import CoreGraphics
import Foundation

/// Policy-v7 input vocabulary. Absolute host times never enter a tensor.
/// Every duration has a fast linear component, a logarithmic component, and
/// an independent validity bit. Zero elapsed time is therefore not padding.
enum TemporalInputContract {
    static let version = 1
    static let embeddingWidth = 64
    static let clockCount = 18
    static let controlWidth = 16
    static let motionWidth = 14
    static let binaryIndices = ActionLayout.binary
    static let motionOffset = clockCount + binaryIndices.count * controlWidth
    static let valueCount = motionOffset + 2 * motionWidth
    static let rowBytes = valueCount * MemoryLayout<Float>.size
    static let controlLookup = Dictionary(uniqueKeysWithValues: binaryIndices.enumerated().map { ($0.element, $0.offset) })

    static func seconds(_ end: UInt64, since start: UInt64) -> Double {
        end >= start ? Double(end - start) / 1_000_000_000 : 0
    }

    static func putTime(_ seconds: Double?, in values: inout [Float], at offset: Int) {
        guard let seconds, seconds.isFinite, seconds >= 0 else { return }
        values[offset] = Float(min(1, seconds / 0.25))
        values[offset + 1] = Float(log1p(seconds) / log(61))
        values[offset + 2] = 1
    }

    static func putClock(
        in values: inout [Float], interval: Double?, perceptionInterval: Double?,
        frameAge: Double?, historyAge: Double? = 0, separation: Double? = nil
    ) {
        values[0] = 1
        putTime(interval, in: &values, at: 1)
        putTime(perceptionInterval, in: &values, at: 4)
        putTime(frameAge, in: &values, at: 7)
        putTime(historyAge, in: &values, at: 10)
        putTime(separation, in: &values, at: 13)
    }

    /// The same disable-only input policy is applied to offline and posted
    /// controls. A blocked control's duration must not become a hidden input.
    static func sanitize(
        _ values: UnsafeMutableBufferPointer<Float>, rows: Int,
        channels: ActionChannels, restrictions: ActionRestrictions
    ) {
        guard values.count >= rows * valueCount else { return }
        let allowed = Set(ActionLayout.learnableBinaryIndices(channels: channels, restrictions: restrictions))
        for row in 0..<rows {
            let base = row * valueCount
            for (slot, index) in binaryIndices.enumerated() where !allowed.contains(index) {
                let start = base + clockCount + slot * controlWidth
                for i in start..<(start + controlWidth) { values[i] = 0 }
            }
            if !channels.mouseMovement {
                values[base + 16] = 0; values[base + 17] = 0
                for i in 0..<motionWidth { values[base + motionOffset + i] = 0 }
            }
            if !channels.scroll {
                for i in 0..<motionWidth { values[base + motionOffset + motionWidth + i] = 0 }
            }
        }
    }

    /// A matched ablation retains clocks, held state, and displacement while
    /// removing information about action timing. Used only by opt-in tests.
    static func ablate(_ values: UnsafeMutableBufferPointer<Float>, rows: Int) {
        for row in 0..<rows {
            let base = row * valueCount
            for slot in binaryIndices.indices {
                let start = base + clockCount + slot * controlWidth
                for i in 1..<controlWidth { values[start + i] = 0 }
            }
            for group in 0..<2 {
                let start = base + motionOffset + group * motionWidth
                for i in 2..<motionWidth { values[start + i] = 0 }
            }
        }
    }
}

/// A streaming, bounded reducer of the controls known before one decision.
/// It integrates holds at event boundaries rather than counting nominal frames.
/// Recording and injection use this exact implementation and clock vocabulary.
struct CausalControlTimeline: Sendable {
    struct Control: Sendable {
        var held = false
        var pressedAt: UInt64?
        var releasedAt: UInt64?
        var completedDuration: Double?
        var onsetKnown = false
        var heldNanos: UInt64 = 0
        var presses = 0
        var releases = 0
    }

    struct Motion: Sendable {
        var x = 0.0, y = 0.0
        var first: UInt64?
        var last: UInt64?
        var count = 0

        mutating func consume(x: Double, y: Double, at timestamp: UInt64) {
            guard x != 0 || y != 0 else { return }
            // Event readers reject non-finite samples, but a malicious finite
            // magnitude must not overflow an aggregate or its rate encoding.
            self.x = min(1e12, max(-1e12, self.x + x))
            self.y = min(1e12, max(-1e12, self.y + y))
            if first == nil { first = timestamp }
            last = timestamp; count += 1
        }

        mutating func endInterval() {
            x = 0; y = 0; first = nil; count = 0
            // Keep last-event recency across idle windows.
        }
    }

    let origin: UInt64
    private let censorInitialOnsets: Bool
    private(set) var intervalStart: UInt64
    private var accountedAt: UInt64
    private(set) var controls = [Control](repeating: Control(), count: TemporalInputContract.binaryIndices.count)
    private var movement = Motion()
    private var scrolling = Motion()
    private var pointer = CGPoint.zero
    private var hasPointer = false

    init(origin: UInt64, censorInitialOnsets: Bool = true) {
        self.origin = origin; intervalStart = origin; accountedAt = origin
        self.censorInitialOnsets = censorInitialOnsets
    }

    private mutating func advance(to timestamp: UInt64) {
        guard timestamp >= accountedAt else { return }
        let elapsed = timestamp - accountedAt
        if elapsed > 0 {
            for slot in controls.indices where controls[slot].held {
                controls[slot].heldNanos += elapsed
            }
        }
        accountedAt = timestamp
    }

    private mutating func set(_ index: Int, down: Bool, at timestamp: UInt64) {
        guard let slot = TemporalInputContract.controlLookup[index], controls[slot].held != down else { return }
        if down {
            controls[slot].pressedAt = timestamp
            controls[slot].onsetKnown = !censorInitialOnsets || timestamp > origin
            controls[slot].presses += 1
        } else {
            if controls[slot].onsetKnown, let start = controls[slot].pressedAt {
                controls[slot].completedDuration = TemporalInputContract.seconds(timestamp, since: start)
            } else { controls[slot].completedDuration = nil }
            controls[slot].releasedAt = timestamp
            controls[slot].releases += 1
        }
        controls[slot].held = down
    }

    mutating func consume(_ sample: InputSample) {
        guard sample.timestampNanos >= accountedAt,
              [sample.x, sample.y, sample.deltaX, sample.deltaY, sample.scrollX, sample.scrollY].allSatisfy(\.isFinite) else { return }
        advance(to: sample.timestampNanos)
        let masks: [CGEventFlags] = [.maskShift, .maskControl, .maskAlternate, .maskCommand]
        for (modifier, mask) in masks.enumerated() {
            set(ActionLayout.modifiers.lowerBound + modifier,
                down: sample.modifiers & mask.rawValue != 0, at: sample.timestampNanos)
        }
        switch sample.kind {
        case .key:
            if sample.keyCode < 128 {
                set(ActionLayout.keyboard.lowerBound + Int(sample.keyCode), down: sample.isDown, at: sample.timestampNanos)
            }
        case .mouseButton:
            if sample.button < 8 { set(ActionLayout.buttons.lowerBound + Int(sample.button), down: sample.isDown, at: sample.timestampNanos) }
            pointer = CGPoint(x: sample.x, y: sample.y); hasPointer = true
        case .mouseMove:
            pointer = CGPoint(x: sample.x, y: sample.y); hasPointer = true
            movement.consume(x: sample.deltaX, y: sample.deltaY, at: sample.timestampNanos)
        case .scroll:
            pointer = CGPoint(x: sample.x, y: sample.y); hasPointer = true
            scrolling.consume(x: sample.scrollX, y: sample.scrollY, at: sample.timestampNanos)
        case .flags: break
        }
    }

    mutating func endInterval(at timestamp: UInt64) {
        advance(to: timestamp)
        intervalStart = max(intervalStart, timestamp)
        for slot in controls.indices {
            controls[slot].heldNanos = 0; controls[slot].presses = 0; controls[slot].releases = 0
        }
        movement.endInterval(); scrolling.endInterval()
    }

    func controlValues(rect: CGRect) -> [Float] {
        var values = [Float](repeating: 0, count: ActionLayout.count)
        if hasPointer {
            values[0] = Float(min(1, max(0, (pointer.x - rect.minX) / max(1, rect.width))))
            values[1] = Float(min(1, max(0, (pointer.y - rect.minY) / max(1, rect.height))))
        }
        values[2] = GameCameraContract.trainingValue(forRawDelta: movement.x)
        values[3] = GameCameraContract.trainingValue(forRawDelta: movement.y)
        values[12] = Float(min(1, max(-1, scrolling.x / 20)))
        values[13] = Float(min(1, max(-1, scrolling.y / 20)))
        for (slot, index) in TemporalInputContract.binaryIndices.enumerated() {
            values[index] = controls[slot].held || controls[slot].presses > 0 ? 1 : 0
        }
        return values
    }

    mutating func features(
        at timestamp: UInt64, frameTime: UInt64?, perceptionInterval: Double?,
        rect: CGRect, historyAge: Double? = 0, separation: Double? = nil
    ) -> [Float] {
        precondition(timestamp >= accountedAt, "Temporal queries must follow consumed events.")
        advance(to: timestamp)
        var values = [Float](repeating: 0, count: TemporalInputContract.valueCount)
        let interval = TemporalInputContract.seconds(timestamp, since: intervalStart)
        let frameAge = frameTime.flatMap { $0 <= timestamp ? TemporalInputContract.seconds(timestamp, since: $0) : nil }
        TemporalInputContract.putClock(in: &values, interval: interval > 0 ? interval : nil,
            perceptionInterval: perceptionInterval, frameAge: frameAge, historyAge: historyAge, separation: separation)
        if hasPointer {
            values[16] = Float(min(1, max(0, (pointer.x - rect.minX) / max(1, rect.width))))
            values[17] = Float(min(1, max(0, (pointer.y - rect.minY) / max(1, rect.height))))
        }
        for slot in controls.indices {
            let control = controls[slot]
            let base = TemporalInputContract.clockCount + slot * TemporalInputContract.controlWidth
            values[base] = control.held ? 1 : 0
            if let press = control.pressedAt, control.onsetKnown {
                let age = TemporalInputContract.seconds(timestamp, since: press)
                if control.held { TemporalInputContract.putTime(age, in: &values, at: base + 1) }
                TemporalInputContract.putTime(age, in: &values, at: base + 4)
            }
            if let release = control.releasedAt {
                TemporalInputContract.putTime(TemporalInputContract.seconds(timestamp, since: release), in: &values, at: base + 7)
            }
            TemporalInputContract.putTime(control.completedDuration, in: &values, at: base + 10)
            values[base + 13] = Float(log1p(Double(control.presses)))
            values[base + 14] = Float(log1p(Double(control.releases)))
            values[base + 15] = interval > 0 ? Float(min(1, Double(control.heldNanos) / (interval * 1e9))) : 0
        }
        putMotion(movement, at: TemporalInputContract.motionOffset, scale: Double(GameCameraContract.deltaScale), time: timestamp, interval: interval, values: &values)
        putMotion(scrolling, at: TemporalInputContract.motionOffset + TemporalInputContract.motionWidth, scale: 20, time: timestamp, interval: interval, values: &values)
        return values
    }

    private func putMotion(_ motion: Motion, at base: Int, scale: Double, time: UInt64, interval: Double, values: inout [Float]) {
        func signedLog(_ value: Double) -> Float { Float((value < 0 ? -1.0 : 1.0) * log1p(abs(value))) }
        values[base] = Float(min(1, max(-1, motion.x / scale)))
        values[base + 1] = Float(min(1, max(-1, motion.y / scale)))
        if interval > 0 {
            values[base + 2] = signedLog(motion.x / scale / interval)
            values[base + 3] = signedLog(motion.y / scale / interval)
        }
        values[base + 4] = Float(log1p(Double(motion.count)))
        if let first = motion.first, let last = motion.last {
            values[base + 5] = Float(TemporalInputContract.seconds(first, since: intervalStart))
            values[base + 6] = Float(TemporalInputContract.seconds(last, since: intervalStart))
            TemporalInputContract.putTime(TemporalInputContract.seconds(last, since: first), in: &values, at: base + 7)
        }
        if let last = motion.last {
            TemporalInputContract.putTime(TemporalInputContract.seconds(time, since: last), in: &values, at: base + 10)
        }
        values[base + 13] = interval > 0 ? 1 : 0
    }
}
