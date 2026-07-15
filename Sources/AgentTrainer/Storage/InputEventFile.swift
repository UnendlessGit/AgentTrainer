import CoreGraphics
import Foundation

final class InputEventWriter: @unchecked Sendable {
    private let handle: FileHandle
    private let lock = NSLock()
    private var buffer = Data()
    private(set) var count = 0
    private var closed = false
    private var failure: Error?

    init(url: URL) throws {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
        try handle.write(contentsOf: Data("ATREVT01".utf8))
        var version = UInt32(1).littleEndian
        try withUnsafeBytes(of: &version) { try handle.write(contentsOf: Data($0)) }
        buffer.reserveCapacity(256 * 1024)
    }

    func append(_ event: InputSample) {
        lock.lock()
        defer { lock.unlock() }
        guard !closed, failure == nil else { return }
        buffer.appendInteger(event.timestampNanos)
        buffer.append(event.kind.rawValue)
        buffer.append(event.isDown ? 1 : 0)
        buffer.append(event.button)
        buffer.append(0)
        buffer.appendInteger(event.keyCode)
        buffer.appendInteger(UInt16(0))
        buffer.appendInteger(event.modifiers)
        buffer.appendDouble(event.x)
        buffer.appendDouble(event.y)
        buffer.appendDouble(event.deltaX)
        buffer.appendDouble(event.deltaY)
        buffer.appendDouble(event.scrollX)
        buffer.appendDouble(event.scrollY)
        count += 1
        if buffer.count >= 256 * 1024 { flushLocked() }
    }

    func finish() throws -> Int {
        lock.lock()
        defer { lock.unlock() }
        if closed {
            if let failure { throw failure }
            return count
        }
        flushLocked()
        if failure == nil {
            do { try handle.synchronize() }
            catch { failure = error }
        }
        do { try handle.close() }
        catch { if failure == nil { failure = error } }
        closed = true
        if let failure {
            throw AgentTrainerError.storage("The recorded input file could not be saved: \(failure.localizedDescription)")
        }
        return count
    }

    private func flushLocked() {
        guard !buffer.isEmpty, failure == nil else { return }
        do { try handle.write(contentsOf: buffer) }
        catch { failure = error }
        buffer.removeAll(keepingCapacity: true)
    }
}

enum InputEventReader {
    static let recordSize = 72

    struct MouseDiagnostics: Sendable, Equatable {
        var moveEventCount = 0
        var nonzeroDeltaCount = 0
        var absolutePositionChangeCount = 0
        var outOfCaptureBoundsCount = 0
        var accumulatedDeltaMagnitude = 0.0
        var maximumDeltaMagnitude = 0.0

        var nonzeroDeltaFraction: Double { Double(nonzeroDeltaCount) / Double(max(1, moveEventCount)) }
        var absolutePositionChangeFraction: Double { Double(absolutePositionChangeCount) / Double(max(1, moveEventCount - 1)) }
        var meanActiveDeltaMagnitude: Double { accumulatedDeltaMagnitude / Double(max(1, nonzeroDeltaCount)) }
        var isGameCamera: Bool {
            moveEventCount >= 20 && nonzeroDeltaCount > 0 && absolutePositionChangeFraction < 0.05
        }
        var positionsAreValid: Bool { outOfCaptureBoundsCount == 0 }
    }

    struct Summary: Sendable {
        var preview: [InputSample]
        var usedKeyCodes: Set<UInt16>
        var keyEventCount: Int
        var mouseEventCount: Int
        var mouse = MouseDiagnostics()
    }

    static func read(url: URL) throws -> [InputSample] {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        var result: [InputSample] = []
        result.reserveCapacity(max(0, (data.count - 12) / recordSize))
        try forEach(in: data) { result.append($0) }
        return result
    }

    static func summarize(url: URL, previewLimit: Int = 80, globalRect: CGRect? = nil) throws -> Summary {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        var summary = Summary(preview: [], usedKeyCodes: [], keyEventCount: 0, mouseEventCount: 0)
        var previousMousePosition: CGPoint?
        summary.preview.reserveCapacity(max(0, previewLimit))
        try forEach(in: data) { event in
            if summary.preview.count < previewLimit { summary.preview.append(event) }
            switch event.kind {
            case .key:
                summary.keyEventCount += 1
                summary.usedKeyCodes.insert(event.keyCode)
            case .flags:
                summary.usedKeyCodes.formUnion(modifierKeyCodes(in: event.modifiers))
            case .mouseMove:
                summary.mouseEventCount += 1
                summary.mouse.moveEventCount += 1
                let magnitude = abs(event.deltaX) + abs(event.deltaY)
                if magnitude > 0 {
                    summary.mouse.nonzeroDeltaCount += 1
                    summary.mouse.accumulatedDeltaMagnitude += magnitude
                    summary.mouse.maximumDeltaMagnitude = max(summary.mouse.maximumDeltaMagnitude, magnitude)
                }
                let position = CGPoint(x: event.x, y: event.y)
                if let previousMousePosition,
                   abs(position.x - previousMousePosition.x) > 0.01 || abs(position.y - previousMousePosition.y) > 0.01 {
                    summary.mouse.absolutePositionChangeCount += 1
                }
                previousMousePosition = position
                if let globalRect, !globalRect.insetBy(dx: -0.5, dy: -0.5).contains(position) {
                    summary.mouse.outOfCaptureBoundsCount += 1
                }
            case .mouseButton, .scroll: summary.mouseEventCount += 1
            }
        }
        return summary
    }

    static func demonstratedKeyCodes(url: URL) throws -> Set<UInt16> {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        var result: Set<UInt16> = []
        try forEach(in: data) { event in
            if event.kind == .key { result.insert(event.keyCode) }
            result.formUnion(modifierKeyCodes(in: event.modifiers))
        }
        return result
    }

    static func mouseDiagnostics(events: [InputSample], globalRect: CGRect? = nil) -> MouseDiagnostics {
        var diagnostics = MouseDiagnostics()
        var previousPosition: CGPoint?
        for event in events where event.kind == .mouseMove {
            diagnostics.moveEventCount += 1
            let magnitude = abs(event.deltaX) + abs(event.deltaY)
            if magnitude > 0 {
                diagnostics.nonzeroDeltaCount += 1
                diagnostics.accumulatedDeltaMagnitude += magnitude
                diagnostics.maximumDeltaMagnitude = max(diagnostics.maximumDeltaMagnitude, magnitude)
            }
            let position = CGPoint(x: event.x, y: event.y)
            if let previousPosition,
               abs(position.x - previousPosition.x) > 0.01 || abs(position.y - previousPosition.y) > 0.01 {
                diagnostics.absolutePositionChangeCount += 1
            }
            previousPosition = position
            if let globalRect, !globalRect.insetBy(dx: -0.5, dy: -0.5).contains(position) {
                diagnostics.outOfCaptureBoundsCount += 1
            }
        }
        return diagnostics
    }

    private static func modifierKeyCodes(in flags: UInt64) -> Set<UInt16> {
        let mappings: [(CGEventFlags, UInt16)] = [(.maskShift, 56), (.maskControl, 59), (.maskAlternate, 58), (.maskCommand, 55)]
        return Set(mappings.compactMap { flags & $0.0.rawValue != 0 ? $0.1 : nil })
    }

    private static func forEach(in data: Data, _ body: (InputSample) -> Void) throws {
        guard data.count >= 12, String(data: data.prefix(8), encoding: .utf8) == "ATREVT01" else { throw AgentTrainerError.storage("Invalid AgentTrainer input event file.") }
        var headerCursor = 8
        let version: UInt32 = data.readInteger(at: &headerCursor)
        guard version == 1, (data.count - 12).isMultiple(of: recordSize) else {
            throw AgentTrainerError.storage("This AgentTrainer input event file is unsupported or incomplete.")
        }
        var cursor = 12
        while cursor + recordSize <= data.count {
            let timestamp: UInt64 = data.readInteger(at: &cursor)
            let kindRaw = data[cursor]; cursor += 1
            let isDown = data[cursor] != 0; cursor += 1
            let button = data[cursor]; cursor += 2
            let keyCode: UInt16 = data.readInteger(at: &cursor); cursor += 2
            let modifiers: UInt64 = data.readInteger(at: &cursor)
            let x = data.readDouble(at: &cursor), y = data.readDouble(at: &cursor)
            let dx = data.readDouble(at: &cursor), dy = data.readDouble(at: &cursor)
            let sx = data.readDouble(at: &cursor), sy = data.readDouble(at: &cursor)
            guard let kind = InputEventKind(rawValue: kindRaw) else { continue }
            body(InputSample(timestampNanos: timestamp, kind: kind, x: x, y: y, deltaX: dx, deltaY: dy, button: button, scrollX: sx, scrollY: sy, keyCode: keyCode, modifiers: modifiers, isDown: isDown))
        }
    }
}

private extension Data {
    mutating func appendInteger<T: FixedWidthInteger>(_ value: T) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
    }

    mutating func appendDouble(_ value: Double) {
        appendInteger(value.bitPattern)
    }

    func readInteger<T: FixedWidthInteger>(at cursor: inout Int) -> T {
        let size = MemoryLayout<T>.size
        let value = self[cursor..<(cursor + size)].withUnsafeBytes { $0.loadUnaligned(as: T.self) }
        cursor += size
        return T(littleEndian: value)
    }

    func readDouble(at cursor: inout Int) -> Double {
        Double(bitPattern: readInteger(at: &cursor) as UInt64)
    }
}
