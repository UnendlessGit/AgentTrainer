import CoreGraphics
import Foundation

/// A recording package (`<name>.atrec`):
/// - `meta.json`            ``RecordingMeta``
/// - `annotations.json`     ``RecordingAnnotations`` (instructions, exclusions; user-editable)
/// - `video.mov`            full-quality HEVC
/// - `frames.bin`           ``FrameRecord`` per captured frame (host-clock timestamps)
/// - `events.bin`           ``InputEventRecord`` per input event (same clock)
/// - `proxy-<S>.atproxy`    lossless S×S RGB frames for training (``ProxyFile``)
/// - `thumb.jpg`
public enum RecordingPackage {
    public static let fileExtension = "atrec"
    public static let meta = "meta.json"
    public static let annotations = "annotations.json"
    public static let video = "video.mov"
    public static let frames = "frames.bin"
    public static let events = "events.bin"
    public static let thumbnail = "thumb.jpg"
    public static func proxy(size: Int) -> String { "proxy-\(size).atproxy" }
}

public struct CaptureTargetInfo: Codable, Hashable, Sendable {
    public enum Kind: String, Codable, Sendable { case display, window, testArena }
    public var kind: Kind
    public var displayID: UInt32?
    public var windowID: UInt32?
    public var appBundleID: String?
    public var appName: String?
    public var title: String?
    public init(kind: Kind, displayID: UInt32? = nil, windowID: UInt32? = nil, appBundleID: String? = nil, appName: String? = nil, title: String? = nil) {
        self.kind = kind; self.displayID = displayID; self.windowID = windowID
        self.appBundleID = appBundleID; self.appName = appName; self.title = title
    }
    public var displayName: String {
        switch kind {
        case .display: title ?? "Display"
        case .window: [appName, title].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " — ")
        case .testArena: "Test Arena"
        }
    }
}

public struct InputCaptureOptions: Codable, Hashable, Sendable {
    public var keyboard = true
    public var mouseMovement = true
    public var mouseButtons = true
    public var scroll = true
    /// Ignore input while another app is in front (window targets).
    public var onlyWhenTargetFocused = true
    public init() {}
}

public struct RecordingMeta: Codable, Hashable, Sendable, Identifiable {
    public var id: UUID
    public var name: String
    public var createdAt: Date
    public var formatVersion = AgentTrainerInfo.formatVersion
    public var appVersion: String
    public var target: CaptureTargetInfo
    /// Captured pixel size and the source area it maps to in global screen points.
    public var captureWidth: Int
    public var captureHeight: Int
    public var sourceRect: CodableRect
    public var captureFPS: Int
    public var showsCursor: Bool
    public var input: InputCaptureOptions
    public var task: String
    public var tags: [String]
    public var duration: Double = 0
    public var frameCount = 0
    public var droppedFrames = 0
    public var eventCount = 0
    public var proxySizes: [Int] = []
    /// Std of pointer deltas per 50 ms window (points), used to scale into P2P mouse units.
    public var mouseDeltaStd: [Double]? = nil
    /// How CGEvent timestamps were mapped to the host clock.
    public var eventClock: String = "uptime-ns"
    public var isCorrectionRun = false
    public var modelName: String? = nil
    /// Regions blacked out in every frame (normalised, top-left origin), e.g. the timecode corner
    /// of an Elkborn replay. Runs of models trained on these recordings black out the same regions.
    public var masks: [CodableRect]? = nil

    public init(id: UUID = UUID(), name: String, createdAt: Date = Date(), appVersion: String, target: CaptureTargetInfo,
                captureWidth: Int, captureHeight: Int, sourceRect: CodableRect, captureFPS: Int, showsCursor: Bool,
                input: InputCaptureOptions, task: String = "", tags: [String] = []) {
        self.id = id; self.name = name; self.createdAt = createdAt; self.appVersion = appVersion; self.target = target
        self.captureWidth = captureWidth; self.captureHeight = captureHeight; self.sourceRect = sourceRect
        self.captureFPS = captureFPS; self.showsCursor = showsCursor; self.input = input; self.task = task; self.tags = tags
    }
}

public struct CodableRect: Codable, Hashable, Sendable {
    public var x, y, width, height: Double
    public init(x: Double, y: Double, width: Double, height: Double) { self.x = x; self.y = y; self.width = width; self.height = height }
    public init(_ r: CGRect) { self.init(x: r.origin.x, y: r.origin.y, width: r.width, height: r.height) }
    public var cgRect: CGRect { CGRect(x: x, y: y, width: width, height: height) }
}

public struct TextSegment: Codable, Hashable, Sendable, Identifiable {
    public enum Source: String, Codable, Sendable { case user, auto, typing, task }
    public var id = UUID()
    public var start: Double
    public var end: Double
    public var text: String
    public var source: Source
    public init(start: Double, end: Double, text: String, source: Source) { self.start = start; self.end = end; self.text = text; self.source = source }
}

public struct TimeRange: Codable, Hashable, Sendable, Identifiable {
    public var id = UUID()
    public var start: Double
    public var end: Double
    public init(start: Double, end: Double) { self.start = start; self.end = end }
    public func contains(_ t: Double) -> Bool { t >= start && t < end }
}

public struct RecordingAnnotations: Codable, Hashable, Sendable {
    public var segments: [TextSegment] = []
    /// Ranges left out of training (menus, idle, mistakes).
    public var excluded: [TimeRange] = []
    /// Ranges where a model (not the human) was in control - kept as history, masked from the loss.
    public var modelControlled: [TimeRange] = []
    public init() {}

    /// Instruction active at time `t` (latest segment containing it).
    public func instruction(at t: Double) -> String? {
        segments.last(where: { t >= $0.start && t < $0.end })?.text
    }
}

/// One captured frame on the host clock.
public struct FrameRecord: Hashable, Sendable {
    public static let size = 24
    public var index: UInt32
    public var flags: UInt32 // bit 0: frame delivered to the proxy
    /// Nanoseconds since the recording's time zero (first frame's display time).
    public var timeNs: Int64
    /// Absolute host time in nanoseconds.
    public var hostNs: UInt64
    public init(index: UInt32, flags: UInt32 = 1, timeNs: Int64, hostNs: UInt64) { self.index = index; self.flags = flags; self.timeNs = timeNs; self.hostNs = hostNs }
}

public enum InputEventKind: UInt8, Codable, Sendable {
    case keyDown = 1, keyUp, flagsChanged, mouseMove, buttonDown, buttonUp, scroll
}

/// One input event on the host clock. Fixed 32-byte binary layout.
public struct InputEventRecord: Hashable, Sendable {
    public static let size = 32
    public var timeNs: Int64
    public var kind: InputEventKind
    /// bit 0: synthetic (posted by AgentTrainer's runtime), bit 1: modifier pressed (flagsChanged)
    public var flags: UInt8
    /// Key code for key events, button number for button events.
    public var code: UInt16
    /// Pointer location in global points (top-left origin).
    public var x: Float
    public var y: Float
    /// Pointer delta (moves) or scroll delta in lines (scroll).
    public var dx: Float
    public var dy: Float
    public var reserved: UInt32 = 0

    public init(timeNs: Int64, kind: InputEventKind, flags: UInt8 = 0, code: UInt16 = 0, x: Float = 0, y: Float = 0, dx: Float = 0, dy: Float = 0) {
        self.timeNs = timeNs; self.kind = kind; self.flags = flags; self.code = code; self.x = x; self.y = y; self.dx = dx; self.dy = dy
    }
    public var isSynthetic: Bool { flags & 1 != 0 }
}

// MARK: Binary encoding

public enum BinaryRecords {
    public static func encode(_ f: FrameRecord) -> Data {
        var d = Data(capacity: FrameRecord.size)
        d.append(le: f.index); d.append(le: f.flags); d.append(le: f.timeNs); d.append(le: f.hostNs)
        return d
    }

    public static func encode(_ e: InputEventRecord) -> Data {
        var d = Data(capacity: InputEventRecord.size)
        d.append(le: e.timeNs); d.append(e.kind.rawValue); d.append(e.flags); d.append(le: e.code)
        d.append(le: e.x.bitPattern); d.append(le: e.y.bitPattern); d.append(le: e.dx.bitPattern); d.append(le: e.dy.bitPattern)
        d.append(le: e.reserved)
        return d
    }

    public static func decodeFrames(_ data: Data) -> [FrameRecord] {
        stride(from: 0, to: data.count - FrameRecord.size + 1, by: FrameRecord.size).map { o in
            FrameRecord(index: data.le(o), flags: data.le(o + 4), timeNs: data.le(o + 8), hostNs: data.le(o + 16))
        }
    }

    public static func decodeEvents(_ data: Data) -> [InputEventRecord] {
        stride(from: 0, to: data.count - InputEventRecord.size + 1, by: InputEventRecord.size).compactMap { o in
            guard let kind = InputEventKind(rawValue: data[data.startIndex + o + 8]) else { return nil }
            return InputEventRecord(timeNs: data.le(o), kind: kind, flags: data[data.startIndex + o + 9], code: data.le(o + 10),
                                    x: Float(bitPattern: data.le(o + 12)), y: Float(bitPattern: data.le(o + 16)),
                                    dx: Float(bitPattern: data.le(o + 20)), dy: Float(bitPattern: data.le(o + 24)))
        }
    }
}

extension Data {
    mutating func append<T: FixedWidthInteger>(le v: T) {
        Swift.withUnsafeBytes(of: v.littleEndian) { append(contentsOf: $0) }
    }

    func le<T: FixedWidthInteger>(_ offset: Int) -> T {
        var v: T = 0
        _ = Swift.withUnsafeMutableBytes(of: &v) { copyBytes(to: $0, from: (startIndex + offset) ..< (startIndex + offset + MemoryLayout<T>.size)) }
        return T(littleEndian: v)
    }
}

/// Reads and writes the JSON parts of a recording package.
public enum RecordingIO {
    public static let encoder: JSONEncoder = {
        let e = JSONEncoder(); e.outputFormatting = [.prettyPrinted, .sortedKeys]; e.dateEncodingStrategy = .iso8601; return e
    }()
    public static let decoder: JSONDecoder = {
        let d = JSONDecoder(); d.dateDecodingStrategy = .iso8601; return d
    }()

    public static func readMeta(_ package: URL) throws -> RecordingMeta {
        try decoder.decode(RecordingMeta.self, from: Data(contentsOf: package.appendingPathComponent(RecordingPackage.meta)))
    }
    public static func writeMeta(_ m: RecordingMeta, to package: URL) throws {
        try encoder.encode(m).write(to: package.appendingPathComponent(RecordingPackage.meta), options: .atomic)
    }
    public static func readAnnotations(_ package: URL) -> RecordingAnnotations {
        (try? decoder.decode(RecordingAnnotations.self, from: Data(contentsOf: package.appendingPathComponent(RecordingPackage.annotations)))) ?? RecordingAnnotations()
    }
    public static func writeAnnotations(_ a: RecordingAnnotations, to package: URL) throws {
        try encoder.encode(a).write(to: package.appendingPathComponent(RecordingPackage.annotations), options: .atomic)
    }
    public static func readFrames(_ package: URL) throws -> [FrameRecord] {
        BinaryRecords.decodeFrames(try Data(contentsOf: package.appendingPathComponent(RecordingPackage.frames)))
    }
    public static func readEvents(_ package: URL) throws -> [InputEventRecord] {
        BinaryRecords.decodeEvents(try Data(contentsOf: package.appendingPathComponent(RecordingPackage.events)))
    }
}
