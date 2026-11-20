import CoreGraphics
import Foundation

/// AgentTrainer's side of Elkborn's on-screen Training Data protocol (in the game:
/// `Framework.Trainer.Protocol`). A stored time trial run is replayed in Roblox, and everything
/// AgentTrainer needs travels through the picture as black and white cells:
///
/// - **Pages** (full screen, before playback): a clock row and a clock column of alternating cells,
///   then data cells row by row, MSB first. Header (big endian): magic `E7B1`, session, kind,
///   page index, page count, length (u16), total length (u32), version, reserved; then the
///   payload slice and a CRC-16/CCITT-FALSE. Both clock lines have an odd number of cells, so
///   they end on a white cell and the grid size can be counted exactly.
/// - **Strip** (bottom-left corner, during playback): a 32-cell clock row and two 32-bit data rows:
///   sync `1011`, session, frame counter, value (replay time in ms), flags and a CRC-8.
///
/// Handshake: a beacon page waits until AgentTrainer presses F8, the timeline pages loop until it
/// presses F8 again, then playback starts with the strip, and an end page closes the session.
public enum ElkbornProtocol {
    public static let pageMagic: UInt16 = 0xE7B1
    public static let payloadMagic: UInt32 = 0x3150_5445 // "ETP1" little endian
    public static let version: UInt8 = 1
    public static let stripColumns = 32
    /// Strip size in cells including its one-cell black margin (34 x 5).
    public static let stripCells = (width: 34.0, height: 5.0)
    /// macOS virtual key codes of the game's handshake keys.
    public static let goKeyCode: UInt16 = 0x64 // F8
    public static let cancelKeyCode: UInt16 = 0x62 // F7

    public enum Kind: UInt8, Sendable { case beacon = 1, data = 2, end = 3, calibration = 4 }

    public struct StripFlags: OptionSet, Sendable {
        public let rawValue: UInt8
        public init(rawValue: UInt8) { self.rawValue = rawValue }
        public static let playing = StripFlags(rawValue: 1)
        public static let end = StripFlags(rawValue: 2)
        public static let calibration = StripFlags(rawValue: 4)
    }

    public struct Strip: Equatable, Sendable {
        public var session: UInt8
        public var frame: UInt16
        /// Replay time in ms during playback; camera pitch while calibrating.
        public var value: UInt32
        public var flags: StripFlags
        /// Where the strip was found, in frame pixels (top-left origin): used to black it out.
        public var rect: CGRect
        public var cell: Double
    }

    public struct Page: Equatable, Sendable {
        public var session: UInt8
        public var kind: Kind
        public var index: Int
        public var count: Int
        public var total: Int
        public var payload: Data
    }

    // MARK: Checksums

    public static func crc8(_ bytes: [UInt8]) -> UInt8 {
        var crc: UInt8 = 0
        for b in bytes {
            crc ^= b
            for _ in 0 ..< 8 { crc = crc & 0x80 != 0 ? (crc << 1) ^ 0x07 : crc << 1 }
        }
        return crc
    }

    public static func crc16(_ bytes: [UInt8]) -> UInt16 {
        var crc: UInt16 = 0xFFFF
        for b in bytes {
            crc ^= UInt16(b) << 8
            for _ in 0 ..< 8 { crc = crc & 0x8000 != 0 ? (crc << 1) ^ 0x1021 : crc << 1 }
        }
        return crc
    }
}

/// A read-only grayscale view of a BGRA frame.
public struct LumaFrame {
    public let width: Int
    public let height: Int
    private let base: UnsafePointer<UInt8>
    private let bytesPerRow: Int

    public init(base: UnsafeRawPointer, width: Int, height: Int, bytesPerRow: Int) {
        self.base = base.assumingMemoryBound(to: UInt8.self)
        self.width = width; self.height = height; self.bytesPerRow = bytesPerRow
    }

    @inline(__always) func luma(_ x: Int, _ y: Int) -> Int {
        let p = base + y * bytesPerRow + x * 4
        return (Int(p[2]) * 77 + Int(p[1]) * 150 + Int(p[0]) * 29) >> 8
    }

    @inline(__always) func white(_ x: Int, _ y: Int) -> Bool { luma(x, y) >= 128 }

    /// Majority of a 3x3 neighbourhood around a cell centre: robust to blur and compression.
    func cellIsWhite(_ cx: Double, _ cy: Double) -> Bool {
        let x = Int(cx.rounded()), y = Int(cy.rounded())
        var sum = 0, n = 0
        for dy in -1 ... 1 {
            for dx in -1 ... 1 {
                let px = x + dx, py = y + dy
                guard px >= 0, py >= 0, px < width, py < height else { continue }
                sum += luma(px, py); n += 1
            }
        }
        return n > 0 && sum / n >= 128
    }
}

extension ElkbornProtocol {
    struct Clock {
        var start: Double
        var end: Double
        var runs: Int
        var cell: Double { (end - start) / Double(runs) }
    }

    static func horizontalClock(_ f: LumaFrame, y: Int, from x0: Int, to x1: Int, minRuns: Int) -> Clock? {
        scanClock(from: x0, to: x1, minRuns: minRuns) { f.white($0, y) }
    }

    static func verticalClock(_ f: LumaFrame, x: Int, from y0: Int, to y1: Int, minRuns: Int) -> Clock? {
        scanClock(from: y0, to: y1, minRuns: minRuns) { f.white(x, $0) }
    }

    /// The first stretch of `minRuns`+ alternating, near-equal runs (white first) along a line,
    /// ending on its last white run. One pass, no allocation.
    @inline(__always)
    static func scanClock(from p0: Int, to p1: Int, minRuns: Int, white: (Int) -> Bool) -> Clock? {
        guard p1 - p0 > minRuns * 2 else { return nil }
        var streakStart = 0, size = 0.0, count = 0, lastWhiteEnd = 0, lastWhiteCount = 0
        func take(_ start: Int, _ length: Int, _ isWhite: Bool) -> Clock? {
            if count > 0, abs(Double(length) - size) <= max(1.6, size * 0.35) {
                count += 1
                if isWhite { lastWhiteEnd = start + length; lastWhiteCount = count }
                return nil
            }
            if lastWhiteCount >= minRuns {
                return Clock(start: Double(streakStart), end: Double(lastWhiteEnd), runs: lastWhiteCount)
            }
            count = 0; lastWhiteCount = 0
            if isWhite, length >= 2 {
                streakStart = start; size = Double(length); count = 1; lastWhiteEnd = start + length; lastWhiteCount = 1
            }
            return nil
        }
        var current = white(p0)
        var runStart = p0
        var p = p0 + 1
        while p < p1 {
            let w = white(p)
            if w != current {
                if let c = take(runStart, p - runStart, current) { return c }
                current = w; runStart = p
            }
            p += 1
        }
        if let c = take(runStart, p1 - runStart, current) { return c }
        // a stretch that reached the end of the line
        return lastWhiteCount >= minRuns ? Clock(start: Double(streakStart), end: Double(lastWhiteEnd), runs: lastWhiteCount) : nil
    }

    static func bytes(fromBits bits: [Bool]) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: bits.count / 8)
        for i in 0 ..< out.count {
            var b: UInt8 = 0
            for k in 0 ..< 8 { b = (b << 1) | (bits[i * 8 + k] ? 1 : 0) }
            out[i] = b
        }
        return out
    }

    /// Where the strip's clock row was found: its first cell's left edge, the cell size and the
    /// row's bottom edge (the data rows are right below it).
    public struct StripGeometry: Equatable, Sendable {
        var x0: Double
        var cell: Double
        var bottom: Double
    }

    /// Where a page grid was found: the clock row's start and cell size, the clock column's start
    /// and cell size, and the number of cells along each.
    public struct PageGeometry: Equatable, Sendable {
        var x0: Double, cellX: Double, cols: Int
        var y0: Double, cellY: Double, rows: Int
    }

    /// Reads the strip at a known place; nil unless the sync nibble and CRC match.
    public static func readStrip(_ f: LumaFrame, _ g: StripGeometry) -> Strip? {
        var bits: [Bool] = []
        bits.reserveCapacity(64)
        for row in 0 ..< 2 {
            let cy = g.bottom + g.cell * (0.5 + Double(row))
            for col in 0 ..< stripColumns { bits.append(f.cellIsWhite(g.x0 + g.cell * (Double(col) + 0.5), cy)) }
        }
        let bytes = Self.bytes(fromBits: bits)
        guard bytes[0] >> 4 == 0xB, crc8(Array(bytes[0 ..< 7])) == bytes[7] else { return nil }
        let session = (bytes[0] << 4) | (bytes[1] >> 4)
        let frame = (UInt16(bytes[1] & 0x0F) << 12) | (UInt16(bytes[2]) << 4) | UInt16(bytes[3] >> 4)
        let value = (UInt32(bytes[3] & 0x0F) << 20) | (UInt32(bytes[4]) << 12) | (UInt32(bytes[5]) << 4) | UInt32(bytes[6] >> 4)
        let top = g.bottom - g.cell * 2
        let rect = CGRect(x: max(0, g.x0 - g.cell), y: top, width: g.cell * stripCells.width, height: Double(f.height) - top)
        return Strip(session: session, frame: frame, value: value, flags: StripFlags(rawValue: bytes[6] & 0x0F), rect: rect, cell: g.cell)
    }

    /// Searches the bottom-left corner for the strip.
    public static func findStrip(_ f: LumaFrame) -> (Strip, StripGeometry)? {
        let x1 = min(f.width, max(64, f.width / 2))
        var y = f.height - 2
        let top = max(0, f.height - f.height / 3)
        while y > top {
            if let c = horizontalClock(f, y: y, from: 0, to: x1, minRuns: 25), c.runs == stripColumns - 1 {
                // scanning upward reaches the clock row's bottom edge first; the data rows are below
                let g = StripGeometry(x0: c.start, cell: c.cell, bottom: Double(y))
                if let strip = readStrip(f, g) { return (strip, g) }
                y -= max(1, Int(c.cell / 2))
                continue
            }
            y -= 1
        }
        return nil
    }

    public static func decodeStrip(_ f: LumaFrame) -> Strip? { findStrip(f)?.0 }

    /// Reads a page at a known grid; nil unless the magic and CRC match.
    public static func readPage(_ f: LumaFrame, _ g: PageGeometry) -> Page? {
        guard g.cols > 1, g.rows > 1 else { return nil }
        var bits: [Bool] = []
        bits.reserveCapacity((g.cols - 1) * (g.rows - 1))
        for r in 1 ..< g.rows {
            let cy = g.y0 + g.cellY * (Double(r) + 0.5)
            guard cy < Double(f.height) else { return nil }
            for c in 1 ..< g.cols { bits.append(f.cellIsWhite(g.x0 + g.cellX * (Double(c) + 0.5), cy)) }
        }
        return parsePage(Self.bytes(fromBits: bits))
    }

    /// Searches the top half of the frame for a page grid.
    public static func findPage(_ f: LumaFrame) -> (Page, PageGeometry)? {
        let limit = f.height / 2
        var y = 0
        while y < limit {
            guard let row = horizontalClock(f, y: y, from: 0, to: f.width, minRuns: 21), row.runs % 2 == 1 else { y += 1; continue }
            let x = Int((row.start + row.cell / 2).rounded())
            guard let column = verticalClock(f, x: x, from: y, to: f.height, minRuns: 5), column.runs % 2 == 1,
                  abs(column.start - Double(y)) <= row.cell else { y += max(1, Int(row.cell / 2)); continue }
            let g = PageGeometry(x0: row.start, cellX: row.cell, cols: row.runs, y0: column.start, cellY: column.cell, rows: column.runs)
            if let page = readPage(f, g) { return (page, g) }
            y += max(1, Int(row.cell / 2))
        }
        return nil
    }

    public static func decodePage(_ f: LumaFrame) -> Page? { findPage(f)?.0 }

    static func parsePage(_ b: [UInt8]) -> Page? {
        guard b.count >= 16, (UInt16(b[0]) << 8 | UInt16(b[1])) == pageMagic else { return nil }
        let length = Int(b[6]) << 8 | Int(b[7])
        guard 14 + length + 2 <= b.count, let kind = Kind(rawValue: b[3]) else { return nil }
        let crc = UInt16(b[14 + length]) << 8 | UInt16(b[15 + length])
        guard crc16(Array(b[0 ..< 14 + length])) == crc else { return nil }
        let total = Int(b[8]) << 24 | Int(b[9]) << 16 | Int(b[10]) << 8 | Int(b[11])
        return Page(session: b[2], kind: kind, index: Int(b[4]), count: Int(b[5]), total: total, payload: Data(b[14 ..< 14 + length]))
    }
}

/// Remembers where the strip and the page grid were last found, so each frame costs a few
/// hundred pixel reads instead of a search. Searches again whenever a read fails.
public struct ElkbornScanner: Sendable {
    private var strip: ElkbornProtocol.StripGeometry?
    private var page: ElkbornProtocol.PageGeometry?
    private var pageMisses = 0
    private var size = (0, 0)

    public init() {}

    private mutating func check(_ f: LumaFrame) {
        if size != (f.width, f.height) { size = (f.width, f.height); strip = nil; page = nil }
    }

    public mutating func strip(_ f: LumaFrame) -> ElkbornProtocol.Strip? {
        check(f)
        if let g = strip, let s = ElkbornProtocol.readStrip(f, g) { return s }
        guard let (s, g) = ElkbornProtocol.findStrip(f) else { return nil }
        strip = g
        return s
    }

    /// `search: false` only reads at the known grid (cheap), for frames where a page is unlikely.
    public mutating func page(_ f: LumaFrame, search: Bool = true) -> ElkbornProtocol.Page? {
        check(f)
        if let g = page, let p = ElkbornProtocol.readPage(f, g) { pageMisses = 0; return p }
        guard search else { return nil }
        // a full search is costly: while nothing is on screen, try it on every third frame
        pageMisses += 1
        if page == nil, pageMisses % 3 != 1 { return nil }
        guard let (p, g) = ElkbornProtocol.findPage(f) else { return nil }
        page = g
        pageMisses = 0
        return p
    }
}

/// Collects the pages of one payload (they loop on screen until all have been seen).
public struct ElkbornPageAssembler: Sendable {
    public private(set) var session: UInt8?
    public private(set) var kind: ElkbornProtocol.Kind?
    public private(set) var count = 0
    private var parts: [Int: Data] = [:]

    public init() {}

    public var received: Int { parts.count }
    public var isComplete: Bool { count > 0 && parts.count == count }

    /// Returns the whole payload once every page has been seen.
    public mutating func add(_ page: ElkbornProtocol.Page) -> Data? {
        if page.session != session || page.kind != kind || page.count != count {
            session = page.session; kind = page.kind; count = page.count; parts = [:]
        }
        parts[page.index] = page.payload
        guard isComplete else { return nil }
        var out = Data()
        for i in 0 ..< count { out.append(parts[i]!) }
        return out
    }
}

// MARK: Payloads

struct ElkbornReader {
    let d: [UInt8]
    var p = 0
    init(_ data: Data) { d = [UInt8](data) }

    enum Failure: Error { case truncated, bad(String) }

    mutating func u8() throws -> UInt8 { guard p < d.count else { throw Failure.truncated }; defer { p += 1 }; return d[p] }
    mutating func u32() throws -> UInt32 {
        guard p + 4 <= d.count else { throw Failure.truncated }
        defer { p += 4 }
        return UInt32(d[p]) | UInt32(d[p + 1]) << 8 | UInt32(d[p + 2]) << 16 | UInt32(d[p + 3]) << 24
    }
    mutating func f64() throws -> Double {
        guard p + 8 <= d.count else { throw Failure.truncated }
        var bits: UInt64 = 0
        for i in 0 ..< 8 { bits |= UInt64(d[p + i]) << (8 * UInt64(i)) }
        p += 8
        return Double(bitPattern: bits)
    }
    mutating func str() throws -> String {
        let n = Int(try u8())
        guard p + n <= d.count else { throw Failure.truncated }
        defer { p += n }
        return String(decoding: d[p ..< p + n], as: UTF8.self)
    }
    mutating func varu() throws -> Int {
        var value = 0, shift = 0
        for _ in 0 ..< 10 {
            let b = try u8()
            value |= Int(b & 0x7F) << shift
            if b < 0x80 { return value }
            shift += 7
        }
        throw Failure.bad("varint")
    }
    mutating func vars() throws -> Int {
        let z = try varu()
        return z & 1 == 0 ? z >> 1 : -((z + 1) >> 1)
    }
}

/// Everything a replay sends before playback: the run's inputs and camera track, and the
/// viewer's keybinds and mouse conversion at replay time.
public struct ElkbornTimeline: Sendable, Equatable {
    public struct Sample: Sendable, Equatable {
        public var timeMs: Int
        /// Camera angles in 1/65536 turns.
        public var pitch: Int
        public var yaw: Int
        /// The camera was turned by the game (a scripted look-at), not by the mouse.
        public var forced: Bool
    }

    public struct Event: Sendable, Equatable {
        public var timeMs: Int
        public var code: UInt8
        public var down: Bool
    }

    /// The runner's mouse input for one frame (version 3 runs): the camera turn it commanded, in
    /// radians, before anything else the game does to the camera.
    public struct MouseInput: Sendable, Equatable {
        public var timeMs: Int
        public var yaw: Double
        public var pitch: Double
    }

    public var session: UInt8
    public var finalTime: Double
    public var sampleHz: Int
    public var runID: String
    public var board: String
    public var runner: String
    public var avatar: String
    public var avatarUserID: Double
    /// Action code -> the viewer's own Roblox key or mouse button for it ("W", "Space", "MouseButton3").
    public var keys: [UInt8: String]
    /// Camera radians per Roblox mouse pixel for the viewer (pitch includes the invert sign).
    public var yawPerPixel: Double
    public var pitchPerPixel: Double
    public var samples: [Sample]
    public var events: [Event]
    /// Empty for runs saved before version 3: their mouse is rebuilt from the camera samples.
    public var mouse: [MouseInput]

    public init(payload: Data) throws {
        var r = ElkbornReader(payload)
        guard try r.u32() == ElkbornProtocol.payloadMagic else { throw ElkbornReader.Failure.bad("magic") }
        guard try r.u8() == ElkbornProtocol.version else { throw ElkbornReader.Failure.bad("version") }
        session = try r.u8()
        let flags = try r.u8()
        _ = try r.u8()
        finalTime = try r.f64()
        sampleHz = Int(try r.u8())
        runID = try r.str(); board = try r.str(); runner = try r.str(); avatar = try r.str()
        avatarUserID = try r.f64()
        var keys: [UInt8: String] = [:]
        for _ in 0 ..< Int(try r.u8()) {
            let code = try r.u8()
            keys[code] = try r.str()
        }
        self.keys = keys
        yawPerPixel = try r.f64()
        pitchPerPixel = try r.f64()
        let frames = try r.varu()
        guard frames < 1_000_000 else { throw ElkbornReader.Failure.bad("frames") }
        var samples: [Sample] = []
        samples.reserveCapacity(frames)
        var t = 0, pitch = 0, yaw = 0
        for _ in 0 ..< frames {
            let head = try r.varu()
            t += head >> 1
            pitch = (pitch + (try r.vars())) & 0xFFFF
            yaw = (yaw + (try r.vars())) & 0xFFFF
            samples.append(Sample(timeMs: t, pitch: pitch, yaw: yaw, forced: head & 1 == 1))
        }
        self.samples = samples
        let count = try r.varu()
        guard count < 1_000_000 else { throw ElkbornReader.Failure.bad("events") }
        var events: [Event] = []
        var et = 0
        for _ in 0 ..< count {
            et += try r.varu()
            let code = try r.u8()
            events.append(Event(timeMs: et, code: code & 0x7F, down: code & 0x80 != 0))
        }
        self.events = events
        // flags bit 2: the runner's mouse input, frame by frame (micro-radians of camera turn)
        var mouse: [MouseInput] = []
        if flags & 2 != 0 {
            let count = try r.varu()
            guard count < 1_000_000 else { throw ElkbornReader.Failure.bad("mouse") }
            mouse.reserveCapacity(count)
            var mt = 0
            for _ in 0 ..< count {
                mt += try r.varu()
                let yaw = Double(try r.vars()) / 1_000_000, pitch = Double(try r.vars()) / 1_000_000
                mouse.append(MouseInput(timeMs: mt, yaw: yaw, pitch: pitch))
            }
        }
        self.mouse = mouse
    }
}

public struct ElkbornCalibrationInfo: Sendable, Equatable {
    public var session: UInt8
    public var yawPerPixel: Double
    public var pitchPerPixel: Double

    public init(payload: Data) throws {
        var r = ElkbornReader(payload)
        guard try r.u32() == ElkbornProtocol.payloadMagic else { throw ElkbornReader.Failure.bad("magic") }
        _ = try r.u8()
        session = try r.u8()
        guard try r.u8() == ElkbornProtocol.Kind.calibration.rawValue else { throw ElkbornReader.Failure.bad("kind") }
        _ = try r.u8()
        yawPerPixel = try r.f64()
        pitchPerPixel = try r.f64()
    }
}

// MARK: Inputs

public enum ElkbornInputs {
    /// Roblox KeyCode / UserInputType names -> AgentTrainer key ids (`KeyCatalog`) or mouse buttons.
    public static func map(_ roblox: String) -> (key: String?, button: Int?) {
        switch roblox {
        case "MouseButton1": return (nil, 0)
        case "MouseButton2": return (nil, 1)
        case "MouseButton3": return (nil, 2)
        default: break
        }
        if roblox.count == 1, let c = roblox.first, c.isLetter { return (roblox.lowercased(), nil) }
        let named: [String: String] = [
            "Zero": "0", "One": "1", "Two": "2", "Three": "3", "Four": "4", "Five": "5", "Six": "6", "Seven": "7",
            "Eight": "8", "Nine": "9", "Space": "space", "LeftShift": "shift", "RightShift": "rshift",
            "LeftControl": "control", "RightControl": "rcontrol", "LeftAlt": "option", "RightAlt": "roption",
            "LeftMeta": "command", "LeftSuper": "command", "RightMeta": "rcommand", "RightSuper": "rcommand",
            "Tab": "tab", "Return": "return", "Escape": "escape", "Backspace": "delete", "Delete": "forwarddelete",
            "CapsLock": "capslock", "Up": "up", "Down": "down", "Left": "left", "Right": "right",
            "Comma": "comma", "Period": "period", "Slash": "slash", "Semicolon": "semicolon", "Quote": "quote",
            "LeftBracket": "lbracket", "RightBracket": "rbracket", "BackSlash": "backslash", "Minus": "minus",
            "Equals": "equal", "Backquote": "grave", "Home": "home", "End": "end", "PageUp": "pageup", "PageDown": "pagedown",
        ]
        if let id = named[roblox] { return (id, nil) }
        if roblox.hasPrefix("F"), let n = Int(roblox.dropFirst()), (1 ... 12).contains(n) { return ("f\(n)", nil) }
        return (nil, nil)
    }

    private static let modifiers: Set<String> = ["shift", "rshift", "control", "rcontrol", "option", "roption", "command", "rcommand", "capslock"]

    /// The run as AgentTrainer input events, on replay time (0 = the run's start).
    /// - `unitsPerPixel`: injected mouse units per Roblox mouse pixel (from calibration; 1 when unknown).
    /// - `pointer`: where the (locked) pointer sits, in global points.
    public static func events(for t: ElkbornTimeline, unitsPerPixel: Double, pointer: CGPoint) -> [InputEventRecord] {
        var out: [InputEventRecord] = []
        let px = Float(pointer.x), py = Float(pointer.y)
        for e in t.events {
            guard let name = t.keys[e.code], !name.isEmpty else { continue }
            let ns = Int64(e.timeMs) * 1_000_000
            let (key, button) = map(name)
            if let button {
                out.append(InputEventRecord(timeNs: ns, kind: e.down ? .buttonDown : .buttonUp, code: UInt16(button), x: px, y: py))
            } else if let key, let info = KeyCatalog.info(key) {
                if modifiers.contains(key) {
                    out.append(InputEventRecord(timeNs: ns, kind: .flagsChanged, flags: e.down ? 2 : 0, code: info.keyCode, x: px, y: py))
                } else {
                    out.append(InputEventRecord(timeNs: ns, kind: e.down ? .keyDown : .keyUp, code: info.keyCode, x: px, y: py))
                }
            }
        }
        // Mouse: each camera step becomes the movement that produces it on the viewer's settings.
        // Runs are sampled at 30 Hz (60 Hz before version 2), so a step longer than a 60 Hz frame
        // is spread evenly over it, the way the replay itself turns the camera between samples.
        let turn = 2 * Double.pi / 65536
        let scale = max(unitsPerPixel, 1e-6)
        let frameMs = 1000.0 / 60
        func wrap(_ v: Int) -> Int { var d = v & 0xFFFF; if d >= 0x8000 { d -= 0x10000 }; return d }
        if !t.mouse.isEmpty, t.yawPerPixel != 0, t.pitchPerPixel != 0 {
            // Version 3: the runner's own mouse input, frame by frame, as the turn it commanded.
            // Turns the camera made for other reasons (look-ats, rides, recoil) are not in it.
            for m in t.mouse where m.yaw != 0 || m.pitch != 0 {
                let dx = -m.yaw / t.yawPerPixel * scale
                let dy = -m.pitch / t.pitchPerPixel * scale
                out.append(InputEventRecord(timeNs: Int64(m.timeMs) * 1_000_000, kind: .mouseMove, x: px, y: py, dx: Float(dx), dy: Float(dy)))
            }
        } else if t.samples.count > 1, t.yawPerPixel != 0, t.pitchPerPixel != 0 {
            for i in 1 ..< t.samples.count {
                let a = t.samples[i - 1], b = t.samples[i]
                if b.forced { continue }
                let dYaw = Double(wrap(b.yaw - a.yaw)) * turn
                let dPitch = Double(wrap(b.pitch - a.pitch)) * turn
                if dYaw == 0 && dPitch == 0 { continue }
                let dx = -dYaw / t.yawPerPixel * scale
                let dy = -dPitch / t.pitchPerPixel * scale
                let span = Double(b.timeMs - a.timeMs)
                let parts = max(1, Int((span / frameMs).rounded()))
                for k in 1 ... parts {
                    let ms = Double(a.timeMs) + span * Double(k) / Double(parts)
                    out.append(InputEventRecord(timeNs: Int64((ms * 1_000_000).rounded()), kind: .mouseMove, x: px, y: py,
                                                dx: Float(dx / Double(parts)), dy: Float(dy / Double(parts))))
                }
            }
        }
        out.sort { $0.timeNs < $1.timeNs }
        return out
    }
}

/// The corner Elkborn replays draw their timecode strip in, as a normalised mask for a window of
/// the given size in points (Roblox GUI units are points). Slightly larger than the strip, like
/// the mask the replay recorder applies.
public enum ElkbornCorner {
    public static func mask(window: CGRect) -> CGRect {
        let cell = 8.0
        let w = (ElkbornProtocol.stripCells.width + 1) * cell, h = (ElkbornProtocol.stripCells.height + 0.5) * cell
        guard window.width > 0, window.height > 0 else { return .zero }
        return CGRect(x: 0, y: 1 - h / window.height, width: min(1, w / window.width), height: min(1, h / window.height))
    }
}
