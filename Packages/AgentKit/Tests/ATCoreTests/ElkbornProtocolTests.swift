@testable import ATCore
import CoreGraphics
import Foundation
import ImageIO
import XCTest

/// Renders Elkborn's screen protocol the way the game draws it (Trainer.Protocol) at a
/// non-integer capture scale, then decodes it.
final class ElkbornProtocolTests: XCTestCase {
    /// A BGRA frame filled with noise, so the decoder has to find the cells itself.
    final class Canvas {
        let width: Int, height: Int
        var pixels: [UInt8]
        init(width: Int, height: Int, seed: UInt64 = 7) {
            self.width = width; self.height = height
            pixels = [UInt8](repeating: 0, count: width * height * 4)
            var s = seed
            for i in stride(from: 0, to: pixels.count, by: 4) {
                s = s &* 6364136223846793005 &+ 1442695040888963407
                let v = UInt8(truncatingIfNeeded: s >> 56)
                pixels[i] = v; pixels[i + 1] = v &+ 40; pixels[i + 2] = v &+ 80; pixels[i + 3] = 255
            }
        }
        func fill(_ x0: Double, _ y0: Double, _ x1: Double, _ y1: Double, white: Bool) {
            let v: UInt8 = white ? 255 : 0
            for y in max(0, Int(y0.rounded())) ..< min(height, Int(y1.rounded())) {
                for x in max(0, Int(x0.rounded())) ..< min(width, Int(x1.rounded())) {
                    let i = (y * width + x) * 4
                    pixels[i] = v; pixels[i + 1] = v; pixels[i + 2] = v
                }
            }
        }
        func decode<R>(_ body: (LumaFrame) -> R) -> R {
            pixels.withUnsafeBytes { body(LumaFrame(base: $0.baseAddress!, width: width, height: height, bytesPerRow: width * 4)) }
        }
    }

    static func bits(_ value: UInt64, _ count: Int) -> [Bool] { (0 ..< count).map { value >> UInt64(count - 1 - $0) & 1 == 1 } }

    /// Same layout as Protocol.showStrip.
    static func drawStrip(_ c: Canvas, scale s: Double, session: UInt8, frame: UInt16, value: UInt32, flags: UInt8) {
        var bits = Self.bits(0xB, 4) + Self.bits(UInt64(session), 8) + Self.bits(UInt64(frame), 16) + Self.bits(UInt64(value), 24) + Self.bits(UInt64(flags), 4)
        let bytes = ElkbornProtocol.bytes(fromBits: bits + [Bool](repeating: false, count: 8))
        bits += Self.bits(UInt64(ElkbornProtocol.crc8(Array(bytes[0 ..< 7]))), 8)
        let S = 8 * s
        let top = Double(c.height) - 5 * S
        c.fill(0, top, 34 * S, Double(c.height), white: false)
        for col in 0 ..< 32 {
            c.fill(Double(col + 1) * S, top + S, Double(col + 2) * S, top + 2 * S, white: col % 2 == 0)
            c.fill(Double(col + 1) * S, top + 2 * S, Double(col + 2) * S, top + 3 * S, white: bits[col])
            c.fill(Double(col + 1) * S, top + 3 * S, Double(col + 2) * S, top + 4 * S, white: bits[32 + col])
        }
    }

    /// Same layout as Protocol.buildPage / showPage / paginate.
    static func drawPage(_ c: Canvas, scale s: Double, logical: CGSize, session: UInt8, kind: UInt8, index: UInt8, count: UInt8, payload: [UInt8], total: Int) {
        let P = 16.0
        var cols = Int(logical.width / P) - 2, rows = Int(logical.height / P) - 1 - 5
        if cols % 2 == 0 { cols -= 1 }
        if rows % 2 == 0 { rows -= 1 }
        var bytes: [UInt8] = [0xE7, 0xB1, session, kind, index, count, UInt8(payload.count >> 8), UInt8(payload.count & 0xFF),
                              UInt8(total >> 24 & 0xFF), UInt8(total >> 16 & 0xFF), UInt8(total >> 8 & 0xFF), UInt8(total & 0xFF), 1, 0] + payload
        let crc = ElkbornProtocol.crc16(bytes)
        bytes += [UInt8(crc >> 8), UInt8(crc & 0xFF)]
        var bits: [Bool] = []
        for b in bytes { bits += Self.bits(UInt64(b), 8) }
        c.fill(0, 0, Double(c.width), Double(c.height), white: false)
        let ox = P * s, oy = 5 * P * s, S = P * s
        var i = 0
        for r in 0 ..< rows {
            for col in 0 ..< cols {
                let white: Bool
                if r == 0 { white = col % 2 == 0 } else if col == 0 { white = r % 2 == 0 } else {
                    white = i < bits.count ? bits[i] : false
                    i += 1
                }
                c.fill(ox + Double(col) * S, oy + Double(r) * S, ox + Double(col + 1) * S, oy + Double(r + 1) * S, white: white)
            }
        }
    }

    func testStripRoundTrip() {
        for scale in [1.0, 1.327, 2.0, 0.89] {
            let c = Canvas(width: Int(1301 * scale), height: Int(814 * scale))
            Self.drawStrip(c, scale: scale, session: 0xA7, frame: 0xBEEF, value: 123_456, flags: 1)
            let strip = c.decode { ElkbornProtocol.decodeStrip($0) }
            XCTAssertEqual(strip?.session, 0xA7, "scale \(scale)")
            XCTAssertEqual(strip?.frame, 0xBEEF)
            XCTAssertEqual(strip?.value, 123_456)
            XCTAssertEqual(strip?.flags, .playing)
            XCTAssertEqual(strip?.cell ?? 0, 8 * scale, accuracy: 0.6)
        }
    }

    func testPageRoundTrip() {
        let payload = (0 ..< 380).map { UInt8(truncatingIfNeeded: $0 &* 37 &+ 11) }
        for scale in [1.0, 1.327, 2.0, 0.89] {
            let logical = CGSize(width: 1301, height: 814)
            let c = Canvas(width: Int(logical.width * scale), height: Int(logical.height * scale))
            Self.drawPage(c, scale: scale, logical: logical, session: 9, kind: 2, index: 3, count: 7, payload: payload, total: 4000)
            let page = c.decode { ElkbornProtocol.decodePage($0) }
            XCTAssertEqual(page?.session, 9, "scale \(scale)")
            XCTAssertEqual(page?.kind, .data)
            XCTAssertEqual(page?.index, 3)
            XCTAssertEqual(page?.count, 7)
            XCTAssertEqual(page?.total, 4000)
            XCTAssertEqual(page.map { [UInt8]($0.payload) }, payload)
        }
    }

    func testCorruptedPageIsRejected() {
        let logical = CGSize(width: 1301, height: 814)
        let c = Canvas(width: 1301, height: 814)
        Self.drawPage(c, scale: 1, logical: logical, session: 1, kind: 2, index: 0, count: 1, payload: [1, 2, 3, 4], total: 4)
        // invert one data cell (row 2 of the grid holds the end of the header and the payload)
        let wasWhite = c.pixels[((16 * 7 + 8) * c.width + 16 * 9 + 8) * 4] == 255
        c.fill(16 * 9, 16 * 7, 16 * 10, 16 * 8, white: !wasWhite)
        XCTAssertNil(c.decode { ElkbornProtocol.decodePage($0) })
    }

    /// The game's payload writer (Codec.writer: little endian, LEB128, zigzag).
    struct Writer {
        var d: [UInt8] = []
        mutating func u8(_ v: Int) { d.append(UInt8(v)) }
        mutating func u32(_ v: UInt32) { for i in 0 ..< 4 { d.append(UInt8(v >> (8 * UInt32(i)) & 0xFF)) } }
        mutating func f64(_ v: Double) { let b = v.bitPattern; for i in 0 ..< 8 { d.append(UInt8(b >> (8 * UInt64(i)) & 0xFF)) } }
        mutating func str(_ s: String) { let u = Array(s.utf8); u8(u.count); d += u }
        mutating func varu(_ v: Int) { var v = v; while v >= 128 { d.append(UInt8(v % 128 + 128)); v /= 128 }; d.append(UInt8(v)) }
        mutating func vars(_ v: Int) { varu(v >= 0 ? v * 2 : -v * 2 - 1) }
    }

    func testTimelineAndInputs() throws {
        var w = Writer()
        w.u32(ElkbornProtocol.payloadMagic); w.u8(1); w.u8(42); w.u8(1); w.u8(0)
        w.f64(12.345); w.u8(60)
        w.str("abc123"); w.str("Ascension"); w.str("xavier8"); w.str("SomeAvatar"); w.f64(1234)
        let keys: [(Int, String)] = [(1, "W"), (7, "Space"), (8, "LeftShift"), (9, "MouseButton3"), (13, "E")]
        w.u8(keys.count)
        for (code, key) in keys { w.u8(code); w.str(key) }
        let yawPerPixel = 0.5 * .pi / 180, pitchPerPixel = -0.5 * 0.77 * .pi / 180
        w.f64(yawPerPixel); w.f64(pitchPerPixel)
        // samples: t, pitch, yaw (1/65536 turns), forced
        let samples: [(Int, Int, Int, Bool)] = [(0, 0, 65530, true), (16, 0, 65530, false), (33, 100, 2, false), (50, 100, 2, false)]
        w.varu(samples.count)
        var pt = 0, pp = 0, py = 0
        for s in samples {
            w.varu((s.0 - pt) * 2 + (s.3 ? 1 : 0))
            w.vars(((s.1 - pp + 32768) % 65536 + 65536) % 65536 - 32768)
            w.vars(((s.2 - py + 32768) % 65536 + 65536) % 65536 - 32768)
            pt = s.0; pp = s.1; py = s.2
        }
        let events: [(Int, Int)] = [(0, 1 + 128), (500, 7 + 128), (600, 7), (700, 8 + 128), (900, 8), (1000, 9 + 128), (1100, 9), (1200, 1)]
        w.varu(events.count)
        var et = 0
        for e in events { w.varu(e.0 - et); w.u8(e.1); et = e.0 }

        let t = try ElkbornTimeline(payload: Data(w.d))
        XCTAssertEqual(t.session, 42)
        XCTAssertEqual(t.finalTime, 12.345)
        XCTAssertEqual(t.board, "Ascension")
        XCTAssertEqual(t.runner, "xavier8")
        XCTAssertEqual(t.keys[7], "Space")
        XCTAssertEqual(t.samples.map(\.yaw), [65530, 65530, 2, 2])
        XCTAssertEqual(t.samples.map(\.forced), [true, false, false, false])
        XCTAssertEqual(t.events.count, 8)

        let records = ElkbornInputs.events(for: t, unitsPerPixel: 1, pointer: CGPoint(x: 100, y: 100))
        let keyDowns = records.filter { $0.kind == .keyDown }.map(\.code)
        XCTAssertEqual(keyDowns, [KeyCatalog.info("w")!.keyCode, KeyCatalog.info("space")!.keyCode])
        let shift = records.filter { $0.kind == .flagsChanged }
        XCTAssertEqual(shift.map(\.flags), [2, 0])
        XCTAssertEqual(records.filter { $0.kind == .buttonDown }.map(\.code), [2])
        // yaw grew by 8 steps (a left turn): the mouse moved left by 8/65536 turn / 0.5 deg = 0.088 px
        let moves = records.filter { $0.kind == .mouseMove }
        XCTAssertEqual(moves.count, 1)
        XCTAssertEqual(Double(moves[0].dx), -(8.0 * 2 * .pi / 65536) / yawPerPixel, accuracy: 1e-4)
        XCTAssertEqual(Double(moves[0].dy), -(100.0 * 2 * .pi / 65536) / pitchPerPixel, accuracy: 1e-4)
        XCTAssertEqual(moves[0].timeNs, 33_000_000)
    }

    /// Version 3 runs carry the runner's own mouse input: the mouse comes from that, frame by frame,
    /// and camera turns the game made on its own (here a 90 degree snap) are not turned into mouse.
    func testRecordedMouseInputIsUsedInsteadOfCameraAngles() throws {
        var w = Writer()
        w.u32(ElkbornProtocol.payloadMagic); w.u8(1); w.u8(3); w.u8(1 + 2); w.u8(0)
        w.f64(0.1); w.u8(30)
        w.str("r"); w.str("Board"); w.str("runner"); w.str("avatar"); w.f64(1)
        w.u8(0)
        let yawPerPixel = 0.5 * .pi / 180, pitchPerPixel = 0.5 * 0.77 * .pi / 180
        w.f64(yawPerPixel); w.f64(pitchPerPixel)
        // camera samples: a 90 degree snap the game made (not flagged), which must be ignored
        let samples: [(Int, Int)] = [(0, 0), (33, 16384), (67, 16384)]
        w.varu(samples.count)
        var pt = 0, py = 0
        for s in samples { w.varu((s.0 - pt) * 2); w.vars(0); w.vars(s.1 - py); pt = s.0; py = s.1 }
        w.varu(0) // no key events
        // mouse input: 5 ms right turn of 0.5 deg (one Roblox pixel), 22 ms a look down
        let mouse: [(Int, Int, Int)] = [(5, -8727, 0), (22, 0, -6720)]
        w.varu(mouse.count)
        var mt = 0
        for m in mouse { w.varu(m.0 - mt); w.vars(m.1); w.vars(m.2); mt = m.0 }
        let t = try ElkbornTimeline(payload: Data(w.d))
        XCTAssertEqual(t.mouse.count, 2)
        let moves = ElkbornInputs.events(for: t, unitsPerPixel: 1, pointer: .zero).filter { $0.kind == .mouseMove }
        XCTAssertEqual(moves.map(\.timeNs), [5_000_000, 22_000_000])
        XCTAssertEqual(Double(moves[0].dx), 8727e-6 / yawPerPixel, accuracy: 1e-3, "a right turn is mouse right")
        XCTAssertEqual(Double(moves[0].dy), 0, accuracy: 1e-9)
        XCTAssertEqual(Double(moves[1].dy), 6720e-6 / pitchPerPixel, accuracy: 1e-3, "looking down is mouse down")
    }

    /// Version 2 runs are sampled at 30 Hz: each camera step is spread over 60 Hz mouse moves.
    func testThirtyHertzCameraStepsBecomeSixtyHertzMoves() throws {
        var w = Writer()
        w.u32(ElkbornProtocol.payloadMagic); w.u8(1); w.u8(7); w.u8(1); w.u8(0)
        w.f64(0.1); w.u8(30)
        w.str("r"); w.str("Board"); w.str("runner"); w.str("avatar"); w.f64(1)
        w.u8(0)
        let yawPerPixel = 0.5 * .pi / 180
        w.f64(yawPerPixel); w.f64(-yawPerPixel)
        let samples: [(Int, Int)] = [(0, 0), (33, 40), (67, 80), (100, 80)]
        w.varu(samples.count)
        var pt = 0, py = 0
        for s in samples { w.varu((s.0 - pt) * 2); w.vars(0); w.vars(s.1 - py); pt = s.0; py = s.1 }
        w.varu(0)
        let t = try ElkbornTimeline(payload: Data(w.d))
        XCTAssertEqual(t.sampleHz, 30)
        let moves = ElkbornInputs.events(for: t, unitsPerPixel: 1, pointer: .zero).filter { $0.kind == .mouseMove }
        XCTAssertEqual(moves.map(\.timeNs), [16_500_000, 33_000_000, 50_000_000, 67_000_000])
        let perStep = -(40.0 * 2 * .pi / 65536) / yawPerPixel
        for m in moves { XCTAssertEqual(Double(m.dx), perStep / 2, accuracy: 1e-4) }
        XCTAssertEqual(moves.reduce(0) { $0 + Double($1.dx) }, perStep * 2, accuracy: 1e-3, "the total turn is unchanged")
    }

    /// A strip captured from the game in Roblox Studio (JPEG), when available on this machine.
    func testCapturedStrip() throws {
        let path = ProcessInfo.processInfo.environment["ELKBORN_STRIP_JPEG"] ?? ""
        guard !path.isEmpty, let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw XCTSkip("no captured strip") }
        let w = image.width, h = image.height
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        let ctx = CGContext(data: &pixels, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        let strip = pixels.withUnsafeBytes { ElkbornProtocol.decodeStrip(LumaFrame(base: $0.baseAddress!, width: w, height: h, bytesPerRow: w * 4)) }
        XCTAssertNotNil(strip)
        if let strip { print("captured strip:", strip.session, strip.frame, strip.value, strip.flags.rawValue, strip.cell) }
    }
}

extension ElkbornProtocolTests {
    /// The scanner finds the grid once, then reads every later page and strip at the same place.
    func testScannerReusesGeometry() {
        let logical = CGSize(width: 1512, height: 945)
        let scale = 0.762 // a 2x Retina window captured at 720 lines
        var scanner = ElkbornScanner()
        for index in 0 ..< 3 {
            let c = Canvas(width: Int(logical.width * scale), height: Int(logical.height * scale), seed: UInt64(index + 1))
            let payload = (0 ..< 200).map { UInt8(truncatingIfNeeded: $0 &+ index) }
            Self.drawPage(c, scale: scale, logical: logical, session: 5, kind: 2, index: UInt8(index), count: 3, payload: payload, total: 600)
            let page = c.decode { scanner.page($0) }
            XCTAssertEqual(page?.index, index)
            XCTAssertEqual(page.map { [UInt8]($0.payload) }, payload)
        }
        var assembler = ElkbornPageAssembler()
        XCTAssertNil(assembler.add(.init(session: 1, kind: .data, index: 1, count: 2, total: 4, payload: Data([3, 4]))))
        XCTAssertEqual(assembler.add(.init(session: 1, kind: .data, index: 0, count: 2, total: 4, payload: Data([1, 2]))), Data([1, 2, 3, 4]))
        for frame: UInt16 in [10, 11, 12] {
            let c = Canvas(width: 1152, height: 720, seed: UInt64(frame))
            Self.drawStrip(c, scale: 0.885, session: 77, frame: frame, value: UInt32(frame) * 16, flags: 1)
            let strip = c.decode { scanner.strip($0) }
            XCTAssertEqual(strip?.frame, frame)
            XCTAssertEqual(strip?.value, UInt32(frame) * 16)
        }
    }
}
