@testable import ATCapture
@testable import ATCore
import AVFoundation
import CoreVideo
import Foundation
import XCTest

/// Feeds a whole replay (ready page, timeline pages, playback strip frames, end page) drawn the
/// way the game draws it through the recorder, without a live capture or any real key presses,
/// and checks the recording it writes.
final class ElkbornReplayRecorderTests: XCTestCase {
    let width = 1152, height = 720
    let scale = 1152.0 / 1512.0 // a 1512x945 point window captured at 720 lines

    func makeBuffer() -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferIOSurfacePropertiesKey: [:], kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary, &pb)
        return pb!
    }

    /// Draws cells into a pixel buffer: `background` grey level, then (x, y, w, h, white) rects.
    func render(_ background: UInt8, _ rects: [(Double, Double, Double, Double, Bool)]) -> CVPixelBuffer {
        let pb = makeBuffer()
        CVPixelBufferLockBaseAddress(pb, [])
        let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
        let bpr = CVPixelBufferGetBytesPerRow(pb)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let p = base + y * bpr + x * 4
                let v = UInt8(truncatingIfNeeded: Int(background) + (x / 37 + y / 23) % 3 * 20)
                p[0] = v; p[1] = v; p[2] = v; p[3] = 255
            }
        }
        for (x0, y0, w, h, white) in rects {
            let v: UInt8 = white ? 255 : 0
            for y in max(0, Int(y0.rounded())) ..< min(height, Int((y0 + h).rounded())) {
                for x in max(0, Int(x0.rounded())) ..< min(width, Int((x0 + w).rounded())) {
                    let p = base + y * bpr + x * 4
                    p[0] = v; p[1] = v; p[2] = v
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(pb, [])
        return pb
    }

    func bits(_ v: UInt64, _ n: Int) -> [Bool] { (0 ..< n).map { v >> UInt64(n - 1 - $0) & 1 == 1 } }

    // Game page geometry for a 1512x945 point window.
    var grid: (cols: Int, rows: Int) {
        var cols = Int(1512.0 / 16) - 2, rows = Int(945.0 / 16) - 1 - 5
        if cols % 2 == 0 { cols -= 1 }
        if rows % 2 == 0 { rows -= 1 }
        return (cols, rows)
    }
    var pageCapacity: Int { (grid.cols - 1) * (grid.rows - 1) / 8 - 16 }

    func page(session: UInt8, kind: UInt8, index: Int, count: Int, slice: [UInt8], total: Int) -> CVPixelBuffer {
        var bytes: [UInt8] = [0xE7, 0xB1, session, kind, UInt8(index), UInt8(count), UInt8(slice.count >> 8), UInt8(slice.count & 0xFF),
                              UInt8(total >> 24 & 0xFF), UInt8(total >> 16 & 0xFF), UInt8(total >> 8 & 0xFF), UInt8(total & 0xFF), 1, 0] + slice
        let crc = ElkbornProtocol.crc16(bytes)
        bytes += [UInt8(crc >> 8), UInt8(crc & 0xFF)]
        var b: [Bool] = []
        for x in bytes { b += bits(UInt64(x), 8) }
        let S = 16 * scale, ox = S, oy = 5 * S
        var rects: [(Double, Double, Double, Double, Bool)] = []
        var i = 0
        for r in 0 ..< grid.rows {
            for c in 0 ..< grid.cols {
                let white: Bool
                if r == 0 { white = c % 2 == 0 } else if c == 0 { white = r % 2 == 0 } else { white = i < b.count ? b[i] : false; i += 1 }
                if white { rects.append((ox + Double(c) * S, oy + Double(r) * S, S, S, true)) }
            }
        }
        return render(0, rects)
    }

    func strip(session: UInt8, frame: UInt16, timeMs: UInt32, flags: UInt8) -> CVPixelBuffer {
        var b = bits(0xB, 4) + bits(UInt64(session), 8) + bits(UInt64(frame), 16) + bits(UInt64(timeMs), 24) + bits(UInt64(flags), 4)
        let bytes = ElkbornProtocol.bytes(fromBits: b + [Bool](repeating: false, count: 8))
        b += bits(UInt64(ElkbornProtocol.crc8(Array(bytes[0 ..< 7]))), 8)
        let S = 8 * scale, top = Double(height) - 5 * S
        var rects: [(Double, Double, Double, Double, Bool)] = [(0, top, 34 * S, 5 * S, false)]
        for c in 0 ..< 32 {
            rects.append((Double(c + 1) * S, top + S, S, S, c % 2 == 0))
            rects.append((Double(c + 1) * S, top + 2 * S, S, S, b[c]))
            rects.append((Double(c + 1) * S, top + 3 * S, S, S, b[32 + c]))
        }
        return render(110, rects)
    }

    /// The game's timeline payload (Trainer.Protocol.timeline).
    func timeline(session: UInt8) -> [UInt8] {
        var d: [UInt8] = []
        func u8(_ v: Int) { d.append(UInt8(v)) }
        func u32(_ v: UInt32) { for i in 0 ..< 4 { d.append(UInt8(v >> (8 * UInt32(i)) & 0xFF)) } }
        func f64(_ v: Double) { let b = v.bitPattern; for i in 0 ..< 8 { d.append(UInt8(b >> (8 * UInt64(i)) & 0xFF)) } }
        func str(_ s: String) { let u = Array(s.utf8); u8(u.count); d += u }
        func varu(_ v: Int) { var v = v; while v >= 128 { d.append(UInt8(v % 128 + 128)); v /= 128 }; d.append(UInt8(v)) }
        func vars(_ v: Int) { varu(v >= 0 ? v * 2 : -v * 2 - 1) }
        u32(ElkbornProtocol.payloadMagic); u8(1); u8(Int(session)); u8(1); u8(0)
        f64(2.0); u8(60); str("run1"); str("Ascension"); str("xavier8"); str("someone"); f64(42)
        let keys: [(Int, String)] = [(1, "W"), (7, "Space"), (9, "MouseButton3")]
        u8(keys.count); for (c, k) in keys { u8(c); str(k) }
        f64(0.5 * .pi / 180); f64(0.5 * 0.77 * .pi / 180)
        // 121 samples over 2 s, yaw turning steadily (a right turn: yaw decreases)
        varu(121)
        for i in 0 ..< 121 { varu((i == 0 ? 0 : 17) * 2); vars(0); vars(i == 0 ? 0 : -10) }
        let events: [(Int, Int)] = [(0, 1 + 128), (500, 7 + 128), (600, 7), (900, 9 + 128), (950, 9), (2000, 1)]
        varu(events.count)
        var t = 0
        for e in events { varu(e.0 - t); u8(e.1); t = e.0 }
        return d
    }

    func testPlaylistBecomesRecordings() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("elkborn-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        let cfg = ElkbornReplayRecorder.Config(targetInfo: CaptureTargetInfo(kind: .window, appName: "Roblox", title: "Roblox"),
                                               targetFrame: CGRect(x: 0, y: 0, width: 1512, height: 945),
                                               settings: CaptureSettings(width: width, height: height, fps: 30, showsCursor: false),
                                               destination: folder, appVersion: "test")
        let recorder = try ElkbornReplayRecorder(config: cfg)
        var keys: [UInt16] = []
        recorder.pressKey = { keys.append($0) }
        var savedURLs: [URL] = []
        let lock = NSLock()
        recorder.onError = { XCTFail("recorder error: \($0)") }
        recorder.startOffline()

        var host: UInt64 = 1_000_000_000
        func feed(_ pb: CVPixelBuffer) { recorder.process(pb, hostNs: host); host += 33_000_000 }
        feed(render(60, []))

        // a playlist: two replays one after the other, each with its own session
        for session: UInt8 in [9, 10] {
            let saved = expectation(description: "saved \(session)")
            recorder.onSaved = { url in lock.withLock { savedURLs.append(url) }; saved.fulfill() }
            let pressed = keys.count

            // the game keeps the ready page up until F8 arrives; idle frames are only searched now and then
            let beacon: [UInt8] = [0x45, 0x54, 0x50, 0x31, 1, session, 1, 0, 0]
            let ready = page(session: session, kind: 1, index: 0, count: 1, slice: beacon, total: beacon.count)
            for _ in 0 ..< 6 { feed(ready) }
            XCTAssertEqual(keys.count, pressed + 1, "the ready page is answered with F8, once")

            let payload = timeline(session: session)
            let cap = pageCapacity
            let count = (payload.count + cap - 1) / cap
            for loop in 0 ..< 2 {
                for i in 0 ..< count {
                    let slice = Array(payload[(i * cap) ..< min(payload.count, (i + 1) * cap)])
                    feed(page(session: session, kind: 2, index: i, count: count, slice: slice, total: payload.count))
                    if loop == 0, i == count - 1 { XCTAssertEqual(keys.count, pressed + 2, "the complete timeline is confirmed with F8") }
                }
            }
            XCTAssertEqual(keys.count, pressed + 2, "pages still on screen right after the confirmation are not answered again")
            // a second of lead-in (not recorded), then 2 s of playback at 30 fps, one frame captured twice
            for f in 0 ..< 30 { feed(strip(session: session, frame: UInt16(f), timeMs: 0, flags: 0)) }
            var frameNo: UInt16 = 30
            // the second replay's first frame arrives 11 ms after the run's zero, like a real capture can
            let offset = session == 10 ? 11 : 0
            for i in 0 ... 60 {
                let pb = strip(session: session, frame: frameNo, timeMs: UInt32(i * 33 + offset), flags: 1)
                feed(pb)
                if i == 10 { feed(pb) }
                frameNo += 1
            }
            let end = page(session: session, kind: 3, index: 0, count: 1, slice: [0x45, 0x54, 0x50, 0x31, 1, session, 3, 0], total: 8)
            for _ in 0 ..< 45 { feed(end) } // held 1.5 s
            wait(for: [saved], timeout: 30)
        }
        XCTAssertTrue(keys.allSatisfy { $0 == ElkbornProtocol.goKeyCode })

        XCTAssertEqual(savedURLs.count, 2)
        XCTAssertEqual(Set(savedURLs).count, 2, "each replay gets its own recording")
        for (n, url) in savedURLs.enumerated() { try checkRecording(url, offsetMs: n == 1 ? 11 : 0) }
    }

    func checkRecording(_ url: URL, offsetMs: Int) throws {
        let meta = try RecordingIO.readMeta(url)
        XCTAssertEqual(meta.frameCount, 61)
        XCTAssertEqual(meta.tags, ["elkborn", "Ascension"])
        XCTAssertEqual(meta.eventClock, "elkborn-replay")
        XCTAssertEqual(meta.duration, 1.98, accuracy: 0.001)
        XCTAssertGreaterThan(meta.mouseDeltaStd?.first ?? 0, 0, "the training timeline sees the camera turning")
        let mask = try XCTUnwrap(meta.masks?.first)
        XCTAssertEqual(mask.x, 0)
        XCTAssertEqual(mask.y + mask.height, 1, accuracy: 0.01, "the mask reaches the bottom edge")
        XCTAssertGreaterThan(mask.width * 1512, 34 * 8, "the mask covers the strip")
        let frames = try RecordingIO.readFrames(url)
        XCTAssertEqual(frames.map(\.timeNs), (0 ... 60).map { Int64($0 * 33 + offsetMs) * 1_000_000 }, "frames are on replay time")
        // the video shows a picture from its very first instant (no black start)
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url.appendingPathComponent(RecordingPackage.video)))
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        let first = try generator.copyCGImage(at: .zero, actualTime: nil)
        XCTAssertGreaterThan(first.width, 0)
        let events = try RecordingIO.readEvents(url)
        let keyEvents = events.filter { $0.kind == .keyDown || $0.kind == .keyUp }
        let w = try XCTUnwrap(KeyCatalog.info("w")).keyCode, space = try XCTUnwrap(KeyCatalog.info("space")).keyCode
        XCTAssertEqual(keyEvents.map(\.code), [w, space, space, w])
        XCTAssertEqual(keyEvents.map(\.timeNs), [0, 500, 600, 2000].map { Int64($0) * 1_000_000 })
        XCTAssertEqual(events.filter { $0.kind == .buttonDown }.map(\.code), [2])
        let moves = events.filter { $0.kind == .mouseMove }
        XCTAssertEqual(moves.count, 120)
        // yaw fell by 10/65536 of a turn per sample (a right turn) with 0.5 degrees per Roblox pixel
        XCTAssertEqual(Double(moves[0].dx), 10.0 / 65536 * 360 / 0.5, accuracy: 1e-3)
        XCTAssertEqual(moves[0].dy, 0)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.appendingPathComponent(RecordingPackage.video).path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.appendingPathComponent(RecordingPackage.thumbnail).path))
        let proxy = try ProxyReader(url: url.appendingPathComponent(RecordingPackage.proxy(size: 192)))
        XCTAssertEqual(proxy.frameCount, 61)
        let image = try proxy.frame(30)
        let ch = proxy.channels
        XCTAssertEqual(image[(190 * 192 + 2) * ch], 0, "the timecode corner is blacked out")
        XCTAssertGreaterThan(image[(100 * 192 + 100) * ch], 60, "the rest of the frame is untouched")
    }
}


// MARK: Calibration

extension ElkbornReplayRecorderTests {
    /// A strip frame drawn over a fixed background (copied, not recomputed, so a test can show
    /// the game's live readout at a frame rate close to the real one).
    func fastStrip(_ background: [UInt8], session: UInt8, frame: UInt16, flags: UInt8) -> CVPixelBuffer {
        let pb = makeBuffer()
        CVPixelBufferLockBaseAddress(pb, [])
        let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
        let bpr = CVPixelBufferGetBytesPerRow(pb)
        background.withUnsafeBytes { src in
            for y in 0 ..< height { memcpy(base + y * bpr, src.baseAddress! + y * width * 4, width * 4) }
        }
        var b = bits(0xB, 4) + bits(UInt64(session), 8) + bits(UInt64(frame), 16) + bits(0, 24) + bits(UInt64(flags), 4)
        let bytes = ElkbornProtocol.bytes(fromBits: b + [Bool](repeating: false, count: 8))
        b += bits(UInt64(ElkbornProtocol.crc8(Array(bytes[0 ..< 7]))), 8)
        let S = 8 * scale, top = Double(height) - 5 * S
        func fill(_ x0: Double, _ y0: Double, _ w: Double, _ h: Double, _ white: Bool) {
            let v: UInt8 = white ? 255 : 0
            for y in max(0, Int(y0.rounded())) ..< min(height, Int((y0 + h).rounded())) {
                for x in max(0, Int(x0.rounded())) ..< min(width, Int((x0 + w).rounded())) {
                    let p = base + y * bpr + x * 4
                    p[0] = v; p[1] = v; p[2] = v
                }
            }
        }
        fill(0, top, 34 * S, 5 * S, false)
        for c in 0 ..< 32 {
            fill(Double(c + 1) * S, top + S, S, S, c % 2 == 0)
            fill(Double(c + 1) * S, top + 2 * S, S, S, b[c])
            fill(Double(c + 1) * S, top + 3 * S, S, S, b[32 + c])
        }
        CVPixelBufferUnlockBaseAddress(pb, [])
        return pb
    }

    /// Roblox re-centres a pointer it finds off-centre on the first movement after the camera
    /// locks, which turns the camera by a random amount. Calibration has to take that jump before
    /// it measures, and still find the true mouse scale.
    func testCalibrationIgnoresTheRecentringSnap() throws {
        let frame = CGRect(x: 100, y: 50, width: 1512, height: 945)
        let cfg = ElkbornReplayRecorder.Config(targetInfo: CaptureTargetInfo(kind: .window, appName: "Roblox", title: "Roblox"),
                                               targetFrame: frame,
                                               settings: CaptureSettings(width: width, height: height, fps: 60, showsCursor: false),
                                               destination: FileManager.default.temporaryDirectory, appVersion: "test")
        let recorder = try ElkbornReplayRecorder(config: cfg)
        let truth = 2.0 // injected mouse units per Roblox pixel
        let yawPerPixel = 0.5 * .pi / 180
        let state = NSLock()
        var yaw = 20000.0 // the game's camera yaw in 1/65536 turns
        var snapped = false
        var placed: CGPoint?
        var keys: [UInt16] = []
        var result: Double?
        var failure: String?
        recorder.pressKey = { code in state.withLock { keys.append(code) } }
        recorder.placePointer = { p in state.withLock { placed = p } }
        recorder.moveMouse = { dx in
            state.withLock {
                if !snapped { snapped = true; yaw += 7000 }
                yaw -= Double(dx) / truth * yawPerPixel / (2 * .pi) * 65536
            }
        }
        recorder.onCalibrated = { units, _ in state.withLock { result = units } }
        recorder.onError = { text in state.withLock { failure = text } }
        recorder.startOffline(calibrate: true)

        let session: UInt8 = 33
        var payload: [UInt8] = [0x45, 0x54, 0x50, 0x31, 1, session, 4, 0]
        for v in [yawPerPixel, yawPerPixel * 0.77] { let b = v.bitPattern; for i in 0 ..< 8 { payload.append(UInt8(b >> (8 * UInt64(i)) & 0xFF)) } }
        var host: UInt64 = 1_000_000_000
        let calibrationPage = page(session: session, kind: 4, index: 0, count: 1, slice: payload, total: payload.count)
        for _ in 0 ..< 6 { recorder.process(calibrationPage, hostNs: host); host += 16_000_000 }
        XCTAssertEqual(state.withLock { keys }, [ElkbornProtocol.goKeyCode], "the calibration page is answered with F8")

        // the game shows the live camera yaw in the strip every frame until F8 ends it
        var background = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let v = UInt8(110 + (x / 37 + y / 23) % 3 * 20), i = (y * width + x) * 4
                background[i] = v; background[i + 1] = v; background[i + 2] = v; background[i + 3] = 255
            }
        }
        let deadline = Date().addingTimeInterval(40)
        while Date() < deadline {
            let (current, finished) = state.withLock { (yaw, result != nil || failure != nil) }
            if finished { break }
            let shown = UInt16(truncatingIfNeeded: Int(current.rounded()) & 0xFFFF)
            recorder.process(fastStrip(background, session: session, frame: shown, flags: 4), hostNs: host)
            host += 16_000_000
            Thread.sleep(forTimeInterval: 0.008)
        }

        let (units, error, point, pressed) = state.withLock { (result, failure, placed, keys) }
        XCTAssertNil(error)
        XCTAssertEqual(try XCTUnwrap(units), truth, accuracy: truth * 0.02)
        XCTAssertEqual(point, CGPoint(x: frame.midX, y: frame.midY), "the pointer is centred on the window first")
        XCTAssertEqual(pressed, [ElkbornProtocol.goKeyCode, ElkbornProtocol.goKeyCode], "F8 again ends the game's calibration screen")
    }
}
