import Foundation
import Testing
@testable import ATCore

@Suite struct RecordingFormatTests {
    func meta() -> RecordingMeta {
        RecordingMeta(name: "t", appVersion: "test", target: CaptureTargetInfo(kind: .display), captureWidth: 640, captureHeight: 360,
                      sourceRect: CodableRect(x: 0, y: 0, width: 1000, height: 500), captureFPS: 60, showsCursor: true, input: InputCaptureOptions())
    }

    @Test func binaryRecordsRoundTrip() {
        let f = FrameRecord(index: 7, flags: 1, timeNs: 123_456_789, hostNs: 987_654_321_000)
        #expect(BinaryRecords.decodeFrames(BinaryRecords.encode(f)) == [f])
        let e = InputEventRecord(timeNs: -5, kind: .scroll, flags: 1, code: 3, x: 1.5, y: -2, dx: 0.25, dy: -3)
        #expect(BinaryRecords.decodeEvents(BinaryRecords.encode(e)) == [e])
    }

    @Test func proxyRoundTripsLosslessly() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("p-\(UUID()).atproxy")
        defer { try? FileManager.default.removeItem(at: url) }
        let w = ProxyWriter.self
        let writer = try w.init(url: url, width: 8, height: 4, chunkFrames: 3)
        var frames: [[UInt8]] = []
        for i in 0 ..< 7 {
            let f = (0 ..< 96).map { UInt8(truncatingIfNeeded: $0 * (i + 1) + i * 7) }
            frames.append(f)
            f.withUnsafeBytes { writer.append($0) }
        }
        try writer.finish()
        let r = try ProxyReader(url: url)
        #expect(r.frameCount == 7 && r.width == 8 && r.height == 4)
        for i in [6, 0, 3, 5, 1] { #expect([UInt8](try r.frame(i)) == frames[i], "frame \(i)") }
    }

    @Test func timelineAlignsInputToTheWindowAfterEachFrame() {
        let ms: Int64 = 1_000_000
        // 60 fps frames for 200 ms.
        let frames = (0 ..< 12).map { FrameRecord(index: UInt32($0), timeNs: Int64($0) * 16_667 * 1000, hostNs: 0) }
        let w = KeyCatalog.info("w")!.keyCode, a = KeyCatalog.info("a")!.keyCode
        let events = [
            InputEventRecord(timeNs: 10 * ms, kind: .keyDown, code: w),           // step 0 (0-50ms)
            InputEventRecord(timeNs: 60 * ms, kind: .keyDown, code: a),           // tap inside step 1
            InputEventRecord(timeNs: 70 * ms, kind: .keyUp, code: a),
            InputEventRecord(timeNs: 80 * ms, kind: .mouseMove, x: 500, y: 250, dx: 4, dy: -2),
            InputEventRecord(timeNs: 90 * ms, kind: .mouseMove, x: 510, y: 250, dx: 6, dy: 1),
            InputEventRecord(timeNs: 120 * ms, kind: .keyUp, code: w),            // step 2
            InputEventRecord(timeNs: 125 * ms, kind: .scroll, dx: 0, dy: 1),
            InputEventRecord(timeNs: 130 * ms, kind: .buttonDown, x: 250, y: 125, dx: 0, dy: 0),
        ]
        let steps = TimelineBuilder.build(frames: frames, events: events, meta: meta(), annotations: RecordingAnnotations(), rateHz: 20)
        #expect(steps.count == 3)
        #expect(steps[0].action.keys == ["w"])
        #expect(steps[1].action.keys == ["w", "a"]) // the short tap is kept
        #expect(steps[1].action.dx == 10 && steps[1].action.dy == -1)
        #expect(steps[2].action.keys.isEmpty == false) // pressed? no - w released, but w was held at window start
        #expect(steps[2].action.pointer.contains(.scrollUp) && steps[2].action.pointer.contains(.left))
        #expect(abs(steps[2].action.absX - 0.25) < 1e-9 && abs(steps[2].action.absY - 0.25) < 1e-9)
        // Frame for step k is the latest frame at or before k*50ms.
        #expect(steps[0].frameIndex == 0 && steps[1].frameIndex == 2 && steps[2].frameIndex == 5)
    }

    @Test func annotationsDriveMasksAndInstructions() {
        let frames = (0 ..< 40).map { FrameRecord(index: UInt32($0), timeNs: Int64($0) * 50_000_000, hostNs: 0) }
        var ann = RecordingAnnotations()
        ann.segments = [TextSegment(start: 0.5, end: 1.0, text: "open the door", source: .user)]
        ann.excluded = [TimeRange(start: 1.0, end: 1.2)]
        ann.modelControlled = [TimeRange(start: 1.5, end: 1.6)]
        let steps = TimelineBuilder.build(frames: frames, events: [], meta: meta(), annotations: ann, rateHz: 20)
        #expect(steps[10].instruction == "open the door" && steps[9].instruction == nil)
        #expect(steps[20].excluded && !steps[24].excluded)
        #expect(steps[30].modelControlled && !steps[29].modelControlled)
    }
}
