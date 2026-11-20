@testable import ATCapture
@testable import ATCore
import AVFoundation
import CoreVideo
import XCTest

/// A run where the AI plays, the person corrects twice: each correction is saved with the 10 s
/// before it marked as the AI's (history), the person's input as the correction, across segments.
final class CorrectionRecorderTests: XCTestCase {
    let width = 320, height = 200

    func frame(_ shade: UInt8) -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferIOSurfacePropertiesKey: [:], kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary, &pb)
        CVPixelBufferLockBaseAddress(pb!, [])
        memset(CVPixelBufferGetBaseAddress(pb!), Int32(shade), CVPixelBufferGetBytesPerRow(pb!) * height)
        CVPixelBufferUnlockBaseAddress(pb!, [])
        return pb!
    }

    func testCorrectionsKeepTheirHistory() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("corrections-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        var cfg = CorrectionRecorder.Config(targetInfo: CaptureTargetInfo(kind: .window, appName: "Game", title: "Game"),
                                            targetFrame: CGRect(x: 0, y: 0, width: 640, height: 400),
                                            settings: CaptureSettings(width: width, height: height, fps: 30, showsCursor: false),
                                            destination: root.appendingPathComponent("Corrections"), scratch: root.appendingPathComponent("scratch"),
                                            appVersion: "test", modelName: "Tester")
        cfg.segmentSeconds = 4
        let recorder = try CorrectionRecorder(config: cfg)
        let saved = expectation(description: "saved")
        saved.expectedFulfillmentCount = 2
        var urls: [URL] = []
        let lock = NSLock()
        recorder.onSaved = { url in lock.withLock { urls.append(url) }; saved.fulfill() }
        recorder.onError = { XCTFail($0) }
        try recorder.startOffline()

        let ms: UInt64 = 1_000_000, t0: UInt64 = 1_000_000_000_000
        func host(_ seconds: Double) -> UInt64 { t0 + UInt64(seconds * 1000) * ms }
        // 30 s of play at 30 fps; the AI holds W (synthetic), the person presses D while correcting
        recorder.ingest(InputEventRecord(timeNs: Int64(host(0.5)), kind: .keyDown, flags: 1, code: KeyCatalog.info("w")!.keyCode))
        for i in 0 ..< 900 {
            let t = Double(i) / 30
            recorder.process(frame(UInt8(i % 250)), hostNs: host(t))
            switch i {
            case 390: recorder.begin(at: host(t)) // 13 s
            case 396: recorder.ingest(InputEventRecord(timeNs: Int64(host(t)), kind: .keyDown, code: KeyCatalog.info("d")!.keyCode))
            case 474: recorder.ingest(InputEventRecord(timeNs: Int64(host(t)), kind: .keyUp, code: KeyCatalog.info("d")!.keyCode))
            case 480: recorder.end(at: host(t)) // 16 s
            case 600: recorder.begin(at: host(t)) // 20 s, soon after the first: its history overlaps it
            case 660: recorder.end(at: host(t)) // 22 s
            default: break
            }
        }
        await fulfillment(of: [saved], timeout: 60)
        await recorder.stop()
        XCTAssertFalse(FileManager.default.fileExists(atPath: cfg.scratch.path), "the rolling segments are removed")

        let sorted = urls.sorted { $0.lastPathComponent < $1.lastPathComponent }
        // first correction: 10 s of history + 3 s of correction
        let first = try XCTUnwrap(sorted.first { (try? RecordingIO.readMeta($0))?.duration ?? 0 > 12 })
        let meta = try RecordingIO.readMeta(first)
        XCTAssertTrue(meta.isCorrectionRun)
        XCTAssertEqual(meta.modelName, "Tester")
        XCTAssertEqual(meta.tags, ["correction"])
        XCTAssertEqual(meta.duration, 13, accuracy: 0.05)
        let ann = RecordingIO.readAnnotations(first)
        XCTAssertEqual(ann.modelControlled.count, 1)
        XCTAssertEqual(ann.modelControlled[0].end, 10, accuracy: 0.05, "the 10 s before the correction are the AI's history")
        let frames = try RecordingIO.readFrames(first)
        XCTAssertEqual(frames.count, 391, accuracy: 2)
        XCTAssertEqual(frames.first?.timeNs, 0)
        let proxy = try ProxyReader(url: first.appendingPathComponent(RecordingPackage.proxy(size: 192)))
        XCTAssertEqual(proxy.frameCount, frames.count, "the proxy spans the segments without gaps")
        // the history is the AI's (W, synthetic), the correction is the person's (D)
        let steps = TimelineBuilder.build(frames: frames, events: try RecordingIO.readEvents(first), meta: meta, annotations: ann, rateHz: 20)
        XCTAssertTrue(steps.filter(\.modelControlled).allSatisfy { $0.action.keys.contains("w") })
        XCTAssertTrue(steps.filter { !$0.modelControlled && $0.time > 10.4 && $0.time < 12.6 }.allSatisfy { $0.action.keys.contains("d") })
        let video = AVURLAsset(url: first.appendingPathComponent(RecordingPackage.video))
        let duration = try await video.load(.duration).seconds
        XCTAssertEqual(duration, 13, accuracy: 0.2, "the video is put together from the segments")
        // second correction: its own full 10 s of history (overlapping the first one), then 2 s
        let second = try XCTUnwrap(sorted.first { $0 != first })
        let m2 = try RecordingIO.readMeta(second)
        XCTAssertEqual(m2.duration, 12, accuracy: 0.05)
        XCTAssertEqual(RecordingIO.readAnnotations(second).modelControlled.first?.end ?? 0, 10, accuracy: 0.05)
    }
}
