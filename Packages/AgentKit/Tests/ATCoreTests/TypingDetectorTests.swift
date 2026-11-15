import Testing
@testable import ATCore

@Suite struct TypingDetectorTests {
    @Test func detectsTypedWordsWithShift() {
        func key(_ id: String, _ t: Double, down: Bool = true) -> InputEventRecord {
            InputEventRecord(timeNs: Int64(t * 1e9), kind: down ? .keyDown : .keyUp, code: KeyCatalog.info(id)!.keyCode)
        }
        var ev: [InputEventRecord] = []
        ev.append(InputEventRecord(timeNs: Int64(0.9e9), kind: .flagsChanged, flags: 2, code: 0x38))
        ev.append(key("h", 1.0))
        ev.append(InputEventRecord(timeNs: Int64(1.05e9), kind: .flagsChanged, flags: 0, code: 0x38))
        for (i, c) in ["i", "space", "y", "o", "u"].enumerated() { ev.append(key(c, 1.2 + Double(i) * 0.15)) }
        ev.append(key("return", 2.2))
        ev.append(key("w", 6.0)) // single key later: not typing
        let segs = TypingDetector.segments(events: ev)
        #expect(segs.count == 1)
        #expect(segs.first?.text == "type \"Hi you\"")
        #expect(segs.first!.start < 1.0 && segs.first!.end > 1.8)
    }

    @Test func oldTrainingSettingsDecodeWithNewFields() throws {
        let json = #"{"learningRate":0.001,"totalSteps":10}"#.data(using: .utf8)!
        let h = try RecordingIO.decoder.decode(TrainingHyperparameters.self, from: json)
        #expect(h.learningRate == 0.001 && h.totalSteps == 10 && h.historyDropout == 0.5)
    }
}
