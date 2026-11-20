@testable import ATCore
@testable import ATLanguage
import XCTest

/// Auto Label's pure parts: answers, clips, input log, variations.
final class AutoLabelTests: XCTestCase {
    // MARK: Answers

    func testParsesAnswerWrappedInProseAndFences() {
        let reply = """
        Sure! Here it is:
        ```json
        {"observation": "A car drives at night and stops at a red light.", "same_goal_as_previous": false,
         "instruction": "Stop at the red light.", "variations": ["Wait at the red light", "stop at the red light", "Halt for the red signal", "", "Brake for the {red} light"]}
        ```
        """
        let a = LabelPrompts.parseAnswer(reply, variations: 5)
        XCTAssertEqual(a?.instruction, "Stop at the red light")
        XCTAssertEqual(a?.observation, "A car drives at night and stops at a red light.")
        XCTAssertEqual(a?.sameAsPrevious, false)
        // The repeat of the instruction and the blank are dropped; braces inside strings don't break parsing.
        XCTAssertEqual(a?.variations, ["Wait at the red light", "Halt for the red signal", "Brake for the {red} light"])
    }

    func testLimitsVariationsAndRejectsJunk() {
        let many = (1 ... 9).map { "\"Go to place number \($0)\"" }.joined(separator: ",")
        let a = LabelPrompts.parseAnswer("{\"instruction\": \"Go to the shop\", \"variations\": [\(many)]}", variations: 3)
        XCTAssertEqual(a?.variations.count, 3)
        XCTAssertNil(LabelPrompts.parseAnswer("I can't tell what is happening.", variations: 3))
        XCTAssertNil(LabelPrompts.parseAnswer("{\"instruction\": \"Go\"}", variations: 3), "one word is not an instruction")
        XCTAssertEqual(LabelPrompts.parseAnswer("<think>hm</think>{\"instruction\": \"Instruction: jump onto the roof\", \"same_goal_as_previous\": \"true\"}", variations: 0)?.instruction, "Jump onto the roof")
        XCTAssertEqual(LabelPrompts.parseAnswer("{\"instruction\": \"Jump onto the roof\", \"same_goal_as_previous\": \"true\"}", variations: 0)?.sameAsPrevious, true)
    }

    func testReview() {
        let ok = LabelPrompts.parseReview("{\"accurate\": true, \"problem\": \"\"}", variations: 3)
        XCTAssertEqual(ok?.accurate, true)
        XCTAssertNil(ok?.instruction)
        let fix = LabelPrompts.parseReview("{\"accurate\": false, \"problem\": \"no tree\", \"instruction\": \"Drive to the gas station\", \"variations\": [\"Head to the gas station\"]}", variations: 3)
        XCTAssertEqual(fix?.accurate, false)
        XCTAssertEqual(fix?.instruction, "Drive to the gas station")
        XCTAssertEqual(fix?.variations, ["Head to the gas station"])
        // "Not accurate" without a fix keeps the original.
        XCTAssertEqual(LabelPrompts.parseReview("{\"accurate\": false}", variations: 3)?.accurate, true)
    }

    func testGuidanceKeepsTheLatestTurns() {
        let chat = (0 ..< 40).map { LabelChatMessage(role: $0 % 2 == 0 ? .user : .assistant, text: "message \($0) " + String(repeating: "x", count: 100)) }
        let g = LabelPrompts.guidance(from: chat, limit: 1000)
        XCTAssertTrue(g.contains("message 39"))
        XCTAssertFalse(g.contains("message 0 "))
        XCTAssertTrue(g.hasPrefix("Person: ") || g.hasPrefix("You: "))
        XCTAssertLessThanOrEqual(g.count, 1100)
    }

    // MARK: Clips

    /// 60 s at 10 Hz: W for 20 s, then D for 20 s, then nothing.
    func steps() -> [TimelineStep] {
        (0 ..< 600).map { i in
            var a = TimestepAction()
            if i < 200 { a.keys = ["w"] } else if i < 400 { a.keys = ["d"] }
            return TimelineStep(frameIndex: i, time: Double(i) / 10, action: a, modelControlled: false, excluded: false, instruction: nil)
        }
    }

    func testClipsCoverTheRecordingAndCutWhereInputChanges() {
        let w = LabelPlanner.windows(steps: steps(), duration: 60, target: 9, skip: [])
        XCTAssertEqual(w.first?.start, 0)
        XCTAssertEqual(w.last?.end ?? 0, 60, accuracy: 0.01)
        for (a, b) in zip(w, w.dropFirst()) { XCTAssertEqual(a.end, b.start, accuracy: 0.001) }
        for c in w { XCTAssertGreaterThan(c.duration, 9 * 0.55); XCTAssertLessThan(c.duration, 9 * 1.6) }
        // The switch from W to D at 20 s is a clip boundary, and so is 40 s.
        XCTAssertTrue(w.contains { abs($0.end - 20) < 0.25 }, "\(w)")
        XCTAssertTrue(w.contains { abs($0.end - 40) < 0.25 }, "\(w)")
    }

    func testSkippedRangesAreNotLabelled() {
        let w = LabelPlanner.windows(steps: steps(), duration: 60, target: 9, skip: [TimeRange(start: 25, end: 35)])
        XCTAssertFalse(w.contains { $0.start < 35 && $0.end > 25 }, "\(w)")
        XCTAssertTrue(w.contains { abs($0.end - 25) < 0.01 })
        XCTAssertTrue(w.contains { abs($0.start - 35) < 0.01 })
    }

    func testShortRecordingIsOneClip() {
        XCTAssertEqual(LabelPlanner.windows(steps: Array(steps().prefix(80)), duration: 8, target: 9, skip: []), [LabelWindow(start: 0, end: 8)])
        XCTAssertEqual(LabelPlanner.windows(steps: [], duration: 0.2, target: 9, skip: []), [])
    }

    func testInputLog() {
        var s: [TimelineStep] = []
        for i in 0 ..< 30 {
            var a = TimestepAction()
            if i < 24 { a.keys = ["w"] }
            if i == 10 { a.keys.insert("space") }
            if i >= 12 && i < 22 { a.dx = 30 }
            s.append(TimelineStep(frameIndex: i, time: 5 + Double(i) / 10, action: a, modelControlled: false, excluded: false, instruction: nil))
        }
        let log = LabelPlanner.inputLog(s, from: 5)
        XCTAssertTrue(log.contains("0.0–2.4 s: holding W"), log)
        XCTAssertTrue(log.contains("1.0 s: tapped SPACE") || log.contains("1.0 s: tapped Space"), log)
        XCTAssertTrue(log.contains("mouse moving right"), log)
        XCTAssertEqual(LabelPlanner.inputLog(Array(s.suffix(5)), from: 7.5), "no input: the person waited or watched")
        let ai = LabelPlanner.inputLog(s, from: 5, aiControlled: [TimeRange(start: 0, end: 6)])
        XCTAssertTrue(ai.contains("0.0–1.0 s: the AI was playing"), ai)
    }

    // MARK: Variations

    func testOldLabelFilesStillLoadAndNewOnesKeepVariations() throws {
        let old = #"{"segments":[{"id":"6B2B0A4E-0C0B-4C47-9C7E-1E8E6F0A1B2C","start":0,"end":5,"text":"Drive","source":"auto"}],"excluded":[],"modelControlled":[]}"#
        let a = try JSONDecoder().decode(RecordingAnnotations.self, from: Data(old.utf8))
        XCTAssertNil(a.segments[0].variants)
        var seg = TextSegment(start: 0, end: 5, text: "Go to the tree", source: .auto, variants: ["Move to the tree", " ", "go to the tree", "Park next to the tree"], observation: "A tree")
        XCTAssertEqual(seg.wordings, ["Go to the tree", "Move to the tree", "Park next to the tree"])
        seg = try JSONDecoder().decode(TextSegment.self, from: JSONEncoder().encode(seg))
        XCTAssertEqual(seg.variants?.count, 4)
        XCTAssertEqual(seg.observation, "A tree")
    }

    func testTextTableKnowsEveryWording() {
        let t = TextTable()
        let base = t.register(TextSegment(start: 0, end: 5, text: "Go to the tree", source: .auto, variants: ["Move to the tree", "Park next to the tree"]))
        XCTAssertEqual(t.id(for: "Go to the tree"), base)
        XCTAssertEqual(t.wordings[base]?.count, 3)
        XCTAssertEqual(Set(t.wordings[base]!.map { t.strings[Int($0)] }), ["Go to the tree", "Move to the tree", "Park next to the tree"])
        // No variations: nothing to pick from.
        let plain = t.register(TextSegment(start: 5, end: 9, text: "Jump", source: .user))
        XCTAssertNil(t.wordings[plain])
    }
}
