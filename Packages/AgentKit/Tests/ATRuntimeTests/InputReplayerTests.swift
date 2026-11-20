@testable import ATCore
@testable import ATRuntime
import CoreGraphics
import Foundation
import XCTest

/// Replay plans and the replayer, without posting a single real event.
final class InputReplayerTests: XCTestCase {
    let ms: Int64 = 1_000_000

    func meta(source: CGRect = CGRect(x: 0, y: 0, width: 1000, height: 500), clock: String = "uptime-ns") -> RecordingMeta {
        var m = RecordingMeta(name: "test", appVersion: "test", target: CaptureTargetInfo(kind: .window, appName: "Game"),
                              captureWidth: 1000, captureHeight: 500, sourceRect: CodableRect(source), captureFPS: 30,
                              showsCursor: false, input: InputCaptureOptions())
        m.eventClock = clock
        return m
    }

    func frames(from start: Int64, to end: Int64, every step: Int64) -> [FrameRecord] {
        stride(from: start, through: end, by: Int(step)).enumerated().map { FrameRecord(index: UInt32($0.offset), timeNs: $0.element, hostNs: UInt64($0.element)) }
    }

    func key(_ id: String) -> UInt16 { KeyCatalog.info(id)!.keyCode }
    func down(_ id: String, _ t: Int64) -> InputEventRecord { InputEventRecord(timeNs: t * ms, kind: .keyDown, code: key(id)) }
    func up(_ id: String, _ t: Int64) -> InputEventRecord { InputEventRecord(timeNs: t * ms, kind: .keyUp, code: key(id)) }
    func mod(_ id: String, _ t: Int64, _ pressed: Bool) -> InputEventRecord { InputEventRecord(timeNs: t * ms, kind: .flagsChanged, flags: pressed ? 2 : 0, code: key(id)) }
    func move(_ t: Int64, _ dx: Float, _ dy: Float, at: (Float, Float) = (500, 250)) -> InputEventRecord {
        InputEventRecord(timeNs: t * ms, kind: .mouseMove, x: at.0, y: at.1, dx: dx, dy: dy)
    }

    // MARK: Plans

    func testPlanStartsAtTheFirstFrameAndEndsAtTheLast() {
        let events = [
            down("w", 10), // held before the video starts: pressed at time zero
            move(20, 5, 0), // movement before the video: already in the first frame
            move(100, 3, -2),
            up("w", 500),
            down("space", 900), up("space", 950),
            down("e", 1100), // after the last frame
        ]
        let plan = ReplayPlan.make(frames: frames(from: 50 * ms, to: 1050 * ms, every: 100 * ms), events: events, meta: meta(), options: ReplayPlan.Options())
        XCTAssertEqual(plan.durationNs, 1000 * ms)
        XCTAssertEqual(plan.steps, [
            ReplayPlan.Step(timeNs: 0, action: .key(code: key("w"), down: true)),
            ReplayPlan.Step(timeNs: 50 * ms, action: .move(dx: 3, dy: -2, to: nil)),
            ReplayPlan.Step(timeNs: 450 * ms, action: .key(code: key("w"), down: false)),
            ReplayPlan.Step(timeNs: 850 * ms, action: .key(code: key("space"), down: true)),
            ReplayPlan.Step(timeNs: 900 * ms, action: .key(code: key("space"), down: false)),
        ])
    }

    func testBlockedModifiersAndAppShortcutsAreNeverSent() {
        var o = ReplayPlan.Options()
        o.hotkeys = [(key("r"), [.maskCommand, .maskShift, .maskAlternate])]
        let events = [
            // ⌘Q with Command blocked (the default): neither ⌘ nor Q is sent
            mod("command", 100, true), down("q", 110), up("q", 120), mod("command", 130, false),
            // ⇧W is fine
            mod("shift", 200, true), down("w", 210), up("w", 220), mod("shift", 230, false),
            // the record shortcut ⌥⇧⌘R: R is never sent
            mod("option", 300, true), mod("shift", 301, true), mod("command", 302, true), down("r", 310), up("r", 320),
            mod("command", 330, false), mod("shift", 331, false), mod("option", 332, false),
        ]
        let plan = ReplayPlan.make(frames: frames(from: 0, to: 400 * ms, every: 100 * ms), events: events, meta: meta(), options: o)
        let keys = plan.steps.compactMap { s -> String? in
            guard case let .key(code, d) = s.action else { return nil }
            return (KeyCatalog.byKeyCode[code]?.id ?? "?") + (d ? "↓" : "↑")
        }
        XCTAssertEqual(keys, ["shift↓", "w↓", "w↑", "shift↑", "option↓", "shift↓", "shift↑", "option↑"])
        XCTAssertEqual(plan.skippedPresses, 4) // ⌘, Q, ⌘ again, R
    }

    func testPointerModeMapsPositionsOntoTheTargetWindow() {
        var o = ReplayPlan.Options()
        o.mouseMode = .pointer
        o.targetFrame = CGRect(x: 100, y: 50, width: 2000, height: 1000)
        let events = [move(10, 4, 2, at: (500, 250)), InputEventRecord(timeNs: 20 * ms, kind: .buttonDown, code: 0, x: 1000, y: 500)]
        let plan = ReplayPlan.make(frames: frames(from: 0, to: 100 * ms, every: 50 * ms), events: events, meta: meta(), options: o)
        XCTAssertEqual(plan.steps.map(\.action), [
            .move(dx: 8, dy: 4, to: CGPoint(x: 1100, y: 550)),
            .button(0, down: true, at: CGPoint(x: 2100, y: 1050)),
        ])
    }

    func testElkbornReplaysAreAlwaysCameraMovement() {
        var o = ReplayPlan.Options()
        o.mouseMode = .pointer
        o.targetFrame = CGRect(x: 0, y: 0, width: 500, height: 250)
        let plan = ReplayPlan.make(frames: frames(from: 0, to: 100 * ms, every: 16 * ms), events: [move(16, 2.5, -1.25)], meta: meta(clock: "elkborn-replay"), options: o)
        XCTAssertEqual(plan.mouseMode, .camera)
        XCTAssertEqual(plan.steps.map(\.action), [.move(dx: 2.5, dy: -1.25, to: nil)], "delta 1:1, at the pointer's place")
    }

    // MARK: Replayer

    final class VirtualClock: @unchecked Sendable {
        var t: UInt64 = 5_000_000_000
        var clock: ReplayClock { ReplayClock(now: { self.t }, wait: { deadline, _ in if deadline > self.t { self.t = deadline } }) }
    }

    func record(_ r: InputReplayer, _ clock: VirtualClock, start: UInt64) -> () -> [(Int64, InputReplayer.Output)] {
        var out: [(Int64, InputReplayer.Output)] = []
        r.emit = { out.append((Int64(clock.t - start) / 1_000_000, $0)) }
        return { out }
    }

    func testReplayerSendsEachStepOnTimeAndLetsGoAtTheEnd() {
        let shift = key("shift"), w = key("w")
        let plan = ReplayPlan(steps: [
            .init(timeNs: 0, action: .key(code: shift, down: true)),
            .init(timeNs: 10 * ms, action: .key(code: w, down: true)),
            .init(timeNs: 20 * ms, action: .button(0, down: true, at: nil)),
            .init(timeNs: 30 * ms, action: .move(dx: 0.4, dy: 0, to: nil)),
            .init(timeNs: 40 * ms, action: .move(dx: 0.4, dy: 0, to: nil)),
            .init(timeNs: 50 * ms, action: .move(dx: 0.4, dy: 0, to: nil)),
        ], durationNs: 100 * ms, mouseMode: .camera, skippedPresses: 0)
        let clock = VirtualClock()
        let r = InputReplayer(plan: plan, clock: clock.clock)
        let outputs = record(r, clock, start: clock.t)
        var ending: InputReplayer.Ending?
        r.onEnd = { ending = $0 }
        r.run()
        let shiftFlag = CGEventFlags.maskShift.rawValue
        let got = outputs()
        XCTAssertEqual(got.map(\.0), [0, 10, 20, 50, 100, 100, 100])
        XCTAssertEqual(got.map(\.1), [
            .key(code: shift, down: true, flags: shiftFlag, flagsChanged: true),
            .key(code: w, down: true, flags: shiftFlag, flagsChanged: false),
            .button(0, down: true, at: nil, flags: shiftFlag),
            .move(dx: 1, dy: 0, to: nil, dragging: 0, flags: shiftFlag), // 0.4 + 0.4 + 0.4: one whole unit, dragging the held button
            .button(0, down: false, at: nil, flags: shiftFlag),
            .key(code: w, down: false, flags: shiftFlag, flagsChanged: false),
            .key(code: shift, down: false, flags: 0, flagsChanged: true),
        ])
        XCTAssertEqual(ending, .finished)
        XCTAssertEqual(r.status.elapsed, 0.1, accuracy: 1e-9)
        XCTAssertFalse(r.status.running)
    }

    func testReplayerHoldsWhileTheTargetIsNotInFront() {
        let w = key("w"), d = key("d")
        let plan = ReplayPlan(steps: [
            .init(timeNs: 0, action: .key(code: w, down: true)),
            .init(timeNs: 300 * ms, action: .key(code: d, down: true)),
        ], durationNs: 400 * ms, mouseMode: .camera, skippedPresses: 0)
        let clock = VirtualClock()
        let start = clock.t
        let r = InputReplayer(plan: plan, clock: clock.clock)
        let outputs = record(r, clock, start: start)
        // the user switches away at 100 ms and comes back at 600 ms
        r.isTargetFrontmost = { let e = clock.t - start; return e < 100_000_000 || e >= 600_000_000 }
        r.run()
        let got = outputs()
        XCTAssertEqual(got.map(\.1), [
            .key(code: w, down: true, flags: 0, flagsChanged: false),
            .key(code: w, down: false, flags: 0, flagsChanged: false), // let go while away
            .key(code: w, down: true, flags: 0, flagsChanged: false), // pressed again on return
            .key(code: d, down: true, flags: 0, flagsChanged: false),
            .key(code: d, down: false, flags: 0, flagsChanged: false),
            .key(code: w, down: false, flags: 0, flagsChanged: false),
        ])
        let times = got.map(\.0)
        XCTAssertEqual(times[1], 100)
        XCTAssertEqual(times[2], 600, accuracy: 50) // noticed within the 50 ms focus poll
        XCTAssertEqual(times[3] - times[2], 200, accuracy: 1, "the clock stood still while away: D is still 200 ms after the return")
    }

    func testStopLetsGoOfEverything() {
        let w = key("w")
        let plan = ReplayPlan(steps: [
            .init(timeNs: 0, action: .key(code: w, down: true)),
            .init(timeNs: 10 * ms, action: .button(1, down: true, at: nil)),
            .init(timeNs: 500 * ms, action: .key(code: key("space"), down: true)),
        ], durationNs: 1000 * ms, mouseMode: .camera, skippedPresses: 0)
        let clock = VirtualClock()
        let r = InputReplayer(plan: plan, clock: clock.clock)
        var out: [InputReplayer.Output] = []
        r.emit = { o in
            out.append(o)
            if case .button(1, true, _, _) = o { r.stop() }
        }
        var ending: InputReplayer.Ending?
        r.onEnd = { ending = $0 }
        r.run()
        XCTAssertEqual(out, [
            .key(code: w, down: true, flags: 0, flagsChanged: false),
            .button(1, down: true, at: nil, flags: 0),
            .button(1, down: false, at: nil, flags: 0),
            .key(code: w, down: false, flags: 0, flagsChanged: false),
        ])
        XCTAssertEqual(ending, .stopped)
    }

    /// On the real clock (posting replaced): steps land within a couple of milliseconds.
    func testHostClockTiming() {
        let steps = (0 ..< 20).map { ReplayPlan.Step(timeNs: Int64($0) * 15 * ms, action: .scroll(dx: 0, dy: 1)) }
        let r = InputReplayer(plan: ReplayPlan(steps: steps, durationNs: 300 * ms, mouseMode: .camera, skippedPresses: 0))
        var times: [UInt64] = []
        r.emit = { _ in times.append(HostClock.now()) }
        let done = expectation(description: "done")
        r.onEnd = { _ in done.fulfill() }
        r.start()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(times.count, 20)
        let errors = times.enumerated().map { abs(Double(Int64($0.element - times[0]) - Int64($0.offset) * 15 * ms)) / 1e6 }
        XCTAssertLessThan(errors.max() ?? 0, 3, "worst step error \(errors.max() ?? 0) ms")
    }
}
