@testable import ATCore
@testable import ATRuntime
import XCTest

/// Your input counted as the AI's: only what the AI has and may use.
final class HumanHistoryTests: XCTestCase {
    func testOnlyKeysAndButtonsTheModelCanUse() {
        let space = ActionSpaceConfig(keys: ["w", "a", "space", "command"], pointerEvents: [.left, .scrollUp])
        var perms = ActionPermissions()
        perms.scroll = false // blocked in Run
        var h = TimestepAction()
        h.keys = ["w", "e", "command"]          // e: not in the model; command: blocked by default
        h.pointer = [.left, .right, .scrollUp]   // right: not in the model; scroll: not allowed
        h.dx = 10; h.dy = -4
        let a = AgentRuntime.asModelAction(h, space: space, permissions: perms, sensitivity: 2)
        XCTAssertEqual(a.keys, ["w"])
        XCTAssertEqual(a.pointer, [.left])
        XCTAssertEqual(a.dx, 5, accuracy: 1e-9)   // in model units, before Run sensitivity
        XCTAssertEqual(a.dy, -2, accuracy: 1e-9)
    }

    func testNoMouseWhenTheModelDoesNotMoveIt() {
        var space = ActionSpaceConfig(keys: ["w"])
        space.movesMouse = false
        var h = TimestepAction(); h.dx = 30; h.dy = 3
        let a = AgentRuntime.asModelAction(h, space: space, permissions: ActionPermissions(), sensitivity: 1)
        XCTAssertEqual(a.dx, 0); XCTAssertEqual(a.dy, 0)
        var perms = ActionPermissions(); perms.mouseMovement = false
        let b = AgentRuntime.asModelAction(h, space: ActionSpaceConfig(keys: ["w"]), permissions: perms, sensitivity: 1)
        XCTAssertEqual(b.dx, 0)
    }

    func testMergeAddsYourInputToTheAIs() {
        var ai = TimestepAction(); ai.keys = ["w"]; ai.dx = 3
        var me = TimestepAction(); me.keys = ["space"]; me.pointer = [.left]; me.dx = 4; me.dy = 1
        let m = AgentRuntime.merge(ai, me, absolute: false)
        XCTAssertEqual(m.keys, ["w", "space"])
        XCTAssertEqual(m.pointer, [.left])
        XCTAssertEqual(m.dx, 7); XCTAssertEqual(m.dy, 1)
    }

    func testMovesMouseRoundTrip() {
        var space = ActionSpaceConfig(keys: ["w"])
        var a = TimestepAction(); a.keys = ["w"]; a.dx = 40; a.dy = -12
        XCTAssertNotEqual(space.decode(space.encode(a)).dx, 0)
        space.movesMouse = false
        let d = space.decode(space.encode(a))
        XCTAssertEqual(d.keys, ["w"]); XCTAssertEqual(d.dx, 0); XCTAssertEqual(d.dy, 0)
        let saved = try! JSONDecoder().decode(ActionSpaceConfig.self, from: JSONEncoder().encode(space))
        XCTAssertFalse(saved.movesMouse)
    }
}
