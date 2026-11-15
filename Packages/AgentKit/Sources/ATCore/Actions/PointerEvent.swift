import Foundation

/// Discrete pointer events predicted in the "mouse button" slots. The first three keep
/// Pixel2Play's class order (left/right/middle) so its pretrained head transfers; scroll is
/// folded into the same vocabulary so the per-timestep token layout stays at 12 tokens.
public enum PointerEvent: String, Codable, CaseIterable, Hashable, Sendable {
    case left, right, middle, back, forward
    case scrollUp, scrollUpFast, scrollDown, scrollDownFast, scrollLeft, scrollRight

    public var isButton: Bool { [.left, .right, .middle, .back, .forward].contains(self) }
    public var isScroll: Bool { !isButton }

    public var label: String {
        switch self {
        case .left: "Left click"
        case .right: "Right click"
        case .middle: "Middle click"
        case .back: "Back button"
        case .forward: "Forward button"
        case .scrollUp: "Scroll up"
        case .scrollUpFast: "Scroll up (fast)"
        case .scrollDown: "Scroll down"
        case .scrollDownFast: "Scroll down (fast)"
        case .scrollLeft: "Scroll left"
        case .scrollRight: "Scroll right"
        }
    }

    /// CGMouseButton number for buttons.
    public var buttonNumber: Int? {
        switch self {
        case .left: 0
        case .right: 1
        case .middle: 2
        case .back: 3
        case .forward: 4
        default: nil
        }
    }

    /// Scroll lines per timestep emitted when this event is predicted (y positive = up).
    public var scrollLines: (x: Int, y: Int) {
        switch self {
        case .scrollUp: (0, 1)
        case .scrollUpFast: (0, 4)
        case .scrollDown: (0, -1)
        case .scrollDownFast: (0, -4)
        case .scrollLeft: (-1, 0)
        case .scrollRight: (1, 0)
        default: (0, 0)
        }
    }

    public static let p2pOrder: [PointerEvent] = [.left, .right, .middle]
    public static let defaultOrder: [PointerEvent] = [.left, .right, .middle, .back, .forward,
                                                      .scrollUp, .scrollUpFast, .scrollDown, .scrollDownFast,
                                                      .scrollLeft, .scrollRight]

    /// Classifies one timestep's summed scroll (in lines, y positive = up) into at most one event.
    public static func fromScroll(lines dx: Double, _ dy: Double) -> PointerEvent? {
        if abs(dy) >= abs(dx), abs(dy) >= 0.5 {
            return dy > 0 ? (dy >= 3 ? .scrollUpFast : .scrollUp) : (dy <= -3 ? .scrollDownFast : .scrollDown)
        }
        if abs(dx) >= 0.5 { return dx > 0 ? .scrollRight : .scrollLeft }
        return nil
    }
}
