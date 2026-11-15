import ATCapture
import ATCore
import CoreGraphics
import Foundation
import Observation

/// What the HUD shows. Fed ~30×/s from recording input (human) or model output (Run).
@MainActor @Observable
final class HUDModel {
    enum Mode { case record, run }
    var mode = Mode.record
    var options = HUDOptions()
    var snapshot = LiveInputState.Snapshot()
    var trail: [CGVector] = []
    /// Model input image (Run mode) and its description.
    var vision: CGImage?
    var visionCaption = ""
    var statusLine = ""
    var instruction: String?
    var paused = false

    func push(_ s: LiveInputState.Snapshot) {
        snapshot = s
        let scale = 1 / 600.0
        trail.append(CGVector(dx: s.velocity.dx * scale, dy: s.velocity.dy * scale))
        if trail.count > 12 { trail.removeFirst(trail.count - 12) }
    }
}
