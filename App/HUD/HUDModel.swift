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
    /// Every key used since the HUD was shown, in the order first pressed. Keys stay once used.
    private(set) var seen: [String] = []

    /// Starts a new session: forgets the keys used so far.
    func reset() {
        seen = []
        snapshot = LiveInputState.Snapshot()
        trail = []
    }

    func push(_ s: LiveInputState.Snapshot) {
        snapshot = s
        for k in s.keys.sorted() where KeyCatalog.info(k) != nil && !seen.contains(k) { seen.append(k) }
        let scale = 1 / 600.0
        trail.append(CGVector(dx: s.velocity.dx * scale, dy: s.velocity.dy * scale))
        if trail.count > 12 { trail.removeFirst(trail.count - 12) }
    }
}
