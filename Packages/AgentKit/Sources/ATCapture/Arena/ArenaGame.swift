import ATCore
import CoreGraphics
import CoreText
import CoreVideo
import Foundation

/// Small deterministic games used to verify that recording → training → running actually works,
/// and to measure closed-loop success (like Pixel2Play's Godot test environments).
public final class ArenaGame: @unchecked Sendable {
    public enum Mode: String, Codable, CaseIterable, Identifiable, Sendable {
        /// Move with WASD to touch the target square (keyboard, vision).
        case chase
        /// Turn with the mouse to center the target and click it (relative mouse, clicks).
        case aim
        /// Two colored squares; go to the one the instruction names (language).
        case colorChase
        /// Click the button whose label the instruction names (absolute pointer, reading text).
        case clickButton
        /// Type the word the instruction gives, then press Return (literal typing).
        case typeWord
        public var id: String { rawValue }
        public var title: String {
            switch self {
            case .chase: "Chase (keyboard)"
            case .aim: "Aim (mouse)"
            case .colorChase: "Go to the named color (instructions)"
            case .clickButton: "Click the named button (pointer + text)"
            case .typeWord: "Type the given word (typing)"
            }
        }
        public var usesInstructions: Bool { self == .colorChase || self == .clickButton || self == .typeWord }
        public var keys: [String] { ["w", "a", "s", "d"] }
    }

    public static let colors: [(name: String, rgb: (CGFloat, CGFloat, CGFloat))] = [
        ("red", (0.95, 0.27, 0.27)), ("green", (0.25, 0.85, 0.4)), ("blue", (0.3, 0.5, 1.0)), ("yellow", (0.98, 0.85, 0.2)),
    ]
    public static let buttonWords = ["Play", "Settings", "Shop", "Quit", "Help", "Map"]
    public static let typingWords = ["apple", "river", "stone", "cloud", "tiger", "lemon", "grass", "piano", "rocket", "forest", "candle", "wizard"]

    public let mode: Mode
    public private(set) var score = 0
    public private(set) var steps = 0
    public private(set) var instruction: String?
    var rng: SplitMix64
    // chase / colorChase
    var player = CGPoint(x: 0.5, y: 0.5)
    var targets: [(pos: CGPoint, color: Int)] = []
    var goal = 0
    // aim
    var view = CGPoint.zero
    var aimTarget = CGPoint(x: 0.3, y: 0.1)
    var flash = 0
    // click
    var buttons: [(rect: CGRect, word: Int)] = []
    public private(set) var pointer = CGPoint(x: 0.5, y: 0.5)
    var expertMemory: (keys: Set<String>, hold: Int) = ([], 0)
    // typeWord
    var word = ""
    public private(set) var typed = ""
    var lastKeys = Set<String>()
    var typeCooldown = 0
    var lastButtonDown = false

    public init(mode: Mode, seed: UInt64) {
        self.mode = mode
        rng = SplitMix64(seed)
        reset()
    }

    func r() -> CGFloat { CGFloat(Double(rng.next() % 1_000_000) / 1_000_000) }

    func reset() {
        switch mode {
        case .chase:
            targets = [(randomPoint(), 0)]
        case .colorChase:
            var a = Int(rng.next() % 4), b = Int(rng.next() % 4)
            if a == b { b = (a + 1 + Int(rng.next() % 3)) % 4 }
            targets = [(randomPoint(), a), (randomPoint(), b)]
            goal = Int(rng.next() % 2)
            instruction = "go to the \(Self.colors[targets[goal].color].name) square"
        case .aim:
            aimTarget = CGPoint(x: view.x + (r() - 0.5) * 1.4, y: view.y + (r() - 0.5) * 0.7)
        case .clickButton:
            var words = Array(0 ..< Self.buttonWords.count)
            words.shuffle(using: &rng)
            buttons = (0 ..< 4).map { i in
                let col = i % 2, row = i / 2
                let w = 0.3 + r() * 0.06, h = 0.13
                let x = 0.08 + CGFloat(col) * 0.48 + r() * 0.08, y = 0.2 + CGFloat(row) * 0.38 + r() * 0.1
                return (CGRect(x: x, y: y, width: w, height: h), words[i])
            }
            goal = Int(rng.next() % 4)
            instruction = "click the \(Self.buttonWords[buttons[goal].word]) button"
        case .typeWord:
            word = Self.typingWords[Int(rng.next() % UInt64(Self.typingWords.count))]
            typed = ""
            instruction = "type \"\(word)\""
        }
    }

    func randomPoint() -> CGPoint {
        var p: CGPoint
        repeat { p = CGPoint(x: 0.08 + r() * 0.84, y: 0.12 + r() * 0.8) } while hypot(p.x - player.x, p.y - player.y) < 0.25
        return p
    }

    /// Distance from the player to the current goal (chase modes), for takeover rules.
    public var goalDistance: Double? {
        switch mode {
        case .chase, .colorChase: let t = targets[mode == .chase ? 0 : goal].pos; return hypot(t.x - player.x, t.y - player.y)
        case .aim: return hypot(aimTarget.x - view.x, aimTarget.y - view.y)
        case .clickButton: return hypot(buttons[goal].rect.midX - pointer.x, buttons[goal].rect.midY - pointer.y)
        case .typeWord: return Double(word.count - typed.count)
        }
    }

    public var nearWall: Bool { mode == .chase || mode == .colorChase ? (player.x < 0.06 || player.x > 0.94 || player.y < 0.1 || player.y > 0.94) : false }

    public var debugState: String {
        switch mode {
        case .chase, .colorChase: String(format: "player (%.2f, %.2f) target (%.2f, %.2f)", player.x, player.y, targets[mode == .chase ? 0 : goal].pos.x, targets[mode == .chase ? 0 : goal].pos.y)
        case .aim: String(format: "aim error (%.3f, %.3f)", aimTarget.x - view.x, aimTarget.y - view.y)
        case .clickButton: String(format: "pointer (%.2f, %.2f) goal %@", pointer.x, pointer.y, "\(buttons[goal].rect)")
        case .typeWord: "typed '\(typed)' of '\(word)'"
        }
    }

    // MARK: Dynamics

    /// Applies one timestep of input. `dx`/`dy` are pointer deltas in points (relative) and
    /// `absX`/`absY` the pointer position (absolute modes).
    public func step(_ a: TimestepAction, dt: Double) {
        steps += 1
        switch mode {
        case .chase, .colorChase:
            var v = CGVector.zero
            if a.keys.contains("w") { v.dy -= 1 }
            if a.keys.contains("s") { v.dy += 1 }
            if a.keys.contains("a") { v.dx -= 1 }
            if a.keys.contains("d") { v.dx += 1 }
            let n = max(1, hypot(v.dx, v.dy))
            let speed = 0.45 * dt
            player.x = min(0.97, max(0.03, player.x + v.dx / n * speed))
            player.y = min(0.97, max(0.07, player.y + v.dy / n * speed))
            for (i, t) in targets.enumerated() where hypot(t.pos.x - player.x, t.pos.y - player.y) < 0.06 {
                if mode == .chase || i == goal { score += 1 } else { score -= 1 }
                reset()
                break
            }
        case .aim:
            view.x += a.dx * 0.0015
            view.y += a.dy * 0.0015
            if flash > 0 { flash -= 1 }
            let down = a.pointer.contains(.left)
            if down && !lastButtonDown {
                if hypot(aimTarget.x - view.x, aimTarget.y - view.y) < 0.045 { score += 1; flash = 6; reset() }
            }
            lastButtonDown = down
        case .clickButton:
            pointer = CGPoint(x: a.absX, y: a.absY)
            let down = a.pointer.contains(.left)
            if down && !lastButtonDown {
                if let i = buttons.firstIndex(where: { $0.rect.contains(pointer) }) {
                    score += i == goal ? 1 : -1
                    reset()
                }
            }
            lastButtonDown = down
        case .typeWord:
            // Newly pressed keys type; held keys don't repeat.
            for k in a.keys.subtracting(lastKeys).sorted() {
                if k == "return" {
                    score += typed == word ? 1 : (typed.isEmpty ? 0 : -1)
                    reset()
                } else if k == "delete" {
                    if !typed.isEmpty { typed.removeLast() }
                } else if let ch = KeyCatalog.info(k)?.character, ch.isLetter, typed.count < 24 {
                    typed.append(ch)
                }
            }
            lastKeys = a.keys
        }
    }

    // MARK: Scripted expert

    /// The demonstrator: near-optimal with a little human-like hesitation.
    public func expertAction() -> TimestepAction {
        var a = TimestepAction()
        switch mode {
        case .chase, .colorChase:
            let t = targets[mode == .chase ? 0 : goal].pos
            let dx = t.x - player.x, dy = t.y - player.y
            var keys = Set<String>()
            if dx > 0.02 { keys.insert("d") } else if dx < -0.02 { keys.insert("a") }
            if dy > 0.02 { keys.insert("s") } else if dy < -0.02 { keys.insert("w") }
            // Humans don't change keys every frame: hold the current choice briefly.
            if expertMemory.hold > 0 && !expertMemory.keys.isEmpty { expertMemory.hold -= 1; keys = expertMemory.keys }
            else if keys != expertMemory.keys { expertMemory = (keys, Int(rng.next() % 3)) }
            a.keys = keys
        case .aim:
            let ex = aimTarget.x - view.x, ey = aimTarget.y - view.y
            let gain = 0.5 + Double(r()) * 0.3
            a.dx = max(-90, min(90, ex / 0.0015 * gain))
            a.dy = max(-60, min(60, ey / 0.0015 * gain))
            if abs(a.dx) < 1.5 { a.dx = 0 }
            if abs(a.dy) < 1.5 { a.dy = 0 }
            if hypot(ex, ey) < 0.03 && !lastButtonDown { a.pointer = [.left] }
        case .clickButton:
            let target = buttons[goal].rect
            let tx = target.midX, ty = target.midY
            let d = hypot(tx - pointer.x, ty - pointer.y)
            let k = min(1, 0.35 + Double(r()) * 0.15)
            a.absX = d < 0.01 ? tx : pointer.x + (tx - pointer.x) * k
            a.absY = d < 0.01 ? ty : pointer.y + (ty - pointer.y) * k
            if target.insetBy(dx: 0.02, dy: 0.02).contains(pointer) && !lastButtonDown { a.pointer = [.left] }
        case .typeWord:
            // Press the next letter for one step, release for at least one, like a quick typist.
            if typeCooldown > 0 { typeCooldown -= 1; a.keys = []; break }
            if !lastKeys.isEmpty { a.keys = []; typeCooldown = Int(rng.next() % 2); break }
            if typed == word { a.keys = ["return"] }
            else if word.hasPrefix(typed) { a.keys = [String(word[word.index(word.startIndex, offsetBy: typed.count)])] }
            else { a.keys = ["delete"] }
        }
        if mode != .clickButton { a.absX = pointer.x; a.absY = pointer.y }
        return a
    }

    // MARK: Rendering

    /// Draws the current state into a BGRA pixel buffer (what a window capture would deliver).
    public func render(into pb: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) else { return }
        render(ctx, width: CGFloat(w), height: CGFloat(h))
    }

    public func render(_ ctx: CGContext, width W: CGFloat, height H: CGFloat) {
        ctx.saveGState()
        ctx.translateBy(x: 0, y: H)
        ctx.scaleBy(x: 1, y: -1) // top-left origin
        ctx.setFillColor(CGColor(red: 0.09, green: 0.1, blue: 0.13, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: W, height: H))
        func color(_ c: Int, _ a: CGFloat = 1) -> CGColor { let v = Self.colors[c].rgb; return CGColor(red: v.0, green: v.1, blue: v.2, alpha: a) }
        switch mode {
        case .chase, .colorChase:
            ctx.setStrokeColor(CGColor(gray: 1, alpha: 0.06))
            for i in 1 ..< 10 {
                ctx.stroke(CGRect(x: CGFloat(i) * W / 10, y: 0, width: 0, height: H))
                ctx.stroke(CGRect(x: 0, y: CGFloat(i) * H / 10, width: W, height: 0))
            }
            for t in targets {
                let s = 0.07 * W
                ctx.setFillColor(mode == .chase ? color(0) : color(t.color))
                ctx.fill(CGRect(x: t.pos.x * W - s / 2, y: t.pos.y * H - s / 2, width: s, height: s))
            }
            ctx.setFillColor(CGColor(gray: 0.96, alpha: 1))
            let r = 0.035 * W
            ctx.fillEllipse(in: CGRect(x: player.x * W - r, y: player.y * H - r, width: 2 * r, height: 2 * r))
        case .aim:
            let ox = -view.x * W, oy = -view.y * H
            ctx.setStrokeColor(CGColor(gray: 1, alpha: 0.12))
            let step = W / 8
            var x = (ox.truncatingRemainder(dividingBy: step) + step).truncatingRemainder(dividingBy: step)
            while x < W { ctx.stroke(CGRect(x: x, y: 0, width: 0, height: H)); x += step }
            var y = (oy.truncatingRemainder(dividingBy: step) + step).truncatingRemainder(dividingBy: step)
            while y < H { ctx.stroke(CGRect(x: 0, y: y, width: W, height: 0)); y += step }
            let tx = (aimTarget.x - view.x) * W + W / 2, ty = (aimTarget.y - view.y) * H + H / 2
            let r = 0.05 * W
            ctx.setFillColor(color(0))
            ctx.fillEllipse(in: CGRect(x: tx - r, y: ty - r, width: 2 * r, height: 2 * r))
            ctx.setStrokeColor(flash > 0 ? color(1) : CGColor(gray: 1, alpha: 0.9))
            ctx.setLineWidth(max(2, W / 200))
            ctx.stroke(CGRect(x: W / 2 - 0.03 * W, y: H / 2, width: 0.06 * W, height: 0))
            ctx.stroke(CGRect(x: W / 2, y: H / 2 - 0.03 * W, width: 0, height: 0.06 * W))
        case .clickButton:
            for (i, b) in buttons.enumerated() {
                let rect = CGRect(x: b.rect.minX * W, y: b.rect.minY * H, width: b.rect.width * W, height: b.rect.height * H)
                ctx.setFillColor(color(i % 4, 0.85))
                ctx.addPath(CGPath(roundedRect: rect, cornerWidth: 8, cornerHeight: 8, transform: nil))
                ctx.fillPath()
                drawText(Self.buttonWords[b.word], in: rect, ctx: ctx, size: rect.height * 0.45)
            }
        case .typeWord:
            let field = CGRect(x: 0.1 * W, y: 0.42 * H, width: 0.8 * W, height: 0.16 * H)
            ctx.setFillColor(CGColor(gray: 0.18, alpha: 1))
            ctx.addPath(CGPath(roundedRect: field, cornerWidth: 10, cornerHeight: 10, transform: nil))
            ctx.fillPath()
            drawText(typed.isEmpty ? " " : typed, in: field, ctx: ctx, size: field.height * 0.5)
            let caretX = field.midX + CGFloat(typed.count) * field.height * 0.15
            ctx.setFillColor(CGColor(gray: 1, alpha: 0.8))
            ctx.fill(CGRect(x: caretX, y: field.minY + field.height * 0.2, width: 2, height: field.height * 0.6))
        }
        ctx.restoreGState()
    }

    func drawText(_ s: String, in rect: CGRect, ctx: CGContext, size: CGFloat) {
        let font = CTFontCreateWithName("Helvetica-Bold" as CFString, size, nil)
        let attr = NSAttributedString(string: s, attributes: [.init(kCTFontAttributeName as String): font,
                                                               .init(kCTForegroundColorAttributeName as String): CGColor(gray: 1, alpha: 1)])
        let line = CTLineCreateWithAttributedString(attr)
        let bounds = CTLineGetImageBounds(line, ctx)
        ctx.saveGState()
        ctx.textMatrix = CGAffineTransform(scaleX: 1, y: -1)
        ctx.textPosition = CGPoint(x: rect.midX - bounds.width / 2, y: rect.midY + bounds.height / 2)
        CTLineDraw(line, ctx)
        ctx.restoreGState()
    }
}

/// Deterministic RNG for reproducible arena episodes.
public struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    public init(_ seed: UInt64) { state = seed &+ 0x1234_5678 }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
