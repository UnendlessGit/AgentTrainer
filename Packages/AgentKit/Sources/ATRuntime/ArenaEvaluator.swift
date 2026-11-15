import ATCapture
import ATLanguage
import ATCore
import ATModel
import Foundation
import MLX

/// Closed-loop evaluation in the simulated Test Arena: the model sees rendered frames (same GPU
/// resize as live play) and its sampled actions drive the game.
public enum ArenaEvaluator {
    public struct Result: Sendable, CustomStringConvertible {
        public var score: Int
        public var expertScore: Int
        public var randomScore: Int
        public var steps: Int
        public var msPerStep: Double
        public var description: String {
            String(format: "model %d · expert %d · random %d over %d steps (%.1f ms/step)", score, expertScore, randomScore, steps, msPerStep)
        }
        public var relative: Double { expertScore > randomScore ? Double(score - randomScore) / Double(expertScore - randomScore) : 0 }
    }

    public static func evaluate(model: PolicyModel, mode: ArenaGame.Mode, seconds: Double, seed: UInt64, temperature: Float = 1, useHistory: Bool = true,
                                pixelSize: Int = 480, embed: ((String) -> [Float]?)? = nil, encoder: InstructionEncoder? = nil,
                                trace: ((Int, TimestepAction, String) -> Void)? = nil) -> Result {
        let rate = model.architecture.stepRateHz, n = Int(seconds * Double(rate))
        let size = model.architecture.frameSize
        let game = ArenaGame(mode: mode, seed: seed)
        let stepper = PolicyStepper(model: model)
        stepper.temperature = temperature
        stepper.useActionHistory = useHistory
        let resizer = FrameResizer()!
        let pb = ArenaRecorder.makePixelBuffer(width: pixelSize, height: pixelSize)!
        var textCache: [String: [Float]] = [:]
        let absolute = model.actions.mouseMode == .absolute
        let feed = model.architecture.usesDecoderMemory ? LanguageFeed(architecture: model.architecture, encoder: encoder) : nil
        let ocrEvery = max(1, Int(ScreenTextReader.interval * Double(rate)))
        var elapsed = 0.0
        for i in 0 ..< n {
            game.render(into: pb)
            if let feed {
                feed.setInstruction(game.instruction)
                if model.architecture.screenText, i % ocrEvery == 0 { feed.setScreenText(ScreenTextReader.read(pb, maxItems: model.architecture.screenTextItems)) }
                if case .some(let l) = feed.update(budget: 32) { stepper.setLanguage(l) }
            }
            let frame: MLXArray = resizer.resize(pb, size: size, cursor: absolute ? game.pointer : nil) { buf in
                MLXArray(Data(buf), [size, size, 3], dtype: .uint8)
            }!
            var text: [Float]?
            if let ins = game.instruction, let embed {
                if textCache[ins] == nil { textCache[ins] = embed(ins) }
                text = textCache[ins]
            }
            let t0 = Date()
            let r = stepper.step(frame: frame, text: text)
            elapsed += Date().timeIntervalSince(t0)
            var a = model.actions.decode(r.classes)
            if !absolute { a.absX = game.pointer.x; a.absY = game.pointer.y }
            trace?(game.steps, a, game.debugState + " expert \(game.expertAction().keys.sorted())")
            game.step(a, dt: 1 / Double(rate))
        }
        let expert = ArenaGame(mode: mode, seed: seed)
        for _ in 0 ..< n { expert.step(expert.expertAction(), dt: 1 / Double(rate)) }
        let random = ArenaGame(mode: mode, seed: seed)
        var rng = SplitMix64(seed &+ 99)
        for _ in 0 ..< n {
            var a = TimestepAction()
            a.keys = Set(["w", "a", "s", "d"].filter { _ in rng.next() % 3 == 0 })
            a.dx = Double(Int(rng.next() % 61) - 30); a.dy = Double(Int(rng.next() % 41) - 20)
            a.absX = Double(rng.next() % 1000) / 1000; a.absY = Double(rng.next() % 1000) / 1000
            if rng.next() % 6 == 0 { a.pointer = [.left] }
            random.step(a, dt: 1 / Double(rate))
        }
        return Result(score: game.score, expertScore: expert.score, randomScore: random.score, steps: n, msPerStep: elapsed / Double(n) * 1000)
    }
}

extension ArenaEvaluator {
    /// Open-loop check on fresh simulator frames while the expert plays: how often the model's
    /// greedy key choice matches the expert, given the true history vs. a blanked history.
    public static func diagnose(model: PolicyModel, mode: ArenaGame.Mode, seconds: Double, seed: UInt64, pixelSize: Int = 480) -> (withHistory: Double, blankHistory: Double, expertKeyRate: Double) {
        let rate = model.architecture.stepRateHz, n = Int(seconds * Double(rate)), size = model.architecture.frameSize
        let game = ArenaGame(mode: mode, seed: seed)
        let a = PolicyStepper(model: model), b = PolicyStepper(model: model)
        a.temperature = 0; b.temperature = 0
        let resizer = FrameResizer()!, pb = ArenaRecorder.makePixelBuffer(width: pixelSize, height: pixelSize)!
        let blank = [Int32](repeating: 0, count: model.actions.slotCount).enumerated().map { i, _ in
            i == model.actions.keySlots + model.actions.pointerSlots ? Int32(model.actions.zeroXClass) : i == model.actions.keySlots + model.actions.pointerSlots + 1 ? Int32(model.actions.zeroYClass) : 0
        }
        var prev: [Int32]?
        var okA = 0, okB = 0, pressed = 0
        for _ in 0 ..< n {
            game.render(into: pb)
            let frame: MLXArray = resizer.resize(pb, size: size) { MLXArray(Data($0), [size, size, 3], dtype: .uint8) }!
            let expert = game.expertAction()
            let ra = a.step(frame: frame, text: nil, previousAction: prev)
            let rb = b.step(frame: frame, text: nil, previousAction: blank)
            let ka = model.actions.decode(ra.classes).keys, kb = model.actions.decode(rb.classes).keys
            if ka == expert.keys { okA += 1 }
            if kb == expert.keys { okB += 1 }
            if !expert.keys.isEmpty { pressed += 1 }
            prev = model.actions.encode(expert)
            game.step(expert, dt: 1 / Double(rate))
        }
        return (Double(okA) / Double(n), Double(okB) / Double(n), Double(pressed) / Double(n))
    }
}
