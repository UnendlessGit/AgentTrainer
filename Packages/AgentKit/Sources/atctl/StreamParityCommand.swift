import ATCore
import ATModel
import Foundation
import MLX
import MLXRandom

/// Checks that the real-time stepper (KV cache, sliding window) reproduces the batch forward.
enum StreamParityCommand {
    static func run(_ args: [String]) throws {
        let opts = Options(args)
        let T = opts.int("--steps", 12), W = opts.int("--window", 5)
        var arch = PolicyArchitecture.p2p150M
        arch.layers = opts.int("--layers", 4)
        arch.historySteps = Array(repeating: W, count: arch.layers)
        let actions = ActionSpaceConfig.gameDefault
        MLXRandom.seed(7)
        let model = PolicyModel(architecture: arch, actions: actions)
        model.train(false)
        let S = actions.slotCount
        let frames = PolicyModel.randomFrames(T, size: arch.frameSize).reshaped(1, T, arch.frameSize, arch.frameSize, 3)
        let counts = actions.slotClassCounts
        let acts = stacked((0 ..< S).map { MLXRandom.randInt(low: 0, high: counts[$0], [1, T]).asType(.int32) }, axis: -1)
        let text = MLXRandom.normal([1, T, 768]) * 0.05
        let present = MLXArray((0 ..< T).map { $0 % 3 != 0 }).reshaped(1, T)
        let batch = model(frames: frames, actionsIn: acts, text: text, textPresent: present)
        eval(batch.keys, batch.pointer, batch.x, batch.y)

        let stepper = PolicyStepper(model: model, contextSteps: W)
        if let r = opts.value("--rebase").flatMap(Int.init) { stepper.rebaseThreshold = r }
        let a = acts.asArray(Int32.self), pres = present.asArray(Bool.self)
        var worst: Float = 0
        for t in 0 ..< T {
            let tv: [Float]? = pres[t] ? text[0, t].asArray(Float.self) : nil
            let r = stepper.step(frame: frames[0, t], text: tv,
                                 previousAction: t > 0 ? Array(a[(t - 1) * S ..< t * S]) : nil,
                                 forced: Array(a[t * S ..< (t + 1) * S]), returnLogits: true)
            let dA = abs(stepper.lastAOutState!.reshaped(-1) - model.lastAOutStates![0, t]).max().item(Float.self)
            if opts.flag("--verbose") { print("t=\(t) a_out Δ=\(dA)") }
            for s in 0 ..< S {
                let ref: MLXArray
                switch actions.kind(ofSlot: s) {
                case .key: ref = batch.keys[0, t, s]
                case .pointer: ref = batch.pointer[0, t, s - actions.keySlots]
                case .x: ref = batch.x[0, t, 0]
                case .y: ref = batch.y[0, t, 0]
                }
                let d = abs(MLXArray(r.logits![s]) - ref).max().item(Float.self)
                worst = max(worst, d)
            }
        }
        print("stream vs batch over \(T) steps (window \(W)): max |Δlogit| = \(worst)")
        print(worst < 1e-3 ? "STREAM PARITY OK" : "STREAM PARITY FAILED")
        if worst >= 1e-3 { exit(2) }
    }
}

struct Options {
    let args: [String]
    init(_ a: [String]) { args = a }
    func value(_ k: String) -> String? { args.firstIndex(of: k).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
    func int(_ k: String, _ d: Int) -> Int { value(k).flatMap(Int.init) ?? d }
    func double(_ k: String, _ d: Double) -> Double { value(k).flatMap(Double.init) ?? d }
    func flag(_ k: String) -> Bool { args.contains(k) }
}
