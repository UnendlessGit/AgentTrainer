import ATCore
import ATModel
import Foundation
import MLX

/// Rough per-stage timing of one real-time step (each stage synchronised separately).
enum ProfileCommand {
    static func run(_ args: [String]) throws {
        let o = Options(args)
        var arch = PolicyArchitecture.p2p150M
        arch.layers = o.int("--layers", 10)
        let model = PolicyModel(architecture: arch, actions: .gameDefault)
        model.train(false)
        model.convert(to: .bfloat16)
        let frames = PolicyModel.randomFrames(4, size: 192)
        eval(frames)
        func time(_ label: String, _ n: Int = 100, _ body: () -> Void) {
            for _ in 0 ..< 10 { body() }
            let t0 = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< n { body() }
            print(String(format: "%-28@ %.2f ms", label as NSString, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6 / Double(n)))
        }
        time("vision (1 frame)") { eval(model.rawImageTokens(frames[0 ..< 1])) }
        let stepper = PolicyStepper(model: model)
        for i in 0 ..< arch.contextSteps + 5 { _ = stepper.step(frame: frames[i % 4], text: nil) }
        time("full step (compiled decode)") { _ = stepper.step(frame: frames[0], text: nil) }
        stepper.temperature = 1
        time("full step + logits (eager)") { _ = stepper.step(frame: frames[0], text: nil, returnLogits: true) }
    }
}
