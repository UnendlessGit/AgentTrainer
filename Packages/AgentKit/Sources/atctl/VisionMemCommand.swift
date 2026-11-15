import ATCore
import ATModel
import Foundation
import MLX
import MLXNN
import MLXRandom

/// Peak memory of vision forward+backward with and without checkpointing.
enum VisionMemCommand {
    static func run(_ args: [String]) throws {
        let o = Options(args)
        let n = o.int("--frames", 120)
        let model = PolicyModel(architecture: .p2p150M, actions: .gameDefault)
        model.setTraining()
        model.gradientCheckpointing = o.flag("--ckpt")
        model.visionGradientFrames = o.int("--k", 32)
        let frames = MLXRandom.randInt(low: 0, high: 256, [n, 192, 192, 3]).asType(.uint8)
        eval(frames)
        let vg = valueAndGrad(model: model) { (m: PolicyModel, a: [MLXArray]) -> [MLXArray] in [m.debugImageTokens(a[0]).square().mean()] }
        GPU.resetPeakMemory()
        let t0 = Date()
        let (l, g) = vg(model, [frames])
        eval(l, g)
        print(String(format: "frames %d k %d: %.2f s, peak %.2f GB", n, model.visionGradientFrames, Date().timeIntervalSince(t0), Double(GPU.peakMemory) / 1e9))
    }
}
