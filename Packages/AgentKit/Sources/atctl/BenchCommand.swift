import ATCore
import ATCapture
import ATModel
import Foundation
import MLX

/// Measures real-time step latency with a full temporal-memory window.
enum BenchCommand {
    static func run(_ args: [String]) throws {
        let o = Options(args)
        var arch = PolicyArchitecture.p2p150M
        arch.layers = o.int("--layers", 10)
        arch.width = o.int("--width", 1024)
        arch.heads = arch.width / 64
        arch.kvHeads = arch.heads
        arch.stepRateHz = o.int("--hz", 30)
        arch.memorySeconds = o.double("--seconds", 10)
        let actions = o.flag("--p2p-actions") ? ActionSpaceConfig.p2p : ActionSpaceConfig.gameDefault
        let model = PolicyModel(architecture: arch, actions: actions)
        if let ckpt = o.value("--ckpt") { try P2PImporter.load(checkpoint: URL(fileURLWithPath: ckpt), into: model) }
        model.train(false)
        if !o.flag("--fp32") { model.convert(to: .bfloat16) }
        if o.flag("--q8") { model.quantizeForInference() }
        let stepper = PolicyStepper(model: model)
        let frames = PolicyModel.randomFrames(8, size: arch.frameSize)
        eval(frames)
        let warm = arch.contextSteps + 20
        let n = o.int("--steps", 300)
        print("model: \(arch.layers) layers × \(arch.width), \(arch.contextSteps) steps memory (\(arch.memorySeconds)s @ \(arch.stepRateHz) Hz), dtype \(model.computeDType)\(o.flag("--q8") ? " q8" : "")")
        print("warming up (\(warm) steps to fill memory)…")
        for i in 0 ..< warm { _ = stepper.step(frame: frames[i % 8], text: nil) }
        var times: [Double] = []
        let paced = o.int("--paced", 0)
        let warmer = o.flag("--warm") ? GPUWarmer() : nil
        let body = {
            var next = HostClock.now()
            for i in 0 ..< n {
                if paced > 0 { next += UInt64(1_000_000_000 / paced); HostClock.sleep(untilNanoseconds: next) }
                warmer?.finish()
                let t0 = DispatchTime.now().uptimeNanoseconds
                _ = stepper.step(frame: frames[i % 8], text: nil, previousAction: o.flag("--prev") ? [0, 0, 0, 0, 0, 0, 11, 8] : nil)
                times.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
                if paced > 0 { warmer?.fill(until: next + UInt64(1_000_000_000 / paced)) }
            }
        }
        if o.flag("--thread") {
            let done = DispatchSemaphore(value: 0)
            let t = Thread { body(); done.signal() }
            t.qualityOfService = .userInteractive
            t.stackSize = 16 << 20
            t.start()
            done.wait()
        } else { body() }
        times.sort()
        func p(_ q: Double) -> Double { times[min(times.count - 1, Int(Double(times.count) * q))] }
        let mean = times.reduce(0, +) / Double(times.count)
        print(String(format: "step latency ms: mean %.2f  p50 %.2f  p90 %.2f  p99 %.2f  max %.2f  →  %.0f FPS (mean)",
                     mean, p(0.5), p(0.9), p(0.99), times.last!, 1000 / mean))
        print(String(format: "GPU memory: active %.0f MB, peak %.0f MB", Double(GPU.activeMemory) / 1e6, Double(GPU.peakMemory) / 1e6))
    }
}
