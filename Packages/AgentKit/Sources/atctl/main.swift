import ATCore
import ATModel
import ATTraining
import Foundation
import MLX

/// Developer command line for AgentTrainer: parity checks, benchmarks, conversions.
let args = Array(CommandLine.arguments.dropFirst())
let command = args.first ?? "help"
let rest = Array(args.dropFirst())

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

do {
    switch command {
    case "smoke":
        print(MLXSmokeTest.run())
    case "elkborn-decode":
        try ElkbornCommand.run(rest)
    case "parity":
        guard rest.count >= 2 else { fail("usage: atctl parity <checkpoint.ckpt> <reference.safetensors>") }
        try ParityCommand.run(checkpoint: URL(fileURLWithPath: rest[0]), reference: URL(fileURLWithPath: rest[1]))
    case "stream-parity":
        try StreamParityCommand.run(rest)
    case "profile":
        try ProfileCommand.run(rest)
    case "train-bench":
        try TrainBenchCommand.run(rest)
    case "vision-mem":
        try VisionMemCommand.run(rest)
    case "arena-gen":
        try ArenaCommands.generate(rest)
    case "train":
        try ArenaCommands.train(rest)
    case "arena-correct":
        try ArenaCommands.corrections(rest)
    case "arena-eval":
        try ArenaCommands.evaluate(rest)
    case "embed":
        try EmbedCommand.run(rest)
    case "label":
        try LabelCommand.run(rest)
    case "attn-bench":
        AttentionBench.run(batch: Int(rest.first ?? "3") ?? 3, steps: Int(rest.dropFirst().first ?? "300") ?? 300, width: 768, heads: 12)
    case "parts-bench":
        LayerPartsBench.run(batch: 3, steps: 300, width: 768, heads: 12)
    case "stem-bench":
        StemBench.run(frames: Int(rest.first ?? "300") ?? 300)
    case "conv-grad":
        DepthwiseCheck.run()
        ConvGradBench.run(n: Int(rest.first ?? "96") ?? 96)
    case "params":
        for (d, l, h, dl, dh) in [(128, 4, 2, 2, 2), (192, 4, 3, 2, 3), (192, 6, 3, 2, 3), (256, 4, 4, 2, 4), (256, 6, 4, 2, 4), (320, 6, 5, 2, 5), (384, 6, 6, 2, 6), (768, 6, 12, 2, 6)] {
            var a = PolicyArchitecture(); a.width = d; a.layers = l; a.heads = h; a.kvHeads = h; a.decoderLayers = dl; a.decoderHeads = dh
            let m = PolicyModel(architecture: a, actions: .gameDefault)
            let n = m.parameters().flattened().filter { !$0.0.contains("running_") }.reduce(0) { $0 + $1.1.size }
            let stem = m.parameters().flattened().filter { $0.0.hasPrefix("vision.stem") && !$0.0.contains("running_") }.reduce(0) { $0 + $1.1.size }
            let proj = m.parameters().flattened().filter { $0.0.hasPrefix("vision.proj") }.reduce(0) { $0 + $1.1.size }
            print(String(format: "D %4d L %d heads %d dec %d×%d: %6.2fM total (stem %.2fM, image proj %.2fM), estimate %.2fM", d, l, h, dl, dh, Double(n) / 1e6, Double(stem) / 1e6, Double(proj) / 1e6, Double(a.parameterCount(actions: .gameDefault)) / 1e6))
        }
    case "opt-check":
        OptimizerCheck.run()
    case "aug-check":
        AugmentCheck.run()
    case "stem-ab":
        StemBench.ab(frames: Int(rest.first ?? "300") ?? 300)
    case "bench":
        try BenchCommand.run(rest)
    default:
        print("""
        atctl smoke                                   MLX sanity check
        atctl parity <ckpt> <reference.safetensors>   compare the MLX port with P2P's PyTorch output
        atctl bench [--layers N] [--ckpt path] [--seconds S] [--hz R] [--steps N] [--bf16]
                                                      streaming inference latency
        """)
    }
} catch {
    fail("error: \(error.localizedDescription)")
}
