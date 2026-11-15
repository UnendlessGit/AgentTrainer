import ATCore
import ATModel
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
    case "parity":
        guard rest.count >= 2 else { fail("usage: atctl parity <checkpoint.ckpt> <reference.safetensors>") }
        try ParityCommand.run(checkpoint: URL(fileURLWithPath: rest[0]), reference: URL(fileURLWithPath: rest[1]))
    case "stream-parity":
        try StreamParityCommand.run(rest)
    case "profile":
        try ProfileCommand.run(rest)
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
