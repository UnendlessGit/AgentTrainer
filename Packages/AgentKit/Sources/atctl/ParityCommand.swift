import ATCore
import ATModel
import Foundation
import MLX

enum ParityCommand {
    static func run(checkpoint: URL, reference: URL) throws {
        let ref = try loadArrays(url: reference)
        let frames = ref["frames"]!, actions = ref["actions"]!.asType(.int32)
        let text = ref["text"]!, present = ref["present"]!.asType(.bool)
        let T = frames.dim(1)
        var arch = PolicyArchitecture.p2p150M
        arch.memorySeconds = Double(T) / Double(arch.stepRateHz)
        let model = PolicyModel(architecture: arch, actions: .p2p)
        let t0 = Date()
        let report = try P2PImporter.load(checkpoint: checkpoint, into: model)
        print("import: \(report) in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")
        if !report.freshlyInitialized.isEmpty { print("  fresh: \(report.freshlyInitialized)") }
        model.train(false)

        let img = model.rawImageTokens(frames.reshaped(T, 192, 192, 3))
        compare("image tokens", img, ref["img_tokens"]!)
        let logits = model(frames: frames, actionsIn: actions, text: text, textPresent: present)
        for k in ["seq_in", "layer0", "seq_out"] { compare(k, model.debugTrace[k]!.reshaped(-1, 1024), ref[k]!) }
        if let dump = ProcessInfo.processInfo.environment["PARITY_DUMP"] {
            try save(arrays: model.debugTrace.mapValues { $0.reshaped(-1, 1024) }, url: URL(fileURLWithPath: dump))
        }
        compare("a_out states", model.lastAOutStates!.reshaped(T, -1), ref["a_out"]!)
        var worst: Float = 0
        worst = max(worst, compare("key logits", logits.keys, ref["logits_keys"]!))
        worst = max(worst, compare("pointer logits", logits.pointer, ref["logits_pointer"]!))
        worst = max(worst, compare("mouse x logits", logits.x, ref["logits_x"]!))
        worst = max(worst, compare("mouse y logits", logits.y, ref["logits_y"]!))
        let argmaxMatch = (argMax(logits.keys, axis: -1) .== argMax(ref["logits_keys"]!, axis: -1)).all().item(Bool.self)
        print(worst < 1e-3 && argmaxMatch ? "PARITY OK (max abs diff \(worst))" : "PARITY FAILED (max abs diff \(worst))")
        if !(worst < 1e-3 && argmaxMatch) { exit(2) }
    }

    @discardableResult
    static func compare(_ name: String, _ a: MLXArray, _ b: MLXArray) -> Float {
        let d = abs(a.asType(.float32) - b.asType(.float32))
        let maxDiff = d.max().item(Float.self)
        let scale = abs(b).max().item(Float.self)
        print(String(format: "  %-16@ max|Δ| %.2e   (max|ref| %.3f)  shape %@", name as NSString, maxDiff, scale, "\(a.shape)" as NSString))
        return maxDiff
    }
}
