import ATCore
import ATLanguage
import Foundation

enum EmbedCommand {
    static func run(_ args: [String]) throws {
        guard let enc = LanguageModels.instructionEncoder(pretrainedRoot: AppPaths.standard.pretrainedRoot) else {
            print("EmbeddingGemma is not downloaded"); exit(1)
        }
        let texts = args.isEmpty ? ["go to the red square", "move to the red box", "go to the blue square", "open the settings menu"] : args
        let t0 = Date()
        let e = try enc.embed(texts)
        print(String(format: "embedded %d texts in %.0f ms, dim %d", texts.count, Date().timeIntervalSince(t0) * 1000, e[0].count))
        func cos(_ a: [Float], _ b: [Float]) -> Float {
            let d = zip(a, b).map(*).reduce(0, +)
            return d / (sqrt(a.map { $0 * $0 }.reduce(0, +)) * sqrt(b.map { $0 * $0 }.reduce(0, +)))
        }
        for i in texts.indices { print(String(format: "  |%@| norm %.2f", texts[i], sqrt(e[i].map { $0 * $0 }.reduce(0, +)))) }
        for i in texts.indices { for j in texts.indices where j > i { print(String(format: "  cos(%d,%d) = %.3f", i, j, cos(e[i], e[j]))) } }
    }
}
