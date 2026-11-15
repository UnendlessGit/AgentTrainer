import ATCore
import ATLanguage
import Foundation

/// atctl label <recording.atrec> [--window 20]
enum LabelCommand {
    static func run(_ args: [String]) throws {
        let o = Options(args)
        guard let path = args.first else { print("usage: atctl label <recording.atrec>"); return }
        let dir = LanguageModels.directory(LanguageModels.labeler, pretrainedRoot: AppPaths.standard.pretrainedRoot)
        let t0 = Date()
        let labeler = try AutoLabeler(directory: dir)
        print(String(format: "loaded labeller in %.1f s", Date().timeIntervalSince(t0)))
        var req = AutoLabelRequest(recordings: [URL(fileURLWithPath: path)], modelDirectory: dir)
        req.windowSeconds = o.double("--window", 20)
        let t1 = Date()
        let segs = try labeler.label(recording: URL(fileURLWithPath: path), request: req) { f, m in print(String(format: "  %3.0f%% %@", f * 100, m)) }
        print(String(format: "labelled in %.1f s:", Date().timeIntervalSince(t1)))
        for s in segs { print(String(format: "  %5.1f–%5.1f  %@", s.start, s.end, s.text)) }
    }
}
