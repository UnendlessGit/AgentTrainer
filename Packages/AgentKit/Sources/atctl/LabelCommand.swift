import ATCore
import ATLanguage
import Foundation

/// atctl label <recording.atrec> [--model best|light] [--detail fine|normal|broad] [--variations 5]
///                                [--no-check] [--guide "what the footage is"] [--seconds 60] [--write]
/// Prints what the labeller sees and writes, with timings. Writes the recording only with --write.
enum LabelCommand {
    static func run(_ args: [String]) throws {
        let o = Options(args)
        guard let path = args.first else { print("usage: atctl label <recording.atrec> [--model best|light] [--write]"); return }
        let model = o.value("--model") == "light" ? LanguageModels.labelerLight : LanguageModels.labelerBest
        let dir = LanguageModels.directory(model, pretrainedRoot: AppPaths.standard.pretrainedRoot)
        let url = URL(fileURLWithPath: path)
        let t0 = Date()
        let labeler = try AutoLabeler(directory: dir)
        print(String(format: "loaded %@ in %.1f s", model.title, Date().timeIntervalSince(t0)))

        var req = AutoLabelRequest(recordings: [url], modelDirectory: dir)
        req.detail = LabelDetail(rawValue: o.value("--detail") ?? "") ?? .normal
        req.variations = o.int("--variations", 5)
        req.doubleCheck = !o.flag("--no-check")
        if let g = o.value("--guide") { req.conversation = [LabelChatMessage(role: .user, text: g)] }
        let limit = o.double("--seconds", .infinity)
        if let px = o.value("--pixels"), let w = Int(px.split(separator: "x").first ?? ""), let h = Int(px.split(separator: "x").last ?? "") { req.maxPixels = w * h }
        req.framesPerSecond = o.double("--fps", req.framesPerSecond)

        // Labels only the first `--seconds` by leaving the rest out.
        var scratch = url
        if limit.isFinite, let meta = try? RecordingIO.readMeta(url), meta.duration > limit {
            scratch = FileManager.default.temporaryDirectory.appendingPathComponent("label-\(UUID().uuidString).atrec")
            try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
            for f in try FileManager.default.contentsOfDirectory(atPath: url.path) where f != RecordingPackage.annotations {
                try FileManager.default.createSymbolicLink(at: scratch.appendingPathComponent(f), withDestinationURL: url.appendingPathComponent(f))
            }
            var ann = RecordingIO.readAnnotations(url)
            ann.excluded.append(TimeRange(start: limit, end: meta.duration + 1))
            try RecordingIO.writeAnnotations(ann, to: scratch)
        }
        defer { if scratch != url { try? FileManager.default.removeItem(at: scratch) } }

        let t1 = Date()
        var last = Date()
        let segs = try labeler.label(recording: scratch, request: req, progress: { p in
            print(String(format: "  %3.0f%% %@", p.fraction * 100, p.status))
        }, onSegment: { s in
            print(String(format: "  [%.1f s] %5.1f–%5.1f  %@", Date().timeIntervalSince(last), s.start, s.end, s.text))
            if let ob = s.observation { print("      seen: \(ob)") }
            for v in s.variants ?? [] { print("      · \(v)") }
            last = Date()
        })
        print(String(format: "labelled %d instructions in %.1f s", segs.count, Date().timeIntervalSince(t1)))
        if o.flag("--write") {
            var ann = RecordingIO.readAnnotations(url)
            ann.segments.removeAll { $0.source == .auto }
            ann.segments = (ann.segments + segs).sorted { $0.start < $1.start }
            try RecordingIO.writeAnnotations(ann, to: url)
            print("written to \(url.lastPathComponent)")
        }
    }
}
