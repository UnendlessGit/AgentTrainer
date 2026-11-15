import ATCapture
import ATCore
import ATLanguage
import ATRuntime
import SwiftUI

struct ArenaView: View {
    @Bindable var app: AppModel
    @State private var mode = ArenaGame.Mode.chase
    @State private var minutes = 10.0
    @State private var generating = false
    @State private var evaluating = false
    @State private var result: String?
    @State private var syncResult: String?
    @State private var selectedModel: UUID?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                PageHeader(title: "Test Arena", subtitle: "Small built-in games with a scripted expert, for checking that recording, training and running really work before you use your own games.")
                Card(title: "Game", symbol: "gamecontroller") {
                    Picker("Mode", selection: $mode) { ForEach(ArenaGame.Mode.allCases) { Text($0.title).tag($0) } }
                    Text(description(mode)).font(.callout).foregroundStyle(.secondary)
                    Button("Open a playable window") { ArenaWindows.show(ArenaGameView(mode: mode), title: "AgentTrainer Test Arena — \(mode.title)") }
                    Text("The window can be recorded in the Recordings tab and controlled from the Run tab, exactly like a real game.").font(.caption).foregroundStyle(.secondary)
                }
                HStack(alignment: .top, spacing: 18) {
                    Card(title: "Expert demonstrations", symbol: "person.fill.checkmark") {
                        LabeledSlider(title: "Length", value: $minutes, range: 2 ... 60, step: 1, format: { "\(Int($0)) min" })
                        Button(generating ? "Generating…" : "Add to Library") { Task { await generate() } }.disabled(generating)
                        Text("Writes real recordings (video, input timeline, training frames, instructions) into Library/Test Arena.").font(.caption).foregroundStyle(.secondary)
                    }
                    Card(title: "Evaluate a model", symbol: "flag.checkered") {
                        Picker("Model", selection: $selectedModel) {
                            ForEach(app.models.models.filter { !$0.checkpoints.isEmpty }) { Text($0.definition.name).tag(Optional($0.id)) }
                        }
                        Button(evaluating ? "Playing…" : "Play 3 × 60 s") { Task { await evaluate() } }.disabled(evaluating || selectedModel == nil)
                        if let result { Text(result).font(.callout.monospaced()).fixedSize(horizontal: false, vertical: true) }
                        Text("Runs the model closed-loop in the simulator (it sees rendered frames and its own actions drive the game) and compares with the expert and with random input.").font(.caption).foregroundStyle(.secondary)
                    }
                }
                Card(title: "Sync check", symbol: "metronome") {
                    HStack {
                        Button("Open Sync Check window") { ArenaWindows.show(SyncCheckView(), title: "AgentTrainer Sync Check") }
                        Button("Analyze last recording") { analyzeSync() }.disabled(app.recorder.lastSaved == nil)
                    }
                    if let syncResult { Text(syncResult).font(.callout.monospaced()) }
                    Text("Record the Sync Check window, press keys about once a second, stop, then analyze. It reports how long after each key event the white flash shows up in a captured frame. A steady value (low jitter) means frames and input are on one clock.").font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(24)
        }
        .onAppear { if selectedModel == nil { selectedModel = app.models.models.first(where: { !$0.checkpoints.isEmpty })?.id } }
    }

    func description(_ m: ArenaGame.Mode) -> String {
        switch m {
        case .chase: "Move the white dot onto the red square with W A S D. Tests vision → keyboard."
        case .aim: "Turn with the mouse to put the red target under the crosshair, then click. Tests relative mouse and clicks."
        case .colorChase: "Two squares; the instruction says which color to reach. Tests instruction following."
        case .clickButton: "Click the button the instruction names. Tests reading on-screen text and pointer control (App control models)."
        case .typeWord: "Type the word in the instruction and press Return. Tests literal typing (App control models)."
        }
    }

    func generate() async {
        generating = true
        defer { generating = false }
        let folder = app.library.fs.root.appendingPathComponent("Test Arena", isDirectory: true).appendingPathComponent(mode.title, isDirectory: true)
        let m = mode, total = minutes
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let episodes = max(1, Int(total / 3))
        await Task.detached {
            for e in 0 ..< episodes {
                _ = try? ArenaRecorder.generate(mode: m, seconds: total * 60 / Double(episodes), seed: UInt64(Date().timeIntervalSince1970) &+ UInt64(e),
                                                destination: folder, name: "\(m.rawValue) \(Date().formatted(.dateTime.hour().minute().second())) \(e + 1)")
            }
        }.value
        app.library.refresh()
    }

    func evaluate() async {
        guard let e = app.models.entry(selectedModel) else { return }
        evaluating = true
        defer { evaluating = false }
        let url = e.url, m = mode, root = app.settings.paths.pretrainedRoot
        result = await Task.detached {
            do {
                let (model, _, ckpt) = try ModelLoader.load(url)
                let enc = LanguageModels.instructionEncoder(pretrainedRoot: root)
                var lines: [String] = []
                var rel = 0.0
                for s in 0 ..< 3 {
                    let r = ArenaEvaluator.evaluate(model: model, mode: m, seconds: 60, seed: UInt64(9000 + s), embed: { t in try? enc?.embed([t]).first }, encoder: enc)
                    rel += r.relative / 3
                    lines.append("Episode \(s + 1): model \(r.score) · expert \(r.expertScore) · random \(r.randomScore)")
                }
                return lines.joined(separator: "\n") + String(format: "\n%@ → %.0f%% of the expert's performance", ckpt, rel * 100)
            } catch { return error.localizedDescription }
        }.value
    }

    func analyzeSync() {
        guard let u = app.recorder.lastSaved else { return }
        syncResult = (try? SyncCheck.analyze(recording: u))?.summary ?? "Couldn't read that recording."
    }
}
