import ATCore
import Charts
import SwiftUI

struct TrainingView: View {
    @Bindable var app: AppModel
    private var tc: TrainingController { app.training }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                PageHeader(title: "Training", subtitle: "Teach a model from your recordings. Training runs on the GPU in a separate engine process.")
                if app.models.models.isEmpty {
                    ContentUnavailableView {
                        Label("No models yet", systemImage: "brain")
                    } description: { Text("Create a model in the AI Models tab first.") } actions: {
                        Button("Go to AI Models") { app.tab = .models }
                    }
                } else if let entry = app.models.entry(app.selectedModelID) ?? app.models.models.first {
                    header(entry)
                    if let e = tc.errorMessage { InlineError(message: e) }
                    if tc.isActive || !tc.progress.isEmpty { liveStats(entry) }
                    HStack(alignment: .top, spacing: 18) {
                        DatasetPicker(app: app, entry: entry).frame(maxWidth: .infinity)
                        CheckpointsCard(app: app, entry: entry).frame(maxWidth: .infinity)
                    }
                    if !tc.progress.isEmpty { LossChart(progress: tc.progress, validation: tc.validation) }
                    if let v = tc.validation.last { ValidationCard(metrics: v, history: tc.validation) }
                    if !tc.logs.isEmpty {
                        Card(title: "Engine log", symbol: "text.alignleft") {
                            ForEach(Array(tc.logs.suffix(12).enumerated()), id: \.offset) { Text($0.element).font(.caption.monospaced()).foregroundStyle(.secondary) }
                        }
                    }
                }
            }
            .padding(24)
        }
        .onAppear { if let e = app.models.entry(app.selectedModelID) ?? app.models.models.first { app.selectedModelID = e.id; tc.loadHistory(for: e) } }
        .onChange(of: app.selectedModelID) { _, id in if let e = app.models.entry(id) { tc.loadHistory(for: e) } }
    }

    @ViewBuilder
    func header(_ entry: ModelEntry) -> some View {
        Card(title: "Session", symbol: "cpu") {
            HStack(spacing: 14) {
                Picker("Model", selection: $app.selectedModelID) {
                    ForEach(app.models.models) { Text($0.definition.name).tag(Optional($0.id)) }
                }
                .frame(maxWidth: 320)
                .disabled(tc.isActive)
                Spacer()
                if tc.isActive && tc.jobModelID == entry.id {
                    if tc.phase == .paused {
                        Button { tc.control("resume") } label: { Label("Resume", systemImage: "play.fill") }.buttonStyle(.borderedProminent)
                    } else {
                        Button { tc.control("pause") } label: { Label("Pause", systemImage: "pause.fill") }
                    }
                    Button { tc.control("checkpoint") } label: { Label("Save checkpoint", systemImage: "square.and.arrow.down") }
                    Button(role: .destructive) { tc.control("stop") } label: { Label("Stop", systemImage: "stop.fill") }
                } else {
                    let ready = app.downloads.isReady(entry.definition.initialization) || !entry.checkpoints.isEmpty
                    Button { tc.start(model: entry) } label: {
                        Label(entry.checkpoints.isEmpty ? "Start training" : "Continue training", systemImage: "play.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(tc.isActive || !ready)
                    .help(ready ? "" : "Download the Pixel2Play weights in AI Models first")
                }
            }
            HStack(spacing: 8) {
                if tc.isActive { ProgressView().controlSize(.small) }
                Text(statusText(entry)).foregroundStyle(.secondary)
            }
        }
    }

    func statusText(_ entry: ModelEntry) -> String {
        if tc.isActive, tc.jobModelID != entry.id { return "Another model is training." }
        if let p = tc.phase { return "\(p.rawValue.capitalized): \(tc.message)" }
        return entry.definition.isTrained ? "Trained for \(entry.definition.trainedSteps) steps. Continue to train more, or start fresh by duplicating the model." : "Ready. Choose recordings and press Start."
    }

    @ViewBuilder
    func liveStats(_ entry: ModelEntry) -> some View {
        let p = tc.progress.last
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 10)], spacing: 10) {
            StatTile(title: "Step", value: p.map { "\($0.step) / \($0.totalSteps)" } ?? "–", symbol: "number")
            StatTile(title: "Loss", value: p.map { String(format: "%.4f", $0.loss) } ?? "–", symbol: "chart.line.downtrend.xyaxis")
            StatTile(title: "Epoch", value: p.map { String(format: "%.2f", $0.epoch) } ?? "–", symbol: "repeat")
            StatTile(title: "Speed", value: p.map { String(format: "%.0f frames/s", $0.framesPerSecond) } ?? "–", symbol: "speedometer")
            StatTile(title: "Time left", value: tc.eta.map { $0.durationText } ?? "–", symbol: "hourglass")
            StatTile(title: "Learning rate", value: p.map { String(format: "%.1e", $0.learningRate) } ?? "–", symbol: "slider.horizontal.3")
            StatTile(title: "GPU", value: tc.gpuUtilization.map { "\(Int($0))%" } ?? "–", symbol: "gpu")
            StatTile(title: "GPU memory", value: p.map { String(format: "%.1f GB (peak %.1f)", $0.gpuMemoryGB, $0.peakMemoryGB) } ?? "–", symbol: "memorychip")
            StatTile(title: "Thermals", value: thermalText, symbol: "thermometer.medium", tint: tc.thermal.rawValue >= 2 ? .orange : .accentColor)
            if let d = tc.dataset {
                StatTile(title: "Data", value: "\(d.trainSeconds.durationText) train · \(d.validationSeconds.durationText) held out", symbol: "film.stack")
                StatTile(title: "Batch", value: "\(d.batchSize) clip\(d.batchSize > 1 ? "s" : "") × \(d.accumulation)", symbol: "square.stack.3d.up")
                StatTile(title: "Instructions", value: "\(Int(d.instructionCoverage * 100))% of steps", symbol: "text.bubble")
            }
        }
    }

    var thermalText: String {
        switch tc.thermal {
        case .nominal: "Normal"
        case .fair: "Warm"
        case .serious: "Hot (throttling)"
        case .critical: "Critical"
        @unknown default: "–"
        }
    }
}

struct DatasetPicker: View {
    let app: AppModel
    let entry: ModelEntry
    var body: some View {
        let sel = app.training.selection[entry.id] ?? []
        let all = app.library.allRecordings()
        let chosen = all.filter { sel.contains($0.url) }
        Card(title: "Recordings to learn from", symbol: "film.stack") {
            Text("\(chosen.count) selected · \(chosen.reduce(0) { $0 + $1.meta.duration }.durationText) · \(Int(entry.definition.training.validationFraction * 100))% of recordings held out for validation")
                .font(.callout).foregroundStyle(.secondary)
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    FolderChecklist(app: app, entry: entry, folder: app.library.tree, depth: 0, all: all)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minHeight: 160, maxHeight: 280)
            .disabled(app.training.isActive)
        }
    }
}

struct FolderChecklist: View {
    let app: AppModel
    let entry: ModelEntry
    let folder: LibraryFolder
    let depth: Int
    let all: [LibraryRecording]

    var body: some View {
        let inFolder = all.filter { $0.folder.standardizedFileURL.path.hasPrefix(folder.url.standardizedFileURL.path) }
        let direct = all.filter { $0.folder.standardizedFileURL == folder.url.standardizedFileURL }
        let sel = app.training.selection[entry.id] ?? []
        let on = !inFolder.isEmpty && inFolder.allSatisfy { sel.contains($0.url) }
        Toggle(isOn: Binding(get: { on }, set: { v in
            var s = app.training.selection[entry.id] ?? []
            for r in inFolder { if v { s.insert(r.url) } else { s.remove(r.url) } }
            app.training.selection[entry.id] = s
            app.training.saveSelection()
        })) {
            Label(depth == 0 ? "Whole library" : folder.name, systemImage: "folder").font(.callout.weight(.medium))
        }
        .toggleStyle(.checkbox)
        .padding(.leading, CGFloat(depth) * 16)
        .disabled(inFolder.isEmpty)
        ForEach(direct) { r in
            Toggle(isOn: Binding(get: { sel.contains(r.url) }, set: { v in
                var s = app.training.selection[entry.id] ?? []
                if v { s.insert(r.url) } else { s.remove(r.url) }
                app.training.selection[entry.id] = s
                app.training.saveSelection()
            })) {
                Text("\(r.meta.name) · \(r.meta.duration.durationText)").font(.callout)
            }
            .toggleStyle(.checkbox)
            .padding(.leading, CGFloat(depth + 1) * 16)
        }
        ForEach(folder.children) { FolderChecklist(app: app, entry: entry, folder: $0, depth: depth + 1, all: all) }
    }
}

struct CheckpointsCard: View {
    let app: AppModel
    let entry: ModelEntry

    static func label(_ name: String, url: URL) -> String {
        if name == "best" {
            guard let st = ModelPackage.checkpointState(url, name: name) else { return "Best" }
            return st.validationLoss.map { String(format: "Best · step %d (held-out loss %.3f)", st.step, $0) } ?? "Best · step \(st.step)"
        }
        guard let n = Int(name.replacingOccurrences(of: "step-", with: "")) else { return name }
        return "Step \(n)"
    }
    var body: some View {
        Card(title: "Checkpoints", symbol: "externaldrive") {
            if entry.checkpoints.isEmpty {
                Text("Checkpoints appear here while training (every \(Int(entry.definition.training.checkpointEveryMinutes)) min, on pause and at the end).")
                    .font(.callout).foregroundStyle(.secondary)
            }
            ForEach(entry.checkpoints.reversed(), id: \.self) { name in
                let active = (entry.definition.activeCheckpoint ?? entry.checkpoints.last) == name
                HStack {
                    Image(systemName: active ? "checkmark.circle.fill" : "circle").foregroundStyle(active ? .green : .secondary)
                    Text(Self.label(name, url: entry.url)).monospacedDigit()
                    Spacer()
                    Button("Use for Run") {
                        var d = entry.definition; d.activeCheckpoint = name; app.models.save(d)
                    }
                    .disabled(active)
                    Button("Resume from") { app.training.start(model: entry, resumeFrom: name) }
                        .disabled(app.training.isActive)
                }
                .font(.callout)
            }
        }
    }
}

struct LossChart: View {
    let progress: [TrainingProgress]
    let validation: [ValidationMetrics]

    var body: some View {
        let smoothed = Self.smooth(progress)
        Card(title: "Loss", symbol: "chart.xyaxis.line") {
            Chart {
                ForEach(smoothed, id: \.0) { s in
                    LineMark(x: .value("Step", s.0), y: .value("Loss", s.1), series: .value("Series", "Training"))
                        .foregroundStyle(by: .value("Series", "Training"))
                }
                ForEach(validation, id: \.step) { v in
                    let total = v.nll.reduce(0, +) / 4
                    PointMark(x: .value("Step", v.step), y: .value("Loss", total))
                        .foregroundStyle(by: .value("Series", "Validation (mean head NLL)"))
                    LineMark(x: .value("Step", v.step), y: .value("Loss", total), series: .value("Series", "Validation"))
                        .foregroundStyle(by: .value("Series", "Validation (mean head NLL)"))
                }
            }
            .chartYScale(type: .log)
            .frame(height: 220)
            Chart {
                ForEach(Array(["Keys", "Mouse buttons & scroll", "Mouse X", "Mouse Y"].enumerated()), id: \.offset) { i, name in
                    ForEach(Self.smoothHead(progress, i), id: \.0) { s in
                        LineMark(x: .value("Step", s.0), y: .value("Loss", s.1), series: .value("Head", name)).foregroundStyle(by: .value("Head", name))
                    }
                }
            }
            .frame(height: 160)
        }
    }

    static func smooth(_ p: [TrainingProgress]) -> [(Int, Double)] {
        var out: [(Int, Double)] = [], s: Double?
        for x in p { s = s.map { $0 * 0.85 + x.loss * 0.15 } ?? x.loss; out.append((x.step, s!)) }
        return out
    }

    static func smoothHead(_ p: [TrainingProgress], _ h: Int) -> [(Int, Double)] {
        var out: [(Int, Double)] = [], s: Double?
        for x in p where x.headLosses.count > h { s = s.map { $0 * 0.85 + x.headLosses[h] * 0.15 } ?? x.headLosses[h]; out.append((x.step, s!)) }
        return out
    }
}

struct ValidationCard: View {
    let metrics: ValidationMetrics
    let history: [ValidationMetrics]

    var body: some View {
        Card(title: "Validation on held-out recordings (step \(metrics.step))", symbol: "checkmark.seal") {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 210), spacing: 10)], spacing: 10) {
                metric("Key press F1", String(format: "%.2f", metrics.keyF1), "Does it press the right keys at the right time (±2 steps)?")
                metric("Click F1", String(format: "%.2f", metrics.clickF1), "Clicks and scrolls at the right moments.")
                metric("Mouse direction", String(format: "%.0f%%", metrics.mouseDirectionAccuracy * 100), "Moves the mouse the same way you did.")
                metric("Mouse correlation", String(format: "%.2f", metrics.mouseCorrelation), "How closely mouse speed matches yours.")
                metric("Beats chance (keys)", String(format: "%.0f%%", metrics.rce[0]), "Relative cross-entropy vs. guessing key frequencies.")
                metric("Beats chance (mouse)", String(format: "%.0f%%", (metrics.rce[2] + metrics.rce[3]) / 2), "Relative cross-entropy for mouse movement.")
                metric("Uses vision", String(format: "%.3f", metrics.visualReliance),
                       "Loss increase when frames are shuffled in time. Near 0 means it copies its own past actions instead of looking (a sign it will play badly).")
                if let h = metrics.historyReliance {
                    metric("Copies itself", String(format: "%.3f", h),
                           "Loss increase when past actions are hidden. Much larger than “Uses vision” means it mostly repeats itself.")
                }
                if let l = metrics.languageReliance {
                    metric("Uses instructions", String(format: "%.3f", l), "Loss decrease when given the right instruction.")
                }
                metric("Perplexity", String(format: "%.3f", metrics.perplexity), "Pixel2Play's validation number (lower is better).")
            }
            if let h = metrics.historyReliance, h > max(0.05, metrics.visualReliance * 3) {
                Label("The model leans on repeating its previous actions more than on what it sees, so it may play poorly even though the scores look good. Raise “Action-history dropout” in the model's settings, record more varied situations, or record corrections in Run.", systemImage: "exclamationmark.triangle.fill")
                    .font(.callout).foregroundStyle(.orange)
            }
            if history.count > 1 {
                Chart {
                    ForEach(history, id: \.step) { v in
                        LineMark(x: .value("Step", v.step), y: .value("Score", v.keyF1), series: .value("M", "Key F1")).foregroundStyle(by: .value("M", "Key F1"))
                        LineMark(x: .value("Step", v.step), y: .value("Score", v.clickF1), series: .value("M", "Click F1")).foregroundStyle(by: .value("M", "Click F1"))
                        LineMark(x: .value("Step", v.step), y: .value("Score", v.mouseDirectionAccuracy), series: .value("M", "Mouse direction")).foregroundStyle(by: .value("M", "Mouse direction"))
                    }
                }
                .chartYScale(domain: 0 ... 1)
                .frame(height: 160)
            }
        }
    }

    func metric(_ title: String, _ value: String, _ help: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title3.weight(.semibold)).monospacedDigit()
            Text(help).font(.caption2).foregroundStyle(.tertiary).fixedSize(horizontal: false, vertical: true)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
    }
}
