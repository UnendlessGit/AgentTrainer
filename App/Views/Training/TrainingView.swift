import ATCore
import Charts
import SwiftUI

struct TrainingView: View {
    @Bindable var app: AppModel
    @State private var chartMode = 0
    private var tc: TrainingController { app.training }

    var body: some View {
        Group {
            if app.models.models.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "brain").font(.system(size: 36, weight: .light)).foregroundStyle(Theme.faint)
                    Text("No models yet").font(Theme.font(17, .semibold))
                    Text("Create a model in the Models tab first.").font(Theme.font(13)).foregroundStyle(Theme.muted)
                    Button("Go to Models") { app.tab = .models }.buttonStyle(.atPrimary).padding(.top, 6)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let entry = app.models.entry(app.selectedModelID) ?? app.models.models.first {
                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {
                        header(entry).riseIn()
                        if let e = tc.errorMessage { InlineError(message: e) }
                        stats(entry).riseIn(0.05)
                        HStack(alignment: .top, spacing: 24) {
                            VStack(spacing: 24) {
                                chartPanel
                                DatasetPicker(app: app, entry: entry)
                            }
                            VStack(spacing: 16) {
                                CheckpointsCard(app: app, entry: entry)
                                earlyStopping(entry)
                                if let v = tc.validation.last { ValidationCard(metrics: v, history: tc.validation) }
                                resources
                                if !tc.logs.isEmpty { logPanel }
                            }
                            .frame(width: 350)
                        }
                        .riseIn(0.1)
                    }
                    .padding(32)
                }
                .scrollIndicators(.never)
            }
        }
        .animation(Theme.spring, value: tc.phase)
        .onAppear { if let e = app.models.entry(app.selectedModelID) ?? app.models.models.first { app.selectedModelID = e.id; tc.loadHistory(for: e) } }
        .onChange(of: app.selectedModelID) { _, id in if let e = app.models.entry(id) { tc.loadHistory(for: e) } }
    }

    // MARK: Header

    func header(_ entry: ModelEntry) -> some View {
        let mine = tc.isActive && tc.jobModelID == entry.id
        return HStack(alignment: .bottom, spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    if mine { LiveDot(color: stateColor) }
                    SectionLabel(stateLabel(entry))
                }
                Menu {
                    ForEach(app.models.models) { m in Button(m.definition.name) { app.selectedModelID = m.id } }
                } label: {
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Text(entry.definition.name).displayStyle(42).lineLimit(1)
                        Image(systemName: "chevron.down").font(.system(size: 14, weight: .semibold)).foregroundStyle(Theme.muted)
                    }
                }
                .menuStyle(.button).buttonStyle(.plain).fixedSize()
                .disabled(tc.isActive)
                Text(dataLine(entry)).font(Theme.font(13)).foregroundStyle(Theme.muted)
            }
            Spacer()
            HStack(spacing: 8) {
                if mine {
                    if tc.phase == .paused {
                        Button { tc.control("resume") } label: { Label("Resume", systemImage: "play.fill") }.buttonStyle(.atPrimary)
                    } else {
                        Button { tc.control("pause") } label: { Label("Pause", systemImage: "pause.fill") }.buttonStyle(.atPrimary)
                    }
                    Button("Save checkpoint") { tc.control("checkpoint") }.buttonStyle(.atSecondary)
                    Button("Stop") { tc.control("stop") }.buttonStyle(.atDestructive)
                } else {
                    let ready = app.downloads.isReady(entry.definition.initialization) || !entry.checkpoints.isEmpty
                    Button { tc.start(model: entry) } label: {
                        Label(entry.checkpoints.isEmpty ? "Start training" : "Continue training", systemImage: "play.fill")
                    }
                    .buttonStyle(.atPrimary)
                    .disabled(tc.isActive || !ready)
                    .help(ready ? "" : "Download the Pixel2Play weights in Models first")
                }
            }
        }
    }

    var stateColor: Color { tc.phase == .paused ? Theme.warning : Theme.live }

    func stateLabel(_ entry: ModelEntry) -> String {
        if tc.isActive, tc.jobModelID != entry.id { return "Another model is training" }
        if let p = tc.phase, tc.jobModelID == entry.id || !tc.isActive {
            switch p {
            case .preparing: return tc.message.isEmpty ? "Preparing" : tc.message
            case .training: return "Training"
            case .validating: return "Validating"
            case .paused: return tc.message.isEmpty ? "Paused" : tc.message
            case .saving: return "Saving"
            case .finished, .stopped, .failed: return tc.message.isEmpty ? p.rawValue.capitalized : tc.message
            }
        }
        return entry.definition.isTrained ? "Trained \(entry.definition.trainedSteps) steps" : "Ready"
    }

    func dataLine(_ entry: ModelEntry) -> String {
        if let d = tc.dataset, tc.jobModelID == entry.id {
            return "\(d.trainRecordings) recordings · \(d.trainSeconds.durationText) of gameplay · \(d.validationRecordings) held out · \(d.batchSize) clip\(d.batchSize == 1 ? "" : "s") per step"
        }
        let sel = app.training.selection[entry.id] ?? []
        return "\(sel.count) recording\(sel.count == 1 ? "" : "s") selected"
    }

    // MARK: Stats strip

    func stats(_ entry: ModelEntry) -> some View {
        let p = tc.progress.last
        let best = ModelPackage.checkpointState(entry.url, name: "best")
        return HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text(p.map { $0.step.formatted() } ?? "\(entry.definition.trainedSteps)").displayStyle(38).contentTransition(.numericText())
                    Text("/ \((p?.totalSteps ?? entry.definition.training.totalSteps).formatted())").font(Theme.font(19, .semibold)).foregroundStyle(Theme.faint)
                }
                GeometryReader { g in
                    let f = Double(p?.step ?? entry.definition.trainedSteps) / Double(max(1, p?.totalSteps ?? entry.definition.training.totalSteps))
                    ZStack(alignment: .leading) {
                        Capsule().fill(Theme.track)
                        Capsule().fill(Theme.text).frame(width: max(4, g.size.width * min(1, f)))
                    }
                }
                .frame(height: 4)
                Text(tc.eta.map { "steps · \($0.durationText) left" } ?? "steps").font(Theme.font(12)).foregroundStyle(Theme.muted)
            }
            .padding(22).frame(maxWidth: .infinity, alignment: .leading)
            divider
            BigStat(value: p.map { String(format: "%.0f", $0.framesPerSecond) } ?? "–", caption: p.map { String(format: "frames / s · %.2f s per step", $0.secondsPerStep) } ?? "frames per second", size: 38)
                .padding(22).frame(maxWidth: .infinity, alignment: .leading)
            divider
            BigStat(value: p.map { String(format: "%.4f", $0.loss) } ?? "–", caption: p.map { String(format: "training loss · lr %.1e", $0.learningRate) } ?? "training loss", size: 38)
                .padding(22).frame(maxWidth: .infinity, alignment: .leading)
            divider
            BigStat(value: best?.validationLoss.map { String(format: "%.3f", $0) } ?? "–", caption: best.map { "best held-out loss · step \($0.step)" } ?? "best held-out loss", size: 38, color: best == nil ? Theme.text : Theme.live)
                .padding(22).frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).strokeBorder(Theme.hairline))
        .animation(Theme.spring, value: p?.step)
    }

    var divider: some View { Rectangle().fill(Color.white.opacity(0.06)).frame(width: 1).padding(.vertical, 18) }

    // MARK: Chart

    var chartPanel: some View {
        Panel(padding: 20) {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Segmented(options: [(0, "Loss"), (1, "Validation"), (2, "Per head")], selection: $chartMode, compact: true)
                    Spacer()
                    if chartMode == 0 {
                        legend(Theme.text, "Training"); legend(Theme.accentLine, "Held-out")
                    }
                }
                Group {
                    if tc.progress.isEmpty {
                        VStack(spacing: 8) {
                            Image(systemName: "chart.xyaxis.line").font(.system(size: 28, weight: .light)).foregroundStyle(Theme.faint)
                            Text("The loss chart appears once training starts.").font(Theme.font(13)).foregroundStyle(Theme.muted)
                        }
                        .frame(maxWidth: .infinity)
                    } else {
                        switch chartMode {
                        case 0: LossChart(progress: tc.progress, validation: tc.validation)
                        case 1: ValidationChart(history: tc.validation)
                        default: HeadChart(progress: tc.progress)
                        }
                    }
                }
                .frame(height: 300)
                .transition(.opacity)
            }
        }
        .animation(Theme.spring, value: chartMode)
    }

    func legend(_ c: Color, _ s: String) -> some View {
        HStack(spacing: 7) { Capsule().fill(c).frame(width: 14, height: 2); Text(s).font(Theme.font(12)).foregroundStyle(Theme.text2) }.padding(.leading, 12)
    }

    // MARK: Side panels

    func earlyStopping(_ entry: ModelEntry) -> some View {
        let hp = entry.definition.training
        let checks = Self.checksWithoutImprovement(tc.validation, space: entry.definition.actions)
        return Panel {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Early stopping").font(Theme.font(14, .semibold))
                    Spacer()
                    if hp.earlyStopping {
                        Text("\(min(checks, hp.earlyStoppingPatience)) of \(hp.earlyStoppingPatience)").font(Theme.font(12, .semibold)).foregroundStyle(checks > 0 ? Theme.warning : Theme.muted)
                    } else {
                        Text("Off").font(Theme.font(12, .semibold)).foregroundStyle(Theme.muted)
                    }
                }
                if hp.earlyStopping {
                    HStack(spacing: 4) {
                        ForEach(0 ..< max(1, hp.earlyStoppingPatience), id: \.self) { i in
                            Capsule().fill(i < checks ? Theme.warning : Theme.track).frame(height: 4)
                        }
                    }
                    .animation(Theme.spring, value: checks)
                    Text(checks == 0 ? "Held-out loss is improving." : "No improvement for \(checks) check\(checks == 1 ? "" : "s"). Stops after \(max(0, hp.earlyStoppingPatience - checks)) more.")
                        .font(Theme.font(12)).foregroundStyle(Theme.muted)
                } else {
                    Text("Turn it on in Models → Training to stop when the held-out loss stops improving.").font(Theme.font(12)).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    static func checksWithoutImprovement(_ v: [ValidationMetrics], space: ActionSpaceConfig) -> Int {
        var best = Double.infinity, since = 0
        for m in v {
            let total = m.nll.enumerated().map { $0.element * ActionSpaceConfig.lossScale(classes: [space.keyClasses, space.pointerClasses, space.xClasses, space.yClasses][$0.offset]) }.reduce(0, +)
            if total < best { best = total; since = 0 } else { since += 1 }
        }
        return since
    }

    var resources: some View {
        Rows(title: "This Mac") {
            Row(title: "GPU") { value(tc.gpuUtilization.map { "\(Int($0))%" } ?? "–") }
            Row(title: "GPU memory") { value(tc.progress.last.map { String(format: "%.1f GB · peak %.1f", $0.gpuMemoryGB, $0.peakMemoryGB) } ?? "–") }
            Row(title: "Thermals") { value(thermalText).foregroundStyle(tc.thermal.rawValue >= 2 ? Theme.warning : Theme.text2) }
        }
    }

    var logPanel: some View {
        Panel(padding: 16) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Engine log").font(Theme.font(14, .semibold)).padding(.bottom, 4)
                ForEach(Array(tc.logs.suffix(8).enumerated()), id: \.offset) {
                    Text($0.element).font(.system(size: 11, design: .monospaced)).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    func value(_ s: String) -> some View { Text(s).font(Theme.font(13)).monospacedDigit().foregroundStyle(Theme.text2) }

    var thermalText: String {
        switch tc.thermal {
        case .nominal: "Normal"
        case .fair: "Warm"
        case .serious: "Hot · throttling"
        case .critical: "Critical"
        @unknown default: "–"
        }
    }
}

// MARK: Charts

struct LossChart: View {
    let progress: [TrainingProgress]
    let validation: [ValidationMetrics]

    var body: some View {
        let smoothed = Self.smooth(progress)
        let val = validation.map { ($0.step, $0.nll.reduce(0, +) / 4) }
        let best = val.min { $0.1 < $1.1 }
        Chart {
            ForEach(smoothed, id: \.0) { s in
                LineMark(x: .value("Step", s.0), y: .value("Loss", s.1), series: .value("S", "Training"))
                    .foregroundStyle(Theme.text).lineStyle(StrokeStyle(lineWidth: 1.8)).interpolationMethod(.monotone)
            }
            ForEach(val, id: \.0) { v in
                LineMark(x: .value("Step", v.0), y: .value("Loss", v.1), series: .value("S", "Held-out"))
                    .foregroundStyle(Theme.accentLine).lineStyle(StrokeStyle(lineWidth: 1.8)).interpolationMethod(.monotone)
                PointMark(x: .value("Step", v.0), y: .value("Loss", v.1)).foregroundStyle(Theme.accentLine).symbolSize(16)
            }
            if let best {
                RuleMark(x: .value("Best", best.0)).foregroundStyle(Theme.live.opacity(0.7)).lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 5]))
                    .annotation(position: .top, alignment: .leading) { Text("best · \(best.0)").font(Theme.font(11, .semibold)).foregroundStyle(Theme.live) }
                PointMark(x: .value("Step", best.0), y: .value("Loss", best.1)).foregroundStyle(Theme.live).symbolSize(60)
            }
        }
        .chartYScale(type: .log)
        .darkAxes()
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

struct HeadChart: View {
    let progress: [TrainingProgress]
    var body: some View {
        let names = ["Keys", "Buttons", "Mouse X", "Mouse Y"]
        let colors = [Theme.text, Theme.accentLine, Theme.live, Theme.warning]
        Chart {
            ForEach(0 ..< 4, id: \.self) { i in
                ForEach(LossChart.smoothHead(progress, i), id: \.0) { s in
                    LineMark(x: .value("Step", s.0), y: .value("Loss", s.1), series: .value("Head", names[i]))
                        .foregroundStyle(by: .value("Head", names[i])).interpolationMethod(.monotone)
                }
            }
        }
        .chartForegroundStyleScale(domain: names, range: colors)
        .chartLegend(position: .top, alignment: .trailing)
        .darkAxes()
    }
}

struct ValidationChart: View {
    let history: [ValidationMetrics]
    var body: some View {
        if history.isEmpty {
            Text("Validation results appear after the first check.").font(Theme.font(13)).foregroundStyle(Theme.muted).frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            Chart {
                ForEach(history, id: \.step) { v in
                    LineMark(x: .value("Step", v.step), y: .value("Score", v.keyF1), series: .value("M", "Key F1")).foregroundStyle(by: .value("M", "Key F1"))
                    LineMark(x: .value("Step", v.step), y: .value("Score", v.clickF1), series: .value("M", "Click F1")).foregroundStyle(by: .value("M", "Click F1"))
                    LineMark(x: .value("Step", v.step), y: .value("Score", v.mouseDirectionAccuracy), series: .value("M", "Mouse direction")).foregroundStyle(by: .value("M", "Mouse direction"))
                }
            }
            .chartForegroundStyleScale(domain: ["Key F1", "Click F1", "Mouse direction"], range: [Theme.text, Theme.accentLine, Theme.live])
            .chartYScale(domain: 0 ... 1)
            .chartLegend(position: .top, alignment: .trailing)
            .darkAxes()
        }
    }
}

extension View {
    /// Quiet axes and grid for charts on black.
    func darkAxes() -> some View {
        self
            .chartXAxis { AxisMarks { _ in AxisGridLine().foregroundStyle(Color.white.opacity(0.05)); AxisValueLabel().foregroundStyle(Theme.faint).font(Theme.font(11)) } }
            .chartYAxis { AxisMarks { _ in AxisGridLine().foregroundStyle(Color.white.opacity(0.06)); AxisValueLabel().foregroundStyle(Theme.faint).font(Theme.font(11)) } }
    }
}

// MARK: Validation

struct ValidationCard: View {
    let metrics: ValidationMetrics
    let history: [ValidationMetrics]

    var body: some View {
        Panel {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Text("Last validation").font(Theme.font(14, .semibold))
                    Spacer()
                    Text("step \(metrics.step)").font(Theme.font(12)).foregroundStyle(Theme.muted)
                }
                LazyVGrid(columns: [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)], spacing: 14) {
                    metric("Key F1", String(format: "%.2f", metrics.keyF1), "Right keys at the right time")
                    metric("Click F1", String(format: "%.2f", metrics.clickF1), "Clicks and scrolls")
                    metric("Mouse direction", String(format: "%.0f%%", metrics.mouseDirectionAccuracy * 100), "Moves like you did")
                    metric("Uses vision", String(format: "%+.3f", metrics.visualReliance), "Loss rise when frames are shuffled")
                    if let h = metrics.historyReliance { metric("Copies itself", String(format: "%+.3f", h), "Loss rise without past actions") }
                    if let l = metrics.languageReliance { metric("Uses instructions", String(format: "%+.3f", l), "Gain from the right instruction") }
                    metric("Beats chance · keys", String(format: "%.0f%%", metrics.rce[0]), "vs. guessing key frequencies")
                    metric("Perplexity", String(format: "%.3f", metrics.perplexity), "Pixel2Play's number, lower is better")
                }
                if let h = metrics.historyReliance, h > max(0.05, metrics.visualReliance * 3) {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                        Text("The model leans on repeating its previous actions more than on what it sees. Raise action-history dropout, record more varied situations, or record corrections in Run.")
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .font(Theme.font(12)).foregroundStyle(Theme.warning)
                }
            }
        }
    }

    func metric(_ title: String, _ value: String, _ help: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(value).font(Theme.font(20, .semibold)).monospacedDigit().contentTransition(.numericText())
            Text(title).font(Theme.font(12, .medium)).foregroundStyle(Theme.text2)
            Text(help).font(Theme.font(11)).foregroundStyle(Theme.faint).lineLimit(2).fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .help(help)
    }
}

// MARK: Checkpoints

struct CheckpointsCard: View {
    let app: AppModel
    let entry: ModelEntry

    static func label(_ name: String, url: URL) -> String {
        if name == "best" {
            guard let st = ModelPackage.checkpointState(url, name: name) else { return "Best" }
            return "Best · step \(st.step)"
        }
        guard let n = Int(name.replacingOccurrences(of: "step-", with: "")) else { return name }
        return "Step \(n.formatted())"
    }

    var body: some View {
        Panel(padding: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Checkpoints").font(Theme.font(14, .semibold)).padding(.horizontal, 10).padding(.top, 8).padding(.bottom, 6)
                if entry.checkpoints.isEmpty {
                    Text("Saved every \(Int(entry.definition.training.checkpointEveryMinutes)) min, on pause, at the end, and whenever validation improves.")
                        .font(Theme.font(12)).foregroundStyle(Theme.muted).padding(.horizontal, 10).padding(.bottom, 8).fixedSize(horizontal: false, vertical: true)
                }
                ForEach(ordered, id: \.self) { name in
                    let active = (entry.definition.activeCheckpoint ?? entry.checkpoints.last) == name
                    let st = ModelPackage.checkpointState(entry.url, name: name)
                    HStack(spacing: 10) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(Self.label(name, url: entry.url)).font(Theme.font(14, .semibold)).foregroundStyle(name == "best" ? Theme.live : Theme.text).lineLimit(1)
                            Text(subtitle(st, active: active)).font(Theme.font(11.5)).foregroundStyle(Theme.muted).lineLimit(1)
                        }
                        .layoutPriority(1)
                        Spacer(minLength: 4)
                        if active {
                            Image(systemName: "checkmark").font(.system(size: 11, weight: .bold)).foregroundStyle(Theme.live).help("Used by Run").frame(width: 30)
                        } else {
                            Button { var d = entry.definition; d.activeCheckpoint = name; app.models.save(d) } label: { Image(systemName: "play.circle") }
                                .buttonStyle(.atQuiet).help("Use for Run")
                        }
                        Button { app.training.start(model: entry, resumeFrom: name) } label: { Image(systemName: "arrow.clockwise") }
                            .buttonStyle(.atQuiet).help("Resume training from here").disabled(app.training.isActive)
                    }
                    .padding(.horizontal, 10).padding(.vertical, 8)
                }
            }
        }
    }

    var ordered: [String] {
        let steps = entry.checkpoints.filter { $0 != "best" }.reversed()
        return (entry.checkpoints.contains("best") ? ["best"] : []) + Array(steps.prefix(4))
    }

    func subtitle(_ st: CheckpointState?, active: Bool) -> String {
        var parts: [String] = []
        if let v = st?.validationLoss { parts.append(String(format: "held-out %.3f", v)) }
        if let d = st?.savedAt { parts.append(d.formatted(.relative(presentation: .named))) }
        if active { parts.append("used by Run") }
        return parts.joined(separator: " · ")
    }
}

// MARK: Dataset

struct DatasetPicker: View {
    let app: AppModel
    let entry: ModelEntry
    var body: some View {
        let sel = app.training.selection[entry.id] ?? []
        let all = app.library.allRecordings()
        let chosen = all.filter { sel.contains($0.url) }
        Panel(padding: 20) {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .firstTextBaseline) {
                    Text("Recordings to learn from").font(Theme.font(14, .semibold))
                    Spacer()
                    Text("\(chosen.count) selected · \(chosen.reduce(0) { $0 + $1.meta.duration }.durationText) · \(Int(entry.definition.training.validationFraction * 100))% held out")
                        .font(Theme.font(12)).foregroundStyle(Theme.muted).contentTransition(.numericText())
                }
                ScrollView {
                    VStack(alignment: .leading, spacing: 2) {
                        FolderChecklist(app: app, entry: entry, folder: app.library.tree, depth: 0, all: all)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(height: 220)
                .disabled(app.training.isActive)
            }
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
        let count = inFolder.filter { sel.contains($0.url) }.count
        check(state: inFolder.isEmpty ? 0 : count == inFolder.count ? 2 : count > 0 ? 1 : 0,
              title: depth == 0 ? "Whole library" : folder.name, icon: depth == 0 ? "books.vertical" : "folder", bold: true,
              trailing: "\(count)/\(inFolder.count)") {
            var s = app.training.selection[entry.id] ?? []
            let on = count != inFolder.count
            for r in inFolder { if on { s.insert(r.url) } else { s.remove(r.url) } }
            app.training.selection[entry.id] = s
            app.training.saveSelection()
        }
        .padding(.leading, CGFloat(depth) * 18)
        .disabled(inFolder.isEmpty)
        ForEach(direct) { r in
            check(state: sel.contains(r.url) ? 2 : 0, title: r.meta.name, icon: nil, bold: false, trailing: r.meta.duration.durationText) {
                var s = app.training.selection[entry.id] ?? []
                if s.contains(r.url) { s.remove(r.url) } else { s.insert(r.url) }
                app.training.selection[entry.id] = s
                app.training.saveSelection()
            }
            .padding(.leading, CGFloat(depth + 1) * 18)
        }
        ForEach(folder.children) { FolderChecklist(app: app, entry: entry, folder: $0, depth: depth + 1, all: all) }
    }

    func check(state: Int, title: String, icon: String?, bold: Bool, trailing: String, action: @escaping () -> Void) -> some View {
        Button { withAnimation(Theme.snappy) { action() } } label: {
            HStack(spacing: 10) {
                ZStack {
                    RoundedRectangle(cornerRadius: 5, style: .continuous).fill(state > 0 ? Theme.text : Color.clear)
                    RoundedRectangle(cornerRadius: 5, style: .continuous).strokeBorder(state > 0 ? Theme.text : Theme.faint, lineWidth: 1.5)
                    if state == 2 { Image(systemName: "checkmark").font(.system(size: 9, weight: .heavy)).foregroundStyle(.black) }
                    if state == 1 { Capsule().fill(.black).frame(width: 8, height: 2) }
                }
                .frame(width: 16, height: 16)
                if let icon { Image(systemName: icon).font(.system(size: 12)).foregroundStyle(Theme.text2) }
                Text(title).font(Theme.font(13, bold ? .semibold : .regular)).lineLimit(1)
                Spacer()
                Text(trailing).font(Theme.font(11.5)).monospacedDigit().foregroundStyle(Theme.muted)
            }
            .padding(.horizontal, 8).frame(height: 30)
        }
        .buttonStyle(RowButtonStyle(radius: 8))
    }
}
