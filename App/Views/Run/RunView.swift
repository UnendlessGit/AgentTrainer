import ATCapture
import ATCore
import ATRuntime
import SwiftUI

struct RunView: View {
    @Bindable var app: AppModel

    var body: some View {
        @Bindable var run = app.runner
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                PageHeader(title: "Run", subtitle: "Let a trained model play or control an app in real time. Touch the keyboard or mouse to take over.")
                controlCard
                if let e = run.errorMessage { InlineError(message: e) }
                if run.isRunning { statsCard }
                Card(title: "What to control", symbol: "macwindow",
                     trailing: AnyView(Button { Task { await run.refreshTargets() } } label: { Label("Refresh", systemImage: "arrow.clockwise") }.disabled(run.isRunning))) {
                    TargetGrid(targets: run.targets, thumbnails: run.thumbnails, selection: $run.selectedTargetID).disabled(run.isRunning)
                }
                if run.selectedModel?.definition.usesInstructions == true {
                    Card(title: "Instruction", symbol: "text.bubble") {
                        HStack {
                            TextField("e.g. “Walk to the door and open it”", text: $run.instruction)
                                .onSubmit { run.applyInstruction() }
                            Button(run.isRunning ? "Send" : "Set") { run.applyInstruction() }
                        }
                        if let a = run.appliedInstruction { Text("Following: “\(a)”").font(.caption).foregroundStyle(.secondary) }
                        if !run.encoderStatus.isEmpty { Text(run.encoderStatus).font(.caption).foregroundStyle(.orange) }
                        Text("You can change the instruction while the model runs; it takes effect on the next frame.").font(.caption).foregroundStyle(.secondary)
                    }
                }
                HStack(alignment: .top, spacing: 18) {
                    Card(title: "Allowed actions", symbol: "hand.raised") {
                        Toggle("Keyboard", isOn: $run.prefs.permissions.keyboard)
                        Toggle("Typing (letters, numbers, symbols)", isOn: $run.prefs.permissions.typingKeys).disabled(!run.prefs.permissions.keyboard)
                        Toggle("Mouse movement", isOn: $run.prefs.permissions.mouseMovement)
                        Toggle("Mouse buttons", isOn: $run.prefs.permissions.mouseButtons)
                        Toggle("Scrolling", isOn: $run.prefs.permissions.scroll)
                        BlockedKeysEditor(blocked: $run.prefs.permissions.blockedKeys)
                        Text("These only filter what is sent. The model and its training are not changed.").font(.caption).foregroundStyle(.secondary)
                    }
                    Card(title: "Behavior", symbol: "slider.horizontal.3") {
                        LabeledSlider(title: "Randomness", value: $run.prefs.temperature, range: 0 ... 1.5, step: 0.05, format: { $0 == 0 ? "off" : String(format: "%.2f", $0) })
                        LabeledSlider(title: "Mouse sensitivity", value: $run.prefs.sensitivity, range: 0.25 ... 3, step: 0.05, format: { String(format: "%.2f×", $0) })
                        Picker("Mouse output", selection: $run.prefs.mouseMode) { ForEach(MouseOutputMode.allCases, id: \.self) { Text($0.label).tag($0) } }
                        Toggle("Only act while the target is in front", isOn: $run.prefs.focusGuard)
                        Toggle("Use action history (turn off if the model keeps repeating itself)", isOn: $run.prefs.useActionHistory)
                        Toggle("Dry run (show actions in the HUD, send nothing)", isOn: $run.prefs.dryRun)
                        Toggle("Record my corrections for training", isOn: $run.prefs.recordCorrections)
                        Button("Test mouse control on the target") { Task { await run.mouseTest() } }
                            .disabled(run.isRunning || run.selectedTarget == nil)
                            .help("Brings the target forward and moves the mouse in a small square. Use it to check that a game accepts synthetic input.")
                    }
                    .disabled(run.isRunning)
                }
                Card(title: "Run HUD", symbol: "rectangle.inset.bottomright.filled") {
                    HUDOptionsEditor(options: $run.prefs.hud, showsVision: true)
                }
            }
            .padding(24)
        }
        .task {
            app.runner.library = app.library
            if run.targets.isEmpty { await run.refreshTargets() }
            if run.selectedModel == nil { run.selectedModelID = app.models.models.first(where: { !$0.checkpoints.isEmpty })?.id }
        }
    }

    private var controlCard: some View {
        let run = app.runner
        return Card(title: "Agent", symbol: "play.circle") {
            HStack(spacing: 16) {
                Button { Task { await run.toggle() } } label: {
                    Image(systemName: run.isRunning ? "stop.fill" : "play.fill").font(.title).frame(width: 64, height: 64)
                        .background(Circle().fill(run.isRunning ? Color.red : Color.green)).foregroundStyle(.white)
                }
                .buttonStyle(.plain)
                .disabled(run.isStarting || run.selectedModel == nil)
                VStack(alignment: .leading, spacing: 6) {
                    Picker("Model", selection: Binding(get: { run.selectedModelID }, set: { run.selectedModelID = $0 })) {
                        ForEach(app.models.models) { m in
                            Text(m.checkpoints.isEmpty ? "\(m.definition.name) (not trained)" : m.definition.name).tag(Optional(m.id))
                        }
                    }
                    .frame(maxWidth: 360)
                    .disabled(run.isRunning)
                    Text(statusText).foregroundStyle(.secondary).font(.callout)
                    Text("Start/stop \(app.settings.stored.runHotkey.display) · pause \(app.settings.stored.pauseHotkey.display) · emergency stop \(app.settings.stored.emergencyHotkey.display)")
                        .font(.caption).foregroundStyle(.tertiary)
                }
                Spacer()
                if run.isRunning {
                    Button { run.togglePause() } label: { Label(run.isPaused ? "Resume" : "Pause", systemImage: run.isPaused ? "play" : "pause") }
                        .controlSize(.large)
                }
            }
        }
    }

    private var statusText: String {
        let run = app.runner
        if run.isStarting { return "Loading model…" }
        guard run.isRunning else {
            if let m = run.selectedModel { return m.checkpoints.isEmpty ? "This model hasn't been trained yet." : "Ready · \(m.definition.architecture.stepRateHz) decisions per second" }
            return "Choose a model"
        }
        if run.stats.paused { return run.stats.pauseReason.isEmpty ? "Paused" : run.stats.pauseReason }
        if !run.stats.pauseReason.isEmpty { return run.stats.pauseReason }
        return "Running \(run.loadedCheckpoint.replacingOccurrences(of: "step-", with: "step "))"
    }

    private var statsCard: some View {
        let s = app.runner.stats
        let hz = app.runner.selectedModel?.definition.architecture.stepRateHz ?? 30
        return LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 10)], spacing: 10) {
            StatTile(title: "Decisions / s", value: String(format: "%.1f of %d", s.fps, hz), symbol: "speedometer", tint: s.fps >= Double(hz) * 0.95 ? .green : .orange)
            StatTile(title: "Model step", value: String(format: "%.1f ms", s.modelMs), symbol: "cpu")
            StatTile(title: "Preprocess", value: String(format: "%.1f ms", s.preprocessMs), symbol: "photo")
            StatTile(title: "Frame age", value: String(format: "%.1f ms", s.frameAgeMs), symbol: "clock")
            StatTile(title: "Screen → action", value: String(format: "%.1f ms", s.endToEndMs), symbol: "bolt")
            StatTile(title: "Missed ticks", value: "\(s.droppedTicks)", symbol: "exclamationmark.triangle", tint: s.droppedTicks > 0 ? .orange : .accentColor)
            StatTile(title: "Takeovers", value: "\(s.takeovers)", symbol: "hand.raised")
            StatTile(title: "Active steps / s", value: String(format: "%.1f", s.actionsPerSecond), symbol: "keyboard")
        }
    }
}

struct BlockedKeysEditor: View {
    @Binding var blocked: Set<String>
    private let common = ["command", "rcommand", "escape", "tab", "return", "option", "control", "delete", "q"]
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Never press").font(.callout)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 92), alignment: .leading)], alignment: .leading) {
                ForEach(common, id: \.self) { k in
                    Toggle(KeyCatalog.info(k).map { $0.id.hasPrefix("r") && $0.id != "return" ? "R " + $0.label : $0.label } ?? k,
                           isOn: Binding(get: { blocked.contains(k) }, set: { if $0 { blocked.insert(k) } else { blocked.remove(k) } }))
                        .toggleStyle(.checkbox).font(.caption)
                }
            }
        }
    }
}
