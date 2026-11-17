import ATCapture
import ATCore
import ATRuntime
import SwiftUI

struct RunView: View {
    @Bindable var app: AppModel

    var body: some View {
        @Bindable var run = app.runner
        VStack(alignment: .leading, spacing: 24) {
            controlPanel.riseIn()
            if let e = run.errorMessage { InlineError(message: e) }
            HStack(alignment: .top, spacing: 24) {
                // Left stays put; it only scrolls on its own if there are more windows than fit.
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        VStack(alignment: .leading, spacing: 12) {
                            HStack {
                                SectionLabel("What to control")
                                Spacer()
                                Button { Task { await run.refreshTargets() } } label: { Image(systemName: "arrow.clockwise").font(.system(size: 12, weight: .semibold)) }
                                    .buttonStyle(.atQuiet).help("Refresh").disabled(run.isRunning)
                            }
                            TargetGrid(targets: run.targets, thumbnails: run.thumbnails, selection: $run.selectedTargetID).disabled(run.isRunning)
                        }
                        if run.selectedModel?.definition.usesInstructions == true { instruction }
                    }
                    .padding(.bottom, 32)
                }
                .scrollIndicators(.never)
                .scrollBounceBehavior(.basedOnSize)
                .riseIn(0.05)
                // Only the settings column scrolls.
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        Rows(title: "Allowed actions", footer: "These only filter what is sent. The model and its training are not changed.") {
                            Toggle("Keyboard", isOn: $run.prefs.permissions.keyboard)
                            Toggle("Typing keys", isOn: $run.prefs.permissions.typingKeys).disabled(!run.prefs.permissions.keyboard)
                            Toggle("Mouse movement", isOn: $run.prefs.permissions.mouseMovement)
                            Toggle("Mouse buttons", isOn: $run.prefs.permissions.mouseButtons)
                            Toggle("Scrolling", isOn: $run.prefs.permissions.scroll)
                            BlockedKeysEditor(blocked: $run.prefs.permissions.blockedKeys)
                        }
                        .toggleStyle(.oled)
                        Rows(title: "Behaviour") {
                            LabeledSlider(title: "Randomness", value: $run.prefs.temperature, range: 0 ... 1.5, step: 0.05, format: { $0 == 0 ? "off" : String(format: "%.2f", $0) })
                            LabeledSlider(title: "Mouse sensitivity", value: $run.prefs.sensitivity, range: 0.25 ... 3, step: 0.05, format: { String(format: "%.2f×", $0) })
                            Row(title: "Mouse output") {
                                Picker("", selection: $run.prefs.mouseMode) { ForEach(MouseOutputMode.allCases, id: \.self) { Text($0.label).tag($0) } }.labelsHidden().fixedSize()
                            }
                            Toggle("Only act while the target is in front", isOn: $run.prefs.focusGuard).toggleStyle(.oled)
                            Toggle("Feed the model its own past actions", isOn: $run.prefs.useActionHistory).toggleStyle(.oled)
                                .help("Pixel2Play does this. With small datasets models often play better without it, because they stop copying their own previous actions.")
                            Toggle("Dry run · show actions, send nothing", isOn: $run.prefs.dryRun).toggleStyle(.oled)
                            Toggle("Record my corrections for training", isOn: $run.prefs.recordCorrections).toggleStyle(.oled)
                            Row(title: "Mouse injection", detail: "Moves the mouse in a small square to check the game accepts it") {
                                Button("Test") { Task { await run.mouseTest() } }.buttonStyle(.atSmall).disabled(run.isRunning || run.selectedTarget == nil)
                            }
                        }
                        .disabled(run.isRunning)
                        HUDRows(title: "Run HUD", options: $run.prefs.hud, showsVision: true)
                    }
                    .padding(.bottom, 32)
                }
                .scrollIndicators(.never)
                .scrollBounceBehavior(.basedOnSize)
                .frame(width: 400)
                .riseIn(0.1)
            }
        }
        .padding([.horizontal, .top], 32)
        .task {
            app.runner.library = app.library
            if run.targets.isEmpty { await run.refreshTargets() }
            if run.selectedModel == nil { run.selectedModelID = app.models.models.first(where: { !$0.checkpoints.isEmpty })?.id }
        }
    }

    // MARK: Control panel

    private var controlPanel: some View {
        let run = app.runner
        let s = run.stats
        let hz = run.selectedModel?.definition.architecture.stepRateHz ?? 30
        return HStack(spacing: 26) {
            GoButton(running: run.isRunning, starting: run.isStarting) { Task { await run.toggle() } }
                .disabled(run.isStarting || run.selectedModel == nil)
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 12) {
                    Menu {
                        ForEach(app.models.models) { m in
                            Button(m.checkpoints.isEmpty ? "\(m.definition.name) (not trained)" : m.definition.name) { run.selectedModelID = m.id }
                                .disabled(m.checkpoints.isEmpty)
                        }
                    } label: {
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            Text(run.selectedModel?.definition.name ?? "Choose a model").displayStyle(34).lineLimit(1)
                            Image(systemName: "chevron.down").font(.system(size: 13, weight: .semibold)).foregroundStyle(Theme.muted)
                        }
                    }
                    .menuStyle(.button).buttonStyle(.plain).fixedSize()
                    .disabled(run.isRunning)
                    if let m = run.selectedModel, let cp = m.definition.activeCheckpoint ?? m.checkpoints.last {
                        Pill(text: CheckpointsCard.label(cp, url: m.url), color: cp == "best" ? Theme.live : Theme.text2)
                    }
                }
                HStack(spacing: 8) {
                    if run.isRunning { LiveDot(color: s.paused ? Theme.warning : Theme.live) }
                    Text(statusText).font(Theme.font(13)).foregroundStyle(run.isRunning ? Theme.text2 : Theme.muted).lineLimit(1)
                }
                Text("Start/stop \(app.settings.stored.runHotkey.display) · Pause \(app.settings.stored.pauseHotkey.display) · Emergency \(app.settings.stored.emergencyHotkey.display)")
                    .font(Theme.font(12)).foregroundStyle(Theme.faint)
            }
            Spacer(minLength: 16)
            HStack(spacing: 36) {
                BigStat(value: run.isRunning ? String(format: "%.1f", s.fps) : "–", caption: "of \(hz) decisions / s", size: 34,
                        color: run.isRunning && s.fps < Double(hz) * 0.95 ? Theme.warning : Theme.text)
                BigStat(value: run.isRunning ? String(format: "%.1f", s.modelMs) : "–", unit: "ms", caption: "model step", size: 34)
                BigStat(value: run.isRunning ? String(format: "%.0f", s.endToEndMs) : "–", unit: "ms", caption: "screen → action", size: 34)
                BigStat(value: run.isRunning ? "\(s.takeovers)" : "–", caption: "takeovers", size: 34)
            }
            if run.isRunning {
                Button { run.togglePause() } label: { Label(run.isPaused ? "Resume" : "Pause", systemImage: run.isPaused ? "play.fill" : "pause.fill") }
                    .buttonStyle(.atSecondary)
                    .transition(.opacity.combined(with: .scale(scale: 0.9)))
            }
        }
        .padding(.horizontal, 26).padding(.vertical, 22)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).strokeBorder(run.isRunning ? Theme.live.opacity(0.35) : Theme.hairline))
        .animation(Theme.spring, value: run.isRunning)
        .animation(Theme.spring, value: s.fps)
    }

    private var instruction: some View {
        @Bindable var run = app.runner
        return VStack(alignment: .leading, spacing: 10) {
            SectionLabel("Instruction")
            HStack(spacing: 10) {
                Image(systemName: "text.bubble").font(.system(size: 14)).foregroundStyle(Theme.muted)
                TextField("e.g. Walk to the door and open it", text: $run.instruction).textFieldStyle(.plain).font(Theme.font(15))
                    .onSubmit { run.applyInstruction() }
                Button(run.isRunning ? "Send" : "Set") { run.applyInstruction() }.buttonStyle(.atSmallPrimary)
            }
            .padding(.leading, 16).padding(.trailing, 8).frame(height: 52)
            .background(Theme.surface, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).strokeBorder(Theme.hairline))
            if let a = run.appliedInstruction { Text("Following “\(a)”").font(Theme.font(12)).foregroundStyle(Theme.live) }
            if !run.encoderStatus.isEmpty { Text(run.encoderStatus).font(Theme.font(12)).foregroundStyle(Theme.warning) }
        }
    }

    private var statusText: String {
        let run = app.runner
        if run.isStarting { return "Loading model…" }
        guard run.isRunning else {
            if let m = run.selectedModel {
                if m.checkpoints.isEmpty { return "This model hasn't been trained yet." }
                return "Ready · \(run.selectedTarget?.name ?? "choose what to control")"
            }
            return "Choose a model"
        }
        if run.stats.paused { return run.stats.pauseReason.isEmpty ? "Paused" : run.stats.pauseReason }
        if !run.stats.pauseReason.isEmpty { return run.stats.pauseReason }
        return "Running on \(run.selectedTarget?.name ?? "the target")"
    }
}

/// The large start/stop control.
struct GoButton: View {
    let running: Bool
    let starting: Bool
    let action: () -> Void
    @State private var hover = false
    @Environment(\.isEnabled) private var enabled
    var body: some View {
        Button(action: action) {
            ZStack {
                Circle().fill(running ? Theme.danger : Theme.text).frame(width: 72, height: 72)
                Circle().strokeBorder((running ? Theme.danger : Color.white).opacity(0.15), lineWidth: 6).frame(width: 86, height: 86)
                if starting {
                    ProgressView().controlSize(.small).tint(.black)
                } else if running {
                    RoundedRectangle(cornerRadius: 5, style: .continuous).fill(.white).frame(width: 22, height: 22)
                } else {
                    Image(systemName: "play.fill").font(.system(size: 26)).foregroundStyle(.black).offset(x: 2)
                }
            }
            .scaleEffect(hover && enabled ? 1.05 : 1)
            .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .opacity(enabled ? 1 : 0.4)
        .onHover { hover = $0 }
        .animation(Theme.spring, value: running)
        .animation(Theme.snappy, value: hover)
        .accessibilityLabel(running ? "Stop" : "Start")
    }
}

struct BlockedKeysEditor: View {
    @Binding var blocked: Set<String>
    private let common = ["command", "rcommand", "escape", "tab", "return", "option", "control", "delete", "q"]
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Never press").font(Theme.font(14))
            FlowChips(items: common.map { k in
                (k, KeyCatalog.info(k).map { $0.id.hasPrefix("r") && $0.id != "return" ? "R " + $0.label : $0.label } ?? k)
            }, isOn: { blocked.contains($0) }) { k in
                withAnimation(Theme.snappy) { if blocked.contains(k) { blocked.remove(k) } else { blocked.insert(k) } }
            }
        }
    }
}
