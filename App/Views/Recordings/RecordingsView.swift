import ATCapture
import ATCore
import SwiftUI

struct RecordingsView: View {
    @Bindable var app: AppModel

    var body: some View {
        @Bindable var settings = app.settings
        @Bindable var rec = app.recorder
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                header(rec).riseIn()
                if let e = rec.errorMessage { InlineError(message: e) }
                HStack(alignment: .top, spacing: 24) {
                    stage(rec)
                        .frame(minWidth: 380, maxWidth: 580)
                        .riseIn(0.05)
                    HStack(alignment: .top, spacing: 16) {
                        VStack(spacing: 16) {
                            Rows(title: "Capture") {
                                Row(title: "Save to") {
                                    Picker("", selection: $rec.destination) {
                                        ForEach(app.library.flatFolders, id: \.url) { f in Text(String(repeating: "   ", count: f.depth) + f.name).tag(f.url) }
                                    }
                                    .labelsHidden().fixedSize()
                                }
                                Row(title: "Resolution") {
                                    Picker("", selection: $settings.stored.captureMaxHeight) {
                                        Text("480p").tag(480); Text("720p").tag(720); Text("1080p").tag(1080); Text("1440p").tag(1440)
                                    }
                                    .labelsHidden().fixedSize()
                                }
                                Row(title: "Frame rate") { Segmented(options: [(30, "30"), (60, "60 fps")], selection: $settings.stored.captureFPS, compact: true) }
                                Row(title: "Task") {
                                    TextField("What are you doing?", text: $rec.task)
                                        .textFieldStyle(.plain).multilineTextAlignment(.trailing).font(Theme.font(14)).frame(maxWidth: 200)
                                }
                                Row(title: "Name") {
                                    TextField("Automatic", text: $rec.name)
                                        .textFieldStyle(.plain).multilineTextAlignment(.trailing).font(Theme.font(14)).frame(maxWidth: 200)
                                }
                            }
                            .disabled(rec.isBusy)
                            HUDRows(title: "Recording HUD", options: $settings.stored.recordHUD, showsVision: false)
                        }
                        VStack(spacing: 16) {
                            Rows(title: "Input") {
                                Toggle("Keyboard", isOn: $settings.stored.input.keyboard)
                                Toggle("Mouse movement", isOn: $settings.stored.input.mouseMovement)
                                Toggle("Mouse buttons", isOn: $settings.stored.input.mouseButtons)
                                Toggle("Scrolling", isOn: $settings.stored.input.scroll)
                                Toggle(isOn: $settings.stored.input.onlyWhenTargetFocused) { Row(title: "Only while in front", detail: "Ignore input while you use other apps") }
                            }
                            .toggleStyle(.oled)
                            .disabled(rec.isBusy)
                            Rows(title: "Video") {
                                Toggle("Show the mouse pointer", isOn: $settings.stored.showsCursor).toggleStyle(.oled)
                                LabeledSlider(title: "Quality", value: $settings.stored.videoQuality, range: 0.1 ... 1, format: { "\(Int($0 * 100))%" })
                            }
                            .disabled(rec.isBusy)
                        }
                    }
                    .riseIn(0.1)
                }
            }
            .padding(32)
        }
        .scrollIndicators(.never)
        .task { if rec.targets.isEmpty { await rec.refreshTargets() } }
        .task(id: rec.selectedTargetID) {
            rec.preview = nil
            await rec.runPreview { [weak rec] in rec?.phase == .idle }
        }
        .onAppear { if !app.library.flatFolders.contains(where: { $0.url == rec.destination }) { rec.destination = app.library.fs.root } }
    }

    // MARK: Header

    func header(_ rec: RecorderController) -> some View {
        @Bindable var rec = rec
        return HStack(alignment: .bottom, spacing: 20) {
            VStack(alignment: .leading, spacing: 8) {
                Text(title(rec)).displayStyle(44).contentTransition(.opacity).animation(Theme.spring, value: rec.phase)
                if let s = rec.plannedSettings {
                    Text(verbatim: "\(s.width) × \(s.height) · \(s.fps) fps · training frames saved losslessly at 192 × 192").font(Theme.font(13)).foregroundStyle(Theme.muted)
                }
            }
            Spacer(minLength: 20)
            TargetPicker(targets: rec.targets, thumbnails: rec.thumbnails, selection: $rec.selectedTargetID, loading: rec.loadingTargets) {
                Task { await rec.refreshTargets() }
            }
            .frame(width: 380)
            .disabled(rec.isBusy)
        }
    }

    // MARK: Stage: preview, record controls, live numbers

    func stage(_ rec: RecorderController) -> some View {
        VStack(alignment: .leading, spacing: 20) {
            preview(rec).aspectRatio(16 / 10, contentMode: .fit)
            HStack(alignment: .center, spacing: 22) {
                RecordButton(phase: rec.phase) { Task { await rec.toggle() } }
                    .help("Start or stop and save (\(app.settings.stored.recordHotkey.display))")
                    .disabled(rec.phase == .saving || (rec.phase == .idle && rec.selectedTarget == nil))
                VStack(alignment: .leading, spacing: 6) {
                    Text(rec.stats.elapsed.durationText).displayStyle(40).contentTransition(.numericText())
                    Text("Save \(app.settings.stored.recordHotkey.display) · Discard \(app.settings.stored.cancelRecordingHotkey.display)")
                        .font(Theme.font(12.5)).foregroundStyle(Theme.muted)
                }
                Spacer(minLength: 0)
            }
            HStack(spacing: 0) {
                stat("\(rec.stats.frames)", "frames")
                stat("\(rec.stats.droppedFrames)", "dropped", color: rec.stats.droppedFrames > 0 ? Theme.warning : Theme.text)
                stat("\(rec.stats.events)", "input events")
                stat(rec.stats.frames > 0 ? String(format: "%.0f", rec.stats.averageFPS) : "–", "fps")
            }
            .padding(.vertical, 14)
            .background(Theme.surface, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).strokeBorder(Theme.hairline))
        }
        .animation(Theme.spring, value: rec.phase)
    }

    func preview(_ rec: RecorderController) -> some View {
        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 18, style: .continuous).fill(Theme.surface)
            if let id = rec.selectedTargetID, let img = rec.preview ?? rec.thumbnails[id] {
                Image(nsImage: img).resizable().aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .padding(14)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 10) {
                    Image(systemName: "macwindow").font(.system(size: 30, weight: .light)).foregroundStyle(Theme.faint)
                    Text(rec.targets.isEmpty ? "Open the game or app, then refresh the list" : "Choose what to record").font(Theme.font(13)).foregroundStyle(Theme.muted)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            if rec.phase == .recording {
                HStack(spacing: 8) { LiveDot(color: Theme.danger); Text("REC").font(Theme.font(12, .bold)).tracking(0.8) }
                    .foregroundStyle(Theme.dangerText)
                    .padding(.horizontal, 12).padding(.vertical, 6)
                    .background(Theme.danger.opacity(0.18), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .padding(16)
                    .transition(.scale(scale: 0.8, anchor: .topLeading).combined(with: .opacity))
            }
            if case .countdown(let n) = rec.phase {
                Text("\(n)").displayStyle(110).frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .contentTransition(.numericText(countsDown: true))
            }
        }
        .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).strokeBorder(rec.phase == .recording ? Theme.danger.opacity(0.55) : Theme.hairline, lineWidth: rec.phase == .recording ? 1.5 : 1))
    }

    func stat(_ v: String, _ caption: String, color: Color = Theme.text) -> some View {
        VStack(spacing: 4) {
            Text(v).font(Theme.font(20, .semibold)).monospacedDigit().foregroundStyle(color).contentTransition(.numericText())
            Text(caption).font(Theme.font(11.5)).foregroundStyle(Theme.muted)
        }
        .frame(maxWidth: .infinity)
    }

    func title(_ r: RecorderController) -> String {
        switch r.phase {
        case .idle: "Ready to record"
        case .countdown: "Get ready"
        case .recording: "Recording"
        case .saving: "Saving…"
        }
    }
}

/// A dropdown of windows and displays, showing the chosen one with its thumbnail.
struct TargetPicker: View {
    let targets: [CaptureTarget]
    let thumbnails: [String: NSImage]
    @Binding var selection: String?
    var loading = false
    var refresh: () -> Void
    @State private var hover = false
    @Environment(\.isEnabled) private var enabled

    var body: some View {
        let sel = targets.first { $0.id == selection }
        HStack(spacing: 8) {
            Menu {
                let displays = targets.filter(\.isDisplay), windows = targets.filter { !$0.isDisplay }
                if !displays.isEmpty {
                    Section("Displays") { ForEach(displays) { item($0) } }
                }
                if !windows.isEmpty {
                    Section("Windows") { ForEach(windows) { item($0) } }
                }
                if targets.isEmpty { Text("No windows found. Open the game or app, then refresh.") }
            } label: {
                HStack(spacing: 12) {
                    Group {
                        if let s = sel, let img = thumbnails[s.id] {
                            Color.clear.overlay { Image(nsImage: img).resizable().scaledToFill() }
                        } else {
                            Placeholder(symbol: sel?.isDisplay == true ? "display" : "macwindow", radius: 6)
                        }
                    }
                    .frame(width: 64, height: 40)
                    .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                    VStack(alignment: .leading, spacing: 2) {
                        Text(sel?.name ?? "Choose a window").font(Theme.font(14, .semibold)).lineLimit(1)
                        Text(sel.map { $0.isDisplay ? "Display" : "Window · \($0.info.appName ?? "")" } ?? "\(targets.count) available")
                            .font(Theme.font(12)).foregroundStyle(Theme.muted).lineLimit(1)
                    }
                    Spacer(minLength: 0)
                    Image(systemName: "chevron.up.chevron.down").font(.system(size: 11, weight: .semibold)).foregroundStyle(Theme.muted)
                }
                .padding(.leading, 8).padding(.trailing, 14).frame(height: 56)
                .background(hover && enabled ? Theme.raised : Theme.surface, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).strokeBorder(Theme.hairline))
                .contentShape(Rectangle())
            }
            .menuStyle(.button).buttonStyle(.plain).menuIndicator(.hidden)
            .onHover { hover = $0 }
            .animation(.easeOut(duration: 0.15), value: hover)
            Button(action: refresh) {
                Image(systemName: "arrow.clockwise").font(.system(size: 13, weight: .semibold))
                    .rotationEffect(.degrees(loading ? 360 : 0))
                    .animation(loading ? .linear(duration: 0.8).repeatForever(autoreverses: false) : .default, value: loading)
                    .frame(width: 30)
            }
            .buttonStyle(ATButtonStyle(kind: .secondary, height: 56))
            .help("Refresh the list")
        }
        .opacity(enabled ? 1 : 0.5)
    }

    func item(_ t: CaptureTarget) -> some View {
        Button { selection = t.id } label: {
            Label(t.name, systemImage: t.id == selection ? "checkmark" : (t.isDisplay ? "display" : "macwindow"))
        }
    }
}

/// The big record button: a red dot that morphs into a stop square.
struct RecordButton: View {
    let phase: RecorderController.Phase
    let action: () -> Void
    @State private var hover = false
    @Environment(\.isEnabled) private var enabled
    var body: some View {
        Button(action: action) {
            ZStack {
                Circle().fill(Theme.text).frame(width: 60, height: 60)
                Circle().strokeBorder(Color.white.opacity(0.12), lineWidth: 6).frame(width: 74, height: 74)
                if phase == .saving {
                    ProgressView().controlSize(.small).tint(.black)
                } else {
                    RoundedRectangle(cornerRadius: phase == .recording ? 6 : 13, style: .continuous)
                        .fill(Theme.danger)
                        .frame(width: phase == .recording ? 22 : 26, height: phase == .recording ? 22 : 26)
                }
            }
            .scaleEffect(hover && enabled ? 1.05 : 1)
            .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .opacity(enabled ? 1 : 0.4)
        .onHover { hover = $0 }
        .animation(Theme.spring, value: phase)
        .animation(Theme.snappy, value: hover)
        .accessibilityLabel(phase == .recording ? "Stop and save" : "Start recording")
    }
}

/// HUD options as rows inside a grouped panel.
struct HUDRows: View {
    let title: String
    @Binding var options: HUDOptions
    var showsVision: Bool
    var body: some View {
        Rows(title: title, footer: "Hidden from recordings and from the model's view.") {
            Toggle("Show HUD", isOn: $options.enabled.animation(Theme.spring)).toggleStyle(.oled)
            if options.enabled {
                FlowChips(items: shows, isOn: { isOn($0) }) { k in withAnimation(Theme.snappy) { toggle(k) } }
                Row(title: "Corner") {
                    Picker("", selection: $options.corner) { ForEach(HUDCorner.allCases) { Text($0.label).tag($0) } }.labelsHidden().fixedSize()
                }
                LabeledSlider(title: "Size", value: $options.scale, range: 0.6 ... 1.8, format: { "\(Int($0 * 100))%" })
                LabeledSlider(title: "Opacity", value: $options.opacity, range: 0.3 ... 1, format: { "\(Int($0 * 100))%" })
            }
        }
    }

    private var shows: [(String, String)] {
        [("keyboard", "Keyboard"), ("mouse", "Mouse"), ("stats", "Stats")] + (showsVision ? [("vision", "What it sees")] : [])
    }

    private func isOn(_ k: String) -> Bool {
        switch k { case "keyboard": options.keyboard; case "mouse": options.mouse; case "stats": options.stats; default: options.vision }
    }

    private func toggle(_ k: String) {
        switch k {
        case "keyboard": options.keyboard.toggle()
        case "mouse": options.mouse.toggle()
        case "stats": options.stats.toggle()
        default: options.vision.toggle()
        }
    }
}
