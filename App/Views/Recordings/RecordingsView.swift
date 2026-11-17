import ATCapture
import ATCore
import SwiftUI

struct RecordingsView: View {
    @Bindable var app: AppModel

    var body: some View {
        @Bindable var settings = app.settings
        @Bindable var rec = app.recorder
        HStack(alignment: .top, spacing: 32) {
            stage(rec).riseIn()
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    targets(rec)
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
                            TextField("e.g. Drive to the gas station", text: $rec.task)
                                .textFieldStyle(.plain).multilineTextAlignment(.trailing).font(Theme.font(14)).frame(maxWidth: 220)
                        }
                        Row(title: "Name") {
                            TextField("Automatic", text: $rec.name)
                                .textFieldStyle(.plain).multilineTextAlignment(.trailing).font(Theme.font(14)).frame(maxWidth: 220)
                        }
                    }
                    .disabled(rec.isBusy)
                    Rows(title: "Input", footer: "Only events while the recorded window is in front are kept when that option is on.") {
                        Toggle("Keyboard", isOn: $settings.stored.input.keyboard)
                        Toggle("Mouse movement", isOn: $settings.stored.input.mouseMovement)
                        Toggle("Mouse buttons", isOn: $settings.stored.input.mouseButtons)
                        Toggle("Scrolling", isOn: $settings.stored.input.scroll)
                        Toggle("Only while the window is in front", isOn: $settings.stored.input.onlyWhenTargetFocused)
                    }
                    .toggleStyle(.oled)
                    .disabled(rec.isBusy)
                    Rows(title: "Video") {
                        Toggle("Show the mouse pointer", isOn: $settings.stored.showsCursor).toggleStyle(.oled)
                        LabeledSlider(title: "Quality", value: $settings.stored.videoQuality, range: 0.1 ... 1, format: { "\(Int($0 * 100))%" })
                    }
                    .disabled(rec.isBusy)
                    HUDRows(title: "Recording HUD", options: $settings.stored.recordHUD, showsVision: false)
                }
                .padding(.bottom, 32)
            }
            .scrollIndicators(.never)
            .frame(width: 380)
            .riseIn(0.08)
        }
        .padding([.horizontal, .top], 32)
        .task { if rec.targets.isEmpty { await rec.refreshTargets() } }
        .task(id: rec.selectedTargetID) {
            rec.preview = nil
            await rec.runPreview { [weak rec] in rec?.phase == .idle }
        }
        .onAppear { if !app.library.flatFolders.contains(where: { $0.url == rec.destination }) { rec.destination = app.library.fs.root } }
    }

    // MARK: Stage: preview, record controls, live numbers

    func stage(_ rec: RecorderController) -> some View {
        VStack(alignment: .leading, spacing: 22) {
            HStack(alignment: .firstTextBaseline) {
                Text(title(rec)).displayStyle(48).contentTransition(.opacity)
                Spacer()
                if let s = rec.plannedSettings {
                    Text(verbatim: "\(rec.selectedTarget?.name ?? "") · \(s.width) × \(s.height) · \(s.fps) fps").font(Theme.font(13)).foregroundStyle(Theme.muted).lineLimit(1)
                }
            }
            if let e = rec.errorMessage { InlineError(message: e) }
            preview(rec)
            HStack(alignment: .center, spacing: 20) {
                recordButton(rec)
                Text("Save \(app.settings.stored.recordHotkey.display) · Discard \(app.settings.stored.cancelRecordingHotkey.display)")
                    .font(Theme.font(12.5)).foregroundStyle(Theme.muted)
                Spacer()
                HStack(spacing: 36) {
                    stat(rec.stats.elapsed.durationText, "time")
                    stat("\(rec.stats.frames)", "frames")
                    stat("\(rec.stats.droppedFrames)", "dropped", color: rec.stats.droppedFrames > 0 ? Theme.warning : Theme.text)
                    stat("\(rec.stats.events)", "input events")
                    stat(rec.stats.frames > 0 ? String(format: "%.0f", rec.stats.averageFPS) : "–", "fps")
                }
            }
            .padding(.bottom, 32)
        }
        .frame(maxWidth: .infinity)
        .animation(Theme.spring, value: rec.phase)
    }

    func preview(_ rec: RecorderController) -> some View {
        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 20, style: .continuous).fill(Theme.surface)
            if let id = rec.selectedTargetID, let img = rec.preview ?? rec.thumbnails[id] {
                Image(nsImage: img).resizable().aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .padding(18)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 10) {
                    Image(systemName: "macwindow").font(.system(size: 34, weight: .light)).foregroundStyle(Theme.faint)
                    Text(rec.targets.isEmpty ? "Open the game or app, then refresh the list" : "Choose what to record").font(Theme.font(13)).foregroundStyle(Theme.muted)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            if rec.phase == .recording {
                HStack(spacing: 8) { LiveDot(color: Theme.danger); Text("REC").font(Theme.font(12, .bold)).tracking(0.8) }
                    .foregroundStyle(Theme.dangerText)
                    .padding(.horizontal, 12).padding(.vertical, 6)
                    .background(Theme.danger.opacity(0.18), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .padding(18)
                    .transition(.opacity.combined(with: .scale(scale: 0.9, anchor: .topLeading)))
            }
            if case .countdown(let n) = rec.phase {
                Text("\(n)").displayStyle(120).frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                    .contentTransition(.numericText(countsDown: true))
                    .transition(.opacity)
            }
        }
        .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).strokeBorder(rec.phase == .recording ? Theme.danger.opacity(0.5) : Theme.hairline, lineWidth: rec.phase == .recording ? 1.5 : 1))
    }

    func recordButton(_ rec: RecorderController) -> some View {
        Button { Task { await rec.toggle() } } label: {
            switch rec.phase {
            case .idle: Label("Record", systemImage: "record.circle")
            case .countdown: Label("Cancel", systemImage: "xmark")
            case .recording: Label("Stop and save", systemImage: "stop.fill")
            case .saving: Label("Saving…", systemImage: "arrow.down.circle")
            }
        }
        .buttonStyle(rec.phase == .recording ? ATButtonStyle.atDestructive : ATButtonStyle.atPrimary)
        .help("Start or stop and save (\(app.settings.stored.recordHotkey.display))")
        .disabled(rec.phase == .saving || (rec.phase == .idle && rec.selectedTarget == nil))
    }

    func stat(_ v: String, _ caption: String, color: Color = Theme.text) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(v).font(Theme.font(22, .semibold)).monospacedDigit().foregroundStyle(color).contentTransition(.numericText())
            Text(caption).font(Theme.font(12)).foregroundStyle(Theme.muted)
        }
    }

    func title(_ r: RecorderController) -> String {
        switch r.phase {
        case .idle: "Ready to record"
        case .countdown: "Get ready"
        case .recording: "Recording"
        case .saving: "Saving…"
        }
    }

    // MARK: Targets

    func targets(_ rec: RecorderController) -> some View {
        @Bindable var rec = rec
        return VStack(alignment: .leading, spacing: 10) {
            HStack {
                SectionLabel("What to record")
                Spacer()
                Button { Task { await rec.refreshTargets() } } label: {
                    Image(systemName: "arrow.clockwise").font(.system(size: 12, weight: .semibold))
                        .rotationEffect(.degrees(rec.loadingTargets ? 360 : 0))
                        .animation(rec.loadingTargets ? .linear(duration: 0.8).repeatForever(autoreverses: false) : .default, value: rec.loadingTargets)
                }
                .buttonStyle(.atQuiet)
                .help("Refresh the list")
                .disabled(rec.isBusy)
            }
            VStack(spacing: 4) {
                ForEach(rec.targets) { t in
                    let on = rec.selectedTargetID == t.id
                    Button { withAnimation(Theme.snappy) { rec.selectedTargetID = t.id } } label: {
                        HStack(spacing: 12) {
                            Group {
                                if let img = rec.thumbnails[t.id] {
                                    Image(nsImage: img).resizable().aspectRatio(contentMode: .fill).frame(width: 56, height: 34).clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                                } else {
                                    Placeholder(symbol: t.isDisplay ? "display" : "macwindow", radius: 6).frame(width: 56, height: 34)
                                }
                            }
                            VStack(alignment: .leading, spacing: 2) {
                                Text(t.name).font(Theme.font(14, .semibold)).lineLimit(1)
                                Text(t.isDisplay ? "Display" : "Window · \(t.info.appName ?? "")").font(Theme.font(12)).foregroundStyle(Theme.muted).lineLimit(1)
                            }
                            Spacer(minLength: 0)
                            if on { Image(systemName: "checkmark").font(.system(size: 12, weight: .bold)).transition(.scale.combined(with: .opacity)) }
                        }
                        .padding(.horizontal, 10).padding(.vertical, 8)
                    }
                    .buttonStyle(RowButtonStyle(selected: on, radius: 12))
                }
                if rec.targets.isEmpty && !rec.loadingTargets {
                    Text("No windows found. Open the game or app, then refresh.").font(Theme.font(13)).foregroundStyle(Theme.muted).padding(10)
                }
            }
            .disabled(rec.isBusy)
        }
    }
}

/// HUD options as rows inside a grouped panel.
struct HUDRows: View {
    let title: String
    @Binding var options: HUDOptions
    var showsVision: Bool
    var body: some View {
        Rows(title: title, footer: "The HUD is hidden from recordings and from the model's view.") {
            Toggle("Show HUD", isOn: $options.enabled.animation(Theme.spring)).toggleStyle(.oled)
            if options.enabled {
                Toggle("Keyboard", isOn: $options.keyboard).toggleStyle(.oled)
                Toggle("Mouse", isOn: $options.mouse).toggleStyle(.oled)
                Toggle("Stats", isOn: $options.stats).toggleStyle(.oled)
                if showsVision { Toggle("What the model sees", isOn: $options.vision).toggleStyle(.oled) }
                Row(title: "Corner") {
                    Picker("", selection: $options.corner) { ForEach(HUDCorner.allCases) { Text($0.label).tag($0) } }.labelsHidden().fixedSize()
                }
                LabeledSlider(title: "Size", value: $options.scale, range: 0.6 ... 1.8, format: { "\(Int($0 * 100))%" })
                LabeledSlider(title: "Opacity", value: $options.opacity, range: 0.3 ... 1, format: { "\(Int($0 * 100))%" })
            }
        }
    }
}

/// Kept for other screens: HUD options as plain controls.
struct HUDOptionsEditor: View {
    @Binding var options: HUDOptions
    var showsVision: Bool
    var body: some View { HUDRows(title: "HUD", options: $options, showsVision: showsVision) }
}

struct TargetGrid: View {
    let targets: [CaptureTarget]
    let thumbnails: [String: NSImage]
    @Binding var selection: String?

    var body: some View {
        if targets.isEmpty {
            Text("No windows found yet. Open the game or app, then press Refresh.").font(Theme.font(13)).foregroundStyle(Theme.muted)
        } else {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 180, maximum: 260), spacing: 14)], spacing: 14) {
                ForEach(targets) { t in
                    let on = selection == t.id
                    Button { withAnimation(Theme.snappy) { selection = t.id } } label: {
                        VStack(alignment: .leading, spacing: 8) {
                            Color.clear
                                .frame(height: 104).frame(maxWidth: .infinity)
                                .overlay {
                                    if let img = thumbnails[t.id] {
                                        Image(nsImage: img).resizable().scaledToFill()
                                    } else {
                                        Placeholder(symbol: t.isDisplay ? "display" : "macwindow", radius: 9)
                                    }
                                }
                                .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
                            VStack(alignment: .leading, spacing: 2) {
                                Text(t.name).font(Theme.font(13, .semibold)).lineLimit(1)
                                Text(t.isDisplay ? "Display" : "Window").font(Theme.font(12)).foregroundStyle(Theme.muted)
                            }
                        }
                        .padding(10)
                        .background(on ? Theme.raised : Theme.surface, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).strokeBorder(on ? Theme.text : Theme.hairline, lineWidth: on ? 1.5 : 1))
                        .scaleEffect(on ? 1 : 0.985)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
}
