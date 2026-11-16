import ATCapture
import ATCore
import SwiftUI

struct RecordingsView: View {
    @Bindable var app: AppModel
    private var rec: RecorderController { app.recorder }

    var body: some View {
        @Bindable var settings = app.settings
        @Bindable var rec = app.recorder
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                PageHeader(title: "Recordings", subtitle: "Record yourself playing or using an app. Screen, keys, mouse and scroll are captured on one clock.")
                RecordControlCard(app: app)
                if let e = rec.errorMessage { InlineError(message: e) }
                Card(title: "What to record", symbol: "macwindow",
                     trailing: AnyView(Button { Task { await rec.refreshTargets() } } label: {
                         if rec.loadingTargets { ProgressView().controlSize(.small) } else { Label("Refresh", systemImage: "arrow.clockwise") }
                     }.disabled(rec.isBusy))) {
                    TargetGrid(targets: rec.targets, thumbnails: rec.thumbnails, selection: $rec.selectedTargetID)
                        .disabled(rec.isBusy)
                }
                HStack(alignment: .top, spacing: 18) {
                    Card(title: "Capture", symbol: "camera.viewfinder") {
                        Picker("Resolution", selection: $settings.stored.captureMaxHeight) {
                            Text("480p").tag(480); Text("720p").tag(720); Text("1080p").tag(1080); Text("1440p").tag(1440)
                        }
                        Picker("Frame rate", selection: $settings.stored.captureFPS) {
                            Text("30 fps").tag(30); Text("60 fps").tag(60)
                        }
                        .pickerStyle(.segmented)
                        Toggle("Show the mouse pointer in the video", isOn: $settings.stored.showsCursor)
                        LabeledSlider(title: "Video quality", value: $settings.stored.videoQuality, range: 0.1 ... 1, format: { "\(Int($0 * 100))%" })
                        if let s = rec.plannedSettings {
                            Text(verbatim: "\(s.width)×\(s.height) at \(s.fps) fps · training frames saved losslessly at 192×192")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .disabled(rec.isBusy)
                    Card(title: "Input", symbol: "keyboard") {
                        Toggle("Keyboard", isOn: $settings.stored.input.keyboard)
                        Toggle("Mouse movement", isOn: $settings.stored.input.mouseMovement)
                        Toggle("Mouse buttons", isOn: $settings.stored.input.mouseButtons)
                        Toggle("Scrolling", isOn: $settings.stored.input.scroll)
                        Toggle("Only while the recorded window is in front", isOn: $settings.stored.input.onlyWhenTargetFocused)
                    }
                    .disabled(rec.isBusy)
                }
                HStack(alignment: .top, spacing: 18) {
                    Card(title: "Save to", symbol: "folder") {
                        Picker("Library folder", selection: $rec.destination) {
                            ForEach(app.library.flatFolders, id: \.url) { f in
                                Text(String(repeating: "    ", count: f.depth) + f.name).tag(f.url)
                            }
                        }
                        TextField("Name (optional)", text: $rec.name)
                        TextField("What are you doing? e.g. “Collect wood and build a house”", text: $rec.task, axis: .vertical)
                            .lineLimit(1 ... 3)
                        Text("The task becomes the instruction for language-enabled models. You can add timed instructions in the Library later.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    .disabled(rec.isBusy)
                    Card(title: "Recording HUD", symbol: "rectangle.inset.bottomright.filled") {
                        HUDOptionsEditor(options: $settings.stored.recordHUD, showsVision: false)
                    }
                }
            }
            .padding(24)
        }
        .task { if rec.targets.isEmpty { await rec.refreshTargets() } }
        .onAppear { if !app.library.flatFolders.contains(where: { $0.url == rec.destination }) { rec.destination = app.library.fs.root } }
    }
}

struct RecordControlCard: View {
    let app: AppModel
    var body: some View {
        let rec = app.recorder
        Card(title: "Session", symbol: "record.circle") {
            HStack(spacing: 18) {
                Button { Task { await rec.toggle() } } label: {
                    ZStack {
                        Circle().fill(rec.phase == .recording ? Color.red : Color.red.opacity(0.85)).frame(width: 64, height: 64)
                        if rec.phase == .recording {
                            RoundedRectangle(cornerRadius: 5).fill(.white).frame(width: 22, height: 22)
                        } else if case .countdown(let n) = rec.phase {
                            Text("\(n)").font(.title.bold()).foregroundStyle(.white)
                        } else if rec.phase == .saving {
                            ProgressView().controlSize(.small).tint(.white)
                        } else {
                            Circle().fill(.white).frame(width: 24, height: 24)
                        }
                    }
                }
                .buttonStyle(.plain)
                .disabled(rec.phase == .saving)
                .help("Start or stop recording (\(app.settings.stored.recordHotkey.display))")
                VStack(alignment: .leading, spacing: 4) {
                    Text(title(rec)).font(.title2.weight(.semibold))
                    Text(rec.selectedTarget.map { "Target: \($0.name)" } ?? "No target selected").foregroundStyle(.secondary)
                    Text("Save: \(app.settings.stored.recordHotkey.display) · Discard: \(app.settings.stored.cancelRecordingHotkey.display)").font(.caption).foregroundStyle(.tertiary)
                }
                Spacer()
                if let last = rec.lastSaved, rec.phase == .idle {
                    Button { app.library.selectedFolder = last.deletingLastPathComponent(); app.tab = .library } label: {
                        Label("Open last recording", systemImage: "film")
                    }
                }
            }
            if rec.phase == .recording || rec.stats.frames > 0 {
                HStack(spacing: 10) {
                    StatTile(title: "Duration", value: rec.stats.elapsed.durationText, symbol: "clock")
                    StatTile(title: "Frames", value: "\(rec.stats.frames)", symbol: "photo.stack")
                    StatTile(title: "Average FPS", value: String(format: "%.1f", rec.stats.averageFPS), symbol: "speedometer")
                    StatTile(title: "Input events", value: "\(rec.stats.events)", symbol: "keyboard")
                    StatTile(title: "Encoder drops", value: "\(rec.stats.droppedFrames)", symbol: "exclamationmark.triangle",
                             tint: rec.stats.droppedFrames > 0 ? .orange : .accentColor)
                }
            }
        }
    }

    func title(_ r: RecorderController) -> String {
        switch r.phase {
        case .idle: "Ready to record"
        case .countdown(let n): "Starting in \(n)…"
        case .recording: "Recording"
        case .saving: "Saving…"
        }
    }
}

struct TargetGrid: View {
    let targets: [CaptureTarget]
    let thumbnails: [String: NSImage]
    @Binding var selection: String?

    var body: some View {
        if targets.isEmpty {
            Text("No windows found yet. Open the game or app, then press Refresh.").foregroundStyle(.secondary)
        } else {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 170, maximum: 230), spacing: 12)], spacing: 12) {
                ForEach(targets) { t in
                    Button { selection = t.id } label: {
                        VStack(alignment: .leading, spacing: 6) {
                            ZStack {
                                RoundedRectangle(cornerRadius: 8).fill(.quaternary)
                                if let img = thumbnails[t.id] {
                                    Image(nsImage: img).resizable().aspectRatio(contentMode: .fit).clipShape(RoundedRectangle(cornerRadius: 6)).padding(4)
                                } else {
                                    Image(systemName: t.isDisplay ? "display" : "macwindow").font(.largeTitle).foregroundStyle(.secondary)
                                }
                            }
                            .frame(height: 100)
                            Label(t.name, systemImage: t.isDisplay ? "display" : "macwindow").font(.callout).lineLimit(1)
                        }
                        .padding(8)
                        .background(RoundedRectangle(cornerRadius: 10).fill(selection == t.id ? Color.accentColor.opacity(0.18) : .clear))
                        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(selection == t.id ? Color.accentColor : .clear, lineWidth: 2))
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
}

struct HUDOptionsEditor: View {
    @Binding var options: HUDOptions
    var showsVision: Bool
    var body: some View {
        Toggle("Show HUD", isOn: $options.enabled)
        Group {
            HStack {
                Toggle("Keyboard", isOn: $options.keyboard)
                Toggle("Mouse", isOn: $options.mouse)
                Toggle("Stats", isOn: $options.stats)
                if showsVision { Toggle("AI vision", isOn: $options.vision) }
            }
            Picker("Corner", selection: $options.corner) { ForEach(HUDCorner.allCases) { Text($0.label).tag($0) } }
            LabeledSlider(title: "Size", value: $options.scale, range: 0.6 ... 1.8, format: { "\(Int($0 * 100))%" })
            LabeledSlider(title: "Opacity", value: $options.opacity, range: 0.3 ... 1, format: { "\(Int($0 * 100))%" })
        }
        .disabled(!options.enabled)
        Text("The HUD is hidden from recordings and from the model's view.").font(.caption).foregroundStyle(.secondary)
    }
}
