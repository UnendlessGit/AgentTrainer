import ATCore
import ATRuntime
import AVKit
import SwiftUI

struct RecordingDetailView: View {
    let app: AppModel
    let recording: LibraryRecording
    var onRename: () -> Void = {}
    var onTrash: () -> Void = {}
    @State private var player: AVPlayer
    @State private var annotations: RecordingAnnotations
    @State private var activity: [ActivityRow] = []
    @State private var currentTime = 0.0
    @State private var timeObserver: Any?
    @State private var comparison: ReplayInspector.Result?
    @State private var comparing = false
    @State private var compareError: String?

    init(app: AppModel, recording: LibraryRecording, onRename: @escaping () -> Void = {}, onTrash: @escaping () -> Void = {}) {
        self.app = app
        self.recording = recording
        self.onRename = onRename
        self.onTrash = onTrash
        _player = State(initialValue: AVPlayer(url: recording.url.appendingPathComponent(RecordingPackage.video)))
        _annotations = State(initialValue: RecordingIO.readAnnotations(recording.url))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                HStack(alignment: .bottom, spacing: 16) {
                    VStack(alignment: .leading, spacing: 8) {
                        SectionLabel(app.library.relativeName(recording.folder))
                        Text(recording.meta.name).displayStyle(38).lineLimit(1).truncationMode(.middle)
                    }
                    Spacer(minLength: 12)
                    HStack(spacing: 8) {
                        Button("Rename", action: onRename).buttonStyle(.atSmall)
                        Menu {
                            ForEach(app.library.flatFolders, id: \.url) { f in
                                Button(String(repeating: "  ", count: f.depth) + f.name) { app.library.move([recording.url], into: f.url) }
                                    .disabled(f.url.standardizedFileURL == recording.folder.standardizedFileURL)
                            }
                        } label: { Text("Move to…") }
                        .menuStyle(.button).buttonStyle(.atSmall).fixedSize()
                        Button { NSWorkspace.shared.activateFileViewerSelecting([recording.url]) } label: { Image(systemName: "folder") }
                            .buttonStyle(.atSmall).help("Show in Finder")
                        Button("Move to Trash", action: onTrash).buttonStyle(.atSmallDestructive)
                    }
                }
                .riseIn()

                HStack(alignment: .top, spacing: 22) {
                    VStack(alignment: .leading, spacing: 14) {
                        PlayerView(player: player)
                            .aspectRatio(CGFloat(recording.meta.captureWidth) / CGFloat(max(recording.meta.captureHeight, 1)), contentMode: .fit)
                            .background(Theme.surface)
                            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).strokeBorder(Theme.hairline))
                        Panel(padding: 14) {
                            VStack(alignment: .leading, spacing: 10) {
                                HStack {
                                    Text("Input activity").font(Theme.font(13, .semibold))
                                    Spacer()
                                    Text("\(currentTime.durationText) / \(recording.meta.duration.durationText)").font(Theme.font(12)).monospacedDigit().foregroundStyle(Theme.muted)
                                }
                                ActivityTimeline(rows: activity, duration: recording.meta.duration, current: currentTime, annotations: annotations) { t in
                                    player.seek(to: CMTime(seconds: t, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
                                }
                                .frame(height: CGFloat(max(activity.count, 1)) * 16 + 30)
                            }
                        }
                        replayInspector
                    }
                    .riseIn(0.06)
                    VStack(alignment: .leading, spacing: 14) {
                        infoRows
                        AnnotationEditor(annotations: $annotations, currentTime: currentTime, duration: recording.meta.duration, app: app, recording: recording)
                    }
                    .frame(width: 340)
                    .riseIn(0.12)
                }
            }
            .padding(.horizontal, 36).padding(.vertical, 28)
        }
        .scrollIndicators(.never)
        .onChange(of: annotations) { _, new in try? RecordingIO.writeAnnotations(new, to: recording.url) }
        .task {
            let url = recording.url, meta = recording.meta
            let rows = await Task.detached { ActivityRow.load(url: url, meta: meta) }.value
            withAnimation(Theme.spring) { activity = rows }
        }
        .onAppear {
            timeObserver = player.addPeriodicTimeObserver(forInterval: CMTime(value: 1, timescale: 20), queue: .main) { t in
                MainActor.assumeIsolated { currentTime = t.seconds }
            }
        }
        .onDisappear {
            player.pause()
            if let o = timeObserver { player.removeTimeObserver(o) }
        }
    }

    /// Replay Inspector: what a trained model would have pressed at each moment, under yours.
    private var replayInspector: some View {
        let trained = app.models.models.filter { !$0.checkpoints.isEmpty }
        return Panel {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Compare with a model").font(Theme.font(14, .semibold))
                        Text(trained.isEmpty ? "Train a model first." : "Runs a model on the next 60 s from the playhead, with your real action history.")
                            .font(Theme.font(12)).foregroundStyle(Theme.muted)
                    }
                    Spacer()
                    Menu {
                        ForEach(trained) { m in Button(m.definition.name) { Task { await compare(m) } } }
                    } label: { Text(comparing ? "Running…" : "Choose model") }
                    .menuStyle(.button).buttonStyle(.atSmall).fixedSize()
                    .disabled(comparing || trained.isEmpty)
                }
                if let e = compareError { Text(e).font(Theme.font(12)).foregroundStyle(Theme.warning) }
                if let c = comparison {
                    Text(String(format: "From %@ the model chose exactly your keys %.0f%% of the time", c.start.durationText, c.keyAgreement * 100)).font(Theme.font(13)).foregroundStyle(Theme.text2)
                    KeyComparisonStrip(result: c).frame(height: 120).transition(.opacity)
                }
            }
        }
        .animation(Theme.spring, value: comparison?.start)
    }

    private func compare(_ m: ModelEntry) async {
        comparing = true; compareError = nil
        defer { comparing = false }
        let url = m.url, rec = recording.url, start = currentTime
        do {
            comparison = try await Task.detached(priority: .userInitiated) { try ReplayInspector.compare(modelURL: url, recording: rec, from: start, seconds: 60) }.value
        } catch { compareError = error.localizedDescription }
    }

    private var infoRows: some View {
        let m = recording.meta
        return Rows {
            Row(title: "Target") { value(m.target.displayName) }
            Row(title: "Video") { value("\(m.captureWidth) × \(m.captureHeight) · \(m.captureFPS) fps") }
            Row(title: "Frames") { value("\(m.frameCount) · \(m.droppedFrames) dropped") }
            Row(title: "Input") { value("\(m.eventCount) events") }
            Row(title: "Recorded") { value(m.createdAt.formatted(date: .abbreviated, time: .shortened)) }
            if let s = m.mouseDeltaStd { Row(title: "Mouse") { value(String(format: "σ %.1f × %.1f", s[0], s[1])) } }
        }
    }

    private func value(_ s: String) -> some View { Text(s).font(Theme.font(13)).foregroundStyle(Theme.text2).lineLimit(1).truncationMode(.middle) }
}

/// Per-key / per-button activity over time, sampled at 10 Hz.
struct ActivityRow: Identifiable, Sendable {
    let id: String
    let label: String
    let active: [Bool]

    static func load(url: URL, meta: RecordingMeta) -> [ActivityRow] {
        guard let frames = try? RecordingIO.readFrames(url), let events = try? RecordingIO.readEvents(url) else { return [] }
        let steps = TimelineBuilder.build(frames: frames, events: events, meta: meta, annotations: RecordingAnnotations(), rateHz: 10)
        var counts: [String: Int] = [:]
        for s in steps { for k in s.action.keys { counts[k, default: 0] += 1 }; for p in s.action.pointer { counts["p:" + p.rawValue, default: 0] += 1 } }
        var rows = counts.sorted { $0.value > $1.value }.prefix(10).map { key, _ -> ActivityRow in
            let isPointer = key.hasPrefix("p:")
            let label = isPointer ? PointerEvent(rawValue: String(key.dropFirst(2)))?.label ?? key : KeyCatalog.info(key)?.label.uppercased() ?? key
            return ActivityRow(id: key, label: label.count > 5 ? String(label.prefix(4)) + "…" : label, active: steps.map { isPointer ? $0.action.pointer.contains(PointerEvent(rawValue: String(key.dropFirst(2)))!) : $0.action.keys.contains(key) })
        }
        let moving = steps.map { abs($0.action.dx) + abs($0.action.dy) > 1 }
        if moving.contains(true) { rows.append(ActivityRow(id: "mouse", label: "Mouse", active: moving)) }
        return rows
    }
}

struct ActivityTimeline: View {
    let rows: [ActivityRow]
    let duration: Double
    let current: Double
    let annotations: RecordingAnnotations
    let seek: (Double) -> Void

    var body: some View {
        GeometryReader { geo in
            let labelW: CGFloat = 56, w = geo.size.width - labelW
            ZStack(alignment: .topLeading) {
                Canvas { ctx, size in
                    // Instruction segments and exclusions along the top band.
                    for s in annotations.segments {
                        let x0 = labelW + w * min(1, s.start / max(duration, 0.01)), x1 = labelW + w * min(1, s.end / max(duration, 0.01))
                        ctx.fill(Path(CGRect(x: x0, y: 0, width: max(2, x1 - x0), height: 8)), with: .color(Theme.accent.opacity(0.55)))
                    }
                    for r in annotations.excluded {
                        let x0 = labelW + w * r.start / max(duration, 0.01), x1 = labelW + w * r.end / max(duration, 0.01)
                        ctx.fill(Path(CGRect(x: x0, y: 0, width: max(2, x1 - x0), height: size.height)), with: .color(Color.white.opacity(0.08)))
                    }
                    for (i, row) in rows.enumerated() {
                        let y = 14 + CGFloat(i) * 14
                        ctx.draw(Text(row.label).font(Theme.font(10, .medium)).foregroundStyle(Theme.muted), at: CGPoint(x: 2, y: y + 5), anchor: .leading)
                        let n = max(row.active.count, 1)
                        var run: Int?
                        for j in 0 ... row.active.count {
                            let on = j < row.active.count && row.active[j]
                            if on, run == nil { run = j }
                            if !on, let st = run {
                                ctx.fill(Path(roundedRect: CGRect(x: labelW + w * CGFloat(st) / CGFloat(n), y: y, width: max(1.5, w * CGFloat(j - st) / CGFloat(n)), height: 10), cornerRadius: 2),
                                         with: .color(row.id == "mouse" ? Theme.accentLine.opacity(0.55) : Theme.accent))
                                run = nil
                            }
                        }
                    }
                    let x = labelW + w * min(1, current / max(duration, 0.01))
                    ctx.fill(Path(CGRect(x: x - 0.75, y: 0, width: 1.5, height: size.height)), with: .color(Theme.text))
                }
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0).onChanged { v in
                    seek(max(0, min(duration, Double((v.location.x - labelW) / w) * duration)))
                })
            }
        }
    }
}

struct AnnotationEditor: View {
    @Binding var annotations: RecordingAnnotations
    let currentTime: Double
    let duration: Double
    let app: AppModel
    let recording: LibraryRecording
    @State private var excludeStart: Double?

    var body: some View {
        Panel {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Instructions").font(Theme.font(14, .semibold))
                    Spacer()
                    AutoLabelButton(app: app, recording: recording, annotations: $annotations)
                }
                if annotations.segments.isEmpty {
                    Text("No instructions yet. Add what you were doing at each point, or let the local AI write them.")
                        .font(Theme.font(12.5)).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
                }
                ForEach($annotations.segments) { $seg in
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 6) {
                            Image(systemName: icon(seg.source)).font(.system(size: 11)).foregroundStyle(Theme.muted).help(seg.source.rawValue)
                            TimeField(value: $seg.start).frame(width: 52)
                            Text("–").foregroundStyle(Theme.muted)
                            TimeField(value: $seg.end).frame(width: 52)
                            Spacer()
                            Button { withAnimation(Theme.spring) { annotations.segments.removeAll { $0.id == seg.id } } } label: { Image(systemName: "xmark").font(.system(size: 10, weight: .bold)) }
                                .buttonStyle(.atQuiet).help("Remove")
                        }
                        TextField("What were you doing?", text: $seg.text, axis: .vertical).textFieldStyle(DarkFieldStyle()).lineLimit(1 ... 3)
                    }
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }
                Button { withAnimation(Theme.spring) { add() } } label: { Label("Add at \(currentTime.durationText)", systemImage: "plus") }.buttonStyle(.atSmall)
            }
        }
        Panel {
            VStack(alignment: .leading, spacing: 12) {
                Text("Left out of training").font(Theme.font(14, .semibold))
                if annotations.excluded.isEmpty && excludeStart == nil {
                    Text("Mark menus, loading screens or mistakes so the model doesn't learn them.").font(Theme.font(12.5)).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
                }
                ForEach(annotations.excluded) { r in
                    HStack {
                        Text("\(r.start.durationText) – \(r.end.durationText)").font(Theme.font(13)).monospacedDigit()
                        Spacer()
                        Button { withAnimation(Theme.spring) { annotations.excluded.removeAll { $0.id == r.id } } } label: { Image(systemName: "xmark").font(.system(size: 10, weight: .bold)) }.buttonStyle(.atQuiet)
                    }
                }
                if let s = excludeStart {
                    HStack(spacing: 8) {
                        Text("\(s.durationText) → \(currentTime.durationText)").font(Theme.font(13)).monospacedDigit()
                        Spacer()
                        Button("Cancel") { excludeStart = nil }.buttonStyle(.atSmall)
                        Button("Exclude") {
                            withAnimation(Theme.spring) { annotations.excluded.append(TimeRange(start: min(s, currentTime), end: max(s, currentTime))) }
                            excludeStart = nil
                        }
                        .buttonStyle(.atSmallPrimary)
                    }
                } else {
                    Button { excludeStart = currentTime } label: { Label("Start at \(currentTime.durationText)", systemImage: "scissors") }.buttonStyle(.atSmall)
                }
            }
        }
    }

    func add() {
        let end = min(duration, currentTime + 10)
        annotations.segments.append(TextSegment(start: currentTime, end: end, text: "", source: .user))
        annotations.segments.sort { $0.start < $1.start }
    }

    func icon(_ s: TextSegment.Source) -> String {
        switch s {
        case .user: "person"
        case .auto: "sparkles"
        case .typing: "keyboard"
        case .task: "flag"
        }
    }
}

struct TimeField: View {
    @Binding var value: Double
    var body: some View {
        TextField("", value: $value, format: .number.precision(.fractionLength(1)))
            .textFieldStyle(.plain).font(Theme.font(12)).monospacedDigit().multilineTextAlignment(.center)
            .frame(height: 24).background(Theme.raised, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
    }
}

/// Your keys (top) vs. the model's (bottom) for the most used keys in the window.
struct KeyComparisonStrip: View {
    let result: ReplayInspector.Result
    var body: some View {
        let counts = result.human.flatMap(\.keys).reduce(into: [String: Int]()) { $0[$1, default: 0] += 1 }
            .merging(result.model.flatMap(\.keys).reduce(into: [String: Int]()) { $0[$1, default: 0] += 1 }, uniquingKeysWith: +)
        let keys = counts.sorted { $0.value > $1.value }.prefix(4).map(\.key)
        Canvas { ctx, size in
            let n = max(result.human.count, 1), labelW: CGFloat = 70, w = size.width - labelW
            let rowH = size.height / CGFloat(max(keys.count * 2, 1))
            for (ki, k) in keys.enumerated() {
                for (j, (seq, name)) in [(result.human, "You"), (result.model, "Model")].enumerated() {
                    let y = CGFloat(ki * 2 + j) * rowH
                    ctx.draw(Text("\(KeyCatalog.info(k)?.label.uppercased() ?? k) · \(name)").font(.caption2).foregroundStyle(.secondary), at: CGPoint(x: 2, y: y + rowH / 2), anchor: .leading)
                    for (t, a) in seq.enumerated() where a.keys.contains(k) {
                        ctx.fill(Path(CGRect(x: labelW + w * CGFloat(t) / CGFloat(n), y: y + 2, width: max(1, w / CGFloat(n)), height: rowH - 4)),
                                 with: .color(j == 0 ? .accentColor : .green))
                    }
                }
            }
        }
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
    }
}

/// AppKit's player view. SwiftUI's `VideoPlayer` (AVKit's SwiftUI overlay) crashes on this macOS
/// version while building its type metadata, so the Library uses `AVPlayerView` directly.
struct PlayerView: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVPlayerView {
        let v = AVPlayerView()
        v.player = player
        v.controlsStyle = .inline
        v.showsFullScreenToggleButton = true
        v.videoGravity = .resizeAspect
        return v
    }

    func updateNSView(_ v: AVPlayerView, context: Context) {
        if v.player !== player { v.player = player }
    }

    static func dismantleNSView(_ v: AVPlayerView, coordinator: ()) {
        v.player?.pause()
        v.player = nil
    }
}
