import ATCore
import AVKit
import SwiftUI

struct RecordingDetailView: View {
    let app: AppModel
    let recording: LibraryRecording
    @State private var player: AVPlayer
    @State private var annotations: RecordingAnnotations
    @State private var activity: [ActivityRow] = []
    @State private var currentTime = 0.0
    @State private var timeObserver: Any?

    init(app: AppModel, recording: LibraryRecording) {
        self.app = app
        self.recording = recording
        _player = State(initialValue: AVPlayer(url: recording.url.appendingPathComponent(RecordingPackage.video)))
        _annotations = State(initialValue: RecordingIO.readAnnotations(recording.url))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Text(recording.meta.name).font(.title2.weight(.semibold)).lineLimit(1)
                    Spacer()
                    Menu {
                        ForEach(app.library.flatFolders, id: \.url) { f in
                            Button(String(repeating: "  ", count: f.depth) + f.name) { app.library.move(recording.url, into: f.url) }
                                .disabled(f.url.standardizedFileURL == recording.folder.standardizedFileURL)
                        }
                    } label: { Label("Move to", systemImage: "folder") }
                    .fixedSize()
                    Button { NSWorkspace.shared.activateFileViewerSelecting([recording.url]) } label: { Image(systemName: "magnifyingglass") }
                        .help("Show in Finder")
                }
                VideoPlayer(player: player)
                    .aspectRatio(CGFloat(recording.meta.captureWidth) / CGFloat(max(recording.meta.captureHeight, 1)), contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                ActivityTimeline(rows: activity, duration: recording.meta.duration, current: currentTime, annotations: annotations) { t in
                    player.seek(to: CMTime(seconds: t, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
                }
                .frame(height: CGFloat(max(activity.count, 1)) * 14 + 34)
                infoGrid
                AnnotationEditor(annotations: $annotations, currentTime: currentTime, duration: recording.meta.duration, app: app, recording: recording)
            }
            .padding(16)
        }
        .onChange(of: annotations) { _, new in try? RecordingIO.writeAnnotations(new, to: recording.url) }
        .task {
            let url = recording.url, meta = recording.meta
            activity = await Task.detached { ActivityRow.load(url: url, meta: meta) }.value
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

    private var infoGrid: some View {
        let m = recording.meta
        return Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
            GridRow { Text("Target").foregroundStyle(.secondary); Text(m.target.displayName) }
            GridRow { Text("Video").foregroundStyle(.secondary); Text("\(m.captureWidth)×\(m.captureHeight) @ \(m.captureFPS) fps · \(m.frameCount) frames") }
            GridRow { Text("Input").foregroundStyle(.secondary); Text("\(m.eventCount) events · clock \(m.eventClock)") }
            GridRow { Text("Recorded").foregroundStyle(.secondary); Text(m.createdAt.formatted(date: .abbreviated, time: .shortened)) }
            if let s = m.mouseDeltaStd { GridRow { Text("Mouse").foregroundStyle(.secondary); Text(String(format: "σ %.1f × %.1f pt per 50 ms", s[0], s[1])) } }
            GridRow { Text("Location").foregroundStyle(.secondary); Text(app.library.relativeName(recording.folder)) }
        }
        .font(.callout)
    }
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
            return ActivityRow(id: key, label: label, active: steps.map { isPointer ? $0.action.pointer.contains(PointerEvent(rawValue: String(key.dropFirst(2)))!) : $0.action.keys.contains(key) })
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
            let labelW: CGFloat = 70, w = geo.size.width - labelW
            ZStack(alignment: .topLeading) {
                Canvas { ctx, size in
                    // Instruction segments and exclusions along the top band.
                    for s in annotations.segments {
                        let x0 = labelW + w * min(1, s.start / max(duration, 0.01)), x1 = labelW + w * min(1, s.end / max(duration, 0.01))
                        ctx.fill(Path(CGRect(x: x0, y: 0, width: max(2, x1 - x0), height: 8)), with: .color(.accentColor.opacity(0.5)))
                    }
                    for r in annotations.excluded {
                        let x0 = labelW + w * r.start / max(duration, 0.01), x1 = labelW + w * r.end / max(duration, 0.01)
                        ctx.fill(Path(CGRect(x: x0, y: 0, width: max(2, x1 - x0), height: size.height)), with: .color(.gray.opacity(0.25)))
                    }
                    for (i, row) in rows.enumerated() {
                        let y = 14 + CGFloat(i) * 14
                        ctx.draw(Text(row.label).font(.caption2).foregroundStyle(.secondary), at: CGPoint(x: 4, y: y + 5), anchor: .leading)
                        let n = max(row.active.count, 1)
                        var run: Int?
                        for j in 0 ... row.active.count {
                            let on = j < row.active.count && row.active[j]
                            if on, run == nil { run = j }
                            if !on, let st = run {
                                ctx.fill(Path(roundedRect: CGRect(x: labelW + w * CGFloat(st) / CGFloat(n), y: y, width: max(1.5, w * CGFloat(j - st) / CGFloat(n)), height: 10), cornerRadius: 2),
                                         with: .color(row.id == "mouse" ? .purple.opacity(0.7) : .accentColor))
                                run = nil
                            }
                        }
                    }
                    let x = labelW + w * min(1, current / max(duration, 0.01))
                    ctx.fill(Path(CGRect(x: x - 0.75, y: 0, width: 1.5, height: size.height)), with: .color(.red))
                }
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0).onChanged { v in
                    seek(max(0, min(duration, Double((v.location.x - labelW) / w) * duration)))
                })
            }
        }
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
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
        Card(title: "Instructions", symbol: "text.bubble",
             trailing: AnyView(HStack {
                 Button { add() } label: { Label("Add at \(currentTime.durationText)", systemImage: "plus") }
                 AutoLabelButton(app: app, recording: recording, annotations: $annotations)
             })) {
            if annotations.segments.isEmpty {
                Text("No instructions yet. Add what you were doing at each point, or let the local AI write them.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            ForEach($annotations.segments) { $seg in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Image(systemName: icon(seg.source)).foregroundStyle(.secondary).help(seg.source.rawValue)
                    TimeField(value: $seg.start).frame(width: 64)
                    Text("–").foregroundStyle(.secondary)
                    TimeField(value: $seg.end).frame(width: 64)
                    TextField("Instruction", text: $seg.text)
                    Button(role: .destructive) { annotations.segments.removeAll { $0.id == seg.id } } label: { Image(systemName: "trash") }
                        .buttonStyle(.borderless)
                }
            }
        }
        Card(title: "Excluded from training", symbol: "scissors") {
            HStack {
                if let s = excludeStart {
                    Text("From \(s.durationText) to \(currentTime.durationText)")
                    Button("Exclude") {
                        annotations.excluded.append(TimeRange(start: min(s, currentTime), end: max(s, currentTime)))
                        excludeStart = nil
                    }
                    Button("Cancel") { excludeStart = nil }
                } else {
                    Button { excludeStart = currentTime } label: { Label("Mark start at \(currentTime.durationText)", systemImage: "arrow.right.to.line") }
                    Text("Use for menus, loading screens or mistakes.").font(.caption).foregroundStyle(.secondary)
                }
            }
            ForEach(annotations.excluded) { r in
                HStack {
                    Text("\(r.start.durationText) – \(r.end.durationText)").monospacedDigit()
                    Spacer()
                    Button(role: .destructive) { annotations.excluded.removeAll { $0.id == r.id } } label: { Image(systemName: "trash") }.buttonStyle(.borderless)
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
        TextField("", value: $value, format: .number.precision(.fractionLength(1))).monospacedDigit().multilineTextAlignment(.trailing)
    }
}
