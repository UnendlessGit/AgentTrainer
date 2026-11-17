import SwiftUI

struct RootView: View {
    @Bindable var app: AppModel

    var body: some View {
        ZStack(alignment: .top) {
            Theme.ground.ignoresSafeArea()
            Group {
                switch app.tab {
                case .recordings: RecordingsView(app: app)
                case .library: LibraryView(app: app)
                case .models: ModelsView(app: app)
                case .training: TrainingView(app: app)
                case .run: RunView(app: app)
                case .arena: ArenaView(app: app)
                case .settings: SettingsView(app: app)
                }
            }
            .id(app.tab)
            .transition(.asymmetric(insertion: .opacity.combined(with: .offset(y: 10)), removal: .opacity))
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            Rectangle().fill(Theme.hairline).frame(height: 1)
        }
        .animation(Theme.spring, value: app.tab)
        .foregroundStyle(Theme.text)
        .tint(Theme.text)
        .preferredColorScheme(.dark)
        .frame(minWidth: 1100, minHeight: 720)
        .navigationTitle("")
        .toolbar(removing: .title)
        .toolbar {
            ToolbarItem(placement: .navigation) { TabStrip(app: app) }
                .sharedBackgroundVisibility(.hidden)
            ToolbarSpacer(.flexible)
            ToolbarItem(placement: .primaryAction) { StatusCluster(app: app) }
                .sharedBackgroundVisibility(.hidden)
        }
        .toolbarBackgroundVisibility(.visible, for: .windowToolbar)
        .toolbarBackground(Theme.ground, for: .windowToolbar)
    }
}

extension AppTab {
    static let topBar: [AppTab] = [.recordings, .library, .models, .training, .run, .settings]
}

/// The tabs, built into the window's title bar: plain labels with a white underline that
/// glides to the selected tab.
struct TabStrip: View {
    @Bindable var app: AppModel
    @Namespace private var ns
    @State private var hover: AppTab?

    var body: some View {
        HStack(spacing: 0) {
            Text("AgentTrainer").font(Theme.font(14, .semibold)).foregroundStyle(Theme.text).padding(.leading, 6).padding(.trailing, 26)
            ForEach(Array(AppTab.topBar.enumerated()), id: \.element) { i, t in
                let on = app.tab == t
                Button { app.tab = t } label: {
                    Text(t.title)
                        .font(Theme.font(13.5, on ? .semibold : .medium))
                        .foregroundStyle(on ? Theme.text : hover == t ? Theme.text2 : Theme.muted)
                        .padding(.horizontal, 12)
                        .frame(height: 34)
                        .overlay(alignment: .bottom) {
                            if on {
                                Capsule().fill(Theme.text).frame(height: 2).padding(.horizontal, 12)
                                    .matchedGeometryEffect(id: "underline", in: ns)
                            }
                        }
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .keyboardShortcut(KeyEquivalent(Character("\(i + 1)")), modifiers: [.command, .option])
                .onHover { hover = $0 ? t : (hover == t ? nil : hover) }
                .animation(.easeOut(duration: 0.15), value: hover)
            }
        }
        .animation(Theme.spring, value: app.tab)
    }
}

/// Live activity on the right of the title bar, plus the Test Arena.
struct StatusCluster: View {
    let app: AppModel
    var body: some View {
        HStack(spacing: 16) {
            if app.recorder.phase == .recording {
                status(color: Theme.danger, "Recording \(app.recorder.stats.elapsed.durationText)")
            }
            if let line = app.training.footerLine {
                Button { app.tab = .training } label: { status(color: app.training.phase == .paused ? Theme.warning : Theme.live, line) }.buttonStyle(.plain)
            }
            if app.runner.isRunning {
                Button { app.tab = .run } label: { status(color: Theme.live, app.runner.footerLine) }.buttonStyle(.plain)
            }
            if app.recorder.phase != .recording, app.training.footerLine == nil, !app.runner.isRunning {
                Text(verbatim: "\(SystemInfo.chipName) · \(ProcessInfo.processInfo.physicalMemory / 1_073_741_824) GB").font(Theme.font(12.5)).foregroundStyle(Theme.muted)
            }
            Rectangle().fill(Theme.hairlineStrong).frame(width: 1, height: 16)
            Button { app.tab = .arena } label: {
                Image(systemName: "gamecontroller").font(.system(size: 14, weight: .medium))
                    .foregroundStyle(app.tab == .arena ? Theme.text : Theme.muted)
                    .frame(width: 30, height: 30)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Test Arena")
            .accessibilityLabel("Test Arena")
        }
        .padding(.trailing, 4)
        .animation(Theme.spring, value: app.recorder.phase == .recording)
        .animation(Theme.spring, value: app.runner.isRunning)
    }

    func status(color: Color, _ text: String) -> some View {
        HStack(spacing: 8) {
            LiveDot(color: color, size: 6)
            Text(text).font(Theme.font(12.5)).foregroundStyle(Theme.text2).lineLimit(1)
        }
        .transition(.opacity.combined(with: .scale(scale: 0.9)))
    }
}
