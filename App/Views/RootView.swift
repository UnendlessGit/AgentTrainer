import SwiftUI

struct RootView: View {
    @Bindable var app: AppModel

    var body: some View {
        NavigationSplitView {
            List(selection: Binding(get: { app.tab }, set: { if let t = $0 { app.tab = t } })) {
                Section {
                    ForEach([AppTab.recordings, .library, .models, .training, .run]) { t in
                        Label(t.title, systemImage: t.symbol).tag(t)
                    }
                }
                Section("Tools") { Label(AppTab.arena.title, systemImage: AppTab.arena.symbol).tag(AppTab.arena) }
                Section { Label(AppTab.settings.title, systemImage: AppTab.settings.symbol).tag(AppTab.settings) }
            }
            .navigationSplitViewColumnWidth(min: 180, ideal: 200, max: 260)
            .safeAreaInset(edge: .bottom) { StatusFooter(app: app).padding(10) }
        } detail: {
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
            .frame(minWidth: 720, minHeight: 560)
        }
    }
}

/// Small live indicator for background activity, visible from every tab.
struct StatusFooter: View {
    let app: AppModel
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if app.recorder.phase == .recording {
                Label("Recording \(app.recorder.stats.elapsed.durationText)", systemImage: "record.circle.fill").foregroundStyle(.red)
            }
            if let line = app.training.footerLine {
                Label(line, systemImage: "chart.line.uptrend.xyaxis").foregroundStyle(.tint)
            }
            if app.runner.isRunning {
                Label(app.runner.footerLine, systemImage: "play.circle.fill").foregroundStyle(.green)
            }
        }
        .font(.caption)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
