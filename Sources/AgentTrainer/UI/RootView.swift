import AppKit
import SwiftUI

struct RootView: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var appearance = UIAppearanceStore.shared
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var effectiveMotion: Bool {
        appearance.motionEnabled && model.isAppActive && !reduceMotion
    }

    var body: some View {
        VStack(spacing: 0) {
            topBar
            Rectangle().fill(ATColor.border).frame(height: max(0.5, appearance.configuration.effectiveBorderWidth))
            HStack(spacing: 0) {
                sidebar
                    .frame(width: CGFloat(appearance.configuration.sidebarWidth))
                Rectangle().fill(ATColor.border).frame(width: max(0.5, appearance.configuration.effectiveBorderWidth))
                detail
            }
        }
        .background(ATColor.canvas)
        .preferredColorScheme(appearance.colorScheme)
        .environment(\.uiMotionEnabled, effectiveMotion)
        .transaction { transaction in
            if !effectiveMotion { transaction.disablesAnimations = true }
        }
        .background(WindowChromeConfigurator(configuration: appearance.configuration))
        .ignoresSafeArea(.container, edges: .top)
        .task {
            model.hudModel.installPanel()
            await model.checkForUpdatesAtLaunch()
        }
        .onAppear { model.isAppActive = NSApplication.shared.isActive }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in model.isAppActive = true }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didResignActiveNotification)) { _ in model.isAppActive = false }
        .alert(
            "AgentTrainer",
            isPresented: Binding(
                get: { model.errorMessage != nil },
                set: { if !$0 { model.errorMessage = nil } }
            )
        ) {
            Button("OK", role: .cancel) { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
    }

    private var detail: some View {
        ZStack {
            selectedPage
                .id(model.selection)
                .transition(.opacity.combined(with: .scale(scale: 0.995)))
        }
        .animation(effectiveMotion ? UIMotion.standard : nil, value: model.selection)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(canvasBackground)
        .foregroundStyle(ATColor.text)
        .font(.system(size: 13 * appearance.configuration.fontScale))
    }

    private var topBar: some View {
        HStack(spacing: 0) {
            HStack(spacing: 8) {
                Image(nsImage: NSApplication.shared.applicationIconImage)
                    .resizable()
                    .frame(width: 21, height: 21)
                    .clipShape(RoundedRectangle(cornerRadius: ATCorner.scaled(5), style: .continuous))
                Text(appearance.configuration.appName)
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(ATColor.text)
                    .lineLimit(1)
            }
            .padding(.leading, 78)
            .padding(.trailing, 12)
            .frame(width: CGFloat(appearance.configuration.sidebarWidth), alignment: .leading)
            .frame(maxHeight: .infinity)
            .background(ATColor.sidebar)

            Rectangle().fill(ATColor.border).frame(width: max(0.5, appearance.configuration.effectiveBorderWidth))

            HStack(spacing: 10) {
                Label(appearance.configuration.label(for: model.selection), systemImage: model.selection.symbol)
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(sectionColor(model.selection))

                Spacer(minLength: 20)
                StatusPill(text: activityTitle, color: activityColor)

                Button {
                    Task { model.isRecording ? await model.stopRecording() : await model.startRecording() }
                } label: {
                    Image(systemName: model.isRecording ? "stop.fill" : "record.circle")
                }
                .primaryButton(color: ATColor.coral)
                .help(model.isRecording ? "Stop and save recording" : "Start recording")
                .disabled(model.isRunning)

                Button {
                    Task { model.isRunning ? await model.stopAgent() : await model.startAgent() }
                } label: {
                    Image(systemName: model.isRunning ? "stop.fill" : "play.fill")
                }
                .primaryButton(color: ATColor.violet)
                .help(model.isRunning ? "Stop AI and release inputs" : "Run selected AI")
                .disabled(!model.isRunning && model.selectedProfile?.activeVersionID == nil)

                Button(role: .destructive) { model.panic() } label: {
                    Image(systemName: "hand.raised.fill")
                }
                .primaryButton(color: ATColor.coral)
                .help("Panic — stop everything and release all inputs")
            }
            .padding(.leading, 14)
            .padding(.trailing, 12)
            .frame(maxHeight: .infinity)
            .background(ATColor.canvas)
        }
        .frame(height: 48)
        .background(WindowDragHandle())
        .accessibilityElement(children: .contain)
    }

    private var sidebar: some View {
        VStack(spacing: 0) {
            ScrollView {
                LazyVStack(spacing: 5) {
                    ForEach(AppSection.allCases) { section in
                        SidebarSectionButton(
                            section: section,
                            selected: model.selection == section,
                            action: { model.selection = section }
                        )
                    }
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 14)
            }

            Spacer(minLength: 8)

            VStack(alignment: .leading, spacing: 8) {
                StatusPill(text: activityTitle, color: activityColor)
                Text(model.activityStatus)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .foregroundStyle(ATColor.text)
        .background(sidebarBackground)
    }

    @ViewBuilder private var selectedPage: some View {
        switch model.selection {
        case .home: HomeView(model: model)
        case .record: RecordView(model: model)
        case .library: LibraryView(model: model)
        case .models: ModelsView(model: model)
        case .training: TrainingView(model: model)
        case .run: RunView(model: model)
        case .diagnostics: DiagnosticsView(model: model)
        case .settings: SettingsView(model: model)
        }
    }


    private var activityTitle: String {
        if model.isRecording { return "Recording" }
        if model.isRunning && model.isTraining { return "Run + train" }
        if model.isRunning { return "AI running" }
        if model.isTraining { return "Training" }
        return "Local only"
    }

    private var activityColor: Color {
        if model.isRecording { return ATColor.coral }
        if model.isRunning { return ATColor.violet }
        if model.isTraining { return ATColor.cyan }
        return ATColor.green
    }

    @ViewBuilder private var sidebarBackground: some View {
        Rectangle().fill(ATColor.sidebar)
    }

    @ViewBuilder private var canvasBackground: some View {
        Rectangle().fill(ATColor.canvas)
    }
}

/// A solid SwiftUI row avoids the focus-dependent NSVisualEffectView used by
/// NavigationSplitView's sidebar on Sequoia.
private struct SidebarSectionButton: View {
    let section: AppSection
    let selected: Bool
    let action: () -> Void
    @ObservedObject private var appearance = UIAppearanceStore.shared
    @Environment(\.uiMotionEnabled) private var motionEnabled
    @State private var hovering = false

    var body: some View {
        let color = sectionColor(section)
        let shape = RoundedRectangle(cornerRadius: ATCorner.scaled(10), style: .continuous)
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: section.symbol)
                    .font(.system(size: 13, weight: .semibold))
                    .frame(width: 20)
                Text(appearance.configuration.label(for: section))
                    .font(.system(size: 13, weight: selected ? .semibold : .medium, design: .rounded))
                    .lineLimit(1)
                Spacer(minLength: 4)
                if selected {
                    Circle().fill(color).frame(width: 5, height: 5)
                }
            }
            .foregroundStyle(selected ? color : ATColor.text.opacity(hovering ? 0.92 : 0.7))
            .padding(.horizontal, 11)
            .frame(height: 36)
            .contentShape(shape)
            .background(shape.fill(selected ? color.opacity(appearance.configuration.effectiveControlOpacity) : hovering ? ATColor.raised.opacity(0.72) : Color.clear))
            .overlay(shape.stroke(selected ? color.opacity(0.5) : hovering ? ATColor.border : Color.clear, lineWidth: 0.8))
        }
        .buttonStyle(.plain)
        .scaleEffect(hovering && motionEnabled ? 1.006 : 1)
        .animation(motionEnabled ? UIMotion.quick : nil, value: hovering)
        .onHover { hovering = motionEnabled ? $0 : false }
        .accessibilityValue(selected ? "Selected" : "")
    }
}

private func sectionColor(_ section: AppSection) -> Color {
    switch section {
    case .home, .diagnostics: ATColor.cyan
    case .record: ATColor.coral
    case .library: ATColor.amber
    case .models, .run: ATColor.violet
    case .training: ATColor.green
    case .settings: ATColor.amber
    }
}

private struct WindowDragHandle: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView { DraggableView(frame: .zero) }
    func updateNSView(_ nsView: NSView, context: Context) {}

    private final class DraggableView: NSView {
        override var mouseDownCanMoveWindow: Bool { true }
        override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    }
}

/// Full-size, transparent AppKit title chrome leaves the traffic lights native
/// while the app draws one identical opaque top bar on Sequoia and Tahoe.
private struct WindowChromeConfigurator: NSViewRepresentable {
    let configuration: UIConfiguration

    func makeNSView(context: Context) -> NSView {
        let view = NSView(frame: .zero)
        DispatchQueue.main.async { configure(view.window) }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        DispatchQueue.main.async { configure(nsView.window) }
    }

    private func configure(_ window: NSWindow?) {
        guard let window else { return }
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.titlebarSeparatorStyle = .none
        window.styleMask.insert(.fullSizeContentView)
        window.isMovableByWindowBackground = false
        window.hasShadow = true
        window.standardWindowButton(.closeButton)?.isHidden = false
        window.standardWindowButton(.miniaturizeButton)?.isHidden = false
        window.standardWindowButton(.zoomButton)?.isHidden = false

        window.isOpaque = true
        window.backgroundColor = NSColor(
            calibratedRed: configuration.canvas.red,
            green: configuration.canvas.green,
            blue: configuration.canvas.blue,
            alpha: 1
        )
        window.contentView?.wantsLayer = true
        window.contentView?.layer?.backgroundColor = window.backgroundColor.cgColor
    }
}
