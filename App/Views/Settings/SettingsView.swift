import ATCapture
import ATCore
import SwiftUI

enum SettingsSection: String, CaseIterable {
    case shortcuts = "Shortcuts", recording = "Recording", performance = "Performance", storage = "Storage", downloads = "Downloads", permissions = "Permissions", about = "About"
    var symbol: String {
        switch self {
        case .shortcuts: "command"
        case .recording: "record.circle"
        case .performance: "gauge.with.dots.needle.67percent"
        case .storage: "externaldrive"
        case .downloads: "arrow.down.circle"
        case .permissions: "lock.shield"
        case .about: "info.circle"
        }
    }
}

struct SettingsView: View {
    @Bindable var app: AppModel
    @State private var permissions: [PrivacyPermission: Bool] = [:]
    @State private var section = SettingsSection.shortcuts
    @Namespace private var ns

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 2) {
                ForEach(SettingsSection.allCases, id: \.self) { s in
                    let on = section == s
                    Button { withAnimation(Theme.spring) { section = s } } label: {
                        HStack(spacing: 10) {
                            Image(systemName: s.symbol).font(.system(size: 13)).frame(width: 18)
                            Text(s.rawValue).font(Theme.font(14, on ? .semibold : .medium))
                            Spacer()
                        }
                        .foregroundStyle(on ? Theme.text : Theme.muted)
                        .padding(.horizontal, 12).frame(height: 38)
                        .background {
                            if on { RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Theme.raised).matchedGeometryEffect(id: "sel", in: ns) }
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
                Spacer()
            }
            .padding(.horizontal, 14).padding(.top, 28)
            .frame(width: 240)
            .riseIn()
            Rectangle().fill(Theme.hairline).frame(width: 1)
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    Text(section.rawValue).displayStyle(42)
                    content.frame(maxWidth: 720, alignment: .leading)
                }
                .id(section)
                .transition(.opacity.combined(with: .offset(y: 8)))
                .padding(.horizontal, 52).padding(.vertical, 32)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .scrollIndicators(.never)
            .animation(Theme.spring, value: section)
        }
        .onAppear { refreshPermissions() }
        .onReceive(Timer.publish(every: 2, on: .main, in: .common).autoconnect()) { _ in refreshPermissions() }
    }

    @ViewBuilder
    private var content: some View {
        @Bindable var settings = app.settings
        switch section {
        case .shortcuts:
            Rows(footer: "Shortcuts work everywhere, even in full-screen games. Click one, then press the new keys (with at least one of ⌘ ⌥ ⌃).") {
                HotkeyRow(title: "Start / stop recording (saves it)", hotkey: $settings.stored.recordHotkey) { app.registerHotkeys() }
                HotkeyRow(title: "Discard recording", hotkey: $settings.stored.cancelRecordingHotkey) { app.registerHotkeys() }
                HotkeyRow(title: "Start / stop the AI", hotkey: $settings.stored.runHotkey) { app.registerHotkeys() }
                HotkeyRow(title: "Pause / resume the AI", hotkey: $settings.stored.pauseHotkey) { app.registerHotkeys() }
                HotkeyRow(title: "Emergency stop", hotkey: $settings.stored.emergencyHotkey) { app.registerHotkeys() }
            }
        case .recording:
            Rows {
                Row(title: "Countdown before recording") { NumberStepper(value: $settings.stored.countdownSeconds, range: 0 ... 10) }
            }
        case .performance:
            Rows(footer: "\(SystemInfo.chipName) · \(ProcessInfo.processInfo.physicalMemory / 1_073_741_824) GB unified memory · \(ProcessInfo.processInfo.activeProcessorCount) CPU cores") {
                LabeledSlider(title: "GPU memory limit", value: $settings.stored.mlxMemoryLimitGB, range: 8 ... 32, step: 1, format: { "\(Int($0)) GB" })
                Toggle(isOn: $settings.stored.keepGPUWarm) { Row(title: "Keep the GPU clocked up while a model runs", detail: "Without it steps can take 2–3× longer after a few seconds. Uses a little more power.") }
                Toggle(isOn: $settings.stored.inferenceQuantized) { Row(title: "8-bit weights when running", detail: "Slightly faster, tiny accuracy cost.") }
                Toggle("Pause the AI when I touch the keyboard or mouse", isOn: $settings.stored.pauseOnHumanInput)
            }
            .toggleStyle(.oled)
        case .storage:
            Rows(footer: "Changing a location doesn't move existing files; move them in Finder to keep them.") {
                PathRow(title: "Recordings library", url: app.settings.paths.libraryRoot) { u in settings.stored.libraryRoot = u.path; app.applyStorageRoots() }
                PathRow(title: "Models and checkpoints", url: app.settings.paths.modelsRoot) { u in settings.stored.modelsRoot = u.path; app.applyStorageRoots() }
                PathRow(title: "Training cache", url: app.settings.paths.cacheRoot) { u in settings.stored.cacheRoot = u.path; app.applyStorageRoots() }
                Row(title: "Default locations") {
                    Button("Reset") {
                        settings.stored.libraryRoot = nil; settings.stored.modelsRoot = nil; settings.stored.cacheRoot = nil
                        app.applyStorageRoots()
                    }
                    .buttonStyle(.atSmall)
                }
            }
        case .downloads:
            Rows(title: "Pixel2Play weights") {
                ForEach(Array(app.downloads.items.values).sorted { $0.id < $1.id }) { item in
                    Row(title: item.title, detail: "\(item.expectedBytes.bytesText) · MIT") {
                        switch item.state {
                        case .done:
                            HStack(spacing: 10) {
                                Text("Downloaded").font(Theme.font(13)).foregroundStyle(Theme.live)
                                Button("Delete") { app.downloads.delete(item.id) }.buttonStyle(.atSmallDestructive)
                            }
                        case .downloading:
                            HStack(spacing: 10) {
                                ProgressView(value: item.fraction).tint(Theme.text).frame(width: 120)
                                Button("Pause") { app.downloads.pause(item.id) }.buttonStyle(.atSmall)
                            }
                        default:
                            Button("Download") { app.downloads.start(item.id) }.buttonStyle(.atSmallPrimary)
                        }
                    }
                }
            }
            Rows(title: "Language models") { LanguageDownloadsRows(app: app) }
        case .permissions:
            HStack(spacing: 14) {
                ForEach(PrivacyPermission.allCases) { p in
                    let ok = permissions[p] == true
                    Panel {
                        VStack(alignment: .leading, spacing: 12) {
                            Image(systemName: p.symbol).font(.system(size: 20)).foregroundStyle(ok ? Theme.live : Theme.warning)
                            Text(p.title).font(Theme.font(14, .semibold))
                            if ok {
                                Text("Granted").font(Theme.font(13)).foregroundStyle(Theme.live)
                            } else {
                                Button("Grant…") { p.request(); p.openSettings() }.buttonStyle(.atSmallPrimary)
                            }
                        }
                    }
                }
            }
            Text("Permissions belong to AgentTrainer's signing identity, so they stay granted when the app is rebuilt or updated.").font(Theme.font(12)).foregroundStyle(Theme.muted)
        case .about:
            Rows {
                Row(title: "Version") { value(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "–") }
                Row(title: "Bundle ID") { value(Bundle.main.bundleIdentifier ?? "–") }
                Row(title: "Model architecture") { value("Pixel2Play · elefant-ai/open-p2p · MIT") }
            }
        }
    }

    func value(_ s: String) -> some View { Text(s).font(Theme.font(13)).foregroundStyle(Theme.text2) }

    func refreshPermissions() { for p in PrivacyPermission.allCases { permissions[p] = p.isGranted } }
}

struct PathRow: View {
    let title: String
    let url: URL
    let choose: (URL) -> Void
    var body: some View {
        Row(title: title, detail: url.path.replacingOccurrences(of: NSHomeDirectory(), with: "~")) {
            HStack(spacing: 8) {
                Button("Choose…") {
                    let panel = NSOpenPanel()
                    panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.canCreateDirectories = true
                    panel.directoryURL = url
                    if panel.runModal() == .OK, let u = panel.url { choose(u) }
                }
                .buttonStyle(.atSmall)
                Button { NSWorkspace.shared.activateFileViewerSelecting([url]) } label: { Image(systemName: "folder") }.buttonStyle(.atSmall).help("Show in Finder")
            }
        }
    }
}

/// Click, then press the new shortcut (needs at least one modifier).
struct HotkeyRow: View {
    let title: String
    @Binding var hotkey: Hotkey
    let changed: () -> Void
    @State private var recording = false
    @State private var monitor: Any?

    var body: some View {
        Row(title: title) {
            Button { recording ? stop() : start() } label: {
                Text(recording ? "Press keys…" : hotkey.display).font(Theme.font(13, .semibold)).tracking(recording ? 0 : 1).frame(minWidth: 110)
            }
            .buttonStyle(ATButtonStyle(kind: recording ? .primary : .secondary, height: 32))
            .animation(Theme.snappy, value: recording)
        }
        .onDisappear { stop() }
    }

    func start() {
        recording = true
        monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { e in
            let f = e.modifierFlags
            guard f.contains(.command) || f.contains(.control) || f.contains(.option) else { return nil }
            hotkey = Hotkey(keyCode: e.keyCode, command: f.contains(.command), shift: f.contains(.shift), option: f.contains(.option), control: f.contains(.control))
            stop()
            changed()
            return nil
        }
    }

    func stop() {
        recording = false
        if let m = monitor { NSEvent.removeMonitor(m) }
        monitor = nil
    }
}

enum SystemInfo {
    static var chipName: String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var buf = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("machdep.cpu.brand_string", &buf, &size, nil, 0)
        return String(cString: buf)
    }
}
