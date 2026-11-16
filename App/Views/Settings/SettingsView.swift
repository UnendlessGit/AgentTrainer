import ATCapture
import ATCore
import SwiftUI

struct SettingsView: View {
    @Bindable var app: AppModel
    @State private var permissions: [PrivacyPermission: Bool] = [:]

    var body: some View {
        @Bindable var settings = app.settings
        Form {
            Section("Storage") {
                PathRow(title: "Recordings library", url: app.settings.paths.libraryRoot) { u in settings.stored.libraryRoot = u.path; app.applyStorageRoots() }
                PathRow(title: "Models and checkpoints", url: app.settings.paths.modelsRoot) { u in settings.stored.modelsRoot = u.path; app.applyStorageRoots() }
                PathRow(title: "Training cache", url: app.settings.paths.cacheRoot) { u in settings.stored.cacheRoot = u.path; app.applyStorageRoots() }
                Button("Reset to default locations") {
                    settings.stored.libraryRoot = nil; settings.stored.modelsRoot = nil; settings.stored.cacheRoot = nil
                    app.applyStorageRoots()
                }
                Text("Changing a location doesn't move existing files; move them in Finder if you want to keep them.").font(.caption).foregroundStyle(.secondary)
            }

            Section("Keyboard shortcuts (work everywhere, even in full-screen games)") {
                HotkeyRow(title: "Start / stop recording (saves it)", hotkey: $settings.stored.recordHotkey) { app.registerHotkeys() }
                HotkeyRow(title: "Cancel recording (discards it)", hotkey: $settings.stored.cancelRecordingHotkey) { app.registerHotkeys() }
                HotkeyRow(title: "Start / stop the AI", hotkey: $settings.stored.runHotkey) { app.registerHotkeys() }
                HotkeyRow(title: "Pause / resume the AI", hotkey: $settings.stored.pauseHotkey) { app.registerHotkeys() }
                HotkeyRow(title: "Emergency stop", hotkey: $settings.stored.emergencyHotkey) { app.registerHotkeys() }
            }

            Section("Recording") {
                Stepper("Countdown: \(settings.stored.countdownSeconds) s", value: $settings.stored.countdownSeconds, in: 0 ... 10)
            }

            Section("Performance") {
                LabeledSlider(title: "GPU memory limit", value: $settings.stored.mlxMemoryLimitGB, range: 8 ... 32, step: 1, format: { "\(Int($0)) GB" })
                Toggle("8-bit weights when running models (slightly faster, tiny accuracy cost)", isOn: $settings.stored.inferenceQuantized)
                Toggle("Pause the AI when I touch the keyboard or mouse", isOn: $settings.stored.pauseOnHumanInput)
                Toggle("Keep the GPU clocked up while a model runs", isOn: $settings.stored.keepGPUWarm)
                    .help("Runs a tiny GPU task between model steps so macOS doesn't lower the GPU clock. Without it, steps can take 2–3× longer once a run has been going for a few seconds. Uses a little more power.")
                Text(verbatim: "\(SystemInfo.chipName) · \(ProcessInfo.processInfo.physicalMemory / 1_073_741_824) GB unified memory · \(ProcessInfo.processInfo.activeProcessorCount) CPU cores")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Downloads") {
                ForEach(Array(app.downloads.items.values).sorted { $0.id < $1.id }) { item in
                    HStack {
                        Text(item.title)
                        Spacer()
                        switch item.state {
                        case .done:
                            Text(item.expectedBytes.bytesText).foregroundStyle(.secondary)
                            Button("Delete", role: .destructive) { app.downloads.delete(item.id) }
                        case .downloading:
                            ProgressView(value: item.fraction).frame(width: 120)
                            Button("Pause") { app.downloads.pause(item.id) }
                        default:
                            Button("Download (\(item.expectedBytes.bytesText))") { app.downloads.start(item.id) }
                        }
                    }
                }
                LanguageDownloadsRows(app: app)
            }

            Section("Permissions") {
                ForEach(PrivacyPermission.allCases) { p in
                    HStack {
                        Label(p.title, systemImage: p.symbol)
                        Spacer()
                        if permissions[p] == true {
                            Label("Granted", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                        } else {
                            Button("Grant…") { p.request(); p.openSettings() }
                        }
                    }
                }
                Text("Permissions belong to AgentTrainer's signing identity, so they stay granted when the app is rebuilt or updated.").font(.caption).foregroundStyle(.secondary)
            }

            Section("About") {
                LabeledContent("Version", value: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "–")
                LabeledContent("Bundle ID", value: Bundle.main.bundleIdentifier ?? "–")
                LabeledContent("Model architecture", value: "Pixel2Play (elefant-ai/open-p2p, MIT)")
            }
        }
        .formStyle(.grouped)
        .onAppear { refreshPermissions() }
        .onReceive(Timer.publish(every: 2, on: .main, in: .common).autoconnect()) { _ in refreshPermissions() }
    }

    func refreshPermissions() { for p in PrivacyPermission.allCases { permissions[p] = p.isGranted } }
}

struct PathRow: View {
    let title: String
    let url: URL
    let choose: (URL) -> Void
    var body: some View {
        LabeledContent(title) {
            HStack {
                Text(url.path.replacingOccurrences(of: NSHomeDirectory(), with: "~")).lineLimit(1).truncationMode(.middle).foregroundStyle(.secondary)
                Button("Choose…") {
                    let panel = NSOpenPanel()
                    panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.canCreateDirectories = true
                    panel.directoryURL = url
                    if panel.runModal() == .OK, let u = panel.url { choose(u) }
                }
                Button { NSWorkspace.shared.activateFileViewerSelecting([url]) } label: { Image(systemName: "arrow.right.circle") }.buttonStyle(.borderless)
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
        LabeledContent(title) {
            Button(recording ? "Press keys…" : hotkey.display) {
                recording ? stop() : start()
            }
            .frame(minWidth: 110)
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
