import AppKit
import SwiftUI

@main
struct AgentTrainerApp: App {
    @StateObject private var model = AppModel()
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup(id: "main") {
            RootView(model: model)
                .frame(minWidth: 1180, minHeight: 760)
                .onAppear { appDelegate.model = model }
        }
        .windowStyle(.hiddenTitleBar)
        .commands {
            CommandMenu("AgentTrainer") {
                Button("Panic Stop") { model.panic() }
                Divider()
                Button(model.isRecording ? "Stop Recording" : "Start Recording") { Task { model.isRecording ? await model.stopRecording() : await model.startRecording() } }
                Button(model.isRunning ? "Stop Agent" : "Run Agent") { Task { model.isRunning ? await model.stopAgent() : await model.startAgent() } }
            }
        }

        MenuBarExtra {
            AgentTrainerMenuBarView(model: model)
        } label: {
            Image(systemName: model.isRecording ? "record.circle.fill" : model.isRunning ? "play.circle.fill" : model.isTraining ? "brain.head.profile.fill" : "waveform.path.ecg.rectangle")
                .symbolRenderingMode(.palette)
                .foregroundStyle(model.isRecording ? Color.red : model.isRunning ? Color.purple : model.isTraining ? Color.cyan : Color.primary)
        }
        .menuBarExtraStyle(.window)
    }
}

private struct AgentTrainerMenuBarView: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var appearance = UIAppearanceStore.shared
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(nsImage: NSApplication.shared.applicationIconImage).resizable().frame(width: 25, height: 25).clipShape(RoundedRectangle(cornerRadius: ATCorner.scaled(6), style: .continuous))
                VStack(alignment: .leading, spacing: 1) { Text(appearance.configuration.appName).font(.headline); Text(model.isRecording ? "Recording" : appearance.configuration.appSubtitle).font(.caption).foregroundStyle(model.isRecording ? Color.red : Color.secondary) }
                Spacer()
                if model.isRecording { Circle().fill(.red).frame(width: 9, height: 9) }
            }
            Divider()
            status("Recording", model.isRecording ? "Active" : "Idle", model.isRecording ? .red : .secondary)
            status("Training", model.isTraining ? model.profiles.first(where: { $0.id == model.trainingProfileID })?.name ?? "Active" : "Idle", model.isTraining ? ATColor.cyan : .secondary)
            status("Agent", model.isRunning ? model.profiles.first(where: { $0.id == model.runningProfileID })?.name ?? "Running" : "Idle", model.isRunning ? ATColor.violet : .secondary)
            Divider()
            Button(model.isRecording ? "Stop & Save Recording" : "Start Recording") { Task { model.isRecording ? await model.stopRecording() : await model.startRecording() } }
                .disabled(model.isRunning)
            Button(model.isRunning ? "Stop Agent & Release Inputs" : "Start Selected Agent") { Task { model.isRunning ? await model.stopAgent() : await model.startAgent() } }
                .disabled(!model.isRunning && model.selectedProfile?.activeVersionID == nil)
            Button("Panic — Stop Everything", role: .destructive) { model.panic() }
            Divider()
            Button("Show AgentTrainer") { openWindow(id: "main"); NSApp.activate(ignoringOtherApps: true) }
            Button("Quit") { NSApp.terminate(nil) }
        }.padding(14).frame(width: 300).preferredColorScheme(appearance.colorScheme)
    }

    private func status(_ name: String, _ value: String, _ color: Color) -> some View {
        HStack { Text(name).foregroundStyle(.secondary); Spacer(); Text(value).foregroundStyle(color).lineLimit(1) }.font(.callout)
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var model: AppModel?

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let model, model.isTraining || model.isRecording || model.isRunning || model.isReplaying else { return .terminateNow }
        if model.isTraining { model.pauseTraining() }
        Task { @MainActor in
            if model.isRunning { await model.stopAgent() }
            if model.isReplaying { model.stopReenactment() }
            if model.isRecording { await model.stopRecording() }
            while model.isTraining { try? await Task.sleep(for: .milliseconds(100)) }
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}
