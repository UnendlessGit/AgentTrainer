import ATCore
import ATRuntime
import SwiftUI

/// The Library's replay controls for one recording: arm it, see where it will play, follow it.
struct ReplayPanel: View {
    let app: AppModel
    let recording: LibraryRecording

    private var replay: ReplayController { app.replay }
    private var here: Bool { replay.isArmed(recording.url) }
    private var key: String { app.settings.stored.replayHotkey.display }

    var body: some View {
        Panel {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 8) {
                    Text("Replay into a game").font(Theme.font(14, .semibold))
                    Spacer()
                    if here { Pill(text: pillText, color: pillColor).transition(.opacity.combined(with: .scale(scale: 0.9))) }
                }
                if here, replay.isActive { active } else if here { armed } else { idle }
                targetLine
                if let m = replay.message, !replay.isActive, here || replay.messageIsError {
                    Text(m).font(Theme.font(12)).foregroundStyle(replay.messageIsError ? Theme.warning : Theme.muted)
                        .fixedSize(horizontal: false, vertical: true)
                        .transition(.opacity)
                }
            }
        }
        .animation(Theme.spring, value: replay.phase)
        .animation(Theme.spring, value: here)
    }

    private var pillText: String {
        switch replay.phase {
        case .idle: "ARMED"
        case .preparing: "STARTING"
        case .playing: "PLAYING"
        case .waitingForFocus: "PAUSED"
        }
    }

    private var pillColor: Color { replay.phase == .waitingForFocus ? Theme.warning : Theme.live }

    private var idle: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Plays this recording's keys and mouse back into the Run tab's window, to check it reproduces what happened. Start it from the same spot the recording starts.")
                .font(Theme.font(12.5)).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
            if replay.armed != nil {
                HStack(spacing: 6) {
                    Image(systemName: "play.circle.fill").font(.system(size: 11)).foregroundStyle(Theme.faint)
                    Text("Armed now: \(replay.armedName)").font(Theme.font(12)).foregroundStyle(Theme.faint).lineLimit(1).truncationMode(.middle)
                }
            }
            Button { withAnimation(Theme.spring) { replay.arm(recording) } } label: { Label("Replay", systemImage: "play.fill") }
                .buttonStyle(.atSmallPrimary)
                .disabled(replay.isActive)
                .help("Arm this recording, then press \(key) in the game to play it")
        }
    }

    private var armed: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Go to the game, stand where the recording starts and press \(key). Press it again to stop.")
                .font(Theme.font(12.5)).foregroundStyle(Theme.text2).fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 8) {
                Button { replay.start(bringToFront: true) } label: { Label("Start Now", systemImage: "play.fill") }
                    .buttonStyle(.atSmallPrimary)
                    .disabled(app.runner.selectedTargetID == nil)
                    .help("Brings the window to the front and starts")
                Button("Disarm") { withAnimation(Theme.spring) { replay.disarm() } }.buttonStyle(.atQuiet)
            }
        }
    }

    private var active: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                LiveDot(color: pillColor)
                Text(statusText).font(Theme.font(13)).foregroundStyle(Theme.text2).monospacedDigit().lineLimit(1)
                Spacer(minLength: 8)
                Button("Stop") { replay.stop() }.buttonStyle(.atSmallDestructive).help("Or press \(key)")
            }
            ProgressTrack(value: replay.duration > 0 ? replay.elapsed / replay.duration : 0, color: pillColor)
        }
    }

    private var statusText: String {
        switch replay.phase {
        case let .preparing(text): text
        case .playing: "\(replay.elapsed.durationText) of \(replay.duration.durationText)"
        case .waitingForFocus: "Paused until \(replay.targetName) is in front"
        case .idle: ""
        }
    }

    /// Where it plays and how the mouse moves.
    private var targetLine: some View {
        let target = app.runner.selectedTarget
        let mouse = ReplayPlan.isElkborn(recording.meta) ? "camera movement (Elkborn replay)" : app.runner.prefs.mouseMode.label.lowercased()
        return HStack(spacing: 8) {
            Image(systemName: "macwindow").font(.system(size: 12)).foregroundStyle(target == nil ? Theme.warning : Theme.muted)
            if let target {
                Text("Into \(target.name) · \(mouse)").font(Theme.font(12)).foregroundStyle(Theme.muted).lineLimit(2).truncationMode(.middle)
            } else {
                Text("Choose the window to play into in the Run tab.").font(Theme.font(12)).foregroundStyle(Theme.warning).lineLimit(2)
                Spacer(minLength: 4)
                Button("Run Tab") { app.tab = .run }.buttonStyle(.atQuiet)
            }
        }
    }
}

/// A thin progress track.
struct ProgressTrack: View {
    let value: Double
    var color: Color = Theme.live
    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(Theme.track)
                Capsule().fill(color).frame(width: max(4, geo.size.width * min(1, max(0, value))))
            }
        }
        .frame(height: 4)
        .animation(.linear(duration: 0.1), value: value)
    }
}
