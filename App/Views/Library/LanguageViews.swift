import ATCore
import ATLanguage
import SwiftUI

/// "Auto-label" in the recording inspector: the local VLM writes instructions for the clip.
struct AutoLabelButton: View {
    let app: AppModel
    let recording: LibraryRecording
    @Binding var annotations: RecordingAnnotations

    var body: some View {
        let labels = app.labels
        if labels.running {
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text(labels.message).font(Theme.font(11)).foregroundStyle(Theme.muted).lineLimit(1).frame(maxWidth: 120)
                Button("Stop") { labels.stop() }.buttonStyle(.atSmall)
            }
        } else {
            Button {
                labels.onFinished = { [weak lib = app.library] in
                    lib?.refresh()
                    annotations = RecordingIO.readAnnotations(recording.url)
                }
                labels.label([recording.url])
            } label: { Label("Auto-label", systemImage: "sparkles") }
            .buttonStyle(.atSmall)
            .disabled(!labels.modelReady)
            .help(labels.modelReady ? "Write instructions with the local vision-language model (replaces earlier automatic ones; yours are kept)" : "Download the labelling model in Settings → Downloads first")
        }
    }
}

struct LanguageDownloadsRows: View {
    let app: AppModel
    var body: some View {
        ForEach([LanguageModels.embeddingGemma, LanguageModels.labeler], id: \.id) { m in
            let st = app.hub.state(m)
            Row(title: m.title, detail: "\(m.repo) · \(m.license)\(st.error.map { " · \($0)" } ?? "")") {
                if st.done {
                    HStack(spacing: 10) {
                        Text("Downloaded").font(Theme.font(13)).foregroundStyle(Theme.live)
                        Button("Delete") { app.hub.delete(m) }.buttonStyle(.atSmallDestructive)
                    }
                } else if st.running {
                    HStack(spacing: 10) {
                        ProgressView(value: st.fraction).tint(Theme.text).frame(width: 120)
                        Button("Cancel") { app.hub.cancel(m) }.buttonStyle(.atSmall)
                    }
                } else {
                    Button("Download ~\(m.approximateBytes.bytesText)") { app.hub.start(m) }.buttonStyle(.atSmallPrimary)
                }
            }
        }
    }
}
