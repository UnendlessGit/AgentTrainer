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
                Text(labels.message).font(.caption).lineLimit(1).frame(maxWidth: 220)
                Button("Stop") { labels.stop() }
            }
        } else {
            Button {
                labels.onFinished = { [weak lib = app.library] in
                    lib?.refresh()
                    annotations = RecordingIO.readAnnotations(recording.url)
                }
                labels.label([recording.url])
            } label: { Label("Auto-label", systemImage: "sparkles") }
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
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text(m.title)
                    Spacer()
                    if st.done {
                        Button("Delete", role: .destructive) { app.hub.delete(m) }
                    } else if st.running {
                        ProgressView(value: st.fraction).frame(width: 120)
                        Button("Cancel") { app.hub.cancel(m) }
                    } else {
                        Button("Download (~\(m.approximateBytes.bytesText))") { app.hub.start(m) }
                    }
                }
                Text("\(m.repo) · \(m.license)\(st.error.map { " · \($0)" } ?? "")").font(.caption).foregroundStyle(st.error == nil ? Color.secondary : Color.orange)
            }
        }
    }
}
