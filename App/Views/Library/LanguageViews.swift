import ATCore
import ATLanguage
import SwiftUI

/// "Auto Label" in a recording's instructions: opens the Auto Label sheet, or shows the progress
/// while this recording is being labelled.
struct AutoLabelButton: View {
    let app: AppModel
    let recording: LibraryRecording

    var body: some View {
        let labels = app.labels
        if labels.isLabelling(recording.url) {
            Button { app.autoLabelTarget = AutoLabelTarget(recordings: labels.recordings) } label: {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Labelling \(Int(labels.progress.fraction * 100))%").font(Theme.font(12, .medium)).monospacedDigit()
                }
            }
            .buttonStyle(.atSmall)
            .help(labels.progress.status)
        } else {
            Button { app.autoLabelTarget = AutoLabelTarget(recordings: [recording.url]) } label: { Label("Auto Label", systemImage: "sparkles") }
                .buttonStyle(.atSmall)
                .help("Let the local vision-language model watch this recording and write its instructions")
        }
    }
}

struct LanguageDownloadsRows: View {
    let app: AppModel
    var body: some View {
        ForEach([LanguageModels.embeddingGemma] + LanguageModels.labelers, id: \.id) { m in
            let st = app.hub.state(m)
            let isLabeler = m.id != LanguageModels.embeddingGemma.id
            Row(title: isLabeler ? "Labelling model (\(m.title))" : m.title,
                detail: "\(isLabeler ? m.summary + " " : "")\(m.repo) · \(m.license)\(st.error.map { " · \($0)" } ?? "")") {
                if st.done {
                    HStack(spacing: 10) {
                        Text("Ready").font(Theme.font(13)).foregroundStyle(Theme.live)
                        Button("Delete") { app.hub.delete(m) }.buttonStyle(.atSmallDestructive)
                            .disabled(isLabeler && app.labels.running)
                    }
                } else if st.running {
                    HStack(spacing: 10) {
                        ProgressView(value: st.fraction).tint(Theme.text).frame(width: 120)
                        Button("Cancel") { app.hub.cancel(m) }.buttonStyle(.atSmall)
                    }
                } else if app.hub.lmStudioCopy(m) != nil {
                    Button("Use LM Studio's copy") { app.hub.importFromLMStudio(m) }.buttonStyle(.atSmallPrimary)
                        .help("LM Studio already has this model. Using its files is instant and takes no extra space.")
                } else {
                    Button("Download ~\(m.approximateBytes.bytesText)") { app.hub.start(m) }.buttonStyle(.atSmallPrimary)
                }
            }
        }
    }
}
