import ATCore
import ATLanguage
import SwiftUI

/// Auto Label: talk to the labelling model about the footage, choose how it labels, and watch the
/// instructions arrive. Labelling keeps going if the sheet is closed.
struct AutoLabelSheet: View {
    let app: AppModel
    let target: AutoLabelTarget
    @Environment(\.dismiss) private var dismiss

    private var labels: LabelController { app.labels }
    private var recordings: [LibraryRecording] { app.library.all.filter { target.recordings.contains($0.url) } }
    private var thisRun: Bool { !labels.recordings.isEmpty && Set(labels.recordings) == Set(target.recordings) }

    var body: some View {
        let recs = recordings
        VStack(alignment: .leading, spacing: 0) {
            header(recs).padding(.horizontal, 28).padding(.top, 26).padding(.bottom, 18)
            HStack(alignment: .top, spacing: 18) {
                Group {
                    if labels.running && thisRun || (!labels.running && thisRun && !labels.recent.isEmpty) {
                        LiveLabelsPanel(app: app)
                    } else {
                        LabelChatPanel(app: app, recordings: target.recordings)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                LabelOptionsPanel(app: app, recordings: recs)
                    .frame(width: 356)
                    .disabled(labels.running)
            }
            .padding(.horizontal, 28)
            .frame(maxHeight: .infinity)
            footer(recs).padding(.horizontal, 28).padding(.vertical, 20)
        }
        .frame(width: 960, height: 740)
        .background(Theme.ground)
        .preferredColorScheme(.dark)
        .animation(Theme.spring, value: labels.running)
    }

    private func header(_ recs: [LibraryRecording]) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 14) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Auto Label").displayStyle(30)
                Text(recs.count == 1 ? "\(recs[0].meta.name) · \(recs[0].meta.duration.durationText)"
                     : "\(recs.count) recordings · \(recs.reduce(0) { $0 + $1.meta.duration }.durationText) of footage")
                    .font(Theme.font(13)).foregroundStyle(Theme.muted).lineLimit(1).truncationMode(.middle)
            }
            Spacer()
        }
    }

    @ViewBuilder
    private func footer(_ recs: [LibraryRecording]) -> some View {
        HStack(spacing: 12) {
            if labels.running && thisRun {
                LabelProgressLine(labels: labels)
                Spacer(minLength: 12)
                Button("Close") { dismiss() }.buttonStyle(.atSecondary).help("Labelling keeps going; its progress shows in the title bar.")
                Button("Stop") { labels.stop() }.buttonStyle(.atDestructive)
            } else {
                if let e = labels.lastError, thisRun { InlineError(message: e) } else if labels.running {
                    Text("Another Auto Label run is going. Wait for it or stop it first.").font(Theme.font(12.5)).foregroundStyle(Theme.warning)
                } else if thisRun, labels.written > 0 {
                    Label("Wrote \(labels.written) instruction\(labels.written == 1 ? "" : "s"). \(labels.progress.status)", systemImage: "checkmark.circle.fill")
                        .font(Theme.font(13)).foregroundStyle(Theme.text2)
                }
                Spacer(minLength: 12)
                Button(thisRun && labels.written > 0 ? "Done" : "Cancel") { dismiss() }.buttonStyle(.atSecondary).keyboardShortcut(.cancelAction)
                Button {
                    labels.label(target.recordings)
                } label: {
                    Label(thisRun && labels.written > 0 ? "Label again" : recs.count == 1 ? "Label recording" : "Label \(recs.count) recordings", systemImage: "sparkles")
                }
                .buttonStyle(.atPrimary)
                .disabled(!labels.modelReady || labels.running || labels.chatBusy || recs.isEmpty || app.training.isActive)
                .help(app.training.isActive ? "A model is training. Labelling needs the same GPU memory." : "")
            }
        }
    }
}

// MARK: Progress

struct LabelProgressLine: View {
    let labels: LabelController
    var body: some View {
        let p = labels.progress
        HStack(spacing: 14) {
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(Theme.track)
                    Capsule().fill(Theme.ai).frame(width: max(6, g.size.width * min(1, max(0, p.fraction))))
                }
            }
            .frame(width: 180, height: 6)
            .animation(Theme.spring, value: p.fraction)
            VStack(alignment: .leading, spacing: 2) {
                Text(p.status).font(Theme.font(13, .medium)).foregroundStyle(Theme.text).lineLimit(1).truncationMode(.middle)
                Text(detail(p)).font(Theme.font(12)).monospacedDigit().foregroundStyle(Theme.muted).lineLimit(1)
            }
        }
    }

    func detail(_ p: LabelController.Progress) -> String {
        var parts = ["\(Int(p.fraction * 100))%"]
        if p.count > 1 { parts.append("recording \(min(p.recording + 1, p.count)) of \(p.count)") }
        if let left = p.secondsLeft { parts.append("about \(Self.left(left)) left") }
        parts.append("\(labels.written) instruction\(labels.written == 1 ? "" : "s")")
        return parts.joined(separator: " · ")
    }

    static func left(_ s: Double) -> String {
        s < 90 ? "1 min" : s < 3600 ? "\(Int((s / 60).rounded())) min" : String(format: "%.1f h", s / 3600)
    }
}

// MARK: Live feed

/// The instructions as they're written, newest at the bottom.
struct LiveLabelsPanel: View {
    let app: AppModel
    var body: some View {
        let labels = app.labels
        Panel(padding: 0) {
            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    Text(labels.running ? "Writing instructions" : "Instructions written").font(Theme.font(14, .semibold))
                    if labels.running { LiveDot(color: Theme.ai, size: 6) }
                    Spacer()
                }
                .padding(.horizontal, 18).padding(.top, 16).padding(.bottom, 10)
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 10) {
                            if labels.recent.isEmpty {
                                HStack(spacing: 10) {
                                    ProgressView().controlSize(.small)
                                    Text(labels.progress.status.isEmpty ? "Starting" : labels.progress.status).font(Theme.font(13)).foregroundStyle(Theme.muted)
                                }
                                .padding(.vertical, 8)
                            }
                            ForEach(labels.recent, id: \.segment.id) { item in
                                LiveLabelRow(name: app.library.all.first { $0.url == item.recording }?.meta.name, showName: Set(labels.recordings).count > 1, segment: item.segment)
                                    .id(item.segment.id)
                                    .transition(.opacity.combined(with: .move(edge: .bottom)))
                            }
                        }
                        .padding(.horizontal, 18).padding(.bottom, 16)
                    }
                    .scrollIndicators(.never)
                    .onChange(of: labels.recent.last?.segment) { _, s in
                        if let id = s?.id { withAnimation(Theme.spring) { proxy.scrollTo(id, anchor: .bottom) } }
                    }
                }
            }
        }
        .animation(Theme.spring, value: labels.recent.map(\.segment))
    }
}

private struct LiveLabelRow: View {
    let name: String?
    let showName: Bool
    let segment: TextSegment
    @State private var open = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text("\(segment.start.durationText)–\(segment.end.durationText)").font(Theme.font(11.5)).monospacedDigit().foregroundStyle(Theme.faint)
                if showName, let name { Text(name).font(Theme.font(11.5)).foregroundStyle(Theme.faint).lineLimit(1).truncationMode(.middle) }
            }
            Text(segment.text).font(Theme.font(14, .medium)).foregroundStyle(Theme.text).fixedSize(horizontal: false, vertical: true)
            if let o = segment.observation, !o.isEmpty {
                Label { Text(o) } icon: { Image(systemName: "eye") }
                    .font(Theme.font(12)).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
            }
            if let v = segment.variants, !v.isEmpty {
                Button { withAnimation(Theme.snappy) { open.toggle() } } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.right").font(.system(size: 9, weight: .bold)).rotationEffect(.degrees(open ? 90 : 0))
                        Text("\(v.count) variation\(v.count == 1 ? "" : "s")")
                    }
                    .font(Theme.font(12, .medium)).foregroundStyle(Theme.ai)
                }
                .buttonStyle(.plain)
                if open {
                    VStack(alignment: .leading, spacing: 3) {
                        ForEach(v, id: \.self) { Text("· \($0)").font(Theme.font(12.5)).foregroundStyle(Theme.text2) }
                    }
                    .transition(.opacity)
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.raised.opacity(0.55), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

// MARK: Chat

/// A conversation with the labelling model about the footage. What you agree on guides every label.
struct LabelChatPanel: View {
    let app: AppModel
    let recordings: [URL]
    @State private var draft = ""
    @FocusState private var focused: Bool

    private var labels: LabelController { app.labels }
    private var messages: [LabelChatMessage] { labels.conversationApplies(to: recordings) ? labels.chat : [] }

    var body: some View {
        Panel(padding: 0) {
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 8) {
                    Text("Tell it about the footage").font(Theme.font(14, .semibold))
                    Spacer()
                    if !messages.isEmpty {
                        Button("New conversation") { labels.newConversation() }.buttonStyle(.atQuiet).disabled(labels.chatBusy)
                    }
                }
                .padding(.horizontal, 18).padding(.top, 14).padding(.bottom, 6)
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 12) {
                            if messages.isEmpty && !labels.chatBusy { intro }
                            ForEach(messages) { m in ChatBubble(role: m.role, text: m.text).id(m.id) }
                            if labels.chatBusy {
                                ChatBubble(role: .assistant, text: labels.reply ?? "", streaming: true).id("reply")
                            }
                            if let e = labels.chatError { InlineError(message: e) }
                            Color.clear.frame(height: 1).id("end")
                        }
                        .padding(.horizontal, 18).padding(.vertical, 8)
                    }
                    .scrollIndicators(.never)
                    .onChange(of: labels.reply) { _, _ in proxy.scrollTo("end", anchor: .bottom) }
                    .onChange(of: labels.chat.count) { _, _ in withAnimation(Theme.spring) { proxy.scrollTo("end", anchor: .bottom) } }
                    .onAppear { proxy.scrollTo("end", anchor: .bottom) }
                }
                Rectangle().fill(Theme.hairline).frame(height: 1)
                composer.padding(12)
            }
        }
        .animation(Theme.spring, value: messages)
        .onAppear { focused = labels.modelReady }
    }

    private var intro: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("It watches an overview of \(recordings.count == 1 ? "the recording" : "these recordings") and answers you. What you tell it here (the game, the goals, names of places, how instructions should sound) guides every label it writes.")
                .font(Theme.font(13)).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
            if !labels.chat.isEmpty {
                Text("Your earlier conversation was about other recordings; a message here starts a new one.")
                    .font(Theme.font(12)).foregroundStyle(Theme.faint).fixedSize(horizontal: false, vertical: true)
            }
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Self.suggestions, id: \.self) { s in
                    Button { draft = s; focused = true } label: {
                        Text(s).font(Theme.font(12.5)).foregroundStyle(Theme.text2)
                            .padding(.horizontal, 12).padding(.vertical, 7)
                            .background(Theme.raised, in: Capsule())
                    }
                    .buttonStyle(.plain)
                }
            }
            .disabled(!labels.modelReady)
        }
        .padding(.vertical, 6)
    }

    static let suggestions = [
        "What do you see in this footage?",
        "This is <game>. The goal in these clips is to …",
        "Name the places and objects you can recognise.",
        "Keep instructions short and about where I'm heading.",
    ]

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField(labels.modelReady ? "Message the labeller" : "Download a labelling model first", text: $draft, axis: .vertical)
                .textFieldStyle(DarkFieldStyle())
                .lineLimit(1 ... 5)
                .focused($focused)
                .onSubmit(send)
                .disabled(!labels.modelReady || labels.running)
            if labels.chatBusy {
                Button { labels.stopReply() } label: { Image(systemName: "stop.fill").font(.system(size: 12, weight: .bold)).frame(width: 36, height: 36) }
                    .buttonStyle(.plain).background(Theme.raised, in: Circle()).help("Stop the reply")
            } else {
                Button(action: send) {
                    Image(systemName: "arrow.up").font(.system(size: 14, weight: .bold)).foregroundStyle(.black).frame(width: 36, height: 36)
                }
                .buttonStyle(.plain)
                .background(canSend ? Theme.text : Theme.off, in: Circle())
                .disabled(!canSend)
                .keyboardShortcut(.return, modifiers: .command)
                .help("Send (Return)")
            }
        }
    }

    private var canSend: Bool { labels.modelReady && !labels.running && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    private func send() {
        guard canSend else { return }
        labels.send(draft, about: recordings)
        draft = ""
    }
}

private struct ChatBubble: View {
    let role: LabelChatMessage.Role
    let text: String
    var streaming = false

    var body: some View {
        if role == .user {
            HStack {
                Spacer(minLength: 60)
                Text(text).font(Theme.font(13.5)).foregroundStyle(Theme.text).textSelection(.enabled)
                    .padding(.horizontal, 14).padding(.vertical, 9)
                    .background(Theme.raised, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            }
        } else {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "sparkles").font(.system(size: 11, weight: .semibold)).foregroundStyle(Theme.ai)
                    .frame(width: 24, height: 24).background(Theme.ai.opacity(0.14), in: Circle())
                if streaming && text.isEmpty {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("Watching the footage…").font(Theme.font(13)).foregroundStyle(Theme.muted)
                    }
                    .padding(.top, 3)
                } else {
                    Text(text + (streaming ? " ▍" : "")).font(Theme.font(13.5)).foregroundStyle(Theme.text2).textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true).padding(.top, 3)
                }
                Spacer(minLength: 40)
            }
        }
    }
}

// MARK: Options

struct LabelOptionsPanel: View {
    let app: AppModel
    let recordings: [LibraryRecording]

    var body: some View {
        @Bindable var settings = app.settings
        let o = settings.stored.labelOptions
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                LabelModelRows(app: app)
                Rows(footer: detailNote(o.detail)) {
                    Row(title: "Detail") {
                        Segmented(options: [(LabelDetail.fine, "Fine"), (.normal, "Normal"), (.broad, "Broad")], selection: $settings.stored.labelOptions.detail, compact: true)
                    }
                    Row(title: "Variations", detail: "Extra wordings each") {
                        NumberStepper(value: $settings.stored.labelOptions.variations, range: 0 ... 8)
                    }
                    Row(title: "Video detail") {
                        Segmented(options: [(LabelVideoDetail.standard, "Standard"), (.high, "High")], selection: $settings.stored.labelOptions.video, compact: true)
                    }
                    Toggle(isOn: $settings.stored.labelOptions.doubleCheck) {
                        Row(title: "Double-check each instruction", detail: "A second look against the clip. About twice as slow.")
                    }
                    Toggle(isOn: $settings.stored.labelOptions.replaceAutomatic) {
                        Row(title: "Replace earlier automatic ones", detail: "Yours are always kept.")
                    }
                }
                .toggleStyle(.oled)
                estimate(o)
            }
            .padding(.bottom, 6)
        }
        .scrollIndicators(.never)
    }

    func detailNote(_ d: LabelDetail) -> String {
        switch d {
        case .fine: "A new instruction every few seconds: each turn, jump or pickup."
        case .normal: "About one instruction per goal, every 5 to 15 seconds."
        case .broad: "Long stretches under one broad goal."
        }
    }

    func secondsPerClip(_ o: LabelOptions) -> Double {
        let light = app.labels.model.id == LanguageModels.labelerLight.id
        let base = (light ? 7.0 : 26.0) * (o.video == .high ? 2 : 1) + Double(o.variations) * (light ? 0.5 : 2)
        return base * (o.doubleCheck ? 1.9 : 1)
    }

    /// Rough time for the whole run, from measured speeds on Apple silicon.
    @ViewBuilder
    func estimate(_ o: LabelOptions) -> some View {
        let footage = recordings.reduce(0) { $0 + $1.meta.duration }
        let clips = max(1, footage / o.detail.seconds)
        let perClip = secondsPerClip(o)
        HStack(spacing: 8) {
            Image(systemName: "clock").font(.system(size: 12))
            Text("About \(LabelProgressLine.left(clips * perClip)) for \(footage.durationText) of footage")
        }
        .font(Theme.font(12.5)).foregroundStyle(Theme.muted)
        .padding(.horizontal, 4)
    }
}

/// Which labelling model to use, and getting it.
struct LabelModelRows: View {
    let app: AppModel
    var body: some View {
        @Bindable var settings = app.settings
        let current = app.labels.model
        Rows(title: "Model") {
            ForEach(LanguageModels.labelers, id: \.id) { m in
                let ready = LanguageModels.isDownloaded(m, pretrainedRoot: app.settings.paths.pretrainedRoot)
                let st = app.hub.state(m)
                Button {
                    if ready { settings.stored.labelerModelID = m.id }
                } label: {
                    HStack(alignment: .top, spacing: 12) {
                        Circle().strokeBorder(current.id == m.id && ready ? Theme.text : Theme.faint, lineWidth: current.id == m.id && ready ? 5 : 1.5)
                            .frame(width: 16, height: 16).padding(.top, 2)
                        VStack(alignment: .leading, spacing: 3) {
                            HStack(spacing: 6) {
                                Text(m.title).font(Theme.font(14, .semibold)).foregroundStyle(Theme.text)
                                if m.id == LanguageModels.labelerBest.id { Pill(text: "Best", color: Theme.ai) }
                            }
                            Text(m.summary).font(Theme.font(12)).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
                            if !ready { download(m, st).padding(.top, 4) }
                        }
                        Spacer(minLength: 0)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
    }

    @ViewBuilder
    func download(_ m: LanguageModels.HubModel, _ st: HubDownloads.State) -> some View {
        if st.running {
            HStack(spacing: 8) {
                ProgressView(value: st.fraction).tint(Theme.text).frame(width: 110)
                Button("Cancel") { app.hub.cancel(m) }.buttonStyle(.atQuiet)
            }
        } else if app.hub.lmStudioCopy(m) != nil {
            Button("Use LM Studio's copy") { app.hub.importFromLMStudio(m) }.buttonStyle(.atSmallPrimary)
                .help("LM Studio already has this model. Using its files is instant and takes no extra space.")
        } else {
            HStack(spacing: 8) {
                Button("Download ~\(m.approximateBytes.bytesText)") { app.hub.start(m) }.buttonStyle(.atSmallPrimary)
                if let e = st.error { Text(e).font(Theme.font(11)).foregroundStyle(Theme.dangerText).lineLimit(1) }
            }
        }
    }
}
