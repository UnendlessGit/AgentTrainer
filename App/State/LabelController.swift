import ATCore
import ATLanguage
import Foundation
import Observation

/// Auto Label: the local vision-language model watches recordings and writes their instructions,
/// with variations, guided by a conversation you have with it about the footage.
@MainActor @Observable
final class LabelController {
    struct Progress: Equatable {
        var recording = 0
        var count = 0
        var fraction = 0.0
        var secondsLeft: Double?
        var status = ""
    }

    private let settings: AppSettings
    private var connection: NSXPCConnection?

    // Labelling
    private(set) var running = false
    private(set) var progress = Progress()
    /// Recordings in the current (or last) run.
    private(set) var recordings: [URL] = []
    /// Instructions written in the current (or last) run.
    var written: Int { writtenIDs.count }
    private var writtenIDs = Set<UUID>()
    var lastError: String?
    /// The latest instructions written in this run, newest last (a merged one replaces its earlier version).
    private(set) var recent: [(recording: URL, segment: TextSegment)] = []
    /// Bumped each time an instruction is written to a recording, so open views can reload it.
    private(set) var revisions: [URL: Int] = [:]
    var onFinished: (() -> Void)?

    // Chat about the footage
    private(set) var chat: [LabelChatMessage] = []
    /// The reply being written, while it streams.
    private(set) var reply: String?
    private(set) var chatBusy = false
    var chatError: String?
    /// Recordings the conversation is about.
    private(set) var chatRecordings: [URL] = []

    /// Self-tests and snapshots must not touch your saved conversation.
    private let persists = !CommandLine.arguments.contains { $0 == "--ui-snapshot" || $0.hasPrefix("--selftest") }

    init(settings: AppSettings) {
        self.settings = settings
        if persists, let d = UserDefaults.standard.data(forKey: "labelChat"), let saved = try? JSONDecoder().decode(SavedChat.self, from: d) {
            chat = saved.messages
            chatRecordings = saved.recordings
        }
    }

    // MARK: Model

    /// The labelling model Auto Label uses: the chosen one, else the best one downloaded.
    var model: LanguageModels.HubModel {
        LanguageModels.labeler(id: settings.stored.labelerModelID)
            ?? LanguageModels.bestDownloadedLabeler(pretrainedRoot: settings.paths.pretrainedRoot)
            ?? LanguageModels.labelerBest
    }

    var modelReady: Bool { LanguageModels.isDownloaded(model, pretrainedRoot: settings.paths.pretrainedRoot) }
    private var modelDirectory: URL { LanguageModels.directory(model, pretrainedRoot: settings.paths.pretrainedRoot) }

    // MARK: Labelling

    func label(_ urls: [URL], options: LabelOptions? = nil) {
        guard !running, !urls.isEmpty, modelReady else { return }
        let o = options ?? settings.stored.labelOptions
        var req = AutoLabelRequest(recordings: urls, modelDirectory: modelDirectory)
        req.detail = o.detail
        req.variations = o.variations
        req.doubleCheck = o.doubleCheck
        req.maxPixels = o.video.maxPixels
        req.replaceAutomatic = o.replaceAutomatic
        // The conversation counts when it's about these recordings (or about some of them).
        if !chat.isEmpty, chatRecordings.isEmpty || !Set(chatRecordings).isDisjoint(with: urls) { req.conversation = chat }
        guard let data = try? JSONEncoder().encode(EngineJob.autoLabel(req)) else { return }
        running = true; lastError = nil; writtenIDs = []; recent = []; recordings = urls
        progress = Progress(recording: 0, count: urls.count, status: "Starting")
        engine()?.submit(data) { [weak self] _, err in
            DispatchQueue.main.async { MainActor.assumeIsolated { if let err { self?.running = false; self?.lastError = err } } }
        }
    }

    func stop() {
        progress.status = "Stopping after this clip…"
        engine()?.control("", command: "stop-labelling") { _ in }
    }

    func isLabelling(_ url: URL) -> Bool { running && recordings.contains(url) }
    func revision(_ url: URL) -> Int { revisions[url] ?? 0 }

    // MARK: Chat

    /// Sends a message about `recordings` (a different set of recordings starts a new conversation).
    func send(_ text: String, about urls: [URL]) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, !chatBusy, !running, modelReady else { return }
        if Set(urls) != Set(chatRecordings) { chat = []; chatRecordings = urls }
        chat.append(LabelChatMessage(role: .user, text: t))
        saveChat()
        let req = LabelChatRequest(recordings: chatRecordings, modelDirectory: modelDirectory, messages: chat)
        guard let data = try? JSONEncoder().encode(EngineJob.labelChat(req)) else { return }
        chatBusy = true; chatError = nil; reply = ""
        engine()?.submit(data) { [weak self] _, err in
            DispatchQueue.main.async { MainActor.assumeIsolated { if let err { self?.chatBusy = false; self?.reply = nil; self?.chatError = err } } }
        }
    }

    func stopReply() { engine()?.control("", command: "stop-labelling") { _ in } }

    func newConversation() {
        guard !chatBusy else { return }
        chat = []; chatRecordings = []; reply = nil; chatError = nil
        saveChat()
    }

    /// UI snapshots only: shows a sample conversation and labels without the model.
    func previewForSnapshot(recordings urls: [URL], live: Bool) {
        chatRecordings = urls
        chat = [
            LabelChatMessage(role: .user, text: "What do you see in this footage?"),
            LabelChatMessage(role: .assistant, text: "A first-person drive through a city at night. The car stops at red lights and carries on when they turn green; the HUD shows a speedometer and a minimap."),
            LabelChatMessage(role: .user, text: "It's a taxi game. Each run I pick up a passenger and drive them to the marker on the minimap."),
        ]
        chatBusy = !live
        reply = live ? nil : "Got it: I'll write instructions about reaching the passenger and following the minimap route, and mention stops"
        guard live else { return }
        running = true
        recordings = urls
        progress = Progress(recording: 0, count: urls.count, fraction: 0.42, secondsLeft: 380, status: "Downtown run · clip 5 of 12")
        recent = [
            (urls[0], TextSegment(start: 0, end: 10.9, text: "Stop at the red light, then drive on when it turns green", source: .auto,
                                  variants: ["Wait for the green light before driving on", "Halt at the red signal and go on green", "Stop for the red light and continue on green"],
                                  observation: "Night-time city street; the car brakes at a red light and pulls away when it turns green.")),
            (urls[0], TextSegment(start: 10.9, end: 19.9, text: "Follow the minimap route to the passenger", source: .auto,
                                  variants: ["Drive to the passenger marker on the minimap", "Head to the pickup point"],
                                  observation: "The car turns onto a wide avenue; the minimap shows a marker ahead.")),
        ]
        for r in recent { writtenIDs.insert(r.segment.id) }
    }

    /// Whether the saved conversation is about these recordings.
    func conversationApplies(to urls: [URL]) -> Bool { !chat.isEmpty && Set(chatRecordings) == Set(urls) }

    private struct SavedChat: Codable { var messages: [LabelChatMessage]; var recordings: [URL] }
    private func saveChat() {
        guard persists else { return }
        if let d = try? JSONEncoder().encode(SavedChat(messages: chat, recordings: chatRecordings)) { UserDefaults.standard.set(d, forKey: "labelChat") }
    }

    // MARK: Engine

    private func engine() -> EngineServiceProtocol? {
        if connection == nil {
            let c = NSXPCConnection(serviceName: EngineXPC.serviceName)
            c.remoteObjectInterface = NSXPCInterface(with: EngineServiceProtocol.self)
            c.exportedInterface = NSXPCInterface(with: EngineClientProtocol.self)
            c.exportedObject = EngineEventReceiver { [weak self] e in DispatchQueue.main.async { MainActor.assumeIsolated { self?.handle(e) } } }
            c.setCodeSigningRequirement(EngineXPC.serviceRequirement)
            c.invalidationHandler = { [weak self] in DispatchQueue.main.async { MainActor.assumeIsolated { self?.engineLost() } } }
            c.interruptionHandler = { [weak self] in DispatchQueue.main.async { MainActor.assumeIsolated { self?.engineLost() } } }
            c.resume()
            connection = c
        }
        return connection?.remoteObjectProxyWithErrorHandler { [weak self] e in
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.lastError = e.localizedDescription } }
        } as? EngineServiceProtocol
    }

    private func engineLost() {
        connection = nil
        if running { running = false; lastError = "The labelling engine stopped unexpectedly. Instructions written so far are kept."; onFinished?() }
        if chatBusy { chatBusy = false; reply = nil; chatError = "The labelling engine stopped unexpectedly." }
    }

    private func handle(_ e: TrainingEvent) {
        switch e {
        case .labelProgress(let r, let n, let f, let left, let status):
            progress = Progress(recording: r, count: n, fraction: f, secondsLeft: left, status: status)
        case .labelled(let url, let seg):
            writtenIDs.insert(seg.id)
            recent.removeAll { $0.segment.id == seg.id }
            recent.append((url, seg))
            if recent.count > 30 { recent.removeFirst(recent.count - 30) }
            revisions[url, default: 0] += 1
        case .phase(let p, let m) where [.finished, .stopped].contains(p):
            running = false
            progress.status = m
            progress.fraction = p == .finished ? 1 : progress.fraction
            onFinished?()
        case .failed(let m):
            running = false
            lastError = m
            onFinished?()
        case .chatDelta(let t):
            if chatBusy { reply = t }
        case .chatDone(let err):
            if let err {
                chatError = err
            } else if let r = reply?.trimmingCharacters(in: .whitespacesAndNewlines), !r.isEmpty {
                chat.append(LabelChatMessage(role: .assistant, text: r))
                saveChat()
            }
            reply = nil
            chatBusy = false
        default: break
        }
    }
}
