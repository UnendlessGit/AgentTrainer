import Foundation

/// How finely recordings are cut into instructions.
public enum LabelDetail: String, Codable, Sendable, CaseIterable {
    /// A new instruction every few seconds: each turn, jump or pickup.
    case fine
    /// About one instruction per goal (the default).
    case normal
    /// Long stretches under one broad goal.
    case broad

    /// Target seconds per instruction.
    public var seconds: Double {
        switch self { case .fine: 5; case .normal: 9; case .broad: 18 }
    }
}

/// How much of each frame the labeller sees. Time per clip grows with it.
public enum LabelVideoDetail: String, Codable, Sendable, CaseIterable {
    case standard, high
    public var maxPixels: Int { self == .standard ? 512 * 288 : 768 * 432 }
}

/// One turn of the conversation you have with the labelling model about the footage.
public struct LabelChatMessage: Codable, Sendable, Hashable, Identifiable {
    public enum Role: String, Codable, Sendable { case user, assistant }
    public var id = UUID()
    public var role: Role
    public var text: String
    public init(role: Role, text: String) { self.role = role; self.text = text }
}

/// Asks the engine to write instructions for recordings with the local vision-language model.
public struct AutoLabelRequest: Codable, Sendable {
    public var jobID = UUID()
    public var recordings: [URL]
    public var modelDirectory: URL
    public var detail = LabelDetail.normal
    /// Extra wordings written for every instruction (0 = just the instruction).
    public var variations = 5
    /// A second look at every instruction against the clip, fixing ones that don't match (about
    /// twice as slow).
    public var doubleCheck = false
    /// What you told the labeller about the footage.
    public var conversation: [LabelChatMessage] = []
    /// Replace earlier automatic labels (yours are always kept).
    public var replaceAutomatic = true
    /// Video the model watches per clip: frames per second and pixels per frame.
    public var framesPerSecond = 2.0
    public var maxPixels = LabelVideoDetail.standard.maxPixels
    public init(recordings: [URL], modelDirectory: URL) { self.recordings = recordings; self.modelDirectory = modelDirectory }

    public init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        let x = AutoLabelRequest(recordings: [], modelDirectory: URL(fileURLWithPath: "/"))
        jobID = (try? c.decode(UUID.self, forKey: .jobID)) ?? x.jobID
        recordings = try c.decode([URL].self, forKey: .recordings)
        modelDirectory = try c.decode(URL.self, forKey: .modelDirectory)
        detail = (try? c.decode(LabelDetail.self, forKey: .detail)) ?? x.detail
        variations = (try? c.decode(Int.self, forKey: .variations)) ?? x.variations
        doubleCheck = (try? c.decode(Bool.self, forKey: .doubleCheck)) ?? x.doubleCheck
        conversation = (try? c.decode([LabelChatMessage].self, forKey: .conversation)) ?? x.conversation
        replaceAutomatic = (try? c.decode(Bool.self, forKey: .replaceAutomatic)) ?? x.replaceAutomatic
        framesPerSecond = (try? c.decode(Double.self, forKey: .framesPerSecond)) ?? x.framesPerSecond
        maxPixels = (try? c.decode(Int.self, forKey: .maxPixels)) ?? x.maxPixels
    }
}

/// A message to the labelling model about the footage it is going to label; it watches a quick
/// overview of the recordings so it can answer about them.
public struct LabelChatRequest: Codable, Sendable {
    public var jobID = UUID()
    public var recordings: [URL]
    public var modelDirectory: URL
    public var messages: [LabelChatMessage]
    public init(recordings: [URL], modelDirectory: URL, messages: [LabelChatMessage]) {
        self.recordings = recordings; self.modelDirectory = modelDirectory; self.messages = messages
    }
}

/// Everything the engine can run.
public enum EngineJob: Codable, Sendable {
    case train(TrainingRequest)
    case autoLabel(AutoLabelRequest)
    case labelChat(LabelChatRequest)
}
