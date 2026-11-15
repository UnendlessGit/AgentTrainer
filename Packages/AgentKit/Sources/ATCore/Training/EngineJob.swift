import Foundation

/// Asks the engine to write instructions for recordings with the local vision-language model.
public struct AutoLabelRequest: Codable, Sendable {
    public var jobID = UUID()
    public var recordings: [URL]
    public var modelDirectory: URL
    /// Seconds per labelled window.
    public var windowSeconds: Double = 20
    public var framesPerWindow: Int = 4
    /// Replace earlier automatic labels (manual ones are always kept).
    public var replaceAutomatic = true
    public init(recordings: [URL], modelDirectory: URL) { self.recordings = recordings; self.modelDirectory = modelDirectory }
}

/// Everything the engine can run.
public enum EngineJob: Codable, Sendable {
    case train(TrainingRequest)
    case autoLabel(AutoLabelRequest)
}
