import Foundation

/// A training job sent to the engine.
public struct TrainingRequest: Codable, Sendable {
    public var jobID = UUID()
    public var modelURL: URL
    public var trainRecordings: [URL]
    public var validationRecordings: [URL]
    public var cacheRoot: URL
    public var pretrainedRoot: URL
    /// Checkpoint folder name to resume from (nil = latest if any, else initialization).
    public var resumeFrom: String?
    public var memoryLimitGB: Double
    public init(modelURL: URL, trainRecordings: [URL], validationRecordings: [URL], cacheRoot: URL, pretrainedRoot: URL,
                resumeFrom: String? = nil, memoryLimitGB: Double = 24) {
        self.modelURL = modelURL; self.trainRecordings = trainRecordings; self.validationRecordings = validationRecordings
        self.cacheRoot = cacheRoot; self.pretrainedRoot = pretrainedRoot; self.resumeFrom = resumeFrom; self.memoryLimitGB = memoryLimitGB
    }
}

public enum TrainingPhase: String, Codable, Sendable {
    case preparing, training, validating, paused, saving, finished, stopped, failed
}

/// Validation results on held-out recordings, aimed at closed-loop performance.
public struct ValidationMetrics: Codable, Hashable, Sendable {
    public var step: Int
    /// Mean cross-entropy per head (keys, pointer, mouse x, mouse y).
    public var nll: [Double]
    /// P2P-style perplexity: exp(Σ CE / log #classes).
    public var perplexity: Double
    /// Relative cross-entropy vs. always predicting label frequencies, per head (%; higher is better, 0 = no better than chance).
    public var rce: [Double]
    public var keyF1: Double
    public var clickF1: Double
    public var mouseDirectionAccuracy: Double
    public var mouseCorrelation: Double
    /// Loss increase when the frames are shuffled in time: how much the model relies on what it sees.
    public var visualReliance: Double
    /// Loss decrease from giving the right instruction vs. none (nil when there are no instructions).
    public var languageReliance: Double?
    /// Loss increase when the past actions are hidden: how much the model leans on copying itself.
    public var historyReliance: Double?
    public var clips: Int
    public init(step: Int, nll: [Double], perplexity: Double, rce: [Double], keyF1: Double, clickF1: Double, mouseDirectionAccuracy: Double,
                mouseCorrelation: Double, visualReliance: Double, languageReliance: Double?, historyReliance: Double? = nil, clips: Int) {
        self.step = step; self.nll = nll; self.perplexity = perplexity; self.rce = rce; self.keyF1 = keyF1; self.clickF1 = clickF1
        self.mouseDirectionAccuracy = mouseDirectionAccuracy; self.mouseCorrelation = mouseCorrelation
        self.visualReliance = visualReliance; self.languageReliance = languageReliance; self.historyReliance = historyReliance; self.clips = clips
    }
}

public struct TrainingProgress: Codable, Hashable, Sendable {
    public var step: Int
    public var totalSteps: Int
    public var epoch: Double
    public var loss: Double
    public var headLosses: [Double]
    public var learningRate: Double
    public var gradNorm: Double
    public var secondsPerStep: Double
    public var framesPerSecond: Double
    public var gpuMemoryGB: Double
    public var peakMemoryGB: Double
    public init(step: Int, totalSteps: Int, epoch: Double, loss: Double, headLosses: [Double], learningRate: Double, gradNorm: Double,
                secondsPerStep: Double, framesPerSecond: Double, gpuMemoryGB: Double, peakMemoryGB: Double) {
        self.step = step; self.totalSteps = totalSteps; self.epoch = epoch; self.loss = loss; self.headLosses = headLosses
        self.learningRate = learningRate; self.gradNorm = gradNorm; self.secondsPerStep = secondsPerStep
        self.framesPerSecond = framesPerSecond; self.gpuMemoryGB = gpuMemoryGB; self.peakMemoryGB = peakMemoryGB
    }
}

public struct DatasetSummary: Codable, Hashable, Sendable {
    public var trainRecordings: Int
    public var validationRecordings: Int
    public var trainSeconds: Double
    public var validationSeconds: Double
    public var trainClips: Int
    public var validationClips: Int
    public var instructionCoverage: Double
    public var mouseUnitScale: Double
    public var batchSize: Int
    public var accumulation: Int
    public init(trainRecordings: Int, validationRecordings: Int, trainSeconds: Double, validationSeconds: Double, trainClips: Int,
                validationClips: Int, instructionCoverage: Double, mouseUnitScale: Double, batchSize: Int, accumulation: Int) {
        self.trainRecordings = trainRecordings; self.validationRecordings = validationRecordings; self.trainSeconds = trainSeconds
        self.validationSeconds = validationSeconds; self.trainClips = trainClips; self.validationClips = validationClips
        self.instructionCoverage = instructionCoverage; self.mouseUnitScale = mouseUnitScale; self.batchSize = batchSize; self.accumulation = accumulation
    }
}

/// Events streamed from the engine to the app (JSON over XPC).
public enum TrainingEvent: Codable, Sendable {
    case phase(TrainingPhase, String)
    case dataset(DatasetSummary)
    case progress(TrainingProgress)
    case validation(ValidationMetrics)
    case checkpoint(name: String, step: Int)
    case log(String)
    case failed(String)
    /// Auto-labelling: one new instruction for a recording (written to it already).
    case labelled(recording: URL, segment: TextSegment)
    /// Auto-labelling progress: recording i of n, fraction of it done, seconds left (estimate).
    case labelProgress(recording: Int, of: Int, fraction: Double, secondsLeft: Double?, status: String)
    /// Chat with the labelling model: more of its reply, or the end of it.
    case chatDelta(String)
    case chatDone(String?)

    public func encoded() -> Data { (try? TrainingJSON.encoder().encode(self)) ?? Data() }
    public static func decode(_ d: Data) -> TrainingEvent? { try? TrainingJSON.decoder().decode(TrainingEvent.self, from: d) }
}

/// JSON for training events and logs. Plain JSON can't hold NaN or infinity, and a single NaN loss
/// used to make every progress event fail to encode, so training looked like it never started.
public enum TrainingJSON {
    public static func encoder() -> JSONEncoder {
        let e = JSONEncoder()
        e.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        return e
    }
    public static func decoder() -> JSONDecoder {
        let d = JSONDecoder()
        d.nonConformingFloatDecodingStrategy = .convertFromString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        return d
    }
}

/// One line of `training-log.jsonl` in the model package (lets the UI show history after restarts).
public struct TrainingLogLine: Codable, Sendable {
    public var date = Date()
    public var progress: TrainingProgress?
    public var validation: ValidationMetrics?
    public init(progress: TrainingProgress? = nil, validation: ValidationMetrics? = nil) { self.progress = progress; self.validation = validation }
}
