import Foundation

public enum ModelInitialization: String, Codable, CaseIterable, Hashable, Sendable {
    case p2p150M, p2p300M, blank
    public var label: String {
        switch self {
        case .p2p150M: "Pixel2Play 150M (pretrained)"
        case .p2p300M: "Pixel2Play 300M (pretrained)"
        case .blank: "Blank (train from scratch)"
        }
    }
    public var checkpointFolder: String? {
        switch self {
        case .p2p150M: "150M"
        case .p2p300M: "300M"
        case .blank: nil
        }
    }
    public var downloadBytes: Int64 {
        switch self {
        case .p2p150M: 2_198_488_343
        case .p2p300M: 3_711_005_207
        case .blank: 0
        }
    }
}

public enum LRSchedule: String, Codable, CaseIterable, Hashable, Sendable { case constant, cosine }

public enum Augmentation: String, Codable, CaseIterable, Hashable, Sendable, Identifiable {
    case spatial, color, planckian, isoNoise, translation, blur, sharpness
    public var id: String { rawValue }
    public var label: String {
        switch self {
        case .spatial: "Perspective"
        case .color: "Color jitter"
        case .planckian: "Color temperature"
        case .isoNoise: "Sensor noise"
        case .translation: "Shift"
        case .blur: "Blur"
        case .sharpness: "Sharpness"
        }
    }
}

public struct TrainingHyperparameters: Codable, Hashable, Sendable {
    public var learningRate = 1e-4
    public var weightDecay = 1e-4
    public var beta1 = 0.9
    public var beta2 = 0.999
    public var warmupSteps = 200
    public var schedule = LRSchedule.cosine
    public var minLRFactor = 0.1
    public var gradientClip = 1.0
    public var zLossWeight = 1e-4
    /// Clips per optimizer step (0 = choose automatically from memory).
    public var batchSize = 0
    public var gradientAccumulation = 1
    public var totalSteps = 4000
    public var validationFraction = 0.1
    public var validateEverySteps = 250
    public var checkpointEveryMinutes = 10.0
    public var augmentations: [Augmentation] = Augmentation.allCases
    public var augmentationStrength = 1.0
    public var instructionDropout = 0.25
    public var bf16 = true
    public var seed = 0
    public init() {}

    public static var fineTune: TrainingHyperparameters {
        var h = TrainingHyperparameters(); h.learningRate = 3e-5; h.warmupSteps = 100; return h
    }
}

/// A model the user created: its architecture, action vocabulary and training settings.
public struct ModelDefinition: Codable, Hashable, Sendable, Identifiable {
    public var id = UUID()
    public var name: String
    public var createdAt = Date()
    public var notes = ""
    public var initialization: ModelInitialization
    public var architecture: PolicyArchitecture
    public var actions: ActionSpaceConfig
    public var training: TrainingHyperparameters
    /// Whether instructions are fed to the model (the text token); off = always "no instruction".
    public var usesInstructions = true
    /// Locked once training has started (changing it would invalidate the weights).
    public var architectureLocked = false
    public var trainedSteps = 0
    public var lastTrainedAt: Date?
    /// Checkpoint used by Run (nil = latest).
    public var activeCheckpoint: String?

    public init(name: String, initialization: ModelInitialization, architecture: PolicyArchitecture,
                actions: ActionSpaceConfig, training: TrainingHyperparameters) {
        self.name = name; self.initialization = initialization; self.architecture = architecture
        self.actions = actions; self.training = training
    }

    public var parameterCount: Int { architecture.parameterCount(actions: actions) }
    public var isTrained: Bool { trainedSteps > 0 }

    public enum Preset: String, CaseIterable, Identifiable, Sendable {
        case gameP2P150, gameP2P300, appControl, compact
        public var id: String { rawValue }
        public var title: String {
            switch self {
            case .gameP2P150: "Game agent · P2P 150M"
            case .gameP2P300: "Game agent · P2P 300M"
            case .appControl: "App control"
            case .compact: "Compact (from scratch)"
            }
        }
        public var subtitle: String {
            switch self {
            case .gameP2P150: "Recommended. 183M parameters, pretrained on 8,300 h of gameplay. ~75 FPS on M3 Max."
            case .gameP2P300: "309M parameters, pretrained. More capacity, ~48 FPS, trains ~2× slower."
            case .appControl: "Full keyboard, absolute pointer, screen-text reading and typing. Starts from P2P 150M."
            case .compact: "60M parameters, blank. Fast to train; needs more recordings to play well."
            }
        }
    }

    public static func make(_ preset: Preset, name: String) -> ModelDefinition {
        switch preset {
        case .gameP2P150:
            return ModelDefinition(name: name, initialization: .p2p150M, architecture: .p2p150M, actions: .gameDefault, training: .fineTune)
        case .gameP2P300:
            return ModelDefinition(name: name, initialization: .p2p300M, architecture: .p2p300M, actions: .gameDefault, training: .fineTune)
        case .appControl:
            var a = PolicyArchitecture.p2p150M
            a.screenText = true; a.literalTyping = true
            return ModelDefinition(name: name, initialization: .p2p150M, architecture: a, actions: .appDefault, training: .fineTune)
        case .compact:
            var t = TrainingHyperparameters(); t.totalSteps = 8000
            return ModelDefinition(name: name, initialization: .blank, architecture: .compact60M, actions: .gameDefault, training: t)
        }
    }
}

/// `<name>.atmodel/` layout.
public enum ModelPackage {
    public static let fileExtension = "atmodel"
    public static let definition = "model.json"
    public static let checkpoints = "checkpoints"
    public static let log = "training-log.jsonl"
    public static let metrics = "metrics.json"
    public static let weights = "weights.safetensors"
    public static let optimizer = "optimizer.safetensors"
    public static let state = "state.json"

    public static func read(_ url: URL) throws -> ModelDefinition {
        try RecordingIO.decoder.decode(ModelDefinition.self, from: Data(contentsOf: url.appendingPathComponent(definition)))
    }
    public static func write(_ m: ModelDefinition, to url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try RecordingIO.encoder.encode(m).write(to: url.appendingPathComponent(definition), options: .atomic)
    }
    public static func checkpointNames(_ url: URL) -> [String] {
        let dir = url.appendingPathComponent(checkpoints)
        return ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
            .filter { FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).appendingPathComponent(weights).path) }
            .sorted()
    }
}

/// Saved alongside each checkpoint.
public struct CheckpointState: Codable, Hashable, Sendable {
    public var step: Int
    public var epoch: Double
    public var trainLoss: Double?
    public var validationLoss: Double?
    public var savedAt = Date()
    public var rngSeed: UInt64
    public var dataCursor: Int
    public init(step: Int, epoch: Double, trainLoss: Double?, validationLoss: Double?, rngSeed: UInt64, dataCursor: Int) {
        self.step = step; self.epoch = epoch; self.trainLoss = trainLoss; self.validationLoss = validationLoss
        self.rngSeed = rngSeed; self.dataCursor = dataCursor
    }
    public static func name(step: Int) -> String { String(format: "step-%07d", step) }
}
