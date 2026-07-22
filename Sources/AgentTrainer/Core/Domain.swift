import Foundation
import CoreGraphics

enum AppSection: String, CaseIterable, Identifiable {
    case home = "Home"
    case record = "Record"
    case library = "Library"
    case models = "AI Models"
    case training = "Training"
    case run = "Run"
    case diagnostics = "Diagnostics"
    case settings = "Settings"

    var id: String { rawValue }
    var symbol: String {
        switch self {
        case .home: "house"
        case .record: "record.circle"
        case .library: "rectangle.stack"
        case .models: "cpu"
        case .training: "chart.xyaxis.line"
        case .run: "play.fill"
        case .diagnostics: "waveform.path.ecg.rectangle"
        case .settings: "gearshape"
        }
    }
}

struct CodableRect: Codable, Hashable, Sendable {
    var x: Double
    var y: Double
    var width: Double
    var height: Double

    init(_ rect: CGRect) {
        x = rect.origin.x
        y = rect.origin.y
        width = rect.width
        height = rect.height
    }

    var cgRect: CGRect { CGRect(x: x, y: y, width: width, height: height) }
}

enum CaptureKind: String, Codable, CaseIterable, Identifiable, Sendable {
    case display = "Display"
    case window = "Window"
    case windowRegion = "Window Region"
    case screenRegion = "Screen Region"
    var id: String { rawValue }
}

struct CaptureSpec: Codable, Hashable, Sendable {
    var kind: CaptureKind = .display
    var displayID: UInt32?
    var windowID: UInt32?
    var region: CodableRect?
    var requestedFPS: Double = 60
    var showsCursor = false
}

enum ColorMode: String, Codable, CaseIterable, Identifiable, Sendable {
    case grayscale = "Grayscale"
    case color = "Color"
    var id: String { rawValue }
}

enum ChromaSubsampling: String, Codable, CaseIterable, Identifiable, Sendable {
    case yuv420 = "4:2:0"
    case yuv422 = "4:2:2"
    case yuv444 = "4:4:4"
    var id: String { rawValue }
}

enum ResizePolicy: String, Codable, CaseIterable, Identifiable, Sendable {
    case fit = "Fit"
    case fill = "Fill"
    case stretch = "Stretch"
    var id: String { rawValue }
}

struct PreprocessingSpec: Codable, Hashable, Sendable {
    var width: Int = 640
    var height: Int = 360
    var colorMode: ColorMode = .color
    var bitDepth: Int = 8
    var chroma: ChromaSubsampling = .yuv444
    var resizePolicy: ResizePolicy = .fit

    var channelCount: Int { colorMode == .grayscale ? 1 : 3 }
    var sampleByteCount: Int {
        guard width > 0, height > 0 else { return 0 }
        let y = saturatedMultiply(width, height)
        if colorMode == .grayscale { return y }
        switch chroma {
        case .yuv420: return saturatedAdd(y, saturatedMultiply(2, saturatedMultiply((width / 2) + (width % 2), (height / 2) + (height % 2))))
        case .yuv422: return saturatedAdd(y, saturatedMultiply(2, saturatedMultiply((width / 2) + (width % 2), height)))
        case .yuv444: return saturatedMultiply(y, 3)
        }
    }

    func validated() throws -> Self {
        guard width > 0, height > 0 else { throw AgentTrainerError.invalidConfiguration("Vision dimensions must be positive.") }
        let pixels = width.multipliedReportingOverflow(by: height)
        guard width <= 8_192, height <= 8_192, !pixels.overflow, pixels.partialValue <= 33_554_432 else {
            throw AgentTrainerError.invalidConfiguration("Model vision may be at most 8,192 pixels per side and 33.5 million pixels total.")
        }
        guard (1...8).contains(bitDepth) else { throw AgentTrainerError.invalidConfiguration("Color detail must be 1 through 8 bits.") }
        guard sampleByteCount < Int.max else { throw AgentTrainerError.invalidConfiguration("The selected vision dimensions exceed this Mac's addressable memory.") }
        return self
    }

    private func saturatedMultiply(_ lhs: Int, _ rhs: Int) -> Int { let result = lhs.multipliedReportingOverflow(by: rhs); return result.overflow ? Int.max : result.partialValue }
    private func saturatedAdd(_ lhs: Int, _ rhs: Int) -> Int { let result = lhs.addingReportingOverflow(rhs); return result.overflow ? Int.max : result.partialValue }
}

struct ActionChannels: Codable, Hashable, Sendable {
    var absoluteMouse = true
    var relativeMouse = false
    var buttons = true
    var scroll = true
    /// Standard keyboard keys plus Shift in training-data schema 7 and later.
    var keyboard = true
    /// Command, Option, and Control in training-data schema 7 and later.
    var modifiers = true

    static let all = ActionChannels(absoluteMouse: true, relativeMouse: true, buttons: true, scroll: true, keyboard: true, modifiers: true)

    /// Mouse demonstrations always contain both normalized cursor position and
    /// raw movement deltas. Training learns both representations; Run chooses
    /// which representation to execute.
    var mouseMovement: Bool {
        get { absoluteMouse || relativeMouse }
        set { absoluteMouse = newValue; relativeMouse = newValue }
    }
}

enum MouseControlMode: String, Codable, CaseIterable, Identifiable, Sendable {
    case automatic = "Auto (Recommended)"
    case absolute = "Absolute Cursor"
    case relative = "Game Camera"
    var id: String { rawValue }
}

enum ModelContract {
    /// Version 4 replaces the shallow globally-pooled encoder with a temporal,
    /// coordinate-aware spatial encoder. The new weights intentionally cannot
    /// be attached to older profiles because their tensor shapes and meaning
    /// are different.
    static let schemaVersion = 4
    static let weightFormat = "AgentTrainer.Policy.v4"
}

/// Version of the causal pairing between a captured frame and the controls the
/// model should perform next. This remains separate from the weight format so
/// a data-only correction can invalidate caches/checkpoints without needlessly
/// changing tensor shapes. Policy v4 itself is an intentional weight break.
enum TrainingDataContract {
    /// Version 7 preserves sub-tick key/button taps and preceding-perception
    /// motion, and assigns Shift to the Keyboard channel while the Modifiers
    /// channel owns only Control, Option, and Command.
    static let schemaVersion = 7
}

/// Stable training/runtime contract for locked-cursor game cameras. Raw HID
/// deltas are divided by this value in the dataset and multiplied by the same
/// value during execution, so the model learns useful values independent of the
/// capture resolution.
enum GameCameraContract {
    static let deltaScale: Float = 80
    static let maximumPostedDelta: CGFloat = 10_000

    static func trainingValue(forRawDelta delta: Double) -> Float {
        Swift.min(1, Swift.max(-1, Float(delta) / deltaScale))
    }

    static func runtimeDelta(forPrediction prediction: Float, sensitivity: Double) -> CGFloat {
        guard prediction.isFinite else { return 0 }
        let safeSensitivity = sensitivity.isFinite ? min(100, max(0.01, sensitivity)) : 1
        let value = CGFloat(prediction) * CGFloat(deltaScale) * safeSensitivity
        return min(maximumPostedDelta, max(-maximumPostedDelta, value))
    }
}

struct GameCameraSettings: Codable, Hashable, Sendable {
    /// Multiplier applied after reversing the fixed training scale.
    var sensitivity = 1.0
    /// Warp to the capture center before and after posting each raw delta. This
    /// matches the locked-camera input path used by games and prevents edges.
    var recenterCursor = true
}

/// Run-only output permissions. These are deliberately separate from an AI's
/// trained channels and per-profile restrictions so they can be changed during
/// a live session without changing the learned-brain contract.
struct RuntimeOutputPermissions: Codable, Hashable, Sendable {
    var cursorMovement = true
    var keyboard = true
}

enum RecurrentKind: String, Codable, CaseIterable, Identifiable, Sendable {
    case gru = "GRU"
    case lstm = "LSTM"
    var id: String { rawValue }
}

enum VisualPoolingKind: String, Codable, CaseIterable, Identifiable, Sendable {
    /// Preserves the exact Policy-v4 projection used by existing brains. It is
    /// expressive, but the first visual dense layer grows with every output
    /// grid position and therefore dominates larger models.
    case flattened = "Flattened Grid (Legacy)"
    /// Learns a small set of spatial keypoints, pools both features and exact
    /// coordinates around them, and retains global context. This keeps layout
    /// information while making parameter count nearly independent of vision
    /// resolution.
    case attention = "Attention Keypoints"

    var id: String { rawValue }
}

struct ArchitectureSpec: Codable, Hashable, Sendable {
    /// The stride-four stem cuts the dominant high-resolution convolution cost,
    /// while the extra stage expands the receptive field before spatial
    /// features are flattened. This is both faster and substantially more
    /// expressive than three stride-two layers followed by a global mean.
    var convolutionChannels: [Int] = [32, 64, 96, 128]
    var kernelSizes: [Int] = [7, 3, 3, 3]
    var strides: [Int] = [4, 2, 2, 2]
    var visualEmbedding = 256
    var recurrentKind: RecurrentKind = .gru
    var recurrentWidth = 192
    var fusionWidths: [Int] = [384, 256]
    var dropout: Double = 0.1

    /// Optional fields keep every existing profile and immutable brain
    /// byte-compatible. A missing value means the historical flattened
    /// projection; newly created profiles use efficient attention pooling.
    var visualPooling: VisualPoolingKind? = .attention
    var attentionHeads: Int? = 8

    var effectiveVisualPooling: VisualPoolingKind { visualPooling ?? .flattened }
    var effectiveAttentionHeads: Int { min(64, max(1, attentionHeads ?? 8)) }

    static let small = ArchitectureSpec(convolutionChannels: [24, 48, 72, 96], visualEmbedding: 192, recurrentWidth: 128, fusionWidths: [256, 192], visualPooling: .attention, attentionHeads: 6)
    static let balanced = ArchitectureSpec()
    static let large = ArchitectureSpec(convolutionChannels: [48, 96, 160, 224], visualEmbedding: 384, recurrentWidth: 256, fusionWidths: [512, 384], dropout: 0.12, visualPooling: .attention, attentionHeads: 12)
}

struct CNNLayerGeometry: Hashable, Sendable {
    var kernelSize: Int
    var effectiveStride: Int
    var receptiveField: Int
}

enum CNNGeometry {
    /// Standard receptive-field accumulation for the policy's unit-dilation
    /// convolution stack. Padding changes edge coverage but not field size.
    static func layer(_ requestedLayer: Int, architecture: ArchitectureSpec) -> CNNLayerGeometry {
        guard !architecture.convolutionChannels.isEmpty else {
            return CNNLayerGeometry(kernelSize: 1, effectiveStride: 1, receptiveField: 1)
        }
        let count = max(1, architecture.convolutionChannels.count)
        let layer = min(count - 1, max(0, requestedLayer))
        var receptiveField = 1
        var effectiveStride = 1
        var currentKernel = 1
        for index in 0...layer {
            currentKernel = architecture.kernelSizes.indices.contains(index) ? max(1, architecture.kernelSizes[index]) : 3
            let stride = architecture.strides.indices.contains(index) ? max(1, architecture.strides[index]) : 2
            receptiveField += (currentKernel - 1) * effectiveStride
            effectiveStride *= stride
        }
        return CNNLayerGeometry(kernelSize: currentKernel, effectiveStride: effectiveStride, receptiveField: receptiveField)
    }

    /// Exact NHWC spatial geometry produced by the configured same-ish padded
    /// convolutions. Keeping this in the domain layer ensures model sizing and
    /// the actual projection layer can never silently disagree.
    static func outputSize(width: Int, height: Int, architecture: ArchitectureSpec) -> (width: Int, height: Int) {
        var width = max(1, width)
        var height = max(1, height)
        for index in architecture.convolutionChannels.indices {
            let kernel = architecture.kernelSizes.indices.contains(index) ? max(1, architecture.kernelSizes[index]) : 3
            let stride = architecture.strides.indices.contains(index) ? max(1, architecture.strides[index]) : 2
            let padding = kernel / 2
            width = convolutionOutput(width, kernel: kernel, stride: stride, padding: padding)
            height = convolutionOutput(height, kernel: kernel, stride: stride, padding: padding)
        }
        return (width, height)
    }

    private static func convolutionOutput(_ input: Int, kernel: Int, stride: Int, padding: Int) -> Int {
        max(1, (max(0, input + 2 * padding - kernel) / max(1, stride)) + 1)
    }
}

enum TrainingPrecision: String, Codable, CaseIterable, Identifiable, Sendable {
    case float16 = "Float16"
    case bfloat16 = "BFloat16"
    case float32 = "Float32"
    var id: String { rawValue }
}

enum FrameMode: String, Codable, CaseIterable, Identifiable, Sendable {
    case newest = "Newest Frame"
    case ordered = "Every Frame"
    var id: String { rawValue }
}

enum LearningRateSchedule: String, Codable, CaseIterable, Identifiable, Sendable {
    /// Linear warmup followed by bounded cosine restarts. Epoch-end validation
    /// (or epoch-average training loss when no validation is possible) reduces
    /// the cycle envelope when progress stalls.
    case adaptiveCosine = "Adaptive Cosine + Plateau"
    /// Compatibility schedule for checkpoints created before adaptive training.
    case legacyInverseSquareRoot = "Legacy Inverse Square Root"

    var id: String { rawValue }
}

/// Run-only CNN inspection modes. They never participate in the learned-brain
/// contract, dataset identity, or saved model weights.
enum CNNVisualizationMode: String, Codable, CaseIterable, Identifiable, Sendable {
    case activationOverlay = "Activation Overlay"
    case featureChannels = "Feature Channels"
    case actionSaliency = "Action Saliency"
    var id: String { rawValue }
}

enum CNNActionFocus: String, Codable, CaseIterable, Identifiable, Sendable {
    case movement = "Movement"
    case mouseButtons = "Mouse Buttons"
    case scroll = "Scrolling"
    case keyboard = "Keyboard"
    case modifiers = "Modifiers"
    var id: String { rawValue }
    var displayName: String { rawValue }
}

/// Presentation-only controls for live CNN inspection. Sanitizing at the
/// runtime boundary prevents malformed persisted values from creating an
/// unbounded diagnostic rate or renderer workload.
struct CNNVisualizationSettings: Codable, Hashable, Sendable {
    var enabled = false
    var mode: CNNVisualizationMode = .activationOverlay
    var framesPerSecond = 4.0
    /// Zero-based convolution index. `-1` follows the final convolution even
    /// when the selected AI uses a different number of layers.
    var convolutionLayer = -1
    var featureChannelCount = 9
    var overlayOpacity = 0.68
    var actionFocus: CNNActionFocus = .movement

    func sanitized(layerCount: Int? = nil) -> Self {
        var value = self
        value.framesPerSecond = framesPerSecond.isFinite ? min(15, max(0.5, framesPerSecond)) : 4
        value.featureChannelCount = min(16, max(4, featureChannelCount))
        value.overlayOpacity = overlayOpacity.isFinite ? min(0.9, max(0.2, overlayOpacity)) : 0.68
        if let layerCount {
            value.convolutionLayer = layerCount > 0 ? min(layerCount - 1, max(0, convolutionLayer < 0 ? layerCount - 1 : convolutionLayer)) : -1
        }
        return value
    }
}

struct TrainingConfiguration: Codable, Hashable, Sendable {
    var epochs = 40
    var batchSize = 32
    var learningRate: Double = 0.0003
    var weightDecay: Double = 0.01
    var historyLength = 16
    var perceptionFPS: Double = 30
    var actionFPS: Double = 60
    var precision: TrainingPrecision = .bfloat16
    var validationSplit: Double = 0.15
    var checkpointInterval = 500
    var seed: UInt64 = 42
    var architecture: ArchitectureSpec = .balanced
    /// Nil is decoded from older profiles. It intentionally maps to the new safe default.
    var maximumSteps: Int? = 10_000

    /// Optional adaptive fields preserve exact behavior for profiles created by
    /// older releases: a missing schedule selects inverse-square-root decay and
    /// a missing focal gamma selects ordinary class-balanced BCE. New profiles
    /// receive the stronger defaults below.
    var learningRateSchedule: LearningRateSchedule? = .adaptiveCosine
    var cosineCycleEpochs: Int? = 8
    var plateauPatience: Int? = 5
    var minimumLearningRateRatio: Double? = 0.05
    var binaryFocalGamma: Double? = 1.5

    var effectiveMaximumSteps: Int { maximumSteps ?? 10_000 }
    var effectiveLearningRateSchedule: LearningRateSchedule { learningRateSchedule ?? .legacyInverseSquareRoot }
    var effectiveCosineCycleEpochs: Int { max(1, cosineCycleEpochs ?? 8) }
    var effectivePlateauPatience: Int { max(1, plateauPatience ?? 5) }
    var effectiveMinimumLearningRateRatio: Double {
        let value = minimumLearningRateRatio ?? 0.05
        return value.isFinite ? min(0.5, max(0.001, value)) : 0.05
    }
    var effectiveBinaryFocalGamma: Double {
        let value = binaryFocalGamma ?? 0
        return value.isFinite ? min(4, max(0, value)) : 0
    }
}

struct TrainingRunSettings: Codable, Hashable, Sendable {
    var maximumSteps = 10_000
    var autosaveSteps = 1_000
}

/// Turns the configured epoch count into a durable continuation goal. A paused
/// run keeps its existing goal; starting again after reaching that goal adds a
/// fresh block of configured epochs.
enum TrainingContinuationPlan {
    static func targetEpoch(completedEpoch: Int, batchOffset: Int, savedTarget: Int?, configuredIncrement: Int) -> Int {
        let completedEpoch = max(0, completedEpoch)
        let increment = max(1, configuredIncrement)
        let currentTarget = max(completedEpoch, savedTarget ?? max(increment, completedEpoch))
        let reachedTarget = batchOffset == 0 && completedEpoch >= currentTarget
        return reachedTarget ? completedEpoch + increment : currentTarget
    }

    static func remainingSteps(completedEpoch: Int, batchOffset: Int, targetEpoch: Int, samplesPerEpoch: Int, batchSize: Int) -> Int {
        let stepsPerEpoch = max(1, Int(ceil(Double(max(1, samplesPerEpoch)) / Double(max(1, batchSize)))))
        let completedBatches = min(stepsPerEpoch, Int(ceil(Double(max(0, batchOffset)) / Double(max(1, batchSize)))))
        let epochsRemainingAfterCurrent = max(0, targetEpoch - max(0, completedEpoch) - (batchOffset > 0 ? 1 : 0))
        let currentEpochRemaining = batchOffset > 0 ? max(0, stepsPerEpoch - completedBatches) : (targetEpoch > completedEpoch ? stepsPerEpoch : 0)
        return currentEpochRemaining + epochsRemainingAfterCurrent * stepsPerEpoch
    }
}

enum InputEventKind: UInt8, Codable, Sendable {
    case mouseMove = 1
    case mouseButton = 2
    case scroll = 3
    case key = 4
    case flags = 5
}

struct InputSample: Codable, Hashable, Sendable {
    var timestampNanos: UInt64
    var kind: InputEventKind
    var x: Double = 0
    var y: Double = 0
    var deltaX: Double = 0
    var deltaY: Double = 0
    var button: UInt8 = 0
    var scrollX: Double = 0
    var scrollY: Double = 0
    var keyCode: UInt16 = 0
    var modifiers: UInt64 = 0
    var isDown = false
}

struct InputState: Equatable, Sendable {
    var keys: Set<UInt16> = []
    var buttons: Set<UInt8> = []
    var modifiers: UInt64 = 0
    var mouseDelta = CGSize.zero
    var scrollDelta = CGSize.zero

    static let empty = InputState()
}

struct RecordingManifest: Codable, Hashable, Identifiable, Sendable {
    var schemaVersion = 2
    var id: UUID
    var name: String
    var createdAt: Date
    var hostStartNanos: UInt64
    var duration: Double
    var capture: CaptureSpec
    var globalRect: CodableRect
    var pixelWidth: Int
    var pixelHeight: Int
    var deliveredFPS: Double
    var eventCount: Int
    var videoFile = "capture.mov"
    var eventFile = "events.atrevents"
    var trimStart: Double = 0
    var trimEnd: Double?
    var folderID: UUID?
    var thumbnailFile: String?
    var excludedKeyCodes: Set<UInt16>?

    var isStructurallyValid: Bool {
        let rect = globalRect.cgRect
        let end = trimEnd ?? duration
        let safeFileNames = [videoFile, eventFile] + (thumbnailFile.map { [$0] } ?? [])
        return (1...2).contains(schemaVersion)
            && !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && duration.isFinite && duration >= 0
            && capture.requestedFPS.isFinite && capture.requestedFPS > 0 && capture.requestedFPS <= 1_000
            && rect.origin.x.isFinite && rect.origin.y.isFinite && rect.width.isFinite && rect.height.isFinite
            && pixelWidth > 0 && pixelHeight > 0 && pixelWidth <= 32_768 && pixelHeight <= 32_768
            && deliveredFPS.isFinite && deliveredFPS >= 0 && deliveredFPS <= 1_000
            && eventCount >= 0
            && trimStart.isFinite && trimStart >= 0 && trimStart <= duration
            && end.isFinite && end >= trimStart && end <= duration
            && safeFileNames.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("/") && !$0.contains("\\") && !$0.contains(":") && !$0.contains("\0") }
    }
}

struct RecordingItem: Identifiable, Hashable, Sendable {
    var manifest: RecordingManifest
    var directory: URL
    var id: UUID { manifest.id }
}

struct RecordingFolder: Codable, Hashable, Identifiable, Sendable {
    var id: UUID
    var name: String
    var createdAt: Date
}

struct ActionRestrictions: Codable, Hashable, Sendable {
    var blockedKeyCodes: Set<UInt16> = []
    var blockedMouseButtons: Set<UInt8> = []

    func allowsKey(_ code: UInt16) -> Bool { !blockedKeyCodes.contains(code) }
    func allowsButton(_ button: UInt8) -> Bool { !blockedMouseButtons.contains(button) }
    func allowsModifier(_ index: Int) -> Bool {
        let equivalentCodes: [[UInt16]] = [[56, 60], [59, 62], [58, 61], [55, 54]]
        guard equivalentCodes.indices.contains(index) else { return true }
        return blockedKeyCodes.isDisjoint(with: equivalentCodes[index])
    }
}

/// Micro-averaged classification quality for one collection of binary control
/// outputs. Raw counts are retained so reports remain auditable and can be
/// recombined without rounding loss.
struct BinaryValidationMetrics: Codable, Hashable, Sendable {
    var truePositives: Int
    var falsePositives: Int
    var falseNegatives: Int
    var trueNegatives: Int

    var precision: Double {
        Double(truePositives) / Double(max(1, truePositives + falsePositives))
    }
    var recall: Double {
        Double(truePositives) / Double(max(1, truePositives + falseNegatives))
    }
    var f1: Double {
        let denominator = 2 * truePositives + falsePositives + falseNegatives
        return Double(2 * truePositives) / Double(max(1, denominator))
    }
    var falsePositiveRate: Double {
        Double(falsePositives) / Double(max(1, falsePositives + trueNegatives))
    }
    var positiveSupport: Int { truePositives + falseNegatives }
    var negativeSupport: Int { falsePositives + trueNegatives }
}

/// Held-out quality is deliberately multi-dimensional. A single weighted loss
/// can improve while a sparse keyboard head collapses or an idle policy starts
/// emitting false actions; these fields make those failure modes visible.
struct ValidationReport: Codable, Hashable, Sendable {
    var sampleCount: Int
    var binary: BinaryValidationMetrics?
    var buttons: BinaryValidationMetrics?
    var keyboard: BinaryValidationMetrics?
    var modifiers: BinaryValidationMetrics?
    var absoluteMouseMAE: Double?
    var activeRelativeMouseMAE: Double?
    var activeScrollMAE: Double?
    var idleContinuousFalseActionRate: Double?

    /// Aggregate weighted loss remains the primary selector, but it may hide a
    /// collapsed sparse head. On the same fixed held-out rows, reject only large
    /// regressions with enough support to be meaningful; small tradeoffs remain
    /// eligible and visible metrics still guide the user.
    func hasSevereBinaryRegression(comparedTo baseline: ValidationReport) -> Bool {
        let pairs = [
            (buttons, baseline.buttons),
            (keyboard, baseline.keyboard),
            (modifiers, baseline.modifiers)
        ]
        return pairs.contains { current, previous in
            guard let current, let previous else { return false }
            let f1Collapsed = previous.positiveSupport >= 5
                && current.f1 + 0.10 < previous.f1
            let falsePositivesSpiked = previous.negativeSupport >= 20
                && current.falsePositiveRate > previous.falsePositiveRate + 0.03
            return f1Collapsed || falsePositivesSpiked
        }
    }
}

struct ModelVersionManifest: Codable, Hashable, Identifiable, Sendable {
    var schemaVersion = ModelContract.schemaVersion
    var id: UUID
    var name: String
    var createdAt: Date
    var globalStep: Int
    var trainingLoss: Double
    var validationLoss: Double?
    var preprocessing: PreprocessingSpec
    var channels: ActionChannels
    var training: TrainingConfiguration
    var weightsFile = "weights.safetensors"
    var optimizerFile: String?
    var trainingStateFile: String?
    var randomStateFile: String?
    var epoch: Int?
    var isAutosave: Bool?
    /// The keyboard capability learned by this immutable brain. Runtime output
    /// is intersected with this set, so a model can never emit an unseen key.
    /// Optional keeps versions created before this invariant decodable.
    var demonstratedKeyCodes: Set<UInt16>? = nil
    /// Optional so schema-1 manifests remain decodable long enough to be
    /// identified and archived by the compatibility migration.
    var relativeMouseScale: Float? = nil
    /// Optional keeps existing runnable brains compatible. New training writes
    /// the dataset/target contract that produced the brain.
    var trainingDataSchema: Int? = nil
    /// Cumulative optimizer wall time represented by this immutable brain.
    /// Optional keeps every version created before timing metrics decodable.
    var trainingDurationSeconds: Double? = nil
    /// Demonstration-time consumed by optimizer batches. A model can process
    /// many hours of examples in a much shorter amount of wall time.
    var experienceDurationSeconds: Double? = nil
    /// Cursor visibility is part of the visual distribution even though it
    /// does not change tensor shape. Runtime reproduces the majority setting
    /// from the immutable training dataset instead of inheriting a Record-tab
    /// toggle that may have changed later.
    var trainingShowsCursor: Bool? = nil
    /// Auto mouse mode is derived from the exact recordings used to train this
    /// immutable brain. Editing a profile's recording selection later cannot
    /// silently switch a game-camera policy into absolute cursor control.
    var recommendedMouseMode: MouseControlMode? = nil
    /// Optional keeps existing manifests decodable. New brains retain the
    /// per-head held-out report that justified best-brain selection.
    var validationReport: ValidationReport? = nil
}

struct TrainingProgressSummary: Codable, Hashable, Sendable {
    var globalStep: Int
    var epoch: Int
    var updatedAt: Date
    var savedBrainCount: Int
    /// Optional for backward-compatible decoding of existing profile.json files.
    var trainingDurationSeconds: Double? = nil
    var experienceDurationSeconds: Double? = nil
}

struct AIProfile: Codable, Hashable, Identifiable, Sendable {
    var schemaVersion = 1
    var id: UUID
    var name: String
    var createdAt: Date
    var preprocessing: PreprocessingSpec
    var channels: ActionChannels
    var training: TrainingConfiguration
    var recordingIDs: [UUID]
    var activeVersionID: UUID?
    var recordingFolderIDs: [UUID]?
    var restrictions: ActionRestrictions?
    /// A cheap list-row summary. It avoids scanning every autosave manifest just
    /// to show how much an AI has trained.
    var trainingProgress: TrainingProgressSummary?
    /// Sticky once Crystal V4 is discovered, so renaming the profile cannot
    /// accidentally remove its user-requested protection.
    var deletionProtected: Bool?

    static func fresh(name: String = "New Agent") -> AIProfile {
        AIProfile(id: UUID(), name: name, createdAt: Date(), preprocessing: PreprocessingSpec(), channels: ActionChannels(), training: TrainingConfiguration(), recordingIDs: [], activeVersionID: nil, recordingFolderIDs: [], restrictions: ActionRestrictions(), trainingProgress: nil, deletionProtected: isProtectedModelName(name))
    }


    var effectiveFolderIDs: [UUID] { recordingFolderIDs ?? [] }
    var effectiveRestrictions: ActionRestrictions { restrictions ?? ActionRestrictions() }
    var isDeletionProtected: Bool {
        deletionProtected == true || Self.isProtectedModelName(name)
    }

    /// User-designated preservation boundary. Exact matching keeps ordinary
    /// Crystal V4 duplicates editable while protecting the two original brains.
    static func isProtectedModelName(_ value: String) -> Bool {
        let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return ["Crystal V4", "Crystal V4 Fine-tuned + glass"].contains {
            normalized.localizedCaseInsensitiveCompare($0) == .orderedSame
        }
    }

    /// Values that define how learned tensors are interpreted. Editing any of
    /// these requires a new brain instead of attaching incompatible weights to
    /// a differently shaped or differently sampled policy.
    var learnedBrainContract: LearnedBrainContract {
        LearnedBrainContract(preprocessing: preprocessing, architecture: training.architecture)
    }

    /// Uses exact persisted counters when available. Older profiles estimate
    /// consumed samples from optimizer steps and batch size. That estimate is
    /// stable even when recordings or folder selections change after training.
    func trainingDurationSummary(recordings _: [RecordingItem]) -> TrainingDurationSummary {
        let rawActual = trainingProgress?.trainingDurationSeconds ?? 0
        let actual = rawActual.isFinite ? max(0, rawActual) : 0
        if let exact = trainingProgress?.experienceDurationSeconds {
            let sanitized = exact.isFinite ? max(0, exact) : 0
            return TrainingDurationSummary(trainingSeconds: actual, experienceSeconds: sanitized, experienceIsEstimated: false)
        }
        let completedSteps = max(0, trainingProgress?.globalStep ?? 0)
        let consumedSamples = Double(completedSteps) * Double(max(1, training.batchSize))
        let actionFPS = training.actionFPS.isFinite ? max(0.0001, training.actionFPS) : 60
        return TrainingDurationSummary(
            trainingSeconds: actual,
            experienceSeconds: consumedSamples / actionFPS,
            experienceIsEstimated: completedSteps > 0
        )
    }
}

struct TrainingDurationSummary: Hashable, Sendable {
    var trainingSeconds: Double
    var experienceSeconds: Double
    var experienceIsEstimated: Bool
}

enum TrainingDurationFormatter {
    /// Training counters intentionally stay in hours until they reach a full
    /// day, matching the way long-running local training sessions are discussed.
    static func string(seconds rawSeconds: Double) -> String {
        let seconds = rawSeconds.isFinite ? max(0, rawSeconds) : 0
        let hours = seconds / 3_600
        if hours >= 24 {
            return formatted(hours / 24) + " days"
        }
        return formatted(hours) + " h"
    }

    private static func formatted(_ value: Double) -> String {
        value.formatted(.number.precision(.fractionLength(value < 10 ? 2 : 1)))
    }
}

struct LearnedBrainContract: Hashable, Sendable {
    var preprocessing: PreprocessingSpec
    var architecture: ArchitectureSpec
}

struct TrainingMetrics: Sendable {
    var epoch = 0
    var totalEpochs = 0
    var batch = 0
    var totalBatches = 0
    var globalStep = 0
    var totalSteps = 0
    var nextAutosaveStep: Int?
    var autosavesPublished = 0
    var trainingLoss = 0.0
    var epochTrainingLoss: Double?
    var validationLoss: Double?
    var validationReport: ValidationReport?
    var effectiveLearningRate = 0.0
    var learningRateScale = 1.0
    var samplesPerSecond = 0.0
    var elapsed = 0.0
    var experienceElapsed = 0.0
    var lossHistory: [Double] = []
    var epochLossHistory: [Double] = []
    var validationHistory: [Double] = []
    var learningRateHistory: [Double] = []
    var mlxActiveMemory = 0
    var mlxCacheMemory = 0
    var mlxPeakMemory = 0
}

struct RuntimeMetrics: Sendable {
    var perceptionFPS = 0.0
    var actionFPS = 0.0
    var latencyMilliseconds = 0.0
    var frameCount = 0
    var actionCount = 0
    var droppedFrames = 0
}

struct VisionPreviewFrame: Sendable {
    var packed: Data
    var spec: PreprocessingSpec
    var timestamp: Double
}

/// A bounded, CPU-renderable snapshot copied from an MLX diagnostic output.
/// Values use NHWC channel order after the singleton batch dimension is removed.
struct CNNFeatureTensor: Sendable {
    var width: Int
    var height: Int
    var channels: Int
    var values: [Float]
    var convolutionLayer: Int
    var kernelSize = 1
    var effectiveStride = 1
    var receptiveField = 1
}

struct CNNVisualizationFrame: Sendable {
    var packed: Data
    var spec: PreprocessingSpec
    var settings: CNNVisualizationSettings
    var tensors: [CNNFeatureTensor]
    var timestamp: Double
}

struct AgentSafetyPolicy: Codable, Hashable, Sendable {
    var stopOnHumanInput = true
    var stopOnFocusLoss = true
    var allowFullMac = false
    var controlRegion: CodableRect?
    var panicKeyCode: UInt16 = 53
    var panicModifiers: UInt64 = 0x1C0000
}

struct HotkeyBinding: Codable, Hashable, Sendable {
    var keyCode: UInt32
    var carbonModifiers: UInt32

    static let panic = HotkeyBinding(keyCode: 53, carbonModifiers: UInt32(1 << 12 | 1 << 11 | 1 << 8))
    static let record = HotkeyBinding(keyCode: 15, carbonModifiers: UInt32(1 << 12 | 1 << 11 | 1 << 8))
    static let run = HotkeyBinding(keyCode: 0, carbonModifiers: UInt32(1 << 12 | 1 << 11 | 1 << 8))
}

struct HotkeySettings: Codable, Hashable, Sendable {
    var panic = HotkeyBinding.panic
    var record = HotkeyBinding.record
    var run = HotkeyBinding.run
}

enum AgentTrainerError: LocalizedError {
    case invalidConfiguration(String)
    case permission(String)
    case capture(String)
    case storage(String)
    case model(String)
    case noData

    var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message), .permission(let message), .capture(let message), .storage(let message), .model(let message): message
        case .noData: "No compatible training data is available."
        }
    }
}

enum ModelSizing {
    static func parameterCount(_ profile: AIProfile) -> Int64 {
        let architecture = profile.training.architecture
        var total: Int64 = 0
        // Current color planes + temporal differences + explicit X/Y.
        var input = profile.preprocessing.channelCount * 2 + 2
        for i in architecture.convolutionChannels.indices {
            let output = max(1, architecture.convolutionChannels[i])
            let kernel = architecture.kernelSizes.indices.contains(i) ? max(1, architecture.kernelSizes[i]) : 3
            total = add(total, multiply(Int64(output), multiply(multiply(Int64(input), Int64(kernel)), Int64(kernel))))
            // Affine GroupNorm scale and bias.
            total = add(total, multiply(2, Int64(output)))
            input = output
        }
        let output = CNNGeometry.outputSize(width: profile.preprocessing.width, height: profile.preprocessing.height, architecture: architecture)
        let finalChannels = Int64(max(1, input))
        let visualProjectionInput: Int64
        switch architecture.effectiveVisualPooling {
        case .flattened:
            visualProjectionInput = multiply(multiply(Int64(output.width), Int64(output.height)), finalChannels)
        case .attention:
            let heads = Int64(architecture.effectiveAttentionHeads)
            // A learned score per head and spatial feature, including bias.
            total = add(total, multiply(add(finalChannels, 1), heads))
            // Each attention head retains pooled features plus exact X/Y, while
            // global mean and maximum features preserve whole-scene context.
            visualProjectionInput = add(multiply(heads, add(finalChannels, 2)), multiply(2, finalChannels))
        }
        total = add(total, multiply(add(visualProjectionInput, 1), Int64(max(1, architecture.visualEmbedding))))
        total = add(total, multiply(2, Int64(max(1, architecture.visualEmbedding))))
        let recurrent = max(1, architecture.recurrentWidth)
        let gates = architecture.recurrentKind == .gru ? 3 : 4
        total = add(total, multiply(multiply(Int64(gates), Int64(recurrent)), add(add(Int64(ActionLayout.count), Int64(recurrent)), 1)))
        // MLX's GRU has the usual three-gate bias plus a separate hidden-state
        // candidate bias (`bhn`) of one value per hidden unit. LSTM uses only
        // its four-gate bias, which is already included above.
        if architecture.recurrentKind == .gru { total = add(total, Int64(recurrent)) }
        var fusionInput = max(1, architecture.visualEmbedding) + recurrent
        for width in architecture.fusionWidths {
            total = add(total, multiply(add(Int64(fusionInput), 1), Int64(max(1, width))))
            total = add(total, multiply(2, Int64(max(1, width))))
            fusionInput = max(1, width)
        }
        total = add(total, multiply(add(Int64(fusionInput), 1), Int64(ActionLayout.count)))
        return total
    }

    /// Conservative peak budget for compiled forward/backward training. AdamW
    /// keeps two Float32 moments in addition to parameters and gradients, while
    /// activations and compiler temporaries multiply the nominal batch input.
    /// This is a safety bound, not a reported MLX memory measurement.
    static func estimatedTrainingWorkingSet(_ profile: AIProfile) -> Int64 {
        let parameters = multiply(parameterCount(profile), 24)
        let batch = multiply(NeuralInputSizing.summary(for: profile).nominalBytesPerTrainingBatch, 8)
        return add(512 * 1_024 * 1_024, add(parameters, batch))
    }

    private static func multiply(_ lhs: Int64, _ rhs: Int64) -> Int64 { let result = lhs.multipliedReportingOverflow(by: rhs); return result.overflow ? Int64.max : result.partialValue }
    private static func add(_ lhs: Int64, _ rhs: Int64) -> Int64 { let result = lhs.addingReportingOverflow(rhs); return result.overflow ? Int64.max : result.partialValue }
}

/// Exact input counts for one policy decision and one optimizer batch. This
/// mirrors `VisionPreprocessor`, `AgentPolicy`, and the fixed action-history
/// layout so the UI never has to approximate the model contract independently.
struct NeuralInputSummary: Hashable, Sendable {
    var pixelCount: Int64
    var lumaValues: Int64
    var chromaValuesPerPlane: Int64
    var packedVisionValues: Int64
    var expandedVisionValues: Int64
    var temporalDifferenceValues: Int64
    var coordinateValues: Int64
    var firstConvolutionValues: Int64
    var historySteps: Int64
    var actionValuesPerHistoryStep: Int64
    var historyValues: Int64
    var historyDurationSeconds: Double
    var valuesPerDecision: Int64
    var runtimeValuesPerSecond: Int64
    var packedVisionBytesPerSecond: Int64
    var batchSize: Int64
    var valuesPerTrainingBatch: Int64
    var quantizationLevels: Int64
    var effectivePackedBits: Int64
    var bytesPerModelValue: Int64
    var nominalBytesPerDecision: Int64
    var nominalBytesPerTrainingBatch: Int64
}

enum NeuralInputCapacityLevel: Hashable, Sendable {
    case comfortable
    case balanced
    case high
    case tooHigh
}

/// A deliberately simple comparison for the model editor. It is a usability
/// guide, not a mathematical validity limit: convolutional weight sharing means
/// image values and learned parameters are not independent one-to-one features.
struct NeuralInputCapacityGuide: Hashable, Sendable {
    var level: NeuralInputCapacityLevel
    var inputValues: Int64
    var parameterCount: Int64
    var inputsPerParameter: Double
}

enum NeuralInputSizing {
    static func capacityGuide(for profile: AIProfile) -> NeuralInputCapacityGuide {
        let inputValues = summary(for: profile).valuesPerDecision
        let parameterCount = max(1, ModelSizing.parameterCount(profile))
        return capacityGuide(inputValues: inputValues, parameterCount: parameterCount)
    }

    static func capacityGuide(inputValues rawInputValues: Int64, parameterCount rawParameterCount: Int64) -> NeuralInputCapacityGuide {
        let inputValues = max(0, rawInputValues)
        let parameterCount = max(1, rawParameterCount)
        let ratio = Double(inputValues) / Double(parameterCount)
        let level: NeuralInputCapacityLevel
        if ratio <= 0.75 { level = .comfortable }
        else if ratio <= 2 { level = .balanced }
        else if ratio <= 5 { level = .high }
        else { level = .tooHigh }
        return NeuralInputCapacityGuide(level: level, inputValues: inputValues, parameterCount: parameterCount, inputsPerParameter: ratio)
    }

    static func summary(for profile: AIProfile) -> NeuralInputSummary {
        let spec = profile.preprocessing
        let width = max(0, Int64(spec.width))
        let height = max(0, Int64(spec.height))
        let pixels = multiply(width, height)
        let luma = pixels

        let chromaPerPlane: Int64
        if spec.colorMode == .grayscale {
            chromaPerPlane = 0
        } else {
            let chromaWidth = spec.chroma == .yuv444 ? width : width / 2 + width % 2
            let chromaHeight = spec.chroma == .yuv420 ? height / 2 + height % 2 : height
            chromaPerPlane = multiply(chromaWidth, chromaHeight)
        }

        let packedVision = add(luma, multiply(2, chromaPerPlane))
        let denseChannels: Int64 = spec.colorMode == .grayscale ? 1 : 3
        let expandedVision = multiply(pixels, denseChannels)
        let coordinates = multiply(pixels, 2)
        let temporalDifference = expandedVision
        let firstConvolution = add(add(expandedVision, temporalDifference), coordinates)

        // Dataset and runtime tensors deliberately retain one zero row when
        // history is disabled so recurrent input always has a valid shape.
        let historySteps = max(1, Int64(profile.training.historyLength))
        let actionValues = Int64(ActionLayout.count)
        let history = multiply(historySteps, actionValues)
        let perDecision = add(firstConvolution, history)
        let actionFPS = profile.training.actionFPS.isFinite ? max(0.0001, profile.training.actionFPS) : 60
        let historyDuration = profile.training.historyLength > 0 ? Double(profile.training.historyLength) / actionFPS : 0
        let perceptionFPS = profile.training.perceptionFPS.isFinite ? max(0, profile.training.perceptionFPS) : 0
        let runtimeValuesPerSecond = rate(perDecision, fps: perceptionFPS)
        let packedVisionBytesPerSecond = rate(packedVision, fps: perceptionFPS)
        let batchSize = max(1, Int64(profile.training.batchSize))
        let perBatch = multiply(perDecision, batchSize)

        let effectiveBitDepth = min(8, max(1, spec.bitDepth))
        let levels = Int64(1 << effectiveBitDepth)
        let meaningfulBits = multiply(packedVision, Int64(effectiveBitDepth))
        let scalarBytes: Int64 = profile.training.precision == .float32 ? 4 : 2

        return NeuralInputSummary(
            pixelCount: pixels,
            lumaValues: luma,
            chromaValuesPerPlane: chromaPerPlane,
            packedVisionValues: packedVision,
            expandedVisionValues: expandedVision,
            temporalDifferenceValues: temporalDifference,
            coordinateValues: coordinates,
            firstConvolutionValues: firstConvolution,
            historySteps: historySteps,
            actionValuesPerHistoryStep: actionValues,
            historyValues: history,
            historyDurationSeconds: historyDuration,
            valuesPerDecision: perDecision,
            runtimeValuesPerSecond: runtimeValuesPerSecond,
            packedVisionBytesPerSecond: packedVisionBytesPerSecond,
            batchSize: batchSize,
            valuesPerTrainingBatch: perBatch,
            quantizationLevels: levels,
            effectivePackedBits: meaningfulBits,
            bytesPerModelValue: scalarBytes,
            nominalBytesPerDecision: multiply(perDecision, scalarBytes),
            nominalBytesPerTrainingBatch: multiply(perBatch, scalarBytes)
        )
    }

    private static func multiply(_ lhs: Int64, _ rhs: Int64) -> Int64 {
        let result = lhs.multipliedReportingOverflow(by: rhs)
        return result.overflow ? Int64.max : result.partialValue
    }

    private static func add(_ lhs: Int64, _ rhs: Int64) -> Int64 {
        let result = lhs.addingReportingOverflow(rhs)
        return result.overflow ? Int64.max : result.partialValue
    }

    private static func rate(_ count: Int64, fps: Double) -> Int64 {
        let value = Double(count) * fps
        guard value.isFinite, value < Double(Int64.max) else { return Int64.max }
        return Int64(max(0, value).rounded())
    }
}
