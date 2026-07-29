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
    /// Version 9 removes teacher-forced action tokens from the policy and
    /// makes the Transformer readout explicitly perception-backed. Policy
    /// v7 could obtain excellent held-out scores by copying the demonstrated
    /// previous action while producing a constant output from perception. Live
    /// inference starts from released controls, making that shortcut an
    /// all-idle absorbing state. An early v8 readout could still settle on its
    /// learned constant token for some initializations. V9 fuses that token
    /// with the parameter-free transformed-visual mean and first visual-token
    /// anchor, so every action head receives a short perception path. Visual memory
    /// provides temporal context and held controls remain stateful in the
    /// injector.
    static let schemaVersion = 9
    static let weightFormat = "AgentTrainer.Policy.v9"
}

/// Policy v9 keeps one zero row at the MLX compatibility boundary, but no
/// demonstrated or predicted action is fed back into the network. Visual
/// memory is the only temporal model input.
enum PolicyInputContract {
    static let actionHistoryLength = 0
    static let placeholderHistoryRows = 1
}

/// Fixed, bounded semantics for visual memory in Policy v9. Past perceptions
/// are sampled at lags `1, 1 + stride, 1 + 2*stride, ...`, so the immediately
/// preceding frame is always present while a few additional slots cover a
/// longer interval. Signed differences and per-slot availability are fused by
/// one pointwise projection before either spatial architecture; Transformer
/// token count and attention cost therefore do not grow with memory length.
enum VisualMemoryContract {
    static let defaultFrameCount = 4
    static let defaultStride = 4
    static let defaultDropout = 0.10
    static let maximumFrameCount = 8
    static let maximumStride = 8
    static let fusionChannels = 8

    static func lags(frameCount: Int, stride: Int) -> [Int] {
        let frames = min(maximumFrameCount, max(0, frameCount))
        let spacing = min(maximumStride, max(1, stride))
        return (0..<frames).map { 1 + $0 * spacing }
    }

    static func maximumLag(frameCount: Int, stride: Int) -> Int {
        lags(frameCount: frameCount, stride: stride).last ?? 0
    }

    static func rawInputChannels(colorChannels: Int, frameCount: Int) -> Int {
        let channels = max(1, colorChannels)
        let frames = min(maximumFrameCount, max(0, frameCount))
        // Current pixels + one signed difference per remembered frame + one
        // availability plane per slot.
        return channels + frames * channels + frames
    }

    static func spatialEncoderInputChannels(colorChannels: Int, frameCount: Int) -> Int {
        max(1, colorChannels) + (frameCount > 0 ? fusionChannels : 0) + 2
    }
}

/// Version of the causal pairing between a captured frame and the controls the
/// model should perform next. This remains separate from the weight format so
/// a data-only correction can invalidate caches/checkpoints without needlessly
/// changing tensor shapes. Policy v9 is tracked separately as an intentional
/// learned-weight break while retaining the same causal dataset contract.
enum TrainingDataContract {
    /// Version 7 preserves sub-tick key/button taps and preceding-perception
    /// motion, and assigns Shift to the Keyboard channel while the Modifiers
    /// channel owns only Control, Option, and Command.
    static let schemaVersion = 7
}

/// Version of optimizer/loss semantics that do not change cached pixels or
/// model tensor shapes. Bumping this restarts incompatible optimizer state
/// while preserving the selected runnable weights for safe fine-tuning.
enum TrainingObjectiveContract {
    /// Version 4 retains objective-v3 binary balancing and applies the same
    /// dataset-level correction to transient relative-mouse and scroll values.
    /// Continuous Smooth L1 is normalized by its beta so small but executable
    /// camera deltas cannot disappear beneath an easy centered-cursor loss.
    static let schemaVersion = 4
}

/// Stable training/runtime contract for locked-cursor game cameras. Raw HID
/// deltas are divided by this value in the dataset and multiplied by the same
/// value during execution, so the model learns useful values independent of the
/// capture resolution.
enum GameCameraContract {
    static let deltaScale: Float = 80
    static let maximumPostedDelta: CGFloat = 10_000
    /// Legacy reports fall back to the conservative measured value. New brains
    /// calibrate the smallest candidate that retains motion while satisfying
    /// the held-out idle ceiling, then persist it in their immutable report.
    static let defaultMinimumPostedMagnitude: CGFloat = 1.5
    static let calibrationMagnitudes: [CGFloat] = [
        0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5
    ]

    static func trainingValue(forRawDelta delta: Double) -> Float {
        Swift.min(1, Swift.max(-1, Float(delta) / deltaScale))
    }

    static func runtimeDelta(forPrediction prediction: Float, sensitivity: Double) -> CGFloat {
        guard prediction.isFinite else { return 0 }
        let safeSensitivity = sensitivity.isFinite ? min(100, max(0.01, sensitivity)) : 1
        let value = CGFloat(prediction) * CGFloat(deltaScale) * safeSensitivity
        return min(maximumPostedDelta, max(-maximumPostedDelta, value))
    }

    static func postedDelta(
        forPrediction prediction: Float,
        sensitivity: Double,
        minimumMagnitude: CGFloat = defaultMinimumPostedMagnitude
    ) -> Int64 {
        let value = runtimeDelta(forPrediction: prediction, sensitivity: sensitivity)
        let finiteMinimum = minimumMagnitude.isFinite
            ? min(4.5, max(0.5, minimumMagnitude))
            : defaultMinimumPostedMagnitude
        guard abs(value) >= finiteMinimum else { return 0 }
        return Int64(value.rounded())
    }

    static func calibratedMinimumPostedMagnitude(
        activeCorrectCounts: [Int],
        activeCount: Int,
        idleFalseCounts: [Int],
        idleCount: Int
    ) -> CGFloat {
        guard activeCount >= ValidationExecutionContract.minimumSupportedPositiveSamples,
              activeCorrectCounts.count == calibrationMagnitudes.count,
              idleFalseCounts.count == calibrationMagnitudes.count else {
            return defaultMinimumPostedMagnitude
        }
        var firstIdleSafe: CGFloat?
        for index in calibrationMagnitudes.indices {
            let recall = Double(activeCorrectCounts[index]) / Double(max(1, activeCount))
            let idleRate = Double(idleFalseCounts[index]) / Double(max(1, idleCount))
            if idleRate <= ValidationExecutionContract.maximumIdleContinuousFalseActionRate {
                firstIdleSafe = firstIdleSafe ?? calibrationMagnitudes[index]
                if recall >= ValidationExecutionContract.minimumContinuousExecutionRecall {
                    return calibrationMagnitudes[index]
                }
            }
        }
        // If no candidate satisfies both constraints, retain the smallest stable
        // one so the report fails on missing motion instead of deploying known
        // idle jitter. If none is stable, use the strongest tested deadzone and
        // let the idle gate reject it.
        return firstIdleSafe ?? calibrationMagnitudes.last
            ?? defaultMinimumPostedMagnitude
    }

    static func minimumPostedMagnitude(from report: ValidationReport?) -> CGFloat {
        guard let value = report?.relativeMouseExecutionDeadzone,
              value.isFinite else {
            return defaultMinimumPostedMagnitude
        }
        return min(4.5, max(0.5, CGFloat(value)))
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
    /// Legacy Policy-v4 settings retained only for profile decoding and model
    /// artifact archival. Current policies never instantiate this projection.
    case flattened = "Flattened Grid (Legacy)"
    /// Legacy Policy-v4 coordinate-aware pooling setting. The modern Hybrid
    /// family evolves these keypoints into individual Transformer tokens.
    case attention = "Attention Keypoints"

    var id: String { rawValue }
}

enum PolicyArchitectureFamily: String, Codable, CaseIterable, Identifiable, Sendable {
    case hybrid = "Hybrid CNN + Transformer"
    case pureTransformer = "Pure Transformer (ViT)"

    var id: String { rawValue }
    var shortName: String { self == .hybrid ? "Hybrid" : "Pure Transformer" }
}

struct ArchitectureSpec: Codable, Hashable, Sendable {
    /// Optional keeps Policy-v5 profiles decodable at the migration boundary.
    /// A missing value is the v5 hybrid, never an ambiguous automatic choice.
    var family: PolicyArchitectureFamily? = .hybrid
    /// Non-overlapping direct image patches for the Pure Transformer path.
    /// Hybrid ignores this field, so tuning it cannot invalidate Hybrid brains.
    var patchSize: Int? = 32
    /// The stride-four stem cuts the dominant high-resolution convolution cost,
    /// while the extra stage expands the receptive field before spatial
    /// features become compact tokens. This is both faster and substantially
    /// more expressive than three stride-two layers followed by a global mean.
    var convolutionChannels: [Int] = [32, 64, 96, 128]
    var kernelSizes: [Int] = [7, 3, 3, 3]
    var strides: [Int] = [4, 2, 2, 2]
    /// Shared width of visual, global, history, and readout tokens.
    var visualEmbedding = 256
    /// Retained only so pre-v5 profile JSON remains decodable long enough for
    /// its model artifacts to be archived and its settings moved to a current
    /// Hybrid preset. Current policies do not instantiate a recurrent module.
    var recurrentKind: RecurrentKind = .gru
    var recurrentWidth = 192
    var fusionWidths: [Int] = [384, 256]
    var dropout: Double = 0.1

    /// Policy-v4 compatibility fields. They remain Codable so migration can
    /// inspect old profiles, but current policies ignore them.
    var visualPooling: VisualPoolingKind? = nil
    var attentionHeads: Int? = nil

    /// Learned coordinate-bearing visual tokens. When decoding a Policy-v4
    /// profile, its old attention-keypoint count is used as the migration
    /// fallback until the profile is rewritten to an explicit Hybrid preset.
    var spatialTokens: Int? = 8

    /// Optional fields keep Policy-v4 profile JSON decodable. Missing values
    /// select the balanced Hybrid Transformer. The model-contract migration
    /// rewrites profiles without a compatible current brain to a tuned preset.
    var transformerLayers: Int? = 3
    var transformerHeads: Int? = 8
    var transformerFeedForward: Int? = 768

    var effectiveSpatialTokens: Int { min(64, max(1, spatialTokens ?? attentionHeads ?? 8)) }
    var effectiveFamily: PolicyArchitectureFamily { family ?? .hybrid }
    var effectivePatchSize: Int { min(64, max(8, patchSize ?? 32)) }
    var effectiveTransformerLayers: Int { min(8, max(1, transformerLayers ?? 3)) }
    var effectiveTransformerHeads: Int { min(32, max(1, transformerHeads ?? 8)) }
    var effectiveTransformerFeedForward: Int { min(16_384, max(1, transformerFeedForward ?? 768)) }

    static let hybridSmall = ArchitectureSpec(
        convolutionChannels: [24, 48, 72, 96],
        visualEmbedding: 192,
        recurrentWidth: 128,
        fusionWidths: [256, 192],
        dropout: 0.08,
        spatialTokens: 6,
        transformerLayers: 2,
        transformerHeads: 6,
        transformerFeedForward: 512
    )
    static let hybridBalanced = ArchitectureSpec()
    static let hybridLarge = ArchitectureSpec(
        convolutionChannels: [48, 96, 160, 224],
        visualEmbedding: 384,
        recurrentWidth: 256,
        fusionWidths: [512, 384],
        dropout: 0.12,
        spatialTokens: 12,
        transformerLayers: 4,
        transformerHeads: 12,
        transformerFeedForward: 1_152
    )

    /// Pure Small/Balanced use 32-pixel patches. At the default 640×360 vision
    /// this produces 240 visual tokens: useful full-patch attention while still
    /// practical for live inference. Large moves to 24 for additional detail.
    static let pureSmall = ArchitectureSpec(
        family: .pureTransformer,
        patchSize: 32,
        convolutionChannels: [],
        kernelSizes: [],
        strides: [],
        visualEmbedding: 192,
        recurrentWidth: 128,
        fusionWidths: [256, 192],
        dropout: 0.08,
        spatialTokens: nil,
        transformerLayers: 3,
        transformerHeads: 6,
        transformerFeedForward: 512
    )
    static let pureBalanced = ArchitectureSpec(
        family: .pureTransformer,
        patchSize: 32,
        convolutionChannels: [],
        kernelSizes: [],
        strides: [],
        visualEmbedding: 256,
        recurrentWidth: 192,
        fusionWidths: [384, 256],
        dropout: 0.1,
        spatialTokens: nil,
        transformerLayers: 4,
        transformerHeads: 8,
        transformerFeedForward: 768
    )
    static let pureLarge = ArchitectureSpec(
        family: .pureTransformer,
        patchSize: 24,
        convolutionChannels: [],
        kernelSizes: [],
        strides: [],
        visualEmbedding: 384,
        recurrentWidth: 256,
        fusionWidths: [512, 384],
        dropout: 0.12,
        spatialTokens: nil,
        transformerLayers: 6,
        transformerHeads: 12,
        transformerFeedForward: 1_152
    )

    // Source-compatible aliases remain the tuned Hybrid presets.
    static let small = hybridSmall
    static let balanced = hybridBalanced
    static let large = hybridLarge

    static func preset(family: PolicyArchitectureFamily, scale: Int) -> ArchitectureSpec {
        switch (family, min(2, max(0, scale))) {
        case (.hybrid, 0): hybridSmall
        case (.hybrid, 1): hybridBalanced
        case (.hybrid, _): hybridLarge
        case (.pureTransformer, 0): pureSmall
        case (.pureTransformer, 1): pureBalanced
        case (.pureTransformer, _): pureLarge
        }
    }
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
        if architecture.effectiveFamily == .pureTransformer {
            let patch = architecture.effectivePatchSize
            return CNNLayerGeometry(kernelSize: patch, effectiveStride: patch, receptiveField: patch)
        }
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
        if architecture.effectiveFamily == .pureTransformer {
            let patch = architecture.effectivePatchSize
            return (
                max(1, (max(1, width) + patch - 1) / patch),
                max(1, (max(1, height) + patch - 1) / patch)
            )
        }
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

/// Fixed-shape token geometry for both Policy-v9 families. Hybrid attention is
/// compact; Pure Transformer attention covers every direct image patch.
enum PolicyTokenGeometry {
    static let globalTokenCount = 2
    static let readoutTokenCount = 1
    static let tokenTypeCount = 3

    static func patchGrid(_ profile: AIProfile) -> (width: Int, height: Int) {
        let patch = profile.training.architecture.effectivePatchSize
        return (
            max(1, (max(1, profile.preprocessing.width) + patch - 1) / patch),
            max(1, (max(1, profile.preprocessing.height) + patch - 1) / patch)
        )
    }

    static func visualTokenCount(_ profile: AIProfile) -> Int {
        let architecture = profile.training.architecture
        if architecture.effectiveFamily == .hybrid {
            return architecture.effectiveSpatialTokens + globalTokenCount
        }
        let grid = patchGrid(profile)
        let product = grid.width.multipliedReportingOverflow(by: grid.height)
        return product.overflow ? Int.max : product.partialValue
    }

    static func sequenceLength(_ profile: AIProfile) -> Int {
        let base = readoutTokenCount.addingReportingOverflow(visualTokenCount(profile))
        return base.overflow ? Int.max : base.partialValue
    }

    static func attentionPairsPerLayer(_ profile: AIProfile) -> Int64 {
        let length = Int64(sequenceLength(profile))
        let result = length.multipliedReportingOverflow(by: length)
        return result.overflow ? Int64.max : result.partialValue
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

/// Run-only policy inspection modes. They never participate in the
/// learned-brain contract, dataset identity, or saved model weights.
enum CNNVisualizationMode: String, Codable, CaseIterable, Identifiable, Sendable {
    case activationOverlay = "Activation Overlay"
    case featureChannels = "Feature Channels"
    case spatialTokens = "Spatial Tokens"
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

/// Presentation-only controls for live policy inspection. Sanitizing at the
/// runtime boundary prevents malformed persisted values from creating an
/// unbounded diagnostic rate or renderer workload.
struct CNNVisualizationSettings: Codable, Hashable, Sendable {
    var enabled = false
    var mode: CNNVisualizationMode = .activationOverlay
    var framesPerSecond = 4.0
    /// Zero-based convolution index. `-1` follows the final convolution even
    /// when the selected AI uses a different number of layers.
    var convolutionLayer = -1
    var featureChannelCount = 8
    var overlayOpacity = 0.68
    var actionFocus: CNNActionFocus = .movement

    func sanitized(layerCount: Int? = nil) -> Self {
        var value = self
        value.framesPerSecond = framesPerSecond.isFinite ? min(15, max(0.5, framesPerSecond)) : 4
        let requestedMaps = min(16, max(4, featureChannelCount))
        value.featureChannelCount = [4, 6, 8, 12, 16].min {
            abs($0 - requestedMaps) < abs($1 - requestedMaps)
        } ?? 8
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
    /// Retained only to decode profiles and immutable manifests from Policy v7.
    /// Policy v9 never consumes demonstrated action history: it is not
    /// deployment-aligned and can turn released runtime state into an idle
    /// feedback loop. Model-contract migration rewrites this value to zero.
    var historyLength = 0
    /// Optional fields keep profiles from older model contracts decodable.
    /// Their effective values intentionally opt into the current defaults;
    /// incompatible weights are archived at the separate model boundary.
    var visualMemoryFrames: Int? = VisualMemoryContract.defaultFrameCount
    var visualMemoryStride: Int? = VisualMemoryContract.defaultStride
    /// Training-only structured dropout for older visual-memory slots. The
    /// immediate predecessor remains present, and inference never drops memory.
    var visualMemoryDropout: Double? = VisualMemoryContract.defaultDropout
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
    var effectiveVisualMemoryFrames: Int {
        min(VisualMemoryContract.maximumFrameCount, max(0, visualMemoryFrames ?? VisualMemoryContract.defaultFrameCount))
    }
    var effectiveVisualMemoryStride: Int {
        min(VisualMemoryContract.maximumStride, max(1, visualMemoryStride ?? VisualMemoryContract.defaultStride))
    }
    var effectiveVisualMemoryDropout: Double {
        let value = visualMemoryDropout ?? VisualMemoryContract.defaultDropout
        return value.isFinite ? min(0.5, max(0, value)) : VisualMemoryContract.defaultDropout
    }
    var visualMemoryLags: [Int] {
        VisualMemoryContract.lags(
            frameCount: effectiveVisualMemoryFrames,
            stride: effectiveVisualMemoryStride
        )
    }
    var visualMemoryMaximumLag: Int {
        visualMemoryLags.last ?? 0
    }
    /// Spacing changes no sampled lag when memory is disabled or contains only
    /// the mandatory immediate predecessor. Normalize it out of learned-brain
    /// compatibility so an irrelevant editor value never archives good weights.
    var learnedVisualMemoryStride: Int {
        effectiveVisualMemoryFrames > 1 ? effectiveVisualMemoryStride : 0
    }
    var visualMemoryDurationSeconds: Double {
        guard perceptionFPS.isFinite, perceptionFPS > 0 else { return 0 }
        return Double(visualMemoryMaximumLag) / perceptionFPS
    }
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
            && duration.isFinite && duration > 0
            && capture.requestedFPS.isFinite && capture.requestedFPS > 0 && capture.requestedFPS <= 1_000
            && rect.origin.x.isFinite && rect.origin.y.isFinite
            && rect.width.isFinite && rect.width > 0 && rect.height.isFinite && rect.height > 0
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

/// Weighted objective values for each enabled action family. The optimizer
/// still minimizes their equal-weight aggregate, while retaining the parts
/// prevents an easy mouse head from hiding a struggling keyboard head.
struct ValidationLossBreakdown: Codable, Hashable, Sendable {
    var mouse: Double?
    var buttons: Double?
    var scroll: Double?
    var keyboard: Double?
    var modifiers: Double?
}

/// Auditable evidence and final positive weight for one binary control.
/// Repeated taps and sustained active duration are both intentional evidence;
/// release counts make the reported class correction match transition loss.
struct BinaryOutputBalance: Codable, Hashable, Identifiable, Sendable {
    var outputIndex: Int
    var positiveSamples: Int
    var pressEpisodes: Int
    /// Optional so objective-v2 reports remain decodable.
    var releaseEpisodes: Int? = nil
    /// Optional so objective-v2 reports remain decodable.
    var activeDurationSeconds: Double? = nil
    var positiveWeight: Double
    var isSupported: Bool
    var id: Int { outputIndex }
}

/// Dataset-level correction for one signed transient output. Active camera and
/// scroll rows are usually much rarer than zero rows, while their normalized
/// magnitudes are intentionally small. Persisting this evidence makes it clear
/// whether a zero-motion head had useful targets and how strongly they were
/// represented in the objective.
struct ContinuousOutputBalance: Codable, Hashable, Identifiable, Sendable {
    var outputIndex: Int
    var activeSamples: Int
    var meanActiveMagnitude: Double
    var activeWeight: Double
    var isSupported: Bool
    var id: Int { outputIndex }
}

struct TrainingBalanceReport: Codable, Hashable, Sendable {
    var outputs: [BinaryOutputBalance]
    /// Optional keeps objective-v2/v3 reports decodable.
    var continuousOutputs: [ContinuousOutputBalance]? = nil

    var supportedOutputs: [BinaryOutputBalance] { outputs.filter(\.isSupported) }
    var ignoredOutputs: [BinaryOutputBalance] { outputs.filter { !$0.isSupported && $0.positiveSamples > 0 } }
}

/// Per-control held-out behavior at both the conventional 0.5 threshold and
/// the precision-safe threshold selected for live execution.
struct BinaryOutputValidation: Codable, Hashable, Identifiable, Sendable {
    var outputIndex: Int
    var decisionThreshold: Double
    var defaultMetrics: BinaryValidationMetrics
    var calibratedMetrics: BinaryValidationMetrics
    var id: Int { outputIndex }
}

/// Identifies whether execution metrics came from genuinely unseen context or
/// from a representative calibration pass over rows that also trained the
/// policy. A zero validation split still needs deployment-aligned thresholds,
/// but in-sample measurements must never be presented as generalization quality
/// or drive best-brain selection.
enum ModelEvaluationScope: String, Codable, Hashable, Sendable {
    case heldOut
    case trainingCalibration

    var isHeldOut: Bool { self == .heldOut }
    var shortLabel: String {
        switch self {
        case .heldOut: "Held-out"
        case .trainingCalibration: "In-sample"
        }
    }
}

enum ValidationExecutionContract {
    static let minimumSupportedPositiveSamples = 8
    /// Require more than a single lucky threshold crossing, while allowing a
    /// brief visually ambiguous control (for example, a handbrake tap) to coexist
    /// with long-held movement keys in the same brain.
    static let minimumPerOutputTruePositives = 4
    static let minimumPerOutputRecall = 0.01
    static let minimumContinuousExecutionRecall = 0.05
    static let maximumIdleContinuousFalseActionRate = 0.10
}

/// Execution behavior is deliberately multi-dimensional. On held-out data, a
/// single weighted loss can improve while a sparse keyboard head collapses or an
/// idle policy starts emitting false actions; the same fields provide explicitly
/// in-sample runtime calibration when no held-out split exists.
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
    /// Pre-rounding Game Camera magnitude selected on active/idle evaluation
    /// axes. `evaluationScope` distinguishes held-out quality from the in-sample
    /// calibration used when the requested validation split is zero.
    var relativeMouseExecutionDeadzone: Double? = nil
    /// Fraction of evaluated active target axes that would produce at least one
    /// correctly directed executable unit at default runtime sensitivity.
    var activeRelativeMouseExecutionRecall: Double? = nil
    var activeScrollExecutionRecall: Double? = nil
    /// Optional fields keep Policy-v6 brains created before objective v2 fully
    /// decodable and runnable with the conservative 0.5 threshold.
    var lossBreakdown: ValidationLossBreakdown? = nil
    var binaryOutputs: [BinaryOutputValidation]? = nil
    var trainingBalance: TrainingBalanceReport? = nil
    /// Optional keeps all earlier manifests decodable; reports created before
    /// execution calibration existed were necessarily held-out reports.
    var evaluationScope: ModelEvaluationScope? = nil

    var effectiveEvaluationScope: ModelEvaluationScope {
        evaluationScope ?? .heldOut
    }

    func decisionThreshold(for outputIndex: Int) -> Float {
        guard let value = binaryOutputs?.first(where: { $0.outputIndex == outputIndex })?.decisionThreshold,
              value.isFinite else { return 0.5 }
        return Float(min(0.95, max(0.5, value)))
    }

    var calibratedThresholdCount: Int {
        binaryOutputs?.count { abs($0.decisionThreshold - 0.5) >= 0.005 } ?? 0
    }

    /// A supported evaluated head with enough positives but no true activation
    /// is the user-visible "press nothing" failure mode. Keep it explicit so the
    /// UI cannot present a decreasing aggregate loss as healthy progress.
    var hasBinaryRecallCollapse: Bool {
        if let binaryOutputs {
            return binaryOutputs.contains {
                $0.calibratedMetrics.positiveSupport
                    >= ValidationExecutionContract.minimumSupportedPositiveSamples
                    && (
                        $0.calibratedMetrics.truePositives
                            < ValidationExecutionContract.minimumPerOutputTruePositives
                            || $0.calibratedMetrics.recall
                                < ValidationExecutionContract.minimumPerOutputRecall
                    )
            }
        }
        // Older reports did not retain per-output metrics.
        return [buttons, keyboard, modifiers].compactMap { $0 }.contains {
            $0.positiveSupport >= ValidationExecutionContract.minimumSupportedPositiveSamples
                && (
                    $0.truePositives < ValidationExecutionContract.minimumPerOutputTruePositives
                        || $0.recall < ValidationExecutionContract.minimumPerOutputRecall
                )
        }
    }

    var hasContinuousExecutionCollapse: Bool {
        let balances = trainingBalance?.continuousOutputs ?? []
        let relativeSupport = balances
            .filter { ActionLayout.relativeMouse.contains($0.outputIndex) }
            .reduce(0) { $0 + $1.activeSamples }
        let scrollSupport = balances
            .filter { ActionLayout.scroll.contains($0.outputIndex) }
            .reduce(0) { $0 + $1.activeSamples }
        return (
            relativeSupport >= ValidationExecutionContract.minimumSupportedPositiveSamples
                && (activeRelativeMouseExecutionRecall ?? 0)
                    < ValidationExecutionContract.minimumContinuousExecutionRecall
        ) || (
            scrollSupport >= ValidationExecutionContract.minimumSupportedPositiveSamples
                && (activeScrollExecutionRecall ?? 0)
                    < ValidationExecutionContract.minimumContinuousExecutionRecall
        )
    }

    var hasContinuousExecutionFailure: Bool {
        hasContinuousExecutionCollapse
            || (idleContinuousFalseActionRate ?? 0)
                > ValidationExecutionContract.maximumIdleContinuousFalseActionRate
    }

    /// Deployment quality gives every evidenced binary control equal influence,
    /// so a common movement key cannot hide a sparse key that still does nothing.
    /// Additive controls contribute their correctly directed executable recall,
    /// discounted by the observed idle execution rate. Weighted validation loss
    /// remains useful for optimization and tie-breaking, but it is not a proxy
    /// for whether an agent will physically perform the demonstrated actions.
    var deploymentQualityScore: Double? {
        var components: [Double] = []
        if let binaryOutputs {
            components.append(contentsOf: binaryOutputs.compactMap { output in
                output.calibratedMetrics.positiveSupport
                    >= ValidationExecutionContract.minimumSupportedPositiveSamples
                    ? output.calibratedMetrics.f1
                    : nil
            })
        } else if let binary,
                  binary.positiveSupport
                    >= ValidationExecutionContract.minimumSupportedPositiveSamples {
            components.append(binary.f1)
        }

        let idleStability = 1 - min(1, max(0, idleContinuousFalseActionRate ?? 0))
        if let activeRelativeMouseExecutionRecall {
            components.append(
                min(1, max(0, activeRelativeMouseExecutionRecall)) * idleStability
            )
        }
        if let activeScrollExecutionRecall {
            components.append(
                min(1, max(0, activeScrollExecutionRecall)) * idleStability
            )
        }
        guard !components.isEmpty else { return nil }
        return components.reduce(0, +) / Double(components.count)
    }

    /// On the same fixed held-out rows, reject only large regressions with enough
    /// support to be meaningful. A false-positive increase remains eligible when
    /// it buys a material F1 improvement instead of merely creating idle noise.
    func hasSevereBinaryRegression(comparedTo baseline: ValidationReport) -> Bool {
        if let currentOutputs = binaryOutputs,
           let previousOutputs = baseline.binaryOutputs {
            let currentByIndex = Dictionary(
                uniqueKeysWithValues: currentOutputs.map { ($0.outputIndex, $0.calibratedMetrics) }
            )
            return previousOutputs.contains { previousOutput in
                guard let current = currentByIndex[previousOutput.outputIndex] else {
                    return false
                }
                let previous = previousOutput.calibratedMetrics
                let lostAllRecall = previous.truePositives > 0
                    && current.positiveSupport >= 8
                    && current.truePositives == 0
                let f1Collapsed = previous.positiveSupport >= 5
                    && current.f1 + 0.10 < previous.f1
                let falsePositivesSpiked = previous.negativeSupport >= 20
                    && current.falsePositiveRate > previous.falsePositiveRate + 0.03
                    && current.f1 < previous.f1 + 0.02
                return lostAllRecall || f1Collapsed || falsePositivesSpiked
            }
        }
        let pairs = [
            (buttons, baseline.buttons),
            (keyboard, baseline.keyboard),
            (modifiers, baseline.modifiers)
        ]
        return pairs.contains { current, previous in
            guard let current, let previous else { return false }
            let lostAllRecall = previous.truePositives > 0
                && current.positiveSupport >= 8
                && current.truePositives == 0
            let f1Collapsed = previous.positiveSupport >= 5
                && current.f1 + 0.10 < previous.f1
            let falsePositivesSpiked = previous.negativeSupport >= 20
                && current.falsePositiveRate > previous.falsePositiveRate + 0.03
                && current.f1 < previous.f1 + 0.02
            return lostAllRecall || f1Collapsed || falsePositivesSpiked
        }
    }

    func hasSevereContinuousRegression(comparedTo baseline: ValidationReport) -> Bool {
        let pairs = [
            (activeRelativeMouseExecutionRecall, baseline.activeRelativeMouseExecutionRecall),
            (activeScrollExecutionRecall, baseline.activeScrollExecutionRecall)
        ]
        let recallRegressed = pairs.contains { current, previous in
            guard let current, let previous else { return false }
            return previous >= ValidationExecutionContract.minimumContinuousExecutionRecall
                && current < ValidationExecutionContract.minimumContinuousExecutionRecall
        }
        let idleExecutionRegressed = (baseline.idleContinuousFalseActionRate ?? 0)
                <= ValidationExecutionContract.maximumIdleContinuousFalseActionRate
            && (idleContinuousFalseActionRate ?? 0)
                > ValidationExecutionContract.maximumIdleContinuousFalseActionRate
        return recallRegressed || idleExecutionRegressed
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
    /// The evidence-verified keyboard capability learned by this immutable
    /// brain. Runtime output is intersected with this set, so a model can never
    /// emit an unseen or under-demonstrated key.
    /// Optional keeps versions created before this invariant decodable.
    var demonstratedKeyCodes: Set<UInt16>? = nil
    /// Optional so schema-1 manifests remain decodable long enough to be
    /// identified and archived by the compatibility migration.
    var relativeMouseScale: Float? = nil
    /// Optional keeps existing runnable brains compatible. New training writes
    /// the dataset/target contract that produced the brain.
    var trainingDataSchema: Int? = nil
    /// Loss/optimizer semantics are versioned independently from cached frames
    /// and tensor shapes. A mismatch restarts optimizer state but can retain the
    /// immutable learned weights.
    var trainingObjectiveSchema: Int? = nil
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
    /// Optional keeps existing manifests decodable. New brains retain either
    /// the per-head held-out report that justified best-brain selection or a
    /// clearly scoped in-sample report used only for runtime calibration.
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
        LearnedBrainContract(
            preprocessing: preprocessing,
            visualMemoryFrames: training.effectiveVisualMemoryFrames,
            visualMemoryStride: training.learnedVisualMemoryStride,
            architecture: LearnedBrainArchitectureContract(training.architecture)
        )
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

/// Only fields that change Policy-v9 tensors or inference semantics belong in
/// this contract. Legacy recurrent/pooling settings and training-only dropout
/// must not force users to discard a compatible brain.
struct LearnedBrainArchitectureContract: Hashable, Sendable {
    var family: PolicyArchitectureFamily
    var patchSize: Int
    var convolutionChannels: [Int]
    var kernelSizes: [Int]
    var strides: [Int]
    var tokenWidth: Int
    var spatialTokens: Int
    var transformerLayers: Int
    var transformerHeads: Int
    var transformerFeedForward: Int
    var fusionWidths: [Int]

    init(_ architecture: ArchitectureSpec) {
        family = architecture.effectiveFamily
        patchSize = family == .pureTransformer ? architecture.effectivePatchSize : 0
        convolutionChannels = family == .hybrid ? architecture.convolutionChannels : []
        kernelSizes = family == .hybrid ? architecture.kernelSizes : []
        strides = family == .hybrid ? architecture.strides : []
        tokenWidth = architecture.visualEmbedding
        spatialTokens = family == .hybrid ? architecture.effectiveSpatialTokens : 0
        transformerLayers = architecture.effectiveTransformerLayers
        transformerHeads = architecture.effectiveTransformerHeads
        transformerFeedForward = architecture.effectiveTransformerFeedForward
        fusionWidths = architecture.fusionWidths
    }
}

struct LearnedBrainContract: Hashable, Sendable {
    var preprocessing: PreprocessingSpec
    var visualMemoryFrames: Int
    var visualMemoryStride: Int
    var architecture: LearnedBrainArchitectureContract
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
    var balanceReport: TrainingBalanceReport?
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
    /// Previous perceptions available to the current decision's bounded
    /// visual-memory window. The Run UI compares this with the saved contract's
    /// maximum lag so startup context is visible instead of implicit.
    var visualMemoryDepth = 0
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
        let colorChannels = profile.preprocessing.channelCount
        let memoryFrames = profile.training.effectiveVisualMemoryFrames
        let memoryEvidenceChannels = memoryFrames * (colorChannels + 1)
        if memoryEvidenceChannels > 0 {
            // Shared pointwise projection over signed visual differences and
            // availability. Current pixels bypass this compression unchanged.
            total = add(
                total,
                multiply(
                    Int64(memoryEvidenceChannels + 1),
                    Int64(VisualMemoryContract.fusionChannels)
                )
            )
        }
        var input = VisualMemoryContract.spatialEncoderInputChannels(
            colorChannels: colorChannels,
            frameCount: memoryFrames
        )
        let tokenWidth = Int64(max(1, architecture.visualEmbedding))
        let feedForward = Int64(architecture.effectiveTransformerFeedForward)

        switch architecture.effectiveFamily {
        case .hybrid:
            for i in architecture.convolutionChannels.indices {
                let output = max(1, architecture.convolutionChannels[i])
                let kernel = architecture.kernelSizes.indices.contains(i) ? max(1, architecture.kernelSizes[i]) : 3
                total = add(total, multiply(Int64(output), multiply(multiply(Int64(input), Int64(kernel)), Int64(kernel))))
                // Affine GroupNorm scale and bias.
                total = add(total, multiply(2, Int64(output)))
                input = output
            }
            let finalChannels = Int64(max(1, input))
            let spatialTokens = Int64(architecture.effectiveSpatialTokens)
            // Learned spatial-token scoring, followed by shared projections
            // for coordinate-bearing local tokens and global mean/max tokens.
            total = add(total, multiply(add(finalChannels, 1), spatialTokens))
            total = add(total, multiply(add(add(finalChannels, 2), 1), tokenWidth))
            total = add(total, multiply(add(finalChannels, 1), tokenWidth))
        case .pureTransformer:
            // A standard ViT tokenizer is one linear projection per direct,
            // non-overlapping patch. It is implemented as kernel=stride Conv2d
            // for an optimized graph, but there is no CNN feature hierarchy.
            let patchArea = multiply(Int64(architecture.effectivePatchSize), Int64(architecture.effectivePatchSize))
            total = add(total, multiply(add(multiply(Int64(input), patchArea), 1), tokenWidth))
            // Affine normalization of direct patch embeddings.
            total = add(total, multiply(2, tokenWidth))
        }

        // Learned readout token and visual token-type embeddings. Positions are
        // deterministic sinusoidal buffers.
        total = add(total, tokenWidth)
        total = add(total, multiply(Int64(PolicyTokenGeometry.tokenTypeCount), tokenWidth))

        for _ in 0..<architecture.effectiveTransformerLayers {
            // Q, K, V, and output projections all use bias in Policy v9.
            total = add(total, multiply(4, multiply(tokenWidth, add(tokenWidth, 1))))
            // Two affine LayerNorms.
            total = add(total, multiply(4, tokenWidth))
            // Gated SiLU feed-forward: gate/value d->ff, then ff->d.
            total = add(total, multiply(2, multiply(add(tokenWidth, 1), feedForward)))
            total = add(total, multiply(add(feedForward, 1), tokenWidth))
        }
        // Final Transformer normalization before the readout token is fused.
        total = add(total, multiply(2, tokenWidth))

        var fusionInput = max(1, architecture.visualEmbedding)
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
        let input = NeuralInputSizing.summary(for: profile)
        let batch = multiply(input.nominalBytesPerTrainingBatch, 8)
        let architecture = profile.training.architecture
        let batchSize = Int64(max(1, profile.training.batchSize))
        let sequence = Int64(PolicyTokenGeometry.sequenceLength(profile))
        let width = Int64(max(1, architecture.visualEmbedding))
        let feedForward = Int64(architecture.effectiveTransformerFeedForward)
        let heads = Int64(architecture.effectiveTransformerHeads)
        let layers = Int64(architecture.effectiveTransformerLayers)
        // Attention softmax is Float32 and backward retains several token/MLP
        // intermediates. This deliberately overestimates those tensors so the
        // UI rejects pathological token/history/batch combinations before MLX
        // can pressure the unified-memory reserve.
        let tokenState = multiply(multiply(sequence, width), 16)
        let feedForwardState = multiply(multiply(sequence, feedForward), 24)
        let attentionState = multiply(multiply(multiply(heads, sequence), sequence), 8)
        let transformerPerBatch = multiply(
            multiply(add(add(tokenState, feedForwardState), attentionState), layers),
            batchSize
        )
        return add(512 * 1_024 * 1_024, add(parameters, add(batch, transformerPerBatch)))
    }

    private static func multiply(_ lhs: Int64, _ rhs: Int64) -> Int64 { let result = lhs.multipliedReportingOverflow(by: rhs); return result.overflow ? Int64.max : result.partialValue }
    private static func add(_ lhs: Int64, _ rhs: Int64) -> Int64 { let result = lhs.addingReportingOverflow(rhs); return result.overflow ? Int64.max : result.partialValue }
}

/// Exact input counts for one policy decision and one optimizer batch. This
/// mirrors `VisionPreprocessor` and the perception-first `AgentPolicy` so the
/// UI never has to approximate the model contract independently.
struct NeuralInputSummary: Hashable, Sendable {
    var pixelCount: Int64
    var lumaValues: Int64
    var chromaValuesPerPlane: Int64
    var packedVisionValues: Int64
    var packedVisualWindowValues: Int64
    var expandedVisionValues: Int64
    var visualMemoryFrames: Int64
    var visualMemoryStride: Int64
    var visualMemoryMaximumLag: Int64
    var visualMemoryDurationSeconds: Double
    var temporalDifferenceValues: Int64
    var visualMemoryAvailabilityValues: Int64
    var visualMemoryFusionValues: Int64
    var coordinateValues: Int64
    var firstVisualProjectionValues: Int64
    var historySteps: Int64
    var actionValuesPerHistoryStep: Int64
    var historyValues: Int64
    var historyDurationSeconds: Double
    var transformerTokenCount: Int64
    var attentionPairsPerLayer: Int64
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
        let memoryFrames = Int64(profile.training.effectiveVisualMemoryFrames)
        let memoryStride = Int64(profile.training.effectiveVisualMemoryStride)
        let memoryMaximumLag = Int64(profile.training.visualMemoryMaximumLag)
        let packedVisualWindow = multiply(packedVision, add(1, memoryFrames))
        let coordinates = multiply(pixels, 2)
        let temporalDifference = multiply(expandedVision, memoryFrames)
        let memoryAvailability = multiply(pixels, memoryFrames)
        let memoryFusion = memoryFrames > 0
            ? multiply(pixels, Int64(VisualMemoryContract.fusionChannels))
            : 0
        let rawVisualInput = add(expandedVision, add(temporalDifference, memoryAvailability))
        let firstVisualProjection = add(add(expandedVision, memoryFusion), coordinates)

        // Policy v9 intentionally excludes teacher-forced actions. The legacy
        // fields remain in this Codable/UI summary as explicit zeroes so older
        // profile JSON can be explained without pretending it still affects
        // training cost or model input.
        let historySteps: Int64 = 0
        let actionValues = Int64(ActionLayout.count)
        let history: Int64 = 0
        // Values entering the graph include raw remembered differences and
        // availability before their pointwise fusion, plus generated X/Y.
        let perDecision = add(rawVisualInput, coordinates)
        let transformerTokens = Int64(PolicyTokenGeometry.sequenceLength(profile))
        let attentionPairs = PolicyTokenGeometry.attentionPairsPerLayer(profile)
        let historyDuration = 0.0
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
            packedVisualWindowValues: packedVisualWindow,
            expandedVisionValues: expandedVision,
            visualMemoryFrames: memoryFrames,
            visualMemoryStride: memoryStride,
            visualMemoryMaximumLag: memoryMaximumLag,
            visualMemoryDurationSeconds: profile.training.visualMemoryDurationSeconds,
            temporalDifferenceValues: temporalDifference,
            visualMemoryAvailabilityValues: memoryAvailability,
            visualMemoryFusionValues: memoryFusion,
            coordinateValues: coordinates,
            firstVisualProjectionValues: firstVisualProjection,
            historySteps: historySteps,
            actionValuesPerHistoryStep: actionValues,
            historyValues: history,
            historyDurationSeconds: historyDuration,
            transformerTokenCount: transformerTokens,
            attentionPairsPerLayer: attentionPairs,
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
