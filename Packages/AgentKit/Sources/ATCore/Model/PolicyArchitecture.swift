import Foundation

/// The Pixel2Play-style policy network shape. All sizes default to P2P "150M" (183M actual).
public struct PolicyArchitecture: Codable, Hashable, Sendable {
    public var width = 1024
    public var layers = 10
    public var heads = 16
    public var kvHeads = 16
    public var decoderLayers = 3
    public var decoderHeads = 8
    public var decoderSinks = 1
    public var thinkingTokens = 1
    public var dropout = 0.1
    /// Vision input (square, P2P supports 192 and 256 with the EfficientNet stem).
    public var frameSize = 192
    public var grayscale = false
    /// Control rate: one model step per 1/stepRateHz seconds.
    public var stepRateHz = 30
    /// Temporal memory in seconds; context = seconds × rate timesteps.
    public var memorySeconds = 10.0
    /// Optional per-layer attention history in timesteps (nil = full memory everywhere).
    public var historySteps: [Int]? = nil
    public var textEmbeddingDim = 768
    public var stochasticDepth = 0.2
    /// Optional language extensions (gated, zero-initialised so pretrained behaviour is unchanged).
    public var screenText = false
    public var screenTextItems = 8
    public var literalTyping = false
    public var typingBytes = 64

    public init() {}

    public var contextSteps: Int { max(2, Int((memorySeconds * Double(stepRateHz)).rounded())) }
    public var headDim: Int { width / heads }
    public var decoderHeadDim: Int { width / decoderHeads }
    public var ffnHidden: Int { Self.swigluHidden(width) }
    public static func swigluHidden(_ d: Int) -> Int { 8 * ((Int(2 * (4 * d) / 3) + 7) / 8) }
    public func tokensPerStep(slots: Int) -> Int { 1 + 1 + thinkingTokens + 1 + slots }
    public var usesDecoderMemory: Bool { screenText || literalTyping }

    /// Approximate parameter count for display.
    public func parameterCount(actions: ActionSpaceConfig) -> Int {
        let d = width, h = ffnHidden
        let block = 3 * d * d + d * d + 2 * d * h + h * d + 4 * d
        let decBlock = block + 2 * decoderSinks * d
        let stem = 850_000 + 112 * (frameSize / 16) * (frameSize / 16) * d + 2 * d
        let tables = (actions.keyClasses + actions.pointerClasses + actions.xClasses + actions.yClasses) * d * 2
        let tokens = d * (4 + thinkingTokens + actions.slotCount) + 768 * d + d * d + (actions.slotCount + 1) * d
        return layers * block + decoderLayers * decBlock + stem + tables + tokens
    }

    public static var p2p150M: PolicyArchitecture { PolicyArchitecture() }
    public static var p2p300M: PolicyArchitecture { var a = PolicyArchitecture(); a.layers = 20; return a }
    public static var compact60M: PolicyArchitecture {
        var a = PolicyArchitecture(); a.width = 768; a.layers = 6; a.heads = 12; a.decoderHeads = 6; a.decoderLayers = 2; return a
    }
}
