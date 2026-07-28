import Foundation
import MLX
import MLXNN
import MLXRandom

/// Shared pre-normalized Transformer block used by both Policy-v9 families.
/// Hybrid feeds it a compact learned sequence; Pure Transformer feeds it every
/// direct image patch. MLX's fused attention implementation handles both paths.
final class PolicyTransformerBlock: Module, @unchecked Sendable {
    @ModuleInfo var attentionNormalization: LayerNorm
    @ModuleInfo var attention: MultiHeadAttention
    @ModuleInfo var feedForwardNormalization: LayerNorm
    @ModuleInfo var gateProjection: Linear
    @ModuleInfo var valueProjection: Linear
    @ModuleInfo var outputProjection: Linear
    @ModuleInfo var attentionDropout: Dropout
    @ModuleInfo var feedForwardDropout: Dropout

    init(dimensions: Int, heads: Int, feedForward: Int, dropout: Float) {
        attentionNormalization = LayerNorm(dimensions: dimensions)
        attention = MultiHeadAttention(dimensions: dimensions, numHeads: heads, bias: true)
        feedForwardNormalization = LayerNorm(dimensions: dimensions)
        gateProjection = Linear(dimensions, feedForward)
        valueProjection = Linear(dimensions, feedForward)
        outputProjection = Linear(feedForward, dimensions)
        attentionDropout = Dropout(p: dropout)
        feedForwardDropout = Dropout(p: dropout)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let normalized = attentionNormalization(input)
        let attended = attention(normalized, keys: normalized, values: normalized)
        let residual = input + attentionDropout(attended)
        let feedForwardInput = feedForwardNormalization(residual)
        // Gated SiLU retains more useful capacity than a plain two-layer MLP at
        // this small token count while remaining a single compact MLX graph.
        let gated = silu(gateProjection(feedForwardInput)) * valueProjection(feedForwardInput)
        return residual + feedForwardDropout(outputProjection(gated))
    }
}

struct PolicyLossComponents {
    var total: MLXArray
    var mouse: MLXArray?
    var buttons: MLXArray?
    var scroll: MLXArray?
    var keyboard: MLXArray?
    var modifiers: MLXArray?
}

final class AgentPolicy: Module, @unchecked Sendable {
    let profile: AIProfile
    let dtype: DType
    // Leading underscore deliberately keeps this deterministic tensor out of
    // MLX Module reflection. It is an input buffer, not a learned parameter;
    // putting it in AdamW would waste two Float32 moment arrays and allow
    // optimization to distort the meaning of X/Y coordinates.
    private let _coordinateGrid: MLXArray
    private let _poolCoordinates: MLXArray
    private let _tokenPositions: MLXArray
    private let _tokenTypeIndices: MLXArray

    @ModuleInfo var convolutions: [Conv2d]
    @ModuleInfo var convolutionNormalizations: [GroupNorm]
    @ModuleInfo var visualMemoryProjections: [Linear]
    @ModuleInfo var patchEmbeddings: [Conv2d]
    @ModuleInfo var patchNormalizations: [LayerNorm]
    @ModuleInfo var spatialAttentionLayers: [Linear]
    @ModuleInfo var spatialTokenProjections: [Linear]
    @ModuleInfo var globalTokenProjections: [Linear]
    @ModuleInfo var tokenTypeEmbedding: Embedding
    @ParameterInfo var readoutToken: MLXArray
    @ModuleInfo var tokenDropout: Dropout
    @ModuleInfo var transformerBlocks: [PolicyTransformerBlock]
    @ModuleInfo var transformerNormalization: LayerNorm
    @ModuleInfo var fusion: [Linear]
    @ModuleInfo var fusionNormalizations: [LayerNorm]
    @ModuleInfo var absoluteMouseHead: Linear
    @ModuleInfo var relativeMouseHead: Linear
    @ModuleInfo var buttonHead: Linear
    @ModuleInfo var scrollHead: Linear
    @ModuleInfo var keyboardHead: Linear
    @ModuleInfo var modifierHead: Linear
    @ModuleInfo var dropout: Dropout

    init(profile: AIProfile) {
        self.profile = profile
        dtype = switch profile.training.precision {
        case .float16: .float16
        case .bfloat16: .bfloat16
        case .float32: .float32
        }
        let architecture = profile.training.architecture
        let width = max(1, profile.preprocessing.width)
        let height = max(1, profile.preprocessing.height)
        let x = broadcast(MLXArray.linspace(Float(-1), Float(1), count: width).reshaped([1, 1, width, 1]), to: [1, height, width, 1])
        let y = broadcast(MLXArray.linspace(Float(-1), Float(1), count: height).reshaped([1, height, 1, 1]), to: [1, height, width, 1])
        _coordinateGrid = concatenated([x, y], axis: -1).asType(dtype)
        let colorChannels = profile.preprocessing.channelCount
        let memoryFrames = profile.training.effectiveVisualMemoryFrames
        let memoryEvidenceChannels = memoryFrames * (colorChannels + 1)
        visualMemoryProjections = memoryEvidenceChannels > 0
            ? [Linear(memoryEvidenceChannels, VisualMemoryContract.fusionChannels)]
            : []
        // Exact current pixels bypass the bounded temporal fusion. Its learned
        // features and deterministic X/Y then enter either spatial family.
        var inputChannels = VisualMemoryContract.spatialEncoderInputChannels(
            colorChannels: colorChannels,
            frameCount: memoryFrames
        )
        let tokenWidth = max(1, architecture.visualEmbedding)
        let visualSize = CNNGeometry.outputSize(width: width, height: height, architecture: architecture)
        let spatialTokenCount: Int
        switch architecture.effectiveFamily {
        case .hybrid:
            var convs: [Conv2d] = []
            var normalizations: [GroupNorm] = []
            for i in architecture.convolutionChannels.indices {
                let output = max(1, architecture.convolutionChannels[i])
                let kernel = architecture.kernelSizes.indices.contains(i) ? max(1, architecture.kernelSizes[i]) : 3
                let stride = architecture.strides.indices.contains(i) ? max(1, architecture.strides[i]) : 2
                convs.append(Conv2d(inputChannels: inputChannels, outputChannels: output, kernelSize: .init(kernel), stride: .init(stride), padding: .init(kernel / 2), bias: false))
                normalizations.append(GroupNorm(groupCount: Self.normalizationGroups(for: output), dimensions: output))
                inputChannels = output
            }
            convolutions = convs
            convolutionNormalizations = normalizations
            patchEmbeddings = []
            patchNormalizations = []
            spatialTokenCount = architecture.effectiveSpatialTokens
            spatialAttentionLayers = [Linear(inputChannels, spatialTokenCount)]
            spatialTokenProjections = [Linear(inputChannels + 2, tokenWidth)]
            globalTokenProjections = [Linear(inputChannels, tokenWidth)]
            let poolX = broadcast(
                MLXArray.linspace(Float(-1), Float(1), count: visualSize.width).reshaped([1, 1, visualSize.width, 1]),
                to: [1, visualSize.height, visualSize.width, 1]
            )
            let poolY = broadcast(
                MLXArray.linspace(Float(-1), Float(1), count: visualSize.height).reshaped([1, visualSize.height, 1, 1]),
                to: [1, visualSize.height, visualSize.width, 1]
            )
            _poolCoordinates = concatenated([poolX, poolY], axis: -1)
                .reshaped([1, visualSize.width * visualSize.height, 2])
                .asType(dtype)
        case .pureTransformer:
            convolutions = []
            convolutionNormalizations = []
            let patch = architecture.effectivePatchSize
            patchEmbeddings = [Conv2d(
                inputChannels: inputChannels,
                outputChannels: tokenWidth,
                kernelSize: .init(patch),
                stride: .init(patch),
                padding: .init(0),
                bias: true
            )]
            patchNormalizations = [LayerNorm(dimensions: tokenWidth)]
            spatialAttentionLayers = []
            spatialTokenProjections = []
            globalTokenProjections = []
            spatialTokenCount = visualSize.width * visualSize.height
            _poolCoordinates = MLXArray.zeros([1, 1, 2], dtype: dtype)
        }
        tokenTypeEmbedding = Embedding(
            embeddingCount: PolicyTokenGeometry.tokenTypeCount,
            dimensions: tokenWidth
        )
        let tokenCount = PolicyTokenGeometry.sequenceLength(profile)
        let visualTypeIndices: [Int]
        if architecture.effectiveFamily == .hybrid {
            visualTypeIndices = Array(repeating: 1, count: spatialTokenCount)
                + Array(repeating: 2, count: PolicyTokenGeometry.globalTokenCount)
        } else {
            visualTypeIndices = Array(repeating: 1, count: spatialTokenCount)
        }
        let typeIndices = [0] + visualTypeIndices
        _tokenTypeIndices = MLXArray(typeIndices)
        let positionValues: [Float]
        if architecture.effectiveFamily == .pureTransformer {
            positionValues = Self.pureTransformerPositions(
                gridWidth: visualSize.width,
                gridHeight: visualSize.height,
                dimensions: tokenWidth
            )
        } else {
            positionValues = Self.sinusoidalPositions(count: tokenCount, dimensions: tokenWidth)
        }
        _tokenPositions = MLXArray(positionValues, [1, tokenCount, tokenWidth]).asType(dtype)
        readoutToken = MLXRandom.normal(
            [1, 1, tokenWidth],
            scale: sqrt(1 / Float(tokenWidth))
        )
        tokenDropout = Dropout(p: Float(min(0.999, max(0, architecture.dropout))))
        transformerBlocks = (0..<architecture.effectiveTransformerLayers).map { _ in
            PolicyTransformerBlock(
                dimensions: tokenWidth,
                heads: architecture.effectiveTransformerHeads,
                feedForward: architecture.effectiveTransformerFeedForward,
                dropout: Float(min(0.999, max(0, architecture.dropout)))
            )
        }
        transformerNormalization = LayerNorm(dimensions: tokenWidth)
        var fusionLayers: [Linear] = []
        var fusionNormalizations: [LayerNorm] = []
        var fusionInput = tokenWidth
        for width in architecture.fusionWidths {
            fusionLayers.append(Linear(fusionInput, max(1, width)))
            fusionNormalizations.append(LayerNorm(dimensions: max(1, width)))
            fusionInput = max(1, width)
        }
        fusion = fusionLayers
        self.fusionNormalizations = fusionNormalizations
        absoluteMouseHead = Linear(fusionInput, 2)
        relativeMouseHead = Linear(fusionInput, 2)
        buttonHead = Linear(fusionInput, 8)
        scrollHead = Linear(fusionInput, 2)
        keyboardHead = Linear(fusionInput, 128)
        modifierHead = Linear(fusionInput, 4)
        dropout = Dropout(p: Float(min(0.999, max(0, architecture.dropout))))
        super.init()
        if dtype != .float32 { update(parameters: mapParameters { $0.asType(self.dtype) }) }
    }

    private static func normalizationGroups(for channels: Int) -> Int {
        [8, 4, 2].first(where: { channels.isMultiple(of: $0) }) ?? 1
    }

    private static func sinusoidalPositions(count: Int, dimensions: Int) -> [Float] {
        guard count > 0, dimensions > 0 else { return [] }
        var values = [Float](repeating: 0, count: count * dimensions)
        for position in 0..<count {
            for dimension in 0..<dimensions {
                let pair = dimension / 2
                let denominator = Foundation.pow(10_000, Double(2 * pair) / Double(dimensions))
                let angle = Double(position) / denominator
                values[position * dimensions + dimension] = Float(dimension.isMultiple(of: 2)
                    ? Foundation.sin(angle)
                    : Foundation.cos(angle))
            }
        }
        return values
    }

    /// Standard separable 2-D sinusoidal positions preserve patch-row and
    /// patch-column identity without adding resolution-specific learned
    /// parameters.
    private static func pureTransformerPositions(
        gridWidth: Int,
        gridHeight: Int,
        dimensions: Int
    ) -> [Float] {
        guard dimensions > 0 else { return [] }
        let verticalDimensions = max(1, dimensions / 2)
        let horizontalDimensions = dimensions - verticalDimensions
        let vertical = sinusoidalPositions(count: gridHeight, dimensions: verticalDimensions)
        let horizontal = sinusoidalPositions(count: gridWidth, dimensions: horizontalDimensions)
        var values = [Float](repeating: 0, count: dimensions) // readout token
        values.reserveCapacity((1 + gridWidth * gridHeight) * dimensions)
        for row in 0..<gridHeight {
            for column in 0..<gridWidth {
                values.append(contentsOf: vertical[(row * verticalDimensions)..<((row + 1) * verticalDimensions)])
                if horizontalDimensions > 0 {
                    values.append(contentsOf: horizontal[(column * horizontalDimensions)..<((column + 1) * horizontalDimensions)])
                }
            }
        }
        return values
    }

    /// Returns each Hybrid CNN stage or the Pure Transformer's direct patch
    /// grid without changing the normal policy graph or its saved parameters.
    /// Runtime diagnostics consume these tensors only when explicitly enabled.
    func visualActivations(images: MLXArray) -> [MLXArray] {
        let memoryFrames = profile.training.effectiveVisualMemoryFrames
        let colorChannels = profile.preprocessing.channelCount
        let expectedChannels = VisualMemoryContract.rawInputChannels(
            colorChannels: colorChannels,
            frameCount: memoryFrames
        )
        precondition(
            images.ndim == 4 && images.dim(3) == expectedChannels,
            "Policy-v9 vision input does not match the saved visual-memory contract."
        )
        let rawVision = images.asType(dtype)
        let current = rawVision[.ellipsis, 0..<colorChannels]
        var vision = current
        if memoryFrames > 0, let projection = visualMemoryProjections.first {
            let differencesEnd = colorChannels + memoryFrames * colorChannels
            var differences = rawVision[.ellipsis, colorChannels..<differencesEnd]
                .reshaped([
                    images.dim(0),
                    profile.preprocessing.height,
                    profile.preprocessing.width,
                    memoryFrames,
                    colorChannels
                ])
            var availability = rawVision[.ellipsis, differencesEnd..<expectedChannels]

            // Keep the immediate predecessor reliable and randomly remove only
            // older slots during training. This makes the policy robust to a
            // newly started run or an uneven capture cadence without creating
            // a train/run magnitude mismatch for retained memory.
            let memoryDropout = profile.training.effectiveVisualMemoryDropout
            if training, memoryFrames > 1, memoryDropout > 0 {
                let immediate = MLXArray.ones(
                    [images.dim(0), 1, 1, 1],
                    dtype: dtype
                )
                let older = MLXRandom.bernoulli(
                    Float(1 - memoryDropout),
                    [images.dim(0), 1, 1, memoryFrames - 1]
                ).asType(dtype)
                let slotMask = concatenated([immediate, older], axis: -1)
                differences = differences * slotMask.expandedDimensions(axis: -1)
                availability = availability * slotMask
            }
            let memoryEvidence = concatenated([
                differences.reshaped([
                    images.dim(0),
                    profile.preprocessing.height,
                    profile.preprocessing.width,
                    memoryFrames * colorChannels
                ]),
                availability
            ], axis: -1)
            vision = concatenated([current, tanh(projection(memoryEvidence))], axis: -1)
        }
        let coordinates = broadcast(_coordinateGrid, to: [images.dim(0), profile.preprocessing.height, profile.preprocessing.width, 2])
        vision = concatenated([vision, coordinates], axis: -1)
        if profile.training.architecture.effectiveFamily == .pureTransformer {
            let patch = profile.training.architecture.effectivePatchSize
            let outputHeight = ((profile.preprocessing.height + patch - 1) / patch) * patch
            let outputWidth = ((profile.preprocessing.width + patch - 1) / patch) * patch
            let padHeight = max(0, outputHeight - profile.preprocessing.height)
            let padWidth = max(0, outputWidth - profile.preprocessing.width)
            if padHeight > 0 || padWidth > 0 {
                let widths: [IntOrPair] = [0, .init((0, padHeight)), .init((0, padWidth)), 0]
                // Edge padding preserves every source pixel and avoids adding a
                // synthetic black border to partial patches.
                vision = padded(vision, widths: widths, mode: .edge)
            }
            guard let embedding = patchEmbeddings.first else { return [vision] }
            vision = embedding(vision)
            if let normalization = patchNormalizations.first {
                vision = normalization(vision)
            }
            return [vision]
        }
        var activations: [MLXArray] = []
        activations.reserveCapacity(max(1, convolutions.count))
        for (index, convolution) in convolutions.enumerated() {
            vision = convolution(vision)
            if convolutionNormalizations.indices.contains(index) {
                vision = convolutionNormalizations[index](vision)
            }
            vision = silu(vision)
            activations.append(vision)
        }
        // A convolution-free custom architecture is still a valid tensor graph.
        // Treat its coordinate-aware input as the only visual stage.
        if activations.isEmpty { activations.append(vision) }
        return activations
    }

    /// Hybrid exposes its exact learned routing maps. Pure Transformer exposes
    /// a normalized direct-patch strength map; its individual embedding planes
    /// remain available through Feature Channels.
    func spatialTokenAttentionMaps(visualFeatures: MLXArray) -> MLXArray {
        let batch = visualFeatures.dim(0)
        let height = visualFeatures.dim(1)
        let width = visualFeatures.dim(2)
        let spatial = visualFeatures.reshaped([batch, -1, visualFeatures.dim(3)])
        if profile.training.architecture.effectiveFamily == .pureTransformer {
            let strength = sqrt((square(spatial.asType(.float32))).mean(axis: -1, keepDims: true) + 1e-8)
            return softmax(strength, axis: 1, precise: true)
                .reshaped([batch, height, width, 1])
        }
        guard let attention = spatialAttentionLayers.first else {
            return MLXArray.zeros([batch, height, width, 1], dtype: dtype)
        }
        return softmax(attention(spatial), axis: 1, precise: true)
            .reshaped([batch, height, width, profile.training.architecture.effectiveSpatialTokens])
    }

    /// Keeps live inspection bounded for custom policies with dozens of visual
    /// tokens. The policy still reasons over every token; only the HUD transfer
    /// is capped.
    func spatialTokensForVisualization(_ visualFeatures: MLXArray, maximumTokens: Int = 16) -> MLXArray {
        let sampled = sampledForVisualization(spatialTokenAttentionMaps(visualFeatures: visualFeatures))
        let count = min(sampled.dim(3), max(1, maximumTokens))
        return sampled[.ellipsis, 0..<count]
    }

    func logits(visualFeatures: MLXArray, history: MLXArray) -> MLXArray {
        let batch = visualFeatures.dim(0)
        let spatial = visualFeatures.reshaped([batch, -1, visualFeatures.dim(3)])
        let visualTokens: [MLXArray]
        switch profile.training.architecture.effectiveFamily {
        case .hybrid:
            guard let attentionLayer = spatialAttentionLayers.first,
                  let spatialProjection = spatialTokenProjections.first,
                  let globalProjection = globalTokenProjections.first else {
                preconditionFailure("Hybrid visual modules are unavailable.")
            }
            let attention = softmax(attentionLayer(spatial), axis: 1, precise: true)
            let coordinates = broadcast(_poolCoordinates, to: [batch, spatial.dim(1), 2])
            let local = spatialProjection(
                attention.transposed(0, 2, 1)
                    .matmul(concatenated([spatial, coordinates], axis: -1))
            )
            let global = globalProjection(stacked([
                spatial.mean(axis: 1),
                spatial.max(axis: 1)
            ], axis: 1))
            visualTokens = [local, global]
        case .pureTransformer:
            // Patch projection already has the shared token width. Keeping all
            // patches is the defining full-ViT path; no CNN pooling intervenes.
            visualTokens = [spatial]
        }

        // Policy v9 is perception-first. Policy v7's ground-truth history
        // tokens let the Transformer copy held controls during validation, but
        // live inference began with an all-released history and could never
        // leave that state. Keep the argument at this compatibility boundary
        // for loss/test call sites; it does not participate in policy logits.
        _ = history
        let readout = broadcast(readoutToken.asType(dtype), to: [batch, 1, profile.training.architecture.visualEmbedding])
        var tokens = concatenated([readout] + visualTokens, axis: 1)
        let types = tokenTypeEmbedding(_tokenTypeIndices).expandedDimensions(axis: 0)
        tokens = tokenDropout(tokens + types + _tokenPositions)
        for block in transformerBlocks {
            tokens = block(tokens)
        }
        let normalizedTokens = transformerNormalization(tokens)
        // A learned readout token alone can minimize a class-balanced objective
        // with nearly constant logits before it learns to attend to perception.
        // That seed-sensitive v8 failure looked different from the v7 history
        // shortcut but had the same user-visible result. A mean summary and the
        // first transformed visual anchor are parameter-free and come only from
        // perception, giving every head a short, unavoidable visual path while
        // preserving the readout's learned cross-token reasoning. Avoid hard
        // maximum or dynamic soft pooling here: their tied/parallel reductions
        // introduce minute nondeterministic gradients that Adam can amplify and
        // would break exact optimizer resume.
        let readoutState = normalizedTokens[0..., 0, 0...]
        let perceptionTokens = normalizedTokens[0..., 1..., 0...]
        let perceptionMean = perceptionTokens.mean(axis: 1)
        let perceptionAnchor = perceptionTokens[0..., 0, 0...]
        var fused = (
            readoutState
                + perceptionMean
                + perceptionAnchor
        ) / 3
        for (index, layer) in fusion.enumerated() {
            fused = layer(fused)
            if fusionNormalizations.indices.contains(index) {
                fused = fusionNormalizations[index](fused)
            }
            fused = dropout(silu(fused))
        }
        return concatenated([
            absoluteMouseHead(fused), relativeMouseHead(fused), buttonHead(fused),
            scrollHead(fused), keyboardHead(fused), modifierHead(fused)
        ], axis: -1)
    }

    func callAsFunction(images: MLXArray, history: MLXArray) -> MLXArray {
        logits(visualFeatures: visualActivations(images: images).last!, history: history)
    }

    func activatedPredictions(logits: MLXArray) -> MLXArray {
        concatenated([
            sigmoid(logits[.ellipsis, ActionLayout.absoluteMouse]),
            tanh(logits[.ellipsis, ActionLayout.relativeMouse]),
            sigmoid(logits[.ellipsis, ActionLayout.buttons]),
            tanh(logits[.ellipsis, ActionLayout.scroll]),
            sigmoid(logits[.ellipsis, ActionLayout.keyboard]),
            sigmoid(logits[.ellipsis, ActionLayout.modifiers])
        ], axis: -1)
    }

    func predictions(images: MLXArray, history: MLXArray) -> MLXArray {
        activatedPredictions(logits: callAsFunction(images: images, history: history))
    }

    /// Caps the longest spatial side copied out of MLX while preserving every
    /// channel. This bounds CPU transfer and HUD rendering for very large model
    /// vision sizes; the model itself always runs at its exact configured size.
    func sampledForVisualization(_ tensor: MLXArray, maximumSide: Int = 96) -> MLXArray {
        guard tensor.ndim == 4 else { return tensor }
        let longest = max(tensor.dim(1), tensor.dim(2))
        let stride = max(1, Int(ceil(Double(longest) / Double(max(1, maximumSide)))))
        return tensor[0..., .stride(by: stride), .stride(by: stride), 0...]
    }

    /// Selects the strongest channels on the GPU so custom very-wide CNNs do
    /// not copy every feature plane into Swift merely to display a small grid.
    func strongestChannelsForVisualization(_ tensor: MLXArray, maximumChannels: Int = 16) -> MLXArray {
        let sampled = sampledForVisualization(tensor)
        guard sampled.ndim == 4, sampled.dim(3) > 0 else { return sampled }
        let count = min(sampled.dim(3), max(1, maximumChannels))
        let scores = sampled.mean(axes: [1, 2])[0]
        let indices = argSort(scores)[.stride(by: -1)][0..<count]
        return sampled[.ellipsis, indices]
    }

    func loss(
        images: MLXArray,
        history: MLXArray,
        targets: MLXArray,
        positiveWeights: MLXArray? = nil,
        previousTargets: MLXArray? = nil
    ) -> MLXArray {
        let logits = callAsFunction(images: images, history: history)
        return loss(
            logits: logits,
            history: history,
            targets: targets,
            positiveWeights: positiveWeights,
            previousTargets: previousTargets
        )
    }

    /// Lets validation share one forward pass between loss and metrics. This is
    /// mathematically identical to `loss(images:...)` and avoids evaluating the
    /// complete Hybrid or Pure Transformer twice for every held-out batch.
    func loss(
        logits: MLXArray,
        history: MLXArray,
        targets: MLXArray,
        positiveWeights: MLXArray? = nil,
        previousTargets: MLXArray? = nil
    ) -> MLXArray {
        lossComponents(
            logits: logits,
            history: history,
            targets: targets,
            positiveWeights: positiveWeights,
            previousTargets: previousTargets
        ).total
    }

    /// Returns the same scalar used by backpropagation plus its action-family
    /// components. Validation exports these existing graph values alongside the
    /// logits, so detailed quality telemetry costs no second model forward.
    func lossComponents(
        logits: MLXArray,
        history: MLXArray,
        targets: MLXArray,
        positiveWeights: MLXArray? = nil,
        previousTargets: MLXArray? = nil
    ) -> PolicyLossComponents {
        let targets = targets.asType(dtype)
        // Training passes the actual preceding cached action explicitly. This
        // remains correct when model history is disabled; using the mandatory
        // zero placeholder row incorrectly marked every held control as a new
        // transition and upweighted long holds fourfold.
        let previous = previousTargets?.asType(dtype)
            ?? history.asType(dtype)[.ellipsis, -1, 0...]
        // Class corrections can legitimately exceed Float16's finite range for
        // very sparse but repeatedly demonstrated controls. Keep binary-loss
        // statistics in Float32 even when model activations use a compact dtype.
        let positiveWeights = positiveWeights?.asType(.float32)
        var losses: [MLXArray] = []
        var result = PolicyLossComponents(total: MLXArray(0, dtype: dtype))
        let channels = profile.channels
        if channels.mouseMovement {
            let absolute = smoothL1Loss(predictions: sigmoid(logits[.ellipsis, ActionLayout.absoluteMouse]), targets: targets[.ellipsis, ActionLayout.absoluteMouse], beta: 0.05)
            let relative = activeContinuousLoss(
                predictions: tanh(logits[.ellipsis, ActionLayout.relativeMouse]),
                targets: targets[.ellipsis, ActionLayout.relativeMouse],
                activeWeights: positiveWeights?[ActionLayout.relativeMouse]
            )
            let mouse = (absolute + relative) / 2
            result.mouse = mouse
            losses.append(mouse)
        }
        if channels.buttons {
            let buttons = binaryControlLoss(logits: logits, targets: targets, previous: previous, positiveWeights: positiveWeights, range: ActionLayout.buttons)
            result.buttons = buttons
            losses.append(buttons)
        }
        if channels.scroll {
            let scroll = activeContinuousLoss(
                predictions: tanh(logits[.ellipsis, ActionLayout.scroll]),
                targets: targets[.ellipsis, ActionLayout.scroll],
                activeWeights: positiveWeights?[ActionLayout.scroll]
            )
            result.scroll = scroll
            losses.append(scroll)
        }
        if channels.keyboard {
            let keyboard = binaryControlLoss(logits: logits, targets: targets, previous: previous, positiveWeights: positiveWeights, indices: ActionLayout.keyboardAndShiftIndices)
            result.keyboard = keyboard
            losses.append(keyboard)
        }
        if channels.modifiers {
            let modifiers = binaryControlLoss(logits: logits, targets: targets, previous: previous, positiveWeights: positiveWeights, range: ActionLayout.commandOptionControl)
            result.modifiers = modifiers
            losses.append(modifiers)
        }
        guard let first = losses.first else { return result }
        result.total = losses.dropFirst().reduce(first, +) / Float(losses.count)
        return result
    }

    private func binaryControlLoss(logits: MLXArray, targets: MLXArray, previous: MLXArray, positiveWeights: MLXArray?, range: Range<Int>) -> MLXArray {
        let selectedLogits = logits[.ellipsis, range].asType(.float32)
        let selectedTargets = targets[.ellipsis, range].asType(.float32)
        let selectedPrevious = previous[.ellipsis, range].asType(.float32)
        let raw = binaryCrossEntropy(logits: selectedLogits, targets: selectedTargets, reduction: .none)
        let classWeights: MLXArray
        let activeOutputs: MLXArray
        if let positiveWeights {
            let positives = positiveWeights[range]
            activeOutputs = (positives .> 0).asType(.float32)
            classWeights = (1 + selectedTargets * (positives - 1)) * activeOutputs
        } else {
            activeOutputs = MLXArray.ones([selectedTargets.dim(1)], dtype: .float32)
            classWeights = MLXArray.ones(like: selectedTargets)
        }
        // Press/release boundaries matter far more than another frame in the
        // middle of a long hold. Upweighting transitions prevents a policy from
        // learning only action persistence.
        let transitionWeights = 1
            + BinaryBalanceContract.transitionBonus * abs(selectedTargets - selectedPrevious)
        let staticWeights = classWeights * transitionWeights
        let focalWeights = binaryFocalWeights(logits: selectedLogits, targets: selectedTargets)
        // Normalize by the fixed number of active output decisions, not by the
        // current batch's class weights. Weight-sum normalization cancels a
        // rare positive's correction precisely in the batches where it appears,
        // then lets the many negative-only batches pull the logit back below
        // 0.5. Dataset-level balance now survives minibatching while expected
        // loss scale stays bounded (roughly two units of static mass/output).
        let decisionCount = activeOutputs.sum() * Float(max(1, selectedTargets.dim(0)))
        return (raw * staticWeights * focalWeights).sum() / (decisionCount + 1e-6)
    }

    private func binaryControlLoss(logits: MLXArray, targets: MLXArray, previous: MLXArray, positiveWeights: MLXArray?, indices: [Int]) -> MLXArray {
        let selection = MLXArray(indices)
        let selectedLogits = logits[.ellipsis, selection].asType(.float32)
        let selectedTargets = targets[.ellipsis, selection].asType(.float32)
        let selectedPrevious = previous[.ellipsis, selection].asType(.float32)
        let raw = binaryCrossEntropy(logits: selectedLogits, targets: selectedTargets, reduction: .none)
        let classWeights: MLXArray
        let activeOutputs: MLXArray
        if let positiveWeights {
            let positives = positiveWeights[selection]
            activeOutputs = (positives .> 0).asType(.float32)
            classWeights = (1 + selectedTargets * (positives - 1)) * activeOutputs
        } else {
            activeOutputs = MLXArray.ones([selectedTargets.dim(1)], dtype: .float32)
            classWeights = MLXArray.ones(like: selectedTargets)
        }
        let transitionWeights = 1
            + BinaryBalanceContract.transitionBonus * abs(selectedTargets - selectedPrevious)
        let staticWeights = classWeights * transitionWeights
        let focalWeights = binaryFocalWeights(logits: selectedLogits, targets: selectedTargets)
        let decisionCount = activeOutputs.sum() * Float(max(1, selectedTargets.dim(0)))
        return (raw * staticWeights * focalWeights).sum() / (decisionCount + 1e-6)
    }

    private func binaryFocalWeights(logits: MLXArray, targets: MLXArray) -> MLXArray {
        let gamma = profile.training.effectiveBinaryFocalGamma
        guard gamma > 0 else { return MLXArray.ones(like: targets) }
        // |target - probability| is p_t's complement. Raising it focuses the
        // already class-balanced loss on mistakes and ambiguous transitions
        // instead of spending most updates on easy idle negatives.
        return pow(abs(targets - sigmoid(logits)), Float(gamma))
    }

    private func activeContinuousLoss(
        predictions: MLXArray,
        targets: MLXArray,
        activeWeights: MLXArray?
    ) -> MLXArray {
        let beta: Float = 0.05
        let predictions = predictions.asType(.float32)
        let targets = targets.asType(.float32)
        // Divide by beta to keep an executable normalized camera delta on the
        // same objective scale as a binary decision. Without this, a perfectly
        // centered absolute cursor can hide a relative head that emits zero for
        // every frame while still reporting a tiny aggregate mouse loss.
        let raw = smoothL1Loss(
            predictions: predictions,
            targets: targets,
            beta: beta,
            reduction: .none
        ) / beta
        let isActive = (abs(targets) .> 0.0001).asType(.float32)
        if let activeWeights {
            let activeWeights = activeWeights.asType(.float32)
            let enabledOutputs = (activeWeights .> 0).asType(.float32)
            let staticWeights = (1 + isActive * (activeWeights - 1)) * enabledOutputs
            let decisions = enabledOutputs.sum() * Float(max(1, targets.dim(0)))
            return (raw * staticWeights).sum() / (decisions + 1e-6)
        }
        // Preserve a sensible standalone loss for tests and callers that do
        // not supply dataset statistics. Production training always supplies
        // exact per-axis active weights.
        let staticWeights = 1 + isActive * 7
        return (raw * staticWeights).sum() / (staticWeights.sum() + 1e-6)
    }

    func saveWeights(to url: URL) throws {
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: parameters().flattened()), metadata: ["format": ModelContract.weightFormat], url: url)
    }

    func loadWeights(from url: URL) throws {
        // A compatible weights-only brain may be fine-tuned at a different
        // configured precision. Cast at the load boundary so the freshly
        // constructed policy—not the file's historical storage dtype—remains
        // authoritative. Exact checkpoint resumes use the same dtype and are
        // therefore unchanged.
        let arrays = try MLX.loadArrays(url: url).mapValues { $0.asType(dtype) }
        try update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
    }
}

final class ResumableAdamW: Updatable, @unchecked Sendable {
    var learningRate: Float
    var weightDecay: Float
    private(set) var warmupSteps: Int
    private(set) var schedule: LearningRateSchedule
    private(set) var cycleSteps: Int
    private(set) var minimumLearningRateRatio: Float
    let beta1: Float = 0.9
    let beta2: Float = 0.999
    let epsilon: Float = 1e-8
    private var stepArray = MLXArray(0, dtype: .float32)
    private var learningRateScaleArray = MLXArray(1, dtype: .float32)
    private var firstMoments: [String: MLXArray] = [:]
    private var secondMoments: [String: MLXArray] = [:]
    private var parameterNames: [String] = []

    var step: Int { MLX.eval(stepArray); return Int(stepArray.item(Float.self).rounded()) }
    var learningRateScale: Float {
        MLX.eval(learningRateScaleArray)
        return learningRateScaleArray.item(Float.self)
    }

    init(
        learningRate: Float,
        weightDecay: Float,
        warmupSteps: Int = 500,
        schedule: LearningRateSchedule = .adaptiveCosine,
        cycleSteps: Int = 4_000,
        minimumLearningRateRatio: Float = 0.05
    ) {
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.warmupSteps = max(1, warmupSteps)
        self.schedule = schedule
        self.cycleSteps = max(1, cycleSteps)
        self.minimumLearningRateRatio = min(0.5, max(0.001, minimumLearningRateRatio))
    }

    func initialize(model: Module) {
        guard parameterNames.isEmpty else { return }
        parameterNames = model.parameters().flattened().map(\.0).sorted()
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        for name in parameterNames {
            guard let parameter = parameters[name] else { continue }
            let value = parameter.asType(.float32)
            firstMoments[name] = MLXArray.zeros(like: value)
            secondMoments[name] = MLXArray.zeros(like: value)
        }
    }

    func update(model: Module, gradients: ModuleParameters, targetType: DType) {
        initialize(model: model)
        stepArray = stepArray + 1
        // Both schedules depend only on persisted optimizer state. New training
        // uses bounded cosine restarts so the update size never silently tends
        // to zero; validation-driven envelope reductions still allow late-stage
        // refinement. Old checkpoints retain inverse-square-root behavior.
        let warmupSteps = MLXArray(Float(self.warmupSteps), dtype: .float32)
        let warmupScale = minimum(stepArray / warmupSteps, MLXArray(1, dtype: .float32))
        let scheduleScale: MLXArray
        switch schedule {
        case .legacyInverseSquareRoot:
            let decayScale = sqrt(warmupSteps / stepArray)
            scheduleScale = minimum(warmupScale, decayScale)
        case .adaptiveCosine:
            let cycleSteps = MLXArray(Float(self.cycleSteps), dtype: .float32)
            let afterWarmup = maximum(stepArray - warmupSteps, MLXArray(0, dtype: .float32))
            let phase = remainder(afterWarmup, cycleSteps) / cycleSteps
            let cosine = (1 + cos(Float.pi * phase)) / 2
            let floor = MLXArray(minimumLearningRateRatio, dtype: .float32)
            let cycleScale = (floor + (1 - floor) * cosine) * learningRateScaleArray
            // Plateau reductions lower the peaks, but the effective rate never
            // falls below the configured global floor. Multiplying two floors
            // would quietly recreate the near-zero learning wall this schedule
            // is designed to remove.
            scheduleScale = warmupScale * maximum(floor, cycleScale)
        }
        let scheduledLearningRate = learningRate * scheduleScale
        let gradientMap = Dictionary(uniqueKeysWithValues: gradients.flattened())
        var updated: [(String, MLXArray)] = []
        for (name, parameter) in model.parameters().flattened() {
            guard let gradient = gradientMap[name] else { updated.append((name, parameter)); continue }
            let p = parameter.asType(.float32)
            let g = gradient.asType(.float32)
            let m = beta1 * (firstMoments[name] ?? MLXArray.zeros(like: p)) + (1 - beta1) * g
            let v = beta2 * (secondMoments[name] ?? MLXArray.zeros(like: p)) + (1 - beta2) * square(g)
            firstMoments[name] = m
            secondMoments[name] = v
            let correction1 = 1 - pow(beta1, stepArray)
            let correction2 = 1 - pow(beta2, stepArray)
            let update = (m / correction1) / (sqrt(v / correction2) + epsilon)
            // AdamW regularizes matrix/kernel weights, while biases,
            // normalization scales, and learned token identities remain
            // unshrunk. This is especially important for a pre-normalized
            // Transformer, where decaying every 1-D affine parameter can erase
            // useful calibration without constraining model complexity.
            let decay = Self.appliesWeightDecay(parameterName: name, dimensions: parameter.ndim)
                ? weightDecay
                : 0
            updated.append((name, (p * (1 - scheduledLearningRate * decay) - scheduledLearningRate * update).asType(targetType)))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
    }

    static func appliesWeightDecay(parameterName: String, dimensions: Int) -> Bool {
        dimensions > 1
            && !parameterName.contains("readoutToken")
            && !parameterName.contains("tokenTypeEmbedding")
    }

    func save(to url: URL) throws {
        MLX.eval(innerState())
        var arrays: [String: MLXArray] = [:]
        for (key, value) in firstMoments { arrays["m.\(key)"] = value }
        for (key, value) in secondMoments { arrays["v.\(key)"] = value }
        try MLX.save(
            arrays: arrays,
            metadata: [
                "step": String(step),
                "learningRate": String(learningRate),
                "weightDecay": String(weightDecay),
                "warmupSteps": String(warmupSteps),
                "schedule": schedule.rawValue,
                "cycleSteps": String(cycleSteps),
                "minimumLearningRateRatio": String(minimumLearningRateRatio),
                "learningRateScale": String(learningRateScale)
            ],
            url: url
        )
    }

    func stateArrays() -> [MLXArray] {
        innerState()
    }

    func innerState() -> [MLXArray] {
        [stepArray, learningRateScaleArray] + parameterNames.flatMap { name in [firstMoments[name], secondMoments[name]].compactMap { $0 } }
    }

    /// Reduces the cosine envelope after a measured plateau. The mutable scalar
    /// is part of the compiled graph's explicit state, so changing it does not
    /// trigger graph recompilation and remains exactly checkpointable.
    @discardableResult
    func reduceLearningRate(factor: Float = 0.5) -> Float {
        let current = learningRateScale
        let next = max(minimumLearningRateRatio, min(1, current * min(0.99, max(0.01, factor))))
        learningRateScaleArray._updateInternal(MLXArray(next, dtype: .float32))
        MLX.eval(learningRateScaleArray)
        return next
    }

    func effectiveLearningRate() -> Float {
        effectiveLearningRate(at: step)
    }

    /// Scalar mirror of the compiled MLX schedule used by telemetry and tests.
    /// Keeping it callable for an explicit step makes restart/floor behavior
    /// auditable without executing a synthetic optimizer update.
    func effectiveLearningRate(at rawStep: Int) -> Float {
        let step = max(1, rawStep)
        let warmup = min(1, Float(step) / Float(max(1, warmupSteps)))
        let scale: Float
        switch schedule {
        case .legacyInverseSquareRoot:
            scale = min(warmup, sqrt(Float(warmupSteps) / Float(step)))
        case .adaptiveCosine:
            let afterWarmup = max(0, step - warmupSteps)
            let phase = Float(afterWarmup % max(1, cycleSteps)) / Float(max(1, cycleSteps))
            let cosine = (1 + Foundation.cos(Float.pi * phase)) / 2
            let cycleScale = (minimumLearningRateRatio + (1 - minimumLearningRateRatio) * cosine) * learningRateScale
            scale = warmup * max(minimumLearningRateRatio, cycleScale)
        }
        return learningRate * scale
    }

    func load(from url: URL) throws {
        let loaded = try MLX.loadArraysAndMetadata(url: url)
        firstMoments = Dictionary(uniqueKeysWithValues: loaded.0.compactMap { key, value in key.hasPrefix("m.") ? (String(key.dropFirst(2)), value) : nil })
        secondMoments = Dictionary(uniqueKeysWithValues: loaded.0.compactMap { key, value in key.hasPrefix("v.") ? (String(key.dropFirst(2)), value) : nil })
        parameterNames = firstMoments.keys.sorted()
        stepArray = MLXArray(Float(Int(loaded.1["step"] ?? "0") ?? 0), dtype: .float32)
        learningRate = Float(loaded.1["learningRate"] ?? "") ?? learningRate
        weightDecay = Float(loaded.1["weightDecay"] ?? "") ?? weightDecay
        // Checkpoints from the fixed-schedule trainer did not save this field;
        // retaining 500 preserves their exact continuation behavior.
        warmupSteps = max(1, Int(loaded.1["warmupSteps"] ?? "500") ?? 500)
        if let rawSchedule = loaded.1["schedule"], let restoredSchedule = LearningRateSchedule(rawValue: rawSchedule) {
            schedule = restoredSchedule
        } else {
            schedule = .legacyInverseSquareRoot
        }
        cycleSteps = max(1, Int(loaded.1["cycleSteps"] ?? "4000") ?? 4_000)
        minimumLearningRateRatio = min(0.5, max(0.001, Float(loaded.1["minimumLearningRateRatio"] ?? "0.05") ?? 0.05))
        let restoredScale = min(1, max(minimumLearningRateRatio, Float(loaded.1["learningRateScale"] ?? "1") ?? 1))
        learningRateScaleArray = MLXArray(restoredScale, dtype: .float32)
    }
}
