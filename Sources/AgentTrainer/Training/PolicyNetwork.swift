import Foundation
import MLX
import MLXNN
import MLXRandom

struct VisualAugmentationParameters {
    let brightness: MLXArray
    let contrast: MLXArray
    let saturation: MLXArray
    let hueRadians: MLXArray
    let gamma: MLXArray
}

final class AgentPolicy: Module, @unchecked Sendable {
    let profile: AIProfile
    let contextSchema: ModelContextSchema
    let dtype: DType
    private let legacyMovementFeedback: Bool
    private let legacyControlFeedback: Bool
    private let _movementSensitivity: Float
    static let maximumHistoryGain: Float = 0.4
    static let maximumControlLogitCorrection: Float = 1
    static let visualActionLossWeight: Float = 0.5
    // Leading underscore deliberately keeps this deterministic tensor out of
    // MLX Module reflection. It is an input buffer, not a learned parameter;
    // putting it in AdamW would waste two Float32 moment arrays and allow
    // optimization to distort the meaning of X/Y coordinates.
    private let _coordinateGrid: MLXArray
    private let _pastCoordinateGrid: MLXArray
    private let _poolCoordinates: MLXArray?
    private let _pastPoolCoordinates: MLXArray?

    @ModuleInfo var convolutions: [Conv2d]
    @ModuleInfo var convolutionNormalizations: [GroupNorm]
    @ModuleInfo var spatialAttention: Linear?
    @ModuleInfo var visualProjection: Linear
    @ModuleInfo var visualNormalization: LayerNorm
    @ModuleInfo var pastVisualProjection: Linear?
    @ModuleInfo var pastVisualNormalization: LayerNorm?
    @ModuleInfo var timingProjection: Linear
    @ModuleInfo var timingNormalization: LayerNorm
    @ModuleInfo var gru: GRU?
    @ModuleInfo var lstm: LSTM?
    @ModuleInfo var fusion: [Linear]
    @ModuleInfo var fusionNormalizations: [LayerNorm]
    @ModuleInfo var absoluteMouseHead: Linear
    @ModuleInfo var relativeMouseHead: Linear
    @ModuleInfo var buttonHead: Linear
    @ModuleInfo var scrollHead: Linear
    @ModuleInfo var keyboardHead: Linear
    @ModuleInfo var modifierHead: Linear
    @ModuleInfo var futureActionHead: Linear
    @ModuleInfo var futureStateHead: Linear
    @ModuleInfo var movementIntentHead: Linear
    @ModuleInfo var futureMovementIntentHead: Linear
    @ModuleInfo var movementHistoryAttention: Linear
    @ModuleInfo var movementHistoryGain: Linear
    @ModuleInfo var dropout: Dropout

    init(profile: AIProfile, contextSchema: ModelContextSchema = .empty, legacyMovementFeedback: Bool = false, legacyControlFeedback: Bool = false, movementSensitivity: Double = 1) {
        self.legacyMovementFeedback = legacyMovementFeedback
        self.legacyControlFeedback = legacyControlFeedback || legacyMovementFeedback
        _movementSensitivity = Float(movementSensitivity.isFinite ? min(100, max(0.01, movementSensitivity)) : 1)
        self.profile = profile
        self.contextSchema = contextSchema
        dtype = switch profile.training.precision {
        case .float16: .float16
        case .bfloat16: .bfloat16
        case .float32: .float32
        }
        let architecture = profile.training.architecture
        let width = max(1, profile.preprocessing.width)
        let height = max(1, profile.preprocessing.height)
        let pastSpec = profile.training.effectiveTemporalVision.pastFrameSpec(from: profile.preprocessing)
        _coordinateGrid = Self.coordinateGrid(width: width, height: height, dtype: dtype)
        _pastCoordinateGrid = Self.coordinateGrid(width: pastSpec.width, height: pastSpec.height, dtype: dtype)
        // Real color planes plus explicit X/Y. Current and past images use the
        // same weights and never include a synthetic difference channel.
        var inputChannels = profile.preprocessing.channelCount + 2
        var convs: [Conv2d] = []
        var convolutionNormalizations: [GroupNorm] = []
        for i in architecture.convolutionChannels.indices {
            let output = architecture.convolutionChannels[i]
            let kernel = architecture.kernelSizes.indices.contains(i) ? max(1, architecture.kernelSizes[i]) : 3
            let stride = architecture.strides.indices.contains(i) ? max(1, architecture.strides[i]) : 2
            convs.append(Conv2d(inputChannels: inputChannels, outputChannels: output, kernelSize: .init(kernel), stride: .init(stride), padding: .init(kernel / 2), bias: false))
            convolutionNormalizations.append(GroupNorm(groupCount: Self.normalizationGroups(for: output), dimensions: output))
            inputChannels = output
        }
        convolutions = convs
        self.convolutionNormalizations = convolutionNormalizations
        let visualSize = CNNGeometry.outputSize(width: width, height: height, architecture: architecture)
        let pastVisualSize = CNNGeometry.outputSize(width: pastSpec.width, height: pastSpec.height, architecture: architecture)
        let visualProjectionInput: Int
        if architecture.effectiveVisualPooling == .attention {
            let heads = architecture.effectiveAttentionHeads
            spatialAttention = Linear(inputChannels, heads)
            _poolCoordinates = Self.poolCoordinates(size: visualSize, dtype: dtype)
            _pastPoolCoordinates = Self.poolCoordinates(size: pastVisualSize, dtype: dtype)
            visualProjectionInput = heads * (inputChannels + 2) + 2 * inputChannels
        } else {
            spatialAttention = nil
            _poolCoordinates = nil
            _pastPoolCoordinates = nil
            visualProjectionInput = max(1, visualSize.width * visualSize.height * inputChannels)
        }
        visualProjection = Linear(visualProjectionInput, architecture.visualEmbedding)
        visualNormalization = LayerNorm(dimensions: architecture.visualEmbedding)
        if architecture.effectiveVisualPooling == .flattened {
            let pastProjectionInput = max(1, pastVisualSize.width * pastVisualSize.height * inputChannels)
            pastVisualProjection = Linear(pastProjectionInput, architecture.visualEmbedding)
            pastVisualNormalization = LayerNorm(dimensions: architecture.visualEmbedding)
        } else {
            pastVisualProjection = nil
            pastVisualNormalization = nil
        }
        timingProjection = Linear(TemporalInputContract.valueCount, TemporalInputContract.embeddingWidth, bias: false)
        timingNormalization = LayerNorm(dimensions: TemporalInputContract.embeddingWidth)
        if architecture.recurrentKind == .gru {
            gru = GRU(inputSize: architecture.visualEmbedding + ActionLayout.count + TemporalInputContract.embeddingWidth, hiddenSize: architecture.recurrentWidth)
            lstm = nil
        } else {
            gru = nil
            lstm = LSTM(inputSize: architecture.visualEmbedding + ActionLayout.count + TemporalInputContract.embeddingWidth, hiddenSize: architecture.recurrentWidth)
        }
        var fusionLayers: [Linear] = []
        var fusionNormalizations: [LayerNorm] = []
        var fusionInput = architecture.visualEmbedding + architecture.recurrentWidth + TemporalInputContract.embeddingWidth
            + contextSchema.inputValueCount
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
        futureActionHead = Linear(
            fusionInput,
            ActionLayout.count * FutureActionPredictionConfiguration.maximumHorizonCount
        )
        futureStateHead = Linear(
            fusionInput + ActionLayout.count,
            architecture.visualEmbedding
        )
        movementIntentHead = Linear(fusionInput, 2)
        futureMovementIntentHead = Linear(fusionInput, 2 * FutureActionPredictionConfiguration.maximumHorizonCount)
        movementHistoryAttention = Linear(fusionInput, 2 * (1 + profile.training.effectiveTemporalVision.pastFrameCount))
        movementHistoryGain = Linear(fusionInput, 2)
        dropout = Dropout(p: Float(min(0.999, max(0, architecture.dropout))))
        super.init()
        if dtype != .float32 { update(parameters: mapParameters { $0.asType(self.dtype) }) }
    }

    private static func normalizationGroups(for channels: Int) -> Int {
        [8, 4, 2].first(where: { channels.isMultiple(of: $0) }) ?? 1
    }

    private static func coordinateGrid(width: Int, height: Int, dtype: DType) -> MLXArray {
        let x = broadcast(
            MLXArray.linspace(Float(-1), Float(1), count: width).reshaped([1, 1, width, 1]),
            to: [1, height, width, 1]
        )
        let y = broadcast(
            MLXArray.linspace(Float(-1), Float(1), count: height).reshaped([1, height, 1, 1]),
            to: [1, height, width, 1]
        )
        return concatenated([x, y], axis: -1).asType(dtype)
    }

    private static func poolCoordinates(size: (width: Int, height: Int), dtype: DType) -> MLXArray {
        coordinateGrid(width: size.width, height: size.height, dtype: dtype)
            .reshaped([1, size.width * size.height, 2])
    }

    /// Samples one conservative photometric transform per supervised row.
    /// Callers reuse each row's parameters for its current, past, and future
    /// images, preserving temporal truth without correlating an entire batch.
    func sampleVisualAugmentation(batchSize: Int = 1) -> VisualAugmentationParameters {
        let configuration = profile.training.effectiveVisualAugmentation
        let count = max(1, batchSize)
        let shape = [count, 1, 1, 1]
        guard training, configuration.enabled, configuration.probability > 0 else {
            let zero = MLXArray.zeros(shape, dtype: .float32)
            let one = MLXArray.ones(shape, dtype: .float32)
            return VisualAugmentationParameters(
                brightness: zero, contrast: one, saturation: one,
                hueRadians: zero, gamma: one
            )
        }
        let applied = MLXRandom.bernoulli(Float(configuration.probability), shape).asType(.float32)
        func variation(_ amount: Double, centeredAt center: Float = 0) -> MLXArray {
            guard amount > 0 else {
                return MLXArray.full(shape, values: MLXArray(center, dtype: .float32))
            }
            return MLXArray.full(shape, values: MLXArray(center, dtype: .float32))
                + applied * MLXRandom.uniform(
                    low: Float(-amount), high: Float(amount), shape, dtype: .float32
                )
        }
        return VisualAugmentationParameters(
            brightness: variation(configuration.brightness),
            contrast: variation(configuration.contrast, centeredAt: 1),
            saturation: variation(configuration.saturation, centeredAt: 1),
            hueRadians: variation(configuration.hueDegrees * .pi / 180),
            gamma: variation(configuration.gamma, centeredAt: 1)
        )
    }

    func augmentedVision(
        _ images: MLXArray,
        parameters: VisualAugmentationParameters
    ) -> MLXArray {
        precondition(images.ndim == 4, "Visual augmentation expects NHWC images.")
        var y = images[.ellipsis, 0..<1].asType(.float32)
        y = clip((y - 0.5) * parameters.contrast + 0.5 + parameters.brightness, min: 0, max: 1)
        y = pow(clip(y, min: 0.0001, max: 1), parameters.gamma)
        guard images.dim(3) >= 3 else { return y }
        let cb = images[.ellipsis, 1..<2].asType(.float32) - 0.5
        let cr = images[.ellipsis, 2..<3].asType(.float32) - 0.5
        let cosine = cos(parameters.hueRadians)
        let sine = sin(parameters.hueRadians)
        let rotatedCB = (cb * cosine - cr * sine) * parameters.saturation
        let rotatedCR = (cb * sine + cr * cosine) * parameters.saturation
        return concatenated([
            y,
            clip(rotatedCB + 0.5, min: 0, max: 1),
            clip(rotatedCR + 0.5, min: 0, max: 1)
        ], axis: -1).asType(images.dtype)
    }

    func augmentedPastVision(
        _ images: MLXArray,
        parameters: VisualAugmentationParameters
    ) -> MLXArray {
        guard images.ndim == 5 else {
            return augmentedVision(images, parameters: parameters)
        }
        let shape = images.shape
        let frameCount = shape[1]
        func repeated(_ values: MLXArray) -> MLXArray {
            broadcast(
                values.reshaped([shape[0], 1, 1, 1, 1]),
                to: [shape[0], frameCount, 1, 1, 1]
            ).reshaped([shape[0] * frameCount, 1, 1, 1])
        }
        let temporalParameters = VisualAugmentationParameters(
            brightness: repeated(parameters.brightness),
            contrast: repeated(parameters.contrast),
            saturation: repeated(parameters.saturation),
            hueRadians: repeated(parameters.hueRadians),
            gamma: repeated(parameters.gamma)
        )
        return augmentedVision(
            images.reshaped([shape[0] * shape[1], shape[2], shape[3], shape[4]]),
            parameters: temporalParameters
        ).reshaped(shape)
    }

    /// Returns every normalized post-SiLU spatial stage without changing the
    /// normal policy graph or its saved parameters. Runtime diagnostics consume
    /// these tensors only when explicitly enabled.
    func visualActivations(images: MLXArray) -> [MLXArray] {
        visualActivations(
            images: images,
            coordinateGrid: _coordinateGrid,
            width: profile.preprocessing.width,
            height: profile.preprocessing.height,
            acceleratedOperators: Self.usesAcceleratedVisionPath(
                batch: images.dim(0),
                width: profile.preprocessing.width,
                height: profile.preprocessing.height
            )
        )
    }

    /// Fused normalization and the split coordinate stem have a fixed launch
    /// cost. Below this measured crossover the original kernels are faster;
    /// larger batches/frames recover substantially more repeated GPU work.
    static func usesAcceleratedVisionPath(batch: Int, width: Int, height: Int) -> Bool {
        let pixels = max(0, width).multipliedReportingOverflow(by: max(0, height))
        guard !pixels.overflow else { return true }
        let spatialSamples = max(0, batch).multipliedReportingOverflow(by: pixels.partialValue)
        return spatialSamples.overflow || spatialSamples.partialValue >= 200_000
    }

    /// The switch is internal solely so the opt-in hardware benchmark can time
    /// the pre-optimization graph against the adaptive production graph with
    /// identical weights.
    func visualActivations(images: MLXArray, acceleratedOperators: Bool) -> [MLXArray] {
        visualActivations(
            images: images,
            coordinateGrid: _coordinateGrid,
            width: profile.preprocessing.width,
            height: profile.preprocessing.height,
            acceleratedOperators: acceleratedOperators
        )
    }

    func pastVisualActivations(images: MLXArray) -> [MLXArray] {
        let spec = profile.training.effectiveTemporalVision.pastFrameSpec(from: profile.preprocessing)
        return visualActivations(
            images: images,
            coordinateGrid: _pastCoordinateGrid,
            width: spec.width,
            height: spec.height,
            acceleratedOperators: Self.usesAcceleratedVisionPath(
                batch: images.dim(0),
                width: spec.width,
                height: spec.height
            )
        )
    }

    private func visualActivations(
        images: MLXArray,
        coordinateGrid: MLXArray,
        width: Int,
        height: Int,
        acceleratedOperators: Bool
    ) -> [MLXArray] {
        var vision = images.asType(dtype)
        var activations: [MLXArray] = []
        activations.reserveCapacity(max(1, convolutions.count))
        for (index, convolution) in convolutions.enumerated() {
            // Splitting the coordinate contribution only removes repeated work
            // when a batch has multiple samples. Batch-one inference retains
            // the original single convolution while still using fused GroupNorm.
            if index == 0, acceleratedOperators, images.dim(0) > 1 {
                vision = Self.sharedCoordinateConvolution(
                    vision,
                    coordinates: coordinateGrid,
                    convolution: convolution
                )
            } else {
                if index == 0 {
                    let coordinates = broadcast(
                        coordinateGrid,
                        to: [images.dim(0), height, width, 2]
                    )
                    vision = concatenated([vision, coordinates], axis: -1)
                }
                vision = convolution(vision)
            }
            if convolutionNormalizations.indices.contains(index) {
                let normalization = convolutionNormalizations[index]
                vision = acceleratedOperators
                    ? Self.acceleratedGroupNorm(vision, normalization: normalization)
                    : normalization(vision)
            }
            vision = silu(vision)
            activations.append(vision)
        }
        // A convolution-free custom architecture is still a valid tensor graph.
        // Treat its coordinate-aware input as the only visual stage.
        if activations.isEmpty {
            let coordinates = broadcast(
                coordinateGrid,
                to: [images.dim(0), height, width, 2]
            )
            activations.append(concatenated([vision, coordinates], axis: -1))
        }
        return activations
    }

    /// The coordinate grid is identical for every item in a batch and the stem
    /// convolution is linear. Convolving its two positional channels once and
    /// broadcasting the result avoids repeating that work for every sample,
    /// while slicing the existing weight preserves the model/checkpoint shape.
    static func sharedCoordinateConvolution(
        _ input: MLXArray,
        coordinates: MLXArray,
        convolution: Conv2d
    ) -> MLXArray {
        let contentChannels = input.dim(-1)
        let contentWeight = convolution.weight[.ellipsis, 0..<contentChannels]
        let coordinateWeight = convolution.weight[.ellipsis, contentChannels..<(contentChannels + 2)]
        var output = conv2d(
            input,
            contentWeight,
            stride: .init(convolution.stride),
            padding: .init(convolution.padding),
            dilation: .init(convolution.dilation),
            groups: convolution.groups
        ) + conv2d(
            coordinates,
            coordinateWeight,
            stride: .init(convolution.stride),
            padding: .init(convolution.padding),
            dilation: .init(convolution.dilation),
            groups: convolution.groups
        )
        if let bias = convolution.bias { output = output + bias }
        return output
    }

    /// MLX's general GroupNorm spells this layout as separate mean, variance,
    /// normalize, scale, and bias operations. The policy's established
    /// interleaved grouping is equivalent to layer normalization after a view
    /// transpose, which lets MLX use its fused Metal kernel without changing
    /// grouping, epsilon, learned parameters, or the saved model contract.
    static func acceleratedGroupNorm(_ input: MLXArray, normalization: GroupNorm) -> MLXArray {
        let batch = input.dim(0)
        let shape = input.shape
        var grouped = input
            .reshaped(batch, -1, normalization.groupCount)
            .transposed(0, 2, 1)
        grouped = MLXFast.layerNorm(
            grouped,
            weight: nil,
            bias: nil,
            eps: normalization.eps
        )
        var output = grouped.transposed(0, 2, 1).reshaped(shape)
        if let weight = normalization.weight { output = weight * output }
        if let bias = normalization.bias { output = output + bias }
        return output
    }

    func visualEmbedding(visualFeatures: MLXArray) -> MLXArray {
        projectedVisualEmbedding(
            visualFeatures: visualFeatures,
            poolCoordinates: _poolCoordinates,
            projection: visualProjection,
            normalization: visualNormalization
        )
    }

    func pastVisualEmbedding(visualFeatures: MLXArray) -> MLXArray {
        projectedVisualEmbedding(
            visualFeatures: visualFeatures,
            poolCoordinates: _pastPoolCoordinates,
            projection: pastVisualProjection ?? visualProjection,
            normalization: pastVisualNormalization ?? visualNormalization
        )
    }

    private func projectedVisualEmbedding(
        visualFeatures: MLXArray,
        poolCoordinates: MLXArray?,
        projection: Linear,
        normalization: LayerNorm
    ) -> MLXArray {
        let batch = visualFeatures.dim(0)
        var vision: MLXArray
        if let spatialAttention, let poolCoordinates {
            let spatial = visualFeatures.reshaped([batch, -1, visualFeatures.dim(3)])
            // Softmax over locations makes each learned head an interpretable
            // spatial keypoint. Pooling exact coordinates retains layout while
            // global mean/max features preserve scene-wide context.
            let coordinates = broadcast(poolCoordinates, to: [batch, spatial.dim(1), 2])
            let attention = softmax(spatialAttention(spatial), axis: 1, precise: true)
            let attended = attention.transposed(0, 2, 1)
                .matmul(concatenated([spatial, coordinates], axis: -1))
                .reshaped([batch, -1])
            vision = concatenated([
                attended,
                spatial.mean(axis: 1),
                spatial.max(axis: 1)
            ], axis: -1)
        } else {
            vision = visualFeatures.reshaped([batch, -1])
        }
        return silu(normalization(projection(vision)))
    }

    func temporalFeatures(
        currentImages: MLXArray,
        pastImages: MLXArray,
        pastControls: MLXArray,
        visionToPast: MLXArray? = nil,
        sampleToVision: MLXArray? = nil,
        currentTiming: MLXArray? = nil,
        pastTiming: MLXArray? = nil
    ) -> MLXArray {
        temporalFeatures(
            currentVisualFeatures: visualActivations(images: currentImages).last!,
            pastImages: pastImages,
            pastControls: pastControls,
            visionToPast: visionToPast,
            sampleToVision: sampleToVision,
            currentTiming: currentTiming,
            pastTiming: pastTiming
        )
    }

    func currentOnlyTemporalFeatures(
        currentImages: MLXArray,
        sampleToVision: MLXArray? = nil,
        currentTiming: MLXArray? = nil
    ) -> MLXArray {
        currentOnlyTemporalFeatures(
            currentVisualEmbedding: visualEmbedding(
                visualFeatures: visualActivations(images: currentImages).last!
            ),
            sampleToVision: sampleToVision,
            currentTiming: currentTiming
        )
    }

    func currentOnlyTemporalFeatures(
        currentVisualFeatures: MLXArray,
        sampleToVision: MLXArray? = nil,
        currentTiming: MLXArray? = nil
    ) -> MLXArray {
        currentOnlyTemporalFeatures(
            currentVisualEmbedding: visualEmbedding(visualFeatures: currentVisualFeatures),
            sampleToVision: sampleToVision,
            currentTiming: currentTiming
        )
    }

    private func currentOnlyTemporalFeatures(
        currentVisualEmbedding: MLXArray,
        sampleToVision: MLXArray?,
        currentTiming: MLXArray? = nil
    ) -> MLXArray {
        var current = currentVisualEmbedding
        if let sampleToVision {
            precondition(sampleToVision.ndim == 1, "The current-frame row map must be one-dimensional.")
            if sampleToVision.dim(0) != current.dim(0) {
                current = current.take(sampleToVision, axis: 0)
            }
        }
        let emptyTemporalState = MLXArray.zeros(
            [current.dim(0), profile.training.architecture.recurrentWidth],
            dtype: dtype
        )
        let keep = training ? MLXRandom.bernoulli(Float(0.5), [current.dim(0), 1]).asType(.float32) : nil
        let decision = concatenated([current, emptyTemporalState,
            timingEmbedding(currentTiming, batch: current.dim(0), keep: keep)], axis: -1)
        let motor = concatenated([current, emptyTemporalState,
            timingEmbedding(motorTiming(currentTiming), batch: current.dim(0), keep: keep)], axis: -1)
        let history = movementHistory(current: currentTiming, past: nil, batch: current.dim(0), keep: keep)
        return concatenated([decision, motor, history], axis: -1)
    }

    func temporalFeatures(
        currentVisualFeatures: MLXArray,
        pastImages: MLXArray,
        pastControls: MLXArray,
        visionToPast: MLXArray? = nil,
        sampleToVision: MLXArray? = nil,
        currentTiming: MLXArray? = nil,
        pastTiming: MLXArray? = nil
    ) -> MLXArray {
        temporalFeatures(
            currentVisualEmbedding: visualEmbedding(visualFeatures: currentVisualFeatures),
            pastImages: pastImages,
            pastControls: pastControls,
            visionToPast: visionToPast,
            sampleToVision: sampleToVision,
            currentTiming: currentTiming,
            pastTiming: pastTiming
        )
    }

    func temporalFeatures(
        currentVisualEmbedding: MLXArray,
        pastImages: MLXArray,
        pastControls: MLXArray,
        visionToPast: MLXArray? = nil,
        sampleToVision: MLXArray? = nil,
        currentTiming: MLXArray? = nil,
        pastTiming: MLXArray? = nil
    ) -> MLXArray {
        let temporal = profile.training.effectiveTemporalVision
        if temporal.pastFrameCount == 0 {
            return currentOnlyTemporalFeatures(
                currentVisualEmbedding: currentVisualEmbedding,
                sampleToVision: sampleToVision,
                currentTiming: currentTiming
            )
        }
        precondition(pastControls.ndim == 3, "Past controls must have shape [batch, frames, controls].")
        let controls = pastControls.asType(dtype)
        let visionCount = currentVisualEmbedding.dim(0)
        let frameCount = pastControls.dim(1)
        let pastSpec = temporal.pastFrameSpec(from: profile.preprocessing)
        precondition(
            frameCount == temporal.pastFrameCount
                && pastControls.dim(0) == visionCount
                && pastControls.dim(1) == frameCount
                && pastControls.dim(2) == ActionLayout.count,
            "Past controls must pair one complete control row with every past image."
        )
        var pastEmbedding: MLXArray
        if let visionToPast {
            precondition(
                pastImages.ndim == 4
                    && pastImages.dim(1) == pastSpec.height
                    && pastImages.dim(2) == pastSpec.width
                    && pastImages.dim(3) == pastSpec.channelCount,
                "Deduplicated past images must have shape [unique frames, height, width, channels]."
            )
            precondition(
                visionToPast.ndim == 2
                    && visionToPast.dim(0) == visionCount
                    && visionToPast.dim(1) == frameCount,
                "The past-frame map must have shape [vision rows, past frames]."
            )
            let uniquePastEmbedding = pastVisualEmbedding(
                visualFeatures: pastVisualActivations(images: pastImages).last!
            )
            let mappedPastEmbedding = uniquePastEmbedding.dim(0) == visionCount * frameCount
                ? uniquePastEmbedding
                : uniquePastEmbedding.take(visionToPast.flattened(), axis: 0)
            pastEmbedding = mappedPastEmbedding.reshaped([
                visionCount,
                frameCount,
                profile.training.architecture.visualEmbedding
            ])
        } else {
            precondition(
                pastImages.ndim == 5
                    && pastImages.dim(0) == visionCount
                    && pastImages.dim(1) == frameCount
                    && pastImages.dim(2) == pastSpec.height
                    && pastImages.dim(3) == pastSpec.width
                    && pastImages.dim(4) == pastSpec.channelCount,
                "Past images do not match this brain's immutable temporal-vision contract."
            )
            let flattenedPast = pastImages.reshaped([
                visionCount * frameCount,
                pastImages.dim(2),
                pastImages.dim(3),
                pastImages.dim(4)
            ])
            pastEmbedding = pastVisualEmbedding(
                visualFeatures: pastVisualActivations(images: flattenedPast).last!
            ).reshaped([
                visionCount,
                frameCount,
                profile.training.architecture.visualEmbedding
            ])
        }

        return encodedTemporalFeatures(current: currentVisualEmbedding,
            past: pastEmbedding, controls: controls, currentTiming: currentTiming,
            pastTiming: pastTiming, sampleToVision: sampleToVision)
    }

    func encodedTemporalFeatures(
        current: MLXArray, past: MLXArray?, controls: MLXArray?,
        currentTiming: MLXArray?, pastTiming: MLXArray?, sampleToVision: MLXArray? = nil
    ) -> MLXArray {
        guard profile.training.effectiveTemporalVision.pastFrameCount > 0,
              var pastEmbedding = past, var controls else {
            return currentOnlyTemporalFeatures(currentVisualEmbedding: current,
                sampleToVision: sampleToVision, currentTiming: currentTiming)
        }
        controls = controls.asType(dtype)
        var currentVisualEmbedding = current
        let visionCount = current.dim(0)
        let frameCount = pastEmbedding.dim(1)
        // CNN features are deterministic and safe to share. During training we
        // gather them back to action-row shape before control masking and the
        // recurrent/fusion stack. This retains one independent mask and one
        // dropout path per label, matching the canonical stochastic objective.
        if let sampleToVision {
            precondition(
                sampleToVision.ndim == 1,
                "The sample-to-vision map must contain one index per action row."
            )
            // The cache assigns unique rows in first-use order, so an equal
            // row count proves this map is the identity and avoids three
            // otherwise redundant Metal gathers at equal action/perception FPS.
            if sampleToVision.dim(0) != visionCount {
                currentVisualEmbedding = currentVisualEmbedding.take(sampleToVision, axis: 0)
                pastEmbedding = pastEmbedding.take(sampleToVision, axis: 0)
                controls = controls.take(sampleToVision, axis: 0)
            }
        }
        let batch = currentVisualEmbedding.dim(0)
        // Complete frame-aligned controls are useful temporal evidence, but
        // held keys and buttons are still an easy shortcut. Mask only the
        // control half for half of training samples; past vision always remains
        // visible and inference always receives the full paired sequence.
        var controlKeep: MLXArray?
        if training {
            let keepProbability: Float = 0.5
            let mask = MLXRandom.bernoulli(keepProbability, [batch, 1, 1]).asType(dtype)
            controls = controls * mask
            controlKeep = mask.asType(.float32)
        }
        var timing = pastTiming ?? MLXArray.zeros([visionCount, frameCount, TemporalInputContract.valueCount], dtype: .float32)
        if timing.dim(0) != batch {
            guard let sampleToVision, timing.dim(0) == visionCount else {
                preconditionFailure("Historical timing requires one causal row per action or visual sequence.")
            }
            timing = timing.take(sampleToVision, axis: 0)
        }
        // Both streams reuse the CNN and learned recurrent/fusion weights.
        // The grounded stream excludes endogenous control feedback. Full timing
        // still informs a bounded correction to visually grounded controls.
        let fullSteps = concatenated([pastEmbedding, controls,
            timingEmbedding(timing, batch: batch, keep: controlKeep)], axis: -1)
        let motorSteps = concatenated([pastEmbedding, motorControls(controls),
            timingEmbedding(motorTiming(timing), batch: batch, keep: controlKeep)], axis: -1)
        let steps = concatenated([fullSteps, motorSteps], axis: 0)
        let recurrent: MLXArray
        if let gru { recurrent = gru(steps)[.ellipsis, -1, 0...] }
        else if let lstm { recurrent = lstm(steps).0[.ellipsis, -1, 0...] }
        else { recurrent = MLXArray.zeros([2 * batch, profile.training.architecture.recurrentWidth], dtype: dtype) }
        let currentKeep = controlKeep?.reshaped([batch, 1])
        let decision = concatenated([currentVisualEmbedding, recurrent[0..<batch],
            timingEmbedding(currentTiming, batch: batch, keep: currentKeep)], axis: -1)
        let motor = concatenated([currentVisualEmbedding, recurrent[batch..<(2 * batch)],
            timingEmbedding(motorTiming(currentTiming), batch: batch, keep: currentKeep)], axis: -1)
        let history = movementHistory(current: currentTiming, past: timing, batch: batch, keep: currentKeep)
        return concatenated([decision, motor, history], axis: -1)
    }

    /// Each history window is a non-expanding average of actual posted deltas.
    /// Never amplify a short/jittered interval. Runtime sensitivity is removed
    /// before reuse so the closed feedback gain remains bounded at any setting.
    private func movementHistory(current: MLXArray?, past: MLXArray?, batch: Int, keep: MLXArray?) -> MLXArray {
        func window(_ values: MLXArray) -> MLXArray {
            let values = values.asType(.float32)
            let seconds = exp(clip(values[.ellipsis, 2..<3], min: 0, max: 2) * Float(Foundation.log(61.0))) - 1
            let nominal = Float(1 / max(0.0001, profile.training.actionFPS))
            let valid = ((values[.ellipsis, 0..<1] .> 0) .&& (values[.ellipsis, 3..<4] .> 0) .&& (seconds .> 0)).asType(.float32)
            // Rates retain raw magnitude beyond the displacement clip. Decode
            // in Float32, average long windows, and never enlarge short ones.
            let rate = values[.ellipsis, (TemporalInputContract.motionOffset + 2)..<(TemporalInputContract.motionOffset + 4)]
            let signedRate = sign(rate) * (exp(minimum(abs(rate), Float(Foundation.log1p(1e12)))) - 1)
            let delta = clip(signedRate * minimum(seconds, nominal) / _movementSensitivity, min: -1, max: 1) * valid
            return concatenated([delta, valid], axis: -1)
        }
        let current = current ?? MLXArray.zeros([batch, TemporalInputContract.valueCount], dtype: .float32)
        var windows = [window(current).expandedDimensions(axis: 1)]
        let count = profile.training.effectiveTemporalVision.pastFrameCount
        if count > 0 {
            windows.append(window(past ?? MLXArray.zeros([batch, count, TemporalInputContract.valueCount], dtype: .float32)))
        }
        var history = concatenated(windows, axis: 1).reshaped([batch, 3 * (count + 1)])
        if let keep { history = history * keep.asType(.float32) }
        return history.asType(dtype)
    }

    /// The grounded stream retains all observation clocks and historical
    /// images. Endogenous controls cannot suppress camera confidence or become
    /// its own evidence for continued turning. Full timing remains available
    /// in the context stream, whose binary correction is explicitly bounded.
    private func motorTiming(_ values: MLXArray?) -> MLXArray? {
        guard let values, !legacyMovementFeedback else { return values }
        if !legacyControlFeedback {
            return concatenated([values[.ellipsis, 0..<16],
                MLXArray.zeros(like: values[.ellipsis, 16...])], axis: -1)
        }
        return concatenated([values[.ellipsis, 0..<16],
            MLXArray.zeros(like: values[.ellipsis, 16..<18]),
            values[.ellipsis, 18..<TemporalInputContract.motionOffset],
            MLXArray.zeros(like: values[.ellipsis, TemporalInputContract.motionOffset...])], axis: -1)
    }

    private func motorControls(_ values: MLXArray) -> MLXArray {
        guard !legacyMovementFeedback else { return values }
        if !legacyControlFeedback { return MLXArray.zeros(like: values) }
        return concatenated([MLXArray.zeros(like: values[.ellipsis, 0..<4]),
            values[.ellipsis, 4..<12], MLXArray.zeros(like: values[.ellipsis, 12..<14]),
            values[.ellipsis, 14...]], axis: -1)
    }

    /// Duration projection accumulates in Float32 even for a low-bit visual
    /// model. Casting only the embedding retains short-tap distinctions.
    func timingEmbedding(_ inputs: MLXArray?, batch: Int, keep: MLXArray? = nil) -> MLXArray {
        var values = inputs ?? MLXArray.zeros([batch, TemporalInputContract.valueCount], dtype: .float32)
        precondition(values.dim(-1) == TemporalInputContract.valueCount)
        values = values.asType(.float32)
        if let keep {
            values = concatenated([values[.ellipsis, 0..<16], values[.ellipsis, 16...] * keep], axis: -1)
        }
        let projected = values.matmul(timingProjection.weight.asType(.float32).T)
        return silu(timingNormalization(projected)).asType(dtype)
    }

    func fusedFeatures(
        temporalFeatures: MLXArray,
        contextValues: MLXArray? = nil,
        sampleToVision: MLXArray? = nil
    ) -> MLXArray {
        let width = profile.training.architecture.visualEmbedding + profile.training.architecture.recurrentWidth + TemporalInputContract.embeddingWidth
        let historyWidth = 3 * (1 + profile.training.effectiveTemporalVision.pastFrameCount)
        precondition(temporalFeatures.dim(-1) == 2 * width + historyWidth)
        let batch = temporalFeatures.dim(0)
        var fused = concatenated([temporalFeatures[.ellipsis, 0..<width], temporalFeatures[.ellipsis, width..<(2 * width)]], axis: 0)
        if contextSchema.inputValueCount > 0 {
            var context = contextValues ?? MLXArray.zeros(
                [temporalFeatures.dim(0), contextSchema.inputValueCount],
                dtype: dtype
            )
            precondition(
                context.ndim == 2
                    && context.dim(1) == contextSchema.inputValueCount,
                "Context values must match this brain's immutable vocabulary."
            )
            if context.dim(0) != temporalFeatures.dim(0) {
                guard let sampleToVision,
                      sampleToVision.ndim == 1,
                      sampleToVision.dim(0) == temporalFeatures.dim(0) else {
                    preconditionFailure(
                        "Shared context rows require one visual index per supervised sample."
                    )
                }
                context = context.take(sampleToVision, axis: 0)
            }
            let pairedContext = concatenated([context, context], axis: 0).asType(dtype)
            fused = concatenated([fused, pairedContext], axis: -1)
        }
        for (index, layer) in fusion.enumerated() {
            fused = layer(fused)
            if fusionNormalizations.indices.contains(index) {
                fused = fusionNormalizations[index](fused)
            }
            fused = dropout(silu(fused))
        }
        return concatenated([fused[0..<batch], fused[batch..<(2 * batch)], temporalFeatures[.ellipsis, (2 * width)...]], axis: -1)
    }

    var fusionWidth: Int {
        profile.training.architecture.fusionWidths.last ?? (profile.training.architecture.visualEmbedding + profile.training.architecture.recurrentWidth + TemporalInputContract.embeddingWidth + contextSchema.inputValueCount)
    }
    func decisionFeatures(_ features: MLXArray) -> MLXArray { features[.ellipsis, 0..<fusionWidth] }
    private func motorFeatures(_ features: MLXArray) -> MLXArray { features[.ellipsis, fusionWidth..<(2 * fusionWidth)] }

    /// Weights and gain depend on visual/discrete/clock context, never on the
    /// motion being fed back. Convex attention and gain < 0.5 prevent both
    /// amplification and a one-pixel rounding fixed point at sensitivity 1.
    private func relativeLogits(_ raw: MLXArray, fused: MLXArray) -> MLXArray {
        guard !legacyMovementFeedback, profile.training.effectiveMovementHistoryMode == .bounded else { return raw }
        let count = 1 + profile.training.effectiveTemporalVision.pastFrameCount
        let history = fused[.ellipsis, (2 * fusionWidth)...].asType(.float32).reshaped([fused.dim(0), count, 3])
        let motor = motorFeatures(fused)
        let attention = movementHistoryAttention(motor).asType(.float32).reshaped([fused.dim(0), count, 2])
        let valid = history[.ellipsis, 2..<3] .> 0
        let weights = softmax(which(valid, attention, MLXArray(Float(-10_000))), axis: 1)
        var prior = (weights * history[.ellipsis, 0..<2]).sum(axis: 1)
            * (Self.maximumHistoryGain * sigmoid(movementHistoryGain(motor).asType(.float32)))
        if raw.ndim == 3 { prior = prior.expandedDimensions(axis: 1) }
        let combined = clip(tanh(raw.asType(.float32)) + prior, min: -0.9999, max: 0.9999)
        return (0.5 * log((1 + combined) / (1 - combined))).asType(dtype)
    }

    /// History can resolve an ambiguous visual decision but cannot overturn a
    /// confident one by copying a held state. Accumulate the correction in
    /// Float32, including for FP16/BF16 models.
    func boundedControlLogits(base: MLXArray, context: MLXArray) -> MLXArray {
        if legacyControlFeedback { return context }
        let base = base.asType(.float32)
        return (base + Self.maximumControlLogitCorrection * tanh(context.asType(.float32) - base)).asType(dtype)
    }

    func visualActionLogits(fusedFeatures fused: MLXArray) -> MLXArray {
        let base = motorFeatures(fused)
        return concatenated([absoluteMouseHead(base), relativeMouseHead(base), buttonHead(base),
            scrollHead(base), keyboardHead(base), modifierHead(base), movementIntentHead(base)], axis: -1)
    }

    /// The same real action labels supervise the visual/clock base. This is a
    /// training-only cost; it adds no CNN or recurrent pass and no new targets.
    func visualActionLoss(fusedFeatures: MLXArray, targets: MLXArray, positiveWeights: MLXArray?,
                          previousTargets: MLXArray?, sampleWeights: MLXArray?) -> MLXArray {
        guard !legacyControlFeedback else { return MLXArray(0, dtype: dtype) }
        return loss(logits: visualActionLogits(fusedFeatures: fusedFeatures), targets: targets,
            positiveWeights: positiveWeights, previousTargets: previousTargets, sampleWeights: sampleWeights)
    }

    /// The complete immediate training objective, shared by canonical and
    /// optimized benchmark paths. Validation still reports executed-head loss
    /// separately, as does TrainingEngine's per-term telemetry.
    func trainingActionLoss(fusedFeatures: MLXArray, targets: MLXArray,
                            positiveWeights: MLXArray? = nil, previousTargets: MLXArray? = nil,
                            sampleWeights: MLXArray? = nil) -> MLXArray {
        loss(logits: logits(fusedFeatures: fusedFeatures), targets: targets,
            positiveWeights: positiveWeights, previousTargets: previousTargets, sampleWeights: sampleWeights)
            + Self.visualActionLossWeight * visualActionLoss(fusedFeatures: fusedFeatures, targets: targets,
                positiveWeights: positiveWeights, previousTargets: previousTargets, sampleWeights: sampleWeights)
    }

    func trainingActionLoss(currentImages: MLXArray, pastImages: MLXArray, pastControls: MLXArray,
                            targets: MLXArray, positiveWeights: MLXArray? = nil,
                            previousTargets: MLXArray? = nil, currentTiming: MLXArray? = nil,
                            pastTiming: MLXArray? = nil) -> MLXArray {
        let features = temporalFeatures(currentImages: currentImages, pastImages: pastImages,
            pastControls: pastControls, currentTiming: currentTiming, pastTiming: pastTiming)
        return trainingActionLoss(fusedFeatures: fusedFeatures(temporalFeatures: features), targets: targets,
            positiveWeights: positiveWeights, previousTargets: previousTargets)
    }

    func logits(fusedFeatures fused: MLXArray) -> MLXArray {
        let decision = decisionFeatures(fused), motor = motorFeatures(fused)
        return concatenated([
            absoluteMouseHead(motor), relativeLogits(relativeMouseHead(motor), fused: fused),
            boundedControlLogits(base: buttonHead(motor), context: buttonHead(decision)),
            scrollHead(motor),
            boundedControlLogits(base: keyboardHead(motor), context: keyboardHead(decision)),
            boundedControlLogits(base: modifierHead(motor), context: modifierHead(decision)), movementIntentHead(motor)
        ], axis: -1)
    }

    func logits(temporalFeatures: MLXArray) -> MLXArray {
        logits(fusedFeatures: fusedFeatures(temporalFeatures: temporalFeatures))
    }

    func logits(
        temporalFeatures: MLXArray,
        contextValues: MLXArray?,
        sampleToVision: MLXArray? = nil
    ) -> MLXArray {
        logits(fusedFeatures: fusedFeatures(
            temporalFeatures: temporalFeatures,
            contextValues: contextValues,
            sampleToVision: sampleToVision
        ))
    }

    func futureActionLogits(fusedFeatures: MLXArray) -> MLXArray {
        let batch = fusedFeatures.dim(0)
        let streams = concatenated([decisionFeatures(fusedFeatures), motorFeatures(fusedFeatures)], axis: 0)
        let raw = futureActionHead(streams).reshaped([2 * batch,
            FutureActionPredictionConfiguration.maximumHorizonCount, ActionLayout.count])
        let decision = raw[0..<batch], motor = raw[batch..<(2 * batch)]
        let intent = futureMovementIntentHead(motorFeatures(fusedFeatures)).reshaped([batch,
            FutureActionPredictionConfiguration.maximumHorizonCount, 2])
        return concatenated([motor[.ellipsis, 0..<2], relativeLogits(motor[.ellipsis, 2..<4], fused: fusedFeatures),
            boundedControlLogits(base: motor[.ellipsis, 4..<12], context: decision[.ellipsis, 4..<12]),
            motor[.ellipsis, 12..<14], boundedControlLogits(base: motor[.ellipsis, 14...], context: decision[.ellipsis, 14...]), intent], axis: -1)
    }

    func futureStateDelta(
        fusedFeatures: MLXArray,
        currentActions: MLXArray
    ) -> MLXArray {
        futureStateHead(concatenated([
            decisionFeatures(fusedFeatures),
            currentActions.asType(dtype)
        ], axis: -1))
    }

    func logits(
        currentVisualFeatures: MLXArray,
        pastImages: MLXArray,
        pastControls: MLXArray,
        contextValues: MLXArray? = nil,
        currentTiming: MLXArray? = nil,
        pastTiming: MLXArray? = nil
    ) -> MLXArray {
        logits(
            temporalFeatures: temporalFeatures(
                currentVisualFeatures: currentVisualFeatures,
                pastImages: pastImages,
                pastControls: pastControls,
                currentTiming: currentTiming,
                pastTiming: pastTiming
            ),
            contextValues: contextValues
        )
    }

    func callAsFunction(
        currentImages: MLXArray,
        pastImages: MLXArray,
        pastControls: MLXArray,
        contextValues: MLXArray? = nil,
        currentTiming: MLXArray? = nil,
        pastTiming: MLXArray? = nil
    ) -> MLXArray {
        logits(
            temporalFeatures: temporalFeatures(
                currentImages: currentImages,
                pastImages: pastImages,
                pastControls: pastControls,
                currentTiming: currentTiming,
                pastTiming: pastTiming
            ),
            contextValues: contextValues
        )
    }

    func activatedPredictions(logits: MLXArray) -> MLXArray {
        concatenated([
            sigmoid(logits[.ellipsis, ActionLayout.absoluteMouse]),
            tanh(logits[.ellipsis, ActionLayout.relativeMouse]) * movementIntent(logits: logits),
            sigmoid(logits[.ellipsis, ActionLayout.buttons]),
            tanh(logits[.ellipsis, ActionLayout.scroll]),
            sigmoid(logits[.ellipsis, ActionLayout.keyboard]),
            sigmoid(logits[.ellipsis, ActionLayout.modifiers])
        ], axis: -1)
    }

    private func movementIntent(logits: MLXArray) -> MLXArray {
        guard !legacyMovementFeedback, logits.dim(-1) == PolicyLogitsLayout.count else {
            return MLXArray.ones(like: logits[.ellipsis, ActionLayout.relativeMouse])
        }
        let intent = logits[.ellipsis, PolicyLogitsLayout.movementIntent]
        if legacyControlFeedback { return (intent .> 0).asType(dtype) }
        // A hard zero/one switch magnifies tiny low-precision encoder changes
        // into a full-size camera command. Interpolate uncertain evidence;
        // confident stops (<= -1) remain exact zero and turns (>= 1) retain
        // their full magnitude. This adds no queue or delay to a reversal.
        let confidence = clip(0.5 + 0.5 * intent.asType(.float32), min: 0, max: 1)
        return (confidence * confidence * (3 - 2 * confidence)).asType(dtype)
    }

    func predictions(
        currentImages: MLXArray,
        pastImages: MLXArray,
        pastControls: MLXArray,
        contextValues: MLXArray? = nil,
        currentTiming: MLXArray? = nil,
        pastTiming: MLXArray? = nil
    ) -> MLXArray {
        activatedPredictions(logits: callAsFunction(
            currentImages: currentImages,
            pastImages: pastImages,
            pastControls: pastControls,
            contextValues: contextValues,
            currentTiming: currentTiming, pastTiming: pastTiming
        ))
    }

    func currentOnlyPredictions(
        currentImages: MLXArray,
        contextValues: MLXArray? = nil,
        currentTiming: MLXArray? = nil
    ) -> MLXArray {
        activatedPredictions(logits: logits(
            temporalFeatures: currentOnlyTemporalFeatures(currentImages: currentImages, currentTiming: currentTiming),
            contextValues: contextValues
        ))
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
        currentImages: MLXArray,
        pastImages: MLXArray,
        pastControls: MLXArray,
        targets: MLXArray,
        positiveWeights: MLXArray? = nil,
        previousTargets: MLXArray? = nil,
        sampleWeights: MLXArray? = nil,
        currentTiming: MLXArray? = nil, pastTiming: MLXArray? = nil
    ) -> MLXArray {
        let logits = callAsFunction(
            currentImages: currentImages,
            pastImages: pastImages,
            pastControls: pastControls,
            currentTiming: currentTiming, pastTiming: pastTiming
        )
        return loss(
            logits: logits,
            targets: targets,
            positiveWeights: positiveWeights,
            previousTargets: previousTargets,
            sampleWeights: sampleWeights
        )
    }

    /// Computes the training objective from an existing forward pass. Validation
    /// uses this overload so loss and per-head metrics share exactly one policy
    /// evaluation instead of running the convolutional/recurrent stack twice.
    func loss(
        logits: MLXArray,
        targets: MLXArray,
        positiveWeights: MLXArray? = nil,
        previousTargets: MLXArray? = nil,
        sampleWeights: MLXArray? = nil
    ) -> MLXArray {
        let targets = targets.asType(dtype)
        // Transition weighting uses the immediately preceding action target,
        // not one of the more widely spaced frame-aligned context controls.
        let previous = previousTargets?.asType(dtype)
            ?? MLXArray.zeros(like: targets)
        let positiveWeights = positiveWeights?.asType(dtype)
        var losses: [MLXArray] = []
        let channels = profile.channels
        if channels.mouseMovement {
            let absolute = continuousLoss(
                predictions: sigmoid(logits[.ellipsis, ActionLayout.absoluteMouse]),
                targets: targets[.ellipsis, ActionLayout.absoluteMouse],
                sampleWeights: sampleWeights,
                activeWeight: 1
            )
            let relative = activeContinuousLoss(
                predictions: tanh(logits[.ellipsis, ActionLayout.relativeMouse]),
                targets: targets[.ellipsis, ActionLayout.relativeMouse],
                sampleWeights: sampleWeights
            )
            var movement = absolute + relative
            if !legacyMovementFeedback, logits.dim(-1) == PolicyLogitsLayout.count {
                let active = (abs(targets[.ellipsis, ActionLayout.relativeMouse].asType(.float32)) .>= (0.5 / GameCameraContract.deltaScale)).asType(dtype)
                let intentLoss = binaryCrossEntropy(logits: logits[.ellipsis, PolicyLogitsLayout.movementIntent], targets: active, reduction: .none)
                let classWeights = positiveWeights?[.ellipsis, ActionLayout.relativeMouse] ?? MLXArray.ones([2], dtype: dtype)
                var weights = 1 + active * (classWeights - 1)
                var normalization = weights
                if let sampleWeights {
                    let rows = sampleWeights.asType(dtype).reshaped([-1, 1])
                    weights = weights * rows
                    normalization = normalization * (rows .> 0).asType(dtype)
                }
                movement = movement + 0.25 * (intentLoss * weights).sum() / (normalization.sum() + 1e-6)
            }
            losses.append(movement / 2)
        }
        if channels.buttons { losses.append(binaryControlLoss(logits: logits, targets: targets, previous: previous, positiveWeights: positiveWeights, sampleWeights: sampleWeights, range: ActionLayout.buttons)) }
        if channels.scroll { losses.append(activeContinuousLoss(predictions: tanh(logits[.ellipsis, ActionLayout.scroll]), targets: targets[.ellipsis, ActionLayout.scroll], sampleWeights: sampleWeights)) }
        if channels.keyboard { losses.append(binaryControlLoss(logits: logits, targets: targets, previous: previous, positiveWeights: positiveWeights, sampleWeights: sampleWeights, indices: ActionLayout.keyboardAndShiftIndices)) }
        if channels.modifiers { losses.append(binaryControlLoss(logits: logits, targets: targets, previous: previous, positiveWeights: positiveWeights, sampleWeights: sampleWeights, range: ActionLayout.commandOptionControl)) }
        guard let first = losses.first else { return MLXArray(0, dtype: dtype) }
        return losses.dropFirst().reduce(first, +) / Float(losses.count)
    }

    private func binaryControlLoss(logits: MLXArray, targets: MLXArray, previous: MLXArray, positiveWeights: MLXArray?, sampleWeights: MLXArray?, range: Range<Int>) -> MLXArray {
        binaryControlLoss(
            logits: logits[.ellipsis, range],
            targets: targets[.ellipsis, range],
            previous: previous[.ellipsis, range],
            positiveWeights: positiveWeights.map { $0[range] },
            sampleWeights: sampleWeights
        )
    }

    private func binaryControlLoss(logits: MLXArray, targets: MLXArray, previous: MLXArray, positiveWeights: MLXArray?, sampleWeights: MLXArray?, indices: [Int]) -> MLXArray {
        let selection = MLXArray(indices)
        return binaryControlLoss(
            logits: logits[.ellipsis, selection],
            targets: targets[.ellipsis, selection],
            previous: previous[.ellipsis, selection],
            positiveWeights: positiveWeights.map { $0[selection] },
            sampleWeights: sampleWeights
        )
    }

    private func binaryControlLoss(
        logits: MLXArray,
        targets: MLXArray,
        previous: MLXArray,
        positiveWeights: MLXArray?,
        sampleWeights: MLXArray?
    ) -> MLXArray {
        let raw = binaryCrossEntropy(logits: logits, targets: targets, reduction: .none)
        let classWeights: MLXArray
        if let positiveWeights {
            let learnedOutput = (positiveWeights .> 0).asType(dtype)
            classWeights = (1 + targets * (positiveWeights - 1)) * learnedOutput
        } else {
            classWeights = MLXArray.ones(like: targets)
        }
        // Press/release boundaries matter far more than another frame in the
        // middle of a long hold. Upweighting transitions prevents a policy from
        // learning only action persistence.
        let transitionWeights = 1 + 3 * abs(targets - previous)
        let baseWeights = classWeights * transitionWeights * binaryFocalWeights(logits: logits, targets: targets)
        var weights = baseWeights
        var normalizationWeights = baseWeights
        if let sampleWeights {
            let rowWeights = sampleWeights.asType(dtype).reshaped([-1, 1])
            weights = weights * rowWeights
            // Masked rows are padding and contribute no denominator.
            normalizationWeights = normalizationWeights * (rowWeights .> 0).asType(dtype)
        }
        return (raw * weights).sum() / (normalizationWeights.sum() + 1e-6)
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
        sampleWeights: MLXArray? = nil
    ) -> MLXArray {
        continuousLoss(
            predictions: predictions,
            targets: targets,
            sampleWeights: sampleWeights,
            activeWeight: 8
        )
    }

    private func continuousLoss(
        predictions: MLXArray,
        targets: MLXArray,
        sampleWeights: MLXArray?,
        activeWeight: Float
    ) -> MLXArray {
        let raw = smoothL1Loss(predictions: predictions, targets: targets, beta: 0.05, reduction: .none)
        let baseWeights = activeWeight > 1
            ? which(abs(targets) .> 0.0001, MLXArray(activeWeight, dtype: dtype), MLXArray(1, dtype: dtype))
            : MLXArray.ones(like: targets)
        var weights = baseWeights
        var normalizationWeights = baseWeights
        if let sampleWeights {
            let rowWeights = sampleWeights.asType(dtype).reshaped([-1, 1])
            weights = weights * rowWeights
            normalizationWeights = normalizationWeights * (rowWeights .> 0).asType(dtype)
        }
        return (raw * weights).sum() / (normalizationWeights.sum() + 1e-6)
    }

    func futureActionLoss(
        fusedFeatures: MLXArray,
        futureTargets: MLXArray,
        futurePreviousTargets: MLXArray,
        masks: MLXArray,
        trainingSampleWeights: MLXArray,
        positiveWeights: MLXArray
    ) -> MLXArray {
        let configuration = profile.training.effectiveFutureActionPrediction
        guard configuration.enabled, configuration.lossWeight > 0,
              !configuration.horizonsSeconds.isEmpty else {
            return MLXArray(0, dtype: dtype)
        }
        let horizonCount = configuration.horizonsSeconds.count
        precondition(
            futureTargets.ndim == 3 && futureTargets.dim(1) == horizonCount
                && futureTargets.dim(2) == ActionLayout.count,
            "Future action targets must match the configured horizons."
        )
        let allLogits = futureActionLogits(fusedFeatures: fusedFeatures)
        var weightedLosses: [MLXArray] = []
        var totalWeight: Float = 0
        for horizon in 0..<horizonCount {
            // Near intent matters most for smooth behavior while farther rows
            // still teach planning. The immediate policy is never blended with
            // these predictions at runtime, preserving reactivity.
            let horizonWeight = 1 / Float(horizon + 1)
            let rowWeights = masks[0..., horizon] * trainingSampleWeights
            weightedLosses.append(horizonWeight * loss(
                logits: allLogits[0..., horizon, 0...],
                targets: futureTargets[0..., horizon, 0...],
                positiveWeights: positiveWeights,
                previousTargets: futurePreviousTargets[0..., horizon, 0...],
                sampleWeights: rowWeights
            ))
            totalWeight += horizonWeight
        }
        guard let first = weightedLosses.first else { return MLXArray(0, dtype: dtype) }
        return weightedLosses.dropFirst().reduce(first, +) / max(0.0001, totalWeight)
    }

    func futureStateLoss(
        fusedFeatures: MLXArray,
        temporalFeatures: MLXArray,
        currentActions: MLXArray,
        futureImages: MLXArray,
        sampleToFuture: MLXArray,
        masks: MLXArray,
        trainingSampleWeights: MLXArray
    ) -> MLXArray {
        let configuration = profile.training.effectiveFutureStatePrediction
        guard configuration.enabled, configuration.lossWeight > 0 else {
            return MLXArray(0, dtype: dtype)
        }
        let uniqueFutureEmbedding = visualEmbedding(
            visualFeatures: visualActivations(images: futureImages).last!
        )
        let futureEmbedding = uniqueFutureEmbedding.take(sampleToFuture, axis: 0)
        let width = profile.training.architecture.visualEmbedding
        let currentEmbedding = temporalFeatures[.ellipsis, 0..<width]
        let targetDelta = stopGradient(futureEmbedding - currentEmbedding)
        let predictedDelta = futureStateDelta(
            fusedFeatures: fusedFeatures,
            currentActions: currentActions
        )
        let raw = smoothL1Loss(
            predictions: predictedDelta,
            targets: targetDelta,
            beta: 0.1,
            reduction: .none
        ).mean(axis: -1)
        // Give visibly changing states more influence without allowing a large
        // latent outlier to dominate the immediate behavior objective.
        let change = clip(abs(targetDelta).mean(axis: -1), min: 0, max: 1)
        let baseWeights = masks.asType(dtype) * (1 + 3 * change)
        let weights = baseWeights * trainingSampleWeights.asType(dtype)
        return (raw * weights).sum() / (baseWeights.sum() + 1e-6)
    }

    func saveWeights(to url: URL) throws {
        guard !legacyControlFeedback else { throw AgentTrainerError.model("A diagnostic legacy policy cannot publish current-format weights.") }
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: parameters().flattened()), metadata: ["format": ModelContract.weightFormat], url: url)
    }

    /// Widen context projections without changing any previously learned column.
    func loadCompatibleWeights(from url: URL, sourceContext: ModelContextSchema) throws {
        if sourceContext.hasSameLayout(as: contextSchema) { try loadWeights(from: url); return }
        let (source, metadata) = try MLX.loadArraysAndMetadata(url: url)
        guard metadata["format"] == ModelContract.weightFormat else { throw AgentTrainerError.model("Incompatible policy weight contract.") }
        let oldIDs = sourceContext.fields.flatMap { field in field.values.map { field.id.uuidString + ":" + $0.id.uuidString } }
        let newIDs = contextSchema.fields.flatMap { field in field.values.map { field.id.uuidString + ":" + $0.id.uuidString } }
        guard Set(oldIDs).isSubset(of: Set(newIDs)) else {
            throw AgentTrainerError.model("The new context vocabulary would remove learned values. The original brain was preserved.")
        }
        var result: [String: MLXArray] = [:]
        for (name, parameter) in parameters().flattened() {
            guard let original = source[name] else { throw AgentTrainerError.model("Missing pretrained parameter: \(name)") }
            let projection = name == "fusion.0.weight" || (fusion.isEmpty && name.hasSuffix(".weight")
                && original.ndim == 2 && original.dim(1) - oldIDs.count == parameter.dim(1) - newIDs.count)
            if projection {
                let tail = name == "futureStateHead.weight" ? ActionLayout.count : 0
                let prefix = original.dim(1) - oldIDs.count - tail
                guard prefix >= 0, parameter.dim(0) == original.dim(0),
                      parameter.dim(1) == prefix + newIDs.count + tail else { throw AgentTrainerError.model("Incompatible context projection.") }
                var columns = [original[0..., 0..<prefix]]
                for id in newIDs {
                    if let old = oldIDs.firstIndex(of: id) { columns.append(original[0..., (prefix + old)..<(prefix + old + 1)]) }
                    else { columns.append(MLXArray.zeros([original.dim(0), 1], dtype: original.dtype)) }
                }
                if tail > 0 { columns.append(original[0..., (original.dim(1) - tail)...]) }
                result[name] = concatenated(columns, axis: 1).asType(parameter.dtype)
            } else {
                guard original.shape == parameter.shape else { throw AgentTrainerError.model("Incompatible pretrained parameter: \(name)") }
                result[name] = original.asType(parameter.dtype)
            }
        }
        try update(parameters: ModuleParameters.unflattened(result), verify: .all)
    }

    func loadWeights(from url: URL) throws {
        let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: url)
        let expectedFormat = legacyMovementFeedback ? "AgentTrainer.Policy.v7"
            : legacyControlFeedback ? "AgentTrainer.Policy.v8" : ModelContract.weightFormat
        guard metadata["format"] == expectedFormat else {
            throw AgentTrainerError.model("These weights use an incompatible policy contract. Retrain from the preserved recordings.")
        }
        let expected = Dictionary(uniqueKeysWithValues: parameters().flattened())
        let converted = Dictionary(uniqueKeysWithValues: arrays.map { key, value in
            (key, value.asType(expected[key]?.dtype ?? value.dtype))
        })
        try update(parameters: ModuleParameters.unflattened(converted), verify: .all)
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
    private var masterWeights: [String: MLXArray] = [:]
    private(set) var usesMasterWeights: Bool
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
        minimumLearningRateRatio: Float = 0.05,
        usesMasterWeights: Bool = true
    ) {
        self.usesMasterWeights = usesMasterWeights
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
            if usesMasterWeights { masterWeights[name] = value }
        }
    }

    /// Warm-start weights can be loaded after moment allocation. Seed the new
    /// accumulator from those selected weights before tracing the first update.
    func synchronizeInitialParameters(model: Module) {
        precondition(step == 0, "Only a fresh optimizer may reseed master weights.")
        guard usesMasterWeights else { return }
        masterWeights = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map {
            ($0.0, $0.1.asType(.float32))
        })
    }

    func update(
        model: Module,
        gradients: ModuleParameters,
        targetType: DType,
        gradientNorm: MLXArray? = nil,
        maxGradientNorm: Float? = nil,
        preserveParameterTypes: Bool = false
    ) {
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
        // These scalars are shared by every parameter. Hoisting them keeps one
        // bias-correction/decay subgraph instead of tracing identical pow and
        // multiply nodes for each tensor in the model.
        let correction1 = 1 - pow(beta1, stepArray)
        let correction2 = 1 - pow(beta2, stepArray)
        let decayScale = 1 - scheduledLearningRate * weightDecay
        let gradientMap = Dictionary(uniqueKeysWithValues: gradients.flattened())
        let clipping: (condition: MLXArray, normalizer: MLXArray)?
        if let gradientNorm, let maxGradientNorm {
            clipping = (
                gradientNorm .< maxGradientNorm,
                maxGradientNorm / (gradientNorm + 1e-6)
            )
        } else {
            clipping = nil
        }
        var updated: [(String, MLXArray)] = []
        for (name, parameter) in model.parameters().flattened() {
            guard let gradient = gradientMap[name] else { updated.append((name, parameter)); continue }
            let p = masterWeights[name] ?? parameter.asType(.float32)
            let clipped = clipping.map {
                which($0.condition, gradient, gradient * $0.normalizer)
            } ?? gradient
            let g = clipped.asType(.float32)
            let m = beta1 * (firstMoments[name] ?? MLXArray.zeros(like: p)) + (1 - beta1) * g
            let v = beta2 * (secondMoments[name] ?? MLXArray.zeros(like: p)) + (1 - beta2) * square(g)
            firstMoments[name] = m
            secondMoments[name] = v
            let update = (m / correction1) / (sqrt(v / correction2) + epsilon)
            let next = p * decayScale - scheduledLearningRate * update
            if usesMasterWeights { masterWeights[name] = next }
            updated.append((name, next.asType(preserveParameterTypes ? parameter.dtype : targetType)))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
    }

    /// Mirrors MLXOptimizers' canonical global norm calculation, but returns
    /// only its shared scalar, avoiding a temporary clipped parameter tree.
    static func globalGradientNorm(_ gradients: ModuleParameters) -> MLXArray {
        sqrt(gradients.reduce(MLXArray(Float(0))) { $0 + $1.asType(.float32).square().sum() })
    }

    func save(to url: URL) throws {
        MLX.eval(innerState())
        var arrays: [String: MLXArray] = [:]
        for (key, value) in firstMoments { arrays["m.\(key)"] = value }
        for (key, value) in secondMoments { arrays["v.\(key)"] = value }
        for (key, value) in masterWeights { arrays["master.\(key)"] = value }
        try MLX.save(
            arrays: arrays,
            metadata: [
                "optimizerFormat": "AgentTrainer.AdamW.v2",
                "usesMasterWeights": String(usesMasterWeights),
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
        [stepArray, learningRateScaleArray] + parameterNames.flatMap { name in [firstMoments[name], secondMoments[name], masterWeights[name]].compactMap { $0 } }
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
        let restoredFirst = Dictionary(uniqueKeysWithValues: loaded.0.compactMap { key, value in key.hasPrefix("m.") ? (String(key.dropFirst(2)), value) : nil })
        let restoredSecond = Dictionary(uniqueKeysWithValues: loaded.0.compactMap { key, value in key.hasPrefix("v.") ? (String(key.dropFirst(2)), value) : nil })
        let restoredMaster = Dictionary(uniqueKeysWithValues: loaded.0.compactMap { key, value in
            key.hasPrefix("master.") ? (String(key.dropFirst(7)), value) : nil
        })
        if let format = loaded.1["optimizerFormat"], format != "AgentTrainer.AdamW.v2" {
            throw AgentTrainerError.model("The optimizer checkpoint uses an unsupported format.")
        }
        let restoredUsesMaster: Bool
        if let flag = loaded.1["usesMasterWeights"] {
            guard let parsed = Bool(flag) else { throw AgentTrainerError.model("Invalid master-weight metadata.") }
            restoredUsesMaster = parsed
        } else {
            // Preserve the exact rounded-update trajectory of older checkpoints.
            restoredUsesMaster = false
        }
        let restoredNames = Set(restoredFirst.keys)
        let expectedNames = Set(parameterNames)
        let expectedShapes = firstMoments.mapValues(\.shape)
        guard !restoredNames.isEmpty,
              restoredNames == Set(restoredSecond.keys),
              restoredUsesMaster ? Set(restoredMaster.keys) == restoredNames : restoredMaster.isEmpty,
              restoredNames.allSatisfy({ !restoredUsesMaster ||
                  (restoredMaster[$0]?.shape == restoredFirst[$0]?.shape && restoredMaster[$0]?.dtype == .float32) }),
              expectedNames.isEmpty || restoredNames == expectedNames,
              restoredNames.allSatisfy({
                  restoredFirst[$0]?.shape == restoredSecond[$0]?.shape
                      && (expectedShapes[$0] == nil || restoredFirst[$0]?.shape == expectedShapes[$0])
              }) else {
            throw AgentTrainerError.model("The optimizer checkpoint is incomplete or does not match this model.")
        }
        firstMoments = restoredFirst
        secondMoments = restoredSecond
        masterWeights = restoredMaster
        usesMasterWeights = restoredUsesMaster
        parameterNames = restoredNames.sorted()
        guard let restoredStep = loaded.1["step"].flatMap(Int.init), restoredStep >= 0,
              let restoredLearningRate = loaded.1["learningRate"].flatMap(Float.init),
              restoredLearningRate.isFinite, restoredLearningRate > 0,
              let restoredWeightDecay = loaded.1["weightDecay"].flatMap(Float.init),
              restoredWeightDecay.isFinite, restoredWeightDecay >= 0 else {
            throw AgentTrainerError.model("The optimizer checkpoint contains invalid core metadata.")
        }
        stepArray = MLXArray(Float(restoredStep), dtype: .float32)
        learningRate = restoredLearningRate
        weightDecay = restoredWeightDecay
        // Checkpoints from the fixed-schedule trainer did not save this field;
        // retaining 500 preserves their exact continuation behavior.
        if let rawWarmup = loaded.1["warmupSteps"] {
            guard let restoredWarmup = Int(rawWarmup), restoredWarmup > 0 else {
                throw AgentTrainerError.model("The optimizer checkpoint contains an invalid warmup length.")
            }
            warmupSteps = restoredWarmup
        } else {
            warmupSteps = 500
        }
        if let rawSchedule = loaded.1["schedule"] {
            guard let restoredSchedule = LearningRateSchedule(rawValue: rawSchedule) else {
                throw AgentTrainerError.model("The optimizer checkpoint contains an unknown learning-rate schedule.")
            }
            schedule = restoredSchedule
        } else {
            schedule = .legacyInverseSquareRoot
        }
        if let rawCycle = loaded.1["cycleSteps"] {
            guard let restoredCycle = Int(rawCycle), restoredCycle > 0 else {
                throw AgentTrainerError.model("The optimizer checkpoint contains an invalid cosine cycle.")
            }
            cycleSteps = restoredCycle
        } else {
            cycleSteps = 4_000
        }
        if let rawRatio = loaded.1["minimumLearningRateRatio"] {
            guard let restoredRatio = Float(rawRatio), restoredRatio.isFinite, (0.001...0.5).contains(restoredRatio) else {
                throw AgentTrainerError.model("The optimizer checkpoint contains an invalid minimum learning-rate ratio.")
            }
            minimumLearningRateRatio = restoredRatio
        } else {
            minimumLearningRateRatio = 0.05
        }
        if let rawScale = loaded.1["learningRateScale"] {
            guard let restoredScale = Float(rawScale), restoredScale.isFinite,
                  (minimumLearningRateRatio...1).contains(restoredScale) else {
                throw AgentTrainerError.model("The optimizer checkpoint contains an invalid learning-rate scale.")
            }
            learningRateScaleArray = MLXArray(restoredScale, dtype: .float32)
        } else {
            learningRateScaleArray = MLXArray(1, dtype: .float32)
        }
    }
}
