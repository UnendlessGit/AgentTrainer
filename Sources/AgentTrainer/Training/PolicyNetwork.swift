import Foundation
import MLX
import MLXNN
import MLXRandom

final class AgentPolicy: Module, @unchecked Sendable {
    let profile: AIProfile
    let dtype: DType
    // Leading underscore deliberately keeps this deterministic tensor out of
    // MLX Module reflection. It is an input buffer, not a learned parameter;
    // putting it in AdamW would waste two Float32 moment arrays and allow
    // optimization to distort the meaning of X/Y coordinates.
    private let _coordinateGrid: MLXArray
    private let _poolCoordinates: MLXArray?

    @ModuleInfo var convolutions: [Conv2d]
    @ModuleInfo var convolutionNormalizations: [GroupNorm]
    @ModuleInfo var spatialAttention: Linear?
    @ModuleInfo var visualProjection: Linear
    @ModuleInfo var visualNormalization: LayerNorm
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
        // Current planes, signed temporal differences, and explicit X/Y.
        var inputChannels = profile.preprocessing.channelCount * 2 + 2
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
        let visualProjectionInput: Int
        if architecture.effectiveVisualPooling == .attention {
            let heads = architecture.effectiveAttentionHeads
            spatialAttention = Linear(inputChannels, heads)
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
            visualProjectionInput = heads * (inputChannels + 2) + 2 * inputChannels
        } else {
            spatialAttention = nil
            _poolCoordinates = nil
            visualProjectionInput = max(1, visualSize.width * visualSize.height * inputChannels)
        }
        visualProjection = Linear(visualProjectionInput, architecture.visualEmbedding)
        visualNormalization = LayerNorm(dimensions: architecture.visualEmbedding)
        if architecture.recurrentKind == .gru {
            gru = GRU(inputSize: ActionLayout.count, hiddenSize: architecture.recurrentWidth)
            lstm = nil
        } else {
            gru = nil
            lstm = LSTM(inputSize: ActionLayout.count, hiddenSize: architecture.recurrentWidth)
        }
        var fusionLayers: [Linear] = []
        var fusionNormalizations: [LayerNorm] = []
        var fusionInput = architecture.visualEmbedding + architecture.recurrentWidth
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

    /// Returns every normalized post-SiLU spatial stage without changing the
    /// normal policy graph or its saved parameters. Runtime diagnostics consume
    /// these tensors only when explicitly enabled.
    func visualActivations(images: MLXArray) -> [MLXArray] {
        visualActivations(
            images: images,
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
                    coordinates: _coordinateGrid,
                    convolution: convolution
                )
            } else {
                if index == 0 {
                    let coordinates = broadcast(
                        _coordinateGrid,
                        to: [images.dim(0), profile.preprocessing.height, profile.preprocessing.width, 2]
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
                _coordinateGrid,
                to: [images.dim(0), profile.preprocessing.height, profile.preprocessing.width, 2]
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
        let batch = visualFeatures.dim(0)
        var vision: MLXArray
        if let spatialAttention, let poolCoordinates = _poolCoordinates {
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
        return silu(visualNormalization(visualProjection(vision)))
    }

    func logits(visualFeatures: MLXArray, history: MLXArray) -> MLXArray {
        logits(
            visualEmbedding: visualEmbedding(visualFeatures: visualFeatures),
            history: history
        )
    }

    func logits(visualEmbedding vision: MLXArray, history: MLXArray) -> MLXArray {
        let batch = vision.dim(0)

        var history = history.asType(dtype)
        // Ground-truth action history is an exceptionally tempting shortcut:
        // a model can copy the previous held key and achieve a tiny loss while
        // ignoring the screen. Drop the complete history branch for half of
        // every training batch independently of ordinary feature dropout, so a
        // user cannot reopen that shortcut by setting Dropout to zero.
        // Inference still gets normal history, but vision must be independently
        // predictive.
        if training, profile.training.historyLength > 0 {
            let keepProbability: Float = 0.5
            let mask = MLXRandom.bernoulli(keepProbability, [batch, 1, 1]).asType(dtype)
            // This is structured branch masking rather than ordinary inverted
            // dropout. Kept histories retain their inference-time magnitude;
            // doubling them would create a train/run distribution mismatch.
            history = history * mask
        }
        let recurrent: MLXArray
        if let gru {
            recurrent = gru(history)[.ellipsis, -1, 0...]
        } else if let lstm {
            recurrent = lstm(history).0[.ellipsis, -1, 0...]
        } else {
            recurrent = MLXArray.zeros(
                [batch, profile.training.architecture.recurrentWidth],
                dtype: dtype
            )
        }
        var fused = concatenated([vision, recurrent], axis: -1)
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

    /// Computes the training objective from an existing forward pass. Validation
    /// uses this overload so loss and per-head metrics share exactly one policy
    /// evaluation instead of running the convolutional/recurrent stack twice.
    func loss(
        logits: MLXArray,
        history: MLXArray,
        targets: MLXArray,
        positiveWeights: MLXArray? = nil,
        previousTargets: MLXArray? = nil
    ) -> MLXArray {
        let targets = targets.asType(dtype)
        // Training passes the actual preceding cached action explicitly. This
        // remains correct when model history is disabled; using the mandatory
        // zero placeholder row incorrectly marked every held control as a new
        // transition and upweighted long holds fourfold.
        let previous = previousTargets?.asType(dtype)
            ?? history.asType(dtype)[.ellipsis, -1, 0...]
        let positiveWeights = positiveWeights?.asType(dtype)
        var losses: [MLXArray] = []
        let channels = profile.channels
        if channels.mouseMovement {
            let absolute = smoothL1Loss(predictions: sigmoid(logits[.ellipsis, ActionLayout.absoluteMouse]), targets: targets[.ellipsis, ActionLayout.absoluteMouse], beta: 0.05)
            let relative = activeContinuousLoss(predictions: tanh(logits[.ellipsis, ActionLayout.relativeMouse]), targets: targets[.ellipsis, ActionLayout.relativeMouse])
            losses.append((absolute + relative) / 2)
        }
        if channels.buttons { losses.append(binaryControlLoss(logits: logits, targets: targets, previous: previous, positiveWeights: positiveWeights, range: ActionLayout.buttons)) }
        if channels.scroll { losses.append(activeContinuousLoss(predictions: tanh(logits[.ellipsis, ActionLayout.scroll]), targets: targets[.ellipsis, ActionLayout.scroll])) }
        if channels.keyboard { losses.append(binaryControlLoss(logits: logits, targets: targets, previous: previous, positiveWeights: positiveWeights, indices: ActionLayout.keyboardAndShiftIndices)) }
        if channels.modifiers { losses.append(binaryControlLoss(logits: logits, targets: targets, previous: previous, positiveWeights: positiveWeights, range: ActionLayout.commandOptionControl)) }
        guard let first = losses.first else { return MLXArray(0, dtype: dtype) }
        return losses.dropFirst().reduce(first, +) / Float(losses.count)
    }

    private func binaryControlLoss(logits: MLXArray, targets: MLXArray, previous: MLXArray, positiveWeights: MLXArray?, range: Range<Int>) -> MLXArray {
        binaryControlLoss(
            logits: logits[.ellipsis, range],
            targets: targets[.ellipsis, range],
            previous: previous[.ellipsis, range],
            positiveWeights: positiveWeights.map { $0[range] }
        )
    }

    private func binaryControlLoss(logits: MLXArray, targets: MLXArray, previous: MLXArray, positiveWeights: MLXArray?, indices: [Int]) -> MLXArray {
        let selection = MLXArray(indices)
        return binaryControlLoss(
            logits: logits[.ellipsis, selection],
            targets: targets[.ellipsis, selection],
            previous: previous[.ellipsis, selection],
            positiveWeights: positiveWeights.map { $0[selection] }
        )
    }

    private func binaryControlLoss(
        logits: MLXArray,
        targets: MLXArray,
        previous: MLXArray,
        positiveWeights: MLXArray?
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
        let weights = classWeights * transitionWeights * binaryFocalWeights(logits: logits, targets: targets)
        return (raw * weights).sum() / (weights.sum() + 1e-6)
    }

    private func binaryFocalWeights(logits: MLXArray, targets: MLXArray) -> MLXArray {
        let gamma = profile.training.effectiveBinaryFocalGamma
        guard gamma > 0 else { return MLXArray.ones(like: targets) }
        // |target - probability| is p_t's complement. Raising it focuses the
        // already class-balanced loss on mistakes and ambiguous transitions
        // instead of spending most updates on easy idle negatives.
        return pow(abs(targets - sigmoid(logits)), Float(gamma))
    }

    private func activeContinuousLoss(predictions: MLXArray, targets: MLXArray) -> MLXArray {
        let raw = smoothL1Loss(predictions: predictions, targets: targets, beta: 0.05, reduction: .none)
        let weights = which(abs(targets) .> 0.0001, MLXArray(8, dtype: dtype), MLXArray(1, dtype: dtype))
        return (raw * weights).sum() / (weights.sum() + 1e-6)
    }

    func saveWeights(to url: URL) throws {
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: parameters().flattened()), metadata: ["format": ModelContract.weightFormat], url: url)
    }

    func loadWeights(from url: URL) throws {
        let arrays = try MLX.loadArrays(url: url)
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

    func update(
        model: Module,
        gradients: ModuleParameters,
        targetType: DType,
        gradientNorm: MLXArray? = nil,
        maxGradientNorm: Float? = nil
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
            let p = parameter.asType(.float32)
            let clipped = clipping.map {
                which($0.condition, gradient, gradient * $0.normalizer)
            } ?? gradient
            let g = clipped.asType(.float32)
            let m = beta1 * (firstMoments[name] ?? MLXArray.zeros(like: p)) + (1 - beta1) * g
            let v = beta2 * (secondMoments[name] ?? MLXArray.zeros(like: p)) + (1 - beta2) * square(g)
            firstMoments[name] = m
            secondMoments[name] = v
            let update = (m / correction1) / (sqrt(v / correction2) + epsilon)
            updated.append((name, (p * decayScale - scheduledLearningRate * update).asType(targetType)))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
    }

    /// Mirrors MLXOptimizers' canonical global norm calculation, but returns
    /// only its shared scalar, avoiding a temporary clipped parameter tree.
    static func globalGradientNorm(_ gradients: ModuleParameters) -> MLXArray {
        sqrt(gradients.reduce(MLXArray(0)) { $0 + $1.square().sum() })
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
        let restoredFirst = Dictionary(uniqueKeysWithValues: loaded.0.compactMap { key, value in key.hasPrefix("m.") ? (String(key.dropFirst(2)), value) : nil })
        let restoredSecond = Dictionary(uniqueKeysWithValues: loaded.0.compactMap { key, value in key.hasPrefix("v.") ? (String(key.dropFirst(2)), value) : nil })
        let restoredNames = Set(restoredFirst.keys)
        let expectedNames = Set(parameterNames)
        let expectedShapes = firstMoments.mapValues(\.shape)
        guard !restoredNames.isEmpty,
              restoredNames == Set(restoredSecond.keys),
              expectedNames.isEmpty || restoredNames == expectedNames,
              restoredNames.allSatisfy({
                  restoredFirst[$0]?.shape == restoredSecond[$0]?.shape
                      && (expectedShapes[$0] == nil || restoredFirst[$0]?.shape == expectedShapes[$0])
              }) else {
            throw AgentTrainerError.model("The optimizer checkpoint is incomplete or does not match this model.")
        }
        firstMoments = restoredFirst
        secondMoments = restoredSecond
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
