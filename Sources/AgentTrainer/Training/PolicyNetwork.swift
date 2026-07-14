import Foundation
import MLX
import MLXNN

final class AgentPolicy: Module, @unchecked Sendable {
    let profile: AIProfile
    let dtype: DType
    let coordinateGrid: MLXArray

    @ModuleInfo var convolutions: [Conv2d]
    @ModuleInfo var visualProjection: Linear
    @ModuleInfo var gru: GRU?
    @ModuleInfo var lstm: LSTM?
    @ModuleInfo var fusion: [Linear]
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
        coordinateGrid = concatenated([x, y], axis: -1).asType(dtype)
        var inputChannels = profile.preprocessing.channelCount + 2
        var convs: [Conv2d] = []
        for i in architecture.convolutionChannels.indices {
            let output = architecture.convolutionChannels[i]
            let kernel = architecture.kernelSizes.indices.contains(i) ? max(1, architecture.kernelSizes[i]) : 3
            let stride = architecture.strides.indices.contains(i) ? max(1, architecture.strides[i]) : 2
            convs.append(Conv2d(inputChannels: inputChannels, outputChannels: output, kernelSize: .init(kernel), stride: .init(stride), padding: .init(kernel / 2)))
            inputChannels = output
        }
        convolutions = convs
        visualProjection = Linear(max(1, inputChannels), architecture.visualEmbedding)
        if architecture.recurrentKind == .gru {
            gru = GRU(inputSize: ActionLayout.count, hiddenSize: architecture.recurrentWidth)
            lstm = nil
        } else {
            gru = nil
            lstm = LSTM(inputSize: ActionLayout.count, hiddenSize: architecture.recurrentWidth)
        }
        var fusionLayers: [Linear] = []
        var fusionInput = architecture.visualEmbedding + architecture.recurrentWidth
        for width in architecture.fusionWidths {
            fusionLayers.append(Linear(fusionInput, max(1, width)))
            fusionInput = max(1, width)
        }
        fusion = fusionLayers
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

    /// Returns every post-ReLU spatial stage without changing the normal policy
    /// graph or its saved parameters. Runtime diagnostics consume these tensors
    /// only when explicitly enabled.
    func visualActivations(images: MLXArray) -> [MLXArray] {
        var vision = images.asType(dtype)
        let coordinates = broadcast(coordinateGrid, to: [images.dim(0), profile.preprocessing.height, profile.preprocessing.width, 2])
        vision = concatenated([vision, coordinates], axis: -1)
        var activations: [MLXArray] = []
        activations.reserveCapacity(max(1, convolutions.count))
        for convolution in convolutions {
            vision = relu(convolution(vision))
            activations.append(vision)
        }
        // A convolution-free custom architecture is still a valid tensor graph.
        // Treat its coordinate-aware input as the only visual stage.
        if activations.isEmpty { activations.append(vision) }
        return activations
    }

    func logits(visualFeatures: MLXArray, history: MLXArray) -> MLXArray {
        var vision = visualFeatures.mean(axes: [1, 2])
        vision = relu(visualProjection(vision))

        let history = history.asType(dtype)
        let recurrent: MLXArray
        if let gru {
            recurrent = gru(history)[.ellipsis, -1, 0...]
        } else if let lstm {
            recurrent = lstm(history).0[.ellipsis, -1, 0...]
        } else {
            recurrent = MLXArray.zeros([visualFeatures.dim(0), profile.training.architecture.recurrentWidth], dtype: dtype)
        }
        var fused = concatenated([vision, recurrent], axis: -1)
        for layer in fusion { fused = dropout(relu(layer(fused))) }
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

    func loss(images: MLXArray, history: MLXArray, targets: MLXArray) -> MLXArray {
        let logits = callAsFunction(images: images, history: history)
        let targets = targets.asType(dtype)
        var losses: [MLXArray] = []
        let channels = profile.channels
        if channels.mouseMovement {
            let absolute = mseLoss(predictions: sigmoid(logits[.ellipsis, ActionLayout.absoluteMouse]), targets: targets[.ellipsis, ActionLayout.absoluteMouse])
            let relative = mseLoss(predictions: tanh(logits[.ellipsis, ActionLayout.relativeMouse]), targets: targets[.ellipsis, ActionLayout.relativeMouse])
            losses.append((absolute + relative) / 2)
        }
        if channels.buttons { losses.append(binaryCrossEntropy(logits: logits[.ellipsis, ActionLayout.buttons], targets: targets[.ellipsis, ActionLayout.buttons])) }
        if channels.scroll { losses.append(mseLoss(predictions: tanh(logits[.ellipsis, ActionLayout.scroll]), targets: targets[.ellipsis, ActionLayout.scroll])) }
        if channels.keyboard { losses.append(binaryCrossEntropy(logits: logits[.ellipsis, ActionLayout.keyboard], targets: targets[.ellipsis, ActionLayout.keyboard])) }
        if channels.modifiers { losses.append(binaryCrossEntropy(logits: logits[.ellipsis, ActionLayout.modifiers], targets: targets[.ellipsis, ActionLayout.modifiers])) }
        guard let first = losses.first else { return MLXArray(0, dtype: dtype) }
        return losses.dropFirst().reduce(first, +) / Float(losses.count)
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
    let beta1: Float = 0.9
    let beta2: Float = 0.999
    let epsilon: Float = 1e-8
    private var stepArray = MLXArray(0, dtype: .float32)
    private var firstMoments: [String: MLXArray] = [:]
    private var secondMoments: [String: MLXArray] = [:]
    private var parameterNames: [String] = []

    var step: Int { MLX.eval(stepArray); return Int(stepArray.item(Float.self).rounded()) }

    init(learningRate: Float, weightDecay: Float) {
        self.learningRate = learningRate
        self.weightDecay = weightDecay
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
            updated.append((name, (p * (1 - learningRate * weightDecay) - learningRate * update).asType(targetType)))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
    }

    func save(to url: URL) throws {
        MLX.eval(innerState())
        var arrays: [String: MLXArray] = [:]
        for (key, value) in firstMoments { arrays["m.\(key)"] = value }
        for (key, value) in secondMoments { arrays["v.\(key)"] = value }
        try MLX.save(arrays: arrays, metadata: ["step": String(step), "learningRate": String(learningRate), "weightDecay": String(weightDecay)], url: url)
    }

    func stateArrays() -> [MLXArray] {
        innerState()
    }

    func innerState() -> [MLXArray] {
        [stepArray] + parameterNames.flatMap { name in [firstMoments[name], secondMoments[name]].compactMap { $0 } }
    }

    func load(from url: URL) throws {
        let loaded = try MLX.loadArraysAndMetadata(url: url)
        firstMoments = Dictionary(uniqueKeysWithValues: loaded.0.compactMap { key, value in key.hasPrefix("m.") ? (String(key.dropFirst(2)), value) : nil })
        secondMoments = Dictionary(uniqueKeysWithValues: loaded.0.compactMap { key, value in key.hasPrefix("v.") ? (String(key.dropFirst(2)), value) : nil })
        parameterNames = firstMoments.keys.sorted()
        stepArray = MLXArray(Float(Int(loaded.1["step"] ?? "0") ?? 0), dtype: .float32)
        learningRate = Float(loaded.1["learningRate"] ?? "") ?? learningRate
        weightDecay = Float(loaded.1["weightDecay"] ?? "") ?? weightDecay
    }
}
