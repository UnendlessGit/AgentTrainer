import ATCore
import ATLanguage
import ATModel
import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// Runs one training job end to end. Blocking; drive it from a background thread.
public final class Trainer: @unchecked Sendable {
    public let request: TrainingRequest
    private let emit: (TrainingEvent) -> Void
    private let encoderFactory: () -> InstructionEncoder?
    private let control = NSCondition()
    private var pauseRequested = false
    private var stopRequested = false
    private var checkpointRequested = false

    public init(request: TrainingRequest, encoderFactory: @escaping () -> InstructionEncoder?, emit: @escaping (TrainingEvent) -> Void) {
        self.request = request
        self.encoderFactory = encoderFactory
        self.emit = emit
    }

    public func pause() { control.lock(); pauseRequested = true; control.unlock() }
    public func resume() { control.lock(); pauseRequested = false; control.broadcast(); control.unlock() }
    public func stop() { control.lock(); stopRequested = true; control.broadcast(); control.unlock() }
    public func checkpointNow() { control.lock(); checkpointRequested = true; control.unlock() }

    private func log(_ s: String) { emit(.log(s)) }

    public func run() {
        do {
            try train()
        } catch is CancellationError {
            emit(.phase(.stopped, "Stopped"))
        } catch {
            emit(.failed(error.localizedDescription))
        }
        MLX.GPU.clearCache()
    }

    // MARK: Setup

    private func train() throws {
        emit(.phase(.preparing, "Reading model"))
        var def = try ModelPackage.read(request.modelURL)
        let hp = def.training
        MLX.GPU.set(memoryLimit: Int(request.memoryLimitGB * 1_073_741_824))
        MLX.GPU.set(cacheLimit: 2 * 1_073_741_824)
        if hp.seed != 0 { MLXRandom.seed(UInt64(hp.seed)) }

        // Mouse units are calibrated once, the first time the model is trained.
        if def.actions.mouseMode == .relative && !def.architectureLocked {
            def.actions.mouseUnitScale = DatasetBuilder.mouseUnitScale(urls: request.trainRecordings, model: def)
            log(String(format: "Mouse units: ×%.2f to match Pixel2Play's bins", def.actions.mouseUnitScale))
        }

        emit(.phase(.preparing, "Preparing recordings"))
        let texts = TextTable()
        let trainRecs = try DatasetBuilder.prepare(urls: request.trainRecordings, model: def, cacheRoot: request.cacheRoot, texts: texts, log: log)
        let valRecs = try DatasetBuilder.prepare(urls: request.validationRecordings, model: def, cacheRoot: request.cacheRoot, texts: texts, log: log)
        guard !trainRecs.isEmpty else { throw TrainerError.noData }

        var textTable: MLXArray? = nil
        if def.usesInstructions, texts.count > 0 {
            emit(.phase(.preparing, "Embedding \(texts.count) instructions"))
            if let enc = encoderFactory() {
                try texts.embedAll(with: enc)
                textTable = concatenated([MLXArray(texts.embeddings.flatMap { $0 }, [texts.count, enc.dimension]), MLXArray.zeros([1, enc.dimension])], axis: 0)
            } else {
                log("The instruction model (EmbeddingGemma) isn't downloaded, so this run trains without instructions. Download it in Settings → Downloads.")
            }
        }

        let a = def.architecture
        let length = a.contextSteps
        let trainSet = TrainingDataset(recordings: trainRecs, length: length, frameSize: a.frameSize, slots: def.actions.slotCount,
                                       absoluteCursor: def.actions.mouseMode == .absolute)
        let valSet = TrainingDataset(recordings: valRecs, length: length, frameSize: a.frameSize, slots: def.actions.slotCount,
                                     absoluteCursor: def.actions.mouseMode == .absolute)
        let batch = hp.batchSize > 0 ? hp.batchSize : Self.autoBatch(a, limitGB: request.memoryLimitGB, bf16: hp.bf16)
        let accumulation = max(1, hp.gradientAccumulation)
        let loader = ClipLoader(dataset: trainSet, seed: UInt64(hp.seed) &+ 17)
        defer { loader.stop() }
        let withText = trainRecs.flatMap(\.text).filter { $0 >= 0 }.count
        emit(.dataset(DatasetSummary(
            trainRecordings: trainRecs.count, validationRecordings: valRecs.count,
            trainSeconds: Double(trainSet.totalSteps) / Double(a.stepRateHz), validationSeconds: Double(valSet.totalSteps) / Double(a.stepRateHz),
            trainClips: loader.clipsPerEpoch, validationClips: valSet.validationClips(limit: 24).count,
            instructionCoverage: Double(withText) / Double(max(1, trainSet.totalSteps)),
            mouseUnitScale: def.actions.mouseUnitScale, batchSize: batch, accumulation: accumulation)))

        // Model: fp32 master weights for the optimizer, a compute copy (bf16) for forward/backward.
        emit(.phase(.preparing, "Loading weights"))
        let master = PolicyModel(architecture: a, actions: def.actions)
        let opt = AdamW(learningRate: Float(hp.learningRate), betas: (Float(hp.beta1), Float(hp.beta2)), eps: 1e-8, weightDecay: Float(hp.weightDecay))
        var startStep = 0
        if let ck = resumeCheckpoint(def) {
            try master.loadWeights(from: ck.appendingPathComponent(ModelPackage.weights))
            let optURL = ck.appendingPathComponent(ModelPackage.optimizer)
            if FileManager.default.fileExists(atPath: optURL.path) {
                opt.update(parameters: ModuleParameters.unflattened(try loadArrays(url: optURL)))
            }
            if let st = try? JSONDecoder().decode(CheckpointState.self, from: Data(contentsOf: ck.appendingPathComponent(ModelPackage.state))) { startStep = st.step }
            log("Resuming from \(ck.lastPathComponent)")
        } else if let pre = AppPathsHelper.pretrained(def.initialization, root: request.pretrainedRoot) {
            let r = try P2PImporter.load(checkpoint: pre, into: master)
            log("Initialized from Pixel2Play (\(r))")
        } else {
            if let stem = AppPathsHelper.pretrained(.p2p150M, root: request.pretrainedRoot) {
                try P2PImporter.loadVisionStem(checkpoint: stem, into: master)
                log("Blank model: vision stem initialized from Pixel2Play's EfficientNet")
            }
            if def.initialization != .blank { throw TrainerError.missingPretrained(def.initialization.label) }
        }
        master.freezeBatchNormStatistics()
        let computeType: DType = hp.bf16 ? .bfloat16 : .float32
        let compute = PolicyModel(architecture: a, actions: def.actions)
        compute.update(parameters: ModuleParameters.unflattened(master.parameters().flattened().map { ($0.0, $0.1.asType(computeType)) }))
        compute.setTraining()
        compute.freezeBatchNormStatistics()
        compute.visionGradientFrames = min(32, length)
        eval(master, compute)

        if !def.architectureLocked {
            def.architectureLocked = true
            try ModelPackage.write(def, to: request.modelURL)
        }

        let validator = valRecs.isEmpty ? nil : Validator(dataset: valSet, space: def.actions,
                                                          frequencies: Validator.labelFrequencies(trainRecs, space: def.actions))
        let augmenter = Augmenter(hp.augmentations, strength: hp.augmentationStrength)
        let space = def.actions
        let embedDim = a.textEmbeddingDim

        func textArrays(_ clips: [ClipData], dropout: Bool) -> (MLXArray, MLXArray) {
            let T = length
            var idx: [Int32] = []
            for c in clips {
                let drop = dropout && Double.random(in: 0 ..< 1) < hp.instructionDropout
                idx += c.text.map { drop || textTable == nil ? -1 : $0 }
            }
            let present = MLXArray(idx.map { $0 >= 0 }, [clips.count, T])
            guard let table = textTable else { return (MLXArray.zeros([clips.count, T, embedDim]), present) }
            let rows = MLXArray(idx.map { $0 >= 0 ? $0 : Int32(table.dim(0) - 1) })
            return (table.take(rows, axis: 0).reshaped(clips.count, T, embedDim), present)
        }

        let vg = valueAndGrad(model: compute) { (m: PolicyModel, b: [MLXArray]) -> [MLXArray] in
            let logits = m(frames: b[0], actionsIn: b[1], text: b[3], textPresent: b[4], historyKeep: b[5])
            let p = PolicyLoss.compute(logits, actions: b[1], mask: b[2], space: space, zWeight: Float(hp.zLossWeight))
            return [p.total] + p.perHead
        }

        // MARK: Loop
        emit(.phase(.training, "Training"))
        var step = startStep
        var lastCheckpoint = Date()
        var lastEmit = Date.distantPast
        var smoothed: Double?
        var lastLoss = 0.0
        var bestValidation = Double.infinity
        let logHandle = Self.openLog(request.modelURL)
        defer { try? logHandle?.close() }
        MLX.GPU.resetPeakMemory()

        while step < hp.totalSteps {
            if try checkControl(master: master, opt: opt, def: &def, step: step, loss: lastLoss, epoch: loader.epochProgress) { break }
            let t0 = Date()
            let lr = learningRate(step: step, hp: hp)
            var accumulated: ModuleParameters?
            var losses = [Double](repeating: 0, count: 5)
            for _ in 0 ..< accumulation {
                var clips: [ClipData] = []
                for _ in 0 ..< batch { clips.append(try loader.next()) }
                let S = a.frameSize, T = length
                let frames = stacked(clips.map { augmenter(MLXArray($0.frames, [T, S, S, 3], dtype: .uint8).asType(.float32)) })
                let acts = MLXArray(clips.flatMap(\.classes), [clips.count, T, space.slotCount])
                let mask = MLXArray(clips.flatMap(\.mask), [clips.count, T])
                let (text, present) = textArrays(clips, dropout: true)
                let keep = MLXArray(clips.flatMap { _ in Self.historyKeepMask(length: T, dropout: hp.historyDropout) }, [clips.count, T])
                let (values, grads) = vg(compute, [frames.asType(computeType), acts, mask, text, present, keep])
                let g32 = grads.mapValues { $0.asType(.float32) }
                accumulated = accumulated.map { acc in
                    ModuleParameters.unflattened(zip(acc.flattened(), g32.flattened()).map { ($0.0.0, $0.0.1 + $0.1.1) })
                } ?? g32
                eval(values, accumulated!)
                for (i, v) in values.enumerated() { losses[i] += Double(v.item(Float.self)) / Double(accumulation) }
            }
            var grads = accumulated!
            if accumulation > 1 { grads = grads.mapValues { $0 / Float(accumulation) } }
            var gradNorm = 0.0
            if hp.gradientClip > 0 {
                let (clipped, norm) = clipGradNorm(gradients: grads, maxNorm: Float(hp.gradientClip))
                grads = clipped
                gradNorm = Double(norm.item(Float.self))
            }
            opt.learningRate = Float(lr)
            opt.update(model: master, gradients: grads)
            compute.update(parameters: ModuleParameters.unflattened(master.trainableParameters().flattened().map { ($0.0, $0.1.asType(computeType)) }))
            eval(master, opt, compute)
            step += 1
            lastLoss = losses[0]
            smoothed = smoothed.map { $0 * 0.9 + losses[0] * 0.1 } ?? losses[0]

            let dt = Date().timeIntervalSince(t0)
            if Date().timeIntervalSince(lastEmit) > 0.4 || step == hp.totalSteps {
                lastEmit = Date()
                let p = TrainingProgress(step: step, totalSteps: hp.totalSteps, epoch: loader.epochProgress, loss: losses[0],
                                         headLosses: Array(losses[1...]), learningRate: lr, gradNorm: gradNorm, secondsPerStep: dt,
                                         framesPerSecond: Double(batch * accumulation * length) / dt,
                                         gpuMemoryGB: Double(MLX.GPU.activeMemory) / 1e9, peakMemoryGB: Double(MLX.GPU.peakMemory) / 1e9)
                emit(.progress(p))
                Self.append(logHandle, TrainingLogLine(progress: p))
            }

            if let v = validator, step % hp.validateEverySteps == 0 || step == hp.totalSteps {
                emit(.phase(.validating, "Validating"))
                if let m = v.run(model: compute, embed: { c in textArrays([c], dropout: false) }, step: step) {
                    emit(.validation(m))
                    Self.append(logHandle, TrainingLogLine(validation: m))
                    let total = m.nll.enumerated().map { $0.element / Foundation.log(Double([space.keyClasses, space.pointerClasses, space.xClasses, space.yClasses][$0.offset])) }.reduce(0, +)
                    if total < bestValidation { bestValidation = total; markBest(step: step) }
                }
                emit(.phase(.training, "Training"))
            }
            if Date().timeIntervalSince(lastCheckpoint) > hp.checkpointEveryMinutes * 60 {
                try saveCheckpoint(master: master, opt: opt, def: &def, step: step, loss: lastLoss, epoch: loader.epochProgress)
                lastCheckpoint = Date()
            }
        }
        emit(.phase(.saving, "Saving"))
        try saveCheckpoint(master: master, opt: opt, def: &def, step: step, loss: lastLoss, epoch: loader.epochProgress)
        emit(.phase(stopRequested ? .stopped : .finished, stopRequested ? "Stopped at step \(step)" : "Finished \(step) steps"))
    }

    /// Handles pause / stop / manual checkpoint. Returns true to stop.
    private func checkControl(master: PolicyModel, opt: AdamW, def: inout ModelDefinition, step: Int, loss: Double, epoch: Double) throws -> Bool {
        control.lock()
        let stop = stopRequested, pause = pauseRequested, ckpt = checkpointRequested
        checkpointRequested = false
        control.unlock()
        if stop { return true }
        if ckpt { try saveCheckpoint(master: master, opt: opt, def: &def, step: step, loss: loss, epoch: epoch) }
        if pause {
            try saveCheckpoint(master: master, opt: opt, def: &def, step: step, loss: loss, epoch: epoch)
            emit(.phase(.paused, "Paused at step \(step)"))
            MLX.GPU.clearCache()
            control.lock()
            while pauseRequested && !stopRequested { control.wait() }
            let s = stopRequested
            control.unlock()
            if s { return true }
            emit(.phase(.training, "Training"))
        }
        return false
    }

    /// Which past-action tokens stay visible: sometimes none for a whole clip, otherwise random
    /// spans (~0.5 s) covering about `dropout` of the steps.
    static func historyKeepMask(length T: Int, dropout p: Double) -> [Bool] {
        guard p > 0 else { return Array(repeating: true, count: T) }
        if Double.random(in: 0 ..< 1) < 0.15 * p / 0.5 { return Array(repeating: false, count: T) }
        let avg = 15.0
        let start = min(0.9, p / (avg * max(1 - p, 0.05)))
        var keep = [Bool](), t = 0
        while t < T {
            if Double.random(in: 0 ..< 1) < start {
                let n = Int.random(in: 1 ... Int(avg * 2))
                keep += Array(repeating: false, count: min(n, T - t)); t += n
            } else { keep.append(true); t += 1 }
        }
        return Array(keep.prefix(T))
    }

    func learningRate(step: Int, hp: TrainingHyperparameters) -> Double {
        if step < hp.warmupSteps { return hp.learningRate * Double(step + 1) / Double(hp.warmupSteps) }
        guard hp.schedule == .cosine else { return hp.learningRate }
        let p = Double(step - hp.warmupSteps) / Double(max(1, hp.totalSteps - hp.warmupSteps))
        let minLR = hp.learningRate * hp.minLRFactor
        return minLR + (hp.learningRate - minLR) * 0.5 * (1 + cos(Double.pi * min(1, p)))
    }

    static func autoBatch(_ a: PolicyArchitecture, limitGB: Double, bf16: Bool) -> Int {
        // Measured on M3 Max: P2P-150M, 10 s @ 30 Hz clip ≈ 9.8 GB (bf16) / 19 GB (fp32).
        let tokens = Double(a.contextSteps) / 300
        let perClip = (bf16 ? 9.8 : 19.4) * Double(a.layers) / 10 * pow(Double(a.width) / 1024, 1.5) * pow(tokens, 1.6) + 1
        return max(1, min(4, Int((limitGB * 0.85 - 2) / perClip)))
    }

    // MARK: Checkpoints

    private func resumeCheckpoint(_ def: ModelDefinition) -> URL? {
        let names = ModelPackage.checkpointNames(request.modelURL)
        let dir = request.modelURL.appendingPathComponent(ModelPackage.checkpoints)
        if let r = request.resumeFrom, names.contains(r) { return dir.appendingPathComponent(r) }
        return names.filter { $0.hasPrefix("step-") }.last.map { dir.appendingPathComponent($0) }
    }

    private func saveCheckpoint(master: PolicyModel, opt: AdamW, def: inout ModelDefinition, step: Int, loss: Double, epoch: Double) throws {
        let name = CheckpointState.name(step: step)
        let dir = request.modelURL.appendingPathComponent(ModelPackage.checkpoints).appendingPathComponent(name)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try master.saveWeights(to: dir.appendingPathComponent(ModelPackage.weights), metadata: ["step": "\(step)"])
        try save(arrays: Dictionary(uniqueKeysWithValues: opt.state().flattened()), url: dir.appendingPathComponent(ModelPackage.optimizer))
        let state = CheckpointState(step: step, epoch: epoch, trainLoss: loss, validationLoss: nil, rngSeed: 0, dataCursor: 0)
        try JSONEncoder().encode(state).write(to: dir.appendingPathComponent(ModelPackage.state))
        var latest = (try? ModelPackage.read(request.modelURL)) ?? def
        latest.trainedSteps = step
        latest.lastTrainedAt = Date()
        latest.actions.mouseUnitScale = def.actions.mouseUnitScale
        latest.architectureLocked = true
        try ModelPackage.write(latest, to: request.modelURL)
        def = latest
        pruneCheckpoints()
        emit(.checkpoint(name: name, step: step))
    }

    private var bestStep: Int?
    private func markBest(step: Int) { bestStep = step }

    /// Keeps the three newest checkpoints, the best-validation one, and any the user pinned for Run.
    private func pruneCheckpoints() {
        let names = ModelPackage.checkpointNames(request.modelURL).filter { $0.hasPrefix("step-") }
        let keep = Set(names.suffix(3) + [bestStep.map(CheckpointState.name(step:))].compactMap { $0 }
                       + [(try? ModelPackage.read(request.modelURL))?.activeCheckpoint].compactMap { $0 })
        let dir = request.modelURL.appendingPathComponent(ModelPackage.checkpoints)
        for n in names where !keep.contains(n) { try? FileManager.default.removeItem(at: dir.appendingPathComponent(n)) }
    }

    static func openLog(_ model: URL) -> FileHandle? {
        let url = model.appendingPathComponent(ModelPackage.log)
        if !FileManager.default.fileExists(atPath: url.path) { FileManager.default.createFile(atPath: url.path, contents: nil) }
        let h = try? FileHandle(forWritingTo: url)
        _ = try? h?.seekToEnd()
        return h
    }

    static func append(_ h: FileHandle?, _ line: TrainingLogLine) {
        guard let h, var d = try? JSONEncoder().encode(line) else { return }
        d.append(0x0A)
        h.write(d)
    }
}

public enum TrainerError: Error, LocalizedError {
    case noData, missingPretrained(String)
    public var errorDescription: String? {
        switch self {
        case .noData: "None of the selected recordings could be used for training."
        case .missingPretrained(let n): "\(n) weights aren't downloaded yet. Download them in the AI Models tab or in Settings."
        }
    }
}

enum AppPathsHelper {
    static func pretrained(_ i: ModelInitialization, root: URL) -> URL? {
        guard let f = i.checkpointFolder else { return nil }
        let u = root.appendingPathComponent("p2p-\(f)").appendingPathComponent("checkpoint.ckpt")
        return FileManager.default.fileExists(atPath: u.path) ? u : nil
    }
}
