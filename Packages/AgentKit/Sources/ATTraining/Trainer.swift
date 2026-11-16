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
        // Training can take hours: don't let App Nap or idle sleep slow it down or stop it.
        let activity = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .idleSystemSleepDisabled], reason: "Training an AI model")
        defer { ProcessInfo.processInfo.endActivity(activity) }
        do {
            // MLX errors (e.g. a shape that doesn't fit) become a failed job instead of a crash.
            try withError { mlxErrors in try train(mlxErrors) }
        } catch is CancellationError {
            emit(.phase(.stopped, "Stopped"))
        } catch {
            emit(.failed(error.localizedDescription))
        }
        MLX.GPU.clearCache()
    }

    // MARK: Setup

    private func train(_ mlxErrors: ErrorBox) throws {
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
        var trainSet = TrainingDataset(recordings: trainRecs, length: length, frameSize: a.frameSize, slots: def.actions.slotCount,
                                       absoluteCursor: def.actions.mouseMode == .absolute)
        var valSet = TrainingDataset(recordings: valRecs, length: length, frameSize: a.frameSize, slots: def.actions.slotCount,
                                     absoluteCursor: def.actions.mouseMode == .absolute)
        if a.screenText {
            trainSet.screenItems = a.screenTextItems
            valSet.screenItems = a.screenTextItems
        }
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
        let computeType: DType = hp.bf16 ? .bfloat16 : .float32
        let opt = FusedAdamW(learningRate: Float(hp.learningRate), betas: (Float(hp.beta1), Float(hp.beta2)), eps: 1e-8,
                             weightDecay: Float(hp.weightDecay), computeType: computeType)
        opt.keepsFP32 = PolicyModel.keepsFP32
        var startStep = 0
        var resumeFrom = resumeCheckpoint(def)
        while let ck = resumeFrom {
            do {
                try master.loadWeights(from: ck.appendingPathComponent(ModelPackage.weights))
                break
            } catch WeightIOError.notFinite {
                log("Skipping \(ck.lastPathComponent): its weights are broken (NaN) from an earlier failed run")
                resumeFrom = previousCheckpoint(before: ck.lastPathComponent)
            }
        }
        if let ck = resumeFrom {
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
        let compute = PolicyModel(architecture: a, actions: def.actions)
        compute.update(parameters: ModuleParameters.unflattened(master.parameters().flattened().map { ($0.0, $0.1.asType(PolicyModel.keepsFP32($0.0) ? .float32 : computeType)) }))
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

        /// Screen-text and typing inputs for a batch of clips (nil for models without them).
        func languageInputs(_ clips: [ClipData]) -> LanguageInputs? {
            guard a.usesDecoderMemory else { return nil }
            let T = length, B = clips.count
            var l = LanguageInputs()
            if a.screenText, let table = textTable {
                let K = a.screenTextItems
                let ids = clips.flatMap(\.screenIds)
                let rows = MLXArray(ids.map { $0 >= 0 ? $0 : Int32(table.dim(0) - 1) })
                l.screenEmbeddings = table.take(rows, axis: 0).reshaped(B, T, K, embedDim)
                l.screenBoxes = MLXArray(clips.flatMap(\.screenBoxes), [B, T, K, 5])
                l.screenMask = MLXArray(ids.map { $0 >= 0 }, [B, T, K])
            }
            if a.literalTyping {
                let bytes = clips.flatMap { c in c.text.flatMap { id in LanguageInputs.bytes(id >= 0 ? texts.strings[Int(id)] : nil, length: a.typingBytes) } }
                l.typingBytes = MLXArray(bytes, [B, T, a.typingBytes])
            }
            return l
        }

        let vgPlain = valueAndGrad(model: compute) { (m: PolicyModel, b: [MLXArray]) -> [MLXArray] in
            var lang: LanguageInputs?
            if b.count > 6 {
                lang = LanguageInputs(screenEmbeddings: b[6].size > 0 ? b[6] : nil, screenBoxes: b[7].size > 0 ? b[7] : nil,
                                      screenMask: b[8].size > 0 ? b[8] : nil, typingBytes: b[9].size > 0 ? b[9] : nil)
            }
            let logits = m(frames: b[0], actionsIn: b[1], text: b[3], textPresent: b[4], historyKeep: b[5], language: lang)
            let p = PolicyLoss.compute(logits, actions: b[1], mask: b[2], space: space, zWeight: Float(hp.zLossWeight))
            return [p.total] + p.perHead
        }
        // The whole forward + backward compiled into one graph: MLX fuses the many small
        // elementwise ops (residuals, dropout, casts, loss) and skips re-tracing every step.
        // Shapes are fixed for the run; randomness flows through the global random state.
        let useCompile = ProcessInfo.processInfo.environment["AT_NO_COMPILE"] == nil
        compute.recordsDebug = !useCompile
        compute.prepareMasks(steps: length)
        var gradKeys: [String] = []
        let compiledStep = compile(inputs: [compute, MLXRandom.globalState], outputs: [MLXRandom.globalState]) { (inputs: [MLXArray]) -> [MLXArray] in
            let (values, grads) = vgPlain(compute, inputs)
            let flat = grads.flattened()
            gradKeys = flat.map(\.0)
            return values + flat.map(\.1)
        }
        func vg(_ m: PolicyModel, _ inputs: [MLXArray]) -> ([MLXArray], ModuleParameters) {
            guard useCompile else { return vgPlain(m, inputs) }
            let out = compiledStep(inputs)
            let nValues = out.count - gradKeys.count
            return (Array(out[0 ..< nValues]), ModuleParameters.unflattened(Array(zip(gradKeys, out[nValues...]))))
        }

        // MARK: Loop
        emit(.phase(.training, "Training"))
        var step = startStep
        var lastCheckpoint = Date()
        var lastEmit = Date.distantPast
        var smoothed: Double?
        var lastLoss = 0.0
        // Best held-out loss so far. Resuming continues the comparison with the saved "best";
        // a fresh run starts over (an old "best" from a previous run would not be comparable).
        var bestValidation = Double.infinity
        let bestDir = request.modelURL.appendingPathComponent(ModelPackage.checkpoints).appendingPathComponent(Self.bestName)
        if startStep > 0, let st = try? JSONDecoder().decode(CheckpointState.self, from: Data(contentsOf: bestDir.appendingPathComponent(ModelPackage.state))),
           let v = st.validationLoss {
            bestValidation = v; bestStep = st.step
        } else if startStep == 0 {
            try? FileManager.default.removeItem(at: bestDir)
        }
        var checksWithoutImprovement = 0
        var stoppedEarly = false
        let logHandle = Self.openLog(request.modelURL)
        defer { try? logHandle?.close() }
        MLX.GPU.resetPeakMemory()
        // AT_TRAIN_PROFILE=1: time each phase (adds sync points, so only for measuring).
        let profile = ProcessInfo.processInfo.environment["AT_TRAIN_PROFILE"] != nil
        var prof: [String: Double] = [:], profSteps = 0
        var tp = Date()
        func mark(_ k: String) { if profile { prof[k, default: 0] += Date().timeIntervalSince(tp); tp = Date() } }

        while step < hp.totalSteps {
            if try checkControl(master: master, opt: opt, def: &def, step: step, loss: lastLoss, epoch: loader.epochProgress) { break }
            let t0 = Date()
            let lr = learningRate(step: step, hp: hp)
            var accumulated: ModuleParameters?
            var losses = [Double](repeating: 0, count: 5)
            var pendingValues: [MLXArray] = []
            tp = Date()
            for _ in 0 ..< accumulation {
                var clips: [ClipData] = []
                for _ in 0 ..< batch { clips.append(try loader.next()) }
                mark("1 wait for data")
                let S = a.frameSize, T = length
                let frames = stacked(clips.map { augmenter(MLXArray($0.frames, [T, S, S, 3], dtype: .uint8), dtype: computeType) })
                let acts = MLXArray(clips.flatMap(\.classes), [clips.count, T, space.slotCount])
                let mask = MLXArray(clips.flatMap(\.mask), [clips.count, T])
                let (text, present) = textArrays(clips, dropout: true)
                let keep = MLXArray(clips.flatMap { _ in Self.historyKeepMask(length: T, dropout: hp.historyDropout) }, [clips.count, T])
                var inputs = [frames, acts, mask, text, present, keep]
                if let l = languageInputs(clips) {
                    let empty = MLXArray.zeros([0])
                    inputs += [l.screenEmbeddings ?? empty, l.screenBoxes ?? empty, l.screenMask ?? empty, l.typingBytes ?? empty]
                }
                if profile { eval(inputs); mark("2 upload + augment") }
                if profile && profSteps == 3 {
                    let f = inputs[0]
                    tp = Date(); eval(compute.profileStem(f)); mark("x stem forward, all frames")
                    tp = Date(); eval(compute(frames: f, actionsIn: inputs[1], text: inputs[3], textPresent: inputs[4], historyKeep: inputs[5]).keys); mark("x full forward only")
                }
                if ProcessInfo.processInfo.environment["AT_LEGACY_CHECK"] != nil, step == startStep {
                    // New training path vs the original one on the same batch and random state.
                    MLXRandom.seed(7); let (v1, g1) = vg(compute, inputs); eval(v1, g1)
                    PolicyModel.useLegacyPaths(true)
                    MLXRandom.seed(7); let (v2, g2) = vgPlain(compute, inputs); eval(v2, g2)
                    PolicyModel.useLegacyPaths(false)
                    let gNew = Dictionary(uniqueKeysWithValues: g1.flattened()), gOld = Dictionary(uniqueKeysWithValues: g2.flattened())
                    var rels: [(Double, String)] = []
                    var num = 0.0, den = 0.0
                    for (k, x) in gNew {
                        guard let y = gOld[k] else { continue }
                        let x32 = x.asType(.float32), y32 = y.asType(.float32)
                        let d2 = Double((x32 - y32).square().sum().item(Float.self)), n2 = Double(x32.square().sum().item(Float.self))
                        num += d2; den += n2
                        rels.append(((d2 / max(n2, 1e-30)).squareRoot(), k))
                    }
                    if computeType != .float32 {
                        // fp32 reference with the same weights: how far is each bf16 path from it?
                        let ref = PolicyModel(architecture: a, actions: def.actions)
                        ref.update(parameters: ModuleParameters.unflattened(master.parameters().flattened()))
                        ref.setTraining(); ref.freezeBatchNormStatistics(); ref.visionGradientFrames = compute.visionGradientFrames; ref.recordsDebug = false
                        let vgRef = valueAndGrad(model: ref) { (m: PolicyModel, b: [MLXArray]) -> [MLXArray] in
                            let logits = m(frames: b[0], actionsIn: b[1], text: b[3], textPresent: b[4], historyKeep: b[5])
                            return [PolicyLoss.compute(logits, actions: b[1], mask: b[2], space: space, zWeight: Float(hp.zLossWeight)).total]
                        }
                        var refInputs = inputs; refInputs[0] = inputs[0].asType(.float32)
                        MLXRandom.seed(7); let (_, gr) = vgRef(ref, refInputs); eval(gr)
                        let r = Dictionary(uniqueKeysWithValues: gr.flattened())
                        func err(_ g: [String: MLXArray], stemOnly: Bool) -> Double {
                            var n = 0.0, d = 0.0
                            for (k, x) in r where !stemOnly || k.hasPrefix("vision.stem") {
                                guard let y = g[k] else { continue }
                                n += Double((x - y.asType(.float32)).square().sum().item(Float.self)); d += Double(x.square().sum().item(Float.self))
                            }
                            return (n / max(d, 1e-30)).squareRoot()
                        }
                        log(String(format: "BF16 ERROR vs fp32 reference | all params: new %.2e old %.2e | vision stem: new %.2e old %.2e",
                                   err(gNew, stemOnly: false), err(gOld, stemOnly: false), err(gNew, stemOnly: true), err(gOld, stemOnly: true)))
                    }
                    rels.sort { $0.0 > $1.0 }
                    log(String(format: "LEGACY CHECK: loss new %.6f old %.6f | all-gradient rel diff %.2e | median %.2e | worst: %@",
                               v1[0].item(Float.self), v2[0].item(Float.self), (num / den).squareRoot(), rels[rels.count / 2].0,
                               rels.prefix(4).map { String(format: "%.1e %@", $0.0, $0.1) }.joined(separator: ", ")))
                }
                if ProcessInfo.processInfo.environment["AT_COMPILE_CHECK"] != nil, step == startStep, useCompile {
                    // Same inputs and random state through the plain and the compiled step.
                    MLXRandom.seed(42); let (v1, g1) = vgPlain(compute, inputs); eval(v1, g1)
                    MLXRandom.seed(42); let (_, g0) = vgPlain(compute, inputs); eval(g0)
                    for key in ["emb_key.weight", "emb_pointer.weight", "emb_x.weight"] {
                        let x = Dictionary(uniqueKeysWithValues: g1.flattened())[key]!.asType(.float32), y = Dictionary(uniqueKeysWithValues: g0.flattened())[key]!.asType(.float32)
                        log(String(format: "PLAIN vs PLAIN %@: rel diff %.2e", key, sqrt((x - y).square().sum()).item(Float.self) / sqrt(x.square().sum()).item(Float.self)))
                    }
                    MLXRandom.seed(42); let (v2, g2) = vg(compute, inputs); eval(v2, g2)
                    let a = Dictionary(uniqueKeysWithValues: g1.flattened()), b = Dictionary(uniqueKeysWithValues: g2.flattened())
                    var rels: [(Double, String)] = []
                    for (k, x) in a {
                        guard let y = b[k] else { log("COMPILE CHECK: missing grad \(k)"); continue }
                        let x32 = x.asType(.float32), y32 = y.asType(.float32)
                        let rel = Double(sqrt((x32 - y32).square().sum()).item(Float.self)) / max(Double(sqrt(x32.square().sum()).item(Float.self)), 1e-20)
                        rels.append((rel, k + String(format: " |g| %.2e", sqrt(x32.square().sum()).item(Float.self))))
                    }
                    rels.sort { $0.0 > $1.0 }
                    let worst = rels.first?.0 ?? 0, worstKey = rels.prefix(6).map { String(format: "%.1e %@", $0.0, $0.1) }.joined(separator: "; ")
                    let median = rels.isEmpty ? 0 : rels[rels.count / 2].0
                    log(String(format: "COMPILE CHECK: median rel grad-norm diff %.2e", median))
                    log(String(format: "COMPILE CHECK: loss plain %.6f compiled %.6f | grads %d vs %d | worst rel grad diff %.2e (%@)",
                               v1[0].item(Float.self), v2[0].item(Float.self), a.count, b.count, worst, worstKey))
                }
                let (values, grads) = vg(compute, inputs)
                try mlxErrors.check()
                mark("3 build graph (CPU)")
                if accumulation == 1 {
                    accumulated = grads // the fused optimizer casts to fp32 itself
                } else {
                    let g32 = grads.mapValues { $0.asType(.float32) }
                    accumulated = accumulated.map { acc in
                        ModuleParameters.unflattened(zip(acc.flattened(), g32.flattened()).map { ($0.0.0, $0.0.1 + $0.1.1) })
                    } ?? g32
                }
                if accumulation > 1 || profile {
                    eval(values, accumulated!)
                    try mlxErrors.check()
                    mark("4 forward + backward (GPU)")
                    for (i, v) in values.enumerated() { losses[i] += Double(v.item(Float.self)) / Double(accumulation) }
                } else {
                    pendingValues = values // evaluated together with the optimizer update below
                }
            }
            if accumulation > 1 || profile {
                guard losses.allSatisfy(\.isFinite) else { throw TrainerError.diverged(step: step + 1) }
            }
            var grads = accumulated!
            if accumulation > 1 { grads = grads.mapValues { $0 / Float(accumulation) } }
            opt.learningRate = Float(lr)
            let norm = opt.update(master: master, compute: compute, gradients: grads, clip: Float(max(0, hp.gradientClip)))
            mark("5a optimizer graph (CPU)")
            // One submission for forward, backward and update: no GPU idle gap between them.
            eval([norm] + opt.arrays + pendingValues, master, compute)
            if !pendingValues.isEmpty {
                try mlxErrors.check()
                for (i, v) in pendingValues.enumerated() { losses[i] = Double(v.item(Float.self)) }
                pendingValues = []
                // The update already ran, but nothing is saved after a non-finite step: training stops here.
                guard losses.allSatisfy(\.isFinite) else { throw TrainerError.diverged(step: step + 1) }
            }
            let gradNorm = hp.gradientClip > 0 ? Double(norm.item(Float.self)) : 0
            try mlxErrors.check()
            mark("5b optimizer (GPU)")
            if profile {
                profSteps += 1
                if profSteps == 2 { prof = [:] } // drop warm-up
                if profSteps > 2, (profSteps - 2) % 5 == 0 {
                    let n = Double(profSteps - 2)
                    log("PROFILE over \(Int(n)) steps (batch \(batch)×\(length)): " + prof.keys.sorted().map { k in String(format: "%@ %.0f ms", k, prof[k]! / (k.hasPrefix("x") ? 1 : n) * 1000) }.joined(separator: " | "))
                }
            }
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
                if let m = v.run(model: compute, embed: { c in textArrays([c], dropout: false) }, language: { c in languageInputs([c]) }, step: step) {
                    emit(.validation(m))
                    Self.append(logHandle, TrainingLogLine(validation: m))
                    let total = m.nll.enumerated().map { $0.element * ActionSpaceConfig.lossScale(classes: [space.keyClasses, space.pointerClasses, space.xClasses, space.yClasses][$0.offset]) }.reduce(0, +)
                    if total < bestValidation {
                        bestValidation = total
                        bestStep = step
                        checksWithoutImprovement = 0
                        if hp.saveBest {
                            try saveBest(master: master, step: step, validationLoss: total, trainLoss: lastLoss, epoch: loader.epochProgress)
                            log(String(format: "New best held-out loss %.4f at step %d (saved as \"best\")", total, step))
                        }
                    } else {
                        checksWithoutImprovement += 1
                        if hp.earlyStopping {
                            log("Held-out loss hasn't improved for \(checksWithoutImprovement) of \(hp.earlyStoppingPatience) checks (best: step \(bestStep ?? 0))")
                        }
                        if hp.earlyStopping && checksWithoutImprovement >= max(1, hp.earlyStoppingPatience) {
                            stoppedEarly = true
                        }
                    }
                }
                if stoppedEarly { break }
                emit(.phase(.training, "Training"))
            }
            if Date().timeIntervalSince(lastCheckpoint) > hp.checkpointEveryMinutes * 60 {
                try saveCheckpoint(master: master, opt: opt, def: &def, step: step, loss: lastLoss, epoch: loader.epochProgress)
                lastCheckpoint = Date()
            }
        }
        emit(.phase(.saving, "Saving"))
        try saveCheckpoint(master: master, opt: opt, def: &def, step: step, loss: lastLoss, epoch: loader.epochProgress)
        // Run uses the best checkpoint unless the user pinned another one.
        if hp.saveBest, FileManager.default.fileExists(atPath: bestDir.path), var latest = try? ModelPackage.read(request.modelURL), latest.activeCheckpoint == nil {
            latest.activeCheckpoint = Self.bestName
            try? ModelPackage.write(latest, to: request.modelURL)
        }
        if stoppedEarly {
            emit(.phase(.finished, "Stopped early at step \(step): the held-out loss stopped improving (best at step \(bestStep ?? step))"))
        } else {
            emit(.phase(stopRequested ? .stopped : .finished, stopRequested ? "Stopped at step \(step)" : "Finished \(step) steps"))
        }
    }

    static let bestName = "best"

    /// Writes the current weights as checkpoint "best" (no optimizer state: it is a model to run,
    /// and training can still resume from it).
    private func saveBest(master: PolicyModel, step: Int, validationLoss: Double, trainLoss: Double, epoch: Double) throws {
        let dir = request.modelURL.appendingPathComponent(ModelPackage.checkpoints).appendingPathComponent(Self.bestName)
        let tmp = request.modelURL.appendingPathComponent(ModelPackage.checkpoints).appendingPathComponent(".best-writing")
        try? FileManager.default.removeItem(at: tmp)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        try master.saveWeights(to: tmp.appendingPathComponent(ModelPackage.weights), metadata: ["step": "\(step)", "best": "1"])
        let state = CheckpointState(step: step, epoch: epoch, trainLoss: trainLoss, validationLoss: validationLoss, rngSeed: 0, dataCursor: 0)
        try JSONEncoder().encode(state).write(to: tmp.appendingPathComponent(ModelPackage.state))
        try? FileManager.default.removeItem(at: dir)
        try FileManager.default.moveItem(at: tmp, to: dir)
        emit(.checkpoint(name: Self.bestName, step: step))
    }

    /// Handles pause / stop / manual checkpoint. Returns true to stop.
    private func checkControl(master: PolicyModel, opt: FusedAdamW, def: inout ModelDefinition, step: Int, loss: Double, epoch: Double) throws -> Bool {
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

    private func previousCheckpoint(before name: String) -> URL? {
        let dir = request.modelURL.appendingPathComponent(ModelPackage.checkpoints)
        return ModelPackage.checkpointNames(request.modelURL).filter { $0.hasPrefix("step-") && $0 < name }.last.map { dir.appendingPathComponent($0) }
    }

    private func resumeCheckpoint(_ def: ModelDefinition) -> URL? {
        let names = ModelPackage.checkpointNames(request.modelURL)
        let dir = request.modelURL.appendingPathComponent(ModelPackage.checkpoints)
        if let r = request.resumeFrom, names.contains(r) { return dir.appendingPathComponent(r) }
        return names.filter { $0.hasPrefix("step-") }.last.map { dir.appendingPathComponent($0) }
    }

    private func saveCheckpoint(master: PolicyModel, opt: FusedAdamW, def: inout ModelDefinition, step: Int, loss: Double, epoch: Double) throws {
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
        guard let h, var d = try? TrainingJSON.encoder().encode(line) else { return }
        d.append(0x0A)
        h.write(d)
    }
}

public enum TrainerError: Error, LocalizedError {
    case noData, missingPretrained(String), diverged(step: Int)
    public var errorDescription: String? {
        switch self {
        case .noData: "None of the selected recordings could be used for training."
        case .missingPretrained(let n): "\(n) weights aren't downloaded yet. Download them in the AI Models tab or in Settings."
        case .diverged(let s): "Training stopped at step \(s): the loss became NaN or infinite. Nothing was saved from this step. Try a lower learning rate or turn gradient clipping on."
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
