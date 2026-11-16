import ATCore
import ATModel
import Foundation
import MLX

/// Held-out evaluation focused on what predicts real play, not just loss.
final class Validator {
    let dataset: TrainingDataset
    let clips: [ClipRef]
    let loader: ClipLoader
    let space: ActionSpaceConfig
    /// Label frequencies per head from the training data (for RCE).
    let frequencies: [[Double]]

    init(dataset: TrainingDataset, space: ActionSpaceConfig, frequencies: [[Double]], maxClips: Int = 24) {
        self.dataset = dataset
        self.space = space
        self.frequencies = frequencies
        clips = dataset.validationClips(limit: maxClips)
        loader = ClipLoader(dataset: dataset, seed: 1)
    }

    static func labelFrequencies(_ recs: [PreparedRecording], space: ActionSpaceConfig) -> [[Double]] {
        let ks = space.keySlots, ps = space.pointerSlots
        var counts = [[Double]](repeating: [], count: 4)
        counts[0] = Array(repeating: 1, count: space.keyClasses); counts[1] = Array(repeating: 1, count: space.pointerClasses)
        counts[2] = Array(repeating: 1, count: space.xClasses); counts[3] = Array(repeating: 1, count: space.yClasses)
        for r in recs {
            for (c, m) in zip(r.classes, r.lossMask) where m {
                for s in 0 ..< ks { counts[0][Int(c[s])] += 1 }
                for s in ks ..< ks + ps { counts[1][Int(c[s])] += 1 }
                counts[2][Int(c[ks + ps])] += 1
                counts[3][Int(c[ks + ps + 1])] += 1
            }
        }
        return counts.map { row in let t = row.reduce(0, +); return row.map { $0 / t } }
    }

    func run(model: PolicyModel, embed: (ClipData) -> (MLXArray, MLXArray), language: (ClipData) -> LanguageInputs?, step: Int) -> ValidationMetrics? {
        guard !clips.isEmpty else { return nil }
        let wasTraining = model.training
        model.train(false)
        defer { if wasTraining { model.setTraining() } }
        let ks = space.keySlots, ps = space.pointerSlots, S = space.slotCount
        var ceSum = [Double](repeating: 0, count: 4), baseSum = [Double](repeating: 0, count: 4)
        var shuffledCE = 0.0, normalCE = 0.0, blindCE = 0.0, textCE = 0.0, noTextCE = 0.0, textSteps = 0
        var nSteps = 0
        var keyStats = OnsetStats(), clickStats = OnsetStats()
        var mx: [(Double, Double)] = []
        let xCenters = binCenters(space.mouseXEdges), yCenters = binCenters(space.mouseYEdges)

        for ref in clips {
            guard let clip = try? loader.load(ref) else { continue }
            let T = dataset.length, Sz = dataset.frameSize
            let frames = MLXArray(clip.frames, [1, T, Sz, Sz, 3], dtype: .uint8)
            let acts = MLXArray(clip.classes, [1, T, S])
            let mask = MLXArray(clip.mask, [1, T])
            let (text, present) = embed(clip)
            let lang = language(clip)
            // The vision stem runs once per clip; all passes below reuse its tokens (eval mode is
            // deterministic, and rolling frames in time just rolls their tokens).
            let img = model.clipImageTokens(frames)
            let logits = model(frames: frames, actionsIn: acts, text: text, textPresent: present, language: lang, imageTokens: img)
            let parts = PolicyLoss.compute(logits, actions: acts, mask: mask, space: space, zWeight: 0)
            // Visual reliance: same clip with frames rolled half a clip in time.
            let rolled = concatenated([img[0..., (T / 2)..., 0..., 0...], img[0..., 0 ..< (T / 2), 0..., 0...]], axis: 1)
            let shuffled = PolicyLoss.compute(model(frames: frames, actionsIn: acts, text: text, textPresent: present, language: lang, imageTokens: rolled), actions: acts, mask: mask, space: space, zWeight: 0)
            let blind = PolicyLoss.compute(model(frames: frames, actionsIn: acts, text: text, textPresent: present, historyKeep: MLXArray.zeros([1, T], dtype: .bool), language: lang, imageTokens: img),
                                           actions: acts, mask: mask, space: space, zWeight: 0)
            var textParts: PolicyLoss.Parts?
            let hasText = clip.text.contains { $0 >= 0 }
            if hasText {
                var noLang = lang
                if let b = lang?.typingBytes { noLang?.typingBytes = MLXArray.full(b.shape, values: MLXArray(Int32(256))) }
                textParts = PolicyLoss.compute(model(frames: frames, actionsIn: acts, text: text, textPresent: MLXArray.zeros([1, T], dtype: .bool), language: noLang, imageTokens: img),
                                               actions: acts, mask: mask, space: space, zWeight: 0)
            }
            let keyPred = argMax(logits.keys, axis: -1), ptrPred = argMax(logits.pointer, axis: -1)
            let xProb = softmax(logits.x.asType(.float32), axis: -1), yProb = softmax(logits.y.asType(.float32), axis: -1)
            eval(parts.total, shuffled.total, blind.total, keyPred, ptrPred, xProb, yProb)
            if let tp = textParts { eval(tp.total) }

            let m = clip.mask.filter { $0 }.count
            guard m > 0 else { continue }
            nSteps += m
            for h in 0 ..< 4 { ceSum[h] += Double(parts.perHead[h].item(Float.self)) * Double(m) }
            normalCE += Double(parts.total.item(Float.self)) * Double(m)
            shuffledCE += Double(shuffled.total.item(Float.self)) * Double(m)
            blindCE += Double(blind.total.item(Float.self)) * Double(m)
            if let tp = textParts {
                let tm = zip(clip.text, clip.mask).filter { $0.0 >= 0 && $0.1 }.count
                textCE += Double(parts.total.item(Float.self)) * Double(tm)
                noTextCE += Double(tp.total.item(Float.self)) * Double(tm)
                textSteps += tm
            }
            // Baseline cross-entropy from label frequencies.
            for t in 0 ..< T where clip.mask[t] {
                let c = Array(clip.classes[t * S ..< (t + 1) * S])
                baseSum[0] += (0 ..< ks).map { -log(frequencies[0][Int(c[$0])]) }.reduce(0, +) / Double(ks)
                baseSum[1] += (ks ..< ks + ps).map { -log(frequencies[1][Int(c[$0])]) }.reduce(0, +) / Double(max(ps, 1))
                baseSum[2] += -log(frequencies[2][Int(c[ks + ps])])
                baseSum[3] += -log(frequencies[3][Int(c[ks + ps + 1])])
            }
            // Onset F1 and mouse quality.
            let kp = keyPred.asArray(Int32.self), pp = ptrPred.asArray(Int32.self)
            let xp = xProb.asArray(Float.self), yp = yProb.asArray(Float.self)
            var predKeys: [Set<Int32>] = [], trueKeys: [Set<Int32>] = [], predPtr: [Set<Int32>] = [], truePtr: [Set<Int32>] = []
            for t in 0 ..< T {
                predKeys.append(Set(kp[t * ks ..< (t + 1) * ks].filter { $0 > 0 }))
                trueKeys.append(Set(clip.classes[t * S ..< t * S + ks].filter { $0 > 0 }))
                predPtr.append(Set(pp[t * ps ..< (t + 1) * ps].filter { $0 > 0 }))
                truePtr.append(Set(clip.classes[t * S + ks ..< t * S + ks + ps].filter { $0 > 0 }))
                if clip.mask[t], space.mouseMode == .relative {
                    let gx = Int(clip.classes[t * S + ks + ps]), gy = Int(clip.classes[t * S + ks + ps + 1])
                    let ex = zip(xp[t * space.xClasses ..< (t + 1) * space.xClasses], xCenters).map { Double($0) * $1 }.reduce(0, +)
                    let ey = zip(yp[t * space.yClasses ..< (t + 1) * space.yClasses], yCenters).map { Double($0) * $1 }.reduce(0, +)
                    if gx != space.zeroXClass { mx.append((ex, xCenters[gx])) }
                    if gy != space.zeroYClass { mx.append((ey, yCenters[gy])) }
                }
            }
            keyStats.add(pred: predKeys, truth: trueKeys, mask: clip.mask)
            clickStats.add(pred: predPtr, truth: truePtr, mask: clip.mask)
        }
        guard nSteps > 0 else { return nil }
        let nll = ceSum.map { $0 / Double(nSteps) }
        let base = baseSum.map { $0 / Double(nSteps) }
        let classes = [space.keyClasses, space.pointerClasses, space.xClasses, space.yClasses]
        let ppl = exp(zip(nll, classes).map { $0 * ActionSpaceConfig.lossScale(classes: $1) }.reduce(0, +))
        let rce = zip(nll, base).map { b in b.1 > 0 ? max(-100, 100 * (b.1 - b.0) / b.1) : 0 }
        let dirAcc = mx.isEmpty ? 0 : Double(mx.filter { ($0.0 > 0) == ($0.1 > 0) }.count) / Double(mx.count)
        return ValidationMetrics(step: step, nll: nll, perplexity: ppl, rce: rce, keyF1: keyStats.f1, clickF1: clickStats.f1,
                                 mouseDirectionAccuracy: dirAcc, mouseCorrelation: pearson(mx),
                                 visualReliance: (shuffledCE - normalCE) / Double(nSteps),
                                 languageReliance: textSteps > 0 ? (noTextCE - textCE) / Double(textSteps) : nil,
                                 historyReliance: (blindCE - normalCE) / Double(nSteps), clips: clips.count)
    }

    func binCenters(_ edges: [Double]) -> [Double] {
        (0 ... edges.count).map { i in
            if i == 0 { return edges[0].rounded(.down) }
            if i == edges.count { return edges.last!.rounded(.down) + 1 }
            return ((edges[i - 1].rounded(.down) + 1) + edges[i].rounded(.down)) / 2
        }
    }

    func pearson(_ p: [(Double, Double)]) -> Double {
        guard p.count > 2 else { return 0 }
        let n = Double(p.count)
        let ma = p.map(\.0).reduce(0, +) / n, mb = p.map(\.1).reduce(0, +) / n
        var cov = 0.0, va = 0.0, vb = 0.0
        for (a, b) in p { cov += (a - ma) * (b - mb); va += (a - ma) * (a - ma); vb += (b - mb) * (b - mb) }
        return va > 0 && vb > 0 ? cov / (va * vb).squareRoot() : 0
    }
}

/// Precision / recall of press onsets (a class appearing that wasn't held the step before), ±2 steps.
struct OnsetStats {
    var tp = 0, fp = 0, fn = 0

    mutating func add(pred: [Set<Int32>], truth: [Set<Int32>], mask: [Bool]) {
        func onsets(_ s: [Set<Int32>]) -> [(Int, Int32)] {
            var out: [(Int, Int32)] = []
            for t in s.indices where mask[t] { for c in s[t] where t == 0 || !s[t - 1].contains(c) { out.append((t, c)) } }
            return out
        }
        let p = onsets(pred), g = onsets(truth)
        var used = Set<Int>()
        for (t, c) in p {
            if let j = g.indices.first(where: { !used.contains($0) && g[$0].1 == c && abs(g[$0].0 - t) <= 2 }) { used.insert(j); tp += 1 } else { fp += 1 }
        }
        fn += g.count - used.count
    }

    var f1: Double {
        let p = tp + fp > 0 ? Double(tp) / Double(tp + fp) : 0, r = tp + fn > 0 ? Double(tp) / Double(tp + fn) : 0
        return p + r > 0 ? 2 * p * r / (p + r) : 0
    }
}
