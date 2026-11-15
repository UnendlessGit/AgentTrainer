import ATCore
import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import ATModel

func tinyArchitecture(window: Int? = nil) -> PolicyArchitecture {
    var a = PolicyArchitecture()
    a.width = 128; a.layers = 2; a.heads = 2; a.kvHeads = 2; a.decoderHeads = 2; a.decoderLayers = 2
    a.frameSize = 32
    if let w = window { a.historySteps = [w, w] }
    return a
}

@Suite(.serialized) struct PolicyTests {
    @Test func maskMatchesPixel2PlayTable() {
        let l = PolicyLayout(thinkingTokens: 1, slots: 8)
        let T = 3, tps = l.tokensPerStep
        let m = l.fullMask(steps: T, history: T - 1).asArray(Bool.self)
        for q in 0 ..< T * tps {
            for k in 0 ..< T * tps {
                #expect(m[q * T * tps + k] == l.allowed(qs: q / tps, qr: q % tps, ks: k / tps, kr: k % tps, history: T - 1))
            }
        }
        // Spot checks from the P2P mask: nothing reads a_out; a_out can't see this step's actions.
        #expect(!l.allowed(qs: 2, qr: 0, ks: 1, kr: 3, history: 9))
        #expect(!l.allowed(qs: 1, qr: 3, ks: 1, kr: 5, history: 9))
        #expect(l.allowed(qs: 1, qr: 3, ks: 0, kr: 5, history: 9)) // past actions are fed back
        #expect(l.allowed(qs: 1, qr: 6, ks: 1, kr: 9, history: 9)) // actions see each other
        #expect(!l.allowed(qs: 1, qr: 1, ks: 1, kr: 6, history: 9)) // image can't see same-step actions
        #expect(!l.allowed(qs: 5, qr: 0, ks: 1, kr: 0, history: 3)) // outside the memory window
    }

    @Test func streamingMatchesBatchAcrossWindowSlides() {
        MLXRandom.seed(3)
        let W = 3, T = 9
        let arch = tinyArchitecture(window: W)
        let actions = ActionSpaceConfig.gameDefault
        let model = PolicyModel(architecture: arch, actions: actions)
        model.train(false)
        let S = actions.slotCount
        let frames = PolicyModel.randomFrames(T, size: 32).reshaped(1, T, 32, 32, 3)
        let counts = actions.slotClassCounts
        let acts = stacked((0 ..< S).map { MLXRandom.randInt(low: 0, high: counts[$0], [1, T]).asType(.int32) }, axis: -1)
        let text = MLXRandom.normal([1, T, 768])
        let present = MLXArray((0 ..< T).map { $0 % 2 == 0 }).reshaped(1, T)
        let batch = model(frames: frames, actionsIn: acts, text: text, textPresent: present)
        let stepper = PolicyStepper(model: model, contextSteps: W)
        stepper.rebaseThreshold = 100 // exercise position rebasing too
        let a = acts.asArray(Int32.self), pres = present.asArray(Bool.self)
        var worst: Float = 0
        for t in 0 ..< T {
            let r = stepper.step(frame: frames[0, t], text: pres[t] ? text[0, t].asArray(Float.self) : nil,
                                 previousAction: t > 0 ? Array(a[(t - 1) * S ..< t * S]) : nil,
                                 forced: Array(a[t * S ..< (t + 1) * S]), returnLogits: true)
            worst = max(worst, abs(MLXArray(r.logits![0]) - batch.keys[0, t, 0]).max().item(Float.self))
            worst = max(worst, abs(MLXArray(r.logits![S - 1]) - batch.y[0, t, 0]).max().item(Float.self))
        }
        #expect(worst < 1e-3)
    }

    @Test func samplingProducesValidClasses() {
        let model = PolicyModel(architecture: tinyArchitecture(), actions: .gameDefault)
        model.train(false)
        let stepper = PolicyStepper(model: model, contextSteps: 4)
        let frame = PolicyModel.randomFrames(1, size: 32)[0]
        let counts = model.actions.slotClassCounts
        for _ in 0 ..< 6 {
            let r = stepper.step(frame: frame, text: nil)
            #expect(r.classes.count == counts.count)
            for (c, n) in zip(r.classes, counts) { #expect(c >= 0 && Int(c) < n) }
        }
    }

    @Test func weightsRoundTrip() throws {
        let a = PolicyModel(architecture: tinyArchitecture(), actions: .p2p)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("at-weights-\(UUID()).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try a.saveWeights(to: url)
        let b = PolicyModel(architecture: tinyArchitecture(), actions: .p2p)
        try b.loadWeights(from: url)
        let pa = Dictionary(uniqueKeysWithValues: a.parameters().flattened())
        for (k, v) in b.parameters().flattened() { #expect(allClose(v, pa[k]!).item(Bool.self), "\(k)") }
    }

    @Test func torchCheckpointReaderReadsRealTorchFile() throws {
        let url = Bundle.module.url(forResource: "Fixtures/tiny_torch", withExtension: "ckpt")!
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: Bundle.module.url(forResource: "Fixtures/tiny_torch", withExtension: "json")!)) as! [String: Any]
        let reader = try TorchCheckpointReader(url: url)
        let sd = try reader.stateDict()
        #expect(Set(sd.keys) == ["a", "b", "c", "p"])
        for name in ["a", "b", "c", "p"] {
            let arr = try reader.load(sd[name]!)
            let expected = (json[name] as! [NSNumber]).map(\.floatValue)
            let got = arr.asType(.float32).flattened().asArray(Float.self)
            #expect(arr.shape == (json["shapes"] as! [String: [Int]])[name]!)
            #expect(zip(got, expected).allSatisfy { abs($0 - $1) < 1e-6 }, "\(name)")
        }
        #expect(sd["b"]!.dtype == .bfloat16 && sd["c"]!.dtype == .int64)
    }
}

@Suite(.serialized) struct LanguagePolicyTests {
    func arch() -> PolicyArchitecture {
        var a = tinyArchitecture(window: 3)
        a.screenText = true; a.screenTextItems = 2; a.literalTyping = true; a.typingBytes = 8
        return a
    }

    @Test func streamingMatchesBatchWithLanguageMemory() {
        MLXRandom.seed(11)
        let a = arch(), T = 6, W = 3, K = 2, L = 8
        let actions = ActionSpaceConfig.appDefault
        let model = PolicyModel(architecture: a, actions: actions)
        // Open the gates so the memory path actually contributes.
        for m in model.decoder.memoryAttention { m.update(parameters: ModuleParameters.unflattened(["gate": MLXArray([Float(0.7)])])) }
        model.screenSummary?.update(parameters: ModuleParameters.unflattened(["weight": MLXRandom.normal([a.width, a.width]) * 0.02]))
        model.train(false)
        let S = actions.slotCount
        let frames = PolicyModel.randomFrames(T, size: 32).reshaped(1, T, 32, 32, 3)
        let counts = actions.slotClassCounts
        let acts = stacked((0 ..< S).map { MLXRandom.randInt(low: 0, high: counts[$0], [1, T]).asType(.int32) }, axis: -1)
        let emb = MLXRandom.normal([1, T, K, 768]), box = MLXRandom.uniform(0 ..< 1, [1, T, K, 5])
        let mask = MLXArray((0 ..< T * K).map { $0 % 3 != 1 }, [1, T, K])
        let bytes = MLXArray((0 ..< T).flatMap { t in LanguageInputs.bytes(t % 2 == 0 ? "type hi" : nil, length: L) }, [1, T, L])
        let lang = LanguageInputs(screenEmbeddings: emb, screenBoxes: box, screenMask: mask, typingBytes: bytes)
        let batch = model(frames: frames, actionsIn: acts, text: MLXArray.zeros([1, T, 768]), textPresent: MLXArray.zeros([1, T], dtype: .bool), language: lang)
        let stepper = PolicyStepper(model: model, contextSteps: W)
        let flat = acts.asArray(Int32.self)
        var worst: Float = 0
        for t in 0 ..< T {
            stepper.setLanguage(LanguageInputs(screenEmbeddings: emb[0, t], screenBoxes: box[0, t], screenMask: mask[0, t], typingBytes: bytes[0, t]))
            let r = stepper.step(frame: frames[0, t], text: nil, previousAction: t > 0 ? Array(flat[(t - 1) * S ..< t * S]) : nil,
                                 forced: Array(flat[t * S ..< (t + 1) * S]), returnLogits: true)
            worst = max(worst, abs(MLXArray(r.logits![0]) - batch.keys[0, t, 0]).max().item(Float.self))
            worst = max(worst, abs(MLXArray(r.logits![S - 1]) - batch.y[0, t, 0]).max().item(Float.self))
        }
        #expect(worst < 1e-3)
    }

    @Test func closedGatesLeaveOutputsUnchanged() {
        MLXRandom.seed(5)
        let a = arch(), T = 3
        let model = PolicyModel(architecture: a, actions: .appDefault)
        model.train(false)
        let frames = PolicyModel.randomFrames(T, size: 32).reshaped(1, T, 32, 32, 3)
        let acts = MLXArray.zeros([1, T, model.actions.slotCount], dtype: .int32)
        let text = MLXArray.zeros([1, T, 768]), present = MLXArray.zeros([1, T], dtype: .bool)
        let lang = LanguageInputs(screenEmbeddings: MLXRandom.normal([1, T, 2, 768]), screenBoxes: MLXRandom.uniform(0 ..< 1, [1, T, 2, 5]),
                                  screenMask: MLXArray.ones([1, T, 2], dtype: .bool), typingBytes: MLXArray(LanguageInputs.bytes("type x", length: 8) + LanguageInputs.bytes("type x", length: 8) + LanguageInputs.bytes("type x", length: 8), [1, T, 8]))
        let with = model(frames: frames, actionsIn: acts, text: text, textPresent: present, language: lang)
        let without = model(frames: frames, actionsIn: acts, text: text, textPresent: present)
        #expect(abs(with.keys - without.keys).max().item(Float.self) < 1e-5)
    }

    @Test func historyDropoutTokenIsNeutralAtInit() {
        let model = PolicyModel(architecture: tinyArchitecture(), actions: .p2p)
        #expect(model.actionUnknown.abs().max().item(Float.self) == 0)
    }
}
