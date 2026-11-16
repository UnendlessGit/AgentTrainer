import Foundation
import MLX
import MLXNN

/// Developer benchmark: where the EfficientNet stem spends its time.
public enum StemBench {
    public static func ab(frames n: Int, size: Int = 192) {
        let tok = ConvTokenizer(frameSize: size, width: 768, stochasticDepth: 0.2)
        // Non-trivial BN statistics so folding is actually exercised.
        tok.update(parameters: ModuleParameters.unflattened(tok.parameters().flattened().map { k, v in
            k.hasSuffix("running_var") ? (k, MLXRandom.uniform(0.5 ..< 2, v.shape)) : k.hasSuffix("running_mean") ? (k, MLXRandom.normal(v.shape) * 0.1) :
            k.hasSuffix("bn.weight") ? (k, MLXRandom.uniform(0.5 ..< 1.5, v.shape)) : k.hasSuffix("bn.bias") ? (k, MLXRandom.normal(v.shape) * 0.1) : (k, v) }))
        let x32 = MLXRandom.uniform(0 ..< 1, [n, size, size, 3])
        tok.train(false)
        ConvBNAct.fused = false; let a = tok.stem(x32[0 ..< 8]); ConvBNAct.fused = true; let b = tok.stem(x32[0 ..< 8])
        print(String(format: "fp32 eval: max |plain − fused| %.2e (mean |y| %.3f)", abs(a - b).max().item(Float.self), abs(a).mean().item(Float.self)))
        tok.update(parameters: ModuleParameters.unflattened(tok.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        let x = x32.asType(.bfloat16)
        tok.train(true)
        for v in [false, true] {
            ConvBNAct.fused = v
            let f = { tok.stem(x) }
            let vg = valueAndGrad(model: tok) { (m: ConvTokenizer, a: [MLXArray]) -> [MLXArray] in [m.stem(a[0]).asType(.float32).square().mean()] }
            eval(f()); eval(vg(tok, [x]).1)
            var t0 = Date(); for _ in 0 ..< 3 { eval(f()) }
            let fw = Date().timeIntervalSince(t0) / 3 * 1000
            t0 = Date(); for _ in 0 ..< 3 { let (l, gr) = vg(tok, [x]); eval(l, gr) }
            let fb = Date().timeIntervalSince(t0) / 3 * 1000
            print(String(format: "%@ stem, %d frames, train mode: forward %6.1f ms   forward+backward %6.1f ms", v ? "fused" : "plain", n, fw, fb))
        }
        ConvBNAct.fused = true
    }

    public static func run(frames n: Int, size: Int = 192, dtype: DType = .bfloat16) {
        let tok = ConvTokenizer(frameSize: size, width: 768, stochasticDepth: 0)
        tok.train(false)
        tok.update(parameters: ModuleParameters.unflattened(tok.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) }))
        let x = (MLXRandom.uniform(0 ..< 1, [n, size, size, 3])).asType(dtype)
        eval(x, tok)
        func time(_ label: String, reps: Int = 3, _ f: () -> MLXArray) -> Double {
            eval(f())
            let t0 = Date()
            for _ in 0 ..< reps { eval(f()) }
            let ms = Date().timeIntervalSince(t0) / Double(reps) * 1000
            print(String(format: "%-48@ %8.1f ms", label as NSString, ms))
            return ms
        }
        _ = time("full stem forward (\(n) frames)") { tok.stem(x) }
        // Per block: conv parts vs elementwise parts.
        var y = tok.stem.stem(x); eval(y)
        var conv = 0.0, dw = 0.0, ew = 0.0, se = 0.0
        conv += time("stem conv 3x3 s2 (3→32)") { tok.stem.stem.conv(x) }
        for (i, b) in tok.stem.blocks.enumerated() {
            var h = y
            if let e = b.expand { let hh = h; conv += time("b\(i) expand 1x1 conv") { e.conv(hh) }; h = e(h); eval(h) }
            let hh = h
            dw += time("b\(i) depthwise \(b.depthwise.conv.weight.dim(1))x\(b.depthwise.conv.weight.dim(1)) conv \(h.shape)") { b.depthwise.conv(hh) }
            let d = b.depthwise.conv(hh); eval(d)
            ew += time("b\(i) BN + SiLU after depthwise") { silu(b.depthwise.bn(d)) }
            let a = b.depthwise(hh); eval(a)
            se += time("b\(i) squeeze-excite") { b.se(a) }
            let s2 = b.se(a); eval(s2)
            conv += time("b\(i) project 1x1 conv") { b.project.conv(s2) }
            y = b(y); eval(y)
        }
        print(String(format: "SUM  1x1/regular conv %.0f ms | depthwise conv %.0f ms | BN+SiLU %.0f ms | SE %.0f ms", conv, dw, ew, se))
    }
}

/// Developer benchmark: one backbone layer forward+backward at training shapes, full vs chunked attention.
public enum AttentionBench {
    public static func run(batch B: Int, steps T: Int, width D: Int, heads H: Int) {
        let layout = PolicyLayout(thinkingTokens: 1, slots: 8)
        let tps = layout.tokensPerStep, N = T * tps
        let layer = TransformerLayer(width: D, heads: H, kvHeads: H, dropout: 0.1, sinks: 0)
        layer.update(parameters: ModuleParameters.unflattened(layer.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        layer.train(true)
        let x = (MLXRandom.normal([B, N, D]) * 0.5).asType(.bfloat16)
        let mask = layout.fullMask(steps: T, history: T - 1)
        eval(x, mask, layer)
        func run(_ label: String, chunks: Int) {
            let ranges: [(Range<Int>, Range<Int>)]? = chunks <= 1 ? nil : (0 ..< chunks).map { c in
                let s0 = T * c / chunks, s1 = T * (c + 1) / chunks
                return (s0 * tps ..< s1 * tps, 0 ..< s1 * tps)
            }
            let vg = valueAndGrad(model: layer) { (m: TransformerLayer, a: [MLXArray]) -> [MLXArray] in
                [m(a[0], mask: a[1], chunks: ranges).asType(.float32).square().mean()]
            }
            let fwd = { layer(x, mask: mask, chunks: ranges) }
            eval(fwd()); eval(vg(layer, [x, mask]).1)
            var t0 = Date()
            for _ in 0 ..< 3 { eval(fwd()) }
            let f = Date().timeIntervalSince(t0) / 3 * 1000
            t0 = Date()
            for _ in 0 ..< 3 { let (v, g) = vg(layer, [x, mask]); eval(v, g) }
            let fb = Date().timeIntervalSince(t0) / 3 * 1000
            print(String(format: "%-28@ forward %7.1f ms   forward+backward %7.1f ms", label as NSString, f, fb))
        }
        print("one layer: B \(B), \(T) steps × \(tps) = \(N) tokens, D \(D), \(H) heads")
        run("full mask", chunks: 1)
        for c in [2, 4, 6, 8, 12] { run("\(c) query chunks", chunks: c) }
        // Check equality.
        let full = layer(x, mask: mask, chunks: nil).asType(.float32)
        let ch = layer(x, mask: mask, chunks: (0 ..< 4).map { c in (T * c / 4 * tps ..< T * (c + 1) / 4 * tps, 0 ..< T * (c + 1) / 4 * tps) }).asType(.float32)
        layer.train(false)
        let a = layer(x, mask: mask, chunks: nil).asType(.float32), b = layer(x, mask: mask, chunks: (0 ..< 4).map { c in (T * c / 4 * tps ..< T * (c + 1) / 4 * tps, 0 ..< T * (c + 1) / 4 * tps) }).asType(.float32)
        _ = (full, ch)
        print(String(format: "max |full − chunked| (eval mode, bf16): %.5f  (mean |x| %.3f)", abs(a - b).max().item(Float.self), abs(a).mean().item(Float.self)))
    }
}

public enum LayerPartsBench {
    public static func run(batch B: Int, steps T: Int, width D: Int, heads H: Int) {
        let layout = PolicyLayout(thinkingTokens: 1, slots: 8)
        let tps = layout.tokensPerStep, N = T * tps, hd = D / H
        let mask = layout.fullMask(steps: T, history: T - 1)
        let q = (MLXRandom.normal([B, H, N, hd])).asType(.bfloat16), k = (MLXRandom.normal([B, H, N, hd])).asType(.bfloat16), v = (MLXRandom.normal([B, H, N, hd])).asType(.bfloat16)
        eval(q, k, v, mask)
        func t(_ label: String, _ f: @escaping ([MLXArray]) -> MLXArray, _ inputs: [MLXArray]) {
            let g = grad({ (a: [MLXArray]) -> MLXArray in f(a).asType(.float32).sum() })
            eval(f(inputs)); eval(g(inputs))
            var t0 = Date(); for _ in 0 ..< 3 { eval(f(inputs)) }
            let fw = Date().timeIntervalSince(t0) / 3 * 1000
            t0 = Date(); for _ in 0 ..< 3 { eval(g(inputs)) }
            let bw = Date().timeIntervalSince(t0) / 3 * 1000
            print(String(format: "%-44@ fwd %7.1f ms   grad %7.1f ms", label as NSString, fw, bw))
        }
        t("SDPA bool mask", { a in MLXFast.scaledDotProductAttention(queries: a[0], keys: a[1], values: a[2], scale: 0.125, mask: .array(mask)) }, [q, k, v])
        t("SDPA additive mask", { a in MLXFast.scaledDotProductAttention(queries: a[0], keys: a[1], values: a[2], scale: 0.125, mask: .array(which(mask, MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(.bfloat16))) }, [q, k, v])
        t("SDPA causal (reference)", { a in MLXFast.scaledDotProductAttention(queries: a[0], keys: a[1], values: a[2], scale: 0.125, mask: .causal) }, [q, k, v])
        let x = (MLXRandom.normal([B, N, D]) * 0.5).asType(.bfloat16)
        let ffn = SwiGLU(width: D); ffn.update(parameters: ModuleParameters.unflattened(ffn.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        let att = SelfAttention(width: D, heads: H, kvHeads: H, dropout: 0, sinks: 0); att.update(parameters: ModuleParameters.unflattened(att.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        let norm = RMSNormF32(dimensions: D)
        eval(x, ffn, att, norm)
        t("FFN (SwiGLU)", { a in ffn(a[0]) }, [x])
        t("qkv proj + QK norm + RoPE", { a in let (q, k, v) = att.projectQKV(a[0], ropeOffset: 0); return q + k + v }, [x])
        t("RMSNormF32", { a in norm(a[0]) }, [x])
        t("output proj", { a in att.cProj(a[0]) }, [x])
    }
}

public enum ConvGradBench {
    public static func run(n: Int) {
        func t(_ label: String, x: MLXArray, w: MLXArray, stride: Int, pad: Int, groups: Int) {
            let f = { (a: [MLXArray]) -> MLXArray in (groups > 1 ? DepthwiseConv.apply(a[0], a[1], stride: stride, padding: pad) : conv2d(a[0], a[1], stride: .init(stride), padding: .init(pad), groups: groups)).asType(.float32).sum() }
            let gx = grad({ (a: [MLXArray]) -> MLXArray in f(a) }, argumentNumbers: [0])
            let gw = grad({ (a: [MLXArray]) -> MLXArray in f(a) }, argumentNumbers: [1])
            eval(f([x, w])); eval(gx([x, w])); eval(gw([x, w]))
            var t0 = Date(); for _ in 0 ..< 3 { eval(f([x, w])) }; let fw = Date().timeIntervalSince(t0) / 3 * 1000
            t0 = Date(); for _ in 0 ..< 3 { eval(gx([x, w])) }; let bx = Date().timeIntervalSince(t0) / 3 * 1000
            t0 = Date(); for _ in 0 ..< 3 { eval(gw([x, w])) }; let bw = Date().timeIntervalSince(t0) / 3 * 1000
            print(String(format: "%-46@ fwd %7.1f  d/dx %7.1f  d/dw %7.1f ms", label as NSString, fw, bx, bw))
        }
        let dt = DType.bfloat16
        t("depthwise 3x3 s1 [n,96,96,32]", x: MLXRandom.normal([n, 96, 96, 32]).asType(dt), w: MLXRandom.normal([32, 3, 3, 1]).asType(dt), stride: 1, pad: 1, groups: 32)
        t("depthwise 3x3 s2 [n,96,96,96]", x: MLXRandom.normal([n, 96, 96, 96]).asType(dt), w: MLXRandom.normal([96, 3, 3, 1]).asType(dt), stride: 2, pad: 1, groups: 96)
        t("depthwise 5x5 s1 [n,48,48,144]", x: MLXRandom.normal([n, 48, 48, 144]).asType(dt), w: MLXRandom.normal([144, 5, 5, 1]).asType(dt), stride: 1, pad: 2, groups: 144)
        t("depthwise 5x5 s1 [n,12,12,672]", x: MLXRandom.normal([n, 12, 12, 672]).asType(dt), w: MLXRandom.normal([672, 5, 5, 1]).asType(dt), stride: 1, pad: 2, groups: 672)
        t("1x1 conv [n,48,48,24]→144", x: MLXRandom.normal([n, 48, 48, 24]).asType(dt), w: MLXRandom.normal([144, 1, 1, 24]).asType(dt), stride: 1, pad: 0, groups: 1)
        t("stem 3x3 s2 [n,192,192,3]→32", x: MLXRandom.normal([n, 192, 192, 3]).asType(dt), w: MLXRandom.normal([32, 3, 3, 3]).asType(dt), stride: 2, pad: 1, groups: 1)
    }
}

public enum DepthwiseCheck {
    public static func run() {
        // Gradient equality vs MLX's own (fp32), for every stride/kernel used by the stem.
        for (k, st, h, c) in [(3, 1, 13, 8), (3, 2, 13, 8), (5, 2, 14, 6), (5, 1, 12, 5), (3, 2, 12, 4)] {
            let x = MLXRandom.normal([2, h, h + 1, c]), w = MLXRandom.normal([c, k, k, 1])
            let p = (k - 1) / 2
            let r = MLXRandom.normal(conv2d(x, w, stride: .init(st), padding: .init(p), groups: c).shape)
            func loss(_ a: [MLXArray]) -> MLXArray { (DepthwiseConv.apply(a[0], a[1], stride: st, padding: p) * r).sum() }
            DepthwiseConv.customBackward = false
            let ref = grad(loss, argumentNumbers: [0, 1])([x, w])
            DepthwiseConv.customBackward = true
            let mine = grad(loss, argumentNumbers: [0, 1])([x, w])
            let ex = abs(ref[0] - mine[0]).max().item(Float.self) / max(abs(ref[0]).max().item(Float.self), 1e-6)
            let ew = abs(ref[1] - mine[1]).max().item(Float.self) / max(abs(ref[1]).max().item(Float.self), 1e-6)
            print(String(format: "k%d s%d %dx%d c%d: rel err dx %.1e  dw %.1e", k, st, h, h + 1, c, ex, ew))
        }
        DepthwiseConv.customBackward = true
    }
}
