import Foundation
import MLX
import MLXFast
import MLXNN

/// RMSNorm computed in fp32 (Pixel2Play `RMSNormF32`, eps 1e-5, learned `scale`).
final class RMSNormF32: Module, UnaryLayer {
    @ParameterInfo(key: "scale") var scale: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-5) {
        _scale.wrappedValue = MLXArray.ones([dimensions])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: scale.asType(x.dtype), eps: eps)
    }
}

/// Multi-head self-attention with a fused qkv projection, QK-RMSNorm over the full width
/// (all heads jointly, as in P2P), interleaved ("traditional") RoPE and optional learned KV sinks.
final class SelfAttention: Module {
    @ModuleInfo(key: "c_attn") var cAttn: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNormF32
    @ModuleInfo(key: "k_norm") var kNorm: RMSNormF32
    @ParameterInfo(key: "k_sinks") var kSinks: MLXArray?
    @ParameterInfo(key: "v_sinks") var vSinks: MLXArray?
    @ModuleInfo(key: "residual_dropout") var dropout: Dropout

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let sinks: Int
    let scale: Float

    init(width: Int, heads: Int, kvHeads: Int, dropout: Float, sinks: Int) {
        self.heads = heads
        self.kvHeads = kvHeads
        self.headDim = width / heads
        self.sinks = sinks
        self.scale = 1 / Float(headDim).squareRoot()
        _cAttn.wrappedValue = Linear(width, heads * headDim + 2 * kvHeads * headDim, bias: false)
        _cProj.wrappedValue = Linear(heads * headDim, width, bias: false)
        _qNorm.wrappedValue = RMSNormF32(dimensions: heads * headDim)
        _kNorm.wrappedValue = RMSNormF32(dimensions: kvHeads * headDim)
        _dropout.wrappedValue = Dropout(p: dropout)
        if sinks > 0 {
            _kSinks.wrappedValue = MLXRandom.normal([1, kvHeads, sinks, width / heads])
            _vSinks.wrappedValue = MLXRandom.normal([1, kvHeads, sinks, width / heads])
        }
    }

    /// Returns q `[B,H,T,hd]` and k, v `[B,Hkv,T,hd]` with norms and RoPE applied.
    /// `ropeOffset` is the position of the first token.
    func projectQKV(_ x: MLXArray, ropeOffset: Int) -> (MLXArray, MLXArray, MLXArray) {
        let (B, T) = (x.dim(0), x.dim(1))
        let qkv = cAttn(x)
        let qw = heads * headDim, kw = kvHeads * headDim
        var q = qNorm(qkv[.ellipsis, 0 ..< qw])
        var k = kNorm(qkv[.ellipsis, qw ..< qw + kw])
        var v = qkv[.ellipsis, (qw + kw)...]
        q = q.reshaped(B, T, heads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(B, T, kvHeads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(B, T, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = MLXFast.RoPE(q, dimensions: headDim, traditional: true, base: 10_000, scale: 1, offset: ropeOffset)
        k = MLXFast.RoPE(k, dimensions: headDim, traditional: true, base: 10_000, scale: 1, offset: ropeOffset)
        return (q, k, v)
    }

    /// Same as `projectQKV` with the RoPE offset as an array (lets the step be compiled once).
    func projectQKV(_ x: MLXArray, ropeOffset: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let (B, T) = (x.dim(0), x.dim(1))
        let qkv = cAttn(x)
        let qw = heads * headDim, kw = kvHeads * headDim
        var q = qNorm(qkv[.ellipsis, 0 ..< qw])
        var k = kNorm(qkv[.ellipsis, qw ..< qw + kw])
        var v = qkv[.ellipsis, (qw + kw)...]
        q = q.reshaped(B, T, heads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(B, T, kvHeads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(B, T, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = MLXFast.RoPE(q, dimensions: headDim, traditional: true, base: 10_000, scale: 1, offset: ropeOffset)
        k = MLXFast.RoPE(k, dimensions: headDim, traditional: true, base: 10_000, scale: 1, offset: ropeOffset)
        return (q, k, v)
    }

    /// Attention over `keys`/`values` (already including any cache), then output projection.
    /// `mask` is boolean `[T, S]` (true = may attend), sinks are prepended and always visible.
    func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXArray?) -> MLXArray {
        let B = q.dim(0), T = q.dim(2)
        var k = k, v = v, mask = mask
        if sinks > 0, let ks = kSinks, let vs = vSinks {
            k = concatenated([broadcast(ks.asType(k.dtype), to: [B, kvHeads, sinks, headDim]), k], axis: 2)
            v = concatenated([broadcast(vs.asType(v.dtype), to: [B, kvHeads, sinks, headDim]), v], axis: 2)
            if let m = mask {
                let allow = MLXArray.ones([m.dim(0), sinks], dtype: .bool)
                mask = concatenated([allow, m], axis: 1)
            }
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale,
            mask: mask.map { .array($0) } ?? .none)
        let y = out.transposed(0, 2, 1, 3).reshaped(B, T, heads * headDim)
        return dropout(cProj(y))
    }
}

extension SelfAttention {
    /// Attention computed per query block over only the key range that block may see. With a
    /// mask that is causal at timestep granularity, every key outside the range is masked for
    /// every query in the block, so the result equals full masked attention exactly while
    /// skipping roughly half of the score matrix.
    func attendChunked(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXArray, chunks: [(Range<Int>, Range<Int>)]) -> MLXArray {
        precondition(sinks == 0, "chunked attention is for the backbone (no sinks)")
        let B = q.dim(0), T = q.dim(2)
        let parts = chunks.map { (qr, kr) -> MLXArray in
            MLXFast.scaledDotProductAttention(
                queries: q[0..., 0..., qr, 0...], keys: k[0..., 0..., kr, 0...], values: v[0..., 0..., kr, 0...], scale: scale,
                mask: .array(mask[qr, kr]))
        }
        let out = concatenated(parts, axis: 2)
        let y = out.transposed(0, 2, 1, 3).reshaped(B, T, heads * headDim)
        return dropout(cProj(y))
    }
}

/// SwiGLU feed-forward with a packed gate/up projection (`w13`), P2P `PackedSwiGLUFFN`.
final class SwiGLU: Module, UnaryLayer {
    @ModuleInfo(key: "w13") var w13: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    let hidden: Int

    init(width: Int) {
        hidden = PolicyArchitectureMath.swigluHidden(width)
        _w13.wrappedValue = Linear(width, 2 * hidden, bias: false)
        _w2.wrappedValue = Linear(hidden, width, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = w13(x)
        return w2(silu(h[.ellipsis, 0 ..< hidden]) * h[.ellipsis, hidden...])
    }
}

/// Pre-norm transformer block (no final norm in P2P).
final class TransformerLayer: Module {
    @ModuleInfo(key: "self_attention") var attention: SelfAttention
    @ModuleInfo(key: "ffn") var ffn: SwiGLU
    @ModuleInfo(key: "self_attention_norm") var attentionNorm: RMSNormF32
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNormF32

    init(width: Int, heads: Int, kvHeads: Int, dropout: Float, sinks: Int) {
        _attention.wrappedValue = SelfAttention(width: width, heads: heads, kvHeads: kvHeads, dropout: dropout, sinks: sinks)
        _ffn.wrappedValue = SwiGLU(width: width)
        _attentionNorm.wrappedValue = RMSNormF32(dimensions: width)
        _ffnNorm.wrappedValue = RMSNormF32(dimensions: width)
    }

    /// Full-sequence forward (training / reference). `chunks` splits attention into query blocks
    /// that each read only the keys they can see (exact for block-causal masks; see `attendChunked`).
    func callAsFunction(_ x: MLXArray, mask: MLXArray?, ropeOffset: Int = 0, chunks: [(Range<Int>, Range<Int>)]? = nil) -> MLXArray {
        let (q, k, v) = attention.projectQKV(attentionNorm(x), ropeOffset: ropeOffset)
        let a: MLXArray
        if let chunks, let mask { a = attention.attendChunked(q, k, v, mask: mask, chunks: chunks) } else { a = attention.attend(q, k, v, mask: mask) }
        let h = x + a
        return h + ffn(ffnNorm(h))
    }
}

enum PolicyArchitectureMath {
    static func swigluHidden(_ d: Int) -> Int { 8 * ((Int(2 * (4 * d) / 3) + 7) / 8) }
}
