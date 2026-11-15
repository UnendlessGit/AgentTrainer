import ATCore
import Foundation
import MLX
import MLXNN
import MLXRandom

/// Pixel2Play-style image augmentation on the GPU. Parameters are drawn once per clip and applied
/// to every frame of it (temporal consistency); only sensor noise varies per frame.
struct Augmenter {
    let enabled: Set<Augmentation>
    let strength: Float

    init(_ list: [Augmentation], strength: Double) {
        enabled = Set(list)
        self.strength = Float(strength)
    }

    private func chance(_ p: Double) -> Bool { Double.random(in: 0 ..< 1) < p }
    private func u(_ a: Float, _ b: Float) -> Float { Float.random(in: a ... b) }

    /// `frames`: `[T, S, S, 3]` float 0-255. Returns the same shape.
    func callAsFunction(_ frames: MLXArray) -> MLXArray {
        guard strength > 0, !enabled.isEmpty else { return frames }
        var x = frames
        let s = strength
        let S = frames.dim(1)

        // Geometry: small rotation/scale always (P2P spatial, p = 1), horizontal shift p = 0.25.
        var angle: Float = 0, scale: Float = 1, tx: Float = 0, ty: Float = 0
        if enabled.contains(.spatial) { angle = u(-3, 3) * s * .pi / 180; scale = 1 + u(-0.04, 0.04) * s; ty = u(-0.01, 0.01) * s }
        if enabled.contains(.translation), chance(0.25) { tx = u(-0.03, 0.03) * s }
        if angle != 0 || scale != 1 || tx != 0 || ty != 0 { x = warp(x, angle: angle, scale: scale, tx: tx * Float(S), ty: ty * Float(S)) }

        if enabled.contains(.color), chance(0.25) {
            let b = 1 + u(-0.2, 0.2) * s, c = 1 + u(-0.2, 0.2) * s, sat = 1 + u(-0.2, 0.2) * s, hue = u(-0.05, 0.05) * s
            x = colorJitter(x, brightness: b, contrast: c, saturation: sat, hue: hue)
        }
        if enabled.contains(.planckian), chance(0.25) {
            let t = u(-1, 1) * min(s, 1.5)
            let gains: [Float] = t > 0 ? [1, 1 - 0.07 * t, 1 - 0.2 * t] : [1 + 0.15 * t, 1 + 0.07 * t, 1]
            x = x * MLXArray(gains)
        }
        if enabled.contains(.blur), chance(0.2) { x = blur(x, sigma: u(0.3, 1.6) * max(s, 0.3)) }
        if enabled.contains(.sharpness), chance(0.15) {
            let amount = u(0.5, 1.5)
            x = x + (x - blur(x, sigma: 1)) * (amount - 1)
        }
        if enabled.contains(.isoNoise), chance(0.1) {
            let sigma = u(2, 12) * s
            x = x + MLXRandom.normal(x.shape) * sigma
        }
        return clip(x, min: 0, max: 255).round()
    }

    /// Bilinear warp about the image center with edge clamping.
    func warp(_ x: MLXArray, angle: Float, scale: Float, tx: Float, ty: Float) -> MLXArray {
        let T = x.dim(0), S = x.dim(1)
        let c = Float(S - 1) / 2
        let idx = MLXArray(0 ..< S).asType(.float32)
        let yy = broadcast(idx.expandedDimensions(axis: 1), to: [S, S]) - c
        let xx = broadcast(idx.expandedDimensions(axis: 0), to: [S, S]) - c
        let ca = cos(-angle) / scale, sa = sin(-angle) / scale
        let sx = clip(xx * ca - yy * sa + c - tx, min: 0, max: Float(S - 1))
        let sy = clip(xx * sa + yy * ca + c - ty, min: 0, max: Float(S - 1))
        let x0 = floor(sx), y0 = floor(sy)
        let x1 = minimum(x0 + 1, MLXArray(Float(S - 1))), y1 = minimum(y0 + 1, MLXArray(Float(S - 1)))
        let wx = (sx - x0).expandedDimensions(axis: -1), wy = (sy - y0).expandedDimensions(axis: -1)
        let flat = x.reshaped(T, S * S, 3)
        func gather(_ yi: MLXArray, _ xi: MLXArray) -> MLXArray {
            let i = (yi * Float(S) + xi).asType(.int32).reshaped(-1)
            return flat.take(i, axis: 1).reshaped(T, S, S, 3)
        }
        let top = gather(y0, x0) * (1 - wx) + gather(y0, x1) * wx
        let bottom = gather(y1, x0) * (1 - wx) + gather(y1, x1) * wx
        return top * (1 - wy) + bottom * wy
    }

    func colorJitter(_ x: MLXArray, brightness b: Float, contrast c: Float, saturation sat: Float, hue: Float) -> MLXArray {
        var y = x * b
        let gray = (y[.ellipsis, 0 ..< 1] * 0.299 + y[.ellipsis, 1 ..< 2] * 0.587 + y[.ellipsis, 2 ..< 3] * 0.114)
        let mean = gray.mean(axes: [1, 2, 3], keepDims: true)
        y = (y - mean) * c + mean
        let g2 = (y[.ellipsis, 0 ..< 1] * 0.299 + y[.ellipsis, 1 ..< 2] * 0.587 + y[.ellipsis, 2 ..< 3] * 0.114)
        y = (y - g2) * sat + g2
        if hue != 0 {
            // Rotate chroma in YIQ space.
            let th = hue * 2 * .pi, cs = cos(th), sn = sin(th)
            let toYIQ = MLXArray([0.299, 0.587, 0.114, 0.596, -0.274, -0.322, 0.211, -0.523, 0.312] as [Float]).reshaped(3, 3)
            let rot = MLXArray([1, 0, 0, 0, cs, -sn, 0, sn, cs] as [Float]).reshaped(3, 3)
            let fromYIQ = MLXArray([1, 0.956, 0.621, 1, -0.272, -0.647, 1, -1.106, 1.703] as [Float]).reshaped(3, 3)
            let m = matmul(fromYIQ, matmul(rot, toYIQ))
            y = matmul(y, m.transposed())
        }
        return y
    }

    /// Separable Gaussian blur via depthwise convolution.
    func blur(_ x: MLXArray, sigma: Float) -> MLXArray {
        let r = 2
        let k = (-r ... r).map { exp(-Float($0 * $0) / (2 * sigma * sigma)) }
        let sum = k.reduce(0, +)
        let kernel = MLXArray(k.map { $0 / sum })
        let kh = broadcast(kernel.reshaped(1, 1, 5, 1), to: [3, 1, 5, 1])
        let kv = broadcast(kernel.reshaped(1, 5, 1, 1), to: [3, 5, 1, 1])
        var y = padded(x, widths: [.init(0), .init((0, 0)), .init((r, r)), .init(0)], mode: .edge)
        y = conv2d(y, kh, groups: 3)
        y = padded(y, widths: [.init(0), .init((r, r)), .init((0, 0)), .init(0)], mode: .edge)
        return conv2d(y, kv, groups: 3)
    }
}
