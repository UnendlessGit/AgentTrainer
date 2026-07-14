import AppKit
import Foundation

struct CNNVisualizationRender {
    var image: NSImage?
    var detail: String
}

/// Coalesces diagnostic frames independently of inference. Slow CPU image
/// conversion can never build a queue or delay the action path.
final class CNNVisualizationRenderer: @unchecked Sendable {
    private let lock = NSLock()
    private let queue = DispatchQueue(label: "AgentTrainer.CNNVisualization", qos: .utility)
    private var newest: (CNNVisualizationFrame, @Sendable (CNNVisualizationRender) -> Void)?
    private var processing = false

    func submit(_ frame: CNNVisualizationFrame, completion: @escaping @Sendable (CNNVisualizationRender) -> Void) {
        lock.lock()
        newest = (frame, completion)
        guard !processing else { lock.unlock(); return }
        processing = true
        lock.unlock()
        queue.async { [weak self] in self?.drain() }
    }

    func cancel() {
        lock.lock()
        newest = nil
        lock.unlock()
    }

    private func drain() {
        while true {
            lock.lock()
            guard let next = newest else { processing = false; lock.unlock(); return }
            newest = nil
            lock.unlock()
            next.1(CNNVisualizationImageRenderer.render(next.0))
        }
    }
}

enum CNNVisualizationImageRenderer {
    private static let maximumImageWidth = 640
    private static let maximumImageHeight = 360

    static func render(_ frame: CNNVisualizationFrame) -> CNNVisualizationRender {
        guard let tensor = frame.tensors.first,
              tensor.width > 0, tensor.height > 0, tensor.channels > 0,
              tensor.values.count == tensor.width * tensor.height * tensor.channels else {
            return CNNVisualizationRender(image: nil, detail: "Waiting for a valid CNN tensor")
        }
        switch frame.settings.mode {
        case .activationOverlay:
            return overlay(frame: frame, tensor: tensor, kind: "combined activation")
        case .featureChannels:
            return featureGrid(frame: frame, tensor: tensor)
        case .actionSaliency:
            return overlay(frame: frame, tensor: tensor, kind: "\(frame.settings.actionFocus.rawValue) influence")
        }
    }

    /// Exposed to regression tests so strongest-channel selection stays bounded
    /// and deterministic without requiring AppKit image inspection.
    static func strongestChannels(in tensor: CNNFeatureTensor, count: Int) -> [Int] {
        guard tensor.channels > 0,
              tensor.width > 0,
              tensor.height > 0,
              tensor.values.count == tensor.width * tensor.height * tensor.channels else { return [] }
        var scores = [Double](repeating: 0, count: tensor.channels)
        let pixels = tensor.width * tensor.height
        for pixel in 0..<pixels {
            let base = pixel * tensor.channels
            for channel in 0..<tensor.channels {
                let value = tensor.values[base + channel]
                if value.isFinite { scores[channel] += Double(abs(value)) }
            }
        }
        return scores.indices.sorted {
            if scores[$0] == scores[$1] { return $0 < $1 }
            return scores[$0] > scores[$1]
        }.prefix(min(tensor.channels, max(1, count))).map { $0 }
    }

    private static func overlay(frame: CNNVisualizationFrame, tensor: CNNFeatureTensor, kind: String) -> CNNVisualizationRender {
        let size = boundedSize(width: frame.spec.width, height: frame.spec.height)
        guard size.width > 0, size.height > 0 else { return CNNVisualizationRender(image: nil, detail: "Invalid model vision size") }
        let channel = 0
        let scale = robustScale(values: tensor.values, channel: channel, channels: tensor.channels)
        var rgba = [UInt8](repeating: 255, count: size.width * size.height * 4)
        frame.packed.withUnsafeBytes { raw in
            guard let bytes = raw.bindMemory(to: UInt8.self).baseAddress else { return }
            for y in 0..<size.height {
                let sourceY = min(frame.spec.height - 1, y * frame.spec.height / size.height)
                for x in 0..<size.width {
                    let sourceX = min(frame.spec.width - 1, x * frame.spec.width / size.width)
                    var base = packedRGB(bytes: bytes, count: raw.count, spec: frame.spec, x: sourceX, y: sourceY)
                    let fx = (Double(x) + 0.5) * Double(tensor.width) / Double(size.width) - 0.5
                    let fy = (Double(y) + 0.5) * Double(tensor.height) / Double(size.height) - 0.5
                    let rawValue = bilinear(tensor, channel: channel, x: fx, y: fy)
                    let normalized = pow(min(1, max(0, rawValue / scale)), 0.72)
                    let heat = heatColor(normalized)
                    let alpha = frame.settings.overlayOpacity * normalized
                    base.0 = base.0 * (1 - alpha) + heat.0 * alpha
                    base.1 = base.1 * (1 - alpha) + heat.1 * alpha
                    base.2 = base.2 * (1 - alpha) + heat.2 * alpha
                    let offset = (y * size.width + x) * 4
                    rgba[offset] = byte(base.0)
                    rgba[offset + 1] = byte(base.1)
                    rgba[offset + 2] = byte(base.2)
                }
            }
        }
        return CNNVisualizationRender(
            image: image(width: size.width, height: size.height, rgba: rgba),
            detail: "\(stageLabel(tensor)) • k\(tensor.kernelSize) • stride ×\(tensor.effectiveStride) • field \(tensor.receptiveField)×\(tensor.receptiveField) • \(kind)"
        )
    }

    private static func featureGrid(frame: CNNVisualizationFrame, tensor: CNNFeatureTensor) -> CNNVisualizationRender {
        let selected = strongestChannels(in: tensor, count: frame.settings.featureChannelCount)
        guard !selected.isEmpty else { return CNNVisualizationRender(image: nil, detail: "No active feature channels") }
        let columns = max(1, Int(ceil(sqrt(Double(selected.count)))))
        let rows = Int(ceil(Double(selected.count) / Double(columns)))
        let width = 600, height = 360, gutter = 3
        let tileWidth = max(1, (width - gutter * (columns + 1)) / columns)
        let tileHeight = max(1, (height - gutter * (rows + 1)) / rows)
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for pixel in 0..<(width * height) {
            rgba[pixel * 4] = 8
            rgba[pixel * 4 + 1] = 8
            rgba[pixel * 4 + 2] = 13
        }
        for (tile, channel) in selected.enumerated() {
            let column = tile % columns, row = tile / columns
            let originX = gutter + column * (tileWidth + gutter)
            let originY = gutter + row * (tileHeight + gutter)
            let fit = min(Double(tileWidth) / Double(tensor.width), Double(tileHeight) / Double(tensor.height))
            let drawWidth = max(1, Int((Double(tensor.width) * fit).rounded()))
            let drawHeight = max(1, Int((Double(tensor.height) * fit).rounded()))
            let drawX = originX + (tileWidth - drawWidth) / 2
            let drawY = originY + (tileHeight - drawHeight) / 2
            let scale = robustScale(values: tensor.values, channel: channel, channels: tensor.channels)
            for y in 0..<drawHeight {
                let fy = (Double(y) + 0.5) * Double(tensor.height) / Double(drawHeight) - 0.5
                for x in 0..<drawWidth {
                    let fx = (Double(x) + 0.5) * Double(tensor.width) / Double(drawWidth) - 0.5
                    let value = bilinear(tensor, channel: channel, x: fx, y: fy)
                    let normalized = pow(min(1, max(0, value / scale)), 0.72)
                    let color = heatColor(normalized)
                    let offset = ((drawY + y) * width + drawX + x) * 4
                    rgba[offset] = byte(color.0)
                    rgba[offset + 1] = byte(color.1)
                    rgba[offset + 2] = byte(color.2)
                }
            }
        }
        return CNNVisualizationRender(
            image: image(width: width, height: height, rgba: rgba),
            detail: "\(stageLabel(tensor)) • stride ×\(tensor.effectiveStride) • field \(tensor.receptiveField)×\(tensor.receptiveField) • top \(selected.count) maps"
        )
    }

    private static func stageLabel(_ tensor: CNNFeatureTensor) -> String {
        tensor.convolutionLayer >= 0 ? "Conv \(tensor.convolutionLayer + 1)" : "Vision input"
    }

    private static func boundedSize(width: Int, height: Int) -> (width: Int, height: Int) {
        guard width > 0, height > 0 else { return (0, 0) }
        let scale = min(1, min(Double(maximumImageWidth) / Double(width), Double(maximumImageHeight) / Double(height)))
        return (max(1, Int((Double(width) * scale).rounded())), max(1, Int((Double(height) * scale).rounded())))
    }

    private static func robustScale(values: [Float], channel: Int, channels: Int) -> Double {
        guard channels > 0, channel >= 0, channel < channels else { return 1 }
        var finite: [Float] = []
        finite.reserveCapacity(values.count / channels)
        var index = channel
        while index < values.count {
            let value = abs(values[index])
            if value.isFinite { finite.append(value) }
            index += channels
        }
        guard !finite.isEmpty else { return 1 }
        finite.sort()
        let percentile = finite[min(finite.count - 1, Int(Double(finite.count - 1) * 0.985))]
        return max(1e-7, Double(percentile))
    }

    private static func bilinear(_ tensor: CNNFeatureTensor, channel: Int, x: Double, y: Double) -> Double {
        let x0 = min(tensor.width - 1, max(0, Int(floor(x))))
        let y0 = min(tensor.height - 1, max(0, Int(floor(y))))
        let x1 = min(tensor.width - 1, x0 + 1), y1 = min(tensor.height - 1, y0 + 1)
        let tx = min(1, max(0, x - Double(x0))), ty = min(1, max(0, y - Double(y0)))
        func value(_ px: Int, _ py: Int) -> Double {
            let raw = tensor.values[(py * tensor.width + px) * tensor.channels + channel]
            return raw.isFinite ? Double(raw) : 0
        }
        let top = value(x0, y0) * (1 - tx) + value(x1, y0) * tx
        let bottom = value(x0, y1) * (1 - tx) + value(x1, y1) * tx
        return top * (1 - ty) + bottom * ty
    }

    private static func packedRGB(bytes: UnsafePointer<UInt8>, count: Int, spec: PreprocessingSpec, x: Int, y: Int) -> (Double, Double, Double) {
        let width = spec.width, height = spec.height
        let lumaCount = width * height
        let yi = min(max(0, count - 1), y * width + x)
        guard count >= lumaCount, yi >= 0 else { return (0, 0, 0) }
        let luma = Double(bytes[yi]) / 255
        guard spec.colorMode == .color else { return (luma, luma, luma) }
        let chromaWidth = spec.chroma == .yuv444 ? width : (width + 1) / 2
        let chromaHeight = spec.chroma == .yuv420 ? (height + 1) / 2 : height
        let chromaCount = chromaWidth * chromaHeight
        let cx = spec.chroma == .yuv444 ? x : x / 2
        let cy = spec.chroma == .yuv420 ? y / 2 : y
        let ci = cy * chromaWidth + cx
        guard lumaCount + chromaCount + ci < count else { return (luma, luma, luma) }
        let cb = Double(bytes[lumaCount + ci]) / 255 - 0.5
        let cr = Double(bytes[lumaCount + chromaCount + ci]) / 255 - 0.5
        return (
            min(1, max(0, luma + 1.5748 * cr)),
            min(1, max(0, luma - 0.1873 * cb - 0.4681 * cr)),
            min(1, max(0, luma + 1.8556 * cb))
        )
    }

    private static func heatColor(_ raw: Double) -> (Double, Double, Double) {
        let value = min(1, max(0, raw))
        if value < 0.5 {
            let t = value * 2
            return (0.12 * (1 - t), 0.08 + 0.72 * t, 0.28 + 0.62 * t)
        }
        let t = (value - 0.5) * 2
        return (t, 0.8 + 0.12 * t, 0.9 * (1 - t) + 0.16 * t)
    }

    private static func byte(_ value: Double) -> UInt8 {
        UInt8(clamping: Int((min(1, max(0, value)) * 255).rounded()))
    }

    private static func image(width: Int, height: Int, rgba: [UInt8]) -> NSImage? {
        guard let bitmap = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: width,
            pixelsHigh: height,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: width * 4,
            bitsPerPixel: 32
        ), let destination = bitmap.bitmapData else { return nil }
        rgba.withUnsafeBytes { source in
            if let base = source.baseAddress { memcpy(destination, base, rgba.count) }
        }
        let image = NSImage(size: NSSize(width: width, height: height))
        image.addRepresentation(bitmap)
        return image
    }
}
