import ATCore
import AVFoundation
import CoreVideo
import Foundation

/// Writes scripted-expert demonstrations as real `.atrec` recordings (same format, same GPU resize
/// and same event timing semantics as live capture), for end-to-end verification.
public enum ArenaRecorder {
    public static func makePixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferIOSurfacePropertiesKey: [:], kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary, &pb)
        return pb
    }

    /// Decides each step's action. Returns the action and whether a model (not the expert/person) chose it.
    public typealias Policy = (_ game: ArenaGame, _ frame: CVPixelBuffer, _ step: Int) -> (action: TimestepAction, byModel: Bool)

    @discardableResult
    public static func generate(mode: ArenaGame.Mode, seconds: Double, rateHz: Int = 30, pixelSize: Int = 480, proxySize: Int = 192,
                                seed: UInt64, destination: URL, name: String, policy: Policy? = nil) throws -> URL {
        let game = ArenaGame(mode: mode, seed: seed)
        let pkg = destination.appendingPathComponent(name).appendingPathExtension(RecordingPackage.fileExtension)
        try? FileManager.default.removeItem(at: pkg)
        try FileManager.default.createDirectory(at: pkg, withIntermediateDirectories: true)
        guard let resizer = FrameResizer(), let pb = makePixelBuffer(width: pixelSize, height: pixelSize) else {
            throw ProxyFileError.io("Metal unavailable")
        }
        let proxy = try ProxyWriter(url: pkg.appendingPathComponent(RecordingPackage.proxy(size: proxySize)), width: proxySize, height: proxySize)
        let video = try AVAssetWriter(outputURL: pkg.appendingPathComponent(RecordingPackage.video), fileType: .mov)
        let vin = AVAssetWriterInput(mediaType: .video, outputSettings: [AVVideoCodecKey: AVVideoCodecType.hevc, AVVideoWidthKey: pixelSize, AVVideoHeightKey: pixelSize])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: vin, sourcePixelBufferAttributes: nil)
        video.add(vin)
        video.startWriting()
        video.startSession(atSourceTime: .zero)

        let stepNs = Int64(1_000_000_000 / rateHz), n = Int(seconds * Double(rateHz))
        let src = CodableRect(x: 0, y: 0, width: Double(pixelSize), height: Double(pixelSize))
        var frames = Data(), events = Data()
        var held = Set<String>(), buttonDown = false
        var annotations = RecordingAnnotations()
        var currentInstruction: (text: String, start: Double)?
        var thumb: CGImage?
        var modelRanges: [TimeRange] = []
        var modelStart: Double?
        for k in 0 ..< n {
            let t = Int64(k) * stepNs
            game.render(into: pb)
            resizer.resize(pb, size: proxySize) { proxy.append($0) }
            while !vin.isReadyForMoreMediaData { usleep(1000) }
            adaptor.append(pb, withPresentationTime: CMTime(value: t, timescale: 1_000_000_000))
            frames.append(BinaryRecords.encode(FrameRecord(index: UInt32(k), flags: 1, timeNs: t, hostNs: UInt64(t))))
            if k == rateHz { thumb = image(pb) }

            let now = Double(t) / 1e9
            if let ins = game.instruction, ins != currentInstruction?.text {
                if let c = currentInstruction { annotations.segments.append(TextSegment(start: c.start, end: now, text: c.text, source: .auto)) }
                currentInstruction = (ins, now)
            }

            let (a, byModel) = policy?(game, pb, k) ?? (game.expertAction(), false)
            if byModel, modelStart == nil { modelStart = now }
            if !byModel, let s0 = modelStart { modelRanges.append(TimeRange(start: s0, end: now)); modelStart = nil }
            let flag: UInt8 = byModel ? 1 : 0
            // Input happens after the frame was shown, inside this step's window.
            for key in held.subtracting(a.keys) {
                events.append(BinaryRecords.encode(InputEventRecord(timeNs: t + 3_000_000, kind: .keyUp, flags: flag, code: KeyCatalog.info(key)!.keyCode)))
            }
            for key in a.keys.subtracting(held) {
                events.append(BinaryRecords.encode(InputEventRecord(timeNs: t + 4_000_000, kind: .keyDown, flags: flag, code: KeyCatalog.info(key)!.keyCode)))
            }
            held = a.keys
            let px = Float(a.absX * Double(pixelSize)), py = Float(a.absY * Double(pixelSize))
            if a.dx != 0 || a.dy != 0 || mode == .clickButton {
                events.append(BinaryRecords.encode(InputEventRecord(timeNs: t + 6_000_000, kind: .mouseMove, x: px, y: py, dx: Float(a.dx), dy: Float(a.dy))))
            }
            let down = a.pointer.contains(.left)
            if down && !buttonDown {
                events.append(BinaryRecords.encode(InputEventRecord(timeNs: t + 9_000_000, kind: .buttonDown, code: 0, x: px, y: py)))
            } else if !down && buttonDown {
                events.append(BinaryRecords.encode(InputEventRecord(timeNs: t + 2_000_000, kind: .buttonUp, code: 0, x: px, y: py)))
            }
            buttonDown = down
            game.step(a, dt: 1 / Double(rateHz))
        }
        if let c = currentInstruction { annotations.segments.append(TextSegment(start: c.start, end: seconds + 1, text: c.text, source: .auto)) }
        if let s0 = modelStart { modelRanges.append(TimeRange(start: s0, end: seconds + 1)) }
        annotations.modelControlled = modelRanges
        try proxy.finish()
        vin.markAsFinished()
        let sem = DispatchSemaphore(value: 0)
        video.finishWriting { sem.signal() }
        sem.wait()
        try frames.write(to: pkg.appendingPathComponent(RecordingPackage.frames))
        try events.write(to: pkg.appendingPathComponent(RecordingPackage.events))
        var meta = RecordingMeta(name: name, appVersion: "arena", target: CaptureTargetInfo(kind: .testArena, title: mode.title),
                                 captureWidth: pixelSize, captureHeight: pixelSize, sourceRect: src, captureFPS: rateHz,
                                 showsCursor: false, input: InputCaptureOptions(), task: mode.title, tags: ["arena", mode.rawValue])
        meta.duration = seconds
        meta.frameCount = n
        meta.eventCount = events.count / InputEventRecord.size
        meta.proxySizes = [proxySize]
        meta.eventClock = "synthetic"
        meta.isCorrectionRun = policy != nil
        try RecordingIO.writeMeta(meta, to: pkg)
        try RecordingIO.writeAnnotations(annotations, to: pkg)
        if let img = thumb, let dest = CGImageDestinationCreateWithURL(pkg.appendingPathComponent(RecordingPackage.thumbnail) as CFURL, "public.jpeg" as CFString, 1, nil) {
            CGImageDestinationAddImage(dest, img, nil)
            CGImageDestinationFinalize(dest)
        }
        _ = game.score
        return pkg
    }

    static func image(_ pb: CVPixelBuffer) -> CGImage? {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb),
                                  bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        return ctx.makeImage()
    }
}
