import ATCore
import AVFoundation
import Foundation

/// Builds a training proxy at another resolution from a recording's video, using the same GPU
/// resize as live capture. Frame order matches `frames.bin` (one proxy frame per decoded video frame).
public enum ProxyBuilder {
    public static func build(recording: URL, size: Int, to dest: URL, progress: ((Double) -> Void)? = nil) throws {
        let asset = AVURLAsset(url: recording.appendingPathComponent(RecordingPackage.video))
        guard let track = asset.tracks(withMediaType: .video).first else { throw ProxyFileError.corrupt("no video track") }
        let reader = try AVAssetReader(asset: asset)
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                                                                          kCVPixelBufferMetalCompatibilityKey as String: true])
        out.alwaysCopiesSampleData = false
        reader.add(out)
        guard reader.startReading(), let resizer = FrameResizer() else { throw ProxyFileError.io("can't decode video") }
        let tmp = dest.appendingPathExtension("partial")
        let writer = try ProxyWriter(url: tmp, width: size, height: size)
        let total = max(1, (try? RecordingIO.readMeta(recording).frameCount) ?? 1)
        var n = 0
        while let sb = out.copyNextSampleBuffer() {
            guard let pb = sb.imageBuffer else { continue }
            resizer.resize(pb, size: size) { writer.append($0) }
            n += 1
            if n % 60 == 0 { progress?(Double(n) / Double(total)) }
        }
        try writer.finish()
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.moveItem(at: tmp, to: dest)
    }
}
