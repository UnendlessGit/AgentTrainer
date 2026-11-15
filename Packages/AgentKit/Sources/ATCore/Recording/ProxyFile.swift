import Compression
import Foundation

/// Lossless training frames: chunks of `chunkFrames` RGB8 frames, delta-coded within the chunk
/// (frame k stores `frame_k - frame_{k-1}` mod 256) and LZFSE-compressed. Random access decodes one chunk.
///
/// Layout: 32-byte header (`ATPX`, version, width, height, channels, chunkFrames), chunks,
/// index (per chunk: offset u64, compressed size u32, frame count u32), footer
/// (index offset u64, chunk count u32, frame count u32, `ATPI`).
public enum ProxyFile {
    static let headerSize = 32
    static let magic: UInt32 = 0x5850_5441 // "ATPX"
    static let indexMagic: UInt32 = 0x4950_5441 // "ATPI"

    static func deltaEncode(_ frames: UnsafeMutableRawBufferPointer, frameBytes: Int, count: Int) {
        let p = frames.bindMemory(to: UInt8.self)
        for f in stride(from: count - 1, to: 0, by: -1) {
            let cur = f * frameBytes, prev = (f - 1) * frameBytes
            for i in 0 ..< frameBytes { p[cur + i] = p[cur + i] &- p[prev + i] }
        }
    }

    static func deltaDecode(_ frames: UnsafeMutableRawBufferPointer, frameBytes: Int, count: Int) {
        let p = frames.bindMemory(to: UInt8.self)
        for f in 1 ..< max(count, 1) {
            let cur = f * frameBytes, prev = (f - 1) * frameBytes
            for i in 0 ..< frameBytes { p[cur + i] = p[cur + i] &+ p[prev + i] }
        }
    }
}

public enum ProxyFileError: Error, LocalizedError {
    case corrupt(String), io(String)
    public var errorDescription: String? {
        switch self {
        case .corrupt(let m): "Training frames are damaged: \(m)"
        case .io(let m): "Could not write training frames: \(m)"
        }
    }
}

/// Appends frames from any thread-confined producer; compression happens on a background queue.
public final class ProxyWriter: @unchecked Sendable {
    public let width: Int, height: Int, channels = 3, chunkFrames: Int
    private let handle: FileHandle
    private let queue = DispatchQueue(label: "agenttrainer.proxy-writer", qos: .utility)
    private var pending = Data()
    private var pendingCount = 0
    private var index: [(offset: UInt64, size: UInt32, count: UInt32)] = []
    private var offset: UInt64
    private var frameCount = 0
    private var error: Error?
    public var frameBytes: Int { width * height * channels }

    public init(url: URL, width: Int, height: Int, chunkFrames: Int = 30) throws {
        self.width = width; self.height = height; self.chunkFrames = chunkFrames
        FileManager.default.createFile(atPath: url.path, contents: nil)
        guard let h = try? FileHandle(forWritingTo: url) else { throw ProxyFileError.io(url.path) }
        handle = h
        var header = Data()
        header.append(le: ProxyFile.magic); header.append(le: UInt32(1)); header.append(le: UInt32(width)); header.append(le: UInt32(height))
        header.append(le: UInt32(3)); header.append(le: UInt32(chunkFrames)); header.append(Data(count: 8))
        handle.write(header)
        offset = UInt64(header.count)
        pending.reserveCapacity(frameBytes * chunkFrames)
    }

    /// Adds one RGB8 frame (`width*height*3` bytes, row-major, top row first).
    public func append(_ rgb: UnsafeRawBufferPointer) {
        precondition(rgb.count == frameBytes)
        pending.append(contentsOf: rgb)
        pendingCount += 1
        frameCount += 1
        if pendingCount == chunkFrames { flushChunk() }
    }

    public var framesWritten: Int { frameCount }

    private func flushChunk() {
        guard pendingCount > 0 else { return }
        var chunk = pending
        let count = pendingCount, fb = frameBytes
        pending = Data(); pending.reserveCapacity(fb * chunkFrames)
        pendingCount = 0
        queue.async { [self] in
            chunk.withUnsafeMutableBytes { ProxyFile.deltaEncode($0, frameBytes: fb, count: count) }
            let compressed = Self.compress(chunk)
            handle.write(compressed)
            index.append((offset, UInt32(compressed.count), UInt32(count)))
            offset += UInt64(compressed.count)
        }
    }

    static func compress(_ input: Data) -> Data {
        let cap = input.count + 4096
        var out = Data(count: cap)
        let n = out.withUnsafeMutableBytes { dst in
            input.withUnsafeBytes { src in
                compression_encode_buffer(dst.bindMemory(to: UInt8.self).baseAddress!, cap,
                                          src.bindMemory(to: UInt8.self).baseAddress!, input.count, nil, COMPRESSION_LZFSE)
            }
        }
        out.count = n
        return out
    }

    /// Flushes the last chunk and writes the index. Safe to call once.
    public func finish() throws {
        flushChunk()
        try queue.sync {
            var tail = Data()
            for e in index { tail.append(le: e.offset); tail.append(le: e.size); tail.append(le: e.count) }
            tail.append(le: offset); tail.append(le: UInt32(index.count)); tail.append(le: UInt32(frameCount)); tail.append(le: ProxyFile.indexMagic)
            handle.write(tail)
            try handle.close()
        }
    }
}

/// Random-access reader with a small chunk cache. Not thread-safe; use one per loader thread.
public final class ProxyReader {
    public let width: Int, height: Int, channels: Int, chunkFrames: Int, frameCount: Int
    private let data: Data
    private var index: [(offset: Int, size: Int, count: Int)] = []
    private var cache: [Int: Data] = [:]
    private var cacheOrder: [Int] = []
    public var frameBytes: Int { width * height * channels }

    public init(url: URL) throws {
        data = try Data(contentsOf: url, options: .alwaysMapped)
        guard data.count >= ProxyFile.headerSize + 20, data.le(0) as UInt32 == ProxyFile.magic else { throw ProxyFileError.corrupt("bad header") }
        width = Int(data.le(8) as UInt32); height = Int(data.le(12) as UInt32)
        channels = Int(data.le(16) as UInt32); chunkFrames = Int(data.le(20) as UInt32)
        let f = data.count - 20
        guard data.le(f + 16) as UInt32 == ProxyFile.indexMagic else { throw ProxyFileError.corrupt("missing index (recording may not have finished)") }
        let indexOffset = Int(data.le(f) as UInt64), chunks = Int(data.le(f + 8) as UInt32)
        frameCount = Int(data.le(f + 12) as UInt32)
        for c in 0 ..< chunks {
            let o = indexOffset + c * 16
            index.append((Int(data.le(o) as UInt64), Int(data.le(o + 8) as UInt32), Int(data.le(o + 12) as UInt32)))
        }
    }

    private func chunk(_ c: Int) throws -> Data {
        if let d = cache[c] { return d }
        let e = index[c]
        var out = Data(count: e.count * frameBytes)
        let n = out.withUnsafeMutableBytes { dst in
            data.withUnsafeBytes { src in
                compression_decode_buffer(dst.bindMemory(to: UInt8.self).baseAddress!, dst.count,
                                          src.bindMemory(to: UInt8.self).baseAddress! + e.offset, e.size, nil, COMPRESSION_LZFSE)
            }
        }
        guard n == out.count else { throw ProxyFileError.corrupt("chunk \(c)") }
        out.withUnsafeMutableBytes { ProxyFile.deltaDecode($0, frameBytes: frameBytes, count: e.count) }
        cache[c] = out
        cacheOrder.append(c)
        if cacheOrder.count > 4 { cache[cacheOrder.removeFirst()] = nil }
        return out
    }

    /// Copies frame `i` into `dst` (frameBytes bytes).
    public func read(frame i: Int, into dst: UnsafeMutableRawPointer) throws {
        guard i >= 0 && i < frameCount else { throw ProxyFileError.corrupt("frame \(i) out of range") }
        let c = try chunk(i / chunkFrames)
        let o = (i % chunkFrames) * frameBytes
        c.withUnsafeBytes { dst.copyMemory(from: $0.baseAddress! + o, byteCount: frameBytes) }
    }

    public func frame(_ i: Int) throws -> Data {
        var d = Data(count: frameBytes)
        try d.withUnsafeMutableBytes { try read(frame: i, into: $0.baseAddress!) }
        return d
    }
}
