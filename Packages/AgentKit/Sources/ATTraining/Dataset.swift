import ATCapture
import ATCore
import ATLanguage
import Foundation

/// One recording prepared for a specific model: per-step frames, action classes, loss mask, text.
final class PreparedRecording {
    let url: URL
    let meta: RecordingMeta
    let proxyURL: URL
    let frameIndex: [Int32]
    let classes: [[Int32]]
    let lossMask: [Bool]
    let text: [Int32]
    let cursor: [(Double, Double)]
    var stepCount: Int { frameIndex.count }

    init(url: URL, meta: RecordingMeta, proxyURL: URL, frameIndex: [Int32], classes: [[Int32]], lossMask: [Bool], text: [Int32], cursor: [(Double, Double)]) {
        self.url = url; self.meta = meta; self.proxyURL = proxyURL; self.frameIndex = frameIndex; self.classes = classes
        self.lossMask = lossMask; self.text = text; self.cursor = cursor
    }
}

struct ClipRef: Hashable { let recording: Int; let start: Int }

/// Recordings resampled to the model's control rate and cut into clips of `length` timesteps.
final class TrainingDataset {
    let recordings: [PreparedRecording]
    let length: Int
    let frameSize: Int
    let slots: Int
    let absoluteCursor: Bool

    init(recordings: [PreparedRecording], length: Int, frameSize: Int, slots: Int, absoluteCursor: Bool) {
        self.recordings = recordings; self.length = length; self.frameSize = frameSize; self.slots = slots; self.absoluteCursor = absoluteCursor
    }

    var totalSteps: Int { recordings.reduce(0) { $0 + $1.stepCount } }

    /// Clips with a random phase per recording (like P2P), skipping clips with almost nothing to learn.
    func clips(seed: UInt64) -> [ClipRef] {
        var rng = SplitMix(seed)
        var out: [ClipRef] = []
        for (ri, r) in recordings.enumerated() {
            guard r.stepCount > length / 4 else { continue }
            let phase = r.stepCount > length ? Int(rng.next() % UInt64(length)) : 0
            var s = phase - (phase > 0 ? length : 0)
            while s < r.stepCount - length / 4 {
                let start = max(0, s)
                let end = min(r.stepCount, start + length)
                let useful = r.lossMask[start ..< end].lazy.filter { $0 }.count
                if useful >= length / 10 { out.append(ClipRef(recording: ri, start: start)) }
                s += length
            }
        }
        out.shuffle(using: &rng)
        return out
    }

    /// Fixed, unshuffled clips for validation.
    func validationClips(limit: Int) -> [ClipRef] {
        var out: [ClipRef] = []
        for (ri, r) in recordings.enumerated() {
            var s = 0
            while s + length / 2 <= r.stepCount { out.append(ClipRef(recording: ri, start: s)); s += length }
        }
        if out.count <= limit { return out }
        let stride = Double(out.count) / Double(limit)
        return (0 ..< limit).map { out[Int(Double($0) * stride)] }
    }
}

/// Builds `PreparedRecording`s for a model.
enum DatasetBuilder {
    static func prepare(urls: [URL], model: ModelDefinition, cacheRoot: URL, texts: TextTable, log: (String) -> Void) throws -> [PreparedRecording] {
        let rate = model.architecture.stepRateHz
        let size = model.architecture.frameSize
        var out: [PreparedRecording] = []
        for url in urls {
            guard let meta = try? RecordingIO.readMeta(url) else { log("Skipped \(url.lastPathComponent): unreadable"); continue }
            let frames = try RecordingIO.readFrames(url), events = try RecordingIO.readEvents(url)
            let ann = RecordingIO.readAnnotations(url)
            let steps = TimelineBuilder.build(frames: frames, events: events, meta: meta, annotations: ann, rateHz: rate)
            guard !steps.isEmpty else { log("Skipped \(meta.name): too short"); continue }
            let proxy = try proxyURL(for: url, meta: meta, size: size, cacheRoot: cacheRoot, log: log)
            var cursor: [(Double, Double)] = []
            var prev = (steps[0].action.absX, steps[0].action.absY)
            for s in steps { cursor.append(prev); prev = (s.action.absX, s.action.absY) }
            out.append(PreparedRecording(
                url: url, meta: meta, proxyURL: proxy,
                frameIndex: steps.map { Int32($0.frameIndex) },
                classes: steps.map { model.actions.encode($0.action) },
                lossMask: steps.map { !$0.modelControlled && !$0.excluded },
                text: steps.map { model.usesInstructions ? texts.id(for: $0.instruction) : -1 },
                cursor: cursor))
        }
        return out
    }

    static func proxyURL(for url: URL, meta: RecordingMeta, size: Int, cacheRoot: URL, log: (String) -> Void) throws -> URL {
        let inPackage = url.appendingPathComponent(RecordingPackage.proxy(size: size))
        if FileManager.default.fileExists(atPath: inPackage.path) { return inPackage }
        let cached = cacheRoot.appendingPathComponent("proxies").appendingPathComponent(meta.id.uuidString).appendingPathComponent("proxy-\(size).atproxy")
        if FileManager.default.fileExists(atPath: cached.path) { return cached }
        log("Preparing \(size)×\(size) frames for \(meta.name)…")
        try FileManager.default.createDirectory(at: cached.deletingLastPathComponent(), withIntermediateDirectories: true)
        try ProxyBuilder.build(recording: url, size: size, to: cached)
        return cached
    }

    /// Scale that maps recorded pointer deltas per step to P2P's mouse units (σx ≈ 96 per window).
    static func mouseUnitScale(urls: [URL], model: ModelDefinition) -> Double {
        var xs: [Double] = []
        for url in urls {
            guard let meta = try? RecordingIO.readMeta(url), let frames = try? RecordingIO.readFrames(url),
                  let events = try? RecordingIO.readEvents(url) else { continue }
            let steps = TimelineBuilder.build(frames: frames, events: events, meta: meta, annotations: RecordingAnnotations(), rateHz: model.architecture.stepRateHz)
            xs += steps.map(\.action.dx).filter { $0 != 0 }
        }
        guard xs.count > 50 else { return 1 }
        let m = xs.reduce(0, +) / Double(xs.count)
        let std = (xs.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(xs.count)).squareRoot()
        return std > 0.5 ? min(50, max(0.05, model.actions.mouseXStd / std)) : 1
    }

    /// Deterministic split by recording.
    static func split(_ urls: [URL], fraction: Double, seed: Int) -> (train: [URL], validation: [URL]) {
        guard fraction > 0, urls.count >= 2 else { return (urls, []) }
        let sorted = urls.sorted { $0.lastPathComponent < $1.lastPathComponent }
        var rng = SplitMix(UInt64(seed) &+ 0x9E37)
        let shuffled = sorted.shuffled(using: &rng)
        let n = max(1, Int((Double(urls.count) * fraction).rounded()))
        return (Array(shuffled.dropFirst(n)), Array(shuffled.prefix(n)))
    }
}

/// A clip assembled on the CPU, ready to upload.
struct ClipData {
    var frames: Data       // T × S × S × 3 uint8
    var classes: [Int32]   // T × slots
    var mask: [Bool]       // T
    var text: [Int32]      // T (-1 = none)
}

/// Background loader: decodes proxy frames and assembles clips ahead of the trainer.
final class ClipLoader: @unchecked Sendable {
    private let dataset: TrainingDataset
    private let queue = DispatchQueue(label: "agenttrainer.loader", qos: .userInitiated, attributes: .concurrent)
    private let lock = NSCondition()
    private var ready: [ClipData] = []
    private var pending: [ClipRef] = []
    private var inFlight = 0
    private var readers: [Int: [ProxyReader]] = [:]
    private let readerLock = NSLock()
    private let capacity: Int
    private(set) var epoch = 0
    private var seed: UInt64
    private var stopped = false
    var clipsPerEpoch: Int

    init(dataset: TrainingDataset, seed: UInt64, capacity: Int = 4) {
        self.dataset = dataset
        self.seed = seed
        self.capacity = capacity
        clipsPerEpoch = max(1, dataset.clips(seed: seed).count)
    }

    func stop() { lock.lock(); stopped = true; lock.broadcast(); lock.unlock() }

    func next() throws -> ClipData {
        lock.lock()
        defer { lock.unlock() }
        refill()
        while ready.isEmpty && !stopped { lock.wait(); refill() }
        if stopped { throw CancellationError() }
        let c = ready.removeFirst()
        refill()
        return c
    }

    var epochProgress: Double { Double(epoch) + 1 - Double(pending.count + ready.count + inFlight) / Double(max(clipsPerEpoch, 1)) }

    private func refill() {
        while ready.count + inFlight < capacity {
            if pending.isEmpty {
                pending = dataset.clips(seed: seed &+ UInt64(epoch) &* 0x9E37_79B9)
                clipsPerEpoch = max(1, pending.count)
                epoch += 1
                if pending.isEmpty { return }
            }
            let ref = pending.removeFirst()
            inFlight += 1
            queue.async { [self] in
                let clip = try? load(ref)
                lock.lock()
                inFlight -= 1
                if let clip { ready.append(clip) }
                lock.broadcast()
                lock.unlock()
            }
        }
    }

    private func reader(_ r: Int) throws -> ProxyReader {
        readerLock.lock()
        if var list = readers[r], !list.isEmpty {
            let x = list.removeLast(); readers[r] = list; readerLock.unlock(); return x
        }
        readerLock.unlock()
        return try ProxyReader(url: dataset.recordings[r].proxyURL)
    }

    private func give(_ r: Int, _ x: ProxyReader) {
        readerLock.lock(); readers[r, default: []].append(x); readerLock.unlock()
    }

    func load(_ ref: ClipRef) throws -> ClipData {
        let rec = dataset.recordings[ref.recording]
        let T = dataset.length, S = dataset.frameSize, fb = S * S * 3
        let rd = try reader(ref.recording)
        defer { give(ref.recording, rd) }
        var frames = Data(count: T * fb)
        var classes = [Int32](), mask = [Bool](), text = [Int32]()
        classes.reserveCapacity(T * dataset.slots)
        try frames.withUnsafeMutableBytes { buf in
            for t in 0 ..< T {
                let i = min(ref.start + t, rec.stepCount - 1)
                let dst = buf.baseAddress! + t * fb
                try rd.read(frame: min(Int(rec.frameIndex[i]), rd.frameCount - 1), into: dst)
                if dataset.absoluteCursor {
                    FrameResizer.drawCursorMarker(UnsafeMutableRawBufferPointer(start: dst, count: fb), size: S,
                                                  at: CGPoint(x: rec.cursor[i].0, y: rec.cursor[i].1))
                }
                let valid = ref.start + t < rec.stepCount
                classes += rec.classes[i]
                mask.append(valid && rec.lossMask[i])
                text.append(rec.text[i])
            }
        }
        return ClipData(frames: frames, classes: classes, mask: mask, text: text)
    }
}

/// Small deterministic RNG.
struct SplitMix: RandomNumberGenerator {
    var state: UInt64
    init(_ seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
