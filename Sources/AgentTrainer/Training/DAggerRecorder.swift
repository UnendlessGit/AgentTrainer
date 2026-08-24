@preconcurrency import AVFoundation
@preconcurrency import CoreVideo
import Foundation

struct DAggerFrame: Sendable {
    var timestampNanos: UInt64
    var packedVision: Data
    var preprocessing: PreprocessingSpec
    var executedControls: [Float]
}

struct DAggerCaptureRequest: Sendable {
    var recordingID: UUID
    var directory: URL
    var profileID: UUID
    var versionID: UUID
    var profileName: String
    var destinationFolderID: UUID
    var excludedKeyCodes: Set<UInt16>
    var preprocessing: PreprocessingSpec
    var capture: CaptureSpec
    var captureRect: CGRect
    var perceptionFPS: Double
    var actionFPS: Double
    var correctionWeight: Double
}

struct DAggerCaptureResult: Sendable {
    var request: DAggerCaptureRequest
    var firstFrameHostNanos: UInt64
    var correctionStartHostNanos: UInt64
    var endHostNanos: UInt64
    var frameCount: Int
    var droppedFrameCount: Int
    var eventCount: Int
    var encodedWidth: Int
    var encodedHeight: Int

    var duration: Double {
        guard endHostNanos >= firstFrameHostNanos else { return 0 }
        return Double(endHostNanos - firstFrameHostNanos) / 1e9
    }

    var supervisionStart: Double {
        guard correctionStartHostNanos >= firstFrameHostNanos else { return 0 }
        return Double(correctionStartHostNanos - firstFrameHostNanos) / 1e9
    }
}

/// Writes the already-preprocessed runtime frames to a compact native recording
/// package. Video work is serialized away from inference and byte-bounded;
/// physical events stream directly to their fixed-width file.
final class DAggerRecorder: @unchecked Sendable {
    private static let maximumPendingVideoBytes = DAggerConfiguration.maximumBufferedContextBytes

    let request: DAggerCaptureRequest
    let correctionStartHostNanos: UInt64

    private let writer: HEVCWriter
    private let eventWriter: InputEventWriter
    private let videoQueue = DispatchQueue(label: "AgentTrainer.DAgger.Video", qos: .userInitiated)
    private let lock = NSLock()
    private var accepting = true
    private var eventAccumulator: ActionAccumulator
    private var lastEventHostNanos: UInt64
    private var pendingVideoBytes = 0
    private var droppedQueuedFrames = 0
    private let firstFrameHostNanos: UInt64
    private var lastFrameHostNanos: UInt64
    private let encodedWidth: Int
    private let encodedHeight: Int

    init(
        request: DAggerCaptureRequest,
        correctionStartHostNanos: UInt64,
        preRoll: [DAggerFrame]
    ) throws {
        guard let first = preRoll.first,
              preRoll.allSatisfy({
                  $0.timestampNanos >= first.timestampNanos
                      && $0.packedVision.count == $0.preprocessing.sampleByteCount
                      && $0.executedControls.count >= ActionLayout.count
              }) else {
            throw AgentTrainerError.model("DAgger needs at least one complete live perception before correction recording can begin.")
        }
        let preRollBytes = preRoll.reduce(into: (value: 0, overflow: false)) { result, frame in
            let addition = result.value.addingReportingOverflow(frame.packedVision.count)
            result.value = addition.partialValue
            result.overflow = result.overflow || addition.overflow
        }
        guard !preRollBytes.overflow,
              preRollBytes.value <= Self.maximumPendingVideoBytes else {
            throw AgentTrainerError.invalidConfiguration("DAgger's buffered visual context exceeds the 512 MB runtime limit.")
        }
        self.request = request
        self.correctionStartHostNanos = max(correctionStartHostNanos, first.timestampNanos)
        encodedWidth = max(2, request.preprocessing.width - request.preprocessing.width % 2)
        encodedHeight = max(2, request.preprocessing.height - request.preprocessing.height % 2)
        firstFrameHostNanos = first.timestampNanos
        lastFrameHostNanos = preRoll.last?.timestampNanos ?? first.timestampNanos
        lastEventHostNanos = first.timestampNanos
        let provisional = RecordingManifest(
            id: UUID(), name: "DAgger", createdAt: Date(),
            hostStartNanos: first.timestampNanos, duration: 1,
            capture: request.capture, globalRect: CodableRect(request.captureRect),
            pixelWidth: encodedWidth, pixelHeight: encodedHeight,
            deliveredFPS: request.perceptionFPS, eventCount: 0
        )
        eventAccumulator = ActionAccumulator(manifest: provisional)
        eventWriter = try InputEventWriter(
            url: request.directory.appendingPathComponent("events.atrevents")
        )
        writer = try HEVCWriter(
            url: request.directory.appendingPathComponent("capture.mov"),
            width: encodedWidth,
            height: encodedHeight,
            fps: request.perceptionFPS
        )

        var synthesizer = DAggerActionEventSynthesizer(rect: request.captureRect)
        for frame in preRoll {
            appendEvents(synthesizer.events(for: frame.executedControls, at: frame.timestampNanos))
            enqueue(frame)
        }
        appendEvents(synthesizer.releaseAll(at: self.correctionStartHostNanos))
    }

    func append(_ frame: DAggerFrame) {
        enqueue(frame, requiresAccepting: true)
    }

    func append(_ sample: InputSample) {
        lock.withLock {
            guard accepting, sample.timestampNanos >= correctionStartHostNanos else { return }
            appendEventLocked(sample)
        }
    }

    func finish(endHostNanos requestedEnd: UInt64) async throws -> DAggerCaptureResult {
        let accepted = lock.withLock { () -> Bool in
            guard accepting else { return false }
            accepting = false
            for sample in balancedReleasesLocked(at: requestedEnd) {
                appendEventLocked(sample)
            }
            return true
        }
        guard accepted else {
            throw AgentTrainerError.storage("This DAgger correction was already finalized.")
        }
        await withCheckedContinuation { continuation in
            videoQueue.async { continuation.resume() }
        }
        let finalTimes = lock.withLock { (lastFrameHostNanos, lastEventHostNanos) }
        let endHostNanos = max(
            requestedEnd,
            max(correctionStartHostNanos, max(finalTimes.0, finalTimes.1))
        )
        let encoded = try await writer.finish(endPTS: CMTime(
            value: Int64(clamping: endHostNanos),
            timescale: 1_000_000_000
        ))
        guard encoded.frames > 0 else {
            throw AgentTrainerError.capture("No DAgger video frames could be encoded.")
        }

        let eventCount = try eventWriter.finish()
        let queuedDrops = lock.withLock { droppedQueuedFrames }
        return DAggerCaptureResult(
            request: request,
            firstFrameHostNanos: firstFrameHostNanos,
            correctionStartHostNanos: correctionStartHostNanos,
            endHostNanos: endHostNanos,
            frameCount: encoded.frames,
            droppedFrameCount: writer.droppedFrameCount + queuedDrops,
            eventCount: eventCount,
            encodedWidth: encodedWidth,
            encodedHeight: encodedHeight
        )
    }

    private func enqueue(_ frame: DAggerFrame, requiresAccepting: Bool = false) {
        let width = encodedWidth
        let height = encodedHeight
        lock.withLock {
            guard !requiresAccepting || accepting else { return }
            let frameBytes = frame.packedVision.count
            guard pendingVideoBytes <= Self.maximumPendingVideoBytes - frameBytes else {
                droppedQueuedFrames += 1
                return
            }
            pendingVideoBytes += frameBytes
            lastFrameHostNanos = max(lastFrameHostNanos, frame.timestampNanos)
            // Queue the frame while holding the same lock finish uses to close
            // admission. Its drain marker therefore cannot overtake an append
            // that already observed accepting=true.
            videoQueue.async { [weak self, writer] in
                autoreleasepool {
                    guard let pixelBuffer = DAggerVideoSurface.make(
                        packed: frame.packedVision,
                        spec: frame.preprocessing,
                        width: width,
                        height: height
                    ) else {
                        self?.completeVideoFrame(bytes: frameBytes, conversionFailed: true)
                        return
                    }
                    writer.append(
                        pixelBuffer,
                        presentationTimeStamp: CMTime(
                            value: Int64(clamping: frame.timestampNanos),
                            timescale: 1_000_000_000
                        )
                    )
                    self?.completeVideoFrame(bytes: frameBytes, conversionFailed: false)
                }
            }
        }
    }

    private func completeVideoFrame(bytes: Int, conversionFailed: Bool) {
        lock.withLock {
            pendingVideoBytes = max(0, pendingVideoBytes - bytes)
            if conversionFailed { droppedQueuedFrames += 1 }
        }
    }

    private func appendEvents(_ samples: [InputSample]) {
        lock.withLock {
            for sample in samples { appendEventLocked(sample) }
        }
    }

    private func appendEventLocked(_ sample: InputSample) {
        eventWriter.append(sample)
        eventAccumulator.consume(sample)
        lastEventHostNanos = max(lastEventHostNanos, sample.timestampNanos)
    }

    private func balancedReleasesLocked(at requestedTimestamp: UInt64) -> [InputSample] {
        var result: [InputSample] = []
        var timestamp = max(requestedTimestamp, lastEventHostNanos &+ 1)
        for key in eventAccumulator.keys.sorted() {
            result.append(InputSample(
                timestampNanos: timestamp, kind: .key, keyCode: key,
                modifiers: eventAccumulator.flags, isDown: false
            ))
            timestamp &+= 1
        }
        for button in eventAccumulator.buttons.sorted() {
            result.append(InputSample(
                timestampNanos: timestamp, kind: .mouseButton,
                x: eventAccumulator.x, y: eventAccumulator.y, button: button,
                modifiers: eventAccumulator.flags, isDown: false
            ))
            timestamp &+= 1
        }
        if eventAccumulator.flags != 0 {
            result.append(InputSample(timestampNanos: timestamp, kind: .flags, modifiers: 0))
        }
        return result
    }
}

private struct DAggerActionEventSynthesizer {
    var rect: CGRect
    var previous = [Float](repeating: 0, count: ActionLayout.count)

    mutating func events(for controls: [Float], at timestamp: UInt64) -> [InputSample] {
        guard controls.count >= ActionLayout.count else { return [] }
        let flags = modifierFlags(controls)
        let previousFlags = modifierFlags(previous)
        let x = rect.minX + CGFloat(controls[0]) * max(1, rect.width)
        let y = rect.minY + CGFloat(controls[1]) * max(1, rect.height)
        var result: [InputSample] = [InputSample(
            timestampNanos: timestamp,
            kind: .mouseMove,
            x: x,
            y: y,
            deltaX: Double(controls[2] * GameCameraContract.deltaScale),
            deltaY: Double(controls[3] * GameCameraContract.deltaScale),
            modifiers: flags
        )]
        if flags != previousFlags {
            result.append(InputSample(timestampNanos: timestamp, kind: .flags, modifiers: flags))
        }
        for button in 0..<8 {
            let index = ActionLayout.buttons.lowerBound + button
            let isDown = controls[index] >= 0.5
            let wasDown = previous[index] >= 0.5
            if isDown != wasDown {
                result.append(InputSample(
                    timestampNanos: timestamp, kind: .mouseButton,
                    x: x, y: y, button: UInt8(button), modifiers: flags, isDown: isDown
                ))
            }
        }
        if controls[12] != 0 || controls[13] != 0 {
            result.append(InputSample(
                timestampNanos: timestamp, kind: .scroll, x: x, y: y,
                scrollX: Double(controls[12] * 20),
                scrollY: Double(controls[13] * 20), modifiers: flags
            ))
        }
        for key in 0..<128 where !ActionLayout.commandOptionControlKeyCodeSet.contains(UInt16(key)) {
            let index = ActionLayout.keyboard.lowerBound + key
            let isDown = controls[index] >= 0.5
            let wasDown = previous[index] >= 0.5
            if isDown != wasDown {
                result.append(InputSample(
                    timestampNanos: timestamp, kind: .key, keyCode: UInt16(key),
                    modifiers: flags, isDown: isDown
                ))
            }
        }
        previous = Array(controls.prefix(ActionLayout.count))
        return result
    }

    mutating func releaseAll(at timestamp: UInt64) -> [InputSample] {
        var zero = [Float](repeating: 0, count: ActionLayout.count)
        zero[0] = previous[0]
        zero[1] = previous[1]
        return events(for: zero, at: timestamp).filter { $0.kind != .mouseMove }
    }

    private func modifierFlags(_ controls: [Float]) -> UInt64 {
        let masks: [CGEventFlags] = [.maskShift, .maskControl, .maskAlternate, .maskCommand]
        return masks.indices.reduce(UInt64(0)) { value, index in
            controls[ActionLayout.modifiers.lowerBound + index] >= 0.5
                ? value | masks[index].rawValue : value
        }
    }
}

private enum DAggerVideoSurface {
    static func make(
        packed: Data,
        spec: PreprocessingSpec,
        width: Int,
        height: Int
    ) -> CVPixelBuffer? {
        guard packed.count >= spec.sampleByteCount else { return nil }
        let attributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:],
            kCVPixelBufferMetalCompatibilityKey: true
        ]
        var optionalBuffer: CVPixelBuffer?
        guard CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            attributes as CFDictionary,
            &optionalBuffer
        ) == kCVReturnSuccess, let buffer = optionalBuffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard CVPixelBufferGetPlaneCount(buffer) == 2,
              let yBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 0),
              let uvBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) else { return nil }
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        let uvStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        packed.withUnsafeBytes { raw in
            guard let source = raw.bindMemory(to: UInt8.self).baseAddress else { return }
            for outputY in 0..<height {
                let sourceY = min(spec.height - 1, outputY * spec.height / height)
                let destination = yBase.advanced(by: outputY * yStride).assumingMemoryBound(to: UInt8.self)
                for outputX in 0..<width {
                    let sourceX = min(spec.width - 1, outputX * spec.width / width)
                    let full = Int(source[sourceY * spec.width + sourceX])
                    destination[outputX] = UInt8(clamping: 16 + (full * 219 + 127) / 255)
                }
            }
            let lumaCount = spec.width * spec.height
            let chromaWidth = spec.chroma == .yuv444 ? spec.width : (spec.width + 1) / 2
            let chromaHeight = spec.chroma == .yuv420 ? (spec.height + 1) / 2 : spec.height
            let chromaCount = chromaWidth * chromaHeight
            func chroma(_ plane: Int, x: Int, y: Int) -> UInt8 {
                guard spec.colorMode == .color else { return 128 }
                let cx = spec.chroma == .yuv444 ? x : x / 2
                let cy = spec.chroma == .yuv420 ? y / 2 : y
                return source[lumaCount + plane * chromaCount + min(chromaHeight - 1, cy) * chromaWidth + min(chromaWidth - 1, cx)]
            }
            for outputY in 0..<(height / 2) {
                let destination = uvBase.advanced(by: outputY * uvStride).assumingMemoryBound(to: UInt8.self)
                for outputX in 0..<(width / 2) {
                    var cb = 0, cr = 0
                    for dy in 0..<2 {
                        for dx in 0..<2 {
                            let x = min(spec.width - 1, (outputX * 2 + dx) * spec.width / width)
                            let y = min(spec.height - 1, (outputY * 2 + dy) * spec.height / height)
                            cb += Int(chroma(0, x: x, y: y))
                            cr += Int(chroma(1, x: x, y: y))
                        }
                    }
                    destination[outputX * 2] = UInt8(clamping: 16 + (((cb + 2) / 4) * 224 + 127) / 255)
                    destination[outputX * 2 + 1] = UInt8(clamping: 16 + (((cr + 2) / 4) * 224 + 127) / 255)
                }
            }
        }
        return buffer
    }
}
