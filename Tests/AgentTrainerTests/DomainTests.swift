import XCTest
import AVFoundation
import CoreVideo
import MLX
import MLXNN
import MLXOptimizers
@preconcurrency import ScreenCaptureKit
@testable import AgentTrainer

private func XCTAssertThrowsErrorAsync<T>(
    _ operation: () async throws -> T,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        _ = try await operation()
        XCTFail("Expected operation to throw an error.", file: file, line: line)
    } catch {}
}

final class DomainTests: XCTestCase {
    private final class EventCollector: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: [(CGEventType, Int64, Int64, Int64)] = []
        private var storedWarps: [CGPoint] = []
        func append(_ event: CGEvent) { lock.lock(); stored.append((event.type, event.getIntegerValueField(.keyboardEventKeycode), event.getIntegerValueField(.mouseEventDeltaX), event.getIntegerValueField(.mouseEventDeltaY))); lock.unlock() }
        func warp(_ point: CGPoint) { lock.lock(); storedWarps.append(point); lock.unlock() }
        var events: [(CGEventType, Int64, Int64, Int64)] { lock.lock(); defer { lock.unlock() }; return stored }
        var warps: [CGPoint] { lock.lock(); defer { lock.unlock() }; return storedWarps }
    }
    private final class OneShotGate: @unchecked Sendable {
        private let lock = NSLock()
        private var used = false
        func claim() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            guard !used else { return false }
            used = true
            return true
        }
    }
    func testUpdateVersionsUseSemanticOrdering() throws {
        XCTAssertLessThan(try XCTUnwrap(AppSemanticVersion("v1.3.9")), try XCTUnwrap(AppSemanticVersion("1.4.0")))
        XCTAssertLessThan(try XCTUnwrap(AppSemanticVersion("1.9")), try XCTUnwrap(AppSemanticVersion("1.10")))
        XCTAssertEqual(try XCTUnwrap(AppSemanticVersion("1.3")), try XCTUnwrap(AppSemanticVersion("1.3.0")))
        XCTAssertLessThan(try XCTUnwrap(AppSemanticVersion("2.0.0-beta.2")), try XCTUnwrap(AppSemanticVersion("2.0.0")))
        XCTAssertNil(AppSemanticVersion("release-next"))
        XCTAssertNil(AppSemanticVersion("1.0-"))
        XCTAssertNil(AppSemanticVersion("1.0-alpha..1"))
        XCTAssertNil(AppSemanticVersion("1.0+"))
        XCTAssertNil(AppSemanticVersion("1.0+build+other"))
    }

    func testReleaseChecksumParserRequiresExactFilenameAndSHA256() {
        let sums = Data("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  AgentTrainer-1.4.dmg\n".utf8)
        XCTAssertEqual(GitHubReleaseUpdater.expectedChecksum(for: "AgentTrainer-1.4.dmg", in: sums), String(repeating: "a", count: 64))
        XCTAssertNil(GitHubReleaseUpdater.expectedChecksum(for: "AgentTrainer-1.4-Compact.dmg", in: sums))
        XCTAssertNil(GitHubReleaseUpdater.expectedChecksum(for: "AgentTrainer-1.4.dmg", in: Data("abc  AgentTrainer-1.4.dmg\n".utf8)))
    }

    func testUpdateMountPointParsingAndProgressBounds() throws {
        let plist: [String: Any] = [
            "system-entities": [
                ["dev-entry": "/dev/disk9"],
                ["mount-point": "/Volumes/AgentTrainer 1.6"]
            ]
        ]
        let data = try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0)
        XCTAssertEqual(try GitHubReleaseUpdater.mountPoint(fromHdiutilPlist: data).path, "/Volumes/AgentTrainer 1.6")
        XCTAssertThrowsError(try GitHubReleaseUpdater.mountPoint(fromHdiutilPlist: Data()))
        XCTAssertEqual(AppUpdateProgress(detail: "low", fraction: -1).fraction, 0)
        XCTAssertEqual(AppUpdateProgress(detail: "high", fraction: 2).fraction, 1)
        XCTAssertEqual(AppUpdateProgress(detail: "nan", fraction: .nan).fraction, 0)
    }

    func testUpdateProcessRunnerDrainsLargeOutputWithoutDeadlocking() async throws {
        let command = "/usr/bin/yes output | /usr/bin/head -c 200000; /usr/bin/yes error | /usr/bin/head -c 200000 >&2"
        let result = try await GitHubReleaseUpdater.runProcess("/bin/zsh", ["-c", command])
        XCTAssertEqual(result.stdout.count, 200_000)
        XCTAssertEqual(result.stderr.count, 200_000)
    }
    func testPackedObservationSizes() {
        var spec = PreprocessingSpec(width: 641, height: 361, colorMode: .color, bitDepth: 6, chroma: .yuv420)
        XCTAssertEqual(spec.sampleByteCount, 641 * 361 + 2 * 321 * 181)
        spec.chroma = .yuv422
        XCTAssertEqual(spec.sampleByteCount, 641 * 361 + 2 * 321 * 361)
        spec.chroma = .yuv444
        XCTAssertEqual(spec.sampleByteCount, 641 * 361 * 3)
        spec.colorMode = .grayscale
        XCTAssertEqual(spec.sampleByteCount, 641 * 361)
    }

    func testPreprocessingAndTrainingMemoryBoundsRejectPathologicalSizes() {
        XCTAssertThrowsError(try PreprocessingSpec(width: 8_193, height: 1).validated())
        XCTAssertThrowsError(try PreprocessingSpec(width: 8_192, height: 8_192).validated())

        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 64, height: 36, colorMode: .grayscale)
        profile.training.batchSize = 2
        profile.training.architecture = .small
        let parameterBytes = ModelSizing.parameterCount(profile) * 24
        XCTAssertGreaterThan(ModelSizing.estimatedTrainingWorkingSet(profile), parameterBytes)
        XCTAssertLessThan(ModelSizing.estimatedTrainingWorkingSet(profile), Int64.max)
    }

    func testMLXMemoryPolicyBoundsInferenceOnlySessions() {
        let gibibyte = 1 << 30
        let eightGiB = MLXMemoryLifecycle.limits(forPhysicalMemory: 8 * gibibyte)
        XCTAssertEqual(eightGiB.memory, 6 * gibibyte)
        XCTAssertEqual(eightGiB.cache, 512 << 20)

        let thirtySixGiB = MLXMemoryLifecycle.limits(forPhysicalMemory: 36 * gibibyte)
        XCTAssertEqual(thirtySixGiB.memory, 36 * gibibyte - Int(Double(36 * gibibyte) * 0.15))
        XCTAssertEqual(thirtySixGiB.cache, 2 * gibibyte)
        XCTAssertLessThanOrEqual(thirtySixGiB.cache, thirtySixGiB.memory)
    }

    func testScreenCaptureFrameStatusesDistinguishStaticFromUnsafeFrames() {
        XCTAssertEqual(CaptureFrameDisposition.classify(nil), .deliver)
        XCTAssertEqual(CaptureFrameDisposition.classify(.complete), .deliver)
        XCTAssertEqual(CaptureFrameDisposition.classify(.started), .deliver)
        XCTAssertEqual(CaptureFrameDisposition.classify(.idle), .reuseLastFrame)
        XCTAssertEqual(CaptureFrameDisposition.classify(.blank), .drop)
        XCTAssertEqual(CaptureFrameDisposition.classify(.suspended), .drop)
        XCTAssertEqual(CaptureFrameDisposition.classify(.stopped), .drop)
    }

    func testOptimizedHEVCBitrateScalesWithWorkloadAndStaysFarBelowRawLikeTargets() async throws {
        let hd30 = HEVCEncodingConfiguration.targetBitRate(width: 1_920, height: 1_080, fps: 30)
        let hd60 = HEVCEncodingConfiguration.targetBitRate(width: 1_920, height: 1_080, fps: 60)
        let uhd60 = HEVCEncodingConfiguration.targetBitRate(width: 3_840, height: 2_160, fps: 60)
        XCTAssertGreaterThan(hd60, hd30)
        XCTAssertGreaterThan(uhd60, hd60)
        let formerUHDTarget = 3_840 * 2_160 * 8
        XCTAssertLessThan(uhd60 * 3, formerUHDTarget)
        XCTAssertLessThanOrEqual(uhd60, 45_000_000)
        XCTAssertEqual(HEVCEncodingConfiguration.targetBitRate(width: 0, height: 0, fps: .nan), 1_500_000)

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("optimized-hevc-\(UUID().uuidString).mov")
        defer { try? FileManager.default.removeItem(at: url) }
        let writer = try HEVCWriter(url: url, width: 1_920, height: 1_080, fps: 60)
        let empty = try await writer.finish()
        XCTAssertEqual(empty.frames, 0)
    }

    func testRecordingMetricsUseExactFramesBytesAndFolderSeconds() {
        let exact = RecordingManifest(
            id: UUID(), name: "Exact", createdAt: Date(), hostStartNanos: 1, duration: 12.5,
            capture: CaptureSpec(), globalRect: CodableRect(CGRect(x: 0, y: 0, width: 100, height: 100)),
            pixelWidth: 100, pixelHeight: 100, deliveredFPS: 60, frameCount: 731, eventCount: 0
        )
        var legacy = exact
        legacy.id = UUID()
        legacy.name = "Legacy"
        legacy.duration = 10
        legacy.deliveredFPS = 2.5
        legacy.frameCount = nil
        XCTAssertEqual(legacy.encodedFrameCount, 25)

        let metrics = RecordingCollectionMetrics.total(for: [
            RecordingItem(manifest: exact, directory: URL(fileURLWithPath: "/tmp/exact"), storageBytes: 1_250_000_000),
            RecordingItem(manifest: legacy, directory: URL(fileURLWithPath: "/tmp/legacy"), storageBytes: 250_000_000)
        ])
        XCTAssertEqual(metrics.recordingCount, 2)
        XCTAssertEqual(metrics.frameCount, 756)
        XCTAssertEqual(metrics.storageBytes, 1_500_000_000)
        XCTAssertEqual(metrics.durationSeconds, 22.5, accuracy: 0.000_001)
    }

    func testTrainingIdentityIgnoresLibraryPresentationButTracksSamplingChanges() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("recording-identity-\(UUID().uuidString).atrrecord", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data([1, 2, 3]).write(to: directory.appendingPathComponent("capture.mov"))
        try Data("ATREVT01".utf8).write(to: directory.appendingPathComponent("events.atrevents"))
        var manifest = RecordingManifest(
            id: UUID(), name: "Before", createdAt: Date(), hostStartNanos: 1, duration: 2,
            capture: CaptureSpec(), globalRect: CodableRect(CGRect(x: 0, y: 0, width: 100, height: 100)),
            pixelWidth: 100, pixelHeight: 100, deliveredFPS: 30, frameCount: 60, eventCount: 0,
            folderID: UUID(), thumbnailFile: "thumbnail.jpg"
        )
        let original = try RecordingTrainingIdentity(recording: RecordingItem(manifest: manifest, directory: directory))
        manifest.name = "After"
        manifest.createdAt = Date().addingTimeInterval(100)
        manifest.folderID = UUID()
        manifest.thumbnailFile = nil
        manifest.frameCount = 999
        let presentationOnly = try RecordingTrainingIdentity(recording: RecordingItem(manifest: manifest, directory: directory))
        XCTAssertEqual(original, presentationOnly)
        manifest.trimStart = 0.25
        let changedSampling = try RecordingTrainingIdentity(recording: RecordingItem(manifest: manifest, directory: directory))
        XCTAssertNotEqual(original, changedSampling)
    }

    func testInputMonitorSessionsJoinBeforeTheyCanRestart() async throws {
        let monitor = InputCaptureService()
        do {
            try monitor.start()
        } catch {
            throw XCTSkip("The test host does not have Input Monitoring permission: \(error.localizedDescription)")
        }
        XCTAssertTrue(monitor.isRunning)
        for _ in 0..<5 {
            monitor.stop()
            XCTAssertFalse(monitor.isRunning)
            try monitor.start()
            XCTAssertTrue(monitor.isRunning)
        }
        await withTaskGroup(of: Void.self) { group in
            group.addTask { monitor.stop() }
            group.addTask { monitor.stop() }
        }
        XCTAssertFalse(monitor.isRunning)
    }

    func testNeuralInputSizingMirrorsCurrentPastAndFrameControlContracts() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 5, height: 3, colorMode: .color, bitDepth: 2, chroma: .yuv420, resizePolicy: .fit)
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 4, frameSpacing: 2, downsampleFactor: 2)
        profile.training.batchSize = 2
        profile.training.precision = .float16
        profile.training.perceptionFPS = 10
        profile.training.actionFPS = 20

        let input = NeuralInputSizing.summary(for: profile)
        XCTAssertEqual(input.currentPixelCount, 15)
        XCTAssertEqual(input.currentLumaValues, 15)
        XCTAssertEqual(input.currentChromaValuesPerPlane, 6)
        XCTAssertEqual(input.currentPackedVisionValues, 27)
        XCTAssertEqual(input.currentExpandedVisionValues, 45)
        XCTAssertEqual(input.currentCoordinateValues, 30)
        XCTAssertEqual(input.currentFirstConvolutionValues, 75)
        XCTAssertEqual(input.pastPixelCountPerFrame, 6)
        XCTAssertEqual(input.pastPackedVisionValuesPerFrame, 10)
        XCTAssertEqual(input.pastExpandedVisionValuesPerFrame, 18)
        XCTAssertEqual(input.pastFirstConvolutionValuesPerFrame, 30)
        XCTAssertEqual(input.pastFrameCount, 4)
        XCTAssertEqual(input.pastControlValues, 4 * 146)
        XCTAssertEqual(input.frameSpacingSeconds, 0.2, accuracy: 0.000_001)
        XCTAssertEqual(input.temporalLookbackSeconds, 0.8, accuracy: 0.000_001)
        XCTAssertEqual(input.valuesPerDecision, 75 + 4 * (30 + 146))
        XCTAssertEqual(input.runtimeEncodedVisionValues, 75 + 30)
        XCTAssertEqual(input.runtimeCachedVisualValues, 4 * 256)
        XCTAssertEqual(input.runtimeValuesPerDecision, 75 + 30 + 4 * (256 + 146))
        XCTAssertEqual(input.runtimeValuesPerSecond, 10 * (75 + 30 + 4 * (256 + 146)))
        XCTAssertEqual(input.packedVisionBytesPerSecond, 370)
        XCTAssertEqual(input.valuesPerTrainingBatch, 2 * (75 + 4 * (30 + 146)))
        XCTAssertEqual(input.quantizationLevels, 4)
        XCTAssertEqual(input.effectivePackedBits, 67 * 2)
        XCTAssertEqual(input.nominalBytesPerTrainingBatch, 2 * 2 * (75 + 4 * (30 + 146)))
    }

    func testNeuralInputSizingKeepsPastGrayscaleAtNativeLowerResolution() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 5, height: 3, colorMode: .grayscale, bitDepth: 8, chroma: .yuv420, resizePolicy: .stretch)
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 1, frameSpacing: 1, downsampleFactor: 2)
        profile.training.batchSize = 1
        profile.training.precision = .float32

        let input = NeuralInputSizing.summary(for: profile)
        XCTAssertEqual(input.currentPackedVisionValues, 15)
        XCTAssertEqual(input.currentChromaValuesPerPlane, 0)
        XCTAssertEqual(input.currentExpandedVisionValues, 15)
        XCTAssertEqual(input.currentFirstConvolutionValues, 45)
        XCTAssertEqual(input.pastPackedVisionValuesPerFrame, 6)
        XCTAssertEqual(input.pastFrameCount, 1)
        XCTAssertEqual(input.pastControlValues, 146)
        XCTAssertEqual(input.valuesPerDecision, 209)
        XCTAssertEqual(input.nominalBytesPerDecision, 209 * 4)
    }

    func testZeroPastFramesIsAValidatedCurrentOnlyContract() throws {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 5, height: 3, colorMode: .grayscale)
        profile.training.temporalVision = TemporalVisionConfiguration(
            pastFrameCount: 0,
            frameSpacing: 240,
            downsampleFactor: 8
        )
        XCTAssertNoThrow(try profile.training.effectiveTemporalVision.validated(current: profile.preprocessing))

        let input = NeuralInputSizing.summary(for: profile)
        XCTAssertEqual(input.pastFrameCount, 0)
        XCTAssertEqual(input.pastControlValues, 0)
        XCTAssertEqual(input.frameSpacingSeconds, 0)
        XCTAssertEqual(input.temporalLookbackSeconds, 0)
        XCTAssertEqual(input.totalPackedVisionValues, input.currentPackedVisionValues)
        XCTAssertEqual(input.valuesPerDecision, input.currentFirstConvolutionValues)
        XCTAssertEqual(input.runtimeEncodedVisionValues, input.currentFirstConvolutionValues)
        XCTAssertEqual(input.runtimeCachedVisualValues, 0)
        XCTAssertEqual(input.runtimeValuesPerDecision, input.currentFirstConvolutionValues)

        profile.training.architecture = .small
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        model.train(false)
        let current = MLXArray.zeros([1, 3, 5, 1], dtype: .float32)
        let predictions = model.currentOnlyPredictions(currentImages: current)
        let genericPredictions = model.predictions(
            currentImages: current,
            pastImages: MLXArray.zeros([1, 1, 1, 1, 1], dtype: .float32),
            pastControls: MLXArray.zeros([1, 1, ActionLayout.count], dtype: .float32)
        )
        let loss = model.loss(
            currentImages: current,
            pastImages: MLXArray.zeros([1, 1, 1, 1, 1], dtype: .float32),
            pastControls: MLXArray.zeros([1, 1, ActionLayout.count], dtype: .float32),
            targets: MLXArray.zeros([1, ActionLayout.count], dtype: .float32)
        )
        MLX.eval(predictions, genericPredictions, loss)
        XCTAssertEqual(predictions.shape, [1, ActionLayout.count])
        XCTAssertTrue(predictions.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertEqual(genericPredictions.asArray(Float.self), predictions.asArray(Float.self))
        XCTAssertTrue(loss.item(Float.self).isFinite)
    }

    func testTemporalValidationBudgetsCachedFeaturesInsteadOfHistoricalPixels() throws {
        let current = PreprocessingSpec(
            width: 8_192,
            height: 4_096,
            colorMode: .grayscale,
            bitDepth: 8,
            chroma: .yuv420,
            resizePolicy: .fit
        )
        let temporal = TemporalVisionConfiguration(
            pastFrameCount: TemporalVisionConfiguration.maximumPastFrameCount,
            frameSpacing: TemporalVisionConfiguration.maximumFrameSpacing,
            downsampleFactor: 1
        )
        XCTAssertNoThrow(try temporal.validated(current: current, cachedEmbeddingWidth: 256))
        XCTAssertThrowsError(try temporal.validated(current: current, cachedEmbeddingWidth: 100_000))
    }

    func testNeuralInputCapacityGuideUsesSimpleConservativeBands() {
        XCTAssertEqual(NeuralInputSizing.capacityGuide(inputValues: 75, parameterCount: 100).level, .comfortable)
        XCTAssertEqual(NeuralInputSizing.capacityGuide(inputValues: 76, parameterCount: 100).level, .balanced)
        XCTAssertEqual(NeuralInputSizing.capacityGuide(inputValues: 200, parameterCount: 100).level, .balanced)
        XCTAssertEqual(NeuralInputSizing.capacityGuide(inputValues: 201, parameterCount: 100).level, .high)
        XCTAssertEqual(NeuralInputSizing.capacityGuide(inputValues: 500, parameterCount: 100).level, .high)
        XCTAssertEqual(NeuralInputSizing.capacityGuide(inputValues: 501, parameterCount: 100).level, .tooHigh)

        let clamped = NeuralInputSizing.capacityGuide(inputValues: -1, parameterCount: 0)
        XCTAssertEqual(clamped.inputValues, 0)
        XCTAssertEqual(clamped.parameterCount, 1)
        XCTAssertEqual(clamped.inputsPerParameter, 0)
    }

    func testBinaryValidationMetricsExposeSparseHeadFailureModes() {
        let metrics = BinaryValidationMetrics(truePositives: 8, falsePositives: 2, falseNegatives: 2, trueNegatives: 88)
        XCTAssertEqual(metrics.precision, 0.8, accuracy: 0.000_001)
        XCTAssertEqual(metrics.recall, 0.8, accuracy: 0.000_001)
        XCTAssertEqual(metrics.f1, 0.8, accuracy: 0.000_001)
        XCTAssertEqual(metrics.falsePositiveRate, 2.0 / 90.0, accuracy: 0.000_001)

        let baseline = ValidationReport(
            sampleCount: 100,
            binary: metrics,
            buttons: nil,
            keyboard: metrics,
            modifiers: nil,
            absoluteMouseMAE: nil,
            activeRelativeMouseMAE: nil,
            activeScrollMAE: nil,
            idleContinuousFalseActionRate: nil
        )
        var collapsed = baseline
        collapsed.keyboard = BinaryValidationMetrics(truePositives: 3, falsePositives: 2, falseNegatives: 7, trueNegatives: 88)
        XCTAssertTrue(collapsed.hasSevereBinaryRegression(comparedTo: baseline))
        var ordinaryNoise = baseline
        ordinaryNoise.keyboard = BinaryValidationMetrics(truePositives: 7, falsePositives: 2, falseNegatives: 3, trueNegatives: 88)
        XCTAssertFalse(ordinaryNoise.hasSevereBinaryRegression(comparedTo: baseline))

        var unsupportedBaseline = baseline
        unsupportedBaseline.keyboard = BinaryValidationMetrics(truePositives: 3, falsePositives: 0, falseNegatives: 1, trueNegatives: 96)
        var unsupportedCollapse = unsupportedBaseline
        unsupportedCollapse.keyboard = BinaryValidationMetrics(truePositives: 0, falsePositives: 0, falseNegatives: 4, trueNegatives: 96)
        XCTAssertFalse(unsupportedCollapse.hasSevereBinaryRegression(comparedTo: unsupportedBaseline), "Four positives are too few for a hard publication gate.")
    }

    func testArchitecturePresetsHaveNoZeroWidths() {
        for architecture in [ArchitectureSpec.small, .balanced, .large] {
            XCTAssertTrue(architecture.convolutionChannels.allSatisfy { $0 > 0 })
            XCTAssertTrue(architecture.fusionWidths.allSatisfy { $0 > 0 })
            XCTAssertEqual(architecture.effectiveVisualPooling, .attention)
            XCTAssertGreaterThan(architecture.effectiveAttentionHeads, 0)
            XCTAssertGreaterThan(architecture.effectiveControlEmbedding, 0)
            XCTAssertEqual(architecture.convolutionChannels.last, architecture.convolutionChannels.dropLast().last)
        }
        XCTAssertEqual(CNNGeometry.layer(0, architecture: .balanced), CNNLayerGeometry(kernelSize: 5, effectiveStride: 2, receptiveField: 5))
        XCTAssertEqual(CNNGeometry.layer(1, architecture: .balanced), CNNLayerGeometry(kernelSize: 3, effectiveStride: 4, receptiveField: 9))
        XCTAssertEqual(CNNGeometry.layer(2, architecture: .balanced), CNNLayerGeometry(kernelSize: 3, effectiveStride: 8, receptiveField: 17))
        XCTAssertEqual(CNNGeometry.layer(3, architecture: .balanced), CNNLayerGeometry(kernelSize: 3, effectiveStride: 16, receptiveField: 33))
        XCTAssertEqual(CNNGeometry.layer(4, architecture: .balanced), CNNLayerGeometry(kernelSize: 3, effectiveStride: 16, receptiveField: 65))
        XCTAssertEqual(CNNGeometry.outputSize(width: 32, height: 24, architecture: .small).width, 2)
        XCTAssertEqual(CNNGeometry.outputSize(width: 32, height: 24, architecture: .small).height, 2)
        XCTAssertEqual(CNNGeometry.outputSize(width: 640, height: 360, architecture: .balanced).width, 40)
        XCTAssertEqual(CNNGeometry.outputSize(width: 640, height: 360, architecture: .balanced).height, 23)
        var convolutionFree = ArchitectureSpec.balanced
        convolutionFree.convolutionChannels = []
        XCTAssertEqual(CNNGeometry.layer(-1, architecture: convolutionFree), CNNLayerGeometry(kernelSize: 1, effectiveStride: 1, receptiveField: 1))
    }

    func testEfficientBackboneAndTemporalCacheCutRedundantCompute() {
        let profile = AIProfile.fresh()
        let architecture = profile.training.architecture
        let efficient = ModelSizing.visualBackboneMultiplyAdds(
            width: profile.preprocessing.width,
            height: profile.preprocessing.height,
            inputChannels: profile.preprocessing.channelCount,
            architecture: architecture
        )
        var width = profile.preprocessing.width
        var height = profile.preprocessing.height
        var input = profile.preprocessing.channelCount + 2
        var denseReference: Int64 = 0
        for index in architecture.convolutionChannels.indices {
            let output = architecture.convolutionChannels[index]
            let kernel = architecture.kernelSizes[index]
            let stride = architecture.strides[index]
            width = (width + 2 * (kernel / 2) - kernel) / stride + 1
            height = (height + 2 * (kernel / 2) - kernel) / stride + 1
            denseReference += Int64(width * height * input * output * kernel * kernel)
            input = output
        }
        XCTAssertLessThan(efficient * 2, denseReference)

        let temporal = profile.training.effectiveTemporalVision
        let past = temporal.pastFrameSpec(from: profile.preprocessing)
        let currentWork = efficient
        let pastWork = ModelSizing.visualBackboneMultiplyAdds(
            width: past.width,
            height: past.height,
            inputChannels: past.channelCount,
            architecture: architecture
        )
        let uncachedRuntimeWork = currentWork + Int64(temporal.pastFrameCount) * pastWork
        XCTAssertEqual(ModelSizing.runtimeVisualBackboneMultiplyAdds(profile), currentWork + pastWork)
        XCTAssertLessThan(ModelSizing.runtimeVisualBackboneMultiplyAdds(profile), uncachedRuntimeWork)

        var currentOnly = profile
        currentOnly.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 0)
        XCTAssertLessThan(ModelSizing.parameterCount(currentOnly), ModelSizing.parameterCount(profile))
    }

    func testAttentionPoolingIsResolutionStableAndMuchSmallerThanLegacyFlattening() {
        var attention = AIProfile.fresh()
        attention.preprocessing = PreprocessingSpec(width: 640, height: 360)
        attention.training.architecture = .balanced
        let attentionParameters = ModelSizing.parameterCount(attention)

        var higherResolution = attention
        higherResolution.preprocessing.width = 1_280
        higherResolution.preprocessing.height = 720
        XCTAssertEqual(ModelSizing.parameterCount(higherResolution), attentionParameters, "Attention pooling must not grow its learned projection with every pixel.")

        var legacy = attention
        legacy.training.architecture.visualPooling = .flattened
        let legacyParameters = ModelSizing.parameterCount(legacy)
        XCTAssertLessThan(attentionParameters * 4, legacyParameters, "The compact architecture should remove the resolution-dominated dense bottleneck.")
    }

    func testLegacyProfileDecodingPreservesExactArchitectureAndSchedule() throws {
        let encoded = try JSONEncoder().encode(TrainingConfiguration())
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        object.removeValue(forKey: "learningRateSchedule")
        object.removeValue(forKey: "cosineCycleEpochs")
        object.removeValue(forKey: "plateauPatience")
        object.removeValue(forKey: "minimumLearningRateRatio")
        object.removeValue(forKey: "binaryFocalGamma")
        object.removeValue(forKey: "generalization")
        var architecture = try XCTUnwrap(object["architecture"] as? [String: Any])
        architecture.removeValue(forKey: "visualPooling")
        architecture.removeValue(forKey: "attentionHeads")
        architecture.removeValue(forKey: "controlEmbedding")
        object["architecture"] = architecture
        let legacyData = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        let legacy = try JSONDecoder().decode(TrainingConfiguration.self, from: legacyData)

        XCTAssertNil(legacy.learningRateSchedule)
        XCTAssertEqual(legacy.effectiveLearningRateSchedule, .legacyInverseSquareRoot)
        XCTAssertEqual(legacy.effectiveBinaryFocalGamma, 0)
        XCTAssertNil(legacy.generalization)
        XCTAssertNil(legacy.architecture.visualPooling)
        XCTAssertEqual(legacy.architecture.effectiveVisualPooling, .flattened)
        XCTAssertEqual(legacy.architecture.effectiveControlEmbedding, 64)
        XCTAssertEqual(legacy.effectiveGeneralization, GeneralizationConfiguration())
        XCTAssertEqual(TrainingConfiguration().effectiveLearningRateSchedule, .adaptiveCosine)
        XCTAssertEqual(TrainingConfiguration().architecture.effectiveVisualPooling, .attention)

        let roundTripped = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(legacy)) as? [String: Any])
        XCTAssertNil(roundTripped["learningRateSchedule"], "Nil compatibility fields must remain absent so legacy checkpoint signatures stay stable.")
        XCTAssertNil(roundTripped["generalization"])
        let roundTrippedArchitecture = try XCTUnwrap(roundTripped["architecture"] as? [String: Any])
        XCTAssertNil(roundTrippedArchitecture["visualPooling"])
    }

    func testDefaultResolutionPolicyProjectionMatchesTheActualCNNGrid() {
        var profile = AIProfile.fresh()
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        model.train(false)
        let images = MLXArray.zeros([
            1,
            profile.preprocessing.height,
            profile.preprocessing.width,
            profile.preprocessing.channelCount
        ], dtype: .float32)
        let temporal = profile.training.effectiveTemporalVision
        let pastSpec = temporal.pastFrameSpec(from: profile.preprocessing)
        let pastImages = MLXArray.zeros([1, temporal.pastFrameCount, pastSpec.height, pastSpec.width, pastSpec.channelCount], dtype: .float32)
        let pastControls = MLXArray.zeros([1, temporal.pastFrameCount, ActionLayout.count], dtype: .float32)
        let predictions = model.predictions(currentImages: images, pastImages: pastImages, pastControls: pastControls)
        MLX.eval(predictions)
        XCTAssertEqual(predictions.shape, [1, ActionLayout.count])
        XCTAssertTrue(predictions.asArray(Float.self).allSatisfy(\.isFinite))
        let actualParameterCount = model.parameters().flattened().reduce(Int64(0)) { total, item in
            total + item.1.shape.reduce(Int64(1)) { $0 * Int64($1) }
        }
        XCTAssertEqual(actualParameterCount, ModelSizing.parameterCount(profile))
    }

    func testModelSizingMatchesActualGRUAndLSTMParameterTrees() {
        for recurrentKind in [RecurrentKind.gru, .lstm] {
            for pooling in VisualPoolingKind.allCases {
                var profile = AIProfile.fresh()
                profile.preprocessing = PreprocessingSpec(width: 12, height: 8, colorMode: .grayscale)
                profile.training.architecture = .small
                profile.training.architecture.recurrentKind = recurrentKind
                profile.training.architecture.visualPooling = pooling
                profile.training.precision = .float32
                let model = AgentPolicy(profile: profile)
                let parameters = model.parameters().flattened()
                let actual = parameters.reduce(Int64(0)) { total, item in
                    total + item.1.shape.reduce(Int64(1)) { $0 * Int64($1) }
                }
                XCTAssertEqual(actual, ModelSizing.parameterCount(profile), "Sizing drifted for \(recurrentKind.rawValue) / \(pooling.rawValue).")
                XCTAssertFalse(parameters.contains { $0.0.contains("coordinate") })
                XCTAssertEqual(parameters.contains { $0.0.contains("spatialAttention") }, pooling == .attention)
            }
        }
    }

    func testFixedThemesIncludePolishedLightDarkAndAlternatePalettes() {
        XCTAssertEqual(AppTheme.allCases, [.midnight, .daylight, .graphite, .ember])
        XCTAssertEqual(AppTheme.daylight.colorScheme, .light)
        XCTAssertEqual(AppTheme.midnight.colorScheme, .dark)
        for theme in AppTheme.allCases {
            let palette = theme.configuration
            XCTAssertNotEqual(palette.canvas, palette.panel)
            XCTAssertNotEqual(palette.text, palette.canvas)
            XCTAssertGreaterThan(palette.cornerRadius, 0)
            XCTAssertEqual(palette.label(for: .diagnostics), "Diagnostics")
        }
    }

    func testAppearanceTuningIsBoundedAndAdjustsTheWholeSurfaceSystem() {
        let extreme = UIAppearanceTuning(cornerRadius: 100, surfaceContrast: 2, accentIntensity: 0.1, sidebarWidth: 90).sanitized
        XCTAssertEqual(extreme.cornerRadius, 28)
        XCTAssertEqual(extreme.surfaceContrast, 1.45)
        XCTAssertEqual(extreme.accentIntensity, 0.65)
        XCTAssertEqual(extreme.sidebarWidth, 205)

        let base = AppTheme.midnight.configuration
        let flatter = UIAppearanceTuning(cornerRadius: 4, surfaceContrast: 0.7, accentIntensity: 0.65, sidebarWidth: 205).applying(to: base)
        let deeper = UIAppearanceTuning(cornerRadius: 28, surfaceContrast: 1.45, accentIntensity: 1.5, sidebarWidth: 300).applying(to: base)
        func separation(_ color: AppearanceColor, from canvas: AppearanceColor) -> Double {
            abs(color.red - canvas.red) + abs(color.green - canvas.green) + abs(color.blue - canvas.blue)
        }
        XCTAssertLessThan(separation(flatter.panel, from: flatter.canvas), separation(deeper.panel, from: deeper.canvas))
        XCTAssertLessThan(flatter.controlOpacity, deeper.controlOpacity)
        XCTAssertEqual(flatter.cornerRadius, 4)
        XCTAssertEqual(deeper.cornerRadius, 28)
        XCTAssertEqual(flatter.sidebarWidth, 205)
        XCTAssertEqual(deeper.sidebarWidth, 300)

        let invalid = UIAppearanceTuning(cornerRadius: .nan, surfaceContrast: .infinity, accentIntensity: -.infinity, sidebarWidth: .nan).sanitized
        XCTAssertEqual(invalid, .standard)
    }

    func testTrainingDurationFormattingSwitchesFromHoursToDays() {
        XCTAssertEqual(TrainingDurationFormatter.string(seconds: 3_600), "1.00 h")
        XCTAssertEqual(TrainingDurationFormatter.string(seconds: 10 * 3_600), "10.0 h")
        XCTAssertEqual(TrainingDurationFormatter.string(seconds: 24 * 3_600), "1.00 days")
        XCTAssertEqual(TrainingDurationFormatter.string(seconds: .infinity), "0.00 h")
    }

    func testLegacyTrainingProgressDecodesWithoutTimingCounters() throws {
        let data = Data(#"{"globalStep":671804,"epoch":3787,"updatedAt":"2026-07-13T10:25:45Z","savedBrainCount":10}"#.utf8)
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601
        let progress = try decoder.decode(TrainingProgressSummary.self, from: data)
        XCTAssertNil(progress.trainingDurationSeconds)
        XCTAssertNil(progress.experienceDurationSeconds)
    }

    func testReinforcementConfigurationIsBoundedAndLegacyProfilesStayOptIn() throws {
        let defaults = ReinforcementConfiguration()
        XCTAssertNoThrow(try defaults.validated())
        XCTAssertFalse(defaults.enabled)
        XCTAssertLessThan(defaults.binaryExploration, 0.01)

        var invalid = defaults
        invalid.rewardHotkey = invalid.punishmentHotkey
        XCTAssertThrowsError(try invalid.validated())
        invalid = defaults
        invalid.creditWindowSeconds = .infinity
        XCTAssertThrowsError(try invalid.validated())
        invalid = defaults
        invalid.scrollCarbonModifiers = 0
        XCTAssertThrowsError(try invalid.validated())
        invalid = defaults
        invalid.rewardHotkey = HotkeyBinding(keyCode: 128, carbonModifiers: 0)
        XCTAssertThrowsError(try invalid.validated())
        invalid.rewardHotkey = .mouse(.max)
        XCTAssertNoThrow(try invalid.validated(), "Vendor-specific side-button numbers remain valid feedback controls.")

        let original = AIProfile.fresh(name: "Legacy")
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(original)) as? [String: Any]
        )
        object.removeValue(forKey: "reinforcement")
        let legacy = try JSONDecoder().decode(
            AIProfile.self,
            from: JSONSerialization.data(withJSONObject: object)
        )
        XCTAssertNil(legacy.reinforcement)
        XCTAssertFalse(legacy.effectiveReinforcement.enabled)
        XCTAssertFalse(legacy.canRunOrLearn)

        var newRL = legacy
        newRL.reinforcement = ReinforcementConfiguration(enabled: true)
        XCTAssertTrue(newRL.canRunOrLearn)
    }

    func testReinforcementScrollAccumulatesExactSignedSteps() {
        var configuration = ReinforcementConfiguration()
        configuration.scrollFeedbackEnabled = true
        configuration.scrollStep = 0.25
        configuration.scrollUpRewards = true
        let modifiers = HotkeyBinding(
            keyCode: 0,
            carbonModifiers: configuration.scrollCarbonModifiers
        ).cgEventModifiers
        var accumulator = ReinforcementScrollAccumulator()

        let partial = accumulator.process(
            InputSample(timestampNanos: 1, kind: .scroll, scrollY: 4, modifiers: modifiers),
            configuration: configuration
        )
        XCTAssertTrue(partial.handled)
        XCTAssertNil(partial.signal)
        let completed = accumulator.process(
            InputSample(timestampNanos: 2, kind: .scroll, scrollY: 6, modifiers: modifiers),
            configuration: configuration
        )
        XCTAssertEqual(completed.signal, 0.25)
        let punishment = accumulator.process(
            InputSample(timestampNanos: 3, kind: .scroll, scrollY: -20, modifiers: modifiers),
            configuration: configuration
        )
        XCTAssertEqual(punishment.signal, -0.5)

        let ordinaryScroll = accumulator.process(
            InputSample(timestampNanos: 4, kind: .scroll, scrollY: 10, modifiers: 0),
            configuration: configuration
        )
        XCTAssertFalse(ordinaryScroll.handled)
        configuration.scrollUpRewards = false
        let reversed = accumulator.process(
            InputSample(timestampNanos: 5, kind: .scroll, scrollY: 10, modifiers: modifiers),
            configuration: configuration
        )
        XCTAssertEqual(reversed.signal, -0.25)
    }

    func testBrandNewReinforcementPolicyStartsNeutralAndSparse() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 8, height: 8, colorMode: .grayscale)
        profile.training.temporalVision = TemporalVisionConfiguration(
            pastFrameCount: 0,
            frameSpacing: 1,
            downsampleFactor: 1
        )
        profile.training.architecture = .small
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        model.initializeForSafeExploration()
        model.train(false)
        let predictions = model.currentOnlyPredictions(
            currentImages: MLXArray.zeros([1, 8, 8, 1], dtype: .float32)
        )
        MLX.eval(predictions)
        let values = predictions.asArray(Float.self)
        for index in ActionLayout.absoluteMouse {
            XCTAssertEqual(values[index], 0.5, accuracy: 0.000_001)
        }
        for index in Array(ActionLayout.relativeMouse) + Array(ActionLayout.scroll) {
            XCTAssertEqual(values[index], 0, accuracy: 0.000_001)
        }
        for index in ActionLayout.binary {
            XCTAssertEqual(values[index], 0.002, accuracy: 0.000_1)
        }
    }

    func testReinforcementObjectiveMovesRewardedAndPunishedActionsInOppositeDirections() {
        var configuration = ReinforcementConfiguration()
        configuration.binaryExploration = 0
        configuration.behaviorAnchor = 0
        configuration.entropyBonus = 0
        let selected = ActionLayout.keyboard.lowerBound + 13
        var actionValues = [Float](repeating: 0, count: ActionLayout.count)
        var maskValues = [Float](repeating: 0, count: ActionLayout.count)
        actionValues[selected] = 1
        maskValues[selected] = 1
        let actions = MLXArray(actionValues, [1, ActionLayout.count])
        let mask = MLXArray(maskValues, [1, ActionLayout.count])
        let behavior = MLXArray.zeros([1, ActionLayout.count], dtype: .float32)

        func gradient(for advantage: Float) -> Float {
            let derivative = grad { logits in
                ReinforcementPolicyObjective.loss(
                    logits: logits,
                    behaviorLogits: behavior,
                    actions: actions,
                    actionMask: mask,
                    advantages: MLXArray([advantage]),
                    configuration: configuration,
                    dtype: .float32
                )
            }(MLXArray.zeros([1, ActionLayout.count], dtype: .float32))
            MLX.eval(derivative)
            return derivative.asArray(Float.self)[selected]
        }

        XCTAssertLessThan(gradient(for: 1), 0, "Gradient descent should increase a rewarded action's logit.")
        XCTAssertGreaterThan(gradient(for: -1), 0, "Gradient descent should decrease a punished action's logit.")
    }

    func testReinforcementTrainerPerformsFiniteOnlineUpdateAndStagesResumeState() throws {
        let snapshotRoot = FileManager.default.temporaryDirectory.appendingPathComponent(
            "rl-trainer-\(UUID().uuidString)",
            isDirectory: true
        )
        defer { try? FileManager.default.removeItem(at: snapshotRoot) }
        try FileManager.default.createDirectory(at: snapshotRoot, withIntermediateDirectories: true)
        var profile = AIProfile.fresh(name: "RL unit")
        profile.preprocessing = PreprocessingSpec(width: 8, height: 8, colorMode: .grayscale)
        profile.training.temporalVision = TemporalVisionConfiguration(
            pastFrameCount: 0,
            frameSpacing: 1,
            downsampleFactor: 1
        )
        profile.training.architecture = .small
        profile.training.precision = .float32
        profile.training.perceptionFPS = 10
        var configuration = ReinforcementConfiguration()
        configuration.maximumCreditFrames = 1
        configuration.creditWindowSeconds = 1
        configuration.autosaveFeedbackCount = 1
        let model = AgentPolicy(profile: profile)
        model.initializeForSafeExploration()
        model.train(false)
        let packed = Data(repeating: 127, count: profile.preprocessing.sampleByteCount)
        let images = VisionPreprocessor.mlxTensor(
            MLXArray(packed, [1, packed.count], dtype: .uint8),
            spec: profile.preprocessing
        )
        let logits = model.logits(temporalFeatures: model.currentOnlyTemporalFeatures(currentImages: images))
        MLX.eval(logits)
        let behavior = logits.asArray(Float.self)
        var action = [Float](repeating: 0, count: ActionLayout.count)
        var mask = [Float](repeating: 0, count: ActionLayout.count)
        action[ActionLayout.buttons.lowerBound] = 1
        mask[ActionLayout.buttons.lowerBound] = 1
        let trainer = try ReinforcementTrainer(
            model: model,
            profile: profile,
            configuration: configuration,
            baseVersion: nil,
            baseVersionDirectory: nil,
            snapshotRoot: snapshotRoot,
            demonstratedKeyCodes: [],
            trainingShowsCursor: false,
            recommendedMouseMode: .relative
        )
        trainer.record(
            timestamp: 1,
            currentPacked: packed,
            pastEmbeddings: [],
            pastControls: [],
            behaviorLogits: behavior,
            action: action,
            allowedMask: mask
        )
        let update = try trainer.apply(ReinforcementSignal(
            timestamp: 1.1,
            value: 1,
            sourceIdentifier: "test"
        ))
        XCTAssertEqual(update.metrics.updateCount, 1)
        XCTAssertEqual(update.metrics.creditedFrames, 1)
        XCTAssertTrue(try XCTUnwrap(update.metrics.lastPolicyLoss).isFinite)
        XCTAssertTrue(update.shouldAutosave)

        let snapshot = try XCTUnwrap(trainer.makeSnapshot(isAutosave: false))
        XCTAssertEqual(snapshot.manifest.reinforcementFeedbackCount, 1)
        XCTAssertEqual(snapshot.manifest.reinforcementUpdateCount, 1)
        XCTAssertEqual(
            snapshot.manifest.trainingDataSchema,
            TrainingDataContract.schemaVersion
        )
        XCTAssertNil(
            snapshot.manifest.trainingObjectiveSchema,
            "A brand-new RL-only brain has no supervised-loss ancestry to claim."
        )
        XCTAssertTrue(FileManager.default.fileExists(atPath: snapshot.stagingDirectory.appendingPathComponent(snapshot.manifest.weightsFile).path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: snapshot.stagingDirectory.appendingPathComponent(try XCTUnwrap(snapshot.manifest.reinforcementOptimizerFile)).path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: snapshot.stagingDirectory.appendingPathComponent(try XCTUnwrap(snapshot.manifest.reinforcementStateFile)).path))
    }

    func testReinforcementPublicationActivatesBrainRemovesOldCheckpointAndRejectsStaleSnapshots() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "workspace-rl-publication-\(UUID().uuidString)",
            isDirectory: true
        )
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()
        var profile = AIProfile.fresh(name: "Online learner")
        var reinforcement = ReinforcementConfiguration()
        reinforcement.enabled = true
        profile.reinforcement = reinforcement
        let oldVersionID = UUID()
        profile.activeVersionID = oldVersionID
        try await store.saveProfile(profile)
        let oldVersion = ModelVersionManifest(
            id: oldVersionID,
            name: "Supervised brain",
            createdAt: Date(timeIntervalSince1970: 1),
            globalStep: 40,
            trainingLoss: 0.2,
            preprocessing: profile.preprocessing,
            channels: profile.channels,
            training: profile.training
        )
        try await store.saveVersionManifest(oldVersion, profileID: profile.id)
        let checkpoint = await store.checkpointDirectory(profileID: profile.id)
        try FileManager.default.createDirectory(at: checkpoint, withIntermediateDirectories: true)
        try Data("old supervised optimizer".utf8).write(to: checkpoint.appendingPathComponent("state.json"))

        let profileRoot = await store.profileDirectory(profile.id)
        let sessionID = UUID()
        let sessionStartedAt = Date(timeIntervalSince1970: 10)
        let staging = profileRoot.appendingPathComponent(
            ".ReinforcementSnapshot.\(sessionID.uuidString).1.tmp",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        for name in ["weights.safetensors", "reinforcement-optimizer.safetensors", "reinforcement-state.json"] {
            try Data(name.utf8).write(to: staging.appendingPathComponent(name))
        }
        let learnedID = UUID()
        let learned = ModelVersionManifest(
            id: learnedID,
            name: "RL brain",
            createdAt: Date(timeIntervalSince1970: 11),
            globalStep: 43,
            trainingLoss: -0.3,
            validationLoss: nil,
            preprocessing: profile.preprocessing,
            channels: profile.channels,
            training: profile.training,
            epoch: 1,
            isAutosave: false,
            demonstratedKeyCodes: [0, 1, 2, 13],
            relativeMouseScale: GameCameraContract.deltaScale,
            trainingDataSchema: TrainingDataContract.schemaVersion,
            reinforcementOptimizerFile: "reinforcement-optimizer.safetensors",
            reinforcementStateFile: "reinforcement-state.json",
            reinforcementSessionID: sessionID,
            reinforcementSessionStartedAt: sessionStartedAt,
            reinforcementSequence: 1,
            reinforcementFeedbackCount: 5,
            reinforcementUpdateCount: 3,
            reinforcementNetReward: 1.5,
            reinforcementTrainingSeconds: 0.12
        )
        let publication = try await store.publishReinforcementSnapshot(
            ReinforcementSnapshot(profileID: profile.id, stagingDirectory: staging, manifest: learned)
        )
        XCTAssertEqual(publication?.profile.activeVersionID, learnedID)
        XCTAssertFalse(FileManager.default.fileExists(atPath: checkpoint.path), "A later supervised run must warm-start from RL weights, not restore the pre-RL checkpoint.")
        let learnedDirectory = await store.versionDirectory(profileID: profile.id, versionID: learnedID)
        XCTAssertTrue(FileManager.default.fileExists(atPath: learnedDirectory.appendingPathComponent("weights.safetensors").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: learnedDirectory.appendingPathComponent("manifest.json").path))
        let profilesAfterPublication = await store.listProfiles()
        let reloaded = try XCTUnwrap(profilesAfterPublication.first { $0.id == profile.id })
        XCTAssertEqual(reloaded.trainingProgress?.reinforcementFeedbackCount, 5)
        XCTAssertEqual(reloaded.trainingProgress?.reinforcementUpdateCount, 3)
        XCTAssertEqual(reloaded.trainingProgress?.reinforcementNetReward, 1.5)

        let staleStaging = profileRoot.appendingPathComponent(
            ".ReinforcementSnapshot.\(sessionID.uuidString).stale.tmp",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: staleStaging, withIntermediateDirectories: false)
        for name in ["weights.safetensors", "reinforcement-optimizer.safetensors", "reinforcement-state.json"] {
            try Data(name.utf8).write(to: staleStaging.appendingPathComponent(name))
        }
        var stale = learned
        stale.id = UUID()
        stale.createdAt = Date(timeIntervalSince1970: 12)
        let stalePublication = try await store.publishReinforcementSnapshot(
            ReinforcementSnapshot(profileID: profile.id, stagingDirectory: staleStaging, manifest: stale)
        )
        XCTAssertNil(stalePublication)
        XCTAssertFalse(FileManager.default.fileExists(atPath: staleStaging.path))
        let profilesAfterStaleSnapshot = await store.listProfiles()
        XCTAssertEqual(profilesAfterStaleSnapshot.first { $0.id == profile.id }?.activeVersionID, learnedID)
    }

    func testLegacyExperienceEstimateUsesOptimizerWorkInsteadOfCurrentRecordings() {
        var profile = AIProfile.fresh()
        profile.training.batchSize = 64
        profile.training.actionFPS = 60
        profile.trainingProgress = TrainingProgressSummary(
            globalStep: 671_804,
            epoch: 3_787,
            updatedAt: Date(),
            savedBrainCount: 10,
            trainingDurationSeconds: 3_749.642
        )

        let summary = profile.trainingDurationSummary(recordings: [])
        XCTAssertEqual(summary.trainingSeconds, 3_749.642, accuracy: 0.001)
        XCTAssertEqual(summary.experienceSeconds, Double(671_804 * 64) / 60, accuracy: 0.001)
        XCTAssertTrue(summary.experienceIsEstimated)
    }

    func testInputEventBinaryRoundTrip() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("events-\(UUID().uuidString).atrevents")
        defer { try? FileManager.default.removeItem(at: url) }
        let writer = try InputEventWriter(url: url)
        let expected = InputSample(timestampNanos: 123_456, kind: .key, x: 10, y: 20, deltaX: 3, deltaY: -4, button: 2, scrollX: 1.5, scrollY: -2.5, keyCode: 12, modifiers: 0x1C0000, isDown: true)
        writer.append(expected)
        XCTAssertEqual(try writer.finish(), 1)
        XCTAssertEqual(try InputEventReader.read(url: url), [expected])
    }

    func testExternalRecorderGoldenManifestAndInputAreNativeCompatible() throws {
        let manifestData = Data(#"""
        {
          "schemaVersion": 2,
          "id": "11111111-2222-3333-4444-555555555555",
          "name": "Mixed platform sample",
          "createdAt": "2026-08-08T09:34:56Z",
          "hostStartNanos": 5000000000,
          "duration": 1,
          "capture": {
            "kind": "Window Region",
            "windowID": 42,
            "region": { "x": 3, "y": 4, "width": 640, "height": 360 },
            "requestedFPS": 60,
            "showsCursor": true
          },
          "globalRect": { "x": 0, "y": 0, "width": 1920, "height": 1080 },
          "pixelWidth": 1920,
          "pixelHeight": 1080,
          "deliveredFPS": 60,
          "eventCount": 1,
          "videoFile": "capture.mov",
          "eventFile": "events.atrevents",
          "trimStart": 0,
          "trimEnd": 1,
          "folderID": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
          "excludedKeyCodes": [0, 49, 56]
        }
        """#.utf8)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let manifest = try decoder.decode(RecordingManifest.self, from: manifestData)
        XCTAssertTrue(manifest.isStructurallyValid)
        XCTAssertEqual(manifest.capture.kind, .windowRegion)
        XCTAssertEqual(manifest.capture.windowID, 42)
        XCTAssertEqual(manifest.excludedKeyCodes, [0, 49, 56])

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("external-golden-\(UUID().uuidString).atrevents")
        defer { try? FileManager.default.removeItem(at: url) }
        let encoded = "QVRSRVZUMDEBAAAACAcGBQQDAgECAQcAfgAAAAAAEgAAAAAAAAAAAACAKEAAAAAAAIAzwAAAAAAAAA5AAAAAAAAAEsAAAAAAAIAYQAAAAAAAgCDA"
        try XCTUnwrap(Data(base64Encoded: encoded)).write(to: url)
        let events = try InputEventReader.read(url: url)
        XCTAssertEqual(events, [InputSample(
            timestampNanos: 0x0102_0304_0506_0708,
            kind: .mouseButton,
            x: 12.25,
            y: -19.5,
            deltaX: 3.75,
            deltaY: -4.5,
            button: 7,
            scrollX: 6.125,
            scrollY: -8.25,
            keyCode: 0x7E,
            modifiers: 0x0012_0000,
            isDown: true
        )])
    }

    func testInputEventWriterKeepsConcurrentHostTimesMonotonic() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("events-monotonic-\(UUID().uuidString).atrevents")
        defer { try? FileManager.default.removeItem(at: url) }
        let writer = try InputEventWriter(url: url)
        writer.append(InputSample(timestampNanos: 20, kind: .key, keyCode: 49, isDown: true))
        writer.append(InputSample(timestampNanos: 19, kind: .key, keyCode: 49, isDown: false))
        _ = try writer.finish()
        XCTAssertEqual(try InputEventReader.read(url: url).map(\.timestampNanos), [20, 20])
    }

    func testInputEventFilePreservesEveryKeyAndExtendedMouseInputField() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("all-inputs-\(UUID().uuidString).atrevents")
        defer { try? FileManager.default.removeItem(at: url) }
        let writer = try InputEventWriter(url: url)
        var expected: [InputSample] = []
        var timestamp: UInt64 = 1
        for keyCode in UInt16(0)..<128 {
            expected.append(InputSample(timestampNanos: timestamp, kind: .key, keyCode: keyCode, isDown: true))
            timestamp += 1
            expected.append(InputSample(timestampNanos: timestamp, kind: .key, keyCode: keyCode, isDown: false))
            timestamp += 1
        }
        let modifierSamples = [
            InputSample(timestampNanos: timestamp, kind: .flags, keyCode: 56, modifiers: CGEventFlags.maskShift.rawValue, isDown: true),
            InputSample(timestampNanos: timestamp + 1, kind: .flags, keyCode: 56, modifiers: 0, isDown: false)
        ]
        expected += modifierSamples
        timestamp += 2
        for button in UInt8(0)..<32 {
            expected.append(InputSample(timestampNanos: timestamp, kind: .mouseButton, x: 100, y: 200, button: button, isDown: true))
            timestamp += 1
            expected.append(InputSample(timestampNanos: timestamp, kind: .mouseButton, x: 101, y: 201, button: button, isDown: false))
            timestamp += 1
        }
        expected.append(InputSample(timestampNanos: timestamp, kind: .mouseMove, x: 300, y: 400, deltaX: -12, deltaY: 9))
        expected.append(InputSample(timestampNanos: timestamp + 1, kind: .scroll, x: 300, y: 400, scrollX: 1.25, scrollY: -6.5))
        for sample in expected { writer.append(sample) }
        XCTAssertEqual(try writer.finish(), expected.count)
        XCTAssertEqual(try InputEventReader.read(url: url), expected)
    }

    func testRecordingPresetValidationAndLegacyKeyboardHotkeyDecoding() throws {
        let preset = RecordingPreset(
            id: UUID(), name: "  Gameplay  ", createdAt: Date(), captureKind: .screenRegion,
            captureFPS: 120, showsCursor: false,
            region: CodableRect(CGRect(x: 10, y: 20, width: 800, height: 450)),
            trimStart: 0.25, trimEnd: 0.5, excludedKeyCodes: [49, 56]
        )
        XCTAssertEqual(try preset.validated().name, "Gameplay")
        var invalid = preset
        invalid.captureFPS = .nan
        XCTAssertThrowsError(try invalid.validated())

        let legacy = Data("{\"keyCode\":15,\"carbonModifiers\":6400}".utf8)
        let binding = try JSONDecoder().decode(HotkeyBinding.self, from: legacy)
        XCTAssertNil(binding.mouseButton)
        XCTAssertEqual(binding.keyCode, 15)
    }

    func testInputEventReaderRejectsUnsupportedAndTruncatedFiles() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("invalid-events-\(UUID().uuidString).atrevents")
        defer { try? FileManager.default.removeItem(at: url) }
        var version = UInt32(2).littleEndian
        var data = Data("ATREVT01".utf8)
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        try data.write(to: url)
        XCTAssertThrowsError(try InputEventReader.read(url: url))

        version = UInt32(1).littleEndian
        data = Data("ATREVT01".utf8)
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        data.append(0)
        try data.write(to: url)
        XCTAssertThrowsError(try InputEventReader.read(url: url))

        let writer = try InputEventWriter(url: url)
        writer.append(InputSample(timestampNanos: 10, kind: .key, keyCode: 1, isDown: true))
        writer.append(InputSample(timestampNanos: 20, kind: .key, keyCode: 1, isDown: false))
        _ = try writer.finish()
        let valid = try Data(contentsOf: url)

        var unknownKind = valid
        unknownKind[12 + 8] = 255
        try unknownKind.write(to: url)
        XCTAssertThrowsError(try InputEventReader.mapped(url: url))

        var nonFinite = valid
        var nanBits = Double.nan.bitPattern.littleEndian
        withUnsafeBytes(of: &nanBits) { nonFinite.replaceSubrange((12 + 24)..<(12 + 32), with: $0) }
        try nonFinite.write(to: url)
        XCTAssertThrowsError(try InputEventReader.mapped(url: url))

        var reversed = valid
        var earlier = UInt64(5).littleEndian
        withUnsafeBytes(of: &earlier) { reversed.replaceSubrange((12 + InputEventReader.recordSize)..<(20 + InputEventReader.recordSize), with: $0) }
        try reversed.write(to: url)
        XCTAssertThrowsError(try InputEventReader.mapped(url: url))
    }

    func testMetalPreprocessingSizeAndQuantization() throws {
        var pixelBuffer: CVPixelBuffer?
        XCTAssertEqual(CVPixelBufferCreate(kCFAllocatorDefault, 8, 8, kCVPixelFormatType_32BGRA, [kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary, &pixelBuffer), kCVReturnSuccess)
        let buffer = try XCTUnwrap(pixelBuffer)
        CVPixelBufferLockBaseAddress(buffer, [])
        memset(CVPixelBufferGetBaseAddress(buffer), 127, CVPixelBufferGetDataSize(buffer))
        CVPixelBufferUnlockBaseAddress(buffer, [])
        let processor = try VisionPreprocessor()
        let spec = PreprocessingSpec(width: 13, height: 7, colorMode: .color, bitDepth: 2, chroma: .yuv420, resizePolicy: .stretch)
        let packed = try processor.process(buffer, spec: spec)
        XCTAssertEqual(packed.count, spec.sampleByteCount)
        XCTAssertTrue(packed.allSatisfy { [0, 85, 170, 255].contains($0) })
    }

    func testPipelinedMetalPreprocessingPreservesSynchronousBytesAndOrder() throws {
        let attributes = [
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ] as CFDictionary
        let buffers: [CVPixelBuffer] = try [19, 67, 113, 181, 239].map { value in
            var pixelBuffer: CVPixelBuffer?
            XCTAssertEqual(
                CVPixelBufferCreate(
                    kCFAllocatorDefault,
                    32,
                    24,
                    kCVPixelFormatType_32BGRA,
                    attributes,
                    &pixelBuffer
                ),
                kCVReturnSuccess
            )
            let buffer = try XCTUnwrap(pixelBuffer)
            CVPixelBufferLockBaseAddress(buffer, [])
            memset(CVPixelBufferGetBaseAddress(buffer), Int32(value), CVPixelBufferGetDataSize(buffer))
            CVPixelBufferUnlockBaseAddress(buffer, [])
            return buffer
        }
        let spec = PreprocessingSpec(
            width: 23,
            height: 13,
            colorMode: .color,
            bitDepth: 8,
            chroma: .yuv420,
            resizePolicy: .fit
        )
        let synchronous = try VisionPreprocessor()
        let expected = try buffers.map { try synchronous.process($0, spec: spec) }
        let pipelined = try VisionPreprocessor()
        let pending = try buffers.map { try pipelined.submit($0, spec: spec) }
        let actual = try pending.map { frame in
            try frame.withPackedBytes { Data($0) }
        }

        XCTAssertEqual(actual, expected)

        let currentSpec = PreprocessingSpec(
            width: 23,
            height: 13,
            colorMode: .color,
            bitDepth: 8,
            chroma: .yuv420,
            resizePolicy: .fit
        )
        let pastSpec = PreprocessingSpec(
            width: 7,
            height: 5,
            colorMode: .grayscale,
            bitDepth: 4,
            chroma: .yuv420,
            resizePolicy: .fill
        )
        let expectedPair = [
            try synchronous.process(buffers[2], spec: currentSpec),
            try synchronous.process(buffers[2], spec: pastSpec)
        ]
        let paired = try pipelined.submit(buffers[2], specs: [currentSpec, pastSpec])
        let actualPair = try paired.map { frame in
            try frame.withPackedBytes { Data($0) }
        }
        XCTAssertEqual(actualPair, expectedPair)
    }

    func testNativeVideoRangePreprocessingMatchesBGRAWithoutQualityLoss() throws {
        let attributes = [
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ] as CFDictionary
        var bgraBuffer: CVPixelBuffer?
        var yuvBuffer: CVPixelBuffer?
        XCTAssertEqual(CVPixelBufferCreate(kCFAllocatorDefault, 16, 16, kCVPixelFormatType_32BGRA, attributes, &bgraBuffer), kCVReturnSuccess)
        XCTAssertEqual(CVPixelBufferCreate(kCFAllocatorDefault, 16, 16, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, attributes, &yuvBuffer), kCVReturnSuccess)
        let bgra = try XCTUnwrap(bgraBuffer)
        let yuv = try XCTUnwrap(yuvBuffer)

        CVPixelBufferLockBaseAddress(bgra, [])
        memset(CVPixelBufferGetBaseAddress(bgra), 128, CVPixelBufferGetDataSize(bgra))
        CVPixelBufferUnlockBaseAddress(bgra, [])
        CVPixelBufferLockBaseAddress(yuv, [])
        memset(CVPixelBufferGetBaseAddressOfPlane(yuv, 0), 126, CVPixelBufferGetBytesPerRowOfPlane(yuv, 0) * CVPixelBufferGetHeightOfPlane(yuv, 0))
        memset(CVPixelBufferGetBaseAddressOfPlane(yuv, 1), 128, CVPixelBufferGetBytesPerRowOfPlane(yuv, 1) * CVPixelBufferGetHeightOfPlane(yuv, 1))
        CVPixelBufferUnlockBaseAddress(yuv, [])
        CVBufferSetAttachment(yuv, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)

        let processor = try VisionPreprocessor()
        let spec = PreprocessingSpec(width: 13, height: 9, colorMode: .color, bitDepth: 8, chroma: .yuv444, resizePolicy: .stretch)
        let bgraPacked = try processor.process(bgra, spec: spec)
        let yuvPacked = try processor.process(yuv, spec: spec)
        XCTAssertEqual(bgraPacked.count, yuvPacked.count)
        XCTAssertLessThanOrEqual(zip(bgraPacked, yuvPacked).map { abs(Int($0) - Int($1)) }.max() ?? .max, 1)
    }

    func testPackedUInt8ExpandsInsideMLXExactlyLikeCPU() {
        let specs = [
            PreprocessingSpec(width: 7, height: 5, colorMode: .grayscale, bitDepth: 8, chroma: .yuv444),
            PreprocessingSpec(width: 7, height: 5, colorMode: .color, bitDepth: 8, chroma: .yuv420),
            PreprocessingSpec(width: 7, height: 5, colorMode: .color, bitDepth: 8, chroma: .yuv422),
            PreprocessingSpec(width: 7, height: 5, colorMode: .color, bitDepth: 8, chroma: .yuv444)
        ]
        for spec in specs {
            let packed = Data((0..<spec.sampleByteCount).map { UInt8($0 % 251) })
            let tensor = VisionPreprocessor.mlxTensor(packed, batch: 1, spec: spec)
            MLX.eval(tensor)
            let gpu = tensor.asArray(Float.self)
            let cpu = VisionPreprocessor.unpackFloats(packed, spec: spec)
            XCTAssertEqual(tensor.shape, [1, spec.height, spec.width, spec.channelCount])
            XCTAssertTrue(zip(gpu, cpu).allSatisfy { abs($0 - $1) < 1e-6 }, "Packed expansion differs for \(spec.chroma.rawValue)")
        }
    }

    func testPackedUInt8BFloat16ExpansionMatchesEstablishedRoundingExactly() {
        let specs = [
            PreprocessingSpec(
                width: 256, height: 1, colorMode: .grayscale,
                bitDepth: 8, chroma: .yuv444
            ),
            PreprocessingSpec(
                width: 7, height: 5, colorMode: .color,
                bitDepth: 8, chroma: .yuv420
            ),
            PreprocessingSpec(
                width: 7, height: 5, colorMode: .color,
                bitDepth: 8, chroma: .yuv422
            ),
            PreprocessingSpec(
                width: 7, height: 5, colorMode: .color,
                bitDepth: 8, chroma: .yuv444
            )
        ]
        for spec in specs {
            let packed = Data((0..<spec.sampleByteCount).map { UInt8($0 % 256) })
            let direct = VisionPreprocessor.mlxTensor(
                packed, batch: 1, spec: spec, dtype: .bfloat16
            )
            let established = VisionPreprocessor.mlxTensor(
                packed, batch: 1, spec: spec
            ).asType(.bfloat16)
            MLX.eval(direct, established)
            XCTAssertEqual(direct.dtype, .bfloat16)
            XCTAssertEqual(
                direct.asData().data,
                established.asData().data,
                "Direct BF16 expansion changed the established rounded bytes for \(spec.chroma.rawValue)."
            )
        }
    }

    func testPackedUInt8Float16RequestRetainsEstablishedFloat32Expansion() {
        let spec = PreprocessingSpec(
            width: 16, height: 16, colorMode: .grayscale,
            bitDepth: 8, chroma: .yuv444
        )
        let packed = Data((0...255).map(UInt8.init))
        let requested = VisionPreprocessor.mlxTensor(
            packed, batch: 1, spec: spec, dtype: .float16
        )
        let established = VisionPreprocessor.mlxTensor(
            packed, batch: 1, spec: spec
        )
        MLX.eval(requested, established)
        XCTAssertEqual(
            requested.dtype,
            .float32,
            "FP16 must keep Float32 normalization until its different rounding is independently quality-gated."
        )
        XCTAssertEqual(requested.asData().data, established.asData().data)
    }

    func testFolderSelectionAndRecursiveDeletion() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("workspace-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()
        let folder = RecordingFolder(id: UUID(), name: "Game", createdAt: Date())
        try await store.saveRecordingFolder(folder)
        let id = UUID(), directory = try await store.createRecordingDirectory(id: id)
        let manifest = RecordingManifest(id: id, name: "Example", createdAt: Date(), hostStartNanos: 1, duration: 2, capture: CaptureSpec(), globalRect: CodableRect(.zero), pixelWidth: 16, pixelHeight: 9, deliveredFPS: 60, eventCount: 3, folderID: folder.id)
        try await store.writeRecording(manifest, to: directory)
        let before = await store.listRecordings()
        XCTAssertEqual(before.first?.manifest.folderID, folder.id)
        XCTAssertGreaterThan(before.first?.storageBytes ?? 0, 0)
        try await store.deleteRecordingFolder(folder, includingRecordings: true)
        let recordingsAfter = await store.listRecordings()
        let foldersAfter = await store.listRecordingFolders()
        XCTAssertTrue(recordingsAfter.isEmpty)
        XCTAssertTrue(foldersAfter.isEmpty)
    }

    func testBulkRecordingMoveAndDeletionAreAllOrNothingAtTheLibraryBoundary() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("workspace-bulk-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()
        let source = RecordingFolder(id: UUID(), name: "Source", createdAt: Date())
        let destination = RecordingFolder(id: UUID(), name: "Destination", createdAt: Date().addingTimeInterval(1))
        try await store.saveRecordingFolder(source)
        try await store.saveRecordingFolder(destination)
        var created: [RecordingItem] = []
        for index in 0..<3 {
            let id = UUID()
            let directory = try await store.createRecordingDirectory(id: id)
            let manifest = RecordingManifest(
                id: id, name: "Recording \(index)", createdAt: Date(), hostStartNanos: 1, duration: 1,
                capture: CaptureSpec(), globalRect: CodableRect(CGRect(x: 0, y: 0, width: 100, height: 100)),
                pixelWidth: 100, pixelHeight: 100, deliveredFPS: 60, eventCount: 0, folderID: source.id
            )
            try await store.writeRecording(manifest, to: directory)
            created.append(RecordingItem(manifest: manifest, directory: directory))
        }
        try await store.assignRecordings(created, to: destination.id)
        let moved = await store.listRecordings()
        XCTAssertEqual(Set(moved.compactMap { $0.manifest.folderID }), [destination.id])
        try await store.deleteRecordings(moved)
        let remaining = await store.listRecordings()
        XCTAssertTrue(remaining.isEmpty)
    }

    func testRecordingExportAndImportPreserveNativeArtifactsFoldersAndCollisionSafety() async throws {
        let container = FileManager.default.temporaryDirectory.appendingPathComponent("recording-transfer-\(UUID().uuidString)", isDirectory: true)
        let sourceRoot = container.appendingPathComponent("Source", isDirectory: true)
        let targetRoot = container.appendingPathComponent("Target", isDirectory: true)
        let exportRoot = container.appendingPathComponent("Recording Export", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: container) }

        let source = WorkspaceStore(root: sourceRoot)
        try await source.prepare()
        let folder = RecordingFolder(id: UUID(), name: "Portable Demonstrations", createdAt: Date())
        try await source.saveRecordingFolder(folder)
        let recordingID = UUID()
        let recordingDirectory = try await source.createRecordingDirectory(id: recordingID)
        let videoURL = recordingDirectory.appendingPathComponent("capture.mov")
        let eventURL = recordingDirectory.appendingPathComponent("events.atrevents")
        try await writeTestMovie(to: videoURL, width: 16, height: 16, frameCount: 6, fps: 30)
        let eventWriter = try InputEventWriter(url: eventURL)
        eventWriter.append(InputSample(timestampNanos: 1_000_000_000, kind: .mouseMove, x: 8, y: 8, deltaX: 2, deltaY: -1))
        eventWriter.append(InputSample(timestampNanos: 1_010_000_000, kind: .key, keyCode: 13, isDown: true))
        eventWriter.append(InputSample(timestampNanos: 1_020_000_000, kind: .key, keyCode: 13, isDown: false))
        let eventCount = try eventWriter.finish()
        let manifest = RecordingManifest(
            id: recordingID,
            name: "Portable",
            createdAt: Date(),
            hostStartNanos: 1_000_000_000,
            duration: 0.2,
            capture: CaptureSpec(requestedFPS: 30),
            globalRect: CodableRect(CGRect(x: 0, y: 0, width: 16, height: 16)),
            pixelWidth: 16,
            pixelHeight: 16,
            deliveredFPS: 30,
            eventCount: eventCount,
            folderID: folder.id,
            excludedKeyCodes: [49]
        )
        try await source.writeRecording(manifest, to: recordingDirectory)
        let sourceVideo = try Data(contentsOf: videoURL)
        let sourceEvents = try Data(contentsOf: eventURL)
        let sourceItems = await source.listRecordings()

        let exported = try await source.exportRecordings(sourceItems, to: exportRoot)
        XCTAssertEqual(exported.exportedCount, 1)
        XCTAssertTrue(FileManager.default.fileExists(atPath: exportRoot.appendingPathComponent("recording-folders.json").path))

        let target = WorkspaceStore(root: targetRoot)
        try await target.prepare()
        let firstImport = try await target.importRecordings(from: [exportRoot], fallbackFolderID: nil)
        XCTAssertEqual(firstImport, RecordingImportResult(importedCount: 1, createdFolderCount: 1, regeneratedIdentifierCount: 0))
        var imported = await target.listRecordings()
        XCTAssertEqual(imported.map(\.id), [recordingID])
        XCTAssertEqual(imported.first?.manifest.excludedKeyCodes, [49])
        XCTAssertEqual(try Data(contentsOf: try XCTUnwrap(imported.first).directory.appendingPathComponent("capture.mov")), sourceVideo)
        XCTAssertEqual(try Data(contentsOf: try XCTUnwrap(imported.first).directory.appendingPathComponent("events.atrevents")), sourceEvents)
        let importedFolders = await target.listRecordingFolders()
        XCTAssertEqual(importedFolders.map(\.name), [folder.name])

        let secondImport = try await target.importRecordings(from: [exportRoot], fallbackFolderID: nil)
        XCTAssertEqual(secondImport.importedCount, 1)
        XCTAssertEqual(secondImport.createdFolderCount, 0)
        XCTAssertEqual(secondImport.regeneratedIdentifierCount, 1)
        imported = await target.listRecordings()
        XCTAssertEqual(imported.count, 2)
        XCTAssertEqual(Set(imported.compactMap { $0.manifest.folderID }).count, 1)
        XCTAssertEqual(Set(imported.map(\.id)).count, 2)
        XCTAssertTrue(imported.allSatisfy {
            (try? Data(contentsOf: $0.directory.appendingPathComponent("capture.mov"))) == sourceVideo
                && (try? Data(contentsOf: $0.directory.appendingPathComponent("events.atrevents"))) == sourceEvents
        })
    }

    func testRecordingImportRejectsCorruptInputBeforeChangingTheLibrary() async throws {
        let container = FileManager.default.temporaryDirectory.appendingPathComponent("recording-import-reject-\(UUID().uuidString)", isDirectory: true)
        let package = container.appendingPathComponent("Broken.atrrecord", isDirectory: true)
        let targetRoot = container.appendingPathComponent("Target", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: container) }
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        try await writeTestMovie(to: package.appendingPathComponent("capture.mov"), width: 16, height: 16, frameCount: 3, fps: 30)
        try Data("ATREVT01".utf8).write(to: package.appendingPathComponent("events.atrevents"))
        let manifest = RecordingManifest(
            id: UUID(), name: "Broken", createdAt: Date(), hostStartNanos: 1, duration: 0.1,
            capture: CaptureSpec(requestedFPS: 30), globalRect: CodableRect(CGRect(x: 0, y: 0, width: 16, height: 16)),
            pixelWidth: 16, pixelHeight: 16, deliveredFPS: 30, eventCount: 0
        )
        let manifestEncoder = JSONEncoder(); manifestEncoder.dateEncodingStrategy = .iso8601
        try manifestEncoder.encode(manifest).write(to: package.appendingPathComponent("manifest.json"))

        let target = WorkspaceStore(root: targetRoot)
        try await target.prepare()
        do {
            _ = try await target.importRecordings(from: [package], fallbackFolderID: nil)
            XCTFail("A truncated event header must be rejected")
        } catch {
            XCTAssertTrue(error.localizedDescription.localizedCaseInsensitiveContains("input"))
        }
        let remainingRecordings = await target.listRecordings()
        let remainingFolders = await target.listRecordingFolders()
        XCTAssertTrue(remainingRecordings.isEmpty)
        XCTAssertTrue(remainingFolders.isEmpty)
    }

    func testLegacyInvalidRecordingIsRecoveredBeforeStrictLibraryScanWithoutTouchingSources() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("recording-recovery-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()
        let id = UUID()
        let directory = try await store.createRecordingDirectory(id: id)
        let videoURL = directory.appendingPathComponent("capture.mov")
        let eventsURL = directory.appendingPathComponent("events.atrevents")
        try await writeTestMovie(to: videoURL, width: 16, height: 16, frameCount: 6, fps: 30)
        let writer = try InputEventWriter(url: eventsURL)
        writer.append(InputSample(timestampNanos: 1_000_000_000, kind: .key, keyCode: 13, isDown: true))
        _ = try writer.finish()
        let videoBefore = try Data(contentsOf: videoURL)
        let eventsBefore = try Data(contentsOf: eventsURL)

        var manifest = RecordingManifest(
            id: id, name: "Legacy", createdAt: Date(), hostStartNanos: 1_000_000_000,
            duration: 0.1, capture: CaptureSpec(requestedFPS: 30),
            globalRect: CodableRect(CGRect(x: 0, y: 0, width: 16, height: 16)),
            pixelWidth: 16, pixelHeight: 16, deliveredFPS: 30, eventCount: 1
        )
        manifest.trimEnd = 0.2 // Valid in the video but beyond the stale manifest duration.
        try await store.writeRecording(manifest, to: directory)
        let originalManifest = try Data(contentsOf: directory.appendingPathComponent("manifest.json"))
        let hiddenBeforeRepair = await store.listRecordings()
        XCTAssertTrue(hiddenBeforeRepair.isEmpty)

        let repairedCount = try await store.repairInvalidRecordingManifests()
        XCTAssertEqual(repairedCount, 1)
        let recovered = await store.listRecordings()
        XCTAssertEqual(recovered.count, 1)
        XCTAssertTrue(recovered[0].manifest.isStructurallyValid)
        XCTAssertGreaterThan(recovered[0].manifest.duration, 0.1)
        XCTAssertLessThanOrEqual(try XCTUnwrap(recovered[0].manifest.trimEnd), recovered[0].manifest.duration)
        XCTAssertEqual(try Data(contentsOf: videoURL), videoBefore)
        XCTAssertEqual(try Data(contentsOf: eventsURL), eventsBefore)
        XCTAssertEqual(try Data(contentsOf: directory.appendingPathComponent("manifest.pre-1.8.1-recovery.json")), originalManifest)
        let secondRepairCount = try await store.repairInvalidRecordingManifests()
        XCTAssertEqual(secondRepairCount, 0)
    }

    func testLegacyRecordingsAreNormalizedIntoRealFolders() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("workspace-normalize-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()
        let id = UUID(), directory = try await store.createRecordingDirectory(id: id)
        let manifest = RecordingManifest(id: id, name: "Legacy", createdAt: Date(), hostStartNanos: 1, duration: 1, capture: CaptureSpec(), globalRect: CodableRect(.zero), pixelWidth: 8, pixelHeight: 8, deliveredFPS: 30, eventCount: 0)
        try await store.writeRecording(manifest, to: directory)

        let folderID = try await store.normalizeRecordingFolders()
        let folders = await store.listRecordingFolders()
        let recordings = await store.listRecordings()
        XCTAssertEqual(folders.count, 1)
        XCTAssertEqual(recordings.first?.manifest.folderID, folderID)
    }

    func testTrainingDataAndModelLibrariesRelocateIndependently() async throws {
        let container = FileManager.default.temporaryDirectory.appendingPathComponent("workspace-relocation-\(UUID().uuidString)", isDirectory: true)
        let original = container.appendingPathComponent("Original", isDirectory: true)
        let trainingDestination = container.appendingPathComponent("Training", isDirectory: true)
        let modelDestination = container.appendingPathComponent("Models", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: container) }

        let store = WorkspaceStore(root: original)
        try await store.prepare()
        let folder = RecordingFolder(id: UUID(), name: "Demonstrations", createdAt: Date())
        try await store.saveRecordingFolder(folder)
        let recordingID = UUID()
        let recordingDirectory = try await store.createRecordingDirectory(id: recordingID)
        let recording = RecordingManifest(id: recordingID, name: "Move me", createdAt: Date(), hostStartNanos: 1, duration: 1, capture: CaptureSpec(), globalRect: CodableRect(.zero), pixelWidth: 8, pixelHeight: 8, deliveredFPS: 30, eventCount: 0, folderID: folder.id)
        try await store.writeRecording(recording, to: recordingDirectory)
        try Data([1, 2, 3]).write(to: recordingDirectory.appendingPathComponent("capture.mov"))

        var profile = AIProfile.fresh(name: "Move this brain")
        let versionID = UUID()
        profile.activeVersionID = versionID
        try await store.saveProfile(profile)
        let version = ModelVersionManifest(id: versionID, name: "Brain", createdAt: Date(), globalStep: 7, trainingLoss: 0.25, preprocessing: profile.preprocessing, channels: profile.channels, training: profile.training)
        try await store.saveVersionManifest(version, profileID: profile.id)
        let versionDirectory = await store.versionDirectory(profileID: profile.id, versionID: versionID)
        try Data([4, 5, 6]).write(to: versionDirectory.appendingPathComponent(version.weightsFile))

        let trainingInspection = try await store.inspectDestination(trainingDestination, for: .trainingData)
        XCTAssertFalse(trainingInspection.containsManagedData)
        let trainingMove = try await store.relocate(.trainingData, to: trainingDestination, useExisting: false)
        XCTAssertTrue(trainingMove.movedExistingData)
        XCTAssertTrue(trainingMove.sourceCleanupComplete)
        var locations = await store.locations()
        XCTAssertEqual(locations.trainingDataRoot.path, trainingDestination.standardizedFileURL.path)
        XCTAssertEqual(locations.modelsRoot.path, original.standardizedFileURL.path)
        let recordingsAfterTrainingMove = await store.listRecordings()
        let profilesAfterTrainingMove = await store.listProfiles()
        XCTAssertEqual(recordingsAfterTrainingMove.map(\.id), [recordingID])
        XCTAssertEqual(profilesAfterTrainingMove.map(\.id), [profile.id])
        XCTAssertFalse(FileManager.default.fileExists(atPath: original.appendingPathComponent("Recordings").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: original.appendingPathComponent("Profiles").path))

        let modelMove = try await store.relocate(.models, to: modelDestination, useExisting: false)
        XCTAssertTrue(modelMove.movedExistingData)
        XCTAssertTrue(modelMove.sourceCleanupComplete)
        locations = await store.locations()
        XCTAssertEqual(locations.trainingDataRoot.path, trainingDestination.standardizedFileURL.path)
        XCTAssertEqual(locations.modelsRoot.path, modelDestination.standardizedFileURL.path)
        let versionsAfterMove = await store.listVersions(profileID: profile.id)
        XCTAssertEqual(versionsAfterMove.map(\.id), [versionID])
        XCTAssertFalse(FileManager.default.fileExists(atPath: original.appendingPathComponent("Profiles").path))
        let usage = await store.storageUsage()
        XCTAssertGreaterThan(usage.trainingDataBytes, 0)
        XCTAssertGreaterThan(usage.modelBytes, 0)
        XCTAssertGreaterThanOrEqual(usage.totalBytes, usage.trainingDataBytes + usage.modelBytes)
    }

    func testSwitchingToExistingLibraryNeverMergesOrDeletesCurrentData() async throws {
        let container = FileManager.default.temporaryDirectory.appendingPathComponent("workspace-switch-\(UUID().uuidString)", isDirectory: true)
        let currentRoot = container.appendingPathComponent("Current", isDirectory: true)
        let existingRoot = container.appendingPathComponent("Existing", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: container) }

        let current = WorkspaceStore(root: currentRoot)
        let existing = WorkspaceStore(root: existingRoot)
        try await current.prepare()
        try await existing.prepare()

        let currentID = UUID()
        let currentDirectory = try await current.createRecordingDirectory(id: currentID)
        try await current.writeRecording(
            RecordingManifest(id: currentID, name: "Current", createdAt: Date(), hostStartNanos: 1, duration: 1, capture: CaptureSpec(), globalRect: CodableRect(.zero), pixelWidth: 8, pixelHeight: 8, deliveredFPS: 30, eventCount: 0),
            to: currentDirectory
        )
        let existingID = UUID()
        let existingDirectory = try await existing.createRecordingDirectory(id: existingID)
        try await existing.writeRecording(
            RecordingManifest(id: existingID, name: "Existing", createdAt: Date(), hostStartNanos: 1, duration: 1, capture: CaptureSpec(), globalRect: CodableRect(.zero), pixelWidth: 8, pixelHeight: 8, deliveredFPS: 30, eventCount: 0),
            to: existingDirectory
        )

        let inspection = try await current.inspectDestination(existingRoot, for: .trainingData)
        XCTAssertTrue(inspection.containsManagedData)
        do {
            _ = try await current.relocate(.trainingData, to: existingRoot, useExisting: false)
            XCTFail("Populated libraries must not be merged implicitly")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("already contains"))
        }

        let switched = try await current.relocate(.trainingData, to: existingRoot, useExisting: true)
        XCTAssertFalse(switched.movedExistingData)
        let switchedRecordings = await current.listRecordings()
        XCTAssertEqual(switchedRecordings.map(\.id), [existingID])
        XCTAssertTrue(FileManager.default.fileExists(atPath: currentDirectory.path), "Switching libraries must leave the previous library intact")
    }

    func testModelContractMigrationPreservesProfilesAndRecordings() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("workspace-contract-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()

        var profile = AIProfile.fresh(name: "Preserved profile")
        profile.activeVersionID = UUID()
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 20, frameSpacing: 3, downsampleFactor: 3)
        profile.training.learningRate = 0.005
        profile.training.architecture = ArchitectureSpec(convolutionChannels: [64, 128, 256], visualEmbedding: 512, recurrentWidth: 384, fusionWidths: [768, 512])
        try await store.saveProfile(profile)
        let profileDirectory = await store.profileDirectory(profile.id)
        let versions = profileDirectory.appendingPathComponent("Versions", isDirectory: true)
        let checkpoint = await store.checkpointDirectory(profileID: profile.id)
        try FileManager.default.createDirectory(at: versions, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: checkpoint, withIntermediateDirectories: true)
        try Data([1]).write(to: versions.appendingPathComponent("old.bin"))
        try Data([2]).write(to: checkpoint.appendingPathComponent("old.bin"))

        var protected = AIProfile.fresh(name: "Crystal V4")
        protected.activeVersionID = UUID()
        try await store.saveProfile(protected)
        let protectedRoot = await store.profileDirectory(protected.id)
        let protectedVersions = protectedRoot.appendingPathComponent("Versions", isDirectory: true)
        let protectedCheckpoint = await store.checkpointDirectory(profileID: protected.id)
        try FileManager.default.createDirectory(at: protectedVersions, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: protectedCheckpoint, withIntermediateDirectories: true)
        try Data([3]).write(to: protectedVersions.appendingPathComponent("old.bin"))
        try Data([4]).write(to: protectedCheckpoint.appendingPathComponent("old.bin"))

        let recordingID = UUID()
        let recordingDirectory = try await store.createRecordingDirectory(id: recordingID)
        let recording = RecordingManifest(id: recordingID, name: "Keep me", createdAt: Date(), hostStartNanos: 1, duration: 1, capture: CaptureSpec(), globalRect: CodableRect(.zero), pixelWidth: 8, pixelHeight: 8, deliveredFPS: 30, eventCount: 0)
        try await store.writeRecording(recording, to: recordingDirectory)

        let archived = try await store.removeObsoleteModelArtifacts(currentSchema: ModelContract.schemaVersion)
        let remainingRecordings = await store.listRecordings()
        let remainingProfiles = await store.listProfiles()
        XCTAssertEqual(archived, 4)
        XCTAssertTrue(FileManager.default.fileExists(atPath: versions.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: versions.appendingPathComponent("old.bin").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: checkpoint.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: protectedVersions.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: protectedVersions.appendingPathComponent("old.bin").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: protectedCheckpoint.path))
        let archivedPaths = [profileDirectory, protectedRoot].flatMap { directory in
            let archive = directory.appendingPathComponent("Archived Model Artifacts", isDirectory: true)
            let enumerator = FileManager.default.enumerator(at: archive, includingPropertiesForKeys: nil)
            return (enumerator?.allObjects as? [URL] ?? []).map(\.lastPathComponent)
        }
        XCTAssertEqual(archivedPaths.filter { $0 == "old.bin" }.count, 4)
        XCTAssertEqual(archivedPaths.filter { $0 == "Checkpoint" }.count, 2)
        XCTAssertEqual(remainingRecordings.count, 1)
        XCTAssertEqual(Set(remainingProfiles.map(\.name)), ["Preserved profile", "Crystal V4"])
        XCTAssertTrue(remainingProfiles.allSatisfy { $0.activeVersionID == nil })
        let migrated = try XCTUnwrap(remainingProfiles.first { $0.name == "Preserved profile" })
        XCTAssertEqual(migrated.training.architecture, .large)
        XCTAssertEqual(migrated.training.effectiveTemporalVision, TemporalVisionConfiguration())
        XCTAssertEqual(migrated.training.learningRate, 0.0003)
        let secondPass = try await store.removeObsoleteModelArtifacts(currentSchema: ModelContract.schemaVersion)
        XCTAssertEqual(secondPass, 0)
    }

    func testModelContractAuditPreservesCompatibleBrainsInMixedCurrentLibrary() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("workspace-contract-audit-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()

        var profile = AIProfile.fresh(name: "Current mixed library")
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 7, frameSpacing: 5, downsampleFactor: 3)
        profile.training.learningRate = 0.00025
        let currentID = UUID()
        profile.activeVersionID = currentID
        profile.trainingProgress = TrainingProgressSummary(globalStep: 321, epoch: 4, updatedAt: Date(), savedBrainCount: 2)
        try await store.saveProfile(profile)

        let current = ModelVersionManifest(
            id: currentID,
            name: "Supported brain",
            createdAt: Date(),
            globalStep: 321,
            trainingLoss: 0.2,
            preprocessing: profile.preprocessing,
            channels: profile.channels,
            training: profile.training
        )
        try await store.saveVersionManifest(current, profileID: profile.id)
        let currentDirectory = await store.versionDirectory(profileID: profile.id, versionID: currentID)
        let currentWeights = Data([9, 8, 7, 6])
        try currentWeights.write(to: currentDirectory.appendingPathComponent(current.weightsFile))

        var legacy = current
        legacy.schemaVersion = ModelContract.schemaVersion - 1
        legacy.id = UUID()
        legacy.name = "Legacy brain"
        try await store.saveVersionManifest(legacy, profileID: profile.id)
        let legacyDirectory = await store.versionDirectory(profileID: profile.id, versionID: legacy.id)
        try Data([1, 2, 3]).write(to: legacyDirectory.appendingPathComponent(legacy.weightsFile))

        let checkpoint = await store.checkpointDirectory(profileID: profile.id)
        try FileManager.default.createDirectory(at: checkpoint, withIntermediateDirectories: true)
        let checkpointWeights = Data([5, 4, 3, 2, 1])
        let checkpointState = Data("current checkpoint".utf8)
        try checkpointWeights.write(to: checkpoint.appendingPathComponent("weights.safetensors"))
        try checkpointState.write(to: checkpoint.appendingPathComponent("state.json"))
        try JSONEncoder().encode(ModelContract.schemaVersion).write(to: root.appendingPathComponent("model-contract.json"), options: .atomic)

        let archived = try await store.removeObsoleteModelArtifacts(currentSchema: ModelContract.schemaVersion)

        XCTAssertEqual(archived, 1)
        XCTAssertEqual(try Data(contentsOf: currentDirectory.appendingPathComponent(current.weightsFile)), currentWeights)
        XCTAssertEqual(try Data(contentsOf: checkpoint.appendingPathComponent("weights.safetensors")), checkpointWeights)
        XCTAssertEqual(try Data(contentsOf: checkpoint.appendingPathComponent("state.json")), checkpointState)
        XCTAssertEqual(try JSONDecoder().decode(Int.self, from: Data(contentsOf: checkpoint.appendingPathComponent("model-schema.json"))), ModelContract.schemaVersion)
        XCTAssertFalse(FileManager.default.fileExists(atPath: legacyDirectory.path))
        let reloadedProfiles = await store.listProfiles()
        let reloaded = try XCTUnwrap(reloadedProfiles.first { $0.id == profile.id })
        XCTAssertEqual(reloaded.activeVersionID, currentID)
        XCTAssertEqual(reloaded.trainingProgress?.globalStep, profile.trainingProgress?.globalStep)
        XCTAssertEqual(reloaded.trainingProgress?.epoch, profile.trainingProgress?.epoch)
        XCTAssertEqual(reloaded.trainingProgress?.savedBrainCount, profile.trainingProgress?.savedBrainCount)
        XCTAssertEqual(reloaded.training.effectiveTemporalVision, profile.training.effectiveTemporalVision)
        XCTAssertEqual(reloaded.training.learningRate, 0.00025)
        let secondPass = try await store.removeObsoleteModelArtifacts(currentSchema: ModelContract.schemaVersion)
        XCTAssertEqual(secondPass, 0)
    }

    func testProfilesAreNewestFirstAndExposeTrainingProgressWithoutLoadingEveryVersion() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("workspace-profile-order-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()
        var older = AIProfile.fresh(name: "Older")
        older.createdAt = Date(timeIntervalSince1970: 10)
        var newer = AIProfile.fresh(name: "Newer")
        newer.createdAt = Date(timeIntervalSince1970: 20)
        let active = UUID()
        newer.activeVersionID = active
        try await store.saveProfile(older)
        try await store.saveProfile(newer)
        let version = ModelVersionManifest(id: active, name: "Brain", createdAt: Date(timeIntervalSince1970: 30), globalStep: 1_234, trainingLoss: 0.1, validationLoss: nil, preprocessing: newer.preprocessing, channels: newer.channels, training: newer.training, epoch: 12, isAutosave: true)
        try await store.saveVersionManifest(version, profileID: newer.id)

        let profiles = await store.listProfiles()
        XCTAssertEqual(profiles.map(\.name), ["Newer", "Older"])
        XCTAssertEqual(profiles.first?.trainingProgress?.globalStep, 1_234)
        XCTAssertEqual(profiles.first?.trainingProgress?.epoch, 12)
        XCTAssertEqual(profiles.first?.trainingProgress?.savedBrainCount, 1)
    }

    func testDuplicateProfileCopiesBrainProgressVersionsAndCheckpoint() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("workspace-duplicate-brain-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()

        var original = AIProfile.fresh(name: "Learned AI")
        let versionID = UUID()
        original.activeVersionID = versionID
        original.trainingProgress = TrainingProgressSummary(globalStep: 498_765, epoch: 1_000, updatedAt: Date(), savedBrainCount: 1)
        try await store.saveProfile(original)
        let version = ModelVersionManifest(id: versionID, name: "Brain", createdAt: Date(), globalStep: 498_765, trainingLoss: 0.1, validationLoss: nil, preprocessing: original.preprocessing, channels: original.channels, training: original.training, epoch: 1_000, isAutosave: false)
        try await store.saveVersionManifest(version, profileID: original.id)
        let versionDirectory = await store.versionDirectory(profileID: original.id, versionID: versionID)
        try Data("learned-weights".utf8).write(to: versionDirectory.appendingPathComponent("weights.safetensors"))
        let checkpoint = await store.checkpointDirectory(profileID: original.id)
        try FileManager.default.createDirectory(at: checkpoint, withIntermediateDirectories: true)
        try Data("resume-state".utf8).write(to: checkpoint.appendingPathComponent("state.json"))

        let copy = try await store.duplicateProfile(original)
        XCTAssertNotEqual(copy.id, original.id)
        XCTAssertEqual(copy.activeVersionID, versionID)
        XCTAssertEqual(copy.trainingProgress?.globalStep, 498_765)
        XCTAssertEqual(copy.trainingProgress?.epoch, 1_000)
        XCTAssertFalse(copy.isDeletionProtected)
        let copiedVersions = await store.listVersions(profileID: copy.id)
        XCTAssertEqual(copiedVersions.map(\.id), [versionID])
        let copiedVersionDirectory = await store.versionDirectory(profileID: copy.id, versionID: versionID)
        XCTAssertEqual(try Data(contentsOf: copiedVersionDirectory.appendingPathComponent("weights.safetensors")), Data("learned-weights".utf8))
        let copiedCheckpoint = await store.checkpointDirectory(profileID: copy.id)
        XCTAssertEqual(try Data(contentsOf: copiedCheckpoint.appendingPathComponent("state.json")), Data("resume-state".utf8))
    }

    func testConfirmedArchitectureResetClearsLearningButNeverProtectedBrains() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("workspace-reset-brain-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()

        var regular = AIProfile.fresh(name: "Regular")
        regular.activeVersionID = UUID()
        regular.trainingProgress = TrainingProgressSummary(globalStep: 10_000, epoch: 100, updatedAt: Date(), savedBrainCount: 2)
        try await store.saveProfile(regular)
        let regularVersions = (await store.profileDirectory(regular.id)).appendingPathComponent("Versions", isDirectory: true)
        let regularCheckpoint = await store.checkpointDirectory(profileID: regular.id)
        try FileManager.default.createDirectory(at: regularVersions, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: regularCheckpoint, withIntermediateDirectories: true)
        regular.training.architecture = .large
        let reset = try await store.resetLearning(for: regular)
        XCTAssertNil(reset.activeVersionID)
        XCTAssertNil(reset.trainingProgress)
        XCTAssertFalse(FileManager.default.fileExists(atPath: regularVersions.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: regularCheckpoint.path))
        let profilesAfterReset = await store.listProfiles()
        XCTAssertEqual(profilesAfterReset.first(where: { $0.id == regular.id })?.training.architecture, .large)

        var crystal = AIProfile.fresh(name: "Crystal V4")
        crystal.activeVersionID = UUID()
        try await store.saveProfile(crystal)
        do {
            _ = try await store.resetLearning(for: crystal)
            XCTFail("Crystal V4 learning reset should be blocked")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("protected"))
        }
        let profilesAfterCrystalAttempt = await store.listProfiles()
        XCTAssertEqual(profilesAfterCrystalAttempt.first(where: { $0.id == crystal.id })?.activeVersionID, crystal.activeVersionID)

        var fineTuned = AIProfile.fresh(name: "Crystal V4 Fine-tuned + glass")
        fineTuned.activeVersionID = UUID()
        try await store.saveProfile(fineTuned)
        do {
            _ = try await store.resetLearning(for: fineTuned)
            XCTFail("The fine-tuned Crystal V4 learning reset should be blocked")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("protected"))
        }
        let fineTunedCopy = try await store.duplicateProfile(fineTuned)
        XCTAssertFalse(fineTunedCopy.isDeletionProtected)
    }

    func testEpochBlocksResumeUntilFinishedThenAddAnotherBlock() {
        XCTAssertEqual(TrainingContinuationPlan.targetEpoch(completedEpoch: 0, batchOffset: 0, savedTarget: nil, configuredIncrement: 1_000), 1_000)
        XCTAssertEqual(TrainingContinuationPlan.targetEpoch(completedEpoch: 500, batchOffset: 0, savedTarget: 1_000, configuredIncrement: 1_000), 1_000)
        XCTAssertEqual(TrainingContinuationPlan.targetEpoch(completedEpoch: 999, batchOffset: 64, savedTarget: 1_000, configuredIncrement: 1_000), 1_000)
        XCTAssertEqual(TrainingContinuationPlan.targetEpoch(completedEpoch: 1_000, batchOffset: 0, savedTarget: 1_000, configuredIncrement: 1_000), 2_000)
        XCTAssertEqual(TrainingContinuationPlan.targetEpoch(completedEpoch: 2_000, batchOffset: 0, savedTarget: 2_000, configuredIncrement: 250), 2_250)
    }

    func testExactVisionTemporalContextAndArchitectureChangeTheLearnedBrainContract() {
        var profile = AIProfile.fresh()
        let original = profile.learnedBrainContract
        profile.training.epochs = 9_999
        profile.training.learningRate = 0.00001
        profile.training.actionFPS = 120
        XCTAssertEqual(profile.learnedBrainContract, original)
        profile.training.architecture.dropout = 0.5
        XCTAssertEqual(profile.learnedBrainContract, original, "Dropout must preserve every learned tensor and existing brain weight.")
        profile.training.generalization = GeneralizationConfiguration(
            visionAugmentationStrength: 0.12,
            randomErasingProbability: 0.15,
            controlHistoryDropout: 0.18,
            temporalFrameDropout: 0.08,
            binaryLabelSmoothing: 0.01
        )
        XCTAssertEqual(profile.learnedBrainContract, original, "Training-only augmentation must preserve the learned tensor layout.")
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 8, frameSpacing: 2, downsampleFactor: 2)
        XCTAssertNotEqual(profile.learnedBrainContract, original)
        profile.training.temporalVision = original.temporalVision
        profile.training.architecture.controlEmbedding = profile.training.architecture.effectiveControlEmbedding + 1
        XCTAssertNotEqual(profile.learnedBrainContract, original)
        profile.training.architecture.controlEmbedding = original.architecture.controlEmbedding
        profile.training.architecture.recurrentWidth += 1
        XCTAssertNotEqual(profile.learnedBrainContract, original)
        profile.training.architecture.recurrentWidth -= 1
        profile.preprocessing.width += 1
        XCTAssertNotEqual(profile.learnedBrainContract, original)
    }

    func testAutosaveRetentionKeepsTenAndNeverTouchesCrystalV4() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("workspace-retention-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()

        var regular = AIProfile.fresh(name: "Regular")
        var crystal = AIProfile.fresh(name: "Crystal V4")
        crystal.name = "Renamed Protected Brain"
        for index in 0..<12 {
            let regularVersion = ModelVersionManifest(id: UUID(), name: "Autosave \(index)", createdAt: Date(timeIntervalSince1970: Double(index)), globalStep: index, trainingLoss: 0.2, validationLoss: nil, preprocessing: regular.preprocessing, channels: regular.channels, training: regular.training, epoch: index, isAutosave: true)
            let crystalVersion = ModelVersionManifest(id: UUID(), name: "Crystal \(index)", createdAt: Date(timeIntervalSince1970: Double(index)), globalStep: index, trainingLoss: 0.2, validationLoss: nil, preprocessing: crystal.preprocessing, channels: crystal.channels, training: crystal.training, epoch: index, isAutosave: true)
            try await store.saveVersionManifest(regularVersion, profileID: regular.id)
            try await store.saveVersionManifest(crystalVersion, profileID: crystal.id)
            if index == 11 { regular.activeVersionID = regularVersion.id; crystal.activeVersionID = crystalVersion.id }
        }
        try await store.saveProfile(regular)
        try await store.saveProfile(crystal)

        let regularRemoved = try await store.pruneAutosaveVersions(profile: regular, keeping: 10)
        let regularCount = await store.listVersions(profileID: regular.id).count
        let crystalRemoved = try await store.pruneAutosaveVersions(profile: crystal, keeping: 10)
        let crystalCount = await store.listVersions(profileID: crystal.id).count
        XCTAssertEqual(regularRemoved, 2)
        XCTAssertEqual(regularCount, 10)
        XCTAssertEqual(crystalRemoved, 0)
        XCTAssertEqual(crystalCount, 12)
        do {
            try await store.deleteProfile(crystal)
            XCTFail("Crystal V4 deletion should be blocked")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("protected"))
        }
    }

    func testFineTunedCrystalV4IsProtectedFromDeletionAndPruning() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("workspace-fine-tuned-protection-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()
        let profile = AIProfile.fresh(name: "Crystal V4 Fine-tuned + glass")
        XCTAssertTrue(profile.isDeletionProtected)
        for index in 0..<12 {
            let version = ModelVersionManifest(id: UUID(), name: "Fine-tuned \(index)", createdAt: Date(timeIntervalSince1970: Double(index)), globalStep: index, trainingLoss: 0.1, validationLoss: nil, preprocessing: profile.preprocessing, channels: profile.channels, training: profile.training, epoch: index, isAutosave: true)
            try await store.saveVersionManifest(version, profileID: profile.id)
        }
        try await store.saveProfile(profile)
        let removed = try await store.pruneAutosaveVersions(profile: profile, keeping: 10)
        XCTAssertEqual(removed, 0)
        let versionCount = await store.listVersions(profileID: profile.id).count
        XCTAssertEqual(versionCount, 12)
        do {
            try await store.deleteProfile(profile)
            XCTFail("The fine-tuned Crystal V4 should not be deletable")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("protected"))
        }
    }

    func testRunnableVersionRestoresExactTrainingCheckpoint() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("versions-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()
        let profile = AIProfile.fresh(), versionID = UUID()
        try await store.saveProfile(profile)
        let directory = await store.versionDirectory(profileID: profile.id, versionID: versionID)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("brain".utf8).write(to: directory.appendingPathComponent("weights.safetensors"))
        try Data("optimizer".utf8).write(to: directory.appendingPathComponent("optimizer.safetensors"))
        try Data("state".utf8).write(to: directory.appendingPathComponent("state.json"))
        let version = ModelVersionManifest(id: versionID, name: "Epoch 120", createdAt: Date(), globalStep: 240, trainingLoss: 0.2, validationLoss: nil, preprocessing: profile.preprocessing, channels: profile.channels, training: profile.training, optimizerFile: "optimizer.safetensors", trainingStateFile: "state.json", epoch: 120, isAutosave: true)
        try await store.saveVersionManifest(version, profileID: profile.id)
        let restored = try await store.activateVersion(profile: profile, versionID: version.id)
        XCTAssertTrue(restored.checkpointIsResumable)
        XCTAssertEqual(restored.profile.activeVersionID, version.id)
        XCTAssertEqual(restored.profile.trainingProgress?.globalStep, 240)
        let checkpoint = await store.checkpointDirectory(profileID: profile.id)
        XCTAssertEqual(try Data(contentsOf: checkpoint.appendingPathComponent("weights.safetensors")), Data("brain".utf8))
        XCTAssertEqual(try Data(contentsOf: checkpoint.appendingPathComponent("optimizer.safetensors")), Data("optimizer".utf8))
        XCTAssertEqual(try Data(contentsOf: checkpoint.appendingPathComponent("state.json")), Data("state".utf8))

        let best = ModelVersionManifest(id: UUID(), name: "Best", createdAt: Date(), globalStep: 120, trainingLoss: 0.1, validationLoss: 0.1, preprocessing: profile.preprocessing, channels: profile.channels, training: profile.training)
        try await store.saveVersionManifest(best, profileID: profile.id)
        let bestDirectory = await store.versionDirectory(profileID: profile.id, versionID: best.id)
        try Data("best-brain".utf8).write(to: bestDirectory.appendingPathComponent(best.weightsFile))
        let staleRestored = try await store.activateVersion(profile: restored.profile, versionID: best.id)
        XCTAssertFalse(staleRestored.checkpointIsResumable)
        XCTAssertEqual(staleRestored.profile.activeVersionID, best.id)
        XCTAssertFalse(FileManager.default.fileExists(atPath: checkpoint.path), "A stale newer checkpoint must not override an explicitly activated weights-only brain")
    }

    func testModelVersionManifestsCannotEscapeTheirVersionDirectory() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("unsafe-version-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()
        let profile = AIProfile.fresh()
        var version = ModelVersionManifest(id: UUID(), name: "Unsafe", createdAt: Date(), globalStep: 1, trainingLoss: 0.1, preprocessing: profile.preprocessing, channels: profile.channels, training: profile.training)
        XCTAssertTrue(version.artifactFileNamesAreSafe)
        version.weightsFile = "../outside.safetensors"
        XCTAssertFalse(version.artifactFileNamesAreSafe)
        await XCTAssertThrowsErrorAsync {
            try await store.saveVersionManifest(version, profileID: profile.id)
        }
    }

    func testProfileAndFolderNamesAreTrimmedAndCannotBeEmpty() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("names-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()

        var profile = AIProfile.fresh(name: "  Polished AI  ")
        try await store.saveProfile(profile)
        let storedProfiles = await store.listProfiles()
        XCTAssertEqual(storedProfiles.first?.name, "Polished AI")
        profile.id = UUID()
        profile.name = "   "
        await XCTAssertThrowsErrorAsync { try await store.saveProfile(profile) }

        let folder = RecordingFolder(id: UUID(), name: "  Sessions  ", createdAt: Date())
        try await store.saveRecordingFolder(folder)
        let storedFolders = await store.listRecordingFolders()
        XCTAssertEqual(storedFolders.first?.name, "Sessions")
        await XCTAssertThrowsErrorAsync {
            try await store.saveRecordingFolder(RecordingFolder(id: UUID(), name: "\n\t", createdAt: Date()))
        }
    }

    func testConfiguredHotkeyIsRemovedFromCapturedInput() {
        var filter = HotkeyInputFilter(bindings: [.record])
        let control = CGEventFlags.maskControl.rawValue
        let controlOption = control | CGEventFlags.maskAlternate.rawValue
        let required = HotkeyBinding.record.cgEventModifiers
        let samples = [
            InputSample(timestampNanos: 1, kind: .flags, modifiers: control),
            InputSample(timestampNanos: 2, kind: .flags, modifiers: controlOption),
            InputSample(timestampNanos: 3, kind: .flags, modifiers: required),
            InputSample(timestampNanos: 4, kind: .key, keyCode: UInt16(HotkeyBinding.record.keyCode), modifiers: required, isDown: true),
            InputSample(timestampNanos: 5, kind: .key, keyCode: UInt16(HotkeyBinding.record.keyCode), modifiers: required, isDown: false),
            InputSample(timestampNanos: 6, kind: .flags, modifiers: controlOption),
            InputSample(timestampNanos: 7, kind: .flags, modifiers: control),
            InputSample(timestampNanos: 8, kind: .flags, modifiers: 0)
        ]
        XCTAssertTrue(samples.flatMap { filter.process($0) }.isEmpty)

        let normalFlags = InputSample(timestampNanos: 9, kind: .flags, modifiers: control)
        let normalKey = InputSample(timestampNanos: 10, kind: .key, keyCode: 0, modifiers: control, isDown: true)
        XCTAssertTrue(filter.process(normalFlags).isEmpty)
        XCTAssertEqual(filter.process(normalKey), [normalFlags, normalKey])
    }

    func testMouseSideButtonHotkeyAndItsModifiersAreRemovedFromCapture() {
        let control = UInt32(1 << 12)
        let binding = HotkeyBinding.mouse(4, carbonModifiers: control)
        var filter = HotkeyInputFilter(bindings: [binding])
        let flags = CGEventFlags.maskControl.rawValue
        let samples = [
            InputSample(timestampNanos: 1, kind: .flags, modifiers: flags),
            InputSample(timestampNanos: 2, kind: .mouseButton, button: 4, modifiers: flags, isDown: true),
            InputSample(timestampNanos: 3, kind: .mouseButton, button: 4, modifiers: flags, isDown: false),
            InputSample(timestampNanos: 4, kind: .flags, modifiers: 0)
        ]
        XCTAssertTrue(samples.flatMap { filter.process($0) }.isEmpty)
        XCTAssertEqual(binding.displayName, "Mouse 5")
    }

    func testGlobalShortcutSuppressionEndsWithItsPhysicalLifecycleAndPreservesForeignInput() {
        let suppression = HotkeySuppression()
        let binding = HotkeyBinding.record
        let shortcutModifiers = binding.cgEventModifiers
        let shift = CGEventFlags.maskShift.rawValue
        let trigger = UInt16(binding.keyCode)

        suppression.activate(binding)
        XCTAssertEqual(
            suppression.process(InputSample(timestampNanos: 1, kind: .key, keyCode: trigger, modifiers: shortcutModifiers, isDown: true)),
            .suppress(binding)
        )
        let foreignFlags = suppression.process(InputSample(
            timestampNanos: 2,
            kind: .flags,
            keyCode: 56,
            modifiers: shortcutModifiers | shift,
            isDown: true
        ))
        guard case let .pass(sanitizedFlags) = foreignFlags else {
            return XCTFail("Shift must survive shortcut-modifier suppression")
        }
        XCTAssertEqual(sanitizedFlags.modifiers & HotkeyBinding.cgModifierMask, shift)

        let foreignKey = suppression.process(InputSample(
            timestampNanos: 3,
            kind: .key,
            keyCode: 0,
            modifiers: shortcutModifiers | shift,
            isDown: true
        ))
        guard case let .pass(sanitizedKey) = foreignKey else {
            return XCTFail("An unrelated key must not be suppressed")
        }
        XCTAssertEqual(sanitizedKey.modifiers & HotkeyBinding.cgModifierMask, shift)

        XCTAssertEqual(
            suppression.process(InputSample(timestampNanos: 4, kind: .key, keyCode: trigger, modifiers: shortcutModifiers | shift, isDown: false)),
            .suppress(binding)
        )
        guard case let .pass(remainingShift) = suppression.process(InputSample(
            timestampNanos: 5,
            kind: .flags,
            keyCode: 55,
            modifiers: shift
        )) else {
            return XCTFail("The foreign held modifier must be preserved")
        }
        XCTAssertEqual(remainingShift.modifiers & HotkeyBinding.cgModifierMask, shift)

        // The shortcut is now fully released. A later standalone Shift release
        // occurs inside the old one-second window but must pass untouched.
        let shiftRelease = InputSample(timestampNanos: 6, kind: .flags, keyCode: 56, modifiers: 0)
        XCTAssertEqual(suppression.process(shiftRelease), .pass(shiftRelease))

        let mouseSuppression = HotkeySuppression()
        let mouseBinding = HotkeyBinding.mouse(15)
        mouseSuppression.activate(mouseBinding)
        XCTAssertEqual(mouseSuppression.process(InputSample(timestampNanos: 7, kind: .mouseButton, button: 15, isDown: true)), .suppress(mouseBinding))
        XCTAssertEqual(mouseSuppression.process(InputSample(timestampNanos: 8, kind: .mouseButton, button: 15, isDown: false)), .suppress(mouseBinding))
        XCTAssertEqual(mouseSuppression.process(shiftRelease), .pass(shiftRelease))
    }

    func testShortcutFilteringPreservesASeparatelyHeldModifierAcrossTheShortcut() {
        let binding = HotkeyBinding.record
        let shortcut = binding.cgEventModifiers
        let shift = CGEventFlags.maskShift.rawValue
        var filter = HotkeyInputFilter(bindings: [binding])
        let shiftDown = InputSample(timestampNanos: 1, kind: .flags, keyCode: 56, modifiers: shift, isDown: true)
        XCTAssertTrue(filter.process(shiftDown).isEmpty)
        XCTAssertTrue(filter.process(InputSample(timestampNanos: 2, kind: .flags, modifiers: shift | CGEventFlags.maskControl.rawValue)).isEmpty)
        XCTAssertTrue(filter.process(InputSample(timestampNanos: 3, kind: .flags, modifiers: shift | shortcut)).isEmpty)

        let triggerDown = InputSample(
            timestampNanos: 4,
            kind: .key,
            keyCode: UInt16(binding.keyCode),
            modifiers: shift | shortcut,
            isDown: true
        )
        let preserved = filter.processGloballySuppressed(triggerDown, shortcut: binding)
        XCTAssertEqual(preserved.count, 1)
        XCTAssertEqual(preserved[0].timestampNanos, shiftDown.timestampNanos)
        XCTAssertEqual(preserved[0].modifiers & HotkeyBinding.cgModifierMask, shift)

        let triggerUp = InputSample(
            timestampNanos: 5,
            kind: .key,
            keyCode: UInt16(binding.keyCode),
            modifiers: shift | shortcut,
            isDown: false
        )
        XCTAssertTrue(filter.processGloballySuppressed(triggerUp, shortcut: binding).isEmpty)
        XCTAssertTrue(filter.processGloballySuppressed(
            InputSample(timestampNanos: 6, kind: .flags, modifiers: shift),
            shortcut: binding
        ).isEmpty)
        let shiftUp = InputSample(timestampNanos: 7, kind: .flags, keyCode: 56, modifiers: 0, isDown: false)
        XCTAssertEqual(filter.process(shiftUp), [shiftUp])
    }

    func testImmediateInputStateTracksEveryKeyModifierAndMouseControl() {
        var timestamp: UInt64 = 1
        var tracker = InputStateTracker()
        let modifierCodes: Set<UInt16> = PhysicalInputSnapshot.modifierKeyCodes

        for keyCode in UInt16(0)..<128 where !modifierCodes.contains(keyCode) {
            XCTAssertTrue(tracker.consume(InputSample(timestampNanos: timestamp, kind: .key, keyCode: keyCode, isDown: true)))
            XCTAssertTrue(tracker.state.keys.contains(keyCode), "Key \(keyCode) did not enter held state")
            timestamp += 1
            XCTAssertTrue(tracker.consume(InputSample(timestampNanos: timestamp, kind: .key, keyCode: keyCode, isDown: false)))
            XCTAssertFalse(tracker.state.keys.contains(keyCode), "Key \(keyCode) remained held")
            timestamp += 1
        }

        let modifierPairs: [([UInt16], UInt64)] = [
            ([56, 60], CGEventFlags.maskShift.rawValue),
            ([59, 62], CGEventFlags.maskControl.rawValue),
            ([58, 61], CGEventFlags.maskAlternate.rawValue),
            ([55, 54], CGEventFlags.maskCommand.rawValue)
        ]
        for (codes, mask) in modifierPairs {
            XCTAssertTrue(tracker.consume(InputSample(timestampNanos: timestamp, kind: .flags, keyCode: codes[0], modifiers: mask, isDown: true)))
            XCTAssertTrue(tracker.state.keys.contains(codes[0]))
            timestamp += 1
            XCTAssertTrue(tracker.consume(InputSample(timestampNanos: timestamp, kind: .flags, keyCode: codes[1], modifiers: mask, isDown: true)))
            XCTAssertTrue(tracker.state.keys.isSuperset(of: codes))
            timestamp += 1
            XCTAssertTrue(tracker.consume(InputSample(timestampNanos: timestamp, kind: .flags, keyCode: codes[0], modifiers: mask, isDown: false)))
            XCTAssertFalse(tracker.state.keys.contains(codes[0]))
            XCTAssertTrue(tracker.state.keys.contains(codes[1]))
            timestamp += 1
            XCTAssertTrue(tracker.consume(InputSample(timestampNanos: timestamp, kind: .flags, keyCode: codes[1], modifiers: 0, isDown: false)))
            XCTAssertTrue(tracker.state.keys.isDisjoint(with: codes))
            timestamp += 1
        }

        for button in UInt8(0)..<32 {
            XCTAssertTrue(tracker.consume(InputSample(timestampNanos: timestamp, kind: .mouseButton, x: 20, y: 30, button: button, isDown: true)))
            XCTAssertTrue(tracker.state.buttons.contains(button), "Mouse button \(button) did not enter held state")
            timestamp += 1
            XCTAssertTrue(tracker.consume(InputSample(timestampNanos: timestamp, kind: .mouseButton, x: 20, y: 30, button: button, isDown: false)))
            XCTAssertFalse(tracker.state.buttons.contains(button), "Mouse button \(button) remained held")
            timestamp += 1
        }

        XCTAssertTrue(tracker.consume(InputSample(timestampNanos: timestamp, kind: .mouseMove, x: 50, y: 60, deltaX: 4, deltaY: -3)))
        XCTAssertEqual(tracker.state.mouseDelta, CGSize(width: 4, height: -3))
        XCTAssertEqual(tracker.pointer, CGPoint(x: 50, y: 60))
        timestamp += 1
        XCTAssertTrue(tracker.consume(InputSample(timestampNanos: timestamp, kind: .scroll, x: 51, y: 61, scrollX: 2.5, scrollY: -7.5)))
        XCTAssertEqual(tracker.state.mouseDelta, .zero)
        XCTAssertEqual(tracker.state.scrollDelta, CGSize(width: 2.5, height: -7.5))
        XCTAssertTrue(tracker.clearTransientDeltas())
        XCTAssertEqual(tracker.state.scrollDelta, .zero)

        let stale = InputSample(timestampNanos: timestamp - 1, kind: .key, keyCode: 49, isDown: true)
        XCTAssertFalse(tracker.consume(stale))
        XCTAssertFalse(tracker.state.keys.contains(49))
    }

    func testFlagOnlyKeyboardEventsNormalizeWithoutLosingCapsLockOrFn() {
        let shift = InputCaptureService.normalizedFlagsChangedSamples(
            timestampNanos: 10,
            keyCode: 56,
            modifiers: CGEventFlags.maskShift.rawValue,
            isPhysicallyDown: true
        )
        XCTAssertEqual(shift.count, 1)
        XCTAssertEqual(shift[0].kind, .flags)
        XCTAssertTrue(shift[0].isDown)

        let caps = InputCaptureService.normalizedFlagsChangedSamples(
            timestampNanos: 20,
            keyCode: 57,
            modifiers: CGEventFlags.maskAlphaShift.rawValue,
            isPhysicallyDown: true
        )
        XCTAssertEqual(caps.map(\.kind), [.key, .key])
        XCTAssertEqual(caps.map(\.keyCode), [57, 57])
        XCTAssertEqual(caps.map(\.isDown), [true, false])
        XCTAssertEqual(caps.map(\.timestampNanos), [20, 21])

        let function = InputCaptureService.normalizedFlagsChangedSamples(
            timestampNanos: 30,
            keyCode: 63,
            modifiers: CGEventFlags.maskSecondaryFn.rawValue,
            isPhysicallyDown: true
        )
        XCTAssertEqual(function.count, 1)
        XCTAssertEqual(function[0].kind, .key)
        XCTAssertEqual(function[0].keyCode, 63)
        XCTAssertTrue(function[0].isDown)
    }

    func testLiveModifierStateIsImmediateWhilePersistedShortcutDecisionRemainsLossless() {
        let shift = CGEventFlags.maskShift.rawValue
        let shiftDown = InputSample(timestampNanos: 1, kind: .flags, keyCode: 56, modifiers: shift, isDown: true)
        let keyDown = InputSample(timestampNanos: 2, kind: .key, keyCode: 0, modifiers: shift, isDown: true)
        let keyUp = InputSample(timestampNanos: 3, kind: .key, keyCode: 0, modifiers: shift, isDown: false)
        let shiftUp = InputSample(timestampNanos: 4, kind: .flags, keyCode: 56, modifiers: 0, isDown: false)

        var live = InputStateTracker()
        var persisted = HotkeyInputFilter(bindings: [.record])
        XCTAssertTrue(live.consume(shiftDown))
        XCTAssertTrue(live.state.keys.contains(56), "The live HUD must show Shift on key-down, not key-up")
        XCTAssertTrue(persisted.process(shiftDown).isEmpty, "Persistence may delay a modifier while deciding whether it starts a shortcut")
        XCTAssertEqual(persisted.process(keyDown), [shiftDown, keyDown])
        XCTAssertEqual(persisted.process(keyUp), [keyUp])
        XCTAssertEqual(persisted.process(shiftUp), [shiftUp])
        XCTAssertTrue(live.consume(shiftUp))
        XCTAssertFalse(live.state.keys.contains(56))

        var seededPersistence = HotkeyInputFilter(bindings: [.record])
        seededPersistence.primePersistedModifiers(shift)
        XCTAssertEqual(seededPersistence.process(shiftUp), [shiftUp], "A modifier seeded at the first frame must release at its real timestamp")
    }

    func testTerminalReleaseBalancingCoversKeysModifiersAndExtraMouseButtons() {
        let shift = CGEventFlags.maskShift.rawValue
        var tracker = InputStateTracker()
        tracker.consume(InputSample(timestampNanos: 10, kind: .flags, keyCode: 56, modifiers: shift, isDown: true))
        tracker.consume(InputSample(timestampNanos: 11, kind: .key, keyCode: 49, modifiers: shift, isDown: true))
        tracker.consume(InputSample(timestampNanos: 12, kind: .mouseButton, x: 80, y: 90, button: 15, modifiers: shift, isDown: true))
        let releases = tracker.terminalReleaseSamples(at: 20)
        XCTAssertEqual(Set(releases.map(\.timestampNanos)), [20])
        XCTAssertTrue(releases.contains { $0.kind == .key && $0.keyCode == 49 && !$0.isDown })
        XCTAssertTrue(releases.contains { $0.kind == .mouseButton && $0.button == 15 && !$0.isDown && $0.x == 80 && $0.y == 90 })
        XCTAssertTrue(releases.contains { $0.kind == .flags && $0.modifiers & HotkeyBinding.cgModifierMask == 0 })

        var missedModifierRelease = InputStateTracker()
        missedModifierRelease.consume(InputSample(timestampNanos: 30, kind: .flags, keyCode: 56, modifiers: shift, isDown: true))
        missedModifierRelease.consume(InputSample(timestampNanos: 31, kind: .mouseMove, modifiers: 0))
        XCTAssertTrue(missedModifierRelease.terminalReleaseSamples(at: 40).contains {
            $0.kind == .flags && $0.modifiers & HotkeyBinding.cgModifierMask == 0
        })
    }

    func testPhysicalStateReconciliationRepairsMissedKeyButtonAndModifierReleases() {
        let shift = CGEventFlags.maskShift.rawValue
        var state = PhysicalInputState()
        state.consume(InputSample(timestampNanos: 10, kind: .flags, modifiers: shift))
        state.consume(InputSample(timestampNanos: 11, kind: .key, keyCode: 49, modifiers: shift, isDown: true))
        state.consume(InputSample(timestampNanos: 12, kind: .mouseButton, button: 3, modifiers: shift, isDown: true))

        let releases = state.reconcile(
            to: PhysicalInputSnapshot(keys: [], buttons: [], modifiers: 0, pointer: CGPoint(x: 50, y: 40)),
            timestampNanos: 20
        )
        XCTAssertTrue(releases.contains { $0.kind == .key && $0.keyCode == 49 && !$0.isDown })
        XCTAssertTrue(releases.contains { $0.kind == .mouseButton && $0.button == 3 && !$0.isDown })
        XCTAssertTrue(releases.contains { $0.kind == .flags && $0.modifiers == 0 })
        XCTAssertTrue(state.reconcile(to: PhysicalInputSnapshot(keys: [], buttons: [], modifiers: 0, pointer: .zero), timestampNanos: 21).isEmpty)

        // A later mouse/key event may already carry the new modifier bits even
        // when the actual flagsChanged event was lost. It must not mask the
        // reconciliation pass that emits the missing modifier release.
        var missedFlagsState = PhysicalInputState()
        missedFlagsState.consume(InputSample(timestampNanos: 30, kind: .flags, modifiers: shift))
        missedFlagsState.consume(InputSample(timestampNanos: 31, kind: .mouseMove, modifiers: 0))
        let repairedFlags = missedFlagsState.reconcile(
            to: PhysicalInputSnapshot(keys: [], buttons: [], modifiers: 0, pointer: .zero),
            timestampNanos: 40
        )
        XCTAssertTrue(repairedFlags.contains { $0.kind == .flags && $0.modifiers == 0 })

        var seededState = PhysicalInputState()
        seededState.seed(
            from: PhysicalInputSnapshot(keys: [49], buttons: [15], modifiers: shift, pointer: .zero),
            timestampNanos: 100
        )
        seededState.consume(InputSample(timestampNanos: 99, kind: .key, keyCode: 49, isDown: false))
        let seededReleases = seededState.reconcile(
            to: PhysicalInputSnapshot(keys: [], buttons: [], modifiers: 0, pointer: .zero),
            timestampNanos: 110
        )
        XCTAssertTrue(seededReleases.contains { $0.kind == .key && $0.keyCode == 49 && !$0.isDown })
        XCTAssertTrue(seededReleases.contains { $0.kind == .mouseButton && $0.button == 15 && !$0.isDown })
    }

    func testRecordingKeyBlacklistDropsKeysAndSanitizesModifiers() {
        var filter = RecordingKeyFilter(excludedKeyCodes: [0, 56])
        let shift = CGEventFlags.maskShift.rawValue
        XCTAssertNil(filter.process(InputSample(timestampNanos: 1, kind: .key, keyCode: 0, modifiers: shift, isDown: true)))
        let mouse = filter.process(InputSample(timestampNanos: 2, kind: .mouseMove, deltaX: 3, modifiers: shift))
        XCTAssertEqual(mouse?.modifiers, 0)
        let allowed = filter.process(InputSample(timestampNanos: 3, kind: .key, keyCode: 13, modifiers: shift, isDown: true))
        XCTAssertEqual(allowed?.keyCode, 13)
        XCTAssertEqual(allowed?.modifiers, 0)
    }

    func testInputSummaryFindsEveryUsedKeyWithoutLoadingTimeline() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("summary-\(UUID().uuidString).atrevents")
        defer { try? FileManager.default.removeItem(at: url) }
        let writer = try InputEventWriter(url: url)
        for index in 0..<120 {
            writer.append(InputSample(timestampNanos: UInt64(index), kind: index % 3 == 0 ? .key : .mouseMove, keyCode: UInt16(index % 20), isDown: true))
        }
        _ = try writer.finish()
        let summary = try InputEventReader.summarize(url: url, previewLimit: 12)
        XCTAssertEqual(summary.preview.count, 12)
        XCTAssertEqual(summary.keyEventCount, 40)
        XCTAssertEqual(summary.mouseEventCount, 80)
        XCTAssertEqual(summary.usedKeyCodes, Set((0..<20).map(UInt16.init)))
    }

    func testInputSummaryIncludesPressedModifiers() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("summary-modifier-\(UUID().uuidString).atrevents")
        defer { try? FileManager.default.removeItem(at: url) }
        let writer = try InputEventWriter(url: url)
        writer.append(InputSample(timestampNanos: 1, kind: .flags, keyCode: 56, modifiers: CGEventFlags.maskShift.rawValue, isDown: true))
        writer.append(InputSample(timestampNanos: 2, kind: .flags, keyCode: 56, modifiers: 0, isDown: false))
        _ = try writer.finish()
        XCTAssertEqual(try InputEventReader.summarize(url: url).usedKeyCodes, [56])
    }

    func testMouseTrainingChannelIsIndependentOfRuntimeMode() {
        var channels = ActionChannels(absoluteMouse: true, relativeMouse: false, buttons: false, scroll: false, keyboard: false, modifiers: false)
        XCTAssertTrue(channels.mouseMovement)
        channels.mouseMovement = false
        XCTAssertFalse(channels.absoluteMouse)
        XCTAssertFalse(channels.relativeMouse)
        channels.mouseMovement = true
        XCTAssertTrue(channels.absoluteMouse)
        XCTAssertTrue(channels.relativeMouse)
        XCTAssertNotEqual(MouseControlMode.absolute, MouseControlMode.relative)
    }

    func testMouseTargetsStartAtRecordedPositionAndTrackEveryPointerEvent() {
        let manifest = RecordingManifest(
            id: UUID(), name: "Pointer", createdAt: Date(), hostStartNanos: 1, duration: 1,
            capture: CaptureSpec(), globalRect: CodableRect(CGRect(x: 100, y: 200, width: 400, height: 200)),
            pixelWidth: 400, pixelHeight: 200, deliveredFPS: 60, eventCount: 1
        )
        let first = InputSample(timestampNanos: 10, kind: .mouseMove, x: 300, y: 250)
        var accumulator = ActionAccumulator(manifest: manifest, events: [first])
        XCTAssertEqual(actionFloat(accumulator.actionData(), index: 0), 0.5, accuracy: 0.0001)
        XCTAssertEqual(actionFloat(accumulator.actionData(), index: 1), 0.25, accuracy: 0.0001)

        accumulator.consume(InputSample(timestampNanos: 20, kind: .mouseButton, x: 420, y: 360, button: 0, isDown: true))
        XCTAssertEqual(actionFloat(accumulator.actionData(), index: 0), 0.8, accuracy: 0.0001)
        XCTAssertEqual(actionFloat(accumulator.actionData(), index: 1), 0.8, accuracy: 0.0001)

        accumulator.consume(InputSample(timestampNanos: 30, kind: .scroll, x: 140, y: 220, scrollY: 1))
        XCTAssertEqual(actionFloat(accumulator.actionData(), index: 0), 0.1, accuracy: 0.0001)
        XCTAssertEqual(actionFloat(accumulator.actionData(), index: 1), 0.1, accuracy: 0.0001)
    }

    func testRecordingManifestValidationRejectsTraversalAndNonFiniteTimelines() {
        var manifest = RecordingManifest(
            id: UUID(), name: "Safe", createdAt: Date(), hostStartNanos: 1, duration: 1,
            capture: CaptureSpec(), globalRect: CodableRect(CGRect(x: 0, y: 0, width: 100, height: 100)),
            pixelWidth: 100, pixelHeight: 100, deliveredFPS: 60, eventCount: 0
        )
        XCTAssertTrue(manifest.isStructurallyValid)
        manifest.eventFile = ".."
        XCTAssertFalse(manifest.isStructurallyValid)
        manifest.eventFile = "events.atrevents"
        manifest.duration = .infinity
        XCTAssertFalse(manifest.isStructurallyValid)
    }

    func testSubTickPressesRemainVisibleInKeyboardButtonAndModifierTargets() {
        let manifest = RecordingManifest(
            id: UUID(), name: "Tap", createdAt: Date(), hostStartNanos: 1, duration: 1,
            capture: CaptureSpec(), globalRect: CodableRect(CGRect(x: 0, y: 0, width: 100, height: 100)),
            pixelWidth: 100, pixelHeight: 100, deliveredFPS: 60, eventCount: 6
        )
        var accumulator = ActionAccumulator(manifest: manifest)
        accumulator.consume(InputSample(timestampNanos: 1, kind: .key, keyCode: 13, modifiers: CGEventFlags.maskShift.rawValue, isDown: true))
        accumulator.consume(InputSample(timestampNanos: 2, kind: .key, keyCode: 13, modifiers: 0, isDown: false))
        accumulator.consume(InputSample(timestampNanos: 3, kind: .mouseButton, button: 1, isDown: true))
        accumulator.consume(InputSample(timestampNanos: 4, kind: .mouseButton, button: 1, isDown: false))
        let pulse = accumulator.actionData()
        XCTAssertEqual(actionFloat(pulse, index: ActionLayout.keyboard.lowerBound + 13), 1)
        XCTAssertEqual(actionFloat(pulse, index: ActionLayout.buttons.lowerBound + 1), 1)
        XCTAssertEqual(actionFloat(pulse, index: ActionLayout.modifiers.lowerBound), 1)

        accumulator.endTick()
        let released = accumulator.actionData()
        XCTAssertEqual(actionFloat(released, index: ActionLayout.keyboard.lowerBound + 13), 0)
        XCTAssertEqual(actionFloat(released, index: ActionLayout.buttons.lowerBound + 1), 0)
        XCTAssertEqual(actionFloat(released, index: ActionLayout.modifiers.lowerBound), 0)
    }

    func testGameCameraContractRoundTripsRawDeltaIndependentOfCaptureSize() {
        let collector = EventCollector()
        let injector = InputInjector(eventSink: { collector.append($0) }, cursorWarp: { collector.warp($0) })
        var profile = AIProfile.fresh()
        profile.channels.buttons = false
        profile.channels.scroll = false
        profile.channels.keyboard = false
        profile.channels.modifiers = false
        var prediction = [Float](repeating: 0, count: ActionLayout.count)
        prediction[2] = GameCameraContract.trainingValue(forRawDelta: 40)
        prediction[3] = GameCameraContract.trainingValue(forRawDelta: -20)

        injector.enable()
        injector.execute(prediction, profile: profile, allowedKeyCodes: [], mouseMode: .relative, captureRect: CGRect(x: 100, y: 50, width: 1_728, height: 1_117), safety: AgentSafetyPolicy(), gameCamera: GameCameraSettings(sensitivity: 1.5, recenterCursor: true))

        let move = collector.events.first { $0.0 == .mouseMoved }
        XCTAssertEqual(move?.2, 60)
        XCTAssertEqual(move?.3, -30)
        XCTAssertEqual(collector.warps, [CGPoint(x: 964, y: 608.5), CGPoint(x: 964, y: 608.5)])
        injector.disableAndReleaseAll()
    }

    func testInjectorDropsNonFinitePredictionsBeforePosting() {
        let collector = EventCollector()
        let injector = InputInjector(eventSink: { collector.append($0) })
        var prediction = [Float](repeating: 0, count: ActionLayout.count)
        prediction[0] = .nan
        injector.enable()
        injector.execute(prediction, profile: .fresh(), allowedKeyCodes: [], mouseMode: .absolute, captureRect: CGRect(x: 0, y: 0, width: 100, height: 100), safety: AgentSafetyPolicy())
        XCTAssertTrue(collector.events.isEmpty)
        injector.disableAndReleaseAll()
    }

    func testRuntimePredictionLatchConsumesTransientOutputsOnce() throws {
        var latch = RuntimePredictionLatch()
        var prediction = [Float](repeating: 0, count: ActionLayout.count)
        prediction[2] = 0.5
        prediction[3] = -0.25
        prediction[4] = 1
        prediction[12] = 0.4
        prediction[14 + 13] = 1

        XCTAssertNil(latch.consume())
        latch.publish(prediction, at: 123.5)
        let first = latch.consume()
        XCTAssertTrue(first?.isFresh == true)
        XCTAssertEqual(first?.values, prediction)
        XCTAssertEqual(first?.publishedAt, 123.5)

        let repeated = latch.consume()
        XCTAssertTrue(repeated?.isFresh == false)
        XCTAssertEqual(repeated?.values, prediction, "The latch retains state while marking reused predictions stale so transient execution can be suppressed separately")

        latch.publish(prediction, at: 124)
        XCTAssertTrue(latch.consume()?.isFresh == true)
        latch.reset()
        XCTAssertNil(latch.consume())
    }

    func testTemporalFrameControlsContainOnlyThresholdedExecutableActions() {
        var prediction = [Float](repeating: 0, count: ActionLayout.count)
        prediction[0] = 1.4
        prediction[2] = -0.8
        prediction[ActionLayout.buttons.lowerBound] = 0.51
        prediction[ActionLayout.buttons.lowerBound + 1] = 0.99
        prediction[ActionLayout.scroll.lowerBound] = 0.25
        prediction[ActionLayout.keyboard.lowerBound + 2] = 0.99  // Never demonstrated.
        prediction[ActionLayout.keyboard.lowerBound + 12] = 0.49
        prediction[ActionLayout.keyboard.lowerBound + 13] = 0.51
        prediction[ActionLayout.keyboard.lowerBound + 59] = 0.99 // Duplicate Control path.
        prediction[ActionLayout.commandOptionControl.lowerBound] = 0.75

        var restrictions = ActionRestrictions()
        restrictions.blockedMouseButtons = [1]
        let controls = RuntimeActionSemantics.temporalControlValues(
            prediction,
            channels: .all,
            restrictions: restrictions,
            allowedKeyCodes: [12, 13, 59],
            outputPermissions: RuntimeOutputPermissions(),
            shiftUsesKeyboardChannel: true
        )
        XCTAssertEqual(controls[0], 1, "Frame controls should use the same bounded value as execution")
        XCTAssertEqual(controls[2], -0.8, accuracy: 0.000_001)
        XCTAssertEqual(controls[ActionLayout.scroll.lowerBound], 0.25, accuracy: 0.000_001, "Fresh frame controls retain transient scroll")
        XCTAssertEqual(controls[ActionLayout.buttons.lowerBound], 1)
        XCTAssertEqual(controls[ActionLayout.buttons.lowerBound + 1], 0)
        XCTAssertEqual(controls[ActionLayout.keyboard.lowerBound + 2], 0, "The demonstrated-key firewall must also guard temporal state")
        XCTAssertEqual(controls[ActionLayout.keyboard.lowerBound + 12], 0)
        XCTAssertEqual(controls[ActionLayout.keyboard.lowerBound + 13], 1)
        XCTAssertEqual(controls[ActionLayout.keyboard.lowerBound + 59], 0, "Duplicate modifier key-code slots must never become hidden state")
        XCTAssertEqual(controls[ActionLayout.commandOptionControl.lowerBound], 1)

        let blocked = RuntimeActionSemantics.temporalControlValues(
            prediction,
            channels: .all,
            allowedKeyCodes: [13, 59],
            outputPermissions: RuntimeOutputPermissions(cursorMovement: false, keyboard: false)
        )
        XCTAssertTrue(ActionLayout.absoluteMouse.allSatisfy { blocked[$0] == 0 })
        XCTAssertTrue(ActionLayout.relativeMouse.allSatisfy { blocked[$0] == 0 })
        XCTAssertTrue(ActionLayout.keyboardAndShift.allSatisfy { blocked[$0] == 0 })
        XCTAssertTrue(ActionLayout.commandOptionControl.allSatisfy { blocked[$0] == 0 })

        var oldChannels = ActionChannels.all
        oldChannels.keyboard = false
        let oldShift = RuntimeActionSemantics.temporalControlValues(
            prediction.enumerated().map { $0.offset == ActionLayout.shift.lowerBound ? 1 : $0.element },
            channels: oldChannels,
            allowedKeyCodes: [56],
            shiftUsesKeyboardChannel: false
        )
        XCTAssertEqual(oldShift[ActionLayout.shift.lowerBound], 1, "Legacy brains route Shift through their saved Modifiers channel")
    }

    func testGameCameraDoesNotReplayStaleOrZeroDeltasAndNeverUsesDragEvents() {
        let collector = EventCollector()
        let injector = InputInjector(eventSink: { collector.append($0) }, cursorWarp: { collector.warp($0) })
        var profile = AIProfile.fresh()
        profile.channels.mouseMovement = true
        profile.channels.buttons = true
        profile.channels.scroll = true
        profile.channels.keyboard = true
        profile.channels.modifiers = false
        var prediction = [Float](repeating: 0, count: ActionLayout.count)
        prediction[2] = GameCameraContract.trainingValue(forRawDelta: 40)
        prediction[3] = GameCameraContract.trainingValue(forRawDelta: -20)
        prediction[4] = 1
        prediction[12] = 0.5
        prediction[14 + 13] = 1

        injector.enable()
        injector.execute(prediction, profile: profile, allowedKeyCodes: [13], mouseMode: .relative, captureRect: CGRect(x: 0, y: 0, width: 200, height: 100), safety: AgentSafetyPolicy(), predictionIsFresh: false)
        XCTAssertFalse(collector.events.contains { $0.0 == .mouseMoved || $0.0 == .scrollWheel })
        XCTAssertTrue(collector.warps.isEmpty)
        XCTAssertTrue(collector.events.contains { $0.0 == .leftMouseDown })
        XCTAssertTrue(collector.events.contains { $0.0 == .keyDown && $0.1 == 13 })

        let heldStateEventCount = collector.events.count
        injector.execute(prediction, profile: profile, allowedKeyCodes: [13], mouseMode: .relative, captureRect: CGRect(x: 0, y: 0, width: 200, height: 100), safety: AgentSafetyPolicy(), predictionIsFresh: false)
        XCTAssertEqual(collector.events.count, heldStateEventCount, "A stale prediction replayed an additive event")

        injector.execute(prediction, profile: profile, allowedKeyCodes: [13], mouseMode: .relative, captureRect: CGRect(x: 0, y: 0, width: 200, height: 100), safety: AgentSafetyPolicy(), predictionIsFresh: true)
        XCTAssertTrue(collector.events.contains { $0.0 == .mouseMoved && $0.2 == 40 && $0.3 == -20 })
        XCTAssertTrue(collector.events.contains { $0.0 == .scrollWheel })
        XCTAssertFalse(collector.events.contains { $0.0 == .leftMouseDragged || $0.0 == .rightMouseDragged || $0.0 == .otherMouseDragged })
        XCTAssertEqual(collector.warps, [CGPoint(x: 100, y: 50), CGPoint(x: 100, y: 50)])
        XCTAssertEqual(collector.events.count { $0.0 == .leftMouseDown }, 1)
        XCTAssertEqual(collector.events.count { $0.0 == .keyDown && $0.1 == 13 }, 1)

        let freshEventCount = collector.events.count
        prediction[2] = 0
        prediction[3] = 0
        prediction[12] = 0
        injector.execute(prediction, profile: profile, allowedKeyCodes: [13], mouseMode: .relative, captureRect: CGRect(x: 0, y: 0, width: 200, height: 100), safety: AgentSafetyPolicy(), predictionIsFresh: true)
        XCTAssertEqual(collector.events.count, freshEventCount, "A rounded zero camera/scroll delta posted an event")
        XCTAssertEqual(collector.warps.count, 2, "A zero camera delta still warped the cursor")
        injector.disableAndReleaseAll()
    }

    func testInputSummaryDetectsLockedGameCameraAndValidMousePositions() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("camera-events-\(UUID().uuidString).atrevents")
        defer { try? FileManager.default.removeItem(at: url) }
        let writer = try InputEventWriter(url: url)
        for index in 0..<40 {
            writer.append(InputSample(timestampNanos: UInt64(index), kind: .mouseMove, x: 500, y: 300, deltaX: index.isMultiple(of: 2) ? 2 : 0, deltaY: index.isMultiple(of: 3) ? -1 : 0))
        }
        _ = try writer.finish()
        let summary = try InputEventReader.summarize(url: url, globalRect: CGRect(x: 0, y: 0, width: 1_000, height: 700))
        XCTAssertTrue(summary.mouse.isGameCamera)
        XCTAssertTrue(summary.mouse.positionsAreValid)
        XCTAssertEqual(summary.mouse.absolutePositionChangeCount, 0)
        XCTAssertGreaterThan(summary.mouse.nonzeroDeltaCount, 0)
    }

    func testInjectorCannotPostAfterDisableAndReleasesHeldControls() {
        let collector = EventCollector()
        let injector = InputInjector(eventSink: { collector.append($0) })
        var profile = AIProfile.fresh()
        profile.channels.mouseMovement = false
        profile.channels.scroll = false
        profile.channels.modifiers = false
        var prediction = [Float](repeating: 0, count: ActionLayout.count)
        prediction[4] = 1
        prediction[14 + 13] = 1 // W

        injector.enable()
        injector.execute(prediction, profile: profile, allowedKeyCodes: [13], mouseMode: .relative, captureRect: CGRect(x: 0, y: 0, width: 100, height: 100), safety: AgentSafetyPolicy())
        injector.disableAndReleaseAll()
        let afterDisable = collector.events
        XCTAssertTrue(afterDisable.contains { $0.0 == .leftMouseDown })
        XCTAssertTrue(afterDisable.contains { $0.0 == .leftMouseUp })
        XCTAssertTrue(afterDisable.contains { $0.0 == .keyDown && $0.1 == 13 })
        XCTAssertTrue(afterDisable.contains { $0.0 == .keyUp && $0.1 == 13 })

        injector.execute(prediction, profile: profile, allowedKeyCodes: [13], mouseMode: .relative, captureRect: CGRect(x: 0, y: 0, width: 100, height: 100), safety: AgentSafetyPolicy())
        XCTAssertEqual(collector.events.count, afterDisable.count, "A late action posted after the injector was disabled")
    }

    func testRuntimeCursorPermissionBlocksMovementWithoutBlockingMouseButtons() {
        let collector = EventCollector()
        let injector = InputInjector(eventSink: { collector.append($0) })
        var profile = AIProfile.fresh()
        profile.channels.scroll = false
        profile.channels.keyboard = false
        profile.channels.modifiers = false
        var prediction = [Float](repeating: 0, count: ActionLayout.count)
        prediction[0] = 0.75
        prediction[1] = 0.25
        prediction[4] = 1

        injector.enable(outputPermissions: RuntimeOutputPermissions(cursorMovement: false, keyboard: true))
        injector.execute(prediction, profile: profile, allowedKeyCodes: [], mouseMode: .absolute, captureRect: CGRect(x: 0, y: 0, width: 100, height: 100), safety: AgentSafetyPolicy())

        XCTAssertFalse(collector.events.contains { $0.0 == .mouseMoved })
        XCTAssertTrue(collector.events.contains { $0.0 == .leftMouseDown })
        injector.disableAndReleaseAll()
    }

    func testDisablingRuntimeKeyboardImmediatelyReleasesKeysAndPreventsRepress() {
        let collector = EventCollector()
        let injector = InputInjector(eventSink: { collector.append($0) })
        var profile = AIProfile.fresh()
        profile.channels.mouseMovement = false
        profile.channels.buttons = false
        profile.channels.scroll = false
        var prediction = [Float](repeating: 0, count: ActionLayout.count)
        prediction[14 + 13] = 1 // W
        prediction[142] = 1 // Shift

        injector.enable()
        injector.execute(prediction, profile: profile, allowedKeyCodes: [13, 56], mouseMode: .absolute, captureRect: CGRect(x: 0, y: 0, width: 100, height: 100), safety: AgentSafetyPolicy())
        XCTAssertTrue(collector.events.contains { $0.0 == .keyDown && $0.1 == 13 })
        let shiftEventsBeforeDisable = collector.events.count { $0.0 == .flagsChanged && $0.1 == 56 }
        XCTAssertEqual(shiftEventsBeforeDisable, 1)

        injector.updateOutputPermissions(RuntimeOutputPermissions(cursorMovement: true, keyboard: false))
        XCTAssertTrue(collector.events.contains { $0.0 == .keyUp && $0.1 == 13 })
        XCTAssertGreaterThan(collector.events.count { $0.0 == .flagsChanged && $0.1 == 56 }, shiftEventsBeforeDisable)
        let afterDisable = collector.events.count

        injector.execute(prediction, profile: profile, allowedKeyCodes: [13, 56], mouseMode: .absolute, captureRect: CGRect(x: 0, y: 0, width: 100, height: 100), safety: AgentSafetyPolicy())
        XCTAssertEqual(collector.events.count, afterDisable)
        injector.disableAndReleaseAll()
    }

    func testKeyboardDisableAndReenableCannotOvertakeTheReleaseEvent() {
        let collector = EventCollector()
        let releaseEntered = DispatchSemaphore(value: 0)
        let allowRelease = DispatchSemaphore(value: 0)
        let disableFinished = DispatchSemaphore(value: 0)
        let repressFinished = DispatchSemaphore(value: 0)
        let gate = OneShotGate()
        let injector = InputInjector(eventSink: { event in
            if event.type == .keyUp, event.getIntegerValueField(.keyboardEventKeycode) == 13 {
                if gate.claim() {
                    releaseEntered.signal()
                    _ = allowRelease.wait(timeout: .now() + 5)
                }
            }
            collector.append(event)
        })
        var profile = AIProfile.fresh()
        profile.channels.mouseMovement = false
        profile.channels.buttons = false
        profile.channels.scroll = false
        profile.channels.modifiers = false
        var prediction = [Float](repeating: 0, count: ActionLayout.count)
        prediction[14 + 13] = 1
        let runtimeProfile = profile
        let action = prediction
        let captureRect = CGRect(x: 0, y: 0, width: 100, height: 100)

        injector.enable()
        injector.execute(action, profile: runtimeProfile, allowedKeyCodes: [13], mouseMode: .absolute, captureRect: captureRect, safety: AgentSafetyPolicy())
        DispatchQueue.global().async {
            injector.updateOutputPermissions(RuntimeOutputPermissions(cursorMovement: true, keyboard: false))
            disableFinished.signal()
        }
        XCTAssertEqual(releaseEntered.wait(timeout: .now() + 2), .success)
        DispatchQueue.global().async {
            injector.updateOutputPermissions(RuntimeOutputPermissions(cursorMovement: true, keyboard: true))
            injector.execute(action, profile: runtimeProfile, allowedKeyCodes: [13], mouseMode: .absolute, captureRect: captureRect, safety: AgentSafetyPolicy())
            repressFinished.signal()
        }
        XCTAssertEqual(repressFinished.wait(timeout: .now() + 0.1), .timedOut, "A new key-down overtook the in-flight release")
        allowRelease.signal()
        XCTAssertEqual(disableFinished.wait(timeout: .now() + 2), .success)
        XCTAssertEqual(repressFinished.wait(timeout: .now() + 2), .success)
        let keyEvents = collector.events.filter { ($0.0 == .keyDown || $0.0 == .keyUp) && $0.1 == 13 }.map(\.0)
        XCTAssertEqual(keyEvents, [.keyDown, .keyUp, .keyDown])
    }

    func testShiftFollowsKeyboardWhileCommandOptionControlFollowModifierChannel() {
        var profile = AIProfile.fresh()
        profile.channels = ActionChannels(absoluteMouse: false, relativeMouse: false, buttons: false, scroll: false, keyboard: true, modifiers: false)
        var prediction = [Float](repeating: 0, count: ActionLayout.count)
        for index in ActionLayout.modifiers { prediction[index] = 1 }
        for code in ActionLayout.commandOptionControlKeyCodes {
            prediction[ActionLayout.keyboard.lowerBound + Int(code)] = 1
        }
        let modifierKeyCodes: Set<UInt16> = [56, 59, 58, 55]

        let keyboardCollector = EventCollector()
        let keyboardInjector = InputInjector(eventSink: { keyboardCollector.append($0) })
        keyboardInjector.enable()
        keyboardInjector.execute(
            prediction,
            profile: profile,
            allowedKeyCodes: modifierKeyCodes,
            mouseMode: .absolute,
            captureRect: CGRect(x: 0, y: 0, width: 100, height: 100),
            safety: AgentSafetyPolicy(),
            shiftUsesKeyboardChannel: true
        )
        let keyboardCodes = Set(keyboardCollector.events.map { UInt16($0.1) })
        XCTAssertTrue(keyboardCodes.contains(56))
        XCTAssertTrue(keyboardCodes.isDisjoint(with: [59, 58, 55]))
        keyboardInjector.disableAndReleaseAll()

        let legacyCollector = EventCollector()
        let legacyInjector = InputInjector(eventSink: { legacyCollector.append($0) })
        legacyInjector.enable()
        legacyInjector.execute(
            prediction,
            profile: profile,
            allowedKeyCodes: modifierKeyCodes,
            mouseMode: .absolute,
            captureRect: CGRect(x: 0, y: 0, width: 100, height: 100),
            safety: AgentSafetyPolicy()
        )
        XCTAssertTrue(legacyCollector.events.isEmpty, "An older brain must not bypass disabled Modifiers through ordinary key outputs")
        legacyInjector.disableAndReleaseAll()

        profile.channels.keyboard = false
        profile.channels.modifiers = true
        let modifierCollector = EventCollector()
        let modifierInjector = InputInjector(eventSink: { modifierCollector.append($0) })
        modifierInjector.enable()
        modifierInjector.execute(
            prediction,
            profile: profile,
            allowedKeyCodes: modifierKeyCodes,
            mouseMode: .absolute,
            captureRect: CGRect(x: 0, y: 0, width: 100, height: 100),
            safety: AgentSafetyPolicy(),
            shiftUsesKeyboardChannel: true
        )
        let modifierCodes = Set(modifierCollector.events.map { UInt16($0.1) })
        XCTAssertFalse(modifierCodes.contains(56))
        XCTAssertTrue(Set([59, 58, 55]).isSubset(of: modifierCodes))
        modifierInjector.disableAndReleaseAll()
    }

    func testTrainingSanitizerRemovesDisabledAndDuplicateModifierPaths() {
        var values = [Float](repeating: 1, count: ActionLayout.count * 2)
        let channels = ActionChannels(absoluteMouse: true, relativeMouse: true, buttons: true, scroll: true, keyboard: true, modifiers: false)
        values.withUnsafeMutableBufferPointer {
            ActionLayout.sanitizeTrainingRows($0, rowCount: 2, channels: channels, restrictions: ActionRestrictions())
        }

        for row in 0..<2 {
            let base = row * ActionLayout.count
            XCTAssertEqual(values[base + ActionLayout.keyboard.lowerBound + 13], 1)
            XCTAssertEqual(values[base + ActionLayout.keyboard.lowerBound + 56], 1)
            XCTAssertEqual(values[base + ActionLayout.shift.lowerBound], 1)
            for index in ActionLayout.commandOptionControlKeyboardIndices { XCTAssertEqual(values[base + index], 0) }
            for index in ActionLayout.commandOptionControl { XCTAssertEqual(values[base + index], 0) }
        }
    }

    func testCurrentModifierToggleCanDisableButNeverEnableAnOldBrainHead() {
        let savedEnabled = ActionChannels(absoluteMouse: true, relativeMouse: true, buttons: true, scroll: true, keyboard: true, modifiers: true)
        var currentDisabled = savedEnabled
        currentDisabled.modifiers = false
        XCTAssertFalse(RuntimeActionSemantics.effectiveChannels(saved: savedEnabled, current: currentDisabled).modifiers)

        var savedDisabled = savedEnabled
        savedDisabled.modifiers = false
        XCTAssertFalse(RuntimeActionSemantics.effectiveChannels(saved: savedDisabled, current: savedEnabled).modifiers)
    }

    func testInjectorCannotEmitAKeyMissingFromTraining() {
        let collector = EventCollector()
        let injector = InputInjector(eventSink: { collector.append($0) })
        var profile = AIProfile.fresh()
        profile.channels.mouseMovement = false
        profile.channels.buttons = false
        profile.channels.scroll = false
        profile.channels.modifiers = false
        var prediction = [Float](repeating: 0, count: ActionLayout.count)
        prediction[14 + 2] = 1  // D was never demonstrated.
        prediction[14 + 13] = 1 // W was demonstrated.

        injector.enable()
        injector.execute(prediction, profile: profile, allowedKeyCodes: [13], mouseMode: .absolute, captureRect: CGRect(x: 0, y: 0, width: 100, height: 100), safety: AgentSafetyPolicy())
        injector.disableAndReleaseAll()

        XCTAssertFalse(collector.events.contains { $0.0 == .keyDown && $0.1 == 2 })
        XCTAssertTrue(collector.events.contains { $0.0 == .keyDown && $0.1 == 13 })
    }

    func testMLXTensorKeepsTheExactSelectedResolution() {
        let grayscale = PreprocessingSpec(width: 128, height: 512, colorMode: .grayscale, bitDepth: 8, chroma: .yuv420, resizePolicy: .fit)
        let grayData = Data(repeating: 127, count: grayscale.sampleByteCount * 2)
        XCTAssertEqual(VisionPreprocessor.mlxTensor(grayData, batch: 2, spec: grayscale).shape, [2, 512, 128, 1])

        let color = PreprocessingSpec(width: 128, height: 512, colorMode: .color, bitDepth: 8, chroma: .yuv420, resizePolicy: .fit)
        let colorData = Data(repeating: 127, count: color.sampleByteCount)
        XCTAssertEqual(VisionPreprocessor.mlxTensor(colorData, batch: 1, spec: color).shape, [1, 512, 128, 3])
    }

    func testVisionPreviewReadsPackedBytesDirectly() throws {
        let grayscale = PreprocessingSpec(width: 2, height: 1, colorMode: .grayscale, bitDepth: 8)
        let image = try XCTUnwrap(VisionPreprocessor.previewImage(Data([0, 255]), spec: grayscale))
        let bitmap = try XCTUnwrap(image.representations.first as? NSBitmapImageRep)
        let bytes = try XCTUnwrap(bitmap.bitmapData)
        XCTAssertEqual(Array(UnsafeBufferPointer(start: bytes, count: 8)), [0, 0, 0, 255, 255, 255, 255, 255])

        let color = PreprocessingSpec(width: 2, height: 2, colorMode: .color, bitDepth: 8, chroma: .yuv420)
        let neutral = try XCTUnwrap(VisionPreprocessor.previewImage(Data([128, 128, 128, 128, 128, 128]), spec: color))
        XCTAssertEqual(neutral.size, NSSize(width: 2, height: 2))
    }

    func testVisionPreviewPreservesAspectRatioWithinDisplayBounds() throws {
        let spec = PreprocessingSpec(width: 8, height: 4, colorMode: .grayscale, bitDepth: 8)
        let image = try XCTUnwrap(VisionPreprocessor.previewImage(Data(repeating: 127, count: 32), spec: spec, maximumWidth: 3, maximumHeight: 3))
        XCTAssertEqual(image.size, NSSize(width: 3, height: 2))
    }

    func testTemporalVisionKeepsCurrentAndPastFramesIndependent() {
        let currentSpec = PreprocessingSpec(width: 4, height: 2, colorMode: .color, bitDepth: 8, chroma: .yuv444)
        let temporal = TemporalVisionConfiguration(pastFrameCount: 2, frameSpacing: 3, downsampleFactor: 2)
        let pastSpec = temporal.pastFrameSpec(from: currentSpec)
        XCTAssertEqual(pastSpec.width, 2)
        XCTAssertEqual(pastSpec.height, 1)
        XCTAssertEqual(pastSpec.colorMode, currentSpec.colorMode)
        XCTAssertEqual(pastSpec.chroma, currentSpec.chroma)
        XCTAssertEqual(pastSpec.bitDepth, currentSpec.bitDepth)

        let current = VisionPreprocessor.mlxTensor(
            Data(repeating: 255, count: currentSpec.sampleByteCount),
            batch: 1,
            spec: currentSpec
        )
        let pastBytes = Data([
            255, 0, 64, 128, 192, 255,
            0, 255, 128, 64, 255, 192
        ])
        let past = VisionPreprocessor.mlxPastFrameTensor(
            MLXArray(pastBytes, [1, 2, pastSpec.sampleByteCount], dtype: .uint8),
            spec: pastSpec
        )
        MLX.eval(current, past)

        XCTAssertEqual(current.shape, [1, 2, 4, 3])
        XCTAssertEqual(past.shape, [1, 2, 1, 2, 3])
        XCTAssertEqual(current.asArray(Float.self), [Float](repeating: 1, count: 24))
        let values = past.asArray(Float.self)
        XCTAssertEqual(values[0], 1, accuracy: 0.000_001)
        XCTAssertEqual(values[1], Float(64) / 255, accuracy: 0.000_001)
        XCTAssertEqual(values[6], 0, accuracy: 0.000_001)
        XCTAssertEqual(values[7], Float(128) / 255, accuracy: 0.000_001)
    }

    func testMetalSharedInputBufferHandsExactBytesToMLX() throws {
        let pool = try MetalArrayBufferPool(maximumCachedBytes: 1 << 20)
        let expected = (0..<4096).map { UInt8(truncatingIfNeeded: $0 &* 31) }
        let array = try pool.makeArrays([.init([16, 256], dtype: .uint8)]) { destinations in
            expected.withUnsafeBytes { destinations[0].copyMemory(from: $0) }
        }[0]
        MLX.eval(array)

        XCTAssertEqual(array.asArray(UInt8.self), expected)
    }

    func testCNNVisualizationSettingsAreStrictlyBounded() {
        var settings = CNNVisualizationSettings(enabled: true, mode: .featureChannels, framesPerSecond: .infinity, convolutionLayer: 99, featureChannelCount: 500, overlayOpacity: -4, actionFocus: .keyboard)
        settings = settings.sanitized(layerCount: 3)
        XCTAssertEqual(settings.framesPerSecond, 4)
        XCTAssertEqual(settings.convolutionLayer, 2)
        XCTAssertEqual(settings.featureChannelCount, 16)
        XCTAssertEqual(settings.overlayOpacity, 0.2)

        var finalLayer = CNNVisualizationSettings()
        finalLayer.convolutionLayer = -1
        XCTAssertEqual(finalLayer.sanitized(layerCount: 4).convolutionLayer, 3)
    }

    func testCNNFeatureGridSelectsStrongestChannelsDeterministically() {
        let tensor = CNNFeatureTensor(
            width: 2,
            height: 2,
            channels: 4,
            values: [
                1, 2, 0, 2,
                1, 2, 0, 2,
                1, 2, 0, 2,
                1, 2, 0, 2
            ],
            convolutionLayer: 1
        )
        XCTAssertEqual(CNNVisualizationImageRenderer.strongestChannels(in: tensor, count: 3), [1, 3, 0])
        XCTAssertEqual(CNNVisualizationImageRenderer.strongestChannels(in: tensor, count: 99), [1, 3, 0, 2])
    }

    func testCNNVisualizationSamplingBoundsSpatialAndChannelTransfers() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 200, height: 120, colorMode: .grayscale, bitDepth: 8)
        profile.training.architecture = .small
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        let tensor = MLXArray([Float](repeating: 0.5, count: 120 * 200 * 32), [1, 120, 200, 32])
        let sampled = model.sampledForVisualization(tensor)
        let channels = model.strongestChannelsForVisualization(tensor)
        MLX.eval(sampled, channels)
        XCTAssertLessThanOrEqual(max(sampled.dim(1), sampled.dim(2)), 96)
        XCTAssertEqual(channels.shape, [1, sampled.dim(1), sampled.dim(2), 16])
    }

    func testCNNVisualizationRendererProducesBoundedOverlayAndGridImages() throws {
        let spec = PreprocessingSpec(width: 80, height: 45, colorMode: .grayscale, bitDepth: 8)
        let packed = Data((0..<spec.sampleByteCount).map { index in UInt8((index % spec.width) * 255 / (spec.width - 1)) })
        var settings = CNNVisualizationSettings(enabled: true)
        let mapWidth = 20, mapHeight = 12
        let overlayValues = (0..<(mapWidth * mapHeight)).map { index -> Float in
            let x = Float(index % mapWidth) / Float(mapWidth - 1), y = Float(index / mapWidth) / Float(mapHeight - 1)
            let dx = x - 0.68, dy = y - 0.42
            return exp(-(dx * dx + dy * dy) * 18)
        }
        let overlayTensor = CNNFeatureTensor(width: mapWidth, height: mapHeight, channels: 1, values: overlayValues, convolutionLayer: 2)
        let overlay = CNNVisualizationImageRenderer.render(CNNVisualizationFrame(packed: packed, spec: spec, settings: settings, tensors: [overlayTensor], timestamp: 0))
        let overlayImage = try XCTUnwrap(overlay.image)
        XCTAssertEqual(overlayImage.size, NSSize(width: 80, height: 45))
        XCTAssertTrue(overlay.detail.contains("Stage 3"))

        settings.mode = .featureChannels
        settings.featureChannelCount = 4
        var gridValues: [Float] = []
        for y in 0..<mapHeight {
            for x in 0..<mapWidth {
                let nx = Float(x) / Float(mapWidth - 1), ny = Float(y) / Float(mapHeight - 1)
                let dx = nx - 0.5, dy = ny - 0.5
                gridValues += [nx, ny, (x / 3 + y / 3).isMultiple(of: 2) ? 1 : 0.05, exp(-(dx * dx + dy * dy) * 16)]
            }
        }
        let gridTensor = CNNFeatureTensor(width: mapWidth, height: mapHeight, channels: 4, values: gridValues, convolutionLayer: 2)
        let grid = CNNVisualizationImageRenderer.render(CNNVisualizationFrame(packed: packed, spec: spec, settings: settings, tensors: [gridTensor], timestamp: 0))
        let gridImage = try XCTUnwrap(grid.image)
        XCTAssertEqual(gridImage.size, NSSize(width: 600, height: 360))
        XCTAssertTrue(grid.detail.contains("top 4 maps"))
    }

    func testAdaptiveWarmupIsBoundedAndPersistsWithOptimizerState() throws {
        XCTAssertEqual(TrainingEngine.recommendedWarmupSteps(stepsPerEpoch: 1), 10)
        XCTAssertEqual(TrainingEngine.recommendedWarmupSteps(stepsPerEpoch: 120), 120)
        XCTAssertEqual(TrainingEngine.recommendedWarmupSteps(stepsPerEpoch: 5_000), 500)
        XCTAssertEqual(TrainingEngine.recommendedCycleSteps(stepsPerEpoch: 120, cycleEpochs: 8), 960)
        XCTAssertEqual(TrainingEngine.recommendedCycleSteps(stepsPerEpoch: Int.max, cycleEpochs: Int.max), 1_000_000)
        XCTAssertEqual(TrainingEngine.recommendedValidationSampleLimit(total: 10_000, batchSize: 32, segmentCount: 20), 3_200)
        XCTAssertEqual(TrainingEngine.recommendedValidationSampleLimit(total: 200, batchSize: 32, segmentCount: 20), 200)

        let baseline = TrainingEngine.adaptivePlateauUpdate(metric: 1, best: nil, plateauEpochs: 0, patience: 2)
        XCTAssertEqual(baseline.best, 1)
        XCTAssertFalse(baseline.shouldReduce)
        let noise = TrainingEngine.adaptivePlateauUpdate(metric: 0.999, best: baseline.best, plateauEpochs: 0, patience: 2)
        XCTAssertEqual(noise.best, 1, "Sub-threshold noise must not continually move the plateau target.")
        XCTAssertEqual(noise.plateauEpochs, 1)
        let plateau = TrainingEngine.adaptivePlateauUpdate(metric: 1.001, best: noise.best, plateauEpochs: noise.plateauEpochs, patience: 2)
        XCTAssertTrue(plateau.shouldReduce)
        XCTAssertEqual(plateau.plateauEpochs, 0)
        let improvement = TrainingEngine.adaptivePlateauUpdate(metric: 0.99, best: plateau.best, plateauEpochs: 1, patience: 2)
        XCTAssertEqual(improvement.best, 0.99)
        XCTAssertEqual(improvement.plateauEpochs, 0)
        XCTAssertFalse(improvement.shouldReduce)

        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 8, height: 8, colorMode: .grayscale)
        profile.training.architecture = .small
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        let saved = ResumableAdamW(
            learningRate: 0.001,
            weightDecay: 0.01,
            warmupSteps: 17,
            schedule: .adaptiveCosine,
            cycleSteps: 80,
            minimumLearningRateRatio: 0.1
        )
        saved.initialize(model: model)
        XCTAssertEqual(saved.effectiveLearningRate(at: 17), 0.001, accuracy: 0.000_001)
        XCTAssertEqual(saved.effectiveLearningRate(at: 97), 0.001, accuracy: 0.000_001, "A completed cosine cycle must restart at peak learning rate.")
        XCTAssertGreaterThanOrEqual(saved.effectiveLearningRate(at: 96), 0.0001)
        XCTAssertEqual(saved.reduceLearningRate(), 0.5, accuracy: 0.000_001)
        XCTAssertEqual(saved.effectiveLearningRate(at: 97), 0.0005, accuracy: 0.000_001)
        for _ in 0..<8 { _ = saved.reduceLearningRate() }
        XCTAssertEqual(saved.learningRateScale, 0.1, accuracy: 0.000_001)
        XCTAssertEqual(saved.effectiveLearningRate(at: 57), 0.0001, accuracy: 0.000_001, "Plateau reduction and cosine troughs must share one global non-zero floor.")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("warmup-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try saved.save(to: url)

        let restored = ResumableAdamW(learningRate: 0.5, weightDecay: 0.5)
        try restored.load(from: url)
        XCTAssertEqual(restored.warmupSteps, 17)
        XCTAssertEqual(restored.learningRate, 0.001, accuracy: 0.000_001)
        XCTAssertEqual(restored.weightDecay, 0.01, accuracy: 0.000_001)
        XCTAssertEqual(restored.schedule, .adaptiveCosine)
        XCTAssertEqual(restored.cycleSteps, 80)
        XCTAssertEqual(restored.minimumLearningRateRatio, 0.1, accuracy: 0.000_001)
        XCTAssertEqual(restored.learningRateScale, 0.1, accuracy: 0.000_001)
    }

    func testOptimizerMetadataWithoutScheduleRestoresLegacyDecay() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("legacy-optimizer-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try MLX.save(
            arrays: ["m.placeholder": MLXArray.zeros([1]), "v.placeholder": MLXArray.zeros([1])],
            metadata: [
                "step": "2000",
                "learningRate": "0.0003",
                "weightDecay": "0.01",
                "warmupSteps": "500"
            ],
            url: url
        )
        let restored = ResumableAdamW(learningRate: 0.5, weightDecay: 0.5)
        try restored.load(from: url)
        XCTAssertEqual(restored.schedule, .legacyInverseSquareRoot)
        XCTAssertEqual(restored.effectiveLearningRate(), 0.00015, accuracy: 0.000_001)
    }

    func testOptimizerCheckpointRejectsMissingMomentPairs() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("incomplete-optimizer-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try MLX.save(arrays: ["m.placeholder": MLXArray.zeros([1])], metadata: ["step": "1"], url: url)
        XCTAssertThrowsError(try ResumableAdamW(learningRate: 0.001, weightDecay: 0.01).load(from: url))
    }

    func testOptimizerCheckpointRejectsMalformedMetadata() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("invalid-optimizer-metadata-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try MLX.save(
            arrays: ["m.placeholder": MLXArray.zeros([1]), "v.placeholder": MLXArray.zeros([1])],
            metadata: ["step": "not-a-step", "learningRate": "nan", "weightDecay": "0.01"],
            url: url
        )
        XCTAssertThrowsError(try ResumableAdamW(learningRate: 0.001, weightDecay: 0.01).load(from: url))
    }

    func testGlobalGradientNormDoesNotOverflowWhenTheTrueNormIsFinite() {
        let gradients = ModuleParameters.unflattened([
            "large": MLXArray([Float(1e20), Float(-1e20)])
        ])

        let norm = ResumableAdamW.globalGradientNorm(gradients)
        MLX.eval(norm)
        let value = norm.item(Float.self)

        XCTAssertTrue(value.isFinite)
        XCTAssertEqual(value / 1e20, Float(2).squareRoot(), accuracy: 0.0001)
    }

    func testOptimizerDropsNonFiniteDerivativesWithoutPoisoningPersistentState() {
        let model = Linear(2, 2)
        let optimizer = ResumableAdamW(learningRate: 0.001, weightDecay: 0.01)
        optimizer.initialize(model: model)
        let gradients = model.mapParameters {
            MLXArray.ones($0.shape) * Float.nan
        }
        let norm = ResumableAdamW.globalGradientNorm(gradients)

        optimizer.update(
            model: model,
            gradients: gradients,
            targetType: .float32,
            gradientNorm: norm,
            maxGradientNorm: 1
        )
        MLX.eval(model.parameters(), optimizer.stateArrays(), norm)

        XCTAssertEqual(norm.item(Float.self), 0)
        for (_, parameter) in model.parameters().flattened() {
            XCTAssertTrue(parameter.asArray(Float.self).allSatisfy(\.isFinite))
        }
        for state in optimizer.stateArrays() {
            XCTAssertTrue(state.asArray(Float.self).allSatisfy(\.isFinite))
        }
    }

    func testIntegratedGlobalGradientClipExactlyMatchesCanonicalClipping() throws {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 16, height: 12, colorMode: .grayscale)
        profile.training.architecture = .small
        profile.training.precision = .bfloat16
        let canonicalModel = AgentPolicy(profile: profile)
        let integratedModel = AgentPolicy(profile: profile)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("integrated-clip-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try canonicalModel.saveWeights(to: url)
        try integratedModel.loadWeights(from: url)
        let canonicalOptimizer = ResumableAdamW(learningRate: 0.0003, weightDecay: 0.01)
        let integratedOptimizer = ResumableAdamW(learningRate: 0.0003, weightDecay: 0.01)
        canonicalOptimizer.initialize(model: canonicalModel)
        integratedOptimizer.initialize(model: integratedModel)
        let canonicalGradients = canonicalModel.mapParameters {
            MLXArray.ones($0.shape) * 0.03125
        }
        let integratedGradients = integratedModel.mapParameters {
            MLXArray.ones($0.shape) * 0.03125
        }

        for _ in 0..<4 {
            let clipped = clipGradNorm(gradients: canonicalGradients, maxNorm: 0.1).0
            canonicalOptimizer.update(
                model: canonicalModel,
                gradients: clipped,
                targetType: canonicalModel.dtype
            )
            integratedOptimizer.update(
                model: integratedModel,
                gradients: integratedGradients,
                targetType: integratedModel.dtype,
                gradientNorm: ResumableAdamW.globalGradientNorm(integratedGradients),
                maxGradientNorm: 0.1
            )
            MLX.eval(
                canonicalModel.parameters(),
                integratedModel.parameters(),
                canonicalOptimizer.stateArrays(),
                integratedOptimizer.stateArrays()
            )
        }

        let expectedParameters = canonicalModel.parameters().flattened().sorted { $0.0 < $1.0 }
        let actualParameters = integratedModel.parameters().flattened().sorted { $0.0 < $1.0 }
        XCTAssertEqual(expectedParameters.map(\.0), actualParameters.map(\.0))
        for (expected, actual) in zip(expectedParameters, actualParameters) {
            XCTAssertEqual(expected.1.asArray(Float.self), actual.1.asArray(Float.self), expected.0)
        }
        XCTAssertEqual(canonicalOptimizer.stateArrays().count, integratedOptimizer.stateArrays().count)
        for (expected, actual) in zip(canonicalOptimizer.stateArrays(), integratedOptimizer.stateArrays()) {
            XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self))
        }
    }

    func testCompiledTrainingStepMatchesUncompiledAdamW() throws {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 16, height: 12, colorMode: .grayscale, bitDepth: 8)
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 1, frameSpacing: 1, downsampleFactor: 2)
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.generalization = .disabled
        profile.training.precision = .float32
        let modelA = AgentPolicy(profile: profile), modelB = AgentPolicy(profile: profile)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("compiled-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let weights = directory.appendingPathComponent("initial.safetensors")
        try modelA.saveWeights(to: weights); try modelB.loadWeights(from: weights)
        let optimizerA = ResumableAdamW(learningRate: 0.001, weightDecay: 0.01)
        let optimizerB = ResumableAdamW(learningRate: 0.001, weightDecay: 0.01)
        optimizerA.initialize(model: modelA); optimizerB.initialize(model: modelB)
        let inputs = temporalModelInputs(profile: profile, batch: 2, value: 0.25)
        let targets = MLXArray([Float](repeating: 0, count: 2 * ActionLayout.count), [2, ActionLayout.count])

        let gradientA = valueAndGrad(model: modelA) { model, arrays in
            [model.loss(currentImages: arrays[0], pastImages: arrays[1], pastControls: arrays[2], targets: arrays[3])]
        }
        let resultA = gradientA(modelA, [inputs.current, inputs.past, inputs.controls, targets])
        optimizerA.update(model: modelA, gradients: resultA.1, targetType: modelA.dtype)
        MLX.eval(resultA.0, modelA.parameters(), optimizerA.stateArrays())

        let compiled = compile(inputs: [modelB, optimizerB], outputs: [modelB, optimizerB]) { (arrays: [MLXArray]) -> [MLXArray] in
            let result = valueAndGrad(model: modelB) { model, arrays in
                [model.loss(currentImages: arrays[0], pastImages: arrays[1], pastControls: arrays[2], targets: arrays[3])]
            }(modelB, arrays)
            optimizerB.update(model: modelB, gradients: result.1, targetType: modelB.dtype)
            return [result.0[0]]
        }
        let lossB = compiled([inputs.current, inputs.past, inputs.controls, targets])[0]
        MLX.asyncEval(lossB, modelB.parameters(), optimizerB.stateArrays())
        // Mirrors the trainer's CPU prefetch window while Metal is executing.
        XCTAssertEqual((0..<10_000).reduce(0, +), 49_995_000)
        MLX.eval(lossB, modelB.parameters(), optimizerB.stateArrays())

        let paramsA = Dictionary(uniqueKeysWithValues: modelA.parameters().flattened())
        let paramsB = Dictionary(uniqueKeysWithValues: modelB.parameters().flattened())
        for key in paramsA.keys {
            let a = try XCTUnwrap(paramsA[key]).asArray(Float.self)
            let b = try XCTUnwrap(paramsB[key]).asArray(Float.self)
            XCTAssertTrue(zip(a, b).allSatisfy { abs($0 - $1) < 1e-5 }, "Compiled update differs at \(key)")
        }
        XCTAssertEqual(optimizerA.step, optimizerB.step)
    }

    func testOrderedCompiledStepQueueStaysBoundedAgainstPerStepSynchronization() throws {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(
            width: 12,
            height: 8,
            colorMode: .grayscale,
            bitDepth: 8
        )
        profile.training.temporalVision = TemporalVisionConfiguration(
            pastFrameCount: 1,
            frameSpacing: 1,
            downsampleFactor: 2
        )
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0.1
        profile.training.generalization = .disabled
        profile.training.precision = .float32
        let synchronizedModel = AgentPolicy(profile: profile)
        let queuedModel = AgentPolicy(profile: profile)
        synchronizedModel.train(true)
        queuedModel.train(true)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("queued-steps-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let weights = directory.appendingPathComponent("initial.safetensors")
        try synchronizedModel.saveWeights(to: weights)
        try queuedModel.loadWeights(from: weights)
        let synchronizedOptimizer = ResumableAdamW(learningRate: 0.001, weightDecay: 0.01)
        let queuedOptimizer = ResumableAdamW(learningRate: 0.001, weightDecay: 0.01)
        let synchronizedRandomState = MLXRandom.RandomState(seed: 81_551)
        let queuedRandomState = MLXRandom.RandomState(seed: 81_551)
        synchronizedOptimizer.initialize(model: synchronizedModel)
        queuedOptimizer.initialize(model: queuedModel)
        let inputs = temporalModelInputs(profile: profile, batch: 2, value: 0.25)
        let targets = MLXArray(
            [Float](repeating: 0, count: 2 * ActionLayout.count),
            [2, ActionLayout.count]
        )

        func compiledStep(
            model: AgentPolicy,
            optimizer: ResumableAdamW,
            randomState: MLXRandom.RandomState
        ) -> @Sendable ([MLXArray]) -> [MLXArray] {
            withRandomState(randomState) {
                compile(
                    inputs: [model, optimizer, randomState],
                    outputs: [model, optimizer, randomState]
                ) { arrays in
                    let result = valueAndGrad(model: model) { candidate, values in
                        [candidate.loss(
                            currentImages: values[0],
                            pastImages: values[1],
                            pastControls: values[2],
                            targets: values[3]
                        )]
                    }(model, arrays)
                    optimizer.update(
                        model: model,
                        gradients: result.1,
                        targetType: model.dtype,
                        maxGradientNorm: 1
                    )
                    return [result.0[0]]
                }
            }
        }

        let synchronizedStep = compiledStep(
            model: synchronizedModel,
            optimizer: synchronizedOptimizer,
            randomState: synchronizedRandomState
        )
        let queuedStep = compiledStep(
            model: queuedModel,
            optimizer: queuedOptimizer,
            randomState: queuedRandomState
        )
        let stepInputs = [inputs.current, inputs.past, inputs.controls, targets]
        var synchronizedLosses: [MLXArray] = []
        var queuedLosses: [MLXArray] = []
        for _ in 0..<4 {
            let loss = withRandomState(synchronizedRandomState) {
                synchronizedStep(stepInputs)[0]
            }
            MLX.eval(
                loss,
                synchronizedModel.parameters(),
                synchronizedOptimizer.stateArrays(),
                synchronizedRandomState.innerState()
            )
            synchronizedLosses.append(loss)
        }
        for _ in 0..<4 {
            let loss = withRandomState(queuedRandomState) {
                queuedStep(stepInputs)[0]
            }
            MLX.asyncEval(
                loss,
                queuedModel.parameters(),
                queuedOptimizer.stateArrays(),
                queuedRandomState.innerState()
            )
            queuedLosses.append(loss)
        }
        MLX.eval(
            queuedLosses,
            queuedModel.parameters(),
            queuedOptimizer.stateArrays(),
            queuedRandomState.innerState()
        )

        let lossDelta = zip(
            synchronizedLosses.map { $0.item(Float.self) },
            queuedLosses.map { $0.item(Float.self) }
        ).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        // Convolution reductions are not bitwise deterministic across separately
        // compiled Metal graphs. The queued state must remain numerically close.
        XCTAssertLessThan(lossDelta, 0.005)
        let synchronizedParameters = synchronizedModel.parameters().flattened().sorted { $0.0 < $1.0 }
        let queuedParameters = queuedModel.parameters().flattened().sorted { $0.0 < $1.0 }
        XCTAssertEqual(synchronizedParameters.map(\.0), queuedParameters.map(\.0))
        for (expected, actual) in zip(synchronizedParameters, queuedParameters) {
            let expectedValues = expected.1.asArray(Float.self)
            let actualValues = actual.1.asArray(Float.self)
            let maximumDelta = zip(expectedValues, actualValues).reduce(Float(0)) {
                max($0, abs($1.0 - $1.1))
            }
            XCTAssertLessThan(maximumDelta, 0.000_1, expected.0)
        }
        XCTAssertEqual(synchronizedOptimizer.stateArrays().count, queuedOptimizer.stateArrays().count)
        for (index, pair) in zip(
            synchronizedOptimizer.stateArrays(),
            queuedOptimizer.stateArrays()
        ).enumerated() {
            let expectedValues = pair.0.asArray(Float.self)
            let actualValues = pair.1.asArray(Float.self)
            XCTAssertEqual(expectedValues.count, actualValues.count, "Optimizer state \(index)")
            XCTAssertTrue(expectedValues.allSatisfy(\.isFinite), "Synchronized optimizer state \(index)")
            XCTAssertTrue(actualValues.allSatisfy(\.isFinite), "Queued optimizer state \(index)")
        }
        XCTAssertEqual(synchronizedOptimizer.step, queuedOptimizer.step)
        XCTAssertEqual(
            synchronizedOptimizer.learningRateScale,
            queuedOptimizer.learningRateScale,
            accuracy: 0.000_001
        )
        XCTAssertEqual(
            synchronizedRandomState.innerState().map { $0.asArray(UInt32.self) },
            queuedRandomState.innerState().map { $0.asArray(UInt32.self) }
        )
    }

    func testOptimizerPipelineDepthUsesAvailableUnifiedMemoryConservatively() {
        var profile = AIProfile.fresh()
        profile.training.precision = .float32
        let workingSet = UInt64(ModelSizing.estimatedTrainingWorkingSet(profile))
        XCTAssertEqual(
            TrainingEngine.recommendedOptimizerPipelineDepth(
                profile: profile,
                physicalMemory: workingSet * 6
            ),
            4
        )
        XCTAssertEqual(
            TrainingEngine.recommendedOptimizerPipelineDepth(
                profile: profile,
                physicalMemory: workingSet * 3
            ),
            3
        )
        XCTAssertEqual(
            TrainingEngine.recommendedOptimizerPipelineDepth(
                profile: profile,
                physicalMemory: workingSet * 2
            ),
            2
        )
        XCTAssertEqual(
            TrainingEngine.recommendedOptimizerPipelineDepth(
                profile: profile,
                physicalMemory: max(1, workingSet - 1)
            ),
            1
        )
    }

    func testPackingConcurrencyUsesCPUAndUnifiedMemoryBounds() {
        let gibibyte = UInt64(1 << 30)
        XCTAssertEqual(
            DatasetCacheBuilder.recommendedPackingConcurrency(
                recordingCount: 1,
                largestDecodedFrameBytes: 64 * 1_024 * 1_024,
                physicalMemory: 64 * gibibyte,
                processorCount: 16
            ),
            1
        )
        XCTAssertEqual(
            DatasetCacheBuilder.recommendedPackingConcurrency(
                recordingCount: 8,
                largestDecodedFrameBytes: 16 * 1_024 * 1_024,
                physicalMemory: 64 * gibibyte,
                processorCount: 16
            ),
            4
        )
        XCTAssertEqual(
            DatasetCacheBuilder.recommendedPackingConcurrency(
                recordingCount: 8,
                largestDecodedFrameBytes: 16 * 1_024 * 1_024,
                physicalMemory: 64 * gibibyte,
                processorCount: 6
            ),
            3
        )
        XCTAssertEqual(
            DatasetCacheBuilder.recommendedPackingConcurrency(
                recordingCount: 8,
                largestDecodedFrameBytes: 64 * 1_024 * 1_024,
                physicalMemory: 512 * 1_024 * 1_024,
                processorCount: 16
            ),
            1
        )
        XCTAssertEqual(
            DatasetCacheBuilder.recommendedPackingLookahead(
                recordingCount: 752,
                concurrency: 4
            ),
            8
        )
        XCTAssertEqual(
            DatasetCacheBuilder.recommendedPackingLookahead(
                recordingCount: 3,
                concurrency: 4
            ),
            3
        )
        XCTAssertEqual(
            DatasetCacheBuilder.recommendedPackingLookahead(
                recordingCount: 0,
                concurrency: 4
            ),
            0
        )
    }

    func testCacheStorageEstimateReservesEveryOutstandingParallelShard() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(
            width: 64,
            height: 36,
            colorMode: .grayscale
        )
        profile.training.actionFPS = 60
        profile.training.perceptionFPS = 30
        profile.training.temporalVision = TemporalVisionConfiguration(
            pastFrameCount: 2,
            frameSpacing: 1,
            downsampleFactor: 2
        )
        let directory = FileManager.default.temporaryDirectory
        let recordings = (0..<3).map { index in
            let manifest = RecordingManifest(
                id: UUID(),
                name: "Estimate \(index)",
                createdAt: Date(),
                hostStartNanos: 1,
                duration: 60,
                capture: CaptureSpec(requestedFPS: 60),
                globalRect: CodableRect(CGRect(x: 0, y: 0, width: 1920, height: 1080)),
                pixelWidth: 1920,
                pixelHeight: 1080,
                deliveredFPS: 60,
                eventCount: 0
            )
            return RecordingItem(manifest: manifest, directory: directory)
        }
        let single = DatasetCacheBuilder.estimatedCacheStorage(
            profile: profile,
            recordings: [recordings[0]]
        )
        let parallel = DatasetCacheBuilder.estimatedCacheStorage(
            profile: profile,
            recordings: recordings
        )
        let reserve = single.peakWorkingBytes - single.cacheBytes

        // Even with a one-worker memory bound, lookahead permits two completed
        // shards to coexist with the growing final cache. A single recording
        // writes directly and exposes the common free-space reserve here.
        XCTAssertGreaterThanOrEqual(
            parallel.peakWorkingBytes,
            parallel.cacheBytes + 2 * single.cacheBytes + reserve
        )
    }

    func testPackedShardWriterRepeatsBytesAndRebasesOnlyRealIndices() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("packed-shard-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let repeatedURL = directory.appendingPathComponent("repeated.bin")
        let repeated = try BufferedFileWriter(url: repeatedURL, capacity: 64 * 1_024)
        let frame = Data([0, 7, 128, 255, 42])
        try frame.withUnsafeBytes { try repeated.appendRepeated($0, count: 2_000) }
        try repeated.finish()
        var expected = Data()
        expected.reserveCapacity(frame.count * 2_000)
        for _ in 0..<2_000 { expected.append(frame) }
        XCTAssertEqual(try Data(contentsOf: repeatedURL), expected)

        let indicesURL = directory.appendingPathComponent("indices.bin")
        let indices = try BufferedFileWriter(url: indicesURL, capacity: 64 * 1_024)
        var raw = Data()
        for value in [UInt32(0), 2, .max, 9] {
            var little = value.littleEndian
            Swift.withUnsafeBytes(of: &little) { raw.append(contentsOf: $0) }
        }
        try indices.appendRebasedObservationIndices(raw, observationBase: 17)
        try indices.finish()
        let rebased = try Data(contentsOf: indicesURL).withUnsafeBytes { bytes in
            stride(from: 0, to: bytes.count, by: MemoryLayout<UInt32>.size).map {
                bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self).littleEndian
            }
        }
        XCTAssertEqual(rebased, [17, 19, UInt32.max, 26])
    }

    func testCompiledTrainingThroughputAndActiveMemoryStayBounded() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 12, height: 8, colorMode: .grayscale, bitDepth: 8)
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 1, frameSpacing: 1, downsampleFactor: 2)
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        let optimizer = ResumableAdamW(learningRate: 0.001, weightDecay: 0.01)
        optimizer.initialize(model: model)
        let inputs = temporalModelInputs(profile: profile, batch: 2, value: 0.25)
        let targets = MLXArray([Float](repeating: 0, count: 2 * ActionLayout.count), [2, ActionLayout.count])
        let step = compile(inputs: [model, optimizer], outputs: [model, optimizer]) { (arrays: [MLXArray]) -> [MLXArray] in
            let result = valueAndGrad(model: model) { model, arrays in
                [model.loss(currentImages: arrays[0], pastImages: arrays[1], pastControls: arrays[2], targets: arrays[3])]
            }(model, arrays)
            optimizer.update(model: model, gradients: result.1, targetType: model.dtype)
            return [result.0[0]]
        }
        var durations: [Double] = []
        var settledMemory = 0
        for iteration in 0..<140 {
            let began = CFAbsoluteTimeGetCurrent()
            let loss = step([inputs.current, inputs.past, inputs.controls, targets])[0]
            MLX.eval(loss, model.parameters(), optimizer.stateArrays())
            if iteration >= 20 { durations.append(CFAbsoluteTimeGetCurrent() - began) }
            if iteration == 60 { settledMemory = Memory.activeMemory }
        }
        let first = durations.prefix(30).reduce(0, +) / 30
        let last = durations.suffix(30).reduce(0, +) / 30
        XCTAssertLessThan(last, first * 3.5 + 0.001, "Fixed-shape compiled training progressively slowed")
        XCTAssertLessThan(abs(Memory.activeMemory - settledMemory), 64 << 20, "Active MLX memory continued growing after warm-up")
    }

    func testTrainingRandomStateRestoresExactly() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("random-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        MLXRandom.seed(912_345)
        try TrainingRandomState.save(to: url)
        let first = MLXRandom.uniform(low: 0, high: 1, [64])
        MLX.eval(first)
        try TrainingRandomState.load(from: url)
        let restored = MLXRandom.uniform(low: 0, high: 1, [64])
        MLX.eval(restored)
        XCTAssertEqual(first.asArray(Float.self), restored.asArray(Float.self))
    }

    func testTrainingRandomStateIsIsolatedFromGlobalInferenceInitialization() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("local-random-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let trainingState = MLXRandom.RandomState(seed: 77)
        try TrainingRandomState.save(trainingState, to: url)
        let expected = withRandomState(trainingState) { MLXRandom.uniform(low: 0, high: 1, [32]) }
        MLX.eval(expected)

        MLXRandom.seed(999)
        _ = AgentPolicy(profile: AIProfile.fresh())

        try TrainingRandomState.load(trainingState, from: url)
        let restored = withRandomState(trainingState) { MLXRandom.uniform(low: 0, high: 1, [32]) }
        MLX.eval(restored)
        XCTAssertEqual(expected.asArray(Float.self), restored.asArray(Float.self))
    }

    func testTrainingRandomStateRejectsAFileWithoutTheStateTensor() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("invalid-random-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try MLX.save(arrays: ["other": MLXArray.zeros([1])], url: url)
        XCTAssertThrowsError(try TrainingRandomState.load(from: url))
    }

    func testPackedTemporalBatchesKeepFramesControlsAndTargetsAligned() throws {
        let sequences: [[UInt32]] = [
            [0, .max, .max],
            [1, .max, 0],
            [2, 0, 1]
        ]
        let actionRows = (1...3).map { row -> [Float] in
            var values = [Float](repeating: 0, count: ActionLayout.count)
            values[0] = Float(row)
            return values
        }
        let frameActionRows = (1...3).map { row -> [Float] in
            var values = [Float](repeating: 0, count: ActionLayout.count)
            values[0] = Float(row * 10)
            return values
        }
        let fixture = try makeSyntheticDataset(
            name: "packed-temporal-batch",
            pastFrameCount: 2,
            sequences: sequences,
            actionRows: actionRows,
            frameActionRows: frameActionRows
        )
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let dataset = fixture.dataset

        XCTAssertEqual(dataset.packedObservations(at: [2, 0, 1]), Data([2, 0, 1]))
        XCTAssertEqual(dataset.pastPackedObservations(at: [2, 0, 1]), Data([0, 1, 0, 0, 1, 0]))
        let controls = MLXArray(
            dataset.pastControlBatch(at: [2, 0, 1]),
            [3, 2, ActionLayout.count],
            type: Float.self
        ).asArray(Float.self)
        XCTAssertEqual(controls[0], 10)
        XCTAssertEqual(controls[ActionLayout.count], 20)
        XCTAssertEqual(controls[2 * ActionLayout.count], 0)
        XCTAssertEqual(controls[4 * ActionLayout.count], 0)
        XCTAssertEqual(controls[5 * ActionLayout.count], 10)
        let targets = MLXArray(dataset.actionBatch(at: [2, 0]), [2, ActionLayout.count], type: Float.self).asArray(Float.self)
        XCTAssertEqual(targets[0], 3); XCTAssertEqual(targets[ActionLayout.count], 1)
        let previous = MLXArray(dataset.previousActionBatch(at: [2, 0, 1]), [3, ActionLayout.count], type: Float.self).asArray(Float.self)
        XCTAssertEqual(previous[0], 2)
        XCTAssertEqual(previous[ActionLayout.count], 0, "A recording boundary must start from released controls.")
        XCTAssertEqual(previous[2 * ActionLayout.count], 1)
    }

    func testFusedTrainingBatchMarksUnavailablePastFrames() throws {
        let sequences: [[UInt32]] = [
            [0, .max, .max],
            [1, .max, 0],
            [2, 0, 1]
        ]
        let rows = sequences.map { _ in
            [Float](repeating: 0, count: ActionLayout.count)
        }
        let fixture = try makeSyntheticDataset(
            name: "past-frame-validity",
            pastFrameCount: 2,
            sequences: sequences,
            actionRows: rows
        )
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let rowBytes = ActionLayout.count * MemoryLayout<Float>.size
        var current = Data(count: sequences.count)
        var past = Data(count: sequences.count * 2)
        var controls = Data(count: sequences.count * 2 * rowBytes)
        var validity = Data(count: sequences.count * 2 * MemoryLayout<Float>.size)
        var actions = Data(count: sequences.count * 2 * rowBytes)
        current.withUnsafeMutableBytes { currentDestination in
            past.withUnsafeMutableBytes { pastDestination in
                controls.withUnsafeMutableBytes { controlDestination in
                    validity.withUnsafeMutableBytes { validityDestination in
                        actions.withUnsafeMutableBytes { actionDestination in
                            fixture.dataset.populateTrainingBatch(
                                at: Array(sequences.indices),
                                observationSequences: sequences.map {
                                    CachedObservationSequence(indices: $0)
                                },
                                packedCurrentObservations: currentDestination,
                                packedPastObservations: pastDestination,
                                pastControlRows: controlDestination,
                                pastFrameValidity: validityDestination,
                                actionRows: actionDestination
                            )
                        }
                    }
                }
            }
        }
        let values = validity.withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        XCTAssertEqual(values, [0, 0, 0, 1, 1, 1])
    }

    func testFusedTrainingBatchMatchesCanonicalCacheReadsAcrossSegments() throws {
        func verify(pastFrameCount: Int) throws {
            let sequences: [[UInt32]] = pastFrameCount == 2 ? [
                [0, .max, .max], [1, .max, 0], [2, 0, 1],
                [3, .max, .max], [4, .max, 3], [5, 3, 4]
            ] : [
                [0, .max], [1, 0], [2, 1],
                [3, .max], [4, 3], [5, 4]
            ]
            let rows = (0..<sequences.count).map { row -> [Float] in
                var values = [Float](repeating: 0, count: ActionLayout.count)
                values[0] = Float(row + 1)
                values[ActionLayout.keyboard.lowerBound + row] = 1
                return values
            }
            let fixture = try makeSyntheticDataset(
                name: "fused-training-batch-\(pastFrameCount)",
                pastFrameCount: pastFrameCount,
                sequences: sequences,
                actionRows: rows,
                segments: [
                    CacheSegment(recordingID: UUID(), start: 0, count: 3),
                    CacheSegment(recordingID: UUID(), start: 3, count: 3)
                ]
            )
            defer { try? FileManager.default.removeItem(at: fixture.directory) }

            let indices = [5, 0, 3, 2, 4, 1]
            let fused = fixture.dataset.trainingBatch(at: indices)
            var expectedCurrent = Data()
            var expectedPast = Data()
            var expectedControls = Data()
            var expectedActionRows = Data()
            for index in indices {
                expectedCurrent.append(fixture.dataset.packedObservations(at: [index]))
                expectedPast.append(fixture.dataset.pastPackedObservations(at: [index]))
                expectedControls.append(fixture.dataset.pastControlBatch(at: [index]))
                expectedActionRows.append(fixture.dataset.actionBatch(at: [index]))
                expectedActionRows.append(fixture.dataset.previousActionBatch(at: [index]))
            }

            XCTAssertEqual(fused.count, indices.count)
            XCTAssertEqual(fused.packedCurrentObservations, expectedCurrent)
            XCTAssertEqual(fused.packedPastObservations, expectedPast)
            XCTAssertEqual(fused.pastControlRows, expectedControls)
            XCTAssertEqual(fused.actionRows, expectedActionRows)
        }

        try verify(pastFrameCount: 2)
        try verify(pastFrameCount: 1)
    }

    func testVisionBatchPlanDeduplicatesTemporalSequencesWithoutChangingActionOrder() throws {
        let sequences: [[UInt32]] = [
            [0, .max], [0, .max], [1, 0], [1, 0], [2, 1], [2, 1]
        ]
        let rows = sequences.indices.map { row -> [Float] in
            var values = [Float](repeating: 0, count: ActionLayout.count)
            values[0] = Float(row)
            return values
        }
        let fixture = try makeSyntheticDataset(
            name: "deduplicated-vision-plan",
            pastFrameCount: 1,
            sequences: sequences,
            actionRows: rows,
            segments: [CacheSegment(recordingID: UUID(), start: 0, count: sequences.count)]
        )
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let indices = [3, 0, 1, 2, 5, 4]
        let plan = fixture.dataset.visionBatchPlan(at: indices)
        XCTAssertEqual(
            plan.uniqueSequences,
            [
                CachedObservationSequence(indices: [1, 0]),
                CachedObservationSequence(indices: [0, .max]),
                CachedObservationSequence(indices: [2, 1])
            ]
        )
        XCTAssertEqual(plan.sampleToVision, [0, 1, 1, 0, 2, 2])
        XCTAssertEqual(fixture.dataset.observationReuseRatio(at: indices), 2, accuracy: 0.000_001)
        XCTAssertEqual(plan.uniquePastObservations, [0, 1])
        XCTAssertEqual(plan.visionToPast, [0, 0, 1])
        XCTAssertEqual(plan.pastFrameReuseRatio, 1.5, accuracy: 0.000_001)
        let plannedGroups = fixture.dataset.observationGroups(at: indices, using: plan)
        XCTAssertEqual(fixture.dataset.observationGroups(at: indices), plannedGroups)
        XCTAssertEqual(plannedGroups, [[3, 2], [0, 1], [5, 4]])

        let rowBytes = ActionLayout.count * MemoryLayout<Float>.size
        var current = Data(count: plan.uniqueSequences.count)
        var past = Data(count: plan.uniqueSequences.count)
        var controls = Data(count: plan.uniqueSequences.count * rowBytes)
        var actions = Data(count: indices.count * 2 * rowBytes)
        current.withUnsafeMutableBytes { currentDestination in
            past.withUnsafeMutableBytes { pastDestination in
                controls.withUnsafeMutableBytes { controlDestination in
                    actions.withUnsafeMutableBytes { actionDestination in
                        fixture.dataset.populateTrainingBatch(
                            at: indices,
                            observationSequences: plan.uniqueSequences,
                            packedCurrentObservations: currentDestination,
                            packedPastObservations: pastDestination,
                            pastControlRows: controlDestination,
                            actionRows: actionDestination
                        )
                    }
                }
            }
        }
        let canonical = fixture.dataset.trainingBatch(at: indices)
        XCTAssertEqual(actions, canonical.actionRows)

        let expandedCurrent = plan.sampleToVision.reduce(into: Data()) { output, visionIndex in
            output.append(current[Int(visionIndex)])
        }
        let expandedPast = plan.sampleToVision.reduce(into: Data()) { output, visionIndex in
            output.append(past[Int(visionIndex)])
        }
        let expandedControls = plan.sampleToVision.reduce(into: Data()) { output, visionIndex in
            let offset = Int(visionIndex) * rowBytes
            output.append(controls[offset..<(offset + rowBytes)])
        }
        XCTAssertEqual(expandedCurrent, canonical.packedCurrentObservations)
        XCTAssertEqual(expandedPast, canonical.packedPastObservations)
        XCTAssertEqual(expandedControls, canonical.pastControlRows)

        var deduplicatedPast = Data(count: plan.uniquePastObservations.count)
        var deduplicatedCurrent = Data(count: plan.uniqueSequences.count)
        var deduplicatedControls = Data(count: plan.uniqueSequences.count * rowBytes)
        var deduplicatedActions = Data(count: indices.count * 2 * rowBytes)
        deduplicatedCurrent.withUnsafeMutableBytes { currentDestination in
            deduplicatedPast.withUnsafeMutableBytes { pastDestination in
                deduplicatedControls.withUnsafeMutableBytes { controlDestination in
                    deduplicatedActions.withUnsafeMutableBytes { actionDestination in
                        fixture.dataset.populateTrainingBatch(
                            at: indices,
                            visionPlan: plan,
                            packedCurrentObservations: currentDestination,
                            packedPastObservations: pastDestination,
                            pastControlRows: controlDestination,
                            actionRows: actionDestination
                        )
                    }
                }
            }
        }
        let restoredPast = plan.visionToPast.reduce(into: Data()) { output, pastIndex in
            output.append(deduplicatedPast[Int(pastIndex)])
        }
        XCTAssertEqual(deduplicatedCurrent, current)
        XCTAssertEqual(restoredPast, past)
        XCTAssertEqual(deduplicatedControls, controls)
        XCTAssertEqual(deduplicatedActions, actions)
    }

    func testGroupedVisionOrderPreservesRowsAndKeepsPairsInsideBatches() {
        let groups = (0..<12).map { group in [group * 2, group * 2 + 1] }
        let salient = Set([0, 6, 12, 18])
        let order = TrainingEngine().groupedVisionTrainingOrder(
            groups: groups,
            batchSize: 8,
            seed: 91,
            salientIndices: salient
        )

        XCTAssertEqual(order.sorted(), Array(0..<24))
        for group in groups {
            let positions = group.compactMap(order.firstIndex)
            XCTAssertEqual(positions.count, 2)
            XCTAssertEqual(positions[0] / 8, positions[1] / 8)
        }
        for start in stride(from: 0, to: order.count, by: 8) {
            let batch = order[start..<min(order.count, start + 8)]
            XCTAssertTrue(batch.contains(where: salient.contains))
        }
    }

    func testLocalityGroupedVisionOrderRandomizesBatchesWithoutLosingLocalityOrSalience() {
        let groups = (0..<12).map { group in [group * 2, group * 2 + 1] }
        let salient = Set([0, 6, 12, 18])
        let order = TrainingEngine().localityGroupedVisionTrainingOrder(
            groups: groups,
            batchSize: 8,
            seed: 91,
            salientIndices: salient
        )

        XCTAssertEqual(order.sorted(), Array(0..<24))
        for group in groups {
            let positions = group.compactMap(order.firstIndex)
            XCTAssertEqual(positions.count, 2)
            XCTAssertEqual(positions[0] / 8, positions[1] / 8)
        }
        for start in stride(from: 0, to: order.count, by: 8) {
            let batch = Array(order[start..<min(order.count, start + 8)])
            let groupIndices = batch.map { $0 / 2 }
            XCTAssertLessThanOrEqual(
                (groupIndices.max() ?? 0) - (groupIndices.min() ?? 0),
                3
            )
            XCTAssertTrue(batch.contains(where: salient.contains))
        }
        XCTAssertNotEqual(order, Array(0..<24))
    }

    func testLargeLocalityBatchesMixSeveralBoundedTemporalLanes() {
        let groups = (0..<256).map { [$0] }
        let order = TrainingEngine().localityGroupedVisionTrainingOrder(
            groups: groups,
            batchSize: 128,
            seed: 92,
            salientIndices: []
        )

        XCTAssertEqual(order.sorted(), Array(0..<256))
        XCTAssertNotEqual(order, Array(0..<256))
        for start in stride(from: 0, to: order.count, by: 128) {
            let sortedBatch = order[start..<min(order.count, start + 128)].sorted()
            let runCount = zip(sortedBatch, sortedBatch.dropFirst()).count { pair in
                pair.1 != pair.0 + 1
            } + 1
            XCTAssertLessThanOrEqual(runCount, 4)
            XCTAssertEqual(sortedBatch.count, 128)
        }
    }

    func testRealVideoCacheDeduplicatesPerceptionFramesAndPreservesSubTickControls() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("video-cache-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()
        let recordingID = UUID()
        let directory = try await store.createRecordingDirectory(id: recordingID)
        let videoURL = directory.appendingPathComponent("capture.mov")
        try await writeTestMovie(to: videoURL, width: 16, height: 16, frameCount: 6, fps: 30)

        let base: UInt64 = 1_000_000_000
        let eventsURL = directory.appendingPathComponent("events.atrevents")
        let writer = try InputEventWriter(url: eventsURL)
        writer.append(InputSample(timestampNanos: base, kind: .mouseMove, x: 8, y: 8))
        writer.append(InputSample(timestampNanos: base + 5_000_000, kind: .key, keyCode: 13, isDown: true))
        writer.append(InputSample(timestampNanos: base + 8_000_000, kind: .key, keyCode: 13, isDown: false))
        let eventCount = try writer.finish()
        let duration = 0.2
        let manifest = RecordingManifest(
            id: recordingID, name: "Integration", createdAt: Date(), hostStartNanos: base, duration: duration,
            capture: CaptureSpec(requestedFPS: 30), globalRect: CodableRect(CGRect(x: 0, y: 0, width: 16, height: 16)),
            pixelWidth: 16, pixelHeight: 16, deliveredFPS: 30, eventCount: eventCount
        )
        try await store.writeRecording(manifest, to: directory)

        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 8, height: 8, colorMode: .grayscale)
        profile.training.actionFPS = 60
        profile.training.perceptionFPS = 30
        profile.training.temporalVision = TemporalVisionConfiguration(
            pastFrameCount: 2,
            frameSpacing: 1,
            downsampleFactor: 2
        )
        let dataset = try await DatasetCacheBuilder(workspace: store).cache(for: profile, recordings: [RecordingItem(manifest: manifest, directory: directory)]) { _, _ in }

        XCTAssertGreaterThan(dataset.count, dataset.manifest.observationCount)
        XCTAssertGreaterThanOrEqual(dataset.manifest.observationCount, 5)
        XCTAssertTrue(dataset.demonstratedKeyCodes().contains(13))
        XCTAssertEqual(dataset.manifest.preprocessing.width, 8)
        XCTAssertEqual(dataset.manifest.pastPreprocessing.width, 4)
        XCTAssertEqual(dataset.pastPackedObservations(at: [0]).count, 2 * dataset.manifest.pastObservationBytesPerSample)
        XCTAssertTrue(
            MLXArray(
                dataset.pastControlBatch(at: Array(0..<dataset.count)),
                [dataset.count, 2, ActionLayout.count],
                type: Float.self
            ).asArray(Float.self).contains { $0 != 0 },
            "Frame-aligned controls should retain short demonstrated inputs instead of storing only model outputs."
        )
    }

    func testDatasetCacheSkipsUnreadableVideoBeforePackingValidNeighbors() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "video-cache-preflight-\(UUID().uuidString)",
            isDirectory: true
        )
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()

        func makeRecording(name: String, validVideo: Bool) async throws -> RecordingItem {
            let id = UUID()
            let directory = try await store.createRecordingDirectory(id: id)
            let videoURL = directory.appendingPathComponent("capture.mov")
            if validVideo {
                try await writeTestMovie(
                    to: videoURL,
                    width: 16,
                    height: 16,
                    frameCount: 4,
                    fps: 20
                )
            } else {
                // Mirrors an interrupted QuickTime writer: bytes exist, but no
                // finalized track index is available for AVFoundation to load.
                try Data("ftypqt  mdat unfinished".utf8).write(to: videoURL)
            }
            let eventWriter = try InputEventWriter(
                url: directory.appendingPathComponent("events.atrevents")
            )
            eventWriter.append(
                InputSample(
                    timestampNanos: 1_000_000_000,
                    kind: .mouseMove,
                    x: 8,
                    y: 8
                )
            )
            let eventCount = try eventWriter.finish()
            let manifest = RecordingManifest(
                id: id,
                name: name,
                createdAt: Date(),
                hostStartNanos: 1_000_000_000,
                duration: 0.2,
                capture: CaptureSpec(requestedFPS: 20),
                globalRect: CodableRect(
                    CGRect(x: 0, y: 0, width: 16, height: 16)
                ),
                pixelWidth: 16,
                pixelHeight: 16,
                deliveredFPS: 20,
                eventCount: eventCount
            )
            try await store.writeRecording(manifest, to: directory)
            return RecordingItem(manifest: manifest, directory: directory)
        }

        let broken = try await makeRecording(
            name: "Interrupted Recording",
            validVideo: false
        )
        let valid = try await makeRecording(name: "Valid Recording", validVideo: true)
        let brokenBytes = try Data(
            contentsOf: broken.directory.appendingPathComponent("capture.mov")
        )
        let readiness = try await DatasetCacheBuilder.trainingReadyRecordings(
            [broken, valid]
        ) { _, _, _ in }
        XCTAssertEqual(readiness.recordings.map(\.id), [valid.id])
        XCTAssertEqual(readiness.failures.map(\.recordingID), [broken.id])
        XCTAssertTrue(readiness.failures[0].diagnosticSummary.contains(broken.manifest.name))
        XCTAssertTrue(readiness.failures[0].reason.localizedCaseInsensitiveContains("video"))

        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(
            width: 8,
            height: 8,
            colorMode: .grayscale
        )
        profile.training.actionFPS = 20
        profile.training.perceptionFPS = 20
        profile.training.temporalVision = TemporalVisionConfiguration(
            pastFrameCount: 0,
            frameSpacing: 1,
            downsampleFactor: 1
        )
        let estimate = DatasetCacheBuilder.estimatedCacheStorage(
            profile: profile,
            recordings: [valid]
        )
        XCTAssertGreaterThan(estimate.cacheBytes, 0)
        XCTAssertGreaterThan(estimate.peakWorkingBytes, estimate.cacheBytes)

        let dataset = try await DatasetCacheBuilder(workspace: store).cache(
            for: profile,
            recordings: [broken, valid]
        ) { _, _ in }
        XCTAssertGreaterThan(dataset.count, 0)
        XCTAssertEqual(dataset.manifest.segments.map(\.recordingID), [valid.id])
        XCTAssertEqual(
            try Data(
                contentsOf: broken.directory.appendingPathComponent("capture.mov")
            ),
            brokenBytes,
            "Preflight must never rewrite or delete an unreadable source package."
        )
    }

    func testRealVideoCacheSupportsCurrentOnlyProfilesWithoutPastArtifacts() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("current-only-video-cache-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()
        let recordingID = UUID()
        let directory = try await store.createRecordingDirectory(id: recordingID)
        try await writeTestMovie(
            to: directory.appendingPathComponent("capture.mov"),
            width: 16,
            height: 16,
            frameCount: 4,
            fps: 20
        )

        let base: UInt64 = 2_000_000_000
        let eventWriter = try InputEventWriter(url: directory.appendingPathComponent("events.atrevents"))
        eventWriter.append(InputSample(timestampNanos: base, kind: .mouseMove, x: 8, y: 8))
        eventWriter.append(InputSample(timestampNanos: base + 50_000_000, kind: .key, keyCode: 13, isDown: true))
        eventWriter.append(InputSample(timestampNanos: base + 100_000_000, kind: .key, keyCode: 13, isDown: false))
        let eventCount = try eventWriter.finish()
        let manifest = RecordingManifest(
            id: recordingID,
            name: "Current only",
            createdAt: Date(),
            hostStartNanos: base,
            duration: 0.2,
            capture: CaptureSpec(requestedFPS: 20),
            globalRect: CodableRect(CGRect(x: 0, y: 0, width: 16, height: 16)),
            pixelWidth: 16,
            pixelHeight: 16,
            deliveredFPS: 20,
            frameCount: 4,
            eventCount: eventCount
        )
        try await store.writeRecording(manifest, to: directory)

        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 8, height: 8, colorMode: .grayscale)
        profile.training.actionFPS = 40
        profile.training.perceptionFPS = 20
        profile.training.temporalVision = TemporalVisionConfiguration(
            pastFrameCount: 0,
            frameSpacing: 240,
            downsampleFactor: 8
        )
        let dataset = try await DatasetCacheBuilder(workspace: store).cache(
            for: profile,
            recordings: [RecordingItem(manifest: manifest, directory: directory)]
        ) { _, _ in }

        XCTAssertGreaterThan(dataset.count, 0)
        XCTAssertEqual(dataset.manifest.temporalVision.pastFrameCount, 0)
        XCTAssertEqual(dataset.manifest.pastObservationBytesPerSample, 0)
        XCTAssertEqual(dataset.pastPackedObservations(at: [0]).count, 0)
        XCTAssertEqual(dataset.pastControlBatch(at: [0]).count, 0)
        let batch = dataset.trainingBatch(at: Array(0..<min(3, dataset.count)))
        XCTAssertEqual(batch.packedCurrentObservations.count, batch.count * profile.preprocessing.sampleByteCount)
        XCTAssertTrue(batch.packedPastObservations.isEmpty)
        XCTAssertTrue(batch.pastControlRows.isEmpty)
        XCTAssertEqual(batch.actionRows.count, batch.count * 2 * ActionLayout.count * MemoryLayout<Float>.size)
        XCTAssertTrue(dataset.demonstratedKeyCodes().contains(13))
    }

    func testCacheDoesNotLeakARecordingFirstFuturePointerEventBackward() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "causal-pointer-cache-\(UUID().uuidString)",
            isDirectory: true
        )
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()
        let recordingID = UUID()
        let directory = try await store.createRecordingDirectory(id: recordingID)
        try await writeTestMovie(
            to: directory.appendingPathComponent("capture.mov"),
            width: 16,
            height: 16,
            frameCount: 4,
            fps: 20
        )

        let base: UInt64 = 3_000_000_000
        let eventWriter = try InputEventWriter(
            url: directory.appendingPathComponent("events.atrevents")
        )
        // Imported recordings are allowed to lack the native time-zero pointer
        // seed. This first pointer sample belongs to the future, not row zero.
        eventWriter.append(InputSample(
            timestampNanos: base + 100_000_000,
            kind: .mouseMove,
            x: 90,
            y: 75
        ))
        let eventCount = try eventWriter.finish()
        let manifest = RecordingManifest(
            id: recordingID,
            name: "Causal pointer",
            createdAt: Date(),
            hostStartNanos: base,
            duration: 0.2,
            capture: CaptureSpec(requestedFPS: 20),
            globalRect: CodableRect(CGRect(x: 0, y: 0, width: 100, height: 100)),
            pixelWidth: 16,
            pixelHeight: 16,
            deliveredFPS: 20,
            frameCount: 4,
            eventCount: eventCount
        )
        try await store.writeRecording(manifest, to: directory)

        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(
            width: 8,
            height: 8,
            colorMode: .grayscale
        )
        profile.training.actionFPS = 20
        profile.training.perceptionFPS = 20
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 0)
        let dataset = try await DatasetCacheBuilder(workspace: store).cache(
            for: profile,
            recordings: [RecordingItem(manifest: manifest, directory: directory)]
        ) { _, _ in }

        XCTAssertGreaterThanOrEqual(dataset.count, 3)
        XCTAssertEqual(dataset.action(at: 0)[ActionLayout.absoluteMouse.lowerBound], 0.5, accuracy: 0.000_1)
        XCTAssertEqual(dataset.action(at: 1)[ActionLayout.absoluteMouse.lowerBound], 0.5, accuracy: 0.000_1)
        XCTAssertEqual(dataset.action(at: 2)[ActionLayout.absoluteMouse.lowerBound], 0.9, accuracy: 0.000_1)
    }

    func testSparseStaticVideoIsSampledAtConfiguredPerceptionCadence() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("static-video-cache-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()
        let recordingID = UUID()
        let directory = try await store.createRecordingDirectory(id: recordingID)
        try await writeTestMovie(to: directory.appendingPathComponent("capture.mov"), width: 16, height: 16, frameCount: 2, fps: 1)
        let eventWriter = try InputEventWriter(url: directory.appendingPathComponent("events.atrevents"))
        eventWriter.append(InputSample(timestampNanos: 1_000_000_000, kind: .mouseMove, x: 8, y: 8))
        let eventCount = try eventWriter.finish()
        let manifest = RecordingManifest(
            id: recordingID, name: "Static", createdAt: Date(), hostStartNanos: 1_000_000_000, duration: 1,
            capture: CaptureSpec(requestedFPS: 1), globalRect: CodableRect(CGRect(x: 0, y: 0, width: 16, height: 16)),
            pixelWidth: 16, pixelHeight: 16, deliveredFPS: 1, eventCount: eventCount
        )
        try await store.writeRecording(manifest, to: directory)
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 8, height: 8, colorMode: .grayscale)
        profile.training.perceptionFPS = 10
        profile.training.actionFPS = 20
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 2, frameSpacing: 1, downsampleFactor: 2)

        let dataset = try await DatasetCacheBuilder(workspace: store).cache(
            for: profile,
            recordings: [RecordingItem(manifest: manifest, directory: directory)]
        ) { _, _ in }
        XCTAssertGreaterThanOrEqual(dataset.manifest.observationCount, 10)
        XCTAssertGreaterThan(dataset.count, dataset.manifest.observationCount)
    }

    func testPositiveClassWeightsUseOnlyRequestedRowsAndRespectRestrictions() throws {
        var actionRows: [[Float]] = []
        for row in 0..<4 {
            var values = [Float](repeating: 0, count: ActionLayout.count)
            if row == 0 { values[ActionLayout.keyboard.lowerBound + 13] = 1 }
            actionRows.append(values)
        }
        let fixture = try makeSyntheticDataset(
            name: "weights",
            pastFrameCount: 1,
            sequences: Array(repeating: [0, .max], count: 4),
            actionRows: actionRows
        )
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let dataset = fixture.dataset
        let weights = dataset.positiveClassWeights(at: [0, 1, 2, 3], restrictions: ActionRestrictions())
        XCTAssertEqual(weights[ActionLayout.keyboard.lowerBound + 13], 3)
        XCTAssertEqual(weights[ActionLayout.keyboard.lowerBound + 12], 0)
        XCTAssertEqual(dataset.demonstratedKeyCodes(at: [0, 1, 2, 3]), [13])
        let blocked = dataset.positiveClassWeights(at: [0, 1, 2, 3], restrictions: ActionRestrictions(blockedKeyCodes: [13]))
        XCTAssertEqual(blocked[ActionLayout.keyboard.lowerBound + 13], 0)
    }

    func testValidationSplitNeverRemovesTheOnlyTrainingExampleOfAControl() throws {
        let segments = (0..<3).map { CacheSegment(recordingID: UUID(), start: $0 * 2, count: 2) }
        var actionRows: [[Float]] = []
        for row in 0..<6 {
            var values = [Float](repeating: 0, count: ActionLayout.count)
            if row == 0 { values[ActionLayout.keyboard.lowerBound + 10] = 1 }
            if row == 2 || row == 4 { values[ActionLayout.keyboard.lowerBound + 13] = 1 }
            actionRows.append(values)
        }
        let fixture = try makeSyntheticDataset(
            name: "split",
            pastFrameCount: 1,
            sequences: Array(repeating: [0, .max], count: 6),
            actionRows: actionRows,
            segments: segments
        )
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let dataset = fixture.dataset
        let representatives = dataset.representativeValidationIndices(from: Array(0..<6), limit: 2)
        XCTAssertEqual(Set(representatives), [0, 2], "Rare positive controls must displace easy zero-only validation rows")
        let split = TrainingEngine().splitIndices(dataset: dataset, fraction: 0.9, seed: 42)
        XCTAssertFalse(split.validation.isEmpty)
        XCTAssertFalse(split.train.isEmpty)
        let trainedKeys = dataset.demonstratedKeyCodes(at: split.train)
        XCTAssertEqual(trainedKeys, [10, 13])
        XCTAssertEqual(dataset.demonstratedKeyCodes(at: split.validation), [13])
    }

    func testWholeRecordingValidationTargetsSamplesAndCoversEverySegment() throws {
        let counts = [2, 10, 50]
        var start = 0
        let segments = counts.map { count -> CacheSegment in
            defer { start += count }
            return CacheSegment(recordingID: UUID(), start: start, count: count)
        }
        let rowCount = counts.reduce(0, +)
        let sequences = Array(repeating: [UInt32(0), .max], count: rowCount)
        let rows = Array(repeating: [Float](repeating: 0, count: ActionLayout.count), count: rowCount)
        let fixture = try makeSyntheticDataset(
            name: "sample-balanced-split",
            pastFrameCount: 1,
            sequences: sequences,
            actionRows: rows,
            segments: segments
        )
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let split = TrainingEngine().splitIndices(dataset: fixture.dataset, fraction: 0.34, seed: 42)
        XCTAssertEqual(split.validation, Array(2..<12), "The recording closest to the requested sample fraction should be held out, not a random huge recording.")
        XCTAssertEqual(split.train.count, 52)

        let representatives = fixture.dataset.representativeValidationIndices(from: Array(0..<rowCount), limit: 6)
        for segment in segments {
            XCTAssertTrue(representatives.contains { (segment.start..<(segment.start + segment.count)).contains($0) }, "Every recording should influence a sufficiently large representative score.")
        }
    }

    func testSingleRecordingValidationPurgesSharedTemporalFrames() throws {
        let sequences: [[UInt32]] = [
            [0, .max, .max, .max], [0, .max, .max, .max], [0, .max, .max, .max],
            [1, .max, .max, 0], [1, .max, .max, 0], [1, .max, .max, 0], [1, .max, .max, 0], [1, .max, .max, 0],
            [2, .max, 0, 1], [2, .max, 0, 1], [2, .max, 0, 1], [2, .max, 0, 1],
            [2, .max, 0, 1], [2, .max, 0, 1], [2, .max, 0, 1], [2, .max, 0, 1],
            [3, 0, 1, 2], [4, 1, 2, 3], [6, 3, 4, 5], [7, 4, 5, 6]
        ]
        let rows = Array(repeating: [Float](repeating: 0, count: ActionLayout.count), count: sequences.count)
        let fixture = try makeSyntheticDataset(name: "purged-validation", pastFrameCount: 3, sequences: sequences, actionRows: rows)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let split = TrainingEngine().splitIndices(dataset: fixture.dataset, fraction: 0.25, seed: 42)
        XCTAssertEqual(split.train, Array(0..<15))
        XCTAssertEqual(split.validation, [18, 19])
        XCTAssertTrue(Set(split.train).isDisjoint(with: split.validation))
        XCTAssertTrue(
            fixture.dataset.firstDisjointValidationIndex(trainingEnd: 15, proposedStart: 15) == 18,
            "Validation may begin only when every selected frame is newer than every training frame."
        )
    }

    func testValidationAvailabilityIgnoresBlockedControls() throws {
        let sequences = (0..<20).map { row -> [UInt32] in
            let current = UInt32(row)
            let past = (1...3).reversed().map { distance -> UInt32 in
                row >= distance ? UInt32(row - distance) : .max
            }
            return [current] + past
        }
        var rows = Array(repeating: [Float](repeating: 0, count: ActionLayout.count), count: sequences.count)
        rows[18][ActionLayout.keyboard.lowerBound + 13] = 1
        let fixture = try makeSyntheticDataset(name: "blocked-validation", pastFrameCount: 3, sequences: sequences, actionRows: rows)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let unrestricted = TrainingEngine().splitIndices(dataset: fixture.dataset, fraction: 0.25, seed: 42)
        XCTAssertTrue(unrestricted.validation.isEmpty, "The only learnable key example must remain in training")

        let blocked = TrainingEngine().splitIndices(
            dataset: fixture.dataset,
            fraction: 0.25,
            seed: 42,
            channels: .all,
            restrictions: ActionRestrictions(blockedKeyCodes: [13])
        )
        XCTAssertEqual(blocked.train, Array(0..<15))
        XCTAssertEqual(blocked.validation, [18, 19], "A blocked output must not erase otherwise honest validation data")
        let representatives = fixture.dataset.representativeValidationIndices(
            from: Array(15..<20),
            limit: 1,
            channels: .all,
            restrictions: ActionRestrictions(blockedKeyCodes: [13])
        )
        XCTAssertNotEqual(representatives, [18], "Blocked positives must not consume the representative-validation budget")
    }

    func testSalienceBalancedTrainingOrderSpreadsTransitionsWithoutResampling() throws {
        let sequences = (0..<12).map { row -> [UInt32] in
            [UInt32(row), row > 0 ? UInt32(row - 1) : .max]
        }
        var rows = Array(repeating: [Float](repeating: 0, count: ActionLayout.count), count: sequences.count)
        rows[1][ActionLayout.keyboard.lowerBound + 13] = 1
        rows[7][ActionLayout.scroll.lowerBound] = 0.5
        let fixture = try makeSyntheticDataset(name: "balanced-order", pastFrameCount: 1, sequences: sequences, actionRows: rows)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        var profile = AIProfile.fresh()
        profile.channels = ActionChannels(absoluteMouse: false, relativeMouse: false, buttons: false, scroll: true, keyboard: true, modifiers: false)
        let indices = Array(0..<sequences.count)
        let salient = fixture.dataset.salientTrainingIndices(at: indices, channels: profile.channels, restrictions: profile.effectiveRestrictions)
        XCTAssertEqual(salient, [1, 2, 7])

        let engine = TrainingEngine()
        let order = engine.trainingOrder(dataset: fixture.dataset, indices: indices, batchSize: 4, seed: 91, profile: profile)
        XCTAssertEqual(order, engine.trainingOrder(dataset: fixture.dataset, indices: indices, batchSize: 4, seed: 91, profile: profile), "Exact resume requires deterministic epoch order")
        XCTAssertEqual(Set(order), Set(indices))
        XCTAssertEqual(order.count, indices.count)
        for start in stride(from: 0, to: order.count, by: 4) {
            XCTAssertFalse(Set(order[start..<min(order.count, start + 4)]).isDisjoint(with: salient), "Every batch should receive a high-signal row when enough exist")
        }
    }

    func testCorruptDatasetCacheSizesThrowInsteadOfOverflowing() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("overflow-cache-\(UUID().uuidString).atrcache", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let spec = PreprocessingSpec(width: 1, height: 1, colorMode: .grayscale, bitDepth: 8)
        let temporal = TemporalVisionConfiguration(pastFrameCount: 1, frameSpacing: 1, downsampleFactor: 1)
        let manifest = DatasetCacheManifest(
            key: "invalid", createdAt: Date(), preprocessing: spec,
            pastPreprocessing: spec, temporalVision: temporal,
            actionFPS: 60, perceptionFPS: 30,
            sampleCount: Int.max, observationCount: 1,
            currentObservationBytesPerSample: 1, pastObservationBytesPerSample: 1,
            actionValuesPerSample: ActionLayout.count,
            segments: [CacheSegment(recordingID: UUID(), start: 0, count: Int.max)]
        )
        let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(manifest).write(to: directory.appendingPathComponent("manifest.json"))
        try Data().write(to: directory.appendingPathComponent("current-observations.bin"))
        try Data().write(to: directory.appendingPathComponent("past-observations.bin"))
        try Data().write(to: directory.appendingPathComponent("observation-indices.bin"))
        try Data().write(to: directory.appendingPathComponent("frame-actions.bin"))
        try Data().write(to: directory.appendingPathComponent("actions.bin"))
        XCTAssertThrowsError(try CachedDataset(directory: directory))
    }

    func testCachedDatasetRejectsOutOfRangeObservationMappings() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("mapping-cache-\(UUID().uuidString).atrcache", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let spec = PreprocessingSpec(width: 1, height: 1, colorMode: .grayscale)
        let temporal = TemporalVisionConfiguration(pastFrameCount: 1, frameSpacing: 1, downsampleFactor: 1)
        let manifest = DatasetCacheManifest(
            key: "mapping", createdAt: Date(), preprocessing: spec,
            pastPreprocessing: spec, temporalVision: temporal,
            actionFPS: 60, perceptionFPS: 30,
            sampleCount: 1, observationCount: 1,
            currentObservationBytesPerSample: 1, pastObservationBytesPerSample: 1,
            actionValuesPerSample: ActionLayout.count,
            segments: [CacheSegment(recordingID: UUID(), start: 0, count: 1)]
        )
        let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(manifest).write(to: directory.appendingPathComponent("manifest.json"))
        try Data([0]).write(to: directory.appendingPathComponent("current-observations.bin"))
        try Data([0]).write(to: directory.appendingPathComponent("past-observations.bin"))
        try observationSequenceData([[1, .max]]).write(to: directory.appendingPathComponent("observation-indices.bin"))
        try Data(count: ActionLayout.count * MemoryLayout<Float>.size).write(to: directory.appendingPathComponent("frame-actions.bin"))
        try Data(count: ActionLayout.count * MemoryLayout<Float>.size).write(to: directory.appendingPathComponent("actions.bin"))
        XCTAssertThrowsError(try CachedDataset(directory: directory))
    }

    func testBlockingEitherModifierSideBlocksModifierChannel() {
        var restrictions = ActionRestrictions()
        XCTAssertTrue(restrictions.allowsModifier(0))
        restrictions.blockedKeyCodes.insert(60) // right shift
        XCTAssertFalse(restrictions.allowsModifier(0))
        restrictions.blockedKeyCodes = [61] // right option
        XCTAssertFalse(restrictions.allowsModifier(2))
        XCTAssertTrue(restrictions.allowsModifier(3))
    }

    func testPolicyForwardAndGradient() throws {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 32, height: 24, colorMode: .grayscale, bitDepth: 8)
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 2, frameSpacing: 2, downsampleFactor: 2)
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        let inputs = temporalModelInputs(profile: profile, batch: 2, value: 0.5)
        let targets = MLXArray([Float](repeating: 0, count: 2 * ActionLayout.count), [2, ActionLayout.count])
        let gradient = valueAndGrad(model: model) { model, arrays in
            [model.loss(currentImages: arrays[0], pastImages: arrays[1], pastControls: arrays[2], targets: arrays[3])]
        }
        let result = gradient(model, [inputs.current, inputs.past, inputs.controls, targets])
        MLX.eval(result.0, result.1)
        XCTAssertEqual(
            model.predictions(currentImages: inputs.current, pastImages: inputs.past, pastControls: inputs.controls).shape,
            [2, ActionLayout.count]
        )
        XCTAssertTrue(result.0[0].item(Float.self).isFinite)
    }

    func testSharedValidationForwardPreservesLossAndPredictions() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 32, height: 24, colorMode: .grayscale, bitDepth: 8)
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 2, frameSpacing: 2, downsampleFactor: 2)
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        model.train(false)
        let inputs = temporalModelInputs(profile: profile, batch: 2, value: 0.375)
        var targetValues = [Float](repeating: 0, count: 2 * ActionLayout.count)
        targetValues[ActionLayout.keyboard.lowerBound + 13] = 1
        targetValues[ActionLayout.count + ActionLayout.buttons.lowerBound] = 1
        let targets = MLXArray(targetValues, [2, ActionLayout.count])
        let weights = MLXArray.ones([ActionLayout.count])

        let expectedLoss = model.loss(
            currentImages: inputs.current,
            pastImages: inputs.past,
            pastControls: inputs.controls,
            targets: targets,
            positiveWeights: weights
        )
        let expectedPredictions = model.predictions(
            currentImages: inputs.current,
            pastImages: inputs.past,
            pastControls: inputs.controls
        )
        let logits = model.callAsFunction(
            currentImages: inputs.current,
            pastImages: inputs.past,
            pastControls: inputs.controls
        )
        let sharedLoss = model.loss(logits: logits, targets: targets, positiveWeights: weights)
        let sharedPredictions = model.activatedPredictions(logits: logits)
        MLX.eval(expectedLoss, expectedPredictions, sharedLoss, sharedPredictions)

        XCTAssertEqual(sharedLoss.item(Float.self), expectedLoss.item(Float.self))
        XCTAssertEqual(sharedPredictions.asArray(Float.self), expectedPredictions.asArray(Float.self))
    }

    func testCachedTemporalEmbeddingsExactlyMatchImageHistoryInference() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 32, height: 24, colorMode: .grayscale)
        profile.training.temporalVision = TemporalVisionConfiguration(
            pastFrameCount: 3,
            frameSpacing: 2,
            downsampleFactor: 2
        )
        profile.training.architecture = .small
        profile.training.precision = .float32
        profile.training.generalization = .disabled
        let model = AgentPolicy(profile: profile)
        model.train(false)
        let inputs = temporalModelInputs(profile: profile, batch: 2, value: 0.375)
        let imagePredictions = model.predictions(
            currentImages: inputs.current,
            pastImages: inputs.past,
            pastControls: inputs.controls
        )

        let temporal = profile.training.effectiveTemporalVision
        let pastSpec = temporal.pastFrameSpec(from: profile.preprocessing)
        let currentEmbedding = model.visualEmbedding(
            visualFeatures: model.visualActivations(images: inputs.current).last!
        )
        let flattenedPast = inputs.past.reshaped([
            2 * temporal.pastFrameCount,
            pastSpec.height,
            pastSpec.width,
            pastSpec.channelCount
        ])
        let cachedPast = model.pastVisualEmbedding(
            visualFeatures: model.pastVisualActivations(images: flattenedPast).last!
        ).reshaped([
            2,
            temporal.pastFrameCount,
            profile.training.architecture.visualEmbedding
        ])
        let cachedPredictions = model.activatedPredictions(logits: model.logits(
            temporalFeatures: model.temporalFeatures(
                currentVisualEmbedding: currentEmbedding,
                pastVisualEmbeddings: cachedPast,
                pastControls: inputs.controls
            )
        ))
        MLX.eval(imagePredictions, cachedPredictions)
        XCTAssertEqual(imagePredictions.asArray(Float.self), cachedPredictions.asArray(Float.self))
    }

    func testUnavailableTrainingHistoryExactlyMatchesRuntimeZeroHistory() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(
            width: 16,
            height: 12,
            colorMode: .grayscale
        )
        profile.training.temporalVision = TemporalVisionConfiguration(
            pastFrameCount: 2,
            frameSpacing: 1,
            downsampleFactor: 2
        )
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        profile.training.generalization = .disabled
        let model = AgentPolicy(profile: profile)
        model.train(false)

        let embeddingWidth = profile.training.architecture.visualEmbedding
        let current = MLXRandom.uniform(low: -1, high: 1, [1, embeddingWidth])
        let substitutedPast = MLXRandom.uniform(low: -1, high: 1, [1, 2, embeddingWidth])
        let substitutedControls = MLXArray.ones([1, 2, ActionLayout.count])
        let masked = model.temporalFeatures(
            currentVisualEmbedding: current,
            pastVisualEmbeddings: substitutedPast,
            pastControls: substitutedControls,
            pastFrameMask: MLXArray.zeros([1, 2, 1])
        )
        let runtimeEquivalent = model.temporalFeatures(
            currentVisualEmbedding: current,
            pastVisualEmbeddings: MLXArray.zeros([1, 2, embeddingWidth]),
            pastControls: MLXArray.zeros([1, 2, ActionLayout.count])
        )
        MLX.eval(masked, runtimeEquivalent)

        let maximumDelta = zip(
            masked.asArray(Float.self),
            runtimeEquivalent.asArray(Float.self)
        ).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        XCTAssertLessThanOrEqual(maximumDelta, 0.000_001)
    }

    func testFinalOnlyRecurrentStatesMatchMLXOutputsAndGradients() {
        MLXRandom.seed(88_200)
        let input = MLXRandom.uniform(low: -1, high: 1, [3, 5, 11])
        let selector = MLXRandom.uniform(low: -1, high: 1, [3, 7])

        let gru = GRU(inputSize: 11, hiddenSize: 7)
        let expectedGRU = gru(input)[.ellipsis, -1, 0...]
        let actualGRU = AgentPolicy.finalGRUState(input, gru: gru)
        let expectedGRUGradient = valueAndGrad(model: gru) { model, values in
            [(model(values[0])[.ellipsis, -1, 0...] * values[1]).sum()]
        }(gru, [input, selector]).1
        let actualGRUGradient = valueAndGrad(model: gru) { model, values in
            [(AgentPolicy.finalGRUState(values[0], gru: model) * values[1]).sum()]
        }(gru, [input, selector]).1

        let lstm = LSTM(inputSize: 11, hiddenSize: 7)
        let expectedLSTM = lstm(input).0[.ellipsis, -1, 0...]
        let actualLSTM = AgentPolicy.finalLSTMState(input, lstm: lstm)
        let expectedLSTMGradient = valueAndGrad(model: lstm) { model, values in
            [(model(values[0]).0[.ellipsis, -1, 0...] * values[1]).sum()]
        }(lstm, [input, selector]).1
        let actualLSTMGradient = valueAndGrad(model: lstm) { model, values in
            [(AgentPolicy.finalLSTMState(values[0], lstm: model) * values[1]).sum()]
        }(lstm, [input, selector]).1
        MLX.eval(
            expectedGRU, actualGRU, expectedGRUGradient, actualGRUGradient,
            expectedLSTM, actualLSTM, expectedLSTMGradient, actualLSTMGradient
        )

        func maximumDelta(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
            zip(lhs.asArray(Float.self), rhs.asArray(Float.self)).reduce(0) {
                max($0, abs($1.0 - $1.1))
            }
        }
        func maximumGradientDelta(
            _ lhs: ModuleParameters,
            _ rhs: ModuleParameters
        ) -> Float {
            let expected = Dictionary(uniqueKeysWithValues: lhs.flattened())
            let actual = Dictionary(uniqueKeysWithValues: rhs.flattened())
            XCTAssertEqual(expected.keys.sorted(), actual.keys.sorted())
            return expected.keys.reduce(0) { maximum, name in
                max(maximum, maximumDelta(expected[name]!, actual[name]!))
            }
        }
        XCTAssertLessThanOrEqual(maximumDelta(expectedGRU, actualGRU), 0.000_001)
        XCTAssertLessThanOrEqual(
            maximumGradientDelta(expectedGRUGradient, actualGRUGradient),
            0.000_001
        )
        XCTAssertLessThanOrEqual(maximumDelta(expectedLSTM, actualLSTM), 0.000_001)
        XCTAssertLessThanOrEqual(
            maximumGradientDelta(expectedLSTMGradient, actualLSTMGradient),
            0.000_001
        )
    }

    func testAcceleratedGroupNormMatchesEstablishedGroupingAndGradient() {
        MLXRandom.seed(88_201)
        let normalization = GroupNorm(groupCount: 8, dimensions: 32)
        let input = MLXRandom.uniform(low: -2, high: 2, [2, 9, 13, 32])
        let selector = MLXRandom.uniform(low: -1, high: 1, input.shape)
        let expected = normalization(input)
        let accelerated = AgentPolicy.acceleratedGroupNorm(input, normalization: normalization)
        let expectedGradient = grad { value in (normalization(value) * selector).sum() }(input)
        let acceleratedGradient = grad {
            value in (AgentPolicy.acceleratedGroupNorm(value, normalization: normalization) * selector).sum()
        }(input)
        MLX.eval(expected, accelerated, expectedGradient, acceleratedGradient)

        let outputDelta = zip(expected.asArray(Float.self), accelerated.asArray(Float.self))
            .reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        let gradientDelta = zip(expectedGradient.asArray(Float.self), acceleratedGradient.asArray(Float.self))
            .reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        XCTAssertLessThanOrEqual(outputDelta, 0.000_002)
        XCTAssertLessThanOrEqual(gradientDelta, 0.000_002)
    }

    func testVisionAccelerationUsesTheMeasuredWorkloadCrossover() {
        XCTAssertFalse(AgentPolicy.usesAcceleratedVisionPath(batch: 16, width: 128, height: 72))
        XCTAssertTrue(AgentPolicy.usesAcceleratedVisionPath(batch: 1, width: 640, height: 360))
        XCTAssertTrue(AgentPolicy.usesAcceleratedVisionPath(batch: 32, width: 640, height: 360))
    }

    func testSharedCoordinateStemMatchesBatchedConvolutionAndWeightGradient() {
        MLXRandom.seed(88_203)
        let batch = 4, height = 31, width = 47, contentChannels = 6, outputChannels = 32
        let input = MLXRandom.uniform(low: -1, high: 1, [batch, height, width, contentChannels])
        let coordinates = MLXRandom.uniform(low: -1, high: 1, [1, height, width, 2])
        let convolution = Conv2d(
            inputChannels: contentChannels + 2,
            outputChannels: outputChannels,
            kernelSize: 7,
            stride: 4,
            padding: 3,
            bias: false
        )
        let expected = convolution(concatenated([
            input,
            broadcast(coordinates, to: [batch, height, width, 2])
        ], axis: -1))
        let actual = AgentPolicy.sharedCoordinateConvolution(
            input,
            coordinates: coordinates,
            convolution: convolution
        )
        let selector = MLXRandom.uniform(low: -1, high: 1, expected.shape)
        let expectedValueAndGradient = valueAndGrad(model: convolution) { model, arrays in
            let batchedCoordinates = broadcast(coordinates, to: [batch, height, width, 2])
            return [(model(concatenated([arrays[0], batchedCoordinates], axis: -1)) * selector).sum()]
        }(convolution, [input])
        let actualValueAndGradient = valueAndGrad(model: convolution) { model, arrays in
            [(
                AgentPolicy.sharedCoordinateConvolution(
                    arrays[0],
                    coordinates: coordinates,
                    convolution: model
                ) * selector
            ).sum()]
        }(convolution, [input])
        MLX.eval(
            expected,
            actual,
            expectedValueAndGradient.0,
            expectedValueAndGradient.1,
            actualValueAndGradient.0,
            actualValueAndGradient.1
        )

        let outputDelta = zip(expected.asArray(Float.self), actual.asArray(Float.self))
            .reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        let expectedGradients = expectedValueAndGradient.1.flattened().map(\.1)
        let actualGradients = actualValueAndGradient.1.flattened().map(\.1)
        let gradientDelta = zip(expectedGradients, actualGradients).reduce(Float(0)) { maximum, pair in
            let delta = zip(pair.0.asArray(Float.self), pair.1.asArray(Float.self))
                .reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
            return max(maximum, delta)
        }
        XCTAssertLessThanOrEqual(outputDelta, 0.000_01)
        XCTAssertLessThanOrEqual(gradientDelta, 0.000_1)
    }

    func testDeduplicatedTemporalVisionPreservesValidationForwardPass() {
        MLXRandom.seed(88_204)
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(
            width: 32,
            height: 24,
            colorMode: .grayscale
        )
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 2, frameSpacing: 1, downsampleFactor: 2)
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0.15
        profile.training.generalization = .disabled
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        model.train(false)
        let uniqueInputs = temporalModelInputs(profile: profile, batch: 2, value: 0.25)
        let uniqueImages = MLXRandom.uniform(low: -1, high: 1, uniqueInputs.current.shape)
        let generatedPast = MLXRandom.uniform(low: -1, high: 1, uniqueInputs.past.shape)
        let pastSpec = profile.training.effectiveTemporalVision.pastFrameSpec(from: profile.preprocessing)
        let flattenedPast = generatedPast.reshaped([
            4, pastSpec.height, pastSpec.width, pastSpec.channelCount
        ])
        let uniquePast = flattenedPast.take(MLXArray([Int32(0), 1, 3]), axis: 0)
        let visionToPast = MLXArray([Int32(0), 1, 1, 2], [2, 2])
        let sequencePast = uniquePast.take(visionToPast.flattened(), axis: 0).reshaped([
            2, 2, pastSpec.height, pastSpec.width, pastSpec.channelCount
        ])
        let mapping = MLXArray([Int32(0), 0, 1, 1])
        let fullImages = uniqueImages.take(mapping, axis: 0)
        let fullPast = sequencePast.take(mapping, axis: 0)
        let fullControls = uniqueInputs.controls.take(mapping, axis: 0)
        var targetValues = [Float](repeating: 0, count: 4 * ActionLayout.count)
        for row in 0..<4 {
            targetValues[row * ActionLayout.count] = Float(row) / 3
            targetValues[row * ActionLayout.count + ActionLayout.keyboard.lowerBound + row] = 1
        }
        let targets = MLXArray(targetValues, [4, ActionLayout.count])
        let previousTargets = MLXArray.zeros(like: targets)
        let classWeights = MLXArray.ones([ActionLayout.count])

        let fullForward = model.logits(temporalFeatures: model.temporalFeatures(
            currentImages: fullImages,
            pastImages: fullPast,
            pastControls: fullControls
        ))
        let reusedForward = model.logits(temporalFeatures: model.temporalFeatures(
            currentImages: uniqueImages,
            pastImages: uniquePast,
            pastControls: uniqueInputs.controls,
            visionToPast: visionToPast,
            sampleToVision: mapping
        ))
        MLX.eval(fullForward, reusedForward)
        let forwardDelta = zip(fullForward.asArray(Float.self), reusedForward.asArray(Float.self))
            .reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        let fullLoss = model.loss(
            logits: fullForward,
            targets: targets,
            positiveWeights: classWeights,
            previousTargets: previousTargets
        )
        let reusedLoss = model.loss(
            logits: reusedForward,
            targets: targets,
            positiveWeights: classWeights,
            previousTargets: previousTargets
        )
        MLX.eval(fullLoss, reusedLoss)

        XCTAssertLessThanOrEqual(forwardDelta, 0.000_01)
        XCTAssertEqual(fullLoss.item(Float.self), reusedLoss.item(Float.self), accuracy: 0.000_01)
    }

    func testAcceleratedBFloat16PolicyForwardStaysWithinStoragePrecision() {
        MLXRandom.seed(88_205)
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 32, height: 24, colorMode: .grayscale)
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 1, frameSpacing: 1, downsampleFactor: 2)
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .bfloat16
        let model = AgentPolicy(profile: profile)
        model.train(false)
        let inputs = temporalModelInputs(profile: profile, batch: 3, value: 0.25)
        let images = MLXRandom.uniform(low: -1, high: 1, inputs.current.shape).asType(.bfloat16)
        let past = inputs.past.asType(.bfloat16)
        let controls = inputs.controls.asType(.bfloat16)
        let legacyLogits = model.logits(
            currentVisualFeatures: model.visualActivations(images: images, acceleratedOperators: false).last!,
            pastImages: past,
            pastControls: controls
        )
        let acceleratedLogits = model.logits(
            currentVisualFeatures: model.visualActivations(images: images, acceleratedOperators: true).last!,
            pastImages: past,
            pastControls: controls
        )
        let legacyPredictions = model.activatedPredictions(logits: legacyLogits)
        let acceleratedPredictions = model.activatedPredictions(logits: acceleratedLogits)
        MLX.eval(
            legacyLogits,
            acceleratedLogits,
            legacyPredictions,
            acceleratedPredictions
        )

        let outputDelta = zip(legacyLogits.asArray(Float.self), acceleratedLogits.asArray(Float.self))
            .reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        let predictionDelta = zip(
            legacyPredictions.asArray(Float.self),
            acceleratedPredictions.asArray(Float.self)
        ).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        XCTAssertLessThanOrEqual(outputDelta, 0.03125)
        XCTAssertLessThanOrEqual(predictionDelta, 0.02)
    }

    func testCombinedActionHeadsMatchSeparateAffineHeadsAndGradients() {
        MLXRandom.seed(731_904)
        var profile = AIProfile.fresh()
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        model.train(false)
        let featureWidth = profile.training.architecture.visualEmbedding
            + profile.training.architecture.recurrentWidth
        let features = MLXRandom.uniform(low: -1, high: 1, [7, featureWidth])
        let selector = MLXRandom.uniform(low: -1, high: 1, [7, ActionLayout.count])

        let combined = model.logits(temporalFeatures: features)
        let reference = model.referenceLogits(temporalFeatures: features)
        let combinedGradient = valueAndGrad(model: model) { candidate, arrays in
            [(candidate.logits(temporalFeatures: arrays[0]) * arrays[1]).sum()]
        }(model, [features, selector]).1
        let referenceGradient = valueAndGrad(model: model) { candidate, arrays in
            [(candidate.referenceLogits(temporalFeatures: arrays[0]) * arrays[1]).sum()]
        }(model, [features, selector]).1
        MLX.eval(combined, reference, combinedGradient, referenceGradient)

        let outputDelta = zip(combined.asArray(Float.self), reference.asArray(Float.self))
            .reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        XCTAssertLessThanOrEqual(outputDelta, 0.000_01)
        let expectedGradients = Dictionary(uniqueKeysWithValues: referenceGradient.flattened())
        let actualGradients = Dictionary(uniqueKeysWithValues: combinedGradient.flattened())
        XCTAssertEqual(expectedGradients.keys.sorted(), actualGradients.keys.sorted())
        for name in expectedGradients.keys {
            let expected = expectedGradients[name]!.asArray(Float.self)
            let actual = actualGradients[name]!.asArray(Float.self)
            let delta = zip(expected, actual).reduce(Float(0)) {
                max($0, abs($1.0 - $1.1))
            }
            XCTAssertLessThanOrEqual(delta, 0.000_01, name)
        }
    }

    func testTemporalHistoryCorruptionIsIndependentOfFeatureDropout() {
        MLXRandom.seed(7_015)
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 12, height: 8, colorMode: .grayscale, bitDepth: 8)
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 1, frameSpacing: 1, downsampleFactor: 2)
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.generalization = GeneralizationConfiguration(
            visionAugmentationStrength: 0,
            randomErasingProbability: 0,
            controlHistoryDropout: 0.5,
            temporalFrameDropout: 0,
            binaryLabelSmoothing: 0
        )
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        let batch = 32
        var controlValues = [Float](repeating: 0, count: batch * ActionLayout.count)
        for row in 0..<batch { controlValues[row * ActionLayout.count + ActionLayout.keyboard.lowerBound + 13] = 1 }
        let controls = MLXArray(controlValues, [batch, 1, ActionLayout.count])
        let embeddings = MLXArray.ones([
            batch,
            1,
            profile.training.architecture.visualEmbedding
        ])

        model.train(false)
        let inference = model.regularizedTemporalHistory(
            pastVisualEmbeddings: embeddings,
            pastControls: controls
        )
        MLX.eval(inference.embeddings, inference.controls)
        XCTAssertEqual(inference.embeddings.asArray(Float.self), embeddings.asArray(Float.self))
        XCTAssertEqual(inference.controls.asArray(Float.self), controls.asArray(Float.self))

        model.train(true)
        let training = model.regularizedTemporalHistory(
            pastVisualEmbeddings: embeddings,
            pastControls: controls
        )
        MLX.eval(training.embeddings, training.controls)
        let trainingControlValues = training.controls.asArray(Float.self)
        let keyValues = stride(
            from: ActionLayout.keyboard.lowerBound + 13,
            to: batch * ActionLayout.count,
            by: ActionLayout.count
        ).map { trainingControlValues[$0] }
        XCTAssertTrue(keyValues.contains(0), "Historical controls must sometimes be hidden even when feature dropout is zero.")
        XCTAssertTrue(keyValues.contains(1), "History corruption must preserve some exact evidence without dropout rescaling.")
        for duplicateIndex in ActionLayout.commandOptionControlKeyboardIndices {
            let values = stride(
                from: duplicateIndex,
                to: batch * ActionLayout.count,
                by: ActionLayout.count
            ).map { trainingControlValues[$0] }
            XCTAssertTrue(
                values.allSatisfy { $0 == 0 },
                "History corruption must never synthesize a duplicate modifier-key path."
            )
        }
        XCTAssertEqual(training.embeddings.asArray(Float.self), embeddings.asArray(Float.self))
    }

    func testVisionAugmentationIsStochasticOnlyDuringTraining() {
        MLXRandom.seed(91_004)
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 24, height: 16, colorMode: .color)
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 0)
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        profile.training.generalization = GeneralizationConfiguration(
            visionAugmentationStrength: 0.25,
            randomErasingProbability: 0.5,
            controlHistoryDropout: 0,
            temporalFrameDropout: 0,
            binaryLabelSmoothing: 0
        )
        let model = AgentPolicy(profile: profile)
        let images = MLXRandom.uniform(low: 0, high: 1, [2, 16, 24, 3])

        model.train(false)
        let inferenceA = model.currentOnlyPredictions(currentImages: images)
        let inferenceB = model.currentOnlyPredictions(currentImages: images)
        MLX.eval(inferenceA, inferenceB)
        XCTAssertEqual(inferenceA.asArray(Float.self), inferenceB.asArray(Float.self))

        model.train(true)
        let trainingA = model.currentOnlyPredictions(currentImages: images)
        let trainingB = model.currentOnlyPredictions(currentImages: images)
        MLX.eval(trainingA, trainingB)
        let first = trainingA.asArray(Float.self)
        let second = trainingB.asArray(Float.self)
        XCTAssertTrue(zip(first, second).contains { abs($0 - $1) > 0.000_001 })
        XCTAssertTrue(first.allSatisfy(\.isFinite))
        XCTAssertTrue(second.allSatisfy(\.isFinite))
    }

    func testBinaryLabelSmoothingIsTrainingOnly() {
        var smoothProfile = AIProfile.fresh()
        smoothProfile.preprocessing = PreprocessingSpec(width: 8, height: 8, colorMode: .grayscale)
        smoothProfile.channels = ActionChannels(
            absoluteMouse: false,
            relativeMouse: false,
            buttons: true,
            scroll: false,
            keyboard: false,
            modifiers: false
        )
        smoothProfile.training.architecture = .small
        smoothProfile.training.generalization = GeneralizationConfiguration(
            visionAugmentationStrength: 0,
            randomErasingProbability: 0,
            controlHistoryDropout: 0,
            temporalFrameDropout: 0,
            binaryLabelSmoothing: 0.1
        )
        var exactProfile = smoothProfile
        exactProfile.training.generalization = .disabled
        let smooth = AgentPolicy(profile: smoothProfile)
        let exact = AgentPolicy(profile: exactProfile)
        let logits = MLXArray([Float](repeating: -10, count: ActionLayout.count), [1, ActionLayout.count])
        let targets = MLXArray.zeros([1, ActionLayout.count])

        smooth.train(true)
        exact.train(true)
        let smoothedTrainingLoss = smooth.loss(logits: logits, targets: targets)
        let exactTrainingLoss = exact.loss(logits: logits, targets: targets)
        smooth.train(false)
        let smoothedValidationLoss = smooth.loss(logits: logits, targets: targets)
        MLX.eval(smoothedTrainingLoss, exactTrainingLoss, smoothedValidationLoss)
        XCTAssertGreaterThan(smoothedTrainingLoss.item(Float.self), exactTrainingLoss.item(Float.self))
        XCTAssertEqual(
            smoothedValidationLoss.item(Float.self),
            exactTrainingLoss.item(Float.self),
            accuracy: 0.000_001
        )
    }

    func testTransitionLossUsesTheRealImmediatelyPreviousAction() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 8, height: 8, colorMode: .grayscale)
        profile.channels = ActionChannels(absoluteMouse: false, relativeMouse: false, buttons: true, scroll: false, keyboard: false, modifiers: false)
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 1, frameSpacing: 1, downsampleFactor: 2)
        profile.training.binaryFocalGamma = 0
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        model.train(false)
        let inputs = temporalModelInputs(profile: profile, batch: 1, value: 0)
        var targetValues = [Float](repeating: 0, count: ActionLayout.count)
        targetValues[ActionLayout.buttons.lowerBound] = 1
        let targets = MLXArray(targetValues, [1, ActionLayout.count])
        let placeholderLoss = model.loss(
            currentImages: inputs.current,
            pastImages: inputs.past,
            pastControls: inputs.controls,
            targets: targets
        )
        let heldActionLoss = model.loss(
            currentImages: inputs.current,
            pastImages: inputs.past,
            pastControls: inputs.controls,
            targets: targets,
            previousTargets: targets
        )
        MLX.eval(placeholderLoss, heldActionLoss)
        XCTAssertNotEqual(
            placeholderLoss.item(Float.self),
            heldActionLoss.item(Float.self),
            accuracy: 0.000_001,
            "A held action and a fresh transition need different weighting independently of the sampled frame context."
        )
    }

    func testClassBalancedBinaryLossKeepsRareKeysBelowTheHeldThreshold() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(
            width: 8,
            height: 8,
            colorMode: .grayscale
        )
        profile.channels = ActionChannels(
            absoluteMouse: false,
            relativeMouse: false,
            buttons: false,
            scroll: false,
            keyboard: true,
            modifiers: false
        )
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 0)
        profile.training.binaryFocalGamma = 0
        profile.training.architecture = .small
        profile.training.precision = .float32
        profile.training.generalization = .disabled
        let model = AgentPolicy(profile: profile)
        model.train(false)

        let rowCount = 100
        let key = ActionLayout.keyboard.lowerBound + 13
        var targetValues = [Float](repeating: 0, count: rowCount * ActionLayout.count)
        var previousValues = targetValues
        targetValues[key] = 1
        previousValues[ActionLayout.count + key] = 1 // The following row is a release.
        let targets = MLXArray(targetValues, [rowCount, ActionLayout.count])
        let previous = MLXArray(previousValues, [rowCount, ActionLayout.count])
        var weightValues = [Float](repeating: 0, count: ActionLayout.count)
        weightValues[key] = 99
        let weights = MLXArray(weightValues, [ActionLayout.count])

        func loss(at rawLogit: Float) -> Float {
            var values = [Float](repeating: 0, count: rowCount * ActionLayout.count)
            for row in 0..<rowCount {
                values[row * ActionLayout.count + key] = rawLogit
            }
            let loss = model.loss(
                logits: MLXArray(values, [rowCount, ActionLayout.count]),
                targets: targets,
                positiveWeights: weights,
                previousTargets: previous
            )
            MLX.eval(loss)
            return loss.item(Float.self)
        }

        // One press receives 4 * 99 static weight; one release receives 4 and
        // the other 98 negatives receive 1. A calibrated loss-only prior shift
        // therefore places the saved/raw optimum at log(4 / 102), far below the
        // runtime's 0.5 held threshold. The former objective optimized near 0.8.
        let calibratedRawLogit = Float(log(4.0 / 102.0))
        let heldRawLogit = Float(log(4.0))
        let calibratedProbability = sigmoid(MLXArray(calibratedRawLogit))
        MLX.eval(calibratedProbability)

        XCTAssertLessThan(calibratedProbability.item(Float.self), 0.05)
        XCTAssertLessThan(loss(at: calibratedRawLogit), loss(at: 0))
        XCTAssertLessThan(loss(at: calibratedRawLogit), loss(at: heldRawLogit))
    }

    func testBinaryFocalGammaActuallyDownweightsUniformEasyNegatives() {
        func loss(gamma: Double) -> Float {
            var profile = AIProfile.fresh()
            profile.preprocessing = PreprocessingSpec(
                width: 8,
                height: 8,
                colorMode: .grayscale
            )
            profile.channels = ActionChannels(
                absoluteMouse: false,
                relativeMouse: false,
                buttons: true,
                scroll: false,
                keyboard: false,
                modifiers: false
            )
            profile.training.binaryFocalGamma = gamma
            profile.training.architecture = .small
            profile.training.precision = .float32
            profile.training.generalization = .disabled
            let model = AgentPolicy(profile: profile)
            model.train(false)
            var weights = [Float](repeating: 0, count: ActionLayout.count)
            weights[ActionLayout.buttons.lowerBound] = 1
            let result = model.loss(
                logits: MLXArray(
                    [Float](repeating: -5, count: 8 * ActionLayout.count),
                    [8, ActionLayout.count]
                ),
                targets: MLXArray.zeros([8, ActionLayout.count]),
                positiveWeights: MLXArray(weights, [ActionLayout.count])
            )
            MLX.eval(result)
            return result.item(Float.self)
        }

        let ordinary = loss(gamma: 0)
        let focused = loss(gamma: 2)
        XCTAssertGreaterThan(ordinary, 0)
        XCTAssertLessThan(
            focused,
            ordinary * 0.01,
            "Prediction-dependent focal weights must not cancel out through their own denominator."
        )
    }

    func testPolicyLearnsAVisualControlSignalInsteadOfAnInertShortcut() {
        MLXRandom.seed(202_607_15)
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 8, height: 8, colorMode: .grayscale)
        profile.channels = ActionChannels(absoluteMouse: false, relativeMouse: false, buttons: false, scroll: false, keyboard: true, modifiers: false)
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 0, frameSpacing: 1, downsampleFactor: 2)
        profile.training.precision = .float32
        profile.training.architecture = ArchitectureSpec(
            convolutionChannels: [8], kernelSizes: [3], strides: [1],
            visualEmbedding: 16, recurrentKind: .gru, recurrentWidth: 8,
            fusionWidths: [16], dropout: 0, visualPooling: .attention,
            attentionHeads: 2, controlEmbedding: 8
        )
        profile.training.generalization = .disabled
        let model = AgentPolicy(profile: profile)
        let optimizer = ResumableAdamW(learningRate: 0.003, weightDecay: 0)
        optimizer.initialize(model: model)
        let pixels = 8 * 8
        var imageValues = [Float](repeating: 0, count: 2 * pixels)
        // Group normalization intentionally removes uniform brightness. Use two
        // opposite spatial signals so the test measures visual learning rather
        // than asking the policy to recover a normalized-away offset.
        for row in 0..<8 {
            for column in 0..<8 {
                let pixel = row * 8 + column
                imageValues[pixel] = column < 4 ? 1 : 0
                imageValues[pixels + pixel] = column >= 4 ? 1 : 0
            }
        }
        let images = MLXArray(imageValues, [2, 8, 8, 1])
        let temporalInputs = temporalModelInputs(profile: profile, batch: 2, value: 0)
        let past = temporalInputs.past
        let controls = temporalInputs.controls
        var targetValues = [Float](repeating: 0, count: 2 * ActionLayout.count)
        let key = ActionLayout.keyboard.lowerBound + 13
        targetValues[ActionLayout.count + key] = 1
        let targets = MLXArray(targetValues, [2, ActionLayout.count])
        var mutableClassWeights = [Float](repeating: 0, count: ActionLayout.count)
        mutableClassWeights[key] = 1
        let classWeightValues = mutableClassWeights
        let weights = MLXArray(classWeightValues, [ActionLayout.count])
        let initial = model.loss(
            currentImages: images,
            pastImages: past,
            pastControls: controls,
            targets: targets,
            positiveWeights: weights
        )
        MLX.eval(initial)

        let step = compile(inputs: [model, optimizer], outputs: [model, optimizer]) { (arrays: [MLXArray]) -> [MLXArray] in
            let tracedWeights = MLXArray(classWeightValues, [ActionLayout.count])
            let result = valueAndGrad(model: model) { model, arrays in
                [model.loss(
                    currentImages: arrays[0],
                    pastImages: arrays[1],
                    pastControls: arrays[2],
                    targets: arrays[3],
                    positiveWeights: tracedWeights
                )]
            }(model, arrays)
            optimizer.update(model: model, gradients: clipGradNorm(gradients: result.1, maxNorm: 1).0, targetType: model.dtype)
            return [result.0[0]]
        }
        var final = initial.item(Float.self)
        for _ in 0..<600 {
            let loss = step([images, past, controls, targets])[0]
            MLX.eval(loss, model.parameters(), optimizer.stateArrays())
            final = loss.item(Float.self)
        }
        model.train(false)
        let predictions = model.predictions(currentImages: images, pastImages: past, pastControls: controls)
        MLX.eval(predictions)
        let values = predictions.asArray(Float.self)
        XCTAssertLessThan(final, initial.item(Float.self) * 0.2)
        XCTAssertLessThan(values[key], 0.2)
        XCTAssertGreaterThan(values[ActionLayout.count + key], 0.8)
    }

    func testKeyboardLossIgnoresRuntimeBlockedUnseenOutputs() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 12, height: 8, colorMode: .grayscale, bitDepth: 8)
        profile.channels = ActionChannels(absoluteMouse: false, relativeMouse: false, buttons: false, scroll: false, keyboard: true, modifiers: false)
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 1, frameSpacing: 1, downsampleFactor: 2)
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        let inputs = temporalModelInputs(profile: profile, batch: 1, value: 0.5)
        var targetValues = [Float](repeating: 0, count: ActionLayout.count)
        targetValues[ActionLayout.keyboard.lowerBound + 13] = 1
        let targets = MLXArray(targetValues, [1, ActionLayout.count])
        let blockedWeights = MLXArray([Float](repeating: 0, count: ActionLayout.count), [ActionLayout.count])
        let blockedLoss = model.loss(currentImages: inputs.current, pastImages: inputs.past, pastControls: inputs.controls, targets: targets, positiveWeights: blockedWeights)
        var learnedValues = [Float](repeating: 0, count: ActionLayout.count)
        learnedValues[ActionLayout.keyboard.lowerBound + 13] = 4
        let learnedLoss = model.loss(currentImages: inputs.current, pastImages: inputs.past, pastControls: inputs.controls, targets: targets, positiveWeights: MLXArray(learnedValues, [ActionLayout.count]))
        MLX.eval(blockedLoss, learnedLoss)
        XCTAssertEqual(blockedLoss.item(Float.self), 0, accuracy: 0.000_001)
        XCTAssertGreaterThan(learnedLoss.item(Float.self), 0)
    }

    func testShiftLossBelongsToKeyboardAndNotModifierChannel() {
        func loss(keyboard: Bool, modifiers: Bool, targetIndex: Int) -> Float {
            var profile = AIProfile.fresh()
            profile.preprocessing = PreprocessingSpec(width: 12, height: 8, colorMode: .grayscale, bitDepth: 8)
            profile.channels = ActionChannels(absoluteMouse: false, relativeMouse: false, buttons: false, scroll: false, keyboard: keyboard, modifiers: modifiers)
            profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 1, frameSpacing: 1, downsampleFactor: 2)
            profile.training.architecture = .small
            profile.training.architecture.dropout = 0
            profile.training.precision = .float32
            let model = AgentPolicy(profile: profile)
            let inputs = temporalModelInputs(profile: profile, batch: 1, value: 0.5)
            var targetValues = [Float](repeating: 0, count: ActionLayout.count)
            targetValues[targetIndex] = 1
            var weightValues = [Float](repeating: 0, count: ActionLayout.count)
            weightValues[targetIndex] = 4
            let result = model.loss(
                currentImages: inputs.current,
                pastImages: inputs.past,
                pastControls: inputs.controls,
                targets: MLXArray(targetValues, [1, ActionLayout.count]),
                positiveWeights: MLXArray(weightValues, [ActionLayout.count])
            )
            MLX.eval(result)
            return result.item(Float.self)
        }

        let shift = ActionLayout.shift.lowerBound
        let control = ActionLayout.commandOptionControl.lowerBound
        let duplicatedControlKey = ActionLayout.keyboard.lowerBound + 59
        XCTAssertGreaterThan(loss(keyboard: true, modifiers: false, targetIndex: shift), 0)
        XCTAssertEqual(loss(keyboard: true, modifiers: false, targetIndex: control), 0, accuracy: 0.000_001)
        XCTAssertEqual(loss(keyboard: true, modifiers: false, targetIndex: duplicatedControlKey), 0, accuracy: 0.000_001)
        XCTAssertEqual(loss(keyboard: true, modifiers: true, targetIndex: duplicatedControlKey), 0, accuracy: 0.000_001)
        XCTAssertEqual(loss(keyboard: false, modifiers: true, targetIndex: shift), 0, accuracy: 0.000_001)
        XCTAssertGreaterThan(loss(keyboard: false, modifiers: true, targetIndex: control), 0)
    }

    func testCompiledCNNDiagnosticsPreservePredictionsAndProduceBoundedMaps() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 32, height: 24, colorMode: .grayscale, bitDepth: 8)
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 2, frameSpacing: 2, downsampleFactor: 2)
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        model.train(false)
        let currentSpec = profile.preprocessing
        let temporal = profile.training.effectiveTemporalVision
        let pastSpec = temporal.pastFrameSpec(from: currentSpec)
        let packedCurrent = MLXArray(
            Data(repeating: 128, count: currentSpec.sampleByteCount),
            [1, currentSpec.sampleByteCount],
            dtype: .uint8
        )
        let packedPast = MLXArray(
            Data(repeating: 128, count: temporal.pastFrameCount * pastSpec.sampleByteCount),
            [1, temporal.pastFrameCount, pastSpec.sampleByteCount],
            dtype: .uint8
        )
        let controls = MLXArray.zeros([1, temporal.pastFrameCount, ActionLayout.count])
        let currentImages = VisionPreprocessor.mlxTensor(packedCurrent, spec: currentSpec)

        let layers = model.visualActivations(images: currentImages)
        XCTAssertEqual(layers.map(\.shape), [[1, 12, 16, 24], [1, 6, 8, 48], [1, 3, 4, 72], [1, 2, 2, 96], [1, 2, 2, 96]])

        let standard = compile(inputs: [model]) { current, past, controls in
            model.predictions(
                currentImages: VisionPreprocessor.mlxTensor(current, spec: currentSpec),
                pastImages: VisionPreprocessor.mlxPastFrameTensor(past, spec: pastSpec),
                pastControls: controls
            )
        }
        let activities = layers.indices.map { selectedLayer in
            compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
                let current = VisionPreprocessor.mlxTensor(inputs[0], spec: currentSpec)
                let past = VisionPreprocessor.mlxPastFrameTensor(inputs[1], spec: pastSpec)
                let visual = model.visualActivations(images: current)
                let logits = model.logits(
                    currentVisualFeatures: visual.last!,
                    pastImages: past,
                    pastControls: inputs[2]
                )
                let map = model.sampledForVisualization(visual[selectedLayer]).mean(axis: -1, keepDims: true)
                return [model.activatedPredictions(logits: logits), map]
            }
        }
        let channels = compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
            let current = VisionPreprocessor.mlxTensor(inputs[0], spec: currentSpec)
            let past = VisionPreprocessor.mlxPastFrameTensor(inputs[1], spec: pastSpec)
            let visual = model.visualActivations(images: current)
            let logits = model.logits(
                currentVisualFeatures: visual.last!,
                pastImages: past,
                pastControls: inputs[2]
            )
            return [model.activatedPredictions(logits: logits), model.strongestChannelsForVisualization(visual.last!)]
        }
        let saliency = compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
            let current = VisionPreprocessor.mlxTensor(inputs[0], spec: currentSpec)
            let past = VisionPreprocessor.mlxPastFrameTensor(inputs[1], spec: pastSpec)
            let visual = model.visualActivations(images: current)
            let logits = model.logits(
                currentVisualFeatures: visual.last!,
                pastImages: past,
                pastControls: inputs[2]
            )
            return [model.activatedPredictions(logits: logits), visual.last!]
        }
        let saliencyGradient = grad({ (inputs: [MLXArray]) -> MLXArray in
            let past = VisionPreprocessor.mlxPastFrameTensor(inputs[1], spec: pastSpec)
            let logits = model.logits(
                currentVisualFeatures: inputs[0],
                pastImages: past,
                pastControls: inputs[2]
            )
            return (logits * inputs[3]).sum()
        }, argumentNumbers: [0])
        var selectorValues = [Float](repeating: 0, count: ActionLayout.count)
        selectorValues[ActionLayout.keyboard.lowerBound] = 1
        let selector = MLXArray(selectorValues, [1, ActionLayout.count])
        let expected = standard(packedCurrent, packedPast, controls)
        let activityResults = activities.map { $0([packedCurrent, packedPast, controls]) }
        let channelResult = channels([packedCurrent, packedPast, controls])
        let saliencyForward = saliency([packedCurrent, packedPast, controls])
        let gradients = saliencyGradient([saliencyForward[1], packedPast, controls, selector])
        let weights = gradients.mean(axes: [1, 2], keepDims: true)
        let saliencyMap = model.sampledForVisualization(relu((saliencyForward[1] * weights).sum(axis: -1, keepDims: true)))
        let saliencyResult = [saliencyForward[0], saliencyMap]
        MLX.eval(expected, activityResults.flatMap { $0 }, channelResult, saliencyForward, saliencyResult)

        XCTAssertEqual(activityResults.map { $0[1].shape }, [[1, 12, 16, 1], [1, 6, 8, 1], [1, 3, 4, 1], [1, 2, 2, 1], [1, 2, 2, 1]])
        XCTAssertEqual(channelResult[1].shape, [1, 2, 2, 16])
        XCTAssertEqual(saliencyForward[1].shape, [1, 2, 2, 96])
        XCTAssertEqual(saliencyResult[1].shape, [1, 2, 2, 1])
        let expectedValues = expected.asArray(Float.self)
        for prediction in activityResults.map({ $0[0] }) + [channelResult[0], saliencyResult[0]] {
            XCTAssertTrue(zip(expectedValues, prediction.asArray(Float.self)).allSatisfy { abs($0 - $1) < 1e-5 })
        }
        XCTAssertTrue(saliencyResult[1].asArray(Float.self).allSatisfy { $0.isFinite && $0 >= 0 })
    }

    func testOptimizerCheckpointResumesExactly() throws {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 16, height: 12, colorMode: .grayscale, bitDepth: 8)
        profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 1, frameSpacing: 1, downsampleFactor: 2)
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.generalization = .disabled
        profile.training.precision = .float32
        let modelA = AgentPolicy(profile: profile)
        let optimizerA = ResumableAdamW(learningRate: 0.001, weightDecay: 0.01)
        let inputs = temporalModelInputs(profile: profile, batch: 1, value: 0.25)
        let targets = MLXArray([Float](repeating: 0, count: ActionLayout.count), [1, ActionLayout.count])
        let gradientA = valueAndGrad(model: modelA) { model, arrays in
            [model.loss(currentImages: arrays[0], pastImages: arrays[1], pastControls: arrays[2], targets: arrays[3])]
        }
        let first = gradientA(modelA, [inputs.current, inputs.past, inputs.controls, targets])
        optimizerA.update(model: modelA, gradients: first.1, targetType: .float32)
        MLX.eval(modelA.parameters(), optimizerA.stateArrays())

        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("resume-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let weights = directory.appendingPathComponent("weights.safetensors")
        let optimizer = directory.appendingPathComponent("optimizer.safetensors")
        try modelA.saveWeights(to: weights); try optimizerA.save(to: optimizer)

        let secondA = gradientA(modelA, [inputs.current, inputs.past, inputs.controls, targets])
        optimizerA.update(model: modelA, gradients: secondA.1, targetType: .float32)
        MLX.eval(modelA.parameters(), optimizerA.stateArrays())

        let modelB = AgentPolicy(profile: profile)
        let optimizerB = ResumableAdamW(learningRate: 0.001, weightDecay: 0.01)
        try modelB.loadWeights(from: weights); try optimizerB.load(from: optimizer)
        let gradientB = valueAndGrad(model: modelB) { model, arrays in
            [model.loss(currentImages: arrays[0], pastImages: arrays[1], pastControls: arrays[2], targets: arrays[3])]
        }
        let secondB = gradientB(modelB, [inputs.current, inputs.past, inputs.controls, targets])
        optimizerB.update(model: modelB, gradients: secondB.1, targetType: .float32)
        MLX.eval(modelB.parameters(), optimizerB.stateArrays())

        let paramsA = Dictionary(uniqueKeysWithValues: modelA.parameters().flattened())
        let paramsB = Dictionary(uniqueKeysWithValues: modelB.parameters().flattened())
        XCTAssertEqual(paramsA.keys.sorted(), paramsB.keys.sorted())
        for key in paramsA.keys {
            let a = try XCTUnwrap(paramsA[key]).asArray(Float.self)
            let b = try XCTUnwrap(paramsB[key]).asArray(Float.self)
            XCTAssertEqual(a.count, b.count)
            let maximumDelta = zip(a, b).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
            // Independent Metal convolution reductions are not bitwise stable,
            // even when weights and every Adam tensor restore exactly.
            XCTAssertLessThan(maximumDelta, 1e-5, "Checkpoint diverged at \(key)")
        }
        XCTAssertEqual(optimizerA.step, optimizerB.step)
    }

    private func actionFloat(_ data: Data, index: Int) -> Float {
        let bits = data.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: index * MemoryLayout<UInt32>.size, as: UInt32.self)
        }
        return Float(bitPattern: UInt32(littleEndian: bits))
    }

    private func writeTestMovie(to url: URL, width: Int, height: Int, frameCount: Int, fps: Int) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height
        ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height
        ])
        guard writer.canAdd(input) else { throw AgentTrainerError.capture("Test movie input was rejected.") }
        writer.add(input)
        guard writer.startWriting() else { throw writer.error ?? AgentTrainerError.capture("Test movie could not start.") }
        writer.startSession(atSourceTime: .zero)
        for frame in 0..<frameCount {
            while !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(1)) }
            var pixelBuffer: CVPixelBuffer?
            let attributes = [kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary
            guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attributes, &pixelBuffer) == kCVReturnSuccess,
                  let pixelBuffer else { throw AgentTrainerError.capture("Test frame allocation failed.") }
            CVPixelBufferLockBaseAddress(pixelBuffer, [])
            let value = UInt8(min(255, 20 + frame * 30))
            if let base = CVPixelBufferGetBaseAddress(pixelBuffer) {
                memset(base, Int32(value), CVPixelBufferGetDataSize(pixelBuffer))
            }
            CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
            guard adaptor.append(pixelBuffer, withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: CMTimeScale(fps))) else {
                throw writer.error ?? AgentTrainerError.capture("Test frame append failed.")
            }
        }
        input.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed else { throw writer.error ?? AgentTrainerError.capture("Test movie did not finish.") }
    }

    private func temporalModelInputs(
        profile: AIProfile,
        batch: Int,
        value: Float
    ) -> (current: MLXArray, past: MLXArray, controls: MLXArray) {
        let temporal = profile.training.effectiveTemporalVision
        let currentSpec = profile.preprocessing
        let pastSpec = temporal.pastFrameSpec(from: currentSpec)
        return (
            MLXArray(
                [Float](repeating: value, count: batch * currentSpec.width * currentSpec.height * currentSpec.channelCount),
                [batch, currentSpec.height, currentSpec.width, currentSpec.channelCount]
            ),
            MLXArray(
                [Float](repeating: value, count: batch * temporal.pastFrameCount * pastSpec.width * pastSpec.height * pastSpec.channelCount),
                [batch, temporal.pastFrameCount, pastSpec.height, pastSpec.width, pastSpec.channelCount]
            ),
            MLXArray.zeros([batch, temporal.pastFrameCount, ActionLayout.count])
        )
    }

    private func makeSyntheticDataset(
        name: String,
        pastFrameCount: Int,
        sequences: [[UInt32]],
        actionRows: [[Float]],
        frameActionRows suppliedFrameActionRows: [[Float]]? = nil,
        segments: [CacheSegment]? = nil
    ) throws -> (dataset: CachedDataset, directory: URL) {
        precondition(sequences.count == actionRows.count)
        precondition(actionRows.allSatisfy { $0.count == ActionLayout.count })
        precondition(sequences.allSatisfy { $0.count == pastFrameCount + 1 })
        let segments = segments ?? [CacheSegment(recordingID: UUID(), start: 0, count: sequences.count)]
        precondition(segments.reduce(0) { $0 + $1.count } == sequences.count)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString).atrcache", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            let observationCount = Int(
                sequences.flatMap { $0 }.filter { $0 != UInt32.max }.max() ?? 0
            ) + 1
            let spec = PreprocessingSpec(width: 1, height: 1, colorMode: .grayscale, bitDepth: 8)
            let temporal = TemporalVisionConfiguration(
                pastFrameCount: pastFrameCount,
                frameSpacing: 1,
                downsampleFactor: 1
            )
            let manifest = DatasetCacheManifest(
                key: name,
                createdAt: Date(),
                preprocessing: spec,
                pastPreprocessing: spec,
                temporalVision: temporal,
                actionFPS: 60,
                perceptionFPS: 30,
                sampleCount: sequences.count,
                observationCount: observationCount,
                currentObservationBytesPerSample: 1,
                pastObservationBytesPerSample: 1,
                actionValuesPerSample: ActionLayout.count,
                segments: segments
            )
            let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(manifest).write(to: directory.appendingPathComponent("manifest.json"))
            let observations = Data((0..<observationCount).map { UInt8(clamping: $0) })
            try observations.write(to: directory.appendingPathComponent("current-observations.bin"))
            try observations.write(to: directory.appendingPathComponent("past-observations.bin"))
            try observationSequenceData(sequences).write(to: directory.appendingPathComponent("observation-indices.bin"))
            let frameActionRows: [[Float]]
            if let suppliedFrameActionRows {
                precondition(suppliedFrameActionRows.count == observationCount)
                precondition(suppliedFrameActionRows.allSatisfy { $0.count == ActionLayout.count })
                frameActionRows = suppliedFrameActionRows
            } else {
                frameActionRows = (0..<observationCount).map { observation in
                    guard let sample = sequences.firstIndex(where: { $0[0] == UInt32(observation) }) else {
                        return [Float](repeating: 0, count: ActionLayout.count)
                    }
                    return actionRows[sample]
                }
            }
            var frameActions = Data(capacity: frameActionRows.count * ActionLayout.count * MemoryLayout<Float>.size)
            for row in frameActionRows { row.withUnsafeBytes { frameActions.append(contentsOf: $0) } }
            try frameActions.write(to: directory.appendingPathComponent("frame-actions.bin"))
            var actions = Data(capacity: actionRows.count * ActionLayout.count * MemoryLayout<Float>.size)
            for row in actionRows { row.withUnsafeBytes { actions.append(contentsOf: $0) } }
            try actions.write(to: directory.appendingPathComponent("actions.bin"))
            return (try CachedDataset(directory: directory), directory)
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    private func observationSequenceData(_ sequences: [[UInt32]]) -> Data {
        var data = Data(capacity: sequences.reduce(0) { $0 + $1.count } * MemoryLayout<UInt32>.size)
        for sequence in sequences {
            for index in sequence {
                var littleEndian = index.littleEndian
                withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
            }
        }
        return data
    }
}
