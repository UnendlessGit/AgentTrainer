import XCTest
import AVFoundation
import CoreVideo
import MLX
import MLXNN
import MLXOptimizers
@preconcurrency import ScreenCaptureKit
@testable import AgentTrainer

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
    func testUpdateVersionsUseSemanticOrdering() throws {
        XCTAssertLessThan(try XCTUnwrap(AppSemanticVersion("v1.3.9")), try XCTUnwrap(AppSemanticVersion("1.4.0")))
        XCTAssertLessThan(try XCTUnwrap(AppSemanticVersion("1.9")), try XCTUnwrap(AppSemanticVersion("1.10")))
        XCTAssertEqual(try XCTUnwrap(AppSemanticVersion("1.3")), try XCTUnwrap(AppSemanticVersion("1.3.0")))
        XCTAssertLessThan(try XCTUnwrap(AppSemanticVersion("2.0.0-beta.2")), try XCTUnwrap(AppSemanticVersion("2.0.0")))
        XCTAssertNil(AppSemanticVersion("release-next"))
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

        var largerAttention = profile
        largerAttention.training.historyLength = 64
        largerAttention.training.batchSize = 8
        largerAttention.training.architecture.transformerLayers = 4
        largerAttention.training.architecture.transformerFeedForward = 1_024
        XCTAssertGreaterThan(
            ModelSizing.estimatedTrainingWorkingSet(largerAttention),
            ModelSizing.estimatedTrainingWorkingSet(profile)
        )
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

    func testNeuralInputSizingMirrorsPerceptionOnlyVisualMemoryContract() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 5, height: 3, colorMode: .color, bitDepth: 2, chroma: .yuv420, resizePolicy: .fit)
        profile.training.historyLength = 4
        profile.training.batchSize = 2
        profile.training.precision = .float16
        profile.training.perceptionFPS = 10
        profile.training.actionFPS = 20

        let input = NeuralInputSizing.summary(for: profile)
        XCTAssertEqual(input.pixelCount, 15)
        XCTAssertEqual(input.lumaValues, 15)
        XCTAssertEqual(input.chromaValuesPerPlane, 6)
        XCTAssertEqual(input.packedVisionValues, 27)
        XCTAssertEqual(input.packedVisualWindowValues, 27 * 5)
        XCTAssertEqual(input.expandedVisionValues, 45)
        XCTAssertEqual(input.visualMemoryFrames, 4)
        XCTAssertEqual(input.visualMemoryStride, 4)
        XCTAssertEqual(input.visualMemoryMaximumLag, 13)
        XCTAssertEqual(input.visualMemoryDurationSeconds, 1.3, accuracy: 0.000_001)
        XCTAssertEqual(input.temporalDifferenceValues, 45 * 4)
        XCTAssertEqual(input.visualMemoryAvailabilityValues, 15 * 4)
        XCTAssertEqual(input.visualMemoryFusionValues, 15 * Int64(VisualMemoryContract.fusionChannels))
        XCTAssertEqual(input.coordinateValues, 30)
        XCTAssertEqual(input.firstVisualProjectionValues, 45 + 15 * Int64(VisualMemoryContract.fusionChannels) + 30)
        XCTAssertEqual(input.historySteps, 0)
        XCTAssertEqual(input.historyValues, 0)
        XCTAssertEqual(input.historyDurationSeconds, 0, accuracy: 0.000_001)
        XCTAssertEqual(input.transformerTokenCount, 11)
        XCTAssertEqual(input.attentionPairsPerLayer, 121)
        let valuesPerDecision: Int64 = 45 + 45 * 4 + 15 * 4 + 30
        XCTAssertEqual(input.valuesPerDecision, valuesPerDecision)
        XCTAssertEqual(input.runtimeValuesPerSecond, 10 * valuesPerDecision)
        XCTAssertEqual(input.packedVisionBytesPerSecond, 270)
        XCTAssertEqual(input.valuesPerTrainingBatch, 2 * valuesPerDecision)
        XCTAssertEqual(input.quantizationLevels, 4)
        XCTAssertEqual(input.effectivePackedBits, 27 * 2)
        XCTAssertEqual(input.nominalBytesPerTrainingBatch, 2 * 2 * valuesPerDecision)
    }

    func testNeuralInputSizingExcludesHistoryAndIgnoresChromaForGrayscale() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 5, height: 3, colorMode: .grayscale, bitDepth: 8, chroma: .yuv420, resizePolicy: .stretch)
        profile.training.historyLength = 0
        profile.training.batchSize = 1
        profile.training.precision = .float32

        let input = NeuralInputSizing.summary(for: profile)
        XCTAssertEqual(input.packedVisionValues, 15)
        XCTAssertEqual(input.chromaValuesPerPlane, 0)
        XCTAssertEqual(input.expandedVisionValues, 15)
        XCTAssertEqual(input.temporalDifferenceValues, 15 * 4)
        XCTAssertEqual(input.visualMemoryAvailabilityValues, 15 * 4)
        XCTAssertEqual(input.firstVisualProjectionValues, 15 + 15 * Int64(VisualMemoryContract.fusionChannels) + 30)
        XCTAssertEqual(input.historySteps, 0)
        XCTAssertEqual(input.historyValues, 0)
        XCTAssertEqual(input.transformerTokenCount, 11)
        XCTAssertEqual(input.attentionPairsPerLayer, 121)
        XCTAssertEqual(input.valuesPerDecision, 165)
        XCTAssertEqual(input.nominalBytesPerDecision, 165 * 4)
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
        var falsePositiveBaseline = baseline
        falsePositiveBaseline.keyboard = BinaryValidationMetrics(
            truePositives: 8,
            falsePositives: 8,
            falseNegatives: 12,
            trueNegatives: 72
        )
        var usefulActivation = falsePositiveBaseline
        usefulActivation.keyboard = BinaryValidationMetrics(
            truePositives: 14,
            falsePositives: 12,
            falseNegatives: 6,
            trueNegatives: 68
        )
        XCTAssertFalse(
            usefulActivation.hasSevereBinaryRegression(comparedTo: falsePositiveBaseline),
            "A false-positive increase is acceptable when held-out F1 improves materially."
        )

        var unsupportedBaseline = baseline
        unsupportedBaseline.keyboard = BinaryValidationMetrics(truePositives: 3, falsePositives: 0, falseNegatives: 1, trueNegatives: 96)
        var unsupportedCollapse = unsupportedBaseline
        unsupportedCollapse.keyboard = BinaryValidationMetrics(truePositives: 0, falsePositives: 0, falseNegatives: 4, trueNegatives: 96)
        XCTAssertFalse(unsupportedCollapse.hasSevereBinaryRegression(comparedTo: unsupportedBaseline), "Four positives are too few for a strong regression diagnosis.")
        XCTAssertFalse(unsupportedCollapse.hasBinaryRecallCollapse)
        var idlePolicy = baseline
        idlePolicy.keyboard = BinaryValidationMetrics(truePositives: 0, falsePositives: 0, falseNegatives: 10, trueNegatives: 90)
        XCTAssertTrue(idlePolicy.hasBinaryRecallCollapse)
        var perOutputCollapse = baseline
        perOutputCollapse.binaryOutputs = [
            BinaryOutputValidation(
                outputIndex: ActionLayout.keyboard.lowerBound,
                decisionThreshold: 0.5,
                defaultMetrics: metrics,
                calibratedMetrics: metrics
            ),
            BinaryOutputValidation(
                outputIndex: ActionLayout.keyboard.lowerBound + 1,
                decisionThreshold: 0.5,
                defaultMetrics: BinaryValidationMetrics(
                    truePositives: 3,
                    falsePositives: 0,
                    falseNegatives: 97,
                    trueNegatives: 92
                ),
                calibratedMetrics: BinaryValidationMetrics(
                    truePositives: 3,
                    falsePositives: 0,
                    falseNegatives: 97,
                    trueNegatives: 92
                )
            )
        ]
        XCTAssertTrue(
            perOutputCollapse.hasBinaryRecallCollapse,
            "One dead supported key must not be hidden by the aggregate keyboard score."
        )
        perOutputCollapse.binaryOutputs?[1].calibratedMetrics = BinaryValidationMetrics(
            truePositives: 4,
            falsePositives: 0,
            falseNegatives: 96,
            trueNegatives: 92
        )
        XCTAssertFalse(perOutputCollapse.hasBinaryRecallCollapse)
        var weakBaseline = baseline
        weakBaseline.keyboard = BinaryValidationMetrics(truePositives: 1, falsePositives: 8, falseNegatives: 9, trueNegatives: 82)
        XCTAssertTrue(
            idlePolicy.hasSevereBinaryRegression(comparedTo: weakBaseline),
            "Best-brain selection must never replace even a weak working key head with zero recall."
        )

        var moving = baseline
        moving.activeRelativeMouseExecutionRecall = 0.6
        moving.trainingBalance = TrainingBalanceReport(
            outputs: [],
            continuousOutputs: [
                ContinuousOutputBalance(
                    outputIndex: ActionLayout.relativeMouse.lowerBound,
                    activeSamples: 20,
                    meanActiveMagnitude: 0.04,
                    activeWeight: 4,
                    isSupported: true
                )
            ]
        )
        var motionCollapsed = moving
        motionCollapsed.activeRelativeMouseExecutionRecall = 0
        XCTAssertTrue(motionCollapsed.hasContinuousExecutionCollapse)
        XCTAssertTrue(motionCollapsed.hasContinuousExecutionFailure)
        XCTAssertTrue(motionCollapsed.hasSevereContinuousRegression(comparedTo: moving))
        var jittering = moving
        jittering.idleContinuousFalseActionRate = 0.11
        XCTAssertFalse(jittering.hasContinuousExecutionCollapse)
        XCTAssertTrue(jittering.hasContinuousExecutionFailure)
        XCTAssertTrue(jittering.hasSevereContinuousRegression(comparedTo: moving))
    }

    func testBinaryThresholdCalibrationReducesFalseActionsWithoutSparseOverfitting() throws {
        let positives = [Float](repeating: 0.8, count: 64)
        let easyNegatives = [Float](repeating: 0.1, count: 200)
        let ambiguousNegatives = [Float](repeating: 0.6, count: 56)
        let calibrated = try XCTUnwrap(BinaryThresholdCalibration.calibrate(
            outputIndex: ActionLayout.keyboard.lowerBound + 13,
            probabilities: positives + easyNegatives + ambiguousNegatives,
            targets: [Bool](repeating: true, count: positives.count)
                + [Bool](repeating: false, count: easyNegatives.count + ambiguousNegatives.count)
        ))
        XCTAssertGreaterThan(calibrated.decisionThreshold, 0.6)
        XCTAssertEqual(calibrated.defaultMetrics.falsePositives, ambiguousNegatives.count)
        XCTAssertEqual(calibrated.calibratedMetrics.falsePositives, 0)
        XCTAssertGreaterThan(calibrated.calibratedMetrics.f1, calibrated.defaultMetrics.f1)
        let persistedReport = ValidationReport(
            sampleCount: positives.count + easyNegatives.count + ambiguousNegatives.count,
            binary: calibrated.calibratedMetrics,
            buttons: nil,
            keyboard: calibrated.calibratedMetrics,
            modifiers: nil,
            absoluteMouseMAE: nil,
            activeRelativeMouseMAE: nil,
            activeScrollMAE: nil,
            idleContinuousFalseActionRate: nil,
            binaryOutputs: [calibrated]
        )
        let decoded = try JSONDecoder().decode(
            ValidationReport.self,
            from: JSONEncoder().encode(persistedReport)
        )
        let denseThresholds = BinaryDecisionThresholds.values(from: decoded)
        XCTAssertEqual(
            denseThresholds[calibrated.outputIndex],
            Float(calibrated.decisionThreshold),
            accuracy: 0.000_001
        )
        var inSampleCalibration = persistedReport
        inSampleCalibration.evaluationScope = .trainingCalibration
        let decodedCalibration = try JSONDecoder().decode(
            ValidationReport.self,
            from: JSONEncoder().encode(inSampleCalibration)
        )
        XCTAssertEqual(
            decodedCalibration.effectiveEvaluationScope,
            .trainingCalibration,
            "Runtime calibration must remain distinguishable from held-out quality after persistence."
        )

        let sparse = try XCTUnwrap(BinaryThresholdCalibration.calibrate(
            outputIndex: ActionLayout.keyboard.lowerBound + 12,
            probabilities: [Float](repeating: 0.9, count: 4) + [Float](repeating: 0.6, count: 100),
            targets: [Bool](repeating: true, count: 4) + [Bool](repeating: false, count: 100)
        ))
        XCTAssertEqual(sparse.decisionThreshold, 0.5)
        XCTAssertEqual(sparse.calibratedMetrics, sparse.defaultMetrics)
    }

    func testLegacyValidationReportDecodesWithoutObjectiveV2Telemetry() throws {
        let json = Data("""
        {
          "sampleCount": 12,
          "absoluteMouseMAE": 0.1,
          "activeRelativeMouseMAE": 0.2,
          "activeScrollMAE": 0.3,
          "idleContinuousFalseActionRate": 0.01
        }
        """.utf8)
        let report = try JSONDecoder().decode(ValidationReport.self, from: json)
        XCTAssertEqual(report.sampleCount, 12)
        XCTAssertNil(report.lossBreakdown)
        XCTAssertNil(report.binaryOutputs)
        XCTAssertNil(report.trainingBalance)
        XCTAssertEqual(report.effectiveEvaluationScope, .heldOut)
        XCTAssertEqual(report.decisionThreshold(for: ActionLayout.keyboard.lowerBound + 13), 0.5)

        let objectiveV2Balance = try JSONDecoder().decode(
            TrainingBalanceReport.self,
            from: Data("""
            {
              "outputs": [{
                "outputIndex": 27,
                "positiveSamples": 20,
                "pressEpisodes": 4,
                "positiveWeight": 2.5,
                "isSupported": true
              }]
            }
            """.utf8)
        )
        XCTAssertNil(objectiveV2Balance.outputs.first?.releaseEpisodes)
        XCTAssertNil(objectiveV2Balance.outputs.first?.activeDurationSeconds)
    }

    func testArchitecturePresetsHaveNoZeroWidths() {
        for architecture in [ArchitectureSpec.small, .balanced, .large] {
            XCTAssertTrue(architecture.convolutionChannels.allSatisfy { $0 > 0 })
            XCTAssertTrue(architecture.fusionWidths.allSatisfy { $0 > 0 })
            XCTAssertGreaterThan(architecture.effectiveSpatialTokens, 0)
            XCTAssertGreaterThan(architecture.effectiveTransformerLayers, 0)
            XCTAssertGreaterThan(architecture.effectiveTransformerHeads, 0)
            XCTAssertTrue(architecture.visualEmbedding.isMultiple(of: architecture.effectiveTransformerHeads))
            XCTAssertGreaterThanOrEqual(architecture.effectiveTransformerFeedForward, architecture.visualEmbedding)
        }
        XCTAssertEqual(CNNGeometry.layer(0, architecture: .balanced), CNNLayerGeometry(kernelSize: 7, effectiveStride: 4, receptiveField: 7))
        XCTAssertEqual(CNNGeometry.layer(1, architecture: .balanced), CNNLayerGeometry(kernelSize: 3, effectiveStride: 8, receptiveField: 15))
        XCTAssertEqual(CNNGeometry.layer(2, architecture: .balanced), CNNLayerGeometry(kernelSize: 3, effectiveStride: 16, receptiveField: 31))
        XCTAssertEqual(CNNGeometry.layer(3, architecture: .balanced), CNNLayerGeometry(kernelSize: 3, effectiveStride: 32, receptiveField: 63))
        XCTAssertEqual(CNNGeometry.outputSize(width: 32, height: 24, architecture: .small).width, 1)
        XCTAssertEqual(CNNGeometry.outputSize(width: 32, height: 24, architecture: .small).height, 1)
        XCTAssertEqual(CNNGeometry.outputSize(width: 640, height: 360, architecture: .balanced).width, 20)
        XCTAssertEqual(CNNGeometry.outputSize(width: 640, height: 360, architecture: .balanced).height, 12)
        var convolutionFree = ArchitectureSpec.balanced
        convolutionFree.convolutionChannels = []
        XCTAssertEqual(CNNGeometry.layer(-1, architecture: convolutionFree), CNNLayerGeometry(kernelSize: 1, effectiveStride: 1, receptiveField: 1))

        for architecture in [ArchitectureSpec.pureSmall, .pureBalanced, .pureLarge] {
            XCTAssertEqual(architecture.effectiveFamily, .pureTransformer)
            XCTAssertTrue(architecture.convolutionChannels.isEmpty)
            XCTAssertTrue((8...64).contains(architecture.effectivePatchSize))
            XCTAssertTrue(architecture.visualEmbedding.isMultiple(of: architecture.effectiveTransformerHeads))
            XCTAssertGreaterThanOrEqual(architecture.effectiveTransformerFeedForward, architecture.visualEmbedding)
        }
        XCTAssertEqual(CNNGeometry.layer(-1, architecture: .pureBalanced), CNNLayerGeometry(kernelSize: 32, effectiveStride: 32, receptiveField: 32))
        XCTAssertEqual(CNNGeometry.outputSize(width: 641, height: 361, architecture: .pureBalanced).width, 21)
        XCTAssertEqual(CNNGeometry.outputSize(width: 641, height: 361, architecture: .pureBalanced).height, 12)
    }

    func testHybridTokenizationIsResolutionStableAndAttentionRemainsCompact() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 640, height: 360)
        profile.training.architecture = .balanced
        let parameters = ModelSizing.parameterCount(profile)

        var higherResolution = profile
        higherResolution.preprocessing.width = 1_280
        higherResolution.preprocessing.height = 720
        XCTAssertEqual(ModelSizing.parameterCount(higherResolution), parameters, "CNN sharing and compact tokens must keep learned parameter count resolution-independent.")
        XCTAssertEqual(PolicyTokenGeometry.sequenceLength(profile), 11)
        XCTAssertEqual(PolicyTokenGeometry.attentionPairsPerLayer(profile), 121)

        var moreReasoning = profile
        moreReasoning.training.architecture.spatialTokens = 12
        moreReasoning.training.architecture.transformerLayers = 4
        XCTAssertGreaterThan(ModelSizing.parameterCount(moreReasoning), parameters)
        XCTAssertEqual(PolicyTokenGeometry.sequenceLength(moreReasoning), 15)
        XCTAssertLessThan(PolicyTokenGeometry.sequenceLength(moreReasoning), 64, "Default hybrid attention should remain a compact token problem, not a raw-pixel Transformer.")
    }

    func testVisualMemoryGrowthIsLinearAndNeverAddsAttentionTokens() {
        var profile = AIProfile.fresh()
        profile.training.architecture = .balanced
        profile.training.visualMemoryFrames = 0
        let sequence = PolicyTokenGeometry.sequenceLength(profile)
        let attention = PolicyTokenGeometry.attentionPairsPerLayer(profile)

        profile.training.visualMemoryFrames = 4
        let fourFrameParameters = ModelSizing.parameterCount(profile)
        XCTAssertEqual(PolicyTokenGeometry.sequenceLength(profile), sequence)
        XCTAssertEqual(PolicyTokenGeometry.attentionPairsPerLayer(profile), attention)

        profile.training.visualMemoryFrames = 8
        let eightFrameParameters = ModelSizing.parameterCount(profile)
        XCTAssertEqual(PolicyTokenGeometry.sequenceLength(profile), sequence)
        XCTAssertEqual(PolicyTokenGeometry.attentionPairsPerLayer(profile), attention)
        XCTAssertEqual(
            eightFrameParameters - fourFrameParameters,
            Int64(
                (8 - 4)
                    * (profile.preprocessing.channelCount + 1)
                    * VisualMemoryContract.fusionChannels
            ),
            "Additional memory slots should grow only the pointwise evidence projection."
        )
    }

    func testPureTransformerTokenGeometryUsesEveryPaddedPatch() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 640, height: 360)
        profile.training.architecture = .pureBalanced
        profile.training.historyLength = 16
        XCTAssertEqual(PolicyTokenGeometry.patchGrid(profile).width, 20)
        XCTAssertEqual(PolicyTokenGeometry.patchGrid(profile).height, 12)
        XCTAssertEqual(PolicyTokenGeometry.visualTokenCount(profile), 240)
        XCTAssertEqual(PolicyTokenGeometry.sequenceLength(profile), 241)
        XCTAssertEqual(PolicyTokenGeometry.attentionPairsPerLayer(profile), 58_081)

        let parameters = ModelSizing.parameterCount(profile)
        profile.preprocessing.width = 1_280
        profile.preprocessing.height = 720
        XCTAssertEqual(ModelSizing.parameterCount(profile), parameters, "Shared direct patch projection must remain resolution-independent.")
        XCTAssertEqual(PolicyTokenGeometry.visualTokenCount(profile), 920)
        XCTAssertGreaterThan(PolicyTokenGeometry.attentionPairsPerLayer(profile), 800_000)
    }

    func testDefaultPureTransformerProducesFiniteFullResolutionPrediction() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 640, height: 360, colorMode: .grayscale)
        profile.training.architecture = .pureBalanced
        profile.training.precision = .float16
        let model = AgentPolicy(profile: profile)
        model.train(false)
        let images = MLXArray.zeros([
            1, 360, 640,
            VisualMemoryContract.rawInputChannels(
                colorChannels: profile.preprocessing.channelCount,
                frameCount: profile.training.effectiveVisualMemoryFrames
            )
        ], dtype: .float32)
        let history = MLXArray.zeros([1, profile.training.historyLength, ActionLayout.count], dtype: .float32)
        let prediction = model.predictions(images: images, history: history)
        MLX.eval(prediction)
        XCTAssertEqual(prediction.shape, [1, ActionLayout.count])
        XCTAssertTrue(prediction.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testLegacyProfileDecodingPreservesMigrationFieldsAndSchedule() throws {
        let encoded = try JSONEncoder().encode(TrainingConfiguration())
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        object.removeValue(forKey: "learningRateSchedule")
        object.removeValue(forKey: "cosineCycleEpochs")
        object.removeValue(forKey: "plateauPatience")
        object.removeValue(forKey: "minimumLearningRateRatio")
        object.removeValue(forKey: "binaryFocalGamma")
        object.removeValue(forKey: "visualMemoryFrames")
        object.removeValue(forKey: "visualMemoryStride")
        object.removeValue(forKey: "visualMemoryDropout")
        var architecture = try XCTUnwrap(object["architecture"] as? [String: Any])
        architecture.removeValue(forKey: "family")
        architecture.removeValue(forKey: "patchSize")
        architecture["visualPooling"] = VisualPoolingKind.attention.rawValue
        architecture["attentionHeads"] = 6
        architecture.removeValue(forKey: "spatialTokens")
        architecture.removeValue(forKey: "transformerLayers")
        architecture.removeValue(forKey: "transformerHeads")
        architecture.removeValue(forKey: "transformerFeedForward")
        object["architecture"] = architecture
        let legacyData = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        let legacy = try JSONDecoder().decode(TrainingConfiguration.self, from: legacyData)

        XCTAssertNil(legacy.learningRateSchedule)
        XCTAssertNil(legacy.visualMemoryFrames)
        XCTAssertNil(legacy.visualMemoryStride)
        XCTAssertNil(legacy.visualMemoryDropout)
        XCTAssertEqual(legacy.effectiveVisualMemoryFrames, VisualMemoryContract.defaultFrameCount)
        XCTAssertEqual(legacy.effectiveVisualMemoryStride, VisualMemoryContract.defaultStride)
        XCTAssertEqual(legacy.effectiveLearningRateSchedule, .legacyInverseSquareRoot)
        XCTAssertEqual(legacy.effectiveBinaryFocalGamma, 0)
        XCTAssertEqual(legacy.architecture.visualPooling, .attention)
        XCTAssertNil(legacy.architecture.family)
        XCTAssertNil(legacy.architecture.patchSize)
        XCTAssertEqual(legacy.architecture.effectiveFamily, .hybrid)
        XCTAssertEqual(legacy.architecture.effectivePatchSize, 32)
        XCTAssertEqual(legacy.architecture.attentionHeads, 6)
        XCTAssertNil(legacy.architecture.spatialTokens)
        XCTAssertEqual(legacy.architecture.effectiveSpatialTokens, 6)
        XCTAssertNil(legacy.architecture.transformerLayers)
        XCTAssertEqual(legacy.architecture.effectiveTransformerLayers, 3)
        XCTAssertEqual(TrainingConfiguration().effectiveLearningRateSchedule, .adaptiveCosine)
        XCTAssertEqual(TrainingConfiguration().architecture.effectiveSpatialTokens, 8)

        let roundTripped = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(legacy)) as? [String: Any])
        XCTAssertNil(roundTripped["learningRateSchedule"], "Nil compatibility fields must remain absent so legacy checkpoint signatures stay stable.")
        XCTAssertNil(roundTripped["visualMemoryFrames"])
        XCTAssertNil(roundTripped["visualMemoryStride"])
        XCTAssertNil(roundTripped["visualMemoryDropout"])
        let roundTrippedArchitecture = try XCTUnwrap(roundTripped["architecture"] as? [String: Any])
        XCTAssertNil(roundTrippedArchitecture["spatialTokens"])
        XCTAssertNil(roundTrippedArchitecture["transformerLayers"])
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
            VisualMemoryContract.rawInputChannels(
                colorChannels: profile.preprocessing.channelCount,
                frameCount: profile.training.effectiveVisualMemoryFrames
            )
        ], dtype: .float32)
        let history = MLXArray.zeros([1, profile.training.historyLength, ActionLayout.count], dtype: .float32)
        let predictions = model.predictions(images: images, history: history)
        MLX.eval(predictions)
        XCTAssertEqual(predictions.shape, [1, ActionLayout.count])
        XCTAssertTrue(predictions.asArray(Float.self).allSatisfy(\.isFinite))
        let actualParameterCount = model.parameters().flattened().reduce(Int64(0)) { total, item in
            total + item.1.shape.reduce(Int64(1)) { $0 * Int64($1) }
        }
        XCTAssertEqual(actualParameterCount, ModelSizing.parameterCount(profile))
    }

    func testModelSizingMatchesActualHybridTransformerParameterTrees() {
        for (presetName, architecture) in [("Small", ArchitectureSpec.small), ("Balanced", .balanced), ("Large", .large)] {
            for historyLength in [0, 3] {
                var profile = AIProfile.fresh()
                profile.preprocessing = PreprocessingSpec(width: 12, height: 8, colorMode: .grayscale)
                profile.training.architecture = architecture
                profile.training.historyLength = historyLength
                profile.training.precision = .float32
                let model = AgentPolicy(profile: profile)
                let parameters = model.parameters().flattened()
                let actual = parameters.reduce(Int64(0)) { total, item in
                    total + item.1.shape.reduce(Int64(1)) { $0 * Int64($1) }
                }
                XCTAssertEqual(actual, ModelSizing.parameterCount(profile), "Sizing drifted for \(presetName) / history \(historyLength).")
                XCTAssertFalse(parameters.contains { $0.0.contains("coordinate") })
                XCTAssertFalse(parameters.contains { $0.0.contains("tokenPositions") })
                XCTAssertFalse(parameters.contains { $0.0.contains("gru") || $0.0.contains("lstm") })
                XCTAssertTrue(parameters.contains { $0.0.contains("spatialAttention") })
                XCTAssertTrue(parameters.contains { $0.0.contains("transformerBlocks") })
                XCTAssertTrue(parameters.contains { $0.0.contains("readoutToken") })
            }
        }
    }

    func testPureTransformerForwardPaddingAndParameterSizingMatch() {
        for (name, architecture) in [("Small", ArchitectureSpec.pureSmall), ("Balanced", .pureBalanced), ("Large", .pureLarge)] {
            var profile = AIProfile.fresh()
            profile.preprocessing = PreprocessingSpec(width: 33, height: 17, colorMode: .grayscale)
            profile.training.architecture = architecture
            profile.training.historyLength = 2
            profile.training.precision = .float32
            let model = AgentPolicy(profile: profile)
            model.train(false)
            let images = MLXArray.zeros([
                1, 17, 33,
                VisualMemoryContract.rawInputChannels(
                    colorChannels: profile.preprocessing.channelCount,
                    frameCount: profile.training.effectiveVisualMemoryFrames
                )
            ], dtype: .float32)
            let history = MLXArray.zeros([1, 2, ActionLayout.count], dtype: .float32)
            let visual = model.visualActivations(images: images)
            let predictions = model.predictions(images: images, history: history)
            let patchMap = model.spatialTokensForVisualization(visual[0])
            MLX.eval(visual, predictions, patchMap)

            let patch = architecture.effectivePatchSize
            XCTAssertEqual(visual.count, 1)
            XCTAssertEqual(visual[0].shape, [1, (17 + patch - 1) / patch, (33 + patch - 1) / patch, architecture.visualEmbedding])
            XCTAssertEqual(predictions.shape, [1, ActionLayout.count])
            XCTAssertTrue(predictions.asArray(Float.self).allSatisfy(\.isFinite), "\(name) produced a non-finite prediction.")
            XCTAssertEqual(patchMap.shape, [1, (17 + patch - 1) / patch, (33 + patch - 1) / patch, 1])
            XCTAssertEqual(patchMap.asArray(Float.self).reduce(0, +), 1, accuracy: 0.000_01)

            let parameters = model.parameters().flattened()
            let actual = parameters.reduce(Int64(0)) { total, item in
                total + item.1.shape.reduce(Int64(1)) { $0 * Int64($1) }
            }
            XCTAssertEqual(actual, ModelSizing.parameterCount(profile), "Pure \(name) sizing drifted.")
            XCTAssertTrue(parameters.contains { $0.0.contains("patchEmbeddings") })
            XCTAssertFalse(parameters.contains { $0.0.contains("spatialAttention") })
            XCTAssertFalse(parameters.contains { $0.0.contains("convolutions") })
        }
    }

    func testLearnedArchitectureContractNormalizesFamilySpecificFields() {
        let hybrid = ArchitectureSpec.hybridBalanced
        var sameHybrid = hybrid
        sameHybrid.patchSize = 8
        XCTAssertEqual(LearnedBrainArchitectureContract(hybrid), LearnedBrainArchitectureContract(sameHybrid))

        let pure = ArchitectureSpec.pureBalanced
        var samePure = pure
        samePure.convolutionChannels = [9, 9]
        samePure.kernelSizes = [9, 9]
        samePure.strides = [9, 9]
        samePure.spatialTokens = 55
        XCTAssertEqual(LearnedBrainArchitectureContract(pure), LearnedBrainArchitectureContract(samePure))
        XCTAssertNotEqual(LearnedBrainArchitectureContract(hybrid), LearnedBrainArchitectureContract(pure))
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

    func testPipelinedMetalPreprocessingIsByteExactAndOrdered() throws {
        let processor = try VisionPreprocessor()
        let spec = PreprocessingSpec(width: 13, height: 7, colorMode: .color, bitDepth: 6, chroma: .yuv420, resizePolicy: .fit)
        var buffers: [CVPixelBuffer] = []
        for value in [UInt8(17), 83, 149, 231] {
            var candidate: CVPixelBuffer?
            XCTAssertEqual(CVPixelBufferCreate(kCFAllocatorDefault, 16, 10, kCVPixelFormatType_32BGRA, [kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary, &candidate), kCVReturnSuccess)
            let buffer = try XCTUnwrap(candidate)
            CVPixelBufferLockBaseAddress(buffer, [])
            memset(CVPixelBufferGetBaseAddress(buffer), Int32(value), CVPixelBufferGetDataSize(buffer))
            CVPixelBufferUnlockBaseAddress(buffer, [])
            buffers.append(buffer)
        }
        let expected = try buffers.map { try processor.process($0, spec: spec) }
        let jobs = try buffers.map { try processor.submitPackedFrame($0, spec: spec) }
        let actual = try jobs.map { job in
            try processor.finishPackedBytes(job) { Data($0) }
        }
        XCTAssertEqual(actual, expected, "Overlapping decode/preprocessing must not change a single packed byte or frame order.")
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
        try await store.deleteRecordingFolder(folder, includingRecordings: true)
        let recordingsAfter = await store.listRecordings()
        let foldersAfter = await store.listRecordingFolders()
        XCTAssertTrue(recordingsAfter.isEmpty)
        XCTAssertTrue(foldersAfter.isEmpty)
    }

    func testPortableWindowsRecordingImportsTransactionallyIntoTheNativeLibrary() async throws {
        let container = FileManager.default.temporaryDirectory.appendingPathComponent("portable-import-\(UUID().uuidString)", isDirectory: true)
        let root = container.appendingPathComponent("Library", isDirectory: true)
        let transfer = container.appendingPathComponent("Transfer", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: container) }
        try FileManager.default.createDirectory(at: transfer, withIntermediateDirectories: true)

        let store = WorkspaceStore(root: root)
        try await store.prepare()
        let destinationFolder = RecordingFolder(id: UUID(), name: "Windows demonstrations", createdAt: Date())
        try await store.saveRecordingFolder(destinationFolder)

        let originalID = UUID()
        let source = transfer.appendingPathComponent("\(originalID.uuidString).atrrecord", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: false)
        try await writeTestMovie(to: source.appendingPathComponent("capture.mov"), width: 16, height: 16, frameCount: 6, fps: 30)
        let eventsURL = source.appendingPathComponent("events.atrevents")
        let eventWriter = try InputEventWriter(url: eventsURL)
        eventWriter.append(InputSample(timestampNanos: 9_000_000_000, kind: .mouseMove, x: 8, y: 8, modifiers: 0))
        eventWriter.append(InputSample(timestampNanos: 9_100_000_000, kind: .key, keyCode: 13, modifiers: 0, isDown: true))
        XCTAssertEqual(try eventWriter.finish(), 2)
        let eventBytesBefore = try Data(contentsOf: eventsURL)
        let videoBytesBefore = try Data(contentsOf: source.appendingPathComponent("capture.mov"))

        // This is intentionally authored as the Windows System.Text.Json shape,
        // including its platform-translated Apple key codes and second-precision
        // ISO date, rather than encoded through Swift's RecordingManifest.
        let manifestJSON = """
        {
          "schemaVersion": 2,
          "id": "\(originalID.uuidString)",
          "name": "Windows example",
          "createdAt": "2026-07-19T12:00:00Z",
          "hostStartNanos": 9000000000,
          "duration": 0.2,
          "capture": {
            "kind": "Display",
            "displayID": 7,
            "requestedFPS": 30,
            "showsCursor": false
          },
          "globalRect": { "x": 0, "y": 0, "width": 16, "height": 16 },
          "pixelWidth": 16,
          "pixelHeight": 16,
          "deliveredFPS": 30,
          "eventCount": 2,
          "videoFile": "capture.mov",
          "eventFile": "events.atrevents",
          "trimStart": 0,
          "trimEnd": 0.2,
          "excludedKeyCodes": [56, 0]
        }
        """
        try Data(manifestJSON.utf8).write(to: source.appendingPathComponent("manifest.json"))

        let imported = try await store.importRecordings(from: [source], into: destinationFolder.id)
        XCTAssertEqual(imported.count, 1)
        let item = try XCTUnwrap(imported.first)
        XCTAssertNotEqual(item.id, originalID, "Imports get a fresh local ID and cannot overwrite an existing recording")
        XCTAssertEqual(item.manifest.folderID, destinationFolder.id)
        XCTAssertEqual(item.manifest.name, "Windows example")
        XCTAssertEqual(item.manifest.excludedKeyCodes, [0, 56])
        XCTAssertEqual(try InputEventReader.read(url: item.directory.appendingPathComponent(item.manifest.eventFile)).map(\.keyCode), [0, 13])
        XCTAssertEqual(try Data(contentsOf: item.directory.appendingPathComponent(item.manifest.eventFile)), eventBytesBefore)
        XCTAssertEqual(try Data(contentsOf: item.directory.appendingPathComponent(item.manifest.videoFile)), videoBytesBefore)
        XCTAssertEqual(try Data(contentsOf: eventsURL), eventBytesBefore, "Import must never modify its source")
        XCTAssertEqual(try Data(contentsOf: source.appendingPathComponent("capture.mov")), videoBytesBefore)

        let archive = transfer.appendingPathComponent("Windows example.atrrecord.zip")
        let zipProcess = Process()
        zipProcess.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        zipProcess.arguments = ["-c", "-k", "--keepParent", source.path, archive.path]
        try zipProcess.run()
        zipProcess.waitUntilExit()
        XCTAssertEqual(zipProcess.terminationStatus, 0)

        let archiveImport = try await store.importRecordings(from: [archive], into: destinationFolder.id)
        let archivedItem = try XCTUnwrap(archiveImport.first)
        XCTAssertEqual(archiveImport.count, 1)
        XCTAssertNotEqual(archivedItem.id, originalID)
        XCTAssertNotEqual(archivedItem.id, item.id)
        XCTAssertEqual(try Data(contentsOf: archivedItem.directory.appendingPathComponent(archivedItem.manifest.eventFile)), eventBytesBefore)
        XCTAssertEqual(try Data(contentsOf: archivedItem.directory.appendingPathComponent(archivedItem.manifest.videoFile)), videoBytesBefore)
    }

    func testPortableRecordingBatchValidationPublishesNothingWhenOneSourceIsCorrupt() async throws {
        let container = FileManager.default.temporaryDirectory.appendingPathComponent("portable-import-reject-\(UUID().uuidString)", isDirectory: true)
        let root = container.appendingPathComponent("Library", isDirectory: true)
        let transfer = container.appendingPathComponent("Transfer", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: container) }
        try FileManager.default.createDirectory(at: transfer, withIntermediateDirectories: true)
        let store = WorkspaceStore(root: root)
        try await store.prepare()
        let folder = RecordingFolder(id: UUID(), name: "Imports", createdAt: Date())
        try await store.saveRecordingFolder(folder)

        var sources: [URL] = []
        for index in 0..<2 {
            let id = UUID()
            let source = transfer.appendingPathComponent("\(id.uuidString).atrrecord", isDirectory: true)
            try FileManager.default.createDirectory(at: source, withIntermediateDirectories: false)
            try await writeTestMovie(to: source.appendingPathComponent("capture.mov"), width: 8, height: 8, frameCount: 3, fps: 30)
            let writer = try InputEventWriter(url: source.appendingPathComponent("events.atrevents"))
            writer.append(InputSample(timestampNanos: 1_000_000_000, kind: .mouseMove, x: 4, y: 4))
            _ = try writer.finish()
            let manifest = RecordingManifest(
                id: id, name: "Import \(index)", createdAt: Date(), hostStartNanos: 1_000_000_000,
                duration: 0.1, capture: CaptureSpec(requestedFPS: 30),
                globalRect: CodableRect(CGRect(x: 0, y: 0, width: 8, height: 8)),
                pixelWidth: 8, pixelHeight: 8, deliveredFPS: 30, eventCount: 1
            )
            try await store.writeRecording(manifest, to: source)
            sources.append(source)
        }
        var corrupt = try Data(contentsOf: sources[1].appendingPathComponent("events.atrevents"))
        corrupt[12 + 8] = 0xFF // First record's event kind.
        try corrupt.write(to: sources[1].appendingPathComponent("events.atrevents"))

        do {
            _ = try await store.importRecordings(from: sources, into: folder.id)
            XCTFail("A corrupt source must reject the complete import batch")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("invalid input stream"))
        }
        let importedRecordings = await store.listRecordings()
        XCTAssertTrue(importedRecordings.isEmpty)
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
        profile.training.historyLength = 128
        profile.training.learningRate = 0.005
        profile.training.learningRateSchedule = nil
        profile.training.cosineCycleEpochs = nil
        profile.training.plateauPatience = nil
        profile.training.minimumLearningRateRatio = nil
        profile.training.binaryFocalGamma = nil
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
        let compatibleCache = (await store.cacheDirectory())
            .appendingPathComponent("preserved.atrcache", isDirectory: true)
        try FileManager.default.createDirectory(at: compatibleCache, withIntermediateDirectories: true)
        let cachedPixels = Data([7, 6, 5, 4])
        try cachedPixels.write(to: compatibleCache.appendingPathComponent("observations.bin"))

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
        XCTAssertEqual(
            try Data(contentsOf: compatibleCache.appendingPathComponent("observations.bin")),
            cachedPixels,
            "A model-contract migration must not discard reusable sampled-video caches."
        )
        XCTAssertEqual(Set(remainingProfiles.map(\.name)), ["Preserved profile", "Crystal V4"])
        XCTAssertTrue(remainingProfiles.allSatisfy { $0.activeVersionID == nil })
        let migrated = try XCTUnwrap(remainingProfiles.first { $0.name == "Preserved profile" })
        XCTAssertEqual(migrated.training.architecture, .large)
        XCTAssertEqual(migrated.training.historyLength, PolicyInputContract.actionHistoryLength)
        XCTAssertEqual(migrated.training.learningRate, 0.0003)
        XCTAssertEqual(migrated.training.effectiveLearningRateSchedule, .adaptiveCosine)
        XCTAssertEqual(migrated.training.effectiveCosineCycleEpochs, 8)
        XCTAssertEqual(migrated.training.effectivePlateauPatience, 5)
        XCTAssertEqual(migrated.training.effectiveMinimumLearningRateRatio, 0.05)
        XCTAssertEqual(migrated.training.effectiveBinaryFocalGamma, 1.5)
        let secondPass = try await store.removeObsoleteModelArtifacts(currentSchema: ModelContract.schemaVersion)
        XCTAssertEqual(secondPass, 0)
    }

    func testModelContractMigrationPreservesExplicitPureFamilyAndFillsVisualMemoryDefaults() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("workspace-pure-contract-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()

        var profile = AIProfile.fresh(name: "Pure legacy brain")
        profile.training.architecture = .pureLarge
        profile.training.visualMemoryFrames = nil
        profile.training.visualMemoryStride = nil
        profile.training.visualMemoryDropout = nil
        let versionID = UUID()
        profile.activeVersionID = versionID
        try await store.saveProfile(profile)

        var legacyVersion = ModelVersionManifest(
            id: versionID,
            name: "Policy v6",
            createdAt: Date(),
            globalStep: 900,
            trainingLoss: 0.2,
            validationLoss: nil,
            preprocessing: profile.preprocessing,
            channels: profile.channels,
            training: profile.training
        )
        legacyVersion.schemaVersion = ModelContract.schemaVersion - 1
        try await store.saveVersionManifest(legacyVersion, profileID: profile.id)
        let versionDirectory = await store.versionDirectory(profileID: profile.id, versionID: versionID)
        let originalWeights = Data("pure-v6-weights".utf8)
        try originalWeights.write(to: versionDirectory.appendingPathComponent(legacyVersion.weightsFile))

        let archivedCount = try await store.removeObsoleteModelArtifacts(
            currentSchema: ModelContract.schemaVersion
        )
        XCTAssertEqual(archivedCount, 1)

        let migratedProfiles = await store.listProfiles()
        let migrated = try XCTUnwrap(migratedProfiles.first)
        XCTAssertNil(migrated.activeVersionID)
        XCTAssertEqual(migrated.training.architecture.effectiveFamily, .pureTransformer)
        XCTAssertEqual(migrated.training.architecture, .pureLarge)
        XCTAssertEqual(migrated.training.visualMemoryFrames, VisualMemoryContract.defaultFrameCount)
        XCTAssertEqual(migrated.training.visualMemoryStride, VisualMemoryContract.defaultStride)
        XCTAssertEqual(migrated.training.visualMemoryDropout, VisualMemoryContract.defaultDropout)

        let archivedWeights = (await store.profileDirectory(profile.id))
            .appendingPathComponent(
                "Archived Model Artifacts/Model Contract \(ModelContract.schemaVersion - 1)/Versions/\(versionID.uuidString)/\(legacyVersion.weightsFile)"
            )
        XCTAssertEqual(try Data(contentsOf: archivedWeights), originalWeights)
        XCTAssertFalse(FileManager.default.fileExists(atPath: versionDirectory.path))
    }

    func testModelContractAuditPreservesCompatibleBrainsInMixedCurrentLibrary() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("workspace-contract-audit-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(root: root)
        try await store.prepare()

        var profile = AIProfile.fresh(name: "Current mixed library")
        profile.training.historyLength = 73
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
        try JSONEncoder().encode(ModelContract.schemaVersion).write(
            to: checkpoint.appendingPathComponent("model-schema.json"),
            options: .atomic
        )

        // Neither a library-wide marker nor another current-schema version is
        // enough to bless an unmarked checkpoint's independent tensor files.
        let uncertain = AIProfile.fresh(name: "Uncertain checkpoint")
        try await store.saveProfile(uncertain)
        let uncertainCheckpoint = await store.checkpointDirectory(profileID: uncertain.id)
        try FileManager.default.createDirectory(at: uncertainCheckpoint, withIntermediateDirectories: true)
        try Data("unmarked".utf8).write(to: uncertainCheckpoint.appendingPathComponent("weights.safetensors"))
        try JSONEncoder().encode(ModelContract.schemaVersion).write(to: root.appendingPathComponent("model-contract.json"), options: .atomic)

        let archived = try await store.removeObsoleteModelArtifacts(currentSchema: ModelContract.schemaVersion)

        XCTAssertEqual(archived, 2)
        XCTAssertEqual(try Data(contentsOf: currentDirectory.appendingPathComponent(current.weightsFile)), currentWeights)
        XCTAssertEqual(try Data(contentsOf: checkpoint.appendingPathComponent("weights.safetensors")), checkpointWeights)
        let archivedLegacy = (await store.profileDirectory(profile.id))
            .appendingPathComponent("Archived Model Artifacts/Model Contract \(ModelContract.schemaVersion - 1)/Versions/\(legacy.id.uuidString)", isDirectory: true)
        XCTAssertTrue(FileManager.default.fileExists(atPath: archivedLegacy.appendingPathComponent(legacy.weightsFile).path))
        let archivedUncertain = (await store.profileDirectory(uncertain.id))
            .appendingPathComponent("Archived Model Artifacts/Model Contract Unknown/Checkpoints/Checkpoint/weights.safetensors")
        XCTAssertTrue(FileManager.default.fileExists(atPath: archivedUncertain.path))
        XCTAssertEqual(try Data(contentsOf: checkpoint.appendingPathComponent("state.json")), checkpointState)
        XCTAssertEqual(try JSONDecoder().decode(Int.self, from: Data(contentsOf: checkpoint.appendingPathComponent("model-schema.json"))), ModelContract.schemaVersion)
        XCTAssertFalse(FileManager.default.fileExists(atPath: legacyDirectory.path))
        let reloadedProfiles = await store.listProfiles()
        let reloaded = try XCTUnwrap(reloadedProfiles.first { $0.id == profile.id })
        XCTAssertEqual(reloaded.activeVersionID, currentID)
        XCTAssertEqual(reloaded.trainingProgress?.globalStep, profile.trainingProgress?.globalStep)
        XCTAssertEqual(reloaded.trainingProgress?.epoch, profile.trainingProgress?.epoch)
        XCTAssertEqual(reloaded.trainingProgress?.savedBrainCount, profile.trainingProgress?.savedBrainCount)
        XCTAssertEqual(reloaded.training.historyLength, 73)
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

    func testConfirmedArchitectureResetArchivesLearningButNeverProtectedBrains() async throws {
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
        try Data("old-version".utf8).write(to: regularVersions.appendingPathComponent("weights.bin"))
        try Data("old-checkpoint".utf8).write(to: regularCheckpoint.appendingPathComponent("state.bin"))
        regular.training.architecture = .large
        let reset = try await store.resetLearning(for: regular)
        XCTAssertNil(reset.activeVersionID)
        XCTAssertNil(reset.trainingProgress)
        XCTAssertFalse(FileManager.default.fileExists(atPath: regularVersions.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: regularCheckpoint.path))
        let archivedRoot = (await store.profileDirectory(regular.id))
            .appendingPathComponent("Archived Model Artifacts/Manual Architecture Changes", isDirectory: true)
        let archivedReset = try XCTUnwrap(FileManager.default.contentsOfDirectory(at: archivedRoot, includingPropertiesForKeys: nil).first)
        XCTAssertEqual(try Data(contentsOf: archivedReset.appendingPathComponent("Versions/weights.bin")), Data("old-version".utf8))
        XCTAssertEqual(try Data(contentsOf: archivedReset.appendingPathComponent("Checkpoint/state.bin")), Data("old-checkpoint".utf8))
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

    func testCompletedTrainingMayUseBestWhileCurrentSnapshotsRemainCurrent() {
        XCTAssertFalse(SnapshotPublicationReason.pause.completed)
        XCTAssertFalse(SnapshotPublicationReason.pause.isAutosave)
        XCTAssertEqual(SnapshotPublicationReason.pause.currentName, "Paused Brain")
        XCTAssertTrue(SnapshotPublicationReason.autosave.isAutosave)
        XCTAssertTrue(SnapshotPublicationReason.completion.completed)

        XCTAssertTrue(RunnableSnapshotSelection.usesValidatedBest(
            preferBest: true,
            bestGlobalStep: 12_345,
            bestWeightsExist: true
        ))
        XCTAssertFalse(RunnableSnapshotSelection.usesValidatedBest(
            preferBest: true,
            bestGlobalStep: nil,
            bestWeightsExist: true
        ))
        XCTAssertFalse(RunnableSnapshotSelection.usesValidatedBest(
            preferBest: false,
            bestGlobalStep: 12_345,
            bestWeightsExist: true
        ))

        XCTAssertTrue(RunnableSnapshotSelection.currentSnapshotHasMatchingEvaluation(
            evaluationGlobalStep: 12_345,
            currentGlobalStep: 12_345,
            hasEvaluation: true
        ))
        XCTAssertFalse(RunnableSnapshotSelection.currentSnapshotHasMatchingEvaluation(
            evaluationGlobalStep: 12_344,
            currentGlobalStep: 12_345,
            hasEvaluation: true
        ))
        XCTAssertFalse(RunnableSnapshotSelection.currentSnapshotHasMatchingEvaluation(
            evaluationGlobalStep: 12_345,
            currentGlobalStep: 12_345,
            hasEvaluation: false
        ))
    }

    func testRunnableBestSelectionPrefersDemonstratedExecutionQualityOverLoss() {
        let weakMetrics = BinaryValidationMetrics(
            truePositives: 4,
            falsePositives: 1,
            falseNegatives: 16,
            trueNegatives: 79
        )
        let strongMetrics = BinaryValidationMetrics(
            truePositives: 14,
            falsePositives: 6,
            falseNegatives: 6,
            trueNegatives: 74
        )
        var weak = ValidationReport(
            sampleCount: 100,
            binary: weakMetrics,
            buttons: nil,
            keyboard: weakMetrics,
            modifiers: nil,
            absoluteMouseMAE: nil,
            activeRelativeMouseMAE: nil,
            activeScrollMAE: nil,
            idleContinuousFalseActionRate: 0.01,
            activeRelativeMouseExecutionRecall: 0.10
        )
        weak.binaryOutputs = [
            BinaryOutputValidation(
                outputIndex: ActionLayout.keyboard.lowerBound,
                decisionThreshold: 0.5,
                defaultMetrics: weakMetrics,
                calibratedMetrics: weakMetrics
            )
        ]
        var strong = weak
        strong.binary = strongMetrics
        strong.keyboard = strongMetrics
        strong.binaryOutputs?[0].defaultMetrics = strongMetrics
        strong.binaryOutputs?[0].calibratedMetrics = strongMetrics
        strong.activeRelativeMouseExecutionRecall = 0.35

        XCTAssertGreaterThan(strong.deploymentQualityScore ?? 0, weak.deploymentQualityScore ?? 0)
        XCTAssertTrue(RunnableSnapshotSelection.improvesValidatedQuality(
            candidateReport: strong,
            candidateLoss: 1.2,
            bestReport: weak,
            bestLoss: 1.0
        ))
        XCTAssertFalse(RunnableSnapshotSelection.improvesValidatedQuality(
            candidateReport: weak,
            candidateLoss: 0.8,
            bestReport: strong,
            bestLoss: 1.0
        ))

        var warningOnly = weak
        let collapsed = BinaryValidationMetrics(
            truePositives: 0,
            falsePositives: 0,
            falseNegatives: 20,
            trueNegatives: 80
        )
        warningOnly.binary = collapsed
        warningOnly.keyboard = collapsed
        warningOnly.binaryOutputs?[0].defaultMetrics = collapsed
        warningOnly.binaryOutputs?[0].calibratedMetrics = collapsed
        XCTAssertTrue(warningOnly.hasBinaryRecallCollapse)
        XCTAssertTrue(
            RunnableSnapshotSelection.improvesValidatedQuality(
                candidateReport: warningOnly,
                candidateLoss: 1.5,
                bestReport: nil,
                bestLoss: nil
            ),
            "Quality warnings must not make the first finite candidate ineligible."
        )

        var inSample = strong
        inSample.evaluationScope = .trainingCalibration
        XCTAssertFalse(
            RunnableSnapshotSelection.improvesValidatedQuality(
                candidateReport: inSample,
                candidateLoss: 0.01,
                bestReport: nil,
                bestLoss: nil
            ),
            "In-sample calibration may tune runtime thresholds but must never select the best brain."
        )
    }

    func testOnlyExactVisionAndArchitectureChangeTheLearnedBrainContract() {
        var profile = AIProfile.fresh()
        let original = profile.learnedBrainContract
        profile.training.epochs = 9_999
        profile.training.learningRate = 0.00001
        profile.training.historyLength = 32
        profile.training.visualMemoryDropout = 0.25
        profile.training.architecture.dropout = 0.2
        profile.training.architecture.recurrentWidth += 1
        profile.training.architecture.recurrentKind = .lstm
        profile.training.architecture.visualPooling = .attention
        XCTAssertEqual(profile.learnedBrainContract, original)
        profile.training.visualMemoryFrames = 3
        XCTAssertNotEqual(profile.learnedBrainContract, original)
        profile.training.visualMemoryFrames = VisualMemoryContract.defaultFrameCount
        profile.training.architecture.transformerFeedForward = 1_024
        XCTAssertNotEqual(profile.learnedBrainContract, original)
        profile.training.architecture.transformerFeedForward = ArchitectureSpec.balanced.transformerFeedForward
        profile.preprocessing.width += 1
        XCTAssertNotEqual(profile.learnedBrainContract, original)
    }

    func testIrrelevantVisualMemorySpacingDoesNotArchiveCompatibleWeights() {
        var profile = AIProfile.fresh()
        profile.training.visualMemoryFrames = 0
        let disabled = profile.learnedBrainContract
        profile.training.visualMemoryStride = VisualMemoryContract.maximumStride
        XCTAssertEqual(profile.learnedBrainContract, disabled)

        profile.training.visualMemoryFrames = 1
        let immediateOnly = profile.learnedBrainContract
        profile.training.visualMemoryStride = 1
        XCTAssertEqual(profile.learnedBrainContract, immediateOnly)

        profile.training.visualMemoryFrames = 2
        let multiTimescale = profile.learnedBrainContract
        profile.training.visualMemoryStride = 2
        XCTAssertNotEqual(profile.learnedBrainContract, multiTimescale)
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
        let directory = await store.versionDirectory(profileID: profile.id, versionID: versionID)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("brain".utf8).write(to: directory.appendingPathComponent("weights.safetensors"))
        try Data("optimizer".utf8).write(to: directory.appendingPathComponent("optimizer.safetensors"))
        try Data("state".utf8).write(to: directory.appendingPathComponent("state.json"))
        let version = ModelVersionManifest(id: versionID, name: "Epoch 120", createdAt: Date(), globalStep: 240, trainingLoss: 0.2, validationLoss: nil, preprocessing: profile.preprocessing, channels: profile.channels, training: profile.training, optimizerFile: "optimizer.safetensors", trainingStateFile: "state.json", epoch: 120, isAutosave: true)
        let restored = try await store.restoreVersionAsCheckpoint(profileID: profile.id, version: version)
        XCTAssertTrue(restored)
        let checkpoint = await store.checkpointDirectory(profileID: profile.id)
        XCTAssertEqual(try Data(contentsOf: checkpoint.appendingPathComponent("weights.safetensors")), Data("brain".utf8))
        XCTAssertEqual(try Data(contentsOf: checkpoint.appendingPathComponent("optimizer.safetensors")), Data("optimizer".utf8))
        XCTAssertEqual(try Data(contentsOf: checkpoint.appendingPathComponent("state.json")), Data("state".utf8))

        let best = ModelVersionManifest(id: UUID(), name: "Best", createdAt: Date(), globalStep: 120, trainingLoss: 0.1, validationLoss: 0.1, preprocessing: profile.preprocessing, channels: profile.channels, training: profile.training)
        let staleRestored = try await store.restoreVersionAsCheckpoint(profileID: profile.id, version: best)
        XCTAssertFalse(staleRestored)
        XCTAssertFalse(FileManager.default.fileExists(atPath: checkpoint.path), "A stale newer checkpoint must not override an explicitly activated weights-only brain")
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
        XCTAssertEqual(
            GameCameraContract.postedDelta(
                forPrediction: GameCameraContract.trainingValue(forRawDelta: 1),
                sensitivity: 1
            ),
            0
        )
        XCTAssertEqual(
            GameCameraContract.postedDelta(
                forPrediction: GameCameraContract.trainingValue(forRawDelta: 2),
                sensitivity: 1
            ),
            2
        )
        XCTAssertEqual(
            GameCameraContract.postedDelta(
                forPrediction: GameCameraContract.trainingValue(forRawDelta: 2),
                sensitivity: 1,
                minimumMagnitude: 2.5
            ),
            0
        )
        XCTAssertEqual(
            GameCameraContract.calibratedMinimumPostedMagnitude(
                activeCorrectCounts: [9, 8, 7, 6, 5, 4, 3, 2, 1],
                activeCount: 10,
                idleFalseCounts: [20, 8, 5, 3, 2, 1, 1, 0, 0],
                idleCount: 100
            ),
            1
        )
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
        XCTAssertEqual(repeated?.values, prediction)

        latch.publish(prediction, at: 124)
        XCTAssertTrue(latch.consume()?.isFresh == true)
        latch.reset()
        XCTAssertNil(latch.consume())
    }

    func testCalibratedThresholdsMatchPhysicalExecution() {
        let collector = EventCollector()
        let injector = InputInjector(eventSink: { collector.append($0) })
        var profile = AIProfile.fresh()
        profile.channels = ActionChannels(
            absoluteMouse: false,
            relativeMouse: false,
            buttons: true,
            scroll: false,
            keyboard: true,
            modifiers: false
        )
        var thresholds = [Float](repeating: 0.5, count: ActionLayout.count)
        thresholds[ActionLayout.buttons.lowerBound] = 0.7
        thresholds[ActionLayout.keyboard.lowerBound + 13] = 0.8
        var prediction = [Float](repeating: 0, count: ActionLayout.count)
        prediction[ActionLayout.buttons.lowerBound] = 0.69
        prediction[ActionLayout.keyboard.lowerBound + 13] = 0.79

        injector.enable()
        injector.execute(
            prediction,
            profile: profile,
            allowedKeyCodes: [13],
            mouseMode: .absolute,
            captureRect: CGRect(x: 0, y: 0, width: 100, height: 100),
            safety: AgentSafetyPolicy(),
            binaryDecisionThresholds: thresholds
        )
        XCTAssertFalse(collector.events.contains { $0.0 == .leftMouseDown || $0.0 == .keyDown })

        prediction[ActionLayout.buttons.lowerBound] = 0.71
        prediction[ActionLayout.keyboard.lowerBound + 13] = 0.81
        injector.execute(
            prediction,
            profile: profile,
            allowedKeyCodes: [13],
            mouseMode: .absolute,
            captureRect: CGRect(x: 0, y: 0, width: 100, height: 100),
            safety: AgentSafetyPolicy(),
            binaryDecisionThresholds: thresholds
        )
        XCTAssertTrue(collector.events.contains { $0.0 == .leftMouseDown })
        XCTAssertTrue(collector.events.contains { $0.0 == .keyDown && $0.1 == 13 })
        injector.disableAndReleaseAll()
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

    func testMouseModeEvidenceIgnoresStationaryRecordingsAndUsesClassifiedVotes() {
        var camera = InputEventReader.MouseDiagnostics()
        camera.moveEventCount = 200
        camera.nonzeroDeltaCount = 120
        camera.absolutePositionChangeCount = 0
        XCTAssertEqual(camera.controlEvidence, .gameCamera)

        var cursor = InputEventReader.MouseDiagnostics()
        cursor.moveEventCount = 80
        cursor.nonzeroDeltaCount = 60
        cursor.absolutePositionChangeCount = 60
        XCTAssertEqual(cursor.controlEvidence, .movingCursor)

        var stationary = InputEventReader.MouseDiagnostics()
        stationary.moveEventCount = 10_000
        XCTAssertEqual(stationary.controlEvidence, .insufficient)

        var evidence = InputEventReader.MouseModeEvidence()
        evidence.include(stationary)
        evidence.include(camera)
        XCTAssertEqual(evidence.recommendedMode, .relative)
        evidence.include(cursor)
        XCTAssertEqual(evidence.recommendedMode, .relative, "A tied recording vote should use classified motion-event evidence.")
        evidence.include(cursor)
        XCTAssertEqual(evidence.recommendedMode, .absolute, "Classified recording count should take precedence over event volume.")
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

    func testTemporalVisionTensorContainsCurrentPixelsAndSignedFrameDifference() {
        let spec = PreprocessingSpec(width: 2, height: 1, colorMode: .grayscale, bitDepth: 8)
        let tensor = VisionPreprocessor.mlxTemporalTensor(
            current: Data([255, 0]),
            previous: Data([0, 255]),
            batch: 1,
            spec: spec
        )
        MLX.eval(tensor)
        XCTAssertEqual(tensor.shape, [1, 1, 2, 2])
        let values = tensor.asArray(Float.self)
        XCTAssertEqual(values[0], 1, accuracy: 0.000_001)
        XCTAssertEqual(values[1], 1, accuracy: 0.000_001)
        XCTAssertEqual(values[2], 0, accuracy: 0.000_001)
        XCTAssertEqual(values[3], -1, accuracy: 0.000_001)

        let firstFrame = VisionPreprocessor.mlxTemporalTensor(current: Data([255, 0]), previous: nil, batch: 1, spec: spec)
        MLX.eval(firstFrame)
        XCTAssertEqual(firstFrame.asArray(Float.self), [1, 0, 0, 0])
    }

    func testVisualMemoryTensorPreservesMultipleLagsAndAvailability() {
        let spec = PreprocessingSpec(width: 2, height: 1, colorMode: .grayscale, bitDepth: 8)
        let memory = PackedVisualMemoryContext(
            packedFrames: Data([
                0, 128,     // lag 1
                255, 255    // older slot
            ]),
            availability: [1, 0.5]
        )
        let tensor = VisionPreprocessor.mlxVisualMemoryTensor(
            current: Data([255, 128]),
            memory: memory,
            batch: 1,
            frameCount: 2,
            spec: spec
        )
        MLX.eval(tensor)
        XCTAssertEqual(tensor.shape, [1, 1, 2, 5])
        let values = tensor.asArray(Float.self)
        XCTAssertEqual(Array(values[0..<5]), [1, 1, 0, 1, 0.5])
        XCTAssertEqual(values[5], Float(128) / 255, accuracy: 0.000_001)
        XCTAssertEqual(values[6], 0, accuracy: 0.000_001)
        XCTAssertEqual(values[7], Float(-127) / 255, accuracy: 0.000_001)
        XCTAssertEqual(values[8], 1, accuracy: 0.000_001)
        XCTAssertEqual(values[9], 0.5, accuracy: 0.000_001)
    }

    func testPackedFrameHistoryClampsStartupWithoutPretendingItIsFull() {
        var empty = PackedFrameHistory(capacity: 5)
        let startup = empty.context(current: Data([40]), lags: [1, 3, 5])
        XCTAssertEqual(startup.packedFrames, Data([40, 40, 40]))
        XCTAssertEqual(startup.availability, [0, 0, 0])

        empty.append(Data([10]))
        empty.append(Data([20]))
        empty.append(Data([30]))
        let context = empty.context(current: Data([40]), lags: [1, 3, 5])
        XCTAssertEqual(context.packedFrames, Data([30, 10, 10]))
        XCTAssertEqual(context.availability[0], 1, accuracy: 0.000_001)
        XCTAssertEqual(context.availability[1], 1, accuracy: 0.000_001)
        XCTAssertEqual(context.availability[2], 0.6, accuracy: 0.000_001)
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
        XCTAssertEqual(finalLayer.featureChannelCount, 8)
        finalLayer.featureChannelCount = 9 // Previous persisted default.
        XCTAssertEqual(finalLayer.sanitized().featureChannelCount, 8)
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
        profile.training.architecture.spatialTokens = 32
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        let tensor = MLXArray([Float](repeating: 0.5, count: 120 * 200 * 32), [1, 120, 200, 32])
        let sampled = model.sampledForVisualization(tensor)
        let channels = model.strongestChannelsForVisualization(tensor)
        let routing = model.spatialTokensForVisualization(MLXArray.zeros([1, 2, 2, 96]))
        MLX.eval(sampled, channels, routing)
        XCTAssertLessThanOrEqual(max(sampled.dim(1), sampled.dim(2)), 96)
        XCTAssertEqual(channels.shape, [1, sampled.dim(1), sampled.dim(2), 16])
        XCTAssertEqual(routing.shape, [1, 2, 2, 16])
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
        XCTAssertTrue(overlay.detail.contains("Conv 3"))

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
        XCTAssertTrue(grid.detail.contains("top 4 feature maps"))

        settings.mode = .spatialTokens
        let tokens = CNNVisualizationImageRenderer.render(CNNVisualizationFrame(packed: packed, spec: spec, settings: settings, tensors: [gridTensor], timestamp: 0))
        XCTAssertEqual(try XCTUnwrap(tokens.image).size, NSSize(width: 600, height: 360))
        XCTAssertTrue(tokens.detail.contains("4 learned spatial-token routing maps"))
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
            arrays: ["m.placeholder": MLXArray.zeros([1])],
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

    func testCompiledTrainingStepMatchesUncompiledAdamW() throws {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 16, height: 12, colorMode: .grayscale, bitDepth: 8)
        profile.training.historyLength = 1
        profile.training.visualMemoryDropout = 0
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
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
        let images = grayscaleTemporalTensor(batch: 2, width: 16, height: 12, value: 0.25)
        let history = MLXArray([Float](repeating: 0, count: 2 * ActionLayout.count), [2, 1, ActionLayout.count])
        let targets = MLXArray([Float](repeating: 0, count: 2 * ActionLayout.count), [2, ActionLayout.count])

        let gradientA = valueAndGrad(model: modelA) { model, arrays in [model.loss(images: arrays[0], history: arrays[1], targets: arrays[2])] }
        let resultA = gradientA(modelA, [images, history, targets])
        optimizerA.update(model: modelA, gradients: resultA.1, targetType: modelA.dtype)
        MLX.eval(resultA.0, modelA.parameters(), optimizerA.stateArrays())

        let compiled = compile(inputs: [modelB, optimizerB], outputs: [modelB, optimizerB]) { images, history, targets in
            let result = valueAndGrad(model: modelB) { model, arrays in [model.loss(images: arrays[0], history: arrays[1], targets: arrays[2])] }(modelB, [images, history, targets])
            optimizerB.update(model: modelB, gradients: result.1, targetType: modelB.dtype)
            return result.0[0]
        }
        let lossB = compiled(images, history, targets)
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

    func testAdamWDecayTargetsOnlyCapacityBearingWeights() {
        XCTAssertTrue(ResumableAdamW.appliesWeightDecay(parameterName: "convolutions.0.weight", dimensions: 4))
        XCTAssertTrue(ResumableAdamW.appliesWeightDecay(parameterName: "transformerBlocks.0.attention.queryProjection.weight", dimensions: 2))
        XCTAssertFalse(ResumableAdamW.appliesWeightDecay(parameterName: "transformerBlocks.0.attention.queryProjection.bias", dimensions: 1))
        XCTAssertFalse(ResumableAdamW.appliesWeightDecay(parameterName: "transformerBlocks.0.attentionNormalization.weight", dimensions: 1))
        XCTAssertFalse(ResumableAdamW.appliesWeightDecay(parameterName: "readoutToken", dimensions: 3))
        XCTAssertFalse(ResumableAdamW.appliesWeightDecay(parameterName: "tokenTypeEmbedding.weight", dimensions: 2))
    }

    func testCompiledTrainingThroughputAndActiveMemoryStayBounded() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 12, height: 8, colorMode: .grayscale, bitDepth: 8)
        profile.training.historyLength = 1
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.visualMemoryDropout = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        let optimizer = ResumableAdamW(learningRate: 0.001, weightDecay: 0.01)
        optimizer.initialize(model: model)
        let images = grayscaleTemporalTensor(batch: 2, width: 12, height: 8, value: 0.25)
        let history = MLXArray([Float](repeating: 0, count: 2 * ActionLayout.count), [2, 1, ActionLayout.count])
        let targets = MLXArray([Float](repeating: 0, count: 2 * ActionLayout.count), [2, ActionLayout.count])
        let step = compile(inputs: [model, optimizer], outputs: [model, optimizer]) { images, history, targets in
            let result = valueAndGrad(model: model) { model, arrays in [model.loss(images: arrays[0], history: arrays[1], targets: arrays[2])] }(model, [images, history, targets])
            optimizer.update(model: model, gradients: result.1, targetType: model.dtype)
            return result.0[0]
        }
        var durations: [Double] = []
        var settledMemory = 0
        for iteration in 0..<140 {
            let began = CFAbsoluteTimeGetCurrent()
            let loss = step(images, history, targets)
            MLX.eval(loss, model.parameters(), optimizer.stateArrays())
            if iteration >= 20 { durations.append(CFAbsoluteTimeGetCurrent() - began) }
            if iteration == 60 { settledMemory = Memory.activeMemory }
        }
        let first = durations.prefix(30).reduce(0, +) / 30
        let last = durations.suffix(30).reduce(0, +) / 30
        XCTAssertLessThan(last, first * 3.5 + 0.001, "Fixed-shape compiled training progressively slowed")
        XCTAssertLessThan(abs(Memory.activeMemory - settledMemory), 64 << 20, "Active MLX memory continued growing after warm-up")
    }

    func testCompiledHybridInferenceThroughputAndMemoryStayBounded() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 64, height: 36, colorMode: .grayscale, bitDepth: 8)
        profile.training.architecture = .small
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        model.train(false)
        let images = grayscaleTemporalTensor(batch: 1, width: 64, height: 36, value: 0.25)
        let history = MLXArray.zeros([1, profile.training.historyLength, ActionLayout.count])
        let inference = compile(inputs: [model]) { images, history in
            model.predictions(images: images, history: history)
        }
        var durations: [Double] = []
        var settledMemory = 0
        var last: MLXArray?
        for iteration in 0..<120 {
            let began = CFAbsoluteTimeGetCurrent()
            let prediction = inference(images, history)
            MLX.eval(prediction)
            last = prediction
            if iteration >= 20 { durations.append(CFAbsoluteTimeGetCurrent() - began) }
            if iteration == 60 { settledMemory = Memory.activeMemory }
        }
        let first = durations.prefix(25).reduce(0, +) / 25
        let final = durations.suffix(25).reduce(0, +) / 25
        XCTAssertLessThan(final, first * 3.5 + 0.001, "Fixed-shape hybrid inference progressively slowed")
        XCTAssertLessThan(abs(Memory.activeMemory - settledMemory), 32 << 20, "Hybrid inference memory continued growing after warm-up")
        XCTAssertTrue(last?.asArray(Float.self).allSatisfy(\.isFinite) == true)
    }

    func testCompiledPureTransformerInferenceThroughputAndMemoryStayBounded() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 64, height: 36, colorMode: .grayscale, bitDepth: 8)
        profile.training.architecture = .pureSmall
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        model.train(false)
        let images = grayscaleTemporalTensor(batch: 1, width: 64, height: 36, value: 0.25)
        let history = MLXArray.zeros([1, profile.training.historyLength, ActionLayout.count])
        let inference = compile(inputs: [model]) { images, history in
            model.predictions(images: images, history: history)
        }
        var durations: [Double] = []
        var settledMemory = 0
        var last: MLXArray?
        for iteration in 0..<100 {
            let began = CFAbsoluteTimeGetCurrent()
            let prediction = inference(images, history)
            MLX.eval(prediction)
            last = prediction
            if iteration >= 20 { durations.append(CFAbsoluteTimeGetCurrent() - began) }
            if iteration == 55 { settledMemory = Memory.activeMemory }
        }
        let first = durations.prefix(20).reduce(0, +) / 20
        let final = durations.suffix(20).reduce(0, +) / 20
        XCTAssertLessThan(final, first * 3.5 + 0.001, "Fixed-shape Pure Transformer inference progressively slowed")
        XCTAssertLessThan(abs(Memory.activeMemory - settledMemory), 32 << 20, "Pure Transformer inference memory continued growing after warm-up")
        XCTAssertTrue(last?.asArray(Float.self).allSatisfy(\.isFinite) == true)
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

    func testPackedActionBatchesPreserveHistoryExactly() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("cache-\(UUID().uuidString).atrcache", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let spec = PreprocessingSpec(width: 1, height: 1, colorMode: .grayscale, bitDepth: 8)
        let manifest = DatasetCacheManifest(key: "test", createdAt: Date(), preprocessing: spec, actionFPS: 60, perceptionFPS: 30, historyLength: 2, sampleCount: 3, observationCount: 3, observationBytesPerSample: 1, actionValuesPerSample: ActionLayout.count, segments: [CacheSegment(recordingID: UUID(), start: 0, count: 3)])
        let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(manifest).write(to: directory.appendingPathComponent("manifest.json"))
        try Data([1, 2, 3]).write(to: directory.appendingPathComponent("observations.bin"))
        try observationMappings([(0, 0), (1, 0), (2, 1)]).write(to: directory.appendingPathComponent("observation-indices.bin"))
        var actions = Data()
        for row in 1...3 {
            var values = [Float](repeating: 0, count: ActionLayout.count); values[0] = Float(row)
            values.withUnsafeBytes { actions.append(contentsOf: $0) }
        }
        try actions.write(to: directory.appendingPathComponent("actions.bin"))
        let dataset = try CachedDataset(directory: directory)
        XCTAssertEqual(dataset.packedObservations(at: [2, 0, 1]), Data([3, 1, 2]))
        XCTAssertEqual(dataset.precedingPackedObservations(at: [2, 0, 1]), Data([2, 1, 1]))
        let targets = MLXArray(dataset.actionBatch(at: [2, 0]), [2, ActionLayout.count], type: Float.self).asArray(Float.self)
        XCTAssertEqual(targets[0], 3); XCTAssertEqual(targets[ActionLayout.count], 1)
        let previous = MLXArray(dataset.previousActionBatch(at: [2, 0, 1]), [3, ActionLayout.count], type: Float.self).asArray(Float.self)
        XCTAssertEqual(previous[0], 2)
        XCTAssertEqual(previous[ActionLayout.count], 0, "A recording boundary must start from released controls.")
        XCTAssertEqual(previous[2 * ActionLayout.count], 1)
        let history = MLXArray(dataset.historyBatch(at: [2]), [1, 2, ActionLayout.count], type: Float.self).asArray(Float.self)
        XCTAssertEqual(history[0], 1); XCTAssertEqual(history[ActionLayout.count], 2)

        let order = [2, 0, 1]
        let fused = dataset.trainingBatch(at: order, historyLength: 2, visualMemoryLags: [1])
        XCTAssertEqual(fused.currentObservations, dataset.packedObservations(at: order))
        XCTAssertEqual(fused.visualMemoryObservations, dataset.precedingPackedObservations(at: order))
        XCTAssertEqual(fused.visualMemoryAvailability, [1, 0, 1])
        XCTAssertEqual(fused.targets, dataset.actionBatch(at: order))
        XCTAssertEqual(fused.previousTargets, dataset.previousActionBatch(at: order))
        XCTAssertEqual(fused.history, dataset.historyBatch(at: order, historyLength: 2))

        let multiLag = dataset.trainingBatch(at: order, historyLength: 0, visualMemoryLags: [1, 3])
        XCTAssertEqual(multiLag.visualMemoryObservations, Data([2, 1, 1, 1, 1, 1]))
        XCTAssertEqual(multiLag.visualMemoryAvailability[0], 1, accuracy: 0.000_001)
        XCTAssertEqual(multiLag.visualMemoryAvailability[1], 2.0 / 3.0, accuracy: 0.000_001)
        XCTAssertEqual(multiLag.visualMemoryAvailability[2], 0, accuracy: 0.000_001)
        XCTAssertEqual(multiLag.visualMemoryAvailability[3], 0, accuracy: 0.000_001)
        XCTAssertEqual(multiLag.visualMemoryAvailability[4], 1, accuracy: 0.000_001)
        XCTAssertEqual(multiLag.visualMemoryAvailability[5], 1.0 / 3.0, accuracy: 0.000_001)

        let retunedHistory = MLXArray(
            dataset.trainingBatch(at: [2], historyLength: 1, visualMemoryLags: [1]).history,
            [1, 1, ActionLayout.count],
            type: Float.self
        ).asArray(Float.self)
        XCTAssertEqual(retunedHistory[0], 2, "History must be derived at batch time rather than baked into video-cache identity.")
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
        for offset: UInt64 in [5_000_000, 45_000_000, 85_000_000, 125_000_000] {
            writer.append(InputSample(timestampNanos: base + offset, kind: .key, keyCode: 13, isDown: true))
            writer.append(InputSample(timestampNanos: base + offset + 3_000_000, kind: .key, keyCode: 13, isDown: false))
        }
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
        profile.training.historyLength = 2
        let builder = DatasetCacheBuilder(workspace: store)
        let recordingItem = RecordingItem(manifest: manifest, directory: directory)
        let dataset = try await builder.cache(for: profile, recordings: [recordingItem]) { _, _ in }

        XCTAssertGreaterThan(dataset.count, dataset.manifest.observationCount)
        XCTAssertGreaterThanOrEqual(dataset.manifest.observationCount, 5)
        XCTAssertTrue(dataset.demonstratedKeyCodes().contains(13))
        XCTAssertEqual(dataset.packedObservation(at: 0), dataset.precedingPackedObservations(at: [0]))

        profile.training.historyLength = 31
        profile.training.visualMemoryFrames = VisualMemoryContract.maximumFrameCount
        profile.training.visualMemoryStride = VisualMemoryContract.maximumStride
        profile.training.architecture = .pureLarge
        let retuned = try await builder.cache(for: profile, recordings: [recordingItem]) { _, _ in }
        XCTAssertEqual(
            retuned.manifest.key,
            dataset.manifest.key,
            "Action history, visual memory, and model family must reuse identical sampled pixels."
        )
        XCTAssertEqual(retuned.manifest.createdAt, dataset.manifest.createdAt)
    }

    func testPositiveClassWeightsUseOnlyRequestedRowsAndRespectRestrictions() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("weights-cache-\(UUID().uuidString).atrcache", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let spec = PreprocessingSpec(width: 1, height: 1, colorMode: .grayscale, bitDepth: 8)
        let sampleCount = 12
        let manifest = DatasetCacheManifest(key: "weights", createdAt: Date(), preprocessing: spec, actionFPS: 60, perceptionFPS: 30, historyLength: 1, sampleCount: sampleCount, observationCount: 1, observationBytesPerSample: 1, actionValuesPerSample: ActionLayout.count, segments: [CacheSegment(recordingID: UUID(), start: 0, count: sampleCount)])
        let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(manifest).write(to: directory.appendingPathComponent("manifest.json"))
        try Data([0]).write(to: directory.appendingPathComponent("observations.bin"))
        try observationMappings(Array(repeating: (0, 0), count: sampleCount)).write(to: directory.appendingPathComponent("observation-indices.bin"))
        var actionRows = Data()
        for row in 0..<sampleCount {
            var values = [Float](repeating: 0, count: ActionLayout.count)
            if [0, 2, 4, 6].contains(row) { values[ActionLayout.keyboard.lowerBound + 13] = 1 }
            if row == 8 { values[ActionLayout.keyboard.lowerBound + 12] = 1 }
            values.withUnsafeBytes { actionRows.append(contentsOf: $0) }
        }
        try actionRows.write(to: directory.appendingPathComponent("actions.bin"))
        let dataset = try CachedDataset(directory: directory)
        let indices = Array(0..<sampleCount)
        let plan = dataset.binaryBalancePlan(at: indices, channels: .all, restrictions: ActionRestrictions())
        let weights = plan.positiveWeights
        // Four press and four release boundaries receive the same 3× bonus:
        // positive mass = 4 + 3×4, negative mass = 8 + 3×4.
        let expected = Float(20) / Float(16)
        XCTAssertEqual(weights[ActionLayout.keyboard.lowerBound + 13], expected, accuracy: 0.000_001)
        XCTAssertEqual(weights[ActionLayout.keyboard.lowerBound + 12], 0)
        XCTAssertEqual(plan.supportedKeyCodes, [13])
        XCTAssertEqual(plan.report.ignoredOutputs.map(\.outputIndex), [ActionLayout.keyboard.lowerBound + 12])
        let key13 = try XCTUnwrap(plan.report.outputs.first { $0.outputIndex == ActionLayout.keyboard.lowerBound + 13 })
        XCTAssertEqual(key13.pressEpisodes, 4)
        XCTAssertEqual(key13.releaseEpisodes, 4)
        XCTAssertEqual(key13.activeDurationSeconds ?? 0, 4.0 / 60.0, accuracy: 0.000_001)
        XCTAssertTrue(key13.isSupported)
        XCTAssertEqual(
            key13.positiveWeight * Double(4 + Int(BinaryBalanceContract.transitionBonus) * 4),
            Double(8 + Int(BinaryBalanceContract.transitionBonus) * 4),
            accuracy: 0.000_001,
            "The 0.5 decision boundary must have equal effective pressed and released mass."
        )
        let blocked = dataset.positiveClassWeights(at: indices, restrictions: ActionRestrictions(blockedKeyCodes: [13]))
        XCTAssertEqual(blocked[ActionLayout.keyboard.lowerBound + 13], 0)
    }

    func testContinuousBalanceUsesExactActiveIdleMassAndReportsMissingAxes() throws {
        let rowCount = 10
        var rows = Array(
            repeating: [Float](repeating: 0, count: ActionLayout.count),
            count: rowCount
        )
        let x = ActionLayout.relativeMouse.lowerBound
        rows[1][x] = 0.02
        rows[7][x] = -0.04
        let fixture = try makeSyntheticDataset(
            name: "continuous-balance",
            historyLength: 0,
            mappings: Array(repeating: (UInt32(0), UInt32(0)), count: rowCount),
            actionRows: rows
        )
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        var channels = ActionChannels.all
        channels.scroll = false
        let plan = fixture.dataset.continuousBalancePlan(
            at: Array(0..<rowCount),
            channels: channels
        )
        XCTAssertEqual(plan.outputs.count, 2)
        let xReport = try XCTUnwrap(plan.outputs.first { $0.outputIndex == x })
        XCTAssertEqual(xReport.activeSamples, 2)
        XCTAssertEqual(xReport.meanActiveMagnitude, 0.03, accuracy: 0.000_001)
        XCTAssertEqual(xReport.activeWeight, 4, accuracy: 0.000_001)
        XCTAssertTrue(xReport.isSupported)
        XCTAssertEqual(plan.activeWeights[x], 4, accuracy: 0.000_001)

        let y = ActionLayout.relativeMouse.lowerBound + 1
        let yReport = try XCTUnwrap(plan.outputs.first { $0.outputIndex == y })
        XCTAssertEqual(yReport.activeSamples, 0)
        XCTAssertFalse(yReport.isSupported)
        XCTAssertEqual(plan.activeWeights[y], 0)
    }

    func testContinuousObjectiveMakesSmallExecutableMotionMaterial() {
        var profile = AIProfile.fresh()
        profile.channels = ActionChannels(
            absoluteMouse: true,
            relativeMouse: true,
            buttons: false,
            scroll: false,
            keyboard: false,
            modifiers: false
        )
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        let rows = 10
        let x = ActionLayout.relativeMouse.lowerBound
        var targetValues = [Float](repeating: 0, count: rows * ActionLayout.count)
        var perfectLogits = [Float](repeating: 0, count: rows * ActionLayout.count)
        for row in 0..<rows {
            targetValues[row * ActionLayout.count + ActionLayout.absoluteMouse.lowerBound] = 0.5
            targetValues[row * ActionLayout.count + ActionLayout.absoluteMouse.lowerBound + 1] = 0.5
        }
        for row in [1, 7] {
            targetValues[row * ActionLayout.count + x] = 0.04
            perfectLogits[row * ActionLayout.count + x] = atanh(0.04)
        }
        var weightValues = [Float](repeating: 0, count: ActionLayout.count)
        weightValues[x] = 4
        let history = MLXArray.zeros([rows, 1, ActionLayout.count], dtype: .float32)
        let targets = MLXArray(targetValues, [rows, ActionLayout.count])
        let weights = MLXArray(weightValues, [ActionLayout.count])
        let idle = model.loss(
            logits: MLXArray.zeros([rows, ActionLayout.count], dtype: .float32),
            history: history,
            targets: targets,
            positiveWeights: weights
        )
        let matching = model.loss(
            logits: MLXArray(perfectLogits, [rows, ActionLayout.count]),
            history: history,
            targets: targets,
            positiveWeights: weights
        )
        MLX.eval(idle, matching)
        XCTAssertGreaterThan(idle.item(Float.self), 0.1)
        XCTAssertLessThan(matching.item(Float.self), 0.000_001)
    }

    func testSustainedHoldQualifiesWithoutFourPressEpisodes() throws {
        let rowCount = 40
        let mappings = Array(repeating: (UInt32(0), UInt32(0)), count: rowCount)
        var rows = Array(
            repeating: [Float](repeating: 0, count: ActionLayout.count),
            count: rowCount
        )
        let qualifyingKey = ActionLayout.keyboard.lowerBound + 13
        let shortKey = ActionLayout.keyboard.lowerBound + 12
        for row in 0..<30 { rows[row][qualifyingKey] = 1 }
        for row in 0..<29 { rows[row][shortKey] = 1 }
        let fixture = try makeSyntheticDataset(
            name: "sustained-key-evidence",
            historyLength: 0,
            mappings: mappings,
            actionRows: rows
        )
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let plan = fixture.dataset.binaryBalancePlan(
            at: Array(0..<rowCount),
            channels: .all,
            restrictions: ActionRestrictions()
        )
        XCTAssertEqual(plan.supportedKeyCodes, [13])
        let qualifying = try XCTUnwrap(plan.report.outputs.first { $0.outputIndex == qualifyingKey })
        let tooShort = try XCTUnwrap(plan.report.outputs.first { $0.outputIndex == shortKey })
        XCTAssertEqual(qualifying.pressEpisodes, 1)
        XCTAssertEqual(qualifying.activeDurationSeconds ?? 0, 0.5, accuracy: 0.000_001)
        XCTAssertTrue(qualifying.isSupported)
        XCTAssertLessThan(qualifying.positiveWeight, 1)
        XCTAssertEqual(
            qualifying.positiveWeight * Double(30 + Int(BinaryBalanceContract.transitionBonus)),
            Double(10 + Int(BinaryBalanceContract.transitionBonus)),
            accuracy: 0.000_001,
            "Mostly-held controls need a correction below one to retain the same neutral boundary."
        )
        XCTAssertFalse(tooShort.isSupported)
        XCTAssertEqual(plan.positiveWeights[shortKey], 0)
    }

    func testZeroValidationKeepsEveryTrainingRowAndCanCarryRuntimeCalibration() throws {
        let rowCount = 12
        let fixture = try makeSyntheticDataset(
            name: "zero-validation-calibration",
            historyLength: 0,
            mappings: Array(repeating: (UInt32(0), UInt32(0)), count: rowCount),
            actionRows: Array(
                repeating: [Float](repeating: 0, count: ActionLayout.count),
                count: rowCount
            )
        )
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let split = TrainingEngine().splitIndices(
            dataset: fixture.dataset,
            fraction: 0,
            seed: 42,
            visualMemoryMaximumLag: 0
        )
        XCTAssertEqual(split.train, Array(0..<rowCount))
        XCTAssertTrue(split.validation.isEmpty)

        let calibration = ValidationReport(
            sampleCount: rowCount,
            binary: nil,
            buttons: nil,
            keyboard: nil,
            modifiers: nil,
            absoluteMouseMAE: nil,
            activeRelativeMouseMAE: 0.01,
            activeScrollMAE: nil,
            idleContinuousFalseActionRate: 0.02,
            relativeMouseExecutionDeadzone: 0.5,
            activeRelativeMouseExecutionRecall: 0.25,
            evaluationScope: .trainingCalibration
        )
        XCTAssertEqual(
            GameCameraContract.minimumPostedMagnitude(from: calibration),
            0.5,
            "Validation zero must not force the legacy 1.5-pixel runtime deadzone once the exact brain has been calibrated."
        )
        XCTAssertFalse(calibration.effectiveEvaluationScope.isHeldOut)
    }

    func testValidationSplitNeverRemovesTheOnlyTrainingExampleOfAControl() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("split-cache-\(UUID().uuidString).atrcache", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let spec = PreprocessingSpec(width: 1, height: 1, colorMode: .grayscale, bitDepth: 8)
        let segments = (0..<3).map { CacheSegment(recordingID: UUID(), start: $0 * 2, count: 2) }
        let manifest = DatasetCacheManifest(key: "split", createdAt: Date(), preprocessing: spec, actionFPS: 60, perceptionFPS: 30, historyLength: 1, sampleCount: 6, observationCount: 1, observationBytesPerSample: 1, actionValuesPerSample: ActionLayout.count, segments: segments)
        let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(manifest).write(to: directory.appendingPathComponent("manifest.json"))
        try Data([0]).write(to: directory.appendingPathComponent("observations.bin"))
        try observationMappings(Array(repeating: (0, 0), count: 6)).write(to: directory.appendingPathComponent("observation-indices.bin"))
        var actionRows = Data()
        for row in 0..<6 {
            var values = [Float](repeating: 0, count: ActionLayout.count)
            if row == 0 { values[ActionLayout.keyboard.lowerBound + 10] = 1 }
            if row == 2 || row == 4 { values[ActionLayout.keyboard.lowerBound + 13] = 1 }
            values.withUnsafeBytes { actionRows.append(contentsOf: $0) }
        }
        try actionRows.write(to: directory.appendingPathComponent("actions.bin"))
        let dataset = try CachedDataset(directory: directory)
        let representatives = dataset.representativeValidationIndices(from: Array(0..<6), limit: 2)
        XCTAssertEqual(Set(representatives), [0, 2], "Rare positive controls must displace easy zero-only validation rows")
        let split = TrainingEngine().splitIndices(
            dataset: dataset,
            fraction: 0.9,
            seed: 42,
            visualMemoryMaximumLag: 0
        )
        XCTAssertFalse(split.validation.isEmpty)
        XCTAssertFalse(split.train.isEmpty)
        let trainCounts = dataset.binaryPositiveCounts(at: split.train)
        let validationCounts = dataset.binaryPositiveCounts(at: split.validation)
        XCTAssertEqual(trainCounts[ActionLayout.keyboard.lowerBound + 10], 1)
        XCTAssertGreaterThan(trainCounts[ActionLayout.keyboard.lowerBound + 13], 0)
        XCTAssertGreaterThan(validationCounts[ActionLayout.keyboard.lowerBound + 13], 0)
        XCTAssertTrue(dataset.demonstratedKeyCodes(at: split.train).isEmpty, "A split-preserved singleton remains visible evidence, but it satisfies neither the repeated-press nor sustained-hold capability gate.")
    }

    func testWholeRecordingSplitPreservesReliableKeyEvidence() throws {
        let segmentCount = 6
        let rowsPerSegment = 2
        let rowCount = segmentCount * rowsPerSegment
        let mappings = Array(repeating: (UInt32(0), UInt32(0)), count: rowCount)
        let segments = (0..<segmentCount).map {
            CacheSegment(
                recordingID: UUID(),
                start: $0 * rowsPerSegment,
                count: rowsPerSegment
            )
        }
        var rows = Array(
            repeating: [Float](repeating: 0, count: ActionLayout.count),
            count: rowCount
        )
        let key = ActionLayout.keyboard.lowerBound + 13
        for segment in 0..<4 {
            rows[segment * rowsPerSegment][key] = 1
        }
        let fixture = try makeSyntheticDataset(
            name: "evidence-preserving-split",
            historyLength: 0,
            mappings: mappings,
            actionRows: rows,
            segments: segments
        )
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let split = TrainingEngine().splitIndices(
            dataset: fixture.dataset,
            fraction: 0.34,
            seed: 42,
            visualMemoryMaximumLag: 0
        )
        XCTAssertEqual(Set(split.validation), Set(8..<12))
        XCTAssertEqual(
            fixture.dataset.binaryBalancePlan(
                at: split.train,
                channels: .all,
                restrictions: ActionRestrictions()
            ).supportedKeyCodes,
            [13],
            "Validation must not turn a four-press demonstrated key into an ignored output."
        )
    }

    func testWholeRecordingValidationTargetsSamplesAndCoversEverySegment() throws {
        let counts = [2, 10, 50]
        var start = 0
        let segments = counts.map { count -> CacheSegment in
            defer { start += count }
            return CacheSegment(recordingID: UUID(), start: start, count: count)
        }
        let rowCount = counts.reduce(0, +)
        let mappings = Array(repeating: (UInt32(0), UInt32(0)), count: rowCount)
        let rows = Array(repeating: [Float](repeating: 0, count: ActionLayout.count), count: rowCount)
        let fixture = try makeSyntheticDataset(
            name: "sample-balanced-split",
            historyLength: 1,
            mappings: mappings,
            actionRows: rows,
            segments: segments
        )
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let split = TrainingEngine().splitIndices(
            dataset: fixture.dataset,
            fraction: 0.34,
            seed: 42,
            visualMemoryMaximumLag: 0
        )
        XCTAssertEqual(split.validation, Array(2..<12), "The recording closest to the requested sample fraction should be held out, not a random huge recording.")
        XCTAssertEqual(split.train.count, 52)

        let representatives = fixture.dataset.representativeValidationIndices(from: Array(0..<rowCount), limit: 6)
        for segment in segments {
            XCTAssertTrue(representatives.contains { (segment.start..<(segment.start + segment.count)).contains($0) }, "Every recording should influence a sufficiently large representative score.")
        }
    }

    func testSingleRecordingValidationPurgesSharedHistoryAndPerceptionFrames() throws {
        let mappings: [(UInt32, UInt32)] = [
            (0, 0), (0, 0), (0, 0), (1, 0), (1, 0),
            (1, 0), (1, 0), (1, 0), (2, 1), (2, 1),
            (2, 1), (2, 1), (2, 1), (2, 1), (2, 1),
            (2, 1), (3, 2), (3, 2), (4, 3), (4, 3)
        ]
        let rows = Array(repeating: [Float](repeating: 0, count: ActionLayout.count), count: mappings.count)
        let fixture = try makeSyntheticDataset(name: "purged-validation", historyLength: 3, mappings: mappings, actionRows: rows)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let split = TrainingEngine().splitIndices(
            dataset: fixture.dataset,
            fraction: 0.25,
            seed: 42,
            visualMemoryMaximumLag: 1
        )
        XCTAssertEqual(split.train, Array(0..<15))
        XCTAssertEqual(split.validation, [18, 19])
        XCTAssertTrue(Set(split.train).isDisjoint(with: split.validation))
        XCTAssertGreaterThanOrEqual(split.validation.first ?? 0, (split.train.last ?? 0) + 1 + 3, "Validation history may not reach back into training")
    }

    func testSingleRecordingSplitExtendsPrefixToPreserveKeyEvidence() throws {
        let rowCount = 50
        let mappings = (0..<rowCount).map {
            (UInt32($0), UInt32(max(0, $0 - 1)))
        }
        var rows = Array(
            repeating: [Float](repeating: 0, count: ActionLayout.count),
            count: rowCount
        )
        let key = ActionLayout.keyboard.lowerBound + 13
        for row in [10, 20, 30, 40] { rows[row][key] = 1 }
        let fixture = try makeSyntheticDataset(
            name: "single-recording-evidence",
            historyLength: 0,
            mappings: mappings,
            actionRows: rows
        )
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let split = TrainingEngine().splitIndices(
            dataset: fixture.dataset,
            fraction: 0.5,
            seed: 42,
            historyLength: 0,
            visualMemoryMaximumLag: 0
        )
        XCTAssertEqual(split.train, Array(0..<41))
        XCTAssertEqual(split.validation, Array(41..<50))
        XCTAssertEqual(
            fixture.dataset.demonstratedKeyCodes(at: split.train),
            [13],
            "The held-out tail must not reduce four complete demonstrations to an ignored two-press control."
        )
    }

    func testSingleRecordingValidationEmbargoCoversCompleteVisualMemory() throws {
        let mappings = (0..<30).map { row in
            (UInt32(row), UInt32(max(0, row - 1)))
        }
        let rows = Array(repeating: [Float](repeating: 0, count: ActionLayout.count), count: mappings.count)
        let fixture = try makeSyntheticDataset(
            name: "visual-memory-embargo",
            historyLength: 3,
            mappings: mappings,
            actionRows: rows
        )
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let actionOnly = TrainingEngine().splitIndices(
            dataset: fixture.dataset,
            fraction: 0.25,
            seed: 42,
            historyLength: 3,
            visualMemoryMaximumLag: 1
        )
        let withVisualMemory = TrainingEngine().splitIndices(
            dataset: fixture.dataset,
            fraction: 0.25,
            seed: 42,
            historyLength: 3,
            visualMemoryMaximumLag: 4
        )
        XCTAssertEqual(actionOnly.train, Array(0..<23))
        XCTAssertEqual(actionOnly.validation.first, 26)
        XCTAssertEqual(withVisualMemory.train, Array(0..<23))
        XCTAssertEqual(withVisualMemory.validation, Array(27..<30))
    }

    func testValidationAvailabilityIgnoresBlockedControls() throws {
        let mappings = (0..<20).map { row -> (UInt32, UInt32) in
            (UInt32(row), UInt32(max(0, row - 1)))
        }
        var rows = Array(repeating: [Float](repeating: 0, count: ActionLayout.count), count: mappings.count)
        rows[18][ActionLayout.keyboard.lowerBound + 13] = 1
        let fixture = try makeSyntheticDataset(name: "blocked-validation", historyLength: 3, mappings: mappings, actionRows: rows)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let unrestricted = TrainingEngine().splitIndices(
            dataset: fixture.dataset,
            fraction: 0.25,
            seed: 42,
            visualMemoryMaximumLag: 1
        )
        XCTAssertTrue(unrestricted.validation.isEmpty, "The only learnable key example must remain in training")

        let blocked = TrainingEngine().splitIndices(
            dataset: fixture.dataset,
            fraction: 0.25,
            seed: 42,
            visualMemoryMaximumLag: 1,
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
        let mappings = (0..<12).map { row -> (UInt32, UInt32) in
            (UInt32(row), UInt32(max(0, row - 1)))
        }
        var rows = Array(repeating: [Float](repeating: 0, count: ActionLayout.count), count: mappings.count)
        rows[1][ActionLayout.keyboard.lowerBound + 13] = 1
        rows[7][ActionLayout.scroll.lowerBound] = 0.5
        let fixture = try makeSyntheticDataset(name: "balanced-order", historyLength: 1, mappings: mappings, actionRows: rows)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        var profile = AIProfile.fresh()
        profile.channels = ActionChannels(absoluteMouse: false, relativeMouse: false, buttons: false, scroll: true, keyboard: true, modifiers: false)
        let indices = Array(0..<mappings.count)
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
        let manifest = DatasetCacheManifest(key: "invalid", createdAt: Date(), preprocessing: spec, actionFPS: 60, perceptionFPS: 30, historyLength: 1, sampleCount: Int.max, observationCount: 1, observationBytesPerSample: 1, actionValuesPerSample: ActionLayout.count, segments: [CacheSegment(recordingID: UUID(), start: 0, count: Int.max)])
        let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(manifest).write(to: directory.appendingPathComponent("manifest.json"))
        try Data().write(to: directory.appendingPathComponent("observations.bin"))
        try Data().write(to: directory.appendingPathComponent("actions.bin"))
        XCTAssertThrowsError(try CachedDataset(directory: directory))
    }

    func testCachedDatasetRejectsOutOfRangeObservationMappings() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("mapping-cache-\(UUID().uuidString).atrcache", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let spec = PreprocessingSpec(width: 1, height: 1, colorMode: .grayscale)
        let manifest = DatasetCacheManifest(key: "mapping", createdAt: Date(), preprocessing: spec, actionFPS: 60, perceptionFPS: 30, historyLength: 1, sampleCount: 1, observationCount: 1, observationBytesPerSample: 1, actionValuesPerSample: ActionLayout.count, segments: [CacheSegment(recordingID: UUID(), start: 0, count: 1)])
        let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(manifest).write(to: directory.appendingPathComponent("manifest.json"))
        try Data([0]).write(to: directory.appendingPathComponent("observations.bin"))
        try observationMappings([(1, 0)]).write(to: directory.appendingPathComponent("observation-indices.bin"))
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
        profile.training.historyLength = 2
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        let images = grayscaleTemporalTensor(batch: 2, width: 32, height: 24, value: 0.5)
        let history = MLXArray([Float](repeating: 0, count: 2 * 2 * ActionLayout.count), [2, 2, ActionLayout.count])
        let targets = MLXArray([Float](repeating: 0, count: 2 * ActionLayout.count), [2, ActionLayout.count])
        let gradient = valueAndGrad(model: model) { model, arrays in [model.loss(images: arrays[0], history: arrays[1], targets: arrays[2])] }
        let result = gradient(model, [images, history, targets])
        MLX.eval(result.0, result.1)
        XCTAssertEqual(model.predictions(images: images, history: history).shape, [2, ActionLayout.count])
        XCTAssertTrue(result.0[0].item(Float.self).isFinite)
    }

    func testPureTransformerForwardAndGradient() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 32, height: 24, colorMode: .grayscale, bitDepth: 8)
        profile.training.historyLength = 2
        profile.training.architecture = .pureSmall
        profile.training.architecture.patchSize = 16
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        let images = grayscaleTemporalTensor(batch: 2, width: 32, height: 24, value: 0.5)
        let history = MLXArray.zeros([2, 2, ActionLayout.count], dtype: .float32)
        let targets = MLXArray.zeros([2, ActionLayout.count], dtype: .float32)
        let gradient = valueAndGrad(model: model) { model, arrays in
            [model.loss(images: arrays[0], history: arrays[1], targets: arrays[2])]
        }
        let result = gradient(model, [images, history, targets])
        MLX.eval(result.0, result.1)
        XCTAssertTrue(result.0[0].item(Float.self).isFinite)
        let patchGradient = result.1.flattened().first { $0.0.contains("patchEmbeddings") }?.1
        XCTAssertNotNil(patchGradient)
        if let patchGradient {
            XCTAssertTrue(patchGradient.asArray(Float.self).allSatisfy(\.isFinite))
        }
    }

    func testPolicyPredictionsAreIndependentOfDemonstratedActionHistory() {
        MLXRandom.seed(7_015)
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 12, height: 8, colorMode: .grayscale, bitDepth: 8)
        profile.training.historyLength = 1
        profile.training.visualMemoryDropout = 0
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        let batch = 8
        let images = MLXArray.zeros([
            batch, 8, 12,
            VisualMemoryContract.rawInputChannels(
                colorChannels: profile.preprocessing.channelCount,
                frameCount: profile.training.effectiveVisualMemoryFrames
            )
        ], dtype: .float32)
        var historyValues = [Float](repeating: 0, count: batch * ActionLayout.count)
        for row in 0..<batch { historyValues[row * ActionLayout.count + ActionLayout.keyboard.lowerBound + 13] = 1 }
        let history = MLXArray(historyValues, [batch, 1, ActionLayout.count])

        model.train(false)
        let inference = model.predictions(images: images, history: history)
        let maskedInference = model.predictions(images: images, history: MLXArray.zeros(like: history))
        MLX.eval(inference, maskedInference)
        XCTAssertEqual(inference.asArray(Float.self), maskedInference.asArray(Float.self))

        model.train(true)
        let training = model.predictions(images: images, history: history)
        let zeroHistoryTraining = model.predictions(images: images, history: MLXArray.zeros(like: history))
        MLX.eval(training, zeroHistoryTraining)
        XCTAssertEqual(
            training.asArray(Float.self),
            zeroHistoryTraining.asArray(Float.self),
            "Policy v9 must never expose demonstrated or predicted previous actions to the model."
        )
        XCTAssertFalse(model.parameters().flattened().contains { $0.0.contains("history") })
    }

    func testFocalLossUsesFixedDecisionNormalization() {
        var profile = AIProfile.fresh()
        profile.channels = ActionChannels(
            absoluteMouse: false,
            relativeMouse: false,
            buttons: true,
            scroll: false,
            keyboard: false,
            modifiers: false
        )
        profile.training.binaryFocalGamma = 1
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        let logits = MLXArray.zeros([1, ActionLayout.count], dtype: .float32)
        let history = MLXArray.zeros([1, 1, ActionLayout.count], dtype: .float32)
        var targetValues = [Float](repeating: 0, count: ActionLayout.count)
        targetValues[ActionLayout.buttons.lowerBound] = 1
        var weightValues = [Float](repeating: 0, count: ActionLayout.count)
        weightValues[ActionLayout.buttons.lowerBound] = 1
        let targets = MLXArray(targetValues, [1, ActionLayout.count])
        let loss = model.loss(
            logits: logits,
            history: history,
            targets: targets,
            positiveWeights: MLXArray(weightValues, [ActionLayout.count]),
            previousTargets: targets
        )
        MLX.eval(loss)
        XCTAssertEqual(loss.item(Float.self), Float(Foundation.log(2.0) / 2), accuracy: 0.000_01)
    }

    func testPositiveClassCorrectionDoesNotCancelInsideItsBatch() {
        var profile = AIProfile.fresh()
        profile.channels = ActionChannels(
            absoluteMouse: false,
            relativeMouse: false,
            buttons: false,
            scroll: false,
            keyboard: true,
            modifiers: false
        )
        profile.training.binaryFocalGamma = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        let key = ActionLayout.keyboard.lowerBound + 13
        let logits = MLXArray.zeros([1, ActionLayout.count], dtype: .float32)
        let history = MLXArray.zeros([1, 1, ActionLayout.count], dtype: .float32)
        var targets = [Float](repeating: 0, count: ActionLayout.count)
        targets[key] = 1
        var weights = [Float](repeating: 0, count: ActionLayout.count)
        weights[key] = 4
        let target = MLXArray(targets, [1, ActionLayout.count])
        let loss = model.loss(
            logits: logits,
            history: history,
            targets: target,
            positiveWeights: MLXArray(weights, [ActionLayout.count]),
            previousTargets: target
        )
        MLX.eval(loss)
        XCTAssertEqual(
            loss.item(Float.self),
            4 * Float(Foundation.log(2.0)),
            accuracy: 0.000_01,
            "Normalizing by the weighted batch mass would reduce this back to log(2) and erase the correction."
        )
    }

    func testSparseClassCorrectionRemainsFiniteWithFloat16Policy() {
        var profile = AIProfile.fresh()
        profile.channels = ActionChannels(
            absoluteMouse: false,
            relativeMouse: false,
            buttons: false,
            scroll: false,
            keyboard: true,
            modifiers: false
        )
        profile.training.binaryFocalGamma = 0
        profile.training.precision = .float16
        let model = AgentPolicy(profile: profile)
        let key = ActionLayout.keyboard.lowerBound + 13
        var targets = [Float](repeating: 0, count: ActionLayout.count)
        targets[key] = 1
        var weights = [Float](repeating: 0, count: ActionLayout.count)
        weights[key] = 100_000
        let target = MLXArray(targets, [1, ActionLayout.count])
        let loss = model.loss(
            logits: MLXArray.zeros([1, ActionLayout.count], dtype: .float16),
            history: MLXArray.zeros([1, 1, ActionLayout.count], dtype: .float16),
            targets: target,
            positiveWeights: MLXArray(weights, [ActionLayout.count]),
            previousTargets: target
        )
        MLX.eval(loss)
        let value = loss.item(Float.self)
        XCTAssertTrue(value.isFinite)
        XCTAssertGreaterThan(value, Float(Float16.greatestFiniteMagnitude))
    }

    func testBalancedSparseControlDoesNotPreferAnIdleConstantPrediction() {
        var profile = AIProfile.fresh()
        profile.channels = ActionChannels(
            absoluteMouse: false,
            relativeMouse: false,
            buttons: false,
            scroll: false,
            keyboard: true,
            modifiers: false
        )
        profile.training.binaryFocalGamma = 1.5
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        let rows = 16
        let key = ActionLayout.keyboard.lowerBound + 13
        var targetValues = [Float](repeating: 0, count: rows * ActionLayout.count)
        for row in [0, 4, 8, 12] {
            targetValues[row * ActionLayout.count + key] = 1
        }
        var previousValues = [Float](repeating: 0, count: rows * ActionLayout.count)
        for row in 1..<rows {
            previousValues[row * ActionLayout.count + key]
                = targetValues[(row - 1) * ActionLayout.count + key]
        }
        var weights = [Float](repeating: 0, count: ActionLayout.count)
        // Positive mass: 4 active + 3×4 presses = 16.
        // Negative mass: 12 idle + 3×4 releases = 24.
        weights[key] = 24.0 / 16.0
        let history = MLXArray.zeros([rows, 1, ActionLayout.count], dtype: .float32)
        let targets = MLXArray(targetValues, [rows, ActionLayout.count])
        let previous = MLXArray(previousValues, [rows, ActionLayout.count])
        let classWeights = MLXArray(weights, [ActionLayout.count])

        func loss(constantProbability: Float) -> Float {
            let logit = Foundation.log(constantProbability / (1 - constantProbability))
            var values = [Float](repeating: 0, count: rows * ActionLayout.count)
            for row in 0..<rows { values[row * ActionLayout.count + key] = logit }
            let result = model.loss(
                logits: MLXArray(values, [rows, ActionLayout.count]),
                history: history,
                targets: targets,
                positiveWeights: classWeights,
                previousTargets: previous
            )
            MLX.eval(result)
            return result.item(Float.self)
        }

        let idleBiased = loss(constantProbability: 0.4)
        let neutral = loss(constantProbability: 0.5)
        let activeBiased = loss(constantProbability: 0.6)
        XCTAssertLessThan(neutral, idleBiased)
        XCTAssertLessThan(neutral, activeBiased)
        XCTAssertEqual(idleBiased, activeBiased, accuracy: 0.000_01)
    }

    func testPerHeadLossesAverageExactlyToTheOptimizationObjective() throws {
        var profile = AIProfile.fresh()
        profile.training.precision = .float32
        profile.training.binaryFocalGamma = 1
        let model = AgentPolicy(profile: profile)
        let logits = MLXArray.zeros([2, ActionLayout.count], dtype: .float32)
        let history = MLXArray.zeros([2, max(1, profile.training.historyLength), ActionLayout.count], dtype: .float32)
        let targets = MLXArray.zeros([2, ActionLayout.count], dtype: .float32)
        let previousTargets = MLXArray.zeros(like: targets)
        let weights = MLXArray.ones([ActionLayout.count], dtype: .float32)
        let components = model.lossComponents(
            logits: logits,
            history: history,
            targets: targets,
            positiveWeights: weights,
            previousTargets: previousTargets
        )
        let objective = model.loss(
            logits: logits,
            history: history,
            targets: targets,
            positiveWeights: weights,
            previousTargets: previousTargets
        )
        let heads = [components.mouse, components.buttons, components.scroll, components.keyboard, components.modifiers]
        let concreteHeads = try heads.map { try XCTUnwrap($0) }
        MLX.eval([components.total, objective] + concreteHeads)
        let expected = concreteHeads.map { $0.item(Float.self) }.reduce(0, +) / Float(concreteHeads.count)
        XCTAssertEqual(components.total.item(Float.self), expected, accuracy: 0.000_001)
        XCTAssertEqual(objective.item(Float.self), expected, accuracy: 0.000_001)
    }

    func testZeroHistoryTransitionLossUsesTheRealPreviousAction() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 8, height: 8, colorMode: .grayscale)
        profile.channels = ActionChannels(absoluteMouse: false, relativeMouse: false, buttons: true, scroll: false, keyboard: false, modifiers: false)
        profile.training.historyLength = 0
        profile.training.binaryFocalGamma = 0
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        model.train(false)
        let images = MLXArray.zeros([
            1, 8, 8,
            VisualMemoryContract.rawInputChannels(
                colorChannels: profile.preprocessing.channelCount,
                frameCount: profile.training.effectiveVisualMemoryFrames
            )
        ], dtype: .float32)
        let history = MLXArray.zeros([1, 1, ActionLayout.count], dtype: .float32)
        var targetValues = [Float](repeating: 0, count: ActionLayout.count)
        targetValues[ActionLayout.buttons.lowerBound] = 1
        let targets = MLXArray(targetValues, [1, ActionLayout.count])
        let placeholderLoss = model.loss(images: images, history: history, targets: targets)
        let heldActionLoss = model.loss(images: images, history: history, targets: targets, previousTargets: targets)
        MLX.eval(placeholderLoss, heldActionLoss)
        XCTAssertNotEqual(
            placeholderLoss.item(Float.self),
            heldActionLoss.item(Float.self),
            accuracy: 0.000_001,
            "A held action and a fresh transition need different weighting even when model history is disabled."
        )
    }

    func testPolicyLearnsAVisualControlSignalInsteadOfAnInertShortcut() {
        MLXRandom.seed(202_607_15)
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 8, height: 8, colorMode: .grayscale)
        profile.channels = ActionChannels(absoluteMouse: false, relativeMouse: false, buttons: false, scroll: false, keyboard: true, modifiers: false)
        profile.training.historyLength = 0
        profile.training.visualMemoryDropout = 0
        profile.training.precision = .float32
        profile.training.architecture = ArchitectureSpec(
            convolutionChannels: [8, 12, 16, 24], kernelSizes: [7, 3, 3, 3], strides: [4, 2, 2, 2],
            visualEmbedding: 32, recurrentKind: .gru, recurrentWidth: 16, fusionWidths: [32], dropout: 0,
            spatialTokens: 4, transformerLayers: 1, transformerHeads: 4, transformerFeedForward: 64
        )
        let model = AgentPolicy(profile: profile)
        let optimizer = ResumableAdamW(learningRate: 0.003, weightDecay: 0)
        optimizer.initialize(model: model)
        let pixels = 8 * 8
        let visualChannels = VisualMemoryContract.rawInputChannels(
            colorChannels: 1,
            frameCount: profile.training.effectiveVisualMemoryFrames
        )
        var imageValues = [Float](repeating: 0, count: 2 * pixels * visualChannels)
        for pixel in 0..<pixels {
            imageValues[(pixels + pixel) * visualChannels] = 1
        }
        let images = MLXArray(imageValues, [2, 8, 8, visualChannels])
        let history = MLXArray.zeros([2, 1, ActionLayout.count])
        var targetValues = [Float](repeating: 0, count: 2 * ActionLayout.count)
        let key = ActionLayout.keyboard.lowerBound + 13
        targetValues[ActionLayout.count + key] = 1
        let targets = MLXArray(targetValues, [2, ActionLayout.count])
        var mutableClassWeights = [Float](repeating: 0, count: ActionLayout.count)
        mutableClassWeights[key] = 1
        let classWeightValues = mutableClassWeights
        let weights = MLXArray(classWeightValues, [ActionLayout.count])
        let initial = model.loss(images: images, history: history, targets: targets, positiveWeights: weights)
        MLX.eval(initial)

        let step = compile(inputs: [model, optimizer], outputs: [model, optimizer]) { images, history, targets in
            let tracedWeights = MLXArray(classWeightValues, [ActionLayout.count])
            let result = valueAndGrad(model: model) { model, arrays in
                [model.loss(images: arrays[0], history: arrays[1], targets: arrays[2], positiveWeights: tracedWeights)]
            }(model, [images, history, targets])
            optimizer.update(model: model, gradients: clipGradNorm(gradients: result.1, maxNorm: 1).0, targetType: model.dtype)
            return result.0[0]
        }
        var final = initial.item(Float.self)
        for _ in 0..<600 {
            let loss = step(images, history, targets)
            MLX.eval(loss, model.parameters(), optimizer.stateArrays())
            final = loss.item(Float.self)
        }
        model.train(false)
        let predictions = model.predictions(images: images, history: history)
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
        profile.training.historyLength = 1
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        let images = grayscaleTemporalTensor(batch: 1, width: 12, height: 8, value: 0.5)
        let history = MLXArray([Float](repeating: 0, count: ActionLayout.count), [1, 1, ActionLayout.count])
        var targetValues = [Float](repeating: 0, count: ActionLayout.count)
        targetValues[ActionLayout.keyboard.lowerBound + 13] = 1
        let targets = MLXArray(targetValues, [1, ActionLayout.count])
        let blockedWeights = MLXArray([Float](repeating: 0, count: ActionLayout.count), [ActionLayout.count])
        let blockedLoss = model.loss(images: images, history: history, targets: targets, positiveWeights: blockedWeights)
        var learnedValues = [Float](repeating: 0, count: ActionLayout.count)
        learnedValues[ActionLayout.keyboard.lowerBound + 13] = 4
        let learnedLoss = model.loss(images: images, history: history, targets: targets, positiveWeights: MLXArray(learnedValues, [ActionLayout.count]))
        MLX.eval(blockedLoss, learnedLoss)
        XCTAssertEqual(blockedLoss.item(Float.self), 0, accuracy: 0.000_001)
        XCTAssertGreaterThan(learnedLoss.item(Float.self), 0)
    }

    func testShiftLossBelongsToKeyboardAndNotModifierChannel() {
        func loss(keyboard: Bool, modifiers: Bool, targetIndex: Int) -> Float {
            var profile = AIProfile.fresh()
            profile.preprocessing = PreprocessingSpec(width: 12, height: 8, colorMode: .grayscale, bitDepth: 8)
            profile.channels = ActionChannels(absoluteMouse: false, relativeMouse: false, buttons: false, scroll: false, keyboard: keyboard, modifiers: modifiers)
            profile.training.historyLength = 1
            profile.training.architecture = .small
            profile.training.architecture.dropout = 0
            profile.training.precision = .float32
            let model = AgentPolicy(profile: profile)
            let images = grayscaleTemporalTensor(batch: 1, width: 12, height: 8, value: 0.5)
            let history = MLXArray([Float](repeating: 0, count: ActionLayout.count), [1, 1, ActionLayout.count])
            var targetValues = [Float](repeating: 0, count: ActionLayout.count)
            targetValues[targetIndex] = 1
            var weightValues = [Float](repeating: 0, count: ActionLayout.count)
            weightValues[targetIndex] = 4
            let result = model.loss(
                images: images,
                history: history,
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
        profile.training.historyLength = 2
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        model.train(false)
        let images = grayscaleTemporalTensor(batch: 1, width: 32, height: 24, value: 0.5)
        let history = MLXArray([Float](repeating: 0, count: 2 * ActionLayout.count), [1, 2, ActionLayout.count])

        let layers = model.visualActivations(images: images)
        XCTAssertEqual(layers.map(\.shape), [[1, 6, 8, 24], [1, 3, 4, 48], [1, 2, 2, 72], [1, 1, 1, 96]])

        let standard = compile(inputs: [model]) { images, history in model.predictions(images: images, history: history) }
        let activities = layers.indices.map { selectedLayer in
            compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
                let visual = model.visualActivations(images: inputs[0])
                let logits = model.logits(visualFeatures: visual.last!, history: inputs[1])
                let map = model.sampledForVisualization(visual[selectedLayer]).mean(axis: -1, keepDims: true)
                return [model.activatedPredictions(logits: logits), map]
            }
        }
        let channels = compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
            let visual = model.visualActivations(images: inputs[0])
            let logits = model.logits(visualFeatures: visual.last!, history: inputs[1])
            return [model.activatedPredictions(logits: logits), model.strongestChannelsForVisualization(visual.last!)]
        }
        let spatialTokens = compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
            let visual = model.visualActivations(images: inputs[0])
            let logits = model.logits(visualFeatures: visual.last!, history: inputs[1])
            return [model.activatedPredictions(logits: logits), model.spatialTokenAttentionMaps(visualFeatures: visual.last!)]
        }
        let saliency = compile(inputs: [model]) { (inputs: [MLXArray]) -> [MLXArray] in
            let visual = model.visualActivations(images: inputs[0])
            let logits = model.logits(visualFeatures: visual.last!, history: inputs[1])
            return [model.activatedPredictions(logits: logits), visual.last!]
        }
        let saliencyGradient = grad({ (inputs: [MLXArray]) -> MLXArray in
            let logits = model.logits(visualFeatures: inputs[0], history: inputs[1])
            return (logits * inputs[2]).sum()
        }, argumentNumbers: [0])
        var selectorValues = [Float](repeating: 0, count: ActionLayout.count)
        selectorValues[ActionLayout.keyboard.lowerBound] = 1
        let selector = MLXArray(selectorValues, [1, ActionLayout.count])
        let expected = standard(images, history)
        let activityResults = activities.map { $0([images, history]) }
        let channelResult = channels([images, history])
        let spatialTokenResult = spatialTokens([images, history])
        let saliencyForward = saliency([images, history])
        let gradients = saliencyGradient([saliencyForward[1], history, selector])
        let weights = gradients.mean(axes: [1, 2], keepDims: true)
        let saliencyMap = model.sampledForVisualization(relu((saliencyForward[1] * weights).sum(axis: -1, keepDims: true)))
        let saliencyResult = [saliencyForward[0], saliencyMap]
        MLX.eval(expected, activityResults.flatMap { $0 }, channelResult, spatialTokenResult, saliencyForward, saliencyResult)

        XCTAssertEqual(activityResults.map { $0[1].shape }, [[1, 6, 8, 1], [1, 3, 4, 1], [1, 2, 2, 1], [1, 1, 1, 1]])
        XCTAssertEqual(channelResult[1].shape, [1, 1, 1, 16])
        XCTAssertEqual(spatialTokenResult[1].shape, [1, 1, 1, ArchitectureSpec.small.effectiveSpatialTokens])
        XCTAssertEqual(saliencyForward[1].shape, [1, 1, 1, 96])
        XCTAssertEqual(saliencyResult[1].shape, [1, 1, 1, 1])
        let expectedValues = expected.asArray(Float.self)
        for prediction in activityResults.map({ $0[0] }) + [channelResult[0], spatialTokenResult[0], saliencyResult[0]] {
            XCTAssertTrue(zip(expectedValues, prediction.asArray(Float.self)).allSatisfy { abs($0 - $1) < 1e-5 })
        }
        XCTAssertTrue(spatialTokenResult[1].asArray(Float.self).allSatisfy { abs($0 - 1) < 1e-6 })
        XCTAssertTrue(saliencyResult[1].asArray(Float.self).allSatisfy { $0.isFinite && $0 >= 0 })
    }

    func testOptimizerCheckpointResumesExactly() throws {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 16, height: 12, colorMode: .grayscale, bitDepth: 8)
        profile.training.historyLength = 1
        profile.training.visualMemoryDropout = 0
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let modelA = AgentPolicy(profile: profile)
        let optimizerA = ResumableAdamW(learningRate: 0.001, weightDecay: 0.01)
        let images = grayscaleTemporalTensor(batch: 1, width: 16, height: 12, value: 0.25)
        let history = MLXArray([Float](repeating: 0, count: ActionLayout.count), [1, 1, ActionLayout.count])
        let targets = MLXArray([Float](repeating: 0, count: ActionLayout.count), [1, ActionLayout.count])
        let gradientA = valueAndGrad(model: modelA) { model, arrays in [model.loss(images: arrays[0], history: arrays[1], targets: arrays[2])] }
        let first = gradientA(modelA, [images, history, targets])
        optimizerA.update(model: modelA, gradients: first.1, targetType: .float32)
        MLX.eval(modelA.parameters(), optimizerA.stateArrays())
        let checkpointParameters = Dictionary(
            uniqueKeysWithValues: modelA.parameters().flattened().map {
                ($0.0, $0.1.asArray(Float.self))
            }
        )
        let checkpointOptimizerState = optimizerA.stateArrays().map {
            $0.asArray(Float.self)
        }

        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("resume-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let weights = directory.appendingPathComponent("weights.safetensors")
        let optimizer = directory.appendingPathComponent("optimizer.safetensors")
        try modelA.saveWeights(to: weights); try optimizerA.save(to: optimizer)

        let secondA = gradientA(modelA, [images, history, targets])
        MLX.eval(secondA.1)
        let secondAGradients = Dictionary(
            uniqueKeysWithValues: secondA.1.flattened().map {
                ($0.0, $0.1.asArray(Float.self))
            }
        )
        optimizerA.update(model: modelA, gradients: secondA.1, targetType: .float32)
        MLX.eval(modelA.parameters(), optimizerA.stateArrays())

        let modelB = AgentPolicy(profile: profile)
        let optimizerB = ResumableAdamW(learningRate: 0.001, weightDecay: 0.01)
        try modelB.loadWeights(from: weights); try optimizerB.load(from: optimizer)
        MLX.eval(modelB.parameters(), optimizerB.stateArrays())
        for (key, parameter) in modelB.parameters().flattened() {
            XCTAssertEqual(
                parameter.asArray(Float.self),
                checkpointParameters[key],
                "Checkpoint load changed \(key)"
            )
        }
        XCTAssertEqual(optimizerB.stateArrays().count, checkpointOptimizerState.count)
        for (loaded, expected) in zip(
            optimizerB.stateArrays(),
            checkpointOptimizerState
        ) {
            XCTAssertEqual(
                loaded.asArray(Float.self),
                expected,
                "Checkpoint load changed optimizer state"
            )
        }
        let gradientB = valueAndGrad(model: modelB) { model, arrays in [model.loss(images: arrays[0], history: arrays[1], targets: arrays[2])] }
        let secondB = gradientB(modelB, [images, history, targets])
        MLX.eval(secondB.1)
        for (key, gradient) in secondB.1.flattened() {
            let loadedGradient = gradient.asArray(Float.self)
            let expectedGradient = try XCTUnwrap(secondAGradients[key])
            XCTAssertTrue(
                zip(loadedGradient, expectedGradient).allSatisfy {
                    abs($0 - $1) < 1e-6
                },
                "Checkpoint resume changed gradient materially at \(key)"
            )
        }
        optimizerB.update(model: modelB, gradients: secondB.1, targetType: .float32)
        MLX.eval(modelB.parameters(), optimizerB.stateArrays())

        let paramsA = Dictionary(uniqueKeysWithValues: modelA.parameters().flattened())
        let paramsB = Dictionary(uniqueKeysWithValues: modelB.parameters().flattened())
        XCTAssertEqual(paramsA.keys.sorted(), paramsB.keys.sorted())
        for key in paramsA.keys {
            let a = try XCTUnwrap(paramsA[key]).asArray(Float.self)
            let b = try XCTUnwrap(paramsB[key]).asArray(Float.self)
            XCTAssertEqual(a.count, b.count)
            XCTAssertTrue(zip(a, b).allSatisfy { abs($0 - $1) < 1e-6 }, "Checkpoint diverged at \(key)")
        }
        XCTAssertEqual(optimizerA.step, optimizerB.step)
    }

    func testPolicyGradientIsDeterministicAtIdenticalWeights() throws {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(
            width: 16,
            height: 12,
            colorMode: .grayscale,
            bitDepth: 8
        )
        profile.training.visualMemoryDropout = 0
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        // MLX parameter initialization is lazy. Materialize the random
        // initializer once so both gradient graphs truly use identical
        // weights rather than re-evaluating the initializer expression.
        MLX.eval(model.parameters())
        let images = grayscaleTemporalTensor(
            batch: 1,
            width: 16,
            height: 12,
            value: 0.25
        )
        let history = MLXArray(
            [Float](repeating: 0, count: ActionLayout.count),
            [1, 1, ActionLayout.count]
        )
        let targets = MLXArray(
            [Float](repeating: 0, count: ActionLayout.count),
            [1, ActionLayout.count]
        )
        let gradient = valueAndGrad(model: model) { model, arrays in
            [model.loss(images: arrays[0], history: arrays[1], targets: arrays[2])]
        }
        let first = gradient(model, [images, history, targets]).1
        MLX.eval(first)
        let second = gradient(model, [images, history, targets]).1
        MLX.eval(second)
        let firstByName = Dictionary(uniqueKeysWithValues: first.flattened())
        let secondByName = Dictionary(uniqueKeysWithValues: second.flattened())
        XCTAssertEqual(firstByName.keys.sorted(), secondByName.keys.sorted())
        for key in firstByName.keys {
            let firstValues = try XCTUnwrap(firstByName[key]).asArray(Float.self)
            let secondValues = try XCTUnwrap(secondByName[key]).asArray(Float.self)
            let differences = zip(firstValues, secondValues).map { abs($0 - $1) }
            let nonFiniteCount = zip(firstValues, secondValues).filter {
                !$0.0.isFinite || !$0.1.isFinite
            }.count
            XCTAssertTrue(
                differences.allSatisfy { $0 < 1e-6 },
                "Repeated gradient diverged at \(key) (max |Δ| \(differences.max() ?? 0), non-finite values \(nonFiniteCount))"
            )
        }
    }

    func testCompatibleWeightWarmStartHonorsTheNewPrecision() throws {
        var sourceProfile = AIProfile.fresh()
        sourceProfile.preprocessing = PreprocessingSpec(width: 12, height: 8, colorMode: .grayscale)
        sourceProfile.training.architecture = .small
        sourceProfile.training.precision = .float32
        let source = AgentPolicy(profile: sourceProfile)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("precision-warm-start-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try source.saveWeights(to: url)

        var destinationProfile = sourceProfile
        destinationProfile.training.precision = .bfloat16
        let destination = AgentPolicy(profile: destinationProfile)
        try destination.loadWeights(from: url)
        XCTAssertTrue(destination.parameters().flattened().allSatisfy { $0.1.dtype == .bfloat16 })
        destination.train(false)
        let predictions = destination.predictions(
            images: MLXArray.zeros([
                1, 8, 12,
                VisualMemoryContract.rawInputChannels(
                    colorChannels: destinationProfile.preprocessing.channelCount,
                    frameCount: destinationProfile.training.effectiveVisualMemoryFrames
                )
            ]),
            history: MLXArray.zeros([1, destinationProfile.training.historyLength, ActionLayout.count])
        )
        MLX.eval(predictions)
        XCTAssertTrue(predictions.asArray(Float.self).allSatisfy(\.isFinite))
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

    private func grayscaleTemporalTensor(batch: Int, width: Int, height: Int, value: Float) -> MLXArray {
        let frames = VisualMemoryContract.defaultFrameCount
        let channels = VisualMemoryContract.rawInputChannels(colorChannels: 1, frameCount: frames)
        var values = [Float](repeating: 0, count: batch * width * height * channels)
        for pixel in 0..<(batch * width * height) {
            values[pixel * channels] = value
            for slot in 0..<frames {
                values[pixel * channels + 1 + frames + slot] = 1
            }
        }
        return MLXArray(values, [batch, height, width, channels])
    }

    private func makeSyntheticDataset(
        name: String,
        historyLength: Int,
        mappings: [(UInt32, UInt32)],
        actionRows: [[Float]],
        segments: [CacheSegment]? = nil,
        actionFPS: Double = 60
    ) throws -> (dataset: CachedDataset, directory: URL) {
        precondition(mappings.count == actionRows.count)
        precondition(actionRows.allSatisfy { $0.count == ActionLayout.count })
        let segments = segments ?? [CacheSegment(recordingID: UUID(), start: 0, count: mappings.count)]
        precondition(segments.reduce(0) { $0 + $1.count } == mappings.count)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString).atrcache", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            let observationCount = Int(mappings.flatMap { [$0.0, $0.1] }.max() ?? 0) + 1
            let spec = PreprocessingSpec(width: 1, height: 1, colorMode: .grayscale, bitDepth: 8)
            let manifest = DatasetCacheManifest(
                key: name,
                createdAt: Date(),
                preprocessing: spec,
                actionFPS: actionFPS,
                perceptionFPS: 30,
                historyLength: historyLength,
                sampleCount: mappings.count,
                observationCount: observationCount,
                observationBytesPerSample: 1,
                actionValuesPerSample: ActionLayout.count,
                segments: segments
            )
            let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(manifest).write(to: directory.appendingPathComponent("manifest.json"))
            try Data((0..<observationCount).map { UInt8(clamping: $0) }).write(to: directory.appendingPathComponent("observations.bin"))
            try observationMappings(mappings).write(to: directory.appendingPathComponent("observation-indices.bin"))
            var actions = Data(capacity: actionRows.count * ActionLayout.count * MemoryLayout<Float>.size)
            for row in actionRows { row.withUnsafeBytes { actions.append(contentsOf: $0) } }
            try actions.write(to: directory.appendingPathComponent("actions.bin"))
            return (try CachedDataset(directory: directory), directory)
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    private func observationMappings(_ pairs: [(UInt32, UInt32)]) -> Data {
        var data = Data(capacity: pairs.count * 2 * MemoryLayout<UInt32>.size)
        for pair in pairs {
            var current = pair.0.littleEndian
            var previous = pair.1.littleEndian
            withUnsafeBytes(of: &current) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &previous) { data.append(contentsOf: $0) }
        }
        return data
    }
}
