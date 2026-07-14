import XCTest
import CoreVideo
import MLX
import MLXNN
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

    func testNeuralInputSizingMirrorsPackedDenseCoordinateAndHistoryContracts() {
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
        XCTAssertEqual(input.expandedVisionValues, 45)
        XCTAssertEqual(input.coordinateValues, 30)
        XCTAssertEqual(input.firstConvolutionValues, 75)
        XCTAssertEqual(input.historySteps, 4)
        XCTAssertEqual(input.historyValues, 4 * 146)
        XCTAssertEqual(input.historyDurationSeconds, 0.2, accuracy: 0.000_001)
        XCTAssertEqual(input.valuesPerDecision, 75 + 4 * 146)
        XCTAssertEqual(input.runtimeValuesPerSecond, 10 * (75 + 4 * 146))
        XCTAssertEqual(input.packedVisionBytesPerSecond, 270)
        XCTAssertEqual(input.valuesPerTrainingBatch, 2 * (75 + 4 * 146))
        XCTAssertEqual(input.quantizationLevels, 4)
        XCTAssertEqual(input.effectivePackedBits, 27 * 2)
        XCTAssertEqual(input.nominalBytesPerTrainingBatch, 2 * 2 * (75 + 4 * 146))
    }

    func testNeuralInputSizingShowsTheZeroHistoryTensorAndIgnoresChromaForGrayscale() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 5, height: 3, colorMode: .grayscale, bitDepth: 8, chroma: .yuv420, resizePolicy: .stretch)
        profile.training.historyLength = 0
        profile.training.batchSize = 1
        profile.training.precision = .float32

        let input = NeuralInputSizing.summary(for: profile)
        XCTAssertEqual(input.packedVisionValues, 15)
        XCTAssertEqual(input.chromaValuesPerPlane, 0)
        XCTAssertEqual(input.expandedVisionValues, 15)
        XCTAssertEqual(input.firstConvolutionValues, 45)
        XCTAssertEqual(input.historySteps, 1)
        XCTAssertEqual(input.historyValues, 146)
        XCTAssertEqual(input.valuesPerDecision, 191)
        XCTAssertEqual(input.nominalBytesPerDecision, 191 * 4)
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

    func testArchitecturePresetsHaveNoZeroWidths() {
        for architecture in [ArchitectureSpec.small, .balanced, .large] {
            XCTAssertTrue(architecture.convolutionChannels.allSatisfy { $0 > 0 })
            XCTAssertTrue(architecture.fusionWidths.allSatisfy { $0 > 0 })
        }
        XCTAssertEqual(CNNGeometry.layer(0, architecture: .balanced), CNNLayerGeometry(kernelSize: 5, effectiveStride: 2, receptiveField: 5))
        XCTAssertEqual(CNNGeometry.layer(1, architecture: .balanced), CNNLayerGeometry(kernelSize: 3, effectiveStride: 4, receptiveField: 9))
        XCTAssertEqual(CNNGeometry.layer(2, architecture: .balanced), CNNLayerGeometry(kernelSize: 3, effectiveStride: 8, receptiveField: 17))
        var convolutionFree = ArchitectureSpec.balanced
        convolutionFree.convolutionChannels = []
        XCTAssertEqual(CNNGeometry.layer(-1, architecture: convolutionFree), CNNLayerGeometry(kernelSize: 1, effectiveStride: 1, receptiveField: 1))
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
        XCTAssertEqual(writer.finish(), 1)
        XCTAssertEqual(try InputEventReader.read(url: url), [expected])
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
        try await store.saveProfile(profile)
        let profileDirectory = await store.profileDirectory(profile.id)
        let versions = profileDirectory.appendingPathComponent("Versions", isDirectory: true)
        let checkpoint = await store.checkpointDirectory(profileID: profile.id)
        try FileManager.default.createDirectory(at: versions, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: checkpoint, withIntermediateDirectories: true)
        try Data([1]).write(to: versions.appendingPathComponent("old.bin"))
        try Data([2]).write(to: checkpoint.appendingPathComponent("old.bin"))

        let recordingID = UUID()
        let recordingDirectory = try await store.createRecordingDirectory(id: recordingID)
        let recording = RecordingManifest(id: recordingID, name: "Keep me", createdAt: Date(), hostStartNanos: 1, duration: 1, capture: CaptureSpec(), globalRect: CodableRect(.zero), pixelWidth: 8, pixelHeight: 8, deliveredFPS: 30, eventCount: 0)
        try await store.writeRecording(recording, to: recordingDirectory)

        let removed = try await store.removeObsoleteModelArtifacts(currentSchema: ModelContract.schemaVersion)
        let remainingRecordings = await store.listRecordings()
        let remainingProfiles = await store.listProfiles()
        XCTAssertEqual(removed, 2)
        XCTAssertFalse(FileManager.default.fileExists(atPath: versions.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: checkpoint.path))
        XCTAssertEqual(remainingRecordings.count, 1)
        XCTAssertEqual(remainingProfiles.first?.name, "Preserved profile")
        XCTAssertNil(remainingProfiles.first?.activeVersionID)
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

    func testOnlyExactVisionAndArchitectureChangeTheLearnedBrainContract() {
        var profile = AIProfile.fresh()
        let original = profile.learnedBrainContract
        profile.training.epochs = 9_999
        profile.training.learningRate = 0.00001
        profile.training.historyLength = 32
        XCTAssertEqual(profile.learnedBrainContract, original)
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
        _ = writer.finish()
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
        _ = writer.finish()
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

    func testRuntimePredictionLatchConsumesTransientOutputsOnce() throws {
        var latch = RuntimePredictionLatch()
        var prediction = [Float](repeating: 0, count: ActionLayout.count)
        prediction[2] = 0.5
        prediction[3] = -0.25
        prediction[4] = 1
        prediction[12] = 0.4
        prediction[14 + 13] = 1

        XCTAssertNil(latch.consume())
        latch.publish(prediction)
        let first = latch.consume()
        XCTAssertTrue(first?.isFresh == true)
        XCTAssertEqual(first?.values, prediction)

        let repeated = latch.consume()
        XCTAssertTrue(repeated?.isFresh == false)
        let repeatedHistory = RuntimeActionSemantics.historyValues(
            try XCTUnwrap(repeated?.values),
            predictionIsFresh: repeated?.isFresh ?? true
        )
        XCTAssertEqual(repeatedHistory[2], 0)
        XCTAssertEqual(repeatedHistory[3], 0)
        XCTAssertEqual(repeatedHistory[12], 0)
        XCTAssertEqual(repeatedHistory[13], 0)
        XCTAssertEqual(repeatedHistory[4], 1, "Held mouse-button state must survive a reused tick")
        XCTAssertEqual(repeatedHistory[14 + 13], 1, "Held keyboard state must survive a reused tick")

        latch.publish(prediction)
        XCTAssertTrue(latch.consume()?.isFresh == true)
        latch.reset()
        XCTAssertNil(latch.consume())
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
        _ = writer.finish()
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
        XCTAssertTrue(grid.detail.contains("top 4 maps"))
    }

    func testCompiledTrainingStepMatchesUncompiledAdamW() throws {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 16, height: 12, colorMode: .grayscale, bitDepth: 8)
        profile.training.historyLength = 1
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
        let images = MLXArray([Float](repeating: 0.25, count: 2 * 16 * 12), [2, 12, 16, 1])
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

    func testCompiledTrainingThroughputAndActiveMemoryStayBounded() {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 12, height: 8, colorMode: .grayscale, bitDepth: 8)
        profile.training.historyLength = 1
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let model = AgentPolicy(profile: profile)
        let optimizer = ResumableAdamW(learningRate: 0.001, weightDecay: 0.01)
        optimizer.initialize(model: model)
        let images = MLXArray([Float](repeating: 0.25, count: 2 * 12 * 8), [2, 8, 12, 1])
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
        let manifest = DatasetCacheManifest(key: "test", createdAt: Date(), preprocessing: spec, actionFPS: 60, perceptionFPS: 30, historyLength: 2, sampleCount: 3, observationBytesPerSample: 1, actionValuesPerSample: ActionLayout.count, segments: [CacheSegment(recordingID: UUID(), start: 0, count: 3)])
        let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(manifest).write(to: directory.appendingPathComponent("manifest.json"))
        try Data([1, 2, 3]).write(to: directory.appendingPathComponent("observations.bin"))
        var actions = Data()
        for row in 1...3 {
            var values = [Float](repeating: 0, count: ActionLayout.count); values[0] = Float(row)
            values.withUnsafeBytes { actions.append(contentsOf: $0) }
        }
        try actions.write(to: directory.appendingPathComponent("actions.bin"))
        let dataset = try CachedDataset(directory: directory)
        XCTAssertEqual(dataset.packedObservations(at: [2, 0, 1]), Data([3, 1, 2]))
        let targets = MLXArray(dataset.actionBatch(at: [2, 0]), [2, ActionLayout.count], type: Float.self).asArray(Float.self)
        XCTAssertEqual(targets[0], 3); XCTAssertEqual(targets[ActionLayout.count], 1)
        let history = MLXArray(dataset.historyBatch(at: [2]), [1, 2, ActionLayout.count], type: Float.self).asArray(Float.self)
        XCTAssertEqual(history[0], 1); XCTAssertEqual(history[ActionLayout.count], 2)
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
        let images = MLXArray([Float](repeating: 0.5, count: 2 * 32 * 24), [2, 24, 32, 1])
        let history = MLXArray([Float](repeating: 0, count: 2 * 2 * ActionLayout.count), [2, 2, ActionLayout.count])
        let targets = MLXArray([Float](repeating: 0, count: 2 * ActionLayout.count), [2, ActionLayout.count])
        let gradient = valueAndGrad(model: model) { model, arrays in [model.loss(images: arrays[0], history: arrays[1], targets: arrays[2])] }
        let result = gradient(model, [images, history, targets])
        MLX.eval(result.0, result.1)
        XCTAssertEqual(model.predictions(images: images, history: history).shape, [2, ActionLayout.count])
        XCTAssertTrue(result.0[0].item(Float.self).isFinite)
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
        let images = MLXArray([Float](repeating: 0.5, count: 32 * 24), [1, 24, 32, 1])
        let history = MLXArray([Float](repeating: 0, count: 2 * ActionLayout.count), [1, 2, ActionLayout.count])

        let layers = model.visualActivations(images: images)
        XCTAssertEqual(layers.map(\.shape), [[1, 12, 16, 16], [1, 6, 8, 32], [1, 3, 4, 48]])

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
        let saliencyForward = saliency([images, history])
        let gradients = saliencyGradient([saliencyForward[1], history, selector])
        let weights = gradients.mean(axes: [1, 2], keepDims: true)
        let saliencyMap = model.sampledForVisualization(relu((saliencyForward[1] * weights).sum(axis: -1, keepDims: true)))
        let saliencyResult = [saliencyForward[0], saliencyMap]
        MLX.eval(expected, activityResults.flatMap { $0 }, channelResult, saliencyForward, saliencyResult)

        XCTAssertEqual(activityResults.map { $0[1].shape }, [[1, 12, 16, 1], [1, 6, 8, 1], [1, 3, 4, 1]])
        XCTAssertEqual(channelResult[1].shape, [1, 3, 4, 16])
        XCTAssertEqual(saliencyForward[1].shape, [1, 3, 4, 48])
        XCTAssertEqual(saliencyResult[1].shape, [1, 3, 4, 1])
        let expectedValues = expected.asArray(Float.self)
        for prediction in activityResults.map({ $0[0] }) + [channelResult[0], saliencyResult[0]] {
            XCTAssertTrue(zip(expectedValues, prediction.asArray(Float.self)).allSatisfy { abs($0 - $1) < 1e-5 })
        }
        XCTAssertTrue(saliencyResult[1].asArray(Float.self).allSatisfy { $0.isFinite && $0 >= 0 })
    }

    func testOptimizerCheckpointResumesExactly() throws {
        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 16, height: 12, colorMode: .grayscale, bitDepth: 8)
        profile.training.historyLength = 1
        profile.training.architecture = .small
        profile.training.architecture.dropout = 0
        profile.training.precision = .float32
        let modelA = AgentPolicy(profile: profile)
        let optimizerA = ResumableAdamW(learningRate: 0.001, weightDecay: 0.01)
        let images = MLXArray([Float](repeating: 0.25, count: 16 * 12), [1, 12, 16, 1])
        let history = MLXArray([Float](repeating: 0, count: ActionLayout.count), [1, 1, ActionLayout.count])
        let targets = MLXArray([Float](repeating: 0, count: ActionLayout.count), [1, ActionLayout.count])
        let gradientA = valueAndGrad(model: modelA) { model, arrays in [model.loss(images: arrays[0], history: arrays[1], targets: arrays[2])] }
        let first = gradientA(modelA, [images, history, targets])
        optimizerA.update(model: modelA, gradients: first.1, targetType: .float32)
        MLX.eval(modelA.parameters(), optimizerA.stateArrays())

        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("resume-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let weights = directory.appendingPathComponent("weights.safetensors")
        let optimizer = directory.appendingPathComponent("optimizer.safetensors")
        try modelA.saveWeights(to: weights); try optimizerA.save(to: optimizer)

        let secondA = gradientA(modelA, [images, history, targets])
        optimizerA.update(model: modelA, gradients: secondA.1, targetType: .float32)
        MLX.eval(modelA.parameters(), optimizerA.stateArrays())

        let modelB = AgentPolicy(profile: profile)
        let optimizerB = ResumableAdamW(learningRate: 0.001, weightDecay: 0.01)
        try modelB.loadWeights(from: weights); try optimizerB.load(from: optimizer)
        let gradientB = valueAndGrad(model: modelB) { model, arrays in [model.loss(images: arrays[0], history: arrays[1], targets: arrays[2])] }
        let secondB = gradientB(modelB, [images, history, targets])
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

    private func actionFloat(_ data: Data, index: Int) -> Float {
        let bits = data.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: index * MemoryLayout<UInt32>.size, as: UInt32.self)
        }
        return Float(bitPattern: UInt32(littleEndian: bits))
    }
}
