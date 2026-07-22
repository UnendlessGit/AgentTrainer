import XCTest
@testable import AgentTrainer

final class WindowsRuntimeSmokeTests: XCTestCase {
    func testCapturedWindowsPackageImportsAndBuildsTrainingCache() async throws {
        guard let sourcePath = ProcessInfo.processInfo.environment["AGENTTRAINER_WINDOWS_SMOKE_PACKAGE"], !sourcePath.isEmpty else {
            throw XCTSkip("Set AGENTTRAINER_WINDOWS_SMOKE_PACKAGE to a captured Windows .atrrecord package.")
        }
        let source = URL(fileURLWithPath: sourcePath, isDirectory: true)
        let container = FileManager.default.temporaryDirectory.appendingPathComponent("windows-runtime-smoke-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: container) }

        let sourceManifestData = try Data(contentsOf: source.appendingPathComponent("manifest.json"))
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let sourceManifest = try decoder.decode(RecordingManifest.self, from: sourceManifestData)
        let sourceEventData = try Data(contentsOf: source.appendingPathComponent(sourceManifest.eventFile))
        let sourceVideoData = try Data(contentsOf: source.appendingPathComponent(sourceManifest.videoFile))

        let store = WorkspaceStore(root: container.appendingPathComponent("Library", isDirectory: true))
        try await store.prepare()
        let folder = RecordingFolder(id: UUID(), name: "Windows smoke", createdAt: Date())
        try await store.saveRecordingFolder(folder)

        let imported = try await store.importRecordings(from: [source], into: folder.id)
        let item = try XCTUnwrap(imported.first)
        XCTAssertEqual(imported.count, 1)
        XCTAssertNotEqual(item.id, sourceManifest.id)
        XCTAssertEqual(item.manifest.eventCount, sourceManifest.eventCount)
        XCTAssertEqual(item.manifest.pixelWidth, sourceManifest.pixelWidth)
        XCTAssertEqual(item.manifest.pixelHeight, sourceManifest.pixelHeight)
        XCTAssertGreaterThan(item.manifest.eventCount, 0)

        let importedEvents = item.directory.appendingPathComponent(item.manifest.eventFile)
        let importedVideo = item.directory.appendingPathComponent(item.manifest.videoFile)
        XCTAssertEqual(try Data(contentsOf: importedEvents), sourceEventData)
        XCTAssertEqual(try Data(contentsOf: importedVideo), sourceVideoData)
        XCTAssertEqual(try InputEventReader.read(url: importedEvents).count, item.manifest.eventCount)

        // Author a second package through the native Mac storage/event APIs so
        // the cache build below exercises a genuinely mixed-platform set.
        let nativeID = UUID()
        let nativeDirectory = try await store.createRecordingDirectory(id: nativeID)
        try FileManager.default.copyItem(
            at: source.appendingPathComponent(sourceManifest.videoFile),
            to: nativeDirectory.appendingPathComponent("capture.mov")
        )
        let nativeEventWriter = try InputEventWriter(url: nativeDirectory.appendingPathComponent("events.atrevents"))
        nativeEventWriter.append(InputSample(timestampNanos: sourceManifest.hostStartNanos, kind: .mouseMove, x: 20, y: 20))
        nativeEventWriter.append(InputSample(timestampNanos: sourceManifest.hostStartNanos + 600_000_000, kind: .key, keyCode: 13, isDown: true))
        nativeEventWriter.append(InputSample(timestampNanos: sourceManifest.hostStartNanos + 1_100_000_000, kind: .key, keyCode: 13, isDown: false))
        let nativeEventCount = try nativeEventWriter.finish()
        let nativeManifest = RecordingManifest(
            id: nativeID,
            name: "Native Mac companion",
            createdAt: Date(),
            hostStartNanos: sourceManifest.hostStartNanos,
            duration: sourceManifest.duration,
            capture: sourceManifest.capture,
            globalRect: sourceManifest.globalRect,
            pixelWidth: sourceManifest.pixelWidth,
            pixelHeight: sourceManifest.pixelHeight,
            deliveredFPS: sourceManifest.deliveredFPS,
            eventCount: nativeEventCount,
            trimStart: sourceManifest.trimStart,
            trimEnd: sourceManifest.trimEnd,
            folderID: folder.id
        )
        try await store.writeRecording(nativeManifest, to: nativeDirectory)
        let nativeItem = RecordingItem(manifest: nativeManifest, directory: nativeDirectory)

        var profile = AIProfile.fresh()
        profile.preprocessing = PreprocessingSpec(width: 32, height: 20, colorMode: .grayscale, bitDepth: 8)
        profile.training.actionFPS = 4
        profile.training.perceptionFPS = 2
        profile.training.historyLength = 2
        let dataset = try await DatasetCacheBuilder(workspace: store).cache(for: profile, recordings: [nativeItem, item]) { _, _ in }
        XCTAssertGreaterThan(dataset.count, 0)
        XCTAssertEqual(dataset.manifest.segments.map(\.recordingID), [nativeID, item.id])
        XCTAssertTrue(dataset.demonstratedKeyCodes().contains(13))

        XCTAssertEqual(try Data(contentsOf: source.appendingPathComponent(sourceManifest.eventFile)), sourceEventData)
        XCTAssertEqual(try Data(contentsOf: source.appendingPathComponent(sourceManifest.videoFile)), sourceVideoData)
    }
}
