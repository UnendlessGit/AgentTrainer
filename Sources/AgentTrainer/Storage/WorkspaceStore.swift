@preconcurrency import AVFoundation
import Foundation

enum WorkspaceDataKind: String, Sendable {
    case trainingData = "Training Data"
    case models = "AI Models"

    fileprivate var managedNames: [String] {
        switch self {
        case .trainingData: ["Recordings", "Caches", "recording-folders.json"]
        case .models: ["Profiles", "model-contract.json", "model-artifact-audit-1.8.2.json"]
        }
    }
}

struct WorkspaceLocations: Equatable, Sendable {
    var supportRoot: URL
    var trainingDataRoot: URL
    var modelsRoot: URL

    var trainingDataIsDefault: Bool { trainingDataRoot.standardizedFileURL == supportRoot.standardizedFileURL }
    var modelsAreDefault: Bool { modelsRoot.standardizedFileURL == supportRoot.standardizedFileURL }
}

struct WorkspaceStorageUsage: Equatable, Sendable {
    var totalBytes: Int64
    var trainingDataBytes: Int64
    var modelBytes: Int64
}

struct WorkspaceDestinationInspection: Equatable, Sendable {
    var url: URL
    var isCurrentLocation: Bool
    var containsManagedData: Bool
}

struct WorkspaceRelocationResult: Equatable, Sendable {
    var destination: URL
    var movedExistingData: Bool
    var sourceCleanupComplete: Bool
}

struct WorkspaceVersionActivation: Sendable {
    var profile: AIProfile
    var version: ModelVersionManifest
    var checkpointIsResumable: Bool
}

struct RecordingImportResult: Equatable, Sendable {
    var importedCount: Int
    var createdFolderCount: Int
    var regeneratedIdentifierCount: Int
}

struct RecordingExportResult: Equatable, Sendable {
    var destination: URL
    var exportedCount: Int
}

actor WorkspaceStore {
    static let shared = WorkspaceStore()

    static let defaultRoot: URL = {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support", isDirectory: true)
        return support.appendingPathComponent("AgentTrainer", isDirectory: true).standardizedFileURL
    }()

    let root: URL
    private var trainingDataRoot: URL
    private var modelsRoot: URL
    private let persistsLocations: Bool

    private var recordingsRoot: URL { trainingDataRoot.appendingPathComponent("Recordings", isDirectory: true) }
    private var profilesRoot: URL { modelsRoot.appendingPathComponent("Profiles", isDirectory: true) }
    private var cachesRoot: URL { trainingDataRoot.appendingPathComponent("Caches", isDirectory: true) }
    private var foldersURL: URL { trainingDataRoot.appendingPathComponent("recording-folders.json") }

    private static let trainingDataRootKey = "AgentTrainer.TrainingDataRoot.v1"
    private static let modelsRootKey = "AgentTrainer.ModelsRoot.v1"

    private let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    private let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    init(root: URL? = nil) {
        let environmentRoot = root == nil
            ? ProcessInfo.processInfo.environment["AGENTTRAINER_WORKSPACE_ROOT"]
                .flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0, isDirectory: true) }
            : nil
        let resolvedRoot = (root ?? environmentRoot ?? Self.defaultRoot).standardizedFileURL
        self.root = resolvedRoot
        persistsLocations = root == nil && environmentRoot == nil
        if persistsLocations {
            trainingDataRoot = UserDefaults.standard.string(forKey: Self.trainingDataRootKey)
                .map { URL(fileURLWithPath: $0, isDirectory: true).standardizedFileURL } ?? resolvedRoot
            modelsRoot = UserDefaults.standard.string(forKey: Self.modelsRootKey)
                .map { URL(fileURLWithPath: $0, isDirectory: true).standardizedFileURL } ?? resolvedRoot
        } else {
            trainingDataRoot = resolvedRoot
            modelsRoot = resolvedRoot
        }
    }

    func prepare() throws {
        try ensureLocationIsAvailable(trainingDataRoot, name: WorkspaceDataKind.trainingData.rawValue)
        if modelsRoot != trainingDataRoot { try ensureLocationIsAvailable(modelsRoot, name: WorkspaceDataKind.models.rawValue) }
        for directory in [root, trainingDataRoot, modelsRoot, recordingsRoot, profilesRoot, cachesRoot] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
    }

    func locations() -> WorkspaceLocations {
        WorkspaceLocations(supportRoot: root, trainingDataRoot: trainingDataRoot, modelsRoot: modelsRoot)
    }

    func cacheDirectory() -> URL { cachesRoot }

    func inspectDestination(_ destination: URL, for kind: WorkspaceDataKind) throws -> WorkspaceDestinationInspection {
        let destination = normalized(destination)
        try validateDestination(destination, for: kind)
        let current = location(for: kind)
        return WorkspaceDestinationInspection(
            url: destination,
            isCurrentLocation: destination == current,
            containsManagedData: kind.managedNames.contains { FileManager.default.fileExists(atPath: destination.appendingPathComponent($0).path) }
        )
    }

    /// Moves a managed library with copy-then-verify semantics, or switches to
    /// an already-populated library without merging it. The source is removed
    /// only after the new location is persisted and prepared, so interruption
    /// can leave a duplicate but never the only copy half-moved.
    func relocate(_ kind: WorkspaceDataKind, to requestedDestination: URL, useExisting: Bool) throws -> WorkspaceRelocationResult {
        let inspection = try inspectDestination(requestedDestination, for: kind)
        if inspection.isCurrentLocation {
            return WorkspaceRelocationResult(destination: inspection.url, movedExistingData: false, sourceCleanupComplete: true)
        }
        if inspection.containsManagedData && !useExisting {
            throw AgentTrainerError.storage("The selected folder already contains \(kind.rawValue.lowercased()). Switch to that library instead of merging two libraries.")
        }

        let source = location(for: kind)
        let destination = inspection.url
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        try verifyWritable(destination, name: kind.rawValue)

        if useExisting {
            try commitLocation(destination, for: kind)
            return WorkspaceRelocationResult(destination: destination, movedExistingData: false, sourceCleanupComplete: true)
        }

        let existingNames = kind.managedNames.filter { FileManager.default.fileExists(atPath: source.appendingPathComponent($0).path) }
        if existingNames.isEmpty {
            try commitLocation(destination, for: kind)
            return WorkspaceRelocationResult(destination: destination, movedExistingData: false, sourceCleanupComplete: true)
        }

        let requiredBytes = existingNames.reduce(Int64(0)) { $0 + logicalBytes(at: source.appendingPathComponent($1)) }
        if let available = try? destination.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage,
           available >= 0, Int64(available) < requiredBytes + 64 * 1_024 * 1_024 {
            throw AgentTrainerError.storage("The selected disk does not have enough free space to move \(kind.rawValue.lowercased()).")
        }

        let stagingName = ".AgentTrainer-\(kind == .trainingData ? "Training" : "Models")-Migration-\(UUID().uuidString)"
        let staging = destination.appendingPathComponent(stagingName, isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        do {
            for name in existingNames {
                let sourceItem = source.appendingPathComponent(name)
                let stagedItem = staging.appendingPathComponent(name)
                try FileManager.default.copyItem(at: sourceItem, to: stagedItem)
                guard contentSummary(at: sourceItem) == contentSummary(at: stagedItem) else {
                    throw AgentTrainerError.storage("The copied \(kind.rawValue.lowercased()) could not be verified. The original library was left unchanged.")
                }
            }
            for name in existingNames {
                try FileManager.default.moveItem(at: staging.appendingPathComponent(name), to: destination.appendingPathComponent(name))
            }
            try FileManager.default.removeItem(at: staging)
        } catch {
            try? FileManager.default.removeItem(at: staging)
            for name in existingNames {
                let destinationItem = destination.appendingPathComponent(name)
                if FileManager.default.fileExists(atPath: destinationItem.path) { try? FileManager.default.removeItem(at: destinationItem) }
            }
            throw error
        }

        try commitLocation(destination, for: kind)
        var cleanupComplete = true
        for name in existingNames {
            do { try FileManager.default.removeItem(at: source.appendingPathComponent(name)) }
            catch { cleanupComplete = false }
        }
        return WorkspaceRelocationResult(destination: destination, movedExistingData: true, sourceCleanupComplete: cleanupComplete)
    }

    /// One-time compatibility boundary for learned action semantics. Every
    /// saved version is inspected independently: compatible brains stay in
    /// place, while incompatible or unreadable artifacts are moved to a
    /// recovery archive instead of being deleted. Profiles and recordings are
    /// never removed by this migration.
    @discardableResult
    func removeObsoleteModelArtifacts(currentSchema: Int) throws -> Int {
        try prepare()
        let marker = modelsRoot.appendingPathComponent("model-contract.json")
        let auditMarker = modelsRoot.appendingPathComponent("model-artifact-audit-2.1.json")
        let storedSchema = (try? Data(contentsOf: marker)).flatMap { try? decoder.decode(Int.self, from: $0) }
        if storedSchema == currentSchema,
           let data = try? Data(contentsOf: auditMarker),
           (try? decoder.decode(Int.self, from: data)) == currentSchema {
            return 0
        }

        var archived = 0
        for var profile in listProfiles() {
            let profileRoot = profileDirectory(profile.id)
            let versionsRoot = profileRoot.appendingPathComponent("Versions", isDirectory: true)
            let versionItems = (try? FileManager.default.contentsOfDirectory(
                at: versionsRoot,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )) ?? []
            var compatibleVersionIDs = Set<UUID>()

            for item in versionItems {
                let manifestURL = item.appendingPathComponent("manifest.json")
                if let data = try? Data(contentsOf: manifestURL),
                   let manifest = try? decoder.decode(ModelVersionManifest.self, from: data),
                   manifest.schemaVersion == currentSchema,
                   manifest.id.uuidString.caseInsensitiveCompare(item.lastPathComponent) == .orderedSame,
                   manifest.artifactFileNamesAreSafe,
                   manifest.trainingDataCoverage?.isValid != false,
                   FileManager.default.fileExists(atPath: item.appendingPathComponent(manifest.weightsFile).path) {
                    compatibleVersionIDs.insert(manifest.id)
                } else {
                    try archiveModelArtifact(
                        item,
                        profileRoot: profileRoot,
                        category: "Versions",
                        storedSchema: storedSchema
                    )
                    archived += 1
                }
            }

            let checkpoint = profileRoot.appendingPathComponent("Checkpoint", isDirectory: true)
            var keepsCheckpoint = false
            if FileManager.default.fileExists(atPath: checkpoint.path) {
                let checkpointMarker = checkpoint.appendingPathComponent("model-schema.json")
                let checkpointSchema = (try? Data(contentsOf: checkpointMarker)).flatMap { try? decoder.decode(Int.self, from: $0) }
                keepsCheckpoint = checkpointSchema == currentSchema
                    || (checkpointSchema == nil && (storedSchema == currentSchema || !compatibleVersionIDs.isEmpty))
                if keepsCheckpoint {
                    // Checkpoints created before 1.8.2 did not carry their own
                    // schema marker. Once compatibility is established from a
                    // current library or version manifest, make it explicit.
                    if checkpointSchema == nil {
                        try atomicWrite(try encoder.encode(currentSchema), to: checkpointMarker)
                    }
                } else {
                    try archiveModelArtifact(
                        checkpoint,
                        profileRoot: profileRoot,
                        category: "Checkpoints",
                        storedSchema: checkpointSchema ?? storedSchema
                    )
                    archived += 1
                }
            }

            let activeIsCompatible = profile.activeVersionID.map(compatibleVersionIDs.contains) ?? false
            if profile.activeVersionID != nil && !activeIsCompatible {
                profile.activeVersionID = nil
                profile.trainingProgress = nil
            }

            // A compatible saved brain or checkpoint proves that this profile
            // already uses the current contract. Never rewrite its settings.
            if compatibleVersionIDs.isEmpty && !keepsCheckpoint {
                profile.training.architecture = migratedArchitecture(profile.training.architecture)
                profile.training.temporalVision = TemporalVisionConfiguration()
                profile.training.generalization = GeneralizationConfiguration()
                if !profile.training.learningRate.isFinite || profile.training.learningRate <= 0 {
                    profile.training.learningRate = 0.0003
                } else {
                    profile.training.learningRate = min(0.0003, profile.training.learningRate)
                }
                if profile.training.perceptionFPS > profile.training.actionFPS {
                    profile.training.perceptionFPS = profile.training.actionFPS
                }
            }
            try saveProfile(profile)
        }
        try clearCaches()
        try atomicWrite(try encoder.encode(currentSchema), to: marker)
        try atomicWrite(try encoder.encode(currentSchema), to: auditMarker)
        return archived
    }

    private func archiveModelArtifact(_ source: URL, profileRoot: URL, category: String, storedSchema: Int?) throws {
        let schemaName = storedSchema.map(String.init) ?? "Unknown"
        let categoryRoot = profileRoot
            .appendingPathComponent("Archived Model Artifacts", isDirectory: true)
            .appendingPathComponent("Model Contract \(schemaName)", isDirectory: true)
            .appendingPathComponent(category, isDirectory: true)
        try FileManager.default.createDirectory(at: categoryRoot, withIntermediateDirectories: true)
        var destination = categoryRoot.appendingPathComponent(source.lastPathComponent, isDirectory: true)
        if FileManager.default.fileExists(atPath: destination.path) {
            destination = categoryRoot.appendingPathComponent("\(source.lastPathComponent)-\(UUID().uuidString)", isDirectory: true)
        }
        try FileManager.default.moveItem(at: source, to: destination)
    }

    private func migratedArchitecture(_ previous: ArchitectureSpec) -> ArchitectureSpec {
        let widestConvolution = previous.convolutionChannels.max() ?? 0
        var architecture: ArchitectureSpec
        if previous.visualEmbedding <= 128, previous.recurrentWidth <= 128, widestConvolution <= 64 {
            architecture = .small
        } else if previous.visualEmbedding >= 384 || previous.recurrentWidth >= 320 || widestConvolution >= 192 {
            architecture = .large
        } else {
            architecture = .balanced
        }
        architecture.recurrentKind = previous.recurrentKind
        return architecture
    }

    func createRecordingDirectory(id: UUID) throws -> URL {
        try prepare()
        let url = recordingsRoot.appendingPathComponent("\(id.uuidString).atrrecord", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        return url
    }

    func writeRecording(_ manifest: RecordingManifest, to directory: URL) throws {
        try atomicWrite(try encoder.encode(manifest), to: directory.appendingPathComponent("manifest.json"))
    }

    /// Imports native recording packages without translating their video or
    /// input stream. A transfer may be one or more `.atrrecord` directories, a
    /// folder containing those packages, or an exported recording library with
    /// `Recordings` and `recording-folders.json` at its root.
    ///
    /// Every source is validated before the managed library changes. The folder
    /// index and all packages then commit as one transaction; a collision gets
    /// a fresh recording identifier while the media and input bytes stay exact.
    func importRecordings(from selectedURLs: [URL], fallbackFolderID requestedFallbackFolderID: UUID?) async throws -> RecordingImportResult {
        try prepare()
        let sources = try recordingImportSources(from: selectedURLs)
        guard !sources.isEmpty else {
            throw AgentTrainerError.storage("No AgentTrainer recording packages were found in the selected location.")
        }

        var validated: [ValidatedRecordingImport] = []
        validated.reserveCapacity(sources.count)
        for source in sources {
            validated.append(try await validateRecordingImport(source))
        }

        let originalFolders = listRecordingFolders()
        var folders = originalFolders
        let needsFallbackFolder = validated.contains { item in
            guard let folderID = item.manifest.folderID else { return true }
            return item.sourceFolders[folderID] == nil
        }
        var fallbackFolderID = requestedFallbackFolderID.flatMap { requested in
            originalFolders.contains(where: { $0.id == requested }) ? requested : nil
        } ?? originalFolders.first?.id
        var folderMapping: [UUID: UUID] = [:]
        var createdFolderCount = 0
        if fallbackFolderID == nil, needsFallbackFolder {
            let fallback = RecordingFolder(id: UUID(), name: "Recordings", createdAt: Date())
            folders.append(fallback)
            fallbackFolderID = fallback.id
            createdFolderCount += 1
        }

        for item in validated {
            guard let sourceFolderID = item.manifest.folderID else { continue }
            if folderMapping[sourceFolderID] != nil { continue }
            guard let sourceFolder = item.sourceFolders[sourceFolderID] else {
                if let fallbackFolderID { folderMapping[sourceFolderID] = fallbackFolderID }
                continue
            }
            if let exact = folders.first(where: { $0.id == sourceFolder.id }) {
                folderMapping[sourceFolderID] = exact.id
            } else if let sameName = folders.first(where: {
                $0.name.localizedCaseInsensitiveCompare(sourceFolder.name) == .orderedSame
            }) {
                folderMapping[sourceFolderID] = sameName.id
            } else {
                var importedFolder = sourceFolder
                importedFolder.name = sourceFolder.name.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !importedFolder.name.isEmpty else {
                    if fallbackFolderID == nil {
                        let fallback = RecordingFolder(id: UUID(), name: "Recordings", createdAt: Date())
                        folders.append(fallback); fallbackFolderID = fallback.id; createdFolderCount += 1
                    }
                    folderMapping[sourceFolderID] = fallbackFolderID
                    continue
                }
                if folders.contains(where: { $0.id == importedFolder.id }) { importedFolder.id = UUID() }
                folders.append(importedFolder)
                folderMapping[sourceFolderID] = importedFolder.id
                createdFolderCount += 1
            }
        }

        let existingIDs = Set(listRecordings().map(\.id))
        var reservedIDs = existingIDs
        var regeneratedIdentifierCount = 0
        let stagingRoot = recordingsRoot.appendingPathComponent(".RecordingImport.\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: stagingRoot, withIntermediateDirectories: false)
        var staged: [(temporary: URL, destination: URL)] = []
        staged.reserveCapacity(validated.count)

        do {
            for item in validated {
                var manifest = item.manifest
                if reservedIDs.contains(manifest.id) {
                    repeat { manifest.id = UUID() } while reservedIDs.contains(manifest.id)
                    regeneratedIdentifierCount += 1
                }
                reservedIDs.insert(manifest.id)
                let mappedFolderID = item.manifest.folderID.flatMap { folderMapping[$0] }
                guard let destinationFolderID = mappedFolderID ?? fallbackFolderID ?? folders.first?.id else {
                    throw AgentTrainerError.storage("A destination folder could not be established for \(manifest.name).")
                }
                manifest.folderID = destinationFolderID

                let temporary = stagingRoot.appendingPathComponent("\(manifest.id.uuidString).atrrecord", isDirectory: true)
                try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false)
                try copyRecordingArtifacts(for: manifest, from: item.directory, to: temporary)
                try atomicWrite(try encoder.encode(manifest), to: temporary.appendingPathComponent("manifest.json"))

                let copiedEvents = temporary.appendingPathComponent(manifest.eventFile)
                let copiedEventCount = try InputEventReader.mapped(url: copiedEvents).count
                guard copiedEventCount == manifest.eventCount else {
                    throw AgentTrainerError.storage("\(manifest.name) declares \(manifest.eventCount) inputs but contains \(copiedEventCount).")
                }
                let destination = recordingsRoot.appendingPathComponent("\(manifest.id.uuidString).atrrecord", isDirectory: true)
                guard !FileManager.default.fileExists(atPath: destination.path) else {
                    throw AgentTrainerError.storage("A recording identifier collision could not be resolved safely.")
                }
                staged.append((temporary, destination))
            }
        } catch {
            try? FileManager.default.removeItem(at: stagingRoot)
            throw error
        }

        let originalFolderData = try? Data(contentsOf: foldersURL)
        var committed: [URL] = []
        do {
            try atomicWrite(try encoder.encode(folders), to: foldersURL)
            for item in staged {
                try FileManager.default.moveItem(at: item.temporary, to: item.destination)
                committed.append(item.destination)
            }
            try FileManager.default.removeItem(at: stagingRoot)
        } catch {
            for directory in committed { try? FileManager.default.removeItem(at: directory) }
            if let originalFolderData { try? atomicWrite(originalFolderData, to: foldersURL) }
            else { try? FileManager.default.removeItem(at: foldersURL) }
            try? FileManager.default.removeItem(at: stagingRoot)
            throw AgentTrainerError.storage("The recordings could not all be imported, so the library was restored: \(error.localizedDescription)")
        }

        return RecordingImportResult(
            importedCount: staged.count,
            createdFolderCount: createdFolderCount,
            regeneratedIdentifierCount: regeneratedIdentifierCount
        )
    }

    /// Exports a self-contained subset of the native recording library. The
    /// package layout is deliberately the same on every platform: a Recordings
    /// directory plus the subset of `recording-folders.json` it references.
    /// Source packages are copied as-is and verified before publication.
    func exportRecordings(_ requestedItems: [RecordingItem], to requestedDestination: URL) throws -> RecordingExportResult {
        try prepare()
        let itemsByID = Dictionary(requestedItems.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        let items = itemsByID.values.sorted { $0.manifest.createdAt < $1.manifest.createdAt }
        guard !items.isEmpty else { throw AgentTrainerError.storage("Select at least one recording to export.") }

        let managedRoot = normalized(recordingsRoot)
        for item in items {
            let directory = normalized(item.directory)
            guard directory.deletingLastPathComponent() == managedRoot,
                  directory.pathExtension == "atrrecord",
                  FileManager.default.fileExists(atPath: directory.path) else {
                throw AgentTrainerError.storage("A selected recording is no longer in the managed library.")
            }
        }

        let destination = requestedDestination.standardizedFileURL
        guard destination.isFileURL else { throw AgentTrainerError.storage("Recording exports require a local or mounted file-system folder.") }
        guard !items.contains(where: { pathsOverlap(destination, $0.directory) }) else {
            throw AgentTrainerError.storage("Choose an export location outside the selected recording packages.")
        }
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: destination.path, isDirectory: &isDirectory) {
            guard isDirectory.boolValue else { throw AgentTrainerError.storage("Choose a folder for the recording export.") }
            let contents = try FileManager.default.contentsOfDirectory(atPath: destination.path)
            guard contents.isEmpty else { throw AgentTrainerError.storage("The export folder must be empty so existing files are never overwritten.") }
        }

        let parent = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let staging = parent.appendingPathComponent(".AgentTrainer-Recording-Export.\(UUID().uuidString).tmp", isDirectory: true)
        let stagedRecordings = staging.appendingPathComponent("Recordings", isDirectory: true)
        try FileManager.default.createDirectory(at: stagedRecordings, withIntermediateDirectories: true)
        do {
            for item in items {
                let copied = stagedRecordings.appendingPathComponent(item.directory.lastPathComponent, isDirectory: true)
                try FileManager.default.copyItem(at: item.directory, to: copied)
                guard contentSummary(at: item.directory) == contentSummary(at: copied) else {
                    throw AgentTrainerError.storage("\(item.manifest.name) could not be verified after copying.")
                }
            }
            let folderIDs = Set(items.compactMap { $0.manifest.folderID })
            let folders = listRecordingFolders().filter { folderIDs.contains($0.id) }
            try atomicWrite(try encoder.encode(folders), to: staging.appendingPathComponent("recording-folders.json"))

            if FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.removeItem(at: destination) }
            try FileManager.default.moveItem(at: staging, to: destination)
        } catch {
            try? FileManager.default.removeItem(at: staging)
            throw error
        }
        return RecordingExportResult(destination: destination, exportedCount: items.count)
    }

    /// Repairs legacy manifests before the strict library scan runs. Earlier
    /// defensive validation filtered some older recordings before the existing
    /// clock repair could discover their stale duration/trim. This pass
    /// enumerates recording directories directly and changes only invalid,
    /// decodable manifests.
    /// Video and input files are never moved, rewritten, or removed.
    @discardableResult
    func repairInvalidRecordingManifests() async throws -> Int {
        try prepare()
        guard let directories = try? FileManager.default.contentsOfDirectory(at: recordingsRoot, includingPropertiesForKeys: [.isDirectoryKey]) else { return 0 }
        var repaired = 0
        for directory in directories where directory.pathExtension == "atrrecord" {
            let manifestURL = directory.appendingPathComponent("manifest.json")
            guard let originalData = try? Data(contentsOf: manifestURL),
                  var manifest = try? decoder.decode(RecordingManifest.self, from: originalData),
                  !manifest.isStructurallyValid,
                  (1...2).contains(manifest.schemaVersion),
                  Self.isSafeLeafName(manifest.videoFile), Self.isSafeLeafName(manifest.eventFile),
                  manifest.thumbnailFile.map(Self.isSafeLeafName) ?? true else { continue }

            let videoURL = directory.appendingPathComponent(manifest.videoFile)
            let eventURL = directory.appendingPathComponent(manifest.eventFile)
            guard FileManager.default.fileExists(atPath: videoURL.path),
                  FileManager.default.fileExists(atPath: eventURL.path) else { continue }
            let asset = AVURLAsset(url: videoURL)
            guard let time = try? await asset.load(.duration) else { continue }
            let videoDuration = CMTimeGetSeconds(time)
            guard videoDuration.isFinite, videoDuration > 0 else { continue }

            if manifest.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { manifest.name = "Recovered Recording" }
            manifest.duration = videoDuration
            if !manifest.capture.requestedFPS.isFinite || manifest.capture.requestedFPS <= 0 || manifest.capture.requestedFPS > 1_000 {
                manifest.capture.requestedFPS = manifest.deliveredFPS.isFinite && manifest.deliveredFPS > 0 && manifest.deliveredFPS <= 1_000 ? manifest.deliveredFPS : 60
            }
            if !manifest.deliveredFPS.isFinite || manifest.deliveredFPS < 0 || manifest.deliveredFPS > 1_000 {
                manifest.deliveredFPS = manifest.capture.requestedFPS
            }
            let rect = manifest.globalRect
            if !rect.x.isFinite || !rect.y.isFinite || !rect.width.isFinite || !rect.height.isFinite {
                manifest.globalRect = CodableRect(CGRect(x: 0, y: 0, width: max(1, manifest.pixelWidth), height: max(1, manifest.pixelHeight)))
            }
            manifest.pixelWidth = min(32_768, max(1, manifest.pixelWidth))
            manifest.pixelHeight = min(32_768, max(1, manifest.pixelHeight))
            manifest.eventCount = max(0, manifest.eventCount)
            manifest.trimStart = manifest.trimStart.isFinite ? min(videoDuration, max(0, manifest.trimStart)) : 0
            if let trimEnd = manifest.trimEnd {
                manifest.trimEnd = trimEnd.isFinite ? min(videoDuration, max(manifest.trimStart, trimEnd)) : videoDuration
            }
            guard manifest.isStructurallyValid else { continue }

            let backup = directory.appendingPathComponent("manifest.pre-1.8.1-recovery.json")
            if !FileManager.default.fileExists(atPath: backup.path) { try originalData.write(to: backup, options: .atomic) }
            try atomicWrite(try encoder.encode(manifest), to: manifestURL)
            repaired += 1
        }
        return repaired
    }

    func listRecordings() -> [RecordingItem] {
        guard let urls = try? FileManager.default.contentsOfDirectory(
            at: recordingsRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }
        return urls.compactMap { directory in
            guard directory.pathExtension == "atrrecord" else { return nil }
            let manifestURL = directory.appendingPathComponent("manifest.json")
            guard let data = try? Data(contentsOf: manifestURL),
                  let manifest = try? decoder.decode(RecordingManifest.self, from: data),
                  manifest.isStructurallyValid else { return nil }
            return RecordingItem(
                manifest: manifest,
                directory: directory,
                storageBytes: contentSummary(at: directory).bytes
            )
        }.sorted { $0.manifest.createdAt > $1.manifest.createdAt }
    }

    func renameRecording(_ item: RecordingItem, to name: String) throws {
        let cleanName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanName.isEmpty else { throw AgentTrainerError.invalidConfiguration("Recording names cannot be empty.") }
        var manifest = item.manifest
        manifest.name = cleanName
        try writeRecording(manifest, to: item.directory)
    }

    func deleteRecording(_ item: RecordingItem) throws {
        try FileManager.default.removeItem(at: item.directory)
    }

    /// Moves the complete set out of the visible library before deleting it.
    /// If any move fails, every prior move is rolled back; a final cleanup
    /// failure leaves a hidden recovery directory rather than a partial library.
    func deleteRecordings(_ items: [RecordingItem]) throws {
        let items = Dictionary(items.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first }).values
        guard !items.isEmpty else { return }
        let root = recordingsRoot.standardizedFileURL
        for item in items {
            guard item.directory.deletingLastPathComponent().standardizedFileURL == root,
                  item.directory.pathExtension == "atrrecord",
                  FileManager.default.fileExists(atPath: item.directory.path) else {
                throw AgentTrainerError.storage("A selected recording is no longer in the managed library.")
            }
        }
        let staging = recordingsRoot.appendingPathComponent(".RecordingDeletion.\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        var moved: [(source: URL, staged: URL)] = []
        do {
            for item in items {
                let destination = staging.appendingPathComponent(item.directory.lastPathComponent, isDirectory: true)
                try FileManager.default.moveItem(at: item.directory, to: destination)
                moved.append((item.directory, destination))
            }
        } catch {
            for item in moved.reversed() { try? FileManager.default.moveItem(at: item.staged, to: item.source) }
            try? FileManager.default.removeItem(at: staging)
            throw AgentTrainerError.storage("The recordings could not all be deleted, so the library was restored: \(error.localizedDescription)")
        }
        do { try FileManager.default.removeItem(at: staging) }
        catch {
            AppLog.write(.warning, category: "Storage", "Deleted recordings remain in a hidden recovery directory", details: staging.path)
        }
    }

    func saveRecordingFolder(_ folder: RecordingFolder) throws {
        var folder = folder
        folder.name = folder.name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !folder.name.isEmpty else { throw AgentTrainerError.invalidConfiguration("Folder names cannot be empty.") }
        var folders = listRecordingFolders()
        if let index = folders.firstIndex(where: { $0.id == folder.id }) { folders[index] = folder } else { folders.append(folder) }
        try atomicWrite(try encoder.encode(folders), to: foldersURL)
    }

    func listRecordingFolders() -> [RecordingFolder] {
        guard let data = try? Data(contentsOf: foldersURL), let folders = try? decoder.decode([RecordingFolder].self, from: data) else { return [] }
        return folders.sorted { $0.createdAt < $1.createdAt }
    }

    /// Migrates legacy/unfiled recordings into a real folder and guarantees a
    /// valid destination for every future recording.
    @discardableResult
    func normalizeRecordingFolders() throws -> UUID {
        var folders = listRecordingFolders()
        let validIDs = Set(folders.map(\.id))
        let orphaned = listRecordings().filter { item in
            guard let folderID = item.manifest.folderID else { return true }
            return !validIDs.contains(folderID)
        }
        if folders.isEmpty || !orphaned.isEmpty {
            let destination: RecordingFolder
            if let existing = folders.first(where: { $0.name.localizedCaseInsensitiveCompare("Recordings") == .orderedSame }) {
                destination = existing
            } else {
                destination = RecordingFolder(id: UUID(), name: "Recordings", createdAt: Date())
                folders.append(destination)
                try atomicWrite(try encoder.encode(folders), to: foldersURL)
            }
            for recording in orphaned { try assignRecording(recording, to: destination.id) }
            return destination.id
        }
        return folders[0].id
    }

    func assignRecording(_ item: RecordingItem, to folderID: UUID?) throws {
        var manifest = item.manifest
        manifest.folderID = folderID
        try writeRecording(manifest, to: item.directory)
    }

    func assignRecordings(_ items: [RecordingItem], to folderID: UUID) throws {
        guard listRecordingFolders().contains(where: { $0.id == folderID }) else {
            throw AgentTrainerError.storage("The destination recording folder no longer exists.")
        }
        let items = Array(Dictionary(items.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first }).values)
        var originals: [(url: URL, data: Data)] = []
        do {
            for item in items {
                let url = item.directory.appendingPathComponent("manifest.json")
                originals.append((url, try Data(contentsOf: url)))
                var manifest = item.manifest
                manifest.folderID = folderID
                try writeRecording(manifest, to: item.directory)
            }
        } catch {
            for original in originals { try? atomicWrite(original.data, to: original.url) }
            throw AgentTrainerError.storage("The recordings could not all be moved, so their original folders were restored: \(error.localizedDescription)")
        }
    }

    func deleteRecordingFolder(_ folder: RecordingFolder, includingRecordings: Bool) throws {
        var remaining = listRecordingFolders().filter { $0.id != folder.id }
        if includingRecordings {
            for recording in listRecordings() where recording.manifest.folderID == folder.id {
                try FileManager.default.removeItem(at: recording.directory)
            }
        } else {
            if remaining.isEmpty {
                remaining = [RecordingFolder(id: UUID(), name: "Recordings", createdAt: Date())]
            }
            let destinationID = remaining[0].id
            for recording in listRecordings() where recording.manifest.folderID == folder.id {
                try assignRecording(recording, to: destinationID)
            }
        }
        try atomicWrite(try encoder.encode(remaining), to: foldersURL)
    }

    private struct RecordingImportSource {
        var directory: URL
        var sourceFolders: [UUID: RecordingFolder]
    }

    private struct ValidatedRecordingImport {
        var directory: URL
        var manifest: RecordingManifest
        var sourceFolders: [UUID: RecordingFolder]
    }

    private func recordingImportSources(from selectedURLs: [URL]) throws -> [RecordingImportSource] {
        var result: [RecordingImportSource] = []
        var seen: Set<String> = []

        func decodedFolders(at root: URL) throws -> [UUID: RecordingFolder] {
            let index = root.appendingPathComponent("recording-folders.json")
            guard FileManager.default.fileExists(atPath: index.path) else { return [:] }
            do {
                let folders = try decoder.decode([RecordingFolder].self, from: Data(contentsOf: index))
                return Dictionary(folders.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
            } catch {
                throw AgentTrainerError.storage("The selected recording folder index is unreadable: \(error.localizedDescription)")
            }
        }

        func appendPackage(_ package: URL, folders: [UUID: RecordingFolder]) throws {
            let unresolvedValues = try package.standardizedFileURL.resourceValues(forKeys: [.isSymbolicLinkKey])
            guard unresolvedValues.isSymbolicLink != true else {
                throw AgentTrainerError.storage("Recording packages cannot be symbolic links.")
            }
            let normalizedPackage = normalized(package)
            let values = try normalizedPackage.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard values.isDirectory == true, values.isSymbolicLink != true,
                  normalizedPackage.pathExtension.localizedCaseInsensitiveCompare("atrrecord") == .orderedSame else { return }
            guard seen.insert(normalizedPackage.path).inserted else { return }
            result.append(RecordingImportSource(directory: normalizedPackage, sourceFolders: folders))
        }

        for selected in selectedURLs {
            let unresolvedValues = try selected.standardizedFileURL.resourceValues(forKeys: [.isSymbolicLinkKey])
            guard unresolvedValues.isSymbolicLink != true else {
                throw AgentTrainerError.storage("Recording imports must be folders, not files or symbolic links.")
            }
            let selected = normalized(selected)
            let values = try selected.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard values.isDirectory == true, values.isSymbolicLink != true else {
                throw AgentTrainerError.storage("Recording imports must be folders, not files or symbolic links.")
            }
            if selected.pathExtension.localizedCaseInsensitiveCompare("atrrecord") == .orderedSame {
                let parent = selected.deletingLastPathComponent()
                let indexRoot = parent.lastPathComponent == "Recordings" ? parent.deletingLastPathComponent() : parent
                try appendPackage(selected, folders: decodedFolders(at: indexRoot))
                continue
            }

            let libraryRecordings = selected.appendingPathComponent("Recordings", isDirectory: true)
            let packageRoot: URL
            let indexRoot: URL
            var packageRootIsDirectory: ObjCBool = false
            if FileManager.default.fileExists(atPath: libraryRecordings.path, isDirectory: &packageRootIsDirectory), packageRootIsDirectory.boolValue {
                packageRoot = libraryRecordings
                indexRoot = selected
            } else {
                packageRoot = selected
                indexRoot = selected
            }
            let folders = try decodedFolders(at: indexRoot)
            let children = try FileManager.default.contentsOfDirectory(
                at: packageRoot,
                includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey],
                options: [.skipsHiddenFiles]
            )
            for child in children { try appendPackage(child, folders: folders) }
        }
        return result
    }

    private func validateRecordingImport(_ source: RecordingImportSource) async throws -> ValidatedRecordingImport {
        let manifestURL = source.directory.appendingPathComponent("manifest.json")
        let manifest: RecordingManifest
        do { manifest = try decoder.decode(RecordingManifest.self, from: Data(contentsOf: manifestURL)) }
        catch { throw AgentTrainerError.storage("A selected recording has an unreadable manifest: \(error.localizedDescription)") }
        guard manifest.isStructurallyValid, manifest.hostStartNanos > 0 else {
            throw AgentTrainerError.storage("\(manifest.name) has an invalid recording manifest or timeline.")
        }
        let artifactNames = [manifest.videoFile, manifest.eventFile] + (manifest.thumbnailFile.map { [$0] } ?? [])
        guard Set(artifactNames).count == artifactNames.count else {
            throw AgentTrainerError.storage("\(manifest.name) reuses an artifact filename and cannot be imported safely.")
        }

        let videoURL = source.directory.appendingPathComponent(manifest.videoFile)
        let eventURL = source.directory.appendingPathComponent(manifest.eventFile)
        guard try regularImportFile(videoURL), try regularImportFile(eventURL) else {
            throw AgentTrainerError.storage("\(manifest.name) is missing its video or synchronized input file.")
        }
        let events: InputEventReader.MappedEvents
        do { events = try InputEventReader.mapped(url: eventURL) }
        catch { throw AgentTrainerError.storage("\(manifest.name) has an invalid synchronized input file: \(error.localizedDescription)") }
        guard events.count == manifest.eventCount else {
            throw AgentTrainerError.storage("\(manifest.name) declares \(manifest.eventCount) inputs but contains \(events.count).")
        }
        if let first = events.first, first.timestampNanos < manifest.hostStartNanos {
            throw AgentTrainerError.storage("\(manifest.name) contains input from before its first video frame.")
        }

        let asset = AVURLAsset(url: videoURL)
        let durationTime: CMTime
        let track: AVAssetTrack
        do {
            durationTime = try await asset.load(.duration)
            guard let loadedTrack = try await asset.loadTracks(withMediaType: .video).first else {
                throw AgentTrainerError.storage("\(manifest.name) does not contain a video track.")
            }
            track = loadedTrack
        } catch {
            throw AgentTrainerError.storage("\(manifest.name) contains video that AgentTrainer cannot decode: \(error.localizedDescription)")
        }
        let videoDuration = CMTimeGetSeconds(durationTime)
        guard videoDuration.isFinite, videoDuration > 0 else {
            throw AgentTrainerError.storage("\(manifest.name) has an empty or invalid video timeline.")
        }
        do {
            let naturalSize = try await track.load(.naturalSize)
            let transform = try await track.load(.preferredTransform)
            let displayed = CGRect(origin: .zero, size: naturalSize).applying(transform).standardized
            let width = Int(abs(displayed.width).rounded())
            let height = Int(abs(displayed.height).rounded())
            guard abs(width - manifest.pixelWidth) <= 1, abs(height - manifest.pixelHeight) <= 1 else {
                throw AgentTrainerError.storage("\(manifest.name)'s manifest dimensions do not match its video.")
            }
        } catch let error as AgentTrainerError { throw error }
        catch {
            throw AgentTrainerError.storage("\(manifest.name)'s video dimensions could not be verified: \(error.localizedDescription)")
        }

        if let thumbnailFile = manifest.thumbnailFile {
            let thumbnail = source.directory.appendingPathComponent(thumbnailFile)
            if FileManager.default.fileExists(atPath: thumbnail.path), try !regularImportFile(thumbnail) {
                throw AgentTrainerError.storage("\(manifest.name) has an unsafe thumbnail artifact.")
            }
        }
        return ValidatedRecordingImport(directory: source.directory, manifest: manifest, sourceFolders: source.sourceFolders)
    }

    private func regularImportFile(_ url: URL) throws -> Bool {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        return values.isRegularFile == true && values.isSymbolicLink != true
    }

    private func copyRecordingArtifacts(for manifest: RecordingManifest, from source: URL, to destination: URL) throws {
        for name in [manifest.videoFile, manifest.eventFile] {
            try FileManager.default.copyItem(at: source.appendingPathComponent(name), to: destination.appendingPathComponent(name))
        }
        if let thumbnailFile = manifest.thumbnailFile {
            let sourceThumbnail = source.appendingPathComponent(thumbnailFile)
            if FileManager.default.fileExists(atPath: sourceThumbnail.path) {
                try FileManager.default.copyItem(at: sourceThumbnail, to: destination.appendingPathComponent(thumbnailFile))
            }
        }
    }

    func saveProfile(_ profile: AIProfile) throws {
        try prepare()
        var profile = profile
        profile.name = profile.name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !profile.name.isEmpty else {
            throw AgentTrainerError.invalidConfiguration("AI names cannot be empty.")
        }
        if profile.isDeletionProtected { profile.deletionProtected = true }
        let directory = profileDirectory(profile.id)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try atomicWrite(try encoder.encode(profile), to: directory.appendingPathComponent("profile.json"))
    }

    func listProfiles() -> [AIProfile] {
        guard let urls = try? FileManager.default.contentsOfDirectory(at: profilesRoot, includingPropertiesForKeys: nil) else { return [] }
        return urls.compactMap { url in
            guard let data = try? Data(contentsOf: url.appendingPathComponent("profile.json")) else { return nil }
            guard var profile = try? decoder.decode(AIProfile.self, from: data) else { return nil }
            if profile.isDeletionProtected { profile.deletionProtected = true }
            // Older profiles predate the list-row summary. Reading only the
            // active manifest is constant work and avoids loading every autosave.
            if profile.trainingProgress == nil,
               let activeVersionID = profile.activeVersionID,
               let active = version(profileID: profile.id, versionID: activeVersionID) {
                profile.trainingProgress = TrainingProgressSummary(
                    globalStep: active.globalStep,
                    epoch: active.epoch ?? 0,
                    updatedAt: active.createdAt,
                    savedBrainCount: versionDirectoryCount(profileID: profile.id),
                    trainingDurationSeconds: active.trainingDurationSeconds,
                    experienceDurationSeconds: active.experienceDurationSeconds,
                    reinforcementFeedbackCount: active.reinforcementFeedbackCount,
                    reinforcementUpdateCount: active.reinforcementUpdateCount,
                    reinforcementNetReward: active.reinforcementNetReward
                )
            }
            // Profiles from earlier builds already have a cheap progress row but
            // no timing fields. Recover wall time from the exact checkpoint and
            // prefer immutable-version timing when available without scanning the
            // full saved-brain list.
            if var progress = profile.trainingProgress {
                let active = profile.activeVersionID.flatMap { version(profileID: profile.id, versionID: $0) }
                let checkpoint = checkpointTiming(profileID: profile.id)
                if progress.trainingDurationSeconds == nil {
                    progress.trainingDurationSeconds = active?.trainingDurationSeconds ?? checkpoint.training
                }
                if progress.experienceDurationSeconds == nil {
                    progress.experienceDurationSeconds = active?.experienceDurationSeconds ?? checkpoint.experience
                }
                profile.trainingProgress = progress
            }
            return profile
        }.sorted { $0.createdAt > $1.createdAt }
    }

    func deleteProfile(_ profile: AIProfile) throws {
        guard !profile.isDeletionProtected else {
            throw AgentTrainerError.storage("This protected AI cannot be deleted.")
        }
        try FileManager.default.removeItem(at: profileDirectory(profile.id))
    }

    func duplicateProfile(_ profile: AIProfile) throws -> AIProfile {
        var copy = profile
        copy.id = UUID()
        copy.name += " Copy"
        copy.createdAt = Date()
        copy.deletionProtected = false
        let source = profileDirectory(profile.id)
        let destination = profileDirectory(copy.id)
        let temporary = profilesRoot.appendingPathComponent(".Duplicate.\(copy.id.uuidString).tmp", isDirectory: true)
        do {
            guard FileManager.default.fileExists(atPath: source.path) else {
                throw AgentTrainerError.storage("The source AI folder is missing, so its brain could not be duplicated.")
            }
            try FileManager.default.copyItem(at: source, to: temporary)
            try atomicWrite(try encoder.encode(copy), to: temporary.appendingPathComponent("profile.json"))
            try FileManager.default.moveItem(at: temporary, to: destination)
            return copy
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            try? FileManager.default.removeItem(at: destination)
            throw error
        }
    }

    /// Explicitly discards learned artifacts after the user confirms a brain-
    /// incompatible configuration change. Crystal V4 remains immutable.
    func resetLearning(for profile: AIProfile) throws -> AIProfile {
        guard !profile.isDeletionProtected else {
            throw AgentTrainerError.storage("This AI is protected. Duplicate it before changing brain architecture or vision settings.")
        }
        var reset = profile
        reset.name = reset.name.trimmingCharacters(in: .whitespacesAndNewlines)
        reset.activeVersionID = nil
        reset.trainingProgress = nil
        let root = profileDirectory(profile.id)
        let profileURL = root.appendingPathComponent("profile.json")
        let previousProfileData = try Data(contentsOf: profileURL)
        let staging = root.appendingPathComponent(".LearningReset.\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        var moved: [(from: URL, to: URL)] = []
        do {
            // Make the persisted profile safe first. A process interruption can
            // then leave only inactive extra artifacts, never an active profile
            // whose selected brain has already disappeared.
            try saveProfile(reset)
            for name in ["Versions", "Checkpoint"] {
                let source = root.appendingPathComponent(name, isDirectory: true)
                guard FileManager.default.fileExists(atPath: source.path) else { continue }
                let destination = staging.appendingPathComponent(name, isDirectory: true)
                try FileManager.default.moveItem(at: source, to: destination)
                moved.append((source, destination))
            }
            try FileManager.default.removeItem(at: staging)
            return reset
        } catch {
            var rollbackError: Error?
            for item in moved.reversed() where FileManager.default.fileExists(atPath: item.to.path) {
                do { try FileManager.default.moveItem(at: item.to, to: item.from) }
                catch { rollbackError = rollbackError ?? error }
            }
            do { try atomicWrite(previousProfileData, to: profileURL) }
            catch { rollbackError = rollbackError ?? error }
            try? FileManager.default.removeItem(at: staging)
            if let rollbackError {
                throw AgentTrainerError.storage("Learning reset failed and its rollback was incomplete: \(rollbackError.localizedDescription)")
            }
            throw error
        }
    }

    func profileDirectory(_ id: UUID) -> URL {
        profilesRoot.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    func versionDirectory(profileID: UUID, versionID: UUID) -> URL {
        profileDirectory(profileID).appendingPathComponent("Versions", isDirectory: true).appendingPathComponent(versionID.uuidString, isDirectory: true)
    }

    func checkpointDirectory(profileID: UUID) -> URL {
        profileDirectory(profileID).appendingPathComponent("Checkpoint", isDirectory: true)
    }

    func saveVersionManifest(_ manifest: ModelVersionManifest, profileID: UUID) throws {
        var manifest = manifest
        manifest.name = manifest.name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !manifest.name.isEmpty,
              manifest.artifactFileNamesAreSafe,
              manifest.trainingDataCoverage?.isValid != false else {
            throw AgentTrainerError.storage("The model version manifest contains an invalid name, artifact filename, or coverage report.")
        }
        let directory = versionDirectory(profileID: profileID, versionID: manifest.id)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try atomicWrite(try encoder.encode(manifest), to: directory.appendingPathComponent("manifest.json"))
    }

    func listVersions(profileID: UUID) -> [ModelVersionManifest] {
        let versions = profileDirectory(profileID).appendingPathComponent("Versions", isDirectory: true)
        guard let urls = try? FileManager.default.contentsOfDirectory(at: versions, includingPropertiesForKeys: nil) else { return [] }
        return urls.compactMap { url in
            guard let data = try? Data(contentsOf: url.appendingPathComponent("manifest.json")),
                  let manifest = try? decoder.decode(ModelVersionManifest.self, from: data),
                  manifest.schemaVersion == ModelContract.schemaVersion,
                  manifest.artifactFileNamesAreSafe,
                  manifest.trainingDataCoverage?.isValid != false,
                  manifest.id.uuidString.caseInsensitiveCompare(url.lastPathComponent) == .orderedSame else { return nil }
            return manifest
        }.sorted { $0.createdAt > $1.createdAt }
    }

    func version(profileID: UUID, versionID: UUID) -> ModelVersionManifest? {
        let url = versionDirectory(profileID: profileID, versionID: versionID).appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: url),
              let manifest = try? decoder.decode(ModelVersionManifest.self, from: data),
              manifest.id == versionID,
              manifest.schemaVersion == ModelContract.schemaVersion,
              manifest.artifactFileNamesAreSafe,
              manifest.trainingDataCoverage?.isValid != false else { return nil }
        return manifest
    }

    /// Publishes one online-learning snapshot as an immutable runnable brain.
    /// The prior supervised checkpoint is removed in the same transaction as
    /// activation: later dataset training must warm-start from these RL weights
    /// instead of restoring an older, otherwise-valid optimizer checkpoint.
    func publishReinforcementSnapshot(
        _ snapshot: ReinforcementSnapshot
    ) throws -> WorkspaceReinforcementPublication? {
        try prepare()
        guard var profile = storedProfile(snapshot.profileID) else {
            throw AgentTrainerError.storage("The AI profile for this RL snapshot no longer exists.")
        }
        let manifest = snapshot.manifest
        let reinforcementSequence = manifest.reinforcementSequence ?? 0
        let feedbackCount = manifest.reinforcementFeedbackCount ?? -1
        let updateCount = manifest.reinforcementUpdateCount ?? -1
        let netReward = manifest.reinforcementNetReward ?? .nan
        let reinforcementSeconds = manifest.reinforcementTrainingSeconds ?? .nan
        guard manifest.schemaVersion == ModelContract.schemaVersion,
              !manifest.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              manifest.globalStep >= 0,
              manifest.trainingLoss.isFinite,
              manifest.artifactFileNamesAreSafe,
              manifest.trainingDataCoverage?.isValid != false,
              manifest.reinforcementSessionID != nil,
              manifest.reinforcementSessionStartedAt != nil,
              reinforcementSequence > 0,
              feedbackCount >= 0,
              updateCount > 0,
              netReward.isFinite,
              reinforcementSeconds.isFinite,
              reinforcementSeconds >= 0,
              manifest.reinforcementOptimizerFile != nil,
              manifest.reinforcementStateFile != nil,
              manifest.optimizerFile == nil,
              manifest.trainingStateFile == nil,
              manifest.randomStateFile == nil else {
            throw AgentTrainerError.storage("The online-learning snapshot has an invalid or ambiguous manifest.")
        }

        let profileRoot = profileDirectory(snapshot.profileID).standardizedFileURL
        let staging = snapshot.stagingDirectory.standardizedFileURL
        guard staging.deletingLastPathComponent() == profileRoot,
              staging.lastPathComponent.hasPrefix(".ReinforcementSnapshot."),
              FileManager.default.fileExists(atPath: staging.path) else {
            throw AgentTrainerError.storage("The online-learning snapshot is outside its managed AI directory.")
        }
        let requiredNames = [
            manifest.weightsFile,
            manifest.reinforcementOptimizerFile!,
            manifest.reinforcementStateFile!
        ]
        guard Set(requiredNames).count == requiredNames.count else {
            throw AgentTrainerError.storage("The online-learning snapshot reuses an artifact filename.")
        }
        for name in requiredNames {
            var isDirectory: ObjCBool = false
            let artifact = staging.appendingPathComponent(name)
            guard FileManager.default.fileExists(
                atPath: artifact.path,
                isDirectory: &isDirectory
            ), !isDirectory.boolValue,
                  ((try? artifact.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) > 0 else {
                throw AgentTrainerError.storage("The online-learning snapshot is incomplete.")
            }
        }

        // Periodic publication is asynchronous. A slower old autosave must
        // never activate after a newer snapshot from the same or a later run.
        if let activeID = profile.activeVersionID,
           let active = version(profileID: profile.id, versionID: activeID),
           let activeStart = active.reinforcementSessionStartedAt,
           let incomingStart = manifest.reinforcementSessionStartedAt {
            let sameSession = active.reinforcementSessionID == manifest.reinforcementSessionID
            let sequenceIsStale = sameSession
                && (active.reinforcementSequence ?? -1) >= (manifest.reinforcementSequence ?? 0)
            if activeStart > incomingStart || sequenceIsStale {
                try? FileManager.default.removeItem(at: staging)
                return nil
            }
        }

        let destination = versionDirectory(profileID: profile.id, versionID: manifest.id)
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            throw AgentTrainerError.storage("The online-learning brain identifier already exists.")
        }
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let profileURL = profileRoot.appendingPathComponent("profile.json")
        let previousProfileData = try Data(contentsOf: profileURL)
        let checkpoint = checkpointDirectory(profileID: profile.id)
        let checkpointBackup = profileRoot.appendingPathComponent(
            ".Checkpoint.before-reinforcement.\(UUID().uuidString)",
            isDirectory: true
        )
        var checkpointWasMoved = false
        var versionWasMoved = false
        do {
            try FileManager.default.moveItem(at: staging, to: destination)
            versionWasMoved = true
            try atomicWrite(
                try encoder.encode(manifest),
                to: destination.appendingPathComponent("manifest.json")
            )
            if FileManager.default.fileExists(atPath: checkpoint.path) {
                try FileManager.default.moveItem(at: checkpoint, to: checkpointBackup)
                checkpointWasMoved = true
            }

            profile.activeVersionID = manifest.id
            profile.trainingProgress = TrainingProgressSummary(
                globalStep: manifest.globalStep,
                epoch: manifest.epoch ?? 0,
                updatedAt: manifest.createdAt,
                savedBrainCount: listVersions(profileID: profile.id).count,
                trainingDurationSeconds: manifest.trainingDurationSeconds,
                experienceDurationSeconds: manifest.experienceDurationSeconds,
                reinforcementFeedbackCount: manifest.reinforcementFeedbackCount,
                reinforcementUpdateCount: manifest.reinforcementUpdateCount,
                reinforcementNetReward: manifest.reinforcementNetReward
            )
            try saveProfile(profile)
        } catch {
            var rollbackError: Error?
            do { try atomicWrite(previousProfileData, to: profileURL) }
            catch { rollbackError = rollbackError ?? error }
            if checkpointWasMoved,
               FileManager.default.fileExists(atPath: checkpointBackup.path),
               !FileManager.default.fileExists(atPath: checkpoint.path) {
                do { try FileManager.default.moveItem(at: checkpointBackup, to: checkpoint) }
                catch { rollbackError = rollbackError ?? error }
            }
            if versionWasMoved, FileManager.default.fileExists(atPath: destination.path) {
                do { try FileManager.default.removeItem(at: destination) }
                catch { rollbackError = rollbackError ?? error }
            }
            if let rollbackError {
                throw AgentTrainerError.storage(
                    "Online-learning publication failed and rollback was incomplete: \(rollbackError.localizedDescription)"
                )
            }
            throw error
        }

        if checkpointWasMoved {
            do { try FileManager.default.removeItem(at: checkpointBackup) }
            catch {
                AppLog.write(
                    .warning,
                    category: "Storage",
                    "An obsolete pre-RL checkpoint remains in a hidden cleanup folder",
                    details: error.localizedDescription
                )
            }
        }
        do {
            _ = try pruneAutosaveVersions(profile: profile, keeping: 10)
        } catch {
            AppLog.write(
                .warning,
                category: "Storage",
                "RL brain published but old autosave cleanup was deferred",
                details: error.localizedDescription
            )
        }
        let publishedProfile = storedProfile(profile.id) ?? profile
        return WorkspaceReinforcementPublication(profile: publishedProfile, version: manifest)
    }

    @discardableResult
    func deleteVersion(profile: AIProfile, versionID: UUID) throws -> AIProfile {
        guard !profile.isDeletionProtected else {
            throw AgentTrainerError.storage("This protected AI's runnable brain cannot be deleted.")
        }
        let source = versionDirectory(profileID: profile.id, versionID: versionID)
        guard FileManager.default.fileExists(atPath: source.path) else {
            throw AgentTrainerError.storage("The selected model version no longer exists.")
        }
        let backup = source.deletingLastPathComponent().appendingPathComponent(".VersionDeletion.\(versionID.uuidString).\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.moveItem(at: source, to: backup)
        do {
            var updated = profile
            if updated.activeVersionID == versionID { updated.activeVersionID = nil }
            if var progress = updated.trainingProgress {
                progress.savedBrainCount = listVersions(profileID: profile.id).count
                updated.trainingProgress = progress
            }
            try saveProfile(updated)
            do { try FileManager.default.removeItem(at: backup) }
            catch {
                AppLog.write(.warning, category: "Storage", "A deleted model version remains in a hidden cleanup folder", details: error.localizedDescription)
            }
            return updated
        } catch {
            if !FileManager.default.fileExists(atPath: source.path) {
                do { try FileManager.default.moveItem(at: backup, to: source) }
                catch {
                    throw AgentTrainerError.storage("Model deletion failed and its rollback was incomplete: \(error.localizedDescription)")
                }
            }
            throw error
        }
    }

    /// Keeps the newest periodic autosaves bounded. Completed/manual brains are
    /// never removed, the active version is always retained, and Crystal V4 is
    /// excluded from automatic cleanup entirely.
    @discardableResult
    func pruneAutosaveVersions(profile: AIProfile, keeping limit: Int = 10) throws -> Int {
        guard !profile.isDeletionProtected else { return 0 }
        let autosaves = listVersions(profileID: profile.id).filter { $0.isAutosave == true }
        guard autosaves.count > max(1, limit) else { return 0 }
        let protectedIDs = Set(autosaves.prefix(max(1, limit)).map(\.id)).union(profile.activeVersionID.map { [$0] } ?? [])
        var removed = 0
        for version in autosaves where !protectedIDs.contains(version.id) {
            let directory = versionDirectory(profileID: profile.id, versionID: version.id)
            guard FileManager.default.fileExists(atPath: directory.path) else { continue }
            try FileManager.default.removeItem(at: directory)
            removed += 1
        }
        if removed > 0, var progress = profile.trainingProgress {
            var updated = profile
            progress.savedBrainCount = versionDirectoryCount(profileID: profile.id)
            updated.trainingProgress = progress
            try saveProfile(updated)
        }
        return removed
    }

    /// Activates the immutable version and its exact checkpoint as one logical
    /// transaction. If profile persistence fails, the previous checkpoint is
    /// restored so the next training session can never resume the wrong brain.
    func activateVersion(profile: AIProfile, versionID: UUID) throws -> WorkspaceVersionActivation {
        guard let version = version(profileID: profile.id, versionID: versionID) else {
            throw AgentTrainerError.storage("The selected model version is missing or has an invalid manifest.")
        }
        let source = versionDirectory(profileID: profile.id, versionID: version.id)
        let sourceWeights = source.appendingPathComponent(version.weightsFile)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: sourceWeights.path, isDirectory: &isDirectory), !isDirectory.boolValue else {
            throw AgentTrainerError.storage("The selected model version is missing its weights.")
        }

        let checkpointMetadata = (version.optimizerFile, version.trainingStateFile, version.randomStateFile)
        let resumable: Bool
        switch checkpointMetadata {
        case (nil, nil, nil):
            resumable = false
        case (.some, .some, _):
            resumable = true
        default:
            throw AgentTrainerError.storage("The selected model version has an incomplete training checkpoint.")
        }

        let destination = checkpointDirectory(profileID: profile.id)
        let parent = destination.deletingLastPathComponent()
        var temporary: URL?
        if resumable, let optimizerFile = version.optimizerFile, let stateFile = version.trainingStateFile {
            let candidate = parent.appendingPathComponent(".Checkpoint.restore.\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: candidate, withIntermediateDirectories: true)
            do {
                try FileManager.default.copyItem(at: sourceWeights, to: candidate.appendingPathComponent("weights.safetensors"))
                try FileManager.default.copyItem(at: source.appendingPathComponent(optimizerFile), to: candidate.appendingPathComponent("optimizer.safetensors"))
                try FileManager.default.copyItem(at: source.appendingPathComponent(stateFile), to: candidate.appendingPathComponent("state.json"))
                try encoder.encode(version.schemaVersion).write(to: candidate.appendingPathComponent("model-schema.json"), options: .atomic)
                if let randomStateFile = version.randomStateFile {
                    try FileManager.default.copyItem(at: source.appendingPathComponent(randomStateFile), to: candidate.appendingPathComponent("random.safetensors"))
                }
                temporary = candidate
            } catch {
                try? FileManager.default.removeItem(at: candidate)
                throw error
            }
        }

        let checkpointExisted = FileManager.default.fileExists(atPath: destination.path)
        let backupName = ".Checkpoint.activation-backup.\(UUID().uuidString)"
        let backup = parent.appendingPathComponent(backupName, isDirectory: true)
        var backupWasCreated = false
        do {
            if checkpointExisted {
                if let temporary {
                    _ = try FileManager.default.replaceItemAt(
                        destination,
                        withItemAt: temporary,
                        backupItemName: backupName,
                        options: .usingNewMetadataOnly
                    )
                } else {
                    try FileManager.default.moveItem(at: destination, to: backup)
                }
                backupWasCreated = FileManager.default.fileExists(atPath: backup.path)
            } else if let temporary {
                try FileManager.default.moveItem(at: temporary, to: destination)
            }

            var updated = profile
            updated.name = updated.name.trimmingCharacters(in: .whitespacesAndNewlines)
            updated.activeVersionID = version.id
            updated.trainingProgress = TrainingProgressSummary(
                globalStep: version.globalStep,
                epoch: version.epoch ?? 0,
                updatedAt: version.createdAt,
                savedBrainCount: listVersions(profileID: profile.id).count,
                trainingDurationSeconds: version.trainingDurationSeconds,
                experienceDurationSeconds: version.experienceDurationSeconds,
                reinforcementFeedbackCount: version.reinforcementFeedbackCount,
                reinforcementUpdateCount: version.reinforcementUpdateCount,
                reinforcementNetReward: version.reinforcementNetReward
            )
            try saveProfile(updated)
            if backupWasCreated {
                do { try FileManager.default.removeItem(at: backup) }
                catch {
                    AppLog.write(.warning, category: "Storage", "An old checkpoint remains in a hidden cleanup folder", details: error.localizedDescription)
                }
            }
            return WorkspaceVersionActivation(profile: updated, version: version, checkpointIsResumable: resumable)
        } catch {
            var rollbackError: Error?
            if backupWasCreated, FileManager.default.fileExists(atPath: backup.path) {
                if FileManager.default.fileExists(atPath: destination.path) {
                    do { try FileManager.default.removeItem(at: destination) }
                    catch { rollbackError = rollbackError ?? error }
                }
                if !FileManager.default.fileExists(atPath: destination.path) {
                    do { try FileManager.default.moveItem(at: backup, to: destination) }
                    catch { rollbackError = rollbackError ?? error }
                }
            } else if !checkpointExisted, FileManager.default.fileExists(atPath: destination.path) {
                do { try FileManager.default.removeItem(at: destination) }
                catch { rollbackError = rollbackError ?? error }
            }
            if let temporary, FileManager.default.fileExists(atPath: temporary.path) {
                try? FileManager.default.removeItem(at: temporary)
            }
            if let rollbackError {
                throw AgentTrainerError.storage("Model activation failed and its checkpoint rollback was incomplete: \(rollbackError.localizedDescription)")
            }
            throw error
        }
    }

    func storageUsage() -> WorkspaceStorageUsage {
        let trainingBytes = WorkspaceDataKind.trainingData.managedNames.reduce(Int64(0)) {
            $0 + allocatedBytes(at: trainingDataRoot.appendingPathComponent($1))
        }
        let modelBytes = WorkspaceDataKind.models.managedNames.reduce(Int64(0)) {
            $0 + allocatedBytes(at: modelsRoot.appendingPathComponent($1))
        }
        let candidates = [root, trainingDataRoot, modelsRoot]
            .map(normalized)
            .reduce(into: [URL]()) { result, url in if !result.contains(url) { result.append(url) } }
            .sorted { $0.pathComponents.count < $1.pathComponents.count }
        var rootsToCount: [URL] = []
        for candidate in candidates where !rootsToCount.contains(where: { isSameOrDescendant(candidate, of: $0) }) {
            rootsToCount.append(candidate)
        }
        let total = rootsToCount.reduce(Int64(0)) { $0 + allocatedBytes(at: $1) }
        return WorkspaceStorageUsage(totalBytes: total, trainingDataBytes: trainingBytes, modelBytes: modelBytes)
    }

    func storageBytes() -> Int64 {
        storageUsage().totalBytes
    }

    private func allocatedBytes(at location: URL) -> Int64 {
        let keys: Set<URLResourceKey> = [.fileAllocatedSizeKey, .totalFileAllocatedSizeKey, .fileSizeKey]
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: location.path, isDirectory: &isDirectory) else { return 0 }
        if !isDirectory.boolValue {
            let values = try? location.resourceValues(forKeys: keys)
            return Int64(values?.totalFileAllocatedSize ?? values?.fileAllocatedSize ?? values?.fileSize ?? 0)
        }
        guard let enumerator = FileManager.default.enumerator(at: location, includingPropertiesForKeys: Array(keys)) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            if let values = try? file.resourceValues(forKeys: keys) {
                let size = values.totalFileAllocatedSize ?? values.fileAllocatedSize ?? values.fileSize ?? 0
                total += Int64(size)
            }
        }
        return total
    }

    func clearCaches() throws {
        if FileManager.default.fileExists(atPath: cachesRoot.path) { try FileManager.default.removeItem(at: cachesRoot) }
        try FileManager.default.createDirectory(at: cachesRoot, withIntermediateDirectories: true)
    }

    @discardableResult
    func removeObsoleteCaches(currentSchema: Int) throws -> Int {
        try prepare()
        guard let directories = try? FileManager.default.contentsOfDirectory(at: cachesRoot, includingPropertiesForKeys: nil) else { return 0 }
        var removed = 0
        for directory in directories where directory.pathExtension == "atrcache" {
            let manifestURL = directory.appendingPathComponent("manifest.json")
            let schema: Int? = (try? Data(contentsOf: manifestURL)).flatMap { data in
                guard let object = try? JSONSerialization.jsonObject(with: data),
                      let manifest = object as? [String: Any] else { return nil }
                return manifest["schemaVersion"] as? Int
            }
            guard schema != currentSchema else { continue }
            try FileManager.default.removeItem(at: directory)
            removed += 1
        }
        return removed
    }

    @discardableResult
    func pruneAllAutosaves(keeping limit: Int = 10) throws -> Int {
        var removed = 0
        for profile in listProfiles() {
            removed += try pruneAutosaveVersions(profile: profile, keeping: limit)
        }
        return removed
    }

    private func versionDirectoryCount(profileID: UUID) -> Int {
        listVersions(profileID: profileID).count
    }

    private func storedProfile(_ id: UUID) -> AIProfile? {
        let url = profileDirectory(id).appendingPathComponent("profile.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? decoder.decode(AIProfile.self, from: data)
    }

    private func checkpointTiming(profileID: UUID) -> (training: Double?, experience: Double?) {
        let stateURL = checkpointDirectory(profileID: profileID).appendingPathComponent("state.json")
        guard let data = try? Data(contentsOf: stateURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return (nil, nil)
        }
        return (object["elapsed"] as? Double, object["experienceSeconds"] as? Double)
    }

    private func location(for kind: WorkspaceDataKind) -> URL {
        switch kind {
        case .trainingData: trainingDataRoot
        case .models: modelsRoot
        }
    }

    private func setLocation(_ destination: URL, for kind: WorkspaceDataKind) {
        switch kind {
        case .trainingData: trainingDataRoot = destination
        case .models: modelsRoot = destination
        }
        guard persistsLocations else { return }
        let key = kind == .trainingData ? Self.trainingDataRootKey : Self.modelsRootKey
        if destination == root { UserDefaults.standard.removeObject(forKey: key) }
        else { UserDefaults.standard.set(destination.path, forKey: key) }
    }

    private func commitLocation(_ destination: URL, for kind: WorkspaceDataKind) throws {
        let previous = location(for: kind)
        setLocation(destination, for: kind)
        do {
            try prepare()
        } catch {
            setLocation(previous, for: kind)
            throw error
        }
    }

    private func normalized(_ url: URL) -> URL {
        url.standardizedFileURL.resolvingSymlinksInPath()
    }

    private func validateDestination(_ destination: URL, for kind: WorkspaceDataKind) throws {
        guard destination.isFileURL else {
            throw AgentTrainerError.storage("Storage locations must be local or mounted file-system folders.")
        }
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: destination.path, isDirectory: &isDirectory), !isDirectory.boolValue {
            throw AgentTrainerError.storage("Choose a folder, not a file, for \(kind.rawValue.lowercased()).")
        }
        guard !destination.pathComponents.contains(where: { $0.lowercased().hasSuffix(".app") }) else {
            throw AgentTrainerError.storage("Store \(kind.rawValue.lowercased()) outside application bundles so an app update can never replace it.")
        }
        let current = location(for: kind)
        if destination != current, pathsOverlap(destination, current) {
            throw AgentTrainerError.storage("The new \(kind.rawValue.lowercased()) folder cannot be inside the current folder, or contain it.")
        }
        let other = location(for: kind == .trainingData ? .models : .trainingData)
        if destination != other, pathsOverlap(destination, other) {
            throw AgentTrainerError.storage("Training data and AI model locations may be the same folder, but one cannot be nested inside the other.")
        }
    }

    private static func isSafeLeafName(_ value: String) -> Bool {
        !value.isEmpty && value != "." && value != ".." && !value.contains("/") && !value.contains("\0")
    }

    private func ensureLocationIsAvailable(_ location: URL, name: String) throws {
        let components = location.standardizedFileURL.pathComponents
        if components.count > 2, components[1] == "Volumes" {
            let volume = URL(fileURLWithPath: "/Volumes", isDirectory: true).appendingPathComponent(components[2], isDirectory: true)
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: volume.path, isDirectory: &isDirectory), isDirectory.boolValue else {
                throw AgentTrainerError.storage("The disk containing \(name.lowercased()) is not connected. Reconnect it or choose another location in Settings.")
            }
        }
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: location.path, isDirectory: &isDirectory), !isDirectory.boolValue {
            throw AgentTrainerError.storage("The saved \(name.lowercased()) location is no longer a folder.")
        }
    }

    private func verifyWritable(_ directory: URL, name: String) throws {
        let probe = directory.appendingPathComponent(".AgentTrainer-write-test-\(UUID().uuidString)")
        do {
            try Data().write(to: probe, options: .atomic)
            try FileManager.default.removeItem(at: probe)
        } catch {
            try? FileManager.default.removeItem(at: probe)
            throw AgentTrainerError.storage("The selected \(name.lowercased()) folder is not writable.")
        }
    }

    private func pathsOverlap(_ lhs: URL, _ rhs: URL) -> Bool {
        isSameOrDescendant(lhs, of: rhs) || isSameOrDescendant(rhs, of: lhs)
    }

    private func isSameOrDescendant(_ candidate: URL, of ancestor: URL) -> Bool {
        let candidatePath = normalized(candidate).path
        let ancestorPath = normalized(ancestor).path
        return candidatePath == ancestorPath || candidatePath.hasPrefix(ancestorPath.hasSuffix("/") ? ancestorPath : ancestorPath + "/")
    }

    private func logicalBytes(at location: URL) -> Int64 {
        contentSummary(at: location).bytes
    }

    private func contentSummary(at location: URL) -> (files: Int, bytes: Int64) {
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .fileSizeKey]
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: location.path, isDirectory: &isDirectory) else { return (0, 0) }
        if !isDirectory.boolValue {
            return (1, Int64((try? location.resourceValues(forKeys: keys).fileSize) ?? 0))
        }
        guard let enumerator = FileManager.default.enumerator(at: location, includingPropertiesForKeys: Array(keys)) else { return (0, 0) }
        var result = (files: 0, bytes: Int64(0))
        for case let file as URL in enumerator {
            guard let values = try? file.resourceValues(forKeys: keys), values.isRegularFile == true else { continue }
            result.files += 1
            result.bytes += Int64(values.fileSize ?? 0)
        }
        return result
    }

    private func atomicWrite(_ data: Data, to destination: URL) throws {
        let temporary = destination.deletingLastPathComponent().appendingPathComponent(".\(destination.lastPathComponent).\(UUID().uuidString).tmp")
        try data.write(to: temporary, options: .atomic)
        do {
            if FileManager.default.fileExists(atPath: destination.path) {
                _ = try FileManager.default.replaceItemAt(destination, withItemAt: temporary, backupItemName: nil, options: .usingNewMetadataOnly)
            } else {
                try FileManager.default.moveItem(at: temporary, to: destination)
            }
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
    }
}
