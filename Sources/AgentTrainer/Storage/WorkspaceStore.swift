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
        let auditMarker = modelsRoot.appendingPathComponent("model-artifact-audit-1.8.2.json")
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
                   manifest.id.uuidString.caseInsensitiveCompare(item.lastPathComponent) == .orderedSame {
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
                profile.training.historyLength = min(32, max(0, profile.training.historyLength))
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

    /// Repairs legacy manifests before the strict library scan runs. Policy
    /// v4 added defensive validation, but filtering first made an older
    /// recording with a stale duration/trim disappear from the UI before the
    /// existing clock repair could discover it. This pass enumerates recording
    /// directories directly and changes only invalid, decodable manifests.
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
        guard let urls = try? FileManager.default.contentsOfDirectory(at: recordingsRoot, includingPropertiesForKeys: nil) else { return [] }
        return urls.compactMap { directory in
            let manifestURL = directory.appendingPathComponent("manifest.json")
            guard let data = try? Data(contentsOf: manifestURL),
                  let manifest = try? decoder.decode(RecordingManifest.self, from: data),
                  manifest.isStructurallyValid else { return nil }
            return RecordingItem(manifest: manifest, directory: directory)
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

    func saveRecordingFolder(_ folder: RecordingFolder) throws {
        guard !folder.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw AgentTrainerError.invalidConfiguration("Folder names cannot be empty.") }
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

    func deleteRecordingFolder(_ folder: RecordingFolder, includingRecordings: Bool) throws {
        var remaining = listRecordingFolders().filter { $0.id != folder.id }
        if includingRecordings {
            for recording in listRecordings() where recording.manifest.folderID == folder.id {
                try? FileManager.default.removeItem(at: recording.directory)
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

    func saveProfile(_ profile: AIProfile) throws {
        try prepare()
        var profile = profile
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
                    experienceDurationSeconds: active.experienceDurationSeconds
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
        reset.activeVersionID = nil
        reset.trainingProgress = nil
        let root = profileDirectory(profile.id)
        let staging = root.appendingPathComponent(".LearningReset.\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        var moved: [(from: URL, to: URL)] = []
        do {
            for name in ["Versions", "Checkpoint"] {
                let source = root.appendingPathComponent(name, isDirectory: true)
                guard FileManager.default.fileExists(atPath: source.path) else { continue }
                let destination = staging.appendingPathComponent(name, isDirectory: true)
                try FileManager.default.moveItem(at: source, to: destination)
                moved.append((source, destination))
            }
            try saveProfile(reset)
            try FileManager.default.removeItem(at: staging)
            return reset
        } catch {
            for item in moved.reversed() where FileManager.default.fileExists(atPath: item.to.path) {
                try? FileManager.default.moveItem(at: item.to, to: item.from)
            }
            try? FileManager.default.removeItem(at: staging)
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
        let directory = versionDirectory(profileID: profileID, versionID: manifest.id)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try atomicWrite(try encoder.encode(manifest), to: directory.appendingPathComponent("manifest.json"))
    }

    func listVersions(profileID: UUID) -> [ModelVersionManifest] {
        let versions = profileDirectory(profileID).appendingPathComponent("Versions", isDirectory: true)
        guard let urls = try? FileManager.default.contentsOfDirectory(at: versions, includingPropertiesForKeys: nil) else { return [] }
        return urls.compactMap { url in
            guard let data = try? Data(contentsOf: url.appendingPathComponent("manifest.json")) else { return nil }
            return try? decoder.decode(ModelVersionManifest.self, from: data)
        }.sorted { $0.createdAt > $1.createdAt }
    }

    func version(profileID: UUID, versionID: UUID) -> ModelVersionManifest? {
        let url = versionDirectory(profileID: profileID, versionID: versionID).appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? decoder.decode(ModelVersionManifest.self, from: data)
    }

    func deleteVersion(profile: AIProfile, versionID: UUID) throws {
        guard !profile.isDeletionProtected else {
            throw AgentTrainerError.storage("This protected AI's runnable brain cannot be deleted.")
        }
        try FileManager.default.removeItem(at: versionDirectory(profileID: profile.id, versionID: versionID))
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

    func restoreVersionAsCheckpoint(profileID: UUID, version: ModelVersionManifest) throws -> Bool {
        let destination = checkpointDirectory(profileID: profileID)
        guard let optimizerFile = version.optimizerFile, let stateFile = version.trainingStateFile else {
            // Explicitly activating a weights-only/best brain means future
            // training should fine-tune that brain with a fresh optimizer, not
            // silently resume an unrelated newer checkpoint left on disk.
            if FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.removeItem(at: destination) }
            return false
        }
        let source = versionDirectory(profileID: profileID, versionID: version.id)
        let temporary = destination.deletingLastPathComponent().appendingPathComponent(".Checkpoint.restore.\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        do {
            try FileManager.default.copyItem(at: source.appendingPathComponent(version.weightsFile), to: temporary.appendingPathComponent("weights.safetensors"))
            try FileManager.default.copyItem(at: source.appendingPathComponent(optimizerFile), to: temporary.appendingPathComponent("optimizer.safetensors"))
            try FileManager.default.copyItem(at: source.appendingPathComponent(stateFile), to: temporary.appendingPathComponent("state.json"))
            try encoder.encode(version.schemaVersion).write(to: temporary.appendingPathComponent("model-schema.json"), options: .atomic)
            if let randomStateFile = version.randomStateFile {
                try FileManager.default.copyItem(at: source.appendingPathComponent(randomStateFile), to: temporary.appendingPathComponent("random.safetensors"))
            }
            if FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.removeItem(at: destination) }
            try FileManager.default.moveItem(at: temporary, to: destination)
            return true
        } catch {
            try? FileManager.default.removeItem(at: temporary)
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
            try? FileManager.default.removeItem(at: directory)
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
        let versions = profileDirectory(profileID).appendingPathComponent("Versions", isDirectory: true)
        return (try? FileManager.default.contentsOfDirectory(at: versions, includingPropertiesForKeys: nil).count) ?? 0
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
