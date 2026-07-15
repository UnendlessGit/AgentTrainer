import CryptoKit
import Foundation

struct AppSemanticVersion: Comparable, CustomStringConvertible, Sendable {
    let components: [Int]
    let prerelease: [String]

    init?(_ value: String) {
        var normalized = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if normalized.first == "v" || normalized.first == "V" { normalized.removeFirst() }
        normalized = normalized.split(separator: "+", maxSplits: 1).first.map(String.init) ?? normalized
        let pieces = normalized.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
        let numericPieces = pieces[0].split(separator: ".", omittingEmptySubsequences: false)
        guard !numericPieces.isEmpty,
              numericPieces.allSatisfy({ !$0.isEmpty && $0.allSatisfy(\.isNumber) }),
              numericPieces.count <= 4 else { return nil }
        components = numericPieces.compactMap { Int($0) }
        guard components.count == numericPieces.count else { return nil }
        prerelease = pieces.count == 2 ? pieces[1].split(separator: ".").map(String.init) : []
    }

    var description: String {
        components.map(String.init).joined(separator: ".") + (prerelease.isEmpty ? "" : "-" + prerelease.joined(separator: "."))
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        let count = max(lhs.components.count, rhs.components.count)
        for index in 0..<count {
            let left = index < lhs.components.count ? lhs.components[index] : 0
            let right = index < rhs.components.count ? rhs.components[index] : 0
            if left != right { return false }
        }
        return lhs.prerelease == rhs.prerelease
    }

    static func < (lhs: Self, rhs: Self) -> Bool {
        let count = max(lhs.components.count, rhs.components.count)
        for index in 0..<count {
            let left = index < lhs.components.count ? lhs.components[index] : 0
            let right = index < rhs.components.count ? rhs.components[index] : 0
            if left != right { return left < right }
        }
        if lhs.prerelease.isEmpty != rhs.prerelease.isEmpty { return !lhs.prerelease.isEmpty }
        for index in 0..<max(lhs.prerelease.count, rhs.prerelease.count) {
            guard index < lhs.prerelease.count else { return true }
            guard index < rhs.prerelease.count else { return false }
            let left = lhs.prerelease[index]
            let right = rhs.prerelease[index]
            if left == right { continue }
            if let leftNumber = Int(left), let rightNumber = Int(right) { return leftNumber < rightNumber }
            if Int(left) != nil { return true }
            if Int(right) != nil { return false }
            return left.localizedStandardCompare(right) == .orderedAscending
        }
        return false
    }
}

struct GitHubRelease: Decodable, Sendable {
    struct Asset: Decodable, Sendable {
        let name: String
        let browserDownloadURL: URL
        let size: Int

        enum CodingKeys: String, CodingKey {
            case name, size
            case browserDownloadURL = "browser_download_url"
        }
    }

    let tagName: String
    let name: String?
    let body: String?
    let htmlURL: URL
    let draft: Bool
    let prerelease: Bool
    let assets: [Asset]

    enum CodingKeys: String, CodingKey {
        case name, body, draft, prerelease, assets
        case tagName = "tag_name"
        case htmlURL = "html_url"
    }

    var version: AppSemanticVersion? { AppSemanticVersion(tagName) }

    var installerAsset: Asset? {
        assets
            .filter { $0.name.lowercased().hasSuffix(".dmg") }
            .sorted {
                let leftCompact = $0.name.localizedCaseInsensitiveContains("compact")
                let rightCompact = $1.name.localizedCaseInsensitiveContains("compact")
                if leftCompact != rightCompact { return !leftCompact }
                return $0.name.localizedStandardCompare($1.name) == .orderedAscending
            }
            .first
    }

    var checksumAsset: Asset? {
        assets.first { $0.name.caseInsensitiveCompare("SHA256SUMS.txt") == .orderedSame }
    }
}

struct GitHubUpdateConfiguration: Sendable {
    let owner: String
    let repository: String
    let currentVersion: AppSemanticVersion

    init?(bundle: Bundle = .main) {
        guard let owner = bundle.object(forInfoDictionaryKey: "ATGitHubOwner") as? String,
              let repository = bundle.object(forInfoDictionaryKey: "ATGitHubRepository") as? String,
              let versionString = bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String,
              let currentVersion = AppSemanticVersion(versionString),
              Self.isSafeRepositoryComponent(owner), Self.isSafeRepositoryComponent(repository) else { return nil }
        self.owner = owner
        self.repository = repository
        self.currentVersion = currentVersion
    }

    init?(owner: String, repository: String, currentVersion: String) {
        guard Self.isSafeRepositoryComponent(owner), Self.isSafeRepositoryComponent(repository),
              let version = AppSemanticVersion(currentVersion) else { return nil }
        self.owner = owner
        self.repository = repository
        self.currentVersion = version
    }

    private static func isSafeRepositoryComponent(_ value: String) -> Bool {
        !value.isEmpty && value.count <= 100 && value.allSatisfy { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" || $0 == "." }
    }
}

struct AppUpdateProgress: Equatable, Sendable {
    let title: String
    let detail: String
    let fraction: Double

    init(title: String = "Updating AgentTrainer", detail: String, fraction: Double) {
        self.title = title
        self.detail = detail
        self.fraction = min(max(fraction.isFinite ? fraction : 0, 0), 1)
    }
}

struct PreparedAppUpdate: Sendable {
    let stagedApplicationURL: URL
    let workingDirectoryURL: URL
    let targetApplicationURL: URL
    let version: AppSemanticVersion
}

enum GitHubUpdateError: LocalizedError {
    case invalidEndpoint
    case invalidResponse
    case requestFailed(Int)
    case releaseHasNoVersion(String)
    case missingInstaller
    case missingChecksum
    case unsafeDownloadURL
    case downloadTooLarge
    case checksumEntryMissing(String)
    case checksumMismatch
    case invalidDiskImage
    case invalidApplication(String)
    case signatureMismatch
    case applicationCannotBeReplaced(String)
    case commandFailed(String, Int32, String)
    case installerHelperFailed

    var errorDescription: String? {
        switch self {
        case .invalidEndpoint: "The configured GitHub release endpoint is invalid."
        case .invalidResponse: "GitHub returned an invalid update response."
        case .requestFailed(let status): "GitHub returned HTTP status \(status)."
        case .releaseHasNoVersion(let tag): "The latest GitHub release tag (\(tag)) is not a supported version."
        case .missingInstaller: "The GitHub release does not contain an AgentTrainer DMG."
        case .missingChecksum: "The GitHub release does not contain SHA256SUMS.txt."
        case .unsafeDownloadURL: "GitHub returned an unexpected download address."
        case .downloadTooLarge: "The update download is unexpectedly large."
        case .checksumEntryMissing(let name): "SHA256SUMS.txt does not contain \(name)."
        case .checksumMismatch: "The downloaded update failed SHA-256 verification."
        case .invalidDiskImage: "The update disk image does not contain a valid AgentTrainer app."
        case .invalidApplication(let reason): "The downloaded AgentTrainer app is invalid: \(reason)"
        case .signatureMismatch: "The update was signed by a different identity and was not installed."
        case .applicationCannotBeReplaced(let path): "AgentTrainer cannot update the app at \(path). Move it to a writable Applications folder and try again."
        case .commandFailed(let command, let status, let details): "\(command) failed with status \(status). \(details)"
        case .installerHelperFailed: "The update installer could not be started."
        }
    }
}

private final class UpdateDownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let progress: @Sendable (Double) -> Void

    init(progress: @escaping @Sendable (Double) -> Void) {
        self.progress = progress
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        progress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
}

struct GitHubReleaseUpdater: Sendable {
    let configuration: GitHubUpdateConfiguration
    var session: URLSession = .shared

    func latestRelease() async throws -> GitHubRelease {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "api.github.com"
        components.path = "/repos/\(configuration.owner)/\(configuration.repository)/releases/latest"
        guard let url = components.url else { throw GitHubUpdateError.invalidEndpoint }
        var request = URLRequest(url: url)
        request.timeoutInterval = 15
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("AgentTrainer/\(configuration.currentVersion)", forHTTPHeaderField: "User-Agent")
        request.setValue("2022-11-28", forHTTPHeaderField: "X-GitHub-Api-Version")
        let data = try await validatedData(for: request, permittedInitialHosts: ["api.github.com"], maximumBytes: 2_000_000)
        let release = try JSONDecoder().decode(GitHubRelease.self, from: data)
        guard !release.draft, !release.prerelease else { throw GitHubUpdateError.invalidResponse }
        guard release.version != nil else { throw GitHubUpdateError.releaseHasNoVersion(release.tagName) }
        return release
    }

    func updateIsAvailable(_ release: GitHubRelease) -> Bool {
        guard let version = release.version else { return false }
        return version > configuration.currentVersion
    }

    func prepareUpdate(
        for release: GitHubRelease,
        targetApplicationURL: URL,
        progress: @escaping @Sendable (Double, String) -> Void
    ) async throws -> PreparedAppUpdate {
        guard let installer = release.installerAsset else { throw GitHubUpdateError.missingInstaller }
        guard let checksum = release.checksumAsset else { throw GitHubUpdateError.missingChecksum }
        guard let releaseVersion = release.version else { throw GitHubUpdateError.releaseHasNoVersion(release.tagName) }
        guard installer.size > 0, installer.size <= 1_500_000_000,
              checksum.size > 0, checksum.size <= 2_000_000 else { throw GitHubUpdateError.downloadTooLarge }

        let targetParent = targetApplicationURL.deletingLastPathComponent()
        guard targetApplicationURL.pathExtension.lowercased() == "app",
              FileManager.default.fileExists(atPath: targetApplicationURL.path),
              FileManager.default.isWritableFile(atPath: targetParent.path) else {
            throw GitHubUpdateError.applicationCannotBeReplaced(targetApplicationURL.path)
        }
        let values = try? targetApplicationURL.resourceValues(forKeys: [.volumeIsReadOnlyKey])
        guard values?.volumeIsReadOnly != true else {
            throw GitHubUpdateError.applicationCannotBeReplaced(targetApplicationURL.path)
        }

        let tag = Self.safePathComponent(release.tagName)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AgentTrainer Updates", isDirectory: true)
            .appendingPathComponent(tag, isDirectory: true)
        try? FileManager.default.removeItem(at: directory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        progress(0.02, "Downloading release checksum…")
        let sums = try await downloadData(asset: checksum, maximumBytes: 2_000_000)
        guard let expected = Self.expectedChecksum(for: installer.name, in: sums) else {
            throw GitHubUpdateError.checksumEntryMissing(installer.name)
        }

        let diskImage = directory.appendingPathComponent(installer.name)
        progress(0.05, "Downloading \(installer.name)…")
        try await downloadFile(asset: installer, destination: diskImage, maximumBytes: 1_500_000_000) { fraction in
            progress(0.05 + fraction * 0.67, "Downloading update… \(Int((fraction * 100).rounded()))%")
        }
        progress(0.74, "Verifying SHA-256 checksum…")
        let actual = try Self.sha256(of: diskImage)
        guard actual.caseInsensitiveCompare(expected) == .orderedSame else { throw GitHubUpdateError.checksumMismatch }

        progress(0.79, "Opening verified disk image…")
        let mountResult = try await Self.runProcess("/usr/bin/hdiutil", ["attach", "-nobrowse", "-readonly", "-plist", diskImage.path])
        let mountPoint = try Self.mountPoint(fromHdiutilPlist: mountResult.stdout)
        do {
            let sourceApplication = mountPoint.appendingPathComponent("AgentTrainer.app", isDirectory: true)
            guard FileManager.default.fileExists(atPath: sourceApplication.path) else { throw GitHubUpdateError.invalidDiskImage }
            progress(0.84, "Validating app identity and signature…")
            try await Self.validateApplication(
                at: sourceApplication,
                expectedVersion: releaseVersion,
                replacing: targetApplicationURL
            )

            let stagedApplication = directory.appendingPathComponent("Staged AgentTrainer.app", isDirectory: true)
            try? FileManager.default.removeItem(at: stagedApplication)
            progress(0.89, "Staging the new application…")
            _ = try await Self.runProcess("/usr/bin/ditto", ["--rsrc", "--extattr", sourceApplication.path, stagedApplication.path])
            try await Self.validateApplication(
                at: stagedApplication,
                expectedVersion: releaseVersion,
                replacing: targetApplicationURL
            )
            _ = try? await Self.runProcess("/usr/bin/hdiutil", ["detach", mountPoint.path])
            progress(0.94, "Update verified and ready to install…")
            return PreparedAppUpdate(
                stagedApplicationURL: stagedApplication,
                workingDirectoryURL: directory,
                targetApplicationURL: targetApplicationURL,
                version: releaseVersion
            )
        } catch {
            _ = try? await Self.runProcess("/usr/bin/hdiutil", ["detach", mountPoint.path])
            throw error
        }
    }

    static func expectedChecksum(for filename: String, in data: Data) -> String? {
        guard let text = String(data: data, encoding: .utf8) else { return nil }
        for line in text.split(whereSeparator: \.isNewline) {
            let fields = line.split(whereSeparator: \.isWhitespace)
            guard fields.count >= 2 else { continue }
            let hash = String(fields[0])
            let listedName = String(fields.last!).trimmingCharacters(in: CharacterSet(charactersIn: "*"))
            if listedName == filename, hash.count == 64, hash.allSatisfy(\.isHexDigit) { return hash.lowercased() }
        }
        return nil
    }

    static func mountPoint(fromHdiutilPlist data: Data) throws -> URL {
        guard let plist = try PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any],
              let entities = plist["system-entities"] as? [[String: Any]],
              let path = entities.compactMap({ $0["mount-point"] as? String }).first,
              !path.isEmpty else { throw GitHubUpdateError.invalidDiskImage }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    private func downloadData(asset: GitHubRelease.Asset, maximumBytes: Int) async throws -> Data {
        guard asset.browserDownloadURL.scheme == "https", asset.browserDownloadURL.host?.lowercased() == "github.com" else {
            throw GitHubUpdateError.unsafeDownloadURL
        }
        var request = URLRequest(url: asset.browserDownloadURL)
        request.timeoutInterval = 120
        request.setValue("application/octet-stream", forHTTPHeaderField: "Accept")
        request.setValue("AgentTrainer/\(configuration.currentVersion)", forHTTPHeaderField: "User-Agent")
        return try await validatedData(for: request, permittedInitialHosts: ["github.com"], maximumBytes: maximumBytes)
    }

    private func downloadFile(
        asset: GitHubRelease.Asset,
        destination: URL,
        maximumBytes: Int,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws {
        guard asset.browserDownloadURL.scheme == "https", asset.browserDownloadURL.host?.lowercased() == "github.com" else {
            throw GitHubUpdateError.unsafeDownloadURL
        }
        var request = URLRequest(url: asset.browserDownloadURL)
        request.timeoutInterval = 180
        request.setValue("application/octet-stream", forHTTPHeaderField: "Accept")
        request.setValue("AgentTrainer/\(configuration.currentVersion)", forHTTPHeaderField: "User-Agent")
        let delegate = UpdateDownloadProgressDelegate(progress: progress)
        let (temporary, response) = try await session.download(for: request, delegate: delegate)
        try Self.validate(response: response, dataSize: nil, maximumBytes: maximumBytes, permittedInitialHosts: ["github.com"], request: request)
        let fileSize = try temporary.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        guard fileSize > 0, fileSize <= maximumBytes else { throw GitHubUpdateError.downloadTooLarge }
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: temporary, to: destination)
    }

    private func validatedData(for request: URLRequest, permittedInitialHosts: Set<String>, maximumBytes: Int) async throws -> Data {
        guard let host = request.url?.host?.lowercased(), permittedInitialHosts.contains(host) else { throw GitHubUpdateError.unsafeDownloadURL }
        let (data, response) = try await session.data(for: request)
        try Self.validate(response: response, dataSize: data.count, maximumBytes: maximumBytes, permittedInitialHosts: permittedInitialHosts, request: request)
        return data
    }

    private static func validate(
        response: URLResponse,
        dataSize: Int?,
        maximumBytes: Int,
        permittedInitialHosts: Set<String>,
        request: URLRequest
    ) throws {
        guard let initialHost = request.url?.host?.lowercased(), permittedInitialHosts.contains(initialHost) else {
            throw GitHubUpdateError.unsafeDownloadURL
        }
        guard let http = response as? HTTPURLResponse else { throw GitHubUpdateError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else { throw GitHubUpdateError.requestFailed(http.statusCode) }
        if let dataSize, dataSize > maximumBytes { throw GitHubUpdateError.downloadTooLarge }
        guard let finalHost = http.url?.host?.lowercased(),
              finalHost == "github.com" || finalHost == "api.github.com" || finalHost.hasSuffix(".githubusercontent.com") else {
            throw GitHubUpdateError.unsafeDownloadURL
        }
    }

    private static func safePathComponent(_ value: String) -> String {
        String(value.map { $0.isLetter || $0.isNumber || $0 == "." || $0 == "-" ? $0 : "-" })
    }

    private static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty { hasher.update(data: data) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func validateApplication(
        at applicationURL: URL,
        expectedVersion: AppSemanticVersion,
        replacing currentApplicationURL: URL
    ) async throws {
        _ = try await runProcess("/usr/bin/codesign", ["--verify", "--deep", "--strict", "--verbose=2", applicationURL.path])
        guard let candidate = Bundle(url: applicationURL), let current = Bundle(url: currentApplicationURL) else {
            throw GitHubUpdateError.invalidApplication("bundle metadata could not be read")
        }
        guard candidate.bundleIdentifier == current.bundleIdentifier else {
            throw GitHubUpdateError.invalidApplication("bundle identifier does not match")
        }
        guard let versionString = candidate.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String,
              let candidateVersion = AppSemanticVersion(versionString), candidateVersion == expectedVersion else {
            throw GitHubUpdateError.invalidApplication("version does not match the GitHub release")
        }
        let currentRequirement = try await designatedRequirement(for: currentApplicationURL)
        let candidateRequirement = try await designatedRequirement(for: applicationURL)
        guard currentRequirement == candidateRequirement else { throw GitHubUpdateError.signatureMismatch }
    }

    private static func designatedRequirement(for applicationURL: URL) async throws -> String {
        let result = try await runProcess("/usr/bin/codesign", ["-d", "-r-", applicationURL.path])
        let text = String(data: result.stderr + result.stdout, encoding: .utf8) ?? ""
        guard let requirement = text.split(whereSeparator: \.isNewline)
            .map(String.init)
            .first(where: { $0.contains("designated =>") })?
            .split(separator: ">", maxSplits: 1)
            .last?
            .trimmingCharacters(in: .whitespacesAndNewlines), !requirement.isEmpty else {
            throw GitHubUpdateError.invalidApplication("designated signing requirement is missing")
        }
        return requirement
    }

    static func runProcess(_ executable: String, _ arguments: [String]) async throws -> (stdout: Data, stderr: Data) {
        try await Task.detached(priority: .utility) {
            let process = Process()
            let stdout = Pipe()
            let stderr = Pipe()
            process.executableURL = URL(fileURLWithPath: executable)
            process.arguments = arguments
            process.standardOutput = stdout
            process.standardError = stderr
            try process.run()
            // Drain both pipes while the child runs. Waiting first can deadlock
            // when a future tool invocation writes more than the pipe buffer.
            async let outputRead = Task.detached(priority: .utility) {
                stdout.fileHandleForReading.readDataToEndOfFile()
            }.value
            async let errorRead = Task.detached(priority: .utility) {
                stderr.fileHandleForReading.readDataToEndOfFile()
            }.value
            process.waitUntilExit()
            let (output, error) = await (outputRead, errorRead)
            guard process.terminationStatus == 0 else {
                let details = String(data: error, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                throw GitHubUpdateError.commandFailed(URL(fileURLWithPath: executable).lastPathComponent, process.terminationStatus, details)
            }
            return (output, error)
        }.value
    }
}

struct SelfUpdateInstaller: Sendable {
    private let prepared: PreparedAppUpdate
    private let incomingApplicationURL: URL
    private let backupApplicationURL: URL
    private let helperScriptURL: URL
    private let readinessURL: URL

    static func prepare(_ prepared: PreparedAppUpdate) async throws -> Self {
        let parent = prepared.targetApplicationURL.deletingLastPathComponent()
        guard FileManager.default.isWritableFile(atPath: parent.path) else {
            throw GitHubUpdateError.applicationCannotBeReplaced(prepared.targetApplicationURL.path)
        }
        let token = UUID().uuidString
        let incoming = parent.appendingPathComponent(".AgentTrainer Update \(token).app", isDirectory: true)
        let backup = parent.appendingPathComponent(".AgentTrainer Backup \(token).app", isDirectory: true)
        try? FileManager.default.removeItem(at: incoming)
        _ = try await GitHubReleaseUpdater.runProcess("/usr/bin/ditto", ["--rsrc", "--extattr", prepared.stagedApplicationURL.path, incoming.path])
        _ = try await GitHubReleaseUpdater.runProcess("/usr/bin/codesign", ["--verify", "--deep", "--strict", "--verbose=2", incoming.path])

        let script = prepared.workingDirectoryURL.appendingPathComponent("install-update.zsh")
        let ready = prepared.workingDirectoryURL.appendingPathComponent("installer-ready")
        try? FileManager.default.removeItem(at: ready)
        let contents = """
        #!/bin/zsh
        set -u
        pid="$1"
        incoming="$2"
        target="$3"
        backup="$4"
        ready="$5"
        cleanup="$6"
        log="$HOME/Library/Logs/AgentTrainer-update.log"
        /bin/mkdir -p "$HOME/Library/Logs"
        exec >>"$log" 2>&1
        /usr/bin/touch "$ready"
        while /bin/kill -0 "$pid" >/dev/null 2>&1; do /bin/sleep 0.1; done
        /bin/rm -rf "$backup"
        if ! /bin/mv "$target" "$backup"; then
            /usr/bin/open "$target" >/dev/null 2>&1 || true
            exit 1
        fi
        if /bin/mv "$incoming" "$target" && /usr/bin/codesign --verify --deep --strict "$target"; then
            /usr/bin/open "$target"
            /bin/sleep 2
            /bin/rm -rf "$backup" "$cleanup"
            exit 0
        fi
        /bin/rm -rf "$target"
        /bin/mv "$backup" "$target"
        /usr/bin/open "$target" >/dev/null 2>&1 || true
        exit 1
        """
        try Data(contents.utf8).write(to: script, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: script.path)
        return Self(
            prepared: prepared,
            incomingApplicationURL: incoming,
            backupApplicationURL: backup,
            helperScriptURL: script,
            readinessURL: ready
        )
    }

    func launch() async throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = [
            helperScriptURL.path,
            String(ProcessInfo.processInfo.processIdentifier),
            incomingApplicationURL.path,
            prepared.targetApplicationURL.path,
            backupApplicationURL.path,
            readinessURL.path,
            prepared.workingDirectoryURL.path
        ]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        for _ in 0..<100 {
            if FileManager.default.fileExists(atPath: readinessURL.path) { return }
            if !process.isRunning { throw GitHubUpdateError.installerHelperFailed }
            try await Task.sleep(for: .milliseconds(20))
        }
        throw GitHubUpdateError.installerHelperFailed
    }
}
