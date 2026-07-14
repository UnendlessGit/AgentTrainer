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
        }
    }
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

    func downloadVerifiedInstaller(for release: GitHubRelease) async throws -> URL {
        guard let installer = release.installerAsset else { throw GitHubUpdateError.missingInstaller }
        guard let checksum = release.checksumAsset else { throw GitHubUpdateError.missingChecksum }
        guard installer.size > 0, installer.size <= 1_500_000_000,
              checksum.size > 0, checksum.size <= 2_000_000 else { throw GitHubUpdateError.downloadTooLarge }

        async let installerData = download(asset: installer, maximumBytes: 1_500_000_000)
        async let checksumData = download(asset: checksum, maximumBytes: 2_000_000)
        let (download, sums) = try await (installerData, checksumData)
        guard let expected = Self.expectedChecksum(for: installer.name, in: sums) else {
            throw GitHubUpdateError.checksumEntryMissing(installer.name)
        }
        let actual = SHA256.hash(data: download).map { String(format: "%02x", $0) }.joined()
        guard actual.caseInsensitiveCompare(expected) == .orderedSame else { throw GitHubUpdateError.checksumMismatch }

        let tag = release.tagName.map { $0.isLetter || $0.isNumber || $0 == "." || $0 == "-" ? $0 : "-" }
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AgentTrainer Updates", isDirectory: true)
            .appendingPathComponent(String(tag), isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let destination = directory.appendingPathComponent(installer.name)
        try download.write(to: destination, options: .atomic)
        return destination
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

    private func download(asset: GitHubRelease.Asset, maximumBytes: Int) async throws -> Data {
        guard asset.browserDownloadURL.scheme == "https", asset.browserDownloadURL.host?.lowercased() == "github.com" else {
            throw GitHubUpdateError.unsafeDownloadURL
        }
        var request = URLRequest(url: asset.browserDownloadURL)
        request.timeoutInterval = 120
        request.setValue("application/octet-stream", forHTTPHeaderField: "Accept")
        request.setValue("AgentTrainer/\(configuration.currentVersion)", forHTTPHeaderField: "User-Agent")
        return try await validatedData(for: request, permittedInitialHosts: ["github.com"], maximumBytes: maximumBytes)
    }

    private func validatedData(for request: URLRequest, permittedInitialHosts: Set<String>, maximumBytes: Int) async throws -> Data {
        guard let host = request.url?.host?.lowercased(), permittedInitialHosts.contains(host) else { throw GitHubUpdateError.unsafeDownloadURL }
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw GitHubUpdateError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else { throw GitHubUpdateError.requestFailed(http.statusCode) }
        guard data.count <= maximumBytes else { throw GitHubUpdateError.downloadTooLarge }
        guard let finalHost = http.url?.host?.lowercased(),
              finalHost == "github.com" || finalHost == "api.github.com" || finalHost.hasSuffix(".githubusercontent.com") else {
            throw GitHubUpdateError.unsafeDownloadURL
        }
        return data
    }
}
