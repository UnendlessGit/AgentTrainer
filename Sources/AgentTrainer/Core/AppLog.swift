import AppKit
import Foundation
import OSLog

enum AppLogLevel: String, Codable, CaseIterable, Sendable {
    case debug = "Debug"
    case info = "Info"
    case warning = "Warning"
    case error = "Error"
}

struct AppLogEntry: Codable, Identifiable, Hashable, Sendable {
    var id = UUID()
    var timestamp = Date()
    var level: AppLogLevel
    var category: String
    var message: String
    var details: String?
}

enum AppLog {
    private static let system = Logger(subsystem: Bundle.main.bundleIdentifier ?? "local.agenttrainer.mac", category: "AgentTrainer")

    static func write(_ level: AppLogLevel = .info, category: String, _ message: String, details: String? = nil) {
        let rendered = details.map { "\(message) — \($0)" } ?? message
        switch level {
        case .debug: system.debug("\(rendered, privacy: .public)")
        case .info: system.info("\(rendered, privacy: .public)")
        case .warning: system.warning("\(rendered, privacy: .public)")
        case .error: system.error("\(rendered, privacy: .public)")
        }
        let entry = AppLogEntry(level: level, category: category, message: message, details: details)
        Task { @MainActor in AppLogStore.shared.append(entry) }
    }
}

@MainActor
final class AppLogStore: ObservableObject {
    static let shared = AppLogStore()

    @Published private(set) var entries: [AppLogEntry]
    private let persistenceQueue = DispatchQueue(label: "AgentTrainer.Diagnostics.Log", qos: .utility)
    private let maximumEntries = 1_500

    static var directory: URL {
        WorkspaceStore.shared.root.appendingPathComponent("Logs", isDirectory: true)
    }
    static var fileURL: URL { directory.appendingPathComponent("app.jsonl") }

    private init() {
        try? FileManager.default.createDirectory(at: Self.directory, withIntermediateDirectories: true)
        entries = Self.readRecentEntries()
    }

    func append(_ entry: AppLogEntry) {
        entries.append(entry)
        if entries.count > maximumEntries { entries.removeFirst(entries.count - maximumEntries) }
        let url = Self.fileURL
        persistenceQueue.async {
            do {
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try Self.rotateIfNeeded(url)
                var line = try JSONEncoder.agentTrainerLog.encode(entry)
                line.append(0x0A)
                if !FileManager.default.fileExists(atPath: url.path) { FileManager.default.createFile(atPath: url.path, contents: nil) }
                let handle = try FileHandle(forWritingTo: url)
                try handle.seekToEnd()
                try handle.write(contentsOf: line)
                try handle.close()
            } catch {
                // OSLog already received the entry; never recurse from logging IO.
            }
        }
    }

    func clear() {
        entries.removeAll(keepingCapacity: true)
        let directory = Self.directory
        persistenceQueue.async {
            try? FileManager.default.removeItem(at: directory)
            try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
    }

    func copyReport(appState: String) {
        let recent = entries.suffix(500).map { entry in
            let details = entry.details.map { " | \($0)" } ?? ""
            return "\(entry.timestamp.formatted(.iso8601)) [\(entry.level.rawValue)] [\(entry.category)] \(entry.message)\(details)"
        }.joined(separator: "\n")
        let report = """
        AgentTrainer diagnostic report
        Generated: \(Date().formatted(.iso8601))
        macOS: \(ProcessInfo.processInfo.operatingSystemVersionString)
        Hardware: \(Self.hardwareName())
        Memory: \(ByteCountFormatter.string(fromByteCount: Int64(ProcessInfo.processInfo.physicalMemory), countStyle: .memory))
        State: \(appState)

        Recent log
        \(recent.isEmpty ? "No entries recorded." : recent)
        """
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(report, forType: .string)
    }

    func revealLogs() {
        try? FileManager.default.createDirectory(at: Self.directory, withIntermediateDirectories: true)
        NSWorkspace.shared.activateFileViewerSelecting([Self.fileURL])
    }

    static func crashReports() -> [URL] {
        let directory = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/DiagnosticReports", isDirectory: true)
        let keys: [URLResourceKey] = [.contentModificationDateKey]
        let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: keys)) ?? []
        return files.filter {
            $0.lastPathComponent.hasPrefix("AgentTrainer") && ["ips", "crash"].contains($0.pathExtension.lowercased())
        }.sorted {
            let lhs = (try? $0.resourceValues(forKeys: Set(keys)).contentModificationDate) ?? .distantPast
            let rhs = (try? $1.resourceValues(forKeys: Set(keys)).contentModificationDate) ?? .distantPast
            return lhs > rhs
        }
    }

    private static func readRecentEntries() -> [AppLogEntry] {
        guard let data = try? Data(contentsOf: fileURL, options: .mappedIfSafe),
              let text = String(data: data.suffix(4 * 1_024 * 1_024), encoding: .utf8) else { return [] }
        return text.split(separator: "\n").suffix(1_500).compactMap {
            try? JSONDecoder.agentTrainerLog.decode(AppLogEntry.self, from: Data($0.utf8))
        }
    }

    nonisolated private static func rotateIfNeeded(_ url: URL) throws {
        let bytes = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.intValue ?? 0
        guard bytes > 8 * 1_024 * 1_024 else { return }
        let previous = url.deletingLastPathComponent().appendingPathComponent("app.previous.jsonl")
        try? FileManager.default.removeItem(at: previous)
        try FileManager.default.moveItem(at: url, to: previous)
    }

    private static func hardwareName() -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var value = [CChar](repeating: 0, count: max(1, size))
        sysctlbyname("machdep.cpu.brand_string", &value, &size, nil, 0)
        return String(decoding: value.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
}

private extension JSONEncoder {
    static var agentTrainerLog: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }
}

private extension JSONDecoder {
    static var agentTrainerLog: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
