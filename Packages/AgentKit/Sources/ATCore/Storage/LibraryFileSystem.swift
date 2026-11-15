import Foundation

/// A folder in the recordings library (a real directory on disk).
public struct LibraryFolder: Identifiable, Hashable, Sendable {
    public var id: URL { url }
    public let url: URL
    public var name: String { url.lastPathComponent }
    public var children: [LibraryFolder]
    public var recordingCount: Int
    public var totalRecordingCount: Int { recordingCount + children.reduce(0) { $0 + $1.totalRecordingCount } }
}

/// A recording package in the library.
public struct LibraryRecording: Identifiable, Hashable, Sendable {
    public var id: URL { url }
    public let url: URL
    public var meta: RecordingMeta
    public var folder: URL { url.deletingLastPathComponent() }
    public var thumbnail: URL { url.appendingPathComponent(RecordingPackage.thumbnail) }
}

public enum LibraryError: Error, LocalizedError {
    case exists(String), invalidName, outsideLibrary, intoItself
    public var errorDescription: String? {
        switch self {
        case .exists(let n): "Something named “\(n)” is already there."
        case .invalidName: "That name can't be used."
        case .outsideLibrary: "That location is outside the library."
        case .intoItself: "A folder can't be moved into itself."
        }
    }
}

/// Folder and recording operations. All moves are real file moves; deletes go to the Trash.
public struct LibraryFileSystem: Sendable {
    public let root: URL
    public init(root: URL) { self.root = root.standardizedFileURL }

    private var fm: FileManager { .default }

    public func folderTree() -> LibraryFolder {
        func scan(_ url: URL) -> LibraryFolder {
            let items = (try? fm.contentsOfDirectory(at: url, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])) ?? []
            var children: [LibraryFolder] = [], count = 0
            for i in items.sorted(by: { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }) {
                if i.pathExtension == RecordingPackage.fileExtension { count += 1 }
                else if (try? i.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true { children.append(scan(i)) }
            }
            return LibraryFolder(url: url, children: children, recordingCount: count)
        }
        return scan(root)
    }

    /// Recordings directly in `folder`, or in it and all subfolders.
    public func recordings(in folder: URL, recursive: Bool) -> [LibraryRecording] {
        var out: [LibraryRecording] = []
        let items = (try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])) ?? []
        for i in items {
            if i.pathExtension == RecordingPackage.fileExtension {
                if let m = try? RecordingIO.readMeta(i) { out.append(LibraryRecording(url: i, meta: m)) }
            } else if recursive, (try? i.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                out += recordings(in: i, recursive: true)
            }
        }
        return out.sorted { $0.meta.createdAt > $1.meta.createdAt }
    }

    func validated(_ name: String) throws -> String {
        let n = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !n.isEmpty, !n.contains("/"), !n.contains(":"), n != ".", n != "..", !n.hasPrefix(".") else { throw LibraryError.invalidName }
        return n
    }

    func inside(_ url: URL) -> Bool { url.standardizedFileURL.path.hasPrefix(root.path) }

    @discardableResult
    public func createFolder(named name: String, in parent: URL) throws -> URL {
        guard inside(parent) else { throw LibraryError.outsideLibrary }
        let url = parent.appendingPathComponent(try validated(name), isDirectory: true)
        guard !fm.fileExists(atPath: url.path) else { throw LibraryError.exists(url.lastPathComponent) }
        try fm.createDirectory(at: url, withIntermediateDirectories: false)
        return url
    }

    /// Renames a folder or a recording (keeps the `.atrec` extension and updates the recording's name).
    @discardableResult
    public func rename(_ url: URL, to name: String) throws -> URL {
        guard inside(url), url.standardizedFileURL != root else { throw LibraryError.outsideLibrary }
        let n = try validated(name)
        let isRecording = url.pathExtension == RecordingPackage.fileExtension
        let dest = url.deletingLastPathComponent().appendingPathComponent(isRecording ? n + "." + RecordingPackage.fileExtension : n)
        guard dest != url else { return url }
        guard !fm.fileExists(atPath: dest.path) else { throw LibraryError.exists(n) }
        try fm.moveItem(at: url, to: dest)
        if isRecording, var m = try? RecordingIO.readMeta(dest) { m.name = n; try RecordingIO.writeMeta(m, to: dest) }
        return dest
    }

    /// Moves a recording or folder into another library folder (changing its parent).
    @discardableResult
    public func move(_ url: URL, into folder: URL) throws -> URL {
        guard inside(url), inside(folder) else { throw LibraryError.outsideLibrary }
        let src = url.standardizedFileURL, dst = folder.standardizedFileURL
        if dst.path == src.path || dst.path.hasPrefix(src.path + "/") { throw LibraryError.intoItself }
        let target = dst.appendingPathComponent(src.lastPathComponent)
        guard target.path != src.path else { return src }
        guard !fm.fileExists(atPath: target.path) else { throw LibraryError.exists(src.lastPathComponent) }
        try fm.moveItem(at: src, to: target)
        return target
    }

    /// Moves to the Trash (recoverable).
    public func trash(_ url: URL) throws {
        guard inside(url), url.standardizedFileURL != root else { throw LibraryError.outsideLibrary }
        try fm.trashItem(at: url, resultingItemURL: nil)
    }
}
