import ATCore
import Foundation
import Observation

/// The recordings library: a folder tree on disk plus the recordings in the selected folder.
@MainActor @Observable
final class LibraryStore {
    private(set) var fs: LibraryFileSystem
    private(set) var tree: LibraryFolder
    var selectedFolder: URL { didSet { reloadRecordings() } }
    var includeSubfolders = true { didSet { reloadRecordings() } }
    private(set) var recordings: [LibraryRecording] = []
    var lastError: String?

    init(root: URL) {
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let f = LibraryFileSystem(root: root)
        fs = f
        tree = f.folderTree()
        selectedFolder = f.root
        recordings = f.recordings(in: f.root, recursive: true)
    }

    func setRoot(_ url: URL) {
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        fs = LibraryFileSystem(root: url)
        selectedFolder = fs.root
        refresh()
    }

    func refresh() {
        tree = fs.folderTree()
        if !FileManager.default.fileExists(atPath: selectedFolder.path) { selectedFolder = fs.root }
        reloadRecordings()
    }

    private func reloadRecordings() { recordings = fs.recordings(in: selectedFolder, recursive: includeSubfolders) }

    /// Every folder as (url, indented display name), for pickers.
    var flatFolders: [(url: URL, name: String, depth: Int)] {
        var out: [(URL, String, Int)] = []
        func walk(_ f: LibraryFolder, _ depth: Int) {
            out.append((f.url, depth == 0 ? "Library" : f.name, depth))
            f.children.forEach { walk($0, depth + 1) }
        }
        walk(tree, 0)
        return out
    }

    func allRecordings() -> [LibraryRecording] { fs.recordings(in: fs.root, recursive: true) }

    func relativeName(_ folder: URL) -> String {
        let r = fs.root.standardizedFileURL.path, p = folder.standardizedFileURL.path
        return p == r ? "Library" : "Library/" + p.dropFirst(r.count + 1)
    }

    private func perform(_ body: () throws -> Void) {
        do { try body(); lastError = nil } catch { lastError = error.localizedDescription }
        refresh()
    }

    func createFolder(_ name: String, in parent: URL) { perform { selectedFolder = try fs.createFolder(named: name, in: parent) } }
    func rename(_ url: URL, to name: String) {
        perform {
            let new = try fs.rename(url, to: name)
            if selectedFolder == url { selectedFolder = new }
        }
    }
    func move(_ url: URL, into folder: URL) { perform { _ = try fs.move(url, into: folder) } }
    func trash(_ url: URL) { perform { try fs.trash(url) } }
}
