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
    /// Every recording in the library (for the folder tree).
    private(set) var all: [LibraryRecording] = []
    var lastError: String?

    init(root: URL) {
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let f = LibraryFileSystem(root: root)
        fs = f
        tree = f.folderTree()
        selectedFolder = f.root
        normalizeSelection()
        reloadRecordings()
    }

    /// The library root itself isn't shown; its loose recordings appear as "Unsorted". Start on the
    /// first folder when there is nothing unsorted.
    private func normalizeSelection() {
        if selectedFolder.standardizedFileURL == fs.root.standardizedFileURL, tree.recordingCount == 0, let first = tree.children.first {
            selectedFolder = first.url
        }
    }

    var isRootSelected: Bool { selectedFolder.standardizedFileURL == fs.root.standardizedFileURL }

    func setRoot(_ url: URL) {
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        fs = LibraryFileSystem(root: url)
        selectedFolder = fs.root
        refresh()
    }

    func refresh() {
        tree = fs.folderTree()
        if !FileManager.default.fileExists(atPath: selectedFolder.path) { selectedFolder = fs.root }
        normalizeSelection()
        reloadRecordings()
        all = fs.recordings(in: fs.root, recursive: true)
    }

    private func reloadRecordings() { recordings = fs.recordings(in: selectedFolder, recursive: includeSubfolders && !isRootSelected) }

    /// Every folder as (url, indented display name), for pickers.
    var flatFolders: [(url: URL, name: String, depth: Int)] {
        var out: [(URL, String, Int)] = []
        func walk(_ f: LibraryFolder, _ depth: Int) {
            out.append((f.url, f.name, depth))
            f.children.forEach { walk($0, depth + 1) }
        }
        // The root holds loose ("Unsorted") recordings; folders are listed at the top level.
        out.append((tree.url, "Unsorted", 0))
        tree.children.forEach { walk($0, 0) }
        return out
    }

    func allRecordings() -> [LibraryRecording] { fs.recordings(in: fs.root, recursive: true) }

    func relativeName(_ folder: URL) -> String {
        let r = fs.root.standardizedFileURL.path, p = folder.standardizedFileURL.path
        return p == r ? "Unsorted" : String(p.dropFirst(r.count + 1))
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

    // MARK: Several recordings at once

    func move(_ urls: [URL], into folder: URL) { perform { for u in urls where u.deletingLastPathComponent().standardizedFileURL != folder.standardizedFileURL { _ = try fs.move(u, into: folder) } } }
    func trash(_ urls: [URL]) { perform { for u in urls { try fs.trash(u) } } }

    /// Renames to "name 1", "name 2", … in the given order. Returns the new URLs.
    @discardableResult
    func rename(_ urls: [URL], base: String) -> [URL] {
        var out: [URL] = []
        perform { for (i, u) in urls.enumerated() { out.append(try fs.rename(u, to: urls.count == 1 ? base : "\(base) \(i + 1)")) } }
        return out
    }

    /// Sets one instruction for the whole of each recording (replacing any previous whole-recording one).
    func setInstruction(_ text: String, for urls: [URL]) {
        perform {
            for u in urls {
                var ann = RecordingIO.readAnnotations(u)
                ann.segments.removeAll { $0.source == .task }
                let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if !t.isEmpty, let meta = try? RecordingIO.readMeta(u) {
                    ann.segments.insert(TextSegment(start: 0, end: meta.duration + 1, text: t, source: .task), at: 0)
                }
                try RecordingIO.writeAnnotations(ann, to: u)
            }
        }
    }

}
