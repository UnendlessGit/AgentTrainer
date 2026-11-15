import ATCore
import SwiftUI

struct LibraryView: View {
    @Bindable var app: AppModel
    @State private var selectedRecording: URL?
    @State private var newFolderParent: URL?
    @State private var renaming: URL?
    @State private var nameField = ""
    @State private var confirmTrash: URL?

    private var lib: LibraryStore { app.library }

    var body: some View {
        HSplitView {
            folderList.frame(minWidth: 200, idealWidth: 230, maxWidth: 320)
            recordingList.frame(minWidth: 280, idealWidth: 340)
            Group {
                if let url = selectedRecording, let r = lib.recordings.first(where: { $0.url == url }) {
                    RecordingDetailView(app: app, recording: r).id(r.url)
                } else {
                    ContentUnavailableView("No recording selected", systemImage: "film", description: Text("Pick a recording to play it, edit its instructions or move it."))
                }
            }
            .frame(minWidth: 380, maxWidth: .infinity)
        }
        .alert("New folder", isPresented: Binding(get: { newFolderParent != nil }, set: { if !$0 { newFolderParent = nil } })) {
            TextField("Name", text: $nameField)
            Button("Create") { if let p = newFolderParent { lib.createFolder(nameField, in: p) }; nameField = "" }
            Button("Cancel", role: .cancel) { nameField = "" }
        }
        .alert("Rename", isPresented: Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })) {
            TextField("Name", text: $nameField)
            Button("Rename") {
                if let u = renaming {
                    let wasSelected = selectedRecording == u
                    lib.rename(u, to: nameField)
                    if wasSelected { selectedRecording = lib.recordings.first { $0.meta.name == nameField }?.url }
                }
                nameField = ""
            }
            Button("Cancel", role: .cancel) { nameField = "" }
        }
        .confirmationDialog("Move to Trash?", isPresented: Binding(get: { confirmTrash != nil }, set: { if !$0 { confirmTrash = nil } })) {
            Button("Move to Trash", role: .destructive) { if let u = confirmTrash { lib.trash(u); if selectedRecording == u { selectedRecording = nil } } }
        } message: {
            Text("“\(confirmTrash?.deletingPathExtension().lastPathComponent ?? "")” will be moved to the Trash. You can restore it from there.")
        }
        .onAppear { lib.refresh() }
    }

    private var folderList: some View {
        VStack(spacing: 0) {
            List(selection: Binding(get: { lib.selectedFolder }, set: { if let u = $0 { lib.selectedFolder = u } })) {
                OutlineGroup([lib.tree], children: \.optionalChildren) { f in
                    folderRow(f)
                }
            }
            Divider()
            HStack {
                Button { newFolderParent = lib.selectedFolder } label: { Label("New Folder", systemImage: "folder.badge.plus") }
                Spacer()
                Toggle("Subfolders", isOn: Binding(get: { lib.includeSubfolders }, set: { lib.includeSubfolders = $0 })).toggleStyle(.checkbox).font(.caption)
            }
            .buttonStyle(.borderless).padding(8)
        }
    }

    private func folderRow(_ f: LibraryFolder) -> some View {
        let isRoot = f.url == lib.fs.root
        return Label {
            HStack {
                Text(isRoot ? "Library" : f.name)
                Spacer()
                Text("\(f.totalRecordingCount)").foregroundStyle(.secondary).font(.caption).monospacedDigit()
            }
        } icon: { Image(systemName: isRoot ? "books.vertical" : "folder") }
        .tag(f.url)
        .dropDestination(for: URL.self) { urls, _ in
            for u in urls where u.pathExtension == RecordingPackage.fileExtension || (try? u.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                lib.move(u, into: f.url)
            }
            return true
        }
        .draggable(f.url)
        .contextMenu {
            Button("New Folder Inside…") { newFolderParent = f.url }
            if !isRoot {
                Button("Rename…") { nameField = f.name; renaming = f.url }
                Menu("Move To") { moveTargets(for: f.url) }
                Divider()
                Button("Move to Trash…", role: .destructive) { confirmTrash = f.url }
            }
            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([f.url]) }
        }
    }

    @ViewBuilder
    func moveTargets(for url: URL) -> some View {
        ForEach(lib.flatFolders, id: \.url) { t in
            Button(String(repeating: "  ", count: t.depth) + t.name) { lib.move(url, into: t.url) }
                .disabled(t.url == url.deletingLastPathComponent().standardizedFileURL || t.url.path.hasPrefix(url.path + "/") || t.url == url)
        }
    }

    private var recordingList: some View {
        VStack(spacing: 0) {
            if let e = lib.lastError { InlineError(message: e).padding(8) }
            if lib.recordings.isEmpty {
                ContentUnavailableView("No recordings here", systemImage: "tray", description: Text("Record in the Recordings tab, or drag recordings onto a folder."))
            } else {
                List(selection: $selectedRecording) {
                    ForEach(lib.recordings) { r in
                        RecordingRow(recording: r, library: lib)
                            .tag(r.url)
                            .draggable(r.url)
                            .contextMenu {
                                Button("Rename…") { nameField = r.meta.name; renaming = r.url }
                                Menu("Move To") { moveTargets(for: r.url) }
                                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([r.url]) }
                                Divider()
                                Button("Move to Trash…", role: .destructive) { confirmTrash = r.url }
                            }
                    }
                }
            }
            Divider()
            Text("\(lib.recordings.count) recordings · \(lib.recordings.reduce(0) { $0 + $1.meta.duration }.durationText) total")
                .font(.caption).foregroundStyle(.secondary).padding(8)
        }
    }
}

extension LibraryFolder {
    var optionalChildren: [LibraryFolder]? { children.isEmpty ? nil : children }
}

struct RecordingRow: View {
    let recording: LibraryRecording
    let library: LibraryStore
    var body: some View {
        HStack(spacing: 10) {
            AsyncThumb(url: recording.thumbnail).frame(width: 72, height: 44)
            VStack(alignment: .leading, spacing: 2) {
                Text(recording.meta.name).lineLimit(1)
                Text("\(recording.meta.duration.durationText) · \(recording.meta.target.displayName)")
                    .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                if library.includeSubfolders, recording.folder.standardizedFileURL != library.selectedFolder.standardizedFileURL {
                    Text(library.relativeName(recording.folder)).font(.caption2).foregroundStyle(.tertiary).lineLimit(1)
                }
            }
            Spacer()
            if recording.meta.isCorrectionRun { Image(systemName: "arrow.uturn.backward.circle").help("Correction run") }
        }
        .padding(.vertical, 2)
    }
}

struct AsyncThumb: View {
    let url: URL
    @State private var image: NSImage?
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 5).fill(.quaternary)
            if let image { Image(nsImage: image).resizable().aspectRatio(contentMode: .fill) } else { Image(systemName: "film").foregroundStyle(.secondary) }
        }
        .clipShape(RoundedRectangle(cornerRadius: 5))
        .task(id: url) { image = await Task.detached { NSImage(contentsOf: url) }.value }
    }
}
