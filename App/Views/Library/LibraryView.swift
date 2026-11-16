import ATCore
import SwiftUI

struct LibraryView: View {
    @Bindable var app: AppModel
    @State private var selection = Set<URL>()
    @State private var newFolderParent: URL?
    @State private var renaming: URL?
    @State private var renamingMany: [URL] = []
    @State private var instructionFor: [URL] = []
    @State private var nameField = ""
    @State private var confirmTrash: [URL] = []

    private var lib: LibraryStore { app.library }
    /// Selected recordings, in list order.
    private var selected: [LibraryRecording] { lib.recordings.filter { selection.contains($0.url) } }

    var body: some View {
        HStack(spacing: 0) {
            folderList.frame(width: 220)
            Divider()
            recordingList.frame(width: 340)
            Divider()
            detail.frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onChange(of: lib.selectedFolder) { selection = [] }
        .onChange(of: lib.recordings) { _, recs in
            let present = Set(recs.map(\.url))
            selection = selection.filter(present.contains)
        }
        .task {
            // UI snapshot hooks: open recordings so the detail panes can be checked.
            if CommandLine.arguments.contains("--open-first-recording") { selection = Set(lib.recordings.prefix(1).map(\.url)) }
            if CommandLine.arguments.contains("--select-three-recordings") { selection = Set(lib.recordings.prefix(3).map(\.url)) }
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
                    let wasSelected = selection.contains(u)
                    let new = lib.rename([u], base: nameField)
                    if wasSelected { selection = Set(new) }
                }
                nameField = ""
            }
            Button("Cancel", role: .cancel) { nameField = "" }
        }
        .alert("Rename \(renamingMany.count) recordings", isPresented: Binding(get: { !renamingMany.isEmpty }, set: { if !$0 { renamingMany = [] } })) {
            TextField("Name", text: $nameField)
            Button("Rename") { selection = Set(lib.rename(renamingMany, base: nameField)); nameField = "" }
            Button("Cancel", role: .cancel) { nameField = "" }
        } message: {
            Text("They will be named “\(nameField.isEmpty ? "Name" : nameField) 1”, “… 2” and so on, in list order.")
        }
        .alert("Instruction for \(instructionFor.count) recording\(instructionFor.count == 1 ? "" : "s")", isPresented: Binding(get: { !instructionFor.isEmpty }, set: { if !$0 { instructionFor = [] } })) {
            TextField("e.g. Drive along the road", text: $nameField)
            Button("Set") { lib.setInstruction(nameField, for: instructionFor); nameField = "" }
            Button("Cancel", role: .cancel) { nameField = "" }
        } message: {
            Text("Applies to the whole of each recording and replaces any earlier whole-recording instruction. Leave empty to remove it. Timed instructions are kept.")
        }
        .confirmationDialog(confirmTrash.count == 1 ? "Move to Trash?" : "Move \(confirmTrash.count) items to Trash?",
                            isPresented: Binding(get: { !confirmTrash.isEmpty }, set: { if !$0 { confirmTrash = [] } })) {
            Button("Move to Trash", role: .destructive) { lib.trash(confirmTrash); selection.subtract(confirmTrash); confirmTrash = [] }
        } message: {
            Text(confirmTrash.count == 1 ? "“\(confirmTrash.first?.deletingPathExtension().lastPathComponent ?? "")” will be moved to the Trash. You can restore it from there."
                                         : "\(confirmTrash.count) recordings will be moved to the Trash. You can restore them from there.")
        }
        .onAppear { lib.refresh() }
    }

    // MARK: Folders

    private var folderList: some View {
        VStack(spacing: 0) {
            List(selection: Binding(get: { lib.selectedFolder }, set: { if let u = $0 { lib.selectedFolder = u } })) {
                if lib.tree.recordingCount > 0 || lib.isRootSelected {
                    unsortedRow
                }
                OutlineGroup(lib.tree.children, children: \.optionalChildren) { f in
                    folderRow(f)
                }
            }
            .contextMenu { Button("New Folder…") { newFolderParent = lib.fs.root } }
            .overlay {
                if lib.tree.children.isEmpty && lib.tree.recordingCount == 0 {
                    ContentUnavailableView("No folders", systemImage: "folder", description: Text("Create a folder to sort your recordings."))
                }
            }
            Divider()
            HStack {
                Button { newFolderParent = lib.fs.root } label: { Label("New Folder", systemImage: "folder.badge.plus") }
                    .help("Create a folder (right-click a folder to create one inside it)")
                Spacer()
                Toggle("Subfolders", isOn: Binding(get: { lib.includeSubfolders }, set: { lib.includeSubfolders = $0 })).toggleStyle(.checkbox).font(.caption)
                    .help("Also show recordings inside subfolders")
            }
            .buttonStyle(.borderless).padding(8)
        }
    }

    private var unsortedRow: some View {
        Label {
            HStack {
                Text("Unsorted")
                Spacer()
                Text("\(lib.tree.recordingCount)").foregroundStyle(.secondary).font(.caption).monospacedDigit()
            }
        } icon: { Image(systemName: "tray") }
        .tag(lib.fs.root)
        .help("Recordings that aren't in a folder")
        .dropDestination(for: URL.self) { urls, _ in
            lib.move(urls.filter { $0.pathExtension == RecordingPackage.fileExtension }, into: lib.fs.root)
            return true
        }
    }

    private func folderRow(_ f: LibraryFolder) -> some View {
        Label {
            HStack {
                Text(f.name)
                Spacer()
                Text("\(f.totalRecordingCount)").foregroundStyle(.secondary).font(.caption).monospacedDigit()
            }
        } icon: { Image(systemName: "folder") }
        .tag(f.url)
        .dropDestination(for: URL.self) { urls, _ in
            lib.move(urls.filter { $0.pathExtension == RecordingPackage.fileExtension || (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }, into: f.url)
            return true
        }
        .draggable(f.url)
        .contextMenu {
            Button("New Folder Inside…") { newFolderParent = f.url }
            Button("Rename…") { nameField = f.name; renaming = f.url }
            Menu("Move To") { moveTargets(for: [f.url]) }
            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([f.url]) }
            Divider()
            Button("Move to Trash…", role: .destructive) { confirmTrash = [f.url] }
        }
    }

    @ViewBuilder
    func moveTargets(for urls: [URL]) -> some View {
        ForEach(lib.flatFolders, id: \.url) { t in
            Button(String(repeating: "    ", count: t.depth) + t.name) { lib.move(urls, into: t.url) }
                .disabled(urls.allSatisfy { u in t.url == u.deletingLastPathComponent().standardizedFileURL || t.url.path.hasPrefix(u.path + "/") || t.url == u })
        }
    }

    // MARK: Recordings

    private var recordingList: some View {
        VStack(spacing: 0) {
            if let e = lib.lastError { InlineError(message: e).padding(8) }
            List(selection: $selection) {
                ForEach(lib.recordings) { r in
                    RecordingRow(recording: r, library: lib)
                        .tag(r.url)
                        .draggable(r.url)
                }
            }
            .contextMenu(forSelectionType: URL.self) { urls in
                recordingMenu(Array(urls))
            }
            .onDeleteCommand { if !selection.isEmpty { confirmTrash = selected.map(\.url) } }
            .overlay {
                if lib.recordings.isEmpty {
                    ContentUnavailableView("No recordings here", systemImage: "tray", description: Text("Record in the Recordings tab, or drag recordings onto a folder."))
                }
            }
            Divider()
            Text(footer).font(.caption).foregroundStyle(.secondary).padding(8)
        }
    }

    private var footer: String {
        let all = "\(lib.recordings.count) recordings · \(lib.recordings.reduce(0) { $0 + $1.meta.duration }.durationText)"
        return selection.count > 1 ? "\(selection.count) selected · " + all : all
    }

    @ViewBuilder
    private func recordingMenu(_ urls: [URL]) -> some View {
        let ordered = lib.recordings.map(\.url).filter(urls.contains)
        if ordered.count == 1, let u = ordered.first {
            Button("Rename…") { nameField = u.deletingPathExtension().lastPathComponent; renaming = u }
        } else if ordered.count > 1 {
            Button("Rename \(ordered.count) Recordings…") { nameField = ""; renamingMany = ordered }
        }
        if !ordered.isEmpty {
            Button(ordered.count == 1 ? "Set Instruction…" : "Set Instruction for \(ordered.count)…") { nameField = ""; instructionFor = ordered }
            Menu(ordered.count == 1 ? "Move To" : "Move \(ordered.count) To") { moveTargets(for: ordered) }
            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting(ordered) }
            Divider()
            Button(ordered.count == 1 ? "Move to Trash…" : "Move \(ordered.count) to Trash…", role: .destructive) { confirmTrash = ordered }
        }
    }

    // MARK: Detail

    @ViewBuilder
    private var detail: some View {
        let sel = selected
        if sel.count == 1, let r = sel.first {
            RecordingDetailView(app: app, recording: r).id(r.url)
        } else if sel.count > 1 {
            VStack(spacing: 14) {
                Image(systemName: "square.stack.3d.up").font(.system(size: 44)).foregroundStyle(.secondary)
                Text("\(sel.count) recordings selected").font(.title2.weight(.semibold))
                Text("\(sel.reduce(0) { $0 + $1.meta.duration }.durationText) in total").foregroundStyle(.secondary)
                HStack {
                    Menu { moveTargets(for: sel.map(\.url)) } label: { Label("Move To", systemImage: "folder") }.fixedSize()
                    Button { nameField = ""; renamingMany = sel.map(\.url) } label: { Label("Rename…", systemImage: "pencil") }
                    Button { nameField = ""; instructionFor = sel.map(\.url) } label: { Label("Set Instruction…", systemImage: "text.bubble") }
                    Button(role: .destructive) { confirmTrash = sel.map(\.url) } label: { Label("Move to Trash…", systemImage: "trash") }
                }
                .controlSize(.large)
                Text("⌘-click or ⇧-click to change the selection.").font(.caption).foregroundStyle(.tertiary)
            }
            .padding(24)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ContentUnavailableView("No recording selected", systemImage: "film",
                                   description: Text("Pick a recording to play it and edit its instructions. ⌘-click or ⇧-click to select several."))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
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
