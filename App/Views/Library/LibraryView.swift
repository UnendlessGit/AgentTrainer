import ATCore
import SwiftUI

struct LibraryView: View {
    @Bindable var app: AppModel
    @State private var selection = Set<URL>()
    @State private var anchor: URL?
    @State private var expanded = Set<URL>()
    @State private var search = ""
    @State private var newFolderParent: URL?
    @State private var renaming: URL?
    @State private var renamingMany: [URL] = []
    @State private var instructionFor: [URL] = []
    @State private var nameField = ""
    @State private var confirmTrash: [URL] = []

    private var lib: LibraryStore { app.library }

    var body: some View {
        HStack(spacing: 0) {
            sidebar.frame(width: 360).riseIn()
            Rectangle().fill(Theme.hairline).frame(width: 1)
            detail.frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .onAppear {
            lib.refresh()
            if expanded.isEmpty, let first = lib.tree.children.first { expanded.insert(first.url) }
            if CommandLine.arguments.contains("--open-first-recording"), let r = visibleOrder.first { selection = [r] }
        }
        .onChange(of: lib.all) { _, recs in
            let present = Set(recs.map(\.url))
            selection = selection.filter(present.contains)
        }
        .alert("New folder", isPresented: Binding(get: { newFolderParent != nil }, set: { if !$0 { newFolderParent = nil } })) {
            TextField("Name", text: $nameField)
            Button("Create") { if let p = newFolderParent { lib.createFolder(nameField, in: p); expanded.insert(p) }; nameField = "" }
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
            Button("Move to Trash", role: .destructive) { withAnimation(Theme.spring) { lib.trash(confirmTrash); selection.subtract(confirmTrash) }; confirmTrash = [] }
        } message: {
            Text(confirmTrash.count == 1 ? "“\(confirmTrash.first?.deletingPathExtension().lastPathComponent ?? "")” will be moved to the Trash. You can restore it from there."
                 : "\(confirmTrash.count) items will be moved to the Trash. You can restore them from there.")
        }
    }

    // MARK: Data

    private var searching: Bool { !search.trimmingCharacters(in: .whitespaces).isEmpty }

    private func recordings(in folder: URL) -> [LibraryRecording] {
        lib.all.filter { $0.folder.standardizedFileURL == folder.standardizedFileURL && matches($0) }
    }

    private func matches(_ r: LibraryRecording) -> Bool {
        !searching || r.meta.name.localizedCaseInsensitiveContains(search) || r.meta.target.displayName.localizedCaseInsensitiveContains(search)
    }

    private func isOpen(_ url: URL) -> Bool { searching || expanded.contains(url) }

    /// Recordings in the order they appear in the tree (for ⇧-click ranges).
    private var visibleOrder: [URL] {
        var out: [URL] = []
        if isOpen(lib.fs.root) { out += recordings(in: lib.fs.root).map(\.url) }
        func walk(_ f: LibraryFolder) {
            guard isOpen(f.url) else { return }
            out += recordings(in: f.url).map(\.url)
            f.children.forEach(walk)
        }
        lib.tree.children.forEach(walk)
        return out
    }

    private var selected: [LibraryRecording] { lib.all.filter { selection.contains($0.url) } }

    private func click(_ url: URL) {
        let flags = NSEvent.modifierFlags
        withAnimation(Theme.snappy) {
            if flags.contains(.command) {
                if selection.contains(url) { selection.remove(url) } else { selection.insert(url) }
                anchor = url
            } else if flags.contains(.shift), let a = anchor, let i = visibleOrder.firstIndex(of: a), let j = visibleOrder.firstIndex(of: url) {
                selection = Set(visibleOrder[min(i, j) ... max(i, j)])
            } else {
                selection = [url]
                anchor = url
            }
        }
    }

    // MARK: Sidebar

    private var sidebar: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                HStack(spacing: 8) {
                    Image(systemName: "magnifyingglass").font(.system(size: 12, weight: .medium)).foregroundStyle(Theme.muted)
                    TextField("Search", text: $search.animation(Theme.spring)).textFieldStyle(.plain).font(Theme.font(13.5))
                    if searching {
                        Button { withAnimation(Theme.spring) { search = "" } } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(Theme.muted) }.buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 12).frame(height: 34)
                .background(Theme.raised, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                Button { newFolderParent = lib.fs.root } label: { Image(systemName: "folder.badge.plus").font(.system(size: 14)) }
                    .buttonStyle(ATButtonStyle(kind: .secondary, height: 34)).help("New folder")
            }
            .padding(.horizontal, 16).padding(.top, 18).padding(.bottom, 10)

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    if !recordings(in: lib.fs.root).isEmpty || lib.tree.recordingCount > 0 {
                        folderRow(url: lib.fs.root, name: "Unsorted", icon: "tray", count: lib.tree.recordingCount, depth: 0, folder: nil)
                    }
                    ForEach(lib.tree.children) { f in node(f, depth: 0) }
                    if lib.tree.children.isEmpty && lib.all.isEmpty {
                        VStack(spacing: 10) {
                            Image(systemName: "film.stack").font(.system(size: 28, weight: .light)).foregroundStyle(Theme.faint)
                            Text("No recordings yet").font(Theme.font(14, .semibold))
                            Text("Record something in the Record tab.").font(Theme.font(12.5)).foregroundStyle(Theme.muted)
                        }
                        .frame(maxWidth: .infinity).padding(.top, 60)
                    }
                }
                .padding(.horizontal, 10).padding(.bottom, 12)
            }
            .scrollIndicators(.never)
            .focusable()
            .focusEffectDisabled()
            .onDeleteCommand { if !selection.isEmpty { confirmTrash = selected.map(\.url) } }
            .onCommand(#selector(NSResponder.selectAll(_:))) { selection = Set(visibleOrder) }

            Rectangle().fill(Theme.hairline).frame(height: 1)
            HStack {
                Text(footer).font(Theme.font(12)).foregroundStyle(Theme.muted).contentTransition(.numericText())
                Spacer()
            }
            .padding(.horizontal, 18).frame(height: 40)
        }
    }

    private var footer: String {
        let total = lib.all.reduce(0) { $0 + $1.meta.duration }
        let base = "\(lib.all.count) recordings · \(total.durationText)"
        return selection.count > 1 ? "\(selection.count) selected · " + base : base + " · ⌘-click to select several"
    }

    private func node(_ f: LibraryFolder, depth: Int) -> AnyView {
        AnyView(VStack(alignment: .leading, spacing: 2) {
            folderRow(url: f.url, name: f.name, icon: "folder", count: f.totalRecordingCount, depth: depth, folder: f)
            if isOpen(f.url) {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(f.children) { c in node(c, depth: depth + 1) }
                }
                .transition(.opacity.combined(with: .offset(y: -4)))
            }
        })
    }

    private func folderRow(url: URL, name: String, icon: String, count: Int, depth: Int, folder: LibraryFolder?) -> some View {
        let open = isOpen(url)
        let recs = recordings(in: url)
        return VStack(alignment: .leading, spacing: 2) {
            Button {
                withAnimation(Theme.spring) { if expanded.contains(url) { expanded.remove(url) } else { expanded.insert(url) } }
            } label: {
                HStack(spacing: 9) {
                    Image(systemName: "chevron.right").font(.system(size: 10, weight: .bold)).foregroundStyle(Theme.muted)
                        .rotationEffect(.degrees(open ? 90 : 0))
                    Image(systemName: open && icon == "folder" ? "folder.fill" : icon).font(.system(size: 14)).foregroundStyle(Theme.text2).frame(width: 18)
                    Text(name).font(Theme.font(14, .semibold)).lineLimit(1)
                    Spacer(minLength: 4)
                    Text("\(count)").font(Theme.font(12)).monospacedDigit().foregroundStyle(Theme.muted)
                }
                .padding(.leading, 10 + CGFloat(depth) * 16).padding(.trailing, 10)
                .frame(height: 38)
            }
            .buttonStyle(RowButtonStyle())
            .dropDestination(for: URL.self) { urls, _ in
                withAnimation(Theme.spring) { lib.move(urls.filter { $0 != url }, into: url); expanded.insert(url) }
                return true
            }
            .contextMenu {
                Button("New Folder Inside…") { newFolderParent = url }
                if folder != nil {
                    Button("Rename…") { nameField = name; renaming = url }
                    Menu("Move To") { moveTargets(for: [url]) }
                }
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                if folder != nil {
                    Divider()
                    Button("Move to Trash…", role: .destructive) { confirmTrash = [url] }
                }
            }
            if open {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(recs) { r in recordingRow(r, depth: depth) }
                }
                .transition(.opacity.combined(with: .offset(y: -4)))
            }
        }
    }

    private func recordingRow(_ r: LibraryRecording, depth: Int) -> some View {
        let on = selection.contains(r.url)
        return Button { click(r.url) } label: {
            HStack(spacing: 10) {
                AsyncThumb(url: r.thumbnail).frame(width: 48, height: 28)
                VStack(alignment: .leading, spacing: 1) {
                    Text(r.meta.name).font(Theme.font(13, .medium)).lineLimit(1).truncationMode(.middle)
                    Text("\(r.meta.duration.durationText)\(r.meta.isCorrectionRun ? " · correction run" : "")").font(Theme.font(11.5)).foregroundStyle(Theme.muted)
                }
                Spacer(minLength: 0)
            }
            .padding(.leading, 34 + CGFloat(depth) * 16).padding(.trailing, 10).padding(.vertical, 6)
        }
        .buttonStyle(RowButtonStyle(selected: on))
        .draggable(r.url)
        .contextMenu {
            let targets = on ? selected.map(\.url) : [r.url]
            if targets.count == 1 {
                Button("Rename…") { nameField = r.meta.name; renaming = r.url }
            } else {
                Button("Rename \(targets.count) Recordings…") { nameField = ""; renamingMany = targets }
            }
            Button(targets.count == 1 ? "Set Instruction…" : "Set Instruction for \(targets.count)…") { nameField = ""; instructionFor = targets }
            Menu(targets.count == 1 ? "Move To" : "Move \(targets.count) To") { moveTargets(for: targets) }
            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting(targets) }
            Divider()
            Button(targets.count == 1 ? "Move to Trash…" : "Move \(targets.count) to Trash…", role: .destructive) { confirmTrash = targets }
        }
    }

    @ViewBuilder
    func moveTargets(for urls: [URL]) -> some View {
        ForEach(lib.flatFolders, id: \.url) { t in
            Button(String(repeating: "    ", count: t.depth) + t.name) { withAnimation(Theme.spring) { lib.move(urls, into: t.url); expanded.insert(t.url) } }
                .disabled(urls.allSatisfy { u in t.url == u.deletingLastPathComponent().standardizedFileURL || t.url.path.hasPrefix(u.path + "/") || t.url == u })
        }
    }

    // MARK: Detail

    @ViewBuilder
    private var detail: some View {
        let sel = selected
        ZStack {
            if sel.count == 1, let r = sel.first {
                RecordingDetailView(app: app, recording: r,
                                    onRename: { nameField = r.meta.name; renaming = r.url },
                                    onTrash: { confirmTrash = [r.url] })
                    .id(r.url)
                    .transition(.opacity)
            } else if sel.count > 1 {
                VStack(spacing: 18) {
                    ZStack {
                        ForEach(0 ..< min(3, sel.count), id: \.self) { i in
                            AsyncThumb(url: sel[i].thumbnail).frame(width: 150, height: 86)
                                .rotationEffect(.degrees(Double(i - 1) * 6))
                                .offset(x: CGFloat(i - 1) * 26)
                                .shadow(color: .black.opacity(0.6), radius: 12, y: 6)
                        }
                    }
                    .frame(height: 120)
                    Text("\(sel.count) recordings selected").displayStyle(32)
                    Text("\(sel.reduce(0) { $0 + $1.meta.duration }.durationText) in total").font(Theme.font(14)).foregroundStyle(Theme.muted)
                    HStack(spacing: 8) {
                        Menu { moveTargets(for: sel.map(\.url)) } label: { Text("Move to…") }.menuStyle(.button).buttonStyle(.atSecondary).fixedSize()
                        Button("Rename…") { nameField = ""; renamingMany = sel.map(\.url) }.buttonStyle(.atSecondary)
                        Button("Set Instruction…") { nameField = ""; instructionFor = sel.map(\.url) }.buttonStyle(.atSecondary)
                        Button("Move to Trash…") { confirmTrash = sel.map(\.url) }.buttonStyle(.atDestructive)
                    }
                    Text("⌘-click or ⇧-click to change the selection.").font(Theme.font(12)).foregroundStyle(Theme.faint)
                }
                .padding(32)
                .transition(.opacity.combined(with: .scale(scale: 0.98)))
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "film").font(.system(size: 36, weight: .light)).foregroundStyle(Theme.faint)
                    Text("No recording selected").font(Theme.font(17, .semibold))
                    Text("Open a folder and pick a recording to play it and edit its instructions.").font(Theme.font(13)).foregroundStyle(Theme.muted)
                }
                .transition(.opacity)
            }
        }
        .animation(Theme.spring, value: sel.map(\.url))
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
                Text(recording.meta.name).font(Theme.font(13, .medium)).lineLimit(1)
                Text("\(recording.meta.duration.durationText) · \(recording.meta.target.displayName)").font(Theme.font(12)).foregroundStyle(Theme.muted).lineLimit(1)
            }
            Spacer()
        }
        .padding(.vertical, 2)
    }
}

struct AsyncThumb: View {
    let url: URL
    @State private var image: NSImage?
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 6, style: .continuous).fill(Theme.raised)
            if let image {
                Image(nsImage: image).resizable().aspectRatio(contentMode: .fill).transition(.opacity)
            } else {
                Image(systemName: "film").font(.system(size: 11)).foregroundStyle(Theme.faint)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).strokeBorder(Color.white.opacity(0.06)))
        .task(id: url) {
            let img = await Task.detached { NSImage(contentsOf: url) }.value
            withAnimation(.easeOut(duration: 0.25)) { image = img }
        }
    }
}
