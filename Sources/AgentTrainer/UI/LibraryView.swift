import AppKit
import AVKit
import SwiftUI

struct LibraryView: View {
    private enum SortOrder: String, CaseIterable, Identifiable {
        case newest = "Newest"
        case oldest = "Oldest"
        case name = "Name"
        case duration = "Duration"
        var id: String { rawValue }
    }

    @ObservedObject var model: AppModel
    @State private var search = ""
    @State private var sortOrder: SortOrder = .newest
    @State private var folderName = ""
    @State private var renameText = ""
    @State private var player: AVPlayer?
    @State private var summary = InputEventReader.Summary(preview: [], usedKeyCodes: [], keyEventCount: 0, mouseEventCount: 0)
    @State private var reenactmentArmed = false
    @State private var loadingDetails = false
    @State private var detailTask: Task<Void, Never>?
    @State private var deleteRecording: RecordingItem?
    @State private var deleteFolder: RecordingFolder?
    @State private var folderToRename: RecordingFolder?
    @State private var folderRenameText = ""
    @State private var expandedFolderIDs: Set<UUID> = []
    @State private var showsTimeline = false

    private var visibleRecordings: [RecordingItem] {
        var result = model.recordings.filter { item in
            (search.isEmpty || item.manifest.name.localizedCaseInsensitiveContains(search))
        }
        switch sortOrder {
        case .newest: result.sort { $0.manifest.createdAt > $1.manifest.createdAt }
        case .oldest: result.sort { $0.manifest.createdAt < $1.manifest.createdAt }
        case .name: result.sort { $0.manifest.name.localizedCaseInsensitiveCompare($1.manifest.name) == .orderedAscending }
        case .duration: result.sort { $0.manifest.duration > $1.manifest.duration }
        }
        return result
    }

    private func recordings(in folder: RecordingFolder) -> [RecordingItem] {
        visibleRecordings.filter { $0.manifest.folderID == folder.id }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            header
            HStack(alignment: .top, spacing: 14) {
                folderBrowser
                inspector
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .padding(24)
        .onAppear { ensureSelection(); if let item = model.selectedRecording { load(item) } }
        .onChange(of: model.selectedRecordingID) { _, _ in if let item = model.selectedRecording { load(item) } }
        .onDisappear { detailTask?.cancel(); player?.pause() }
        .alert("Delete recording?", isPresented: Binding(get: { deleteRecording != nil }, set: { if !$0 { deleteRecording = nil } })) {
            Button("Cancel", role: .cancel) { deleteRecording = nil }
            Button("Delete", role: .destructive) { if let item = deleteRecording { Task { await model.deleteRecording(item) } }; deleteRecording = nil }
        } message: { Text("This permanently removes “\(deleteRecording?.manifest.name ?? "")”, its video, and recorded inputs.") }
        .alert("Delete folder and recordings?", isPresented: Binding(get: { deleteFolder != nil }, set: { if !$0 { deleteFolder = nil } })) {
            Button("Cancel", role: .cancel) { deleteFolder = nil }
            Button("Delete Folder", role: .destructive) { if let folder = deleteFolder { Task { await model.deleteRecordingFolder(folder) } }; deleteFolder = nil }
        } message: { Text("Every recording inside “\(deleteFolder?.name ?? "")” will be permanently removed.") }
        .alert("Rename folder", isPresented: Binding(get: { folderToRename != nil }, set: { if !$0 { folderToRename = nil } })) {
            TextField("Folder name", text: $folderRenameText)
            Button("Cancel", role: .cancel) { folderToRename = nil }
            Button("Rename") {
                if let folder = folderToRename { Task { await model.renameRecordingFolder(folder, name: folderRenameText) } }
                folderToRename = nil
            }
        }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            SectionTitle("Library", "Expand folders to browse recordings, inspect captured controls, and replay locally.")
            TextField("Search recordings", text: $search).textFieldStyle(.roundedBorder).frame(width: 240)
            Picker("Sort", selection: $sortOrder) { ForEach(SortOrder.allCases) { Text($0.rawValue).tag($0) } }.frame(width: 130)
            Button { Task { await model.refreshLibrary(); ensureSelection() } } label: { Image(systemName: "arrow.clockwise") }.primaryButton()
        }
    }

    private var folderBrowser: some View {
        OLEDCard(padding: 12) {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Label("Recording folders", systemImage: "folder.fill").font(.headline).foregroundStyle(ATColor.violet)
                    Spacer()
                    Text("\(visibleRecordings.count) recording\(visibleRecordings.count == 1 ? "" : "s")").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                HStack {
                    TextField("New folder", text: $folderName).textFieldStyle(.roundedBorder)
                    Button("Create") {
                        let name = folderName.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !name.isEmpty else { return }
                        folderName = ""; Task { await model.createRecordingFolder(name: name) }
                    }.primaryButton()
                }
                Divider()
                List {
                    ForEach(model.recordingFolders) { folder in
                        let items = recordings(in: folder)
                        DisclosureGroup(isExpanded: Binding(
                            get: { expandedFolderIDs.contains(folder.id) || !search.isEmpty },
                            set: { expanded in if expanded { expandedFolderIDs.insert(folder.id) } else { expandedFolderIDs.remove(folder.id) } }
                        )) {
                            if items.isEmpty {
                                Text(search.isEmpty ? "No recordings in this folder" : "No matching recordings")
                                    .font(.caption).foregroundStyle(.tertiary).padding(.vertical, 8)
                            } else {
                                ForEach(items) { item in
                                    HStack(spacing: 5) {
                                        Button { model.selectedRecordingID = item.id } label: {
                                            LibraryRecordingRow(item: item, selected: model.selectedRecordingID == item.id)
                                        }
                                        .buttonStyle(.plain)
                                        .frame(maxWidth: .infinity)
                                        Button { deleteRecording = item } label: {
                                            Image(systemName: "trash")
                                                .frame(width: 30, height: 30)
                                                .contentShape(Rectangle())
                                        }
                                        .buttonStyle(.borderless)
                                        .foregroundStyle(ATColor.coral)
                                        .help("Delete this recording")
                                    }
                                }
                            }
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: "folder.fill").foregroundStyle(ATColor.violet)
                                Text(folder.name).font(.headline).lineLimit(1)
                                Text("\(model.recordings.count { $0.manifest.folderID == folder.id })").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                                Spacer()
                                Menu {
                                    Button("Use for new recordings") { model.recordingDestinationFolderID = folder.id }
                                    Button("Rename folder") { folderRenameText = folder.name; folderToRename = folder }
                                    Button("Delete folder and recordings", role: .destructive) { deleteFolder = folder }
                                } label: { Image(systemName: "ellipsis.circle").foregroundStyle(.secondary) }.menuStyle(.borderlessButton).fixedSize()
                            }
                        }
                        .tint(ATColor.cyan)
                    }
                    if model.recordingFolders.isEmpty {
                        ContentUnavailableView("No recording folders", systemImage: "folder.badge.plus")
                    }
                    if !search.isEmpty && visibleRecordings.isEmpty {
                        ContentUnavailableView("No matching recordings", systemImage: "magnifyingglass")
                    }
                }
                .listStyle(.inset)
                .scrollContentBackground(.hidden)
            }
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder private var inspector: some View {
        if let item = model.selectedRecording {
            OLEDCard(padding: 13) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        VideoPlayer(player: player)
                            .aspectRatio(CGFloat(item.manifest.pixelWidth) / CGFloat(max(1, item.manifest.pixelHeight)), contentMode: .fit)
                            .frame(maxWidth: .infinity, minHeight: 150).background(Color.black).clipShape(RoundedRectangle(cornerRadius: ATCorner.scaled(10), style: .continuous))
                        TextField("Recording name", text: $renameText).font(.title3.bold()).textFieldStyle(.plain)
                            .onSubmit { Task { await model.renameRecording(item, name: renameText) } }
                        HStack { StatusPill(text: durationString(effectiveDuration(item)), color: ATColor.cyan); StatusPill(text: "\(item.manifest.pixelWidth)×\(item.manifest.pixelHeight)", color: ATColor.violet); StatusPill(text: "\(Int(item.manifest.deliveredFPS.rounded())) FPS", color: ATColor.green) }
                        HStack { Label("\(item.manifest.eventCount) inputs", systemImage: "cursorarrow.click.2"); Spacer(); Label(item.manifest.capture.kind.rawValue, systemImage: "viewfinder") }.font(.caption).foregroundStyle(.secondary)
                        if item.manifest.trimStart > 0 || (item.manifest.trimEnd ?? item.manifest.duration) < item.manifest.duration {
                            Label("Uses \(item.manifest.trimStart.formatted(.number.precision(.fractionLength(2))))s – \((item.manifest.trimEnd ?? item.manifest.duration).formatted(.number.precision(.fractionLength(2))))s", systemImage: "scissors").font(.caption).foregroundStyle(ATColor.amber)
                        }
                        Divider()
                        HStack { Text("Recorded keyboard").font(.subheadline.bold()); Spacer(); if loadingDetails { ProgressView().controlSize(.small) } else { Text("\(summary.usedKeyCodes.count) keys").font(.caption).foregroundStyle(.secondary) } }
                        Text(summary.usedKeyCodes.sorted().map(KeyNames.name).joined(separator: "  "))
                            .font(.caption.bold()).foregroundStyle(ATColor.cyan)
                        HStack { Label("\(summary.keyEventCount) key events", systemImage: "keyboard"); Label("\(summary.mouseEventCount) pointer events", systemImage: "computermouse") }.font(.caption2).foregroundStyle(.secondary)
                        if summary.mouse.moveEventCount > 0 {
                            HStack(spacing: 7) {
                                StatusPill(text: summary.mouse.isGameCamera ? "Game camera detected" : "Moving cursor detected", color: summary.mouse.isGameCamera ? ATColor.green : ATColor.cyan)
                                StatusPill(text: summary.mouse.positionsAreValid ? "Positions valid" : "\(summary.mouse.outOfCaptureBoundsCount) invalid positions", color: summary.mouse.positionsAreValid ? ATColor.green : ATColor.coral)
                            }
                            Text("Raw delta active on \((summary.mouse.nonzeroDeltaFraction * 100).formatted(.number.precision(.fractionLength(1))))% of move samples • mean active |Δ| \(summary.mouse.meanActiveDeltaMagnitude.formatted(.number.precision(.fractionLength(2)))) px • max \(summary.mouse.maximumDeltaMagnitude.formatted(.number.precision(.fractionLength(1)))) px")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                        Button {
                            showsTimeline.toggle()
                        } label: {
                            HStack {
                                Text("Input timeline preview").font(.subheadline)
                                Spacer()
                                Image(systemName: showsTimeline ? "chevron.up" : "chevron.down").font(.caption.bold())
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(ATColor.cyan)
                        if showsTimeline {
                            ScrollView(.vertical) {
                                VStack(spacing: 4) {
                                ForEach(Array(summary.preview.enumerated()), id: \.offset) { _, event in
                                    HStack { Text(eventLabel(event)).lineLimit(1); Spacer(); Text(eventTime(event, manifest: item.manifest)).font(.caption2.monospacedDigit()).foregroundStyle(.secondary) }.font(.caption).padding(.vertical, 1)
                                }
                                }
                                .padding(8)
                            }
                            .frame(height: 190)
                            .raisedGlassSurface(cornerRadius: 9)
                        }
                        Divider()
                        Picker("Folder", selection: Binding(get: { item.manifest.folderID }, set: { folder in Task { await model.moveRecording(item, to: folder) } })) { ForEach(model.recordingFolders) { Text($0.name).tag(Optional($0.id)) } }
                        Toggle("Enable real-input reenactment", isOn: $reenactmentArmed).font(.caption)
                        HStack {
                            if model.isReplaying { Button("Stop Replay") { model.stopReenactment() }.primaryButton(color: ATColor.coral) }
                            else { Button("Reenact") { model.startReenactment() }.primaryButton(color: ATColor.amber).disabled(!reenactmentArmed) }
                            Spacer(); Button("Delete", role: .destructive) { deleteRecording = item }.primaryButton(color: ATColor.coral)
                        }
                    }
                }
            }.frame(width: 380).frame(maxHeight: .infinity)
        } else {
            OLEDCard { ContentUnavailableView("Select a recording", systemImage: "play.rectangle") }.frame(width: 380).frame(maxHeight: .infinity)
        }
    }

    private func ensureSelection() {
        if model.selectedRecording == nil { model.selectedRecordingID = visibleRecordings.first?.id }
    }

    private func load(_ item: RecordingItem) {
        detailTask?.cancel(); player?.pause()
        showsTimeline = false
        renameText = item.manifest.name
        player = AVPlayer(url: item.directory.appendingPathComponent(item.manifest.videoFile))
        summary = .init(preview: [], usedKeyCodes: [], keyEventCount: 0, mouseEventCount: 0)
        loadingDetails = true
        let id = item.id, url = item.directory.appendingPathComponent(item.manifest.eventFile)
        detailTask = Task {
            let rect = item.manifest.globalRect.cgRect
            let value = await Task.detached(priority: .utility) { try? InputEventReader.summarize(url: url, globalRect: rect) }.value
            guard !Task.isCancelled, model.selectedRecordingID == id else { return }
            summary = value ?? .init(preview: [], usedKeyCodes: [], keyEventCount: 0, mouseEventCount: 0)
            loadingDetails = false
        }
    }

    private func effectiveDuration(_ item: RecordingItem) -> Double { max(0, (item.manifest.trimEnd ?? item.manifest.duration) - item.manifest.trimStart) }
    private func durationString(_ seconds: Double) -> String { let value = Int(ceil(seconds)); return String(format: "%d:%02d", value / 60, value % 60) }
    private func eventLabel(_ event: InputSample) -> String { switch event.kind { case .key: "\(KeyNames.name(for: event.keyCode)) \(event.isDown ? "down" : "up")"; case .mouseButton: "Mouse \(Int(event.button) + 1) \(event.isDown ? "down" : "up")"; case .mouseMove: "Mouse Δ\(Int(event.deltaX)), \(Int(event.deltaY))"; case .scroll: "Scroll \(Int(event.scrollY))"; case .flags: "Modifiers" } }
    private func eventTime(_ event: InputSample, manifest: RecordingManifest) -> String { guard event.timestampNanos >= manifest.hostStartNanos else { return "—" }; return String(format: "%.3fs", Double(event.timestampNanos - manifest.hostStartNanos) / 1e9) }
}

private struct LibraryRecordingRow: View {
    let item: RecordingItem
    let selected: Bool

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "play.rectangle.fill")
                .font(.title2).foregroundStyle(selected ? ATColor.cyan : ATColor.violet)
                .frame(width: 38, height: 32)
                .background(RoundedRectangle(cornerRadius: ATCorner.scaled(8), style: .continuous).fill(ATColor.raised))
            VStack(alignment: .leading, spacing: 3) {
                Text(item.manifest.name).font(.subheadline.bold()).lineLimit(1)
                HStack(spacing: 10) {
                    Label(duration, systemImage: "clock")
                    Label("\(item.manifest.pixelWidth)×\(item.manifest.pixelHeight)", systemImage: "rectangle.inset.filled")
                    Label("\(item.manifest.eventCount)", systemImage: "cursorarrow.click.2")
                }.font(.caption2).foregroundStyle(.secondary)
            }
            Spacer()
            Text(item.manifest.createdAt.formatted(date: .abbreviated, time: .omitted)).font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: ATCorner.scaled(14), style: .continuous).fill(selected ? ATColor.cyan.opacity(0.12) : ATColor.raised.opacity(0.58)))
        .overlay(RoundedRectangle(cornerRadius: ATCorner.scaled(14), style: .continuous).stroke(selected ? ATColor.cyan : ATColor.border, lineWidth: selected ? 1.2 : 0.7))
        .uiHoverResponse(scale: 1.006)
    }

    private var duration: String { let seconds = Int(ceil(max(0, (item.manifest.trimEnd ?? item.manifest.duration) - item.manifest.trimStart))); return String(format: "%d:%02d", seconds / 60, seconds % 60) }
}
