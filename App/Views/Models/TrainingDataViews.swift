import ATCore
import SwiftUI

/// A checkbox: 0 off, 1 some (a folder with some subfolders chosen), 2 on.
struct TriCheck: View {
    let state: Int
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 5, style: .continuous).fill(state > 0 ? Theme.text : Color.clear)
            RoundedRectangle(cornerRadius: 5, style: .continuous).strokeBorder(state > 0 ? Theme.text : Theme.faint, lineWidth: 1.5)
            if state == 2 { Image(systemName: "checkmark").font(.system(size: 9, weight: .heavy)).foregroundStyle(.black) }
            if state == 1 { Capsule().fill(.black).frame(width: 8, height: 2) }
        }
        .frame(width: 16, height: 16)
    }
}

/// What a set of chosen folders adds up to.
struct TrainingDataTotals {
    var recordings = 0
    var seconds = 0.0
    var corrections = 0

    init(_ recs: [LibraryRecording]) {
        recordings = recs.count
        seconds = recs.reduce(0) { $0 + $1.meta.duration }
        corrections = recs.filter(\.meta.isCorrectionRun).count
    }

    @MainActor
    init(folders: [String]?, library: LibraryStore) {
        let chosen = Set(folders ?? [])
        self.init(library.all.filter { library.folderKey($0.folder).map(chosen.contains) ?? false })
    }

    var line: String {
        guard recordings > 0 else { return "No recordings chosen" }
        var s = "\(recordings) recording\(recordings == 1 ? "" : "s") · \(seconds.durationText)"
        if corrections > 0 { s += " · \(corrections) correction\(corrections == 1 ? "" : "s")" }
        return s
    }
}

/// Models → Data: the library folders a model learns from. Whole folders only, so anything
/// recorded into them later (corrections included) is used next time it trains.
struct TrainingDataSection: View {
    let app: AppModel
    @Binding var folders: [String]?
    let validationFraction: Double
    @State private var collapsed = Set<String>()

    private struct Node: Identifiable {
        let key: String
        let name: String
        let depth: Int
        var id: String { key }
    }

    var body: some View {
        let lib = app.library
        let nodes = self.nodes(lib)
        let chosen = Set(folders ?? [])
        let byFolder = Dictionary(grouping: lib.all) { lib.folderKey($0.folder) ?? "\u{0}" }
        let totals = TrainingDataTotals(chosen.flatMap { byFolder[$0] ?? [] })
        VStack(alignment: .leading, spacing: 18) {
            Panel {
                HStack(alignment: .bottom, spacing: 32) {
                    BigStat(value: "\(totals.recordings)", caption: totals.recordings == 1 ? "recording" : "recordings", size: 30)
                    BigStat(value: totals.seconds.durationText, caption: "of gameplay", size: 30)
                    BigStat(value: "\(totals.corrections)", caption: totals.corrections == 1 ? "correction" : "corrections", size: 30,
                            color: totals.corrections > 0 ? Theme.correction : Theme.text)
                    Spacer(minLength: 0)
                    Text("\(Int(validationFraction * 100))% held out to check progress")
                        .font(Theme.font(12)).foregroundStyle(Theme.muted).multilineTextAlignment(.trailing)
                }
            }
            if nodes.count <= 1 && lib.all.isEmpty {
                Panel {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("No recordings yet").font(Theme.font(14, .semibold))
                        Text("Record some gameplay in the Record tab, then choose its folder here.").font(Theme.font(13)).foregroundStyle(Theme.muted)
                    }
                }
            } else {
                Rows(title: "Folders to learn from",
                     footer: "The model learns from every recording directly in the chosen folders, including ones you add later. Checking a folder also checks its subfolders.") {
                    ForEach(visible(nodes)) { n in row(n, nodes: nodes, chosen: chosen, byFolder: byFolder) }
                }
            }
        }
    }

    private func nodes(_ lib: LibraryStore) -> [Node] {
        var out: [Node] = []
        if lib.tree.recordingCount > 0 || (folders ?? []).contains("") { out.append(Node(key: "", name: "Unsorted", depth: 0)) }
        func walk(_ f: LibraryFolder, _ depth: Int) {
            guard let k = lib.folderKey(f.url) else { return }
            out.append(Node(key: k, name: f.name, depth: depth))
            f.children.forEach { walk($0, depth + 1) }
        }
        lib.tree.children.forEach { walk($0, 0) }
        return out
    }

    /// The folder and every folder below it ("" — Unsorted — is only itself).
    private func subtree(_ key: String, _ nodes: [Node]) -> [String] {
        key.isEmpty ? [""] : [key] + nodes.map(\.key).filter { $0.hasPrefix(key + "/") }
    }

    private func visible(_ nodes: [Node]) -> [Node] {
        nodes.filter { n in !collapsed.contains { !$0.isEmpty && n.key.hasPrefix($0 + "/") } }
    }

    private func row(_ n: Node, nodes: [Node], chosen: Set<String>, byFolder: [String: [LibraryRecording]]) -> some View {
        let sub = subtree(n.key, nodes)
        let on = sub.filter(chosen.contains).count
        let state = on == 0 ? 0 : on == sub.count ? 2 : 1
        let recs = sub.flatMap { byFolder[$0] ?? [] }
        let hasChildren = sub.count > 1
        let open = !collapsed.contains(n.key)
        return HStack(spacing: 10) {
            Button {
                withAnimation(Theme.snappy) {
                    var s = Set(folders ?? [])
                    if state == 2 { s.subtract(sub) } else { s.formUnion(sub) }
                    folders = s.sorted()
                }
            } label: {
                HStack(spacing: 10) {
                    TriCheck(state: state)
                    Image(systemName: n.key.isEmpty ? "tray" : "folder").font(.system(size: 13)).foregroundStyle(Theme.text2)
                    Text(n.name).font(Theme.font(14, n.depth == 0 ? .medium : .regular)).foregroundStyle(Theme.text).lineLimit(1)
                    Spacer(minLength: 8)
                    Text(recs.isEmpty ? "empty" : "\(recs.count) · \(recs.reduce(0) { $0 + $1.meta.duration }.durationText)")
                        .font(Theme.font(12)).monospacedDigit().foregroundStyle(Theme.muted)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            Button {
                withAnimation(Theme.snappy) { if open { collapsed.insert(n.key) } else { collapsed.remove(n.key) } }
            } label: {
                Image(systemName: "chevron.right").font(.system(size: 11, weight: .semibold)).foregroundStyle(Theme.muted)
                    .rotationEffect(.degrees(open ? 90 : 0)).frame(width: 18, height: 18).contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .opacity(hasChildren ? 1 : 0)
            .disabled(!hasChildren)
        }
        .padding(.leading, CGFloat(n.depth) * 22)
    }
}

/// Training tab: which folders the selected model learns from (chosen in Models → Data).
struct TrainingDataSummary: View {
    let app: AppModel
    let entry: ModelEntry
    var body: some View {
        let folders = (entry.definition.dataFolders ?? []).sorted()
        let totals = TrainingDataTotals(folders: folders, library: app.library)
        Panel(padding: 20) {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .firstTextBaseline) {
                    Text("Training data").font(Theme.font(14, .semibold))
                    Text(totals.line).font(Theme.font(12)).foregroundStyle(Theme.muted).contentTransition(.numericText())
                    Spacer()
                    Button("Choose folders…") {
                        app.selectedModelID = entry.id
                        app.pendingModelSection = .data
                        app.tab = .models
                    }
                    .buttonStyle(.atSmall)
                    .disabled(app.training.isActive && app.training.jobModelID == entry.id)
                }
                if folders.isEmpty {
                    Text("No folders chosen yet. Pick the recordings this model learns from in its settings.")
                        .font(Theme.font(13)).foregroundStyle(Theme.muted)
                } else {
                    FolderChips(items: folders.map { $0.isEmpty ? "Unsorted" : $0 })
                }
            }
        }
    }
}

/// Folder names as small chips that wrap.
private struct FolderChips: View {
    let items: [String]
    var body: some View {
        WrapLayout(spacing: 6) {
            ForEach(items, id: \.self) { name in
                Label(name, systemImage: name == "Unsorted" ? "tray" : "folder")
                    .font(Theme.font(12, .medium)).foregroundStyle(Theme.text2).lineLimit(1)
                    .padding(.horizontal, 10).padding(.vertical, 5)
                    .background(Theme.raised, in: Capsule())
            }
        }
    }
}

private struct WrapLayout: Layout {
    var spacing: CGFloat
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = arrange(width: proposal.width ?? .infinity, subviews)
        return CGSize(width: proposal.width ?? rows.map(\.width).max() ?? 0, height: rows.last.map { $0.y + $0.height } ?? 0)
    }
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        for row in arrange(width: bounds.width, subviews) {
            var x = bounds.minX
            for i in row.items {
                let s = subviews[i].sizeThatFits(.unspecified)
                subviews[i].place(at: CGPoint(x: x, y: bounds.minY + row.y), proposal: ProposedViewSize(s))
                x += s.width + spacing
            }
        }
    }
    private func arrange(width: CGFloat, _ subviews: Subviews) -> [(items: [Int], y: CGFloat, width: CGFloat, height: CGFloat)] {
        var rows: [(items: [Int], y: CGFloat, width: CGFloat, height: CGFloat)] = []
        var cur: [Int] = [], x: CGFloat = 0, h: CGFloat = 0, y: CGFloat = 0
        for i in subviews.indices {
            let s = subviews[i].sizeThatFits(.unspecified)
            if !cur.isEmpty, x + s.width > width {
                rows.append((cur, y, x - spacing, h)); y += h + spacing; cur = []; x = 0; h = 0
            }
            cur.append(i); x += s.width + spacing; h = max(h, s.height)
        }
        if !cur.isEmpty { rows.append((cur, y, x - spacing, h)) }
        return rows
    }
}
