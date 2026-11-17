import ATCore
import ATModel
import MLX
import SwiftUI

struct ModelsView: View {
    @Bindable var app: AppModel
    @State private var showingNew = false

    var body: some View {
        HStack(spacing: 0) {
            modelList.frame(width: 300).riseIn()
            Rectangle().fill(Theme.hairline).frame(width: 1)
            ZStack {
                if let e = app.models.entry(app.selectedModelID) {
                    ModelEditor(app: app, entry: e).id(e.id)
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "brain").font(.system(size: 36, weight: .light)).foregroundStyle(Theme.faint)
                        Text("No model selected").font(Theme.font(17, .semibold))
                        Text("Create a model to train on your recordings.").font(Theme.font(13)).foregroundStyle(Theme.muted)
                        Button("New Model…") { showingNew = true }.buttonStyle(.atPrimary).padding(.top, 6)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .sheet(isPresented: $showingNew) { NewModelSheet(app: app) }
        .onAppear { if app.selectedModelID == nil { app.selectedModelID = app.models.models.first?.id } }
    }

    private var modelList: some View {
        VStack(spacing: 0) {
            ScrollView {
                LazyVStack(spacing: 2) {
                    ForEach(app.models.models) { m in
                        let on = app.selectedModelID == m.id
                        Button { withAnimation(Theme.snappy) { app.selectedModelID = m.id } } label: {
                            HStack(spacing: 10) {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(m.definition.name).font(Theme.font(14, .semibold)).lineLimit(1)
                                    HStack(spacing: 6) {
                                        if app.training.isActive && app.training.jobModelID == m.id { LiveDot(size: 5) }
                                        Text(subtitle(m)).font(Theme.font(12)).foregroundStyle(Theme.muted).lineLimit(1)
                                    }
                                }
                                Spacer(minLength: 4)
                                Text(String(format: "%.0fM", Double(m.definition.parameterCount) / 1e6)).font(Theme.font(12)).monospacedDigit().foregroundStyle(Theme.muted)
                            }
                            .padding(.horizontal, 14).padding(.vertical, 11)
                        }
                        .buttonStyle(RowButtonStyle(selected: on, radius: 12))
                        .contextMenu {
                            Button("Duplicate") { app.selectedModelID = app.models.duplicate(m)?.id }
                            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([m.url]) }
                            Divider()
                            Button("Move to Trash", role: .destructive) { app.models.trash(m) }
                        }
                    }
                }
                .padding(10)
            }
            .scrollIndicators(.never)
            Rectangle().fill(Theme.hairline).frame(height: 1)
            Button { showingNew = true } label: { Label("New Model", systemImage: "plus").frame(maxWidth: .infinity) }
                .buttonStyle(.atPrimary)
                .padding(14)
        }
    }

    func subtitle(_ m: ModelEntry) -> String {
        if m.checkpoints.contains("best"), let st = ModelPackage.checkpointState(m.url, name: "best") { return "\(m.definition.trainedSteps) steps · best \(st.step)" }
        if m.definition.isTrained { return "\(m.definition.trainedSteps) steps trained" }
        return m.definition.initialization == .blank ? "Not trained" : "Pretrained · not trained"
    }
}

// MARK: New model

struct NewModelSheet: View {
    let app: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var name = "My Agent"
    @State private var preset = ModelDefinition.Preset.gameP2P150

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("New model").displayStyle(30)
            TextField("Name", text: $name).textFieldStyle(DarkFieldStyle())
            ScrollView {
                VStack(spacing: 8) {
                    ForEach(ModelDefinition.Preset.allCases) { p in
                        let on = preset == p
                        Button { withAnimation(Theme.snappy) { preset = p } } label: {
                            HStack(alignment: .top, spacing: 12) {
                                ZStack {
                                    Circle().strokeBorder(on ? Theme.text : Theme.faint, lineWidth: on ? 5 : 1.5).frame(width: 18, height: 18)
                                }
                                .padding(.top, 2)
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(p.title).font(Theme.font(14, .semibold))
                                    Text(p.subtitle).font(Theme.font(12.5)).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true).multilineTextAlignment(.leading)
                                }
                                Spacer(minLength: 0)
                            }
                            .padding(14)
                        }
                        .buttonStyle(RowButtonStyle(selected: on, radius: 12))
                    }
                }
            }
            .frame(maxHeight: 420)
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.buttonStyle(.atSecondary).keyboardShortcut(.cancelAction)
                Button("Create") {
                    if let e = app.models.create(ModelDefinition.make(preset, name: name)) { app.selectedModelID = e.id }
                    dismiss()
                }
                .buttonStyle(.atPrimary)
                .keyboardShortcut(.defaultAction)
                .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(28)
        .frame(width: 540)
        .background(Theme.surface)
        .preferredColorScheme(.dark)
    }
}

// MARK: Editor

enum ModelSection: String, CaseIterable { case general = "General", network = "Network", visionLanguage = "Vision & language", actions = "Actions", training = "Training" }

struct ModelEditor: View {
    let app: AppModel
    let entry: ModelEntry
    @State private var def: ModelDefinition
    @State private var section = ModelSection.general
    @State private var bench: String?
    @State private var benchmarking = false

    init(app: AppModel, entry: ModelEntry) {
        self.app = app
        self.entry = entry
        _def = State(initialValue: entry.definition)
    }

    private var locked: Bool { def.architectureLocked }
    private var shapeLocked: Bool { locked || def.initialization != .blank }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header.padding(.horizontal, 40).padding(.top, 28)
            Segmented(options: ModelSection.allCases.map { ($0, $0.rawValue) }, selection: $section)
                .padding(.horizontal, 40).padding(.top, 22).padding(.bottom, 4)
            ScrollView {
                Group {
                    switch section {
                    case .general: general
                    case .network: network
                    case .visionLanguage: visionLanguage
                    case .actions: actionsSection
                    case .training: TrainingPanels(training: $def.training)
                    }
                }
                .id(section)
                .transition(.opacity.combined(with: .offset(y: 8)))
                .padding(.horizontal, 40).padding(.top, 18).padding(.bottom, 40)
            }
            .scrollIndicators(.never)
            .animation(Theme.spring, value: section)
        }
        .onChange(of: def) { _, d in app.models.save(d) }
    }

    private var header: some View {
        HStack(alignment: .bottom, spacing: 16) {
            VStack(alignment: .leading, spacing: 10) {
                Text(def.name.isEmpty ? "Untitled" : def.name).displayStyle(42).lineLimit(1)
                HStack(spacing: 14) {
                    Text(String(format: "%.0fM parameters", Double(def.parameterCount) / 1e6))
                    Text(def.isTrained ? "\(def.trainedSteps) steps trained" : "Not trained yet")
                    if let st = ModelPackage.checkpointState(entry.url, name: "best") { Pill(text: "Best · step \(st.step)") }
                }
                .font(Theme.font(13)).foregroundStyle(Theme.muted)
            }
            Spacer()
            Button("Duplicate") { app.selectedModelID = app.models.duplicate(entry)?.id }.buttonStyle(.atSmall)
            Button { app.selectedModelID = entry.id; app.tab = .training } label: { Label("Train", systemImage: "play.fill") }.buttonStyle(.atSmallPrimary)
        }
    }

    // MARK: General

    private var general: some View {
        VStack(alignment: .leading, spacing: 18) {
            if locked {
                HStack(spacing: 10) {
                    Image(systemName: "lock.fill")
                    Text("Network, vision, memory and actions are fixed once training starts. Duplicate the model to change them.")
                }
                .font(Theme.font(13)).foregroundStyle(Theme.warning)
                .padding(.horizontal, 16).padding(.vertical, 12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Theme.warning.opacity(0.1), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            }
            HStack(alignment: .top, spacing: 18) {
                Panel {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Starting point").font(Theme.font(14, .semibold))
                        ForEach(ModelInitialization.allCases, id: \.self) { i in
                            let on = def.initialization == i
                            Button {
                                withAnimation(Theme.snappy) { def.initialization = i; applyInitialization(i) }
                            } label: {
                                HStack(alignment: .top, spacing: 12) {
                                    Circle().strokeBorder(on ? Theme.text : Theme.faint, lineWidth: on ? 5 : 1.5).frame(width: 16, height: 16).padding(.top, 2)
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(title(i)).font(Theme.font(14, .semibold))
                                        Text(detail(i)).font(Theme.font(12)).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
                                    }
                                    Spacer(minLength: 0)
                                }
                                .padding(12)
                            }
                            .buttonStyle(RowButtonStyle(selected: on, radius: 12))
                            .disabled(locked)
                        }
                        if def.initialization != .blank { PretrainedStatus(app: app, initialization: def.initialization).padding(.top, 4) }
                    }
                }
                Panel {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("About").font(Theme.font(14, .semibold))
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Name").font(Theme.font(12)).foregroundStyle(Theme.muted)
                            TextField("Name", text: $def.name).textFieldStyle(DarkFieldStyle())
                        }
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Notes").font(Theme.font(12)).foregroundStyle(Theme.muted)
                            TextField("What is this model for?", text: $def.notes, axis: .vertical).textFieldStyle(DarkFieldStyle()).lineLimit(3 ... 6)
                        }
                        HStack(spacing: 28) {
                            BigStat(value: String(format: "%.0f", Double(def.parameterCount) / 1e6), unit: "M", caption: "parameters", size: 30)
                            BigStat(value: "\(entry.checkpoints.count)", caption: "checkpoints", size: 30)
                        }
                        .padding(.top, 6)
                    }
                }
            }
        }
    }

    func title(_ i: ModelInitialization) -> String {
        switch i { case .blank: "Blank"; case .p2p150M: "Pixel2Play 150M"; case .p2p300M: "Pixel2Play 300M" }
    }

    func detail(_ i: ModelInitialization) -> String {
        switch i {
        case .blank: "Vision stem from Pixel2Play, the rest learned from your recordings."
        case .p2p150M: "Pretrained on 8,300 h of gameplay. 183M parameters."
        case .p2p300M: "Pretrained, twice as deep. 309M parameters, slower to train."
        }
    }

    // MARK: Network

    private var network: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 12) {
                SectionLabel("Size")
                HStack(spacing: 10) {
                    ForEach(Array(sizes.enumerated()), id: \.offset) { _, s in
                        let on = def.architecture.width == s.arch.width && def.architecture.layers == s.arch.layers
                        Button { withAnimation(Theme.snappy) { applySize(s.arch) } } label: {
                            VStack(alignment: .leading, spacing: 8) {
                                Text(s.size).displayStyle(28)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(s.name).font(Theme.font(13, .semibold))
                                    Text("\(s.arch.width) × \(s.arch.layers)").font(Theme.font(12)).foregroundStyle(Theme.muted)
                                }
                            }
                            .padding(16)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(on ? Theme.raised : Theme.surface, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).strokeBorder(on ? Theme.text : Theme.hairline, lineWidth: on ? 1.5 : 1))
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .disabled(shapeLocked)
                        .opacity(shapeLocked && !on ? 0.45 : 1)
                    }
                }
                if def.initialization != .blank {
                    Text("Pretrained models keep Pixel2Play's shape (\(def.initialization == .p2p300M ? 20 : 10) layers × 1024).").font(Theme.font(12)).foregroundStyle(Theme.muted)
                }
            }
            Rows(title: "Shape") {
                Row(title: "Transformer layers") { NumberStepper(value: $def.architecture.layers, range: 2 ... 32) }
                Row(title: "Width") {
                    Picker("", selection: $def.architecture.width) { ForEach([128, 192, 256, 384, 512, 768, 1024, 1536, 2048], id: \.self) { Text("\($0)").tag($0) } }
                        .labelsHidden().fixedSize()
                        .onChange(of: def.architecture.width) { _, w in def.architecture.heads = w / 64; def.architecture.kvHeads = w / 64; def.architecture.decoderHeads = max(1, w / 128) }
                }
                Row(title: "Action decoder layers") { NumberStepper(value: $def.architecture.decoderLayers, range: 1 ... 6) }
                LabeledSlider(title: "Dropout", value: $def.architecture.dropout, range: 0 ... 0.3, step: 0.01)
            }
            .disabled(shapeLocked)
            Rows(title: "Memory", footer: "\(def.architecture.contextSteps) timesteps · \(def.architecture.contextSteps * def.architecture.tokensPerStep(slots: def.actions.slotCount)) tokens in memory. Each decision only processes the new frame.") {
                Row(title: "Decisions per second") { Segmented(options: [20, 30, 40, 60].map { ($0, "\($0)") }, selection: $def.architecture.stepRateHz, compact: true) }
                LabeledSlider(title: "Memory", value: $def.architecture.memorySeconds, range: 5 ... 30, step: 1, format: { "\(Int($0)) s" })
            }
            .disabled(locked)
            Rows(title: "Speed on this Mac", footer: "One real-time step with a full memory window, random weights, bf16.") {
                Row(title: bench ?? "Not measured yet") {
                    Button(benchmarking ? "Measuring…" : "Run benchmark") { Task { await benchmark() } }.buttonStyle(.atSmall).disabled(benchmarking)
                }
            }
        }
    }

    private var sizes: [(size: String, name: String, arch: PolicyArchitecture)] {
        [("5M", "Tiny", .tiny5M), ("10M", "Small", .small10M), ("22M", "Medium", .medium22M), ("60M", "Compact", .compact60M), ("183M", "P2P 150M", .p2p150M)]
    }

    func applySize(_ a: PolicyArchitecture) {
        def.architecture.width = a.width; def.architecture.layers = a.layers; def.architecture.heads = a.heads; def.architecture.kvHeads = a.kvHeads
        def.architecture.decoderLayers = a.decoderLayers; def.architecture.decoderHeads = a.decoderHeads
    }

    // MARK: Vision & language

    private var visionLanguage: some View {
        HStack(alignment: .top, spacing: 18) {
            Rows(title: "Vision", footer: "Frames are squashed to a square like Pixel2Play; the same GPU resize is used for training and running.") {
                Row(title: "Input size") {
                    Picker("", selection: $def.architecture.frameSize) { Text("192 × 192").tag(192); Text("256 × 256").tag(256) }.labelsHidden().fixedSize()
                        .disabled(shapeLocked)
                }
                Toggle("Grayscale", isOn: $def.architecture.grayscale).toggleStyle(.oled).disabled(locked)
            }
            Rows(title: "Language", footer: "Instructions and on-screen text are embedded on device with EmbeddingGemma. Typing lets the model copy characters from an instruction like “type hello”.") {
                Toggle("Follow instructions", isOn: $def.usesInstructions.animation(Theme.spring)).toggleStyle(.oled)
                if def.usesInstructions {
                    LabeledSlider(title: "Instruction dropout", value: $def.training.instructionDropout, range: 0 ... 0.6, step: 0.05, format: { "\(Int($0 * 100))%" })
                }
                Toggle("Read on-screen text", isOn: $def.architecture.screenText.animation(Theme.spring)).toggleStyle(.oled).disabled(locked)
                if def.architecture.screenText {
                    Row(title: "Text items per frame") { NumberStepper(value: $def.architecture.screenTextItems, range: 2 ... 16) }.disabled(locked)
                }
                Toggle("Type text from instructions", isOn: $def.architecture.literalTyping).toggleStyle(.oled).disabled(locked)
            }
        }
    }

    // MARK: Actions

    private var actionsSection: some View {
        VStack(alignment: .leading, spacing: 18) {
            Panel {
                KeyVocabularyPicker(keys: $def.actions.keys).disabled(locked)
            }
            HStack(alignment: .top, spacing: 18) {
                Panel {
                    PointerVocabularyPicker(events: $def.actions.pointerEvents).disabled(locked)
                }
                Rows(title: "Mouse", footer: "The Run tab can still block any action without retraining.") {
                    Row(title: "Movement") {
                        Segmented(options: [(MouseMode.relative, "Relative"), (MouseMode.absolute, "Pointer")], selection: $def.actions.mouseMode, compact: true)
                    }
                    .disabled(locked)
                    if def.actions.mouseMode == .absolute {
                        Row(title: "Pointer grid", detail: "\(def.actions.absoluteColumns) × \(def.actions.absoluteRows)") {
                            NumberStepper(value: $def.actions.absoluteColumns, range: 32 ... 256, step: 16)
                                .onChange(of: def.actions.absoluteColumns) { _, c in def.actions.absoluteRows = c * 9 / 16 }
                        }
                        .disabled(locked)
                    }
                    Row(title: "Sampling") {
                        Segmented(options: [(MouseSampling.truncatedNormal, "Smooth"), (MouseSampling.center, "Bin center")], selection: $def.actions.mouseSampling, compact: true)
                    }
                    Row(title: "Simultaneous keys") { NumberStepper(value: $def.actions.keySlots, range: 1 ... 8) }.disabled(shapeLocked)
                }
            }
        }
    }

    func applyInitialization(_ i: ModelInitialization) {
        switch i {
        case .p2p150M: def.architecture.layers = 10; def.architecture.width = 1024; def.architecture.heads = 16; def.architecture.kvHeads = 16
            def.architecture.decoderLayers = 3; def.architecture.decoderHeads = 8; def.architecture.frameSize = 192; def.actions.keySlots = 4
        case .p2p300M: def.architecture.layers = 20; def.architecture.width = 1024; def.architecture.heads = 16; def.architecture.kvHeads = 16
            def.architecture.decoderLayers = 3; def.architecture.decoderHeads = 8; def.architecture.frameSize = 192; def.actions.keySlots = 4
        case .blank: break
        }
    }

    func benchmark() async {
        benchmarking = true
        defer { benchmarking = false }
        let arch = def.architecture, actions = def.actions
        let ms: Double = await Task.detached(priority: .userInitiated) {
            let model = PolicyModel(architecture: arch, actions: actions)
            model.train(false)
            model.convert(to: .bfloat16)
            let stepper = PolicyStepper(model: model)
            let frames = PolicyModel.randomFrames(2, size: arch.frameSize)
            for i in 0 ..< arch.contextSteps + 5 { _ = stepper.step(frame: frames[i % 2], text: nil) }
            let t0 = Date()
            for i in 0 ..< 60 { _ = stepper.step(frame: frames[i % 2], text: nil) }
            let r = Date().timeIntervalSince(t0) / 60 * 1000
            MLX.GPU.clearCache()
            return r
        }.value
        let budget = 1000 / Double(arch.stepRateHz)
        withAnimation(Theme.spring) {
            bench = String(format: "%.1f ms per decision · up to %.0f per second %@", ms, 1000 / ms, ms < budget * 0.8 ? "✓" : "· too slow for \(arch.stepRateHz) Hz")
        }
    }
}

/// − value + stepper.
struct NumberStepper: View {
    @Binding var value: Int
    let range: ClosedRange<Int>
    var step = 1
    @Environment(\.isEnabled) private var enabled
    var body: some View {
        HStack(spacing: 10) {
            Button { withAnimation(Theme.snappy) { value = max(range.lowerBound, value - step) } } label: { Image(systemName: "minus").font(.system(size: 11, weight: .bold)).frame(width: 28, height: 28) }
                .buttonStyle(ATButtonStyle(kind: .secondary, height: 28)).disabled(value <= range.lowerBound)
            Text("\(value)").font(Theme.font(14, .semibold)).monospacedDigit().frame(minWidth: 30).contentTransition(.numericText())
            Button { withAnimation(Theme.snappy) { value = min(range.upperBound, value + step) } } label: { Image(systemName: "plus").font(.system(size: 11, weight: .bold)).frame(width: 28, height: 28) }
                .buttonStyle(ATButtonStyle(kind: .secondary, height: 28)).disabled(value >= range.upperBound)
        }
        .opacity(enabled ? 1 : 0.5)
    }
}

struct PretrainedStatus: View {
    let app: AppModel
    let initialization: ModelInitialization
    var body: some View {
        if let item = app.downloads.item(for: initialization) {
            HStack(spacing: 10) {
                switch item.state {
                case .done:
                    Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.live)
                    Text("Weights downloaded").font(Theme.font(13)).foregroundStyle(Theme.live)
                case .downloading:
                    ProgressView(value: item.fraction).tint(Theme.text).frame(width: 140)
                    Text("\(item.received.bytesText) of \(item.expectedBytes.bytesText)").font(Theme.font(12)).monospacedDigit().foregroundStyle(Theme.muted)
                    Button("Pause") { app.downloads.pause(item.id) }.buttonStyle(.atSmall)
                case .paused, .notStarted, .failed:
                    if case .failed(let m) = item.state { Text(m).font(Theme.font(12)).foregroundStyle(Theme.warning) }
                    Button("Download \(item.expectedBytes.bytesText)") { app.downloads.start(item.id) }.buttonStyle(.atSmallPrimary)
                    Text("MIT · elefantai/open-p2p").font(Theme.font(12)).foregroundStyle(Theme.muted)
                }
            }
        }
    }
}

struct KeyVocabularyPicker: View {
    @Binding var keys: [String]
    @Environment(\.isEnabled) private var enabled
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Keys the model can press").font(Theme.font(14, .semibold))
                    Text("\(keys.count) selected · keys outside this set are never learned or pressed").font(Theme.font(12)).foregroundStyle(Theme.muted).contentTransition(.numericText())
                }
                Spacer()
                Button("Pixel2Play set") { withAnimation(Theme.snappy) { keys = KeyCatalog.p2pKeys } }.buttonStyle(.atSmall)
                Button("Game set") { withAnimation(Theme.snappy) { keys = KeyCatalog.p2pKeys + KeyCatalog.gameExtras } }.buttonStyle(.atSmall)
                Button("Full keyboard") { withAnimation(Theme.snappy) { keys = KeyCatalog.p2pKeys + KeyCatalog.fullKeyboard.filter { !KeyCatalog.p2pKeys.contains($0) } } }.buttonStyle(.atSmall)
            }
            let unit: CGFloat = 40
            ZStack(alignment: .topLeading) {
                ForEach(KeyCatalog.all) { k in
                    let on = keys.contains(k.id)
                    Button {
                        withAnimation(Theme.snappy) { if on { keys.removeAll { $0 == k.id } } else { keys.append(k.id) } }
                    } label: {
                        Text(k.label.count > 3 ? String(k.label.prefix(5)) : k.label.uppercased())
                            .font(Theme.font(k.label.count > 3 ? 9.5 : 12, .semibold))
                            .frame(width: k.width * unit - 5, height: unit - 5)
                            .background(RoundedRectangle(cornerRadius: 7, style: .continuous).fill(on ? Theme.text : Theme.raised))
                            .foregroundStyle(on ? Color.black : Theme.text2)
                            .scaleEffect(on ? 1 : 0.96)
                    }
                    .buttonStyle(.plain)
                    .offset(x: k.x * unit, y: k.y * unit)
                }
            }
            .frame(width: 16.5 * unit, height: 6.6 * unit, alignment: .topLeading)
            .opacity(enabled ? 1 : 0.55)
        }
    }
}

struct PointerVocabularyPicker: View {
    @Binding var events: [PointerEvent]
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Buttons and scrolling").font(Theme.font(14, .semibold))
            FlowChips(items: PointerEvent.allCases.map { ($0, $0.label) }, isOn: { events.contains($0) }) { e in
                withAnimation(Theme.snappy) {
                    if events.contains(e) { events.removeAll { $0 == e } } else { events.append(e); events = PointerEvent.allCases.filter(events.contains) }
                }
            }
        }
    }
}

/// Wrapping toggle chips.
struct FlowChips<T: Hashable>: View {
    let items: [(T, String)]
    let isOn: (T) -> Bool
    let toggle: (T) -> Void
    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 120), spacing: 6, alignment: .leading)], alignment: .leading, spacing: 6) {
            ForEach(items, id: \.0) { item, label in
                let on = isOn(item)
                Button { toggle(item) } label: {
                    Text(label).font(Theme.font(12.5, on ? .semibold : .regular)).lineLimit(1)
                        .foregroundStyle(on ? Color.black : Theme.text2)
                        .padding(.horizontal, 12).frame(height: 30)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(on ? Theme.text : Theme.raised, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
                .buttonStyle(.plain)
            }
        }
    }
}

/// Training hyperparameters, grouped.
struct TrainingPanels: View {
    @Binding var training: TrainingHyperparameters
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top, spacing: 18) {
                Rows(title: "Schedule") {
                    Row(title: "Learning rate") {
                        TextField("", value: $training.learningRate, format: .number.notation(.scientific)).textFieldStyle(.plain)
                            .font(Theme.font(14)).multilineTextAlignment(.trailing).frame(width: 90)
                    }
                    Row(title: "Schedule") { Segmented(options: [(LRSchedule.cosine, "Cosine"), (LRSchedule.constant, "Constant")], selection: $training.schedule, compact: true) }
                    Row(title: "Warmup steps") { NumberStepper(value: $training.warmupSteps, range: 0 ... 5000, step: 50) }
                    Row(title: "Total steps") { NumberStepper(value: $training.totalSteps, range: 100 ... 500_000, step: 500) }
                    Row(title: "Clips per step") {
                        Picker("", selection: $training.batchSize) { Text("Automatic").tag(0); ForEach([1, 2, 4, 8], id: \.self) { Text("\($0)").tag($0) } }.labelsHidden().fixedSize()
                    }
                    Row(title: "Gradient accumulation") { NumberStepper(value: $training.gradientAccumulation, range: 1 ... 16) }
                }
                VStack(alignment: .leading, spacing: 18) {
                    Rows(title: "Validation", footer: (training.saveBest || training.earlyStopping) && training.validationFraction == 0 ? "Both need held-out recordings above 0%." : "The best model is saved as “Best” and used by Run unless you pick another checkpoint.") {
                        LabeledSlider(title: "Held-out recordings", value: $training.validationFraction, range: 0 ... 0.4, step: 0.05, format: { "\(Int($0 * 100))%" })
                        Row(title: "Validate every") { NumberStepper(value: $training.validateEverySteps, range: 50 ... 10000, step: 50) }
                        Toggle("Keep the best model", isOn: $training.saveBest).toggleStyle(.oled)
                        Toggle("Early stopping", isOn: $training.earlyStopping.animation(Theme.spring)).toggleStyle(.oled)
                        if training.earlyStopping {
                            Row(title: "Stop after", detail: "\(training.earlyStoppingPatience * training.validateEverySteps) steps without improvement") {
                                NumberStepper(value: $training.earlyStoppingPatience, range: 1 ... 50)
                            }
                        }
                    }
                    Rows(title: "Checkpoints") {
                        LabeledSlider(title: "Save every", value: $training.checkpointEveryMinutes, range: 2 ... 60, step: 1, format: { "\(Int($0)) min" })
                    }
                }
            }
            HStack(alignment: .top, spacing: 18) {
                Rows(title: "Regularisation", footer: "History dropout hides past actions so the model acts on what it sees instead of copying itself. Raise it if the model repeats actions in Run.") {
                    LabeledSlider(title: "Action-history dropout", value: $training.historyDropout, range: 0 ... 0.9, step: 0.05, format: { "\(Int($0 * 100))%" })
                    Row(title: "Weight decay") {
                        TextField("", value: $training.weightDecay, format: .number.notation(.scientific)).textFieldStyle(.plain)
                            .font(Theme.font(14)).multilineTextAlignment(.trailing).frame(width: 90)
                    }
                    LabeledSlider(title: "Gradient clip", value: $training.gradientClip, range: 0 ... 5, step: 0.25)
                    Toggle("bf16 mixed precision", isOn: $training.bf16).toggleStyle(.oled)
                }
                Panel {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Augmentations").font(Theme.font(14, .semibold))
                        Text("The same for every frame of a clip, as in Pixel2Play.").font(Theme.font(12)).foregroundStyle(Theme.muted)
                        FlowChips(items: Augmentation.allCases.map { ($0, $0.label) }, isOn: { training.augmentations.contains($0) }) { a in
                            withAnimation(Theme.snappy) {
                                if training.augmentations.contains(a) { training.augmentations.removeAll { $0 == a } } else { training.augmentations.append(a) }
                            }
                        }
                        LabeledSlider(title: "Strength", value: $training.augmentationStrength, range: 0 ... 2, step: 0.1)
                    }
                }
            }
        }
    }
}

/// Kept for compatibility with older call sites.
struct TrainingSettingsSection: View {
    @Binding var training: TrainingHyperparameters
    var body: some View { TrainingPanels(training: $training) }
}
