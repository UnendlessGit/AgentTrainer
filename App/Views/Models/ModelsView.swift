import ATCore
import ATModel
import MLX
import SwiftUI

struct ModelsView: View {
    @Bindable var app: AppModel
    @State private var showingNew = false

    var body: some View {
        HSplitView {
            VStack(spacing: 0) {
                List(selection: $app.selectedModelID) {
                    ForEach(app.models.models) { m in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(m.definition.name)
                            Text(m.definition.isTrained ? "\(m.definition.trainedSteps) steps trained" : "Not trained yet")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        .tag(m.id)
                        .contextMenu {
                            Button("Duplicate") { app.selectedModelID = app.models.duplicate(m)?.id }
                            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([m.url]) }
                            Divider()
                            Button("Move to Trash", role: .destructive) { app.models.trash(m) }
                        }
                    }
                }
                Divider()
                HStack {
                    Button { showingNew = true } label: { Label("New Model", systemImage: "plus") }
                    Spacer()
                }
                .buttonStyle(.borderless).padding(8)
            }
            .frame(minWidth: 200, idealWidth: 230, maxWidth: 300)
            Group {
                if let e = app.models.entry(app.selectedModelID) {
                    ModelEditor(app: app, entry: e).id(e.id)
                } else {
                    ContentUnavailableView {
                        Label("No model selected", systemImage: "brain")
                    } description: {
                        Text("Create a model to train on your recordings.")
                    } actions: {
                        Button("New Model…") { showingNew = true }.buttonStyle(.borderedProminent)
                    }
                }
            }
            .frame(minWidth: 520, maxWidth: .infinity)
        }
        .sheet(isPresented: $showingNew) { NewModelSheet(app: app) }
    }
}

struct NewModelSheet: View {
    let app: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var name = "My Agent"
    @State private var preset = ModelDefinition.Preset.gameP2P150

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("New AI Model").font(.title2.bold())
            TextField("Name", text: $name)
            ForEach(ModelDefinition.Preset.allCases) { p in
                Button { preset = p } label: {
                    HStack(alignment: .top, spacing: 12) {
                        Image(systemName: preset == p ? "largecircle.fill.circle" : "circle").foregroundStyle(.tint)
                        VStack(alignment: .leading, spacing: 3) {
                            Text(p.title).font(.headline)
                            Text(p.subtitle).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                        }
                        Spacer()
                    }
                    .padding(12)
                    .background(RoundedRectangle(cornerRadius: 10).fill(preset == p ? Color.accentColor.opacity(0.12) : Color.secondary.opacity(0.06)))
                }
                .buttonStyle(.plain)
            }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Create") {
                    if let e = app.models.create(ModelDefinition.make(preset, name: name)) { app.selectedModelID = e.id }
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(24)
        .frame(width: 520)
    }
}

struct ModelEditor: View {
    let app: AppModel
    let entry: ModelEntry
    @State private var def: ModelDefinition
    @State private var bench: String?
    @State private var benchmarking = false

    init(app: AppModel, entry: ModelEntry) {
        self.app = app
        self.entry = entry
        _def = State(initialValue: entry.definition)
    }

    private var locked: Bool { def.architectureLocked }

    var body: some View {
        Form {
            Section {
                TextField("Name", text: $def.name)
                TextField("Notes", text: $def.notes, axis: .vertical).lineLimit(1 ... 4)
                LabeledContent("Parameters") { Text(String(format: "%.0fM", Double(def.parameterCount) / 1e6)).monospacedDigit() }
                LabeledContent("Status") {
                    Text(def.isTrained ? "\(def.trainedSteps) training steps · \(entry.checkpoints.count) checkpoints" : "Not trained yet")
                }
                if locked {
                    Label("The network shape is fixed because training has started. Duplicate the model to change it.", systemImage: "lock")
                        .font(.callout).foregroundStyle(.secondary)
                }
            } header: { Text("Overview") }

            Section("Starting point") {
                Picker("Initialize from", selection: $def.initialization) {
                    ForEach(ModelInitialization.allCases, id: \.self) { Text($0.label).tag($0) }
                }
                .disabled(locked)
                .onChange(of: def.initialization) { _, i in applyInitialization(i) }
                if def.initialization != .blank { PretrainedStatus(app: app, initialization: def.initialization) }
            }

            Section("Network") {
                Stepper("Transformer layers: \(def.architecture.layers)", value: $def.architecture.layers, in: 2 ... 32)
                Picker("Width", selection: $def.architecture.width) { ForEach([128, 192, 256, 384, 512, 768, 1024, 1536, 2048], id: \.self) { Text("\($0)").tag($0) } }
                    .onChange(of: def.architecture.width) { _, w in def.architecture.heads = w / 64; def.architecture.kvHeads = w / 64; def.architecture.decoderHeads = max(1, w / 128) }
                Stepper("Action decoder layers: \(def.architecture.decoderLayers)", value: $def.architecture.decoderLayers, in: 1 ... 6)
                LabeledSlider(title: "Dropout", value: $def.architecture.dropout, range: 0 ... 0.3, step: 0.01)
                if def.initialization != .blank {
                    Text("Pretrained models keep Pixel2Play's shape (\(def.initialization == .p2p300M ? 20 : 10) layers × 1024).").font(.caption).foregroundStyle(.secondary)
                }
            }
            .disabled(locked || def.initialization != .blank)

            Section("Vision") {
                Picker("Input size", selection: $def.architecture.frameSize) { Text("192 × 192 (Pixel2Play)").tag(192); Text("256 × 256").tag(256) }
                    .disabled(locked || def.initialization != .blank)
                Toggle("Grayscale", isOn: $def.architecture.grayscale).disabled(locked)
                Text("Frames are squashed to a square like Pixel2Play; the same GPU resize is used for training and running.").font(.caption).foregroundStyle(.secondary)
            }

            Section("Temporal memory") {
                Picker("Decisions per second", selection: $def.architecture.stepRateHz) {
                    ForEach([20, 30, 40, 60], id: \.self) { Text("\($0) Hz").tag($0) }
                }
                .pickerStyle(.segmented)
                LabeledSlider(title: "Memory", value: $def.architecture.memorySeconds, range: 5 ... 30, step: 1, format: { "\(Int($0)) s" })
                Text("\(def.architecture.contextSteps) timesteps · \(def.architecture.contextSteps * def.architecture.tokensPerStep(slots: def.actions.slotCount)) tokens kept in the KV cache; each step only processes the new frame.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .disabled(locked)

            Section("Language") {
                Toggle("Follow instructions", isOn: $def.usesInstructions)
                LabeledSlider(title: "Instruction dropout", value: $def.training.instructionDropout, range: 0 ... 0.6, step: 0.05)
                Toggle("Read text on screen (OCR)", isOn: $def.architecture.screenText).disabled(locked)
                if def.architecture.screenText {
                    Stepper("Screen text items: \(def.architecture.screenTextItems)", value: $def.architecture.screenTextItems, in: 2 ... 16).disabled(locked)
                }
                Toggle("Type text from instructions", isOn: $def.architecture.literalTyping).disabled(locked)
                Text("Instructions and on-screen text are embedded with EmbeddingGemma (on device). Typing lets the model copy characters from an instruction like “type hello”.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Actions the model can use") {
                KeyVocabularyPicker(keys: $def.actions.keys).disabled(locked)
                Stepper("Simultaneous keys: \(def.actions.keySlots)", value: $def.actions.keySlots, in: 1 ... 8).disabled(locked || def.initialization != .blank)
                PointerVocabularyPicker(events: $def.actions.pointerEvents).disabled(locked)
                Picker("Mouse", selection: $def.actions.mouseMode) {
                    Text("Relative movement (games)").tag(MouseMode.relative)
                    Text("Pointer position (apps)").tag(MouseMode.absolute)
                }
                .disabled(locked)
                if def.actions.mouseMode == .absolute {
                    Stepper("Pointer grid: \(def.actions.absoluteColumns) × \(def.actions.absoluteRows)", value: $def.actions.absoluteColumns, in: 32 ... 256, step: 16)
                        .onChange(of: def.actions.absoluteColumns) { _, c in def.actions.absoluteRows = c * 9 / 16 }
                        .disabled(locked)
                }
                Picker("Mouse sampling", selection: $def.actions.mouseSampling) {
                    Text("Smooth (truncated normal)").tag(MouseSampling.truncatedNormal)
                    Text("Bin center").tag(MouseSampling.center)
                }
                Text("These define what the model can output. The Run tab can still block any of them without retraining.").font(.caption).foregroundStyle(.secondary)
            }

            TrainingSettingsSection(training: $def.training)

            Section("Speed on this Mac") {
                HStack {
                    Button(benchmarking ? "Measuring…" : "Run benchmark") { Task { await benchmark() } }.disabled(benchmarking)
                    if let bench { Text(bench).monospacedDigit() }
                }
                Text("Measures one real-time step with a full memory window, random weights, bf16.").font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .onChange(of: def) { _, d in app.models.save(d) }
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
        bench = String(format: "%.1f ms per step → %.0f FPS max · %@ for %d Hz", ms, 1000 / ms, ms < budget * 0.8 ? "✓ fast enough" : "⚠︎ too slow", arch.stepRateHz)
    }
}

struct PretrainedStatus: View {
    let app: AppModel
    let initialization: ModelInitialization
    var body: some View {
        if let item = app.downloads.item(for: initialization) {
            HStack {
                switch item.state {
                case .done: Label("Weights downloaded", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                case .downloading:
                    ProgressView(value: item.fraction).frame(width: 160)
                    Text("\(item.received.bytesText) of \(item.expectedBytes.bytesText)").font(.caption).monospacedDigit()
                    Button("Pause") { app.downloads.pause(item.id) }
                case .paused, .notStarted, .failed:
                    if case .failed(let m) = item.state { Text(m).font(.caption).foregroundStyle(.orange) }
                    Button("Download \(item.expectedBytes.bytesText)") { app.downloads.start(item.id) }
                    Text("MIT license · elefantai/open-p2p").font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }
}

struct KeyVocabularyPicker: View {
    @Binding var keys: [String]
    @Environment(\.isEnabled) private var enabled
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Keys (\(keys.count)) — click to add or remove").font(.callout)
            let unit: CGFloat = 26
            ZStack(alignment: .topLeading) {
                ForEach(KeyCatalog.all) { k in
                    let on = keys.contains(k.id)
                    Button {
                        if on { keys.removeAll { $0 == k.id } } else { keys.append(k.id) }
                    } label: {
                        Text(k.label.count > 3 ? String(k.label.prefix(4)) : k.label.uppercased())
                            .font(.system(size: 9, weight: .medium))
                            .frame(width: k.width * unit - 3, height: unit - 3)
                            .background(RoundedRectangle(cornerRadius: 4).fill(on ? Color.accentColor : Color.secondary.opacity(0.12)))
                            .foregroundStyle(on ? .white : .primary)
                    }
                    .buttonStyle(.plain)
                    .offset(x: k.x * unit, y: k.y * unit)
                }
            }
            .frame(width: 16.5 * unit, height: 6.6 * unit, alignment: .topLeading)
            .opacity(enabled ? 1 : 0.6)
            HStack {
                Button("Pixel2Play set") { keys = KeyCatalog.p2pKeys }
                Button("Game set") { keys = KeyCatalog.p2pKeys + KeyCatalog.gameExtras }
                Button("Full keyboard") { keys = KeyCatalog.p2pKeys + KeyCatalog.fullKeyboard.filter { !KeyCatalog.p2pKeys.contains($0) } }
            }
            .controlSize(.small)
        }
    }
}

struct PointerVocabularyPicker: View {
    @Binding var events: [PointerEvent]
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Mouse buttons and scrolling").font(.callout)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), alignment: .leading)], alignment: .leading) {
                ForEach(PointerEvent.allCases, id: \.self) { e in
                    Toggle(e.label, isOn: Binding(get: { events.contains(e) }, set: { on in
                        if on { events.append(e); events = PointerEvent.allCases.filter(events.contains) } else { events.removeAll { $0 == e } }
                    }))
                    .toggleStyle(.checkbox)
                }
            }
        }
    }
}

struct TrainingSettingsSection: View {
    @Binding var training: TrainingHyperparameters
    var body: some View {
        Section("Training") {
            LabeledContent("Learning rate") {
                TextField("", value: $training.learningRate, format: .number.notation(.scientific)).frame(width: 90).multilineTextAlignment(.trailing)
            }
            Picker("Schedule", selection: $training.schedule) { Text("Cosine").tag(LRSchedule.cosine); Text("Constant (P2P)").tag(LRSchedule.constant) }
            Stepper("Warmup: \(training.warmupSteps) steps", value: $training.warmupSteps, in: 0 ... 5000, step: 50)
            Stepper("Total steps: \(training.totalSteps)", value: $training.totalSteps, in: 100 ... 500_000, step: 500)
            Picker("Clips per step", selection: $training.batchSize) {
                Text("Automatic").tag(0); ForEach([1, 2, 4, 8], id: \.self) { Text("\($0)").tag($0) }
            }
            Stepper("Gradient accumulation: \(training.gradientAccumulation)×", value: $training.gradientAccumulation, in: 1 ... 16)
            LabeledContent("Weight decay") {
                TextField("", value: $training.weightDecay, format: .number.notation(.scientific)).frame(width: 90).multilineTextAlignment(.trailing)
            }
            LabeledSlider(title: "Gradient clip", value: $training.gradientClip, range: 0 ... 5, step: 0.25)
            LabeledSlider(title: "Held-out recordings", value: $training.validationFraction, range: 0 ... 0.4, step: 0.05, format: { "\(Int($0 * 100))%" })
            Stepper("Validate every \(training.validateEverySteps) steps", value: $training.validateEverySteps, in: 50 ... 10000, step: 50)
            LabeledSlider(title: "Checkpoint every", value: $training.checkpointEveryMinutes, range: 2 ... 60, step: 1, format: { "\(Int($0)) min" })
            Toggle("Keep the best model", isOn: $training.saveBest)
                .help("Saves the model from the validation check with the lowest held-out loss as checkpoint “Best”. It is never pruned, and Run uses it unless you pick another checkpoint.")
            Toggle("Early stopping", isOn: $training.earlyStopping)
                .help("Stops training when the held-out loss hasn't improved for several validation checks in a row.")
            if training.earlyStopping {
                Stepper("Stop after \(training.earlyStoppingPatience) checks without improvement (\(training.earlyStoppingPatience * training.validateEverySteps) steps)",
                        value: $training.earlyStoppingPatience, in: 1 ... 50)
            }
            if (training.saveBest || training.earlyStopping) && training.validationFraction == 0 {
                Text("Both need held-out recordings: set “Held-out recordings” above 0%.").font(.caption).foregroundStyle(.orange)
            }
            LabeledSlider(title: "Action-history dropout", value: $training.historyDropout, range: 0 ... 0.9, step: 0.05, format: { "\(Int($0 * 100))%" })
            Text("Hides past actions from the model during training so it learns to act on what it sees instead of copying itself. Raise it if the model keeps repeating actions in Run.")
                .font(.caption).foregroundStyle(.secondary)
            Toggle("bf16 mixed precision", isOn: $training.bf16)
            VStack(alignment: .leading) {
                Text("Augmentations (same for every frame of a clip, as in Pixel2Play)").font(.callout)
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 140), alignment: .leading)], alignment: .leading) {
                    ForEach(Augmentation.allCases) { a in
                        Toggle(a.label, isOn: Binding(get: { training.augmentations.contains(a) }, set: { on in
                            if on { training.augmentations.append(a) } else { training.augmentations.removeAll { $0 == a } }
                        }))
                        .toggleStyle(.checkbox)
                    }
                }
                LabeledSlider(title: "Strength", value: $training.augmentationStrength, range: 0 ... 2, step: 0.1)
            }
        }
    }
}
