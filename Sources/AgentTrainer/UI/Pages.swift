import AppKit
import MLX
import SwiftUI

struct HomeView: View {
    @ObservedObject var model: AppModel
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                SectionTitle("Home", "Record how you work, train locally, then let your AI act anywhere on this Mac.")
                HStack(spacing: 14) {
                    MetricCard(title: "Recordings", value: "\(model.recordings.count)", symbol: "record.circle", color: ATColor.coral)
                    MetricCard(title: "AI profiles", value: "\(model.profiles.count)", symbol: "cpu", color: ATColor.violet)
                    MetricCard(title: "Local storage", value: ByteCountFormatter.string(fromByteCount: model.storageBytes, countStyle: .file), symbol: "internaldrive", color: ATColor.amber)
                    MetricCard(title: "MLX memory", value: ByteCountFormatter.string(fromByteCount: Int64(Memory.activeMemory), countStyle: .memory), symbol: "memorychip", color: ATColor.green)
                }
                OLEDCard {
                    HStack(spacing: 20) {
                        VStack(alignment: .leading, spacing: 8) { Text("Private by design").font(.title3.bold()).foregroundStyle(ATColor.green); Text("Recordings, packed caches, checkpoints, and model versions stay on this Mac. The only network request is a launch-time check for new GitHub releases.").foregroundStyle(.secondary) }
                        Spacer()
                        Button("Start Recording") { model.selection = .record }.primaryButton(color: ATColor.coral)
                        Button("Train an AI") { model.selection = .training }.primaryButton(color: ATColor.cyan)
                        Button("Run an AI") { model.selection = .run }.primaryButton(color: ATColor.violet)
                    }
                }
                PermissionStrip(model: model)
                HStack(alignment: .top, spacing: 14) {
                    OLEDCard { VStack(alignment: .leading, spacing: 10) { Label("Exact vision contract", systemImage: "viewfinder").foregroundStyle(ATColor.cyan).font(.headline); Text("Live capture is configured to the exact width, height, color detail, and chroma format stored in the trained model. Incompatible runs are blocked.").foregroundStyle(.secondary) } }
                    OLEDCard { VStack(alignment: .leading, spacing: 10) { Label("Immediate safety", systemImage: "hand.raised.fill").foregroundStyle(ATColor.coral).font(.headline); Text("Your customizable global panic shortcut stops every workflow. Every held key and mouse button is released immediately.").foregroundStyle(.secondary) } }
                }
            }.padding(28)
        }
    }
}

struct PermissionStrip: View {
    @ObservedObject var model: AppModel
    var body: some View {
        OLEDCard {
            HStack(spacing: 24) {
                Text("Permissions").font(.headline)
                PermissionBadge(name: "Screen", granted: model.screenPermission)
                PermissionBadge(name: "Input", granted: model.inputPermission)
                PermissionBadge(name: "Accessibility", granted: model.accessibilityPermission)
                Spacer()
                Button("Refresh") { model.refreshPermissions() }.primaryButton()
            }
        }
    }
}

private struct PermissionBadge: View {
    let name: String; let granted: Bool
    var body: some View { Label(name, systemImage: granted ? "checkmark.circle.fill" : "exclamationmark.circle.fill").foregroundStyle(granted ? ATColor.green : ATColor.amber) }
}

struct RecordView: View {
    @ObservedObject var model: AppModel
    @State private var presetName = ""
    @State private var presetToDelete: RecordingPreset?
    private var sources: [CaptureSourceOption] {
        model.captureSources.filter { source in
            switch model.captureKind { case .display, .screenRegion: source.kind == .display; case .window, .windowRegion: source.kind == .window }
        }
    }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                SectionTitle("Record", "Hardware HEVC video with frame-accurate input synchronization and configurable cleanup at both ends.")
                recordingPresets
                HStack(alignment: .top, spacing: 16) {
                    OLEDCard {
                        VStack(alignment: .leading, spacing: 16) {
                            Text("Capture source").font(.headline)
                            Picker("Type", selection: $model.captureKind) { ForEach(CaptureKind.allCases) { Text($0.rawValue).tag($0) } }.pickerStyle(.segmented).onChange(of: model.captureKind) { _, _ in Task { await model.refreshSources() } }
                            Picker("Source", selection: $model.selectedSourceID) { Text("Select…").tag(UInt32?.none); ForEach(sources) { Text("\($0.name) — \($0.detail)").tag(Optional($0.id)) } }
                            HStack {
                                Picker("Save in folder", selection: $model.recordingDestinationFolderID) { ForEach(model.recordingFolders) { Text($0.name).tag(Optional($0.id)) } }
                                Button { Task { await model.createRecordingFolder() } } label: { Image(systemName: "folder.badge.plus") }.buttonStyle(.plain).foregroundStyle(ATColor.cyan)
                            }
                            if model.captureKind == .screenRegion || model.captureKind == .windowRegion {
                                Divider(); Text("Region").font(.subheadline.bold())
                                HStack { LabeledNumber("X", value: $model.regionX); LabeledNumber("Y", value: $model.regionY); LabeledNumber("Width", value: $model.regionWidth); LabeledNumber("Height", value: $model.regionHeight) }
                                if model.captureKind == .screenRegion { Button("Select Region on Screen") { model.selectScreenRegion() }.primaryButton(color: ATColor.cyan) }
                                Text(model.captureKind == .windowRegion ? "Window-region coordinates are relative to the window." : "Screen-region coordinates use global display points.").font(.caption).foregroundStyle(.secondary)
                            }
                            Divider()
                            HStack { LabeledNumber("FPS", value: $model.captureFPS); Toggle("Show cursor in video", isOn: $model.showsCursor); Spacer() }
                            HStack { LabeledNumber("Trim first (seconds)", value: $model.recordingTrimStart); LabeledNumber("Trim last (seconds)", value: $model.recordingTrimEnd); InfoTip("Trimming is non-destructive. Replay and training use only the retained time range; the original HEVC video remains intact."); Spacer() }
                        }
                    }.frame(maxWidth: .infinity).disabled(model.recordingIsActiveOrStarting)
                    OLEDCard {
                        VStack(alignment: .leading, spacing: 14) {
                            Label("Synchronized inputs", systemImage: "keyboard.badge.ellipsis").foregroundStyle(ATColor.cyan).font(.headline)
                            InputCapability("Mouse position + raw deltas", "cursorarrow.motionlines")
                            InputCapability("Buttons, holds, and drags", "computermouse")
                            InputCapability("Two-axis scrolling", "arrow.up.and.down.and.arrow.left.and.right")
                            InputCapability("Keys, chords, and modifiers", "keyboard")
                            InputCapability("Relative game-camera movement", "scope")
                            Divider()
                            Text("Recording key blacklist").font(.subheadline.bold()).foregroundStyle(ATColor.coral)
                            RecordingKeyBlacklistEditor(keys: $model.recordingExcludedKeyCodes, model: model)
                        }
                    }.frame(width: 330).disabled(model.recordingIsActiveOrStarting)
                }
                HStack {
                    Label("The menu bar shows recording status; a compact capture-excluded keyboard shows only keys used in this recording.", systemImage: "keyboard").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    if model.recordingIsActiveOrStarting { Button(model.isRecording ? "Stop & Save" : "Cancel Start") { Task { await model.stopRecording() } }.primaryButton(color: ATColor.coral) }
                    else { Button("Record") { Task { await model.startRecording() } }.primaryButton(color: ATColor.coral).disabled(model.agentIsActiveOrStarting || model.isReplaying) }
                }
            }.padding(28)
        }
        .onAppear {
            if let id = model.selectedRecordingPresetID,
               let preset = model.recordingPresets.first(where: { $0.id == id }) {
                presetName = preset.name
            }
        }
        .alert("Delete recording preset?", isPresented: Binding(get: { presetToDelete != nil }, set: { if !$0 { presetToDelete = nil } })) {
            Button("Cancel", role: .cancel) { presetToDelete = nil }
            Button("Delete", role: .destructive) {
                if let presetToDelete { model.deleteRecordingPreset(presetToDelete) }
                self.presetToDelete = nil
                presetName = ""
            }
        } message: {
            Text("This deletes only the preset “\(presetToDelete?.name ?? "")”. Existing recordings are unchanged.")
        }
    }

    private var recordingPresets: some View {
        OLEDCard {
            HStack(spacing: 10) {
                Label("Recording presets", systemImage: "slider.horizontal.3").font(.headline).foregroundStyle(ATColor.violet)
                Picker("Preset", selection: Binding(
                    get: { model.selectedRecordingPresetID },
                    set: { id in
                        guard let id, let preset = model.recordingPresets.first(where: { $0.id == id }) else {
                            model.selectedRecordingPresetID = nil
                            presetName = ""
                            return
                        }
                        presetName = preset.name
                        model.applyRecordingPreset(preset)
                    }
                )) {
                    Text("Unsaved settings").tag(UUID?.none)
                    ForEach(model.recordingPresets) { preset in Text(preset.name).tag(Optional(preset.id)) }
                }
                .frame(minWidth: 190)
                TextField("Preset name", text: $presetName).textFieldStyle(.roundedBorder).frame(width: 190)
                Button("Save New") {
                    model.createRecordingPreset(name: presetName)
                }
                .primaryButton(color: ATColor.violet)
                .disabled(presetName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Button("Update") {
                    guard let id = model.selectedRecordingPresetID,
                          let preset = model.recordingPresets.first(where: { $0.id == id }) else { return }
                    model.updateRecordingPreset(preset, name: presetName)
                }
                .primaryButton(color: ATColor.cyan)
                .disabled(model.selectedRecordingPresetID == nil || presetName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Button("Delete", role: .destructive) {
                    guard let id = model.selectedRecordingPresetID else { return }
                    presetToDelete = model.recordingPresets.first { $0.id == id }
                }
                .primaryButton(color: ATColor.coral)
                .disabled(model.selectedRecordingPresetID == nil)
                Spacer(minLength: 0)
                InfoTip("Save the complete capture type, source, destination, FPS, cursor, region, trim, and input blacklist. Update overwrites the selected preset with the settings currently shown below.")
            }
        }
        .disabled(model.recordingIsActiveOrStarting)
    }
}

private struct InputCapability: View { let text: String; let symbol: String; init(_ text: String, _ symbol: String) { self.text = text; self.symbol = symbol }; var body: some View { Label(text, systemImage: symbol).foregroundStyle(.secondary) } }
private struct LabeledNumber: View { let title: String; @Binding var value: Double; init(_ title: String, value: Binding<Double>) { self.title = title; _value = value }; var body: some View { VStack(alignment: .leading, spacing: 5) { Text(title).font(.caption).foregroundStyle(.secondary); TextField(title, value: $value, format: .number).textFieldStyle(.roundedBorder).frame(minWidth: 82) } } }

struct ModelsView: View {
    @ObservedObject var model: AppModel
    @State private var profileToDelete: AIProfile?
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionTitle("AI Models", "Create profiles, select control channels and recordings, and manage immutable model versions.")
            HStack(alignment: .top, spacing: 14) {
                OLEDCard {
                    VStack(spacing: 10) {
                        List(model.profiles, selection: $model.selectedProfileID) { profile in
                            HStack(spacing: 8) {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(profile.name).font(.headline).lineLimit(1)
                                    let progress = profile.trainingProgress
                                    Text("\(progress?.globalStep ?? 0) steps • \(progress?.epoch ?? 0) epochs")
                                        .font(.caption.bold()).foregroundStyle(progress == nil ? Color.secondary : ATColor.green)
                                    let timing = profile.trainingDurationSummary(recordings: model.recordings)
                                    Text("\(TrainingDurationFormatter.string(seconds: timing.trainingSeconds)) trained • \(timing.experienceIsEstimated ? "~" : "")\(TrainingDurationFormatter.string(seconds: timing.experienceSeconds)) experience")
                                        .font(.caption2.monospacedDigit()).foregroundStyle(.secondary).lineLimit(1)
                                    Text("\(profile.preprocessing.width) × \(profile.preprocessing.height) • \(profile.training.precision.rawValue)")
                                        .font(.caption2).foregroundStyle(.secondary)
                                    if let feedback = progress?.reinforcementFeedbackCount {
                                        Text("RL \(feedback) feedback • \(progress?.reinforcementUpdateCount ?? 0) updates • net \((progress?.reinforcementNetReward ?? 0).formatted(.number.sign(strategy: .always()).precision(.fractionLength(0...2))))")
                                            .font(.caption2.bold().monospacedDigit())
                                            .foregroundStyle((progress?.reinforcementNetReward ?? 0) >= 0 ? ATColor.green : ATColor.coral)
                                    } else if profile.effectiveReinforcement.enabled {
                                        Text("Real-time RL enabled").font(.caption2.bold()).foregroundStyle(ATColor.violet)
                                    }
                                }
                                Spacer(minLength: 4)
                                if profile.isDeletionProtected {
                                    Image(systemName: "shield.fill").foregroundStyle(ATColor.green).help("This AI is protected from deletion, reset, migration cleanup, and automatic autosave cleanup.")
                                } else {
                                    Button { profileToDelete = profile } label: {
                                        Image(systemName: "trash").frame(width: 30, height: 30).contentShape(Rectangle())
                                    }
                                    .buttonStyle(.borderless)
                                    .foregroundStyle(ATColor.coral)
                                    .help("Delete this AI and its saved brains")
                                }
                            }
                            .tag(profile.id)
                            .padding(.vertical, 5)
                            .uiHoverResponse(scale: 1.006)
                        }
                        .scrollContentBackground(.hidden)
                        .onChange(of: model.selectedProfileID) { _, _ in model.unloadVersions() }
                        Button("New Profile") { model.createProfile(name: "New Agent") }.primaryButton()
                    }
                }.frame(width: 280)
                if let profile = model.selectedProfile { ProfileEditor(profile: profile, model: model).id(profile.id) } else { ContentUnavailableView("No AI profile", systemImage: "cpu") }
            }
        }
        .padding(28)
        .alert("Delete AI?", isPresented: Binding(get: { profileToDelete != nil }, set: { if !$0 { profileToDelete = nil } })) {
            Button("Cancel", role: .cancel) { profileToDelete = nil }
            Button("Delete", role: .destructive) { if let profileToDelete { Task { await model.deleteProfile(profileToDelete) } }; profileToDelete = nil }
        } message: {
            Text("This permanently removes \(profileToDelete?.name ?? "this AI") and its saved brains. Recordings are not deleted.")
        }
    }
}

private struct ProfileEditor: View {
    private enum TrainingConfigurationTab: String, CaseIterable, Identifiable {
        case trainingData = "Training Data"
        case reinforcement = "RL Configuration"
        var id: String { rawValue }
    }

    @State var draft: AIProfile
    @State private var acceptedDraft: AIProfile
    @State private var pendingBrainReset: AIProfile?
    @State private var suppressDraftObserver = false
    @State private var trainingConfigurationTab: TrainingConfigurationTab = .trainingData
    @ObservedObject var model: AppModel
    init(profile: AIProfile, model: AppModel) {
        _draft = State(initialValue: profile)
        _acceptedDraft = State(initialValue: profile)
        self.model = model
    }
    var body: some View {
        ScrollView {
            VStack(spacing: 14) {
                if model.isProfileBusy(draft.id) { HStack { Image(systemName: "lock.fill"); Text(model.trainingProfileID == draft.id ? "This brain is training. You can select and run another AI." : "This brain is running. You can select and train another AI.") }.font(.caption.bold()).foregroundStyle(ATColor.amber).frame(maxWidth: .infinity, alignment: .leading).padding(10).raisedGlassSurface(cornerRadius: 10, tint: ATColor.amber) }
                TrainingTimeOverview(profile: draft, recordings: model.recordings)
                OLEDCard {
                    VStack(alignment: .leading, spacing: 14) {
                        HStack { TextField("Profile name", text: $draft.name).font(.title2.bold()).textFieldStyle(.plain); StatusPill(text: "Autosaved", color: ATColor.green); if draft.isDeletionProtected { StatusPill(text: "Protected", color: ATColor.green) }; Spacer(); Button("Duplicate") { Task { await model.duplicateProfile(draft) } }.primaryButton() }
                        Divider(); HStack { Text("Exact model vision").font(.headline).foregroundStyle(ATColor.cyan); InfoTip("These dimensions are the AI's actual eyesight. Training and live running always use this exact width and height.") }
                        HStack { IntField("Width", value: $draft.preprocessing.width, help: "Exact pixels the model sees horizontally."); IntField("Height", value: $draft.preprocessing.height, help: "Exact pixels the model sees vertically."); IntField("Bit detail", value: $draft.preprocessing.bitDepth, help: "Quantization detail from 1 to 8 bits per stored channel.") }
                        HStack { Picker("Mode", selection: $draft.preprocessing.colorMode) { ForEach(ColorMode.allCases) { Text($0.rawValue).tag($0) } }; Picker("Chroma", selection: $draft.preprocessing.chroma) { ForEach(ChromaSubsampling.allCases) { Text($0.rawValue).tag($0) } }.disabled(draft.preprocessing.colorMode == .grayscale); Picker("Resize", selection: $draft.preprocessing.resizePolicy) { ForEach(ResizePolicy.allCases) { Text($0.rawValue).tag($0) } } }
                        ForEach(draft.preprocessing.trainingQualityWarnings, id: \.self) { warning in
                            Label(warning, systemImage: "exclamationmark.triangle.fill")
                                .font(.caption).foregroundStyle(ATColor.amber)
                        }
                        HStack { Spacer(); InfoTip("Chroma controls color-position detail. 4:2:0 is lighter and faster; 4:4:4 preserves color at every pixel. Grayscale stores luminance only.") }
                    }
                }
                OLEDCard {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack { Text("Control channels").font(.headline).foregroundStyle(ATColor.violet); InfoTip("Mouse demonstrations train absolute position and relative delta together. The execution mode is chosen only in Run.") }
                        HStack { Toggle("Mouse movement", isOn: Binding(get: { draft.channels.mouseMovement }, set: { draft.channels.mouseMovement = $0 })); Toggle("Buttons", isOn: $draft.channels.buttons) }
                        HStack { Toggle("Scroll", isOn: $draft.channels.scroll); Toggle("Keyboard", isOn: $draft.channels.keyboard); Toggle("Modifiers", isOn: $draft.channels.modifiers) }
                        Divider(); Text("Blocked outputs").font(.subheadline.bold()).foregroundStyle(ATColor.coral)
                        KeyRestrictionGrid(restrictions: Binding(get: { draft.effectiveRestrictions }, set: { draft.restrictions = $0 }), model: model)
                    }
                }
                OLEDCard {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack { Text("Training configuration").font(.headline).foregroundStyle(ATColor.green); InfoTip("Maximum Steps and Autosave Steps are universal run controls in the Training tab, so they do not change a model's learning identity. Architecture, exact vision, and temporal-context changes require a fresh brain, and the app asks before clearing training.") }
                        Picker("Configuration", selection: $trainingConfigurationTab) {
                            ForEach(TrainingConfigurationTab.allCases) { tab in Text(tab.rawValue).tag(tab) }
                        }
                        .pickerStyle(.segmented)
                        if trainingConfigurationTab == .trainingData {
                        HStack { IntField("Epochs per block", value: $draft.training.epochs, help: "How many complete dataset passes to add. If a block is paused, Train finishes it first; after a completed block, Train adds another block of this size."); IntField("Batch", value: $draft.training.batchSize, help: "Samples evaluated together. Larger batches use more unified memory."); Spacer() }
                        VStack(alignment: .leading, spacing: 9) {
                            HStack {
                                Text("Temporal frame context").font(.subheadline.bold()).foregroundStyle(ATColor.cyan)
                                InfoTip("Set Past frames to 0 for current-frame-only vision. Otherwise each decision adds real lower-resolution past frames paired with all controls demonstrated during their perception intervals. No motion-difference image is generated.")
                                Spacer()
                            }
                            let temporal = draft.training.effectiveTemporalVision
                            HStack {
                                IntField("Past frames", value: temporalIntBinding(\.pastFrameCount), help: "Number of causal screen frames supplied before the current frame (0–\(TemporalVisionConfiguration.maximumPastFrameCount)). Zero disables temporal memory.")
                                IntField("Frames apart", value: temporalIntBinding(\.frameSpacing), help: "Distance between selected frames in Perception FPS intervals (1–\(TemporalVisionConfiguration.maximumFrameSpacing)). 1 means consecutive perceptions.")
                                    .disabled(temporal.pastFrameCount == 0)
                                IntField("Past downscale", value: temporalIntBinding(\.downsampleFactor), help: "Linear resolution reduction for every past frame (1×–\(TemporalVisionConfiguration.maximumDownsampleFactor)×). 2 means half width and half height while preserving aspect, color mode, chroma, and bit detail.")
                                    .disabled(temporal.pastFrameCount == 0)
                            }
                            let pastSpec = temporal.pastFrameSpec(from: draft.preprocessing)
                            HStack(spacing: 10) {
                                StatusPill(text: "Current \(draft.preprocessing.width) × \(draft.preprocessing.height)", color: ATColor.green)
                                if temporal.pastFrameCount == 0 {
                                    StatusPill(text: "Temporal memory off", color: ATColor.amber)
                                    Text("Only the exact current frame is cached, trained, and used at runtime.")
                                        .font(.caption).foregroundStyle(.secondary)
                                } else {
                                    StatusPill(text: "Past \(pastSpec.width) × \(pastSpec.height)", color: ATColor.cyan)
                                    Text("nominally every \(temporal.spacingSeconds(perceptionFPS: draft.training.perceptionFPS).formatted(.number.precision(.fractionLength(3))))s • \(temporal.lookbackSeconds(perceptionFPS: draft.training.perceptionFPS).formatted(.number.precision(.fractionLength(3))))s total lookback")
                                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                                }
                                Spacer()
                            }
                            Text(temporalFormatDetail)
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                        .padding(11)
                        .raisedGlassSurface(cornerRadius: 11, tint: ATColor.cyan)
                        HStack { DoubleField("Learning rate", value: $draft.training.learningRate, help: "Peak AdamW update size. Adaptive scheduling warms up, uses bounded cosine restarts so updates never decay silently to zero, and reduces its envelope only after a measured plateau."); DoubleField("Weight decay", value: $draft.training.weightDecay, help: "Regularization applied by AdamW."); DoubleField("Validation", value: $draft.training.validationSplit, help: "Fraction of whole recordings held out. Splits target sample count, keep rare controls in training, purge shared visual context, and stress-test the brain with zero prior actions. Per-head threshold misses are reported as quality warnings, not whole-model failures.") }
                        HStack {
                            Picker("Learning-rate schedule", selection: Binding(get: { draft.training.effectiveLearningRateSchedule }, set: { draft.training.learningRateSchedule = $0 })) { ForEach(LearningRateSchedule.allCases) { Text($0.rawValue).tag($0) } }
                            IntField("Cosine cycle epochs", value: Binding(get: { draft.training.effectiveCosineCycleEpochs }, set: { draft.training.cosineCycleEpochs = $0 }), help: "Length of each cosine restart. Restarts periodically restore useful update size instead of letting learning freeze.")
                            IntField("Plateau patience", value: Binding(get: { draft.training.effectivePlateauPatience }, set: { draft.training.plateauPatience = $0 }), help: "Completed epochs without a meaningful validation improvement before the learning-rate envelope is halved.")
                        }
                        HStack {
                            DoubleField("Minimum LR ratio", value: Binding(get: { draft.training.effectiveMinimumLearningRateRatio }, set: { draft.training.minimumLearningRateRatio = $0 }), help: "Lowest fraction of the peak rate inside a cosine cycle and the final adaptive envelope floor.")
                            DoubleField("Binary focal gamma", value: Binding(get: { draft.training.effectiveBinaryFocalGamma }, set: { draft.training.binaryFocalGamma = $0 }), help: "Focuses button, key, and modifier learning on current mistakes instead of easy idle frames. Use 0 for calibrated class-balanced BCE without focal modulation.")
                            Spacer()
                            InfoTip("Changing scheduler or loss settings keeps the active brain weights but safely starts a new optimizer sequence. Older profiles remain on their exact legacy schedule until you explicitly select Adaptive Cosine + Plateau.")
                        }
                        VStack(alignment: .leading, spacing: 9) {
                            HStack {
                                Text("Generalization and anti-memorization").font(.subheadline.bold()).foregroundStyle(ATColor.amber)
                                InfoTip("These perturbations run only while training. They preserve cursor geometry, make pixels and prior actions less exact, and force the brain to use stable visual evidence. Held-out validation and live inference remain unmodified.")
                                Spacer()
                                Button("Recommended") { draft.training.generalization = GeneralizationConfiguration() }.primaryButton(color: ATColor.amber)
                                Button("Disable") { draft.training.generalization = .disabled }.primaryButton()
                            }
                            HStack {
                                DoubleField("Vision variation", value: generalizationDoubleBinding(\.visionAugmentationStrength), help: "Luminance, contrast, chroma, and structured sensor variation from 0–0.5. It never moves pixels or pointer labels.")
                                DoubleField("Random erasing", value: generalizationDoubleBinding(\.randomErasingProbability), help: "Chance from 0–0.5 of covering a small neutral rectangle, preventing reliance on one exact pixel patch.")
                                DoubleField("History dropout", value: generalizationDoubleBinding(\.controlHistoryDropout), help: "Chance from 0–0.8 of hiding a prior control's entire trajectory. Rare derived bit flips and bounded continuous noise model imperfect self-predictions; autonomous-start zero history is always trained independently.")
                                    .disabled(draft.training.effectiveTemporalVision.pastFrameCount == 0)
                                DoubleField("Frame dropout", value: generalizationDoubleBinding(\.temporalFrameDropout), help: "Chance from 0–0.5 of hiding an entire historical visual/control token, including missing startup context.")
                                    .disabled(draft.training.effectiveTemporalVision.pastFrameCount == 0)
                                DoubleField("Label smoothing", value: generalizationDoubleBinding(\.binaryLabelSmoothing), help: "Training-only binary target smoothing from 0–0.2. Small values reduce brittle overconfidence; validation targets remain exact.")
                            }
                        }
                        .padding(11)
                        .raisedGlassSurface(cornerRadius: 11, tint: ATColor.amber)
                        HStack { DoubleField("Perception FPS", value: $draft.training.perceptionFPS, help: "How often the AI receives and records a new screen perception. Temporal spacing is measured in these frame intervals. It cannot exceed Action FPS."); DoubleField("Action FPS", value: $draft.training.actionFPS, help: "How often the AI may update mouse, keyboard, button, and scroll output."); Picker("Precision", selection: $draft.training.precision) { ForEach(TrainingPrecision.allCases) { Text($0.rawValue).tag($0) } } }
                        HStack { Text("Architecture preset").foregroundStyle(.secondary); Button("Small") { draft.training.architecture = .small }.primaryButton(); Button("Balanced") { draft.training.architecture = .balanced }.primaryButton(); Button("Large") { draft.training.architecture = .large }.primaryButton() }
                        HStack { IntField("Visual width", value: $draft.training.architecture.visualEmbedding, help: "How much visual information is kept after the shared efficient visual stages."); IntField("Control history width", value: Binding(get: { draft.training.architecture.effectiveControlEmbedding }, set: { draft.training.architecture.controlEmbedding = $0 }), help: "Compresses the sparse 146-value historical action row before temporal processing, reducing compute and shortcut capacity.").disabled(draft.training.effectiveTemporalVision.pastFrameCount == 0); IntField("Recurrent width", value: $draft.training.architecture.recurrentWidth, help: "Capacity used to integrate paired past-frame embeddings and compressed controls.").disabled(draft.training.effectiveTemporalVision.pastFrameCount == 0); Picker("Temporal encoder", selection: $draft.training.architecture.recurrentKind) { ForEach(RecurrentKind.allCases) { Text($0.rawValue).tag($0) } }.disabled(draft.training.effectiveTemporalVision.pastFrameCount == 0) }
                        HStack {
                            LabeledContent("Spatial pooling", value: "Attention Keypoints")
                            IntField("Attention keypoints", value: Binding(get: { draft.training.architecture.effectiveAttentionHeads }, set: { draft.training.architecture.attentionHeads = $0 }), help: "Independent learned spatial queries. Each retains visual features and exact X/Y while global mean/max preserve whole-screen context.")
                            Spacer()
                            InfoTip("Policy v6 uses a dense coordinate-aware stem, depthwise spatial filters, pointwise channel mixing, and a same-width residual stage. Attention pooling preserves exact X/Y with a compact projection whose parameter count barely changes with resolution.")
                        }
                        HStack { InfoTip("Training encodes real current and reduced causal frames. Live inference caches each reduced visual embedding when first seen, so history length no longer multiplies visual-encoder work."); Spacer() }
                        HStack { IntArrayField("Stage channels", values: $draft.training.architecture.convolutionChannels); IntArrayField("Spatial kernels", values: $draft.training.architecture.kernelSizes); IntArrayField("Stage strides", values: $draft.training.architecture.strides); IntArrayField("Fusion widths", values: $draft.training.architecture.fusionWidths) }
                        HStack { DoubleField("Feature dropout", value: $draft.training.architecture.dropout, help: "Randomly hides fusion features and, at half strength, visual-embedding features during training. Inference disables it completely."); Spacer(); Text("Estimated parameters: \(ModelSizing.parameterCount(draft).formatted()) • runtime visual MACs: \(ModelSizing.runtimeVisualBackboneMultiplyAdds(draft).formatted())").font(.caption.monospacedDigit()).foregroundStyle(ATColor.cyan) }
                        } else {
                            ReinforcementConfigurationEditor(
                                configuration: Binding(
                                    get: { draft.effectiveReinforcement },
                                    set: { draft.reinforcement = $0 }
                                ),
                                hasExistingBrain: draft.activeVersionID != nil,
                                model: model
                            )
                        }
                    }
                }
                if trainingConfigurationTab == .trainingData {
                    NeuralNetworkInputOverview(profile: draft)
                    OLEDCard {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("Training recordings").font(.headline).foregroundStyle(ATColor.amber)
                            Text("Open a folder to select individual recordings, or select the folder to keep every recording in it included automatically.")
                                .font(.caption).foregroundStyle(.secondary)
                            ProfileRecordingPicker(
                                folders: model.recordingFolders,
                                recordings: model.recordings,
                                recordingIDs: $draft.recordingIDs,
                                folderIDs: Binding(get: { draft.effectiveFolderIDs }, set: { draft.recordingFolderIDs = $0 })
                            )
                        }
                    }
                }
                OLEDCard {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Saved brains and autosaves").font(.headline).foregroundStyle(ATColor.violet)
                            InfoTip("This list is loaded only when you ask for it, which keeps the page fast after long runs. Training Data brains preserve exact continuation checkpoints; RL brains preserve their separate online optimizer and resume from the selected immutable snapshot.")
                            Spacer()
                            if model.versionsLoadedForProfileID == draft.id {
                                Button("Hide List") { model.unloadVersions() }.primaryButton(color: ATColor.violet)
                            } else {
                                Button("Load Saved Brains") { Task { await model.refreshVersions() } }.primaryButton(color: ATColor.violet)
                            }
                        }
                        if model.isLoadingVersions { ProgressView("Loading saved brains…") }
                        else if model.versionsLoadedForProfileID != draft.id {
                            Text("The version list is hidden to avoid scanning autosaves during normal model editing.\(draft.trainingProgress.map { " \($0.savedBrainCount) saved brain\($0.savedBrainCount == 1 ? "" : "s") recorded." } ?? "")")
                                .font(.caption).foregroundStyle(.secondary)
                        } else if model.versions.isEmpty {
                            Text("Training snapshots and completed versions appear here.").foregroundStyle(.secondary)
                        } else {
                            ForEach(model.versions) { version in
                                HStack {
                                    VStack(alignment: .leading, spacing: 3) {
                                        HStack {
                                            Text(version.name).font(.subheadline.bold())
                                            if draft.activeVersionID == version.id {
                                                StatusPill(text: "Active", color: ATColor.green)
                                            }
                                            if version.reinforcementOptimizerFile != nil {
                                                StatusPill(text: "RL", color: ATColor.violet)
                                            }
                                        }
                                        Text("\(version.globalStep) steps • \(version.epoch ?? 0) epochs • train \(version.trainingLoss.formatted(.number.precision(.fractionLength(4))))\(version.validationLoss.map { " • \((version.trainingObjectiveSchema ?? 0) >= 2 ? "zero-history " : "")validation \($0.formatted(.number.precision(.fractionLength(4))))" } ?? "")\(version.validationReport?.binary.map { " • F1 \((100 * $0.f1).formatted(.number.precision(.fractionLength(1))))% over \(version.validationReport?.sampleCount ?? 0) samples" } ?? "")\(version.reinforcementFeedbackCount.map { " • RL \($0) feedback / \(version.reinforcementUpdateCount ?? 0) updates / net \((version.reinforcementNetReward ?? 0).formatted(.number.sign(strategy: .always()).precision(.fractionLength(0...2))))" } ?? "") • \(version.demonstratedKeyCodes.map { "\($0.count) learned keys" } ?? "legacy key set derived at run") • \(version.createdAt.formatted(date: .abbreviated, time: .shortened))")
                                            .font(.caption).foregroundStyle(.secondary)
                                        ForEach(version.autonomousRunQualityWarnings, id: \.self) { warning in
                                            Text("⚠︎ \(warning)").font(.caption2).foregroundStyle(ATColor.amber)
                                        }
                                        if let validationStep = version.validationGlobalStep,
                                           validationStep != version.globalStep {
                                            Text("Validation metrics were measured at step \(validationStep); this autosave contains newer exact weights from step \(version.globalStep).")
                                                .font(.caption2).foregroundStyle(.tertiary)
                                        }
                                    }
                                    Spacer(); Button(version.reinforcementOptimizerFile != nil ? "Resume RL" : version.optimizerFile == nil ? "Run this" : "Revert & Resume") {
                                        Task {
                                            if await model.activateVersion(version) { draft.activeVersionID = version.id }
                                        }
                                    }.primaryButton(color: ATColor.violet)
                                    if !draft.isDeletionProtected { Button("Delete") { Task { await model.deleteVersion(version) } }.primaryButton(color: ATColor.coral) }
                                }.padding(.vertical, 4)
                            }
                        }
                    }
                }
            }
            .disabled(model.isProfileBusy(draft.id))
        }
        .onChange(of: draft) { _, value in handleDraftChange(value) }
        .alert(acceptedDraft.isDeletionProtected ? "Duplicate the protected AI first" : "Change configuration and reset training?", isPresented: Binding(get: { pendingBrainReset != nil }, set: { if !$0 { pendingBrainReset = nil } })) {
            Button("Cancel", role: .cancel) { pendingBrainReset = nil }
            if acceptedDraft.isDeletionProtected {
                Button("Duplicate First") {
                    pendingBrainReset = nil
                    Task { await model.duplicateProfile(acceptedDraft) }
                }
            } else {
                Button("Change & Reset Brain", role: .destructive) { confirmBrainReset() }
            }
        } message: {
            if acceptedDraft.isDeletionProtected {
                Text("This protected AI cannot lose or replace its brain. Duplicate it with its training intact, then make the architecture, exact-vision, or temporal-context change on the copy.")
            } else {
                Text("This changes the AI's architecture, exact vision, or temporal-context contract. Its existing weights, optimizer state, saved brains, steps, and epochs cannot be attached safely and will be permanently cleared. Recordings are kept.")
            }
        }
    }

    private func temporalIntBinding(_ keyPath: WritableKeyPath<TemporalVisionConfiguration, Int>) -> Binding<Int> {
        Binding(
            get: { draft.training.effectiveTemporalVision[keyPath: keyPath] },
            set: { value in
                var temporal = draft.training.effectiveTemporalVision
                temporal[keyPath: keyPath] = value
                draft.training.temporalVision = temporal
            }
        )
    }

    private func generalizationDoubleBinding(
        _ keyPath: WritableKeyPath<GeneralizationConfiguration, Double>
    ) -> Binding<Double> {
        Binding(
            get: { draft.training.effectiveGeneralization[keyPath: keyPath] },
            set: { value in
                var configuration = draft.training.effectiveGeneralization
                configuration[keyPath: keyPath] = value
                draft.training.generalization = configuration
            }
        )
    }

    private var temporalFormatDetail: String {
        if draft.training.effectiveTemporalVision.pastFrameCount == 0 {
            return "Temporal payloads and recurrent parameters are omitted entirely; the network receives current vision only."
        }
        let colorDetail = draft.preprocessing.colorMode == .color
            ? "color mode and \(draft.preprocessing.chroma.rawValue) chroma"
            : "grayscale luminance"
        return "Past frames use the same \(colorDetail), \(draft.preprocessing.bitDepth)-bit detail, and resize policy as the current frame."
    }

    private func handleDraftChange(_ value: AIProfile) {
        if suppressDraftObserver {
            suppressDraftObserver = false
            return
        }
        if acceptedDraft.activeVersionID != nil, value.learnedBrainContract != acceptedDraft.learnedBrainContract {
            pendingBrainReset = value
            suppressDraftObserver = true
            draft = acceptedDraft
            return
        }
        acceptedDraft = value
        model.scheduleProfileAutosave(value)
    }

    private func confirmBrainReset() {
        guard let pending = pendingBrainReset else { return }
        pendingBrainReset = nil
        Task {
            guard let reset = await model.resetLearningAndSave(pending) else { return }
            acceptedDraft = reset
            suppressDraftObserver = true
            draft = reset
        }
    }
}

private struct ReinforcementConfigurationEditor: View {
    @Binding var configuration: ReinforcementConfiguration
    let hasExistingBrain: Bool
    @ObservedObject var model: AppModel
    @State private var showsAdvanced = false

    var body: some View {
        VStack(alignment: .leading, spacing: 13) {
            HStack(spacing: 12) {
                Toggle("Enable real-time reinforcement learning", isOn: $configuration.enabled)
                    .tint(ATColor.green)
                    .font(.headline)
                Spacer()
                StatusPill(
                    text: configuration.enabled ? "Run & Learn" : "Off",
                    color: configuration.enabled ? ATColor.green : .secondary
                )
            }
            Text(hasExistingBrain
                 ? "The active brain continues from its current weights. RL snapshots become normal runnable brains, and later Training Data runs warm-start from the latest RL brain without resetting it."
                 : "This AI can run immediately from a neutral, low-action policy. Reward and Punish teach it online; the first credited update creates its first runnable brain.")
                .font(.caption)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Label("Manual feedback", systemImage: "hand.tap.fill")
                        .font(.subheadline.bold())
                        .foregroundStyle(ATColor.green)
                    Spacer()
                    InfoTip("Feedback is timestamped the instant the shortcut, button, or wheel gesture occurs. The bounded causal window is credited even if the GPU update finishes after the scene changes.")
                }
                HStack(alignment: .bottom, spacing: 12) {
                    DoubleField("Reward amount", value: $configuration.rewardAmount, help: "Exact positive value submitted by the Reward shortcut and Run-page button (up to 100).")
                    ProfileHotkeyCaptureButton(
                        title: "Reward shortcut",
                        binding: $configuration.rewardHotkey,
                        color: ATColor.green,
                        model: model
                    )
                    DoubleField("Punish amount", value: $configuration.punishmentAmount, help: "Exact magnitude submitted as a negative value by the Punish shortcut and Run-page button (up to 100).")
                    ProfileHotkeyCaptureButton(
                        title: "Punish shortcut",
                        binding: $configuration.punishmentHotkey,
                        color: ATColor.coral,
                        model: model
                    )
                }
                if shortcutCollision {
                    Label("Reward and Punish must differ and cannot match Panic, Record, or Run in Settings.", systemImage: "exclamationmark.triangle.fill")
                        .font(.caption.bold())
                        .foregroundStyle(ATColor.coral)
                }
                Divider()
                Toggle("Modifier + scroll wheel gives exact feedback steps", isOn: $configuration.scrollFeedbackEnabled)
                    .tint(ATColor.cyan)
                if configuration.scrollFeedbackEnabled {
                    HStack(alignment: .bottom, spacing: 12) {
                        ReinforcementModifierChordEditor(modifiers: $configuration.scrollCarbonModifiers)
                        DoubleField("Amount per detent", value: $configuration.scrollStep, help: "Trackpad deltas accumulate to the same fixed step as a physical wheel detent, so the submitted amount is exact.")
                        Toggle("Scroll up rewards", isOn: $configuration.scrollUpRewards)
                            .tint(ATColor.green)
                        Spacer()
                    }
                    if configuration.scrollCarbonModifiers == 0 {
                        Text("Select at least one modifier so ordinary scrolling remains available to you and the AI.")
                            .font(.caption.bold()).foregroundStyle(ATColor.coral)
                    }
                }
            }
            .padding(12)
            .raisedGlassSurface(cornerRadius: 11, tint: ATColor.green)

            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Credit assignment").font(.subheadline.bold()).foregroundStyle(ATColor.cyan)
                    Spacer()
                    InfoTip("Recent decisions receive decayed credit, newest first. Updates are serialized with inference and clipped to keep manual training stable.")
                }
                HStack {
                    DoubleField("Window (seconds)", value: $configuration.creditWindowSeconds, help: "How far before your feedback the learner looks for causal actions (0.05–10 seconds).")
                    DoubleField("Older-frame decay", value: $configuration.creditDecay, help: "Multiplier applied for each older credited decision (0.1–1).")
                    IntField("Maximum frames", value: $configuration.maximumCreditFrames, help: "Hard cap on transitions updated by one feedback event (1–64).")
                    IntField("Autosave feedback", value: $configuration.autosaveFeedbackCount, help: "Publish an immutable RL autosave after this many feedback events that produced an update.")
                }
                Toggle("Learn from inactive keys and buttons", isOn: $configuration.learnFromInaction)
                    .tint(ATColor.amber)
                Text(configuration.learnFromInaction
                     ? "Both action and inaction are credited. Punishment can therefore make previously inactive controls more likely; use this only for deliberate stillness/waiting shaping."
                     : "Recommended: credit follows activated keys/buttons and meaningful movement, avoiding punishment that accidentally promotes every inactive control.")
                    .font(.caption).foregroundStyle(configuration.learnFromInaction ? ATColor.amber : .secondary)
            }
            .padding(12)
            .raisedGlassSurface(cornerRadius: 11, tint: ATColor.cyan)

            ReinforcementAllowedKeysEditor(keys: $configuration.allowedKeyCodes)

            DisclosureGroup("Advanced stability and exploration", isExpanded: $showsAdvanced) {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        DoubleField("RL learning rate", value: $configuration.learningRate, help: "AdamW update size for online learning. This optimizer is separate from Training Data continuation.")
                        DoubleField("Binary exploration", value: $configuration.binaryExploration, help: "Probability mix with an unbiased binary action sample. New policies otherwise begin with a low 0.2% binary-action probability.")
                        DoubleField("Continuous noise", value: $configuration.continuousExplorationStandardDeviation, help: "Standard deviation for bounded mouse and scroll exploration.")
                    }
                    HStack {
                        DoubleField("PPO clip", value: $configuration.policyClip, help: "Bounds the likelihood-ratio change from the behavior policy that generated each action.")
                        DoubleField("Behavior anchor", value: $configuration.behaviorAnchor, help: "KL-style penalty that resists abrupt drift from the action-time policy.")
                        DoubleField("Entropy bonus", value: $configuration.entropyBonus, help: "Small incentive to preserve exploration in binary controls.")
                        DoubleField("Gradient norm", value: $configuration.maximumGradientNorm, help: "Global gradient clipping threshold for each online update.")
                    }
                    HStack {
                        Button("Recommended defaults") {
                            let enabled = configuration.enabled
                            let allowedKeys = configuration.allowedKeyCodes
                            configuration = ReinforcementConfiguration()
                            configuration.enabled = enabled
                            configuration.allowedKeyCodes = allowedKeys
                        }
                        .primaryButton(color: ATColor.amber)
                        Spacer()
                        Text("Online updates use clipped policy gradients, a fixed exploration distribution, behavior anchoring, finite checks, and a bounded 256 MB transition budget.")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
                .padding(.top, 10)
            }

            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "viewfinder.circle")
                    .foregroundStyle(ATColor.violet)
                VStack(alignment: .leading, spacing: 3) {
                    Text("Ready for automatic rewards").font(.subheadline.bold()).foregroundStyle(ATColor.violet)
                    Text("Future screen-element, shape, color, OCR, or game-state detectors plug into the same timestamped signal interface as manual controls. They do not require a new optimizer, brain format, or persistence path.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(11)
            .raisedGlassSurface(cornerRadius: 11, tint: ATColor.violet)
        }
        .opacity(configuration.enabled ? 1 : 0.78)
    }

    private var shortcutCollision: Bool {
        let feedback = Set([configuration.rewardHotkey, configuration.punishmentHotkey])
        return feedback.count != 2
            || !feedback.isDisjoint(with: [model.hotkeys.panic, model.hotkeys.record, model.hotkeys.run])
    }
}

private struct ProfileHotkeyCaptureButton: View {
    let title: String
    @Binding var binding: HotkeyBinding
    let color: Color
    @ObservedObject var model: AppModel
    @State private var isListening = false
    @State private var keyMonitor: Any?
    @State private var mouseMonitor: Any?

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Button(isListening ? "Press shortcut…" : binding.shortcutDisplayName) {
                isListening ? stop() : begin()
            }
            .primaryButton(color: isListening ? ATColor.amber : color)
            .help("Keyboard and mouse-button shortcuts are supported. Add modifiers to avoid intercepting normal input.")
        }
        .onDisappear { stop() }
    }

    private func begin() {
        stop(resume: false)
        model.suspendGlobalHotkeys()
        isListening = true
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            guard !event.isARepeat else { return nil }
            binding = HotkeyBinding(
                keyCode: UInt32(event.keyCode),
                carbonModifiers: event.modifierFlags.carbonHotkeyModifiers
            )
            stop()
            return nil
        }
        mouseMonitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown, .otherMouseDown]) { event in
            binding = .mouse(
                UInt8(clamping: event.buttonNumber),
                carbonModifiers: event.modifierFlags.carbonHotkeyModifiers
            )
            stop()
            return nil
        }
    }

    private func stop(resume: Bool = true) {
        if let keyMonitor { NSEvent.removeMonitor(keyMonitor); self.keyMonitor = nil }
        if let mouseMonitor { NSEvent.removeMonitor(mouseMonitor); self.mouseMonitor = nil }
        let wasListening = isListening
        isListening = false
        if resume, wasListening { model.resumeGlobalHotkeys() }
    }
}

private struct ReinforcementModifierChordEditor: View {
    @Binding var modifiers: UInt32
    private let options: [(String, UInt32)] = [
        ("⌃ Control", UInt32(1 << 12)),
        ("⌥ Option", UInt32(1 << 11)),
        ("⇧ Shift", UInt32(1 << 9)),
        ("⌘ Command", UInt32(1 << 8))
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("Scroll modifiers").font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 5) {
                ForEach(options, id: \.1) { option in
                    Button(option.0) { modifiers ^= option.1 }
                        .buttonStyle(.plain)
                        .font(.caption.bold())
                        .padding(.horizontal, 7)
                        .padding(.vertical, 6)
                        .background(RoundedRectangle(cornerRadius: 7).fill(modifiers & option.1 != 0 ? ATColor.cyan.opacity(0.22) : Color.white.opacity(0.04)))
                        .overlay(RoundedRectangle(cornerRadius: 7).stroke(modifiers & option.1 != 0 ? ATColor.cyan : ATColor.border, lineWidth: 0.8))
                }
            }
        }
    }
}

private struct ReinforcementAllowedKeysEditor: View {
    @Binding var keys: Set<UInt16>
    private let keyRows: [[UInt16]] = [
        [18, 19, 20, 21, 23, 22, 26, 28, 25, 29],
        [48, 12, 13, 14, 15, 17, 16, 32, 34, 31, 35, 33, 30],
        [57, 0, 1, 2, 3, 5, 4, 38, 40, 37, 41, 39, 36],
        [56, 6, 7, 8, 9, 11, 45, 46, 43, 47, 44, 49],
        [59, 58, 55, 123, 125, 124, 126]
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Text("Brand-new AI keyboard firewall").font(.subheadline.bold()).foregroundStyle(ATColor.violet)
                InfoTip("A new policy may emit only these explicitly allowed keys. Existing trained brains use the union of this set and demonstrated keys. Per-AI blocked outputs and the Run keyboard switch still take priority.")
                Spacer()
                StatusPill(text: "\(keys.count) allowed", color: keys.isEmpty ? ATColor.amber : ATColor.violet)
                Button("WASD + Space") { keys.formUnion([0, 1, 2, 13, 49]) }.primaryButton(color: ATColor.violet)
                Button("Clear") { keys.removeAll() }.primaryButton()
            }
            ForEach(keyRows.indices, id: \.self) { row in
                HStack(spacing: 5) {
                    ForEach(keyRows[row], id: \.self) { code in
                        let selected = keys.contains(code)
                        Button(KeyNames.name(for: code)) {
                            if selected { keys.remove(code) } else { keys.insert(code) }
                        }
                        .buttonStyle(.plain)
                        .font(.system(size: 10, weight: .bold, design: .rounded))
                        .frame(minWidth: code == 49 ? 74 : 28, minHeight: 25)
                        .background(RoundedRectangle(cornerRadius: 7).fill(selected ? ATColor.violet.opacity(0.25) : Color.white.opacity(0.035)))
                        .overlay(RoundedRectangle(cornerRadius: 7).stroke(selected ? ATColor.violet : ATColor.border, lineWidth: 0.8))
                    }
                    Spacer(minLength: 0)
                }
            }
            if keys.isEmpty {
                Text("No keyboard key can be explored by a brand-new AI. Mouse, buttons, and scroll still follow the Control Channels and runtime firewalls above.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .raisedGlassSurface(cornerRadius: 11, tint: ATColor.violet)
    }
}

private extension NSEvent.ModifierFlags {
    var carbonHotkeyModifiers: UInt32 {
        var result: UInt32 = 0
        if contains(.command) { result |= UInt32(1 << 8) }
        if contains(.shift) { result |= UInt32(1 << 9) }
        if contains(.option) { result |= UInt32(1 << 11) }
        if contains(.control) { result |= UInt32(1 << 12) }
        return result
    }
}

private struct NeuralNetworkInputOverview: View {
    let profile: AIProfile
    @State private var showsTechnicalDetails = false

    var body: some View {
        let input = NeuralInputSizing.summary(for: profile)
        let capacity = NeuralInputSizing.capacityGuide(for: profile)
        let color = statusColor(capacity.level)
        OLEDCard {
            VStack(alignment: .leading, spacing: 13) {
                HStack(spacing: 7) {
                    Text("Input size check").font(.headline).foregroundStyle(ATColor.cyan)
                    InfoTip("This is a quick comparison between the values used for one decision and the number of learned parameters in the selected network. Comfortable or Balanced means there is no obvious size mismatch. High or Too high suggests choosing a larger architecture preset, reducing resolution, using fewer past frames, or increasing their downscale. It is a practical guide, not a guarantee of training quality.")
                    Spacer()
                    StatusPill(text: statusTitle(capacity.level), color: color)
                }

                HStack(alignment: .top, spacing: 12) {
                    Image(systemName: statusSymbol(capacity.level))
                        .font(.system(size: 28, weight: .semibold)).foregroundStyle(color)
                        .frame(width: 42, height: 42)
                        .background(Circle().fill(color.opacity(0.13)))
                    VStack(alignment: .leading, spacing: 4) {
                        Text(statusHeadline(capacity.level)).font(.title3.bold()).foregroundStyle(color)
                        Text(statusExplanation(capacity.level)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer(minLength: 0)
                }
                .padding(12)
                .raisedGlassSurface(cornerRadius: 12, tint: color)

                HStack(spacing: 10) {
                    NeuralInputMetric(
                        title: "Inputs each decision",
                        value: shortCount(capacity.inputValues),
                        detail: "\(capacity.inputValues.formatted()) exact values",
                        color: ATColor.cyan
                    )
                    Image(systemName: "arrow.right").font(.headline).foregroundStyle(.tertiary)
                    NeuralInputMetric(
                        title: "Selected network size",
                        value: shortCount(capacity.parameterCount),
                        detail: "\(capacity.parameterCount.formatted()) learned parameters",
                        color: color
                    )
                }

                DisclosureGroup(isExpanded: $showsTechnicalDetails) {
                    technicalDetails(input, capacity: capacity)
                        .padding(.top, 9)
                } label: {
                    HStack {
                        Label(showsTechnicalDetails ? "Hide technical details" : "Show technical details", systemImage: "slider.horizontal.3")
                            .font(.caption.bold()).foregroundStyle(ATColor.cyan)
                        Spacer()
                        Text("current vision, past frames, controls, memory, and rates").font(.caption2).foregroundStyle(.secondary)
                    }
                }
                .tint(ATColor.cyan)
            }
        }
    }

    @ViewBuilder
    private func technicalDetails(_ input: NeuralInputSummary, capacity: NeuralInputCapacityGuide) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                NeuralInputMetric(
                    title: "Current-frame convolution",
                    value: input.currentFirstConvolutionValues.formatted(),
                    detail: "\(profile.preprocessing.width) × \(profile.preprocessing.height) × \(profile.preprocessing.channelCount + 2)",
                    color: ATColor.violet
                )
                NeuralInputMetric(
                    title: "Whole training batch",
                    value: input.valuesPerTrainingBatch.formatted(),
                    detail: "\(input.valuesPerDecision.formatted()) × batch \(input.batchSize.formatted())",
                    color: ATColor.green
                )
            }

            LazyVGrid(columns: [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)], alignment: .leading, spacing: 10) {
                NeuralInputBreakdown(
                    title: "Packed current frame",
                    value: "\(input.currentPackedVisionValues.formatted()) UInt8 values",
                    detail: currentPackedVisionDetail(input),
                    color: ATColor.amber
                )
                NeuralInputBreakdown(
                    title: "Packed past frames",
                    value: "\((input.pastPackedVisionValuesPerFrame * input.pastFrameCount).formatted()) UInt8 values",
                    detail: pastPackedVisionDetail(input),
                    color: ATColor.amber
                )
                NeuralInputBreakdown(
                    title: "Expanded current vision",
                    value: "\(input.currentExpandedVisionValues.formatted()) values",
                    detail: "\(profile.preprocessing.width) × \(profile.preprocessing.height) × \(profile.preprocessing.channelCount) after chroma expansion",
                    color: ATColor.cyan
                )
                NeuralInputBreakdown(
                    title: "Expanded past vision",
                    value: "\((input.pastExpandedVisionValuesPerFrame * input.pastFrameCount).formatted()) values",
                    detail: "\(input.pastFrameCount.formatted()) real lower-resolution frames; no motion extraction or upscaling",
                    color: ATColor.cyan
                )
                NeuralInputBreakdown(
                    title: "Generated coordinates",
                    value: "\((input.currentCoordinateValues + input.pastCoordinateValuesPerFrame * input.pastFrameCount).formatted()) values",
                    detail: "Independent X/Y planes for the current frame and every native-size past frame",
                    color: ATColor.violet
                )
                NeuralInputBreakdown(
                    title: "Frame-aligned controls",
                    value: "\(input.pastControlValues.formatted()) values",
                    detail: "\(input.pastFrameCount.formatted()) frames × \(input.actionValuesPerPastFrame.formatted()) complete outputs • \(input.temporalLookbackSeconds.formatted(.number.precision(.fractionLength(3))))s lookback",
                    color: ATColor.green
                )
            }

            HStack(spacing: 8) {
                Image(systemName: "memorychip").foregroundStyle(ATColor.cyan)
                Text("Nominal input payload at \(profile.training.precision.rawValue): \(bytes(input.nominalBytesPerDecision)) per decision • \(bytes(input.nominalBytesPerTrainingBatch)) per batch")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                Spacer()
            }
            HStack(spacing: 8) {
                Image(systemName: "speedometer").foregroundStyle(ATColor.green)
                Text("Live cache at \(profile.training.perceptionFPS.formatted()) Perception FPS: \(input.runtimeEncodedVisionValues.formatted()) newly encoded vision values + \(input.runtimeCachedVisualValues.formatted()) cached visual features + \(input.pastControlValues.formatted()) controls per decision • \(input.runtimeValuesPerSecond.formatted()) graph values/s from \(bytes(input.packedVisionBytesPerSecond)) packed vision/s")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                Spacer()
            }
            Text("The \(statusTitle(capacity.level).lowercased()) capacity status is a conservative guide based on \(capacity.inputsPerParameter.formatted(.number.precision(.fractionLength(2)))) input values per learned parameter. Convolutional sharing means this is not a hard limit or a promise of training quality. Resize changes framing; enabled controls change losses; architecture widths change the network size.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func statusTitle(_ level: NeuralInputCapacityLevel) -> String {
        switch level {
        case .comfortable: "Comfortable"
        case .balanced: "Balanced"
        case .high: "High"
        case .tooHigh: "Too high"
        }
    }

    private func statusHeadline(_ level: NeuralInputCapacityLevel) -> String {
        switch level {
        case .comfortable: "Comfortable for this network"
        case .balanced: "A reasonable match"
        case .high: "Input is large for this network"
        case .tooHigh: "Probably too many inputs"
        }
    }

    private func statusExplanation(_ level: NeuralInputCapacityLevel) -> String {
        switch level {
        case .comfortable:
            "No change needed. The selected network is comfortably sized for this input."
        case .balanced:
            "This input and network size are reasonably matched. You can train with these settings."
        case .high:
            "Consider a larger architecture preset, lower the resolution, use fewer past frames, or increase their downscale."
        case .tooHigh:
            "Choose a larger architecture preset or reduce current/past visual input before training."
        }
    }

    private func statusSymbol(_ level: NeuralInputCapacityLevel) -> String {
        switch level {
        case .comfortable: "checkmark.circle.fill"
        case .balanced: "equal.circle.fill"
        case .high: "exclamationmark.triangle.fill"
        case .tooHigh: "xmark.octagon.fill"
        }
    }

    private func statusColor(_ level: NeuralInputCapacityLevel) -> Color {
        switch level {
        case .comfortable: ATColor.green
        case .balanced: ATColor.cyan
        case .high: ATColor.amber
        case .tooHigh: ATColor.coral
        }
    }

    private func shortCount(_ count: Int64) -> String {
        let value = Double(max(0, count))
        if value >= 1_000_000 {
            let scaled = value / 1_000_000
            return String(format: scaled < 10 ? "%.1fM" : "%.0fM", scaled)
        }
        if value >= 1_000 {
            let scaled = value / 1_000
            return String(format: scaled < 10 ? "%.1fK" : "%.0fK", scaled)
        }
        return count.formatted()
    }

    private func currentPackedVisionDetail(_ input: NeuralInputSummary) -> String {
        let storedBytes = bytes(input.currentPackedVisionValues)
        let levels = input.quantizationLevels.formatted()
        if profile.preprocessing.colorMode == .grayscale {
            return "\(input.currentLumaValues.formatted()) Y • \(storedBytes) stored • \(levels) levels"
        }
        return "\(input.currentLumaValues.formatted()) Y + 2 × \(input.currentChromaValuesPerPlane.formatted()) chroma at \(profile.preprocessing.chroma.rawValue) • \(storedBytes) stored • \(levels) levels"
    }

    private func pastPackedVisionDetail(_ input: NeuralInputSummary) -> String {
        if input.pastFrameCount == 0 {
            return "Disabled • no past-image or frame-control payload"
        }
        let pastSpec = profile.training.effectiveTemporalVision.pastFrameSpec(from: profile.preprocessing)
        let colorDetail = profile.preprocessing.colorMode == .color
            ? "color/\(profile.preprocessing.chroma.rawValue)"
            : "grayscale luminance"
        return "\(input.pastFrameCount.formatted()) × \(pastSpec.width) × \(pastSpec.height) • same \(colorDetail) and \(profile.preprocessing.bitDepth)-bit detail • every \(input.frameSpacingSeconds.formatted(.number.precision(.fractionLength(3))))s"
    }

    private func bytes(_ count: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: count, countStyle: .memory)
    }
}

private struct NeuralInputMetric: View {
    let title: String
    let value: String
    let detail: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(value).font(.title3.bold().monospacedDigit()).foregroundStyle(color).lineLimit(1).minimumScaleFactor(0.72)
            Text(title).font(.caption.bold())
            Text(detail).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
        }
        .padding(11)
        .frame(maxWidth: .infinity, alignment: .leading)
        .raisedGlassSurface(cornerRadius: 10, tint: color)
    }
}

private struct NeuralInputBreakdown: View {
    let title: String
    let value: String
    let detail: String
    let color: Color

    var body: some View {
        HStack(alignment: .top, spacing: 9) {
            Circle().fill(color).frame(width: 7, height: 7).padding(.top, 5)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.caption.bold()).foregroundStyle(color)
                Text(value).font(.subheadline.monospacedDigit())
                Text(detail).font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .raisedGlassSurface(cornerRadius: 10)
    }
}

private struct TrainingTimeOverview: View {
    let profile: AIProfile
    let recordings: [RecordingItem]

    var body: some View {
        let timing = profile.trainingDurationSummary(recordings: recordings)
        OLEDCard(padding: 14) {
            HStack(spacing: 24) {
                Label {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(TrainingDurationFormatter.string(seconds: timing.trainingSeconds))
                            .font(.title3.bold().monospacedDigit()).foregroundStyle(ATColor.cyan)
                        Text("Actual training time").font(.caption).foregroundStyle(.secondary)
                    }
                } icon: { Image(systemName: "clock.fill").foregroundStyle(ATColor.cyan) }
                Divider().frame(height: 34)
                Label {
                    VStack(alignment: .leading, spacing: 3) {
                        Text((timing.experienceIsEstimated ? "~" : "") + TrainingDurationFormatter.string(seconds: timing.experienceSeconds))
                            .font(.title3.bold().monospacedDigit()).foregroundStyle(ATColor.violet)
                        Text("Equivalent demonstration experience").font(.caption).foregroundStyle(.secondary)
                    }
                } icon: { Image(systemName: "brain.head.profile.fill").foregroundStyle(ATColor.violet) }
                InfoTip("Actual time is optimizer wall-clock time saved with this brain. Experience is the duration of demonstration samples the optimizer has consumed across repeated epochs; for example, one real hour can process ten hours of examples.")
                Spacer()
            }
        }
    }
}

private struct ProfileRecordingPicker: View {
    let folders: [RecordingFolder]
    let recordings: [RecordingItem]
    @Binding var recordingIDs: [UUID]
    @Binding var folderIDs: [UUID]
    @State private var expandedFolderIDs: Set<UUID> = []

    var body: some View {
        VStack(spacing: 8) {
            ForEach(folders) { folder in
                let items = recordings.filter { $0.manifest.folderID == folder.id }
                DisclosureGroup(isExpanded: Binding(
                    get: { expandedFolderIDs.contains(folder.id) },
                    set: { expanded in
                        if expanded { expandedFolderIDs.insert(folder.id) }
                        else { expandedFolderIDs.remove(folder.id) }
                    }
                )) {
                    VStack(spacing: 5) {
                        if items.isEmpty {
                            Text("No recordings in this folder")
                                .font(.caption).foregroundStyle(.tertiary)
                                .frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 6)
                        } else {
                            ForEach(items) { recording in
                                Toggle(isOn: individualBinding(recording, folder: folder)) {
                                    HStack {
                                        Image(systemName: "play.rectangle.fill").foregroundStyle(ATColor.cyan)
                                        Text(recording.manifest.name).lineLimit(1)
                                        Spacer()
                                        Text(duration(recording)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                                    }
                                }
                                .disabled(folderIDs.contains(folder.id))
                                .padding(.leading, 20)
                                .help(folderIDs.contains(folder.id) ? "This recording is included because the whole folder is selected." : "Include only this recording")
                            }
                        }
                    }
                    .padding(.top, 7)
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: expandedFolderIDs.contains(folder.id) ? "folder.fill.badge.minus" : "folder.fill")
                            .foregroundStyle(ATColor.violet)
                        Text(folder.name).font(.subheadline.bold()).lineLimit(1)
                        Text("\(items.count)").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                        Spacer()
                        Toggle("All", isOn: folderBinding(folder, items: items))
                            .toggleStyle(.switch).controlSize(.small).fixedSize()
                    }
                }
                .tint(ATColor.cyan)
                .padding(10)
                .raisedGlassSurface(cornerRadius: 11, tint: folderIDs.contains(folder.id) ? ATColor.violet : nil)
            }
            if folders.isEmpty {
                ContentUnavailableView("No recording folders", systemImage: "folder.badge.plus", description: Text("Record examples before training."))
                    .frame(maxWidth: .infinity).padding(.vertical, 18)
            }
        }
    }

    private func folderBinding(_ folder: RecordingFolder, items: [RecordingItem]) -> Binding<Bool> {
        Binding(
            get: { folderIDs.contains(folder.id) },
            set: { enabled in
                if enabled {
                    if !folderIDs.contains(folder.id) { folderIDs.append(folder.id) }
                    let itemIDs = Set(items.map(\.id))
                    recordingIDs.removeAll { itemIDs.contains($0) }
                    expandedFolderIDs.insert(folder.id)
                } else {
                    folderIDs.removeAll { $0 == folder.id }
                }
            }
        )
    }

    private func individualBinding(_ recording: RecordingItem, folder: RecordingFolder) -> Binding<Bool> {
        Binding(
            get: { folderIDs.contains(folder.id) || recordingIDs.contains(recording.id) },
            set: { enabled in
                if enabled {
                    if !recordingIDs.contains(recording.id) { recordingIDs.append(recording.id) }
                } else {
                    recordingIDs.removeAll { $0 == recording.id }
                }
            }
        )
    }

    private func duration(_ recording: RecordingItem) -> String {
        let seconds = Int(ceil(max(0, (recording.manifest.trimEnd ?? recording.manifest.duration) - recording.manifest.trimStart)))
        return String(format: "%d:%02d", seconds / 60, seconds % 60)
    }
}

struct InfoTip: View {
    let text: String
    @State private var isPresented = false
    init(_ text: String) { self.text = text }
    var body: some View {
        Button { isPresented.toggle() } label: {
            Image(systemName: "info.circle")
                .font(.caption)
                .frame(width: 28, height: 28)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(ATColor.cyan.opacity(0.9))
        .uiHoverResponse(scale: 1.08)
        .help(text)
        .popover(isPresented: $isPresented) {
            VStack(alignment: .leading, spacing: 8) {
                Label("How this works", systemImage: "info.circle.fill").font(.headline).foregroundStyle(ATColor.cyan)
                Text(text).font(.callout).fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
            }
            .padding(14)
            .frame(width: 330, alignment: .leading)
        }
        .accessibilityLabel("Information")
        .accessibilityHint(text)
    }
}

private struct KeyRestrictionGrid: View {
    @Binding var restrictions: ActionRestrictions
    @ObservedObject var model: AppModel
    @State private var keyMonitor: Any?
    @State private var mouseMonitor: Any?
    @State private var listeningForKeys = false
    @State private var listeningForMouse = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Add only the controls the AI must never emit. Restrictions apply to training targets and live execution.").font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 8) {
                Button { listeningForKeys ? stopKeyCapture() : startKeyCapture() } label: { Label(listeningForKeys ? "Done adding keys" : "Press keys to block", systemImage: listeningForKeys ? "checkmark.circle.fill" : "keyboard.badge.ellipsis") }.primaryButton(color: listeningForKeys ? ATColor.green : ATColor.cyan)
                Button { startMouseCapture() } label: { Label(listeningForMouse ? "Click a mouse button…" : "Capture mouse button", systemImage: "computermouse") }.primaryButton(color: listeningForMouse ? ATColor.amber : ATColor.violet).disabled(listeningForMouse)
                if !restrictions.blockedKeyCodes.isEmpty || !restrictions.blockedMouseButtons.isEmpty { Button("Clear All") { restrictions = ActionRestrictions() }.buttonStyle(.plain).foregroundStyle(ATColor.coral).font(.caption.bold()) }
            }

            if restrictions.blockedKeyCodes.isEmpty && restrictions.blockedMouseButtons.isEmpty {
                Text(listeningForKeys ? "Listening — press one or more keyboard keys." : listeningForMouse ? "Listening — click the mouse button to block." : "No blocked controls.").font(.caption).foregroundStyle(listeningForKeys || listeningForMouse ? ATColor.amber : Color.secondary)
            } else {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 76), spacing: 6)], alignment: .leading, spacing: 6) {
                    ForEach(restrictions.blockedKeyCodes.sorted(), id: \.self) { code in RestrictionChip(label: KeyNames.name(for: code), symbol: "keyboard") { restrictions.blockedKeyCodes.remove(code) } }
                    ForEach(restrictions.blockedMouseButtons.sorted(), id: \.self) { button in RestrictionChip(label: mouseName(button), symbol: "computermouse") { restrictions.blockedMouseButtons.remove(button) } }
                }
            }
        }
        .padding(10)
        .raisedGlassSurface(cornerRadius: 11, tint: listeningForKeys || listeningForMouse ? ATColor.amber : ATColor.raised)
        .overlay(RoundedRectangle(cornerRadius: ATCorner.scaled(11), style: .continuous).stroke(listeningForKeys || listeningForMouse ? ATColor.amber.opacity(0.65) : ATColor.border))
        .onDisappear { stopKeyCapture(); stopMouseCapture() }
    }

    private func startKeyCapture() {
        stopMouseCapture(); stopKeyCapture(resumeHotkeys: false); model.suspendGlobalHotkeys(); listeningForKeys = true
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            restrictions.blockedKeyCodes.insert(event.keyCode)
            return nil
        }
    }

    private func stopKeyCapture(resumeHotkeys: Bool = true) {
        if let keyMonitor { NSEvent.removeMonitor(keyMonitor); self.keyMonitor = nil }
        let wasListening = listeningForKeys; listeningForKeys = false
        if wasListening && resumeHotkeys { model.resumeGlobalHotkeys() }
    }

    private func startMouseCapture() {
        stopKeyCapture(); stopMouseCapture(); listeningForMouse = true
        mouseMonitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown, .otherMouseDown]) { event in
            restrictions.blockedMouseButtons.insert(UInt8(clamping: event.buttonNumber))
            stopMouseCapture()
            return nil
        }
    }

    private func stopMouseCapture() {
        if let mouseMonitor { NSEvent.removeMonitor(mouseMonitor); self.mouseMonitor = nil }
        listeningForMouse = false
    }

    private func mouseName(_ button: UInt8) -> String {
        switch button { case 0: "Left Mouse"; case 1: "Right Mouse"; case 2: "Middle Mouse"; default: "Mouse \(Int(button) + 1)" }
    }
}

private struct RecordingKeyBlacklistEditor: View {
    @Binding var keys: Set<UInt16>
    @ObservedObject var model: AppModel
    @State private var monitor: Any?
    @State private var listening = false

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Button(listening ? "Done" : "Press keys to exclude") { listening ? stop() : start() }.primaryButton(color: listening ? ATColor.green : ATColor.coral)
                if !keys.isEmpty { Button("Clear") { keys.removeAll() }.buttonStyle(.plain).foregroundStyle(ATColor.coral).font(.caption.bold()) }
                InfoTip("Excluded keys never enter the recording event file. If a modifier key is excluded, its modifier flag is also removed from every recorded input sample.")
            }
            if keys.isEmpty { Text(listening ? "Listening — press any keys to blacklist." : "No excluded keys.").font(.caption).foregroundStyle(listening ? ATColor.amber : Color.secondary) }
            else {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 55), spacing: 5)], alignment: .leading, spacing: 5) {
                    ForEach(keys.sorted(), id: \.self) { code in RestrictionChip(label: KeyNames.name(for: code), symbol: "keyboard") { keys.remove(code) } }
                }
            }
        }.onDisappear { stop() }
    }

    private func start() {
        stop(resumeHotkeys: false); model.suspendGlobalHotkeys(); listening = true
        monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in keys.insert(event.keyCode); return nil }
    }
    private func stop(resumeHotkeys: Bool = true) { if let monitor { NSEvent.removeMonitor(monitor); self.monitor = nil }; let wasListening = listening; listening = false; if wasListening && resumeHotkeys { model.resumeGlobalHotkeys() } }
}

private struct RestrictionChip: View {
    let label: String; let symbol: String; let remove: () -> Void
    var body: some View {
        HStack(spacing: 5) { Image(systemName: symbol).font(.caption2); Text(label).lineLimit(1); Button(action: remove) { Image(systemName: "xmark.circle.fill") }.buttonStyle(.plain).foregroundStyle(.secondary) }
            .font(.caption.bold()).padding(.horizontal, 8).padding(.vertical, 6).background(Capsule().fill(ATColor.coral.opacity(0.14))).overlay(Capsule().stroke(ATColor.coral.opacity(0.55)))
            .uiHoverResponse(scale: 1.025)
    }
}

private struct IntField: View { let title: String; @Binding var value: Int; var help: String?; init(_ title: String, value: Binding<Int>, help: String? = nil) { self.title = title; _value = value; self.help = help }; var body: some View { VStack(alignment: .leading, spacing: 4) { HStack(spacing: 4) { Text(title); if let help { InfoTip(help) } }.font(.caption).foregroundStyle(.secondary); TextField(title, value: $value, format: .number).textFieldStyle(.roundedBorder) }.frame(minWidth: 92) } }
private struct DoubleField: View { let title: String; @Binding var value: Double; var help: String?; init(_ title: String, value: Binding<Double>, help: String? = nil) { self.title = title; _value = value; self.help = help }; var body: some View { VStack(alignment: .leading, spacing: 4) { HStack(spacing: 4) { Text(title); if let help { InfoTip(help) } }.font(.caption).foregroundStyle(.secondary); TextField(title, value: $value, format: .number).textFieldStyle(.roundedBorder) }.frame(minWidth: 110) } }
private struct IntArrayField: View {
    let title: String
    @Binding var values: [Int]
    @State private var text: String
    init(_ title: String, values: Binding<[Int]>) { self.title = title; _values = values; _text = State(initialValue: values.wrappedValue.map(String.init).joined(separator: ", ")) }
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            TextField("comma separated", text: $text).textFieldStyle(.roundedBorder).onSubmit { commit() }.onDisappear { commit() }
        }.frame(minWidth: 120)
    }
    private func commit() { let parsed = text.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }; if !parsed.isEmpty { values = parsed; text = parsed.map(String.init).joined(separator: ", ") } }
}

struct TrainingView: View {
    private enum GraphKind: String, CaseIterable, Identifiable {
        case current = "Batch Loss"
        case epoch = "Epoch Average"
        case validation = "Validation"
        case learningRate = "Learning Rate"
        var id: String { rawValue }
    }
    @ObservedObject var model: AppModel
    @State private var graphKind: GraphKind = .current
    @State private var zoomFraction = 0.25

    private var displayedProfile: AIProfile? {
        model.trainingProfileID.flatMap { id in model.profiles.first(where: { $0.id == id }) } ?? model.selectedProfile
    }

    private var displayedTiming: TrainingDurationSummary {
        guard let profile = displayedProfile else { return TrainingDurationSummary(trainingSeconds: 0, experienceSeconds: 0, experienceIsEstimated: false) }
        let persisted = profile.trainingDurationSummary(recordings: model.recordings)
        guard model.isTraining, model.trainingProfileID == profile.id else { return persisted }
        return TrainingDurationSummary(
            trainingSeconds: max(persisted.trainingSeconds, model.trainingMetrics.elapsed),
            experienceSeconds: max(persisted.experienceSeconds, model.trainingMetrics.experienceElapsed),
            experienceIsEstimated: false
        )
    }

    private var displayedSteps: Int {
        guard let profile = displayedProfile else { return 0 }
        if model.isTraining, model.trainingProfileID == profile.id {
            return max(profile.trainingProgress?.globalStep ?? 0, model.trainingMetrics.globalStep)
        }
        return profile.trainingProgress?.globalStep ?? 0
    }

    private var graphValues: [Double] {
        switch graphKind {
        case .current: model.trainingMetrics.lossHistory
        case .epoch: model.trainingMetrics.epochLossHistory
        case .validation: model.trainingMetrics.validationHistory
        case .learningRate: model.trainingMetrics.learningRateHistory
        }
    }

    private var graphAccent: Color {
        switch graphKind {
        case .current: ATColor.cyan
        case .epoch: ATColor.green
        case .validation: ATColor.violet
        case .learningRate: ATColor.amber
        }
    }

    private func lossText(_ value: Double?) -> String {
        value.map { $0.formatted(.number.precision(.fractionLength(6))) } ?? "—"
    }

    private func validationSummary(_ report: ValidationReport) -> String {
        var components = ["Zero-history stress validation: \(report.sampleCount.formatted()) diverse held-out samples"]
        if let binary = report.binary {
            components.append("binary precision \((100 * binary.precision).formatted(.number.precision(.fractionLength(1))))%")
            components.append("recall \((100 * binary.recall).formatted(.number.precision(.fractionLength(1))))%")
            components.append("F1 \((100 * binary.f1).formatted(.number.precision(.fractionLength(1))))%")
            components.append("false-positive rate \((100 * binary.falsePositiveRate).formatted(.number.precision(.fractionLength(2))))%")
        }
        if let value = report.activeRelativeMouseMAE { components.append("active mouse MAE \(value.formatted(.number.precision(.fractionLength(4))))") }
        if let value = report.activeScrollMAE { components.append("active scroll MAE \(value.formatted(.number.precision(.fractionLength(4))))") }
        return components.joined(separator: " • ")
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionTitle("Training", "Compiled MLX training uses an efficient residual vision encoder, causal temporal memory, autonomous-start history exposure, pixel and trajectory perturbations, class-balanced sparse controls, transition-balanced batches, disjoint visual validation, and leak-resistant best-brain selection. Another trained AI can run simultaneously.")
                HStack {
                    if model.isTraining, let profile = displayedProfile {
                        Text(profile.name).font(.title2.bold())
                        StatusPill(text: model.isAutoTraining ? "Auto training" : "GPU training", color: ATColor.green)
                    } else {
                        Picker("AI to train", selection: $model.selectedProfileID) {
                            Text("Select an AI…").tag(UUID?.none)
                            ForEach(model.profiles) { profile in
                                Text("\(profile.name) — \(profile.trainingProgress?.globalStep ?? 0) steps").tag(Optional(profile.id))
                            }
                        }
                        .frame(width: 360)
                        InfoTip("Choose the AI whose recordings, vision settings, and network configuration should be trained. You can switch here without visiting AI Models.")
                        if let profile = displayedProfile { StatusPill(text: profile.activeVersionID == nil ? "Untrained" : "Runnable brain ready", color: ATColor.violet) }
                    }
                    Spacer()
                    if model.isTraining { Button("Pause → Runnable Brain") { model.pauseTraining() }.primaryButton(color: ATColor.amber); Button("Stop") { model.stopTraining() }.primaryButton(color: ATColor.coral) }
                    else {
                        Button("Start / Exact Resume") { model.startTraining() }.primaryButton(color: ATColor.green)
                        Button("Auto Train") { model.startAutoTraining() }.primaryButton(color: ATColor.cyan).help("Keep starting another configured epoch block whenever training completes, until paused or stopped.")
                    }
                }
                OLEDCard {
                    HStack(alignment: .bottom, spacing: 14) {
                        VStack(alignment: .leading, spacing: 5) { HStack { Text("Universal training controls").font(.headline).foregroundStyle(ATColor.green); InfoTip("These controls apply to the next training session regardless of AI profile. Maximum Steps is added to the AI's current global step, so it limits only this session and never blocks later continuation.") }; Text("Maximum Steps limits this session; 0 runs to the current epoch-block goal.").font(.caption).foregroundStyle(.secondary) }
                        Spacer()
                        IntField("Maximum Steps", value: $model.trainingRunSettings.maximumSteps, help: "Optimizer-update budget for this session. It starts from the AI's restored global step; use 0 for epoch-only control.").frame(width: 165)
                        IntField("Autosave Steps", value: $model.trainingRunSettings.autosaveSteps, help: "Publish a runnable brain and exact resumable state after this many optimizer updates.").frame(width: 155)
                    }.disabled(model.isTraining)
                }
                OLEDCard {
                    VStack(alignment: .leading, spacing: 14) {
                        HStack {
                            VStack(alignment: .leading) { Text("Learning curves").font(.headline); Text("Newest data stays in view; only visible points are rendered.").font(.caption).foregroundStyle(.secondary) }
                            Spacer()
                            Picker("Graph", selection: $graphKind) { ForEach(GraphKind.allCases) { Text($0.rawValue).tag($0) } }.pickerStyle(.segmented).frame(width: 540)
                            Button { zoomFraction = max(0.02, zoomFraction / 1.6) } label: { Image(systemName: "plus.magnifyingglass") }.primaryButton().help("Zoom in")
                            Button { zoomFraction = min(1, zoomFraction * 1.6) } label: { Image(systemName: "minus.magnifyingglass") }.primaryButton().help("Zoom out")
                            Text("\(Int(zoomFraction * 100))%").font(.caption.monospacedDigit()).foregroundStyle(.secondary).frame(width: 38)
                        }
                        LossChart(values: graphValues, zoomFraction: zoomFraction, color: graphAccent, isActive: model.isAppActive)
                            .id("\(graphKind.rawValue)-\(zoomFraction)")
                            .frame(height: 250)
                        HStack { Text(graphKind.rawValue).foregroundStyle(graphAccent).font(.caption.bold()); Spacer(); Text(graphValues.last?.formatted(.number.precision(.fractionLength(8))) ?? "No data yet").font(.caption.monospacedDigit()).foregroundStyle(.secondary) }
                    }
                }
                HStack(spacing: 12) {
                    MetricCard(title: "Batch loss", value: lossText(model.trainingMetrics.lossHistory.last), symbol: "waveform.path.ecg", color: ATColor.cyan)
                    MetricCard(title: "Epoch-average loss", value: lossText(model.trainingMetrics.epochTrainingLoss), symbol: "chart.xyaxis.line", color: ATColor.green)
                    MetricCard(title: "Zero-history loss", value: lossText(model.trainingMetrics.validationLoss), symbol: "checkmark.shield.fill", color: ATColor.violet)
                    MetricCard(title: "Effective LR • envelope", value: model.trainingMetrics.effectiveLearningRate > 0 ? String(format: "%.2e • %.3fx", model.trainingMetrics.effectiveLearningRate, model.trainingMetrics.learningRateScale) : "—", symbol: "speedometer", color: ATColor.amber)
                    MetricCard(title: "Zero-history binary F1", value: model.trainingMetrics.validationReport?.binary.map { "\((100 * $0.f1).formatted(.number.precision(.fractionLength(1))))%" } ?? "—", symbol: "scope", color: ATColor.coral)
                }
                HStack(spacing: 12) {
                    MetricCard(title: "Actual training time", value: TrainingDurationFormatter.string(seconds: displayedTiming.trainingSeconds), symbol: "clock.fill", color: ATColor.cyan)
                    MetricCard(title: "Equivalent experience", value: (displayedTiming.experienceIsEstimated ? "~" : "") + TrainingDurationFormatter.string(seconds: displayedTiming.experienceSeconds), symbol: "brain.head.profile.fill", color: ATColor.violet)
                    MetricCard(title: "Samples / sec", value: model.trainingMetrics.samplesPerSecond.formatted(.number.precision(.fractionLength(0))), symbol: "bolt.fill", color: ATColor.amber)
                    MetricCard(title: "Optimizer steps", value: displayedSteps.formatted(), symbol: "arrow.triangle.2.circlepath", color: ATColor.green)
                }
                HStack(spacing: 12) {
                    MetricCard(title: "Pipelined step", value: "\(model.trainingMetrics.trainingStepMilliseconds.formatted(.number.precision(.fractionLength(1)))) ms", symbol: "timer", color: ATColor.cyan)
                    MetricCard(title: "Mapped input gather", value: "\(model.trainingMetrics.batchPreparationMilliseconds.formatted(.number.precision(.fractionLength(1)))) ms", symbol: "externaldrive.fill", color: ATColor.violet)
                    MetricCard(title: "Throughput retained", value: "\((100 * model.trainingMetrics.throughputRetention).formatted(.number.precision(.fractionLength(0))))%", symbol: "chart.line.uptrend.xyaxis", color: model.trainingMetrics.throughputRetention < 0.8 ? ATColor.coral : ATColor.green)
                    MetricCard(title: "Thermal pressure", value: model.trainingMetrics.thermalState.rawValue, symbol: "thermometer.medium", color: model.trainingMetrics.thermalState == .nominal ? ATColor.green : ATColor.coral)
                }
                HStack(spacing: 12) {
                    MetricCard(title: "MLX active unified", value: ByteCountFormatter.string(fromByteCount: Int64(model.trainingMetrics.mlxActiveMemory), countStyle: .memory), symbol: "memorychip", color: ATColor.green)
                    MetricCard(title: "MLX reusable cache", value: ByteCountFormatter.string(fromByteCount: Int64(model.trainingMetrics.mlxCacheMemory), countStyle: .memory), symbol: "internaldrive", color: ATColor.violet)
                    MetricCard(title: "MLX peak active", value: ByteCountFormatter.string(fromByteCount: Int64(model.trainingMetrics.mlxPeakMemory), countStyle: .memory), symbol: "chart.bar.fill", color: ATColor.amber)
                    MetricCard(title: "Epoch", value: "\(model.trainingMetrics.epoch) / \(model.trainingMetrics.totalEpochs)", symbol: "chart.line.uptrend.xyaxis", color: ATColor.cyan)
                    MetricCard(title: "Batch", value: "\(model.trainingMetrics.batch) / \(model.trainingMetrics.totalBatches)", symbol: "square.stack.3d.up.fill", color: ATColor.violet)
                }
                if let profile = displayedProfile {
                    HStack(spacing: 12) {
                        MetricCard(title: "Parameters", value: ModelSizing.parameterCount(profile).formatted(), symbol: "circle.grid.cross", color: ATColor.cyan)
                        MetricCard(title: "Packed sample", value: ByteCountFormatter.string(fromByteCount: Int64(profile.preprocessing.sampleByteCount), countStyle: .memory), symbol: "shippingbox.fill", color: ATColor.violet)
                        MetricCard(title: "Workspace storage", value: ByteCountFormatter.string(fromByteCount: model.storageBytes, countStyle: .file), symbol: "internaldrive", color: ATColor.green)
                    }
                }
                OLEDCard {
                    VStack(alignment: .leading, spacing: 9) {
                        HStack {
                            Text(model.trainingStatus).foregroundStyle(ATColor.cyan)
                            Spacer()
                            Text("Step \(model.trainingMetrics.globalStep) / \(model.trainingMetrics.totalSteps)").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                        }
                        ProgressView(value: Double(model.trainingMetrics.globalStep), total: Double(max(1, model.trainingMetrics.totalSteps))).tint(ATColor.cyan)
                        if model.isTraining, let next = model.trainingMetrics.nextAutosaveStep {
                            Label("Next periodic autosave at step \(next) • \(model.trainingMetrics.autosavesPublished) published this run", systemImage: "externaldrive.badge.timemachine")
                                .font(.caption).foregroundStyle(ATColor.amber)
                        }
                        if let report = model.trainingMetrics.validationReport {
                            Text(validationSummary(report)).font(.caption).foregroundStyle(ATColor.green)
                            ForEach(report.executableBinaryCollapseWarnings, id: \.self) { warning in
                                Label(warning, systemImage: "exclamationmark.triangle.fill")
                                    .font(.caption).foregroundStyle(ATColor.amber)
                            }
                        }
                        if let coverage = model.trainingMetrics.trainingDataCoverage,
                           let profile = displayedProfile {
                            ForEach(coverage.warnings(for: profile.channels), id: \.self) { warning in
                                Label(warning, systemImage: "exclamationmark.triangle.fill")
                                    .font(.caption).foregroundStyle(ATColor.amber)
                            }
                        }
                        if let profile = displayedProfile {
                            Text("\(profile.preprocessing.width) × \(profile.preprocessing.height) • \(profile.preprocessing.bitDepth)-bit \(profile.preprocessing.chroma.rawValue) • \(profile.training.precision.rawValue) • perception \(profile.training.perceptionFPS.formatted()) FPS • action \(profile.training.actionFPS.formatted()) FPS").font(.caption).foregroundStyle(.secondary)
                            ForEach(profile.preprocessing.trainingQualityWarnings, id: \.self) { warning in
                                Label(warning, systemImage: "exclamationmark.triangle.fill")
                                    .font(.caption).foregroundStyle(ATColor.amber)
                            }
                        }
                        Text("Exact continuation always uses the latest optimizer and adaptive-scheduler checkpoint. With validation data, completed training prefers a brain whose supported discrete heads cross the execution threshold, then the lowest-loss non-regressing checkpoint. If a head remains weak, the best available brain is still saved and the limitation is shown as a quality warning.").font(.caption2).foregroundStyle(.secondary)
                        Text("MLX reports allocator-backed unified memory: active arrays, reusable cache, and process-lifetime peak. It is not separate VRAM on Apple silicon.").font(.caption2).foregroundStyle(.tertiary)
                    }
                }
            }.padding(28)
        }
    }
}

private struct LossChart: View {
    let values: [Double]
    let zoomFraction: Double
    let color: Color
    let isActive: Bool
    @Environment(\.uiMotionEnabled) private var motionEnabled
    @State private var previousValues: [Double] = []
    @State private var renderedValues: [Double] = []
    @State private var transitionProgress = 1.0
    @State private var revealed = true

    var body: some View {
        ZStack {
            Canvas(opaque: false, rendersAsynchronously: true) { context, size in
                let inset: CGFloat = 10
                let chartSize = CGSize(width: max(1, size.width - inset * 2), height: max(1, size.height - inset * 2))
                let chartRect = CGRect(origin: CGPoint(x: inset, y: inset), size: chartSize)

                for row in 0...4 {
                    let y = chartRect.minY + chartRect.height * CGFloat(row) / 4
                    var line = Path(); line.move(to: CGPoint(x: chartRect.minX, y: y)); line.addLine(to: CGPoint(x: chartRect.maxX, y: y))
                    context.stroke(line, with: .color(ATColor.border.opacity(row == 4 ? 0.75 : 0.45)), lineWidth: 0.7)
                }
                for column in 0...5 {
                    let x = chartRect.minX + chartRect.width * CGFloat(column) / 5
                    var line = Path(); line.move(to: CGPoint(x: x, y: chartRect.minY)); line.addLine(to: CGPoint(x: x, y: chartRect.maxY))
                    context.stroke(line, with: .color(ATColor.border.opacity(0.25)), lineWidth: 0.6)
                }

                drawCurve(
                    context: context,
                    chartRect: chartRect,
                    previous: previousValues,
                    current: renderedValues,
                    progress: transitionProgress
                )
            }
            .mask(alignment: .leading) {
                Rectangle().scaleEffect(x: revealed ? 1 : 0, anchor: .leading)
            }

            if renderedValues.isEmpty {
                VStack(spacing: 7) {
                    Image(systemName: "waveform.path.ecg").font(.title2).foregroundStyle(color.opacity(0.7))
                    Text("Curve appears as training publishes metrics").font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .background(RoundedRectangle(cornerRadius: ATCorner.scaled(13), style: .continuous).fill(ATColor.raised.opacity(0.72)))
        .overlay(RoundedRectangle(cornerRadius: ATCorner.scaled(13), style: .continuous).stroke(ATColor.border, lineWidth: 0.8))
        .onAppear { synchronize(values, reveal: true) }
        // The published histories are bounded suffixes. Once a suffix reaches
        // its cap, its count stays constant while every optimizer update shifts
        // in a new value. Observe the snapshot itself so a long run continues
        // to repaint in real time instead of waiting for the view to remount.
        .onChange(of: values) { _, updatedValues in synchronize(updatedValues, reveal: false) }
        .onChange(of: isActive) { _, active in if active { synchronize(values, reveal: true) } }
    }

    private func synchronize(_ updatedValues: [Double], reveal: Bool) {
        guard isActive else { return }
        let morphsLiveUpdate = motionEnabled && !reveal && !renderedValues.isEmpty && !updatedValues.isEmpty && renderedValues != updatedValues
        if morphsLiveUpdate {
            previousValues = renderedValues
            renderedValues = updatedValues
            transitionProgress = 0
            DispatchQueue.main.async {
                withAnimation(UIMotion.chartUpdate) { transitionProgress = 1 }
            }
        } else {
            previousValues = []
            renderedValues = updatedValues
            transitionProgress = 1
        }
        guard reveal, motionEnabled, !updatedValues.isEmpty else { revealed = true; return }
        revealed = false
        DispatchQueue.main.async {
            withAnimation(UIMotion.reveal) { revealed = true }
        }
    }

    /// Morphs only while a new metrics snapshot arrives. There is no continuous
    /// timeline, so an idle, hidden, or inactive chart consumes no animation work.
    private func drawCurve(context sourceContext: GraphicsContext, chartRect: CGRect, previous: [Double], current: [Double], progress: Double) {
        let limit = max(80, Int(chartRect.width * 1.25))
        let currentSeries = sampledVisibleSeries(current, limit: limit)
        guard currentSeries.count > 1 else { return }
        let previousSeries = sampledVisibleSeries(previous, limit: limit)
        let clampedProgress = min(1, max(0, progress))
        let displayed = previousSeries.count > 1 && clampedProgress < 1
            ? interpolatedSeries(from: previousSeries, to: currentSeries, progress: clampedProgress)
            : currentSeries
        let scaleSeries = previousSeries.count > 1 ? previousSeries + currentSeries : currentSeries
        let minimum = scaleSeries.min() ?? 0, maximum = scaleSeries.max() ?? 1
        let padding = max(0.000_001, (maximum - minimum) * 0.14)
        let low = max(0, minimum - padding), range = max(0.000_001, maximum + padding - low)
        let points = displayed.enumerated().map { index, value in
            CGPoint(
                x: chartRect.minX + chartRect.width * CGFloat(index) / CGFloat(max(1, displayed.count - 1)),
                y: chartRect.maxY - chartRect.height * CGFloat((value - low) / range)
            )
        }
        let line = smoothPath(points)
        var area = line
        area.addLine(to: CGPoint(x: points.last?.x ?? chartRect.maxX, y: chartRect.maxY))
        area.addLine(to: CGPoint(x: points.first?.x ?? chartRect.minX, y: chartRect.maxY))
        area.closeSubpath()
        let context = sourceContext
        context.fill(area, with: .linearGradient(
            Gradient(colors: [color.opacity(0.28), color.opacity(0.015)]),
            startPoint: CGPoint(x: chartRect.midX, y: chartRect.minY),
            endPoint: CGPoint(x: chartRect.midX, y: chartRect.maxY)
        ))
        context.stroke(line, with: .color(color.opacity(0.18)), style: StrokeStyle(lineWidth: 6, lineCap: .round, lineJoin: .round))
        context.stroke(line, with: .color(color), style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
        if let last = points.last {
            let dot = Path(ellipseIn: CGRect(x: last.x - 3.5, y: last.y - 3.5, width: 7, height: 7))
            context.fill(dot, with: .color(ATColor.panel))
            context.stroke(dot, with: .color(color), lineWidth: 2)
        }
    }

    private func sampledVisibleSeries(_ source: [Double], limit: Int) -> [Double] {
        guard !source.isEmpty else { return [] }
        let visibleCount = max(20, Int(Double(source.count) * min(1, max(0.02, zoomFraction))))
        let visible = source.suffix(visibleCount).filter(\.isFinite)
        return downsamplePreservingExtrema(Array(visible), limit: limit)
    }

    private func interpolatedSeries(from: [Double], to: [Double], progress: Double) -> [Double] {
        let count = max(from.count, to.count)
        guard count > 1 else { return to }
        return (0..<count).map { index in
            let position = Double(index) / Double(count - 1)
            let oldValue = interpolatedValue(in: from, position: position)
            let newValue = interpolatedValue(in: to, position: position)
            return oldValue + (newValue - oldValue) * progress
        }
    }

    private func interpolatedValue(in values: [Double], position: Double) -> Double {
        guard values.count > 1 else { return values.first ?? 0 }
        let scaled = position * Double(values.count - 1)
        let lower = min(values.count - 1, max(0, Int(scaled.rounded(.down))))
        let upper = min(values.count - 1, lower + 1)
        let fraction = scaled - Double(lower)
        return values[lower] + (values[upper] - values[lower]) * fraction
    }

    /// Min/max bucket sampling retains brief spikes while bounding Canvas work.
    private func downsamplePreservingExtrema(_ input: [Double], limit: Int) -> [Double] {
        guard input.count > limit, limit >= 4 else { return input }
        let bucketCount = max(1, limit / 2)
        let bucketSize = Double(input.count) / Double(bucketCount)
        var result: [(index: Int, value: Double)] = [(0, input[0])]
        result.reserveCapacity(limit + 2)
        for bucket in 0..<bucketCount {
            let start = max(1, Int(Double(bucket) * bucketSize))
            let end = min(input.count - 1, max(start + 1, Int(Double(bucket + 1) * bucketSize)))
            guard start < end else { continue }
            let range = start..<end
            if let minimum = range.min(by: { input[$0] < input[$1] }),
               let maximum = range.max(by: { input[$0] < input[$1] }) {
                for index in [minimum, maximum].sorted() where result.last?.index != index {
                    result.append((index, input[index]))
                }
            }
        }
        if result.last?.index != input.count - 1 { result.append((input.count - 1, input[input.count - 1])) }
        return result.map(\.value)
    }

    private func smoothPath(_ points: [CGPoint]) -> Path {
        var path = Path()
        guard let first = points.first else { return path }
        path.move(to: first)
        guard points.count > 2 else { path.addLine(to: points.last ?? first); return path }
        for index in 1..<points.count {
            let previous = points[index - 1]
            let current = points[index]
            let midpoint = CGPoint(x: (previous.x + current.x) / 2, y: (previous.y + current.y) / 2)
            path.addQuadCurve(to: midpoint, control: previous)
        }
        path.addLine(to: points.last ?? first)
        return path
    }
}

struct RunView: View {
    @ObservedObject var model: AppModel
    private var sources: [CaptureSourceOption] { model.captureSources.filter { source in switch model.captureKind { case .display, .screenRegion: source.kind == .display; case .window, .windowRegion: source.kind == .window } } }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionTitle("Run", "Run an existing brain or start a new neutral one with live Reward/Punish learning. Capture, causal credit, inference, and input safety stay local.")
                OLEDCard {
                    HStack {
                        Picker("AI to run", selection: $model.selectedProfileID) {
                            Text("Select an AI…").tag(UUID?.none)
                            ForEach(model.profiles) { profile in
                                Text("\(profile.name) — \(profile.activeVersionID == nil ? (profile.effectiveReinforcement.enabled ? "new RL brain" : "untrained") : "\(profile.trainingProgress?.globalStep ?? 0) steps")").tag(Optional(profile.id))
                            }
                        }
                        .frame(maxWidth: 430)
                        .disabled(model.agentIsActiveOrStarting)
                        InfoTip("Choose an AI with an active saved brain, or a brand-new AI with RL enabled. Saved weights load directly; a new RL AI starts neutral and publishes its first brain after credited feedback.")
                        Spacer()
                        if let progress = model.selectedProfile?.trainingProgress {
                            Text("\(progress.globalStep) steps • \(progress.epoch) epochs").font(.caption.bold()).foregroundStyle(ATColor.green)
                        }
                    }
                }
                if let profile = model.selectedProfile {
                    OLEDCard {
                        HStack { VStack(alignment: .leading, spacing: 7) { Text(profile.name).font(.title2.bold()); Text("Model vision \(profile.preprocessing.width) × \(profile.preprocessing.height) — live capture will be exactly the same").foregroundStyle(ATColor.cyan); Text("\(profile.preprocessing.colorMode.rawValue), \(profile.preprocessing.bitDepth)-bit, \(profile.preprocessing.chroma.rawValue)").foregroundStyle(.secondary) }; Spacer(); StatusPill(text: runReadiness(profile), color: model.agentIsActiveOrStarting || profile.effectiveReinforcement.enabled ? ATColor.green : ATColor.violet) }
                    }
                    OLEDCard {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("What the AI watches").font(.headline).foregroundStyle(ATColor.cyan)
                            Picker("Type", selection: $model.captureKind) { ForEach(CaptureKind.allCases) { Text($0.rawValue).tag($0) } }.pickerStyle(.segmented).onChange(of: model.captureKind) { _, _ in Task { await model.refreshSources() } }
                            Picker("Source", selection: $model.selectedSourceID) { Text("Select…").tag(UInt32?.none); ForEach(sources) { Text("\($0.name) — \($0.detail)").tag(Optional($0.id)) } }
                            if model.captureKind == .screenRegion || model.captureKind == .windowRegion {
                                HStack { LabeledNumber("X", value: $model.regionX); LabeledNumber("Y", value: $model.regionY); LabeledNumber("Width", value: $model.regionWidth); LabeledNumber("Height", value: $model.regionHeight); if model.captureKind == .screenRegion { Button("Draw Region") { model.selectScreenRegion() }.primaryButton(color: ATColor.cyan) } }
                            }
                                Text("The persistent stream is preprocessed to the model's exact \(profile.preprocessing.width) × \(profile.preprocessing.height) vision contract. Reduced historical frames are encoded once and reused as compact embeddings. AgentTrainer and its floating HUD are excluded.").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    if profile.effectiveReinforcement.enabled {
                        LiveReinforcementCard(profile: profile, model: model)
                    }
                    HStack(alignment: .top, spacing: 14) {
                        OLEDCard {
                            VStack(alignment: .leading, spacing: 14) {
                            Text("Perception and action").font(.headline)
                            HStack { Picker("Frame mode", selection: $model.frameMode) { ForEach(FrameMode.allCases) { Text($0.rawValue).tag($0) } }.pickerStyle(.segmented); InfoTip("Newest Frame stays responsive by skipping old frames when inference is busy. Every Frame preserves order by slowing capture instead of building a large queue.") }
                            HStack { Picker("Mouse execution", selection: $model.runMouseMode) { ForEach(MouseControlMode.allCases) { Text($0.rawValue).tag($0) } }.pickerStyle(.segmented); InfoTip("Auto checks the selected AI's recordings and chooses normal cursor movement or signed Game Camera deltas. A brand-new RL brain starts in neutral Game Camera mode because absolute coordinates have no no-movement value. Choose Absolute Cursor explicitly when the task uses a visible pointer.") }
                                if model.runMouseMode != .absolute {
                                    VStack(alignment: .leading, spacing: 9) {
                                        HStack { Text(model.runMouseMode == .automatic ? "Game Camera settings (when Auto detects it)" : "Game Camera").font(.subheadline.bold()).foregroundStyle(ATColor.violet); Spacer(); Text("\(model.gameCamera.sensitivity.formatted(.number.precision(.fractionLength(2))))× sensitivity").font(.caption.monospacedDigit()).foregroundStyle(.secondary) }
                                        Slider(value: $model.gameCamera.sensitivity, in: 0.1...3, step: 0.05).tint(ATColor.violet)
                                        Toggle("Recenter before and after every raw delta", isOn: $model.gameCamera.recenterCursor).tint(ATColor.cyan)
                                        Text(model.gameCamera.recenterCursor ? "Recommended for Roblox and locked-camera games. A HID-system event is posted at the capture center and the cursor is restored immediately, preventing screen-edge damping." : "Raw deltas are posted without warping. Use this only for software that rejects cursor recentering.")
                                            .font(.caption).foregroundStyle(.secondary)
                                    }
                                    .padding(10)
                                    .raisedGlassSurface(cornerRadius: 10, tint: ATColor.amber)
                                }
                                LabeledContent("Perception", value: "\(profile.training.perceptionFPS.formatted()) FPS")
                                LabeledContent("Action", value: "\(profile.training.actionFPS.formatted()) FPS")
                                Text(profile.training.actionFPS > profile.training.perceptionFPS ? "Held keys/buttons persist between perception updates; camera and scroll deltas run once per new prediction." : "Actions execute independently; additive camera and scroll deltas never replay on a stale prediction.").font(.caption).foregroundStyle(.secondary)
                                Divider(); Toggle("Stop immediately on my input", isOn: $model.safety.stopOnHumanInput); Toggle("Allow full-Mac control", isOn: $model.safety.allowFullMac).tint(ATColor.coral)
                                Divider(); Toggle("Show exact AI-vision PIP", isOn: $model.showVisionPreview).tint(ATColor.green); if model.showVisionPreview { Toggle("Update exactly with AI perception", isOn: $model.visionPreviewMatchesPerception).tint(ATColor.cyan); if !model.visionPreviewMatchesPerception { LabeledNumber("Independent PIP FPS", value: $model.visionPreviewFPS) }; Text(model.visionPreviewMatchesPerception ? "The preview updates on the same processed frames as the AI." : "Preview refresh is independent; AI perception remains \(profile.training.perceptionFPS.formatted()) FPS.").font(.caption).foregroundStyle(.secondary) }
                                Divider()
                                Toggle("Show live CNN internals", isOn: $model.cnnVisualizationSettings.enabled).tint(ATColor.cyan)
                                if model.cnnVisualizationSettings.enabled {
                                    Picker("CNN view", selection: $model.cnnVisualizationSettings.mode) { ForEach(CNNVisualizationMode.allCases) { Text($0.rawValue).tag($0) } }.pickerStyle(.segmented)
                                    HStack {
                                        Text("Diagnostic rate").font(.caption).foregroundStyle(.secondary)
                                        Slider(value: $model.cnnVisualizationSettings.framesPerSecond, in: 0.5...15, step: 0.5).tint(ATColor.cyan)
                                        Text("\(model.cnnVisualizationSettings.framesPerSecond.formatted(.number.precision(.fractionLength(1)))) FPS").font(.caption.monospacedDigit()).foregroundStyle(.secondary).frame(width: 58, alignment: .trailing)
                                    }
                                    switch model.cnnVisualizationSettings.mode {
                                    case .activationOverlay:
                                        HStack {
                                            Picker("Stage", selection: $model.cnnVisualizationSettings.convolutionLayer) {
                                                Text(profile.training.architecture.convolutionChannels.isEmpty ? "Vision input" : "Final visual stage").tag(-1)
                                                ForEach(profile.training.architecture.convolutionChannels.indices, id: \.self) { index in Text("Stage \(index + 1)").tag(index) }
                                            }
                                            CNNOverlayOpacityControl(value: $model.cnnVisualizationSettings.overlayOpacity)
                                        }
                                    case .featureChannels:
                                        Picker("Channels shown", selection: $model.cnnVisualizationSettings.featureChannelCount) {
                                            ForEach([4, 6, 9, 12, 16], id: \.self) { count in Text("\(count)").tag(count) }
                                        }
                                    case .actionSaliency:
                                        HStack {
                                            Picker("Action head", selection: $model.cnnVisualizationSettings.actionFocus) {
                                                ForEach(CNNActionFocus.allCases) { focus in
                                                    Text(focus.displayName + (actionFocusIsTrained(focus, by: profile) ? "" : " (not trained)")).tag(focus)
                                                }
                                            }
                                            CNNOverlayOpacityControl(value: $model.cnnVisualizationSettings.overlayOpacity)
                                        }
                                    }
                                    Text(model.cnnVisualizationSettings.mode == .actionSaliency ? "Grad-CAM traces positive influence on the selected action head. It is the most expensive view and runs only at the diagnostic rate." : model.cnnVisualizationSettings.mode == .featureChannels ? "Shows the strongest final-layer filters as individually normalized maps." : "Combines the selected layer's activations and overlays them on the exact processed input.").font(.caption).foregroundStyle(.secondary)
                                    if model.cnnVisualizationSettings.mode == .actionSaliency, !actionFocusIsTrained(model.cnnVisualizationSettings.actionFocus, by: profile) {
                                        Text("This action head was not trained for the selected AI, so its saliency is not meaningful.").font(.caption).foregroundStyle(ATColor.amber)
                                    }
                                    Text("These controls remain live while running. Turning the view off immediately restores the standard inference-only path.").font(.caption).foregroundStyle(ATColor.violet)
                                }
                                Text("When full-Mac control is off, pointer actions are clamped to the selected capture region.").font(.caption).foregroundStyle(.secondary)
                            }
                        }.frame(maxWidth: .infinity)
                        OLEDCard {
                            VStack(alignment: .leading, spacing: 12) {
                                Text("Live runtime").font(.headline)
                                LabeledContent("Perception", value: model.runtimeMetrics.perceptionFPS.formatted(.number.precision(.fractionLength(1))) + " FPS")
                                LabeledContent("Actions", value: model.runtimeMetrics.actionFPS.formatted(.number.precision(.fractionLength(1))) + " FPS")
                                LabeledContent("Inference latency", value: model.runtimeMetrics.latencyMilliseconds.formatted(.number.precision(.fractionLength(1))) + " ms")
                                LabeledContent("Dropped frames", value: "\(model.runtimeMetrics.droppedFrames)")
                                if let progress = profile.trainingProgress { LabeledContent("Training", value: "\(progress.globalStep) steps • epoch \(progress.epoch)") }
                                Text("The bottom-right HUD displays AI-generated inputs only and is excluded from the AI's capture filter.").font(.caption).foregroundStyle(ATColor.violet)
                            }
                        }.frame(width: 330)
                    }
                    OLEDCard {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("Runtime output firewall").font(.headline).foregroundStyle(ATColor.coral)
                            HStack(spacing: 24) {
                                Toggle("Allow cursor movement", isOn: $model.runtimeOutputPermissions.cursorMovement)
                                    .tint(ATColor.cyan)
                                Toggle("Allow keyboard", isOn: $model.runtimeOutputPermissions.keyboard)
                                    .tint(ATColor.violet)
                            }
                            Text("These run-only permissions take effect immediately, including during an active run. Turning off Keyboard releases every AI-held key and modifier. Cursor movement does not disable mouse buttons or scrolling.")
                                .font(.caption).foregroundStyle(.secondary)
                            Divider()
                            Text("Per-AI key and mouse-button restrictions").font(.subheadline.bold())
                            KeyRestrictionGrid(restrictions: Binding(get: { profile.effectiveRestrictions }, set: { value in var changed = profile; changed.restrictions = value; model.saveProfile(changed) }), model: model).disabled(model.agentIsActiveOrStarting)
                        }
                    }
                    HStack { StatusPill(text: model.isTraining ? "Background training remains active" : "Custom panic shortcut armed", color: model.isTraining ? ATColor.cyan : ATColor.coral); Spacer(); if model.agentIsActiveOrStarting { Button(model.isStartingAgent ? "Cancel AI Start & Release Inputs" : "Stop AI & Save Learning") { Task { await model.stopAgent() } }.primaryButton(color: ATColor.coral) } else { Button(profile.effectiveReinforcement.enabled ? "Run & Learn" : "Run AI") { Task { await model.startAgent() } }.primaryButton(color: ATColor.green).disabled(!profile.canRunOrLearn || model.trainingProfileID == profile.id || model.recordingIsActiveOrStarting || model.isReplaying) } }
                } else { ContentUnavailableView("No AI profile selected", systemImage: "cpu") }
            }.padding(28)
        }
    }

    private func actionFocusIsTrained(_ focus: CNNActionFocus, by profile: AIProfile) -> Bool {
        switch focus {
        case .movement: profile.channels.mouseMovement
        case .mouseButtons: profile.channels.buttons
        case .scroll: profile.channels.scroll
        case .keyboard: profile.channels.keyboard
        case .modifiers: profile.channels.modifiers
        }
    }

    private func runReadiness(_ profile: AIProfile) -> String {
        if model.isStartingAgent { return "Starting / stopping" }
        if model.isRunning { return profile.effectiveReinforcement.enabled ? "AI running & learning" : "AI running" }
        if profile.activeVersionID != nil { return profile.effectiveReinforcement.enabled ? "Ready to run & learn" : "Ready" }
        return profile.effectiveReinforcement.enabled ? "New neutral RL brain" : "Training required"
    }
}

private struct LiveReinforcementCard: View {
    let profile: AIProfile
    @ObservedObject var model: AppModel

    private var configuration: ReinforcementConfiguration { profile.effectiveReinforcement }
    private var controlsAreLive: Bool {
        model.isRunning && model.runningProfileID == profile.id && model.reinforcementMetrics.isActive
    }

    var body: some View {
        OLEDCard {
            VStack(alignment: .leading, spacing: 13) {
                HStack {
                    Label("Live reinforcement", systemImage: "bolt.heart.fill")
                        .font(.headline)
                        .foregroundStyle(ATColor.green)
                    StatusPill(text: controlsAreLive ? "Feedback armed" : "Arms when running", color: controlsAreLive ? ATColor.green : ATColor.amber)
                    Spacer()
                    Text("Session net \(signed(model.reinforcementMetrics.netReward))")
                        .font(.callout.bold().monospacedDigit())
                        .foregroundStyle(model.reinforcementMetrics.netReward >= 0 ? ATColor.green : ATColor.coral)
                }

                HStack(spacing: 12) {
                    feedbackButton(
                        title: "REWARD",
                        amount: configuration.rewardAmount,
                        shortcut: configuration.rewardHotkey.shortcutDisplayName,
                        color: ATColor.green,
                        symbol: "hand.thumbsup.fill",
                        action: model.rewardRunningAgent
                    )
                    feedbackButton(
                        title: "PUNISH",
                        amount: -configuration.punishmentAmount,
                        shortcut: configuration.punishmentHotkey.shortcutDisplayName,
                        color: ATColor.coral,
                        symbol: "hand.thumbsdown.fill",
                        action: model.punishRunningAgent
                    )
                }
                .disabled(!controlsAreLive)

                HStack(spacing: 10) {
                    ReinforcementMetricChip(title: "Feedback", value: "\(model.reinforcementMetrics.feedbackCount)", color: ATColor.violet)
                    ReinforcementMetricChip(title: "Updates", value: "\(model.reinforcementMetrics.updateCount)", color: ATColor.green)
                    ReinforcementMetricChip(title: "Last credit", value: "\(model.reinforcementMetrics.creditedFrames) frames", color: ATColor.cyan)
                    ReinforcementMetricChip(title: "Update time", value: model.reinforcementMetrics.updateCount == 0 ? "—" : "\(model.reinforcementMetrics.lastUpdateMilliseconds.formatted(.number.precision(.fractionLength(1)))) ms", color: ATColor.amber)
                    if let loss = model.reinforcementMetrics.lastPolicyLoss {
                        ReinforcementMetricChip(title: "Policy loss", value: loss.formatted(.number.precision(.fractionLength(4))), color: ATColor.violet)
                    }
                }

                HStack(alignment: .top, spacing: 12) {
                    Label("Global shortcuts: \(configuration.rewardHotkey.shortcutDisplayName) reward • \(configuration.punishmentHotkey.shortcutDisplayName) punish", systemImage: "keyboard")
                    if configuration.scrollFeedbackEnabled {
                        Label("\(scrollChord) + wheel: \(configuration.scrollStep.formatted(.number.precision(.fractionLength(0...2)))) per detent", systemImage: "computermouse")
                    }
                    Spacer()
                }
                .font(.caption.bold())
                .foregroundStyle(.secondary)

                Text(controlsAreLive
                     ? "Feedback is acknowledged immediately in the capture-excluded HUD. It credits up to \(configuration.maximumCreditFrames) decisions from the previous \(configuration.creditWindowSeconds.formatted(.number.precision(.fractionLength(2)))) seconds; learning snapshots are saved automatically and once more when you stop."
                     : "Start this AI to arm global feedback. You can switch scenes immediately after pressing a shortcut—the signal keeps its input-time timestamp.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func feedbackButton(
        title: String,
        amount: Double,
        shortcut: String,
        color: Color,
        symbol: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: symbol).font(.title2)
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(title) \(signed(amount))").font(.title3.bold())
                    Text(shortcut).font(.caption.bold().monospaced())
                }
                Spacer()
            }
            .foregroundStyle(color)
            .padding(.horizontal, 16)
            .frame(maxWidth: .infinity, minHeight: 62)
            .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(color.opacity(0.12)))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(color.opacity(0.65), lineWidth: 1))
        }
        .buttonStyle(.plain)
    }

    private var scrollChord: String {
        HotkeyBinding(keyCode: 0, carbonModifiers: configuration.scrollCarbonModifiers).shortcutDisplayName
            .replacingOccurrences(of: KeyNames.name(for: 0), with: "")
    }

    private func signed(_ value: Double) -> String {
        value.formatted(.number.sign(strategy: .always()).precision(.fractionLength(0...2)))
    }
}

private struct ReinforcementMetricChip: View {
    let title: String
    let value: String
    let color: Color
    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title.uppercased()).font(.system(size: 9, weight: .bold)).foregroundStyle(color)
            Text(value).font(.caption.bold().monospacedDigit())
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(RoundedRectangle(cornerRadius: 9).fill(color.opacity(0.08)))
    }
}

private struct CNNOverlayOpacityControl: View {
    @Binding var value: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Overlay").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text(value.formatted(.percent.precision(.fractionLength(0)))).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Slider(value: $value, in: 0.2...0.9, step: 0.05).tint(ATColor.cyan)
        }
        .frame(minWidth: 180)
    }
}

struct DiagnosticsView: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var logs = AppLogStore.shared
    @State private var search = ""
    @State private var selectedLevel: AppLogLevel?

    private var filteredEntries: [AppLogEntry] {
        logs.entries.reversed().filter { entry in
            (selectedLevel == nil || entry.level == selectedLevel) &&
            (search.isEmpty || entry.message.localizedCaseInsensitiveContains(search) || entry.category.localizedCaseInsensitiveContains(search) || (entry.details?.localizedCaseInsensitiveContains(search) ?? false))
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    SectionTitle("Diagnostics", "Persistent app events, errors, crash reports, permissions, MLX memory, and storage health.")
                    Button("Copy Report") { logs.copyReport(appState: appState) }.primaryButton()
                    Button("Reveal Logs") { logs.revealLogs() }.primaryButton(color: ATColor.violet)
                }
                PermissionStrip(model: model)
                HStack(spacing: 12) { MetricCard(title: "MLX active", value: ByteCountFormatter.string(fromByteCount: Int64(Memory.activeMemory), countStyle: .memory), symbol: "memorychip", color: ATColor.green); MetricCard(title: "MLX peak", value: ByteCountFormatter.string(fromByteCount: Int64(Memory.peakMemory), countStyle: .memory), symbol: "chart.bar.fill", color: ATColor.amber); MetricCard(title: "MLX cache", value: ByteCountFormatter.string(fromByteCount: Int64(Memory.cacheMemory), countStyle: .memory), symbol: "shippingbox", color: ATColor.violet) }
                OLEDCard { VStack(alignment: .leading, spacing: 10) { LabeledContent("Chip", value: hardwareName()); LabeledContent("MLX device", value: Device.defaultDevice().deviceType == .gpu ? "Apple GPU" : "CPU"); LabeledContent("Physical unified memory", value: ByteCountFormatter.string(fromByteCount: Int64(ProcessInfo.processInfo.physicalMemory), countStyle: .memory)); LabeledContent("Local workspace", value: ByteCountFormatter.string(fromByteCount: model.storageBytes, countStyle: .file)); LabeledContent("Bundle identifier", value: Bundle.main.bundleIdentifier ?? "local.agenttrainer.mac"); LabeledContent("Networking", value: "GitHub Releases update check only") } }
                OLEDCard { HStack { VStack(alignment: .leading) { HStack { Text("Packed dataset caches").font(.headline); InfoTip("Caches store each perception once at current and configured past-frame resolution, map action ticks to exact causal frame sequences, and retain complete controls paired with every perception interval. Clearing them never deletes recordings, profiles, or checkpoints.") }; Text("Delete reusable decoded observations without deleting recordings or models.").foregroundStyle(.secondary) }; Spacer(); Button("Clear Caches") { Task { await model.clearCaches() } }.primaryButton(color: ATColor.amber) } }
                let crashReports = AppLogStore.crashReports()
                OLEDCard {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack { Text("macOS crash reports").font(.headline).foregroundStyle(crashReports.isEmpty ? ATColor.green : ATColor.coral); InfoTip("macOS writes a local crash report when the app exits unexpectedly. AgentTrainer only lists its own reports and never sends them anywhere."); Spacer(); StatusPill(text: crashReports.isEmpty ? "None found" : "\(crashReports.count) found", color: crashReports.isEmpty ? ATColor.green : ATColor.coral) }
                        if crashReports.isEmpty {
                            Text("No AgentTrainer .ips or .crash reports were found in your user DiagnosticReports folder.").font(.caption).foregroundStyle(.secondary)
                        } else {
                            ForEach(crashReports.prefix(8), id: \.path) { report in
                                Button { NSWorkspace.shared.activateFileViewerSelecting([report]) } label: {
                                    HStack { Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(ATColor.coral); Text(report.lastPathComponent).lineLimit(1); Spacer(); Image(systemName: "arrow.forward.circle") }
                                        .padding(.vertical, 6).contentShape(Rectangle())
                                }.buttonStyle(.plain)
                            }
                        }
                    }
                }
                OLEDCard {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            VStack(alignment: .leading, spacing: 3) {
                                Text("Application log").font(.headline).foregroundStyle(ATColor.cyan)
                                Text("Errors and important lifecycle events persist across launches. Copy Report includes the latest 500 entries and system details.").font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            TextField("Search logs", text: $search).textFieldStyle(.roundedBorder).frame(width: 220)
                            Picker("Level", selection: $selectedLevel) {
                                Text("All levels").tag(Optional<AppLogLevel>.none)
                                ForEach(AppLogLevel.allCases, id: \.self) { level in Text(level.rawValue).tag(Optional(level)) }
                            }.frame(width: 135)
                            Button("Clear") { logs.clear() }.primaryButton(color: ATColor.coral)
                        }
                        Divider()
                        if filteredEntries.isEmpty {
                            ContentUnavailableView("No matching log entries", systemImage: "text.magnifyingglass")
                                .frame(maxWidth: .infinity).frame(height: 120)
                        } else {
                            LazyVStack(alignment: .leading, spacing: 0) {
                                ForEach(Array(filteredEntries.prefix(400))) { entry in
                                    DiagnosticLogRow(entry: entry)
                                    Divider().opacity(0.45)
                                }
                            }
                        }
                    }
                }
            }.padding(28)
        }
    }
    private var appState: String { "recording=\(model.isRecording), recordingStarting=\(model.isStartingRecording), recordingStopping=\(model.isStoppingRecording), training=\(model.isTraining), running=\(model.isRunning), agentStarting=\(model.isStartingAgent), replaying=\(model.isReplaying), activity=\(model.activityStatus)" }
    private func hardwareName() -> String { var size = 0; sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0); var value = [CChar](repeating: 0, count: max(1, size)); sysctlbyname("machdep.cpu.brand_string", &value, &size, nil, 0); return String(decoding: value.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self) }
}

private struct DiagnosticLogRow: View {
    let entry: AppLogEntry
    private var color: Color {
        switch entry.level {
        case .debug: .secondary
        case .info: ATColor.cyan
        case .warning: ATColor.amber
        case .error: ATColor.coral
        }
    }
    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Text(entry.timestamp.formatted(date: .omitted, time: .standard)).font(.caption2.monospacedDigit()).foregroundStyle(.secondary).frame(width: 82, alignment: .leading)
            Text(entry.level.rawValue.uppercased()).font(.caption2.bold()).foregroundStyle(color).frame(width: 58, alignment: .leading)
            Text(entry.category).font(.caption2.bold()).foregroundStyle(ATColor.violet).frame(width: 90, alignment: .leading)
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.message).font(.caption).textSelection(.enabled)
                if let details = entry.details { Text(details).font(.caption2.monospaced()).foregroundStyle(.secondary).textSelection(.enabled) }
            }
            Spacer(minLength: 0)
        }.padding(.vertical, 7)
    }
}

struct SettingsView: View {
    @ObservedObject var model: AppModel
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionTitle("Settings", "Permissions, shortcuts, safety, diagnostics, storage, and local runtime behavior.")
                OLEDCard { VStack(alignment: .leading, spacing: 14) { Text("macOS permissions").font(.headline); PermissionSetting("Screen Recording", granted: model.screenPermission) { model.openPrivacyPane("Privacy_ScreenCapture") }; PermissionSetting("Input Monitoring", granted: model.inputPermission) { model.openPrivacyPane("Privacy_ListenEvent") }; PermissionSetting("Accessibility", granted: model.accessibilityPermission) { model.openPrivacyPane("Privacy_Accessibility") }; Button("Refresh Permission Status") { model.refreshPermissions() }.primaryButton() } }
                OLEDCard {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack { Text("Global keybinds").font(.headline).foregroundStyle(ATColor.cyan); InfoTip("Click a shortcut, then press a keyboard key or any mouse button, with optional modifiers. Middle and additional side buttons are supported. Shortcuts work globally and are removed from recordings and human-interruption safety.") }
                        HotkeySettingsEditor(model: model).disabled(model.agentIsActiveOrStarting || model.recordingIsActiveOrStarting)
                        if model.agentIsActiveOrStarting || model.recordingIsActiveOrStarting { Text("Shortcuts are locked until the active recording or agent session stops.").font(.caption).foregroundStyle(ATColor.amber) }
                    }
                }
                OLEDCard { VStack(alignment: .leading, spacing: 12) { HStack { Text("Global safety").font(.headline).foregroundStyle(ATColor.coral); InfoTip("The panic shortcut disables capture/action hooks, drains background work, releases every held control, and posts a neutral mouse event.") }; Toggle("Stop AI on any physical human input", isOn: $model.safety.stopOnHumanInput); Toggle("Allow full-Mac control by default", isOn: $model.safety.allowFullMac).tint(ATColor.coral) } }
                ThemeSettingsView()
                OLEDCard { HStack { VStack(alignment: .leading, spacing: 4) { Text("Diagnostics and app logs").font(.headline).foregroundStyle(ATColor.cyan); Text("Open the dedicated tab for persistent errors, prints, crash reports, MLX memory, and a copyable support report.").foregroundStyle(.secondary) }; Spacer(); Button("Open Diagnostics") { model.selection = .diagnostics }.primaryButton() } }
                StorageSettingsView(model: model)
                HStack { Spacer(); Text("AgentTrainer v\(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "development") (\(Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unversioned"))").font(.caption2.monospacedDigit()).foregroundStyle(.tertiary) }
            }.padding(28)
        }
    }
}

private struct StorageSettingsView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        OLEDCard {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Storage locations").font(.headline).foregroundStyle(ATColor.amber)
                        Text("Keep large recordings and model brains on separate internal or external disks.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if model.isChangingStorageLocation {
                        ProgressView().controlSize(.small)
                        Text("Copying and verifying…").font(.caption.bold()).foregroundStyle(ATColor.amber)
                    } else {
                        StatusPill(text: ByteCountFormatter.string(fromByteCount: model.storageUsage.totalBytes, countStyle: .file), color: ATColor.amber)
                    }
                }

                StorageLocationRow(
                    title: "Training data",
                    detail: "Recordings, input events, videos, thumbnails, and rebuildable packed caches",
                    symbol: "externaldrive.badge.timemachine",
                    color: ATColor.cyan,
                    location: model.storageLocations.trainingDataRoot,
                    bytes: model.storageUsage.trainingDataBytes,
                    isDefault: model.storageLocations.trainingDataIsDefault,
                    canChange: model.canChangeStorageLocations,
                    reveal: { model.revealStorageLocation(.trainingData) },
                    change: { Task { await model.chooseStorageLocation(.trainingData) } },
                    restore: { Task { await model.restoreDefaultStorageLocation(.trainingData) } }
                )
                StorageLocationRow(
                    title: "AI models",
                    detail: "Profiles, runnable brains, exact checkpoints, optimizer state, and saved versions",
                    symbol: "brain.head.profile",
                    color: ATColor.violet,
                    location: model.storageLocations.modelsRoot,
                    bytes: model.storageUsage.modelBytes,
                    isDefault: model.storageLocations.modelsAreDefault,
                    canChange: model.canChangeStorageLocations,
                    reveal: { model.revealStorageLocation(.models) },
                    change: { Task { await model.chooseStorageLocation(.models) } },
                    restore: { Task { await model.restoreDefaultStorageLocation(.models) } }
                )

                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "checkmark.shield.fill").foregroundStyle(ATColor.green)
                    Text("Moving to an empty location copies and verifies every managed file before switching, then removes the old copy. Selecting a populated AgentTrainer library switches without merging. Logs remain in Application Support so diagnostics are always available.")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                }
                if !model.canChangeStorageLocations && !model.isChangingStorageLocation {
                    Label("Storage locations unlock after recording, training, running, and replay have stopped.", systemImage: "lock.fill")
                        .font(.caption.bold()).foregroundStyle(ATColor.amber)
                }
            }
        }
    }
}

private struct StorageLocationRow: View {
    let title: String
    let detail: String
    let symbol: String
    let color: Color
    let location: URL
    let bytes: Int64
    let isDefault: Bool
    let canChange: Bool
    let reveal: () -> Void
    let change: () -> Void
    let restore: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(spacing: 10) {
                Image(systemName: symbol).font(.title3).foregroundStyle(color).frame(width: 24)
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 7) {
                        Text(title).font(.subheadline.bold())
                        Text(isDefault ? "DEFAULT" : "CUSTOM")
                            .font(.system(size: 9, weight: .bold, design: .rounded))
                            .foregroundStyle(isDefault ? ATColor.green : color)
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(Capsule().fill((isDefault ? ATColor.green : color).opacity(0.12)))
                    }
                    Text(detail).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Text(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file))
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            HStack(spacing: 8) {
                Text(location.path)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                    .help(location.path)
                    .padding(.horizontal, 9).padding(.vertical, 7)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .raisedGlassSurface(cornerRadius: 8)
                Button("Reveal", action: reveal).primaryButton(color: color)
                Button("Change…", action: change).primaryButton(color: color).disabled(!canChange)
                if !isDefault {
                    Button("Use Default", action: restore).primaryButton(color: ATColor.amber).disabled(!canChange)
                }
            }
        }
        .padding(12)
        .raisedGlassSurface(cornerRadius: 12, tint: color)
    }
}

private struct ThemeSettingsView: View {
    @ObservedObject private var appearance = UIAppearanceStore.shared

    var body: some View {
        OLEDCard {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Appearance").font(.headline).foregroundStyle(ATColor.violet)
                        Text("Balanced themes plus global shape, depth, accent, width, and motion controls.").font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    StatusPill(text: appearance.selectedTheme.name, color: ATColor.cyan)
                    Button("Reset UI") { appearance.resetTuning() }.primaryButton(color: ATColor.amber)
                }

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 10)], spacing: 10) {
                    ForEach(AppTheme.allCases) { theme in
                        Button { appearance.select(theme) } label: {
                            ThemePreview(theme: theme, selected: appearance.selectedTheme == theme)
                        }
                        .buttonStyle(.plain)
                        .uiHoverResponse(scale: 1.01)
                    }
                }

                Divider()
                LazyVGrid(columns: [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)], spacing: 10) {
                    AppearanceTuningControl(
                        title: "Corner radius",
                        detail: "Applies proportionally to cards, controls, rows, previews, and the live HUD.",
                        symbol: "square.on.square",
                        color: ATColor.violet,
                        value: $appearance.tuning.cornerRadius,
                        range: 4...28,
                        step: 1,
                        renderedValue: "\(Int(appearance.tuning.cornerRadius.rounded())) pt"
                    )
                    AppearanceTuningControl(
                        title: "Surface separation",
                        detail: "Controls the visual depth between the canvas, sidebar, cards, and raised controls.",
                        symbol: "square.3.layers.3d",
                        color: ATColor.cyan,
                        value: $appearance.tuning.surfaceContrast,
                        range: 0.7...1.45,
                        step: 0.05,
                        renderedValue: appearance.tuning.surfaceContrast.formatted(.percent.precision(.fractionLength(0)))
                    )
                    AppearanceTuningControl(
                        title: "Accent fill",
                        detail: "Adjusts tinted button and status fills without reducing text or focus contrast.",
                        symbol: "paintbrush.pointed.fill",
                        color: ATColor.green,
                        value: $appearance.tuning.accentIntensity,
                        range: 0.65...1.5,
                        step: 0.05,
                        renderedValue: appearance.tuning.accentIntensity.formatted(.percent.precision(.fractionLength(0)))
                    )
                    AppearanceTuningControl(
                        title: "Sidebar width",
                        detail: "Keeps long section names comfortable while preserving more workspace when compact.",
                        symbol: "sidebar.left",
                        color: ATColor.amber,
                        value: $appearance.tuning.sidebarWidth,
                        range: 205...300,
                        step: 5,
                        renderedValue: "\(Int(appearance.tuning.sidebarWidth.rounded())) pt"
                    )
                }

                HStack(spacing: 14) {
                    Label("Interface animations", systemImage: appearance.motionEnabled ? "sparkles" : "pause.circle.fill")
                        .font(.subheadline.bold()).foregroundStyle(ATColor.cyan)
                    Toggle("", isOn: $appearance.motionEnabled).labelsHidden().tint(ATColor.cyan)
                    Text("Subtle hover, press, page, and chart-reveal transitions use opacity and transforms only. They stop when AgentTrainer is inactive and honor Reduce Motion.")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                }
                .padding(11)
                .raisedGlassSurface(cornerRadius: 11, tint: appearance.motionEnabled ? ATColor.cyan : nil)

                Label("The top bar and sidebar use solid, theme-matched surfaces on macOS 15 and later, avoiding version-specific glass and inactive-window gray states.", systemImage: "macwindow")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

private struct AppearanceTuningControl: View {
    let title: String
    let detail: String
    let symbol: String
    let color: Color
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    let renderedValue: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(title, systemImage: symbol).font(.subheadline.bold()).foregroundStyle(color)
                Spacer()
                Text(renderedValue).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Slider(value: $value, in: range, step: step).tint(color)
            Text(detail).font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
        .padding(11)
        .raisedGlassSurface(cornerRadius: 11, tint: color)
    }
}

private struct ThemePreview: View {
    let theme: AppTheme
    let selected: Bool
    @ObservedObject private var appearance = UIAppearanceStore.shared

    var body: some View {
        let palette = appearance.tuning.applying(to: theme.configuration)
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: theme.symbol).foregroundStyle(palette.cyan.color)
                Text(theme.name).font(.subheadline.bold()).foregroundStyle(palette.text.color)
                Spacer()
                Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(selected ? palette.green.color : palette.border.color)
            }
            HStack(spacing: 5) {
                ForEach([palette.cyan, palette.violet, palette.green, palette.amber, palette.coral], id: \.self) { swatch in
                    Capsule().fill(swatch.color).frame(height: 7)
                }
            }
            Text(theme.detail).font(.caption2).foregroundStyle(palette.text.color.opacity(0.7)).lineLimit(2)
        }
        .padding(12)
        .frame(maxWidth: .infinity, minHeight: 92, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: CGFloat(palette.cornerRadius), style: .continuous).fill(palette.panel.color))
        .overlay(RoundedRectangle(cornerRadius: CGFloat(palette.cornerRadius), style: .continuous).stroke(selected ? palette.cyan.color : palette.border.color, lineWidth: selected ? 1.5 : 0.8))
    }
}

private struct HotkeySettingsEditor: View {
    enum Field { case panic, record, run }
    @ObservedObject var model: AppModel
    @State private var listening: Field?
    @State private var keyMonitor: Any?
    @State private var mouseMonitor: Any?
    var body: some View {
        VStack(spacing: 8) {
            row("Panic and release everything", field: .panic, binding: model.hotkeys.panic)
            row("Start / stop recording", field: .record, binding: model.hotkeys.record)
            row("Start / stop agent", field: .run, binding: model.hotkeys.run)
        }
            .onDisappear { stop() }
    }

    private func row(_ title: String, field: Field, binding: HotkeyBinding) -> some View {
        HStack { Text(title); Spacer(); Button(listening == field ? "Press shortcut…" : display(binding)) { listening == field ? stop() : begin(field) }.primaryButton(color: listening == field ? ATColor.amber : ATColor.violet) }
    }

    private func begin(_ field: Field) {
        stop(resume: false); model.suspendGlobalHotkeys(); listening = field
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            guard !event.isARepeat else { return nil }
            commit(HotkeyBinding(keyCode: UInt32(event.keyCode), carbonModifiers: carbonModifiers(event.modifierFlags)), to: field)
            return nil
        }
        mouseMonitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown, .otherMouseDown]) { event in
            commit(.mouse(UInt8(clamping: event.buttonNumber), carbonModifiers: carbonModifiers(event.modifierFlags)), to: field)
            return nil
        }
    }
    private func commit(_ captured: HotkeyBinding, to field: Field) {
        var settings = model.hotkeys
        switch field { case .panic: settings.panic = captured; case .record: settings.record = captured; case .run: settings.run = captured }
        stop(resume: false)
        model.saveHotkeys(settings)
    }
    private func stop(resume: Bool = true) {
        if let keyMonitor { NSEvent.removeMonitor(keyMonitor); self.keyMonitor = nil }
        if let mouseMonitor { NSEvent.removeMonitor(mouseMonitor); self.mouseMonitor = nil }
        listening = nil
        if resume { model.resumeGlobalHotkeys() }
    }
    private func carbonModifiers(_ flags: NSEvent.ModifierFlags) -> UInt32 {
        var modifiers: UInt32 = 0
        if flags.contains(.command) { modifiers |= UInt32(1 << 8) }
        if flags.contains(.shift) { modifiers |= UInt32(1 << 9) }
        if flags.contains(.option) { modifiers |= UInt32(1 << 11) }
        if flags.contains(.control) { modifiers |= UInt32(1 << 12) }
        return modifiers
    }
    private func display(_ value: HotkeyBinding) -> String {
        var result = ""
        if value.carbonModifiers & UInt32(1 << 12) != 0 { result += "⌃" }
        if value.carbonModifiers & UInt32(1 << 11) != 0 { result += "⌥" }
        if value.carbonModifiers & UInt32(1 << 9) != 0 { result += "⇧" }
        if value.carbonModifiers & UInt32(1 << 8) != 0 { result += "⌘" }
        return result + value.displayName
    }
}

private struct PermissionSetting: View { let name: String; let granted: Bool; let action: () -> Void; init(_ name: String, granted: Bool, action: @escaping () -> Void) { self.name = name; self.granted = granted; self.action = action }; var body: some View { HStack { Label(name, systemImage: granted ? "checkmark.circle.fill" : "exclamationmark.circle.fill").foregroundStyle(granted ? ATColor.green : ATColor.amber); Spacer(); Button(granted ? "Open Settings" : "Grant") { action() }.primaryButton(color: granted ? .secondary : ATColor.amber) } } }
