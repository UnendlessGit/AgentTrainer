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
    private var sources: [CaptureSourceOption] {
        model.captureSources.filter { source in
            switch model.captureKind { case .display, .screenRegion: source.kind == .display; case .window, .windowRegion: source.kind == .window }
        }
    }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                SectionTitle("Record", "Hardware HEVC video with frame-accurate input synchronization and configurable cleanup at both ends.")
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
    @State var draft: AIProfile
    @State private var acceptedDraft: AIProfile
    @State private var pendingBrainReset: AIProfile?
    @State private var suppressDraftObserver = false
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
                        HStack { Text("Training configuration").font(.headline).foregroundStyle(ATColor.green); InfoTip("Maximum Steps and Autosave Steps are universal run controls in the Training tab, so they do not change a model's learning identity. Architecture and exact vision changes require a fresh brain, and the app asks before clearing training.") }
                        HStack { IntField("Epochs per block", value: $draft.training.epochs, help: "How many complete dataset passes to add. If a block is paused, Train finishes it first; after a completed block, Train adds another block of this size."); IntField("Batch", value: $draft.training.batchSize, help: "Samples evaluated together. Larger batches use more unified memory."); IntField("History", value: $draft.training.historyLength, help: "Earlier actions supplied to the recurrent encoder. Keep this short: motion comes directly from consecutive screen frames, and training masks history on half the samples so the policy cannot succeed by copying it.") }
                        HStack { DoubleField("Learning rate", value: $draft.training.learningRate, help: "Peak AdamW update size. Training warms up for one epoch (bounded to 10–500 steps), then decays automatically so small datasets learn promptly and long runs keep refining."); DoubleField("Weight decay", value: $draft.training.weightDecay, help: "Regularization applied by AdamW."); DoubleField("Validation", value: $draft.training.validationSplit, help: "Fraction of whole recordings held out. Rare controls stay in training. A lone recording gets a history/frame embargo; if no honest held-out tail remains, validation is disabled instead of reporting a leaked score.") }
                        HStack { DoubleField("Perception FPS", value: $draft.training.perceptionFPS, help: "How often the AI receives a new screen frame and signed motion difference. It cannot exceed Action FPS."); DoubleField("Action FPS", value: $draft.training.actionFPS, help: "How often the AI may update mouse, keyboard, button, and scroll output."); Picker("Precision", selection: $draft.training.precision) { ForEach(TrainingPrecision.allCases) { Text($0.rawValue).tag($0) } } }
                        HStack { Text("Architecture preset").foregroundStyle(.secondary); Button("Small") { draft.training.architecture = .small }.primaryButton(); Button("Balanced") { draft.training.architecture = .balanced }.primaryButton(); Button("Large") { draft.training.architecture = .large }.primaryButton() }
                        HStack { IntField("Visual width", value: $draft.training.architecture.visualEmbedding, help: "How much visual information is kept after the convolution layers."); IntField("Recurrent width", value: $draft.training.architecture.recurrentWidth, help: "How much capacity is used to remember the recent action history."); Picker("History encoder", selection: $draft.training.architecture.recurrentKind) { ForEach(RecurrentKind.allCases) { Text($0.rawValue).tag($0) } } }
                        HStack { InfoTip("The four-stage spatial encoder sees the current frame, signed frame-to-frame motion, and screen coordinates. It preserves layout instead of averaging the screen into one vector. Convolution channels control visual capacity; recurrent width controls short action history."); Spacer() }
                        HStack { IntArrayField("Conv channels", values: $draft.training.architecture.convolutionChannels); IntArrayField("Kernels", values: $draft.training.architecture.kernelSizes); IntArrayField("Strides", values: $draft.training.architecture.strides); IntArrayField("Fusion widths", values: $draft.training.architecture.fusionWidths) }
                        HStack { DoubleField("Dropout", value: $draft.training.architecture.dropout, help: "Randomly hides a small share of features during training to reduce memorization. It is disabled while the AI runs."); Spacer(); Text("Estimated parameters: \(ModelSizing.parameterCount(draft).formatted())").font(.caption.monospacedDigit()).foregroundStyle(ATColor.cyan) }
                    }
                }
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
                OLEDCard {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Saved brains and autosaves").font(.headline).foregroundStyle(ATColor.violet)
                            InfoTip("This list is loaded only when you ask for it, which keeps the page fast after long runs. Completed training activates the lowest held-out validation-loss brain; the separate latest checkpoint remains available for exact continuation.")
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
                                    VStack(alignment: .leading) { HStack { Text(version.name).font(.subheadline.bold()); if draft.activeVersionID == version.id { StatusPill(text: "Active", color: ATColor.green) } }; Text("\(version.globalStep) steps • \(version.epoch ?? 0) epochs • train \(version.trainingLoss.formatted(.number.precision(.fractionLength(4))))\(version.validationLoss.map { " • validation \($0.formatted(.number.precision(.fractionLength(4))))" } ?? "") • \(version.demonstratedKeyCodes.map { "\($0.count) learned keys" } ?? "legacy key set derived at run") • \(version.createdAt.formatted(date: .abbreviated, time: .shortened))").font(.caption).foregroundStyle(.secondary) }
                                    Spacer(); Button(version.optimizerFile == nil ? "Run this" : "Revert & Resume") { Task { await model.activateVersion(version); draft.activeVersionID = version.id } }.primaryButton(color: ATColor.violet)
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
                Text("This protected AI cannot lose or replace its brain. Duplicate it with its training intact, then make the architecture or exact-vision change on the copy.")
            } else {
                Text("This changes the AI's architecture or exact vision contract. Its existing weights, optimizer state, saved brains, steps, and epochs cannot be attached safely and will be permanently cleared. Recordings are kept.")
            }
        }
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
                    InfoTip("This is a quick comparison between the values used for one decision and the number of learned parameters in the selected network. Comfortable or Balanced means there is no obvious size mismatch. High or Too high suggests choosing a larger architecture preset or reducing resolution or history. It is a practical guide, not a guarantee of training quality.")
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
                        Text("vision, history, memory, and rates").font(.caption2).foregroundStyle(.secondary)
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
                    title: "First convolution",
                    value: input.firstConvolutionValues.formatted(),
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
                    title: "Packed vision source",
                    value: "\(input.packedVisionValues.formatted()) UInt8 values",
                    detail: packedVisionDetail(input),
                    color: ATColor.amber
                )
                NeuralInputBreakdown(
                    title: "Expanded model vision",
                    value: "\(input.expandedVisionValues.formatted()) values",
                    detail: "\(profile.preprocessing.width) × \(profile.preprocessing.height) × \(profile.preprocessing.channelCount) after chroma expansion",
                    color: ATColor.cyan
                )
                NeuralInputBreakdown(
                    title: "Signed visual motion",
                    value: "\(input.temporalDifferenceValues.formatted()) values",
                    detail: "Current frame minus the immediately preceding perception frame",
                    color: ATColor.amber
                )
                NeuralInputBreakdown(
                    title: "Generated coordinates",
                    value: "\(input.coordinateValues.formatted()) values",
                    detail: "X and Y at every one of \(input.pixelCount.formatted()) pixels",
                    color: ATColor.violet
                )
                NeuralInputBreakdown(
                    title: "Previous-action history",
                    value: "\(input.historyValues.formatted()) values",
                    detail: historyDetail(input),
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
                Text("At \(profile.training.perceptionFPS.formatted()) Perception FPS: up to \(input.runtimeValuesPerSecond.formatted()) input values/s from \(bytes(input.packedVisionBytesPerSecond)) of packed vision/s")
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
            "Consider a larger architecture preset, or lower the resolution or History value."
        case .tooHigh:
            "Choose a larger architecture preset or reduce resolution or History before training."
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

    private func packedVisionDetail(_ input: NeuralInputSummary) -> String {
        let storedBytes = bytes(input.packedVisionValues)
        let levels = input.quantizationLevels.formatted()
        if profile.preprocessing.colorMode == .grayscale {
            return "\(input.lumaValues.formatted()) Y • \(storedBytes) stored • \(levels) levels (\(input.effectivePackedBits.formatted()) meaningful bits)"
        }
        return "\(input.lumaValues.formatted()) Y + 2 × \(input.chromaValuesPerPlane.formatted()) chroma at \(profile.preprocessing.chroma.rawValue) • \(storedBytes) stored • \(levels) levels (\(input.effectivePackedBits.formatted()) meaningful bits)"
    }

    private func historyDetail(_ input: NeuralInputSummary) -> String {
        if profile.training.historyLength == 0 {
            return "History disabled; one all-zero row × \(input.actionValuesPerHistoryStep.formatted()) keeps a valid recurrent shape"
        }
        return "\(input.historySteps.formatted()) earlier actions × \(input.actionValuesPerHistoryStep.formatted()) values • \(input.historyDurationSeconds.formatted(.number.precision(.fractionLength(3))))s at \(profile.training.actionFPS.formatted()) Action FPS"
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
    private enum GraphKind: String, CaseIterable, Identifiable { case current = "Current Loss"; case validation = "Validation Loss"; var id: String { rawValue } }
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

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionTitle("Training", "Compiled MLX training uses temporal vision, class-balanced controls, transition-balanced batches, anti-shortcut history masking, and leak-resistant best-brain selection. Another trained AI can run simultaneously.")
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
                            Picker("Graph", selection: $graphKind) { ForEach(GraphKind.allCases) { Text($0.rawValue).tag($0) } }.pickerStyle(.segmented).frame(width: 290)
                            Button { zoomFraction = max(0.02, zoomFraction / 1.6) } label: { Image(systemName: "plus.magnifyingglass") }.primaryButton().help("Zoom in")
                            Button { zoomFraction = min(1, zoomFraction * 1.6) } label: { Image(systemName: "minus.magnifyingglass") }.primaryButton().help("Zoom out")
                            Text("\(Int(zoomFraction * 100))%").font(.caption.monospacedDigit()).foregroundStyle(.secondary).frame(width: 38)
                        }
                        let values = graphKind == .current ? model.trainingMetrics.lossHistory : model.trainingMetrics.validationHistory
                        let accent = graphKind == .current ? ATColor.cyan : ATColor.violet
                        LossChart(values: values, zoomFraction: zoomFraction, color: accent, isActive: model.isAppActive)
                            .id("\(graphKind.rawValue)-\(zoomFraction)")
                            .frame(height: 250)
                        HStack { Text(graphKind.rawValue).foregroundStyle(accent).font(.caption.bold()); Spacer(); Text(values.last?.formatted(.number.precision(.fractionLength(6))) ?? "No data yet").font(.caption.monospacedDigit()).foregroundStyle(.secondary) }
                    }
                }
                HStack(spacing: 12) {
                    MetricCard(title: "Actual training time", value: TrainingDurationFormatter.string(seconds: displayedTiming.trainingSeconds), symbol: "clock.fill", color: ATColor.cyan)
                    MetricCard(title: "Equivalent experience", value: (displayedTiming.experienceIsEstimated ? "~" : "") + TrainingDurationFormatter.string(seconds: displayedTiming.experienceSeconds), symbol: "brain.head.profile.fill", color: ATColor.violet)
                    MetricCard(title: "Samples / sec", value: model.trainingMetrics.samplesPerSecond.formatted(.number.precision(.fractionLength(0))), symbol: "bolt.fill", color: ATColor.amber)
                    MetricCard(title: "Optimizer steps", value: displayedSteps.formatted(), symbol: "arrow.triangle.2.circlepath", color: ATColor.green)
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
                        if let profile = displayedProfile { Text("\(profile.preprocessing.width) × \(profile.preprocessing.height) • \(profile.preprocessing.bitDepth)-bit \(profile.preprocessing.chroma.rawValue) • \(profile.training.precision.rawValue) • perception \(profile.training.perceptionFPS.formatted()) FPS • action \(profile.training.actionFPS.formatted()) FPS").font(.caption).foregroundStyle(.secondary) }
                        Text("Exact continuation always uses the latest optimizer checkpoint. When a validation split exists, completed training runs the lowest-loss held-out brain so late regression is never silently activated.").font(.caption2).foregroundStyle(.secondary)
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
                SectionTitle("Run", "A persistent ScreenCaptureKit stream sees the exact training resolution while action execution runs independently.")
                OLEDCard {
                    HStack {
                        Picker("AI to run", selection: $model.selectedProfileID) {
                            Text("Select an AI…").tag(UUID?.none)
                            ForEach(model.profiles) { profile in
                                Text("\(profile.name) — \(profile.trainingProgress?.globalStep ?? 0) steps").tag(Optional(profile.id))
                            }
                        }
                        .frame(maxWidth: 430)
                        .disabled(model.agentIsActiveOrStarting)
                        InfoTip("Choose the trained AI to run here. The active saved brain for that AI is loaded directly, so the full autosave list does not need to be opened.")
                        Spacer()
                        if let progress = model.selectedProfile?.trainingProgress {
                            Text("\(progress.globalStep) steps • \(progress.epoch) epochs").font(.caption.bold()).foregroundStyle(ATColor.green)
                        }
                    }
                }
                if let profile = model.selectedProfile {
                    OLEDCard {
                        HStack { VStack(alignment: .leading, spacing: 7) { Text(profile.name).font(.title2.bold()); Text("Model vision \(profile.preprocessing.width) × \(profile.preprocessing.height) — live capture will be exactly the same").foregroundStyle(ATColor.cyan); Text("\(profile.preprocessing.colorMode.rawValue), \(profile.preprocessing.bitDepth)-bit, \(profile.preprocessing.chroma.rawValue)").foregroundStyle(.secondary) }; Spacer(); StatusPill(text: model.isStartingAgent ? "Starting / stopping" : model.isRunning ? "AI running" : profile.activeVersionID == nil ? "Training required" : "Ready", color: model.agentIsActiveOrStarting ? ATColor.green : ATColor.violet) }
                    }
                    OLEDCard {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("What the AI watches").font(.headline).foregroundStyle(ATColor.cyan)
                            Picker("Type", selection: $model.captureKind) { ForEach(CaptureKind.allCases) { Text($0.rawValue).tag($0) } }.pickerStyle(.segmented).onChange(of: model.captureKind) { _, _ in Task { await model.refreshSources() } }
                            Picker("Source", selection: $model.selectedSourceID) { Text("Select…").tag(UInt32?.none); ForEach(sources) { Text("\($0.name) — \($0.detail)").tag(Optional($0.id)) } }
                            if model.captureKind == .screenRegion || model.captureKind == .windowRegion {
                                HStack { LabeledNumber("X", value: $model.regionX); LabeledNumber("Y", value: $model.regionY); LabeledNumber("Width", value: $model.regionWidth); LabeledNumber("Height", value: $model.regionHeight); if model.captureKind == .screenRegion { Button("Draw Region") { model.selectScreenRegion() }.primaryButton(color: ATColor.cyan) } }
                            }
                            Text("The persistent stream is preprocessed to the model's exact \(profile.preprocessing.width) × \(profile.preprocessing.height) vision contract. AgentTrainer and its floating HUD are excluded.").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    HStack(alignment: .top, spacing: 14) {
                        OLEDCard {
                            VStack(alignment: .leading, spacing: 14) {
                            Text("Perception and action").font(.headline)
                            HStack { Picker("Frame mode", selection: $model.frameMode) { ForEach(FrameMode.allCases) { Text($0.rawValue).tag($0) } }.pickerStyle(.segmented); InfoTip("Newest Frame stays responsive by skipping old frames when inference is busy. Every Frame preserves order by slowing capture instead of building a large queue.") }
                            HStack { Picker("Mouse execution", selection: $model.runMouseMode) { ForEach(MouseControlMode.allCases) { Text($0.rawValue).tag($0) } }.pickerStyle(.segmented); InfoTip("Auto checks the selected AI's recordings and chooses normal cursor movement or signed Game Camera deltas. This prevents a locked-camera recording from being run as positive screen coordinates, which can look like the AI only moves right and down. Use Absolute Cursor only for a visibly moving pointer.") }
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
                                            Picker("Layer", selection: $model.cnnVisualizationSettings.convolutionLayer) {
                                                Text(profile.training.architecture.convolutionChannels.isEmpty ? "Vision input" : "Final convolution").tag(-1)
                                                ForEach(profile.training.architecture.convolutionChannels.indices, id: \.self) { index in Text("Conv \(index + 1)").tag(index) }
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
                    HStack { StatusPill(text: model.isTraining ? "Background training remains active" : "Custom panic shortcut armed", color: model.isTraining ? ATColor.cyan : ATColor.coral); Spacer(); if model.agentIsActiveOrStarting { Button(model.isStartingAgent ? "Cancel AI Start & Release Inputs" : "Stop AI & Disable All Hooks") { Task { await model.stopAgent() } }.primaryButton(color: ATColor.coral) } else { Button("Run AI") { Task { await model.startAgent() } }.primaryButton(color: ATColor.green).disabled(profile.activeVersionID == nil || model.trainingProfileID == profile.id || model.recordingIsActiveOrStarting || model.isReplaying) } }
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
                OLEDCard { HStack { VStack(alignment: .leading) { HStack { Text("Packed dataset caches").font(.headline); InfoTip("Caches store each perception frame once, map faster action ticks to compact frame indices, and preserve the exact previous frame for motion input. Clearing them never deletes recordings, profiles, or checkpoints.") }; Text("Delete reusable decoded observations without deleting recordings or models.").foregroundStyle(.secondary) }; Spacer(); Button("Clear Caches") { Task { await model.clearCaches() } }.primaryButton(color: ATColor.amber) } }
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
    private var appState: String { "recording=\(model.isRecording), recordingStarting=\(model.isStartingRecording), training=\(model.isTraining), running=\(model.isRunning), agentStarting=\(model.isStartingAgent), replaying=\(model.isReplaying), activity=\(model.activityStatus)" }
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
                        HStack { Text("Global keybinds").font(.headline).foregroundStyle(ATColor.cyan); InfoTip("Click a shortcut, then press a new combination. Shortcuts work globally and are removed from recordings and human-interruption safety.") }
                        HotkeySettingsEditor(model: model).disabled(model.agentIsActiveOrStarting || model.recordingIsActiveOrStarting)
                        if model.agentIsActiveOrStarting || model.recordingIsActiveOrStarting { Text("Shortcuts are locked until the active recording or agent session stops.").font(.caption).foregroundStyle(ATColor.amber) }
                    }
                }
                OLEDCard { VStack(alignment: .leading, spacing: 12) { HStack { Text("Global safety").font(.headline).foregroundStyle(ATColor.coral); InfoTip("The panic shortcut disables capture/action hooks, drains background work, releases every held control, and posts a neutral mouse event.") }; Toggle("Stop AI on any physical human input", isOn: $model.safety.stopOnHumanInput); Toggle("Allow full-Mac control by default", isOn: $model.safety.allowFullMac).tint(ATColor.coral) } }
                ThemeSettingsView()
                OLEDCard { HStack { VStack(alignment: .leading, spacing: 4) { Text("Diagnostics and app logs").font(.headline).foregroundStyle(ATColor.cyan); Text("Open the dedicated tab for persistent errors, prints, crash reports, MLX memory, and a copyable support report.").foregroundStyle(.secondary) }; Spacer(); Button("Open Diagnostics") { model.selection = .diagnostics }.primaryButton() } }
                StorageSettingsView(model: model)
                HStack { Spacer(); Text("AgentTrainer v\(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.8.6") (\(Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "17"))").font(.caption2.monospacedDigit()).foregroundStyle(.tertiary) }
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

                Label("The top bar and sidebar use solid, theme-matched surfaces on macOS Sequoia, Tahoe, and macOS 27, avoiding version-specific glass and inactive-window gray states.", systemImage: "macwindow")
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
    @State private var monitor: Any?
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
        monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            var modifiers: UInt32 = 0
            if event.modifierFlags.contains(.command) { modifiers |= UInt32(1 << 8) }
            if event.modifierFlags.contains(.shift) { modifiers |= UInt32(1 << 9) }
            if event.modifierFlags.contains(.option) { modifiers |= UInt32(1 << 11) }
            if event.modifierFlags.contains(.control) { modifiers |= UInt32(1 << 12) }
            var settings = model.hotkeys
            let captured = HotkeyBinding(keyCode: UInt32(event.keyCode), carbonModifiers: modifiers)
            switch field { case .panic: settings.panic = captured; case .record: settings.record = captured; case .run: settings.run = captured }
            stop(resume: false)
            model.saveHotkeys(settings)
            return nil
        }
    }
    private func stop(resume: Bool = true) { if let monitor { NSEvent.removeMonitor(monitor); self.monitor = nil }; listening = nil; if resume { model.resumeGlobalHotkeys() } }
    private func display(_ value: HotkeyBinding) -> String {
        var result = ""
        if value.carbonModifiers & UInt32(1 << 12) != 0 { result += "⌃" }
        if value.carbonModifiers & UInt32(1 << 11) != 0 { result += "⌥" }
        if value.carbonModifiers & UInt32(1 << 9) != 0 { result += "⇧" }
        if value.carbonModifiers & UInt32(1 << 8) != 0 { result += "⌘" }
        return result + KeyNames.name(for: UInt16(clamping: value.keyCode))
    }
}

private struct PermissionSetting: View { let name: String; let granted: Bool; let action: () -> Void; init(_ name: String, granted: Bool, action: @escaping () -> Void) { self.name = name; self.granted = granted; self.action = action }; var body: some View { HStack { Label(name, systemImage: granted ? "checkmark.circle.fill" : "exclamationmark.circle.fill").foregroundStyle(granted ? ATColor.green : ATColor.amber); Spacer(); Button(granted ? "Open Settings" : "Grant") { action() }.primaryButton(color: granted ? .secondary : ATColor.amber) } } }
