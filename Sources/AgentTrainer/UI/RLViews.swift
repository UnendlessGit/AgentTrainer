import AppKit
import Charts
import SwiftUI

struct RLSettingsView: View {
    @Binding var profile: AIProfile
    @ObservedObject var model: AppModel
    @State private var preset: RLPreset = .balanced
    private var settings: Binding<RLConfiguration> {
        Binding(get: { profile.effectiveRL }, set: { profile.rlTraining = $0 })
    }

    var body: some View {
        VStack(spacing: 14) {
            OLEDCard {
                VStack(alignment: .leading, spacing: 14) {
                    HStack {
                        Label("Reinforcement Learning", systemImage: "brain.head.profile").font(.headline).foregroundStyle(ATColor.green)
                        Spacer(); StatusPill(text: "PPO • MLX", color: ATColor.cyan)
                        Picker("Settings", selection: settings.advanced) { Text("Beginner").tag(false); Text("Advanced").tag(true) }
                            .pickerStyle(.segmented).labelsHidden().frame(width: 210)
                    }
                    Text("Learn by acting, receiving screen rewards, and updating the current brain. Switching training methods preserves compatible learned weights.")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack {
                        Picker("Learning preset", selection: $preset) { ForEach(RLPreset.allCases) { Text($0.rawValue).tag($0) } }
                        Button("Apply Preset") { var value = profile.effectiveRL; value.apply(preset); profile.rlTraining = value }.primaryButton(color: ATColor.green)
                    }
                    Text(preset == .stable ? "Smaller changes and less random exploration for an established skill." : preset == .balanced ? "A measured balance of exploration and conservative updates." : "Try more varied controls and larger updates while discovering a new task.")
                        .font(.caption).foregroundStyle(.secondary)
                    if !profile.effectiveRL.advanced {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack { Text("Exploration").font(.caption); Slider(value: settings.binaryExploration, in: 0...0.2); Text(profile.effectiveRL.binaryExploration.formatted(.percent.precision(.fractionLength(0)))).font(.caption.monospacedDigit()) }
                            if profile.effectiveRL.mouseMovement || profile.effectiveRL.scroll {
                                HStack { Text("Mouse variation").font(.caption); Slider(value: settings.explorationScale, in: 0.05...2.5); Text(profile.effectiveRL.explorationScale.formatted(.number.precision(.fractionLength(2))) + "×").font(.caption.monospacedDigit()) }
                            }
                            HStack {
                                Text("Learning stability").font(.caption)
                                Slider(value: Binding(get: { min(1, max(0, (-3 - log10(profile.effectiveRL.learningRate)) / 2)) }, set: { value in
                                    var c = profile.effectiveRL; c.learningRate = pow(10, -3 - 2 * value)
                                    c.clip = 0.3 - 0.2 * value; c.targetKL = 0.05 - 0.04 * value; profile.rlTraining = c
                                }), in: 0...1)
                            }
                            Text("Higher stability reduces the update size and tightens how far the policy may change.").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Divider()
                    RLControlPicker(configuration: settings, model: model)
                }
            }
            OLEDCard {
                VStack(alignment: .leading, spacing: 12) {
                    Label("Episodes and rewards", systemImage: "flag.checkered").font(.headline).foregroundStyle(ATColor.violet)
                    Picker("Reward format", selection: settings.rewardMode) { ForEach(RLRewardMode.allCases) { Text($0.rawValue).tag($0) } }
                    Text(profile.effectiveRL.rewardMode == .cumulative
                         ? "The counter resets to zero for each episode. Only changes in the counter become rewards, including negative changes."
                         : "The visible value is a reward for each action interval. Unread intervals are excluded instead of guessed.")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack {
                        RLEpisodeKey(title: "Episode Start Key", key: settings.startKey, model: model)
                        RLEpisodeKey(title: "Episode End Key", key: settings.endKey, model: model)
                    }
                    Text("These keys belong to AgentTrainer and are excluded from AI actions. Start waits for fresh Reward: 0 and a positive Episode Length. Keep terminal text visible until the next start key.")
                        .font(.caption).foregroundStyle(.secondary)
                    Toggle("Automatic checkpoints", isOn: settings.automaticCheckpoints).tint(ATColor.green)
                    HStack {
                        RLNumber("Between episodes (s)", value: settings.episodeDelay, help: "0–60 seconds without normal AI actions between episodes.")
                        RLNumber("Reset timeout (s)", value: settings.resetTimeout, help: "1–120 seconds to acknowledge reset before pausing safely.")
                        RLNumber("Autosave (s)", value: settings.autosaveSeconds, help: "5–86,400 seconds between automatic checkpoints.")
                    }
                }
            }
            if profile.effectiveRL.advanced { advancedSettings }
            RLSourceSelection(model: model)
            RLRewardSetup(model: model)
            if (try? profile.effectiveRL.validated()) == nil {
                Label("A setting is outside its supported range. Correct it before starting.", systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(ATColor.coral)
            }
        }
        .task(id: profile.id) {
            if !profile.effectiveRL.controlsInitialized {
                var value = profile.effectiveRL
                if let id = profile.activeVersionID, let version = await WorkspaceStore.shared.version(profileID: profile.id, versionID: id) {
                    value.allowedKeys = version.outputCapabilities?.allowedKeyCodes ?? version.demonstratedKeyCodes ?? []
                    value.allowedButtons = version.outputCapabilities?.allowedMouseButtons ?? []
                    value.mouseMovement = version.channels.mouseMovement; value.scroll = version.channels.scroll
                }
                value.controlsInitialized = true; profile.rlTraining = value
            }
            await model.refreshSources()
        }
    }

    private var advancedSettings: some View {
        OLEDCard {
            VStack(alignment: .leading, spacing: 14) {
                Label("PPO optimization", systemImage: "slider.horizontal.3").font(.headline).foregroundStyle(ATColor.cyan)
                HStack {
                    RLNumber("Learning rate", value: settings.learningRate, help: "0.0000001–0.003. Size of an optimizer update; smaller changes usually preserve an established policy more conservatively.")
                    RLInteger("Rollout actions", value: settings.rolloutSize, help: "2–4096 executed decisions before an update. Updates can happen during an episode.")
                    RLInteger("Minibatch", value: settings.batchSize, help: "1–512 reward intervals per optimizer step, no larger than the rollout. GPU work is accumulated in small microbatches.")
                    RLInteger("Epochs", value: settings.epochs, help: "1–20 passes over each rollout. KL early stopping may end optimization sooner.")
                }
                HStack {
                    RLNumber("Discount", value: settings.discount, help: "0.8–1 per nominal Action FPS interval. Higher values credit rewards further into the future; actual elapsed durations are respected.")
                    RLNumber("GAE lambda", value: settings.gaeLambda, help: "0–1. Balances longer reward traces against noisy estimates.")
                    RLNumber("PPO clip", value: settings.clip, help: "0.01–0.5. Limits the incentive for large changes in action probability.")
                    RLNumber("Target KL", value: settings.targetKL, help: "0.0001–0.5. Stop an update early when the policy has changed too much.")
                }
                HStack {
                    RLNumber("Value loss", value: settings.valueCoefficient, help: "0–10. Weight of the critic's reward prediction objective.")
                    RLNumber("Entropy", value: settings.entropyCoefficient, help: "0–1. Encourages varied actions. Reported entropy is normalized by enabled action dimensions.")
                    RLNumber("Gradient limit", value: settings.maximumGradientNorm, help: "0.01–100. Clips the global gradient norm before applying an update.")
                    RLNumber("Weight decay", value: settings.weightDecay, help: "0–1. AdamW regularization; zero is the default for PPO.")
                }
                HStack { Text("Random seed").font(.caption).foregroundStyle(.secondary); TextField("Seed", value: settings.seed, format: .number).textFieldStyle(.roundedBorder); InfoTip("Saved random state makes pause/resume reproducible. A changed seed starts a new optimizer sequence without clearing learned weights.") }
                Divider(); Text("Exploration and reward scaling").font(.subheadline.bold())
                HStack {
                    RLNumber("Binary exploration", value: settings.binaryExploration, help: "0–0.5. Mixes a small random probability into allowed key/button choices, including previously rare actions.")
                    RLNumber("Mouse exploration", value: settings.explorationScale, help: "0.05–5. Multiplies learned Gaussian exploration for enabled mouse controls.")
                    RLNumber("Reward scale", value: settings.rewardScale, help: "0.000001–1,000,000. Multiplies training rewards; dashboard rewards remain in the original units.")
                }
                HStack {
                    Toggle("Normalize advantages", isOn: settings.normalizeAdvantages)
                    Toggle("Normalize rewards", isOn: settings.normalizeRewards)
                }
                HStack {
                    RLNumber("Reset settling (s)", value: settings.resetSettlingSeconds, help: "0–5 seconds after the reset key before accepting a fresh zero reward. Normal AI actions remain blocked.")
                    RLNumber("Detection confidence", value: settings.minimumConfidence, help: "0.1–1. Lower-confidence recognition is rejected instead of interpreted as zero.")
                    RLNumber("Detection timeout (s)", value: settings.detectionTimeout, help: "0.25–30 seconds without a recent valid pair before releasing inputs and saving.")
                    RLNumber("Training compute share", value: settings.trainingComputeFraction, help: "0.1–0.8. Bounds learner duty cycle; overrunning inference receives additional time. This is a scheduling budget, not a GPU reservation.")
                    RLInteger("Retain autosaves", value: settings.retainedAutosaves, help: "1–100 periodic checkpoints. Active, manual, and protected versions are retained.")
                }
            }
        }
    }
}

private struct RLNumber: View {
    var title: String
    @Binding var value: Double
    var help: String
    init(_ title: String, value: Binding<Double>, help: String) { self.title = title; _value = value; self.help = help }
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 4) { Text(title); InfoTip(help) }.font(.caption).foregroundStyle(.secondary)
            TextField(title, value: $value, format: .number).textFieldStyle(.roundedBorder)
        }.frame(maxWidth: .infinity)
    }
}

private struct RLInteger: View {
    var title: String
    @Binding var value: Int
    var help: String
    init(_ title: String, value: Binding<Int>, help: String) { self.title = title; _value = value; self.help = help }
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 4) { Text(title); InfoTip(help) }.font(.caption).foregroundStyle(.secondary)
            TextField(title, value: $value, format: .number).textFieldStyle(.roundedBorder)
        }.frame(maxWidth: .infinity)
    }
}

private struct RLControlPicker: View {
    @Binding var configuration: RLConfiguration
    @ObservedObject var model: AppModel
    @State private var monitor: Any?
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Allowed AI keys").font(.subheadline.bold())
                Spacer(); Button(monitor == nil ? "Add Keys" : "Done Adding Keys") { monitor == nil ? listen() : stop() }.primaryButton(color: monitor == nil ? ATColor.violet : ATColor.amber)
            }
            if monitor != nil { Text("Press every key you want to allow, including modifiers. Click Done when finished.").font(.caption).foregroundStyle(ATColor.amber) }
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 80))], alignment: .leading) {
                ForEach(configuration.allowedKeys.sorted(), id: \.self) { code in
                    Button { configuration.allowedKeys.remove(code) } label: {
                        HStack { Text(KeyNames.name(for: code)); Image(systemName: "xmark.circle.fill") }.font(.caption)
                    }.buttonStyle(.bordered)
                }
            }
            if configuration.allowedKeys.isEmpty { Text("No keys selected. Mouse-only learning is supported.").font(.caption).foregroundStyle(.secondary) }
            Text("Mouse buttons").font(.subheadline.bold())
            HStack {
                ForEach(UInt8(0)..<8, id: \.self) { button in
                    Toggle(button < 3 ? ["Left", "Right", "Middle"][Int(button)] : "\(button + 1)", isOn: Binding(
                        get: { configuration.allowedButtons.contains(button) },
                        set: { if $0 { configuration.allowedButtons.insert(button) } else { configuration.allowedButtons.remove(button) } }))
                        .toggleStyle(.button)
                }
            }
            HStack { Toggle("Mouse movement", isOn: $configuration.mouseMovement); Toggle("Scrolling", isOn: $configuration.scroll) }
            Text("Shared control-channel switches, blocked outputs, and Run permissions still apply.").font(.caption).foregroundStyle(.secondary)
        }.onDisappear { stop() }
    }
    private func listen() {
        model.suspendGlobalHotkeys()
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .flagsChanged]) { event in
            if event.keyCode < 128 { configuration.allowedKeys.insert(event.keyCode); configuration.controlsInitialized = true }
            return nil
        }
    }
    private func stop() {
        if let monitor { NSEvent.removeMonitor(monitor); self.monitor = nil; model.resumeGlobalHotkeys() }
    }
}

private struct RLEpisodeKey: View {
    var title: String
    @Binding var key: UInt16?
    @ObservedObject var model: AppModel
    @State private var monitor: Any?
    var body: some View {
        HStack {
            Text(title).font(.caption)
            Button(monitor != nil ? "Press key…" : key.map { KeyNames.name(for: $0) } ?? "Unassigned") {
                if monitor != nil { stop(); return }
                model.suspendGlobalHotkeys()
                monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .flagsChanged]) { event in key = event.keyCode; stop(); return nil }
            }.buttonStyle(.bordered)
            if key != nil { Button { key = nil } label: { Image(systemName: "xmark.circle") }.buttonStyle(.plain) }
        }.frame(maxWidth: .infinity, alignment: .leading).onDisappear { stop() }
    }
    private func stop() { if let monitor { NSEvent.removeMonitor(monitor); self.monitor = nil; model.resumeGlobalHotkeys() } }
}

struct RLRewardSetup: View {
    @ObservedObject var model: AppModel
    var body: some View {
        OLEDCard {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Label("Reward detection", systemImage: "viewfinder").font(.headline).foregroundStyle(ATColor.cyan)
                    Spacer()
                    Button("Select Reward Region") { Task { await model.selectRewardRegion() } }.primaryButton(color: ATColor.cyan)
                    Button("Test Detection") { Task { await model.testRewardDetection() } }.primaryButton()
                }.disabled(model.agentIsActiveOrStarting || model.isSelectingRewardRegion)
                Text("Select one small area containing white text on black: Reward: 125.5 and Episode Length: 60. AI vision is masked; this reader uses native pixels.")
                    .font(.caption).foregroundStyle(.secondary)
                if let preview = model.rewardPreview {
                    Image(nsImage: NSImage(cgImage: preview.image, size: NSSize(width: preview.image.width, height: preview.image.height))).resizable().interpolation(.none).scaledToFit()
                        .frame(maxWidth: .infinity, maxHeight: 160).background(.black).clipShape(RoundedRectangle(cornerRadius: 10))
                    HStack {
                        LabeledContent("Reward", value: preview.reading.map { $0.reward.formatted() } ?? "—")
                        LabeledContent("Episode length", value: preview.reading.map { $0.episodeLength.formatted() + " s" } ?? "—")
                        Text(preview.framesPerSecond.formatted(.number.precision(.fractionLength(1))) + " reads/s").font(.caption.monospacedDigit())
                    }
                }
                if let confidence = model.rewardPreview?.reading?.confidence {
                    Text("Detection confidence: \(Int(confidence * 100))%").font(.caption).foregroundStyle(.secondary)
                }
                Label(model.rewardTestStatus, systemImage: "text.viewfinder").font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

struct RLDashboard: View {
    @ObservedObject var model: AppModel
    private var m: RLMetrics { model.rlMetrics }
    var body: some View {
        VStack(spacing: 14) {
            RLRewardSetup(model: model)
            OLEDCard {
                VStack(alignment: .leading, spacing: 14) {
                    HStack { Label("Live reinforcement learning", systemImage: "chart.xyaxis.line").font(.headline).foregroundStyle(ATColor.green); Spacer(); StatusPill(text: "PPO", color: ATColor.violet) }
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 180))], alignment: .leading, spacing: 16) {
                        metric("Current episode", "\(m.episode)")
                        metric("Episode reward", m.episodeReward.formatted())
                        metric("Best reward", m.best?.formatted() ?? "—")
                        metric("Average • last 100", m.average?.formatted() ?? "—")
                        metric("Total episodes", "\(m.totalEpisodes)")
                        metric("Time remaining", m.remaining.formatted(.number.precision(.fractionLength(1))) + " s")
                        metric("Training time", TrainingDurationFormatter.string(seconds: m.trainingSeconds))
                        metric(m.legacyUpdateAttempts > 0 ? "Applied updates since upgrade" : "Applied learning",
                            "\(max(0, m.updates - m.legacyUpdateAttempts)) updates • \(m.optimizerSteps) total steps")
                        metric("Rollouts attempted / rejected", "\(m.attemptedUpdates) / \(m.rejectedUpdates)")
                        metric("Action / perception FPS", String(format: "%.1f / %.1f", model.runtimeMetrics.actionFPS, model.runtimeMetrics.perceptionFPS))
                        metric("Reward FPS / OCR latency", String(format: "%.1f / %.1f ms", m.rewardFPS, m.recognitionLatency))
                        metric("Policy / value loss", String(format: "%.4f / %.4f", m.policyLoss, m.valueLoss))
                        metric("Entropy / KL", String(format: "%.4f / %.4f", m.entropy, m.kl))
                    }
                    Divider()
                    Text("Live RL pauses and saves on physical input or target focus loss to avoid learning from another controller.").font(.caption).foregroundStyle(.secondary)
                    Text(m.status).font(.subheadline.bold()).foregroundStyle(model.isRLTraining ? ATColor.green : .secondary)
                    Text(m.controls).font(.caption).foregroundStyle(.secondary)
                    Text("Backlog: \(m.backlogActions) • stale/buffer exclusions: \(m.discardedActions) • KL stopped actions: \(m.klRejectedActions) • unread actions: \(m.unobservedActions)").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    if !m.resumeNotice.isEmpty { Text(m.resumeNotice).font(.caption).foregroundStyle(.secondary) }
                    Text(String(format: "Inference p95 • actions %.1f ms • vision %.1f ms", model.runtimeMetrics.actionP95Milliseconds, model.runtimeMetrics.encodingP95Milliseconds))
                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    Text(m.detectionStatus + (m.skippedIntervals > 0 ? " • \(m.skippedIntervals) unread immediate intervals excluded" : "")).font(.caption).foregroundStyle(.secondary)
                    Chart {
                        ForEach(m.history.suffix(500).filter { !$0.interrupted }) { episode in
                            LineMark(x: .value("Episode", episode.id), y: .value("Reward", episode.reward))
                                .foregroundStyle(by: .value("Series", "Episode reward"))
                                .symbol(.circle).symbolSize(18)
                        }
                        ForEach(rollingHistory) { episode in
                            LineMark(x: .value("Episode", episode.id), y: .value("Average", episode.reward))
                                .foregroundStyle(by: .value("Series", "Average • last 20"))
                        }
                        ForEach(m.history.suffix(500).filter(\.interrupted)) { episode in
                            PointMark(x: .value("Episode", episode.id), y: .value("Reward", episode.reward))
                                .foregroundStyle(by: .value("Series", "Interrupted")).symbol(.cross)
                        }
                    }.chartForegroundStyleScale(["Episode reward": ATColor.green, "Average • last 20": ATColor.cyan, "Interrupted": ATColor.amber])
                        .frame(height: 190).chartXAxisLabel("Episode").chartYAxisLabel("Reward")
                    HStack {
                        if model.isRLTraining {
                            Button("Pause & Save") { Task { await model.stopRL() } }.primaryButton(color: ATColor.amber)
                            Button("Save Snapshot") { Task { await model.saveRL() } }.primaryButton(color: ATColor.cyan)
                            Spacer()
                            Button("Stop & Save") { Task { await model.stopRL() } }.primaryButton(color: ATColor.coral)
                        } else if model.hasUnsavedRLProgress {
                            Text("Progress is retained in memory. Save before closing the app.").font(.caption).foregroundStyle(ATColor.amber)
                            Spacer(); Button("Retry Saving") { Task { await model.saveRL() } }.primaryButton(color: ATColor.amber)
                        } else if model.isStartingAgent {
                            ProgressView().controlSize(.small); Text(model.runtimeStatus).font(.caption)
                            Spacer(); Button("Cancel RL Start") { Task { await model.stopRL() } }.primaryButton(color: ATColor.coral)
                        } else {
                            Text("Configure RL controls and learning in AI Models.").font(.caption).foregroundStyle(.secondary)
                            Spacer()
                            Button(m.updates > 0 ? "Resume RL Training" : "Start RL Training") { Task { await model.startRL() } }.primaryButton(color: ATColor.green)
                                .disabled(model.agentIsActiveOrStarting || model.isTraining || model.recordingIsActiveOrStarting || model.isReplaying)
                        }
                    }
                }
            }
        }
    }
    private var rollingHistory: [RLEpisodeRecord] {
        let completed = m.history.filter { !$0.interrupted }
        return completed.indices.suffix(500).map { index in
            let window = completed[max(0, index - 19)...index]
            return RLEpisodeRecord(id: completed[index].id,
                reward: window.reduce(0) { $0 + $1.reward } / Double(window.count), duration: 0)
        }
    }
    private func metric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 5) { Text(title).font(.caption).foregroundStyle(.secondary); Text(value).font(.system(.body, design: .monospaced).weight(.semibold)) }
    }
}

struct RLSourceSelection: View {
    @ObservedObject var model: AppModel
    var body: some View {
        OLEDCard {
            HStack {
                Picker("Capture type", selection: $model.captureKind) { ForEach(CaptureKind.allCases) { Text($0.rawValue).tag($0) } }
                Picker("Source", selection: $model.selectedSourceID) {
                    Text("Select a source…").tag(UInt32?.none)
                    ForEach(model.captureSources.filter { ($0.kind == .window) == (model.captureKind == .window || model.captureKind == .windowRegion) }) {
                        Text($0.name + " • " + $0.detail).tag(Optional($0.id))
                    }
                }
                Button { Task { await model.refreshSources() } } label: { Image(systemName: "arrow.clockwise") }.buttonStyle(.bordered)
            }
        }.disabled(model.agentIsActiveOrStarting)
    }
}
