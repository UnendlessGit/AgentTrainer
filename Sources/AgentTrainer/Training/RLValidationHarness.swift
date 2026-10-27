import AppKit
import Foundation
import Darwin
import SwiftUI

/// Explicit local QA entry point. It refuses to use the normal workspace and
/// only sends the fixture's two action keys and reset key to its focused window.
@MainActor
enum RLValidationHarness {
    private static var hasRun = false
    static func run(model: AppModel) async {
        if ProcessInfo.processInfo.environment["AGENTTRAINER_RL_AUDIT_MANIFEST"] != nil {
            await RLLiveAudit.run(model: model)
            return
        }
        guard !hasRun, ProcessInfo.processInfo.environment["AGENTTRAINER_RL_VALIDATION"] == "1",
              let workspace = ProcessInfo.processInfo.environment["AGENTTRAINER_WORKSPACE_ROOT"],
              let report = ProcessInfo.processInfo.environment["AGENTTRAINER_RL_VALIDATION_REPORT"],
              workspace.contains("rl-validation") else { return }
        hasRun = true
        let savedWorkflow = UserDefaults.standard.data(forKey: "AgentTrainer.WorkflowSettings")
        func restorePreferences() {
            if let savedWorkflow { UserDefaults.standard.set(savedWorkflow, forKey: "AgentTrainer.WorkflowSettings") }
            else { UserDefaults.standard.removeObject(forKey: "AgentTrainer.WorkflowSettings") }
        }
        defer { restorePreferences() }
        do {
            await model.bootstrap()
            let sources = try await CaptureService.availableSources()
            guard let source = sources.first(where: { $0.kind == .window && $0.name == "AgentTrainer RL Validation Game" }) else {
                throw AgentTrainerError.capture("Launch the signed RL validation fixture first.")
            }
            var profile = AIProfile.fresh(name: "RL Screen Validation")
            profile.preprocessing.width = 96; profile.preprocessing.height = 64; profile.preprocessing.colorMode = .grayscale
            profile.training.architecture = .small; profile.training.temporalVision = TemporalVisionConfiguration(pastFrameCount: 0)
            profile.training.actionFPS = Double(ProcessInfo.processInfo.environment["AGENTTRAINER_RL_VALIDATION_FPS"] ?? "30") ?? 30
            profile.training.perceptionFPS = min(30, profile.training.actionFPS)
            if ProcessInfo.processInfo.environment["AGENTTRAINER_RL_VALIDATION_DEFAULT"] == "1" {
                profile.preprocessing = PreprocessingSpec(); profile.training.architecture = .balanced
                profile.training.temporalVision = TemporalVisionConfiguration()
            }
            profile.channels.mouseMovement = false; profile.channels.buttons = false
            profile.channels.scroll = false; profile.channels.modifiers = false
            var rl = RLConfiguration(); rl.allowedKeys = [0, 2]; rl.endKey = 14; rl.controlsInitialized = true
            if ProcessInfo.processInfo.environment["AGENTTRAINER_RL_VALIDATION_PRODUCTION"] != "1" {
                rl.rolloutSize = 64; rl.batchSize = 8; rl.epochs = 2; rl.learningRate = 0.0003; rl.autosaveSeconds = 30
            }
            rl.rewardRegion = RLRewardRegion(sourceID: source.id, isWindow: true,
                normalized: CodableRect(CGRect(x: 0.01, y: 0.70, width: 0.98, height: 0.29)))
            profile.rlTraining = rl
            try await WorkspaceStore.shared.saveProfile(profile)
            await model.refreshLibrary(); model.selectedProfileID = profile.id
            model.captureSources = sources; model.captureKind = .window; model.selectedSourceID = source.id
            model.safety.stopOnHumanInput = true; model.safety.stopOnFocusLoss = true
            model.selection = .run; model.runUsesRL = true
            var selectionChecks: [String: Bool] = [:]
            if ProcessInfo.processInfo.environment["AGENTTRAINER_RL_VALIDATE_SELECTION"] == "1" {
                selectionChecks = try await validateSelection(model: model, source: source)
            }
            if ProcessInfo.processInfo.environment["AGENTTRAINER_RL_CAPTURE_UI"] == "1" {
                try await captureSettings(profile: model.selectedProfile ?? profile, model: model, report: report)
            }
            await model.startRL()
            let began = Date()
            let duration = Double(ProcessInfo.processInfo.environment["AGENTTRAINER_RL_VALIDATION_SECONDS"] ?? "70") ?? 70
            var snapshots: [[String: Any]] = []
            var interruptions: [[String: Any]] = []
            var paused = false
            while model.isRLTraining, Date().timeIntervalSince(began) < duration {
                try await Task.sleep(for: .seconds(1))
                if snapshots.count == 10, ProcessInfo.processInfo.environment["AGENTTRAINER_RL_CAPTURE_UI"] == "1",
                   let window = NSApp.windows.first(where: { $0.isVisible && $0.frame.width >= 1000 }) {
                    if let image = try? await CaptureService.snapshot(spec: CaptureSpec(kind: .window, windowID: UInt32(window.windowNumber))),
                       let bytes = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]) {
                        try? bytes.write(to: URL(fileURLWithPath: report).deletingPathExtension().appendingPathExtension("png"))
                    }
                }
                let m = model.rlMetrics, runtime = model.runtimeMetrics
                snapshots.append(["elapsed": Date().timeIntervalSince(began), "episode": m.episode,
                    "reward": m.episodeReward, "rewardFPS": m.rewardFPS, "updates": m.updates,
                    "steps": m.optimizerSteps, "actionFPS": runtime.actionFPS,
                    "perceptionFPS": runtime.perceptionFPS, "ocrMilliseconds": m.recognitionLatency,
                    "actionMilliseconds": runtime.actionInferenceMilliseconds, "encodingMilliseconds": runtime.latencyMilliseconds,
                    "actionP95Milliseconds": runtime.actionP95Milliseconds, "encodingP95Milliseconds": runtime.encodingP95Milliseconds,
                    "backlog": m.backlogActions, "discardedActions": m.discardedActions, "residentBytes": residentBytes(),
                    "policyLoss": m.policyLoss, "valueLoss": m.valueLoss, "kl": m.kl,
                    "detectedReward": model.rewardPreview?.reading.map { $0.reward as Any } ?? NSNull(),
                    "detectedLength": model.rewardPreview?.reading.map { $0.episodeLength as Any } ?? NSNull(),
                    "status": m.status, "detection": m.detectionStatus])
                if !paused, Date().timeIntervalSince(began) >= 15,
                   ProcessInfo.processInfo.environment["AGENTTRAINER_RL_VALIDATION_PAUSE"] == "1" {
                    paused = true; await model.stopRL()
                    interruptions.append(["updates": model.rlMetrics.updates, "steps": model.rlMetrics.optimizerSteps,
                        "episodes": model.rlMetrics.totalEpisodes, "status": model.runtimeStatus])
                    await model.startRL()
                    interruptions.append(["updates": model.rlMetrics.updates, "steps": model.rlMetrics.optimizerSteps,
                        "episodes": model.rlMetrics.totalEpisodes, "status": model.runtimeStatus])
                }
                if Date().timeIntervalSince(began) >= 12, ProcessInfo.processInfo.environment["AGENTTRAINER_RL_VALIDATE_PANIC"] == "1" {
                    model.panic()
                    while model.isRLTraining { try await Task.sleep(for: .milliseconds(20)) }
                }
            }
            await model.stopRL()
            let history = model.rlMetrics.history.map { ["episode": $0.id, "reward": $0.reward, "duration": $0.duration, "interrupted": $0.interrupted] as [String: Any] }
            let result: [String: Any] = ["status": model.runtimeStatus, "updates": model.rlMetrics.updates,
                "samples": snapshots, "episodes": history, "workspace": workspace,
                "interruptions": interruptions, "defaultProfile": ProcessInfo.processInfo.environment["AGENTTRAINER_RL_VALIDATION_DEFAULT"] == "1",
                "productionPPOSettings": ProcessInfo.processInfo.environment["AGENTTRAINER_RL_VALIDATION_PRODUCTION"] == "1",
                "selectionChecks": selectionChecks,
                "error": model.errorMessage ?? "", "actionFPSRequested": profile.training.actionFPS]
            try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
                .write(to: URL(fileURLWithPath: report), options: .atomic)
        } catch {
            try? JSONSerialization.data(withJSONObject: ["error": error.localizedDescription], options: [.prettyPrinted])
                .write(to: URL(fileURLWithPath: report), options: .atomic)
        }
        restorePreferences()
        // Leave the SwiftUI task's main-actor turn before AppKit enters its
        // termination loop, so asynchronous safety/save cleanup can finish.
        DispatchQueue.main.async { NSApp.terminate(nil) }
    }

    private static func validateSelection(model: AppModel, source: CaptureSourceOption) async throws -> [String: Bool] {
        let original = model.selectedProfile?.effectiveRL.rewardRegion
        let visible = NSApp.windows.filter(\.isVisible)
        await model.selectRewardRegion()
        guard let panel = NSApp.keyWindow, model.isSelectingRewardRegion else { throw AgentTrainerError.model("Reward selection did not open.") }
        sendKey(53, window: panel)
        let cancelled = !model.isSelectingRewardRegion && model.selectedProfile?.effectiveRL.rewardRegion == original
        let restored = visible.allSatisfy(\.isVisible)
        await model.selectRewardRegion()
        guard let panel = NSApp.keyWindow else { throw AgentTrainerError.model("Reward selection could not reopen.") }
        let size = panel.contentView!.bounds.size
        for (kind, point) in [(NSEvent.EventType.leftMouseDown, CGPoint(x: size.width * 0.01, y: size.height * 0.01)),
                              (.leftMouseDragged, CGPoint(x: size.width * 0.99, y: size.height * 0.30)),
                              (.leftMouseUp, CGPoint(x: size.width * 0.99, y: size.height * 0.30))] {
            if let event = NSEvent.mouseEvent(with: kind, location: point, modifierFlags: [], timestamp: 0,
                windowNumber: panel.windowNumber, context: nil, eventNumber: 0, clickCount: 1, pressure: 1) { panel.sendEvent(event) }
        }
        sendKey(36, window: panel)
        try await Task.sleep(for: .seconds(1))
        let selected = model.selectedProfile?.effectiveRL.rewardRegion
        return ["escapeCancelled": cancelled, "visibilityRestored": restored,
            "enterConfirmed": !model.isSelectingRewardRegion && selected?.matches(source) == true,
            "nativeDetectionPassed": model.rewardPreview?.reading != nil]
    }

    private static func sendKey(_ code: UInt16, window: NSWindow) {
        if let event = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [], timestamp: 0,
            windowNumber: window.windowNumber, context: nil, characters: "", charactersIgnoringModifiers: "", isARepeat: false, keyCode: code) { window.sendEvent(event) }
    }

    private static func captureSettings(profile: AIProfile, model: AppModel, report: String) async throws {
        for advanced in [false, true] {
            var preview = profile; preview.rlTraining?.advanced = advanced
            let view = ScrollView { RLSettingsView(profile: .constant(preview), model: model).padding(24) }.preferredColorScheme(.dark)
            let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 1100, height: advanced ? 1400 : 1100),
                styleMask: [.titled], backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            window.contentView = NSHostingView(rootView: view); window.orderFront(nil)
            try await Task.sleep(for: .milliseconds(250))
            if let image = try? await CaptureService.snapshot(spec: CaptureSpec(kind: .window, windowID: UInt32(window.windowNumber))),
               let bytes = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]) {
                let path = URL(fileURLWithPath: report).deletingPathExtension().path + (advanced ? "-advanced.png" : "-beginner.png")
                try bytes.write(to: URL(fileURLWithPath: path))
            }
            window.close()
        }
    }

    private static func residentBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.resident_size : 0
    }
}
