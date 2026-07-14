import AppKit
import Combine
import SwiftUI

enum HUDInputSource: String {
    case human = "YOUR INPUT"
    case agent = "AI INPUT"
    var color: Color { self == .human ? .cyan : .purple }
}

@MainActor
final class InputHUDModel: ObservableObject {
    @Published private(set) var state = InputState.empty
    @Published private(set) var source: HUDInputSource = .human
    @Published private(set) var isVisible = false
    @Published private(set) var recentKeys: [UInt16] = []
    @Published private(set) var recentButtons: [UInt8] = []
    @Published private(set) var visionImage: NSImage?
    @Published private(set) var visionSpec: PreprocessingSpec?
    @Published private(set) var showsVision = false
    @Published private(set) var cnnVisualizationImage: NSImage?
    @Published private(set) var cnnVisualizationDetail = ""
    @Published private(set) var cnnVisualizationMode: CNNVisualizationMode = .activationOverlay
    @Published private(set) var showsCNNVisualization = false
    private var controller: InputHUDController?
    private let previewRenderer = VisionPreviewRenderer()
    private let cnnRenderer = CNNVisualizationRenderer()

    func installPanel() {
        guard controller == nil else { return }
        controller = InputHUDController(model: self)
    }

    func show(source: HUDInputSource, vision: Bool = false, cnnVisualization: CNNVisualizationSettings = CNNVisualizationSettings()) {
        self.source = source; state = .empty; recentKeys = []; recentButtons = []; visionImage = nil; visionSpec = nil; showsVision = vision; cnnVisualizationImage = nil; cnnVisualizationDetail = ""; cnnVisualizationMode = cnnVisualization.mode; showsCNNVisualization = cnnVisualization.enabled; isVisible = true
    }
    func hide() { previewRenderer.cancel(); cnnRenderer.cancel(); state = .empty; visionImage = nil; cnnVisualizationImage = nil; showsVision = false; showsCNNVisualization = false; isVisible = false }
    func update(state: InputState, source: HUDInputSource) {
        for key in state.keys.sorted() where !recentKeys.contains(key) { recentKeys.append(key) }
        for button in state.buttons.sorted() where !recentButtons.contains(button) { recentButtons.append(button) }
        if recentKeys.count > 128 { recentKeys.removeFirst(recentKeys.count - 128) }
        if recentButtons.count > 3 { recentButtons.removeFirst(recentButtons.count - 3) }
        self.state = state; self.source = source
    }
    func updateVision(_ frame: VisionPreviewFrame) {
        guard showsVision else { return }
        previewRenderer.submit(frame) { [weak self] image, spec in
            Task { @MainActor in guard let self, self.showsVision else { return }; self.visionSpec = spec; self.visionImage = image }
        }
    }
    func configureCNNVisualization(_ settings: CNNVisualizationSettings) {
        let changedMode = cnnVisualizationMode != settings.mode
        cnnVisualizationMode = settings.mode
        showsCNNVisualization = isVisible && source == .agent && settings.enabled
        if !showsCNNVisualization || changedMode {
            cnnRenderer.cancel(); cnnVisualizationImage = nil; cnnVisualizationDetail = ""
        }
    }
    func updateCNNVisualization(_ frame: CNNVisualizationFrame) {
        guard showsCNNVisualization else { return }
        cnnRenderer.submit(frame) { [weak self] render in
            Task { @MainActor in
                guard let self, self.showsCNNVisualization, self.cnnVisualizationMode == frame.settings.mode else { return }
                self.cnnVisualizationImage = render.image
                self.cnnVisualizationDetail = render.detail
            }
        }
    }
}

private final class VisionPreviewRenderer: @unchecked Sendable {
    private let lock = NSLock()
    private let queue = DispatchQueue(label: "AgentTrainer.VisionPreview", qos: .utility)
    private var newest: (VisionPreviewFrame, @Sendable (NSImage?, PreprocessingSpec) -> Void)?
    private var processing = false

    func submit(_ frame: VisionPreviewFrame, completion: @escaping @Sendable (NSImage?, PreprocessingSpec) -> Void) {
        lock.lock(); newest = (frame, completion)
        guard !processing else { lock.unlock(); return }
        processing = true; lock.unlock()
        queue.async { [weak self] in self?.drain() }
    }

    func cancel() { lock.lock(); newest = nil; lock.unlock() }

    private func drain() {
        while true {
            lock.lock()
            guard let next = newest else { processing = false; lock.unlock(); return }
            newest = nil; lock.unlock()
            next.1(VisionPreprocessor.previewImage(next.0.packed, spec: next.0.spec), next.0.spec)
        }
    }
}

@MainActor
private final class InputHUDController {
    private let panel: NSPanel
    private var cancellables: Set<AnyCancellable> = []

    init(model: InputHUDModel) {
        panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 400, height: 127), styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        panel.level = .floating
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.ignoresMouseEvents = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        panel.contentView = NSHostingView(rootView: InputHUDView(model: model))
        model.$isVisible.sink { [weak self] visible in
            guard let self else { return }
            if visible { position(); panel.orderFrontRegardless() } else { panel.orderOut(nil) }
        }.store(in: &cancellables)
        model.$showsVision.combineLatest(model.$showsCNNVisualization, model.$recentKeys).sink { [weak self] visionVisible, cnnVisible, keys in
            guard let self else { return }
            let rowSets: [Set<UInt16>] = [
                [53, 122, 120, 99, 118, 96, 97, 98, 100, 101, 109, 103, 111],
                [50, 18, 19, 20, 21, 23, 22, 26, 28, 25, 29, 27, 24, 51],
                [48, 12, 13, 14, 15, 17, 16, 32, 34, 31, 35, 33, 30, 42],
                [57, 0, 1, 2, 3, 5, 4, 38, 40, 37, 41, 39, 36],
                [56, 6, 7, 8, 9, 11, 45, 46, 43, 47, 44, 60, 126],
                [59, 58, 55, 49, 54, 61, 62, 123, 125, 124]
            ]
            let keySet = Set(keys)
            let rows = max(1, rowSets.count { !$0.isDisjoint(with: keySet) })
            let keyboardHeight = CGFloat(rows * 17)
            panel.setContentSize(NSSize(width: 400, height: 110 + keyboardHeight + (visionVisible ? 206 : 0) + (cnnVisible ? 206 : 0)))
            if model.isVisible { position() }
        }.store(in: &cancellables)
    }

    private func position() {
        guard let screen = NSScreen.main ?? NSScreen.screens.first else { return }
        let visible = screen.visibleFrame
        panel.setFrameOrigin(NSPoint(x: visible.maxX - panel.frame.width - 22, y: visible.minY + 22))
    }
}

private struct InputHUDView: View {
    @ObservedObject var model: InputHUDModel
    @ObservedObject private var appearance = UIAppearanceStore.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if model.showsCNNVisualization {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Label("CNN INTERNALS", systemImage: "waveform.path.ecg").font(.system(size: 9, weight: .bold)).foregroundStyle(.cyan)
                        Spacer()
                        Text(model.cnnVisualizationMode.rawValue.uppercased()).font(.system(size: 9, design: .monospaced)).foregroundStyle(.secondary)
                    }
                    Group {
                        if let image = model.cnnVisualizationImage { Image(nsImage: image).resizable().interpolation(model.cnnVisualizationMode == .featureChannels ? .none : .medium).scaledToFit() }
                        else { ZStack { Color.black; ProgressView().controlSize(.small) } }
                    }.frame(maxWidth: .infinity, minHeight: 152, maxHeight: 152).background(Color.black).clipShape(RoundedRectangle(cornerRadius: ATCorner.scaled(10), style: .continuous))
                    Text(model.cnnVisualizationDetail.isEmpty ? "Waiting for the next diagnostic frame" : model.cnnVisualizationDetail).font(.system(size: 9)).foregroundStyle(.secondary).lineLimit(1)
                }
                Divider().overlay(Color.white.opacity(0.1))
            }
            if model.showsVision {
                VStack(alignment: .leading, spacing: 6) {
                    HStack { Label("AI VISION", systemImage: "eye.fill").font(.system(size: 9, weight: .bold)).foregroundStyle(.green); Spacer(); if let spec = model.visionSpec { Text("\(spec.width)×\(spec.height) • \(spec.bitDepth)-bit • \(spec.chroma.rawValue)").font(.system(size: 9, design: .monospaced)).foregroundStyle(.secondary) } }
                    Group {
                        if let image = model.visionImage { Image(nsImage: image).resizable().interpolation(.none).scaledToFit() }
                        else { ZStack { Color.black; ProgressView().controlSize(.small) } }
                    }.frame(maxWidth: .infinity, minHeight: 152, maxHeight: 152).background(Color.black).clipShape(RoundedRectangle(cornerRadius: ATCorner.scaled(10), style: .continuous))
                    if let spec = model.visionSpec { Text("\(spec.colorMode.rawValue) • \(spec.chroma.rawValue) chroma • exact model frame").font(.system(size: 9)).foregroundStyle(.secondary) }
                }
                Divider().overlay(Color.white.opacity(0.1))
            }
            HStack {
                Circle().fill(model.source.color).frame(width: 7, height: 7)
                Text(model.source.rawValue).font(.system(size: 11, weight: .bold, design: .rounded)).foregroundStyle(model.source.color)
                Spacer()
                Text("Capture-excluded").font(.system(size: 9, weight: .medium)).foregroundStyle(.secondary)
            }
            VisualKeyboard(usedKeys: Set(model.recentKeys), activeKeys: model.state.keys, accent: model.source.color, compact: true)
            HStack(spacing: 8) {
                HUDGroup(title: "MOUSE", accent: .purple) {
                    HStack(spacing: 5) {
                        ForEach(model.recentButtons.suffix(3), id: \.self) { button in HUDKey(label: mouseLabel(button), active: model.state.buttons.contains(button), color: .purple) }
                        if model.recentButtons.isEmpty { Text("—").foregroundStyle(.tertiary) }
                    }
                }
                Divider().overlay(Color.white.opacity(0.12))
                HUDGroup(title: "MOVE", accent: .green) {
                    Text("x\(Int(model.state.mouseDelta.width)) y\(Int(model.state.mouseDelta.height))")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced)).foregroundStyle(.green)
                }
                HUDGroup(title: "SCROLL", accent: .cyan) {
                    Text("x\(Int(model.state.scrollDelta.width)) y\(Int(model.state.scrollDelta.height))")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced)).foregroundStyle(.cyan)
                }
            }
        }
        .padding(11)
        .background(RoundedRectangle(cornerRadius: ATCorner.scaled(18), style: .continuous).fill(Color(red: 0.025, green: 0.03, blue: 0.045).opacity(0.97)))
        .overlay(RoundedRectangle(cornerRadius: ATCorner.scaled(18), style: .continuous).stroke(model.source.color.opacity(0.42), lineWidth: 0.9))
        .environment(\.uiMotionEnabled, appearance.motionEnabled)
    }

    private func mouseLabel(_ button: UInt8) -> String { switch button { case 0: "L"; case 1: "R"; case 2: "M"; default: "M\(Int(button) + 1)" } }
}

private struct HUDGroup<Content: View>: View {
    let title: String
    let accent: Color
    @ViewBuilder let content: Content
    var body: some View { VStack(alignment: .leading, spacing: 6) { Text(title).font(.system(size: 9, weight: .bold)).foregroundStyle(accent.opacity(0.8)); content }.frame(maxWidth: .infinity, alignment: .leading) }
}

private struct HUDKey: View {
    let label: String
    let active: Bool
    let color: Color
    var body: some View {
        Text(label).font(.system(size: 9, weight: .bold, design: .rounded)).foregroundStyle(active ? .white : .secondary).frame(minWidth: 21, minHeight: 20)
            .background(RoundedRectangle(cornerRadius: ATCorner.scaled(7), style: .continuous).fill(active ? color.opacity(0.3) : Color.white.opacity(0.04)))
            .overlay(RoundedRectangle(cornerRadius: ATCorner.scaled(7), style: .continuous).stroke(active ? color : Color.white.opacity(0.1), lineWidth: 1))
    }
}

enum KeyNames {
    static func name(for code: UInt16) -> String {
        let names: [UInt16: String] = [0:"A",1:"S",2:"D",3:"F",4:"H",5:"G",6:"Z",7:"X",8:"C",9:"V",11:"B",12:"Q",13:"W",14:"E",15:"R",16:"Y",17:"T",18:"1",19:"2",20:"3",21:"4",22:"6",23:"5",24:"=",25:"9",26:"7",27:"-",28:"8",29:"0",30:"]",31:"O",32:"U",33:"[",34:"I",35:"P",36:"↩",37:"L",38:"J",39:"'",40:"K",41:";",42:"\\",43:",",44:"/",45:"N",46:"M",47:".",48:"⇥",49:"Space",50:"`",51:"⌫",53:"Esc",55:"⌘",56:"⇧",57:"⇪",58:"⌥",59:"⌃",123:"←",124:"→",125:"↓",126:"↑"]
        return names[code] ?? "K\(code)"
    }
}
