import ATCapture
import ATCore
import SwiftUI

/// The overlay shown while recording (your input) and while running (the model's output).
/// Keys appear once used and stay; the held key grows and pushes its neighbours aside; the mouse leans into its motion.
struct HUDView: View {
    @Bindable var model: HUDModel
    private var s: CGFloat { model.options.scale }
    private var isRun: Bool { model.mode == .run }
    /// Run and replay both send input; they share the accent look.
    private var sends: Bool { model.mode != .record }
    private var fill: Color { sends ? Theme.accent : Theme.text }
    private var ink: Color { sends ? .white : .black }
    private var right: Bool { model.options.corner == .bottomRight || model.options.corner == .topRight }

    /// Fixed panel size: the HUD grows inside it towards its corner, so the window never jumps.
    static func panelSize(scale s: CGFloat) -> CGSize { CGSize(width: 680 * s, height: 280 * s) }

    var body: some View {
        let size = Self.panelSize(scale: s)
        VStack(alignment: right ? .trailing : .leading, spacing: 10 * s) {
            if model.options.stats { status }
            if let n = model.notice {
                Label(n, systemImage: "checkmark.circle.fill").font(Theme.font(12 * s, .medium))
                    .foregroundStyle(Theme.text2).lineLimit(1)
                    .transition(.opacity.combined(with: .move(edge: right ? .trailing : .leading)))
            }
            if isRun, let i = model.instruction, !i.isEmpty {
                Text("“\(i)”").font(Theme.font(12 * s)).foregroundStyle(Theme.text2)
                    .lineLimit(1).truncationMode(.tail)
                    .frame(maxWidth: 380 * s, alignment: right ? .trailing : .leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(alignment: .bottom, spacing: 16 * s) {
                if isRun, model.options.vision, let v = model.vision { vision(v) }
                if model.options.keyboard {
                    KeyboardHUD(seen: model.seen, pressed: model.snapshot.keys, unit: 22 * s, fill: fill, ink: ink,
                                anchor: right ? .bottomTrailing : .bottomLeading,
                                emptyText: model.mode == .run ? "Keys appear as the model uses them" : model.mode == .replay ? "Keys appear as the replay presses them" : "Keys appear as they’re pressed")
                }
                if model.options.mouse {
                    MouseHUD(velocity: model.snapshot.velocity, buttons: model.snapshot.buttons, scroll: model.snapshot.scroll, fill: fill, scale: s)
                }
            }
        }
        .padding(.horizontal, 16 * s).padding(.vertical, 14 * s)
        .background(RoundedRectangle(cornerRadius: 16 * s, style: .continuous).fill(Color.black.opacity(0.74)))
        .overlay(RoundedRectangle(cornerRadius: 16 * s, style: .continuous).strokeBorder(Color.white.opacity(0.1)))
        .fixedSize()
        // New keys widen the layout; the panel follows smoothly.
        .animation(.spring(response: 0.45, dampingFraction: 0.86), value: model.seen)
        .animation(Theme.snappy, value: model.notice)
        .animation(Theme.snappy, value: model.correcting)
        .foregroundStyle(Theme.text)
        .opacity(model.options.opacity)
        .padding(14 * s)
        .frame(width: size.width, height: size.height, alignment: alignment)
        .environment(\.colorScheme, .dark)
    }

    private var alignment: Alignment {
        switch model.options.corner {
        case .bottomRight: .bottomTrailing
        case .bottomLeft: .bottomLeading
        case .topRight: .topTrailing
        case .topLeft: .topLeading
        }
    }

    private var status: some View {
        let correcting = isRun && model.correcting
        let color = correcting ? Theme.correction : sends ? (model.paused ? Theme.warning : Theme.live) : Theme.danger
        let label = switch model.mode {
        case .record: "REC"
        case .run: correcting ? "CORRECTING" : model.paused ? "AI PAUSED" : "AI"
        case .replay: model.paused ? "REPLAY PAUSED" : "REPLAY"
        }
        return HStack(spacing: 8 * s) {
            LiveDot(color: color, size: 7 * s)
            Text(label).font(Theme.font(12 * s, .bold)).tracking(0.8)
                .foregroundStyle(sends ? color : Theme.dangerText)
                .contentTransition(.opacity)
            if !model.statusLine.isEmpty {
                Text(model.statusLine).font(.system(size: 12 * s, design: .monospaced)).monospacedDigit()
                    .foregroundStyle(Theme.text2).lineLimit(1)
            }
        }
    }

    private func vision(_ v: CGImage) -> some View {
        VStack(spacing: 5 * s) {
            Image(decorative: v, scale: 1).interpolation(.none).resizable()
                .frame(width: 78 * s, height: 78 * s)
                .clipShape(RoundedRectangle(cornerRadius: 9 * s, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 9 * s, style: .continuous).strokeBorder(Theme.hairlineStrong))
            Text(model.visionCaption).font(.system(size: 10 * s, design: .monospaced)).foregroundStyle(Theme.muted)
        }
    }
}

/// Keys in their real keyboard positions, but only the ones used so far, laid out tightly around them.
struct KeyboardHUD: View {
    let seen: [String]
    let pressed: Set<String>
    let unit: CGFloat
    let fill: Color
    let ink: Color
    let anchor: UnitPoint
    let emptyText: String

    private struct Cap: Identifiable {
        let id: String
        let label: String
        let frame: CGRect
        let on: Bool
        let push: CGSize
        let half: Bool
    }

    private static let halfHeight: Set<String> = ["left", "right", "up", "down"]
    private static let shortLabels = ["control": "ctrl", "rcontrol": "ctrl", "option": "opt", "roption": "opt",
                                      "command": "cmd", "rcommand": "cmd", "space": "", "capslock": "caps"]

    var body: some View {
        let layout = Self.layout(seen: seen, pressed: pressed, unit: unit)
        Group {
            if layout.caps.isEmpty {
                Text(emptyText).font(Theme.font(unit * 0.55)).foregroundStyle(Theme.muted)
                    .frame(height: unit)
            } else {
                ZStack(alignment: .topLeading) {
                    ForEach(layout.caps) { c in
                        key(c)
                            .offset(x: c.frame.minX + c.push.width, y: c.frame.minY + c.push.height)
                            .zIndex(c.on ? 2 : 1)
                            .transition(.asymmetric(insertion: .scale(scale: 0.35).combined(with: .opacity), removal: .identity))
                    }
                }
                .frame(width: layout.size.width, height: layout.size.height, alignment: .topLeading)
                // While anything is held the whole keyboard swells a touch.
                .scaleEffect(pressed.isDisjoint(with: seen) ? 1 : 1.04, anchor: anchor)
                // Room for the swell and for keys pushed outward, so nothing pokes past the panel.
                .padding(.horizontal, unit * 0.35).padding(.top, unit * 0.3).padding(.bottom, unit * 0.1)
            }
        }
        .animation(.spring(response: 0.3, dampingFraction: 0.58), value: pressed)
    }

    private func key(_ c: Cap) -> some View {
        let r = c.half ? unit * 0.18 : unit * 0.3
        let shape = RoundedRectangle(cornerRadius: r, style: .continuous)
        let size: CGFloat = c.label.count > 1 ? 0.4 : (c.half ? 0.36 : 0.52)
        return Text(c.label)
            .font(Theme.font(unit * size, .semibold))
            .foregroundStyle(c.on ? ink : Theme.text2)
            .frame(width: c.frame.width, height: c.frame.height)
            .background(shape.fill(c.on ? fill : Theme.raised))
            .overlay(shape.strokeBorder(c.on ? Color.clear : Theme.hairline))
            .shadow(color: .black.opacity(c.on ? 0.55 : 0), radius: unit * 0.3, y: unit * 0.22)
            .scaleEffect(c.on ? 1.18 : 1)
    }

    private static func layout(seen: [String], pressed: Set<String>, unit u: CGFloat) -> (caps: [Cap], size: CGSize) {
        let infos = seen.compactMap { id in KeyCatalog.info(id).map { (id, $0) } }
        guard !infos.isEmpty else { return ([], .zero) }
        func height(_ id: String) -> Double { halfHeight.contains(id) ? 0.5 : 1 }
        let minX = infos.map(\.1.x).min()!, maxX = infos.map { $0.1.x + $0.1.width }.max()!
        let minY = infos.map(\.1.y).min()!, maxY = infos.map { $0.1.y + height($0.0) }.max()!
        let held = infos.filter { pressed.contains($0.0) }
        let caps = infos.map { id, k -> Cap in
            let h = height(id)
            let cx = k.x + k.width / 2, cy = k.y + h / 2
            // Held keys nudge the keys around them outward, more the closer they are.
            var px = 0.0, py = 0.0
            for (pid, p) in held where pid != id {
                let dx = cx - (p.x + p.width / 2), dy = cy - (p.y + height(pid) / 2)
                let d = (dx * dx + dy * dy).squareRoot()
                guard d > 0, d < 2.3 else { continue }
                let f = (2.3 - d) / 2.3 * 0.42
                px += f * dx / d
                py += f * dy / d * 0.8
            }
            let frame = CGRect(x: (k.x - minX) * u + 2, y: (k.y - minY) * u + 2, width: k.width * u - 4, height: h * u - 4)
            return Cap(id: id, label: shortLabels[id] ?? k.label, frame: frame, on: pressed.contains(id),
                       push: CGSize(width: px * u, height: py * u), half: h < 1)
        }
        return (caps, CGSize(width: (maxX - minX) * u, height: (maxY - minY) * u))
    }
}

/// A mouse that tilts and drifts the way it is moving, then springs back upright.
struct MouseHUD: View {
    let velocity: CGVector
    let buttons: Set<Int>
    let scroll: CGVector
    let fill: Color
    let scale: CGFloat

    var body: some View {
        let s = scale
        // Velocity is in points per second; quantised so tiny jitter doesn't restart the spring.
        let tilt = (min(11, max(-11, velocity.dx / 40)) * 2).rounded() / 2
        let lean = CGSize(width: (min(5, max(-5, velocity.dx / 120)) * 2).rounded() / 2 * s,
                          height: (min(5, max(-5, velocity.dy / 120)) * 2).rounded() / 2 * s)
        let body = RoundedRectangle(cornerRadius: 23 * s, style: .continuous)
        let scrolling = scroll.dy != 0 || scroll.dx != 0
        ZStack(alignment: .top) {
            body.fill(Theme.raised)
            HStack(spacing: 0) {
                Rectangle().fill(buttons.contains(0) ? fill : .clear)
                Rectangle().fill(buttons.contains(1) ? fill : .clear)
            }
            .frame(height: 30 * s)
            Rectangle().fill(Theme.hairlineStrong).frame(width: 1, height: 30 * s)
            Rectangle().fill(Theme.hairlineStrong).frame(height: 1).padding(.top, 30 * s)
            Capsule().fill(scrolling || buttons.contains(2) ? fill : Theme.muted)
                .frame(width: 6 * s, height: 13 * s)
                .padding(.top, 8 * s)
                .offset(y: scroll.dy > 0 ? -2 * s : scroll.dy < 0 ? 2 * s : 0)
        }
        .frame(width: 46 * s, height: 70 * s)
        .clipShape(body)
        .overlay(body.strokeBorder(Theme.hairlineStrong))
        .rotationEffect(.degrees(tilt))
        .offset(lean)
        .animation(.spring(response: 0.5, dampingFraction: 0.5), value: tilt)
        .animation(.spring(response: 0.5, dampingFraction: 0.5), value: lean)
        .animation(.easeOut(duration: 0.12), value: buttons)
        .animation(.spring(response: 0.2, dampingFraction: 0.6), value: scrolling)
    }
}
