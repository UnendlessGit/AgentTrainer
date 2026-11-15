import ATCapture
import ATCore
import SwiftUI

struct HUDView: View {
    @Bindable var model: HUDModel
    private var u: CGFloat { 15 * model.options.scale }

    var body: some View {
        VStack(alignment: .trailing, spacing: 8 * model.options.scale) {
            if model.options.stats, !model.statusLine.isEmpty || model.mode == .run {
                HStack(spacing: 6) {
                    Circle().fill(model.mode == .record ? Color.red : (model.paused ? .orange : .green)).frame(width: 7, height: 7)
                    Text(model.mode == .record ? "REC" : (model.paused ? "AI PAUSED" : "AI"))
                        .font(.system(size: 10 * model.options.scale, weight: .bold, design: .rounded))
                    if !model.statusLine.isEmpty {
                        Text(model.statusLine).font(.system(size: 10 * model.options.scale, design: .monospaced)).foregroundStyle(.secondary)
                    }
                }
            }
            if let i = model.instruction, !i.isEmpty, model.mode == .run {
                Text("“\(i)”").font(.system(size: 10 * model.options.scale)).foregroundStyle(.secondary).lineLimit(1)
            }
            HStack(alignment: .bottom, spacing: 10 * model.options.scale) {
                if model.mode == .run, model.options.vision, let v = model.vision {
                    VStack(spacing: 3) {
                        Image(decorative: v, scale: 1).interpolation(.none).resizable()
                            .frame(width: 96 * model.options.scale, height: 96 * model.options.scale)
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                        Text(model.visionCaption).font(.system(size: 8 * model.options.scale, design: .monospaced)).foregroundStyle(.secondary)
                    }
                }
                if model.options.keyboard { KeyboardHUD(pressed: model.snapshot.keys, unit: u) }
                if model.options.mouse { MouseHUD(snapshot: model.snapshot, trail: model.trail, unit: u) }
            }
        }
        .padding(10 * model.options.scale)
        .background(RoundedRectangle(cornerRadius: 12 * model.options.scale).fill(Color.black.opacity(0.55)))
        .foregroundStyle(.white)
        .opacity(model.options.opacity)
        .fixedSize()
    }
}

/// ANSI Mac keyboard where only pressed keys are drawn, each in its real position.
struct KeyboardHUD: View {
    let pressed: Set<String>
    let unit: CGFloat
    private static let keys = KeyCatalog.all.filter { $0.x < 14.6 }

    var body: some View {
        Canvas { ctx, _ in
            for k in Self.keys where pressed.contains(k.id) {
                let isArrowHalf = k.category == .navigation && ["left", "right", "down"].contains(k.id) || k.id == "up"
                let h = isArrowHalf ? unit * 0.48 : unit * 0.92
                let r = CGRect(x: k.x * unit + 1, y: k.y * unit + 1, width: k.width * unit - 2, height: h)
                ctx.fill(Path(roundedRect: r, cornerRadius: unit * 0.18), with: .color(.accentColor))
                let label = Text(k.label).font(.system(size: unit * (k.label.count > 2 ? 0.34 : 0.5), weight: .semibold, design: .rounded))
                ctx.draw(label, at: CGPoint(x: r.midX, y: r.midY))
            }
        }
        .frame(width: 14.75 * unit, height: 6.1 * unit)
        .background(RoundedRectangle(cornerRadius: unit * 0.3).strokeBorder(Color.white.opacity(0.12), lineWidth: 1))
    }
}

struct MouseHUD: View {
    let snapshot: LiveInputState.Snapshot
    let trail: [CGVector]
    let unit: CGFloat

    var body: some View {
        HStack(alignment: .center, spacing: unit * 0.5) {
            // Mouse body with buttons
            ZStack(alignment: .top) {
                RoundedRectangle(cornerRadius: unit * 1.2).strokeBorder(Color.white.opacity(0.35), lineWidth: 1.5)
                    .frame(width: unit * 2.4, height: unit * 3.6)
                HStack(spacing: 2) {
                    UnevenRoundedRectangle(topLeadingRadius: unit * 1.1).fill(snapshot.buttons.contains(0) ? Color.accentColor : .clear)
                    UnevenRoundedRectangle(topTrailingRadius: unit * 1.1).fill(snapshot.buttons.contains(1) ? Color.accentColor : .clear)
                }
                .frame(width: unit * 2.2, height: unit * 1.5).padding(.top, 2)
                Capsule().fill(snapshot.buttons.contains(2) ? Color.accentColor : Color.white.opacity(0.35))
                    .frame(width: unit * 0.28, height: unit * 0.7).padding(.top, unit * 0.35)
                if snapshot.scroll.dy != 0 || snapshot.scroll.dx != 0 {
                    Image(systemName: snapshot.scroll.dy > 0 ? "chevron.up" : snapshot.scroll.dy < 0 ? "chevron.down" : snapshot.scroll.dx > 0 ? "chevron.right" : "chevron.left")
                        .font(.system(size: unit * 0.6, weight: .bold)).foregroundStyle(Color.accentColor)
                        .padding(.top, unit * 1.7)
                }
            }
            // Movement: velocity vector with a fading trail
            Canvas { ctx, size in
                let c = CGPoint(x: size.width / 2, y: size.height / 2), r = size.width / 2 - 2
                ctx.stroke(Path(ellipseIn: CGRect(x: 2, y: 2, width: r * 2, height: r * 2)), with: .color(.white.opacity(0.2)), lineWidth: 1)
                func point(_ v: CGVector) -> CGPoint {
                    let m = (v.dx * v.dx + v.dy * v.dy).squareRoot()
                    let k = m > 1 ? 1 / m : 1
                    return CGPoint(x: c.x + v.dx * k * r, y: c.y + v.dy * k * r)
                }
                for (i, v) in trail.enumerated() {
                    let p = point(v), a = Double(i + 1) / Double(max(trail.count, 1))
                    ctx.fill(Path(ellipseIn: CGRect(x: p.x - 1.5, y: p.y - 1.5, width: 3, height: 3)), with: .color(.accentColor.opacity(a * 0.6)))
                }
                if let last = trail.last {
                    var line = Path(); line.move(to: c); line.addLine(to: point(last))
                    ctx.stroke(line, with: .color(.accentColor), style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                }
                ctx.fill(Path(ellipseIn: CGRect(x: c.x - 2, y: c.y - 2, width: 4, height: 4)), with: .color(.white.opacity(0.6)))
            }
            .frame(width: unit * 3.4, height: unit * 3.4)
        }
    }
}

