import ATCore
import CoreGraphics
import Foundation
import ImageIO

/// `atctl elkborn-decode <image>...`: decodes Elkborn screen-protocol frames (pages and strips)
/// from screenshots, assembles complete payloads and prints what they contain.
enum ElkbornCommand {
    static func run(_ paths: [String]) throws {
        var assembler = ElkbornPageAssembler()
        var scanner = ElkbornScanner()
        for path in paths where path.hasSuffix(".hex") {
            // a raw timeline payload, as hex (read from the game's page cells)
            let hex = try String(contentsOfFile: path, encoding: .utf8).filter(\.isHexDigit)
            var bytes: [UInt8] = []
            var i = hex.startIndex
            while i < hex.endIndex, let j = hex.index(i, offsetBy: 2, limitedBy: hex.endIndex) { bytes.append(UInt8(hex[i ..< j], radix: 16)!); i = j }
            let t = try ElkbornTimeline(payload: Data(bytes))
            print("\(path): \(bytes.count) bytes: \(t.runner) on \(t.board), run \(t.runID), \(String(format: "%.3f", t.finalTime)) s, avatar \(t.avatar) (\(Int(t.avatarUserID)))")
            print("  \(t.samples.count) camera samples over \(t.samples.last?.timeMs ?? 0) ms, \(t.events.count) inputs, yaw/px \(t.yawPerPixel), pitch/px \(t.pitchPerPixel)")
            print("  keys: " + t.keys.keys.sorted().map { "\($0)=\(t.keys[$0]!)" }.joined(separator: " "))
            for e in t.events { print("  \(e.timeMs) ms action \(e.code) \(e.down ? "down" : "up") -> \(t.keys[e.code] ?? "?")") }
            let records = ElkbornInputs.events(for: t, unitsPerPixel: 1, pointer: .zero)
            print("  as AgentTrainer events: \(records.filter { $0.kind != .mouseMove }.count) keys/buttons, \(records.filter { $0.kind == .mouseMove }.count) mouse moves")
        }
        for path in paths where !path.hasSuffix(".hex") {
            guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else { print("\(path): not an image"); continue }
            let w = image.width, h = image.height
            var pixels = [UInt8](repeating: 0, count: w * h * 4)
            let ctx = CGContext(data: &pixels, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            let name = URL(fileURLWithPath: path).lastPathComponent
            pixels.withUnsafeBytes { raw in
                let frame = LumaFrame(base: raw.baseAddress!, width: w, height: h, bytesPerRow: w * 4)
                if let page = scanner.page(frame) {
                    print("\(name): page kind \(page.kind) session \(page.session) \(page.index + 1)/\(page.count), \(page.payload.count) of \(page.total) bytes")
                    if page.kind == .data, let payload = assembler.add(page) {
                        do {
                            let t = try ElkbornTimeline(payload: payload)
                            print("  timeline: \(t.runner) on \(t.board), run \(t.runID), \(String(format: "%.3f", t.finalTime)) s, avatar \(t.avatar)")
                            print("  \(t.samples.count) camera samples, \(t.events.count) inputs, yaw/px \(t.yawPerPixel), pitch/px \(t.pitchPerPixel)")
                            print("  keys: " + t.keys.keys.sorted().map { "\($0)=\(t.keys[$0]!)" }.joined(separator: " "))
                            let events = ElkbornInputs.events(for: t, unitsPerPixel: 1, pointer: .zero)
                            let names = events.prefix(12).map { e -> String in
                                let key = KeyCatalog.byKeyCode[e.code]?.id ?? "\(e.code)"
                                switch e.kind {
                                case .keyDown: return "\(e.timeNs / 1_000_000)ms \(key)↓"
                                case .keyUp: return "\(e.timeNs / 1_000_000)ms \(key)↑"
                                case .flagsChanged: return "\(e.timeNs / 1_000_000)ms \(key)\(e.flags & 2 != 0 ? "↓" : "↑")"
                                case .buttonDown: return "\(e.timeNs / 1_000_000)ms button\(e.code)↓"
                                case .buttonUp: return "\(e.timeNs / 1_000_000)ms button\(e.code)↑"
                                case .mouseMove: return String(format: "%lldms move(%.1f,%.1f)", e.timeNs / 1_000_000, e.dx, e.dy)
                                default: return "\(e.kind)"
                                }
                            }
                            print("  first inputs: " + names.joined(separator: ", "))
                        } catch {
                            print("  timeline did not parse: \(error)")
                        }
                    }
                } else if let strip = scanner.strip(frame) {
                    print("\(name): strip session \(strip.session) frame \(strip.frame) value \(strip.value) flags \(strip.flags.rawValue) cell \(String(format: "%.2f", strip.cell))px")
                } else {
                    print("\(name): nothing found")
                }
            }
        }
        if assembler.count > 0, !assembler.isComplete { print("timeline incomplete: \(assembler.received) of \(assembler.count) pages") }
    }
}
