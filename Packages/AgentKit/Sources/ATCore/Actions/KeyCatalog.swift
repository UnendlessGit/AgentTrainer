import Foundation

/// One physical key: a stable identifier, its macOS virtual key code and its place on
/// an ANSI Mac keyboard (used by the HUD and the key-vocabulary picker).
public struct KeyInfo: Hashable, Sendable, Identifiable {
    public enum Category: String, Codable, Sendable { case letter, digit, symbol, modifier, function, navigation, editing, whitespace }
    public let id: String
    public let label: String
    public let keyCode: UInt16
    public let category: Category
    /// Layout position in key units (1 unit = one letter key).
    public let x: Double
    public let y: Double
    public let width: Double
    /// Character produced without/with shift on a US layout (nil for non-printing keys).
    public let character: Character?
    public let shiftedCharacter: Character?
}

public enum KeyCatalog {
    public static let all: [KeyInfo] = build()
    public static let byID: [String: KeyInfo] = Dictionary(uniqueKeysWithValues: all.map { ($0.id, $0) })
    public static let byKeyCode: [UInt16: KeyInfo] = Dictionary(all.map { ($0.keyCode, $0) }, uniquingKeysWith: { a, _ in a })

    public static func info(_ id: String) -> KeyInfo? { byID[id] }

    /// Pixel2Play's 19-key vocabulary, in its class order (class 0 is "no key").
    public static let p2pKeys = ["space", "1", "2", "3", "4", "a", "d", "e", "f", "q", "w", "s", "z",
                                 "down", "up", "left", "right", "shift", "rshift"]

    /// Extra keys most games need on top of P2P's set (Minecraft, Roblox, shooters).
    public static let gameExtras = ["5", "6", "7", "8", "9", "0", "c", "r", "t", "g", "x", "v", "b", "tab", "escape",
                                    "return", "control", "option", "slash", "f1", "f2", "f3", "f5"]

    /// Every key, for app-control and typing models.
    public static var fullKeyboard: [String] { all.map(\.id) }

    /// Characters a model can type and the key + shift state that produces each.
    public static let typingMap: [Character: (key: String, shift: Bool)] = {
        var m: [Character: (String, Bool)] = [:]
        for k in all {
            if let c = k.character { m[c] = (k.id, false) }
            if let s = k.shiftedCharacter, m[s] == nil { m[s] = (k.id, true) }
        }
        return m
    }()

    private static func build() -> [KeyInfo] {
        var keys: [KeyInfo] = []
        func add(_ id: String, _ label: String, _ code: UInt16, _ cat: KeyInfo.Category, _ x: Double, _ y: Double,
                 _ w: Double = 1, _ ch: Character? = nil, _ sh: Character? = nil) {
            keys.append(KeyInfo(id: id, label: label, keyCode: code, category: cat, x: x, y: y, width: w,
                                character: ch, shiftedCharacter: sh))
        }
        // Function row
        add("escape", "esc", 0x35, .function, 0, 0, 1.5)
        let fCodes: [UInt16] = [0x7A, 0x78, 0x63, 0x76, 0x60, 0x61, 0x62, 0x64, 0x65, 0x6D, 0x67, 0x6F]
        for (i, c) in fCodes.enumerated() { add("f\(i + 1)", "F\(i + 1)", c, .function, 1.5 + Double(i) * (13.0 / 12.0), 0, 13.0 / 12.0) }
        // Number row
        add("grave", "`", 0x32, .symbol, 0, 1, 1, "`", "~")
        let digits: [(String, UInt16, Character)] = [("1", 0x12, "!"), ("2", 0x13, "@"), ("3", 0x14, "#"), ("4", 0x15, "$"),
                                                    ("5", 0x17, "%"), ("6", 0x16, "^"), ("7", 0x1A, "&"), ("8", 0x1C, "*"),
                                                    ("9", 0x19, "("), ("0", 0x1D, ")")]
        for (i, d) in digits.enumerated() { add(d.0, d.0, d.1, .digit, 1 + Double(i), 1, 1, Character(d.0), d.2) }
        add("minus", "-", 0x1B, .symbol, 11, 1, 1, "-", "_")
        add("equal", "=", 0x18, .symbol, 12, 1, 1, "=", "+")
        add("delete", "delete", 0x33, .editing, 13, 1, 1.5)
        // Top letter row
        add("tab", "tab", 0x30, .whitespace, 0, 2, 1.5, "\t", nil)
        let top: [(String, UInt16)] = [("q", 0x0C), ("w", 0x0D), ("e", 0x0E), ("r", 0x0F), ("t", 0x11), ("y", 0x10),
                                       ("u", 0x20), ("i", 0x22), ("o", 0x1F), ("p", 0x23)]
        for (i, k) in top.enumerated() { add(k.0, k.0.uppercased(), k.1, .letter, 1.5 + Double(i), 2, 1, Character(k.0), Character(k.0.uppercased())) }
        add("lbracket", "[", 0x21, .symbol, 11.5, 2, 1, "[", "{")
        add("rbracket", "]", 0x1E, .symbol, 12.5, 2, 1, "]", "}")
        add("backslash", "\\", 0x2A, .symbol, 13.5, 2, 1, "\\", "|")
        // Home row
        add("capslock", "caps", 0x39, .modifier, 0, 3, 1.75)
        let home: [(String, UInt16)] = [("a", 0x00), ("s", 0x01), ("d", 0x02), ("f", 0x03), ("g", 0x05), ("h", 0x04),
                                        ("j", 0x26), ("k", 0x28), ("l", 0x25)]
        for (i, k) in home.enumerated() { add(k.0, k.0.uppercased(), k.1, .letter, 1.75 + Double(i), 3, 1, Character(k.0), Character(k.0.uppercased())) }
        add("semicolon", ";", 0x29, .symbol, 10.75, 3, 1, ";", ":")
        add("quote", "'", 0x27, .symbol, 11.75, 3, 1, "'", "\"")
        add("return", "return", 0x24, .whitespace, 12.75, 3, 1.75, "\n", nil)
        // Bottom letter row
        add("shift", "shift", 0x38, .modifier, 0, 4, 2.25)
        let bottom: [(String, UInt16)] = [("z", 0x06), ("x", 0x07), ("c", 0x08), ("v", 0x09), ("b", 0x0B), ("n", 0x2D), ("m", 0x2E)]
        for (i, k) in bottom.enumerated() { add(k.0, k.0.uppercased(), k.1, .letter, 2.25 + Double(i), 4, 1, Character(k.0), Character(k.0.uppercased())) }
        add("comma", ",", 0x2B, .symbol, 9.25, 4, 1, ",", "<")
        add("period", ".", 0x2F, .symbol, 10.25, 4, 1, ".", ">")
        add("slash", "/", 0x2C, .symbol, 11.25, 4, 1, "/", "?")
        add("rshift", "shift", 0x3C, .modifier, 12.25, 4, 2.25)
        // Bottom row
        add("fn", "fn", 0x3F, .modifier, 0, 5, 1)
        add("control", "control", 0x3B, .modifier, 1, 5, 1)
        add("option", "option", 0x3A, .modifier, 2, 5, 1)
        add("command", "command", 0x37, .modifier, 3, 5, 1.25)
        add("space", "space", 0x31, .whitespace, 4.25, 5, 5, " ", " ")
        add("rcommand", "command", 0x36, .modifier, 9.25, 5, 1.25)
        add("roption", "option", 0x3D, .modifier, 10.5, 5, 1)
        add("left", "←", 0x7B, .navigation, 11.5, 5.5, 1)
        add("up", "↑", 0x7E, .navigation, 12.5, 5, 1)
        add("down", "↓", 0x7D, .navigation, 12.5, 5.5, 1)
        add("right", "→", 0x7C, .navigation, 13.5, 5.5, 1)
        // Keys without a slot on the compact layout (shown in a side column by the HUD).
        add("rcontrol", "control", 0x3E, .modifier, 14.5, 5, 1)
        add("forwarddelete", "del", 0x75, .editing, 14.5, 1, 1)
        add("home", "home", 0x73, .navigation, 14.5, 2, 1)
        add("end", "end", 0x77, .navigation, 14.5, 3, 1)
        add("pageup", "pg up", 0x74, .navigation, 14.5, 4, 1)
        add("pagedown", "pg dn", 0x79, .navigation, 15.5, 4, 1)
        return keys
    }
}
