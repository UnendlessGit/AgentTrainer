import Foundation

public enum MouseMode: String, Codable, CaseIterable, Hashable, Sendable {
    /// Per-timestep mouse deltas (games with a locked cursor). Pixel2Play's mode.
    case relative
    /// Absolute pointer position on a grid over the captured area (apps with a visible cursor).
    case absolute
}

public enum MouseSampling: String, Codable, CaseIterable, Hashable, Sendable {
    /// Pixel2Play's default: a truncated normal inside the chosen bin (smoother control).
    case truncatedNormal
    /// The bin's representative value.
    case center
}

/// The model's discrete action vocabulary. Slot order per timestep is
/// `keys[keySlots], pointer[pointerSlots], mouseX, mouseY` exactly like Pixel2Play's
/// `keys[4], mouse_buttons[2], dx, dy`, so P2P weights map onto the first classes.
public struct ActionSpaceConfig: Codable, Hashable, Sendable {
    /// Key ids (see `KeyCatalog`) in class order. Class 0 means "no key".
    public var keys: [String]
    public var keySlots: Int
    /// Pointer events in class order. Class 0 means "none".
    public var pointerEvents: [PointerEvent]
    public var pointerSlots: Int
    public var mouseMode: MouseMode
    /// Bucket edges (torch.bucketize semantics, right=False) for relative mode, in P2P units.
    public var mouseXEdges: [Double]
    public var mouseYEdges: [Double]
    public var absoluteColumns: Int
    public var absoluteRows: Int
    /// Converts recorded pointer deltas per timestep into model units (P2P's bins were fit to
    /// Windows raw counts per 50 ms). Measured from recordings; inverted at inference.
    public var mouseUnitScale: Double
    public var mouseXStd: Double
    public var mouseYStd: Double
    public var mouseSampling: MouseSampling
    /// Stored as an optional so models saved before this setting existed still load (nil = on).
    public var mouseMovement: Bool?
    /// Whether the model moves the mouse. Off: it learns "no movement" and never moves it, while
    /// its keys and buttons work as usual (the mouse heads stay, so weights keep their shape).
    public var movesMouse: Bool {
        get { mouseMovement != false }
        set { mouseMovement = newValue }
    }

    public static let p2pXEdges: [Double] = [-500.1, -144.1, -76.1, -46.1, -29.1, -18.1, -11.1, -6.1, -3.1, -1.1, -0.1, 0,
                                             1, 3, 6, 11, 18, 29, 46, 76, 144, 500]
    public static let p2pYEdges: [Double] = [-150.1, -24.1, -12.1, -7.1, -4.1, -3.1, -1.1, -0.1, 0, 1, 3, 4, 7, 12, 24, 150]

    public init(keys: [String], keySlots: Int = 4, pointerEvents: [PointerEvent] = PointerEvent.defaultOrder,
                pointerSlots: Int = 2, mouseMode: MouseMode = .relative,
                mouseXEdges: [Double] = ActionSpaceConfig.p2pXEdges, mouseYEdges: [Double] = ActionSpaceConfig.p2pYEdges,
                absoluteColumns: Int = 128, absoluteRows: Int = 72, mouseUnitScale: Double = 1,
                mouseXStd: Double = 96, mouseYStd: Double = 22, mouseSampling: MouseSampling = .truncatedNormal) {
        self.keys = keys
        self.keySlots = keySlots
        self.pointerEvents = pointerEvents
        self.pointerSlots = pointerSlots
        self.mouseMode = mouseMode
        self.mouseXEdges = mouseXEdges
        self.mouseYEdges = mouseYEdges
        self.absoluteColumns = absoluteColumns
        self.absoluteRows = absoluteRows
        self.mouseUnitScale = mouseUnitScale
        self.mouseXStd = mouseXStd
        self.mouseYStd = mouseYStd
        self.mouseSampling = mouseSampling
    }

    /// Exactly Pixel2Play's action space (19 keys, L/R/M buttons, P2P mouse bins).
    public static var p2p: ActionSpaceConfig {
        ActionSpaceConfig(keys: KeyCatalog.p2pKeys, pointerEvents: PointerEvent.p2pOrder)
    }

    /// P2P's vocabulary first (so pretrained rows transfer) plus common game keys and scroll.
    public static var gameDefault: ActionSpaceConfig {
        ActionSpaceConfig(keys: KeyCatalog.p2pKeys + KeyCatalog.gameExtras)
    }

    /// Full keyboard, absolute pointer: for controlling desktop apps and typing.
    public static var appDefault: ActionSpaceConfig {
        let rest = KeyCatalog.fullKeyboard.filter { !KeyCatalog.p2pKeys.contains($0) }
        return ActionSpaceConfig(keys: KeyCatalog.p2pKeys + rest, keySlots: 4, mouseMode: .absolute)
    }

    // MARK: Sizes

    public var slotCount: Int { keySlots + pointerSlots + 2 }
    public var keyClasses: Int { keys.count + 1 }
    public var pointerClasses: Int { pointerEvents.count + 1 }
    /// Pixel2Play divides each head's cross-entropy by log(#classes). A head with a single class
    /// (e.g. no mouse buttons selected) is always right and contributes 0 instead of 0 / log(1).
    public static func lossScale(classes: Int) -> Double { classes > 1 ? 1 / Foundation.log(Double(classes)) : 0 }
    public var xClasses: Int { mouseMode == .relative ? mouseXEdges.count + 1 : absoluteColumns }
    public var yClasses: Int { mouseMode == .relative ? mouseYEdges.count + 1 : absoluteRows }
    public var zeroXClass: Int { mouseMode == .relative ? bucketize(0, mouseXEdges) : 0 }
    public var zeroYClass: Int { mouseMode == .relative ? bucketize(0, mouseYEdges) : 0 }

    /// Classes for each slot, in slot order.
    public var slotClassCounts: [Int] {
        Array(repeating: keyClasses, count: keySlots) + Array(repeating: pointerClasses, count: pointerSlots) + [xClasses, yClasses]
    }

    public enum SlotKind: Sendable { case key, pointer, x, y }
    public func kind(ofSlot s: Int) -> SlotKind {
        if s < keySlots { return .key }
        if s < keySlots + pointerSlots { return .pointer }
        return s == keySlots + pointerSlots ? .x : .y
    }

    public var keyIndex: [String: Int] { Dictionary(uniqueKeysWithValues: keys.enumerated().map { ($1, $0 + 1) }) }
    public var pointerIndex: [PointerEvent: Int] { Dictionary(uniqueKeysWithValues: pointerEvents.enumerated().map { ($1, $0 + 1) }) }

    // MARK: Encoding

    /// Encodes one timestep into class labels, one per slot. `dx`/`dy` are recorded pointer
    /// deltas (points) for relative mode; `absX`/`absY` are normalized [0,1] cursor coordinates
    /// in the captured area for absolute mode.
    public func encode(_ step: TimestepAction) -> [Int32] {
        var out = [Int32]()
        out.reserveCapacity(slotCount)
        let kIdx = keyIndex
        let keyClassesSorted = step.keys.compactMap { kIdx[$0] }.sorted()
        for s in 0 ..< keySlots { out.append(s < keyClassesSorted.count ? Int32(keyClassesSorted[s]) : 0) }
        let pIdx = pointerIndex
        let ptr = step.pointer.compactMap { pIdx[$0] }.sorted()
        for s in 0 ..< pointerSlots { out.append(s < ptr.count ? Int32(ptr[s]) : 0) }
        if !movesMouse {
            out.append(Int32(zeroXClass))
            out.append(Int32(zeroYClass))
            return out
        }
        switch mouseMode {
        case .relative:
            out.append(Int32(bucketize(step.dx * mouseUnitScale, mouseXEdges)))
            out.append(Int32(bucketize(step.dy * mouseUnitScale, mouseYEdges)))
        case .absolute:
            out.append(Int32(gridIndex(step.absX, absoluteColumns)))
            out.append(Int32(gridIndex(step.absY, absoluteRows)))
        }
        return out
    }

    /// Decodes sampled classes into an executable action.
    public func decode(_ classes: [Int32], sampleMouse: (Double, Double, Double) -> Double = ActionSpaceConfig.truncatedNormal) -> TimestepAction {
        precondition(classes.count == slotCount)
        var a = TimestepAction()
        for s in 0 ..< keySlots where classes[s] > 0 && Int(classes[s]) <= keys.count {
            a.keys.insert(keys[Int(classes[s]) - 1])
        }
        for s in keySlots ..< keySlots + pointerSlots where classes[s] > 0 && Int(classes[s]) <= pointerEvents.count {
            a.pointer.insert(pointerEvents[Int(classes[s]) - 1])
        }
        let cx = Int(classes[keySlots + pointerSlots]), cy = Int(classes[keySlots + pointerSlots + 1])
        if !movesMouse { return a }
        switch mouseMode {
        case .relative:
            a.dx = binValue(cx, edges: mouseXEdges, std: mouseXStd, sample: sampleMouse) / mouseUnitScale
            a.dy = binValue(cy, edges: mouseYEdges, std: mouseYStd, sample: sampleMouse) / mouseUnitScale
        case .absolute:
            a.absX = (Double(cx) + 0.5) / Double(absoluteColumns)
            a.absY = (Double(cy) + 0.5) / Double(absoluteRows)
        }
        return a
    }

    private func gridIndex(_ v: Double, _ n: Int) -> Int { max(0, min(n - 1, Int((v * Double(n)).rounded(.down)))) }

    /// Value for a relative-mode bin: 0 for the zero bin, ±(outer edge + 1) for the outer bins
    /// (P2P's ±501 / ±151), otherwise a sample inside the bin's integer range.
    func binValue(_ bin: Int, edges: [Double], std: Double, sample: (Double, Double, Double) -> Double) -> Double {
        if bin <= 0 { return (edges.first!.rounded(.down)) - 0 }
        if bin >= edges.count { return edges.last!.rounded(.down) + 1 }
        let lo = edges[bin - 1].rounded(.down) + 1, hi = edges[bin].rounded(.down)
        if lo >= hi { return hi }
        switch mouseSampling {
        case .center: return ((lo + hi) / 2).rounded()
        case .truncatedNormal: return min(hi, max(lo, sample(lo, hi, std).rounded()))
        }
    }

    /// Samples N(0, std) truncated to [lo, hi] by inverse CDF.
    public static func truncatedNormal(lo: Double, hi: Double, std: Double) -> Double {
        func cdf(_ x: Double) -> Double { 0.5 * (1 + erf(x / (std * 2.0.squareRoot()))) }
        let a = cdf(lo - 0.5), b = cdf(hi + 0.5)
        let u = a + Double.random(in: 0 ..< 1) * max(b - a, 1e-12)
        return std * 2.0.squareRoot() * inverseErf(2 * u - 1)
    }
}

/// `torch.bucketize(v, edges, right=False)`: the number of edges strictly below `v`.
public func bucketize(_ v: Double, _ edges: [Double]) -> Int {
    var lo = 0, hi = edges.count
    while lo < hi {
        let mid = (lo + hi) / 2
        if edges[mid] < v { lo = mid + 1 } else { hi = mid }
    }
    return lo
}

/// Inverse error function (Giles' approximation, refined with two Newton steps).
public func inverseErf(_ x: Double) -> Double {
    let x = max(-0.999999999, min(0.999999999, x))
    var w = -log((1 - x) * (1 + x)), p: Double
    if w < 5 {
        w -= 2.5
        p = 2.81022636e-08
        for c in [3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087, -0.00125372503, -0.00417768164, 0.246640727, 1.50140941] { p = c + p * w }
    } else {
        w = w.squareRoot() - 3
        p = -0.000200214257
        for c in [0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773, -0.0076224613, 0.00943887047, 1.00167406, 2.83297682] { p = c + p * w }
    }
    var y = p * x
    for _ in 0 ..< 2 { y -= (erf(y) - x) / (2 / Double.pi.squareRoot() * exp(-y * y)) }
    return y
}

/// One timestep of human or model input, before discretization.
public struct TimestepAction: Hashable, Sendable, Codable {
    public var keys: Set<String> = []
    public var pointer: Set<PointerEvent> = []
    /// Relative pointer movement during the step (points, y down).
    public var dx: Double = 0
    public var dy: Double = 0
    /// Absolute pointer position at the end of the step, normalized to the capture area.
    public var absX: Double = 0.5
    public var absY: Double = 0.5
    public init() {}
}
