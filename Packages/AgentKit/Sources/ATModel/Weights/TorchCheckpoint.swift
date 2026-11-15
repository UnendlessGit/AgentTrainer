import Foundation
import MLX

public enum TorchCheckpointError: Error, LocalizedError {
    case notAZip, missingEntry(String), compressedEntry(String), pickle(String), unsupportedDType(String)

    public var errorDescription: String? {
        switch self {
        case .notAZip: "The checkpoint is not a PyTorch zip archive."
        case .missingEntry(let n): "The checkpoint is missing \(n)."
        case .compressedEntry(let n): "Entry \(n) is compressed; PyTorch checkpoints store tensors uncompressed."
        case .pickle(let m): "Could not read the checkpoint index: \(m)"
        case .unsupportedDType(let t): "Unsupported tensor storage type \(t)."
        }
    }
}

/// A tensor described by a PyTorch checkpoint (not yet loaded).
public struct TorchTensorRef: Sendable {
    public let storageKey: String
    public let dtype: DType
    public let elementOffset: Int
    public let shape: [Int]
    public let stride: [Int]

    var isContiguous: Bool {
        var expected = 1
        for (d, s) in zip(shape, stride).reversed() {
            if d != 1 && s != expected { return false }
            expected *= d
        }
        return true
    }
}

/// Reads PyTorch `torch.save` zip checkpoints (including Lightning `.ckpt`) without Python.
///
/// The pickle stream is interpreted by a restricted VM: only tensor/storage reconstructors and
/// plain containers are honoured; every other global becomes an inert placeholder, so loading
/// a checkpoint can never execute code.
public final class TorchCheckpointReader {
    private let data: Data
    private var entries: [String: (offset: Int, size: Int)] = [:]
    private var prefix = ""

    public init(url: URL) throws {
        data = try Data(contentsOf: url, options: .alwaysMapped)
        try parseZip()
        guard let pkl = entries.keys.first(where: { $0.hasSuffix("/data.pkl") || $0 == "data.pkl" }) else {
            throw TorchCheckpointError.missingEntry("data.pkl")
        }
        prefix = String(pkl.dropLast("data.pkl".count))
    }

    // MARK: Zip

    private func u16(_ o: Int) -> Int { Int(data[data.startIndex + o]) | Int(data[data.startIndex + o + 1]) << 8 }
    private func u32(_ o: Int) -> Int { u16(o) | u16(o + 2) << 16 }
    private func u64(_ o: Int) -> Int { u32(o) | u32(o + 4) << 32 }

    private func parseZip() throws {
        let n = data.count
        guard n > 22 else { throw TorchCheckpointError.notAZip }
        var eocd = -1
        var i = n - 22
        while i >= max(0, n - 65_557) {
            if u32(i) == 0x0605_4B50 { eocd = i; break }
            i -= 1
        }
        guard eocd >= 0 else { throw TorchCheckpointError.notAZip }
        var count = u16(eocd + 10)
        var cdOffset = u32(eocd + 16)
        if (count == 0xFFFF || cdOffset == 0xFFFF_FFFF), eocd >= 20, u32(eocd - 20) == 0x0706_4B50 {
            let z64 = u64(eocd - 20 + 8)
            guard u32(z64) == 0x0606_4B50 else { throw TorchCheckpointError.notAZip }
            count = u64(z64 + 32)
            cdOffset = u64(z64 + 48)
        }
        var p = cdOffset
        for _ in 0 ..< count {
            guard u32(p) == 0x0201_4B50 else { throw TorchCheckpointError.notAZip }
            let method = u16(p + 10)
            var size = u32(p + 24)
            let nameLen = u16(p + 28), extraLen = u16(p + 30), commentLen = u16(p + 32)
            var local = u32(p + 42)
            let name = String(decoding: data[(data.startIndex + p + 46) ..< (data.startIndex + p + 46 + nameLen)], as: UTF8.self)
            // Zip64 extra field: values present only for fields saturated at 0xFFFFFFFF.
            var e = p + 46 + nameLen
            let eEnd = e + extraLen
            while e + 4 <= eEnd {
                let id = u16(e), len = u16(e + 2)
                if id == 0x0001 {
                    var q = e + 4
                    if u32(p + 24) == 0xFFFF_FFFF { size = u64(q); q += 8 }
                    if u32(p + 20) == 0xFFFF_FFFF { q += 8 }
                    if u32(p + 42) == 0xFFFF_FFFF { local = u64(q) }
                }
                e += 4 + len
            }
            if method != 0 && size > 0 { throw TorchCheckpointError.compressedEntry(name) }
            let lName = u16(local + 26), lExtra = u16(local + 28)
            entries[name] = (local + 30 + lName + lExtra, size)
            p += 46 + nameLen + extraLen + commentLen
        }
    }

    private func entry(_ name: String) throws -> Data {
        guard let e = entries[prefix + name] else { throw TorchCheckpointError.missingEntry(name) }
        return data.subdata(in: (data.startIndex + e.offset) ..< (data.startIndex + e.offset + e.size))
    }

    // MARK: Tensors

    /// The model weights: `state_dict` for Lightning checkpoints, otherwise the top-level dict.
    public func stateDict() throws -> [String: TorchTensorRef] {
        var unpickler = Unpickler(data: try entry("data.pkl"))
        let root = try unpickler.load()
        var dict: [String: PickleValue] = [:]
        if case .dict(let d) = root {
            if case .dict(let sd)? = d.items["state_dict"] { dict = sd.items } else { dict = d.items }
        }
        var out: [String: TorchTensorRef] = [:]
        for (k, v) in dict { if case .tensor(let t) = v { out[k] = t } }
        guard !out.isEmpty else { throw TorchCheckpointError.pickle("no tensors found") }
        return out
    }

    /// Loads one tensor as a contiguous MLX array of its stored dtype.
    public func load(_ t: TorchTensorRef) throws -> MLXArray {
        let raw = try entry("data/\(t.storageKey)")
        let count = t.shape.reduce(1, *)
        if t.isContiguous {
            let start = t.elementOffset * t.dtype.size
            let bytes = raw.subdata(in: (raw.startIndex + start) ..< (raw.startIndex + start + count * t.dtype.size))
            return MLXArray(bytes, t.shape.isEmpty ? [1] : t.shape, dtype: t.dtype).reshaped(t.shape)
        }
        let all = MLXArray(raw, [raw.count / t.dtype.size], dtype: t.dtype)
        return asStrided(all, t.shape, strides: t.stride, offset: t.elementOffset)
    }
}

// MARK: - Restricted pickle VM

final class PickleDict { var items: [String: PickleValue] = [:] }
final class PickleList { var items: [PickleValue] = [] }

indirect enum PickleValue {
    case none, bool(Bool), int(Int), float(Double), string(String), bytes(Data)
    case tuple([PickleValue]), list(PickleList), dict(PickleDict)
    case global(String), storage(key: String, dtype: DType), tensor(TorchTensorRef), opaque(String)
}

struct Unpickler {
    let data: Data
    private var pos = 0
    private var stack: [PickleValue] = []
    private var marks: [Int] = []
    private var memo: [Int: PickleValue] = [:]

    init(data: Data) { self.data = data }

    private mutating func byte() throws -> UInt8 {
        guard pos < data.count else { throw TorchCheckpointError.pickle("unexpected end") }
        defer { pos += 1 }
        return data[data.startIndex + pos]
    }

    private mutating func bytes(_ n: Int) throws -> Data {
        guard pos + n <= data.count else { throw TorchCheckpointError.pickle("unexpected end") }
        defer { pos += n }
        return data.subdata(in: (data.startIndex + pos) ..< (data.startIndex + pos + n))
    }

    private mutating func uint(_ n: Int) throws -> Int {
        var v = 0
        for i in 0 ..< n { v |= Int(try byte()) << (8 * i) }
        return v
    }

    private mutating func line() throws -> String {
        var s = Data()
        while true {
            let b = try byte()
            if b == 0x0A { break }
            s.append(b)
        }
        return String(decoding: s, as: UTF8.self)
    }

    private mutating func pop() throws -> PickleValue {
        guard let v = stack.popLast() else { throw TorchCheckpointError.pickle("stack underflow") }
        return v
    }

    private mutating func popMark() throws -> [PickleValue] {
        guard let m = marks.popLast() else { throw TorchCheckpointError.pickle("mark underflow") }
        let items = Array(stack[m...])
        stack.removeSubrange(m...)
        return items
    }

    private static func key(_ v: PickleValue) -> String {
        switch v {
        case .string(let s): s
        case .int(let i): String(i)
        default: "\(v)"
        }
    }

    private static func dtype(forStorage name: String) throws -> DType {
        switch name.split(separator: ".").last.map(String.init) ?? name {
        case "FloatStorage": .float32
        case "BFloat16Storage": .bfloat16
        case "HalfStorage": .float16
        case "DoubleStorage": .float64
        case "LongStorage": .int64
        case "IntStorage": .int32
        case "ShortStorage": .int16
        case "CharStorage": .int8
        case "ByteStorage": .uint8
        case "BoolStorage": .bool
        default: throw TorchCheckpointError.unsupportedDType(name)
        }
    }

    private static func ints(_ v: PickleValue) -> [Int] {
        if case .tuple(let t) = v { return t.compactMap { if case .int(let i) = $0 { i } else { nil } } }
        return []
    }

    private func reduce(_ callable: PickleValue, _ args: PickleValue) throws -> PickleValue {
        guard case .global(let name) = callable else { return .opaque("call") }
        let a: [PickleValue] = { if case .tuple(let t) = args { t } else { [] } }()
        switch name {
        case "torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_tensor":
            guard a.count >= 4, case .storage(let key, let dtype) = a[0], case .int(let offset) = a[1] else {
                throw TorchCheckpointError.pickle("bad tensor record")
            }
            return .tensor(TorchTensorRef(storageKey: key, dtype: dtype, elementOffset: offset,
                                          shape: Self.ints(a[2]), stride: Self.ints(a[3])))
        case "torch._utils._rebuild_parameter", "torch._utils._rebuild_parameter_with_state":
            return a.first ?? .none
        case "collections.OrderedDict", "builtins.dict":
            return .dict(PickleDict())
        case "builtins.list":
            return .list(PickleList())
        default:
            return .opaque(name) // inert: never executed
        }
    }

    mutating func load() throws -> PickleValue {
        while true {
            let op = try byte()
            switch op {
            case 0x80: _ = try byte() // PROTO
            case 0x95: pos += 8 // FRAME
            case 0x2E: return try pop() // STOP
            case 0x28: marks.append(stack.count) // MARK
            case 0x30: _ = try pop() // POP
            case 0x31: _ = try popMark() // POP_MARK
            case 0x32: stack.append(stack.last ?? .none) // DUP
            case 0x4E: stack.append(.none)
            case 0x88: stack.append(.bool(true))
            case 0x89: stack.append(.bool(false))
            case 0x4A: stack.append(.int(Int(Int32(truncatingIfNeeded: try uint(4))))) // BININT
            case 0x4B: stack.append(.int(try uint(1))) // BININT1
            case 0x4D: stack.append(.int(try uint(2))) // BININT2
            case 0x8A, 0x8B: // LONG1 / LONG4
                let n = op == 0x8A ? try uint(1) : try uint(4)
                let b = try bytes(n)
                var v = 0
                for (i, x) in b.enumerated() where i < 8 { v |= Int(x) << (8 * i) }
                if n > 0, n < 8, let last = b.last, last & 0x80 != 0 { v -= 1 << (8 * n) }
                stack.append(.int(v))
            case 0x47: // BINFLOAT (big-endian)
                let b = try bytes(8)
                stack.append(.float(Double(bitPattern: b.reduce(UInt64(0)) { $0 << 8 | UInt64($1) })))
            case 0x58, 0x8C, 0x8D: // BINUNICODE / SHORT_BINUNICODE / BINUNICODE8
                let n = op == 0x58 ? try uint(4) : op == 0x8C ? try uint(1) : try uint(8)
                stack.append(.string(String(decoding: try bytes(n), as: UTF8.self)))
            case 0x54, 0x55, 0x42, 0x43: // BINSTRING / SHORT_BINSTRING / BINBYTES / SHORT_BINBYTES
                let n = (op == 0x54 || op == 0x42) ? try uint(4) : try uint(1)
                let b = try bytes(n)
                stack.append(op == 0x54 || op == 0x55 ? .string(String(decoding: b, as: UTF8.self)) : .bytes(b))
            case 0x29: stack.append(.tuple([]))
            case 0x74: stack.append(.tuple(try popMark()))
            case 0x85: let a = try pop(); stack.append(.tuple([a]))
            case 0x86: let b = try pop(), a = try pop(); stack.append(.tuple([a, b]))
            case 0x87: let c = try pop(), b = try pop(), a = try pop(); stack.append(.tuple([a, b, c]))
            case 0x5D: stack.append(.list(PickleList()))
            case 0x6C: let l = PickleList(); l.items = try popMark(); stack.append(.list(l))
            case 0x61: // APPEND
                let v = try pop()
                if case .list(let l) = stack.last { l.items.append(v) }
            case 0x65: // APPENDS
                let items = try popMark()
                if case .list(let l) = stack.last { l.items.append(contentsOf: items) }
            case 0x7D: stack.append(.dict(PickleDict()))
            case 0x64: // DICT
                let items = try popMark()
                let d = PickleDict()
                for i in stride(from: 0, to: items.count - 1, by: 2) { d.items[Self.key(items[i])] = items[i + 1] }
                stack.append(.dict(d))
            case 0x73: // SETITEM
                let v = try pop(), k = try pop()
                if case .dict(let d) = stack.last { d.items[Self.key(k)] = v }
            case 0x75: // SETITEMS
                let items = try popMark()
                if case .dict(let d) = stack.last {
                    for i in stride(from: 0, to: items.count - 1, by: 2) { d.items[Self.key(items[i])] = items[i + 1] }
                }
            case 0x8F: stack.append(.opaque("set")) // EMPTY_SET
            case 0x90: _ = try popMark() // ADDITEMS
            case 0x91: _ = try popMark(); stack.append(.opaque("frozenset"))
            case 0x63: // GLOBAL
                let module = try line(), name = try line()
                stack.append(.global("\(module).\(name)"))
            case 0x93: // STACK_GLOBAL
                let n = try pop(), m = try pop()
                if case .string(let ms) = m, case .string(let ns) = n { stack.append(.global("\(ms).\(ns)")) } else { stack.append(.opaque("global")) }
            case 0x52: // REDUCE
                let args = try pop(), callable = try pop()
                stack.append(try reduce(callable, args))
            case 0x81: // NEWOBJ
                let args = try pop(), cls = try pop()
                stack.append(try reduce(cls, args))
            case 0x92: // NEWOBJ_EX
                _ = try pop(); let args = try pop(), cls = try pop()
                stack.append(try reduce(cls, args))
            case 0x62: // BUILD
                let state = try pop()
                if case .dict(let d) = stack.last, case .dict(let s) = state {
                    for (k, v) in s.items { d.items[k] = v }
                }
            case 0x51: // BINPERSID
                let pid = try pop()
                guard case .tuple(let t) = pid, t.count >= 3 else { throw TorchCheckpointError.pickle("bad persistent id") }
                guard case .global(let storageType) = t[1] else { throw TorchCheckpointError.pickle("bad storage type") }
                let key: String = { if case .string(let s) = t[2] { s } else { Self.key(t[2]) } }()
                stack.append(.storage(key: key, dtype: try Self.dtype(forStorage: storageType)))
            case 0x71: memo[try uint(1)] = stack.last // BINPUT
            case 0x72: memo[try uint(4)] = stack.last // LONG_BINPUT
            case 0x94: memo[memo.count] = stack.last // MEMOIZE
            case 0x70: memo[Int(try line()) ?? -1] = stack.last // PUT
            case 0x68: stack.append(memo[try uint(1)] ?? .none) // BINGET
            case 0x6A: stack.append(memo[try uint(4)] ?? .none) // LONG_BINGET
            case 0x67: stack.append(memo[Int(try line()) ?? -1] ?? .none) // GET
            default:
                throw TorchCheckpointError.pickle(String(format: "unsupported opcode 0x%02x at %d", op, pos - 1))
            }
        }
    }
}
