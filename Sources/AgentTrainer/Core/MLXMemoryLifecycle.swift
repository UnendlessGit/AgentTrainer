import Foundation
import MLX

/// Process-wide MLX allocator policy. Apple-silicon GPU allocations use the
/// same unified memory as macOS, WindowServer, and the HID/input stack, so the
/// library default (whose cache may grow to the full memory limit) is not a
/// safe desktop-app policy.
enum MLXMemoryLifecycle {
    struct Limits: Equatable, Sendable {
        let memory: Int
        let cache: Int
    }

    private static let operationLock = NSLock()
    nonisolated(unsafe) private static var configured = false

    static func limits(forPhysicalMemory physicalMemory: Int) -> Limits {
        let gibibyte = 1 << 30
        let mebibyte = 1 << 20
        let physical = max(gibibyte, physicalMemory)
        let percentageReserve = Int(Double(physical) * 0.15)
        let reservedForSystem = max(2 * gibibyte, percentageReserve)
        let memory = max(gibibyte, physical - reservedForSystem)
        let cache = max(512 * mebibyte, min(2 * gibibyte, Int(Double(physical) * 0.06)))
        return Limits(memory: memory, cache: min(memory, cache))
    }

    static func configure(physicalMemory: Int = Int(ProcessInfo.processInfo.physicalMemory)) {
        operationLock.lock()
        defer { operationLock.unlock() }
        guard !configured else { return }
        let limits = limits(forPhysicalMemory: physicalMemory)
        Memory.memoryLimit = limits.memory
        Memory.cacheLimit = limits.cache
        configured = true
        AppLog.write(
            category: "Memory",
            "Configured bounded MLX unified-memory allocator",
            details: "memory limit \(formatted(limits.memory)), reusable cache limit \(formatted(limits.cache))"
        )
    }

    /// Call only after session-owned models and compiled closures have been
    /// released. Active tensors belonging to a simultaneous training/run task
    /// remain valid; MLX discards only currently reusable allocator buffers.
    static func reclaimCaches(after context: String) {
        operationLock.lock()
        let before = Memory.snapshot()
        Memory.clearCache()
        let after = Memory.snapshot()
        operationLock.unlock()

        guard before.cacheMemory > 0 || before.activeMemory > 0 else { return }
        AppLog.write(
            category: "Memory",
            "Reclaimed MLX cache after \(context)",
            details: "cache \(formatted(before.cacheMemory)) → \(formatted(after.cacheMemory)); active \(formatted(after.activeMemory))"
        )
    }

    private static func formatted(_ bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }
}
