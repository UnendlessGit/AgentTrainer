import Foundation
import Metal
import MLX

/// Supplies MLX inputs from Metal shared memory. `MLXArray(Data, ...)` first
/// copies every byte into MLX-owned storage; large temporal vision batches can
/// therefore cross host memory twice before the GPU reads them. These buffers
/// are populated directly from the mapped dataset and handed to MLX with an
/// ownership finalizer, which keeps the single shared allocation alive until
/// the graph has finished using it.
final class MetalArrayBufferPool: @unchecked Sendable {
    struct Descriptor: Sendable {
        let shape: [Int]
        let dtype: DType

        init(_ shape: [Int], dtype: DType) {
            self.shape = shape
            self.dtype = dtype
        }
    }

    private let device: any MTLDevice
    private let maximumCachedBytes: Int
    private let lock = NSLock()
    private var available: [any MTLBuffer] = []
    private var cachedBytes = 0

    init(maximumCachedBytes: Int = 128 * 1_024 * 1_024) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw AgentTrainerError.model("Metal is unavailable on this Mac.")
        }
        self.device = device
        self.maximumCachedBytes = max(0, maximumCachedBytes)
    }

    /// Allocates all inputs before invoking `populate`, then transfers each
    /// buffer to an MLX array without another host copy. The finalizers recycle
    /// only buffers whose combined retained size stays within the fixed cap.
    func makeArrays(
        _ descriptors: [Descriptor],
        populate: ([UnsafeMutableRawBufferPointer]) throws -> Void
    ) throws -> [MLXArray] {
        let byteCounts = try descriptors.map(Self.byteCount)
        var buffers: [any MTLBuffer] = []
        buffers.reserveCapacity(descriptors.count)
        do {
            for byteCount in byteCounts {
                guard byteCount > 0, let buffer = acquire(byteCount: byteCount) else {
                    throw AgentTrainerError.model("Metal could not allocate a shared MLX input buffer.")
                }
                buffers.append(buffer)
            }
            try populate(zip(buffers, byteCounts).map { buffer, byteCount in
                UnsafeMutableRawBufferPointer(start: buffer.contents(), count: byteCount)
            })
        } catch {
            buffers.forEach(recycle)
            throw error
        }

        return zip(buffers, descriptors).map { buffer, descriptor in
            MLXArray(
                rawPointer: buffer.contents(),
                descriptor.shape,
                dtype: descriptor.dtype
            ) { [self, buffer] in
                recycle(buffer)
            }
        }
    }

    private static func byteCount(for descriptor: Descriptor) throws -> Int {
        var elements = 1
        for dimension in descriptor.shape {
            guard dimension >= 0 else {
                throw AgentTrainerError.model("An MLX input has an invalid negative dimension.")
            }
            let product = elements.multipliedReportingOverflow(by: dimension)
            guard !product.overflow else {
                throw AgentTrainerError.model("An MLX input is too large for this Mac.")
            }
            elements = product.partialValue
        }
        let bytes = elements.multipliedReportingOverflow(by: descriptor.dtype.size)
        guard !bytes.overflow else {
            throw AgentTrainerError.model("An MLX input is too large for this Mac.")
        }
        return bytes.partialValue
    }

    private func acquire(byteCount: Int) -> (any MTLBuffer)? {
        if let reused = lock.withLock({ () -> (any MTLBuffer)? in
            let candidates = available.indices.filter { available[$0].length >= byteCount }
            guard let index = candidates.min(by: { available[$0].length < available[$1].length }) else {
                return nil
            }
            let buffer = available.remove(at: index)
            cachedBytes -= buffer.length
            return buffer
        }) {
            return reused
        }
        return device.makeBuffer(length: byteCount, options: .storageModeShared)
    }

    private func recycle(_ buffer: any MTLBuffer) {
        lock.withLock {
            guard buffer.length <= maximumCachedBytes,
                  cachedBytes <= maximumCachedBytes - buffer.length else { return }
            available.append(buffer)
            cachedBytes += buffer.length
        }
    }
}
