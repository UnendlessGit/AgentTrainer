import ATCore
import Foundation
import Metal

/// Keeps the GPU clocked up between real-time steps. Apple GPUs lower their clock when they are
/// only partly busy (a 30 Hz model leaves the GPU idle ~55% of the time), which makes each step
/// 2-3× slower. After a step, `fill(until:)` runs a tiny single-threadgroup spin kernel that ends
/// just before the next tick, so the GPU looks busy to the power controller without competing
/// with the model.
public final class GPUWarmer: @unchecked Sendable {
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private var nsPerIteration = 50.0
    private var pending: MTLCommandBuffer?

    public init?(device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        guard let device, let q = device.makeCommandQueue() else { return nil }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void spin(device float *out [[buffer(0)]], constant uint &n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
            float x = float(tid);
            for (uint i = 0; i < n; i++) { x = fma(x, 1.0000001f, 0.0000001f); }
            if (x == 12345.0f) { out[0] = x; }
        }
        """
        guard let lib = try? device.makeLibrary(source: src, options: nil), let f = lib.makeFunction(name: "spin"),
              let p = try? device.makeComputePipelineState(function: f) else { return nil }
        queue = q
        pipeline = p
        out = device.makeBuffer(length: 16, options: .storageModeShared)!
    }

    private let out: MTLBuffer

    /// Waits for the previous fill to finish (so it never overlaps a step).
    public func finish() {
        pending?.waitUntilCompleted()
        pending = nil
    }

    /// Busies the GPU until shortly before `deadlineNs` (host clock).
    public func fill(until deadlineNs: UInt64, margin: Double = 2.5e6) {
        let now = HostClock.now()
        guard deadlineNs > now + UInt64(margin) + 1_000_000 else { return }
        let budget = Double(deadlineNs - now) - margin
        var n = UInt32(max(1000, min(50_000_000, budget / nsPerIteration)))
        guard let cmd = queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(out, offset: 0, index: 0)
        enc.setBytes(&n, length: 4, index: 1)
        enc.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cmd.addCompletedHandler { [weak self] c in
            guard let self, c.gpuEndTime > c.gpuStartTime else { return }
            let measured = (c.gpuEndTime - c.gpuStartTime) * 1e9 / Double(n)
            // Smoothly track how long one iteration takes at the current clock.
            self.nsPerIteration = self.nsPerIteration * 0.7 + measured * 0.3
        }
        cmd.commit()
        pending = cmd
    }
}
