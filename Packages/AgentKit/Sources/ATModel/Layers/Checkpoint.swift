import Cmlx
import Foundation
import MLX
import MLXNN

/// Gradient checkpointing (`mlx::core::checkpoint`): intermediates of the wrapped function are
/// not kept for the backward pass; they are recomputed. Trades ~30% extra compute for a large
/// memory saving on long clips.
final class CheckpointedFunction {
    private var closure: mlx_closure

    init(_ f: @escaping ([MLXArray]) -> [MLXArray]) {
        let inner = Self.makeClosure(f)
        var res = mlx_closure_new()
        mlx_checkpoint(&res, inner)
        mlx_closure_free(inner)
        closure = res
    }

    deinit { mlx_closure_free(closure) }

    func callAsFunction(_ args: [MLXArray]) -> [MLXArray] {
        let input = withExtendedLifetime(args) { mlx_vector_array_new_data(args.map(\.ctx), args.count) }
        defer { mlx_vector_array_free(input) }
        var out = mlx_vector_array_new()
        let status = mlx_closure_apply(&out, closure, input)
        precondition(status == 0, "checkpointed function failed")
        defer { mlx_vector_array_free(out) }
        return Self.values(out)
    }

    static func values(_ v: mlx_vector_array) -> [MLXArray] {
        (0 ..< mlx_vector_array_size(v)).map { i in
            var a = mlx_array_new()
            mlx_vector_array_get(&a, v, i)
            return MLXArray(a)
        }
    }

    private static func makeClosure(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> mlx_closure {
        let payload = Unmanaged.passRetained(CheckpointBox(f)).toOpaque()
        return mlx_closure_new_func_payload(checkpointTrampoline, payload, checkpointFree)
    }
}

private final class CheckpointBox {
    let f: ([MLXArray]) -> [MLXArray]
    init(_ f: @escaping ([MLXArray]) -> [MLXArray]) { self.f = f }
}

private func checkpointTrampoline(out: UnsafeMutablePointer<mlx_vector_array>?, input: mlx_vector_array, payload: UnsafeMutableRawPointer?) -> Int32 {
    let box = Unmanaged<CheckpointBox>.fromOpaque(payload!).takeUnretainedValue()
    let result = box.f(CheckpointedFunction.values(input))
    out?.pointee = withExtendedLifetime(result) { mlx_vector_array_new_data(result.map(\.ctx), result.count) }
    return 0
}

private func checkpointFree(ptr: UnsafeMutableRawPointer?) {
    Unmanaged<CheckpointBox>.fromOpaque(ptr!).release()
}

/// Checkpoints a module call. The module's trainable parameters are passed as explicit inputs
/// (like `mlx.nn.utils.checkpoint`) so their gradients flow through the recomputation.
final class ModuleCheckpoint<M: Module> {
    let module: M
    private var fn: CheckpointedFunction?
    private var keys: [String] = []
    private let body: (M, [MLXArray]) -> [MLXArray]

    init(_ module: M, _ body: @escaping (M, [MLXArray]) -> [MLXArray]) {
        self.module = module
        self.body = body
    }

    func callAsFunction(_ args: [MLXArray]) -> [MLXArray] {
        let params = module.trainableParameters().flattened()
        if fn == nil || params.map(\.0) != keys {
            keys = params.map(\.0)
            let keys = keys, module = module, body = body
            fn = CheckpointedFunction { inputs in
                module.update(parameters: ModuleParameters.unflattened(Array(zip(keys, inputs[0 ..< keys.count]))))
                return body(module, Array(inputs[keys.count...]))
            }
        }
        return fn!(params.map(\.1) + args)
    }
}
