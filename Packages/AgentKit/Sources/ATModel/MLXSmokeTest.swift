import Foundation
import MLX

/// Verifies that MLX can find its Metal library and run a kernel in the current process.
public enum MLXSmokeTest {
    public static func run() -> String {
        let a = MLXArray(0 ..< 16, [4, 4]).asType(.float32)
        let b = matmul(a, a.transposed())
        let s = b.sum().item(Float.self)
        let info = GPU.deviceInfo()
        return "MLX ok: sum=\(s) device=\(info.architecture) mem=\(info.memorySize / 1_073_741_824) GB"
    }
}
