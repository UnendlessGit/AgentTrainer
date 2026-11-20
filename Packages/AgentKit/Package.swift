// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "AgentKit",
    platforms: [.macOS("26.0")],
    products: [
        .library(name: "ATCore", targets: ["ATCore"]),
        .library(name: "ATModel", targets: ["ATModel"]),
        .library(name: "ATCapture", targets: ["ATCapture"]),
        .library(name: "ATTraining", targets: ["ATTraining"]),
        .library(name: "ATLanguage", targets: ["ATLanguage"]),
        .library(name: "ATRuntime", targets: ["ATRuntime"]),
        .executable(name: "atctl", targets: ["atctl"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.33.0")),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", revision: "6352907543270ae9c61fcaf59cb3e07b0b989ea1", traits: []),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.4.0")),
    ],
    targets: [
        .target(name: "ATCore"),
        .target(
            name: "ATModel",
            dependencies: [
                "ATCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ]),
        .target(name: "ATCapture", dependencies: ["ATCore"]),
        .target(
            name: "ATLanguage",
            dependencies: [
                "ATCore", "ATModel",
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]),
        .target(name: "ATTraining", dependencies: ["ATCore", "ATModel", "ATLanguage", "ATCapture"]),
        .target(name: "ATRuntime", dependencies: ["ATCore", "ATModel", "ATCapture", "ATLanguage"]),
        .executableTarget(name: "atctl", dependencies: ["ATCore", "ATModel", "ATTraining", "ATLanguage", "ATCapture", "ATRuntime"]),
        .testTarget(name: "ATCoreTests", dependencies: ["ATCore"]),
        .testTarget(name: "ATCaptureTests", dependencies: ["ATCapture", "ATCore"]),
        .testTarget(name: "ATRuntimeTests", dependencies: ["ATRuntime", "ATCapture", "ATCore"]),
        .testTarget(name: "ATModelTests", dependencies: ["ATModel", "ATCore"], resources: [.copy("Fixtures")]),
        .testTarget(name: "ATTrainingTests", dependencies: ["ATTraining", "ATModel", "ATCore", .product(name: "MLXOptimizers", package: "mlx-swift")]),
    ],
    swiftLanguageModes: [.v5]
)
