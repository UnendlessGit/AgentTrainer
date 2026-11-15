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
        .target(name: "ATLanguage", dependencies: ["ATCore", "ATModel"]),
        .target(name: "ATTraining", dependencies: ["ATCore", "ATModel", "ATLanguage"]),
        .target(name: "ATRuntime", dependencies: ["ATCore", "ATModel", "ATCapture", "ATLanguage"]),
        .executableTarget(name: "atctl", dependencies: ["ATCore", "ATModel", "ATTraining", "ATLanguage"]),
        .testTarget(name: "ATCoreTests", dependencies: ["ATCore"]),
        .testTarget(name: "ATModelTests", dependencies: ["ATModel", "ATCore"], resources: [.copy("Fixtures")]),
    ],
    swiftLanguageModes: [.v5]
)
