// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "AgentTrainer",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "AgentTrainer", targets: ["AgentTrainer"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.31.3")
    ],
    targets: [
        .executableTarget(
            name: "AgentTrainer",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ],
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("SwiftUI"),
                .linkedFramework("ScreenCaptureKit"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("AVKit"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("VideoToolbox"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("Accelerate"),
                .linkedFramework("Carbon"),
                .linkedFramework("IOKit")
            ]
        ),
        .testTarget(
            name: "AgentTrainerTests",
            dependencies: ["AgentTrainer"]
        )
    ]
)
