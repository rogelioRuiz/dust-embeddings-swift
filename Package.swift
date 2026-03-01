// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "dust-embeddings-swift",
    platforms: [.iOS(.v16), .macOS(.v14)],
    products: [
        .library(name: "DustEmbeddings", targets: ["DustEmbeddings"])
    ],
    dependencies: [
        .package(url: "https://github.com/rogelioRuiz/dust-core-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/rogelioRuiz/dust-llm-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/rogelioRuiz/dust-onnx-swift.git", from: "0.1.0"),
    ],
    targets: [
        .target(
            name: "DustEmbeddings",
            dependencies: [
                .product(name: "DustCore", package: "dust-core-swift"),
                .product(name: "DustLlm", package: "dust-llm-swift"),
                .product(name: "DustOnnx", package: "dust-onnx-swift"),
            ],
            path: "Sources/DustEmbeddings"
        ),
        .testTarget(
            name: "DustEmbeddingsTests",
            dependencies: [
                "DustEmbeddings",
                .product(name: "DustCore", package: "dust-core-swift"),
                .product(name: "DustOnnx", package: "dust-onnx-swift"),
            ],
            path: "Tests/DustEmbeddingsTests",
            resources: [
                .copy("Fixtures"),
            ]
        ),
    ],
    swiftLanguageVersions: [.v5]
)
