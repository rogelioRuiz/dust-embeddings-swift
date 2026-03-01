// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "dust-embeddings-swift",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "DustEmbeddings", targets: ["DustEmbeddings"])
    ],
    dependencies: [
        .package(name: "dust-core-swift", path: "../dust-core-swift"),
        .package(name: "dust-llm-swift", path: "../dust-llm-swift"),
        .package(name: "dust-onnx-swift", path: "../dust-onnx-swift"),
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
    ]
)
