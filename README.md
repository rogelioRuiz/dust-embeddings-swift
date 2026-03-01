# dust-embeddings-swift

Standalone tokenizers and embedding runtime primitives for Dust — iOS/macOS.

**Version: 0.1.0**

## Overview

Provides tokenization and embedding generation for Apple platforms. Builds on [dust-core-swift](../dust-core-swift), [dust-onnx-swift](../dust-onnx-swift), and [dust-llm-swift](../dust-llm-swift). Requires iOS 16+ / macOS 13+.

```
dust-embeddings-swift/
├── Package.swift                                # SPM: product "DustEmbeddings", iOS 16+ / macOS 13+
├── DustEmbeddings.podspec                       # CocoaPods spec (module name: DustEmbeddings)
├── Sources/DustEmbeddings/
│   ├── BPETokenizer.swift
│   ├── WordPieceTokenizer.swift
│   ├── EmbeddingSession.swift
│   ├── EmbeddingSessionManager.swift
│   ├── PoolingStrategy.swift
│   └── VectorMath.swift
└── Tests/DustEmbeddingsTests/
    └── Fixtures/                                # Tokenizer test fixtures (vocab files, etc.)
```

## Install

### Swift Package Manager — local

```swift
// Package.swift
dependencies: [
    .package(name: "dust-embeddings-swift", path: "../dust-embeddings-swift"),
],
targets: [
    .target(
        name: "MyTarget",
        dependencies: [
            .product(name: "DustEmbeddings", package: "dust-embeddings-swift"),
        ]
    )
]
```

### Swift Package Manager — remote (when published)

```swift
.package(url: "https://github.com/rogelioRuiz/dust-embeddings-swift.git", from: "0.1.0")
```

### CocoaPods

```ruby
pod 'DustEmbeddings', '~> 0.1'
```

## Dependencies

- [dust-core-swift](../dust-core-swift) (DustCore)
- [dust-onnx-swift](../dust-onnx-swift) (DustOnnx)
- [dust-llm-swift](../dust-llm-swift) (DustLlm)

## Usage

```swift
import DustEmbeddings

// 1. Tokenize text
let tokenizer = try BPETokenizer(vocabURL: vocabFileURL)
let tokens = tokenizer.encode("Hello, world!")

// 2. Generate embeddings
let session = try EmbeddingSession(modelPath: modelURL)
let vector = try await session.embed("Hello, world!", pooling: .mean)

// 3. Compare vectors
let similarity = VectorMath.cosineSimilarity(vectorA, vectorB)

// 4. Clean up
session.close()
```

## Test

```bash
cd dust-embeddings-swift
swift test    # 17 XCTest tests
```

Tests use tokenizer fixtures in `Tests/DustEmbeddingsTests/Fixtures/`. Requires macOS with Swift toolchain — no Xcode project needed.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding conventions, and PR guidelines.

## License

Copyright 2026 T6X. Licensed under the [Apache License 2.0](LICENSE).
