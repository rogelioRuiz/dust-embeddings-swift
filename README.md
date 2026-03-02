<p align="center">
  <img alt="dust" src="assets/dust_banner.png" width="400">
</p>

<p align="center">
  <strong>Device Unified Serving Toolkit</strong><br>
  <a href="https://github.com/rogelioRuiz/dust">dust ecosystem</a> · v0.1.0 · Apache 2.0
</p>

<p align="center">
  <a href="https://github.com/rogelioRuiz/dust/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-informational">
  <img alt="SPM" src="https://img.shields.io/badge/SPM-DustEmbeddings-F05138">
  <img alt="CocoaPods" src="https://img.shields.io/badge/CocoaPods-DustEmbeddings-EE3322">
  <a href="https://swift.org"><img alt="Swift" src="https://img.shields.io/badge/Swift-5.9-orange.svg"></a>
  <img alt="Platforms" src="https://img.shields.io/badge/Platforms-iOS_16+_|_macOS_13+-lightgrey">
  <a href="https://github.com/rogelioRuiz/dust-embeddings-swift/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/rogelioRuiz/dust-embeddings-swift/actions/workflows/ci.yml/badge.svg?branch=main"></a>
</p>

---

<p align="center">
<strong>dust ecosystem</strong> —
<a href="../capacitor-core/README.md">capacitor-core</a> ·
<a href="../capacitor-llm/README.md">capacitor-llm</a> ·
<a href="../capacitor-onnx/README.md">capacitor-onnx</a> ·
<a href="../capacitor-serve/README.md">capacitor-serve</a> ·
<a href="../capacitor-embeddings/README.md">capacitor-embeddings</a>
<br>
<a href="../dust-core-kotlin/README.md">dust-core-kotlin</a> ·
<a href="../dust-llm-kotlin/README.md">dust-llm-kotlin</a> ·
<a href="../dust-onnx-kotlin/README.md">dust-onnx-kotlin</a> ·
<a href="../dust-embeddings-kotlin/README.md">dust-embeddings-kotlin</a> ·
<a href="../dust-serve-kotlin/README.md">dust-serve-kotlin</a>
<br>
<a href="../dust-core-swift/README.md">dust-core-swift</a> ·
<a href="../dust-llm-swift/README.md">dust-llm-swift</a> ·
<a href="../dust-onnx-swift/README.md">dust-onnx-swift</a> ·
<strong>dust-embeddings-swift</strong> ·
<a href="../dust-serve-swift/README.md">dust-serve-swift</a>
</p>

---

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

Copyright 2026 Rogelio Ruiz Perez. Licensed under the [Apache License 2.0](LICENSE).
