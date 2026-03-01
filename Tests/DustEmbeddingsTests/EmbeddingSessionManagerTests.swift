import XCTest
import DustCore
import DustOnnx
@testable import DustEmbeddings

final class EmbeddingSessionManagerTests: XCTestCase {
    func testE2T8LoadAndUnloadUpdatesRefCount() throws {
        let manager = makeManager()
        let modelURL = temporaryModelURL()

        let first = try manager.loadModel(
            modelPath: modelURL.path,
            modelId: "mini-embed",
            vocabPath: fixturePath("bert-vocab-mini", ext: "txt"),
            mergesPath: nil,
            config: makeConfig(),
            onnxConfig: ONNXConfig(),
            priority: .interactive
        )
        let second = try manager.loadModel(
            modelPath: modelURL.path,
            modelId: "mini-embed",
            vocabPath: fixturePath("bert-vocab-mini", ext: "txt"),
            mergesPath: nil,
            config: makeConfig(),
            onnxConfig: ONNXConfig(),
            priority: .interactive
        )

        XCTAssertEqual(ObjectIdentifier(first), ObjectIdentifier(second))
        XCTAssertEqual(manager.refCount(for: "mini-embed"), 2)

        try manager.unloadModel(id: "mini-embed")

        XCTAssertEqual(manager.refCount(for: "mini-embed"), 1)
    }

    func testE2T9CriticalPressureEvictsUnreferencedSessions() async throws {
        let manager = makeManager()
        let modelURL = temporaryModelURL()

        _ = try manager.loadModel(
            modelPath: modelURL.path,
            modelId: "mini-embed",
            vocabPath: fixturePath("bert-vocab-mini", ext: "txt"),
            mergesPath: nil,
            config: makeConfig(),
            onnxConfig: ONNXConfig(),
            priority: .background
        )
        try manager.unloadModel(id: "mini-embed")

        await manager.evictUnderPressure(level: .critical)

        XCTAssertFalse(manager.hasCachedSession(for: "mini-embed"))
        XCTAssertEqual(manager.sessionCount, 0)
    }

    func testE6T1LoadGGUFModelCachesSession() throws {
        let manager = makeManager()
        let config = makeConfig(dims: 4)
        let engine1 = MockGGUFEngine()

        let first = manager.loadGGUFModel(
            modelId: "gguf-model",
            engine: engine1,
            config: config
        )

        XCTAssertTrue(manager.hasCachedSession(for: "gguf-model"))
        XCTAssertEqual(manager.sessionCount, 1)
        let cached = try XCTUnwrap(manager.session(for: "gguf-model"))
        XCTAssertEqual(ObjectIdentifier(first), ObjectIdentifier(cached))
        XCTAssertEqual(manager.refCount(for: "gguf-model"), 1)

        let engine2 = MockGGUFEngine()
        let second = manager.loadGGUFModel(
            modelId: "gguf-model",
            engine: engine2,
            config: config
        )

        XCTAssertEqual(ObjectIdentifier(first), ObjectIdentifier(second))
        XCTAssertEqual(manager.refCount(for: "gguf-model"), 2)
        XCTAssertTrue(engine2.closed)
        XCTAssertFalse(engine1.closed)
    }

    func testE6T2ForceUnloadRemovesFromCache() async throws {
        let manager = makeManager()
        let modelURL = temporaryModelURL()

        _ = try manager.loadModel(
            modelPath: modelURL.path,
            modelId: "mini-embed",
            vocabPath: fixturePath("bert-vocab-mini", ext: "txt"),
            mergesPath: nil,
            config: makeConfig(),
            onnxConfig: ONNXConfig(),
            priority: .interactive
        )

        try await manager.forceUnloadModel(id: "mini-embed")

        XCTAssertFalse(manager.hasCachedSession(for: "mini-embed"))
        XCTAssertEqual(manager.sessionCount, 0)
        XCTAssertNil(manager.session(for: "mini-embed"))
    }

    func testE6T3ForceUnloadUnknownModelThrows() async throws {
        let manager = makeManager()

        do {
            try await manager.forceUnloadModel(id: "nonexistent")
            XCTFail("Expected DustCoreError.modelNotFound")
        } catch let error as DustCoreError {
            XCTAssertEqual(error, .modelNotFound)
        } catch {
            XCTFail("Expected DustCoreError.modelNotFound, got \(error)")
        }
    }

    func testE6T4UnloadUnknownModelThrows() throws {
        let manager = makeManager()

        XCTAssertThrowsError(try manager.unloadModel(id: "nonexistent")) { error in
            XCTAssertEqual(error as? DustCoreError, .modelNotFound)
        }
    }

    func testE6T5AllModelIdsReturnsSortedIds() throws {
        let manager = makeManager()

        for modelId in ["charlie", "alpha", "bravo"] {
            let modelURL = temporaryModelURL()
            _ = try manager.loadModel(
                modelPath: modelURL.path,
                modelId: modelId,
                vocabPath: fixturePath("bert-vocab-mini", ext: "txt"),
                mergesPath: nil,
                config: makeConfig(),
                onnxConfig: ONNXConfig(),
                priority: .interactive
            )
        }

        XCTAssertEqual(manager.allModelIds(), ["alpha", "bravo", "charlie"])
    }

    func testE6T6StatusIdleWhenEmptyReadyWhenLoaded() async throws {
        let manager = makeManager()
        XCTAssertEqual(manager.status(), .idle)

        let modelURL = temporaryModelURL()
        _ = try manager.loadModel(
            modelPath: modelURL.path,
            modelId: "mini-embed",
            vocabPath: fixturePath("bert-vocab-mini", ext: "txt"),
            mergesPath: nil,
            config: makeConfig(),
            onnxConfig: ONNXConfig(),
            priority: .interactive
        )
        XCTAssertEqual(manager.status(), .ready)

        try await manager.forceUnloadModel(id: "mini-embed")

        XCTAssertEqual(manager.status(), .idle)
    }

    func testE6T7EmbeddingDimensionReturnsConfiguredDims() async throws {
        let manager = makeManager()
        XCTAssertEqual(manager.embeddingDimension(), 0)

        let modelURL = temporaryModelURL()
        _ = try manager.loadModel(
            modelPath: modelURL.path,
            modelId: "mini-embed",
            vocabPath: fixturePath("bert-vocab-mini", ext: "txt"),
            mergesPath: nil,
            config: makeConfig(dims: 4),
            onnxConfig: ONNXConfig(),
            priority: .interactive
        )
        XCTAssertEqual(manager.embeddingDimension(), 4)

        try await manager.forceUnloadModel(id: "mini-embed")

        XCTAssertEqual(manager.embeddingDimension(), 0)
    }

    func testE6T8StandardPressureOnlyEvictsBackgroundSessions() async throws {
        let manager = makeManager()
        let interactiveURL = temporaryModelURL()
        let backgroundURL = temporaryModelURL()

        _ = try manager.loadModel(
            modelPath: interactiveURL.path,
            modelId: "interactive-model",
            vocabPath: fixturePath("bert-vocab-mini", ext: "txt"),
            mergesPath: nil,
            config: makeConfig(),
            onnxConfig: ONNXConfig(),
            priority: .interactive
        )
        _ = try manager.loadModel(
            modelPath: backgroundURL.path,
            modelId: "background-model",
            vocabPath: fixturePath("bert-vocab-mini", ext: "txt"),
            mergesPath: nil,
            config: makeConfig(),
            onnxConfig: ONNXConfig(),
            priority: .background
        )
        try manager.unloadModel(id: "interactive-model")
        try manager.unloadModel(id: "background-model")

        await manager.evictUnderPressure(level: .standard)

        XCTAssertTrue(manager.hasCachedSession(for: "interactive-model"))
        XCTAssertFalse(manager.hasCachedSession(for: "background-model"))
        XCTAssertEqual(manager.sessionCount, 1)
    }

    private func makeManager() -> EmbeddingSessionManager {
        let onnxManager = ONNXSessionManager(
            sessionFactory: { _, modelId, _, priority in
                ONNXSession(
                    sessionId: modelId,
                    metadata: ONNXModelMetadataValue(
                        inputs: [],
                        outputs: [],
                        accelerator: "cpu",
                        opset: 17
                    ),
                    priority: priority
                )
            }
        )

        return EmbeddingSessionManager(
            onnxSessionManager: onnxManager,
            sessionFactory: { onnxSession, modelId, config, _, _, _ in
                EmbeddingSession(
                    sessionId: modelId,
                    tokenizer: MockTokenizer(),
                    onnxSession: onnxSession,
                    config: config
                )
            }
        )
    }

    private func makeConfig(dims: Int = 2) -> EmbeddingSessionConfig {
        EmbeddingSessionConfig(
            dims: dims,
            maxSequenceLength: 8,
            tokenizerType: "wordpiece",
            pooling: "mean",
            normalize: true
        )
    }

    private func temporaryModelURL() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("onnx")
        FileManager.default.createFile(atPath: url.path, contents: Data(), attributes: nil)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: url)
        }
        return url
    }

    private func fixturePath(_ name: String, ext: String) -> String {
        let url = try! XCTUnwrap(
            Bundle.module.url(forResource: name, withExtension: ext, subdirectory: "Fixtures")
        )
        return url.path
    }
}
