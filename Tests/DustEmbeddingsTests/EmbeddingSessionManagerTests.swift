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

    private func makeConfig() -> EmbeddingSessionConfig {
        EmbeddingSessionConfig(
            dims: 2,
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
