import XCTest
import DustCore
import DustOnnx
@testable import DustEmbeddings

final class EmbeddingSessionTests: XCTestCase {
    func testE2T6EmbedRunsTokenizeInferPoolAndNormalize() throws {
        let engine = MockONNXEngine()
        let session = EmbeddingSession(
            sessionId: "mini-embed",
            tokenizer: MockTokenizer(),
            onnxSession: ONNXSession(
                sessionId: "mini-embed",
                engine: engine,
                metadata: ONNXModelMetadataValue(
                    inputs: engine.inputMetadata,
                    outputs: engine.outputMetadata,
                    accelerator: "cpu",
                    opset: 17
                ),
                priority: .interactive
            ),
            config: makeConfig()
        )

        let result = try session.embed(text: "hello")

        XCTAssertEqual(result.modelId, "mini-embed")
        XCTAssertEqual(result.tokenCount, 3)
        XCTAssertFalse(result.truncated)
        XCTAssertEqual(result.embedding[0], 1, accuracy: 0.0001)
        XCTAssertEqual(result.embedding[1], 0, accuracy: 0.0001)
        XCTAssertEqual(engine.lastInputs?["input_ids"]?.dtype, "int64")
        XCTAssertEqual(engine.lastInputs?["attention_mask"]?.dtype, "int64")
    }

    func testE2T7CountTokensReturnsCountAndTruncationFlag() throws {
        let session = EmbeddingSession(
            sessionId: "mini-embed",
            tokenizer: MockTokenizer(tokenCount: 7),
            onnxSession: ONNXSession(
                sessionId: "mini-embed",
                engine: MockONNXEngine(),
                metadata: ONNXModelMetadataValue(
                    inputs: [],
                    outputs: [],
                    accelerator: "cpu",
                    opset: 17
                ),
                priority: .interactive
            ),
            config: EmbeddingSessionConfig(
                dims: 2,
                maxSequenceLength: 6,
                tokenizerType: "wordpiece",
                pooling: "mean",
                normalize: true
            )
        )

        let result = session.countTokens(text: "hello")

        XCTAssertEqual(result.count, 7)
        XCTAssertTrue(result.truncated)
    }

    func testE2T5NormalizeFalseReturnsRawPooledVector() throws {
        let engine = MockONNXEngine()
        let session = EmbeddingSession(
            sessionId: "mini-embed",
            tokenizer: MockTokenizer(),
            onnxSession: ONNXSession(
                sessionId: "mini-embed",
                engine: engine,
                metadata: ONNXModelMetadataValue(
                    inputs: engine.inputMetadata,
                    outputs: engine.outputMetadata,
                    accelerator: "cpu",
                    opset: 17
                ),
                priority: .interactive
            ),
            config: EmbeddingSessionConfig(
                dims: 2,
                maxSequenceLength: 6,
                tokenizerType: "wordpiece",
                pooling: "mean",
                normalize: false
            )
        )

        let result = try session.embed(text: "hello")

        let norm = sqrt(result.embedding[0] * result.embedding[0] + result.embedding[1] * result.embedding[1])
        XCTAssertNotEqual(norm, 1.0, accuracy: 0.001)
    }

    func testE2T8InferenceErrorIsPropagated() throws {
        let engine = MockONNXEngine(
            errorToThrow: NSError(domain: "test", code: 1, userInfo: [NSLocalizedDescriptionKey: "GPU out of memory"])
        )
        let session = EmbeddingSession(
            sessionId: "mini-embed",
            tokenizer: MockTokenizer(),
            onnxSession: ONNXSession(
                sessionId: "mini-embed",
                engine: engine,
                metadata: ONNXModelMetadataValue(
                    inputs: engine.inputMetadata,
                    outputs: engine.outputMetadata,
                    accelerator: "cpu",
                    opset: 17
                ),
                priority: .interactive
            ),
            config: makeConfig()
        )

        XCTAssertThrowsError(try session.embed(text: "hello")) { error in
            guard case ONNXError.inferenceError(let detail) = error else {
                XCTFail("Expected ONNXError.inferenceError, got \(error)")
                return
            }
            XCTAssertTrue(detail.contains("GPU out of memory"))
        }
    }

    func testE2T9EvictedSessionThrowsModelEvicted() throws {
        let engine = MockONNXEngine()
        let session = EmbeddingSession(
            sessionId: "mini-embed",
            tokenizer: MockTokenizer(),
            onnxSession: ONNXSession(
                sessionId: "mini-embed",
                engine: engine,
                metadata: ONNXModelMetadataValue(
                    inputs: engine.inputMetadata,
                    outputs: engine.outputMetadata,
                    accelerator: "cpu",
                    opset: 17
                ),
                priority: .interactive
            ),
            config: makeConfig()
        )

        session.evict()

        XCTAssertThrowsError(try session.embed(text: "hello")) { error in
            XCTAssertEqual(error as? ONNXError, ONNXError.modelEvicted)
        }
    }

    func testE3T1EmbedBatchPadsShorterInputsToChunkMaxLength() throws {
        let engine = MockONNXEngine()
        let session = makeSession(
            tokenizer: BatchMockTokenizer(),
            engine: engine
        )

        _ = try session.embedBatch(texts: ["Hello", "Hello world"])

        XCTAssertEqual(engine.lastInputs?["input_ids"]?.shape, [2, 4])
        XCTAssertEqual(
            engine.lastInputs?["input_ids"]?.data,
            [101, 200, 102, 0, 101, 200, 201, 102].map(Double.init)
        )
        XCTAssertEqual(
            engine.lastInputs?["attention_mask"]?.data,
            [1, 1, 1, 0, 1, 1, 1, 1].map(Double.init)
        )
    }

    func testE3T2EmbedBatchReturnsNormalizedEmbeddingsWithConfiguredDims() throws {
        let session = makeSession(
            tokenizer: BatchMockTokenizer(),
            engine: MockONNXEngine()
        )

        let results = try session.embedBatch(texts: ["Hello", "Hello world"])

        XCTAssertEqual(results.count, 2)
        for result in results {
            XCTAssertEqual(result.embedding.count, 2)
            let norm = sqrt(result.embedding[0] * result.embedding[0] + result.embedding[1] * result.embedding[1])
            XCTAssertEqual(norm, 1, accuracy: 0.0001)
        }
    }

    func testE3T3EmbedBatchSingleItemMatchesSingleEmbed() throws {
        let singleSession = makeSession(
            tokenizer: BatchMockTokenizer(),
            engine: MockONNXEngine()
        )
        let batchSession = makeSession(
            tokenizer: BatchMockTokenizer(),
            engine: MockONNXEngine()
        )

        let single = try singleSession.embed(text: "Hello")
        let batch = try XCTUnwrap(try batchSession.embedBatch(texts: ["Hello"]).first)

        XCTAssertEqual(single.tokenCount, batch.tokenCount)
        XCTAssertEqual(single.truncated, batch.truncated)
        XCTAssertEqual(single.modelId, batch.modelId)
        for (expected, actual) in zip(single.embedding, batch.embedding) {
            XCTAssertEqual(expected, actual, accuracy: 0.0001)
        }
    }

    func testE3T4EmbedBatchReturnsEmptyWithoutRunningInference() throws {
        let engine = MockONNXEngine()
        let session = makeSession(
            tokenizer: BatchMockTokenizer(),
            engine: engine
        )

        let results = try session.embedBatch(texts: [])

        XCTAssertTrue(results.isEmpty)
        XCTAssertEqual(engine.runCallCount, 0)
    }

    func testE3T5EmbedBatchMarksTruncatedInputsWhenAllowed() throws {
        let session = makeSession(
            tokenizer: BatchMockTokenizer(),
            engine: MockONNXEngine(),
            config: makeConfig(maxSequenceLength: 4)
        )

        let results = try session.embedBatch(texts: ["Hello", "overflow_text"], truncate: true)

        XCTAssertEqual(results.count, 2)
        XCTAssertTrue(results[1].truncated)
    }

    func testE3T6EmbedBatchFailsFastWhenTruncationIsDisabled() throws {
        let engine = MockONNXEngine()
        let session = makeSession(
            tokenizer: BatchMockTokenizer(),
            engine: engine,
            config: makeConfig(maxSequenceLength: 4)
        )

        XCTAssertThrowsError(try session.embedBatch(texts: ["Hello", "overflow_text"], truncate: false)) { error in
            XCTAssertEqual(
                error as? DustCoreError,
                DustCoreError.invalidInput(detail: "Input exceeds maxSequenceLength of 4")
            )
        }
        XCTAssertEqual(engine.runCallCount, 0)
    }

    func testE3T7EmbedBatchChunksBatchesLargerThan64Items() throws {
        let engine = MockONNXEngine()
        let session = makeSession(
            tokenizer: BatchMockTokenizer(),
            engine: engine
        )

        let results = try session.embedBatch(texts: Array(repeating: "Hello", count: 65))

        XCTAssertEqual(results.count, 65)
        XCTAssertEqual(engine.runCallCount, 2)
    }

    func testE3T8EmbedBatchWrapsBatchInferenceErrors() throws {
        let engine = MockONNXEngine(
            outputGenerator: { _ in
                throw NSError(
                    domain: "test",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "synthetic batch failure"]
                )
            }
        )
        let session = makeSession(
            tokenizer: BatchMockTokenizer(),
            engine: engine
        )

        XCTAssertThrowsError(try session.embedBatch(texts: ["Hello", "Hello world"])) { error in
            guard case DustCoreError.inferenceFailed(let detail) = error else {
                XCTFail("Expected DustCoreError.inferenceFailed, got \(error)")
                return
            }
            XCTAssertTrue((detail ?? "").contains("Batch inference failed"))
        }
    }

    func testE4T1GGUFEmbedReturnsVectorFromEngine() throws {
        let session = EmbeddingSession(
            sessionId: "gguf-embed",
            tokenizer: MockTokenizer(),
            onnxSession: nil,
            config: makeConfig(dims: 4, maxSequenceLength: 32),
            ggufEngine: MockGGUFEngine(dims: 4)
        )

        let result = try session.embed(text: "hello")

        XCTAssertEqual(result.modelId, "gguf-embed")
        XCTAssertEqual(result.embedding.count, 4)
        XCTAssertGreaterThan(result.tokenCount, 0)
    }

    func testE4T2GGUFEmbedDoesNotCallONNXSession() throws {
        let session = EmbeddingSession(
            sessionId: "gguf-embed",
            tokenizer: MockTokenizer(),
            onnxSession: nil,
            config: makeConfig(dims: 4, maxSequenceLength: 32),
            ggufEngine: MockGGUFEngine(dims: 4)
        )

        let result = try session.embed(text: "hello")

        XCTAssertEqual(result.embedding.count, 4)
    }

    func testE4T3GGUFCountTokensUsesEngineNotTokenizer() {
        let session = EmbeddingSession(
            sessionId: "gguf-embed",
            tokenizer: MockTokenizer(tokenCount: 3),
            onnxSession: nil,
            config: makeConfig(dims: 4, maxSequenceLength: 64),
            ggufEngine: MockGGUFEngine(dims: 4, tokenCounts: { _ in 42 })
        )

        let result = session.countTokens(text: "anything")

        XCTAssertEqual(result.count, 42)
    }

    func testE4T4GGUFEmbedBatchFallsBackToSequential() throws {
        let engine = MockGGUFEngine(dims: 4)
        let session = EmbeddingSession(
            sessionId: "gguf-embed",
            tokenizer: MockTokenizer(),
            onnxSession: nil,
            config: makeConfig(dims: 4, maxSequenceLength: 32),
            ggufEngine: engine
        )

        let results = try session.embedBatch(texts: ["a", "b", "c"])

        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(engine.embedCallCount, 3)
    }

    func testE4T5GGUFTruncateFalseThrowsWhenExceedsMax() {
        let session = EmbeddingSession(
            sessionId: "gguf-embed",
            tokenizer: MockTokenizer(),
            onnxSession: nil,
            config: makeConfig(dims: 4, maxSequenceLength: 4),
            ggufEngine: MockGGUFEngine(dims: 4, tokenCounts: { _ in 10 })
        )

        XCTAssertThrowsError(try session.embed(text: "long", truncate: false)) { error in
            XCTAssertEqual(
                error as? DustCoreError,
                DustCoreError.invalidInput(detail: "Input exceeds maxSequenceLength of 4")
            )
        }
    }

    func testE4T6GGUFTruncateTrueMarksTruncated() throws {
        let session = EmbeddingSession(
            sessionId: "gguf-embed",
            tokenizer: MockTokenizer(),
            onnxSession: nil,
            config: makeConfig(dims: 4, maxSequenceLength: 4),
            ggufEngine: MockGGUFEngine(dims: 4, tokenCounts: { _ in 10 })
        )

        let result = try session.embed(text: "long", truncate: true)

        XCTAssertTrue(result.truncated)
    }

    func testE4T7GGUFEngineErrorWrappedAsInferenceFailed() {
        let session = EmbeddingSession(
            sessionId: "gguf-embed",
            tokenizer: MockTokenizer(),
            onnxSession: nil,
            config: makeConfig(dims: 4, maxSequenceLength: 32),
            ggufEngine: MockGGUFEngine(
                dims: 4,
                errorToThrow: NSError(
                    domain: "test",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "GPU OOM"]
                )
            )
        )

        XCTAssertThrowsError(try session.embed(text: "hello")) { error in
            guard case DustCoreError.inferenceFailed(let detail) = error else {
                XCTFail("Expected DustCoreError.inferenceFailed, got \(error)")
                return
            }
            XCTAssertTrue((detail ?? "").contains("Embedding extraction failed"))
        }
    }

    func testE4T8GGUFEvictReleasesEngine() throws {
        let engine = MockGGUFEngine(dims: 4)
        let session = EmbeddingSession(
            sessionId: "gguf-embed",
            tokenizer: MockTokenizer(),
            onnxSession: nil,
            config: makeConfig(dims: 4, maxSequenceLength: 32),
            ggufEngine: engine
        )

        session.evict()

        XCTAssertTrue(engine.evicted)
        XCTAssertThrowsError(try session.embed(text: "hello")) { error in
            XCTAssertEqual(error as? ONNXError, ONNXError.modelEvicted)
        }
    }

    func testE5T1ImageEmbedReturnsNormalizedVector() throws {
        let engine = makeImageEngine(
            dims: 4,
            cannedOutput: imageOutput(
                shape: [1, 4],
                values: [3, 4, 0, 0]
            )
        )
        let onnxSession = makeONNXSession(engine: engine)
        let session = EmbeddingSession(
            sessionId: "mini-embed",
            tokenizer: MockTokenizer(),
            onnxSession: onnxSession,
            config: makeConfig(dims: 4)
        )

        let result = try session.inferImageTensor(
            session: onnxSession,
            tensorName: "pixel_values",
            preprocessed: makeImageTensor()
        )

        XCTAssertEqual(result.embedding.count, 4)
        XCTAssertEqual(result.embedding[0], 0.6, accuracy: 0.0001)
        XCTAssertEqual(result.embedding[1], 0.8, accuracy: 0.0001)
        XCTAssertEqual(result.tokenCount, 0)
        XCTAssertFalse(result.truncated)
        XCTAssertEqual(result.modelId, "mini-embed")
        XCTAssertEqual(engine.lastInputs?["pixel_values"]?.shape, [1, 3, 224, 224])
    }

    func testE5T2ImageEmbedPools3DOutputTensor() throws {
        let engine = makeImageEngine(
            dims: 4,
            cannedOutput: imageOutput(
                shape: [1, 2, 4],
                values: [
                    2, 0, 0, 0,
                    0, 2, 0, 0
                ]
            )
        )
        let onnxSession = makeONNXSession(engine: engine)
        let session = EmbeddingSession(
            sessionId: "mini-embed",
            tokenizer: MockTokenizer(),
            onnxSession: onnxSession,
            config: makeConfig(dims: 4)
        )

        let result = try session.inferImageTensor(
            session: onnxSession,
            tensorName: "pixel_values",
            preprocessed: makeImageTensor()
        )

        let expected = Float(sqrt(0.5))
        XCTAssertEqual(result.embedding[0], expected, accuracy: 0.0001)
        XCTAssertEqual(result.embedding[1], expected, accuracy: 0.0001)
        XCTAssertEqual(result.embedding[2], 0, accuracy: 0.0001)
        XCTAssertEqual(result.embedding[3], 0, accuracy: 0.0001)
    }

    func testE5T3ImageEmbedNormalizeFalseReturnsRawVector() throws {
        let engine = makeImageEngine(
            dims: 4,
            cannedOutput: imageOutput(
                shape: [1, 4],
                values: [3, 4, 0, 0]
            )
        )
        let onnxSession = makeONNXSession(engine: engine)
        let session = EmbeddingSession(
            sessionId: "mini-embed",
            tokenizer: MockTokenizer(),
            onnxSession: onnxSession,
            config: makeConfig(dims: 4, normalize: false)
        )

        let result = try session.inferImageTensor(
            session: onnxSession,
            tensorName: "pixel_values",
            preprocessed: makeImageTensor()
        )

        XCTAssertEqual(result.embedding, [3, 4, 0, 0])
    }

    func testE5T4ResolveImageInputFinds4DTensor() throws {
        let session = makeSession()

        let imageInput = try session.resolveImageInput(from: [
            ONNXTensorMetadata(name: "input_ids", shape: [-1, -1], dtype: "int64"),
            ONNXTensorMetadata(name: "pixel_values", shape: [1, 3, 128, 256], dtype: "float32"),
            ONNXTensorMetadata(name: "attention_mask", shape: [-1, -1], dtype: "int64"),
        ])

        XCTAssertEqual(imageInput.metadata.name, "pixel_values")
        XCTAssertEqual(imageInput.width, 256)
        XCTAssertEqual(imageInput.height, 128)
    }

    func testE5T5ResolveImageInputDefaultsTo224WhenDynamic() throws {
        let session = makeSession()

        let imageInput = try session.resolveImageInput(from: [
            ONNXTensorMetadata(name: "pixel_values", shape: [-1, 3, -1, -1], dtype: "float32")
        ])

        XCTAssertEqual(imageInput.width, 224)
        XCTAssertEqual(imageInput.height, 224)
    }

    func testE5T6ResolveImageInputThrowsWhenNoImageTensor() {
        let session = makeSession()

        XCTAssertThrowsError(try session.resolveImageInput(from: [
            ONNXTensorMetadata(name: "input_ids", shape: [-1, -1], dtype: "int64"),
            ONNXTensorMetadata(name: "attention_mask", shape: [-1, -1], dtype: "int64"),
        ])) { error in
            XCTAssertEqual(
                error as? DustCoreError,
                DustCoreError.invalidInput(detail: "Model does not expose an image input tensor")
            )
        }
    }

    func testE5T7ImageEmbedInferenceErrorPropagates() throws {
        let engine = makeImageEngine(
            dims: 4,
            errorToThrow: NSError(
                domain: "test",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "synthetic image failure"]
            )
        )
        let onnxSession = makeONNXSession(engine: engine)
        let session = EmbeddingSession(
            sessionId: "mini-embed",
            tokenizer: MockTokenizer(),
            onnxSession: onnxSession,
            config: makeConfig(dims: 4)
        )

        XCTAssertThrowsError(try session.inferImageTensor(
            session: onnxSession,
            tensorName: "pixel_values",
            preprocessed: makeImageTensor()
        )) { error in
            guard case ONNXError.inferenceError(let detail) = error else {
                XCTFail("Expected ONNXError.inferenceError, got \(error)")
                return
            }
            XCTAssertTrue(detail.contains("synthetic image failure"))
        }
    }

    func testE5T8ImageEmbedDimensionMismatchThrows() throws {
        let engine = makeImageEngine(
            dims: 8,
            cannedOutput: imageOutput(
                shape: [1, 8],
                values: [1, 2, 3, 4, 5, 6, 7, 8]
            )
        )
        let onnxSession = makeONNXSession(engine: engine)
        let session = EmbeddingSession(
            sessionId: "mini-embed",
            tokenizer: MockTokenizer(),
            onnxSession: onnxSession,
            config: makeConfig(dims: 4)
        )

        XCTAssertThrowsError(try session.inferImageTensor(
            session: onnxSession,
            tensorName: "pixel_values",
            preprocessed: makeImageTensor()
        )) { error in
            XCTAssertEqual(
                error as? DustCoreError,
                DustCoreError.inferenceFailed(detail: "Expected embedding dimension 4, got 8")
            )
        }
    }

    private func makeConfig(
        dims: Int = 2,
        maxSequenceLength: Int = 6,
        normalize: Bool = true
    ) -> EmbeddingSessionConfig {
        EmbeddingSessionConfig(
            dims: dims,
            maxSequenceLength: maxSequenceLength,
            tokenizerType: "wordpiece",
            pooling: "mean",
            normalize: normalize
        )
    }

    private func makeSession(
        tokenizer: any EmbeddingTokenizer = MockTokenizer(),
        engine: MockONNXEngine = MockONNXEngine(),
        config: EmbeddingSessionConfig = EmbeddingSessionConfig(
            dims: 2,
            maxSequenceLength: 6,
            tokenizerType: "wordpiece",
            pooling: "mean",
            normalize: true
        )
    ) -> EmbeddingSession {
        EmbeddingSession(
            sessionId: "mini-embed",
            tokenizer: tokenizer,
            onnxSession: makeONNXSession(engine: engine),
            config: config
        )
    }

    private func makeONNXSession(
        engine: MockONNXEngine,
        sessionId: String = "mini-embed"
    ) -> ONNXSession {
        ONNXSession(
            sessionId: sessionId,
            engine: engine,
            metadata: ONNXModelMetadataValue(
                inputs: engine.inputMetadata,
                outputs: engine.outputMetadata,
                accelerator: "cpu",
                opset: 17
            ),
            priority: .interactive
        )
    }

    private func makeImageEngine(
        dims: Int = 4,
        imageWidth: Int = 224,
        imageHeight: Int = 224,
        cannedOutput: [String: TensorData]? = nil,
        outputGenerator: (([String: TensorData]) throws -> [String: TensorData])? = nil,
        errorToThrow: (any Error)? = nil
    ) -> MockONNXEngine {
        MockONNXEngine(
            inputMetadata: [
                ONNXTensorMetadata(name: "pixel_values", shape: [1, 3, imageHeight, imageWidth], dtype: "float32")
            ],
            outputMetadata: [
                ONNXTensorMetadata(name: "last_hidden_state", shape: [1, dims], dtype: "float32")
            ],
            cannedOutput: cannedOutput,
            outputGenerator: outputGenerator,
            errorToThrow: errorToThrow
        )
    }

    private func makeImageTensor(
        width: Int = 224,
        height: Int = 224
    ) -> TensorData {
        TensorData(
            name: "pixel_values",
            dtype: "float32",
            shape: [1, 3, height, width],
            data: Array(repeating: 0.5, count: 3 * height * width)
        )
    }

    private func imageOutput(
        shape: [Int],
        values: [Double],
        name: String = "last_hidden_state"
    ) -> [String: TensorData] {
        [
            name: TensorData(
                name: name,
                dtype: "float32",
                shape: shape,
                data: values
            )
        ]
    }
}

struct MockTokenizer: EmbeddingTokenizer {
    let vocabSize = 1_000
    let tokenCount: Int

    init(tokenCount: Int = 3) {
        self.tokenCount = tokenCount
    }

    func tokenize(text: String, maxLength: Int) -> TokenizerOutput {
        let inputIds: [Int32] = [101, 200, 102, 0]
        let attentionMask: [Int32] = [1, 1, 1, 0]
        let tokenTypeIds: [Int32] = [0, 0, 0, 0]
        return TokenizerOutput(
            inputIds: Array(inputIds.prefix(maxLength)),
            attentionMask: Array(attentionMask.prefix(maxLength)),
            tokenTypeIds: Array(tokenTypeIds.prefix(maxLength))
        )
    }

    func countTokens(text: String) -> Int {
        tokenCount
    }
}

struct BatchMockTokenizer: EmbeddingTokenizer {
    let vocabSize = 1_000

    func tokenize(text: String, maxLength: Int) -> TokenizerOutput {
        let inputIds = Array(fullInputIds(for: text).prefix(maxLength))
        return TokenizerOutput(
            inputIds: inputIds,
            attentionMask: Array(repeating: Int32(1), count: inputIds.count),
            tokenTypeIds: Array(repeating: Int32(0), count: inputIds.count)
        )
    }

    func countTokens(text: String) -> Int {
        fullInputIds(for: text).count
    }

    private func fullInputIds(for text: String) -> [Int32] {
        if text == "overflow_text" {
            return [101, 200, 201, 202, 203, 204, 205, 102]
        }
        if text.contains(" ") {
            return [101, 200, 201, 102]
        }
        return [101, 200, 102]
    }
}

final class MockGGUFEngine: GGUFEmbeddingEngine, @unchecked Sendable {
    let dims: Int
    var errorToThrow: Error?
    private let tokenCounts: (String) -> Int
    private let embeddings: (String) -> [Float]

    private(set) var embedCallCount = 0
    private(set) var closed = false
    private(set) var evicted = false

    init(
        dims: Int = 4,
        errorToThrow: Error? = nil,
        tokenCounts: @escaping (String) -> Int = { $0.split(separator: " ").count + 2 },
        embeddings: ((String) -> [Float])? = nil
    ) {
        self.dims = dims
        self.errorToThrow = errorToThrow
        self.tokenCounts = tokenCounts
        self.embeddings = embeddings ?? { text in
            let hash = text.hashValue
            return (0..<dims).map { index in
                Float((hash >> (index * 8)) & 0xFF) / 255.0
            }
        }
    }

    func embed(text: String) throws -> [Float] {
        embedCallCount += 1
        if let errorToThrow {
            throw errorToThrow
        }
        return embeddings(text)
    }

    func countTokens(text: String) throws -> Int {
        tokenCounts(text)
    }

    func close() {
        closed = true
    }

    func evict() {
        evicted = true
    }
}
