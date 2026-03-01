import Foundation
import DustCore
import DustOnnx

private let batchSizeLimit = 64

public struct EmbeddingResult: Sendable {
    public let embedding: [Float]
    public let tokenCount: Int
    public let truncated: Bool
    public let modelId: String

    public init(embedding: [Float], tokenCount: Int, truncated: Bool, modelId: String) {
        self.embedding = embedding
        self.tokenCount = tokenCount
        self.truncated = truncated
        self.modelId = modelId
    }
}

public struct TokenCountResult: Sendable {
    public let count: Int
    public let truncated: Bool

    public init(count: Int, truncated: Bool) {
        self.count = count
        self.truncated = truncated
    }
}

public final class EmbeddingSession: @unchecked Sendable {
    public let sessionId: String
    public let config: EmbeddingSessionConfig

    private let lock = NSLock()
    private let tokenizer: any EmbeddingTokenizer
    private var onnxSession: ONNXSession?
    private var ggufEngine: (any GGUFEmbeddingEngine)?
    private let ggufBackend: Bool
    private var evicted = false

    public init(
        sessionId: String,
        tokenizer: any EmbeddingTokenizer,
        onnxSession: ONNXSession?,
        config: EmbeddingSessionConfig,
        ggufEngine: (any GGUFEmbeddingEngine)? = nil
    ) {
        self.sessionId = sessionId
        self.tokenizer = tokenizer
        self.onnxSession = onnxSession
        self.config = config
        self.ggufEngine = ggufEngine
        self.ggufBackend = ggufEngine != nil
    }

    public func embed(text: String, truncate: Bool = true) throws -> EmbeddingResult {
        if ggufBackend {
            let engine = try activeGGUFEngine()
            let rawTokenCount = try engine.countTokens(text: text)
            let truncated = rawTokenCount > ggufMaxTokenCount
            if truncated && !truncate {
                throw DustCoreError.invalidInput(detail: "Input exceeds maxSequenceLength of \(config.maxSequenceLength)")
            }

            let embedding: [Float]
            do {
                embedding = try engine.embed(text: text)
            } catch {
                throw DustCoreError.inferenceFailed(detail: "Embedding extraction failed: \(Self.errorDetail(from: error))")
            }

            try validate(embedding: embedding)
            return EmbeddingResult(
                embedding: embedding,
                tokenCount: rawTokenCount,
                truncated: truncated,
                modelId: sessionId
            )
        }

        let rawTokenCount = tokenizer.countTokens(text: text)
        let truncated = rawTokenCount > maxTokenCount
        if truncated && !truncate {
            throw DustCoreError.invalidInput(detail: "Input exceeds maxSequenceLength of \(config.maxSequenceLength)")
        }

        let tokenOutput = tokenizer.tokenize(text: text, maxLength: config.maxSequenceLength)
        let outputTensor = try runTextInference(tokenOutput: tokenOutput)
        var embedding = try extractEmbedding(
            from: outputTensor,
            attentionMask: normalizedMask(tokenOutput.attentionMask, count: sequenceLength(from: outputTensor))
        )

        if config.normalize {
            VectorMath.l2Normalize(&embedding)
        }

        return EmbeddingResult(
            embedding: embedding,
            tokenCount: rawTokenCount,
            truncated: truncated,
            modelId: sessionId
        )
    }

    public func embedBatch(texts: [String], truncate: Bool = true) throws -> [EmbeddingResult] {
        guard !texts.isEmpty else {
            return []
        }

        if ggufBackend {
            return try texts.map { try embed(text: $0, truncate: truncate) }
        }

        var results: [EmbeddingResult] = []
        results.reserveCapacity(texts.count)

        for start in stride(from: 0, to: texts.count, by: batchSizeLimit) {
            let end = min(start + batchSizeLimit, texts.count)
            results += try embedBatchChunk(texts: Array(texts[start..<end]), truncate: truncate)
        }

        return results
    }

    public func embedImage(imageData: Data) throws -> EmbeddingResult {
        let session = try activeSession()
        let imageInput = try resolveImageInput(from: session.metadata.inputs)
        let preprocessed = try ImagePreprocessor.preprocess(
            imageData: imageData,
            targetWidth: imageInput.width,
            targetHeight: imageInput.height,
            resize: "crop_center",
            normalization: "imagenet",
            customMean: nil,
            customStd: nil
        )
        return try inferImageTensor(
            session: session,
            tensorName: imageInput.metadata.name,
            preprocessed: preprocessed
        )
    }

    func inferImageTensor(
        session: ONNXSession,
        tensorName: String,
        preprocessed: TensorData
    ) throws -> EmbeddingResult {
        let imageTensor = TensorData(
            name: tensorName,
            dtype: preprocessed.dtype,
            shape: preprocessed.shape,
            data: preprocessed.data
        )
        let outputs = try session.runInference(
            inputs: [imageTensor.name: imageTensor],
            outputNames: [config.outputName]
        )
        guard let outputTensor = outputs[config.outputName] else {
            throw DustCoreError.inferenceFailed(detail: "Output tensor '\(config.outputName)' was not returned")
        }

        var embedding = try extractEmbedding(
            from: outputTensor,
            attentionMask: Array(repeating: 1, count: sequenceLength(from: outputTensor))
        )

        if config.normalize {
            VectorMath.l2Normalize(&embedding)
        }

        return EmbeddingResult(
            embedding: embedding,
            tokenCount: 0,
            truncated: false,
            modelId: sessionId
        )
    }

    public func countTokens(text: String) -> TokenCountResult {
        let count: Int
        if ggufBackend {
            count = (try? activeGGUFEngine().countTokens(text: text)) ?? 0
        } else {
            count = tokenizer.countTokens(text: text)
        }
        let maxCount = ggufBackend ? ggufMaxTokenCount : maxTokenCount
        return TokenCountResult(count: count, truncated: count > maxCount)
    }

    public func tokenize(text: String, maxLength: Int? = nil) -> TokenizerOutput {
        tokenizer.tokenize(text: text, maxLength: maxLength ?? config.maxSequenceLength)
    }

    public func close() async throws {
        if ggufBackend {
            let engine = detachGGUF(evicted: false)
            engine?.close()
            return
        }

        let session = detach(evicted: false)
        if let session {
            try await session.close()
        }
    }

    public func evict() {
        if ggufBackend {
            let engine = detachGGUF(evicted: true)
            engine?.evict()
            return
        }

        let session = detach(evicted: true)
        session?.evict()
    }

    func detach(evicted: Bool) -> ONNXSession? {
        lock.lock()
        defer { lock.unlock() }

        let session = onnxSession
        onnxSession = nil
        self.evicted = evicted
        return session
    }

    var usesGGUFBackend: Bool {
        ggufBackend
    }

    private var maxTokenCount: Int {
        max(0, config.maxSequenceLength - 2)
    }

    private var ggufMaxTokenCount: Int {
        config.maxSequenceLength
    }

    private func activeSession() throws -> ONNXSession {
        lock.lock()
        defer { lock.unlock() }

        if let onnxSession {
            return onnxSession
        }

        if evicted {
            throw ONNXError.modelEvicted
        }

        throw DustCoreError.sessionClosed
    }

    private func activeGGUFEngine() throws -> any GGUFEmbeddingEngine {
        lock.lock()
        defer { lock.unlock() }

        if let ggufEngine {
            return ggufEngine
        }

        if evicted {
            throw ONNXError.modelEvicted
        }

        throw DustCoreError.sessionClosed
    }

    private func detachGGUF(evicted: Bool) -> (any GGUFEmbeddingEngine)? {
        lock.lock()
        defer { lock.unlock() }

        let engine = ggufEngine
        ggufEngine = nil
        onnxSession = nil
        self.evicted = evicted
        return engine
    }

    private func runTextInference(tokenOutput: TokenizerOutput) throws -> TensorData {
        let session = try activeSession()
        let inputNames = Set(session.metadata.inputs.map(\.name))
        let seqLen = tokenOutput.inputIds.count

        var inputs: [String: TensorData] = [:]
        inputs[config.inputNames.inputIds] = TensorData(
            name: config.inputNames.inputIds,
            dtype: "int64",
            shape: [1, seqLen],
            data: tokenOutput.inputIds.map(Double.init)
        )
        if inputNames.contains(config.inputNames.attentionMask) {
            inputs[config.inputNames.attentionMask] = TensorData(
                name: config.inputNames.attentionMask,
                dtype: "int64",
                shape: [1, seqLen],
                data: tokenOutput.attentionMask.map(Double.init)
            )
        }
        if inputNames.contains(config.inputNames.tokenTypeIds) {
            inputs[config.inputNames.tokenTypeIds] = TensorData(
                name: config.inputNames.tokenTypeIds,
                dtype: "int64",
                shape: [1, seqLen],
                data: tokenOutput.tokenTypeIds.map(Double.init)
            )
        }

        let outputs = try session.runInference(inputs: inputs, outputNames: [config.outputName])
        guard let outputTensor = outputs[config.outputName] else {
            throw DustCoreError.inferenceFailed(detail: "Output tensor '\(config.outputName)' was not returned")
        }

        return outputTensor
    }

    private func embedBatchChunk(texts: [String], truncate: Bool) throws -> [EmbeddingResult] {
        let rawTokenCounts = texts.map { tokenizer.countTokens(text: $0) }
        if !truncate, rawTokenCounts.contains(where: { $0 > maxTokenCount }) {
            throw DustCoreError.invalidInput(detail: "Input exceeds maxSequenceLength of \(config.maxSequenceLength)")
        }

        let tokenOutputs = texts.map { tokenizer.tokenize(text: $0, maxLength: config.maxSequenceLength) }
        let maxSeqLen = tokenOutputs.map(\.inputIds.count).max() ?? 0
        let paddedOutputs = tokenOutputs.map { padTokenizerOutput($0, targetLength: maxSeqLen) }
        let outputTensor = try runBatchTextInference(tokenOutputs: paddedOutputs, seqLen: maxSeqLen)
        let embeddings = try extractBatchEmbeddings(
            from: outputTensor,
            tokenOutputs: paddedOutputs,
            batchSize: texts.count
        )

        return texts.indices.map { index in
            var embedding = embeddings[index]
            if config.normalize {
                VectorMath.l2Normalize(&embedding)
            }

            return EmbeddingResult(
                embedding: embedding,
                tokenCount: rawTokenCounts[index],
                truncated: rawTokenCounts[index] > maxTokenCount,
                modelId: sessionId
            )
        }
    }

    private func padTokenizerOutput(_ tokenOutput: TokenizerOutput, targetLength: Int) -> TokenizerOutput {
        guard tokenOutput.inputIds.count < targetLength else {
            return tokenOutput
        }

        let padding = Array(repeating: Int32.zero, count: targetLength - tokenOutput.inputIds.count)
        return TokenizerOutput(
            inputIds: tokenOutput.inputIds + padding,
            attentionMask: tokenOutput.attentionMask + padding,
            tokenTypeIds: tokenOutput.tokenTypeIds + padding
        )
    }

    private func runBatchTextInference(
        tokenOutputs: [TokenizerOutput],
        seqLen: Int
    ) throws -> TensorData {
        let session = try activeSession()
        let inputNames = Set(session.metadata.inputs.map(\.name))
        let batchSize = tokenOutputs.count

        var inputs: [String: TensorData] = [:]
        inputs[config.inputNames.inputIds] = TensorData(
            name: config.inputNames.inputIds,
            dtype: "int64",
            shape: [batchSize, seqLen],
            data: tokenOutputs.flatMap { $0.inputIds }.map(Double.init)
        )
        if inputNames.contains(config.inputNames.attentionMask) {
            inputs[config.inputNames.attentionMask] = TensorData(
                name: config.inputNames.attentionMask,
                dtype: "int64",
                shape: [batchSize, seqLen],
                data: tokenOutputs.flatMap { $0.attentionMask }.map(Double.init)
            )
        }
        if inputNames.contains(config.inputNames.tokenTypeIds) {
            inputs[config.inputNames.tokenTypeIds] = TensorData(
                name: config.inputNames.tokenTypeIds,
                dtype: "int64",
                shape: [batchSize, seqLen],
                data: tokenOutputs.flatMap { $0.tokenTypeIds }.map(Double.init)
            )
        }

        let outputs: [String: TensorData]
        do {
            outputs = try session.runInference(inputs: inputs, outputNames: [config.outputName])
        } catch {
            throw DustCoreError.inferenceFailed(detail: "Batch inference failed: \(Self.errorDetail(from: error))")
        }

        guard let outputTensor = outputs[config.outputName] else {
            throw DustCoreError.inferenceFailed(detail: "Output tensor '\(config.outputName)' was not returned")
        }

        return outputTensor
    }

    private func extractBatchEmbeddings(
        from outputTensor: TensorData,
        tokenOutputs: [TokenizerOutput],
        batchSize: Int
    ) throws -> [[Float]] {
        if outputTensor.shape.count == 2,
           outputTensor.shape.first == batchSize {
            let hiddenDim = outputTensor.shape[1]
            let values = outputTensor.data.map(Float.init)

            return try (0..<batchSize).map { index in
                let start = index * hiddenDim
                let end = start + hiddenDim
                let embedding = Array(values[start..<end])
                try validate(embedding: embedding)
                return embedding
            }
        }

        if outputTensor.shape.count == 3,
           outputTensor.shape[0] == batchSize {
            let seqLen = outputTensor.shape[1]
            let hiddenDim = outputTensor.shape[2]
            let values = outputTensor.data.map(Float.init)

            return try (0..<batchSize).map { index in
                let start = index * seqLen * hiddenDim
                let end = start + (seqLen * hiddenDim)
                let pooled = Pooling.apply(
                    strategy: config.pooling,
                    hiddenStates: Array(values[start..<end]),
                    attentionMask: normalizedMask(tokenOutputs[index].attentionMask, count: seqLen),
                    seqLen: seqLen,
                    hiddenDim: hiddenDim
                )
                try validate(embedding: pooled)
                return pooled
            }
        }

        throw DustCoreError.inferenceFailed(
            detail: "Unsupported embedding tensor shape: \(outputTensor.shape)"
        )
    }

    private func extractEmbedding(from outputTensor: TensorData, attentionMask: [Int32]) throws -> [Float] {
        if outputTensor.shape.count == 2,
           outputTensor.shape.first == 1 {
            let embedding = outputTensor.data.map(Float.init)
            try validate(embedding: embedding)
            return embedding
        }

        if outputTensor.shape.count == 3,
           outputTensor.shape[0] == 1 {
            let seqLen = outputTensor.shape[1]
            let hiddenDim = outputTensor.shape[2]
            let hiddenStates = outputTensor.data.map(Float.init)
            let pooled = Pooling.apply(
                strategy: config.pooling,
                hiddenStates: hiddenStates,
                attentionMask: normalizedMask(attentionMask, count: seqLen),
                seqLen: seqLen,
                hiddenDim: hiddenDim
            )
            try validate(embedding: pooled)
            return pooled
        }

        throw DustCoreError.inferenceFailed(
            detail: "Unsupported embedding tensor shape: \(outputTensor.shape)"
        )
    }

    private func validate(embedding: [Float]) throws {
        guard embedding.count == config.dims else {
            throw DustCoreError.inferenceFailed(
                detail: "Expected embedding dimension \(config.dims), got \(embedding.count)"
            )
        }
    }

    private func sequenceLength(from outputTensor: TensorData) -> Int {
        if outputTensor.shape.count == 3 {
            return max(outputTensor.shape[1], 1)
        }

        return 1
    }

    private func normalizedMask(_ mask: [Int32], count: Int) -> [Int32] {
        if count <= 0 {
            return []
        }

        if mask.count == count {
            return mask
        }

        if mask.count > count {
            return Array(mask.prefix(count))
        }

        return mask + Array(repeating: 0, count: count - mask.count)
    }

    func resolveImageInput(from metadata: [ONNXTensorMetadata]) throws -> (metadata: ONNXTensorMetadata, width: Int, height: Int) {
        guard let imageInput = metadata.first(where: { $0.shape.count == 4 }) else {
            throw DustCoreError.invalidInput(detail: "Model does not expose an image input tensor")
        }

        let height = imageInput.shape[2] > 0 ? imageInput.shape[2] : 224
        let width = imageInput.shape[3] > 0 ? imageInput.shape[3] : 224
        return (imageInput, width, height)
    }

    private static func errorDetail(from error: any Error) -> String {
        let nsError = error as NSError
        if let detail = nsError.userInfo[NSLocalizedDescriptionKey] as? String,
           !detail.isEmpty {
            return detail
        }

        let description = String(describing: error)
        if !description.isEmpty {
            return description
        }

        return nsError.localizedDescription
    }
}
