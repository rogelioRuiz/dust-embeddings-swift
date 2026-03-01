public struct EmbeddingSessionConfig: Equatable, Sendable {
    public let dims: Int
    public let maxSequenceLength: Int
    public let tokenizerType: String
    public let pooling: String
    public let normalize: Bool
    public let inputNames: InputNames
    public let outputName: String

    public struct InputNames: Equatable, Sendable {
        public let inputIds: String
        public let attentionMask: String
        public let tokenTypeIds: String

        public init(
            inputIds: String = "input_ids",
            attentionMask: String = "attention_mask",
            tokenTypeIds: String = "token_type_ids"
        ) {
            self.inputIds = inputIds
            self.attentionMask = attentionMask
            self.tokenTypeIds = tokenTypeIds
        }
    }

    public init(
        dims: Int,
        maxSequenceLength: Int,
        tokenizerType: String,
        pooling: String,
        normalize: Bool,
        inputNames: InputNames = InputNames(),
        outputName: String = "last_hidden_state"
    ) {
        self.dims = dims
        self.maxSequenceLength = maxSequenceLength
        self.tokenizerType = tokenizerType
        self.pooling = pooling
        self.normalize = normalize
        self.inputNames = inputNames
        self.outputName = outputName
    }
}
