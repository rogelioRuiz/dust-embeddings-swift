public struct TokenizerOutput: Equatable, Sendable {
    public let inputIds: [Int32]
    public let attentionMask: [Int32]
    public let tokenTypeIds: [Int32]

    public init(inputIds: [Int32], attentionMask: [Int32], tokenTypeIds: [Int32]) {
        self.inputIds = inputIds
        self.attentionMask = attentionMask
        self.tokenTypeIds = tokenTypeIds
    }
}
