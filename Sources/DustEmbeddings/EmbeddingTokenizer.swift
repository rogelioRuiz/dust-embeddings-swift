public protocol EmbeddingTokenizer: Sendable {
    var vocabSize: Int { get }

    func tokenize(text: String, maxLength: Int) -> TokenizerOutput
    func countTokens(text: String) -> Int
}
