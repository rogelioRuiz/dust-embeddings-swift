import DustCore

public enum TokenizerFactory {
    public static func makeTokenizer(
        type: String,
        vocabPath: String,
        mergesPath: String? = nil
    ) throws -> any EmbeddingTokenizer {
        switch type.lowercased() {
        case "wordpiece":
            return try WordPieceTokenizer(vocabPath: vocabPath)
        case "bpe":
            guard let mergesPath else {
                throw DustCoreError.invalidInput(detail: "mergesPath is required for bpe tokenizers")
            }
            return try BPETokenizer(vocabPath: vocabPath, mergesPath: mergesPath)
        default:
            throw DustCoreError.invalidInput(detail: "Unsupported tokenizer type: \(type)")
        }
    }
}
