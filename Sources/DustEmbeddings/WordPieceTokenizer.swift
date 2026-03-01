import Foundation

public final class WordPieceTokenizer: EmbeddingTokenizer, @unchecked Sendable {
    private static let padTokenId: Int32 = 0
    private static let unkTokenId: Int32 = 100
    private static let clsTokenId: Int32 = 101
    private static let sepTokenId: Int32 = 102
    private static let maxPieceLength = 200

    private let vocab: [String: Int32]

    public let vocabSize: Int

    public init(vocabPath: String) throws {
        let contents = try String(contentsOfFile: vocabPath, encoding: .utf8)
        var loadedVocab: [String: Int32] = [:]
        for (index, line) in contents.split(whereSeparator: \.isNewline).enumerated() {
            loadedVocab[String(line)] = Int32(index)
        }
        self.vocab = loadedVocab
        self.vocabSize = loadedVocab.count
    }

    public func tokenize(text: String, maxLength: Int) -> TokenizerOutput {
        guard maxLength > 0 else {
            return TokenizerOutput(inputIds: [], attentionMask: [], tokenTypeIds: [])
        }

        let contentTokenIds = tokenIds(for: text)
        var inputIds = [Self.clsTokenId] + contentTokenIds + [Self.sepTokenId]
        if inputIds.count > maxLength {
            inputIds = Array(inputIds.prefix(maxLength))
            inputIds[maxLength - 1] = Self.sepTokenId
        }

        let realTokenCount = inputIds.count
        if inputIds.count < maxLength {
            inputIds.append(contentsOf: Array(repeating: Self.padTokenId, count: maxLength - inputIds.count))
        }

        let attentionMask = Array(repeating: Int32(1), count: realTokenCount)
            + Array(repeating: Int32(0), count: maxLength - realTokenCount)
        let tokenTypeIds = Array(repeating: Int32(0), count: maxLength)
        return TokenizerOutput(inputIds: inputIds, attentionMask: attentionMask, tokenTypeIds: tokenTypeIds)
    }

    public func countTokens(text: String) -> Int {
        tokenIds(for: text).count
    }

    private func tokenIds(for text: String) -> [Int32] {
        normalizedWords(from: text).flatMap(wordPieceTokenIds(for:))
    }

    private func normalizedWords(from text: String) -> [String] {
        let lowercased = text.lowercased()
        let stripped = stripAccents(in: lowercased)
        let cjkSeparated = insertSpacesAroundCJK(in: stripped)
        return cjkSeparated
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
    }

    private func stripAccents(in text: String) -> String {
        let decomposed = text.decomposedStringWithCanonicalMapping
        let scalars = decomposed.unicodeScalars.filter { !CharacterSet.nonBaseCharacters.contains($0) }
        return String(String.UnicodeScalarView(scalars))
    }

    private func insertSpacesAroundCJK(in text: String) -> String {
        var pieces: [String] = []
        pieces.reserveCapacity(text.count * 2)
        for scalar in text.unicodeScalars {
            let scalarString = String(scalar)
            if isCJK(scalar) {
                pieces.append(" ")
                pieces.append(scalarString)
                pieces.append(" ")
            } else {
                pieces.append(scalarString)
            }
        }
        return pieces.joined()
    }

    private func isCJK(_ scalar: UnicodeScalar) -> Bool {
        switch scalar.value {
        case 0x3400...0x4DBF,
             0x4E00...0x9FFF,
             0xF900...0xFAFF,
             0x20000...0x2A6DF,
             0x2A700...0x2B73F,
             0x2B740...0x2B81F,
             0x2B820...0x2CEAF,
             0x2F800...0x2FA1F:
            return true
        default:
            return false
        }
    }

    private func wordPieceTokenIds(for word: String) -> [Int32] {
        guard !word.isEmpty else { return [] }

        let characters = Array(word)
        var tokenIds: [Int32] = []
        var start = 0

        while start < characters.count {
            var end = min(characters.count, start + Self.maxPieceLength)
            var matchedTokenId: Int32?

            while end > start {
                var piece = String(characters[start..<end])
                if start > 0 {
                    piece = "##" + piece
                }
                if let tokenId = vocab[piece] {
                    matchedTokenId = tokenId
                    break
                }
                end -= 1
            }

            guard let tokenId = matchedTokenId else {
                return [Self.unkTokenId]
            }

            tokenIds.append(tokenId)
            start = end
        }

        return tokenIds
    }
}
