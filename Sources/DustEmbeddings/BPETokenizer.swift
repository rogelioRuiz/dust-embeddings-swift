import Foundation

public final class BPETokenizer: EmbeddingTokenizer, @unchecked Sendable {
    private static let padTokenId: Int32 = 0
    private static let startTokenId: Int32 = 49406
    private static let endTokenId: Int32 = 49407

    private let vocab: [String: Int32]
    private let merges: [String: Int]
    private let byteEncoder: [UInt8: String]

    public let vocabSize: Int

    public init(vocabPath: String, mergesPath: String) throws {
        let vocabData = try Data(contentsOf: URL(fileURLWithPath: vocabPath))
        let mergesContents = try String(contentsOfFile: mergesPath, encoding: .utf8)

        guard let rawVocab = try JSONSerialization.jsonObject(with: vocabData) as? [String: NSNumber] else {
            throw CocoaError(.fileReadCorruptFile)
        }

        var loadedVocab: [String: Int32] = [:]
        for (token, value) in rawVocab {
            loadedVocab[token] = value.int32Value
        }

        var loadedMerges: [String: Int] = [:]
        var rank = 0
        for line in mergesContents.split(whereSeparator: \.isNewline) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty || trimmed.hasPrefix("#") {
                continue
            }

            let parts = trimmed.split(separator: " ")
            guard parts.count == 2 else { continue }
            loadedMerges["\(parts[0]) \(parts[1])"] = rank
            rank += 1
        }

        self.vocab = loadedVocab
        self.merges = loadedMerges
        self.byteEncoder = Self.makeByteEncoder()
        self.vocabSize = loadedVocab.count
    }

    public func tokenize(text: String, maxLength: Int) -> TokenizerOutput {
        guard maxLength > 0 else {
            return TokenizerOutput(inputIds: [], attentionMask: [], tokenTypeIds: [])
        }

        let contentTokenIds = tokenIds(for: text)
        var inputIds = [Self.startTokenId] + contentTokenIds + [Self.endTokenId]
        if inputIds.count > maxLength {
            inputIds = Array(inputIds.prefix(maxLength))
            inputIds[maxLength - 1] = Self.endTokenId
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
        text.lowercased()
            .split(whereSeparator: \.isWhitespace)
            .flatMap { bpeTokenStrings(for: String($0)) }
            .map { vocab[$0] ?? Self.padTokenId }
    }

    private func bpeTokenStrings(for word: String) -> [String] {
        let encodedPieces = word.utf8.compactMap { byteEncoder[$0] }
        guard let lastPiece = encodedPieces.last else { return [] }

        var symbols = Array(encodedPieces.dropLast())
        symbols.append(lastPiece + "</w>")

        while symbols.count > 1 {
            let bestPair = bestMergePair(in: symbols)
            guard let bestPair else { break }
            symbols = merge(symbols, using: bestPair)
        }

        return symbols
    }

    private func bestMergePair(in symbols: [String]) -> (String, String)? {
        guard symbols.count > 1 else { return nil }

        var bestPair: (String, String)?
        var bestRank = Int.max

        for index in 0..<(symbols.count - 1) {
            let left = symbols[index]
            let right = symbols[index + 1]
            guard let rank = merges["\(left) \(right)"] else { continue }
            if rank < bestRank {
                bestRank = rank
                bestPair = (left, right)
            }
        }

        return bestPair
    }

    private func merge(_ symbols: [String], using pair: (String, String)) -> [String] {
        var merged: [String] = []
        var index = 0

        while index < symbols.count {
            if index < symbols.count - 1, symbols[index] == pair.0, symbols[index + 1] == pair.1 {
                merged.append(symbols[index] + symbols[index + 1])
                index += 2
            } else {
                merged.append(symbols[index])
                index += 1
            }
        }

        return merged
    }

    private static func makeByteEncoder() -> [UInt8: String] {
        var baseValues = Array(UInt32(33)...UInt32(126))
        baseValues += Array(UInt32(161)...UInt32(172))
        baseValues += Array(UInt32(174)...UInt32(255))

        var extraValues = baseValues
        var next = UInt32(256)
        for byte in UInt32(0)...UInt32(255) where !baseValues.contains(byte) {
            baseValues.append(byte)
            extraValues.append(next)
            next += 1
        }

        var encoder: [UInt8: String] = [:]
        for (index, byte) in baseValues.enumerated() {
            encoder[UInt8(byte)] = String(UnicodeScalar(extraValues[index])!)
        }
        return encoder
    }
}
