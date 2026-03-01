import XCTest
@testable import DustEmbeddings

final class BPETokenizerTests: XCTestCase {
    private func makeTokenizer() throws -> BPETokenizer {
        let vocabURL = try XCTUnwrap(
            Bundle.module.url(forResource: "clip-vocab-mini", withExtension: "json", subdirectory: "Fixtures")
        )
        let mergesURL = try XCTUnwrap(
            Bundle.module.url(forResource: "clip-merges-mini", withExtension: "txt", subdirectory: "Fixtures")
        )
        return try BPETokenizer(vocabPath: vocabURL.path, mergesPath: mergesURL.path)
    }

    private func makeWordPieceTokenizer() throws -> WordPieceTokenizer {
        let vocabURL = try XCTUnwrap(
            Bundle.module.url(forResource: "bert-vocab-mini", withExtension: "txt", subdirectory: "Fixtures")
        )
        return try WordPieceTokenizer(vocabPath: vocabURL.path)
    }

    func testClipTextUsesExpectedIds() throws {
        let tokenizer = try makeTokenizer()

        let output = tokenizer.tokenize(text: "a photo of a cat", maxLength: 77)

        XCTAssertEqual(Array(output.inputIds.prefix(7)), [49406, 320, 1125, 539, 320, 2368, 49407])
    }

    func testTruncatesToSeventySevenAndEndsWithEndToken() throws {
        let tokenizer = try makeTokenizer()
        let longText = Array(repeating: "cat", count: 80).joined(separator: " ")

        let output = tokenizer.tokenize(text: longText, maxLength: 77)

        XCTAssertEqual(output.inputIds.count, 77)
        XCTAssertEqual(output.attentionMask.count, 77)
        XCTAssertEqual(output.tokenTypeIds.count, 77)
        XCTAssertEqual(output.inputIds.last, 49407)
    }

    func testCountTokensMatchesNonSpecialTokens() throws {
        let wordPieceTokenizer = try makeWordPieceTokenizer()
        let wordPieceText = "the quick brown fox"
        let wordPieceOutput = wordPieceTokenizer.tokenize(text: wordPieceText, maxLength: 8)
        let wordPieceNonSpecialCount = wordPieceOutput.inputIds.filter { $0 != 0 && $0 != 101 && $0 != 102 }.count

        XCTAssertEqual(wordPieceTokenizer.countTokens(text: wordPieceText), wordPieceNonSpecialCount)

        let bpeTokenizer = try makeTokenizer()
        let bpeText = "a photo of a cat"
        let bpeOutput = bpeTokenizer.tokenize(text: bpeText, maxLength: 16)
        let bpeNonSpecialCount = bpeOutput.inputIds.filter { $0 != 0 && $0 != 49406 && $0 != 49407 }.count

        XCTAssertEqual(bpeTokenizer.countTokens(text: bpeText), bpeNonSpecialCount)
    }
}
