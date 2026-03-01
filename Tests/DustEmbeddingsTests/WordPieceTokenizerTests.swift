import XCTest
@testable import DustEmbeddings

final class WordPieceTokenizerTests: XCTestCase {
    private func makeTokenizer() throws -> WordPieceTokenizer {
        let vocabURL = try XCTUnwrap(
            Bundle.module.url(forResource: "bert-vocab-mini", withExtension: "txt", subdirectory: "Fixtures")
        )
        return try WordPieceTokenizer(vocabPath: vocabURL.path)
    }

    func testHelloWorldUsesKnownIds() throws {
        let tokenizer = try makeTokenizer()

        let output = tokenizer.tokenize(text: "Hello world", maxLength: 8)

        XCTAssertEqual(Array(output.inputIds.prefix(4)), [101, 7592, 2088, 102])
    }

    func testUnaffableUsesExpectedSubwords() throws {
        let tokenizer = try makeTokenizer()

        let output = tokenizer.tokenize(text: "unaffable", maxLength: 8)

        XCTAssertEqual(Array(output.inputIds.prefix(5)), [101, 4895, 4273, 3085, 102])
    }

    func testTruncatesAndKeepsSepAtEnd() throws {
        let tokenizer = try makeTokenizer()

        let output = tokenizer.tokenize(text: "hello world the quick brown fox hello", maxLength: 8)

        XCTAssertEqual(output.inputIds.count, 8)
        XCTAssertEqual(output.attentionMask.count, 8)
        XCTAssertEqual(output.tokenTypeIds.count, 8)
        XCTAssertEqual(output.inputIds.last, 102)
    }

    func testChineseCharactersAreSeparated() throws {
        let tokenizer = try makeTokenizer()

        let output = tokenizer.tokenize(text: "中文", maxLength: 8)
        let nonPadTokens = output.inputIds.filter { $0 != 0 }

        XCTAssertGreaterThanOrEqual(nonPadTokens.count, 4)
        XCTAssertEqual(Array(nonPadTokens.prefix(4)), [101, 1746, 1861, 102])
    }

    func testUnknownTokenFallsBackToUnk() throws {
        let tokenizer = try makeTokenizer()

        let output = tokenizer.tokenize(text: "xyzzyplugh", maxLength: 8)

        XCTAssertTrue(output.inputIds.contains(100))
    }
}
