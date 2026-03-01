import XCTest
@testable import DustEmbeddings

final class VectorMathTests: XCTestCase {
    func testE2T4L2NormalizedReturnsUnitVector() {
        let normalized = VectorMath.l2Normalized([3, 4])

        XCTAssertEqual(normalized[0], 0.6, accuracy: 0.0001)
        XCTAssertEqual(normalized[1], 0.8, accuracy: 0.0001)
    }

    func testE2T5CosineSimilarityMatchesOrthogonalAndIdenticalVectors() {
        XCTAssertEqual(
            VectorMath.cosineSimilarity([1, 0], [0, 1]),
            0,
            accuracy: 0.0001
        )
        XCTAssertEqual(
            VectorMath.cosineSimilarity([2, 2], [2, 2]),
            1,
            accuracy: 0.0001
        )
    }
}
