import XCTest
@testable import DustEmbeddings

final class PoolingTests: XCTestCase {
    func testE2T1MeanPoolingUsesAttentionMask() {
        let hiddenStates: [Float] = [
            1, 0, 1,
            9, 9, 9,
            2, 0, 2,
        ]
        let pooled = Pooling.mean(
            hiddenStates: hiddenStates,
            attentionMask: [1, 0, 1],
            seqLen: 3,
            hiddenDim: 3
        )

        XCTAssertEqual(pooled[0], 1.5, accuracy: 0.0001)
        XCTAssertEqual(pooled[1], 0, accuracy: 0.0001)
        XCTAssertEqual(pooled[2], 1.5, accuracy: 0.0001)
    }

    func testE2T2ClsPoolingReturnsFirstRow() {
        let pooled = Pooling.cls(hiddenStates: [1, 2, 3, 4, 5, 6], hiddenDim: 3)

        XCTAssertEqual(pooled, [1, 2, 3])
    }

    func testE2T3EosPoolingReturnsLastAttendedToken() {
        let hiddenStates: [Float] = [
            1, 1,
            2, 2,
            3, 3,
            9, 9,
            10, 10,
        ]
        let pooled = Pooling.eos(
            hiddenStates: hiddenStates,
            attentionMask: [1, 1, 1, 0, 0],
            seqLen: 5,
            hiddenDim: 2
        )

        XCTAssertEqual(pooled, [3, 3])
    }
}
