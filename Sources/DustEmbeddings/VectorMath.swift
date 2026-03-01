import Foundation

#if canImport(Accelerate)
import Accelerate
#endif

public enum VectorMath {
    public static func l2Normalize(_ vector: inout [Float]) {
        let norm = l2Norm(vector)
        guard norm >= 1e-12 else {
            return
        }

        #if canImport(Accelerate)
        vector = vDSP.divide(vector, norm)
        #else
        for index in vector.indices {
            vector[index] /= norm
        }
        #endif
    }

    public static func l2Normalized(_ vector: [Float]) -> [Float] {
        var normalized = vector
        l2Normalize(&normalized)
        return normalized
    }

    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else {
            return 0
        }

        let normA = l2Norm(a)
        let normB = l2Norm(b)
        guard normA >= 1e-12, normB >= 1e-12 else {
            return 0
        }

        #if canImport(Accelerate)
        let dot = vDSP.dot(a, b)
        #else
        var dot: Float = 0
        for index in a.indices {
            dot += a[index] * b[index]
        }
        #endif

        return dot / (normA * normB)
    }

    private static func l2Norm(_ vector: [Float]) -> Float {
        #if canImport(Accelerate)
        return sqrt(vDSP.sumOfSquares(vector))
        #else
        var sum: Float = 0
        for value in vector {
            sum += value * value
        }
        return sqrt(sum)
        #endif
    }
}
