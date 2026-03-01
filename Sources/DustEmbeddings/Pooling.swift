public enum Pooling {
    public static func mean(
        hiddenStates: [Float],
        attentionMask: [Int32],
        seqLen: Int,
        hiddenDim: Int
    ) -> [Float] {
        let activeTokenCount = attentionMask.reduce(0) { partialResult, value in
            partialResult + (value == 0 ? 0 : 1)
        }

        guard activeTokenCount > 0 else {
            return Array(repeating: 0, count: hiddenDim)
        }

        var result = Array(repeating: Float.zero, count: hiddenDim)
        for tokenIndex in 0..<seqLen where attentionMask[tokenIndex] != 0 {
            let base = tokenIndex * hiddenDim
            for hiddenIndex in 0..<hiddenDim {
                result[hiddenIndex] += hiddenStates[base + hiddenIndex]
            }
        }

        let divisor = Float(activeTokenCount)
        for hiddenIndex in 0..<hiddenDim {
            result[hiddenIndex] /= divisor
        }

        return result
    }

    public static func cls(hiddenStates: [Float], hiddenDim: Int) -> [Float] {
        guard hiddenStates.count >= hiddenDim else {
            return Array(repeating: 0, count: hiddenDim)
        }

        return Array(hiddenStates.prefix(hiddenDim))
    }

    public static func eos(
        hiddenStates: [Float],
        attentionMask: [Int32],
        seqLen: Int,
        hiddenDim: Int
    ) -> [Float] {
        let lastIndex = attentionMask.lastIndex(where: { $0 != 0 }) ?? 0
        return slice(
            hiddenStates: hiddenStates,
            tokenIndex: min(max(lastIndex, 0), max(seqLen - 1, 0)),
            hiddenDim: hiddenDim
        )
    }

    public static func lastToken(
        hiddenStates: [Float],
        attentionMask: [Int32],
        seqLen: Int,
        hiddenDim: Int
    ) -> [Float] {
        eos(
            hiddenStates: hiddenStates,
            attentionMask: attentionMask,
            seqLen: seqLen,
            hiddenDim: hiddenDim
        )
    }

    public static func apply(
        strategy: String,
        hiddenStates: [Float],
        attentionMask: [Int32],
        seqLen: Int,
        hiddenDim: Int
    ) -> [Float] {
        switch strategy.lowercased() {
        case "mean":
            return mean(
                hiddenStates: hiddenStates,
                attentionMask: attentionMask,
                seqLen: seqLen,
                hiddenDim: hiddenDim
            )
        case "cls":
            return cls(hiddenStates: hiddenStates, hiddenDim: hiddenDim)
        case "eos":
            return eos(
                hiddenStates: hiddenStates,
                attentionMask: attentionMask,
                seqLen: seqLen,
                hiddenDim: hiddenDim
            )
        case "last_token":
            return lastToken(
                hiddenStates: hiddenStates,
                attentionMask: attentionMask,
                seqLen: seqLen,
                hiddenDim: hiddenDim
            )
        default:
            return mean(
                hiddenStates: hiddenStates,
                attentionMask: attentionMask,
                seqLen: seqLen,
                hiddenDim: hiddenDim
            )
        }
    }

    private static func slice(hiddenStates: [Float], tokenIndex: Int, hiddenDim: Int) -> [Float] {
        guard hiddenDim > 0 else {
            return []
        }

        let start = tokenIndex * hiddenDim
        let end = min(start + hiddenDim, hiddenStates.count)
        guard start >= 0, start < end else {
            return Array(repeating: 0, count: hiddenDim)
        }

        let values = Array(hiddenStates[start..<end])
        if values.count == hiddenDim {
            return values
        }

        return values + Array(repeating: 0, count: hiddenDim - values.count)
    }
}
