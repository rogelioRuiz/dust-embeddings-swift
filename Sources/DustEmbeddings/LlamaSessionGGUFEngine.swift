import DustCore
import DustLlm

public final class LlamaSessionGGUFEngine: GGUFEmbeddingEngine, @unchecked Sendable {
    private var session: LlamaSession?
    private let onRelease: (@Sendable () -> Void)?

    public init(
        session: LlamaSession?,
        onRelease: (@Sendable () -> Void)? = nil
    ) {
        self.session = session
        self.onRelease = onRelease
    }

    public func embed(text: String) throws -> [Float] {
        guard let session else {
            throw DustCoreError.sessionClosed
        }

        return try session.getEmbedding(text: text)
    }

    public func countTokens(text: String) throws -> Int {
        guard let session else {
            throw DustCoreError.sessionClosed
        }

        return try session.countTokens(text: text)
    }

    public var dims: Int {
        session?.embeddingDims ?? 0
    }

    public func close() {
        release()
    }

    public func evict() {
        release()
    }

    private func release() {
        guard session != nil else {
            return
        }

        session = nil
        onRelease?()
    }
}
