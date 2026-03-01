import Foundation
import DustCore
import DustOnnx

public final class EmbeddingSessionManager: DustEmbeddingService, @unchecked Sendable {
    public typealias SessionFactory = @Sendable (
        _ onnxSession: ONNXSession,
        _ modelId: String,
        _ config: EmbeddingSessionConfig,
        _ vocabPath: String,
        _ mergesPath: String?,
        _ priority: DustSessionPriority
    ) throws -> EmbeddingSession

    public static let inferenceQueue = DispatchQueue(
        label: "io.t6x.dust.embeddings.inference",
        qos: .userInitiated
    )

    private let lock = NSLock()
    private let onnxSessionManager: ONNXSessionManager
    private let sessionFactory: SessionFactory
    private var cachedSessions: [String: CachedSession] = [:]
    private var configs: [String: EmbeddingSessionConfig] = [:]

    public init(
        onnxSessionManager: ONNXSessionManager,
        sessionFactory: SessionFactory? = nil
    ) {
        self.onnxSessionManager = onnxSessionManager
        self.sessionFactory = sessionFactory ?? { onnxSession, modelId, config, vocabPath, mergesPath, _ in
            let tokenizer = try TokenizerFactory.makeTokenizer(
                type: config.tokenizerType,
                vocabPath: vocabPath,
                mergesPath: mergesPath
            )
            return EmbeddingSession(
                sessionId: modelId,
                tokenizer: tokenizer,
                onnxSession: onnxSession,
                config: config
            )
        }
    }

    public func loadModel(
        modelPath: String,
        modelId: String,
        vocabPath: String,
        mergesPath: String?,
        config: EmbeddingSessionConfig,
        onnxConfig: ONNXConfig,
        priority: DustSessionPriority
    ) throws -> EmbeddingSession {
        lock.lock()
        defer { lock.unlock() }

        if var cached = cachedSessions[modelId] {
            cached.refCount += 1
            cached.lastAccessTime = DispatchTime.now().uptimeNanoseconds
            cachedSessions[modelId] = cached
            return cached.session
        }

        let onnxSession = try onnxSessionManager.loadModel(
            path: modelPath,
            modelId: modelId,
            config: onnxConfig,
            priority: priority
        )
        let session = try sessionFactory(
            onnxSession,
            modelId,
            config,
            vocabPath,
            mergesPath,
            priority
        )

        cachedSessions[modelId] = CachedSession(
            session: session,
            priority: priority,
            refCount: 1,
            lastAccessTime: DispatchTime.now().uptimeNanoseconds
        )
        configs[modelId] = config
        return session
    }

    public func loadGGUFModel(
        modelId: String,
        engine: any GGUFEmbeddingEngine,
        config: EmbeddingSessionConfig
    ) -> EmbeddingSession {
        lock.lock()
        defer { lock.unlock() }

        if var cached = cachedSessions[modelId] {
            cached.refCount += 1
            cached.lastAccessTime = DispatchTime.now().uptimeNanoseconds
            cachedSessions[modelId] = cached
            engine.close()
            return cached.session
        }

        let session = EmbeddingSession(
            sessionId: modelId,
            tokenizer: NoopTokenizer(),
            onnxSession: nil,
            config: config,
            ggufEngine: engine
        )

        cachedSessions[modelId] = CachedSession(
            session: session,
            priority: .interactive,
            refCount: 1,
            lastAccessTime: DispatchTime.now().uptimeNanoseconds
        )
        configs[modelId] = config
        return session
    }

    public func unloadModel(id: String) throws {
        lock.lock()
        defer { lock.unlock() }

        guard var cached = cachedSessions[id], cached.refCount > 0 else {
            throw DustCoreError.modelNotFound
        }

        cached.refCount -= 1
        cached.lastAccessTime = DispatchTime.now().uptimeNanoseconds
        cachedSessions[id] = cached
    }

    public func forceUnloadModel(id: String) async throws {
        let session: EmbeddingSession

        lock.lock()
        guard let removed = cachedSessions.removeValue(forKey: id) else {
            lock.unlock()
            throw DustCoreError.modelNotFound
        }
        configs.removeValue(forKey: id)
        session = removed.session
        lock.unlock()

        if session.usesGGUFBackend {
            try await session.close()
            return
        }

        defer {
            _ = session.detach(evicted: false)
        }

        do {
            try await onnxSessionManager.forceUnloadModel(id: id)
        } catch let error as DustCoreError where error == .modelNotFound {
            // The embedding cache is authoritative here. If ONNX was already gone,
            // we still invalidate the higher-level session.
        }
    }

    public func session(for id: String) -> EmbeddingSession? {
        lock.lock()
        defer { lock.unlock() }

        guard var cached = cachedSessions[id] else {
            return nil
        }

        cached.lastAccessTime = DispatchTime.now().uptimeNanoseconds
        cachedSessions[id] = cached
        return cached.session
    }

    public func allModelIds() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return cachedSessions.keys.sorted()
    }

    public func evictUnderPressure(level: MemoryPressureLevel) async {
        let evicted: [(String, EmbeddingSession)]

        lock.lock()
        let eligible = cachedSessions.filter { _, cached in
            guard cached.refCount == 0 else {
                return false
            }

            switch level {
            case .standard:
                return cached.priority == .background
            case .critical:
                return true
            }
        }
        let sorted = eligible.sorted { $0.value.lastAccessTime < $1.value.lastAccessTime }
        evicted = sorted.map { ($0.key, $0.value.session) }
        for (id, _) in sorted {
            cachedSessions.removeValue(forKey: id)
            configs.removeValue(forKey: id)
        }
        lock.unlock()

        for (id, session) in evicted {
            if session.usesGGUFBackend {
                session.evict()
            } else {
                _ = await onnxSessionManager.evict(modelId: id)
                _ = session.detach(evicted: true)
            }
        }
    }

    public func embed(texts: [String]) async throws -> [[Float]] {
        guard let session = firstSession() else {
            throw DustCoreError.modelNotFound
        }

        return try session.embedBatch(texts: texts).map(\.embedding)
    }

    public func embeddingDimension() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return configs.values.first?.dims ?? 0
    }

    public func status() -> DustEmbeddingStatus {
        lock.lock()
        defer { lock.unlock() }
        return cachedSessions.isEmpty ? .idle : .ready
    }

    public func refCount(for id: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        return cachedSessions[id]?.refCount ?? 0
    }

    public func hasCachedSession(for id: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return cachedSessions[id] != nil
    }

    public var sessionCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return cachedSessions.count
    }

    private func firstSession() -> EmbeddingSession? {
        lock.lock()
        defer { lock.unlock() }
        return cachedSessions.values.first?.session
    }
}

private struct NoopTokenizer: EmbeddingTokenizer {
    let vocabSize = 0

    func tokenize(text: String, maxLength: Int) -> TokenizerOutput {
        TokenizerOutput(inputIds: [], attentionMask: [], tokenTypeIds: [])
    }

    func countTokens(text: String) -> Int {
        0
    }
}

private struct CachedSession {
    let session: EmbeddingSession
    let priority: DustSessionPriority
    var refCount: Int
    var lastAccessTime: UInt64
}
