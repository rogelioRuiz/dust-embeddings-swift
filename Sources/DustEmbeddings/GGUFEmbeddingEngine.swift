import Foundation

public protocol GGUFEmbeddingEngine: AnyObject, Sendable {
    func embed(text: String) throws -> [Float]
    func countTokens(text: String) throws -> Int
    var dims: Int { get }
    func close()
    func evict()
}
