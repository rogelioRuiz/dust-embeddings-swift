import DustOnnx

final class MockONNXEngine: ONNXEngine {
    let inputMetadata: [ONNXTensorMetadata]
    let outputMetadata: [ONNXTensorMetadata]
    let accelerator: String

    var cannedOutput: [String: TensorData]?
    var outputGenerator: (([String: TensorData]) throws -> [String: TensorData])?
    var errorToThrow: (any Error)?
    private(set) var lastInputs: [String: TensorData]?
    private(set) var runCallCount = 0
    private(set) var closeCallCount = 0

    init(
        inputMetadata: [ONNXTensorMetadata] = [
            ONNXTensorMetadata(name: "input_ids", shape: [-1, -1], dtype: "int64"),
            ONNXTensorMetadata(name: "attention_mask", shape: [-1, -1], dtype: "int64"),
            ONNXTensorMetadata(name: "token_type_ids", shape: [-1, -1], dtype: "int64"),
        ],
        outputMetadata: [ONNXTensorMetadata] = [
            ONNXTensorMetadata(name: "last_hidden_state", shape: [-1, -1, 2], dtype: "float32"),
        ],
        accelerator: String = "cpu",
        cannedOutput: [String: TensorData]? = nil,
        outputGenerator: (([String: TensorData]) throws -> [String: TensorData])? = nil,
        errorToThrow: (any Error)? = nil
    ) {
        self.inputMetadata = inputMetadata
        self.outputMetadata = outputMetadata
        self.accelerator = accelerator
        self.cannedOutput = cannedOutput
        self.outputGenerator = outputGenerator
        self.errorToThrow = errorToThrow
    }

    func run(inputs: [String: TensorData]) throws -> [String: TensorData] {
        lastInputs = inputs
        runCallCount += 1
        if let error = errorToThrow { throw error }
        if let outputGenerator {
            return try outputGenerator(inputs)
        }
        if let cannedOutput {
            return cannedOutput
        }
        return dynamicOutput(from: inputs)
    }

    func close() {
        closeCallCount += 1
    }

    private func dynamicOutput(from inputs: [String: TensorData]) -> [String: TensorData] {
        let inputTensor = inputs["input_ids"] ?? inputs.values.first
        let batchSize = inputTensor?.shape.first ?? 1
        let seqLen = inputTensor?.shape.dropFirst().first ?? 1
        let outputName = outputMetadata.first?.name ?? "last_hidden_state"

        var data: [Double] = []
        data.reserveCapacity(batchSize * seqLen * 2)

        for _ in 0..<batchSize {
            for tokenIndex in 0..<seqLen {
                if tokenIndex == seqLen - 1 {
                    data.append(100)
                    data.append(100)
                } else {
                    data.append(Double(tokenIndex + 1))
                    data.append(0)
                }
            }
        }

        return [
            outputName: TensorData(
                name: outputName,
                dtype: "float32",
                shape: [batchSize, seqLen, 2],
                data: data
            )
        ]
    }
}
