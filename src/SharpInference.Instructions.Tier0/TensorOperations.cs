using SharpInference.Graphs;

namespace SharpInference.Instructions;

public static class TensorOperations
{
    public static readonly OperationId Fill = new("core.tensor.fill");
    public static readonly OperationId CastFp16ToFp32 = new("core.tensor.cast-f16-f32");
    public static readonly OperationId Reshape = new("core.tensor.reshape");
    public static readonly OperationId Slice = new("core.tensor.slice");
    public static readonly OperationId Broadcast = new("core.tensor.broadcast");
    public static readonly OperationId BatchedMatVec = new("core.tensor.batched-mat-vec");
    public static readonly OperationId ReduceLastSum = new("core.tensor.reduce-last-sum");
    public static readonly OperationId ReduceLastMean = new("core.tensor.reduce-last-mean");
    public static readonly OperationId HeadOuter = new("core.tensor.head-outer");
}
