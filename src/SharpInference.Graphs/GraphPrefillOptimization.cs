namespace SharpInference.Graphs;

/// <summary>Inputs available when selecting and building an architecture-specific prefill graph.</summary>
public sealed record GraphPrefillOptimizationContext(
    LogicalGraph? LogicalGraph,
    ExecutionGraph InferenceGraph,
    GraphModelSignature Model,
    IModelTensorCatalog? Tensors = null,
    int TokenCapacity = 64,
    int ChunkSize = 64);

/// <summary>Builds a prefill execution graph from a model's existing inference graph.</summary>
public interface IPrefillGraphOptimizer
{
    bool CanOptimize(GraphPrefillOptimizationContext context);
    ExecutionGraph Optimize(GraphPrefillOptimizationContext context);
}
