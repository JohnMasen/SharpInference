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
    /// <summary>Determines whether the optimizer can build a prefill graph for the supplied context.</summary>
    bool CanOptimize(GraphPrefillOptimizationContext context);

    /// <summary>Builds an optimized prefill execution graph.</summary>
    ExecutionGraph Optimize(GraphPrefillOptimizationContext context);
}
