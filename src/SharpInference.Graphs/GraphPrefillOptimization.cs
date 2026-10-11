namespace SharpInference.Graphs;

/// <summary>Inputs available when selecting and building an architecture-specific prefill graph.</summary>
/// <param name="LogicalGraph">The optional original logical graph.</param>
/// <param name="InferenceGraph">The execution graph used for single-token inference.</param>
/// <param name="Model">The model dimensions, attributes, and state ABI signature.</param>
/// <param name="Tensors">The model tensor catalog.</param>
/// <param name="TokenCapacity">The maximum token capacity of the prefill plan.</param>
/// <param name="ChunkSize">The maximum number of tokens processed in one prefill chunk.</param>
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
