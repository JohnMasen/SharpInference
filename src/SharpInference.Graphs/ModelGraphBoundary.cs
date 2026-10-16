
namespace SharpInference.Graphs;

/// <summary>Resources crossing into or out of a selected operation set.</summary>
/// <typeparam name="TNode">The model node type used for the resource results.</typeparam>
/// <param name="Inputs">Distinct resources read by selected operations that are inputs/in-outs or have an outside writer.</param>
/// <param name="Outputs">Distinct resources written by selected operations that are outputs/in-outs or have an outside reader.</param>
public sealed record ModelGraphBoundary<TNode>(
    IReadOnlyList<TNode> Inputs,
    IReadOnlyList<TNode> Outputs)
    where TNode : IModelGraphNode;
