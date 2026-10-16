
namespace SharpInference.Graphs.Logical;

/// <summary>Pairs a resource declaration with its scope in the containing call graph.</summary>
/// <param name="Resource">The bound resource declaration.</param>
/// <param name="Owner">The parent scope or owning call instance, independently of the declaration's graph owner.</param>
internal sealed record GraphCallBoundResource(
    GraphDefinitionResource Resource,
    GraphBindingOwner Owner);
