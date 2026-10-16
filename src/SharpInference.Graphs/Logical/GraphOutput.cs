
namespace SharpInference.Graphs.Logical;

/// <summary>A typed handle for a public port that writes its bound resource.</summary>
/// <typeparam name="T">The caller's resource type marker; it does not validate the tensor descriptor.</typeparam>
/// <param name="resource">The underlying resource declaration.</param>
public sealed class GraphOutput<T>(GraphDefinitionResource resource) : GraphResource<T>(resource);
