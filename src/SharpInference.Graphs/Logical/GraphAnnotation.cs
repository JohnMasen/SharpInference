
namespace SharpInference.Graphs.Logical;

/// <summary>Optional descriptive metadata attached to a graph definition or operation.</summary>
/// <param name="Name">The display name, if supplied.</param>
/// <param name="Role">The application-defined role, if supplied.</param>
/// <param name="Description">The descriptive text, if supplied.</param>
/// <param name="Attributes">Additional string metadata, or null when absent.</param>
public sealed record GraphAnnotation(
    string? Name = null,
    string? Role = null,
    string? Description = null,
    IReadOnlyDictionary<string, string>? Attributes = null);
