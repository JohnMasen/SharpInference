
namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Carries optional descriptive architecture metadata without prescribing execution behavior.</summary>
/// <param name="Name">An optional architecture name.</param>
/// <param name="Role">An optional architectural role.</param>
/// <param name="Description">An optional human-readable description.</param>
/// <param name="Attributes">Optional additional named metadata; the record retains the supplied dictionary.</param>
public sealed record GraphArchitecture(
    string? Name = null,
    string? Role = null,
    string? Description = null,
    IReadOnlyDictionary<string, string>? Attributes = null);
