namespace SharpInference;

/// <summary>A named dense float32 buffer exposed by a model state implementation.</summary>
public sealed record Float32StateView(string Name, IReadOnlyList<int> Dimensions, float[] Values);

/// <summary>Optional host access for state implementations supporting dense float32 buffers.</summary>
public interface INamedFloat32ModelState : IModelState
{
    IReadOnlyList<Float32StateView> Views { get; }
    void CommitViews();
}
