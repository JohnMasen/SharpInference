namespace SharpInference;

/// <summary>A named dense float32 buffer exposed by a model state implementation.</summary>
/// <param name="Name">The stable state entry name.</param>
/// <param name="Dimensions">The dense tensor shape.</param>
/// <param name="Values">The mutable single-precision state values.</param>
public sealed record Float32StateView(string Name, IReadOnlyList<int> Dimensions, float[] Values);

/// <summary>Optional host access for state implementations supporting dense float32 buffers.</summary>
public interface INamedFloat32ModelState : IModelState
{
    /// <summary>Gets the current host-visible buffers, synchronizing device state when required.</summary>
    IReadOnlyList<Float32StateView> Views
    {
        get;
    }
    /// <summary>Publishes changes made through the host-visible state buffers.</summary>
    void CommitViews();
}
