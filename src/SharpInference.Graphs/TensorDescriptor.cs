namespace SharpInference.Graphs;

/// <summary>Describes a tensor's element type, positive dimensions, and logical layout.</summary>
public sealed record TensorDescriptor
{
    /// <summary>Creates a tensor descriptor.</summary>
    /// <param name="elementType">The scalar representation to store; this constructor does not validate the enum value.</param>
    /// <param name="dimensions">A nonempty sequence of positive extents, copied once into a read-only list.</param>
    /// <param name="layout">A nonblank layout identifier, stored without interpreting its meaning.</param>
    /// <exception cref="ArgumentNullException">The dimension sequence is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The shape is empty or contains a nonpositive extent.</exception>
    /// <exception cref="ArgumentException">The layout is null, empty, or whitespace.</exception>
    public TensorDescriptor(ElementType elementType, IEnumerable<int> dimensions, string layout = "dense")
    {
        ArgumentNullException.ThrowIfNull(dimensions);
        var values = dimensions.ToArray();
        if (values.Length == 0 || values.Any(value => value <= 0))
        {
            throw new ArgumentOutOfRangeException(nameof(dimensions), "Tensor dimensions must be positive.");
        }

        ElementType = elementType;
        Dimensions = Array.AsReadOnly(values);
        Layout = string.IsNullOrWhiteSpace(layout)
            ? throw new ArgumentException("A tensor layout is required.", nameof(layout))
            : layout;
    }

    /// <summary>Gets the scalar element type.</summary>
    public ElementType ElementType
    {
        get;
    }
    /// <summary>Gets tensor dimensions in axis order.</summary>
    public IReadOnlyList<int> Dimensions
    {
        get;
    }
    /// <summary>Gets the logical tensor layout.</summary>
    public string Layout
    {
        get;
    }
}
