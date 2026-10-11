namespace SharpInference.Graphs;

/// <summary>Represents either a fixed tensor size or a named symbolic dimension.</summary>
public abstract record TensorDimension;

/// <summary>Represents a strictly positive fixed tensor dimension.</summary>
public sealed record FixedTensorDimension : TensorDimension
{
    /// <summary>Creates a strictly positive fixed tensor dimension.</summary>
    /// <param name="size">The strictly positive fixed dimension size.</param>
    public FixedTensorDimension(int size)
    {
        if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size));
        Size = size;
    }

    /// <summary>Gets the strictly positive fixed dimension size.</summary>
    public int Size { get; }
}

/// <summary>Represents a tensor dimension resolved by a named shape binding.</summary>
public sealed record SymbolicTensorDimension : TensorDimension
{
    /// <summary>Creates a dimension reference with a nonempty symbolic name.</summary>
    /// <param name="name">The symbolic dimension name.</param>
    public SymbolicTensorDimension(string name)
    {
        Name = string.IsNullOrWhiteSpace(name)
            ? throw new ArgumentException("A symbolic dimension name is required.", nameof(name))
            : name;
    }

    /// <summary>Gets the name of this element.</summary>
    public string Name { get; }
}

/// <summary>Constrains a named dimension&apos;s range, divisibility, and optional specialization buckets.</summary>
public sealed class ShapeConstraint
{
    /// <summary>Creates a named dimension constraint with validated range, divisibility, and optional buckets.</summary>
    /// <param name="dimension">The symbolic dimension name.</param>
    /// <param name="minimum">The inclusive minimum dimension size.</param>
    /// <param name="maximum">The optional inclusive maximum dimension size.</param>
    /// <param name="multipleOf">The positive divisor required for specialized dimension sizes.</param>
    /// <param name="buckets">The optional allowed padded specialization sizes.</param>
    public ShapeConstraint(
        string dimension,
        int minimum = 1,
        int? maximum = null,
        int multipleOf = 1,
        IEnumerable<int>? buckets = null)
    {
        if (string.IsNullOrWhiteSpace(dimension))
            throw new ArgumentException("A dimension name is required.", nameof(dimension));
        if (minimum <= 0) throw new ArgumentOutOfRangeException(nameof(minimum));
        if (maximum is <= 0 || maximum < minimum) throw new ArgumentOutOfRangeException(nameof(maximum));
        if (multipleOf <= 0) throw new ArgumentOutOfRangeException(nameof(multipleOf));
        var bucketValues = buckets?.Distinct().Order().ToArray() ?? [];
        if (bucketValues.Any(value => value < minimum || value > maximum || value % multipleOf != 0))
            throw new ArgumentException("Shape buckets must satisfy the dimension range and multiple.",
                nameof(buckets));

        Dimension = dimension;
        Minimum = minimum;
        Maximum = maximum;
        MultipleOf = multipleOf;
        Buckets = Array.AsReadOnly(bucketValues);
    }

    /// <summary>Gets the symbolic dimension name governed by this constraint.</summary>
    public string Dimension { get; }
    /// <summary>Gets the inclusive minimum dimension size.</summary>
    public int Minimum { get; }
    /// <summary>Gets the optional inclusive maximum dimension size.</summary>
    public int? Maximum { get; }
    /// <summary>Gets the required positive divisor of specialized dimension sizes.</summary>
    public int MultipleOf { get; }
    /// <summary>Gets the sorted allowed padded specialization sizes.</summary>
    public IReadOnlyList<int> Buckets { get; }

    internal int SelectSpecializedSize(int actual)
    {
        if (actual < Minimum || actual > Maximum)
            throw new ArgumentOutOfRangeException(Dimension,
                $"Dimension '{Dimension}' must be in the range [{Minimum}, {Maximum?.ToString() ?? "unbounded"}].");
        if (Buckets.Count == 0)
        {
            if (actual % MultipleOf != 0)
                throw new ArgumentException(
                    $"Dimension '{Dimension}' must be a multiple of {MultipleOf}.", Dimension);
            return actual;
        }

        foreach (var bucket in Buckets)
        {
            if (bucket >= actual) return bucket;
        }
        throw new ArgumentOutOfRangeException(Dimension,
            $"Dimension '{Dimension}' does not fit an available shape bucket.");
    }
}

/// <summary>Maps actual dimension sizes to validated, optionally bucket-padded specialization sizes.</summary>
public sealed class ShapeBinding
{
    private ShapeBinding(
        IReadOnlyDictionary<string, int> actualDimensions,
        IReadOnlyDictionary<string, int> specializedDimensions)
    {
        ActualDimensions = actualDimensions;
        SpecializedDimensions = specializedDimensions;
    }

    /// <summary>Gets the actual sizes keyed by symbolic dimension name.</summary>
    public IReadOnlyDictionary<string, int> ActualDimensions { get; }
    /// <summary>Gets the validated, optionally bucket-padded sizes keyed by dimension name.</summary>
    public IReadOnlyDictionary<string, int> SpecializedDimensions { get; }
    /// <summary>Returns the actual size bound to a symbolic dimension.</summary>
    /// <param name="dimension">The symbolic dimension name.</param>
    public int GetActual(string dimension) => Get(ActualDimensions, dimension);
    /// <summary>Returns the validated specialization size bound to a symbolic dimension.</summary>
    /// <param name="dimension">The symbolic dimension name.</param>
    public int GetSpecialized(string dimension) => Get(SpecializedDimensions, dimension);
    /// <summary>Returns the difference between a dimension&apos;s specialized size and its actual size.</summary>
    /// <param name="dimension">The symbolic dimension name.</param>
    public int GetPadding(string dimension) => checked(GetSpecialized(dimension) - GetActual(dimension));

    /// <summary>Validates actual dimensions against their constraints and selects padded specialization sizes.</summary>
    /// <param name="constraints">The constraints for every symbolic dimension.</param>
    /// <param name="actualDimensions">The actual sizes keyed by symbolic dimension name.</param>
    public static ShapeBinding Create(
        IEnumerable<ShapeConstraint> constraints,
        IReadOnlyDictionary<string, int> actualDimensions)
    {
        ArgumentNullException.ThrowIfNull(constraints);
        ArgumentNullException.ThrowIfNull(actualDimensions);
        var byName = constraints.ToDictionary(value => value.Dimension, StringComparer.Ordinal);
        if (byName.Count == 0)
            throw new ArgumentException("At least one shape constraint is required.", nameof(constraints));
        if (actualDimensions.Count != byName.Count ||
            actualDimensions.Keys.Any(name => !byName.ContainsKey(name)))
            throw new ArgumentException("Actual dimensions must exactly match the constrained dimensions.",
                nameof(actualDimensions));

        var actual = new Dictionary<string, int>(StringComparer.Ordinal);
        var specialized = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var (name, constraint) in byName)
        {
            if (!actualDimensions.TryGetValue(name, out var value))
                throw new ArgumentException($"Actual dimension '{name}' is missing.", nameof(actualDimensions));
            actual.Add(name, value);
            specialized.Add(name, constraint.SelectSpecializedSize(value));
        }
        return new ShapeBinding(actual, specialized);
    }

    private static int Get(IReadOnlyDictionary<string, int> dimensions, string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        return dimensions.TryGetValue(name, out var value)
            ? value
            : throw new KeyNotFoundException($"Shape dimension '{name}' is not bound.");
    }
}

/// <summary>Describes a tensor with fixed or symbolic dimensions that can be specialized to concrete sizes.</summary>
public sealed class SymbolicTensorDescriptor
{
    /// <summary>Creates a tensor descriptor containing fixed or symbolic dimensions.</summary>
    /// <param name="elementType">The tensor scalar element type.</param>
    /// <param name="dimensions">The fixed or symbolic tensor dimensions in axis order.</param>
    /// <param name="layout">The tensor storage layout name.</param>
    public SymbolicTensorDescriptor(
        GraphElementType elementType,
        IEnumerable<TensorDimension> dimensions,
        string layout = "dense")
    {
        ArgumentNullException.ThrowIfNull(dimensions);
        var values = dimensions.ToArray();
        if (values.Length == 0 || values.Any(static value => value is null))
            throw new ArgumentException("A symbolic tensor requires dimensions.", nameof(dimensions));
        if (!Enum.IsDefined(elementType)) throw new ArgumentOutOfRangeException(nameof(elementType));
        if (string.IsNullOrWhiteSpace(layout))
            throw new ArgumentException("A tensor layout is required.", nameof(layout));
        ElementType = elementType;
        Dimensions = Array.AsReadOnly(values);
        Layout = layout;
    }

    /// <summary>Gets the tensor scalar element type.</summary>
    public GraphElementType ElementType { get; }
    /// <summary>Gets the tensor dimensions in axis order.</summary>
    public IReadOnlyList<TensorDimension> Dimensions { get; }
    /// <summary>Gets the tensor storage layout name.</summary>
    public string Layout { get; }

    /// <summary>Resolves symbolic dimensions to specialized sizes and creates a concrete tensor descriptor.</summary>
    /// <param name="binding">The concrete shape binding used for specialization.</param>
    public TensorDescriptor Specialize(ShapeBinding binding)
    {
        ArgumentNullException.ThrowIfNull(binding);
        return new TensorDescriptor(ElementType, Dimensions.Select(dimension => dimension switch
        {
            FixedTensorDimension fixedDimension => fixedDimension.Size,
            SymbolicTensorDimension symbolic => binding.GetSpecialized(symbolic.Name),
            _ => throw new InvalidDataException($"Unknown tensor dimension '{dimension.GetType().Name}'."),
        }), Layout);
    }
}

/// <summary>Describes a versioned graph port with symbolic tensor dimensions and an access contract.</summary>
/// <param name="Name">The port name.</param>
/// <param name="AbiVersion">The port ABI version.</param>
/// <param name="Tensor">The tensor element type, dimensions, and layout.</param>
/// <param name="Access">The permitted read and write access.</param>
public sealed record SymbolicGraphPort(
    string Name,
    int AbiVersion,
    SymbolicTensorDescriptor Tensor,
    GraphResourceAccess Access);

/// <summary>Caches compiled shape specializations with bounded least-recently-used eviction and shared in-flight compilation.</summary>
/// <typeparam name="TKey">The shape specialization cache key type.</typeparam>
/// <typeparam name="TValue">The compiled shape specialization type.</typeparam>
public sealed class ShapeBucketCache<TKey, TValue> where TKey : notnull
{
    private sealed record CacheEntry(TValue Value, LinkedListNode<TKey> Node);

    private readonly object gate = new();
    private readonly int capacity;
    private readonly Dictionary<TKey, CacheEntry> cache = [];
    private readonly Dictionary<TKey, Task<TValue>> compiling = [];
    private readonly LinkedList<TKey> lru = [];

    /// <summary>Creates a shape specialization cache with a positive completed-entry capacity.</summary>
    /// <param name="capacity">The maximum number of completed specializations retained.</param>
    public ShapeBucketCache(int capacity)
    {
        if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        this.capacity = capacity;
    }

    /// <summary>Gets the maximum number of completed shape specializations retained.</summary>
    public int Capacity => capacity;
    /// <summary>Gets the number of entries currently stored.</summary>
    public int Count { get { lock (gate) return cache.Count; } }

    /// <summary>Returns a cached specialization or shares asynchronous compilation, applying cancellation only to this caller&apos;s wait.</summary>
    /// <param name="key">The shape specialization cache key.</param>
    /// <param name="compile">The asynchronous factory compiling a specialization for a key.</param>
    /// <param name="cancellationToken">The token cancelling this caller&apos;s wait without cancelling shared compilation.</param>
    public ValueTask<TValue> GetOrAddAsync(
        TKey key,
        Func<TKey, CancellationToken, ValueTask<TValue>> compile,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(compile);
        Task<TValue> task;
        TaskCompletionSource<TValue>? completion = null;
        lock (gate)
        {
            if (cache.TryGetValue(key, out var cached))
            {
                Touch(cached.Node);
                return ValueTask.FromResult(cached.Value);
            }
            if (!compiling.TryGetValue(key, out task!))
            {
                completion = new TaskCompletionSource<TValue>(
                    TaskCreationOptions.RunContinuationsAsynchronously);
                task = completion.Task;
                compiling.Add(key, task);
            }
        }
        if (completion is not null)
            _ = CompileAsync(key, compile, completion);
        return new ValueTask<TValue>(task.WaitAsync(cancellationToken));
    }

    private async Task CompileAsync(
        TKey key,
        Func<TKey, CancellationToken, ValueTask<TValue>> compile,
        TaskCompletionSource<TValue> completion)
    {
        try
        {
            var value = await compile(key, CancellationToken.None).ConfigureAwait(false);
            lock (gate)
            {
                compiling.Remove(key);
                if (cache.Count == capacity)
                {
                    var oldest = lru.First!;
                    lru.RemoveFirst();
                    cache.Remove(oldest.Value);
                }
                var node = lru.AddLast(key);
                cache.Add(key, new CacheEntry(value, node));
            }
            completion.SetResult(value);
        }
        catch (Exception error)
        {
            lock (gate) compiling.Remove(key);
            completion.SetException(error);
        }
    }

    private void Touch(LinkedListNode<TKey> node)
    {
        if (node.List is null || node == lru.Last) return;
        lru.Remove(node);
        lru.AddLast(node);
    }
}
