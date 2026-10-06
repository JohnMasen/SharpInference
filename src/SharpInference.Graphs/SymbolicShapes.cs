namespace SharpInference.Graphs;

public abstract record TensorDimension;

public sealed record FixedTensorDimension : TensorDimension
{
    public FixedTensorDimension(int size)
    {
        if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size));
        Size = size;
    }

    public int Size { get; }
}

public sealed record SymbolicTensorDimension : TensorDimension
{
    public SymbolicTensorDimension(string name)
    {
        Name = string.IsNullOrWhiteSpace(name)
            ? throw new ArgumentException("A symbolic dimension name is required.", nameof(name))
            : name;
    }

    public string Name { get; }
}

public sealed class ShapeConstraint
{
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

    public string Dimension { get; }
    public int Minimum { get; }
    public int? Maximum { get; }
    public int MultipleOf { get; }
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

public sealed class ShapeBinding
{
    private ShapeBinding(
        IReadOnlyDictionary<string, int> actualDimensions,
        IReadOnlyDictionary<string, int> specializedDimensions)
    {
        ActualDimensions = actualDimensions;
        SpecializedDimensions = specializedDimensions;
    }

    public IReadOnlyDictionary<string, int> ActualDimensions { get; }
    public IReadOnlyDictionary<string, int> SpecializedDimensions { get; }
    public int GetActual(string dimension) => Get(ActualDimensions, dimension);
    public int GetSpecialized(string dimension) => Get(SpecializedDimensions, dimension);
    public int GetPadding(string dimension) => checked(GetSpecialized(dimension) - GetActual(dimension));

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

public sealed class SymbolicTensorDescriptor
{
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

    public GraphElementType ElementType { get; }
    public IReadOnlyList<TensorDimension> Dimensions { get; }
    public string Layout { get; }

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

public sealed record SymbolicGraphPort(
    string Name,
    int AbiVersion,
    SymbolicTensorDescriptor Tensor,
    GraphResourceAccess Access);

public sealed class ShapeBucketCache<TKey, TValue> where TKey : notnull
{
    private sealed record CacheEntry(TValue Value, LinkedListNode<TKey> Node);

    private readonly object gate = new();
    private readonly int capacity;
    private readonly Dictionary<TKey, CacheEntry> cache = [];
    private readonly Dictionary<TKey, Task<TValue>> compiling = [];
    private readonly LinkedList<TKey> lru = [];

    public ShapeBucketCache(int capacity)
    {
        if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        this.capacity = capacity;
    }

    public int Capacity => capacity;
    public int Count { get { lock (gate) return cache.Count; } }

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
