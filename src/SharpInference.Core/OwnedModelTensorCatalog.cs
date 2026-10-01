namespace SharpInference;

internal sealed class OwnedModelTensorCatalog : IModelTensorCatalog
{
    private readonly Dictionary<string, OwnedTensor> tensors;

    public OwnedModelTensorCatalog(IModelTensorCatalog source, bool copyWeights)
    {
        ArgumentNullException.ThrowIfNull(source);
        VocabularySize = source.VocabularySize;
        EmbeddingSize = source.EmbeddingSize;
        LayerCount = source.LayerCount;
        tensors = new Dictionary<string, OwnedTensor>(StringComparer.Ordinal);
        foreach (var name in source.Names)
        {
            tensors.Add(name, new OwnedTensor(source.GetRequired(name), copyWeights));
        }

        Names = tensors.Keys.ToArray();
    }

    public int VocabularySize { get; }
    public int EmbeddingSize { get; }
    public int LayerCount { get; }
    public IReadOnlyCollection<string> Names { get; }

    public bool TryGet(string name, out IModelTensor tensor)
    {
        var found = tensors.TryGetValue(name, out var value);
        tensor = value!;
        return found;
    }

    public IModelTensor GetRequired(string name) =>
        tensors.TryGetValue(name, out var value)
            ? value
            : throw new InvalidDataException($"The model is missing required tensor '{name}'.");

    public void ReleaseSource()
    {
        foreach (var tensor in tensors.Values)
        {
            tensor.ReleaseSource();
        }
    }

    private sealed class OwnedTensor : IModelTensor
    {
        private IModelTensor? source;
        private readonly float[]? floats;
        private readonly Half[]? halves;
        private float[]? convertedFloats;

        public OwnedTensor(IModelTensor source, bool copyWeights)
        {
            Name = source.Name;
            DataType = source.DataType;
            Dimensions = source.Dimensions.ToArray();
            if (!copyWeights)
            {
                this.source = source;
            }
            else if (DataType == RwkvTensorDataType.Float32)
            {
                floats = source.FloatValues.ToArray();
            }
            else if (DataType == RwkvTensorDataType.Float16)
            {
                halves = source.HalfValues.ToArray();
            }
            else
            {
                throw new InvalidDataException($"Tensor '{Name}' has unsupported type {(uint)DataType}.");
            }
        }

        public string Name { get; }
        public RwkvTensorDataType DataType { get; }
        public IReadOnlyList<int> Dimensions { get; }

        public ReadOnlySpan<float> FloatValues
        {
            get
            {
                if (source is { } tensor)
                {
                    return tensor.FloatValues;
                }

                if (floats is not null)
                {
                    return floats;
                }

                if (halves is not null)
                {
                    if (convertedFloats is null)
                    {
                        var converted = new float[halves.Length];
                        for (var index = 0; index < converted.Length; index++)
                        {
                            converted[index] = (float)halves[index];
                        }

                        Interlocked.CompareExchange(ref convertedFloats, converted, null);
                    }

                    return convertedFloats;
                }

                throw new InvalidOperationException($"GPU tensor '{Name}' is no longer available on the CPU.");
            }
        }

        public ReadOnlySpan<Half> HalfValues
        {
            get
            {
                if (DataType != RwkvTensorDataType.Float16)
                {
                    return ReadOnlySpan<Half>.Empty;
                }

                if (source is { } tensor)
                {
                    return tensor.HalfValues;
                }

                return halves ?? throw new InvalidOperationException(
                    $"GPU tensor '{Name}' is no longer available on the CPU.");
            }
        }

        public void ReleaseSource() => source = null;
    }
}
