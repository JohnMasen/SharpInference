namespace SharpInference;

/// <summary>Provides model tensor metadata and access while optionally owning CPU copies of the tensor data.</summary>
internal sealed class OwnedModelTensorCatalog : IModelTensorCatalog
{
    private readonly Dictionary<string, OwnedTensor> tensors;

    /// <summary>Creates a catalog containing the tensors exposed by <paramref name="source"/>.</summary>
    /// <param name="source">The catalog whose metadata and tensors are copied or referenced.</param>
    /// <param name="copyWeights">Whether to copy supported CPU tensor values instead of retaining the source tensors.</param>
    public OwnedModelTensorCatalog(IModelTensorCatalog source, bool copyWeights)
    {
        ArgumentNullException.ThrowIfNull(source);
        tensors = new Dictionary<string, OwnedTensor>(StringComparer.Ordinal);
        foreach (var name in source.Names)
        {
            tensors.Add(name, new OwnedTensor(source.GetRequired(name), copyWeights));
        }

        Names = tensors.Keys.ToArray();
    }

    /// <summary>Gets the names of the tensors held by this catalog.</summary>
    public IReadOnlyCollection<string> Names { get; }

    /// <summary>Attempts to retrieve a tensor by name.</summary>
    /// <param name="name">The tensor name to find.</param>
    /// <param name="tensor">When this method returns, the matching tensor if it was found; otherwise, the default value.</param>
    /// <returns><see langword="true"/> when a tensor with the specified name exists; otherwise, <see langword="false"/>.</returns>
    public bool TryGet(string name, out IModelTensor tensor)
    {
        var found = tensors.TryGetValue(name, out var value);
        tensor = value!;
        return found;
    }

    /// <summary>Gets a tensor by name and throws when the tensor is not present.</summary>
    /// <param name="name">The required tensor name.</param>
    /// <returns>The tensor associated with <paramref name="name"/>.</returns>
    /// <exception cref="InvalidDataException">The catalog does not contain <paramref name="name"/>.</exception>
    public IModelTensor GetRequired(string name) =>
        tensors.TryGetValue(name, out var value)
            ? value
            : throw new InvalidDataException($"The model is missing required tensor '{name}'.");

    /// <summary>Releases references to source tensors retained by this catalog.</summary>
    public void ReleaseSource()
    {
        foreach (var tensor in tensors.Values)
        {
            tensor.ReleaseSource();
        }
    }

    /// <summary>Represents one tensor backed by a source tensor or by copied CPU values.</summary>
    private sealed class OwnedTensor : IModelTensor
    {
        private IModelTensor? source;
        private readonly float[]? floats;
        private readonly Half[]? halves;
        private float[]? convertedFloats;

        /// <summary>Creates a tensor wrapper using either source storage or copied values.</summary>
        /// <param name="source">The tensor supplying metadata and, when requested, values.</param>
        /// <param name="copyWeights">Whether to copy supported float32 or float16 values.</param>
        /// <exception cref="InvalidDataException">The tensor uses a data type other than float32 or float16 when copying is requested.</exception>
        public OwnedTensor(IModelTensor source, bool copyWeights)
        {
            Name = source.Name;
            DataType = source.DataType;
            Dimensions = source.Dimensions.ToArray();
            if (!copyWeights)
            {
                this.source = source;
            }
            else if (DataType == TensorDataType.Float32)
            {
                floats = source.FloatValues.ToArray();
            }
            else if (DataType == TensorDataType.Float16)
            {
                halves = source.HalfValues.ToArray();
            }
            else
            {
                throw new InvalidDataException($"Tensor '{Name}' has unsupported type {(uint)DataType}.");
            }
        }

        /// <summary>Gets the tensor name.</summary>
        public string Name { get; }

        /// <summary>Gets the tensor data type.</summary>
        public TensorDataType DataType { get; }

        /// <summary>Gets the tensor dimensions copied from the source tensor.</summary>
        public IReadOnlyList<int> Dimensions { get; }

        /// <summary>Gets the tensor values as float32, converting copied float16 values on first access when necessary.</summary>
        /// <exception cref="InvalidOperationException">The tensor is backed only by a released source and has no copied CPU values.</exception>
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

        /// <summary>Gets the tensor values as float16 when the tensor data type is float16.</summary>
        /// <returns>The float16 values, or an empty span for tensors with another data type.</returns>
        /// <exception cref="InvalidOperationException">The tensor is backed only by a released source and has no copied float16 values.</exception>
        public ReadOnlySpan<Half> HalfValues
        {
            get
            {
                if (DataType != TensorDataType.Float16)
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

        /// <summary>Stops retaining the source tensor so copied storage, if present, is used instead.</summary>
        public void ReleaseSource() => source = null;
    }
}
