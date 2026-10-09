using SharpInference.Gguf;

namespace SharpInference.Architectures.Phi4;

public sealed record Phi4TextGraphParameters(int Layers, int QueryHeads, int KeyValueHeads, int RotarySize,
    float RopeBase, float RmsEpsilon, float RopeScale = 1)
{
    public void Validate(int embeddingWidth)
    {
        if (Layers is <= 0 or > 32 || QueryHeads <= 0 || KeyValueHeads <= 0 || QueryHeads % KeyValueHeads != 0 ||
            embeddingWidth <= 0 || embeddingWidth % QueryHeads != 0 || RotarySize <= 0 || RotarySize % 2 != 0 ||
            RotarySize > embeddingWidth / QueryHeads || !float.IsFinite(RopeBase) || RopeBase <= 0 ||
            !float.IsFinite(RmsEpsilon) || RmsEpsilon <= 0 || !float.IsFinite(RopeScale) || RopeScale <= 0)
            throw new InvalidDataException("Phi4 text graph parameters have incompatible heads, rotary dimensions or normalization constants.");
    }
}

public sealed class Phi4PackageTensorCatalog : IModelFile
{
    private readonly Dictionary<string, IModelTensor> tensors = new(StringComparer.Ordinal);
    private readonly IModelFile[] sources;
    private bool disposed;

    /// <summary>Takes ownership of the supplied files after successful construction without copying tensor data.</summary>
    public Phi4PackageTensorCatalog(string path, Phi4TextGraphParameters parameters, IModelFile text, IModelFile omni,
        IModelFile visionAdapter, IModelFile speechAdapter, Phi4Tokenizer? tokenizer = null,
        float? visionAdapterAlpha = null, float? speechAdapterAlpha = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(omni);
        ArgumentNullException.ThrowIfNull(visionAdapter);
        ArgumentNullException.ThrowIfNull(speechAdapter);
        var embedding = text.GetRequired("token_embd.weight");
        if (embedding.Dimensions.Count != 2 || embedding.Dimensions.Any(size => size <= 0))
            throw new InvalidDataException("Phi4 text embeddings require a positive vocabulary-by-width matrix.");
        parameters.Validate(embedding.Dimensions[1]);
        Path = path;
        TextParameters = parameters;
        Tokenizer = tokenizer;
        if (visionAdapterAlpha is { } visionAlpha && (!float.IsFinite(visionAlpha) || visionAlpha <= 0) ||
            speechAdapterAlpha is { } speechAlpha && (!float.IsFinite(speechAlpha) || speechAlpha <= 0))
            throw new InvalidDataException("Phi4 adapter alpha must be finite and positive.");
        VisionAdapterAlpha = visionAdapterAlpha;
        SpeechAdapterAlpha = speechAdapterAlpha;
        sources = [text, omni, visionAdapter, speechAdapter];
        Add("text", text, aliases: false);
        Add("omni", omni, aliases: true);
        Add("vision-adapter", visionAdapter, aliases: false);
        Add("speech-adapter", speechAdapter, aliases: false);
        var frequencies = Enumerable.Range(0, parameters.RotarySize / 2)
            .Select(index => 1f / MathF.Pow(parameters.RopeBase, 2f * index / parameters.RotarySize)).ToArray();
        if (frequencies.Any(value => !float.IsFinite(value) || value <= 0))
            throw new InvalidDataException("Phi4 rotary frequencies must be finite and positive.");
        if (!tensors.TryAdd("text.rope_frequencies", new FrequencyTensor(this, frequencies)))
            throw new InvalidDataException("Phi4 package contains a reserved generated tensor name.");
        Names = Array.AsReadOnly(tensors.Keys.ToArray());

        void Add(string scope, IModelFile file, bool aliases)
        {
            foreach (var name in file.Names)
            {
                var source = file.GetRequired(name);
                var scoped = scope + "." + name;
                if (!tensors.TryAdd(scoped, new Tensor(this, scoped, source)) ||
                    aliases && !tensors.TryAdd(name, new Tensor(this, name, source)))
                    throw new InvalidDataException($"Duplicate Phi4 package tensor name '{scoped}'.");
            }
        }
    }

    public string Path { get; }
    public Phi4TextGraphParameters TextParameters { get; }
    public Phi4Tokenizer? Tokenizer { get; }
    public float? VisionAdapterAlpha { get; }
    public float? SpeechAdapterAlpha { get; }
    public IReadOnlyCollection<string> Names { get; }
    public bool TryGet(string name, out IModelTensor tensor) => tensors.TryGetValue(name, out tensor!);
    public IModelTensor GetRequired(string name) => tensors.TryGetValue(name, out var tensor) ? tensor :
        throw new InvalidDataException($"Phi4 package tensor '{name}' was not found.");

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        List<Exception>? errors = null;
        var released = new HashSet<IModelFile>(ReferenceEqualityComparer.Instance);
        foreach (var source in sources.Reverse())
        {
            if (!released.Add(source)) continue;
            try { source.Dispose(); }
            catch (Exception error) { (errors ??= []).Add(error); }
        }
        if (errors is not null) throw new AggregateException("Phi4 package source disposal failed.", errors);
    }

    private sealed class Tensor(Phi4PackageTensorCatalog owner, string name, IModelTensor source) : IModelTensor
    {
        public string Name { get; } = name;
        public TensorDataType DataType { get; } = source.DataType;
        public IReadOnlyList<int> Dimensions { get; } = Array.AsReadOnly(source.Dimensions.ToArray());
        public ReadOnlySpan<float> FloatValues
        {
            get { ObjectDisposedException.ThrowIf(owner.disposed, owner); return source.FloatValues; }
        }
        public ReadOnlySpan<Half> HalfValues
        {
            get { ObjectDisposedException.ThrowIf(owner.disposed, owner); return source.HalfValues; }
        }
    }

    private sealed class FrequencyTensor(Phi4PackageTensorCatalog owner, float[] values) : IModelTensor
    {
        public string Name => "text.rope_frequencies";
        public TensorDataType DataType => TensorDataType.Float32;
        public IReadOnlyList<int> Dimensions { get; } = Array.AsReadOnly(new[] { values.Length });
        public ReadOnlySpan<float> FloatValues
        {
            get { ObjectDisposedException.ThrowIf(owner.disposed, owner); return values; }
        }
        public ReadOnlySpan<Half> HalfValues => throw new InvalidOperationException("Rotary frequencies use FP32 storage.");
    }
}

public sealed class Phi4PackageGraphReader : IModelReader
{
    public IModelFile Open(string path)
    {
        var package = Phi4ModelPackage.Open(path);
        try
        {
            var text = package.Text;
            var parameters = new Phi4TextGraphParameters(checked((int)text.GetMetadata<uint>("phi3.block_count")),
                checked((int)text.GetMetadata<uint>("phi3.attention.head_count")),
                checked((int)text.GetMetadata<uint>("phi3.attention.head_count_kv")),
                text.Metadata.ContainsKey("phi3.rope.dimension_count") ? checked((int)text.GetMetadata<uint>("phi3.rope.dimension_count")) : 96,
                text.GetMetadata<float>("phi3.rope.freq_base"), text.GetMetadata<float>("phi3.attention.layer_norm_rms_epsilon"),
                text.GetMetadata<float>("phi3.rope.scaling.attn_factor"));
            return new Phi4PackageTensorCatalog(package.Directory, parameters,
                new GgufCatalog(package.Text), new GgufCatalog(package.Omni), new GgufCatalog(package.VisionAdapter),
                new GgufCatalog(package.SpeechAdapter), package.Tokenizer,
                package.VisionAdapter.GetMetadata<float>("adapter.lora.alpha"), package.SpeechAdapter.GetMetadata<float>("adapter.lora.alpha"));
        }
        catch (Exception error)
        {
            try { package.Dispose(); }
            catch (Exception cleanup) { throw new AggregateException("Phi4 package creation and cleanup failed.", error, cleanup); }
            throw;
        }
    }

    private sealed class GgufCatalog : IModelFile
    {
        private readonly GgufModelFile file;
        private readonly Dictionary<string, IModelTensor> tensors;
        private bool disposed;
        public GgufCatalog(GgufModelFile file)
        {
            this.file = file;
            tensors = file.Tensors.ToDictionary(value => value.Key, value => (IModelTensor)new Tensor(this, value.Value), StringComparer.Ordinal);
            Names = Array.AsReadOnly(tensors.Keys.ToArray());
        }
        public string Path => file.Path;
        public IReadOnlyCollection<string> Names { get; }
        public bool TryGet(string name, out IModelTensor tensor) => tensors.TryGetValue(name, out tensor!);
        public IModelTensor GetRequired(string name) => tensors.TryGetValue(name, out var tensor) ? tensor : throw new InvalidDataException(name);
        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
            file.Dispose();
        }
        private sealed class Tensor : IModelTensor
        {
            private readonly GgufCatalog owner;
            private readonly GgufModelTensor tensor;
            public Tensor(GgufCatalog owner, GgufModelTensor tensor)
            {
                this.owner = owner;
                this.tensor = tensor;
                DataType = tensor.Type switch
                {
                    GgufModelTensorType.Float16 => TensorDataType.Float16,
                    GgufModelTensorType.Float32 => TensorDataType.Float32,
                    _ => throw new NotSupportedException($"Phi4 graph tensor '{tensor.Name}' requires FP16 or FP32 storage."),
                };
                Dimensions = Array.AsReadOnly(tensor.Dimensions.Reverse().Select(value => checked((int)value)).ToArray());
            }
            public string Name => tensor.Name;
            public TensorDataType DataType { get; }
            public IReadOnlyList<int> Dimensions { get; }
            public ReadOnlySpan<float> FloatValues
            {
                get { ObjectDisposedException.ThrowIf(owner.disposed, owner); return tensor.FloatValues; }
            }
            public ReadOnlySpan<Half> HalfValues
            {
                get { ObjectDisposedException.ThrowIf(owner.disposed, owner); return tensor.HalfValues; }
            }
        }
    }
}
