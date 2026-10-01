namespace SharpInference.Tests;

public sealed class OwnedModelTensorCatalogTests
{
    [Theory]
    [InlineData(RwkvTensorDataType.Float32)]
    [InlineData(RwkvTensorDataType.Float16)]
    public void CpuWeightsRemainReadableAfterSourceIsReleased(RwkvTensorDataType type)
    {
        var source = new SourceTensor(type);
        var catalog = new OwnedModelTensorCatalog(new SourceCatalog(source), copyWeights: true);
        catalog.ReleaseSource();
        source.Dispose();

        var tensor = catalog.GetRequired("weight");
        Assert.Equal([1f, 2f], tensor.FloatValues.ToArray());
        if (type == RwkvTensorDataType.Float16)
        {
            Assert.Equal([(Half)1, (Half)2], tensor.HalfValues.ToArray());
        }
    }

    [Fact]
    public void GpuMetadataCannotReadReleasedSource()
    {
        var source = new SourceTensor(RwkvTensorDataType.Float16);
        var catalog = new OwnedModelTensorCatalog(new SourceCatalog(source), copyWeights: false);
        Assert.Equal([1f, 2f], catalog.GetRequired("weight").FloatValues.ToArray());
        catalog.ReleaseSource();
        source.Dispose();

        var tensor = catalog.GetRequired("weight");
        Assert.Equal([2], tensor.Dimensions);
        Assert.Throws<InvalidOperationException>(() => tensor.FloatValues.ToArray());
        Assert.Throws<InvalidOperationException>(() => tensor.HalfValues.ToArray());
    }

    [Fact]
    public void LoadedModelCanInferAndSnapshotAfterOriginalFileIsDeleted()
    {
        var path = Path.Combine(Directory.GetCurrentDirectory(), $"rwkv-owned-{Guid.NewGuid():N}.ggml");
        try
        {
            File.Copy(TestModelLoader.GetPath(TestModel.Rwkv7Fp32), path);
            using var model = SharpInference.Runtime.Processor.Load(path);
            File.Delete(path);
            Assert.False(File.Exists(path));
            using var session = model.CreateSession();
            Assert.All(session.ForwardToken(0).ToArray(), value => Assert.True(float.IsFinite(value)));
            using var snapshot = new MemoryStream();
            session.SaveState(snapshot);
            Assert.True(snapshot.Length > 0);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private sealed class SourceCatalog(SourceTensor tensor) : IModelTensorCatalog
    {
        public int VocabularySize => 2;
        public int EmbeddingSize => 2;
        public int LayerCount => 1;
        public IReadOnlyCollection<string> Names => ["weight"];
        public bool TryGet(string name, out IModelTensor result)
        {
            result = tensor;
            return name == "weight";
        }
        public IModelTensor GetRequired(string name) =>
            name == "weight" ? tensor : throw new InvalidDataException(name);
    }

    private sealed class SourceTensor(RwkvTensorDataType type) : IModelTensor, IDisposable
    {
        private bool disposed;
        public string Name => "weight";
        public RwkvTensorDataType DataType => type;
        public IReadOnlyList<int> Dimensions => [2];
        public ReadOnlySpan<float> FloatValues
        {
            get
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                return [1, 2];
            }
        }
        public ReadOnlySpan<Half> HalfValues
        {
            get
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                return new Half[] { (Half)1, (Half)2 };
            }
        }
        public void Dispose() => disposed = true;
    }
}
