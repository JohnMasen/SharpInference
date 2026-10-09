using System.Buffers.Binary;
using System.Runtime.InteropServices;
using SharpInference.Architectures.Phi4;

namespace SharpInference.Tests;

public sealed class Phi4PackageGraphTests
{
    private static readonly Phi4TextGraphParameters Parameters = new(2, 2, 1, 2, 10000, 1e-5f);

    [Fact]
    public void PackageCatalogScopesAdaptersRetainsOmniAliasesAndOwnsGeneratedFrequencies()
    {
        var text = new File("token_embd.weight", [8, 4]);
        var omni = new File("a.global_mean", [4]);
        var vision = new File("blk.0.attn_qkv.weight.lora_a", [2, 4]);
        var speech = new File("blk.0.attn_qkv.weight.lora_a", [3, 4]);
        using var catalog = new Phi4PackageTensorCatalog("package", Parameters, text, omni, vision, speech);
        Assert.Equal("text.token_embd.weight", catalog.GetRequired("text.token_embd.weight").Name);
        Assert.Equal([2, 4], catalog.GetRequired("vision-adapter.blk.0.attn_qkv.weight.lora_a").Dimensions);
        Assert.Equal([3, 4], catalog.GetRequired("speech-adapter.blk.0.attn_qkv.weight.lora_a").Dimensions);
        Assert.Equal("a.global_mean", catalog.GetRequired("a.global_mean").Name);
        Assert.Equal(omni.Values, catalog.GetRequired("omni.a.global_mean").FloatValues.ToArray());
        text.Values[0] = 9;
        Assert.Equal(9f, catalog.GetRequired("text.token_embd.weight").FloatValues[0]);
        var frequencies = catalog.GetRequired("text.rope_frequencies");
        Assert.Equal(TensorDataType.Float32, frequencies.DataType);
        Assert.Equal([1f], frequencies.FloatValues.ToArray());
        catalog.Dispose();
        Assert.All(new[] { text, omni, vision, speech }, file => Assert.Equal(1, file.Disposals));
        Assert.Equal([8, 4], catalog.GetRequired("text.token_embd.weight").Dimensions);
        Assert.Throws<ObjectDisposedException>(() => catalog.GetRequired("text.token_embd.weight").FloatValues.ToArray());
        Assert.Throws<ObjectDisposedException>(() => frequencies.FloatValues.ToArray());
    }

    [Fact]
    public void PackageDisposalReleasesEverySourceAfterErrorsAndDoesNotDoubleReleaseAliases()
    {
        var text = new File("token_embd.weight", [8, 4]);
        var omni = new File("a.global_mean", [4]);
        var adapter = new File("lora", [2, 4]) { FailDispose = true };
        var catalog = new Phi4PackageTensorCatalog("package", Parameters, text, omni, adapter, adapter);
        var failure = Assert.Throws<AggregateException>(catalog.Dispose);
        Assert.Single(failure.InnerExceptions);
        Assert.Equal(1, adapter.Disposals);
        Assert.Equal(1, omni.Disposals);
        Assert.Equal(1, text.Disposals);
        catalog.Dispose();
        Assert.Equal(1, adapter.Disposals);
    }

    [Fact]
    public void ReservedNameCollisionIsRejectedWithoutTransferringOwnership()
    {
        var text = new File("token_embd.weight", [8, 4]);
        var omni = new File("text.rope_frequencies", [1]);
        var adapter = new File("lora", [2, 4]);
        Assert.Throws<InvalidDataException>(() => new Phi4PackageTensorCatalog("package", Parameters, text, omni, adapter, adapter));
        Assert.All(new[] { text, omni, adapter }, file => Assert.Equal(0, file.Disposals));
    }

    [Theory]
    [InlineData(0, 2, 1, 2, 10000f, 1e-5f)]
    [InlineData(2, 3, 1, 2, 10000f, 1e-5f)]
    [InlineData(2, 2, 3, 2, 10000f, 1e-5f)]
    [InlineData(2, 2, 1, 3, 10000f, 1e-5f)]
    [InlineData(2, 2, 1, 4, 10000f, 1e-5f)]
    [InlineData(2, 2, 1, 2, float.NaN, 1e-5f)]
    [InlineData(2, 2, 1, 2, 10000f, 0f)]
    public void TextMetadataRejectsInvalidConfiguration(int layers, int queryHeads, int keyValueHeads, int rotary, float ropeBase, float epsilon)
    {
        var parameters = new Phi4TextGraphParameters(layers, queryHeads, keyValueHeads, rotary, ropeBase, epsilon);
        Assert.Throws<InvalidDataException>(() => parameters.Validate(4));
    }

    [Fact]
    public void PackedTokenAndEmbeddingInputsRoundTripThroughModelOwnedTypedCodec()
    {
        var packedToken = Phi4TextInputCodec.EncodeTokenControl(7, 2, 8, 16);
        var token = Phi4TextInputCodec.DecodeTokenControl(packedToken, 8, 16);
        Assert.Equal(7, BinaryPrimitives.ReadInt32LittleEndian(token["token"].Span));
        Assert.Equal(2, BinaryPrimitives.ReadInt32LittleEndian(token["position"].Span));
        float[] values = [1, -2, 0.25f, 0];
        var packedEmbedding = Phi4TextInputCodec.EncodeEmbeddingControl(values, 3, 16);
        var embedding = Phi4TextInputCodec.DecodeEmbeddingControl(packedEmbedding, 4, 16);
        Assert.Equal(3, BinaryPrimitives.ReadInt32LittleEndian(embedding["position"].Span));
        Assert.Equal(values, MemoryMarshal.Cast<byte, float>(embedding["embedding"].Span).ToArray());
        packedEmbedding[4] ^= 1;
        Assert.Equal(values, MemoryMarshal.Cast<byte, float>(embedding["embedding"].Span).ToArray());
    }

    [Fact]
    public void ModelOwnedInputCodecRejectsMalformedPayloadsAndNonfiniteValues()
    {
        Assert.Throws<ArgumentException>(() => Phi4TextInputCodec.DecodeTokenControl(new byte[4], 8, 16));
        Assert.Throws<ArgumentOutOfRangeException>(() => Phi4TextInputCodec.EncodeTokenControl(-1, 0, 8, 16));
        Assert.Throws<ArgumentOutOfRangeException>(() => Phi4TextInputCodec.EncodeTokenControl(8, 0, 8, 16));
        Assert.Throws<ArgumentOutOfRangeException>(() => Phi4TextInputCodec.EncodeTokenControl(0, 16, 8, 16));
        Assert.Throws<ArgumentException>(() => Phi4TextInputCodec.DecodeEmbeddingControl(new byte[4], 4, 16));
        Assert.Throws<ArgumentOutOfRangeException>(() => Phi4TextInputCodec.EncodeEmbeddingControl([1f], -1, 16));
        Assert.Throws<ArgumentException>(() => Phi4TextInputCodec.EncodeEmbeddingControl([float.NaN], 0, 16));
        Assert.Throws<ArgumentException>(() => Phi4TextInputCodec.EncodeEmbeddingControl([float.PositiveInfinity], 0, 16));
    }

    private sealed class File : IModelFile
    {
        private readonly Tensor tensor;
        public File(string name, int[] dimensions)
        {
            tensor = new(this, name, dimensions);
            Names = [name];
            Values = new float[dimensions.Aggregate(1, (count, size) => count * size)];
        }
        public float[] Values { get; }
        public int Disposals { get; private set; }
        public bool FailDispose { get; init; }
        public string Path => "synthetic";
        public IReadOnlyCollection<string> Names { get; }
        public bool TryGet(string name, out IModelTensor value)
        {
            value = tensor;
            return name == tensor.Name;
        }
        public IModelTensor GetRequired(string name) => name == tensor.Name ? tensor : throw new InvalidDataException(name);
        public void Dispose()
        {
            Disposals++;
            if (FailDispose) throw new InvalidOperationException("Synthetic disposal failure.");
        }
    }

    private sealed class Tensor(File file, string name, int[] dimensions) : IModelTensor
    {
        public string Name => name;
        public TensorDataType DataType => TensorDataType.Float32;
        public IReadOnlyList<int> Dimensions => dimensions;
        public ReadOnlySpan<float> FloatValues => file.Disposals == 0 ? file.Values : throw new ObjectDisposedException(nameof(File));
        public ReadOnlySpan<Half> HalfValues => throw new InvalidOperationException();
    }
}
