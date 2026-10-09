using System.Collections.ObjectModel;
using System.IO.MemoryMappedFiles;
using Microsoft.Win32.SafeHandles;

namespace SharpInference;

/// <summary>Reads supported GGML RWKV model files through a read-only memory mapping.</summary>
public sealed unsafe class GgmlModelFile : IModelFile
{
    private const uint Magic = 0x67676d66;
    private const uint Version = 101;
    private const int HeaderSize = 6 * sizeof(uint);
    private readonly MemoryMappedFile mapping;
    private readonly MemoryMappedViewAccessor accessor;
    private readonly byte* data;
    private readonly Dictionary<string, IModelTensor> tensors;
    private bool disposed;

    private GgmlModelFile(
        string path,
        MemoryMappedFile mapping,
        MemoryMappedViewAccessor accessor,
        byte* data,
        int vocabularySize,
        int embeddingSize,
        int layerCount,
        Dictionary<string, IModelTensor> tensors)
    {
        Path = path;
        this.mapping = mapping;
        this.accessor = accessor;
        this.data = data;
        VocabularySize = vocabularySize;
        EmbeddingSize = embeddingSize;
        LayerCount = layerCount;
        this.tensors = tensors;
        Names = new ReadOnlyCollection<string>(tensors.Keys.OrderBy(static name => name, StringComparer.Ordinal).ToArray());
    }

    /// <summary>Gets the vocabulary size stored in the GGML header.</summary>
    public int VocabularySize { get; }

    /// <summary>Gets the embedding size stored in the GGML header.</summary>
    public int EmbeddingSize { get; }

    /// <summary>Gets the layer count stored in the GGML header.</summary>
    public int LayerCount { get; }

    /// <summary>Gets the path of the mapped model file.</summary>
    public string Path { get; }

    /// <summary>Gets the names of tensors found in the file, sorted ordinally.</summary>
    public IReadOnlyCollection<string> Names { get; }

    /// <summary>Validates the fixed GGML header without retaining a file mapping.</summary>
    /// <param name="path">The path to the model file.</param>
    public static void ValidateHeader(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("The GGML model file does not exist.", path);
        }

        Span<byte> header = stackalloc byte[HeaderSize];
        using var stream = File.OpenRead(path);
        try
        {
            stream.ReadExactly(header);
        }
        catch (EndOfStreamException exception)
        {
            throw new InvalidDataException("The GGML file ends unexpectedly while reading its header.", exception);
        }

        var offset = 0;
        var magic = ReadHeaderUInt32(header, ref offset);
        var version = ReadHeaderUInt32(header, ref offset);
        _ = CheckedInt(ReadHeaderUInt32(header, ref offset), "vocabulary size");
        _ = CheckedInt(ReadHeaderUInt32(header, ref offset), "embedding size");
        _ = CheckedInt(ReadHeaderUInt32(header, ref offset), "layer count");
        var defaultType = (RwkvTensorDataType)ReadHeaderUInt32(header, ref offset);
        ValidateHeaderFields(magic, version, defaultType);
    }

    /// <summary>Opens and memory-maps a supported GGML model file for read-only tensor access.</summary>
    /// <param name="path">The path to the model file.</param>
    /// <returns>A mapped model file that must be disposed after use.</returns>
    public static GgmlModelFile Open(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("The GGML model file does not exist.", path);
        }

        var mapping = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        MemoryMappedViewAccessor? accessor = null;
        byte* data = null;
        try
        {
            accessor = mapping.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref data);
            var length = new FileInfo(path).Length;
            var offset = 0L;
            var magic = ReadUInt32(data, ref offset, length);
            var version = ReadUInt32(data, ref offset, length);
            var vocabularySize = CheckedInt(ReadUInt32(data, ref offset, length), "vocabulary size");
            var embeddingSize = CheckedInt(ReadUInt32(data, ref offset, length), "embedding size");
            var layerCount = CheckedInt(ReadUInt32(data, ref offset, length), "layer count");
            var defaultType = (RwkvTensorDataType)ReadUInt32(data, ref offset, length);

            ValidateHeaderFields(magic, version, defaultType);

            var tensors = new Dictionary<string, IModelTensor>(StringComparer.Ordinal);
            while (offset < length)
            {
                var dimensionCount = CheckedInt(ReadUInt32(data, ref offset, length), "tensor dimension count");
                var nameLength = CheckedInt(ReadUInt32(data, ref offset, length), "tensor name length");
                var type = (RwkvTensorDataType)ReadUInt32(data, ref offset, length);
                if (dimensionCount is < 1 or > 3 || nameLength <= 0 || type is not RwkvTensorDataType.Float32 and not RwkvTensorDataType.Float16)
                {
                    throw new InvalidDataException($"Unsupported GGML tensor metadata at byte {offset}.");
                }

                var dimensions = new int[dimensionCount];
                long elementCount = 1;
                for (var i = 0; i < dimensionCount; i++)
                {
                    dimensions[i] = CheckedInt(ReadUInt32(data, ref offset, length), "tensor dimension");
                    elementCount = checked(elementCount * dimensions[i]);
                }

                EnsureRange(offset, nameLength, length);
                var name = System.Text.Encoding.UTF8.GetString(new ReadOnlySpan<byte>(data + offset, nameLength));
                offset += nameLength;
                var bytes = checked(elementCount * (type == RwkvTensorDataType.Float32 ? sizeof(float) : sizeof(Half)));
                EnsureRange(offset, bytes, length);
                if (!tensors.TryAdd(name, new MappedTensor(name, type, dimensions, data + offset, elementCount)))
                {
                    throw new InvalidDataException($"The GGML file contains the tensor '{name}' more than once.");
                }

                offset += bytes;
            }

            return new GgmlModelFile(path, mapping, accessor, data, vocabularySize, embeddingSize, layerCount, tensors);
        }
        catch
        {
            if (data is not null && accessor is not null)
            {
                accessor.SafeMemoryMappedViewHandle.ReleasePointer();
            }

            accessor?.Dispose();
            mapping.Dispose();
            throw;
        }
    }

    /// <summary>Attempts to retrieve a tensor by name.</summary>
    /// <param name="name">The tensor name to find.</param>
    /// <param name="tensor">The matching tensor when found.</param>
    /// <returns><see langword="true"/> when the tensor exists; otherwise, <see langword="false"/>.</returns>
    public bool TryGet(string name, out IModelTensor tensor) => tensors.TryGetValue(name, out tensor!);

    /// <summary>Retrieves a tensor by name.</summary>
    /// <param name="name">The required tensor name.</param>
    /// <returns>The matching tensor.</returns>
    /// <exception cref="InvalidDataException">The tensor is not present in the file.</exception>
    public IModelTensor GetRequired(string name) =>
        tensors.TryGetValue(name, out var tensor)
            ? tensor
            : throw new InvalidDataException($"The GGML model is missing required tensor '{name}'.");

    /// <summary>Releases the memory mapping and view accessor held by this model file.</summary>
    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        accessor.SafeMemoryMappedViewHandle.ReleasePointer();
        accessor.Dispose();
        mapping.Dispose();
        disposed = true;
    }

    private static uint ReadUInt32(byte* data, ref long offset, long length)
    {
        EnsureRange(offset, sizeof(uint), length);
        var value = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(new ReadOnlySpan<byte>(data + offset, sizeof(uint)));
        offset += sizeof(uint);
        return value;
    }

    private static int CheckedInt(uint value, string field) =>
        value <= int.MaxValue ? (int)value : throw new InvalidDataException($"Invalid {field}: {value}.");

    private static uint ReadHeaderUInt32(ReadOnlySpan<byte> header, ref int offset)
    {
        var value = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(header[offset..]);
        offset += sizeof(uint);
        return value;
    }

    private static void ValidateHeaderFields(uint magic, uint version, RwkvTensorDataType defaultType)
    {
        if (magic != Magic || version != Version)
        {
            throw new InvalidDataException($"Unsupported GGML header: magic=0x{magic:X8}, version={version}.");
        }

        if (defaultType is not RwkvTensorDataType.Float32 and not RwkvTensorDataType.Float16)
        {
            throw new InvalidDataException($"Unsupported default tensor type {(uint)defaultType}.");
        }
    }

    private static void EnsureRange(long offset, long count, long length)
    {
        if (offset < 0 || count < 0 || offset > length - count)
        {
            throw new InvalidDataException("The GGML file ends unexpectedly.");
        }
    }

    private sealed unsafe class MappedTensor : IModelTensor
    {
        private readonly byte* values;
        private readonly int elementCount;
        private float[]? convertedFloatValues;

        /// <summary>Creates a tensor view over bytes in the mapped GGML file.</summary>
        /// <exception cref="InvalidDataException">The tensor contains more elements than the managed runtime can index.</exception>
        public MappedTensor(string name, RwkvTensorDataType dataType, int[] dimensions, byte* values, long elementCount)
        {
            if (elementCount > int.MaxValue)
            {
                throw new InvalidDataException($"Tensor '{name}' is too large for the managed runtime.");
            }

            Name = name;
            DataType = dataType;
            Dimensions = dimensions;
            this.values = values;
            this.elementCount = (int)elementCount;
        }

        /// <summary>Gets the tensor name.</summary>
        public string Name { get; }

        /// <summary>Gets the tensor data type.</summary>
        public RwkvTensorDataType DataType { get; }

        /// <summary>Gets the tensor dimensions from the file metadata.</summary>
        public IReadOnlyList<int> Dimensions { get; }

        /// <summary>Gets float32 values, converting float16 mapped values lazily when necessary.</summary>
        public ReadOnlySpan<float> FloatValues
        {
            get
            {
                if (DataType == RwkvTensorDataType.Float32)
                {
                    return new ReadOnlySpan<float>((float*)values, elementCount);
                }

                return GetOrCreateFloatValues();
            }
        }

        /// <summary>Gets the mapped float16 values, or an empty span for other data types.</summary>
        public ReadOnlySpan<Half> HalfValues =>
            DataType == RwkvTensorDataType.Float16
                ? new ReadOnlySpan<Half>((Half*)values, elementCount)
                : ReadOnlySpan<Half>.Empty;

        private ReadOnlySpan<float> GetOrCreateFloatValues()
        {
            var result = convertedFloatValues;
            if (result is null)
            {
                var source = HalfValues;
                result = new float[source.Length];
                for (var index = 0; index < source.Length; index++)
                {
                    result[index] = (float)source[index];
                }

                Interlocked.CompareExchange(ref convertedFloatValues, result, null);
                result = convertedFloatValues!;
            }

            return result;
        }
    }
}
