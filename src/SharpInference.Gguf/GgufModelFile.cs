using SharpInference.Graphs;
using System.Collections.ObjectModel;
using System.IO.MemoryMappedFiles;
using System.Text;

namespace SharpInference.Gguf;

/// <summary>Identifies the GGUF metadata value representation.</summary>
public enum GgufMetadataType : uint
{
    /// <summary>An unsigned 8-bit integer.</summary>
    UInt8 = 0,
    /// <summary>A signed 8-bit integer.</summary>
    Int8 = 1,
    /// <summary>An unsigned 16-bit integer.</summary>
    UInt16 = 2,
    /// <summary>A signed 16-bit integer.</summary>
    Int16 = 3,
    /// <summary>An unsigned 32-bit integer.</summary>
    UInt32 = 4,
    /// <summary>A signed 32-bit integer.</summary>
    Int32 = 5,
    /// <summary>An IEEE 754 single-precision value.</summary>
    Float32 = 6,
    /// <summary>A Boolean value.</summary>
    Bool = 7,
    /// <summary>A length-prefixed UTF-8 string.</summary>
    String = 8,
    /// <summary>An array of values with a common metadata type.</summary>
    Array = 9,
    /// <summary>An unsigned 64-bit integer.</summary>
    UInt64 = 10,
    /// <summary>A signed 64-bit integer.</summary>
    Int64 = 11,
    /// <summary>An IEEE 754 double-precision value.</summary>
    Float64 = 12,
}

/// <summary>Identifies the storage type of a tensor in a GGUF model file.</summary>
public enum GgufModelTensorType : uint
{
    /// <summary>IEEE 754 single-precision elements.</summary>
    Float32 = 0,
    /// <summary>IEEE 754 half-precision elements.</summary>
    Float16 = 1,
    /// <summary>GGML Q4_0 blocks with 4-bit quantized values.</summary>
    Q4_0 = 2,
    /// <summary>GGML Q4_1 blocks with 4-bit quantized values.</summary>
    Q4_1 = 3,
    /// <summary>GGML Q5_0 blocks with 5-bit quantized values.</summary>
    Q5_0 = 6,
    /// <summary>GGML Q5_1 blocks with 5-bit quantized values.</summary>
    Q5_1 = 7,
    /// <summary>GGML Q8_0 blocks with 8-bit quantized values.</summary>
    Q8_0 = 8,
    /// <summary>GGML Q8_1 blocks with 8-bit quantized values.</summary>
    Q8_1 = 9,
    /// <summary>GGML 2-bit K-quant superblocks.</summary>
    Q2_K = 10,
    /// <summary>GGML 3-bit K-quant superblocks.</summary>
    Q3_K = 11,
    /// <summary>GGML 4-bit K-quant superblocks.</summary>
    Q4_K = 12,
    /// <summary>GGML 5-bit K-quant superblocks.</summary>
    Q5_K = 13,
    /// <summary>GGML 6-bit K-quant superblocks.</summary>
    Q6_K = 14,
    /// <summary>GGML 8-bit K-quant superblocks.</summary>
    Q8_K = 15,
    /// <summary>Signed 8-bit integer elements.</summary>
    Int8 = 24,
    /// <summary>Signed 16-bit integer elements.</summary>
    Int16 = 25,
    /// <summary>Signed 32-bit integer elements.</summary>
    Int32 = 26,
    /// <summary>Signed 64-bit integer elements.</summary>
    Int64 = 27,
    /// <summary>IEEE 754 double-precision elements.</summary>
    Float64 = 28,
    /// <summary>Brain floating-point 16-bit elements.</summary>
    BFloat16 = 30,
}

/// <summary>Reads GGUF v3 model metadata and exposes memory-mapped tensor data.</summary>
public sealed unsafe class GgufModelFile : IDisposable
{
    private const uint Magic = 0x46554747;
    private const uint SupportedVersion = 3;
    private static readonly UTF8Encoding Utf8 = new(false, true);
    private readonly MemoryMappedFile mapping;
    private readonly MemoryMappedViewAccessor accessor;
    private readonly byte* pointer;
    private bool disposed;

    private GgufModelFile(
        string path,
        IReadOnlyDictionary<string, object> metadata,
        IReadOnlyDictionary<string, GgufModelTensor> tensors,
        MemoryMappedFile mapping,
        MemoryMappedViewAccessor accessor,
        byte* pointer)
    {
        Path = path;
        Metadata = metadata;
        Tensors = tensors;
        this.mapping = mapping;
        this.accessor = accessor;
        this.pointer = pointer;
    }

    /// <summary>Gets the normalized path of the mapped GGUF file.</summary>
    public string Path { get; }

    /// <summary>Gets the GGUF metadata values keyed by metadata name.</summary>
    public IReadOnlyDictionary<string, object> Metadata { get; }

    /// <summary>Gets the tensor descriptors and mapped tensor views keyed by tensor name.</summary>
    public IReadOnlyDictionary<string, GgufModelTensor> Tensors { get; }

    /// <summary>Opens a GGUF v3 model file and maps its tensor data for read-only access.</summary>
    /// <param name="path">The path to the GGUF model file.</param>
    /// <returns>An opened model file that must be disposed.</returns>
    public static GgufModelFile Open(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using var stream = new FileStream(
            path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20,
            FileOptions.RandomAccess);
        using var reader = new BinaryReader(stream, Utf8, leaveOpen: true);
        if (reader.ReadUInt32() != Magic || reader.ReadUInt32() != SupportedVersion)
        {
            throw new InvalidDataException("Expected a GGUF v3 model file.");
        }

        var tensorCount = CheckedCount(reader.ReadUInt64(), "tensor");
        var metadataCount = CheckedCount(reader.ReadUInt64(), "metadata");

        var metadata = new Dictionary<string, object>(metadataCount, StringComparer.Ordinal);
        for (var index = 0; index < metadataCount; index++)
        {
            var key = ReadString(reader);
            object value;
            try
            {
                value = ReadValue(reader);
            }
            catch (Exception exception) when (exception is InvalidDataException or EndOfStreamException or OverflowException)
            {
                throw new InvalidDataException($"Invalid GGUF metadata '{key}'.", exception);
            }
            if (string.IsNullOrWhiteSpace(key) || !metadata.TryAdd(key, value))
            {
                throw new InvalidDataException($"Empty or duplicate GGUF metadata key '{key}'.");
            }
        }

        var descriptors = new TensorDescriptor[tensorCount];
        var tensorNames = new HashSet<string>(StringComparer.Ordinal);
        for (var index = 0; index < tensorCount; index++)
        {
            var name = ReadString(reader);
            if (string.IsNullOrWhiteSpace(name) || !tensorNames.Add(name))
            {
                throw new InvalidDataException($"Empty or duplicate GGUF tensor name '{name}'.");
            }

            var rank = reader.ReadUInt32();
            if (rank is < 1 or > 8)
            {
                throw new InvalidDataException($"Tensor '{name}' has unsupported rank {rank}.");
            }

            var dimensions = new ulong[rank];
            for (var dimension = 0; dimension < dimensions.Length; dimension++)
            {
                dimensions[dimension] = reader.ReadUInt64();
                if (dimensions[dimension] == 0)
                {
                    throw new InvalidDataException($"Tensor '{name}' has an empty dimension.");
                }
            }
            descriptors[index] = new TensorDescriptor(
                name, (GgufModelTensorType)reader.ReadUInt32(), dimensions, reader.ReadUInt64());
        }

        var alignment = metadata.TryGetValue("general.alignment", out var alignmentValue)
            ? checked((uint)Convert.ToUInt64(alignmentValue))
            : 32;
        if (alignment == 0 || !System.Numerics.BitOperations.IsPow2(alignment))
        {
            throw new InvalidDataException($"Invalid GGUF alignment {alignment}.");
        }

        var dataStart = Align(stream.Position, alignment);
        var fileLength = stream.Length;

        var mapping = MemoryMappedFile.CreateFromFile(
            path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        MemoryMappedViewAccessor? accessor = null;
        byte* pointer = null;
        try
        {
            accessor = mapping.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
            var tensors = new Dictionary<string, GgufModelTensor>(tensorCount, StringComparer.Ordinal);
            foreach (var descriptor in descriptors)
            {
                var byteLength = GetByteLength(descriptor.Type, descriptor.Dimensions);
                var absoluteOffset = checked(dataStart + checked((long)descriptor.Offset));
                if (absoluteOffset < dataStart || byteLength > fileLength - absoluteOffset)
                {
                    throw new InvalidDataException($"Tensor '{descriptor.Name}' exceeds the GGUF file.");
                }

                tensors.Add(descriptor.Name, new GgufModelTensor(
                    descriptor.Name, descriptor.Type, descriptor.Dimensions,
                    absoluteOffset, byteLength, pointer));
            }
            return new GgufModelFile(
                System.IO.Path.GetFullPath(path),
                new ReadOnlyDictionary<string, object>(metadata),
                new ReadOnlyDictionary<string, GgufModelTensor>(tensors),
                mapping, accessor, pointer);
        }
        catch
        {
            if (pointer is not null && accessor is not null)
            {
                accessor.SafeMemoryMappedViewHandle.ReleasePointer();
            }

            accessor?.Dispose();
            mapping.Dispose();
            throw;
        }
    }

    /// <summary>Reads a required metadata value, converting compatible scalar values when necessary.</summary>
    /// <typeparam name="T">The requested metadata value type.</typeparam>
    /// <param name="key">The metadata key.</param>
    /// <returns>The metadata value in the requested type.</returns>
    /// <exception cref="InvalidDataException">The key is absent or its value cannot be converted.</exception>
    public T GetMetadata<T>(string key)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (!Metadata.TryGetValue(key, out var value))
        {
            throw new InvalidDataException($"GGUF metadata '{key}' is missing.");
        }

        if (value is T typed)
        {
            return typed;
        }

        try
        {
            return (T)Convert.ChangeType(value, typeof(T), System.Globalization.CultureInfo.InvariantCulture);
        }
        catch (Exception exception) when (exception is InvalidCastException or FormatException or OverflowException)
        {
            throw new InvalidDataException($"GGUF metadata '{key}' is not a {typeof(T).Name}.", exception);
        }
    }

    /// <summary>Retrieves a required memory-mapped tensor by name.</summary>
    /// <param name="name">The tensor name.</param>
    /// <returns>The tensor descriptor and mapped data view.</returns>
    public GgufModelTensor GetTensor(string name)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return Tensors.TryGetValue(name, out var tensor)
            ? tensor
            : throw new InvalidDataException($"GGUF tensor '{name}' is missing.");
    }

    /// <summary>Releases the file mapping and invalidates all tensor views obtained from it.</summary>
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

    private static object ReadValue(BinaryReader reader)
    {
        var type = (GgufMetadataType)reader.ReadUInt32();
        return type switch
        {
            GgufMetadataType.UInt8 => reader.ReadByte(),
            GgufMetadataType.Int8 => reader.ReadSByte(),
            GgufMetadataType.UInt16 => reader.ReadUInt16(),
            GgufMetadataType.Int16 => reader.ReadInt16(),
            GgufMetadataType.UInt32 => reader.ReadUInt32(),
            GgufMetadataType.Int32 => reader.ReadInt32(),
            GgufMetadataType.Float32 => reader.ReadSingle(),
            GgufMetadataType.Bool => reader.ReadByte() switch
            {
                0 => false,
                1 => true,
                var value => throw new InvalidDataException($"Invalid GGUF Boolean value {value}."),
            },
            GgufMetadataType.String => ReadString(reader),
            GgufMetadataType.Array => ReadArray(reader),
            GgufMetadataType.UInt64 => reader.ReadUInt64(),
            GgufMetadataType.Int64 => reader.ReadInt64(),
            GgufMetadataType.Float64 => reader.ReadDouble(),
            _ => throw new InvalidDataException($"Unsupported GGUF metadata type {(uint)type}."),
        };
    }

    private static Array ReadArray(BinaryReader reader)
    {
        var elementType = (GgufMetadataType)reader.ReadUInt32();
        if (elementType == GgufMetadataType.Array)
        {
            throw new InvalidDataException("Nested GGUF metadata arrays are unsupported.");
        }

        var count = ReadArrayCount(reader);
        return elementType switch
        {
            GgufMetadataType.UInt8 => Read(count, reader.ReadByte),
            GgufMetadataType.Int8 => Read(count, reader.ReadSByte),
            GgufMetadataType.UInt16 => Read(count, reader.ReadUInt16),
            GgufMetadataType.Int16 => Read(count, reader.ReadInt16),
            GgufMetadataType.UInt32 => Read(count, reader.ReadUInt32),
            GgufMetadataType.Int32 => Read(count, reader.ReadInt32),
            GgufMetadataType.Float32 => Read(count, reader.ReadSingle),
            GgufMetadataType.Bool => Read(count, () => reader.ReadByte() switch
            {
                0 => false,
                1 => true,
                var value => throw new InvalidDataException($"Invalid GGUF Boolean value {value}."),
            }),
            GgufMetadataType.String => Read(count, () => ReadString(reader)),
            GgufMetadataType.UInt64 => Read(count, reader.ReadUInt64),
            GgufMetadataType.Int64 => Read(count, reader.ReadInt64),
            GgufMetadataType.Float64 => Read(count, reader.ReadDouble),
            _ => throw new InvalidDataException($"Unsupported GGUF array type {(uint)elementType}."),
        };
    }

    private static T[] Read<T>(int count, Func<T> read)
    {
        var result = new T[count];
        for (var index = 0; index < result.Length; index++)
        {
            result[index] = read();
        }

        return result;
    }

    private static string ReadString(BinaryReader reader)
    {
        var length = CheckedCount(reader.ReadUInt64(), "string byte");
        var bytes = reader.ReadBytes(length);
        if (bytes.Length != length)
        {
            throw new EndOfStreamException("GGUF string is truncated.");
        }

        return Utf8.GetString(bytes);
    }

    private static int CheckedCount(ulong value, string kind) =>
        value <= int.MaxValue
            ? (int)value
            : throw new InvalidDataException($"GGUF {kind} count {value} is too large.");

    private static int ReadArrayCount(BinaryReader reader)
    {
        var value = reader.ReadUInt64();
        if (value <= int.MaxValue)
        {
            return (int)value;
        }

        // Some older community converters emitted a 32-bit array count.
        var legacyCount = (uint)value;
        if (legacyCount > int.MaxValue)
        {
            throw new InvalidDataException($"GGUF array count {value} is too large.");
        }

        reader.BaseStream.Position -= sizeof(uint);
        return (int)legacyCount;
    }

    private static long Align(long value, uint alignment) =>
        checked((value + alignment - 1) & -(long)alignment);

    private static long GetByteLength(GgufModelTensorType type, IReadOnlyList<ulong> dimensions)
    {
        ulong elements = 1;
        foreach (var dimension in dimensions)
        {
            elements = checked(elements * dimension);
        }

        var bytes = type switch
        {
            GgufModelTensorType.Float32 => checked(elements * 4),
            GgufModelTensorType.Float16 or GgufModelTensorType.BFloat16 => checked(elements * 2),
            GgufModelTensorType.Int8 => elements,
            GgufModelTensorType.Int16 => checked(elements * 2),
            GgufModelTensorType.Int32 => checked(elements * 4),
            GgufModelTensorType.Int64 or GgufModelTensorType.Float64 => checked(elements * 8),
            GgufModelTensorType.Q4_0 => BlockBytes(elements, 32, 18),
            GgufModelTensorType.Q4_1 => BlockBytes(elements, 32, 20),
            GgufModelTensorType.Q5_0 => BlockBytes(elements, 32, 22),
            GgufModelTensorType.Q5_1 => BlockBytes(elements, 32, 24),
            GgufModelTensorType.Q8_0 => BlockBytes(elements, 32, 34),
            GgufModelTensorType.Q8_1 => BlockBytes(elements, 32, 36),
            _ => throw new NotSupportedException($"GGUF tensor type '{type}' is not supported."),
        };
        return checked((long)bytes);

        static ulong BlockBytes(ulong elements, ulong blockSize, ulong bytes)
        {
            if (elements % blockSize != 0)
            {
                throw new InvalidDataException(
                    $"Quantized tensor element count {elements} is not divisible by block size {blockSize}.");
            }

            return checked(elements / blockSize * bytes);
        }
    }

    private sealed record TensorDescriptor(
        string Name,
        GgufModelTensorType Type,
        ulong[] Dimensions,
        ulong Offset);
}

/// <summary>Provides a read-only tensor view valid only while its owning model file remains open.</summary>
public sealed unsafe class GgufModelTensor
{
    private readonly byte* filePointer;
    private readonly long offset;
    private readonly long byteLength;

    internal GgufModelTensor(
        string name,
        GgufModelTensorType type,
        ulong[] dimensions,
        long offset,
        long byteLength,
        byte* filePointer)
    {
        Name = name;
        Type = type;
        Dimensions = Array.AsReadOnly(dimensions);
        this.offset = offset;
        this.byteLength = byteLength;
        this.filePointer = filePointer;
    }

    /// <summary>Gets the tensor name in the model file.</summary>
    public string Name { get; }
    /// <summary>Gets the stored element or quantization type.</summary>
    public GgufModelTensorType Type { get; }
    /// <summary>Gets dimensions in GGUF order, with the fastest-varying dimension first.</summary>
    public IReadOnlyList<ulong> Dimensions { get; }
    /// <summary>Gets the stored tensor length in bytes, including quantization metadata.</summary>
    public long ByteLength => byteLength;
    /// <summary>Gets the address of mapped tensor storage; the owning file must remain open.</summary>
    public nint DataAddress => (nint)(filePointer + offset);

    /// <summary>Gets a zero-copy byte view, rejecting tensors larger than a .NET span can represent.</summary>
    public ReadOnlySpan<byte> Bytes =>
        byteLength <= int.MaxValue
            ? new ReadOnlySpan<byte>(filePointer + offset, checked((int)byteLength))
            : throw new NotSupportedException(
                $"Tensor '{Name}' is larger than the maximum .NET span length.");

    /// <summary>Gets a zero-copy half-precision view; requires Float16 storage.</summary>
    public ReadOnlySpan<Half> HalfValues =>
        Type == GgufModelTensorType.Float16
            ? System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Half>(Bytes)
            : throw new InvalidOperationException($"Tensor '{Name}' is not Float16.");

    /// <summary>Gets a zero-copy single-precision view; requires Float32 storage.</summary>
    public ReadOnlySpan<float> FloatValues =>
        Type == GgufModelTensorType.Float32
            ? System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(Bytes)
            : throw new InvalidOperationException($"Tensor '{Name}' is not Float32.");

    /// <summary>Reads one Float16 element and converts it to single precision.</summary>
    /// <param name="index">The zero-based linear element index.</param>
    /// <returns>The selected value as a single-precision number.</returns>
    public float GetHalfValue(int index)
    {
        if (Type != GgufModelTensorType.Float16)
        {
            throw new InvalidOperationException($"Tensor '{Name}' is not Float16.");
        }

        var count = byteLength / sizeof(ushort);
        if ((uint)index >= (ulong)count)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        return (float)((Half*)(filePointer + offset))[index];
    }

    /// <summary>Reads one Float32 element without creating a span for the entire tensor.</summary>
    /// <param name="index">The zero-based linear element index.</param>
    /// <returns>The selected single-precision value.</returns>
    public float GetFloatValue(int index)
    {
        if (Type != GgufModelTensorType.Float32)
        {
            throw new InvalidOperationException($"Tensor '{Name}' is not Float32.");
        }

        var count = byteLength / sizeof(float);
        if ((uint)index >= (ulong)count)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        return ((float*)(filePointer + offset))[index];
    }
}
