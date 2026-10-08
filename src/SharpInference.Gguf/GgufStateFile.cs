using System.Text;

namespace SharpInference.Gguf;

/// <summary>Unquantized GGML tensor types supported by state files.</summary>
public enum GgufTensorType : uint
{
    Float32 = 0,
    Float16 = 1,
    Int8 = 24,
    Int16 = 25,
    Int32 = 26,
    Int64 = 27,
    Float64 = 28,
}

/// <summary>A named tensor in GGUF dimension order, with its raw little-endian bytes.</summary>
public sealed class GgufStateTensor
{
    /// <summary>Creates a state tensor with copied dimensions and supplied raw data.</summary>
    /// <param name="name">The tensor name.</param>
    /// <param name="type">The unquantized tensor storage type.</param>
    /// <param name="dimensions">The tensor dimensions in GGUF order.</param>
    /// <param name="data">The tensor's raw little-endian bytes.</param>
    public GgufStateTensor(string name, GgufTensorType type, IEnumerable<ulong> dimensions, ReadOnlyMemory<byte> data)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(dimensions);
        Name = name;
        Type = type;
        Dimensions = Array.AsReadOnly(dimensions.ToArray());
        Data = data;
    }

    /// <summary>Gets the tensor name.</summary>
    public string Name { get; }

    /// <summary>Gets the tensor storage type.</summary>
    public GgufTensorType Type { get; }

    /// <summary>Gets the tensor dimensions in GGUF order.</summary>
    public IReadOnlyList<ulong> Dimensions { get; }

    /// <summary>Gets the raw little-endian tensor bytes.</summary>
    public ReadOnlyMemory<byte> Data { get; }
}

/// <summary>A state schema name and the exact set of persisted tensors.</summary>
public sealed class GgufState
{
    /// <summary>Creates a named state containing the supplied tensors.</summary>
    /// <param name="schemaName">The state schema identifier.</param>
    /// <param name="tensors">The tensors persisted by the state.</param>
    public GgufState(string schemaName, IEnumerable<GgufStateTensor> tensors)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(schemaName);
        ArgumentNullException.ThrowIfNull(tensors);
        SchemaName = schemaName;
        Tensors = Array.AsReadOnly(tensors.ToArray());
    }

    /// <summary>Gets the state schema identifier.</summary>
    public string SchemaName { get; }

    /// <summary>Gets the tensors persisted by the state.</summary>
    public IReadOnlyList<GgufStateTensor> Tensors { get; }
}

/// <summary>Reads and writes state-only GGUF v3 files without owning the supplied stream.</summary>
public static class GgufStateFile
{
    private const uint Magic = 0x46554747;
    private const uint Version = 3;
    private const uint StringValueType = 8;
    private const int Alignment = 32;
    private const string ArchitectureKey = "general.architecture";
    private const string Architecture = "sharpinference";
    private const string SchemaKey = "sharpinference.state_schema.name";
    private static readonly UTF8Encoding Utf8 = new(false, true);

    /// <summary>Writes a state as a GGUF v3 file at the stream's current position.</summary>
    /// <param name="stream">A writable, seekable stream that remains open.</param>
    /// <param name="state">The state to serialize.</param>
    public static void Write(Stream stream, GgufState state)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(state);
        if (!stream.CanWrite || !stream.CanSeek)
            throw new ArgumentException("The stream must be writable and seekable.", nameof(stream));
        var start = stream.Position;

        var names = new HashSet<string>(StringComparer.Ordinal);
        var offsets = new ulong[state.Tensors.Count];
        ulong end = 0;
        for (var i = 0; i < state.Tensors.Count; i++)
        {
            var tensor = state.Tensors[i];
            ArgumentNullException.ThrowIfNull(tensor);
            if (!names.Add(tensor.Name)) throw new ArgumentException($"Duplicate tensor name '{tensor.Name}'.", nameof(state));
            var bytes = TensorLength(tensor.Type, tensor.Dimensions);
            if (bytes != (ulong)tensor.Data.Length)
                throw new ArgumentException($"Tensor '{tensor.Name}' has {tensor.Data.Length} bytes, expected {bytes}.", nameof(state));
            offsets[i] = Align(end);
            end = checked(offsets[i] + bytes);
        }

        using var writer = new BinaryWriter(stream, Utf8, leaveOpen: true);
        writer.Write(Magic);
        writer.Write(Version);
        writer.Write((ulong)state.Tensors.Count);
        writer.Write(2UL);
        WriteStringMetadata(writer, ArchitectureKey, Architecture);
        WriteStringMetadata(writer, SchemaKey, state.SchemaName);
        for (var i = 0; i < state.Tensors.Count; i++)
        {
            var tensor = state.Tensors[i];
            WriteString(writer, tensor.Name);
            writer.Write((uint)tensor.Dimensions.Count);
            foreach (var dimension in tensor.Dimensions) writer.Write(dimension);
            writer.Write((uint)tensor.Type);
            writer.Write(offsets[i]);
        }

        WritePadding(writer, (int)((Alignment - (stream.Position - start) % Alignment) % Alignment));
        ulong position = 0;
        for (var i = 0; i < state.Tensors.Count; i++)
        {
            WritePadding(writer, checked((int)(offsets[i] - position)));
            writer.Write(state.Tensors[i].Data.Span);
            position = checked(offsets[i] + (ulong)state.Tensors[i].Data.Length);
        }
        writer.Flush();
    }

    /// <summary>Reads a GGUF v3 state from the stream's current position.</summary>
    /// <param name="stream">A readable, seekable stream that remains open.</param>
    /// <returns>The deserialized state.</returns>
    public static GgufState Read(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanRead || !stream.CanSeek) throw new ArgumentException("The stream must be readable and seekable.", nameof(stream));
        try
        {
            using var reader = new BinaryReader(stream, Utf8, leaveOpen: true);
            var start = stream.Position;
            if (reader.ReadUInt32() != Magic || reader.ReadUInt32() != Version)
                throw new InvalidDataException("Expected a GGUF v3 header.");
            var tensorCount = reader.ReadUInt64();
            var metadataCount = reader.ReadUInt64();
            if (tensorCount > (ulong)(stream.Length - stream.Position) / 24 ||
                metadataCount > (ulong)(stream.Length - stream.Position) / 16)
                throw new InvalidDataException("GGUF entry count exceeds file length.");

            string? architecture = null, schema = null;
            var keys = new HashSet<string>(StringComparer.Ordinal);
            for (ulong i = 0; i < metadataCount; i++)
            {
                var key = ReadString(reader);
                if (!keys.Add(key)) throw new InvalidDataException($"Duplicate GGUF metadata key '{key}'.");
                if (reader.ReadUInt32() != StringValueType)
                    throw new InvalidDataException($"Metadata '{key}' must be a GGUF string.");
                var value = ReadString(reader);
                if (key == ArchitectureKey) architecture = value;
                else if (key == SchemaKey) schema = value;
                else throw new InvalidDataException($"Unexpected state metadata '{key}'.");
            }
            if (architecture != Architecture || string.IsNullOrWhiteSpace(schema))
                throw new InvalidDataException("GGUF state requires sharpinference architecture and a schema name.");

            var count = checked((int)tensorCount);
            var entries = new (string Name, GgufTensorType Type, ulong[] Dimensions, ulong Offset, ulong Length)[count];
            var names = new HashSet<string>(StringComparer.Ordinal);
            for (var i = 0; i < count; i++)
            {
                var name = ReadString(reader);
                if (string.IsNullOrWhiteSpace(name) || !names.Add(name))
                    throw new InvalidDataException("Empty or duplicate GGUF tensor name.");
                var dimensionCount = reader.ReadUInt32();
                if (dimensionCount is < 1 or > 4) throw new InvalidDataException("Unsupported GGUF tensor dimension count.");
                var dimensions = new ulong[dimensionCount];
                for (var j = 0; j < dimensions.Length; j++) dimensions[j] = reader.ReadUInt64();
                var type = (GgufTensorType)reader.ReadUInt32();
                var offset = reader.ReadUInt64();
                ulong length;
                try { length = TensorLength(type, dimensions); }
                catch (ArgumentException exception) { throw new InvalidDataException($"Invalid tensor '{name}'.", exception); }
                catch (OverflowException exception) { throw new InvalidDataException($"Tensor '{name}' is too large.", exception); }
                entries[i] = (name, type, dimensions, offset, length);
            }

            var headerLength = stream.Position - start;
            var dataStart = checked(start + (long)Align((ulong)headerLength));
            if (dataStart > stream.Length) throw new InvalidDataException("Truncated GGUF header padding.");
            CheckZeros(reader, dataStart - stream.Position);
            ulong expected = 0;
            var tensors = new GgufStateTensor[count];
            for (var i = 0; i < count; i++)
            {
                var entry = entries[i];
                if (entry.Offset != Align(expected) || entry.Length > int.MaxValue ||
                    entry.Offset > (ulong)(stream.Length - dataStart) ||
                    entry.Length > (ulong)(stream.Length - dataStart) - entry.Offset)
                    throw new InvalidDataException($"Invalid GGUF tensor offset or data length for '{entry.Name}'.");
                CheckZeros(reader, checked((long)(entry.Offset - expected)));
                var data = reader.ReadBytes((int)entry.Length);
                if (data.Length != (int)entry.Length) throw new InvalidDataException($"Truncated tensor '{entry.Name}'.");
                tensors[i] = new GgufStateTensor(entry.Name, entry.Type, entry.Dimensions, data);
                expected = checked(entry.Offset + entry.Length);
            }
            if (stream.Position != stream.Length)
                throw new InvalidDataException("GGUF tensor data length does not match the file length.");
            return new GgufState(schema, tensors);
        }
        catch (EndOfStreamException exception)
        {
            throw new InvalidDataException("Truncated GGUF state file.", exception);
        }
        catch (OverflowException exception)
        {
            throw new InvalidDataException("Invalid GGUF size or offset.", exception);
        }
        catch (DecoderFallbackException exception)
        {
            throw new InvalidDataException("Invalid GGUF UTF-8 string.", exception);
        }
    }

    private static ulong TensorLength(GgufTensorType type, IReadOnlyList<ulong> dimensions)
    {
        if (dimensions.Count is < 1 or > 4 || dimensions.Any(static d => d == 0))
            throw new ArgumentException("GGUF tensors require one to four positive dimensions.");
        var size = type switch
        {
            GgufTensorType.Int8 => 1UL,
            GgufTensorType.Float16 or GgufTensorType.Int16 => 2UL,
            GgufTensorType.Float32 or GgufTensorType.Int32 => 4UL,
            GgufTensorType.Float64 or GgufTensorType.Int64 => 8UL,
            _ => throw new ArgumentException($"Unsupported GGML tensor type {(uint)type}."),
        };
        foreach (var dimension in dimensions) size = checked(size * dimension);
        return size;
    }

    private static ulong Align(ulong size) => checked((size + Alignment - 1) / Alignment * Alignment);

    private static void WriteString(BinaryWriter writer, string value)
    {
        var bytes = Utf8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    private static string ReadString(BinaryReader reader)
    {
        var size = reader.ReadUInt64();
        if (size > int.MaxValue || size > (ulong)(reader.BaseStream.Length - reader.BaseStream.Position))
            throw new InvalidDataException("GGUF string exceeds remaining file length.");
        return Utf8.GetString(reader.ReadBytes((int)size));
    }

    private static void WriteStringMetadata(BinaryWriter writer, string key, string value)
    {
        WriteString(writer, key);
        writer.Write(StringValueType);
        WriteString(writer, value);
    }

    private static void WritePadding(BinaryWriter writer, int count)
    {
        if (count > 0) writer.Write(new byte[count]);
    }

    private static void CheckZeros(BinaryReader reader, long count)
    {
        while (count-- > 0)
            if (reader.ReadByte() != 0) throw new InvalidDataException("Nonzero GGUF alignment padding.");
    }
}
