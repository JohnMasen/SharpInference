using System.Security.Cryptography;
using System.Text;

namespace SharpInference.Vm;

public static class VmStateStream
{
    private static readonly byte[] Magic = "SIVMSTATE"u8.ToArray();
    private static readonly Encoding TextEncoding = new UTF8Encoding(false, true);

    public static void Export(Stream destination, VmProgram program, VmExecutionLease access, string contextId)
    {
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentNullException.ThrowIfNull(program);
        ArgumentNullException.ThrowIfNull(access);
        ArgumentException.ThrowIfNullOrWhiteSpace(contextId);
        if (!access.IsStateAccess || !ReferenceEquals(program, access.Program) || !access.StateValid)
            throw new InvalidOperationException("State export requires a valid lease for this program.");
        if (!destination.CanWrite)
            throw new ArgumentException("The state destination must be writable.", nameof(destination));
        using var writer = new BinaryWriter(destination, TextEncoding, leaveOpen: true);
        writer.Write(Magic);
        writer.Write(1);
        WriteText(writer, contextId);
        WriteText(writer, program.State.Schema);
        writer.Write(program.State.Version);
        writer.Write(program.State.Entries.Count);
        var slots = program.Slots.ToDictionary(slot => slot.Id, StringComparer.Ordinal);
        var chunk = new byte[64 * 1024];
        foreach (var entry in program.State.Entries)
        {
            var tensor = slots[entry.Slot].Tensor;
            WriteText(writer, entry.Name);
            writer.Write((int)tensor.ElementType);
            writer.Write(tensor.Dimensions.Count);
            foreach (var size in tensor.Dimensions) writer.Write(size);
            writer.Write(tensor.ByteLength);
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            for (ulong offset = 0; offset < tensor.ByteLength;)
            {
                var count = (int)Math.Min((ulong)chunk.Length, tensor.ByteLength - offset);
                access.Read(entry.Slot, offset, chunk.AsSpan(0, count));
                writer.Write(chunk.AsSpan(0, count));
                hash.AppendData(chunk.AsSpan(0, count));
                offset += (ulong)count;
            }
            writer.Write(hash.GetHashAndReset());
        }
    }

    public static void Import(Stream source, VmProgram program, VmExecutionLease access,
        string contextId, ulong maximumStagingBytes)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(program);
        ArgumentNullException.ThrowIfNull(access);
        ArgumentException.ThrowIfNullOrWhiteSpace(contextId);
        if (!access.IsStateAccess || !ReferenceEquals(program, access.Program))
            throw new InvalidOperationException("State import requires a lease for this program.");
        if (!source.CanRead)
            throw new ArgumentException("The state source must be readable.", nameof(source));
        var slots = program.Slots.ToDictionary(slot => slot.Id, StringComparer.Ordinal);
        var staged = new List<(string Slot, byte[] Bytes)>();
        using var reader = new BinaryReader(source, TextEncoding, leaveOpen: true);
        try
        {
            if (!ReadBytes(reader, Magic.Length).AsSpan().SequenceEqual(Magic) ||
                reader.ReadInt32() != 1 || ReadText(reader) != contextId ||
                ReadText(reader) != program.State.Schema || reader.ReadInt32() != program.State.Version ||
                reader.ReadInt32() != program.State.Entries.Count)
                throw new InvalidDataException("State header, context or schema is incompatible.");
            ulong total = 0;
            foreach (var entry in program.State.Entries)
            {
                var slot = slots[entry.Slot];
                var tensor = slot.Tensor;
                if (!access.CanRestoreState(entry.Slot) || ReadText(reader) != entry.Name ||
                    reader.ReadInt32() != (int)tensor.ElementType ||
                    reader.ReadInt32() != tensor.Dimensions.Count)
                    throw new InvalidDataException($"State entry '{entry.Name}' is incompatible.");
                foreach (var size in tensor.Dimensions)
                    if (reader.ReadInt32() != size)
                        throw new InvalidDataException($"State entry '{entry.Name}' has an incompatible shape.");
                if (reader.ReadUInt64() != tensor.ByteLength ||
                    tensor.ByteLength > int.MaxValue || tensor.ByteLength > maximumStagingBytes - total)
                    throw new InvalidDataException("State payload length exceeds its descriptor or staging budget.");
                total += tensor.ByteLength;
                var bytes = ReadBytes(reader, (int)tensor.ByteLength);
                if (!CryptographicOperations.FixedTimeEquals(SHA256.HashData(bytes), ReadBytes(reader, 32)))
                    throw new InvalidDataException($"State entry '{entry.Name}' failed integrity validation.");
                staged.Add((entry.Slot, bytes));
            }
        }
        catch (Exception error) when (error is EndOfStreamException or DecoderFallbackException)
        {
            throw new InvalidDataException("Truncated or malformed state stream.", error);
        }
        try
        {
            foreach (var (slot, bytes) in staged)
                access.WriteState(slot, bytes);
            access.ValidateState();
        }
        catch
        {
            access.InvalidateState();
            throw;
        }
    }

    private static byte[] ReadBytes(BinaryReader reader, int count)
    {
        var bytes = reader.ReadBytes(count);
        if (bytes.Length != count)
            throw new EndOfStreamException();
        return bytes;
    }

    private static string ReadText(BinaryReader reader)
    {
        var length = reader.ReadInt32();
        if (length is < 1 or > 4096)
            throw new InvalidDataException("Invalid state text length.");
        return TextEncoding.GetString(ReadBytes(reader, length));
    }

    private static void WriteText(BinaryWriter writer, string text)
    {
        var bytes = TextEncoding.GetBytes(text);
        if (bytes.Length is < 1 or > 4096)
            throw new InvalidDataException("State identifiers must contain 1 to 4096 UTF-8 bytes.");
        writer.Write(bytes.Length);
        writer.Write(bytes);
    }
}
