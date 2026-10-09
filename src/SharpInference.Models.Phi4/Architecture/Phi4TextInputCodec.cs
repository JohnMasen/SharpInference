using System.Buffers.Binary;

namespace SharpInference.Architectures.Phi4;

public enum Phi4TextInputKind
{
    Token,
    Embedding,
}

public static class Phi4TextInputCodec
{
    public static IReadOnlyDictionary<string, ReadOnlyMemory<byte>> DecodeTokenControl(ReadOnlySpan<byte> control,
        int vocabularySize, int maximumContext)
    {
        RequireLimit(vocabularySize, nameof(vocabularySize));
        RequireLimit(maximumContext, nameof(maximumContext));
        if (control.Length != 2 * sizeof(int)) throw new ArgumentException("Token control requires exactly two Int32 values.", nameof(control));
        var token = BinaryPrimitives.ReadInt32LittleEndian(control);
        var position = BinaryPrimitives.ReadInt32LittleEndian(control[sizeof(int)..]);
        ValidateToken(token, vocabularySize, nameof(control));
        ValidatePosition(position, maximumContext, nameof(control));
        return new Dictionary<string, ReadOnlyMemory<byte>>
        {
            ["token"] = Int32(token), ["position"] = Int32(position),
        };
    }

    public static IReadOnlyDictionary<string, ReadOnlyMemory<byte>> DecodeEmbeddingControl(ReadOnlySpan<byte> control,
        int embeddingWidth, int maximumContext)
    {
        RequireLimit(embeddingWidth, nameof(embeddingWidth));
        RequireLimit(maximumContext, nameof(maximumContext));
        if (control.Length != checked((embeddingWidth + 1) * sizeof(float)))
            throw new ArgumentException("Embedding control requires one Int32 position followed by the complete FP32 embedding.", nameof(control));
        var position = BinaryPrimitives.ReadInt32LittleEndian(control);
        ValidatePosition(position, maximumContext, nameof(control));
        for (var index = 0; index < embeddingWidth; index++)
            if (!float.IsFinite(BinaryPrimitives.ReadSingleLittleEndian(control.Slice((index + 1) * sizeof(float), sizeof(float)))))
                throw new ArgumentException("Embedding values must be finite.", nameof(control));
        return new Dictionary<string, ReadOnlyMemory<byte>>
        {
            ["position"] = Int32(position), ["embedding"] = control[sizeof(int)..].ToArray(),
        };
    }

    public static byte[] EncodeTokenControl(int token, int position, int vocabularySize, int maximumContext)
    {
        RequireLimit(vocabularySize, nameof(vocabularySize));
        RequireLimit(maximumContext, nameof(maximumContext));
        ValidateToken(token, vocabularySize, nameof(token));
        ValidatePosition(position, maximumContext, nameof(position));
        var bytes = new byte[2 * sizeof(int)];
        BinaryPrimitives.WriteInt32LittleEndian(bytes, token);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(sizeof(int)), position);
        return bytes;
    }

    public static byte[] EncodeEmbeddingControl(ReadOnlySpan<float> embedding, int position, int maximumContext)
    {
        RequireLimit(embedding.Length, nameof(embedding));
        RequireLimit(maximumContext, nameof(maximumContext));
        ValidatePosition(position, maximumContext, nameof(position));
        foreach (var value in embedding)
            if (!float.IsFinite(value)) throw new ArgumentException("Embedding values must be finite.", nameof(embedding));
        var bytes = new byte[checked((embedding.Length + 1) * sizeof(float))];
        BinaryPrimitives.WriteInt32LittleEndian(bytes, position);
        for (var index = 0; index < embedding.Length; index++)
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan((index + 1) * sizeof(float)), embedding[index]);
        return bytes;
    }

    private static byte[] Int32(int value)
    {
        var bytes = new byte[sizeof(int)];
        BinaryPrimitives.WriteInt32LittleEndian(bytes, value);
        return bytes;
    }

    private static void ValidateToken(int token, int vocabularySize, string name)
    {
        if ((uint)token >= (uint)vocabularySize) throw new ArgumentOutOfRangeException(name, "Token is outside the vocabulary.");
    }

    private static void ValidatePosition(int position, int maximumContext, string name)
    {
        if ((uint)position >= (uint)maximumContext) throw new ArgumentOutOfRangeException(name, "Position is outside the context.");
    }

    private static void RequireLimit(int value, string name)
    {
        if (value <= 0) throw new ArgumentOutOfRangeException(name);
    }
}
