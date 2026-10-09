namespace SharpInference;

/// <summary>Exposes a token vocabulary whose output pieces represent UTF-8 bytes.</summary>
public interface IByteTextTokenizer
{
    IReadOnlyList<int> TokenIds { get; }
    IReadOnlyList<int> Encode(string text);
    ReadOnlyMemory<byte> DecodeBytes(int token);
}
