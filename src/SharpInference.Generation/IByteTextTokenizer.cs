namespace SharpInference;

/// <summary>Exposes a token vocabulary whose output pieces represent UTF-8 bytes.</summary>
public interface IByteTextTokenizer
{
    /// <summary>Gets the vocabulary's valid token identifiers.</summary>
    IReadOnlyList<int> TokenIds { get; }
    /// <summary>Encodes text into an ordered sequence of vocabulary tokens.</summary>
    /// <param name="text">The input text.</param>
    /// <returns>The encoded token identifiers.</returns>
    IReadOnlyList<int> Encode(string text);
    /// <summary>Returns the UTF-8 bytes represented by one token.</summary>
    /// <param name="token">The vocabulary token identifier.</param>
    /// <returns>The token's decoded byte piece.</returns>
    ReadOnlyMemory<byte> DecodeBytes(int token);
}
