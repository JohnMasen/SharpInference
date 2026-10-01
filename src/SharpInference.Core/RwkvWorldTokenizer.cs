using System.Text;

namespace SharpInference;

public sealed class RwkvWorldTokenizer
{
    private const string BundledVocabularyResourceName = "SharpInference.Resources.rwkv_vocab_v20230424.txt";
    private readonly Dictionary<int, byte[]> tokens;
    private readonly int[] tokenIds;
    private readonly TrieNode root = new();

    private RwkvWorldTokenizer(Dictionary<int, byte[]> tokens)
    {
        this.tokens = tokens;
        tokenIds = tokens.Keys.Order().ToArray();
        foreach (var (id, value) in tokens)
        {
            var node = root;
            foreach (var valueByte in value)
            {
                node = node.Children[valueByte] ??= new TrieNode();
            }

            // The official World tokenizer builds a byte-to-id dictionary first, so a
            // later vocabulary entry intentionally wins for duplicate byte sequences.
            node.TokenId = id;
        }
    }

    public IReadOnlyList<int> TokenIds => tokenIds;

    public static RwkvWorldTokenizer Load(string vocabularyPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(vocabularyPath);
        using var stream = File.OpenRead(vocabularyPath);
        return Load(stream);
    }

    public static RwkvWorldTokenizer LoadBundled()
    {
        using var stream = typeof(RwkvWorldTokenizer).Assembly.GetManifestResourceStream(BundledVocabularyResourceName)
            ?? throw new InvalidOperationException($"The bundled World vocabulary resource '{BundledVocabularyResourceName}' is unavailable.");
        return Load(stream);
    }

    private static RwkvWorldTokenizer Load(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        var tokens = new Dictionary<int, byte[]>();
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, leaveOpen: false);
        while (reader.ReadLine() is { } line)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var firstSpace = line.IndexOf(' ');
            var lastSpace = line.LastIndexOf(' ');
            if (firstSpace <= 0 || lastSpace <= firstSpace)
            {
                throw new InvalidDataException($"Invalid vocabulary line: '{line}'.");
            }

            if (!int.TryParse(line.AsSpan(0, firstSpace), out var id) || id < 0 ||
                !int.TryParse(line.AsSpan(lastSpace + 1), out var expectedLength) || expectedLength < 0)
            {
                throw new InvalidDataException($"Invalid vocabulary token metadata: '{line}'.");
            }

            var bytes = ParsePythonLiteral(line.AsSpan(firstSpace + 1, lastSpace - firstSpace - 1));
            if (bytes.Length != expectedLength || !tokens.TryAdd(id, bytes))
            {
                throw new InvalidDataException($"Invalid or duplicate vocabulary token {id}.");
            }
        }

        if (tokens.Count == 0)
        {
            throw new InvalidDataException("The vocabulary has no tokens.");
        }

        return new RwkvWorldTokenizer(tokens);
    }

    public IReadOnlyList<int> Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return Encode(Encoding.UTF8.GetBytes(text));
    }

    public IReadOnlyList<int> Encode(ReadOnlySpan<byte> bytes)
    {
        var result = new List<int>();
        for (var offset = 0; offset < bytes.Length;)
        {
            var node = root;
            int? token = null;
            var tokenEnd = offset;
            for (var cursor = offset; cursor < bytes.Length && node.Children[bytes[cursor]] is { } child; cursor++)
            {
                node = child;
                if (node.TokenId is { } id)
                {
                    token = id;
                    tokenEnd = cursor + 1;
                }
            }

            if (token is null)
            {
                throw new InvalidDataException($"The vocabulary does not contain a token for byte 0x{bytes[offset]:X2}.");
            }

            result.Add(token.Value);
            offset = tokenEnd;
        }

        return result;
    }

    public ReadOnlyMemory<byte> DecodeBytes(int token)
    {
        return tokens.TryGetValue(token, out var bytes)
            ? bytes
            : throw new ArgumentOutOfRangeException(nameof(token), $"The vocabulary does not contain token {token}.");
    }

    public string Decode(IEnumerable<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        using var stream = new MemoryStream();
        foreach (var token in tokenIds)
        {
            stream.Write(DecodeBytes(token).Span);
        }

        return Encoding.UTF8.GetString(stream.ToArray());
    }

    private static byte[] ParsePythonLiteral(ReadOnlySpan<char> literal)
    {
        literal = literal.Trim();
        var byteLiteral = literal.StartsWith("b'", StringComparison.Ordinal) || literal.StartsWith("b\"", StringComparison.Ordinal);
        if (byteLiteral)
        {
            literal = literal[1..];
        }

        if (literal.Length < 2 || (literal[0] != '\'' && literal[0] != '"') || literal[^1] != literal[0])
        {
            throw new InvalidDataException($"Unsupported vocabulary literal '{literal.ToString()}'.");
        }

        var values = new List<byte>();
        for (var index = 1; index < literal.Length - 1; index++)
        {
            var character = literal[index];
            if (character != '\\')
            {
                if (char.IsHighSurrogate(character) && index + 1 < literal.Length - 1 && char.IsLowSurrogate(literal[index + 1]))
                {
                    AppendUtf8(values, literal.Slice(index, 2).ToString());
                    index++;
                    continue;
                }

                if (char.IsSurrogate(character))
                {
                    throw new InvalidDataException("The vocabulary contains an unpaired UTF-16 surrogate.");
                }

                AppendUtf8(values, character.ToString());
                continue;
            }

            if (++index >= literal.Length - 1)
            {
                throw new InvalidDataException("A vocabulary literal ends with an escape character.");
            }

            character = literal[index];
            switch (character)
            {
                case '\\': values.Add((byte)'\\'); break;
                case '\'': values.Add((byte)'\''); break;
                case '"': values.Add((byte)'"'); break;
                case 'a': values.Add(0x07); break;
                case 'b': values.Add(0x08); break;
                case 'f': values.Add(0x0C); break;
                case 'n': values.Add(0x0A); break;
                case 'r': values.Add(0x0D); break;
                case 't': values.Add(0x09); break;
                case 'v': values.Add(0x0B); break;
                case 'x':
                    AppendEscapedByte(values, ParseHexByte(literal, ref index, 2), byteLiteral);
                    break;
                case 'u':
                    AppendUtf8(values, char.ConvertFromUtf32(ParseHexScalar(literal, ref index, 4)));
                    break;
                case 'U':
                    AppendUtf8(values, char.ConvertFromUtf32(ParseHexScalar(literal, ref index, 8)));
                    break;
                default:
                    if (character is >= '0' and <= '7')
                    {
                        AppendEscapedByte(values, ParseOctalByte(literal, ref index), byteLiteral);
                    }
                    else
                    {
                        throw new InvalidDataException($"Unsupported vocabulary escape '\\{character}'.");
                    }

                    break;
            }
        }

        return values.ToArray();
    }

    private static byte ParseHexByte(ReadOnlySpan<char> value, ref int index, int digits)
    {
        var scalar = ParseHexScalar(value, ref index, digits);
        if (scalar > byte.MaxValue) throw new InvalidDataException("A byte escape exceeds 255.");
        return (byte)scalar;
    }

    private static int ParseHexScalar(ReadOnlySpan<char> value, ref int index, int digits)
    {
        if (index + digits >= value.Length) throw new InvalidDataException("A vocabulary escape is truncated.");
        var start = ++index;
        var text = value.Slice(start, digits);
        if (!int.TryParse(text, System.Globalization.NumberStyles.AllowHexSpecifier, null, out var scalar))
        {
            throw new InvalidDataException($"Invalid hexadecimal escape '{text.ToString()}'.");
        }

        index += digits - 1;
        return scalar;
    }

    private static byte ParseOctalByte(ReadOnlySpan<char> value, ref int index)
    {
        var start = index;
        while (index + 1 < value.Length - 1 && index - start < 2 && value[index + 1] is >= '0' and <= '7') index++;
        var scalar = Convert.ToInt32(value.Slice(start, index - start + 1).ToString(), 8);
        if (scalar > byte.MaxValue) throw new InvalidDataException("An octal escape exceeds 255.");
        return (byte)scalar;
    }

    private static void AppendEscapedByte(List<byte> destination, byte value, bool byteLiteral)
    {
        if (byteLiteral)
        {
            destination.Add(value);
        }
        else
        {
            AppendUtf8(destination, ((char)value).ToString());
        }
    }

    private static void AppendUtf8(List<byte> destination, string value) => destination.AddRange(Encoding.UTF8.GetBytes(value));

    private sealed class TrieNode
    {
        public TrieNode?[] Children { get; } = new TrieNode[256];
        public int? TokenId { get; set; }
    }
}
