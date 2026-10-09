using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace SharpInference.Architectures.Phi4;

public sealed class Phi4Tokenizer
{
    private const string PretokenizerPattern =
        @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    private static readonly Regex Pretokenizer = new(
        PretokenizerPattern, RegexOptions.Compiled | RegexOptions.CultureInvariant);
    private static readonly IReadOnlyDictionary<byte, char> ByteEncoder = CreateByteEncoder();
    private static readonly IReadOnlyDictionary<char, byte> ByteDecoder =
        ByteEncoder.ToDictionary(static item => item.Value, static item => item.Key);

    private readonly IReadOnlyDictionary<string, int> vocabulary;
    private readonly IReadOnlyDictionary<int, string> tokensById;
    private readonly IReadOnlyDictionary<(string Left, string Right), int> mergeRanks;
    private readonly IReadOnlyDictionary<string, int> addedTokens;
    private readonly string[] addedTokenOrder;
    private readonly ConcurrentDictionary<string, string[]> cache = new(StringComparer.Ordinal);

    private Phi4Tokenizer(
        IReadOnlyDictionary<string, int> vocabulary,
        IReadOnlyDictionary<(string Left, string Right), int> mergeRanks,
        IReadOnlyDictionary<string, int> addedTokens)
    {
        this.vocabulary = vocabulary;
        var inverse = vocabulary.ToDictionary(static item => item.Value, static item => item.Key);
        foreach (var (token, id) in addedTokens)
            inverse[id] = token;
        tokensById = inverse;
        this.mergeRanks = mergeRanks;
        this.addedTokens = addedTokens;
        addedTokenOrder = addedTokens.Keys
            .OrderByDescending(static value => value.Length)
            .ThenBy(static value => value, StringComparer.Ordinal)
            .ToArray();
    }

    public const int EndOfTextTokenId = 199999;
    public const int ImageTokenId = 200010;
    public const int AudioTokenId = 200011;
    public const int AssistantTokenId = 200019;
    public const int EndTokenId = 200020;
    public const int UserTokenId = 200021;
    public const int SystemTokenId = 200022;

    public static Phi4Tokenizer Load(string tokenizerJsonPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tokenizerJsonPath);
        using var stream = File.OpenRead(tokenizerJsonPath);
        using var document = JsonDocument.Parse(stream);
        var root = document.RootElement;
        var model = root.GetProperty("model");

        var vocabulary = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var property in model.GetProperty("vocab").EnumerateObject())
            vocabulary.Add(property.Name, property.Value.GetInt32());

        var mergeRanks = new Dictionary<(string, string), int>();
        var rank = 0;
        foreach (var merge in model.GetProperty("merges").EnumerateArray())
        {
            string left;
            string right;
            if (merge.ValueKind == JsonValueKind.Array)
            {
                var parts = merge.EnumerateArray().ToArray();
                if (parts.Length != 2)
                    throw new InvalidDataException("A tokenizer merge must contain exactly two symbols.");
                left = parts[0].GetString()!;
                right = parts[1].GetString()!;
            }
            else
            {
                var parts = merge.GetString()!.Split(' ', 2);
                if (parts.Length != 2)
                    throw new InvalidDataException("A tokenizer merge must contain exactly two symbols.");
                left = parts[0];
                right = parts[1];
            }
            if (!mergeRanks.TryAdd((left, right), rank++))
                throw new InvalidDataException($"Duplicate tokenizer merge '{left} {right}'.");
        }

        var addedTokens = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var token in root.GetProperty("added_tokens").EnumerateArray())
        {
            var content = token.GetProperty("content").GetString()
                ?? throw new InvalidDataException("An added token has no content.");
            addedTokens[content] = token.GetProperty("id").GetInt32();
        }
        return new Phi4Tokenizer(vocabulary, mergeRanks, addedTokens);
    }

    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var result = new List<int>();
        var position = 0;
        while (position < text.Length)
        {
            var token = FindAddedToken(text, position);
            if (token is not null)
            {
                result.Add(addedTokens[token]);
                position += token.Length;
                continue;
            }

            var next = NextAddedTokenIndex(text, position);
            EncodeOrdinary(text[position..next], result);
            position = next;
        }
        return result.ToArray();
    }

    public string Decode(IEnumerable<int> tokenIds, bool skipSpecialTokens = false)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        var text = new StringBuilder();
        var bytes = new List<byte>();
        void FlushBytes()
        {
            if (bytes.Count == 0)
                return;
            text.Append(Encoding.UTF8.GetString(bytes.ToArray()));
            bytes.Clear();
        }

        foreach (var id in tokenIds)
        {
            if (!tokensById.TryGetValue(id, out var token))
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Unknown token ID {id}.");
            if (token.StartsWith("<|", StringComparison.Ordinal) &&
                token.EndsWith("|>", StringComparison.Ordinal))
            {
                FlushBytes();
                if (!skipSpecialTokens)
                    text.Append(token);
                continue;
            }
            foreach (var character in token)
            {
                if (!ByteDecoder.TryGetValue(character, out var value))
                    throw new InvalidDataException($"Token {id} contains an invalid ByteLevel character.");
                bytes.Add(value);
            }
        }
        FlushBytes();
        return text.ToString();
    }

    public string FormatChat(
        IEnumerable<Phi4ChatMessage> messages,
        bool addGenerationPrompt = true)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var builder = new StringBuilder();
        foreach (var message in messages)
        {
            ArgumentNullException.ThrowIfNull(message);
            if (string.IsNullOrWhiteSpace(message.Role) || message.Content is null)
                throw new ArgumentException("Chat messages require a role and content.", nameof(messages));
            builder.Append("<|").Append(message.Role).Append("|>")
                .Append(message.Content).Append("<|end|>");
        }
        return addGenerationPrompt
            ? builder.Append("<|assistant|>").ToString()
            : builder.Append("<|endoftext|>").ToString();
    }

    private void EncodeOrdinary(string text, List<int> destination)
    {
        foreach (Match match in Pretokenizer.Matches(text))
        {
            var bytes = Encoding.UTF8.GetBytes(match.Value);
            var encoded = string.Create(bytes.Length, bytes, static (span, source) =>
            {
                for (var index = 0; index < source.Length; index++)
                    span[index] = ByteEncoder[source[index]];
            });
            foreach (var piece in cache.GetOrAdd(encoded, ApplyBpe))
            {
                if (!vocabulary.TryGetValue(piece, out var id))
                    throw new InvalidDataException($"Tokenizer vocabulary does not contain BPE piece '{piece}'.");
                destination.Add(id);
            }
        }
    }

    private string[] ApplyBpe(string token)
    {
        var symbols = token.Select(static character => character.ToString()).ToList();
        while (symbols.Count > 1)
        {
            var bestRank = int.MaxValue;
            (string Left, string Right) best = default;
            for (var index = 0; index < symbols.Count - 1; index++)
            {
                var pair = (symbols[index], symbols[index + 1]);
                if (mergeRanks.TryGetValue(pair, out var candidate) && candidate < bestRank)
                {
                    bestRank = candidate;
                    best = pair;
                }
            }
            if (bestRank == int.MaxValue)
                break;

            var merged = new List<string>(symbols.Count);
            for (var index = 0; index < symbols.Count;)
            {
                if (index + 1 < symbols.Count &&
                    symbols[index] == best.Left && symbols[index + 1] == best.Right)
                {
                    merged.Add(best.Left + best.Right);
                    index += 2;
                }
                else
                {
                    merged.Add(symbols[index++]);
                }
            }
            symbols = merged;
        }
        return symbols.ToArray();
    }

    private string? FindAddedToken(string text, int position)
    {
        foreach (var token in addedTokenOrder)
            if (text.AsSpan(position).StartsWith(token, StringComparison.Ordinal))
                return token;
        return null;
    }

    private int NextAddedTokenIndex(string text, int position)
    {
        var result = text.Length;
        foreach (var token in addedTokenOrder)
        {
            var index = text.IndexOf(token, position, StringComparison.Ordinal);
            if (index >= 0 && index < result)
                result = index;
        }
        return result;
    }

    private static IReadOnlyDictionary<byte, char> CreateByteEncoder()
    {
        var values = Enumerable.Range('!', '~' - '!' + 1)
            .Concat(Enumerable.Range('¡', '¬' - '¡' + 1))
            .Concat(Enumerable.Range('®', 'ÿ' - '®' + 1))
            .ToList();
        var characters = new List<int>(values);
        var extra = 0;
        for (var value = 0; value < 256; value++)
        {
            if (values.Contains(value))
                continue;
            values.Add(value);
            characters.Add(256 + extra++);
        }
        return values.Select((value, index) => (value, character: characters[index]))
            .ToDictionary(static item => (byte)item.value, static item => (char)item.character);
    }
}

public sealed record Phi4ChatMessage(string Role, string Content);
