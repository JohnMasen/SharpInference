using System.Runtime.CompilerServices;
using System.Text;

namespace SharpInference;

public interface IRwkvGenerationSession
{
    ReadOnlyMemory<float> Prefill(ReadOnlySpan<int> tokens);
    ReadOnlyMemory<float> ForwardToken(int token);
}

public static class RwkvTextGenerator
{
    public static async IAsyncEnumerable<char> GenerateCharactersAsync(
        IRwkvGenerationSession session,
        RwkvWorldTokenizer tokenizer,
        string prompt,
        RwkvGenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var text in GenerateAsync(session, tokenizer, prompt, options, cancellationToken))
        {
            foreach (var character in text)
            {
                yield return character;
            }
        }
    }

    public static async IAsyncEnumerable<string> GenerateAsync(
        IRwkvGenerationSession session,
        RwkvWorldTokenizer tokenizer,
        string prompt,
        RwkvGenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(prompt);

        await foreach (var text in GenerateAsync(
            session,
            tokenizer,
            tokenizer.Encode(prompt),
            options,
            cancellationToken))
        {
            yield return text;
        }
    }

    public static async IAsyncEnumerable<string> GenerateAsync(
        IRwkvGenerationSession session,
        RwkvWorldTokenizer tokenizer,
        IReadOnlyList<int> promptTokens,
        RwkvGenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(promptTokens);
        options ??= new RwkvGenerationOptions();
        options.Validate();

        if (promptTokens.Count == 0)
        {
            throw new ArgumentException("The prompt must contain at least one token.", nameof(promptTokens));
        }

        var logits = session.Prefill(promptTokens.ToArray());
        await foreach (var text in GenerateFromPrefilledAsync(
            session,
            tokenizer,
            logits,
            options,
            cancellationToken))
        {
            yield return text;
        }
    }

    public static async IAsyncEnumerable<string> GenerateFromPrefilledAsync(
        IRwkvGenerationSession session,
        RwkvWorldTokenizer tokenizer,
        ReadOnlyMemory<float> initialLogits,
        RwkvGenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (initialLogits.IsEmpty)
        {
            throw new ArgumentException("The initial logits cannot be empty.", nameof(initialLogits));
        }

        options ??= new RwkvGenerationOptions();
        options.Validate();
        var sampler = new RwkvSampler(options, tokenizer.TokenIds);
        var decoder = new IncrementalUtf8Decoder();
        var logits = initialLogits;
        var stopMatcher = options.StopStrings.Count == 0 ? null : new Utf8StopMatcher(options.StopStrings);
        for (var generated = 0; generated < options.MaxTokens; generated++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var token = sampler.Sample(logits.Span);
            if (options.StopTokenIds.Contains(token))
            {
                yield break;
            }

            var tokenBytes = tokenizer.DecodeBytes(token);
            if (stopMatcher is null)
            {
                var text = decoder.Append(tokenBytes.Span);
                logits = session.ForwardToken(token);
                if (text.Length > 0)
                {
                    yield return text;
                }
            }
            else
            {
                logits = session.ForwardToken(token);
                var textStop = stopMatcher.Append(tokenBytes.Span, out var safeBytes);
                var text = decoder.Append(safeBytes.Span);
                if (text.Length > 0)
                {
                    yield return text;
                }

                if (textStop)
                {
                    yield break;
                }
            }

            await Task.Yield();
        }

        var remaining = stopMatcher is null
            ? decoder.Flush()
            : decoder.Append(stopMatcher.Flush().Span) + decoder.Flush();
        if (remaining.Length > 0)
        {
            yield return remaining;
        }
    }

    private sealed class IncrementalUtf8Decoder
    {
        private readonly Decoder decoder = new UTF8Encoding(false, false).GetDecoder();

        public string Append(ReadOnlySpan<byte> bytes) => Decode(bytes, flush: false);
        public string Flush() => Decode(ReadOnlySpan<byte>.Empty, flush: true);

        private string Decode(ReadOnlySpan<byte> bytes, bool flush)
        {
            if (bytes.IsEmpty && !flush) return string.Empty;
            var chars = new char[Encoding.UTF8.GetMaxCharCount(bytes.Length)];
            decoder.Convert(bytes, chars, flush, out _, out var charsUsed, out _);
            return new string(chars, 0, charsUsed);
        }
    }

    private sealed class Utf8StopMatcher
        {
            private readonly byte[][] patterns;
            private readonly List<byte> pending = [];

            public Utf8StopMatcher(IReadOnlyList<string> stopStrings)
            {
                patterns = stopStrings.Select(Encoding.UTF8.GetBytes).ToArray();
            }

            public bool Append(ReadOnlySpan<byte> bytes, out ReadOnlyMemory<byte> safeBytes)
            {
                List<byte>? safe = null;
                foreach (var value in bytes)
                {
                    pending.Add(value);
                    if (TryGetMatchingPatternLength(out var matchingLength))
                    {
                        var stopStart = pending.Count - matchingLength;
                        if (stopStart > 0)
                        {
                            safe ??= [];
                            safe.AddRange(pending.GetRange(0, stopStart));
                        }

                        pending.Clear();
                        safeBytes = safe?.ToArray() ?? Array.Empty<byte>();
                        return true;
                    }

                    var retainedPrefixLength = GetLongestRetainedPrefixLength();
                    var safeCount = pending.Count - retainedPrefixLength;
                    if (safeCount > 0)
                    {
                        safe ??= [];
                        safe.AddRange(pending.GetRange(0, safeCount));
                        pending.RemoveRange(0, safeCount);
                    }
                }

                safeBytes = safe?.ToArray() ?? Array.Empty<byte>();
                return false;
            }

            public ReadOnlyMemory<byte> Flush()
            {
                var remaining = pending.ToArray();
                pending.Clear();
                return remaining;
            }

            private bool TryGetMatchingPatternLength(out int matchingLength)
            {
                matchingLength = 0;
                foreach (var pattern in patterns)
                {
                    if (pattern.Length > matchingLength &&
                        pattern.Length <= pending.Count &&
                        EndsWith(pattern))
                    {
                        matchingLength = pattern.Length;
                    }
                }

                return matchingLength > 0;
            }

            private int GetLongestRetainedPrefixLength()
            {
                for (var length = pending.Count; length > 0; length--)
                {
                    foreach (var pattern in patterns)
                    {
                        if (length < pattern.Length && StartsWith(pattern, length))
                        {
                            return length;
                        }
                    }
                }

                return 0;
            }

            private bool EndsWith(ReadOnlySpan<byte> pattern)
            {
                var start = pending.Count - pattern.Length;
                for (var index = 0; index < pattern.Length; index++)
                {
                    if (pending[start + index] != pattern[index])
                    {
                        return false;
                    }
                }

                return true;
            }

            private bool StartsWith(ReadOnlySpan<byte> pattern, int length)
            {
                for (var index = 0; index < length; index++)
                {
                    if (pending[pending.Count - length + index] != pattern[index])
                    {
                        return false;
                    }
                }

                return true;
            }
    }

    private sealed class RwkvSampler
    {
        private readonly RwkvGenerationOptions options;
        private readonly Random random;
        private readonly LogitEntry[] ranking;
        private readonly int[] tokenIds;

        public RwkvSampler(RwkvGenerationOptions options, IReadOnlyList<int> tokenIds)
        {
            this.options = options;
            random = options.Seed is { } seed ? new Random(seed) : Random.Shared;
            this.tokenIds = tokenIds.ToArray();
            ranking = new LogitEntry[this.tokenIds.Length];
        }

        public int Sample(ReadOnlySpan<float> logits)
        {
            if (tokenIds.Length == 0 || tokenIds[^1] >= logits.Length) throw new ArgumentException("The logits do not cover every tokenizer token.", nameof(logits));
            if (options.Temperature == 0) return ArgMax(logits, tokenIds);

            var count = options.TopK == 0 ? ranking.Length : Math.Min(options.TopK, ranking.Length);
            for (var index = 0; index < tokenIds.Length; index++) ranking[index] = new LogitEntry(tokenIds[index], logits[tokenIds[index]]);
            Array.Sort(ranking, static (left, right) => right.Logit.CompareTo(left.Logit));

            var maximum = ranking[0].Logit;
            double total = 0;
            for (var index = 0; index < count; index++)
            {
                total += Math.Exp((ranking[index].Logit - maximum) / options.Temperature);
            }

            var threshold = total * options.TopP;
            double cumulative = 0;
            var retained = count;
            for (var index = 0; index < count; index++)
            {
                cumulative += Math.Exp((ranking[index].Logit - maximum) / options.Temperature);
                if (cumulative >= threshold)
                {
                    retained = index + 1;
                    total = cumulative;
                    break;
                }
            }

            var sample = random.NextDouble() * total;
            cumulative = 0;
            for (var index = 0; index < retained; index++)
            {
                cumulative += Math.Exp((ranking[index].Logit - maximum) / options.Temperature);
                if (sample < cumulative) return ranking[index].Token;
            }

            return ranking[retained - 1].Token;
        }

        private static int ArgMax(ReadOnlySpan<float> values, ReadOnlySpan<int> candidates)
        {
            var result = candidates[0];
            for (var index = 1; index < candidates.Length; index++)
            {
                if (values[candidates[index]] > values[result]) result = candidates[index];
            }

            return result;
        }

        private readonly record struct LogitEntry(int Token, float Logit);
    }
}
