namespace SharpInference.WebApi;

public enum PromptStateKind
{
    Prefill,
    Complete,
}

public sealed class PromptStateManagerOptions
{
    public bool Enabled { get; init; } = true;
    public int Capacity { get; init; } = 10;

    public void Validate()
    {
        if (Capacity < 0)
        {
            throw new InvalidOperationException("Rwkv:StateManager:Capacity cannot be negative.");
        }
    }
}

public sealed record PromptStateMatch(
    string Input,
    ReadOnlyMemory<byte> State,
    int StartCharacterIndex,
    int Score,
    PromptStateKind Kind);

public sealed class PromptStateManager
{
    private readonly object gate = new();
    private readonly int capacity;
    private readonly Dictionary<(string Input, PromptStateKind Kind), Entry> entries = new();
    private long sequence;

    public PromptStateManager(PromptStateManagerOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        capacity = options.Enabled ? options.Capacity : 0;
    }

    public int Capacity => capacity;

    public int Count
    {
        get
        {
            lock (gate)
            {
                return entries.Count;
            }
        }
    }

    public PromptStateMatch? FindLongestPrefix(string input)
    {
        ArgumentNullException.ThrowIfNull(input);
        lock (gate)
        {
            Entry? best = null;
            foreach (var entry in entries.Values)
            {
                if (entry.Input.Length >= input.Length ||
                    !input.StartsWith(entry.Input, StringComparison.Ordinal) ||
                    (entry.Kind == PromptStateKind.Complete &&
                     !input.AsSpan(entry.Input.Length).StartsWith("\n\n", StringComparison.Ordinal)) ||
                    (best is not null && (entry.Input.Length < best.Input.Length ||
                     entry.Input.Length == best.Input.Length && entry.Kind != PromptStateKind.Complete)))
                {
                    continue;
                }

                best = entry;
            }

            if (best is null)
            {
                return null;
            }

            best.Score++;
            best.LastAccessSequence = ++sequence;
            return new PromptStateMatch(
                best.Input,
                best.State,
                best.Input.Length,
                best.Score,
                best.Kind);
        }
    }

    public void Store(string input, ReadOnlySpan<byte> state, PromptStateKind kind)
    {
        ArgumentException.ThrowIfNullOrEmpty(input);
        if (kind is not (PromptStateKind.Prefill or PromptStateKind.Complete))
        {
            throw new ArgumentOutOfRangeException(nameof(kind));
        }
        if (state.IsEmpty)
        {
            throw new ArgumentException("The cached state cannot be empty.", nameof(state));
        }

        if (capacity == 0)
        {
            return;
        }

        var snapshot = state.ToArray();
        lock (gate)
        {
            if (entries.TryGetValue((input, kind), out var existing))
            {
                existing.State = snapshot;
                existing.LastAccessSequence = ++sequence;
                return;
            }

            if (entries.Count == capacity)
            {
                var victim = entries.Values
                    .OrderBy(static entry => entry.Score)
                    .ThenBy(static entry => entry.LastAccessSequence)
                    .First();
                entries.Remove((victim.Input, victim.Kind));
            }

            entries.Add((input, kind), new Entry(input, snapshot, kind, ++sequence));
        }
    }

    private sealed class Entry(string input, byte[] state, PromptStateKind kind, long sequence)
    {
        public string Input { get; } = input;
        public byte[] State { get; set; } = state;
        public PromptStateKind Kind { get; } = kind;
        public int Score { get; set; }
        public long LastAccessSequence { get; set; } = sequence;
    }
}
