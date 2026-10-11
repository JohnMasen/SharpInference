namespace SharpInference.WebApi;

/// <summary>Distinguishes prompt-prefill snapshots from snapshots containing a completed assistant answer.</summary>
public enum PromptStateKind
{
    /// <summary>Identifies state captured immediately after prompt prefill.</summary>
    Prefill,
    /// <summary>Identifies state captured after a canonical completed answer.</summary>
    Complete,
}

/// <summary>Configures the bounded, scored prompt-state cache.</summary>
public sealed class PromptStateManagerOptions
{
    /// <summary>Gets whether prompt-state lookup and snapshot storage are enabled.</summary>
    public bool Enabled { get; init; } = true;
    /// <summary>Gets the maximum number of cached states; zero disables storage.</summary>
    public int Capacity { get; init; } = 10;

    /// <summary>Rejects a negative prompt-state cache capacity.</summary>
    public void Validate()
    {
        if (Capacity < 0)
        {
            throw new InvalidOperationException("Rwkv:StateManager:Capacity cannot be negative.");
        }
    }
}

/// <summary>Describes a reusable cached prefix, its state snapshot, and the prompt position at which prefill resumes.</summary>
/// <param name="Input">The cached prompt prefix represented by this state.</param>
/// <param name="State">The serialized session-state snapshot for the cached prefix.</param>
/// <param name="StartCharacterIndex">The character position where the uncached prompt suffix starts.</param>
/// <param name="Score">The cache reuse score after this match was selected.</param>
/// <param name="Kind">Whether the cached snapshot contains prefill state or a completed answer.</param>
public sealed record PromptStateMatch(
    string Input,
    ReadOnlyMemory<byte> State,
    int StartCharacterIndex,
    int Score,
    PromptStateKind Kind);

/// <summary>Caches scored prompt-state snapshots and selects the longest compatible proper prefix.</summary>
public sealed class PromptStateManager
{
    private readonly object gate = new();
    private readonly int capacity;
    private readonly Dictionary<(string Input, PromptStateKind Kind), Entry> entries = new();
    private long sequence;

    /// <summary>Creates a prompt-state cache with validated enablement and capacity settings.</summary>
    /// <param name="options">The cache enablement flag and non-negative entry capacity.</param>
    public PromptStateManager(PromptStateManagerOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        capacity = options.Enabled ? options.Capacity : 0;
    }

    /// <summary>Gets the effective cache capacity, which is zero when caching is disabled.</summary>
    public int Capacity => capacity;

    /// <summary>Gets the current number of cached prompt-state entries.</summary>
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

    /// <summary>Finds and scores the longest reusable proper prefix, requiring a message boundary after completed-answer entries.</summary>
    /// <param name="input">The full prompt text whose reusable proper prefix is requested.</param>
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

    /// <summary>Copies a state snapshot into the cache, evicting the lowest-scored least-recently-used entry when full.</summary>
    /// <param name="input">The non-empty canonical prompt prefix represented by the snapshot.</param>
    /// <param name="state">The non-empty serialized session snapshot copied into the cache.</param>
    /// <param name="kind">Whether the snapshot contains prefill state or a completed answer.</param>
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
