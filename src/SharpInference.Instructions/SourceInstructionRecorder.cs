namespace SharpInference.Instructions;

public sealed class SourceInstructionRecorder(InstructionTarget target) : IInstructionRecorder
{
    private readonly List<InstructionRecording> recordings = [];
    private readonly Dictionary<string, InstructionHelper> helpers = new(StringComparer.Ordinal);
    public InstructionTarget Target { get; } = target;
    public IReadOnlyList<InstructionRecording> Recordings => recordings.AsReadOnly();
    public IReadOnlyList<InstructionHelper> Helpers => Array.AsReadOnly(helpers.Values.ToArray());
    public IReadOnlyList<System.Reflection.Assembly> References =>
        Array.AsReadOnly(recordings.SelectMany(recording => recording.References).Distinct().ToArray());

    public void Record(InstructionRecording recording)
    {
        ArgumentNullException.ThrowIfNull(recording);
        if (!Enum.IsDefined(recording.Synchronization))
            throw new InvalidDataException("Unknown instruction synchronization.");
        if (Target == InstructionTarget.Cpu && recording.Synchronization != InstructionSynchronization.None)
            throw new NotSupportedException("CPU recorder does not support GPU group synchronization.");
        var additions = new Dictionary<string, InstructionHelper>(StringComparer.Ordinal);
        foreach (var helper in recording.Helpers)
        {
            if (string.IsNullOrWhiteSpace(helper.Name) || string.IsNullOrWhiteSpace(helper.Source))
                throw new InvalidDataException("Instruction helpers require a name and source.");
            if (helpers.TryGetValue(helper.Name, out var existing) && existing.Source != helper.Source ||
                additions.TryGetValue(helper.Name, out existing) && existing.Source != helper.Source)
                throw new InvalidDataException($"Conflicting instruction helper '{helper.Name}'.");
            additions.TryAdd(helper.Name, helper);
        }
        foreach (var helper in additions.Values) helpers.TryAdd(helper.Name, helper);
        recordings.Add(recording);
    }

    public string GetSource() => string.Join(Environment.NewLine, recordings.Select(recording =>
        recording.Source + (recording.Synchronization == InstructionSynchronization.GroupMemoryBarrier
            ? Environment.NewLine + "GroupMemoryBarrierWithGroupSync();" : "")));
}
