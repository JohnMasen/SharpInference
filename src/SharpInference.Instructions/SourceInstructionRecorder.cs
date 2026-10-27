using SharpInference.Graphs.Execution;
namespace SharpInference.Instructions;

/// <summary>Collects source recordings and deduplicates helper declarations and assembly references.</summary>
/// <param name="target">The CPU or Direct3D12 VM execution backend.</param>
public sealed class SourceInstructionRecorder(ExecutionTarget target) : IInstructionRecorder
{
    private readonly List<InstructionRecording> recordings = [];
    private readonly Dictionary<string, InstructionHelper> helpers = new(StringComparer.Ordinal);
    /// <summary>Gets the execution architecture required by this contract.</summary>
    public ExecutionTarget Target { get; } = target;
    /// <summary>Gets the instruction recordings in invocation order.</summary>
    public IReadOnlyList<InstructionRecording> Recordings => recordings.AsReadOnly();
    /// <summary>Gets the named helper declarations required by the recorded source.</summary>
    public IReadOnlyList<InstructionHelper> Helpers => Array.AsReadOnly(helpers.Values.ToArray());
    /// <summary>Gets the distinct assemblies required to compile the recorded source.</summary>
    public IReadOnlyList<System.Reflection.Assembly> References =>
        Array.AsReadOnly(recordings.SelectMany(recording => recording.References).Distinct().ToArray());

    /// <summary>Appends a recording after validating synchronization and rejecting conflicting helper definitions.</summary>
    /// <param name="recording">The source, helpers, synchronization, and references to record.</param>
    public void Record(InstructionRecording recording)
    {
        ArgumentNullException.ThrowIfNull(recording);
        if (!Enum.IsDefined(recording.Synchronization))
        {
            throw new InvalidDataException("Unknown instruction synchronization.");
        }

        if (Target == ExecutionTarget.Cpu && recording.Synchronization != InstructionSynchronization.None)
        {
            throw new NotSupportedException("CPU recorder does not support GPU group synchronization.");
        }

        var additions = new Dictionary<string, InstructionHelper>(StringComparer.Ordinal);
        foreach (var helper in recording.Helpers)
        {
            if (string.IsNullOrWhiteSpace(helper.Name) || string.IsNullOrWhiteSpace(helper.Source))
            {
                throw new InvalidDataException("Instruction helpers require a name and source.");
            }

            if (helpers.TryGetValue(helper.Name, out var existing) && existing.Source != helper.Source ||
                additions.TryGetValue(helper.Name, out existing) && existing.Source != helper.Source)
            {
                throw new InvalidDataException($"Conflicting instruction helper '{helper.Name}'.");
            }

            additions.TryAdd(helper.Name, helper);
        }
        foreach (var helper in additions.Values)
        {
            helpers.TryAdd(helper.Name, helper);
        }

        recordings.Add(recording);
    }

    /// <summary>Concatenates recorded fragments in invocation order and appends required GPU group synchronization.</summary>
    public string GetSource() => string.Join(Environment.NewLine, recordings.Select(recording =>
        recording.Source + (recording.Synchronization == InstructionSynchronization.GroupMemoryBarrier
            ? Environment.NewLine + "GroupMemoryBarrierWithGroupSync();" : "")));
}
