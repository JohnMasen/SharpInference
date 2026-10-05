namespace SharpInference.Instructions;

public sealed record InstructionImplementationId
{
    public InstructionImplementationId(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);
        if (value.Length > 128 || value.Any(character =>
                !char.IsAsciiLetterOrDigit(character) && character is not ('.' or '-' or '_')))
            throw new ArgumentException("Implementation identifiers require ASCII letters, digits, '.', '-' or '_'.", nameof(value));
        Value = value;
    }

    public string Value { get; }
    public override string ToString() => Value;
}

public abstract record InstructionExecutionConfiguration
{
    protected InstructionExecutionConfiguration(InstructionImplementationId implementation)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        Implementation = implementation;
    }

    public InstructionImplementationId Implementation { get; }
    public abstract InstructionTarget Target { get; }
}

public sealed record CpuInstructionExecutionConfiguration : InstructionExecutionConfiguration
{
    public CpuInstructionExecutionConfiguration(InstructionImplementationId implementation) : base(implementation) { }
    public override InstructionTarget Target => InstructionTarget.Cpu;
}

public sealed record D3D12InstructionExecutionConfiguration : InstructionExecutionConfiguration
{
    public D3D12InstructionExecutionConfiguration(InstructionImplementationId implementation) : base(implementation) { }
    public override InstructionTarget Target => InstructionTarget.Direct3D12;
}
