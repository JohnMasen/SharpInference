namespace SharpInference.Instructions;

/// <summary>Identifies a concrete execution implementation using a bounded ASCII identifier.</summary>
public sealed record InstructionImplementationId
{
    /// <summary>Creates an implementation identifier of at most 128 ASCII letters, digits, dots, hyphens, or underscores.</summary>
    /// <param name="value">The textual value of this identifier or attribute.</param>
    public InstructionImplementationId(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);
        if (value.Length > 128 || value.Any(character =>
                !char.IsAsciiLetterOrDigit(character) && character is not ('.' or '-' or '_')))
            throw new ArgumentException("Implementation identifiers require ASCII letters, digits, '.', '-' or '_'.", nameof(value));
        Value = value;
    }

    /// <summary>Gets the textual value of this identifier or attribute.</summary>
    public string Value { get; }
    /// <inheritdoc/>
    public override string ToString() => Value;
}

/// <summary>Selects a concrete instruction implementation for an execution target.</summary>
public abstract record InstructionExecutionConfiguration
{
    /// <summary>Creates an execution configuration for a nonnull implementation identity.</summary>
    /// <param name="implementation">The identity of the selected execution implementation.</param>
    protected InstructionExecutionConfiguration(InstructionImplementationId implementation)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        Implementation = implementation;
    }

    /// <summary>Gets the identity of the selected execution implementation.</summary>
    public InstructionImplementationId Implementation { get; }
    /// <summary>Gets the execution architecture required by this contract.</summary>
    public abstract InstructionTarget Target { get; }
}

/// <summary>Selects an instruction implementation for managed CPU execution.</summary>
public sealed record CpuInstructionExecutionConfiguration : InstructionExecutionConfiguration
{
    /// <summary>Selects the specified implementation for managed CPU execution.</summary>
    /// <param name="implementation">The identity of the selected execution implementation.</param>
    public CpuInstructionExecutionConfiguration(InstructionImplementationId implementation) : base(implementation) { }
    /// <inheritdoc/>
    public override InstructionTarget Target => InstructionTarget.Cpu;
}

/// <summary>Selects an instruction implementation for Direct3D12 execution.</summary>
public sealed record D3D12InstructionExecutionConfiguration : InstructionExecutionConfiguration
{
    /// <summary>Selects the specified implementation for Direct3D12 execution.</summary>
    /// <param name="implementation">The identity of the selected execution implementation.</param>
    public D3D12InstructionExecutionConfiguration(InstructionImplementationId implementation) : base(implementation) { }
    /// <inheritdoc/>
    public override InstructionTarget Target => InstructionTarget.Direct3D12;
}
