
namespace SharpInference.Instructions;

/// <summary>Classifies the benefits offered by an instruction optimization.</summary>
[Flags]
public enum InstructionBenefitKind
{
    /// <summary>Declares no optimization benefit.</summary>
    None = 0,
    /// <summary>Offers a computational benefit.</summary>
    Compute = 1,
    /// <summary>Reduces execution overhead or intermediate materialization.</summary>
    Execution = 2
}

/// <summary>Describes an operand's tensor, access, and resource identity for adaptation checks.</summary>
/// <param name="Name">The instruction port name associated with the operand.</param>
/// <param name="Tensor">The tensor descriptor defining element representation and shape.</param>
/// <param name="Access">The permitted resource access.</param>
/// <param name="Source">The source resource identity.</param>
public sealed record InstructionOperandDescription(
    string Name, TensorDescriptor Tensor, ResourceAccess Access, string Source);

/// <summary>Describes a target-specific tier-one implementation and its fusion eligibility contract.</summary>
public sealed class InstructionOptimizationCapability
{
    /// <summary>Creates and validates a consistent tier-one implementation and fusion contract.</summary>
    /// <param name="collectionId">The identity of the collection providing this implementation.</param>
    /// <param name="name">The registered tier-one pointwise instruction name.</param>
    /// <param name="target">The CPU or Direct3D12 VM execution backend.</param>
    /// <param name="configuration">The selected implementation configuration, or null for the default implementation.</param>
    /// <param name="implementationFingerprint">The implementation identity used to match trusted measurements.</param>
    /// <param name="benefits">The categories of optimization benefit offered by the implementation.</param>
    /// <param name="definition">The pointwise fusion definition supported by this implementation.</param>
    public InstructionOptimizationCapability(Guid collectionId, string name, InstructionTarget target,
        InstructionExecutionConfiguration configuration, string implementationFingerprint,
        InstructionBenefitKind benefits, TierOnePointwiseDefinition definition)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentException.ThrowIfNullOrWhiteSpace(implementationFingerprint);
        ArgumentNullException.ThrowIfNull(configuration);
        ArgumentNullException.ThrowIfNull(definition);
        if (configuration.Target != target || name != definition.Name ||
            collectionId != InstructionCollectionIds.TierOneFloat32 ||
            benefits == InstructionBenefitKind.None || (benefits & ~(InstructionBenefitKind.Compute | InstructionBenefitKind.Execution)) != 0)
        {
            throw new ArgumentException("Inconsistent optimization capability.");
        }

        CollectionId = collectionId;
        Name = name;
        Target = target;
        Configuration = configuration;
        ImplementationFingerprint = implementationFingerprint;
        Benefits = benefits;
        Definition = definition;
    }

    /// <summary>Gets the identity of the collection providing this implementation.</summary>
    public Guid CollectionId { get; }
    /// <summary>Gets the registered tier-one pointwise instruction name.</summary>
    public string Name { get; }
    /// <summary>Gets the execution architecture required by this contract.</summary>
    public InstructionTarget Target { get; }
    /// <summary>Gets the concrete target execution configuration.</summary>
    public InstructionExecutionConfiguration Configuration { get; }
    /// <summary>Gets the implementation identity used to match trusted measurements.</summary>
    public string ImplementationFingerprint { get; }
    /// <summary>Gets the categories of optimization benefit offered by the implementation.</summary>
    public InstructionBenefitKind Benefits { get; }
    /// <summary>Gets the pointwise fusion definition supported by this implementation.</summary>
    public TierOnePointwiseDefinition Definition { get; }
    /// <summary>Gets the internal arithmetic and accumulator precision contract.</summary>
    public KernelPrecisionProfile Precision => TierOnePointwiseCatalog.Precision;

    /// <summary>Checks whether operand types, shapes, access, aliases, and precision permit this pointwise fusion.</summary>
    /// <param name="operands">The named tensor operands, including the writable output.</param>
    /// <param name="requirement">The minimum arithmetic and accumulator precision required by the caller.</param>
    /// <param name="diagnostic">Receives an empty string on success or the incompatibility reason on failure.</param>
    /// <returns><see langword="true"/> when the operands satisfy the fusion contract; otherwise, <see langword="false"/>.</returns>
    public bool TryAdapt(IReadOnlyList<InstructionOperandDescription> operands, PrecisionRequirement requirement,
        out string diagnostic) => Definition.TryAdapt(operands, requirement, out diagnostic);
}

/// <summary>Discovers instruction implementations available for graph optimization.</summary>
public interface IInstructionOptimizationProvider
{
    /// <summary>Returns instruction implementations available for graph fusion selection.</summary>
    IReadOnlyList<InstructionOptimizationCapability> QueryOptimizationCapabilities();
}
