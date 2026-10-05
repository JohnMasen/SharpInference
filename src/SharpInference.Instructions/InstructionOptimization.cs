using SharpInference.Graphs;

namespace SharpInference.Instructions;

[Flags]
public enum InstructionBenefitKind { None = 0, Compute = 1, Execution = 2 }

public sealed record InstructionOperandDescription(
    string Name, TensorDescriptor Tensor, GraphResourceAccess Access, string Source);

public sealed class InstructionOptimizationCapability
{
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
            throw new ArgumentException("Inconsistent optimization capability.");
        CollectionId = collectionId;
        Name = name;
        Target = target;
        Configuration = configuration;
        ImplementationFingerprint = implementationFingerprint;
        Benefits = benefits;
        Definition = definition;
    }

    public Guid CollectionId { get; }
    public string Name { get; }
    public InstructionTarget Target { get; }
    public InstructionExecutionConfiguration Configuration { get; }
    public string ImplementationFingerprint { get; }
    public InstructionBenefitKind Benefits { get; }
    public TierOnePointwiseDefinition Definition { get; }
    public KernelPrecisionProfile Precision => TierOnePointwiseCatalog.Precision;

    public bool TryAdapt(IReadOnlyList<InstructionOperandDescription> operands, PrecisionRequirement requirement,
        out string diagnostic) => Definition.TryAdapt(operands, requirement, out diagnostic);
}

public interface IInstructionOptimizationProvider
{
    IReadOnlyList<InstructionOptimizationCapability> QueryOptimizationCapabilities();
}
