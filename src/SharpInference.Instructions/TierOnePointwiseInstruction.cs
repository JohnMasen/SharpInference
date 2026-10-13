using System.Security.Cryptography;
using System.Text;
using SharpInference.Graphs;

namespace SharpInference.Instructions;

/// <summary>Adapts catalog-defined pointwise fusions to a configured target implementation.</summary>
public abstract class TierOnePointwiseInstruction : Instruction
{
    /// <summary>Selects a catalog fusion and configures its FP32 input and output signature.</summary>
    /// <param name="name">The catalog name of the FP32 pointwise fusion to implement.</param>
    /// <param name="configuration">The selected implementation configuration, or null for the default implementation.</param>
    protected TierOnePointwiseInstruction(string name, InstructionExecutionConfiguration configuration)
    {
        Definition = TierOnePointwiseCatalog.Get(name);
        Configuration = configuration;
        Signatures = Array.AsReadOnly<InstructionSignature>(
        [
            new(Definition.Inputs.Select(input => new InstructionPort(input, GraphElementType.Float32, GraphResourceAccess.Read))
                .Append(new("output", GraphElementType.Float32, GraphResourceAccess.Write)).ToArray(),
                [], TierOnePointwiseCatalog.Precision),
        ]);
    }

    /// <summary>Gets the catalog pointwise definition implemented by this instruction.</summary>
    public TierOnePointwiseDefinition Definition { get; }
    /// <summary>Gets the concrete target execution configuration.</summary>
    public InstructionExecutionConfiguration Configuration { get; }
    /// <inheritdoc/>
    public override Guid CollectionId => InstructionCollectionIds.TierOneFloat32;
    /// <inheritdoc/>
    public override string Name => Definition.Name;
    /// <inheritdoc/>
    public override InstructionTarget Target => Configuration.Target;
    /// <inheritdoc/>
    public override bool RequiresDispatchIsolation => false;
    /// <inheritdoc/>
    public override IReadOnlyList<InstructionSignature> Signatures { get; }

    /// <summary>Gets the numerical implementation discriminator included in optimization fingerprints.</summary>
    protected virtual string NumericalImplementationIdentity => "";

    /// <summary>Creates a fusion capability with a fingerprint including module and numerical implementation identities.</summary>
    public InstructionOptimizationCapability OptimizationCapability()
    {
        var identity = $"{GetType().Assembly.ManifestModule.ModuleVersionId:D}|" +
            $"{typeof(TierOnePointwiseCatalog).Assembly.ManifestModule.ModuleVersionId:D}|" +
            $"{Target.Architecture}|{Name}|{Configuration.Implementation.Value}|{NumericalImplementationIdentity}";
        return new(CollectionId, Name, Target, Configuration,
            Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(identity))),
            InstructionBenefitKind.Execution, Definition);
    }

    /// <inheritdoc/>
    protected sealed override void ValidateExecutionConfiguration(InstructionExecutionConfiguration? configuration)
    {
        if (configuration != Configuration)
        {
            throw new InstructionAdaptationException(CollectionId, Name, "Missing or unsupported execution configuration.");
        }
    }

    /// <inheritdoc/>
    protected sealed override void Validate(InstructionParameter[] parameters)
    {
        var tensors = parameters.OfType<InstructionTensorParameter>().ToArray();
        if (tensors.Any(parameter => parameter.ByteOffset != 0))
        {
            throw new InstructionAdaptationException(CollectionId, Name, "Put typed offsets on calls, not instruction parameters.");
        }

        var operands = tensors.Select(parameter =>
            new InstructionOperandDescription(parameter.Name, parameter.Tensor, parameter.Access, parameter.Source)).ToArray();
        if (!Definition.TryAdapt(operands, new(GraphElementType.Float32, GraphElementType.Float32), out var diagnostic))
        {
            throw new InstructionAdaptationException(CollectionId, Name, diagnostic);
        }
    }
}
