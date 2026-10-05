using System.Security.Cryptography;
using System.Text;
using SharpInference.Graphs;

namespace SharpInference.Instructions;

public abstract class TierOnePointwiseInstruction : Instruction
{
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

    public TierOnePointwiseDefinition Definition { get; }
    public InstructionExecutionConfiguration Configuration { get; }
    public override Guid CollectionId => InstructionCollectionIds.TierOneFloat32;
    public override string Name => Definition.Name;
    public override InstructionTarget Target => Configuration.Target;
    public override bool RequiresDispatchIsolation => false;
    public override IReadOnlyList<InstructionSignature> Signatures { get; }

    protected virtual string NumericalImplementationIdentity => "";

    public InstructionOptimizationCapability OptimizationCapability()
    {
        var identity = $"{GetType().Assembly.ManifestModule.ModuleVersionId:D}|" +
            $"{typeof(TierOnePointwiseCatalog).Assembly.ManifestModule.ModuleVersionId:D}|" +
            $"{Target.Architecture}|{Name}|{Configuration.Implementation.Value}|{NumericalImplementationIdentity}";
        return new(CollectionId, Name, Target, Configuration,
            Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(identity))),
            InstructionBenefitKind.Execution, Definition);
    }

    protected sealed override void ValidateExecutionConfiguration(InstructionExecutionConfiguration? configuration)
    {
        if (configuration != Configuration)
            throw new InstructionAdaptationException(CollectionId, Name, "Missing or unsupported execution configuration.");
    }

    protected sealed override void Validate(InstructionParameter[] parameters)
    {
        var tensors = parameters.OfType<InstructionTensorParameter>().ToArray();
        if (tensors.Any(parameter => parameter.ByteOffset != 0))
            throw new InstructionAdaptationException(CollectionId, Name, "Put typed offsets on calls, not instruction parameters.");
        var operands = tensors.Select(parameter =>
            new InstructionOperandDescription(parameter.Name, parameter.Tensor, parameter.Access, parameter.Source)).ToArray();
        if (!Definition.TryAdapt(operands, new(GraphElementType.Float32, GraphElementType.Float32), out var diagnostic))
            throw new InstructionAdaptationException(CollectionId, Name, diagnostic);
    }
}
