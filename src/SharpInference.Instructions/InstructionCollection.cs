using SharpInference.Graphs;
using System.Reflection;

namespace SharpInference.Instructions;

public static class InstructionCollectionIds
{
    public static readonly Guid TierZeroFloat32 = Guid.Empty;
    public static readonly Guid TierZeroFloat16 = new("00000000-0000-0000-0000-000000000001");
}

public readonly record struct InstructionTarget
{
    public InstructionTarget(string architecture)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(architecture);
        Architecture = architecture;
    }
    public string Architecture { get; }
    public static readonly InstructionTarget Cpu = new("cpu.managed");
    public static readonly InstructionTarget Direct3D12 = new("direct3d12");
    public override string ToString() => Architecture;
}
public enum InstructionSynchronization { None, GroupMemoryBarrier }
public sealed record InstructionCollectionDescription(Guid Id, string Name, int Tier, InstructionTarget Architecture);
public sealed record InstructionPort(string Name, GraphElementType ElementType, GraphResourceAccess Access);
public sealed record InstructionSignature
{
    public InstructionSignature(IReadOnlyList<InstructionPort> ports, IReadOnlyList<string> attributes,
        KernelPrecisionProfile precision)
    {
        ArgumentNullException.ThrowIfNull(ports);
        ArgumentNullException.ThrowIfNull(attributes);
        ArgumentNullException.ThrowIfNull(precision);
        Ports = Array.AsReadOnly(ports.ToArray());
        Attributes = Array.AsReadOnly(attributes.ToArray());
        Precision = precision;
    }
    public IReadOnlyList<InstructionPort> Ports { get; }
    public IReadOnlyList<string> Attributes { get; }
    public KernelPrecisionProfile Precision { get; }
}
public sealed record InstructionIndexBound(string IndexPort, string TensorPort, int Axis);

public abstract record InstructionParameter(string Name);
public sealed record InstructionTensorParameter(
    string Name, TensorDescriptor Tensor, GraphResourceAccess Access, string Source,
    string Expression, string OffsetExpression, ulong ByteOffset = 0) : InstructionParameter(Name);
public sealed record InstructionAttributeParameter(string Name, string Value) : InstructionParameter(Name);

public sealed record InstructionHelper(string Name, string Source);
public sealed class InstructionRecording
{
    public InstructionRecording(string source, IEnumerable<InstructionHelper>? helpers = null,
        InstructionSynchronization synchronization = InstructionSynchronization.None,
        IEnumerable<Assembly>? references = null)
    {
        Source = source ?? throw new ArgumentNullException(nameof(source));
        Helpers = Array.AsReadOnly(helpers?.ToArray() ?? []);
        Synchronization = synchronization;
        References = Array.AsReadOnly(references?.Distinct().ToArray() ?? []);
    }
    public string Source { get; }
    public IReadOnlyList<InstructionHelper> Helpers { get; }
    public InstructionSynchronization Synchronization { get; }
    public IReadOnlyList<Assembly> References { get; }
}

public interface IInstructionRecorder
{
    InstructionTarget Target { get; }
    void Record(InstructionRecording recording);
}

public abstract class Instruction
{
    public abstract Guid CollectionId { get; }
    public abstract string Name { get; }
    public abstract InstructionTarget Target { get; }
    public abstract IReadOnlyList<InstructionSignature> Signatures { get; }
    public virtual bool RequiresDispatchIsolation => true;
    public virtual IReadOnlyList<InstructionIndexBound> IndexBounds => [];
    public void Invoke(IInstructionRecorder recorder, InstructionParameter[] parameters, PrecisionRequirement precision)
    {
        ArgumentNullException.ThrowIfNull(recorder);
        if (recorder.Target != Target)
            throw new InstructionAdaptationException(CollectionId, Name, $"requires {Target}, recorder is {recorder.Target}.");
        GetSignature(parameters, precision);
        Validate(parameters);
        recorder.Record(Generate(parameters));
    }

    protected virtual void Validate(InstructionParameter[] parameters) { }
    protected abstract InstructionRecording Generate(InstructionParameter[] parameters);

    public InstructionSignature GetSignature(InstructionParameter[] parameters, PrecisionRequirement precision)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(precision);
        if (parameters.Any(parameter => parameter is null) ||
            parameters.Select(parameter => parameter.Name).Distinct(StringComparer.Ordinal).Count() != parameters.Length ||
            parameters.Any(parameter => parameter is not (InstructionTensorParameter or InstructionAttributeParameter)))
            throw new InstructionAdaptationException(CollectionId, Name, "Invalid or duplicate parameters.");
        var tensors = parameters.OfType<InstructionTensorParameter>().ToArray();
        var attributes = parameters.OfType<InstructionAttributeParameter>().Select(parameter => parameter.Name).ToHashSet(StringComparer.Ordinal);
        var matches = Signatures.Where(signature => signature.Ports.Count == tensors.Length &&
            attributes.SetEquals(signature.Attributes) && signature.Ports.All(port => tensors.Any(parameter =>
                parameter.Name == port.Name && parameter.Tensor.ElementType == port.ElementType &&
                (port.Access == GraphResourceAccess.Read || parameter.Access != GraphResourceAccess.Read)))).ToArray();
        if (matches.Length == 0)
            throw new InstructionAdaptationException(CollectionId, Name, "No supported parameter signature.");
        var compatible = matches.Where(signature => NumericTypeCompatibility.Satisfies(signature.Precision, precision)).ToArray();
        if (compatible.Length == 0)
            throw new InstructionAdaptationException(CollectionId, Name,
                $"On {Target}, required arithmetic={precision.MinimumArithmeticType}, accumulator={precision.MinimumAccumulatorType}; " +
                "provided " + string.Join("; ", matches.Select(signature =>
                    $"arithmetic={signature.Precision.ArithmeticType}, accumulator={signature.Precision.AccumulatorType}")) + ".");
        if (compatible.Length != 1)
            throw new InstructionAdaptationException(CollectionId, Name, "Ambiguous supported parameter/precision signatures.");
        return compatible[0];
    }
}

public interface IInstructionCollectionProvider
{
    IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection();
    IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName);
}

public sealed class InstructionAdaptationException : InvalidOperationException
{
    public InstructionAdaptationException(Guid collectionId, string name, string reason, Exception? inner = null)
        : base($"Instruction '{collectionId:D}/{name}' cannot adapt parameters: {reason}", inner) { }
}

public sealed class InstructionRegistry : IInstructionCollectionProvider
{
    private readonly IReadOnlyList<IInstructionCollectionProvider> providers;
    private readonly IReadOnlyList<InstructionCollectionDescription> collections;

    public InstructionRegistry(IEnumerable<IInstructionCollectionProvider> providers)
    {
        ArgumentNullException.ThrowIfNull(providers);
        this.providers = Array.AsReadOnly(providers.ToArray());
        if (this.providers.Any(provider => provider is null))
            throw new ArgumentException("IC providers cannot contain null.", nameof(providers));
        var descriptions = this.providers.SelectMany(provider => provider.QueryInstructionCollection()).ToArray();
        foreach (var group in descriptions.GroupBy(description => (description.Id, description.Architecture)))
            if (group.Distinct().Count() != 1)
                throw new InvalidDataException($"Conflicting IC descriptions for '{group.Key}'.");
        if (descriptions.Any(description => string.IsNullOrWhiteSpace(description.Name) || description.Tier < 0 ||
            string.IsNullOrWhiteSpace(description.Architecture.Architecture)))
            throw new InvalidDataException("IC descriptions require a name and non-negative Tier.");
        collections = Array.AsReadOnly(descriptions.Distinct().OrderBy(description => description.Id)
            .ThenBy(description => description.Architecture.Architecture, StringComparer.Ordinal).ToArray());
    }

    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() => collections;

    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(instructionName);
        if (!collections.Any(collection => collection.Id == collectionId))
            throw new NotSupportedException($"Instruction collection '{collectionId:D}' is not installed.");
        var values = providers.Where(provider => provider.QueryInstructionCollection()
                .Any(collection => collection.Id == collectionId))
            .SelectMany(provider => provider.QueryInstruction(collectionId, instructionName)).ToArray();
        if (values.Any(value => value.CollectionId != collectionId || value.Name != instructionName ||
            !collections.Any(collection => collection.Id == collectionId && collection.Architecture == value.Target)))
            throw new InvalidDataException("Provider returned an instruction with a different identity.");
        if (values.GroupBy(value => value.Target).Any(group => group.Count() > 1))
            throw new InvalidDataException($"Ambiguous implementations for '{collectionId:D}/{instructionName}'.");
        if (values.SelectMany(value => value.Signatures).Any(signature =>
            signature.Ports.Select(port => port.Name).Distinct(StringComparer.Ordinal).Count() != signature.Ports.Count ||
            signature.Attributes.Distinct(StringComparer.Ordinal).Count() != signature.Attributes.Count ||
            signature.Ports.Any(port => string.IsNullOrWhiteSpace(port.Name) || !Enum.IsDefined(port.ElementType) ||
                !Enum.IsDefined(port.Access))))
            throw new InvalidDataException($"Invalid signatures for '{collectionId:D}/{instructionName}'.");
        return Array.AsReadOnly(values);
    }

    public Instruction Resolve(Guid collectionId, string instructionName, InstructionTarget target) =>
        QueryInstruction(collectionId, instructionName).SingleOrDefault(instruction => instruction.Target == target)
        ?? throw new NotSupportedException($"No {target} implementation for '{collectionId:D}/{instructionName}'.");
}
