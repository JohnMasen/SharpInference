using SharpInference.Graphs.Execution;
using SharpInference.Graphs;
using System.Reflection;

namespace SharpInference.Instructions;

/// <summary>Specifies synchronization required after a recorded instruction.</summary>
public enum InstructionSynchronization
{
    /// <summary>Requires no additional synchronization.</summary>
    None,
    /// <summary>Requires a group-memory barrier and thread-group synchronization.</summary>
    GroupMemoryBarrier
}
/// <summary>Describes an instruction collection's identity, tier, and target architecture.</summary>
/// <param name="Id">The stable instruction collection identity.</param>
/// <param name="Name">The human-readable instruction collection name.</param>
/// <param name="Tier">The nonnegative instruction optimization tier.</param>
/// <param name="Architecture">The architecture identifier used to select implementations.</param>
public sealed record InstructionCollectionDescription(
    Guid Id, string Name, int Tier, ExecutionTarget Architecture, string Domain = "core",
    InstructionCategory? Category = null, IReadOnlyList<string>? RequiredInstructions = null);
/// <summary>Declares a named tensor port's element type and resource access.</summary>
/// <param name="Name">The tensor port name used to match invocation parameters.</param>
/// <param name="ElementType">The scalar storage representation of each tensor element.</param>
/// <param name="Access">The permitted resource access.</param>
public sealed record InstructionPort(
    string Name, ElementType ElementType, ResourceAccess Access,
    string? Meaning = null, ElementType? InternalPrecision = null,
    string? ShapeConstraint = null);
/// <summary>Describes supported tensor ports, attribute names, and internal arithmetic precision.</summary>
public sealed record InstructionSignature
{
    /// <summary>Snapshots supported tensor ports, required attribute names, and internal precision.</summary>
    /// <param name="ports">The tensor ports supported by this signature.</param>
    /// <param name="attributes">The attribute names required by the signature.</param>
    /// <param name="precision">The supported internal arithmetic and accumulator precision.</param>
    public InstructionSignature(IReadOnlyList<InstructionPort> ports, IReadOnlyList<string> attributes,
        ExecutionPrecision precision)
    {
        ArgumentNullException.ThrowIfNull(ports);
        ArgumentNullException.ThrowIfNull(attributes);
        ArgumentNullException.ThrowIfNull(precision);
        Ports = Array.AsReadOnly(ports.ToArray());
        Attributes = Array.AsReadOnly(attributes.ToArray());
        Precision = precision;
    }
    /// <summary>Gets the tensor ports supported by this signature.</summary>
    public IReadOnlyList<InstructionPort> Ports { get; }
    /// <summary>Gets the attribute names required by the signature.</summary>
    public IReadOnlyList<string> Attributes { get; }
    /// <summary>Gets the supported internal arithmetic and accumulator precision.</summary>
    public ExecutionPrecision Precision { get; }
}
/// <summary>Constrains an index port using a tensor dimension and a minimum valid index.</summary>
/// <param name="IndexPort">The port supplying index values to validate.</param>
/// <param name="TensorPort">The port whose dimension provides the exclusive upper index bound.</param>
/// <param name="Axis">The zero-based tensor dimension used as the upper index bound.</param>
/// <param name="MinimumIndex">The inclusive lower bound for valid index values.</param>
public sealed record InstructionIndexBound(string IndexPort, string TensorPort, int Axis, int MinimumIndex = 0);

/// <summary>Represents a named tensor or attribute argument supplied to an instruction.</summary>
/// <param name="Name">The instruction parameter name matched against signature ports or attributes.</param>
public abstract record InstructionParameter(string Name);
/// <summary>Describes a tensor argument and the source expressions used by code generation.</summary>
/// <param name="Name">The tensor port name matched against the instruction signature.</param>
/// <param name="Tensor">The tensor descriptor defining element representation and shape.</param>
/// <param name="Access">The permitted resource access.</param>
/// <param name="Source">The source resource identity.</param>
/// <param name="Expression">The generated expression used to access tensor elements.</param>
/// <param name="OffsetExpression">The generated expression for the tensor's typed element offset.</param>
/// <param name="ByteOffset">The byte offset within the source resource.</param>
public sealed record InstructionTensorParameter(
    string Name, TensorDescriptor Tensor, ResourceAccess Access, string Source,
    string Expression, string OffsetExpression, ulong ByteOffset = 0) : InstructionParameter(Name);
/// <summary>Supplies a named string attribute to an instruction.</summary>
/// <param name="Name">The attribute name required by the instruction signature.</param>
/// <param name="Value">The textual attribute value supplied to code generation.</param>
public sealed record InstructionAttributeParameter(string Name, string Value) : InstructionParameter(Name);

/// <summary>Contains a named helper declaration required by generated instruction source.</summary>
/// <param name="Name">The helper declaration name used to deduplicate generated source.</param>
/// <param name="Source">The source code defining the helper.</param>
public sealed record InstructionHelper(string Name, string Source);
/// <summary>Contains generated instruction source, helper declarations, synchronization, and assembly references.</summary>
public sealed class InstructionRecording
{
    /// <summary>Snapshots generated source, helpers, synchronization, and distinct assembly references.</summary>
    /// <param name="source">The generated source fragment implementing the invocation.</param>
    /// <param name="helpers">The named helper declarations required by the recorded source.</param>
    /// <param name="synchronization">The synchronization required after the recorded source executes.</param>
    /// <param name="references">The distinct assemblies required to compile the recorded source.</param>
    public InstructionRecording(string source, IEnumerable<InstructionHelper>? helpers = null,
        InstructionSynchronization synchronization = InstructionSynchronization.None,
        IEnumerable<Assembly>? references = null)
    {
        Source = source ?? throw new ArgumentNullException(nameof(source));
        Helpers = Array.AsReadOnly(helpers?.ToArray() ?? []);
        Synchronization = synchronization;
        References = Array.AsReadOnly(references?.Distinct().ToArray() ?? []);
    }
    /// <summary>Gets the generated source fragment implementing the invocation.</summary>
    public string Source { get; }
    /// <summary>Gets the named helper declarations required by the recorded source.</summary>
    public IReadOnlyList<InstructionHelper> Helpers { get; }
    /// <summary>Gets the synchronization required after the recorded source executes.</summary>
    public InstructionSynchronization Synchronization { get; }
    /// <summary>Gets the distinct assemblies required to compile the recorded source.</summary>
    public IReadOnlyList<Assembly> References { get; }
}

/// <summary>Receives instruction source recordings for a particular execution architecture.</summary>
public interface IInstructionRecorder
{
    /// <summary>Gets the execution architecture required by this contract.</summary>
    ExecutionTarget Target { get; }
    /// <summary>Records an instruction's generated source and compilation dependencies for this target.</summary>
    /// <param name="recording">The source, helpers, synchronization, and references to record.</param>
    void Record(InstructionRecording recording);
}

/// <summary>Adapts invocation parameters and precision requirements to a target-specific source implementation.</summary>
public abstract class Instruction
{
    public virtual string? Comment => null;
    public virtual IReadOnlyList<InstructionAttributeDescription> AttributeDescriptions => [];
    /// <summary>Gets the identity of the collection providing this implementation.</summary>
    public abstract Guid CollectionId { get; }
    /// <summary>Gets the registered instruction name within its collection.</summary>
    public abstract string Name { get; }
    /// <summary>Gets the execution architecture required by this contract.</summary>
    public abstract ExecutionTarget Target { get; }
    /// <summary>Gets the parameter and precision signatures supported by this implementation.</summary>
    public abstract IReadOnlyList<InstructionSignature> Signatures { get; }
    /// <summary>Gets whether the instruction must execute in a kernel without other operations.</summary>
    public virtual bool RequiresDispatchIsolation => true;
    /// <summary>Gets the index-port bounds required by the instruction.</summary>
    public virtual IReadOnlyList<InstructionIndexBound> IndexBounds => [];
    /// <summary>Adapts invocation parameters to the target and records the configured implementation source.</summary>
    /// <param name="recorder">The target-specific recorder receiving generated instruction source.</param>
    /// <param name="parameters">The named tensor and attribute parameters to adapt.</param>
    /// <param name="precision">The minimum arithmetic and accumulator precision required by the invocation.</param>
    /// <param name="executionConfiguration">The selected execution implementation, or null when no configuration is required.</param>
    public void Invoke(IInstructionRecorder recorder, InstructionParameter[] parameters, PrecisionRequirement precision,
        InstructionExecutionConfiguration? executionConfiguration = null)
    {
        ArgumentNullException.ThrowIfNull(recorder);
        if (recorder.Target != Target)
        {
            throw new InstructionAdaptationException(CollectionId, Name, $"requires {Target}, recorder is {recorder.Target}.");
        }

        Adapt(parameters, precision, executionConfiguration);
        recorder.Record(GenerateConfigured(parameters, executionConfiguration));
    }

    /// <summary>Selects a compatible signature and validates invocation-specific tensor constraints.</summary>
    /// <param name="parameters">The named tensor and attribute parameters to adapt.</param>
    /// <param name="precision">The minimum arithmetic and accumulator precision required by the invocation.</param>
    /// <param name="executionConfiguration">The selected execution implementation, or null when no configuration is required.</param>
    /// <returns>The supported signature after invocation-specific validation succeeds.</returns>
    /// <exception cref="InstructionAdaptationException">The invocation violates a parameter, precision, or implementation constraint.</exception>
    public InstructionSignature Adapt(InstructionParameter[] parameters, PrecisionRequirement precision,
        InstructionExecutionConfiguration? executionConfiguration = null)
    {
        var signature = GetSignature(parameters, precision, executionConfiguration);
        Validate(parameters);
        return signature;
    }

    /// <summary>Checks that the requested execution implementation is supported by this instruction.</summary>
    /// <param name="configuration">The selected implementation configuration, or null for the default implementation.</param>
    protected virtual void ValidateExecutionConfiguration(InstructionExecutionConfiguration? configuration)
    {
        if (configuration is not null)
        {
            throw new InstructionAdaptationException(CollectionId, Name, "This instruction does not accept an execution configuration.");
        }
    }

    /// <summary>Validates invocation-specific tensor constraints after signature selection.</summary>
    /// <param name="parameters">The named tensor and attribute parameters to adapt.</param>
    protected virtual void Validate(InstructionParameter[] parameters) { }
    /// <summary>Generates source and helper declarations for validated instruction parameters.</summary>
    /// <param name="parameters">The named tensor and attribute parameters to adapt.</param>
    protected abstract InstructionRecording Generate(InstructionParameter[] parameters);
    /// <summary>Generates source for validated parameters using the requested execution implementation.</summary>
    /// <param name="parameters">The named tensor and attribute parameters to adapt.</param>
    /// <param name="configuration">The selected implementation configuration, or null for the default implementation.</param>
    protected virtual InstructionRecording GenerateConfigured(InstructionParameter[] parameters,
        InstructionExecutionConfiguration? configuration) => Generate(parameters);

    /// <summary>Selects the unique signature satisfying port types, access, attributes, and precision requirements.</summary>
    /// <param name="parameters">The named tensor and attribute parameters to adapt.</param>
    /// <param name="precision">The minimum arithmetic and accumulator precision required by the invocation.</param>
    /// <param name="executionConfiguration">The selected execution implementation, or null when no configuration is required.</param>
    /// <returns>The unique supported signature satisfying the supplied parameters and precision requirement.</returns>
    /// <exception cref="InstructionAdaptationException">No unique compatible signature or supported execution configuration exists.</exception>
    public InstructionSignature GetSignature(InstructionParameter[] parameters, PrecisionRequirement precision,
        InstructionExecutionConfiguration? executionConfiguration = null)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(precision);
        if (executionConfiguration is not null && executionConfiguration.Target != Target)
        {
            throw new InstructionAdaptationException(CollectionId, Name, "Execution configuration has a different target.");
        }

        ValidateExecutionConfiguration(executionConfiguration);
        if (parameters.Any(parameter => parameter is null) ||
            parameters.Select(parameter => parameter.Name).Distinct(StringComparer.Ordinal).Count() != parameters.Length ||
            parameters.Any(parameter => parameter is not (InstructionTensorParameter or InstructionAttributeParameter)))
        {
            throw new InstructionAdaptationException(CollectionId, Name, "Invalid or duplicate parameters.");
        }

        var tensors = parameters.OfType<InstructionTensorParameter>().ToArray();
        var attributes = parameters.OfType<InstructionAttributeParameter>().Select(parameter => parameter.Name).ToHashSet(StringComparer.Ordinal);
        var matches = Signatures.Where(signature => signature.Ports.Count == tensors.Length &&
            attributes.SetEquals(signature.Attributes) && signature.Ports.All(port => tensors.Any(parameter =>
                parameter.Name == port.Name && parameter.Tensor.ElementType == port.ElementType &&
                (port.Access == ResourceAccess.Read || parameter.Access != ResourceAccess.Read)))).ToArray();
        if (matches.Length == 0)
        {
            throw new InstructionAdaptationException(CollectionId, Name, "No supported parameter signature.");
        }

        var compatible = matches.Where(signature => InstructionPrecisionCompatibility.Satisfies(signature.Precision, precision)).ToArray();
        if (compatible.Length == 0)
        {
            throw new InstructionAdaptationException(CollectionId, Name,
                $"On {Target}, required arithmetic={precision.MinimumArithmeticType}, accumulator={precision.MinimumAccumulatorType}; " +
                "provided " + string.Join("; ", matches.Select(signature =>
                    $"arithmetic={signature.Precision.ArithmeticType}, accumulator={signature.Precision.AccumulatorType}")) + ".");
        }

        if (compatible.Length != 1)
        {
            throw new InstructionAdaptationException(CollectionId, Name, "Ambiguous supported parameter/precision signatures.");
        }

        return compatible[0];
    }
}

/// <summary>Discovers installed instruction collections and their named implementations.</summary>
public interface IInstructionCollectionProvider
{
    IReadOnlyList<Instruction> EnumerateInstructions();
    /// <summary>Returns descriptions of instruction collections exposed by the provider.</summary>
    IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection();
    /// <summary>Returns available target implementations of the specified collection and instruction name.</summary>
    /// <param name="collectionId">The identity of the collection providing this implementation.</param>
    /// <param name="instructionName">The registered instruction name.</param>
    IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName);
}

/// <summary>Reports why a registered instruction cannot accept an invocation.</summary>
public sealed class InstructionAdaptationException : InvalidOperationException
{
    /// <summary>Creates an adaptation failure identifying the instruction and rejection reason.</summary>
    /// <param name="collectionId">The identity of the collection providing this implementation.</param>
    /// <param name="name">The registered name of the instruction whose adaptation failed.</param>
    /// <param name="reason">The reason the instruction cannot adapt the invocation.</param>
    /// <param name="inner">The underlying adaptation failure, if present.</param>
    public InstructionAdaptationException(Guid collectionId, string name, string reason, Exception? inner = null)
        : base($"Instruction '{collectionId:D}/{name}' cannot adapt parameters: {reason}", inner) { }
}

/// <summary>Combines providers and validates unique instruction identities, signatures, and graph mappings.</summary>
public sealed class InstructionRegistry : IInstructionCatalog, IInstructionOptimizationProvider, IGraphInstructionProvider
{
    private readonly IReadOnlyList<IInstructionCollectionProvider> providers;
    private readonly IReadOnlyList<InstructionCollectionDescription> collections;
    private readonly IReadOnlyList<Instruction> instructions;
    public IReadOnlyList<InstructionCatalogRow> Rows { get; }
    public IReadOnlyList<InstructionParameterRow> Parameters { get; }
    public IReadOnlyList<InstructionSignatureRow> Signatures { get; }
    public IReadOnlyList<Instruction> EnumerateInstructions() => instructions;

    /// <summary>Snapshots providers and validates their collection identities and target descriptions.</summary>
    /// <param name="providers">The instruction collection providers to combine.</param>
    public InstructionRegistry(IEnumerable<IInstructionCollectionProvider> providers)
    {
        ArgumentNullException.ThrowIfNull(providers);
        this.providers = Array.AsReadOnly(providers.ToArray());
        if (this.providers.Any(provider => provider is null))
        {
            throw new ArgumentException("IC providers cannot contain null.", nameof(providers));
        }

        var descriptions = this.providers.SelectMany(provider => provider.QueryInstructionCollection()).ToArray();
        foreach (var group in descriptions.GroupBy(description => (description.Id, description.Architecture)))
        {
            if (group.Count() != 1)
            {
                throw new InvalidDataException($"Multiple providers own instruction set '{group.Key}'. Sets cannot be merged or overridden.");
            }
        }

        if (descriptions.Any(description => string.IsNullOrWhiteSpace(description.Name) || description.Tier < 0 ||
            string.IsNullOrWhiteSpace(description.Architecture.Architecture) ||
            string.IsNullOrWhiteSpace(description.Domain) ||
            description.Category is { } category && !Enum.IsDefined(category) ||
            description.RequiredInstructions is { } required &&
                (required.Any(string.IsNullOrWhiteSpace) || required.Distinct(StringComparer.Ordinal).Count() != required.Count)))
        {
            throw new InvalidDataException("IC descriptions require a name and non-negative Tier.");
        }

        collections = Array.AsReadOnly(descriptions.Select(description => description with
            { RequiredInstructions = description.RequiredInstructions is { } names ? Array.AsReadOnly(names.ToArray()) : null })
            .OrderBy(description => description.Id)
            .ThenBy(description => description.Architecture.Architecture, StringComparer.Ordinal).ToArray());
        var all = new List<Instruction>();
        foreach (var provider in this.providers)
        {
            var owned = provider.QueryInstructionCollection();
            var supplied = provider.EnumerateInstructions();
            if (supplied.Any(instruction => instruction is null ||
                    !owned.Any(collection => collection.Id == instruction.CollectionId &&
                        collection.Architecture == instruction.Target)) ||
                supplied.GroupBy(instruction => (instruction.CollectionId, instruction.Target, instruction.Name))
                    .Any(group => group.Count() != 1))
            {
                throw new InvalidDataException($"Provider '{provider.GetType().FullName}' returned duplicate or unowned instructions.");
            }
            foreach (var collection in owned)
            {
                if (provider is IInstructionProvider && collection.RequiredInstructions is null)
                {
                    throw new InvalidDataException(
                        $"Provider '{provider.GetType().FullName}' must declare the complete manifest for '{collection.Name}'.");
                }
                var names = supplied.Where(instruction => instruction.CollectionId == collection.Id &&
                    instruction.Target == collection.Architecture).Select(instruction => instruction.Name)
                    .ToHashSet(StringComparer.Ordinal);
                if (collection.RequiredInstructions is { } required &&
                    !names.SetEquals(required))
                {
                    throw new InvalidDataException(
                        $"Incomplete instruction set '{collection.Name}' from '{provider.GetType().FullName}'. " +
                        $"Missing: {string.Join(", ", required.Except(names))}; unexpected: {string.Join(", ", names.Except(required))}.");
                }
            }
            all.AddRange(supplied);
        }
        instructions = Array.AsReadOnly(all.OrderBy(instruction => instruction.Target.Architecture, StringComparer.Ordinal)
            .ThenBy(instruction => instruction.CollectionId).ThenBy(instruction => instruction.Name, StringComparer.Ordinal).ToArray());
        foreach (var instruction in instructions)
        {
            _ = QueryInstruction(instruction.CollectionId, instruction.Name);
        }
        Rows = Array.AsReadOnly(instructions.Select(instruction =>
        {
            var collection = collections.Single(collection => collection.Id == instruction.CollectionId &&
                collection.Architecture == instruction.Target);
            return new InstructionCatalogRow(instruction.Target, instruction.CollectionId, instruction.Name,
                collection.Category ?? (collection.Tier == 0 ? InstructionCategory.Tier0 : InstructionCategory.Tier1),
                collection.Domain, instruction.Comment);
        }).ToArray());
        Signatures = Array.AsReadOnly(instructions.SelectMany(instruction => instruction.Signatures.Select(
            (signature, index) => new InstructionSignatureRow(instruction.Target, instruction.CollectionId,
                instruction.Name, index, signature.Precision.ArithmeticType, signature.Precision.AccumulatorType))).ToArray());
        Parameters = Array.AsReadOnly(instructions.SelectMany(instruction => instruction.Signatures
            .SelectMany((signature, index) => signature.Ports.Select(port => new InstructionParameterRow(
                instruction.Target, instruction.CollectionId, instruction.Name, index,
                port.Name, port.Meaning, "Tensor", port.Access, port.ElementType.ToString(),
                port.InternalPrecision ?? (port.ElementType is ElementType.Float16 or ElementType.Float32
                    ? signature.Precision.ArithmeticType : null), port.ShapeConstraint))
                .Concat(signature.Attributes.Select(name =>
                {
                    var description = instruction.AttributeDescriptions.SingleOrDefault(attribute => attribute.Name == name);
                    return new InstructionParameterRow(instruction.Target, instruction.CollectionId, instruction.Name,
                        index, name, description?.Meaning, "Attribute", null,
                        description?.DataType ?? "String", null, description?.Constraints);
                })))).ToArray());
    }

    /// <summary>Returns descriptions of instruction collections exposed by the provider.</summary>
    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() => collections;

    /// <summary>Returns registered mappings from graph operations to target instructions.</summary>
    public IReadOnlyList<GraphInstructionBinding> QueryGraphInstructionBindings()
    {
        var bindings = providers.OfType<IGraphInstructionProvider>()
            .SelectMany(provider => provider.QueryGraphInstructionBindings()).ToArray();
        if (bindings.Any(binding => binding is null || string.IsNullOrWhiteSpace(binding.Operation.Name) ||
            binding.Operation.Version <= 0 || string.IsNullOrWhiteSpace(binding.InstructionName)))
        {
            throw new InvalidDataException("Graph instruction bindings require a versioned operation and instruction name.");
        }

        if (bindings.GroupBy(binding => (binding.Operation, binding.Target)).Any(group => group.Count() != 1))
        {
            throw new InvalidDataException("Ambiguous graph instruction bindings for the same operation and target.");
        }

        foreach (var binding in bindings)
        {
            var instruction = Resolve(binding.CollectionId, binding.InstructionName, binding.Target);
            var ports = binding.ReadWritePorts ?? [];
            if (ports.Any(string.IsNullOrWhiteSpace) || ports.Distinct(StringComparer.Ordinal).Count() != ports.Count ||
                !instruction.Signatures.Any(signature => ports.ToHashSet(StringComparer.Ordinal).SetEquals(
                    signature.Ports.Where(port => port.Access == ResourceAccess.ReadWrite).Select(port => port.Name))))
            {
                throw new InvalidDataException($"Graph mapping '{binding.Operation}' has incompatible read/write ports.");
            }
        }
        return Array.AsReadOnly(bindings);
    }

    /// <summary>Returns instruction implementations available for graph fusion selection.</summary>
    public IReadOnlyList<InstructionOptimizationCapability> QueryOptimizationCapabilities()
    {
        var capabilities = providers.OfType<IInstructionOptimizationProvider>()
            .SelectMany(provider => provider.QueryOptimizationCapabilities()).ToArray();
        if (capabilities.GroupBy(value => (value.CollectionId, value.Name, value.Target, value.Configuration))
            .Any(group => group.Count() != 1))
        {
            throw new InvalidDataException("Ambiguous optimization capabilities.");
        }

        foreach (var capability in capabilities)
        {
            Resolve(capability.CollectionId, capability.Name, capability.Target);
        }

        return Array.AsReadOnly(capabilities);
    }

    /// <summary>Returns available target implementations of the specified collection and instruction name.</summary>
    /// <param name="collectionId">The identity of the collection providing this implementation.</param>
    /// <param name="instructionName">The registered instruction name.</param>
    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(instructionName);
        if (!collections.Any(collection => collection.Id == collectionId))
        {
            throw new NotSupportedException($"Instruction collection '{collectionId:D}' is not installed.");
        }

        var values = instructions.Where(instruction =>
            instruction.CollectionId == collectionId && instruction.Name == instructionName).ToArray();
        if (values.Any(value => value.CollectionId != collectionId || value.Name != instructionName ||
            !collections.Any(collection => collection.Id == collectionId && collection.Architecture == value.Target)))
        {
            throw new InvalidDataException("Provider returned an instruction with a different identity.");
        }

        if (values.GroupBy(value => value.Target).Any(group => group.Count() > 1))
        {
            throw new InvalidDataException($"Ambiguous implementations for '{collectionId:D}/{instructionName}'.");
        }

        if (values.SelectMany(value => value.Signatures).Any(signature =>
            signature.Ports.Select(port => port.Name).Distinct(StringComparer.Ordinal).Count() != signature.Ports.Count ||
            signature.Attributes.Distinct(StringComparer.Ordinal).Count() != signature.Attributes.Count ||
            signature.Ports.Any(port => string.IsNullOrWhiteSpace(port.Name) || !Enum.IsDefined(port.ElementType) ||
                !Enum.IsDefined(port.Access))))
        {
            throw new InvalidDataException($"Invalid signatures for '{collectionId:D}/{instructionName}'.");
        }
        if (values.Any(value => string.IsNullOrWhiteSpace(value.Name) || value.Signatures.Count == 0 ||
                value.AttributeDescriptions.Select(attribute => attribute.Name).Distinct(StringComparer.Ordinal).Count() !=
                    value.AttributeDescriptions.Count))
        {
            throw new InvalidDataException($"Missing or invalid instruction contract for '{collectionId:D}/{instructionName}'.");
        }

        return Array.AsReadOnly(values);
    }

    /// <summary>Returns the unique implementation matching the collection, name, and execution target.</summary>
    /// <param name="collectionId">The identity of the collection providing this implementation.</param>
    /// <param name="instructionName">The registered instruction name.</param>
    /// <param name="target">The instruction architecture for implementation resolution.</param>
    public Instruction Resolve(Guid collectionId, string instructionName, ExecutionTarget target) =>
        QueryInstruction(collectionId, instructionName).SingleOrDefault(instruction => instruction.Target == target)
        ?? throw new NotSupportedException($"No {target} implementation for '{collectionId:D}/{instructionName}'.");
}
