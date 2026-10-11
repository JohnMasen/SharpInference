namespace SharpInference.Graphs;

/// <summary>Provides operation metadata and bound tensor descriptors to an operation validator.</summary>
/// <param name="NodeId">The failing execution node identifier, or null for a graph-level failure.</param>
/// <param name="Operation">The operation contract identifier.</param>
/// <param name="Bindings">The node or call port resource bindings.</param>
/// <param name="Resources">The graph resource descriptors.</param>
/// <param name="Attributes">The operation or model attributes keyed by name.</param>
/// <param name="Requirements">The operation precision requirements.</param>
public sealed record GraphOperationValidationContext(
    string NodeId,
    GraphOperationId Operation,
    IReadOnlyList<NodeResourceBinding> Bindings,
    IReadOnlyDictionary<ResourceId, GraphResource> Resources,
    IReadOnlyDictionary<string, string> Attributes,
    PrecisionRequirement Requirements)
{
    /// <summary>Returns the bound tensor descriptor for a port, applying its view when present.</summary>
    /// <param name="port">The node port name.</param>
    public TensorDescriptor Tensor(string port)
    {
        var binding = Bindings.Single(binding => binding.Port == port);
        var resource = Resources[binding.Resource];
        binding.View?.Validate(resource.Tensor);
        return binding.View?.Tensor ?? resource.Tensor;
    }
}

/// <summary>Supplies registered operation semantics independently of structural graph validation.</summary>
public interface IGraphOperationValidator
{
    /// <summary>Validates structural, binding, and operation contracts, throwing when a contract is violated.</summary>
    /// <param name="context">The graph and operation metadata to inspect.</param>
    void Validate(GraphOperationValidationContext context);
}
