namespace SharpInference.Graphs;

public sealed record GraphOperationValidationContext(
    string NodeId,
    GraphOperationId Operation,
    IReadOnlyList<NodeResourceBinding> Bindings,
    IReadOnlyDictionary<ResourceId, GraphResource> Resources,
    IReadOnlyDictionary<string, string> Attributes,
    PrecisionRequirement Requirements)
{
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
    void Validate(GraphOperationValidationContext context);
}
