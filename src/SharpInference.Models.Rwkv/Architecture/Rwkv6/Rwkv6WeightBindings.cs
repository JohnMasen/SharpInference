using SharpInference.Graphs;

namespace SharpInference.Architectures.Rwkv6;

internal sealed class Rwkv6WeightBindings
{
    private readonly IReadOnlyDictionary<string, IModelTensor> weights;

    public Rwkv6WeightBindings(ExecutionGraph graph, IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(tensors);
        var modelDimensions = RwkvTensorCatalogDimensions.Read(tensors);
        if (graph.Identity.ArchitectureId != "rwkv-6" ||
            graph.Model.Dimensions["vocabulary"] != modelDimensions.VocabularySize ||
            graph.Model.Dimensions["embedding"] != modelDimensions.EmbeddingSize ||
            graph.Model.Dimensions["layers"] != modelDimensions.LayerCount)
        {
            throw new InvalidDataException("The RWKV-6 execution graph does not match the model catalog.");
        }

        var resolved = new Dictionary<string, IModelTensor>(StringComparer.Ordinal);
        foreach (var resource in graph.Resources.Where(resource => resource.Kind == GraphResourceKind.Weight))
        {
            if (resource.Lifetime != GraphResourceLifetime.Model ||
                string.IsNullOrWhiteSpace(resource.BindingKey))
            {
                throw new InvalidDataException($"Weight resource '{resource.Id}' lacks a model binding key.");
            }

            var tensor = tensors.GetRequired(resource.BindingKey);
            var elementType = tensor.DataType switch
            {
                TensorDataType.Float16 => GraphElementType.Float16,
                TensorDataType.Float32 => GraphElementType.Float32,
                _ => throw new InvalidDataException($"Tensor '{tensor.Name}' has an unsupported data type."),
            };
            if (resource.Tensor.Layout != "dense" ||
                resource.Tensor.ElementType != elementType ||
                !resource.Tensor.Dimensions.SequenceEqual(tensor.Dimensions) ||
                !resolved.TryAdd(resource.BindingKey, tensor))
            {
                throw new InvalidDataException($"Weight resource '{resource.Id}' does not match tensor '{tensor.Name}'.");
            }
        }

        weights = resolved;
    }

    public IModelTensor GetRequired(string role) =>
        weights.TryGetValue(role, out var tensor)
            ? tensor
            : throw new InvalidDataException($"RWKV-6 execution graph is missing named weight role '{role}'.");

    public IModelTensor GetLayer(int index, string suffix) => GetRequired($"blocks.{index}.{suffix}");
}
