using SharpInference.Graphs;

namespace SharpInference.Architectures.Rwkv7;

internal sealed class Rwkv7WeightBindings
{
    private readonly IReadOnlyDictionary<string, IModelTensor> weights;

    public Rwkv7WeightBindings(ExecutionGraph graph, IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(tensors);
        if (graph.Identity.ArchitectureId != "rwkv-7" ||
            graph.Model.VocabularySize != tensors.VocabularySize ||
            graph.Model.EmbeddingSize != tensors.EmbeddingSize ||
            graph.Model.LayerCount != tensors.LayerCount)
        {
            throw new InvalidDataException("The RWKV-7 execution graph does not match the model catalog.");
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
                RwkvTensorDataType.Float16 => GraphElementType.Float16,
                RwkvTensorDataType.Float32 => GraphElementType.Float32,
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
            : throw new InvalidDataException($"RWKV-7 execution graph is missing named weight role '{role}'.");

    public IModelTensor GetLayer(int index, string suffix) => GetRequired($"blocks.{index}.{suffix}");
}
