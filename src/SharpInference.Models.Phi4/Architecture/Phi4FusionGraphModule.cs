using SharpInference.Graphs;

namespace SharpInference.Architectures.Phi4;

public sealed record Phi4FusionSegment(int Start, int TokenCount, string? EmbeddingPort = null);

public sealed class Phi4FusionGraphModule : IModelGraphModule
{
    private readonly int tokenCount;
    private readonly Phi4FusionSegment[] segments;

    public Phi4FusionGraphModule(int tokenCount, IEnumerable<Phi4FusionSegment> segments)
    {
        if (tokenCount <= 0) throw new ArgumentOutOfRangeException(nameof(tokenCount));
        ArgumentNullException.ThrowIfNull(segments);
        this.tokenCount = tokenCount;
        this.segments = segments.OrderBy(segment => segment.Start).ToArray();
        var end = 0;
        var ports = new HashSet<string>(StringComparer.Ordinal);
        foreach (var segment in this.segments)
        {
            if (segment.Start != end || segment.TokenCount <= 0 || segment.TokenCount > tokenCount - end)
                throw new ArgumentException("Fusion segments must cover the prompt exactly without gaps or overlaps.", nameof(segments));
            if (segment.EmbeddingPort is { } port && (string.IsNullOrWhiteSpace(port) || port is "token_ids" or "output" || !ports.Add(port)))
                throw new ArgumentException("Media ports must have unique non-reserved names.", nameof(segments));
            end = checked(end + segment.TokenCount);
        }
        if (end != tokenCount) throw new ArgumentException("Fusion segments do not cover the prompt.", nameof(segments));
    }

    public string ArchitectureId => "phi4.fusion";
    public bool CanLoad(IModelTensorCatalog tensors) => tensors is Phi4PackageTensorCatalog;
    public ModelMetadata ReadMetadata(IModelTensorCatalog tensors)
    {
        var embedding = tensors.GetRequired("text.token_embd.weight");
        if (embedding.Dimensions.Count != 2 || embedding.DataType is not (TensorDataType.Float16 or TensorDataType.Float32))
            throw new InvalidDataException("Phi4 fusion requires FP16/FP32 vocabulary-by-width embeddings.");
        return new(ArchitectureId, new Dictionary<string, long>
        {
            ["tokens"] = tokenCount,
            ["embedding_width"] = embedding.Dimensions[1],
            ["vocabulary"] = embedding.Dimensions[0],
        });
    }

    public LogicalGraph Build(IModelTensorCatalog tensors)
    {
        var dimensions = ReadMetadata(tensors).Dimensions.ToDictionary(pair => pair.Key, pair => checked((int)pair.Value));
        var width = dimensions["embedding_width"];
        var embedding = tensors.GetRequired("text.token_embd.weight");
        var builder = new LogicalGraphBuilder(new(ArchitectureId, 2, "forward"), new GraphModelSignature(ArchitectureId, "phi4.fusion.empty@1", dimensions))
            .AddRegion("root", GraphRegionTypes.Graph, "Dense token-major embedding fusion", architecture: new(
                Description: "Assembles text and media embeddings into explicit dense prompt row ranges.",
                DefaultView: GraphArchitectureView.Architecture))
            .AddResource("output", "Fused prompt", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [tokenCount, width]), graphOutput: true);
        if (segments.Any(segment => segment.EmbeddingPort is null))
        {
            builder.AddResource("token_ids", "Expanded prompt token ids", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [tokenCount]), graphInput: true);
            builder.AddResource("token_embedding", "Token embeddings", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(embedding.DataType == TensorDataType.Float16 ? GraphElementType.Float16 : GraphElementType.Float32, embedding.Dimensions), embedding.Name);
        }
        var index = 0;
        builder.AddRegion("fusion", GraphRegionTypes.Stage, "Fusion", "root",
            architecture: new(Description: "Writes each prompt segment to its disjoint tensor view."));
        foreach (var segment in segments)
        {
            builder.WithRegion($"segment.{segment.Start}", GraphRegionTypes.Architecture,
                segment.EmbeddingPort is { } media ? media + " Embeddings" : "Text Embeddings", _ =>
            {
                if (segment.EmbeddingPort is { } port)
                {
                    var shape = new TensorDescriptor(GraphElementType.Float32, [segment.TokenCount, width]);
                    builder.AddResource(port, port, GraphResourceKind.Input, GraphResourceLifetime.External, shape, graphInput: true);
                    Emit(PrimitiveGraphOperations.Copy,
                        [GraphBindings.Read("input", port), new("output", new("output"), GraphResourceAccess.Write,
                        View: new(checked((ulong)segment.Start * (ulong)width * sizeof(float)), shape))]);
                }
                else
                {
                    for (var row = segment.Start; row < segment.Start + segment.TokenCount; row++)
                        Emit(PrimitiveGraphOperations.GatherRow,
                            [GraphBindings.Read("table", "token_embedding"), new("index", new("token_ids"), GraphResourceAccess.Read,
                            View: new(checked((ulong)row * sizeof(int)), new TensorDescriptor(GraphElementType.Int32, [1]))),
                         new("output", new("output"), GraphResourceAccess.Write,
                            View: new(checked((ulong)row * (ulong)width * sizeof(float)), new TensorDescriptor(GraphElementType.Float32, [width])))]);
                }
            }, parentId: "fusion",
                architecture: new(Description: $"Writes prompt rows [{segment.Start}, {segment.Start + segment.TokenCount}) using the actual tensor-view bindings."));
        }
        return builder.BuildSequential();

        void Emit(GraphOperationId operation, NodeResourceBinding[] bindings)
        {
            var id = "fusion.node." + index++;
            builder.AddNode(id, operation, bindings, null);
        }
    }

    public static TensorDescriptor ToDenseEmbeddingPort(TensorDescriptor legacy)
    {
        ArgumentNullException.ThrowIfNull(legacy);
        if (legacy.ElementType != GraphElementType.Float32 || legacy.Dimensions.Count != 2 || legacy.Layout is not ("dense" or "token-major"))
            throw new ArgumentException("Phi4 embedding ports require contiguous FP32 token-by-width storage.", nameof(legacy));
        return new(GraphElementType.Float32, legacy.Dimensions);
    }
}
