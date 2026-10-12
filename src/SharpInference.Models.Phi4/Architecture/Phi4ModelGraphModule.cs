using SharpInference.Graphs;

namespace SharpInference.Architectures.Phi4;

/// <summary>Composes optional media encoders and embedding fusion with the stateful Phi-4 text decoder.</summary>
public sealed class Phi4ModelGraphModule : IPhi4GraphModule
{
    private readonly Phi4TextGraphModule text;
    private readonly Phi4FusionGraphModule? fusion;
    private readonly IReadOnlyDictionary<string, IPhi4GraphModule> encoders;

    /// <summary>Initializes decoder, fusion, and optional media encoder definition composition.</summary>
    public Phi4ModelGraphModule(
        int maximumContext = 4096,
        Phi4Adapter adapter = Phi4Adapter.None,
        float? adapterScale = null,
        Phi4FusionGraphModule? fusion = null,
        IReadOnlyDictionary<string, IPhi4GraphModule>? encoders = null)
    {
        if (fusion is null && encoders is { Count: > 0 })
            throw new ArgumentException("Encoder connections require a fusion graph.", nameof(encoders));
        this.encoders = (encoders ?? new Dictionary<string, IPhi4GraphModule>())
            .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
        if (this.encoders.Any(pair =>
                string.IsNullOrWhiteSpace(pair.Key) ||
                pair.Value is not (Phi4AudioGraphModule or Phi4VisionGraphModule)))
            throw new ArgumentException(
                "Phi4 encoder connections must explicitly name audio or vision modules.",
                nameof(encoders));
        this.fusion = fusion;
        text = new(maximumContext, embeddingInput: fusion is not null, adapter, adapterScale);
    }

    /// <inheritdoc/>
    public string ArchitectureId => "phi4";

    /// <summary>
    /// The legacy runtime operation validator. The definition IR is not currently consumable by
    /// the Runtime/D3D12 pipeline.
    /// </summary>

    /// <inheritdoc/>
    public bool CanLoad(IModelTensorCatalog tensors) =>
        text.CanLoad(tensors) && encoders.Values.All(module => module.CanLoad(tensors));

    /// <inheritdoc/>
    public ModelMetadata ReadMetadata(IModelTensorCatalog tensors)
    {
        var metadata = text.ReadMetadata(tensors);
        foreach (var module in encoders.Values) _ = module.ReadMetadata(tensors);
        if (fusion is not null) _ = fusion.ReadMetadata(tensors);
        return new(ArchitectureId, metadata.Dimensions, metadata.Attributes);
    }

    /// <inheritdoc/>
    public LogicalGraphDefinition Build(IModelTensorCatalog tensors)
    {
        var metadata = ReadMetadata(tensors);
        var dimensions = metadata.Dimensions.ToDictionary(
            pair => pair.Key, pair => checked((int)pair.Value), StringComparer.Ordinal);
        var decoder = text.Build(tensors);
        var builder = new Phi4GraphDefinitionBuilder(
                new(ArchitectureId, 2, "forward"),
                new GraphModelSignature(ArchitectureId, "phi4.definition@1", dimensions))
            .Root("root", GraphSectionTypes.Graph, "Phi4 model-owned component composition",
                architecture: new(
                    Description: "Model-owned audio/vision encoders, dense embedding fusion and a stateful text decoder.",
                    DefaultView: GraphArchitectureView.Architecture,
                    Step: GraphArchitectureStep.Token));

        if (fusion is not null)
        {
            var fused = fusion.Build(tensors);
            var aliases = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["output"] = "fusion.output",
                ["token_ids"] = "token_ids",
            };
            foreach (var (port, module) in encoders)
            {
                var input = fused.Resources.SingleOrDefault(resource =>
                    resource.Id.Value == port && resource.Kind == GraphDefinitionResourceKind.Input)
                    ?? throw new InvalidDataException($"Fusion has no encoder port '{port}'.");
                var component = module.Build(tensors);
                var output = component.Resources.Single(resource =>
                    resource.Id.Value == "output" &&
                    resource.Kind == GraphDefinitionResourceKind.Output);
                if (!Same(output.Tensor, input.Tensor))
                    throw new InvalidDataException(
                        $"Encoder '{port}' does not match the dense fusion ABI.");
                var destination = $"{port}.output";
                builder.AppendDefinition(component, port,
                    new Dictionary<string, string> { ["output"] = destination });
                aliases.Add(port, destination);
            }
            builder.AppendDefinition(fused, "fusion", aliases);
            builder
                .AddResource(
                    "select_prompt_embedding_output",
                    "Selected fused token",
                    GraphResourceKind.Temporary,
                    GraphResourceLifetime.Invocation,
                    new TensorDescriptor(
                        GraphElementType.Float32,
                        [dimensions["embedding_width"]]))
                .AddResource(
                    "prompt_index",
                    "Prompt row",
                    GraphResourceKind.Input,
                    GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Int32, [1]),
                    graphInput: true);
            builder.Section(
                "prompt-selection",
                GraphSectionTypes.Architecture,
                "Prompt Row Selection",
                scope => scope.AddNode(
                    "select_prompt_embedding",
                    PrimitiveGraphOperations.GatherRow,
                    [
                        GraphBindings.Read("table", aliases["output"]),
                        GraphBindings.Read("index", "prompt_index"),
                        GraphBindings.Write(
                            "output",
                            "select_prompt_embedding_output"),
                    ]),
                architecture: new(
                    Description: "Selects the fused prompt embedding for the current decoder invocation."));
        }

        builder.AppendDefinition(
            decoder,
            "decoder",
            new Dictionary<string, string>
            {
                ["embedding"] = "select_prompt_embedding_output",
                ["position"] = "position",
                ["token"] = "token",
                ["logits"] = "logits",
            },
            exposeOutputs: true);
        return builder.Build();
    }

    private static bool Same(TensorDescriptor left, TensorDescriptor right) =>
        left.ElementType == right.ElementType &&
        left.Layout == right.Layout &&
        left.Dimensions.SequenceEqual(right.Dimensions);
}
