using SharpInference.Graphs;

namespace SharpInference.Architectures.Phi4;

public sealed class Phi4ModelGraphModule : IModelGraphModule, IGraphOperationValidator
{
    private readonly Phi4TextGraphModule text;
    private readonly Phi4FusionGraphModule? fusion;
    private readonly IReadOnlyDictionary<string, IModelGraphModule> encoders;

    public Phi4ModelGraphModule(int maximumContext = 4096, Phi4Adapter adapter = Phi4Adapter.None,
        float? adapterScale = null, Phi4FusionGraphModule? fusion = null,
        IReadOnlyDictionary<string, IModelGraphModule>? encoders = null)
    {
        if (fusion is null && encoders is { Count: > 0 })
            throw new ArgumentException("Encoder connections require a fusion graph.", nameof(encoders));
        this.encoders = (encoders ?? new Dictionary<string, IModelGraphModule>()).ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
        if (this.encoders.Any(pair => string.IsNullOrWhiteSpace(pair.Key) || pair.Value is not (Phi4AudioGraphModule or Phi4VisionGraphModule)))
            throw new ArgumentException("Phi4 encoder connections must explicitly name audio or vision modules.", nameof(encoders));
        this.fusion = fusion;
        text = new(maximumContext, embeddingInput: fusion is not null, adapter, adapterScale);
    }

    public string ArchitectureId => "phi4";
    public IGraphOperationValidator OperationValidator => this;
    public bool CanLoad(IModelTensorCatalog tensors) => text.CanLoad(tensors) && encoders.Values.All(module => module.CanLoad(tensors));
    public ModelMetadata ReadMetadata(IModelTensorCatalog tensors)
    {
        var metadata = text.ReadMetadata(tensors);
        foreach (var module in encoders.Values) _ = module.ReadMetadata(tensors);
        if (fusion is not null) _ = fusion.ReadMetadata(tensors);
        return new(ArchitectureId, metadata.Dimensions, metadata.Attributes);
    }

    public LogicalGraph Build(IModelTensorCatalog tensors)
    {
        var decoder = text.Build(tensors);
        var builder = new LogicalGraphBuilder(new(ArchitectureId, 2, "forward"),
                new GraphModelSignature(ArchitectureId, decoder.Model.StateAbiId, decoder.Model.Dimensions, decoder.Model.Attributes))
            .AddRegion("root", GraphRegionTypes.Graph, "Phi4 model-owned component composition", architecture: new(
                Description: "Model-owned audio/vision encoders, dense embedding fusion and a stateful text decoder.",
                DefaultView: GraphArchitectureView.Architecture, Step: GraphArchitectureStep.Token))
            .SetStateSchema(decoder.GraphState.Schema);
        var resources = new Dictionary<string, TensorDescriptor>(StringComparer.Ordinal);
        if (fusion is not null)
        {
            var fused = fusion.Build(tensors);
            var aliases = new Dictionary<string, string>(StringComparer.Ordinal) { ["output"] = "fusion." + fused.Nodes[^1].Id.Value + "_output", ["token_ids"] = "token_ids" };
            foreach (var (port, module) in encoders)
            {
                if (!fused.Inputs.Contains(new ResourceId(port))) throw new InvalidDataException($"Fusion has no encoder port '{port}'.");
                var component = module.Build(tensors);
                var output = component.Resources.Single(resource => resource.Id.Value == "output").Tensor;
                var target = fused.Resources.Single(resource => resource.Id.Value == port).Tensor;
                if (!Same(output, target)) throw new InvalidDataException($"Encoder '{port}' does not match the dense fusion ABI.");
                var destination = port + "." + component.Nodes[^1].Id.Value + "_output";
                Append(component, port, new Dictionary<string, string> { ["output"] = destination });
                aliases.Add(port, destination);
            }
            Append(fused, "fusion", aliases);
            var width = decoder.Model.Dimensions["embedding_width"];
            var shape = new TensorDescriptor(GraphElementType.Float32, [width]);
            builder.AddResource("select_prompt_embedding_output", "Selected fused token", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation, shape)
                .AddResource("prompt_index", "Prompt row", GraphResourceKind.Input, GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Int32, [1]), graphInput: true);
            resources.Add("select_prompt_embedding_output", shape);
            builder.AddRegion("prompt-selection", GraphRegionTypes.Architecture, "Prompt Row Selection", "root",
                architecture: new(Description: "Selects the fused prompt embedding for the current decoder invocation."));
            builder.AddNode("select_prompt_embedding", PrimitiveGraphOperations.GatherRow, "prompt-selection",
                [GraphBindings.Read("table", aliases["output"]), GraphBindings.Read("index", "prompt_index"), GraphBindings.Write("output", "select_prompt_embedding_output")],
                null);
        }
        Append(decoder, "decoder", new Dictionary<string, string>
        {
            ["embedding"] = "select_prompt_embedding_output", ["position"] = "position", ["token"] = "token", ["logits"] = "logits",
        }, exposeOutputs: true);
        var graph = GraphLayerReuse.Extract(builder.BuildSequential(), region => region.Type != GraphRegionTypes.Layer ? null :
            region.Id.Value.StartsWith("decoder.", StringComparison.Ordinal) ? "Phi4.Text.Layer" :
            region.Id.Value.StartsWith("audio.", StringComparison.Ordinal) ? "Phi4.Audio.Layer" : "Phi4.Vision.Layer");
        GraphValidator.Validate(graph, this);
        return graph;

        void Append(LogicalGraph component, string tag, IReadOnlyDictionary<string, string> aliases, bool exposeOutputs = false)
        {
            string Map(ResourceId id)
            {
                if (aliases.TryGetValue(id.Value, out var alias)) return alias;
                var resource = component.Resources.Single(resource => resource.Id == id);
                return resource.Kind == GraphResourceKind.Weight && resource.BindingKey is { } key ? "weight." + key : tag + "." + id.Value;
            }
            foreach (var resource in component.Resources)
            {
                var id = Map(resource.Id);
                if (resources.TryGetValue(id, out var existing))
                {
                    if (!Same(existing, resource.Tensor)) throw new InvalidDataException($"Phi4 component '{tag}' has an incompatible port '{resource.Id}'.");
                    continue;
                }
                var output = component.Outputs.Contains(resource.Id);
                var internalOutput = output && !exposeOutputs;
                builder.AddResource(id, resource.Name, internalOutput ? GraphResourceKind.Temporary : resource.Kind,
                    internalOutput ? GraphResourceLifetime.Invocation : resource.Lifetime, resource.Tensor, resource.BindingKey,
                    component.Inputs.Contains(resource.Id), output && exposeOutputs, resource.DeclaredScope);
                resources.Add(id, resource.Tensor);
            }
            foreach (var region in component.Regions)
                builder.AddRegion(tag + "." + region.Id.Value, region.Type == GraphRegionTypes.Graph ? GraphRegionTypes.Stage : region.Type,
                    region.Name, region.ParentId is { } parent ? tag + "." + parent.Value : "root", region.Role, region.Attributes,
                    region.Architecture?.MapResources(id => new ResourceId(Map(id))));
            foreach (var node in component.Nodes)
            {
                var id = tag + "." + node.Id.Value;
                builder.AddNode(id, node.Operation, tag + "." + node.Region.Value, node.Resources.Select(binding => binding with { Resource = new(Map(binding.Resource)) }),
                    null, node.Attributes, node.Requirements);
            }
            foreach (var state in component.GraphState) builder.AddStateSlot(tag + "." + state.Name, Map(state.Resource));
        }
    }

    private static bool Same(TensorDescriptor left, TensorDescriptor right) => left.ElementType == right.ElementType &&
        left.Layout == right.Layout && left.Dimensions.SequenceEqual(right.Dimensions);
    public void Validate(GraphOperationValidationContext context)
    {
        Phi4TextOperationValidation.Validate(context);
        Phi4VisionOperationValidation.Validate(context);
        Phi4AudioOperationValidation.Validate(context);
    }
}
