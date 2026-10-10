using System.Runtime.InteropServices;
using SharpInference.Architectures.Phi4;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Instructions.Phi4;
using SharpInference.Instructions.Phi4.Cpu;
using SharpInference.Instructions.Phi4.D3D12;
using SharpInference.Runtime;
using SharpInference.Runtime.Cpu;
using SharpInference.Runtime.D3D12;

namespace SharpInference.Tests;

[Collection(VmGraphGpuCollection.Name)]
public sealed class Phi4TextGraphTests
{
    [Theory]
    [InlineData(false, false, Phi4Adapter.None)]
    [InlineData(true, true, Phi4Adapter.None)]
    [InlineData(false, true, Phi4Adapter.Vision)]
    [InlineData(true, false, Phi4Adapter.Vision)]
    [InlineData(false, false, Phi4Adapter.Speech)]
    [InlineData(true, true, Phi4Adapter.Speech)]
    public void GeneratedGraphUsesExplicitOutputResources(bool half, bool embeddings, Phi4Adapter adapter)
    {
        using var file = Package(half, 2);
        var module = new Phi4TextGraphModule(4, embeddings, adapter);
        var graph = module.Build(file);
        Assert.NotEmpty(graph.Structure!.LayerDefinitions);
        Assert.Equal(2, ModelGraphNamingAssertions.Calls(graph.Structure.Root).Count());
        ModelGraphNamingAssertions.Validate(graph, module);
        var model = new Phi4ModelGraphModule(4, adapter);
        var composed = model.Build(file);
        Assert.NotEmpty(composed.Structure!.LayerDefinitions);
        ModelGraphNamingAssertions.Validate(composed, model);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ComposedGraphUsesExplicitOutputResources(bool half)
    {
        var audio = Phi4AudioGraphTests.CreateSyntheticFile(half, 2, 4);
        var vision = Phi4VisionGraphTests.CreateSyntheticFile(half, 2, 4);
        using var file = Package(half, 2, new CombinedFile([audio, vision]));
        var fusion = new Phi4FusionGraphModule(15, [new(0, 1), new(1, 10, "image"), new(11, 3, "audio"), new(14, 1)]);
        ModelGraphNamingAssertions.Validate(fusion.Build(file));
        var module = new Phi4ModelGraphModule(16, Phi4Adapter.Vision, fusion: fusion, encoders: new Dictionary<string, IModelGraphModule>
        {
            ["audio"] = new Phi4AudioGraphModule(Phi4AudioProjector.Vision, frames: 24, layers: 2),
            ["image"] = new Phi4VisionGraphModule(2, 10, heads: 2, layers: 2),
        });
        ModelGraphNamingAssertions.Validate(module.Build(file), module);
    }

    [Theory]
    [InlineData(false, false, false, Phi4Adapter.None, 2)]
    [InlineData(false, true, false, Phi4Adapter.None, 32)]
    [InlineData(false, true, true, Phi4Adapter.Vision, 2)]
    [InlineData(false, false, false, Phi4Adapter.Speech, 2)]
    [InlineData(true, false, false, Phi4Adapter.None, 2)]
    [InlineData(true, true, true, Phi4Adapter.Vision, 2)]
    [InlineData(true, false, false, Phi4Adapter.Speech, 2)]
    public void DecoderExecutesStatefulSyntheticReference(bool gpu, bool half, bool embeddings, Phi4Adapter adapter, int layers)
    {
        using var file = Package(half, layers);
        var values = file.Names.ToDictionary(name => name, name => Values(file.GetRequired(name)));
        var reference = new Reference(values, layers, adapter);
        var module = new Phi4TextGraphModule(4, embeddings, adapter);
        var registry = new ModelGraphModuleRegistry();
        registry.Register(module);
        IInstructionCollectionProvider[] collections = gpu
            ? [.. D3D12InstructionCollections.Create(), new Phi4D3D12TextInstructionCollection()]
            : [.. CpuInstructionCollections.Create(), new Phi4CpuTextInstructionCollection()];
        using var backend = gpu ? D3D12VmBackendFactory.Create(instructionCollections: collections) : CpuVmBackendFactory.Create(instructionCollections: collections);
        using var processor = Processor.Load("text", new Reader(file), registry, backend);
        var graph = processor.LogicalGraph!;
        ModelGraphNamingAssertions.Validate(graph, module);
        Assert.Equal(layers * 2, graph.GraphState.Count);
        Assert.Equal(layers, graph.Nodes.Count(node => node.Operation == Phi4TextGraphOperations.RopeKeyValueWrite));
        GraphValidator.Validate(GraphXml.DeserializeLogical(GraphXml.Serialize(graph)), module);
        using var session = Assert.IsAssignableFrom<ITensorProcessorSession>(((IProcessor)processor).CreateSession());
        using var independent = Assert.IsAssignableFrom<ITensorProcessorSession>(((IProcessor)processor).CreateSession());
        int[] tokens = [1, 3, 2, 0];
        float[]? first = null;
        for (var position = 0; position < tokens.Length; position++)
        {
            var embedding = values["text.token_embd.weight"].AsSpan(tokens[position] * 4, 4).ToArray();
            var expected = reference.Forward(embedding, position);
            var actual = Run(session, tokens[position], position);
            Check(expected, actual);
            if (position == 0) { first = actual; Check(first, Run(independent, tokens[0], 0)); }
            if (position == 1)
            {
                Assert.Throws<ArgumentOutOfRangeException>(() => Run(session, tokens[2], -1));
                Assert.Throws<ArgumentOutOfRangeException>(() => Run(session, tokens[2], 4));
                if (!embeddings) Assert.Throws<ArgumentOutOfRangeException>(() => Run(session, 7, 2));
            }
        }
        session.Reset();
        Check(first!, Run(session, tokens[0], 0));

        float[] Run(ITensorProcessorSession target, int token, int position)
        {
            var inputs = new Dictionary<string, ReadOnlyMemory<byte>> { ["position"] = BitConverter.GetBytes(position) };
            if (embeddings) inputs["embedding"] = MemoryMarshal.AsBytes(values["text.token_embd.weight"].AsSpan(token * 4, 4)).ToArray();
            else inputs["token"] = BitConverter.GetBytes(token);
            return MemoryMarshal.Cast<byte, float>(target.Execute(inputs)["logits"]).ToArray();
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void UnifiedModelConnectsDenseFusionToStatefulDecoder(bool gpu)
    {
        using var file = Package(true, 2);
        var values = file.Names.ToDictionary(name => name, name => Values(file.GetRequired(name)));
        var fusion = new Phi4FusionGraphModule(4, [new(0, 1), new(1, 2, "image"), new(3, 1)]);
        var module = new Phi4ModelGraphModule(4, fusion: fusion);
        var registry = new ModelGraphModuleRegistry();
        registry.Register(module);
        IInstructionCollectionProvider[] collections = gpu
            ? [.. D3D12InstructionCollections.Create(), new Phi4D3D12TextInstructionCollection()]
            : [.. CpuInstructionCollections.Create(), new Phi4CpuTextInstructionCollection()];
        using var backend = gpu ? D3D12VmBackendFactory.Create(instructionCollections: collections) : CpuVmBackendFactory.Create(instructionCollections: collections);
        using var processor = Processor.Load("model", new Reader(file), registry, backend);
        Assert.Equal("phi4", processor.Metadata.ArchitectureId);
        ModelGraphNamingAssertions.Validate(processor.LogicalGraph!, module);
        GraphValidator.Validate(GraphXml.DeserializeLogical(GraphXml.Serialize(processor.LogicalGraph!)), module);
        using var session = Assert.IsAssignableFrom<ITensorProcessorSession>(((IProcessor)processor).CreateSession());
        int[] tokens = [1, -1, -1, 2];
        float[] media = [0.3f, -0.2f, 0.1f, 0.5f, -0.1f, 0.4f, -0.3f, 0.2f];
        var reference = new Reference(values, 2, Phi4Adapter.None);
        for (var row = 0; row < 4; row++)
        {
            var embedding = row is 1 or 2 ? media.AsSpan((row - 1) * 4, 4).ToArray() : values["text.token_embd.weight"].AsSpan(tokens[row] * 4, 4).ToArray();
            var inputs = new Dictionary<string, ReadOnlyMemory<byte>>
            {
                ["token_ids"] = MemoryMarshal.AsBytes(tokens.AsSpan()).ToArray(),
                ["fusion.image"] = MemoryMarshal.AsBytes(media.AsSpan()).ToArray(),
                ["prompt_index"] = BitConverter.GetBytes(row), ["position"] = BitConverter.GetBytes(row),
            };
            Check(reference.Forward(embedding, row), MemoryMarshal.Cast<byte, float>(session.Execute(inputs)["logits"]).ToArray());
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void UnifiedModelConnectsEncodersToFusionAndDecoder(bool gpu, bool half)
    {
        var audio = Phi4AudioGraphTests.CreateSyntheticFile(half, 2, 4);
        var vision = gpu ? Phi4VisionGraphTests.CreateSyntheticFile(half, 2, 4) : null;
        var projector = gpu ? Phi4AudioProjector.Vision : Phi4AudioProjector.Speech;
        var audioModule = new Phi4AudioGraphModule(projector, frames: 24, layers: 2);
        var visionModule = gpu ? new Phi4VisionGraphModule(2, 10, heads: 2, layers: 2) : null;
        float[] features = Enumerable.Range(0, 24 * 8).Select(i => (i % 17 - 7) / 16f).ToArray();
        var audioInputs = new Dictionary<string, ReadOnlyMemory<byte>>
        {
            ["audio_features"] = MemoryMarshal.AsBytes(features.AsSpan()).ToArray(), ["frame_count"] = BitConverter.GetBytes(24),
        };
        var imageInputs = new Dictionary<string, ReadOnlyMemory<byte>>
        {
            ["pixels"] = MemoryMarshal.AsBytes(Enumerable.Range(0, 96).Select(i => (i % 13 - 6) / 16f).ToArray().AsSpan()).ToArray(),
            ["mask"] = MemoryMarshal.AsBytes(Enumerable.Repeat(1f, 32).ToArray().AsSpan()).ToArray(),
            ["mapping"] = MemoryMarshal.AsBytes(new int[] { 4, 5, -1, -2, 0, 1, -1, 2, 3, -1 }.AsSpan()).ToArray(),
        };
        IInstructionCollectionProvider[] collections = gpu
            ? [.. D3D12InstructionCollections.Create(), new Phi4D3D12TextInstructionCollection(), new Phi4D3D12AudioInstructionCollection(), new Phi4D3D12VisionInstructionCollection()]
            : [.. CpuInstructionCollections.Create(), new Phi4CpuTextInstructionCollection(), new Phi4CpuAudioInstructionCollection()];
        var audioEmbeddings = Encode(audioModule, audio, audioInputs);
        var imageEmbeddings = gpu ? Encode(visionModule!, vision!, imageInputs) : [];
        using var package = Package(half, 2, new CombinedFile(vision is null ? [audio] : [audio, vision]));
        var weights = package.Names.ToDictionary(name => name, name => Values(package.GetRequired(name)));
        var count = gpu ? 15 : 5;
        var segments = gpu ? new Phi4FusionSegment[] { new(0, 1), new(1, 10, "image"), new(11, 3, "audio"), new(14, 1) }
            : [new(0, 1), new(1, 3, "audio"), new(4, 1)];
        var encoders = new Dictionary<string, IModelGraphModule> { ["audio"] = audioModule };
        if (gpu) encoders.Add("image", visionModule!);
        var adapter = gpu ? Phi4Adapter.Vision : Phi4Adapter.Speech;
        var module = new Phi4ModelGraphModule(16, adapter, 0.75f, new(count, segments), encoders);
        var registry = new ModelGraphModuleRegistry();
        registry.Register(module);
        using var backend = gpu ? D3D12VmBackendFactory.Create(instructionCollections: collections) : CpuVmBackendFactory.Create(instructionCollections: collections);
        using var processor = Processor.Load("multimodal", new Reader(package), registry, backend);
        Assert.Equal("phi4", processor.LogicalGraph!.Identity.ArchitectureId);
        ModelGraphNamingAssertions.Validate(processor.LogicalGraph!, module);
        Assert.Contains(processor.LogicalGraph.Nodes, node => node.Operation == Phi4AudioGraphOperations.RelativeAttention);
        if (gpu) Assert.Contains(processor.LogicalGraph.Nodes, node => node.Operation == Phi4VisionGraphOperations.Attention);
        Assert.Equal(1, processor.LogicalGraph.Resources.Count(resource => resource.BindingKey == "text.token_embd.weight"));
        using var session = Assert.IsAssignableFrom<ITensorProcessorSession>(((IProcessor)processor).CreateSession());
        var tokens = Enumerable.Repeat(-1, count).ToArray(); tokens[0] = 1; tokens[^1] = 2;
        var inputs = audioInputs.ToDictionary(pair => "audio." + pair.Key, pair => pair.Value);
        if (gpu) foreach (var pair in imageInputs) inputs.Add("image." + pair.Key, pair.Value);
        inputs.Add("token_ids", MemoryMarshal.AsBytes(tokens.AsSpan()).ToArray());
        var reference = new Reference(weights, 2, adapter);
        for (var row = 0; row < count; row++)
        {
            var embedding = row == 0 || row == count - 1 ? weights["text.token_embd.weight"].AsSpan(tokens[row] * 4, 4).ToArray()
                : gpu && row <= 10 ? imageEmbeddings.AsSpan((row - 1) * 4, 4).ToArray()
                : audioEmbeddings.AsSpan((row - (gpu ? 11 : 1)) * 4, 4).ToArray();
            inputs["prompt_index"] = BitConverter.GetBytes(row); inputs["position"] = BitConverter.GetBytes(row);
            Check(reference.Forward(embedding, row), MemoryMarshal.Cast<byte, float>(session.Execute(inputs)["logits"]).ToArray());
        }

        float[] Encode(IModelGraphModule component, IModelFile file, Dictionary<string, ReadOnlyMemory<byte>> inputs)
        {
            using var backend = gpu ? D3D12VmBackendFactory.Create(instructionCollections: collections) : CpuVmBackendFactory.Create(instructionCollections: collections);
            var graph = component.Build(file);
            backend.Prepare(graph, modelModule: component);
            using var session = backend.CreateGraphSession(file, new GraphTensorState(graph));
            return MemoryMarshal.Cast<byte, float>(session.Execute(inputs.ToDictionary(pair => new ResourceId(pair.Key), pair => pair.Value))[new("output")]).ToArray();
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FusedPromptStateContinuesThroughTokenDecoderWithoutReencoding(bool gpu)
    {
        using var package = Package(true, 2);
        var weights = package.Names.ToDictionary(name => name, name => Values(package.GetRequired(name)));
        var fusedModule = new Phi4ModelGraphModule(4, fusion: new(2, [new(0, 2, "media")]));
        var decodeModule = new Phi4ModelGraphModule(4);
        var fusedGraph = fusedModule.Build(package);
        var decodeGraph = decodeModule.Build(package);
        var state = new GraphTensorState(fusedGraph);
        IInstructionCollectionProvider[] collections = gpu
            ? [.. D3D12InstructionCollections.Create(), new Phi4D3D12TextInstructionCollection()]
            : [.. CpuInstructionCollections.Create(), new Phi4CpuTextInstructionCollection()];
        using var encoderBackend = gpu ? D3D12VmBackendFactory.Create(instructionCollections: collections) : CpuVmBackendFactory.Create(instructionCollections: collections);
        using var decoderBackend = gpu ? D3D12VmBackendFactory.Create(instructionCollections: collections) : CpuVmBackendFactory.Create(instructionCollections: collections);
        encoderBackend.Prepare(fusedGraph, modelModule: fusedModule);
        decoderBackend.Prepare(decodeGraph, modelModule: decodeModule);
        using var prompt = encoderBackend.CreateGraphSession(package, state);
        using var decoder = decoderBackend.CreateGraphSession(package, state);
        float[] media = [0.3f, -0.2f, 0.1f, 0.5f, -0.1f, 0.4f, -0.3f, 0.2f];
        var reference = new Reference(weights, 2, Phi4Adapter.None);
        for (var row = 0; row < 2; row++)
        {
            var inputs = new Dictionary<ResourceId, ReadOnlyMemory<byte>>
            {
                [new("fusion.media")] = MemoryMarshal.AsBytes(media.AsSpan()).ToArray(),
                [new("prompt_index")] = BitConverter.GetBytes(row), [new("position")] = BitConverter.GetBytes(row),
            };
            Check(reference.Forward(media.AsSpan(row * 4, 4).ToArray(), row), MemoryMarshal.Cast<byte, float>(prompt.Execute(inputs)[new("logits")]).ToArray());
        }
        var continuation = new Dictionary<ResourceId, ReadOnlyMemory<byte>> { [new("token")] = BitConverter.GetBytes(2), [new("position")] = BitConverter.GetBytes(2) };
        Check(reference.Forward(weights["text.token_embd.weight"].AsSpan(8, 4).ToArray(), 2), MemoryMarshal.Cast<byte, float>(decoder.Execute(continuation)[new("logits")]).ToArray());
    }

    [Fact]
    public void FusionRejectsIncompletePlansAndConvertsOnlyExplicitContiguousLegacyAbi()
    {
        Assert.Throws<ArgumentException>(() => new Phi4FusionGraphModule(4, [new(0, 1), new(2, 2)]));
        Assert.Throws<ArgumentException>(() => new Phi4FusionGraphModule(4, [new(0, 3), new(2, 2)]));
        Assert.Throws<ArgumentException>(() => new Phi4FusionGraphModule(4, [new(0, 2, "image"), new(2, 2, "image")]));
        var dense = Phi4FusionGraphModule.ToDenseEmbeddingPort(new(GraphElementType.Float32, [4, 4], "token-major"));
        Assert.Equal("dense", dense.Layout);
        Assert.Equal([4, 4], dense.Dimensions);
        Assert.Throws<ArgumentException>(() => Phi4FusionGraphModule.ToDenseEmbeddingPort(new(GraphElementType.Float32, [4, 4], "strided")));
        Assert.Throws<ArgumentException>(() => Phi4FusionGraphModule.ToDenseEmbeddingPort(new(GraphElementType.Float16, [4, 4], "token-major")));
    }

    [Fact]
    public void MissingNativeRegistrationRejectsPreparationAndMalformedContractsAreRejected()
    {
        using var file = Package(false, 2);
        var module = new Phi4TextGraphModule(4);
        var graph = module.Build(file);
        using var cpu = CpuVmBackendFactory.Create();
        Assert.Throws<BackendPreparationException>(() => cpu.Prepare(graph));
        var node = graph.Nodes.First(node => node.Operation == Phi4TextGraphOperations.RopeKeyValueWrite);
        var attributes = node.Attributes.ToDictionary(pair => pair.Key, pair => pair.Value);
        attributes["rotary_size"] = "3";
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        Assert.Throws<InvalidDataException>(() => module.Validate(new(node.Id.Value, node.Operation, node.Resources, resources, attributes, node.Requirements)));
        Assert.Throws<NotSupportedException>(() => module.Validate(new(node.Id.Value, node.Operation with { Version = 99 }, node.Resources, resources, node.Attributes, node.Requirements)));
        var downgraded = node.Resources.Select(binding => binding.Port == "key_cache" ? binding with { Access = GraphResourceAccess.Write } : binding).ToArray();
        Assert.Throws<InvalidDataException>(() => module.Validate(new(node.Id.Value, node.Operation, downgraded, resources, node.Attributes, node.Requirements)));
        Assert.Throws<ArgumentException>(() => new Phi4TextGraphModule(adapter: Phi4Adapter.Vision, adapterScale: float.NaN));
    }

    private static void Check(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < actual.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]));
            Assert.InRange(MathF.Abs(expected[i] - actual[i]), 0, 1e-4f + 1e-4f * MathF.Abs(expected[i]));
        }
        Assert.Contains(actual, value => value != 0);
    }
    private static float[] Values(IModelTensor tensor) => tensor.DataType == TensorDataType.Float32
        ? tensor.FloatValues.ToArray() : tensor.HalfValues.ToArray().Select(value => (float)value).ToArray();

    private static Phi4PackageTensorCatalog Package(bool half, int layers, IModelFile? omni = null)
    {
        var text = new File();
        var vision = new File();
        var speech = new File();
        text.Add("token_embd.weight", [7, 4], half);
        text.Add("output_norm.weight", [4], half, norm: true);
        for (var layer = 0; layer < layers; layer++)
        {
            var prefix = $"blk.{layer}.";
            text.Add(prefix + "attn_norm.weight", [4], half, norm: true);
            text.Add(prefix + "ffn_norm.weight", [4], half, norm: true);
            foreach (var (name, output, input) in new[] { ("attn_qkv", 8, 4), ("attn_output", 4, 4), ("ffn_up", 12, 4), ("ffn_down", 4, 6) })
            {
                text.Add(prefix + name + ".weight", [output, input], half);
                vision.Add(prefix + name + ".weight.lora_a", [2, input], half);
                vision.Add(prefix + name + ".weight.lora_b", [output, 2], half);
                speech.Add(prefix + name + ".weight.lora_a", [2, input], half);
                speech.Add(prefix + name + ".weight.lora_b", [output, 2], half);
            }
        }
        return new("synthetic", new(layers, 2, 1, 2, 10000, 1e-5f, 1.1f), text, omni ?? new File(), vision, speech,
            visionAdapterAlpha: 1.5f, speechAdapterAlpha: 1.5f);
    }

    private sealed class CombinedFile(IModelFile[] sources) : IModelFile
    {
        public string Path => "synthetic-omni";
        public IReadOnlyCollection<string> Names => sources.SelectMany(source => source.Names).ToArray();
        public IModelTensor GetRequired(string name) => sources.Single(source => source.Names.Contains(name)).GetRequired(name);
        public bool TryGet(string name, out IModelTensor tensor)
        {
            foreach (var source in sources) if (source.TryGet(name, out tensor)) return true;
            tensor = null!; return false;
        }
        public void Dispose() { foreach (var source in sources) source.Dispose(); }
    }
    private sealed class Reader(IModelFile file) : IModelReader { public IModelFile Open(string path) => file; }
    private sealed class File : IModelFile
    {
        private readonly Dictionary<string, Tensor> tensors = [];
        public string Path => "synthetic";
        public IReadOnlyCollection<string> Names => tensors.Keys;
        public bool TryGet(string name, out IModelTensor tensor) { var found = tensors.TryGetValue(name, out var value); tensor = value!; return found; }
        public IModelTensor GetRequired(string name) => tensors[name];
        public void Dispose() { }
        public void Add(string name, int[] shape, bool half, bool norm = false)
        {
            var count = shape.Aggregate(1, (total, size) => total * size);
            var data = Enumerable.Range(0, count).Select(i => norm ? 0.9f + (i % 3) * 0.05f : ((i * 7 + name.Length) % 19 - 9) / 32f).ToArray();
            tensors.Add(name, new(name, shape, half, data));
        }
    }
    private sealed class Tensor(string name, int[] shape, bool half, float[] data) : IModelTensor
    {
        private readonly Half[] halves = data.Select(value => (Half)value).ToArray();
        public string Name => name;
        public TensorDataType DataType => half ? TensorDataType.Float16 : TensorDataType.Float32;
        public IReadOnlyList<int> Dimensions => shape;
        public ReadOnlySpan<float> FloatValues => !half ? data : throw new InvalidOperationException();
        public ReadOnlySpan<Half> HalfValues => half ? halves : throw new InvalidOperationException();
    }

    private sealed class Reference(Dictionary<string, float[]> weights, int layers, Phi4Adapter adapter)
    {
        private readonly List<double[]>[] keys = Enumerable.Range(0, layers).Select(_ => new List<double[]>()).ToArray();
        private readonly List<double[]>[] values = Enumerable.Range(0, layers).Select(_ => new List<double[]>()).ToArray();
        public float[] Forward(float[] embedding, int position)
        {
            var hidden = embedding.Select(value => (double)value).ToArray();
            for (var layer = 0; layer < layers; layer++)
            {
                var prefix = $"text.blk.{layer}.";
                var qkv = Linear(Norm(hidden, prefix + "attn_norm.weight"), prefix + "attn_qkv.weight", 8);
                for (var offset = 0; offset < 6; offset += 2)
                {
                    var left = qkv[offset]; var right = qkv[offset + 1];
                    qkv[offset] = 1.1 * (left * Math.Cos(position) - right * Math.Sin(position));
                    qkv[offset + 1] = 1.1 * (right * Math.Cos(position) + left * Math.Sin(position));
                }
                keys[layer].Add(qkv[4..6]);
                values[layer].Add(qkv[6..8]);
                var attention = new double[4];
                for (var head = 0; head < 2; head++)
                {
                    var scores = keys[layer].Select(key => (qkv[head * 2] * key[0] + qkv[head * 2 + 1] * key[1]) / Math.Sqrt(2)).ToArray();
                    var maximum = scores.Max();
                    var probabilities = scores.Select(score => Math.Exp(score - maximum)).ToArray();
                    var total = probabilities.Sum();
                    for (var token = 0; token < probabilities.Length; token++)
                    for (var d = 0; d < 2; d++) attention[head * 2 + d] += probabilities[token] / total * values[layer][token][d];
                }
                hidden = hidden.Zip(Linear(attention, prefix + "attn_output.weight", 4), (a, b) => a + b).ToArray();
                var up = Linear(Norm(hidden, prefix + "ffn_norm.weight"), prefix + "ffn_up.weight", 12);
                var activated = Enumerable.Range(0, 6).Select(i => up[i] / (1 + Math.Exp(-up[i])) * up[6 + i]).ToArray();
                hidden = hidden.Zip(Linear(activated, prefix + "ffn_down.weight", 4), (a, b) => a + b).ToArray();
            }
            return Multiply(Norm(hidden, "text.output_norm.weight"), weights["text.token_embd.weight"], 7).Select(value => (float)value).ToArray();
        }
        private double[] Norm(double[] input, string name)
        {
            var scale = 1 / Math.Sqrt(input.Sum(value => value * value) / input.Length + 1e-5);
            return input.Select((value, i) => value * scale * weights[name][i]).ToArray();
        }
        private double[] Linear(double[] input, string name, int output)
        {
            var result = Multiply(input, weights[name], output);
            if (adapter == Phi4Adapter.None) return result;
            var prefix = (adapter == Phi4Adapter.Vision ? "vision-adapter." : "speech-adapter.") + name[5..];
            var low = Multiply(input, weights[prefix + ".lora_a"], 2);
            var update = Multiply(low, weights[prefix + ".lora_b"], output);
            return result.Zip(update, (a, b) => a + 0.75 * b).ToArray();
        }
        private static double[] Multiply(double[] input, float[] matrix, int rows) => Enumerable.Range(0, rows)
            .Select(row => input.Select((value, column) => value * matrix[row * input.Length + column]).Sum()).ToArray();
    }
}
