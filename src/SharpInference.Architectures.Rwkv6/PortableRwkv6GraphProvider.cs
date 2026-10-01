using SharpInference.Graphs;
using TensorOps = SharpInference.Graphs.PortableTensorOperationContracts;

namespace SharpInference.Architectures.Rwkv6;

/// <summary>
/// Unfused RWKV-6 single-token graph. Tensor operations require a backend
/// implementing the portable tensor contracts.
/// </summary>
public sealed class PortableRwkv6GraphProvider : ILogicalGraphProvider
{
    public string ArchitectureId => "rwkv-6";

    public LogicalGraph Build(IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        if (!tensors.TryGet("blocks.0.att.time_maa_x", out _) ||
            !tensors.TryGet("blocks.0.att.time_faaaa", out var first) ||
            tensors.TryGet("blocks.0.att.k_k", out _))
            throw new InvalidDataException("The tensor catalog does not satisfy the RWKV-6 x060 contract.");

        var width = tensors.EmbeddingSize;
        if (first.Dimensions.Count != 3 || first.Dimensions[0] != 1 ||
            first.Dimensions[1] <= 0 || first.Dimensions[2] <= 0 ||
            checked(first.Dimensions[1] * first.Dimensions[2]) != width)
            throw new InvalidDataException("RWKV-6 time_faaaa has incompatible head dimensions.");
        var size = first.Dimensions[1];
        var heads = first.Dimensions[2];
        var builder = new LogicalGraphBuilder(
            new GraphIdentity("rwkv-6", 1, "rwkv6.forward-token.portable"),
            new GraphModelSignature(tensors.VocabularySize, width, tensors.LayerCount,
                heads, size, "rwkv-6.state.fp32@1"))
            .SetStateSchema(new StateSchema("RWKV6_State"));
        builder.AddRegion("graph", GraphRegionTypes.Graph, "Portable forward token");
        builder.AddResource("token", "Token", GraphResourceKind.Input, GraphResourceLifetime.External,
            new TensorDescriptor(GraphElementType.Int32, [1]), graphInput: true);
        builder.AddResource("logits", "Logits", GraphResourceKind.Output, GraphResourceLifetime.External,
            new TensorDescriptor(GraphElementType.Float32, [tensors.VocabularySize]), graphOutput: true);

        var shapes = new Dictionary<string, int[]>(StringComparer.Ordinal)
        {
            ["token"] = [1],
            ["logits"] = [tensors.VocabularySize],
        };
        var types = new Dictionary<string, GraphElementType>(StringComparer.Ordinal);
        foreach (var name in tensors.Names.Order(StringComparer.Ordinal))
        {
            var tensor = tensors.GetRequired(name);
            if (tensor.DataType is not (RwkvTensorDataType.Float16 or RwkvTensorDataType.Float32))
                throw new InvalidDataException($"Tensor '{name}' must be FP16 or FP32.");
            var id = $"weight.{name}";
            var type = tensor.DataType == RwkvTensorDataType.Float16
                ? GraphElementType.Float16 : GraphElementType.Float32;
            builder.AddResource(id, name, GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(type, tensor.Dimensions), name);
            shapes.Add(id, tensor.Dimensions.ToArray());
            types.Add(id, type);
        }
        for (var layer = 0; layer < tensors.LayerCount; layer++)
        {
            foreach (var (suffix, dimensions) in new (string, int[])[]
            {
                ("ffn-previous", [width]), ("att-previous", [width]),
                ("wkv", [heads, size, size]),
            })
            {
                var id = $"state.{layer}.{suffix}";
                builder.AddResource(id, id, GraphResourceKind.SessionState, GraphResourceLifetime.Session,
                    new TensorDescriptor(GraphElementType.Float32, dimensions)).AddStateSlot(id, id);
                shapes.Add(id, dimensions);
            }
        }

        var serial = 0;
        string? previous = null;
        string Node(GraphOperationId operation, (string Port, string Resource)[] inputs,
            int[] resultShape, string? resultId = null, IReadOnlyDictionary<string, string>? attributes = null,
            GraphElementType type = GraphElementType.Float32)
        {
            var id = $"portable.{serial++:D5}";
            resultId ??= id;
            if (shapes.TryGetValue(resultId, out var existing))
            {
                if (!existing.SequenceEqual(resultShape) ||
                    (types.TryGetValue(resultId, out var existingType) && existingType != type))
                    throw new InvalidDataException($"Resource '{resultId}' has incompatible dimensions or type.");
            }
            else
            {
                builder.AddResource(resultId, resultId, GraphResourceKind.Temporary,
                    GraphResourceLifetime.Invocation, new TensorDescriptor(type, resultShape));
                shapes.Add(resultId, resultShape);
                types.Add(resultId, type);
            }
            builder.AddNode(id, operation, "graph",
                inputs.Select(input => GraphBindings.Read(input.Port, input.Resource))
                    .Append(GraphBindings.Write("output", resultId)),
                previous is null ? null : [previous], attributes,
                new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32));
            previous = id;
            return resultId;
        }
        int[] Shape(string resource) => shapes[resource];
        static long Count(int[] dimensions) =>
            dimensions.Aggregate(1L, (product, dimension) => checked(product * dimension));
        string Reshape(string input, params int[] dimensions)
        {
            if (Count(Shape(input)) != Count(dimensions))
                throw new InvalidDataException($"Cannot reshape '{input}'.");
            return Shape(input).SequenceEqual(dimensions) ? input :
                Node(TensorOps.Reshape, [("input", input)], dimensions);
        }
        string Broadcast(string input, params int[] dimensions) =>
            Node(TensorOps.Broadcast, [("input", input)], dimensions);
        string Fill(float value, params int[] dimensions) =>
            Node(TensorOps.Fill, [], dimensions, attributes: new TensorFillValue(value).ToAttributes());
        string Weight(string name)
        {
            var id = $"weight.{name}";
            if (!types.TryGetValue(id, out var type))
                throw new InvalidDataException($"Missing RWKV-6 tensor '{name}'.");
            return type == GraphElementType.Float32 ? id :
                Node(TensorOps.CastFp16ToFp32, [("input", id)], Shape(id));
        }
        string VectorWeight(string name, int length)
        {
            var weight = Weight(name);
            if (Count(Shape(weight)) != length)
                throw new InvalidDataException($"Tensor '{name}' must contain {length} values.");
            return Reshape(weight, length);
        }
        string Unary(GraphOperationId op, string input) =>
            Node(op, [("input", input)], Shape(input));
        string Binary(GraphOperationId op, string left, string right)
        {
            if (!Shape(left).SequenceEqual(Shape(right)))
                throw new InvalidDataException($"Incompatible elementwise dimensions: '{left}', '{right}'.");
            return Node(op, [("left", left), ("right", right)], Shape(left));
        }
        string Add(string left, string right) => Binary(PrimitiveGraphOperations.Add, left, right);
        string Sub(string left, string right) => Binary(PrimitiveGraphOperations.Subtract, left, right);
        string Mul(string left, string right) => Binary(PrimitiveGraphOperations.Multiply, left, right);
        string Mean(string input) => Node(PrimitiveGraphOperations.ReduceMean, [("input", input)], [1]);
        string HeadMean(string input) => Node(TensorOps.ReduceLastMean,
            [("input", input)], Shape(input)[..^1]);
        string ExpandHead(string input) => Broadcast(Reshape(input, heads, 1), heads, size);
        string Matrix(string name, int inputLength)
        {
            var weight = Weight(name);
            var dimensions = Shape(weight);
            if (dimensions.Length != 2 || dimensions[0] != inputLength)
                throw new InvalidDataException($"Tensor '{name}' has invalid projection dimensions.");
            // GGML dimensions [input,output] store consecutive rows of input elements.
            return Reshape(weight, dimensions[1], dimensions[0]);
        }
        string Project(string name, string input)
        {
            if (Shape(input).Length != 1)
                throw new InvalidDataException($"Projection input '{input}' must be a vector.");
            var matrix = Matrix(name, Shape(input)[0]);
            return Node(PrimitiveGraphOperations.MatVec,
                [("matrix", matrix), ("input", input)], [Shape(matrix)[0]]);
        }
        string Normalize(string input, float epsilon)
        {
            var centered = Sub(input, Broadcast(Mean(input), Shape(input)));
            var variance = Mean(Unary(PrimitiveGraphOperations.Square, centered));
            var inverse = Unary(PrimitiveGraphOperations.ReciprocalSquareRoot,
                Add(variance, Fill(epsilon, 1)));
            return Mul(centered, Broadcast(inverse, Shape(input)));
        }
        string GroupNormalize(string input)
        {
            var grouped = Reshape(input, heads, size);
            var centered = Sub(grouped, ExpandHead(HeadMean(grouped)));
            var variance = HeadMean(Unary(PrimitiveGraphOperations.Square, centered));
            var inverse = Unary(PrimitiveGraphOperations.ReciprocalSquareRoot,
                Add(variance, Fill(64e-5f, heads)));
            return Reshape(Mul(centered, ExpandHead(inverse)), width);
        }
        string Affine(string input, string weight, string bias) =>
            Add(Mul(input, VectorWeight(weight, Shape(input)[0])),
                VectorWeight(bias, Shape(input)[0]));
        string Mix(string x, string delta, string coefficient) => Add(x, Mul(delta, coefficient));
        string Slice(string input, int axis, int start, int length, params int[] dimensions) =>
            Node(TensorOps.Slice, [("input", input)], dimensions,
                attributes: new TensorSlice(axis, start, length).ToAttributes());
        string Outer(string left, string right) => Node(TensorOps.HeadOuter,
            [("left", left), ("right", right)], [heads, size, size]);
        void Store(string source, string state) =>
            Node(PrimitiveGraphOperations.Copy, [("input", source)], Shape(state), state);

        var embedding = Matrix("emb.weight", width);
        if (Shape(embedding)[0] != tensors.VocabularySize)
            throw new InvalidDataException("The embedding does not match the vocabulary size.");
        var x = Node(PrimitiveGraphOperations.GatherRow,
            [("table", embedding), ("index", "token")], [width]);
        x = Affine(Normalize(x, 1e-5f), "blocks.0.ln0.weight", "blocks.0.ln0.bias");
        for (var layer = 0; layer < tensors.LayerCount; layer++)
        {
            var prefix = $"blocks.{layer}.";
            string Att(string suffix) => prefix + "att." + suffix;
            var attPrevious = $"state.{layer}.att-previous";
            var ffnPrevious = $"state.{layer}.ffn-previous";
            var wkvState = $"state.{layer}.wkv";
            var normalized = Affine(Normalize(x, 1e-5f), prefix + "ln1.weight", prefix + "ln1.bias");
            var delta = Sub(attPrevious, normalized);
            var maaInput = Mix(normalized, delta, VectorWeight(Att("time_maa_x"), width));
            var w1 = Project(Att("time_maa_w1"), maaInput);
            if (Shape(w1)[0] % 5 != 0)
                throw new InvalidDataException($"Tensor '{Att("time_maa_w1")}' must have five hidden groups.");
            var hidden = Shape(w1)[0] / 5;
            var w2 = Weight(Att("time_maa_w2"));
            if (!Shape(w2).SequenceEqual(new[] { hidden, width, 5 }))
                throw new InvalidDataException($"Tensor '{Att("time_maa_w2")}' has invalid grouped projection dimensions.");
            var groupedWeights = Reshape(w2, 5, width, hidden);
            var groupedInput = Reshape(Unary(PrimitiveGraphOperations.Tanh, w1), 5, hidden);
            var maa = Node(TensorOps.BatchedMatVec,
                [("matrix", groupedWeights), ("vector", groupedInput)], [5, width]);
            string DynamicMix(int group) => Reshape(Slice(maa, 0, group, 1, 1, width), width);
            string Mixed(int group, string suffix) =>
                Mix(normalized, delta, Add(DynamicMix(group), VectorWeight(Att(suffix), width)));
            var xw = Mixed(0, "time_maa_w");
            var xk = Mixed(1, "time_maa_k");
            var xv = Mixed(2, "time_maa_v");
            var xr = Mixed(3, "time_maa_r");
            var xg = Mixed(4, "time_maa_g");

            var r = Project(Att("receptance.weight"), xr);
            var k = Project(Att("key.weight"), xk);
            var v = Project(Att("value.weight"), xv);
            var gateLinear = Project(Att("gate.weight"), xg);
            var gate = Mul(gateLinear, Unary(PrimitiveGraphOperations.Sigmoid, gateLinear));
            var decayOffset = Add(Project(Att("time_decay_w2"),
                    Unary(PrimitiveGraphOperations.Tanh, Project(Att("time_decay_w1"), xw))),
                VectorWeight(Att("time_decay"), width));
            var decay = Unary(PrimitiveGraphOperations.Exp,
                Sub(Fill(0, width), Unary(PrimitiveGraphOperations.Exp, decayOffset)));

            var keyed = Reshape(k, heads, size);
            var valued = Reshape(v, heads, size);
            var received = Reshape(r, heads, size);
            var firstValues = Reshape(VectorWeight(Att("time_faaaa"), width), heads, size);
            var products = Outer(keyed, valued);
            var firstMatrix = Broadcast(Reshape(firstValues, heads, size, 1), heads, size, size);
            var responseMatrix = Broadcast(Reshape(received, heads, size, 1), heads, size, size);
            var contributions = Mul(Add(wkvState, Mul(products, firstMatrix)), responseMatrix);
            // Sum across the matrix row axis using fixed-size slices; ReduceLastSum
            // only reduces columns, and reshaping cannot transpose the state.
            string? attentionHeads = null;
            for (var row = 0; row < size; row++)
            {
                var term = Reshape(Slice(contributions, 1, row, 1, heads, 1, size), heads, size);
                attentionHeads = attentionHeads is null ? term : Add(attentionHeads, term);
            }
            var decayMatrix = Broadcast(Reshape(Reshape(decay, heads, size), heads, size, 1),
                heads, size, size);
            var updated = Add(Mul(wkvState, decayMatrix), products);
            Store(updated, wkvState);

            var attention = Affine(GroupNormalize(Reshape(attentionHeads!, width)),
                Att("ln_x.weight"), Att("ln_x.bias"));
            x = Add(x, Project(Att("output.weight"), Mul(attention, gate)));
            Store(normalized, attPrevious);

            var ffnNormalized = Affine(Normalize(x, 1e-5f),
                prefix + "ln2.weight", prefix + "ln2.bias");
            var ffnDelta = Sub(ffnPrevious, ffnNormalized);
            var ffnKeyInput = Mix(ffnNormalized, ffnDelta,
                VectorWeight(prefix + "ffn.time_maa_k", width));
            var ffnReceptanceInput = Mix(ffnNormalized, ffnDelta,
                VectorWeight(prefix + "ffn.time_maa_r", width));
            var ffnGate = Unary(PrimitiveGraphOperations.Sigmoid,
                Project(prefix + "ffn.receptance.weight", ffnReceptanceInput));
            var hiddenValue = Unary(PrimitiveGraphOperations.Square,
                Unary(PrimitiveGraphOperations.Relu, Project(prefix + "ffn.key.weight", ffnKeyInput)));
            x = Add(x, Mul(Project(prefix + "ffn.value.weight", hiddenValue), ffnGate));
            Store(ffnNormalized, ffnPrevious);
        }
        var output = Affine(Normalize(x, 1e-5f), "ln_out.weight", "ln_out.bias");
        Store(Project("head.weight", output), "logits");

        var graph = builder.Build();
        TensorOps.ValidateGraph(graph);
        return graph;
    }
}
