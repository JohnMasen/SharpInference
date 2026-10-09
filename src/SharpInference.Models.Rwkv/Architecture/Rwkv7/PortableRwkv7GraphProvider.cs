using SharpInference.Graphs;

namespace SharpInference.Architectures.Rwkv7;

/// <summary>
/// Unfused RWKV-7 single-token logical graph. Its tensor operations
/// require backend implementations before execution.
/// </summary>
public sealed class PortableRwkv7GraphProvider : ILogicalGraphProvider
{
    public string ArchitectureId => "rwkv-7";

    public LogicalGraph Build(IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var modelDimensions = RwkvTensorCatalogDimensions.Read(tensors);
        if (!tensors.TryGet("blocks.0.att.r_k", out var receptanceKey) ||
            !tensors.TryGet("blocks.0.att.x_rwkvag", out _))
            throw new InvalidDataException("The tensor catalog does not satisfy the RWKV-7 x070 contract.");
        var width = modelDimensions.EmbeddingSize;
        if (receptanceKey.Dimensions.Count != 2 || width <= 0 ||
            receptanceKey.Dimensions[1] <= 0 || width % receptanceKey.Dimensions[1] != 0)
            throw new InvalidDataException("RWKV-7 head dimensions do not match the embedding size.");
        var heads = receptanceKey.Dimensions[1];
        var headSize = width / heads;
        if (receptanceKey.Dimensions[0] != headSize)
            throw new InvalidDataException("RWKV-7 receptance-key dimensions do not match the head shape.");

        var builder = new LogicalGraphBuilder(
            new GraphIdentity("rwkv-7", 1, "rwkv7.forward-token.portable"),
            RwkvGraphSignatures.Create(modelDimensions.VocabularySize, width, modelDimensions.LayerCount, heads,
                headSize, "rwkv-7.state.fp32@1"))
            .SetStateSchema(new StateSchema("RWKV7_State"));
        builder.AddRegion("graph", GraphRegionTypes.Graph, "Portable forward token");
        builder.AddResource("token", "Token", GraphResourceKind.Input, GraphResourceLifetime.External,
            new TensorDescriptor(GraphElementType.Int32, [1]), graphInput: true);
        builder.AddResource("logits", "Logits", GraphResourceKind.Output, GraphResourceLifetime.External,
            new TensorDescriptor(GraphElementType.Float32, [modelDimensions.VocabularySize]), graphOutput: true);
        var shapes = new Dictionary<string, int[]>(StringComparer.Ordinal)
        {
            ["token"] = [1],
            ["logits"] = [modelDimensions.VocabularySize],
        };
        var types = new Dictionary<string, GraphElementType>(StringComparer.Ordinal);
        foreach (var name in tensors.Names.Order(StringComparer.Ordinal))
        {
            var tensor = tensors.GetRequired(name);
            if (tensor.DataType is not (TensorDataType.Float16 or TensorDataType.Float32))
                throw new InvalidDataException($"Tensor '{name}' must have FP16 or FP32 values.");
            var id = $"weight.{name}";
            var type = tensor.DataType == TensorDataType.Float16
                ? GraphElementType.Float16 : GraphElementType.Float32;
            builder.AddResource(id, name, GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(type, tensor.Dimensions), name);
            shapes.Add(id, tensor.Dimensions.ToArray());
            types.Add(id, type);
        }
        for (var layer = 0; layer < modelDimensions.LayerCount; layer++)
        {
            foreach (var (suffix, dimensions) in new (string, int[])[]
            {
                ("ffn-previous", [width]), ("att-previous", [width]),
                ("wkv", [heads, headSize, headSize]),
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
            int[] resultShape, string? resultId = null, IReadOnlyDictionary<string, string>? attributes = null)
        {
            var id = $"portable.{serial++:D5}";
            resultId ??= id;
            if (shapes.TryGetValue(resultId, out var existing))
            {
                if (!existing.SequenceEqual(resultShape))
                    throw new InvalidDataException($"Resource '{resultId}' has incompatible dimensions.");
            }
            else
            {
                builder.AddResource(resultId, resultId, GraphResourceKind.Temporary,
                    GraphResourceLifetime.Invocation,
                    new TensorDescriptor(GraphElementType.Float32, resultShape));
                shapes.Add(resultId, resultShape);
            }
            var bindings = inputs.Select(input => GraphBindings.Read(input.Port, input.Resource))
                .Append(GraphBindings.Write("output", resultId));
            builder.AddNode(id, operation, "graph", bindings, previous is null ? null : [previous],
                attributes, new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32));
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
                Node(PortableTensorOperationContracts.Reshape, [("input", input)], dimensions);
        }
        string Broadcast(string input, params int[] dimensions) =>
            Node(PortableTensorOperationContracts.Broadcast, [("input", input)], dimensions);
        string Fill(float value, params int[] dimensions) =>
            Node(PortableTensorOperationContracts.Fill, [], dimensions,
                attributes: new TensorFillValue(value).ToAttributes());
        string Weight(string name)
        {
            var id = $"weight.{name}";
            if (types[id] == GraphElementType.Float32) return id;
            return Node(PortableTensorOperationContracts.CastFp16ToFp32,
                [("input", id)], Shape(id));
        }
        string VectorWeight(string name, int length)
        {
            var tensor = Weight(name);
            if (Count(Shape(tensor)) != length)
                throw new InvalidDataException($"Tensor '{name}' does not have {length} elements.");
            return Reshape(tensor, length);
        }
        string Unary(GraphOperationId operation, string input) =>
            Node(operation, [("input", input)], Shape(input));
        string Binary(GraphOperationId operation, string left, string right)
        {
            if (!Shape(left).SequenceEqual(Shape(right)))
                throw new InvalidDataException($"Incompatible elementwise dimensions: '{left}', '{right}'.");
            return Node(operation, [("left", left), ("right", right)], Shape(left));
        }
        string Add(string left, string right) => Binary(PrimitiveGraphOperations.Add, left, right);
        string Sub(string left, string right) => Binary(PrimitiveGraphOperations.Subtract, left, right);
        string Mul(string left, string right) => Binary(PrimitiveGraphOperations.Multiply, left, right);
        string Mean(string input) => Node(PrimitiveGraphOperations.ReduceMean,
            [("input", input)], [1]);
        string ReduceHead(string input, bool mean) => Node(
            mean ? PortableTensorOperationContracts.ReduceLastMean :
                PortableTensorOperationContracts.ReduceLastSum,
            [("input", input)], Shape(input)[..^1]);
        string ExpandHead(string input) => Broadcast(Reshape(input, heads, 1), heads, headSize);
        string Matrix(string name, int inputLength)
        {
            var weight = Weight(name);
            var dimensions = Shape(weight);
            if (dimensions.Length != 2 || dimensions[0] != inputLength)
                throw new InvalidDataException($"Tensor '{name}' has invalid projection dimensions.");
            // GGML [input,output] uses row-major flattened [output,input] values.
            return Reshape(weight, dimensions[1], dimensions[0]);
        }
        string Project(string name, string input)
        {
            var dimensions = Shape(input);
            if (dimensions.Length != 1)
                throw new InvalidDataException($"Projection input '{input}' must be a vector.");
            var matrix = Matrix(name, dimensions[0]);
            return Node(PrimitiveGraphOperations.MatVec,
                [("matrix", matrix), ("input", input)], [Shape(matrix)[0]]);
        }
        string Normalize(string input, float epsilon)
        {
            var dimensions = Shape(input);
            var centered = Sub(input, Broadcast(Mean(input), dimensions));
            var variance = Mean(Unary(PrimitiveGraphOperations.Square, centered));
            var invStd = Unary(PrimitiveGraphOperations.ReciprocalSquareRoot,
                Add(variance, Fill(epsilon, 1)));
            return Mul(centered, Broadcast(invStd, dimensions));
        }
        string GroupNormalize(string input)
        {
            var grouped = Reshape(input, heads, headSize);
            var centered = Sub(grouped, ExpandHead(ReduceHead(grouped, mean: true)));
            var variance = ReduceHead(Unary(PrimitiveGraphOperations.Square, centered), mean: true);
            var invStd = Unary(PrimitiveGraphOperations.ReciprocalSquareRoot,
                Add(variance, Fill(64e-5f, heads)));
            return Reshape(Mul(centered, ExpandHead(invStd)), width);
        }
        string Affine(string input, string weight, string bias) =>
            Add(Mul(input, VectorWeight(weight, Shape(input)[0])),
                VectorWeight(bias, Shape(input)[0]));
        string Mix(string input, string previousValue, string mix) =>
            Add(input, Mul(Sub(previousValue, input), mix));
        string SliceMix(string mixes, int row) =>
            Reshape(Node(PortableTensorOperationContracts.Slice,
                    [("input", mixes)], [1, width],
                    attributes: new TensorSlice(0, row, 1).ToAttributes()), width);
        string Outer(string left, string right) => Node(
            PortableTensorOperationContracts.HeadOuter,
            [("left", left), ("right", right)], [heads, headSize, headSize]);
        string BatchedMatVec(string matrix, string vector) => Node(
            PortableTensorOperationContracts.BatchedMatVec,
            [("matrix", matrix), ("vector", vector)], [heads, headSize]);
        void Store(string source, string state) =>
            Node(PrimitiveGraphOperations.Copy, [("input", source)], Shape(state), state);

        var embedding = Matrix("emb.weight", width);
        if (Shape(embedding)[0] != modelDimensions.VocabularySize)
            throw new InvalidDataException("The embedding does not match the vocabulary size.");
        var x = Node(PrimitiveGraphOperations.GatherRow,
            [("table", embedding), ("index", "token")], [width]);
        x = Affine(Normalize(x, 1e-5f), "blocks.0.ln0.weight", "blocks.0.ln0.bias");
        string? firstValue = null;
        for (var layer = 0; layer < modelDimensions.LayerCount; layer++)
        {
            var prefix = $"blocks.{layer}.";
            string Att(string suffix) => prefix + "att." + suffix;
            var attPrevious = $"state.{layer}.att-previous";
            var ffnPrevious = $"state.{layer}.ffn-previous";
            var wkvState = $"state.{layer}.wkv";
            var normalized = Affine(Normalize(x, 1e-5f),
                prefix + "ln1.weight", prefix + "ln1.bias");
            var mixes = Reshape(Weight(Att("x_rwkvag")), 6, width);
            var xr = Mix(normalized, attPrevious, SliceMix(mixes, 0));
            var xw = Mix(normalized, attPrevious, SliceMix(mixes, 1));
            var xk = Mix(normalized, attPrevious, SliceMix(mixes, 2));
            var xv = Mix(normalized, attPrevious, SliceMix(mixes, 3));
            var xa = Mix(normalized, attPrevious, SliceMix(mixes, 4));
            var xg = Mix(normalized, attPrevious, SliceMix(mixes, 5));
            var r = Project(Att("receptance.weight"), xr);
            var key = Project(Att("key.weight"), xk);
            var value = Project(Att("value.weight"), xv);
            var decayOffset = Add(Project(Att("w2"),
                Unary(PrimitiveGraphOperations.Tanh, Project(Att("w1"), xw))),
                VectorWeight(Att("w0"), width));
            var decay = Unary(PrimitiveGraphOperations.Exp,
                Mul(Fill(-0.606531f, width),
                    Unary(PrimitiveGraphOperations.Sigmoid, decayOffset)));
            var adaptation = Unary(PrimitiveGraphOperations.Sigmoid,
                Add(Project(Att("a2"), Project(Att("a1"), xa)),
                    VectorWeight(Att("a0"), width)));
            var gate = Project(Att("g2"),
                Unary(PrimitiveGraphOperations.Sigmoid, Project(Att("g1"), xg)));

            var scaledKey = Mul(key, VectorWeight(Att("k_k"), width));
            var keyHeads = Reshape(scaledKey, heads, headSize);
            var squares = ReduceHead(Unary(PrimitiveGraphOperations.Square, keyHeads), mean: false);
            var inverseMagnitude = Unary(PrimitiveGraphOperations.ReciprocalSquareRoot,
                Binary(PrimitiveGraphOperations.Maximum, squares, Fill(1e-24f, heads)));
            var normalizedKey = Mul(keyHeads, ExpandHead(inverseMagnitude));
            var ka = Mul(key, VectorWeight(Att("k_a"), width));
            key = Sub(Add(key, Mul(adaptation, ka)), ka);
            if (firstValue is null)
                firstValue = value;
            else
            {
                var valueGate = Unary(PrimitiveGraphOperations.Sigmoid,
                    Add(Project(Att("v2"), Project(Att("v1"), xv)),
                        VectorWeight(Att("v0"), width)));
                value = Add(value, Mul(Sub(firstValue, value), valueGate));
            }

            var keyed = Reshape(key, heads, headSize);
            var valued = Reshape(value, heads, headSize);
            var received = Reshape(r, heads, headSize);
            var adaptedKey = Mul(normalizedKey, Reshape(adaptation, heads, headSize));
            var negativeProjection = Sub(Fill(0, heads, headSize),
                BatchedMatVec(wkvState, normalizedKey));
            var decayMatrix = Broadcast(Reshape(decay, heads, 1, headSize),
                heads, headSize, headSize);
            // S'[h,i,j] = S[h,i,j]*w[h,j] + v[h,i]*k[h,j]
            //             - (sum_l S[h,i,l]*kn[h,l])*kn[h,j]*a[h,j].
            var updated = Add(
                Add(Mul(wkvState, decayMatrix), Outer(valued, keyed)),
                Outer(negativeProjection, adaptedKey));
            var attention = Reshape(BatchedMatVec(updated, received), width);
            Store(updated, wkvState);
            attention = Affine(GroupNormalize(attention),
                Att("ln_x.weight"), Att("ln_x.bias"));
            var bonusFactors = Mul(Mul(key, r), VectorWeight(Att("r_k"), width));
            var bonus = ExpandHead(ReduceHead(Reshape(bonusFactors, heads, headSize), mean: false));
            attention = Mul(Add(attention, Mul(value, Reshape(bonus, width))), gate);
            x = Add(x, Project(Att("output.weight"), attention));
            Store(normalized, attPrevious);

            var ffnNormalized = Affine(Normalize(x, 1e-5f),
                prefix + "ln2.weight", prefix + "ln2.bias");
            var keyInput = Mix(ffnNormalized, ffnPrevious,
                VectorWeight(prefix + "ffn.x_k", width));
            var hidden = Unary(PrimitiveGraphOperations.Relu,
                Project(prefix + "ffn.key.weight", keyInput));
            x = Add(x, Project(prefix + "ffn.value.weight",
                Unary(PrimitiveGraphOperations.Square, hidden)));
            Store(ffnNormalized, ffnPrevious);
        }
        var output = Affine(Normalize(x, 1e-5f), "ln_out.weight", "ln_out.bias");
        var logits = Project("head.weight", output);
        Node(PrimitiveGraphOperations.Copy, [("input", logits)], [modelDimensions.VocabularySize], "logits");

        var graph = builder.Build();
        PortableTensorOperationContracts.ValidateGraph(graph);
        return graph;
    }
}
