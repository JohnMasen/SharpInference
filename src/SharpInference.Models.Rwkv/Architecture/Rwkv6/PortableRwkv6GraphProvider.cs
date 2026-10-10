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
        var modelDimensions = RwkvTensorCatalogDimensions.Read(tensors);
        if (!tensors.TryGet("blocks.0.att.time_maa_x", out _) ||
            !tensors.TryGet("blocks.0.att.time_faaaa", out var first) ||
            tensors.TryGet("blocks.0.att.k_k", out _))
            throw new InvalidDataException("The tensor catalog does not satisfy the RWKV-6 x060 contract.");

        var width = modelDimensions.EmbeddingSize;
        if (first.Dimensions.Count != 3 || first.Dimensions[0] != 1 ||
            first.Dimensions[1] <= 0 || first.Dimensions[2] <= 0 ||
            checked(first.Dimensions[1] * first.Dimensions[2]) != width)
            throw new InvalidDataException("RWKV-6 time_faaaa has incompatible head dimensions.");
        var size = first.Dimensions[1];
        var heads = first.Dimensions[2];
        var builder = new LogicalGraphBuilder(
            new GraphIdentity("rwkv-6", 1, "rwkv6.forward-token.portable"),
            RwkvGraphSignatures.Create(modelDimensions.VocabularySize, width, modelDimensions.LayerCount,
                heads, size, "rwkv-6.state.fp32@1"))
            .SetStateSchema(new StateSchema("RWKV6_State"));
        builder.AddRegion("graph", GraphRegionTypes.Graph, "Portable forward token", architecture: new(
            Description: "Embedding, recurrent RWKV6 Blocks and language-model head.",
            DefaultView: GraphArchitectureView.Architecture, Step: GraphArchitectureStep.Token));
        builder.AddRegion("blocks", GraphRegionTypes.Architecture, "RWKV6 Blocks", "graph",
            architecture: new(Description: "Ordered recurrent Block instances.", DefaultCollapsed: true, RepeatGroup: true));
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
                throw new InvalidDataException($"Tensor '{name}' must be FP16 or FP32.");
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
        string Node(GraphOperationId operation, (string Port, string Resource)[] inputs,
            int[] resultShape, string? resultId = null, IReadOnlyDictionary<string, string>? attributes = null,
            GraphElementType type = GraphElementType.Float32)
        {
            var id = $"portable.{serial++:D5}";
            resultId ??= $"{id}_output";
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
            builder.AddNode(id, operation,
                inputs.Select(input => GraphBindings.Read(input.Port, input.Resource))
                    .Append(GraphBindings.Write("output", resultId)),
                null, attributes,
                new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32));
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

        var x = builder.WithRegion("embedding", GraphRegionTypes.Stage, "Embedding", _ =>
        {
            var embedding = Matrix("emb.weight", width);
            if (Shape(embedding)[0] != modelDimensions.VocabularySize)
                throw new InvalidDataException("The embedding does not match the vocabulary size.");
            var embedded = Node(PrimitiveGraphOperations.GatherRow,
                [("table", embedding), ("index", "token")], [width]);
            return Affine(Normalize(embedded, 1e-5f), "blocks.0.ln0.weight", "blocks.0.ln0.bias");
        }, parentId: "graph", architecture: new(Description: "Token embedding and initial affine LayerNorm."));
        for (var layer = 0; layer < modelDimensions.LayerCount; layer++)
        {
            var layerRegion = $"layer.{layer}";
            builder.AddRegion(layerRegion, GraphRegionTypes.Layer, "RWKV6 Block", "blocks",
                architecture: new(Description: "Pre-normalized TimeMix and gated ChannelMix, residual connections and history stores."));
            var attentionRegion = layerRegion + ".attention";
            builder.AddRegion(attentionRegion, GraphRegionTypes.Stage, "Attention", layerRegion, role: "attention",
                architecture: new(DefaultCollapsed: false));
            var prefix = $"blocks.{layer}.";
            string Att(string suffix) => prefix + "att." + suffix;
            var attPrevious = $"state.{layer}.att-previous";
            var ffnPrevious = $"state.{layer}.ffn-previous";
            var wkvState = $"state.{layer}.wkv";
            var timeMix = layerRegion + ".time-mix";
            var normalized = builder.WithRegion(layerRegion + ".norm1", GraphRegionTypes.Architecture, "LayerNorm 1", _ => Affine(Normalize(x, 1e-5f), prefix + "ln1.weight", prefix + "ln1.bias"), parentId: attentionRegion, architecture: new("normalization", "Normalizes the input before TimeMix.", null));
            var timeMixOutput = builder.WithRegion(timeMix, GraphRegionTypes.Architecture, "TimeMix", _ =>
            {
                var projections = timeMix + ".projections";
                var (xw, xk, xv, xr, xg) = builder.WithRegion(timeMix + ".input-mixing", GraphRegionTypes.Architecture, "Dynamic Input Mixing", _ =>
                {
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
                    var maa = Node(TensorOps.BatchedMatVec, [("matrix", groupedWeights), ("vector", groupedInput)], [5, width]);
                    string DynamicMix(int group) => Reshape(Slice(maa, 0, group, 1, 1, width), width);
                    string Mixed(int group, string suffix) => Mix(normalized, delta, Add(DynamicMix(group), VectorWeight(Att(suffix), width)));
                    var xw = Mixed(0, "time_maa_w");
                    var xk = Mixed(1, "time_maa_k");
                    var xv = Mixed(2, "time_maa_v");
                    var xr = Mixed(3, "time_maa_r");
                    var xg = Mixed(4, "time_maa_g");
                    return (xw, xk, xv, xr, xg);
                }, architecture: new("temporal-mixing", "Computes low-rank mixing coefficients and W/K/V/R/G inputs.", "x_q = x + (x_prev - x) .* (m_q + dynamic_m_q)"));
                var (r, k, v, gate, decay) = builder.WithRegion(projections, GraphRegionTypes.Architecture, "Projections and Gates", _ =>
                {
                    var r = builder.WithRegion(projections + ".r", GraphRegionTypes.Architecture, "R: Receptance", _ => Project(Att("receptance.weight"), xr), architecture: new("receptance", "Projects receptance.", "r = W_r x_r"));
                    var k = builder.WithRegion(projections + ".k", GraphRegionTypes.Architecture, "K: Key", _ => Project(Att("key.weight"), xk), architecture: new("key", "Projects the key.", "k = W_k x_k"));
                    var v = builder.WithRegion(projections + ".v", GraphRegionTypes.Architecture, "V: Value", _ => Project(Att("value.weight"), xv), architecture: new("value", "Projects the value.", "v = W_v x_v"));
                    var gate = builder.WithRegion(projections + ".g", GraphRegionTypes.Architecture, "G: Output Gate", _ =>
                    {
                        var gateLinear = Project(Att("gate.weight"), xg);
                        var gate = Mul(gateLinear, Unary(PrimitiveGraphOperations.Sigmoid, gateLinear));
                        return gate;
                    }, architecture: new("output-gate", "Applies SiLU to the gate projection.", "g = silu(W_g x_g)"));
                    var decay = builder.WithRegion(projections + ".w", GraphRegionTypes.Architecture, "W: Decay", _ =>
                    {
                        var decayOffset = Add(Project(Att("time_decay_w2"), Unary(PrimitiveGraphOperations.Tanh, Project(Att("time_decay_w1"), xw))), VectorWeight(Att("time_decay"), width));
                        var decay = Unary(PrimitiveGraphOperations.Exp, Sub(Fill(0, width), Unary(PrimitiveGraphOperations.Exp, decayOffset)));
                        return decay;
                    }, architecture: new("decay", "Computes low-rank time decay.", "w = exp(-exp(w0 + W2 tanh(W1 x_w)))"));
                    return (r, k, v, gate, decay);
                }, architecture: new("projection-and-gating", "R/K/V projections, SiLU output gate and exponential decay.", null));
                var (products, firstMatrix, responseMatrix) = builder.WithRegion(timeMix + ".head-preparation", GraphRegionTypes.Architecture, "Head Preparation", _ =>
                {
                    var keyed = Reshape(k, heads, size);
                    var valued = Reshape(v, heads, size);
                    var received = Reshape(r, heads, size);
                    var firstValues = Reshape(VectorWeight(Att("time_faaaa"), width), heads, size);
                    var products = Outer(keyed, valued);
                    var firstMatrix = Broadcast(Reshape(firstValues, heads, size, 1), heads, size, size);
                    var responseMatrix = Broadcast(Reshape(received, heads, size, 1), heads, size, size);
                    return (products, firstMatrix, responseMatrix);
                }, architecture: new("head-preparation", "Reshapes projections, constructs key/value outer products and first-value weights.", "P = k v^T"));
                var attentionHeads = builder.WithRegion(timeMix + ".state-read", GraphRegionTypes.Architecture, "Memory Read", _ =>
                {
                    var contributions = Mul(Add(wkvState, Mul(products, firstMatrix)), responseMatrix);
                    // Sum across the matrix row axis using fixed-size slices; ReduceLastSum
                    // only reduces columns, and reshaping cannot transpose the state.
                    string? attentionHeads = null;
                    for (var row = 0; row < size; row++)
                    {
                        var term = Reshape(Slice(contributions, 1, row, 1, heads, 1, size), heads, size);
                        attentionHeads = attentionHeads is null ? term : Add(attentionHeads, term);
                    }

                    return attentionHeads;
                }, architecture: new("state-read", "Reads the old memory with the current key/value bonus, reducing the row axis.", "y_j = sum_i r_i * (S_prev[i,j] + u_i * k_i * v_j)"));
                builder.WithRegion(timeMix + ".state-update", GraphRegionTypes.Architecture, "Memory Update / Commit", _ =>
                {
                    var decayMatrix = Broadcast(Reshape(Reshape(decay, heads, size), heads, size, 1), heads, size, size);
                    var updated = Add(Mul(wkvState, decayMatrix), products);
                    Store(updated, wkvState);
                }, architecture: new("state-update", "Decays matrix rows, adds the current key/value outer product and commits the state.", "S_new = diag(w) S_prev + k v^T"));
                var timeMixOutput = builder.WithRegion(timeMix + ".output", GraphRegionTypes.Architecture, "Output Processing", _ =>
                {
                    var attention = Affine(GroupNormalize(Reshape(attentionHeads!, width)), Att("ln_x.weight"), Att("ln_x.bias"));
                    var timeMixOutput = Project(Att("output.weight"), Mul(attention, gate));
                    return timeMixOutput;
                }, architecture: new("output-processing", "Head-wise normalization, output gating and projection.", null));
                return timeMixOutput;
            }, parentId: attentionRegion, architecture: new("time-mix", "Dynamic temporal mixing, gated projections, old-memory read, memory update and output processing.", null));
            builder.WithRegion(layerRegion + ".residual1", GraphRegionTypes.Architecture, "Residual Add 1", _ =>
            {
                x = Add(x, timeMixOutput);
            }, parentId: attentionRegion, architecture: new("residual-add", "Adds the TimeMix output.", "x <- x + TimeMix(x)"));
            builder.WithRegion(layerRegion + ".attention-history", GraphRegionTypes.Architecture, "Attention History Store", _ =>
            {
                Store(normalized, attPrevious);
            }, parentId: attentionRegion, architecture: new("state-store", "Stores the normalized input for the next token.", null));

            var ffnRegion = layerRegion + ".ffn";
            builder.AddRegion(ffnRegion, GraphRegionTypes.Stage, "Feed forward", layerRegion, role: "ffn",
                architecture: new(DefaultCollapsed: false));
            var channelMix = layerRegion + ".channel-mix";
            var ffnNormalized = builder.WithRegion(layerRegion + ".norm2", GraphRegionTypes.Architecture, "LayerNorm 2", _ => Affine(Normalize(x, 1e-5f), prefix + "ln2.weight", prefix + "ln2.bias"), parentId: ffnRegion, architecture: new("normalization", "Normalizes the input before ChannelMix.", null));
            var channelMixOutput = builder.WithRegion(channelMix, GraphRegionTypes.Architecture, "ChannelMix", _ =>
            {
                var (ffnKeyInput, ffnReceptanceInput) = builder.WithRegion(channelMix + ".input-mixing", GraphRegionTypes.Architecture, "Input Mixing", _ =>
                {
                    var ffnDelta = Sub(ffnPrevious, ffnNormalized);
                    var ffnKeyInput = Mix(ffnNormalized, ffnDelta, VectorWeight(prefix + "ffn.time_maa_k", width));
                    var ffnReceptanceInput = Mix(ffnNormalized, ffnDelta, VectorWeight(prefix + "ffn.time_maa_r", width));
                    return (ffnKeyInput, ffnReceptanceInput);
                }, architecture: new("temporal-mixing", "Produces key and receptance inputs from FFN history.", null));
                var ffnGate = builder.WithRegion(channelMix + ".receptance", GraphRegionTypes.Architecture, "Receptance Gate", _ => Unary(PrimitiveGraphOperations.Sigmoid, Project(prefix + "ffn.receptance.weight", ffnReceptanceInput)), architecture: new("receptance", "Computes the sigmoid FFN gate.", "r = sigmoid(W_r x_r)"));
                var projectedKey = builder.WithRegion(channelMix + ".key", GraphRegionTypes.Architecture, "Key Projection", _ => Project(prefix + "ffn.key.weight", ffnKeyInput), architecture: new("key-projection", "Projects to the FFN hidden width.", null));
                var hiddenValue = builder.WithRegion(channelMix + ".activation", GraphRegionTypes.Architecture, "ReLU Squared", _ => Unary(PrimitiveGraphOperations.Square, Unary(PrimitiveGraphOperations.Relu, projectedKey)), architecture: new("activation", "Applies squared ReLU.", "h = relu(W_k x_k)^2"));
                var channelMixOutput = builder.WithRegion(channelMix + ".value", GraphRegionTypes.Architecture, "Gated Value Projection", _ => Mul(Project(prefix + "ffn.value.weight", hiddenValue), ffnGate), architecture: new("value-projection", "Projects back to the embedding width and applies receptance.", "y = (W_v h) .* r"));
                return channelMixOutput;
            }, parentId: ffnRegion, architecture: new("channel-mix", "Temporal input mixing and receptance-gated squared-ReLU feed forward.", null));
            builder.WithRegion(layerRegion + ".residual2", GraphRegionTypes.Architecture, "Residual Add 2", _ =>
            {
                x = Add(x, channelMixOutput);
            }, parentId: ffnRegion, architecture: new("residual-add", "Adds the ChannelMix output.", "x <- x + ChannelMix(x)"));
            builder.WithRegion(layerRegion + ".ffn-history", GraphRegionTypes.Architecture, "FFN History Store", _ =>
            {
                Store(ffnNormalized, ffnPrevious);
            }, parentId: ffnRegion, architecture: new("state-store", "Stores the normalized FFN input for the next token.", null));
        }
        builder.WithRegion("output", GraphRegionTypes.Stage, "LM Head", _ =>
        {
            var output = Affine(Normalize(x, 1e-5f), "ln_out.weight", "ln_out.bias");
            Store(Project("head.weight", output), "logits");
        }, parentId: "graph", architecture: new(Description: "Final affine LayerNorm and vocabulary projection."));

        var graph = GraphLayerReuse.Extract(builder.BuildSequential(), region => region.Id.Value switch
        {
            "embedding" => "Rwkv6.Embedding",
            "output" => "Rwkv6.Output",
            _ when region.Type == GraphRegionTypes.Layer => "Rwkv6.Layer",
            _ => null,
        });
        TensorOps.ValidateGraph(graph);
        return graph;
    }
}
