using SharpInference.Graphs;
using TensorOps = SharpInference.Architectures.RwkvPortableTensorOperations;
using SharpInference.Graphs.Logical;

namespace SharpInference.Architectures.Rwkv6;

/// <summary>
/// Unfused RWKV-6 single-token graph. Tensor operations require a backend
/// implementing the portable tensor contracts.
/// </summary>
public sealed class PortableRwkv6GraphProvider
{
    /// <inheritdoc/>
    public string ArchitectureId => "rwkv-6";

    /// <inheritdoc/>
    public LogicalGraphDefinition Build(IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var modelDimensions = RwkvTensorCatalogDimensions.Read(tensors);
        if (!tensors.TryGet("blocks.0.att.time_maa_x", out _) ||
            !tensors.TryGet("blocks.0.att.time_faaaa", out var first) ||
            tensors.TryGet("blocks.0.att.k_k", out _))
        {
            throw new InvalidDataException("The tensor catalog does not satisfy the RWKV-6 x060 contract.");
        }

        var width = modelDimensions.EmbeddingSize;
        if (first.Dimensions.Count != 3 || first.Dimensions[0] != 1 ||
            first.Dimensions[1] <= 0 || first.Dimensions[2] <= 0 ||
            checked(first.Dimensions[1] * first.Dimensions[2]) != width)
        {
            throw new InvalidDataException("RWKV-6 time_faaaa has incompatible head dimensions.");
        }

        var size = first.Dimensions[1];
        var heads = first.Dimensions[2];
        return PortableRwkvGraphDefinitionComposer.Define(
            "rwkv6.forward-token.portable",
            modelDimensions.VocabularySize,
            builder =>
        {
        var graphScope = builder.Root("graph", "Portable forward token", new(
            Description: "Embedding, recurrent RWKV6 Blocks and language-model head."));
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
                {
                    throw new InvalidDataException($"Tensor '{name}' must be FP16 or FP32.");
                }

                var id = $"weight.{name}";
            var type = tensor.DataType == TensorDataType.Float16
                ? GraphElementType.Float16 : GraphElementType.Float32;
            builder.AddWeight(id, name, new TensorDescriptor(type, tensor.Dimensions), name);
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
                builder.AddState(id, new TensorDescriptor(GraphElementType.Float32, dimensions));
                shapes.Add(id, dimensions);
            }
        }

        var serial = 0;
        T ArchitectureValue<T>(RwkvArchitectureScope parent, string id, string name,
            Func<RwkvArchitectureScope, T> build, RwkvArchitectureAnnotation? architecture = null) =>
            parent.Value(id, name, build, architecture);
        static void Within(RwkvArchitectureScope scope, Action build) => build();
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
                    {
                        throw new InvalidDataException($"Resource '{resultId}' has incompatible dimensions or type.");
                    }
                }
            else
            {
                builder.AddTemporary(resultId, new TensorDescriptor(type, resultShape));
                shapes.Add(resultId, resultShape);
                types.Add(resultId, type);
            }
            builder.AddNode(id, operation, inputs, resultId, attributes,
                new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32));
            return resultId;
        }
        int[] Shape(string resource) => shapes[resource];
        static long Count(int[] dimensions) =>
            dimensions.Aggregate(1L, (product, dimension) => checked(product * dimension));
        string Reshape(string input, params int[] dimensions)
        {
            if (Count(Shape(input)) != Count(dimensions))
                {
                    throw new InvalidDataException($"Cannot reshape '{input}'.");
                }

                return Shape(input).SequenceEqual(dimensions) ? input :
                Node(TensorOps.Reshape, [("input", input)], dimensions);
        }
        string Broadcast(string input, params int[] dimensions) =>
            Node(TensorOps.Broadcast, [("input", input)], dimensions);
        string Fill(float value, params int[] dimensions) =>
            Node(TensorOps.Fill, [], dimensions, attributes: TensorOps.FillAttributes(value));
        string Weight(string name)
        {
            var id = $"weight.{name}";
            if (!types.TryGetValue(id, out var type))
                {
                    throw new InvalidDataException($"Missing RWKV-6 tensor '{name}'.");
                }

                return type == GraphElementType.Float32 ? id :
                Node(TensorOps.CastFp16ToFp32, [("input", id)], Shape(id));
        }
        string VectorWeight(string name, int length)
        {
            var weight = Weight(name);
            if (Count(Shape(weight)) != length)
                {
                    throw new InvalidDataException($"Tensor '{name}' must contain {length} values.");
                }

                return Reshape(weight, length);
        }
        string Unary(GraphOperationId op, string input) =>
            Node(op, [("input", input)], Shape(input));
        string Binary(GraphOperationId op, string left, string right)
        {
            if (!Shape(left).SequenceEqual(Shape(right)))
                {
                    throw new InvalidDataException($"Incompatible elementwise dimensions: '{left}', '{right}'.");
                }

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
                {
                    throw new InvalidDataException($"Tensor '{name}' has invalid projection dimensions.");
                }
                // GGML dimensions [input,output] store consecutive rows of input elements.
                return Reshape(weight, dimensions[1], dimensions[0]);
        }
        string Project(string name, string input)
        {
            if (Shape(input).Length != 1)
                {
                    throw new InvalidDataException($"Projection input '{input}' must be a vector.");
                }

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
                attributes: TensorOps.SliceAttributes(axis, start, length));
        string Outer(string left, string right) => Node(TensorOps.HeadOuter,
            [("left", left), ("right", right)], [heads, size, size]);
        void Store(string source, string state) =>
            Node(PrimitiveGraphOperations.Copy, [("input", source)], Shape(state), state);

        var x = ArchitectureValue(graphScope, "embedding", "Embedding", _ =>
        {
            var embedding = Matrix("emb.weight", width);
            if (Shape(embedding)[0] != modelDimensions.VocabularySize)
            {
                throw new InvalidDataException("The embedding does not match the vocabulary size.");
            }

            var embedded = Node(PrimitiveGraphOperations.GatherRow,
                [("table", embedding), ("index", "token")], [width]);
            return Affine(Normalize(embedded, 1e-5f), "blocks.0.ln0.weight", "blocks.0.ln0.bias");
        }, new(Description: "Token embedding and initial affine LayerNorm."));
        graphScope.Child("blocks", "RWKV6 Blocks", blocksScope =>
        {
            for (var layer = 0; layer < modelDimensions.LayerCount; layer++)
            {
                var layerPath = $"layer.{layer}";
                blocksScope.Child(layerPath, "RWKV6 Block", layerScope =>
                {
                    var prefix = $"blocks.{layer}.";
                    string Att(string suffix) => prefix + "att." + suffix;
                    var attPrevious = $"state.{layer}.att-previous";
                    var ffnPrevious = $"state.{layer}.ffn-previous";
                    var wkvState = $"state.{layer}.wkv";
                    var attentionPath = layerPath + ".attention";
                    layerScope.Child(attentionPath, "Attention", attentionScope =>
                {
                    var timeMix = layerPath + ".time-mix";
                    var normalized = ArchitectureValue(attentionScope, layerPath + ".norm1", "LayerNorm 1", _ => Affine(Normalize(x, 1e-5f), prefix + "ln1.weight", prefix + "ln1.bias"), new("normalization", "Normalizes the input before TimeMix.", null));
                    string timeMixOutput = null!;
                    attentionScope.Child(timeMix, "TimeMix", timeMixScope =>
                {
                    var projections = timeMix + ".projections";
                    var (xw, xk, xv, xr, xg) = ArchitectureValue(timeMixScope, timeMix + ".input-mixing", "Dynamic Input Mixing", _ =>
                    {
                        var delta = Sub(attPrevious, normalized);
                        var maaInput = Mix(normalized, delta, VectorWeight(Att("time_maa_x"), width));
                        var w1 = Project(Att("time_maa_w1"), maaInput);
                        if (Shape(w1)[0] % 5 != 0)
                        {
                            throw new InvalidDataException($"Tensor '{Att("time_maa_w1")}' must have five hidden groups.");
                        }

                        var hidden = Shape(w1)[0] / 5;
                        var w2 = Weight(Att("time_maa_w2"));
                        if (!Shape(w2).SequenceEqual(new[] { hidden, width, 5 }))
                        {
                            throw new InvalidDataException($"Tensor '{Att("time_maa_w2")}' has invalid grouped projection dimensions.");
                        }

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
                    }, new("temporal-mixing", "Computes low-rank mixing coefficients and W/K/V/R/G inputs.", "x_q = x + (x_prev - x) .* (m_q + dynamic_m_q)"));
                    string r = null!, k = null!, v = null!, gate = null!, decay = null!;
                    timeMixScope.Child(projections, "Projections and Gates", projectionsScope =>
                    {
                        r = ArchitectureValue(projectionsScope, projections + ".r", "R: Receptance", _ => Project(Att("receptance.weight"), xr), new("receptance", "Projects receptance.", "r = W_r x_r"));
                        k = ArchitectureValue(projectionsScope, projections + ".k", "K: Key", _ => Project(Att("key.weight"), xk), new("key", "Projects the key.", "k = W_k x_k"));
                        v = ArchitectureValue(projectionsScope, projections + ".v", "V: Value", _ => Project(Att("value.weight"), xv), new("value", "Projects the value.", "v = W_v x_v"));
                        gate = ArchitectureValue(projectionsScope, projections + ".g", "G: Output Gate", _ =>
                        {
                            var gateLinear = Project(Att("gate.weight"), xg);
                            var gate = Mul(gateLinear, Unary(PrimitiveGraphOperations.Sigmoid, gateLinear));
                            return gate;
                        }, new("output-gate", "Applies SiLU to the gate projection.", "g = silu(W_g x_g)"));
                        decay = ArchitectureValue(projectionsScope, projections + ".w", "W: Decay", _ =>
                        {
                            var decayOffset = Add(Project(Att("time_decay_w2"), Unary(PrimitiveGraphOperations.Tanh, Project(Att("time_decay_w1"), xw))), VectorWeight(Att("time_decay"), width));
                            var decay = Unary(PrimitiveGraphOperations.Exp, Sub(Fill(0, width), Unary(PrimitiveGraphOperations.Exp, decayOffset)));
                            return decay;
                        }, new("decay", "Computes low-rank time decay.", "w = exp(-exp(w0 + W2 tanh(W1 x_w)))"));
                    }, architecture: new("projection-and-gating", "R/K/V projections, SiLU output gate and exponential decay.", null));
                    var (products, firstMatrix, responseMatrix) = ArchitectureValue(timeMixScope, timeMix + ".head-preparation", "Head Preparation", _ =>
                    {
                        var keyed = Reshape(k, heads, size);
                        var valued = Reshape(v, heads, size);
                        var received = Reshape(r, heads, size);
                        var firstValues = Reshape(VectorWeight(Att("time_faaaa"), width), heads, size);
                        var products = Outer(keyed, valued);
                        var firstMatrix = Broadcast(Reshape(firstValues, heads, size, 1), heads, size, size);
                        var responseMatrix = Broadcast(Reshape(received, heads, size, 1), heads, size, size);
                        return (products, firstMatrix, responseMatrix);
                    }, new("head-preparation", "Reshapes projections, constructs key/value outer products and first-value weights.", "P = k v^T"));
                    var attentionHeads = ArchitectureValue(timeMixScope, timeMix + ".state-read", "Memory Read", _ =>
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
                    }, new("state-read", "Reads the old memory with the current key/value bonus, reducing the row axis.", "y_j = sum_i r_i * (S_prev[i,j] + u_i * k_i * v_j)"));
                    timeMixScope.Child(timeMix + ".state-update", "Memory Update / Commit", stateUpdateScope =>
                        Within(stateUpdateScope, () =>
                        {
                            var decayMatrix = Broadcast(Reshape(Reshape(decay, heads, size), heads, size, 1), heads, size, size);
                            var updated = Add(Mul(wkvState, decayMatrix), products);
                            Store(updated, wkvState);
                        }), architecture: new("state-update", "Decays matrix rows, adds the current key/value outer product and commits the state.", "S_new = diag(w) S_prev + k v^T"));
                    timeMixOutput = ArchitectureValue(timeMixScope, timeMix + ".output", "Output Processing", _ =>
                    {
                        var attention = Affine(GroupNormalize(Reshape(attentionHeads!, width)), Att("ln_x.weight"), Att("ln_x.bias"));
                        var timeMixOutput = Project(Att("output.weight"), Mul(attention, gate));
                        return timeMixOutput;
                    }, new("output-processing", "Head-wise normalization, output gating and projection.", null));
                }, architecture: new("time-mix", "Dynamic temporal mixing, gated projections, old-memory read, memory update and output processing.", null));
                    attentionScope.Child(layerPath + ".residual1", "Residual Add 1", residualScope =>
                    Within(residualScope, () =>
                    {
                        x = Add(x, timeMixOutput);
                    }), architecture: new("residual-add", "Adds the TimeMix output.", "x <- x + TimeMix(x)"));
                    attentionScope.Child(layerPath + ".attention-history", "Attention History Store", historyScope =>
                    Within(historyScope, () =>
                    {
                        Store(normalized, attPrevious);
                    }), architecture: new("state-store", "Stores the normalized input for the next token.", null));
                }, architecture: new("attention"));

                    var ffnPath = layerPath + ".ffn";
                    layerScope.Child(ffnPath, "Feed forward", ffnScope =>
                {
                    var channelMix = layerPath + ".channel-mix";
                    var ffnNormalized = ArchitectureValue(ffnScope, layerPath + ".norm2", "LayerNorm 2", _ => Affine(Normalize(x, 1e-5f), prefix + "ln2.weight", prefix + "ln2.bias"), new("normalization", "Normalizes the input before ChannelMix.", null));
                    string channelMixOutput = null!;
                    ffnScope.Child(channelMix, "ChannelMix", channelMixScope =>
                {
                    var (ffnKeyInput, ffnReceptanceInput) = ArchitectureValue(channelMixScope, channelMix + ".input-mixing", "Input Mixing", _ =>
                    {
                        var ffnDelta = Sub(ffnPrevious, ffnNormalized);
                        var ffnKeyInput = Mix(ffnNormalized, ffnDelta, VectorWeight(prefix + "ffn.time_maa_k", width));
                        var ffnReceptanceInput = Mix(ffnNormalized, ffnDelta, VectorWeight(prefix + "ffn.time_maa_r", width));
                        return (ffnKeyInput, ffnReceptanceInput);
                    }, new("temporal-mixing", "Produces key and receptance inputs from FFN history.", null));
                    var ffnGate = ArchitectureValue(channelMixScope, channelMix + ".receptance", "Receptance Gate", _ => Unary(PrimitiveGraphOperations.Sigmoid, Project(prefix + "ffn.receptance.weight", ffnReceptanceInput)), new("receptance", "Computes the sigmoid FFN gate.", "r = sigmoid(W_r x_r)"));
                    var projectedKey = ArchitectureValue(channelMixScope, channelMix + ".key", "Key Projection", _ => Project(prefix + "ffn.key.weight", ffnKeyInput), new("key-projection", "Projects to the FFN hidden width.", null));
                    var hiddenValue = ArchitectureValue(channelMixScope, channelMix + ".activation", "ReLU Squared", _ => Unary(PrimitiveGraphOperations.Square, Unary(PrimitiveGraphOperations.Relu, projectedKey)), new("activation", "Applies squared ReLU.", "h = relu(W_k x_k)^2"));
                    channelMixOutput = ArchitectureValue(channelMixScope, channelMix + ".value", "Gated Value Projection", _ => Mul(Project(prefix + "ffn.value.weight", hiddenValue), ffnGate), new("value-projection", "Projects back to the embedding width and applies receptance.", "y = (W_v h) .* r"));
                }, architecture: new("channel-mix", "Temporal input mixing and receptance-gated squared-ReLU feed forward.", null));
                    ffnScope.Child(layerPath + ".residual2", "Residual Add 2", residualScope =>
                    Within(residualScope, () =>
                    {
                        x = Add(x, channelMixOutput);
                    }), architecture: new("residual-add", "Adds the ChannelMix output.", "x <- x + ChannelMix(x)"));
                    ffnScope.Child(layerPath + ".ffn-history", "FFN History Store", historyScope =>
                    Within(historyScope, () =>
                    {
                        Store(ffnNormalized, ffnPrevious);
                    }), architecture: new("state-store", "Stores the normalized FFN input for the next token.", null));
                }, architecture: new("ffn"));
                }, architecture: new(Description: "Pre-normalized TimeMix and gated ChannelMix, residual connections and history stores."));
            }
        }, architecture: new(Description: "Ordered recurrent Block instances."));
        graphScope.Child("output", "LM Head", outputScope =>
            Within(outputScope, () =>
            {
                var output = Affine(Normalize(x, 1e-5f), "ln_out.weight", "ln_out.bias");
                Store(Project("head.weight", output), "logits");
            }), architecture: new(Description: "Final affine LayerNorm and vocabulary projection."));

        }, new GraphAnnotation(
            Name: "RWKV-6 forward token",
            Role: "forward-token",
            Description: "RWKV-6 single-token portable logical graph."));
    }
}
