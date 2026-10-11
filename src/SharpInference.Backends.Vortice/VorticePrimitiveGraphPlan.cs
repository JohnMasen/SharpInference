using SharpInference.Graphs;

namespace SharpInference.Backends.Vortice;

/// <summary>Describes one compiled primitive GPU dispatch and its tensor bindings, dimensions, and optional fused expression.</summary>
/// <param name="Node">The originating execution node identifier.</param>
/// <param name="Kernel">The GPU kernel name selected for this dispatch.</param>
/// <param name="Input0">The primary input resource identifier.</param>
/// <param name="Input1">The optional second input resource identifier.</param>
/// <param name="Output">The destination resource identifier.</param>
/// <param name="Index">The optional Int32 scalar resource used for row selection.</param>
/// <param name="ElementCount">The number of output elements processed by the dispatch.</param>
/// <param name="Rows">The kernel-specific row count, source rank, or slice axis.</param>
/// <param name="Columns">The kernel-specific column count, reduction width, or destination rank.</param>
/// <param name="DispatchX">The linear thread-group count, split across X and Y for kernels supporting large grids.</param>
/// <param name="Parameter">The kernel-specific scalar, such as a fill value&apos;s Float32 bits or a slice offset.</param>
/// <param name="SourceDimensions">Optional source dimensions used by portable tensor kernels.</param>
/// <param name="OutputDimensions">Optional destination dimensions used by portable tensor kernels.</param>
/// <param name="FusedInputs">The ordered input resources for an optional fused elementwise expression.</param>
/// <param name="Expression">The optional fused elementwise expression evaluated by the dispatch.</param>
public sealed record VorticePrimitiveGraphStep(
    ExecutionNodeId Node,
    string Kernel,
    ResourceId Input0,
    ResourceId? Input1,
    ResourceId Output,
    ResourceId? Index,
    uint ElementCount,
    uint Rows,
    uint Columns,
    uint DispatchX,
    uint Parameter = 0,
    IReadOnlyList<uint>? SourceDimensions = null,
    IReadOnlyList<uint>? OutputDimensions = null,
    IReadOnlyList<ResourceId>? FusedInputs = null,
    FusedElementwiseExpression? Expression = null);

/// <summary>Contains a validated execution graph and its ordered GPU dispatch steps.</summary>
public sealed class VorticePrimitiveGraphPlan : IBackendExecutablePlan
{
    private VorticePrimitiveGraphPlan(ExecutionGraph graph, IReadOnlyList<VorticePrimitiveGraphStep> steps)
    {
        Graph = graph;
        Steps = steps;
    }

    /// <inheritdoc/>
    public ExecutionGraph Graph { get; }
    /// <summary>Gets the ordered dispatch steps produced by graph compilation.</summary>
    public IReadOnlyList<VorticePrimitiveGraphStep> Steps { get; }

    /// <summary>Validates a dense primitive execution graph and resolves its nodes into GPU dispatch steps.</summary>
    /// <param name="graph">The dense execution graph whose resources and nodes are prepared.</param>
    public static VorticePrimitiveGraphPlan Compile(ExecutionGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        if (graph.Nodes.Any(node => node.Resources.Any(binding => binding.View is not null)))
            throw new NotSupportedException("The primitive reference backend does not support tensor-view bindings; use a VM adapter.");
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        var completed = new HashSet<ExecutionNodeId>();
        var initialized = graph.Resources
            .Where(resource => graph.Inputs.Contains(resource.Id) ||
                resource.Kind is GraphResourceKind.Weight or GraphResourceKind.Constant or GraphResourceKind.SessionState)
            .Select(resource => resource.Id).ToHashSet();
        var written = new HashSet<ResourceId>();
        var steps = new List<VorticePrimitiveGraphStep>();
        foreach (var resource in graph.Resources)
        {
            if (resource.Tensor.Layout != "dense" ||
                resource.Tensor.ElementType is not (GraphElementType.Float32 or GraphElementType.Float16 or GraphElementType.Int32) ||
                resource.Scope == GraphResourceScope.Global &&
                resource.Kind is not (GraphResourceKind.Weight or GraphResourceKind.Constant) ||
                resource.Tensor.ElementType == GraphElementType.Float16 &&
                (resource.Scope != GraphResourceScope.Global ||
                    resource.Kind is not (GraphResourceKind.Weight or GraphResourceKind.Constant)) ||
                resource.Tensor.ElementType == GraphElementType.Int32 &&
                (resource.Kind != GraphResourceKind.Input || resource.Tensor.Dimensions.Aggregate(1L, (a, b) => checked(a * b)) != 1))
                throw new NotSupportedException($"Vortice primitive graph resource '{resource.Id}' requires dense FP32 data, a model FP16 weight, or a scalar Int32 graph input.");
        }

        foreach (var node in graph.Nodes)
        {
            NotSupportedException Unsupported(string reason) =>
                new($"Vortice primitive graph node '{node.Id}' ({node.Operation}): {reason}");
            if (node.Dependencies.Any(dependency => !completed.Contains(dependency)))
                throw Unsupported("dependencies must precede the node in execution order.");
            if (node.InternalResources.Count != 0)
                throw Unsupported("internal/fused resources are not supported.");
            if (node.Requirements.MinimumArithmeticType != GraphElementType.Float32 ||
                node.Requirements.MinimumAccumulatorType != GraphElementType.Float32)
                throw Unsupported("only FP32 arithmetic and accumulation are supported.");

            if (node.Operation == FusedElementwiseExpressionContract.Operation)
            {
                var fused = VorticeFusedExpressionKernel.Compile(node, resources, initialized);
                steps.Add(fused);
                initialized.Add(fused.Output);
                written.Add(fused.Output);
                completed.Add(node.Id);
                continue;
            }

            if (node.Operation.Name.StartsWith("core.tensor.", StringComparison.Ordinal))
            {
                var portable = VorticePortableTensorPlanner.Compile(node, resources, initialized);
                steps.Add(portable);
                initialized.Add(portable.Output);
                written.Add(portable.Output);
                completed.Add(node.Id);
                continue;
            }

            var kernel = node.Operation switch
            {
                var op when op == PrimitiveGraphOperations.Copy => "Copy",
                var op when op == PrimitiveGraphOperations.Add => "Add",
                var op when op == PrimitiveGraphOperations.Subtract => "Subtract",
                var op when op == PrimitiveGraphOperations.Multiply => "Multiply",
                var op when op == PrimitiveGraphOperations.Divide => "Divide",
                var op when op == PrimitiveGraphOperations.Maximum => "Maximum",
                var op when op == PrimitiveGraphOperations.Relu => "Relu",
                var op when op == PrimitiveGraphOperations.Sigmoid => "Sigmoid",
                var op when op == PrimitiveGraphOperations.Exp => "Exp",
                var op when op == PrimitiveGraphOperations.Tanh => "Tanh",
                var op when op == PrimitiveGraphOperations.ReciprocalSquareRoot => "ReciprocalSquareRoot",
                var op when op == PrimitiveGraphOperations.Square => "Square",
                var op when op == PrimitiveGraphOperations.ReduceSum => "ReduceSum",
                var op when op == PrimitiveGraphOperations.ReduceMean => "ReduceMean",
                var op when op == PrimitiveGraphOperations.MatVec => "MatVec",
                var op when op == PrimitiveGraphOperations.GatherRow => "GatherRow",
                _ => throw Unsupported("no GPU graph kernel is available for this operation."),
            };
            var binary = kernel is "Add" or "Subtract" or "Multiply" or "Divide" or "Maximum" or "MatVec";
            var gather = kernel == "GatherRow";
            var reduce = kernel is "ReduceSum" or "ReduceMean";
            var matrixPort = node.Resources.Any(binding => binding.Port == "weight") ? "weight" : "matrix";
            var inputPorts = kernel == "MatVec" ? new[] { matrixPort, "input" } :
                gather ? ["table", "index"] :
                binary ? ["left", "right"] : ["input"];
            const string outputPort = "output";
            if (node.Resources.Count != inputPorts.Length + 1 ||
                node.Resources.Any(binding => !inputPorts.Contains(binding.Port, StringComparer.Ordinal) && binding.Port != outputPort))
                throw Unsupported($"expected ports {string.Join(", ", inputPorts)} (read) and {outputPort} (write).");
            ResourceId Port(string port, GraphResourceAccess access)
            {
                var binding = node.Resources.SingleOrDefault(item => item.Port == port);
                if (binding is null || binding.Access != access)
                    throw Unsupported($"port '{port}' must have {access} access.");
                return binding.Resource;
            }
            var input0 = Port(inputPorts[0], GraphResourceAccess.Read);
            var input1 = binary ? Port(inputPorts[1], GraphResourceAccess.Read) : (ResourceId?)null;
            var index = gather ? Port("index", GraphResourceAccess.Read) : (ResourceId?)null;
            var output = Port(outputPort, GraphResourceAccess.Write);
            if (input0 == output || input1 == output || index == output)
                throw Unsupported("in-place writes are not supported.");
            foreach (var input in node.Resources.Where(binding => binding.Access == GraphResourceAccess.Read))
                if (!initialized.Contains(input.Resource))
                    throw Unsupported($"resource '{input.Resource}' is read before it is produced.");
            if (resources[input0].Tensor.ElementType != GraphElementType.Float32 ||
                input1 is ResourceId second && resources[second].Tensor.ElementType != GraphElementType.Float32 ||
                resources[output].Tensor.ElementType != GraphElementType.Float32 ||
                index is ResourceId scalar && (resources[scalar].Tensor.ElementType != GraphElementType.Int32 ||
                    !graph.Inputs.Contains(scalar)))
                throw Unsupported("data ports require FP32 tensors; gather index requires an Int32 graph input.");
            uint Count(ResourceId id) => checked((uint)resources[id].Tensor.Dimensions.Aggregate(1L, (a, b) => checked(a * b)));
            var outCount = Count(output);
            var columns = gather ? outCount : 0u;
            uint rows = 0;
            if (kernel == "MatVec")
            {
                var matrix = resources[input0].Tensor.Dimensions;
                if (matrix.Count != 2)
                    throw Unsupported("matrix must have shape [rows, columns] in row-major dense layout.");
                rows = checked((uint)matrix[0]);
                columns = checked((uint)matrix[1]);
                if (Count(input1!.Value) != columns || outCount != rows)
                    throw Unsupported("matrix, vector, and output dimensions do not match.");
            }
            else if (gather)
            {
                var table = resources[input0].Tensor.Dimensions;
                if (table.Count != 2 || table[1] != outCount)
                    throw Unsupported("table must have shape [rows, output length] in row-major dense layout.");
                rows = checked((uint)table[0]);
            }
            else if (reduce)
            {
                if (outCount != 1)
                    throw Unsupported("primitive reduction output must contain exactly one FP32 element.");
            }
            else if (!resources[input0].Tensor.Dimensions.SequenceEqual(resources[output].Tensor.Dimensions) ||
                input1 is ResourceId other &&
                !resources[other].Tensor.Dimensions.SequenceEqual(resources[output].Tensor.Dimensions))
                throw Unsupported("elementwise input and output lengths must match.");
            initialized.Add(output);
            written.Add(output);
            completed.Add(node.Id);
            if (kernel == "MatVec" && rows > 65535)
                kernel = "MatVecLarge";
            var dispatchX = kernel is "MatVec" or "MatVecLarge" ? rows : reduce ? 1u :
                checked((uint)(((ulong)outCount + 63) / 64));
            if (dispatchX > 65535 && kernel != "MatVecLarge")
                throw Unsupported($"dispatch requires {dispatchX} thread groups, exceeding the D3D12 X-axis limit of 65535.");
            steps.Add(new VorticePrimitiveGraphStep(node.Id, kernel, input0, input1, output, index,
                reduce ? Count(input0) : outCount, rows, columns, dispatchX));
        }
        foreach (var output in graph.Outputs)
            if (!written.Contains(output) || resources[output].Tensor.ElementType != GraphElementType.Float32)
                throw new NotSupportedException($"Vortice primitive graph output '{output}' must be produced as FP32.");
        return new VorticePrimitiveGraphPlan(graph, steps);
    }
}
