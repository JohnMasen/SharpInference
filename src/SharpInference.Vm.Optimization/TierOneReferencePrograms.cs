using SharpInference.Graphs;
using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization;

public static class TierOneReferencePrograms
{
    public static LogicalGraph CreateGraph(string operation, IReadOnlyList<int> dimensions)
    {
        var definition = TierOnePointwiseCatalog.Get(operation);
        var tensor = new TensorDescriptor(GraphElementType.Float32, dimensions);
        var count = dimensions.Aggregate(1, (value, dimension) => checked(value * dimension));
        var resources = definition.Inputs.Select(port => new GraphResource(new(port), port, GraphResourceKind.Input,
            GraphResourceLifetime.External, tensor)).ToList();
        resources.Add(new(new("output"), "output", GraphResourceKind.Output, GraphResourceLifetime.External, tensor));
        var nodes = new List<LogicalNode>();
        for (var index = 0; index < definition.Stages.Count; index++)
        {
            var stage = definition.Stages[index];
            var output = index == definition.Stages.Count - 1 ? "output" : $"t{index}";
            if (output != "output")
                resources.Add(new(new(output), output, GraphResourceKind.Temporary, GraphResourceLifetime.Invocation, tensor));
            var inputs = stage.Operands.Select(operand => operand.IsInput ? definition.Inputs[operand.Index] :
                $"t{operand.Index}").ToArray();
            var bindings = inputs.Select((source, port) => new NodeResourceBinding(
                inputs.Length == 1 ? "input" : port == 0 ? "left" : "right", new(source), GraphResourceAccess.Read))
                .Append(new("output", new(output), GraphResourceAccess.Write)).ToArray();
            nodes.Add(new(new($"stage{index}"), stage.Operation, new("pointwise"), bindings,
                stage.Operands.Where(operand => !operand.IsInput).Select(operand => new LogicalNodeId($"stage{operand.Index}"))
                    .Distinct().ToArray(), new Dictionary<string, string>(),
                new(GraphElementType.Float32, GraphElementType.Float32)));
        }
        return new(new("t1-reference", 1, $"t1-{operation}"),
            new("pointwise", "t1-reference", new Dictionary<string, int> { ["elementCount"] = count }), resources,
            [new(new("root"), null, GraphRegionTypes.Graph, null, "root", new Dictionary<string, string>()),
                new(new("pointwise"), new("root"), GraphRegionTypes.Stage, null, "pointwise", new Dictionary<string, string>())],
            nodes, definition.Inputs.Select(port => new ResourceId(port)).ToArray(), [new("output")],
            new GraphState(new("t1-reference"), []));
    }

    public static VmProgram CreateFused(string operation, VmTarget target, IReadOnlyList<int> dimensions,
        uint threadsPerGroup = 64)
    {
        var definition = TierOnePointwiseCatalog.Get(operation);
        var tensor = new VmTensor(VmElementType.Float32, dimensions);
        var parameters = definition.Inputs.Select(port => new VmParameter(port, VmAccess.ReadOnly, tensor))
            .Append(new("output", VmAccess.ReadWrite, tensor)).ToArray();
        var arguments = parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)).ToArray();
        InstructionExecutionConfiguration configuration = target == VmTarget.Cpu
            ? new CpuInstructionExecutionConfiguration(new("adaptive-pointwise"))
            : new D3D12InstructionExecutionConfiguration(new("register-pointwise"));
        var body = new VmDefinition("body", target == VmTarget.Cpu ? VmDefinitionKind.Function : VmDefinitionKind.Kernel,
            parameters, [new("compute", new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32),
                InstructionCollectionIds.TierOneFloat32, operation, arguments, executionConfiguration: configuration))],
            target == VmTarget.Cpu ? null : new(threadsPerGroup));
        var calls = new List<VmNode>();
        if (target == VmTarget.Cpu)
            calls.Add(new("compute", new VmCall("body", arguments)));
        else
        {
            calls.Add(new("compute", new VmDispatch("body", arguments,
                new(checked((uint)((tensor.ElementCount + threadsPerGroup - 1) / threadsPerGroup))))));
            calls.Add(new("barrier", new VmBarrier(["output"]), ["compute"]));
        }
        return new($"t1-{operation}", "vm:t1-reference", target,
            parameters.Select(parameter => new VmSlot(parameter.Name, VmSlotScope.Local, parameter.Access, tensor)),
            [body, new("forward", VmDefinitionKind.Orchestration, parameters, calls)],
            [new("forward", "forward", arguments)], new("t1-reference", 1, []));
    }
}
