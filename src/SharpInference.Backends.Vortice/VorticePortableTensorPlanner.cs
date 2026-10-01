using System.Globalization;
using SharpInference.Graphs;

namespace SharpInference.Backends.Vortice;

internal static class VorticePortableTensorPlanner
{
    public static VorticePrimitiveGraphStep Compile(
        ExecutionNode node,
        IReadOnlyDictionary<ResourceId, GraphResource> resources,
        IReadOnlySet<ResourceId> initialized)
    {
        NotSupportedException Unsupported(string reason) =>
            new($"Vortice primitive graph node '{node.Id}' ({node.Operation}): {reason}");
        try
        {
            PortableTensorOperationContracts.ValidateNode(
                new LogicalNode(new LogicalNodeId(node.Id.Value), node.Operation, node.Region,
                    node.Resources, [], node.Attributes, node.Requirements), resources);
        }
        catch (InvalidDataException error)
        {
            throw Unsupported(error.Message);
        }
        ResourceId Port(string port) => node.Resources.Single(binding => binding.Port == port).Resource;
        uint Count(ResourceId id) => checked((uint)resources[id].Tensor.Dimensions.Aggregate(
            1L, (size, dimension) => checked(size * dimension)));
        var contract = PortableTensorOperationContracts.Contracts.Single(item => item.Operation == node.Operation);
        var output = Port("output");
        var input0 = contract.InputPorts.Count > 0 ? Port(contract.InputPorts[0]) : output;
        var input1 = contract.InputPorts.Count > 1 ? Port(contract.InputPorts[1]) : (ResourceId?)null;
        foreach (var port in contract.InputPorts)
            if (!initialized.Contains(Port(port)))
                throw Unsupported($"resource '{Port(port)}' is read before it is produced.");
        var cast = node.Operation == PortableTensorOperationContracts.CastFp16ToFp32;
        if (resources[output].Tensor.ElementType != GraphElementType.Float32 ||
            contract.InputPorts.Any(port => resources[Port(port)].Tensor.ElementType !=
                (cast ? GraphElementType.Float16 : GraphElementType.Float32)))
            throw Unsupported("the tensor kernel requires FP32 data ports.");
        if (cast && resources[input0].Kind is not (GraphResourceKind.Weight or GraphResourceKind.Constant))
            throw Unsupported("FP16 cast input must be a model weight or constant.");

        var outputDims = resources[output].Tensor.Dimensions;
        var inputDims = resources[input0].Tensor.Dimensions;
        var kernel = "";
        uint rows = 0, columns = 0, parameter = 0;
        if (cast)
            kernel = "CastFp16ToFp32";
        else if (node.Operation == PortableTensorOperationContracts.Fill)
        {
            kernel = "Fill";
            parameter = BitConverter.SingleToUInt32Bits(
                float.Parse(node.Attributes["value"], CultureInfo.InvariantCulture));
        }
        else if (node.Operation == PortableTensorOperationContracts.Reshape)
            kernel = "PortableCopy";
        else if (node.Operation == PortableTensorOperationContracts.Slice)
        {
            if (inputDims.Count is < 1 or > 4)
                throw Unsupported("the GPU slice kernel supports ranks 1 through 4.");
            var axis = int.Parse(node.Attributes["axis"], CultureInfo.InvariantCulture);
            kernel = "Slice";
            rows = checked((uint)axis);
            columns = checked((uint)outputDims.Count);
            parameter = uint.Parse(node.Attributes["start"], CultureInfo.InvariantCulture);
        }
        else if (node.Operation == PortableTensorOperationContracts.Broadcast)
        {
            if (outputDims.Count is < 1 or > 4 || inputDims.Count is < 1 or > 4)
                throw Unsupported("the GPU broadcast kernel supports ranks 1 through 4.");
            kernel = "Broadcast";
            rows = checked((uint)inputDims.Count);
            columns = checked((uint)outputDims.Count);
        }
        else if (node.Operation == PortableTensorOperationContracts.BatchedMatVec)
        {
            kernel = "BatchedMatVec";
            rows = checked((uint)inputDims[1]);
            columns = checked((uint)inputDims[2]);
        }
        else if (node.Operation == PortableTensorOperationContracts.ReduceLastSum ||
                 node.Operation == PortableTensorOperationContracts.ReduceLastMean)
        {
            kernel = node.Operation == PortableTensorOperationContracts.ReduceLastSum
                ? "ReduceLastSum" : "ReduceLastMean";
            columns = checked((uint)inputDims[^1]);
        }
        else if (node.Operation == PortableTensorOperationContracts.HeadOuter)
        {
            kernel = "HeadOuter";
            rows = checked((uint)inputDims[1]);
            columns = checked((uint)resources[input1!.Value].Tensor.Dimensions[1]);
        }
        else
            throw Unsupported("no GPU kernel is available for this tensor operation.");

        var count = Count(output);
        var groups = checked((uint)(((ulong)count + 63) / 64));
        if (groups > 65535UL * 65535UL)
            throw Unsupported($"dispatch requires {groups} thread groups, exceeding the D3D12 XY grid limit.");
        return new VorticePrimitiveGraphStep(node.Id, kernel, input0, input1, output, null,
            count, rows, columns, groups, parameter,
            kernel is "Slice" or "Broadcast"
                ? inputDims.Select(value => checked((uint)value)).ToArray() : null,
            kernel is "Slice" or "Broadcast"
                ? outputDims.Select(value => checked((uint)value)).ToArray() : null);
    }
}
