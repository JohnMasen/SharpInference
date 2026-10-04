using SharpInference.Vm;

namespace SharpInference.Backends.D3D12Vm;

internal sealed record D3D12VmBinding(int Slot, ulong Offset);
internal abstract record D3D12VmCommand;
internal sealed record D3D12VmDispatch(string Kernel, VmThreadGroup Groups,
    D3D12VmBinding[] Bindings) : D3D12VmCommand;
internal sealed record D3D12VmBarrier(int[] Slots) : D3D12VmCommand;
internal sealed record D3D12VmGatherIndex(int Slot, ulong ByteOffset, int Rows);

internal static class D3D12VmSchedule
{
    public static D3D12VmGatherIndex[] GatherIndices(VmProgram program, string? entry = null)
    {
        var definitions = program.Definitions.ToDictionary(d => d.Id, StringComparer.Ordinal);
        var result = new HashSet<D3D12VmGatherIndex>();
        void Walk(VmDefinition definition, Dictionary<string, D3D12VmBinding> bindings)
        {
            foreach (var node in definition.Nodes)
                if (node.Instruction is VmCall call)
                    Walk(definitions[call.Definition], call.Arguments.ToDictionary(a => a.Parameter,
                        a => bindings[a.Source] with { Offset = checked(bindings[a.Source].Offset + a.ByteOffset) }, StringComparer.Ordinal));
                else if (node.Instruction is VmOperator op)
                foreach (var bound in op.IndexBounds)
                {
                    var indexArgument = op.Arguments.Single(a => a.Parameter == bound.IndexPort);
                    var index = bindings[indexArgument.Source];
                    var table = definition.Parameters.Single(p => p.Name ==
                        op.Arguments.Single(a => a.Parameter == bound.TensorPort).Source);
                    result.Add(new(index.Slot, checked(index.Offset + indexArgument.ByteOffset), table.Tensor.Dimensions[bound.Axis]));
                }
        }
        var schedules = Create(program);
        var commands = entry is null ? schedules.Values.SelectMany(v => v) : schedules[entry];
        foreach (var dispatch in commands.OfType<D3D12VmDispatch>())
        {
            var definition = definitions[dispatch.Kernel];
            Walk(definition, definition.Parameters.Select((p, i) => (p.Name, Binding: dispatch.Bindings[i]))
                .ToDictionary(p => p.Name, p => p.Binding, StringComparer.Ordinal));
        }
        return result.ToArray();
    }

    public static IReadOnlyDictionary<string, D3D12VmCommand[]> Create(VmProgram program)
    {
        var definitions = program.Definitions.ToDictionary(d => d.Id, StringComparer.Ordinal);
        var slots = program.Slots.Select((s, i) => (s.Id, i)).ToDictionary(v => v.Id, v => v.i, StringComparer.Ordinal);
        var result = new Dictionary<string, D3D12VmCommand[]>(StringComparer.Ordinal);
        foreach (var entry in program.Entries)
        {
            if (definitions[entry.Definition].Kind != VmDefinitionKind.Orchestration)
                throw new NotSupportedException($"GPU entry '{entry.Name}' requires an explicit orchestration definition.");
            var commands = new List<D3D12VmCommand>();
            var pendingWrites = new HashSet<int>();
            void Walk(VmDefinition definition, Dictionary<string, D3D12VmBinding> bindings)
            {
                Dictionary<string, D3D12VmBinding> Bind(IReadOnlyList<VmArgument> arguments) =>
                    arguments.ToDictionary(a => a.Parameter, a =>
                    {
                        var source = bindings[a.Source];
                        return source with { Offset = checked(source.Offset + a.ByteOffset) };
                    }, StringComparer.Ordinal);
                foreach (var node in definition.Nodes)
                    switch (node.Instruction)
                    {
                        case VmCall call:
                            Walk(definitions[call.Definition], Bind(call.Arguments));
                            break;
                        case VmBarrier barrier:
                            var resources = barrier.Resources.Select(r => bindings[r].Slot).Distinct().ToArray();
                            commands.Add(new D3D12VmBarrier(resources));
                            pendingWrites.ExceptWith(resources);
                            break;
                        case VmDispatch dispatch:
                            var kernel = definitions[dispatch.Definition];
                            if ((ulong)dispatch.Groups.X * kernel.Threads!.X * dispatch.Groups.Y * kernel.Threads.Y *
                                dispatch.Groups.Z * kernel.Threads.Z > uint.MaxValue)
                                throw new NotSupportedException($"Dispatch '{node.Id}' exceeds 32-bit linear thread indexing.");
                            var map = Bind(dispatch.Arguments);
                            var bound = kernel.Parameters.Select(p => map[p.Name]).ToArray();
                            foreach (var binding in bound)
                                if (pendingWrites.Contains(binding.Slot))
                                    throw new InvalidDataException($"Dispatch '{node.Id}' accesses a pending UAV write; an explicit VmBarrier is required.");
                            for (var i = 0; i < bound.Length; i++)
                            {
                                var p = kernel.Parameters[i];
                                for (var j = i + 1; j < bound.Length; j++)
                                    if (bound[i].Slot == bound[j].Slot &&
                                        (p.Access == VmAccess.ReadWrite || kernel.Parameters[j].Access == VmAccess.ReadWrite) &&
                                        bound[i].Offset < bound[j].Offset + kernel.Parameters[j].Tensor.ByteLength &&
                                        bound[j].Offset < bound[i].Offset + p.Tensor.ByteLength)
                                        throw new NotSupportedException($"Dispatch '{node.Id}' aliases writable kernel parameters.");
                                if (p.Access == VmAccess.ReadWrite)
                                    pendingWrites.Add(bound[i].Slot);
                            }
                            commands.Add(new D3D12VmDispatch(kernel.Id, dispatch.Groups, bound));
                            break;
                        default:
                            throw new NotSupportedException($"Orchestration '{definition.Id}' contains an unsupported instruction.");
                    }
            }
            Walk(definitions[entry.Definition], entry.Arguments.ToDictionary(a => a.Parameter,
                a => new D3D12VmBinding(slots[a.Source], a.ByteOffset), StringComparer.Ordinal));
            result.Add(entry.Name, commands.ToArray());
        }
        return result;
    }
}
