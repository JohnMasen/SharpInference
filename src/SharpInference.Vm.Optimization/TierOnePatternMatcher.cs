using SharpInference.Graphs;
using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization;

public sealed class TierOneFusionCandidate
{
    internal TierOneFusionCandidate(InstructionOptimizationCapability capability, IReadOnlyList<int> nodes,
        IReadOnlyList<ResourceId> inputs, ResourceId output, IReadOnlyList<ResourceId> intermediates,
        IReadOnlyDictionary<ResourceId, ResourceId>? aliases)
    {
        Capability = capability;
        Nodes = Array.AsReadOnly(nodes.Order().ToArray());
        Inputs = Array.AsReadOnly(inputs.ToArray());
        Output = output;
        Intermediates = Array.AsReadOnly(intermediates.ToArray());
        var identities = new Dictionary<ResourceId, int>();
        InputAliases = string.Join(",", inputs.Select(input => aliases?.GetValueOrDefault(input, input) ?? input).Select(input =>
        {
            if (!identities.TryGetValue(input, out var index)) identities.Add(input, index = identities.Count);
            return index;
        }));
    }

    public InstructionOptimizationCapability Capability { get; }
    public IReadOnlyList<int> Nodes { get; }
    public IReadOnlyList<ResourceId> Inputs { get; }
    public ResourceId Output { get; }
    public IReadOnlyList<ResourceId> Intermediates { get; }
    public string InputAliases { get; }
}

public static class TierOnePatternMatcher
{
    public static IReadOnlyList<TierOneFusionCandidate> Find(ExecutionGraph graph,
        IReadOnlyList<InstructionOptimizationCapability> capabilities, VmTarget target,
        IReadOnlyDictionary<ResourceId, ResourceId>? aliases = null)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(capabilities);
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        var reaching = new Dictionary<ResourceId, int>();
        var reads = new Dictionary<int, Value[]>();
        var consumers = new Dictionary<Value, HashSet<int>>();
        var writes = new Dictionary<ResourceId, int>();
        var barriers = new HashSet<int>();
        var endpoints = graph.Inputs.Concat(graph.Outputs).ToHashSet();
        for (var index = 0; index < graph.Nodes.Count; index++)
        {
            var node = graph.Nodes[index];
            var bindings = node.Resources.Where(binding => binding.Access == GraphResourceAccess.Read)
                .OrderBy(binding => binding.Port == "input" || binding.Port == "left" ? 0 : 1).ToArray();
            reads[index] = bindings.Select(binding =>
                new Value(binding.Resource, reaching.GetValueOrDefault(binding.Resource, -1))).ToArray();
            foreach (var value in reads[index])
            {
                if (!consumers.TryGetValue(value, out var uses)) consumers.Add(value, uses = []);
                uses.Add(index);
            }
            foreach (var binding in node.Resources.Where(binding => binding.Access != GraphResourceAccess.Read))
            {
                reaching[binding.Resource] = index;
                writes[binding.Resource] = writes.GetValueOrDefault(binding.Resource) + 1;
                if (resources[binding.Resource].Scope != GraphResourceScope.Local) barriers.Add(index);
            }
            if (node.Operation == PrimitiveGraphOperations.Copy) barriers.Add(index);
        }
        var result = new List<TierOneFusionCandidate>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var capability in capabilities.Where(capability => capability.Target ==
                     (target == VmTarget.Cpu ? InstructionTarget.Cpu : InstructionTarget.Direct3D12)))
        {
            var definition = capability.Definition;
            for (var root = 0; root < graph.Nodes.Count; root++)
                foreach (var state in Match(definition.Stages.Count - 1, root, new MatchState()))
                {
                    if (state.Steps.Count != definition.Stages.Count ||
                        state.Inputs.Count != definition.Inputs.Count ||
                        state.Steps.Values.Distinct().Count() != definition.Stages.Count)
                        continue;
                    var selected = state.Steps.Values.ToHashSet();
                    var node = graph.Nodes[root];
                    var output = node.Resources.Single(binding => binding.Access == GraphResourceAccess.Write).Resource;
                    var intermediates = selected.Where(index => index != root).Select(index =>
                        graph.Nodes[index].Resources.Single(binding => binding.Access == GraphResourceAccess.Write).Resource).ToArray();
                    if (barriers.Any(index => index >= selected.Min() && index <= selected.Max()) ||
                        selected.Any(index => graph.Nodes[index].Region != node.Region) ||
                        intermediates.Any(resource => endpoints.Contains(resource) ||
                            resources[resource].Scope != GraphResourceScope.Local || writes[resource] != 1) ||
                        selected.Where(index => index != root).Any(index =>
                            consumers.GetValueOrDefault(new Value(graph.Nodes[index].Resources
                                .Single(binding => binding.Access == GraphResourceAccess.Write).Resource, index), [])
                                .Any(consumer => !selected.Contains(consumer))) ||
                        state.Inputs.Values.Any(value => selected.Contains(value.Producer)) ||
                        state.Inputs.Values.GroupBy(value => value.Resource).Any(group => group.Distinct().Count() != 1))
                        continue;
                    var inputs = Enumerable.Range(0, definition.Inputs.Count).Select(index => state.Inputs[index].Resource).ToArray();
                    var descriptions = inputs.Select((input, index) => new InstructionOperandDescription(definition.Inputs[index],
                            resources[input].Tensor, GraphResourceAccess.Read, input.Value))
                        .Append(new("output", resources[output].Tensor, GraphResourceAccess.Write, output.Value)).ToArray();
                    if (!capability.TryAdapt(descriptions, node.Requirements, out _)) continue;
                    var key = $"{capability.Name}|{capability.Configuration.Implementation.Value}|{string.Join(",", selected.Order())}";
                    if (seen.Add(key))
                        result.Add(new(capability, selected.Order().ToArray(), inputs, output, intermediates, aliases));
                }

            IEnumerable<MatchState> Match(int step, int index, MatchState current)
            {
                if (index < 0) yield break;
                if (current.Steps.TryGetValue(step, out var assigned))
                {
                    if (assigned == index) yield return current;
                    yield break;
                }
                var node = graph.Nodes[index];
                var stage = definition.Stages[step];
                var operands = reads[index];
                if (stage.Operation == PrimitiveGraphOperations.Square && node.Operation == PrimitiveGraphOperations.Multiply &&
                    operands.Length == 2 && operands[0] == operands[1])
                    operands = [operands[0]];
                else if (node.Operation != stage.Operation) yield break;
                if (node.Attributes.Count != 0 || operands.Length != stage.Operands.Count ||
                    node.Resources.Count(binding => binding.Access == GraphResourceAccess.Write) != 1 ||
                    node.Resources.Any(binding => binding.Access == GraphResourceAccess.ReadWrite) ||
                    !NumericTypeCompatibility.Satisfies(capability.Precision, node.Requirements) ||
                    node.Resources.Any(binding => resources[binding.Resource].Tensor.ElementType != GraphElementType.Float32))
                    yield break;
                var orders = new List<Value[]> { operands };
                if (operands.Length == 2 && (stage.Operation == PrimitiveGraphOperations.Add ||
                    stage.Operation == PrimitiveGraphOperations.Multiply) && operands[0] != operands[1])
                    orders.Add([operands[1], operands[0]]);
                foreach (var order in orders)
                {
                    var seed = current.Copy();
                    seed.Steps.Add(step, index);
                    IEnumerable<MatchState> states = [seed];
                    for (var port = 0; port < order.Length; port++)
                    {
                        var operand = stage.Operands[port];
                        var value = order[port];
                        states = states.SelectMany(state => Bind(state, operand, value)).ToArray();
                    }
                    foreach (var state in states) yield return state;
                }
            }

            IEnumerable<MatchState> Bind(MatchState state, TierOneOperand operand, Value value)
            {
                if (!operand.IsInput)
                    return Match(operand.Index, value.Producer, state);
                if (state.Inputs.TryGetValue(operand.Index, out var assigned))
                    return assigned == value ? [state] : [];
                var next = state.Copy();
                next.Inputs.Add(operand.Index, value);
                return [next];
            }
        }
        return Array.AsReadOnly(result.ToArray());
    }

    private readonly record struct Value(ResourceId Resource, int Producer);
    private sealed class MatchState
    {
        public Dictionary<int, int> Steps { get; } = [];
        public Dictionary<int, Value> Inputs { get; } = [];
        public MatchState Copy()
        {
            var result = new MatchState();
            foreach (var pair in Steps) result.Steps.Add(pair.Key, pair.Value);
            foreach (var pair in Inputs) result.Inputs.Add(pair.Key, pair.Value);
            return result;
        }
    }
}
