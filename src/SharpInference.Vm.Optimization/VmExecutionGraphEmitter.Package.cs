using SharpInference.Graphs;
using SharpInference.Graphs.Core;
using SharpInference.Graphs.Execution;
using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization;

public sealed partial class VmExecutionGraphEmitter
{
    public VmProgram Emit(GraphPackage<ExecutionGraph> package, VmExecutionGraphEmitterOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(package);
        options ??= new();
        if (string.IsNullOrWhiteSpace(options.Abi) || string.IsNullOrWhiteSpace(options.Entry) ||
            string.IsNullOrWhiteSpace(options.StateSchema) || options.StateVersion <= 0)
        {
            throw new ArgumentException("VM artifact, entry and state identities must be nonempty and versioned.", nameof(options));
        }

        var source = package.Snapshot();
        ExecutionPackageCompilation.ValidatePlan(source);
        if (source.GetTarget() is not { } backend || ExecutionImplementationSelector.Target(backend) != instructionTarget)
        {
            throw new InvalidDataException("The execution package target does not match this VM emitter.");
        }

        implementationValidator.Validate(source, backend);
        var graphs = source.ToDictionary(graph => graph.Id, StringComparer.Ordinal);
        var definitions = graphs.ToDictionary(pair => pair.Key, pair => (SharpInference.Graphs.Graph)pair.Value, StringComparer.Ordinal);
        var frames = new Dictionary<string, PackageFrame>(StringComparer.Ordinal);
        var main = source.MainGraphId ?? throw new InvalidDataException("The main graph is missing.");
        bool Calls(string caller, string target, HashSet<string> visited) =>
            visited.Add(caller) && graphs[caller].Actions.OfType<ExecutionCall>()
                .Any(call => call.Target == target || Calls(call.Target, target, visited));

        static string SharedAddress(Resource resource) =>
            (resource.Scope == ResourceScope.Global ? "global:" : "session:") +
            (resource.BindingKey is { } key ? "binding:" + Encode(key)
                : "resource:" + Encode(resource.Graph.Id) + "/" + Encode(resource.Id));

        PackageFrame BuildFrame(string id)
        {
            if (frames.TryGetValue(id, out var existing))
            {
                return existing;
            }

            var graph = graphs[id];
            var frame = new PackageFrame(graph);
            foreach (var resource in graph.Resources)
            {
                if (!graph.Storage.TryGetValue(resource.Id, out var storage))
                {
                    throw new InvalidDataException($"Resource '{id}/{resource.Id}' has no selected storage.");
                }

                var address = resource.Scope switch
                {
                    ResourceScope.Call => $"local:{Encode(id)}/{Encode(ExecutionPackageCompilation.StorageSlot(graph, resource.Id))}",
                    ResourceScope.Session or ResourceScope.Global => SharedAddress(resource),
                    _ => throw new InvalidDataException($"Unknown scope at '{id}/{resource.Id}'."),
                };
                frame.Parameters.Add(new(resource.Id, resource, address));
                frame.Names.Add(resource.Id);
                if (resource.IsState)
                {
                    frame.StateNames.Add(resource.Id);
                }
            }
            foreach (var call in graph.Actions.OfType<ExecutionCall>())
            {
                var child = BuildFrame(call.Target);
                var arguments = new Dictionary<string, string>(StringComparer.Ordinal);
                frame.Calls.Add(call.Id, arguments);
                foreach (var parameter in child.Parameters)
                {
                    var binding = call.Bindings.SingleOrDefault(binding => binding.SourceProperty == parameter.Name);
                    if (binding is not null)
                    {
                        continue;
                    }

                    if (parameter.Ambient)
                    {
                        var local = parameter.Resource.Graph.Id == graph.Id
                            ? frame.Parameters.Single(item => item.Name == parameter.Resource.Id)
                            : frame.Parameters.SingleOrDefault(item => item.Ambient &&
                                item.Resource.Graph.Id == parameter.Resource.Graph.Id && item.Resource.Id == parameter.Resource.Id);
                        if (local is not null)
                        {
                            arguments.Add(parameter.Name, local.Name);
                            if (child.StateNames.Contains(parameter.Name))
                            {
                                frame.StateNames.Add(local.Name);
                            }

                            continue;
                        }
                    }
                    var name = frame.Allocate(call.Id + "/" + parameter.Name);
                    var address = parameter.Resource.Scope == ResourceScope.Call && !parameter.Ambient
                        ? $"call:{Encode(call.Id)}/{parameter.Address}" : parameter.Address;
                    frame.Parameters.Add(parameter with { Name = name, Address = address });
                    arguments.Add(parameter.Name, name);
                    if (child.StateNames.Contains(parameter.Name))
                    {
                        frame.StateNames.Add(name);
                    }
                }
            }
            frames.Add(id, frame);
            foreach (var call in graph.Actions.OfType<ExecutionCall>())
            {
                var child = frames[call.Target];
                foreach (var parameter in child.Parameters)
                {
                    var binding = call.Bindings.SingleOrDefault(binding => binding.SourceProperty == parameter.Name);
                    if (binding is not null)
                    {
                        var actual = Resolve(frame, binding.Target, []);
                        frame.Calls[call.Id].Add(parameter.Name, actual);
                        if (child.StateNames.Contains(parameter.Name))
                        {
                            frame.StateNames.Add(actual);
                        }
                    }
                }
            }
            foreach (var reference in graph.Actions.SelectMany(action => action.Bindings).Select(binding => binding.Target).Distinct())
            {
                _ = Resolve(frame, reference, []);
            }

            return frame;
        }

        string Resolve(PackageFrame frame, ResourceRef reference, HashSet<ResourceRef> visited)
        {
            if (!visited.Add(reference))
            {
                throw new InvalidDataException($"Resource alias cycle at '{frame.Graph.Id}/{reference}'.");
            }

            if (reference.TargetKind == ResourceTarget.Graph &&
                (reference.Target is null || reference.Target == frame.Graph.Id))
            {
                return frame.Parameters.SingleOrDefault(parameter => parameter.Name == reference.TargetProperty)?.Name
                    ?? throw new InvalidDataException($"Unknown resource '{frame.Graph.Id}/{reference.TargetProperty}'.");
            }

            if (reference.TargetKind == ResourceTarget.Call)
            {
                var owner = frame.Graph.Actions.OfType<ExecutionCall>().Single(call => call.Id == reference.Target);
                var actual = owner.Bindings.SingleOrDefault(binding => binding.SourceProperty == reference.TargetProperty);
                if (actual is not null)
                {
                    return Resolve(frame, actual.Target, visited);
                }

                return frame.Calls[owner.Id].GetValueOrDefault(reference.TargetProperty)
                    ?? throw new InvalidDataException($"Unknown instance resource '{owner.Id}/{reference.TargetProperty}'.");
            }
            if (!GraphValidation.TryResolve(frame.Graph, reference, definitions, out var resource))
            {
                throw new InvalidDataException($"Unknown resource '{reference}'.");
            }

            if (resource.Scope == ResourceScope.Call)
            {
                if (Calls(resource.Graph.Id, frame.Graph.Id, []))
                {
                    if (frame.External.TryGetValue(reference, out var ambient))
                    {
                        return ambient;
                    }

                    ambient = frame.Allocate(resource.Graph.Id + "/" + resource.Id);
                    frame.Parameters.Add(new(ambient, resource,
                        "ambient:" + Encode(resource.Graph.Id) + "/" + Encode(resource.Id), Ambient: true));
                    if (resource.IsState)
                    {
                        frame.StateNames.Add(ambient);
                    }

                    frame.External.Add(reference, ambient);
                    return ambient;
                }
                var candidates = frame.Graph.Actions.OfType<ExecutionCall>().SelectMany(call =>
                    frames[call.Target].Parameters.Where(parameter => parameter.Resource.Graph.Id == resource.Graph.Id &&
                        parameter.Resource.Id == resource.Id).Select(parameter =>
                    {
                        var binding = call.Bindings.SingleOrDefault(binding => binding.SourceProperty == parameter.Name);
                        return binding is null ? frame.Calls[call.Id][parameter.Name]
                            : Resolve(frame, binding.Target, new(visited));
                    })).GroupBy(name => frame.Parameters.Single(parameter => parameter.Name == name).Address)
                    .Select(group => group.First()).Take(2).ToArray();
                if (candidates.Length != 1)
                {
                    throw new InvalidDataException($"Resource '{resource.Graph.Id}/{resource.Id}' requires a unique active instance.");
                }

                return candidates[0];
            }
            if (frame.External.TryGetValue(reference, out var name))
            {
                return name;
            }

            name = frame.Allocate(resource.Graph.Id + "/" + resource.Id);
            var address = SharedAddress(resource);
            frame.Parameters.Add(new(name, resource, address));
            if (resource.IsState)
            {
                frame.StateNames.Add(name);
            }

            frame.External.Add(reference, name);
            return name;
        }

        var root = BuildFrame(main);
        var slots = new List<VmSlot>();
        var addresses = new Dictionary<string, VmSlot>(StringComparer.Ordinal);
        var slotNames = new HashSet<string>(StringComparer.Ordinal);
        var entryArguments = new List<VmArgument>();
        var state = new List<VmStateEntry>();
        var stateSlots = new HashSet<string>(StringComparer.Ordinal);
        foreach (var parameter in root.Parameters)
        {
            var resource = parameter.Resource;
            if (!addresses.TryGetValue(parameter.Address, out var slot))
            {
                var name = parameter.Name;
                for (var suffix = 1; !slotNames.Add(name); suffix++)
                {
                    name = parameter.Name + "." + suffix;
                }

                slot = new(name, resource.Scope switch
                {
                    ResourceScope.Call => VmSlotScope.Local,
                    ResourceScope.Session => VmSlotScope.Session,
                    ResourceScope.Global => VmSlotScope.Global,
                    _ => throw new InvalidDataException("Unknown resource scope."),
                }, ParameterAccess(resource), Tensor(resource.Tensor), resource.BindingKey);
                addresses.Add(parameter.Address, slot);
                slots.Add(slot);
            }
            else if (slot.Tensor.ElementType != Tensor(resource.Tensor).ElementType ||
                !slot.Tensor.Dimensions.SequenceEqual(resource.Tensor.Dimensions) || slot.Access != ParameterAccess(resource))
            {
                throw new InvalidDataException($"Shared storage '{parameter.Address}' has incompatible contracts.");
            }

            entryArguments.Add(new(parameter.Name, slot.Id));
            if (root.StateNames.Contains(parameter.Name) && stateSlots.Add(slot.Id))
            {
                state.Add(new(parameter.Address, slot.Id));
            }
        }

        var emitted = new List<VmDefinition>();
        foreach (var frame in frames.Values)
        {
            var parameters = frame.Parameters.Select(parameter =>
                new VmParameter(parameter.Name, ParameterAccess(parameter.Resource), Tensor(parameter.Resource.Tensor))).ToArray();
            var nodes = new List<VmNode>();
            var completion = new Dictionary<string, string>(StringComparer.Ordinal);
            var nodeNames = frame.Graph.Actions.Select(action => action.Id).ToHashSet(StringComparer.Ordinal);
            foreach (var action in frame.Graph.DependencyGraph().TopologicalOrder().Select(node => node.Action))
            {
                var dependencies = frame.Graph.Dependencies(action).Select(dependency => completion[dependency.Action]).Distinct().ToArray();
                if (action is ExecutionCall call)
                {
                    nodes.Add(new(call.Id, new VmCall("graph." + call.Target,
                        frames[call.Target].Parameters.Select(parameter =>
                            new VmArgument(parameter.Name, frame.Calls[call.Id][parameter.Name]))), dependencies));
                    completion.Add(call.Id, call.Id);
                    continue;
                }
                if (action is not ExecutionOperation operation || operation.SelectedImplementation is not { } implementation)
                {
                    throw new InvalidDataException($"Action '{frame.Graph.Id}/{action.Id}' has no selected implementation.");
                }

                if (implementation.Target != instructionTarget)
                {
                    throw new InvalidDataException($"Action '{frame.Graph.Id}/{action.Id}' targets a different backend.");
                }

                var arguments = action.Bindings.Select(binding =>
                    new VmArgument(binding.SourceProperty, Resolve(frame, binding.Target, []))).ToArray();
                var precision = operation.Precision ?? new PrecisionRequirement(
                    implementation.Precision.ArithmeticType, implementation.Precision.AccumulatorType);
                if (target == VmTarget.Cpu)
                {
                    nodes.Add(new(action.Id, new VmOperator(precision, implementation.CollectionId,
                        implementation.InstructionName, arguments, operation.Attributes,
                        resolvedPrecision: implementation.Precision), dependencies));
                    completion.Add(action.Id, action.Id);
                }
                else
                {
                    var dispatch = implementation.Dispatch
                        ?? throw new InvalidDataException($"GPU action '{action.Id}' has no dispatch configuration.");
                    var kernel = "kernel." + Encode(frame.Graph.Id) + "/" + Encode(action.Id);
                    var kernelParameters = action.Bindings.Select(binding =>
                    {
                        var parameter = frame.Parameters.Single(parameter => parameter.Name == Resolve(frame, binding.Target, []));
                        return new VmParameter(binding.SourceProperty, Access(binding.Access), Tensor(parameter.Resource.Tensor));
                    }).ToArray();
                    emitted.Add(new(kernel, VmDefinitionKind.Kernel, kernelParameters,
                        [new VmNode("body", new VmOperator(precision, implementation.CollectionId,
                            implementation.InstructionName, action.Bindings.Select(binding =>
                                new VmArgument(binding.SourceProperty, binding.SourceProperty)),
                            operation.Attributes, resolvedPrecision: implementation.Precision))],
                        new(dispatch.ThreadsX, dispatch.ThreadsY, dispatch.ThreadsZ)));
                    nodes.Add(new(action.Id, new VmDispatch(kernel, arguments,
                        new(dispatch.GroupsX, dispatch.GroupsY, dispatch.GroupsZ)), dependencies));
                    var writes = action.Bindings.Where(binding => binding.Access != ResourceAccess.Read)
                        .Select(binding => Resolve(frame, binding.Target, [])).Distinct().ToArray();
                    if (writes.Length > 0)
                    {
                        var barrier = action.Id + ".complete";
                        for (var suffix = 1; !nodeNames.Add(barrier); suffix++)
                        {
                            barrier = action.Id + ".complete." + suffix;
                        }

                        nodes.Add(new(barrier, new VmBarrier(writes), [action.Id]));
                        completion.Add(action.Id, barrier);
                    }
                    else
                    {
                        completion.Add(action.Id, action.Id);
                    }
                }
            }
            emitted.Add(new("graph." + frame.Graph.Id,
                target == VmTarget.Cpu ? VmDefinitionKind.Function : VmDefinitionKind.Orchestration, parameters, nodes));
        }
        var program = new VmProgram(options.Name ?? main, options.Abi, target, slots, emitted,
            [new VmEntry(options.Entry, "graph." + main, entryArguments)],
            new VmState(options.StateSchema, options.StateVersion, state));
        VmProgramValidator.Validate(program);
        return VmInstructionContracts.Bind(program, registry);
    }

    private static VmAccess ParameterAccess(Resource resource) =>
        resource.Permission == ResourcePermission.ReadOnly || resource.GraphIO == GraphIO.In
            ? VmAccess.ReadOnly : VmAccess.ReadWrite;

    private static string Encode(string value) => Uri.EscapeDataString(value);

    private sealed record PackageParameter(string Name, Resource Resource, string Address, bool Ambient = false);

    private sealed class PackageFrame(ExecutionGraph graph)
    {
        public ExecutionGraph Graph { get; } = graph;
        public List<PackageParameter> Parameters { get; } = [];
        public HashSet<string> Names { get; } = new(StringComparer.Ordinal);
        public HashSet<string> StateNames { get; } = new(StringComparer.Ordinal);
        public Dictionary<string, Dictionary<string, string>> Calls { get; } = new(StringComparer.Ordinal);
        public Dictionary<ResourceRef, string> External { get; } = [];

        public string Allocate(string candidate)
        {
            var name = candidate;
            for (var suffix = 1; !Names.Add(name); suffix++)
            {
                name = candidate + "." + suffix;
            }

            return name;
        }
    }
}
