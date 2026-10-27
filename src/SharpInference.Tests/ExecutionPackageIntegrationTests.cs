using System.Runtime.InteropServices;
using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Graphs.Execution;
using SharpInference.Graphs.Logical;
using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Instructions.D3D12;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;
using SharpInference.Vm.Optimization.GraphHelper;

namespace SharpInference.Tests;

public sealed class ExecutionPackageIntegrationTests
{
    private static readonly TensorDescriptor Vector = new(ElementType.Float32, [8]);
    private static readonly IInstructionCollectionProvider[] Cpu = [new CpuFloat32InstructionCollection()];

    [Fact]
    public void DirectCompilationPlansAndExecutesWithoutChangingTheEditableSource()
    {
        var graph = new ExecutionGraph { Id = "main" };
        var input = graph.Resource("input", Vector, GraphIO.In);
        var temporary = graph.Resource("temporary", Vector);
        var output = graph.Resource("output", Vector, GraphIO.Out);
        Copy(graph, "produce", new(input), new(temporary));
        Copy(graph, "consume", new(temporary), new(output));
        var package = new ExecutionGraphPackage(graph);

        var program = package.Compile(VmTarget.Cpu, Cpu);
        Assert.Equal(Values(), Execute(program));
        Assert.Empty(graph.GraphFlow);
        Assert.Empty(graph.Storage);
        Assert.All(graph.Actions.OfType<ExecutionOperation>(), operation => Assert.Null(operation.SelectedImplementation));
        var unplanned = package.Snapshot();
        foreach (var resource in unplanned.MainGraph!.Resources)
        {
            unplanned.MainGraph.Storage.Add(resource.Id, new(ResourceStorageKind.LocalBuffer));
        }

        new ExecutionImplementationSelector(Cpu).Select(unplanned, "cpu-vm");
        Assert.Throws<InvalidDataException>(() => new VmExecutionGraphEmitter(VmTarget.Cpu, Cpu).Emit(unplanned));
    }

    [Theory]
    [InlineData("extract")]
    [InlineData("inline")]
    [InlineData("fuse")]
    [InlineData("replace")]
    public void StructuralRewritesPreserveObservableCpuResults(string rewrite)
    {
        var graph = new ExecutionGraph { Id = "main" };
        var input = graph.Resource("input", Vector, GraphIO.In);
        var middle = graph.Resource("middle", Vector);
        var output = graph.Resource("output", Vector, GraphIO.Out);
        Copy(graph, "first", new(input), new(middle));
        Copy(graph, "second", new(middle), new(output));
        var source = new ExecutionGraphPackage(graph);
        ExecutionGraphPackage result;
        if (rewrite == "fuse")
        {
            result = ExecutionGraphRewriter.FuseNodesToOperation(source, "main", ["first", "second"], "fused",
                new(Operations.Copy), [new("input", new(input), ResourceAccess.Read), new("output", new(output), ResourceAccess.Write)]);
        }
        else
        {
            result = ExecutionGraphRewriter.ExtractNodesToGraph(source, "main", ["first", "second"], "extracted", "group");
            if (rewrite == "inline")
            {
                result = ExecutionGraphRewriter.InlineGraphCall(result, "main", "group");
            }
            else if (rewrite == "replace")
            {
                result = ExecutionGraphRewriter.ReplaceGraphCallWithOperation(result, "main", "group", new(Operations.Copy));
            }
        }
        Assert.Equal(Values(), Execute(result.Compile(VmTarget.Cpu, Cpu)));
        Assert.Equal(2, graph.Actions.Count);
        Assert.Empty(graph.Storage);
        Assert.Empty(graph.GraphFlow);
    }

    [Fact]
    public void NestedCallsForwardAncestorPrivateResourcesAndSummarizeInitializedScratch()
    {
        var main = new ExecutionGraph { Id = "main" };
        var input = main.Resource("input", Vector, GraphIO.In);
        var hidden = main.Resource("hidden", Vector);
        var output = main.Resource("output", Vector, GraphIO.Out);
        Copy(main, "prepare", new(input), new(hidden));
        var leaf = new ExecutionGraph { Id = "leaf" };
        var scratch = leaf.Resource("scratch", Vector);
        Copy(leaf, "read", new(hidden), new(scratch));
        Copy(leaf, "finish", new(scratch), new(output));
        var middle = new ExecutionGraph { Id = "middle" };
        middle.Call("inner", leaf);
        main.Call("outer", middle);
        var package = new ExecutionGraphPackage(main, middle, leaf);

        var program = package.Compile(VmTarget.Cpu, Cpu);
        Assert.Equal(3, program.Definitions.Count);
        Assert.Equal(Values(), Execute(program));
        Assert.Equal(["prepare"], program.Definitions.Single(definition => definition.Id == "graph.main")
            .Nodes.Single(node => node.Id == "outer").Dependencies);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DescendantPrivateResourcesResolveOnlyWhenTheirActiveInstanceIsUnambiguous(bool repeated)
    {
        var leaf = new ExecutionGraph { Id = "leaf" };
        var leafInput = leaf.Resource("input", Vector, GraphIO.In);
        var scratch = leaf.Resource("scratch", Vector);
        Copy(leaf, "produce", new(leafInput), new(scratch));
        var middle = new ExecutionGraph { Id = "middle" };
        var middleInput = middle.Resource("input", Vector, GraphIO.In);
        middle.Call("inner", leaf).Bind("input", middleInput, ResourceAccess.Read);
        var main = new ExecutionGraph { Id = "main" };
        var input = main.Resource("input", Vector, GraphIO.In);
        var output = main.Resource("output", Vector, GraphIO.Out);
        main.Call("outer", middle).Bind("input", input, ResourceAccess.Read);
        if (repeated)
        {
            main.Call("other", middle).Bind("input", input, ResourceAccess.Read);
        }

        Copy(main, "consume", new(scratch), new(output));
        var package = new ExecutionGraphPackage(main, middle, leaf);
        if (repeated)
        {
            Assert.Contains("A0006", Assert.Throws<InvalidDataException>(() => package.Compile(VmTarget.Cpu, Cpu)).Message);
        }
        else
        {
            Assert.Equal(Values(), Execute(package.Compile(VmTarget.Cpu, Cpu)));
        }
    }

    [Theory]
    [InlineData(ResourceScope.Global)]
    [InlineData(ResourceScope.Session)]
    public void SharedBindingAliasesParticipateInConflictAnalysis(ResourceScope scope)
    {
        var graph = new LogicalGraph { Id = "main" };
        var first = graph.Resource("first", Vector, scope: scope, bindingKey: "shared.value");
        var second = graph.Resource("second", Vector, scope: scope, bindingKey: "shared.value");
        var write = graph.Operation("write", Operations.Copy);
        write.Bind("output", first, ResourceAccess.Write);
        var read = graph.Operation("read", Operations.Copy);
        read.Bind("input", second, ResourceAccess.Read);
        var package = new LogicalGraphPackage(graph);
        Assert.Contains(package.Analyze(graph).Diagnostics, diagnostic => diagnostic.Code == "A0003");
        write.Then(read);
        Assert.Empty(package.Analyze(graph).Diagnostics);
        second.Permission = ResourcePermission.ReadOnly;
        Assert.Contains(package.Validate(), diagnostic => diagnostic.Code == "P0011");
    }

    [Fact]
    public void TemporaryStorageCannotAliasConcurrentLifetimes()
    {
        var graph = new ExecutionGraph { Id = "main" };
        var input = graph.Resource("input", Vector, GraphIO.In);
        var left = graph.Resource("left", Vector);
        var right = graph.Resource("right", Vector);
        var output = graph.Resource("output", Vector, GraphIO.Out);
        Copy(graph, "left", new(input), new(left));
        Copy(graph, "right", new(input), new(right));
        var add = graph.Operation("add", Operations.Add);
        add.Bind("left", left, ResourceAccess.Read);
        add.Bind("right", right, ResourceAccess.Read);
        add.Bind("output", output, ResourceAccess.Write);
        graph.Storage.Add(right.Id, new(ResourceStorageKind.LocalBuffer, left.Id));
        Assert.Throws<InvalidDataException>(() => new ExecutionGraphPackage(graph).Compile(VmTarget.Cpu, Cpu));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DeclaredWritesCannotHideAnImplementationRead(bool alreadySelected)
    {
        var graph = new ExecutionGraph { Id = "main" };
        var input = graph.Resource("input", Vector, GraphIO.InOut);
        var output = graph.Resource("output", Vector, GraphIO.Out);
        Copy(graph, "copy", new(input), new(output));
        var package = new ExecutionGraphPackage(graph);
        if (alreadySelected)
        {
            foreach (var resource in graph.Resources)
            {
                graph.Storage.Add(resource.Id, new(ResourceStorageKind.ExternalArgument));
            }

            new ExecutionImplementationSelector(Cpu).Select(package, "cpu-vm");
        }
        graph.Actions[0].Bindings[0] = graph.Actions[0].Bindings[0] with { Access = ResourceAccess.Write };
        if (alreadySelected)
        {
            Assert.Throws<InvalidDataException>(() => new VmExecutionGraphEmitter(VmTarget.Cpu, Cpu).Emit(package));
        }
        else
        {
            Assert.Throws<NotSupportedException>(() => package.Compile(VmTarget.Cpu, Cpu));
        }
    }

    [Fact]
    public void GpuCompletionNamesCannotCollideWithActionNames()
    {
        var graph = new ExecutionGraph { Id = "main" };
        var input = graph.Resource("input", Vector, GraphIO.In);
        var first = graph.Resource("first", Vector, GraphIO.Out);
        var second = graph.Resource("second", Vector, GraphIO.Out);
        Copy(graph, "copy", new(input), new(first));
        Copy(graph, "copy.complete", new(input), new(second));
        IInstructionCollectionProvider[] gpu = [new GpuFloat32InstructionCollection()];
        var program = new ExecutionGraphPackage(graph).Compile(VmTarget.Direct3D12, gpu);
        var nodes = program.Definitions.Single(definition => definition.Kind == VmDefinitionKind.Orchestration).Nodes;
        Assert.Equal(4, nodes.Select(node => node.Id).Distinct().Count());
        Assert.Contains(nodes, node => node.Id == "copy.complete.1");
        Assert.NotEmpty(new D3D12VmCompiler(gpu).GenerateSources(program));
    }

    [Fact]
    public void StateViewsAcrossCallsMarkTheActualSlotWithoutChangingItsLifetime()
    {
        var child = new ExecutionGraph { Id = "child" };
        child.Resource("value", Vector, GraphIO.InOut, isState: true);
        var main = new ExecutionGraph { Id = "main" };
        var value = main.Resource("value", Vector, GraphIO.InOut, ResourceScope.Session);
        main.Call("borrow", child).Bind("value", value, ResourceAccess.ReadWrite);
        var program = new ExecutionGraphPackage(main, child).Compile(VmTarget.Cpu, Cpu);
        Assert.Equal("value", Assert.Single(program.State.Entries).Slot);
        Assert.Equal(VmSlotScope.Session, Assert.Single(program.Slots).Scope);
        Assert.False(value.IsState);
    }

    [Fact]
    public void StateKeysDistinguishEqualBindingNamesInDifferentRuntimeScopes()
    {
        var graph = new ExecutionGraph { Id = "main" };
        graph.Resource("session", Vector, GraphIO.InOut, ResourceScope.Session, isState: true, bindingKey: "shared");
        graph.Resource("global", Vector, GraphIO.InOut, ResourceScope.Global, isState: true, bindingKey: "shared");
        var program = new ExecutionGraphPackage(graph).Compile(VmTarget.Cpu, Cpu);
        Assert.Equal(2, program.State.Entries.Select(entry => entry.Name).Distinct().Count());
        Assert.Contains(program.State.Entries, entry => entry.Name == "session:binding:shared");
        Assert.Contains(program.State.Entries, entry => entry.Name == "global:binding:shared");
    }

    [Fact]
    public void DependencyProjectionCapturesIdentitiesWhileInterfaceEnumerationRemainsLive()
    {
        var graph = new ExecutionGraph { Id = "main" };
        var resource = graph.Resource("value", Vector);
        var action = graph.Operation("before", Operations.Copy);
        var projection = graph.DependencyGraph();
        var inputs = graph.Inputs;
        action.Id = "after";
        resource.GraphIO = GraphIO.In;
        Assert.Equal("before", Assert.Single(projection.Nodes).Id.Value);
        Assert.Same(resource, Assert.Single(inputs));
    }

    [Fact]
    public void PackageInterfaceEnumeratorsFollowLaterMainGraphSelection()
    {
        var package = new ExecutionGraphPackage();
        var inputs = package.Inputs;
        var first = new ExecutionGraph { Id = "first" };
        first.Resource("first", Vector, GraphIO.In);
        var second = new ExecutionGraph { Id = "second" };
        var value = second.Resource("second", Vector, GraphIO.InOut);
        package.Add(first);
        package.Add(second);
        package.MainGraphId = "first";
        Assert.Equal("first", Assert.Single(inputs).Id);
        package.MainGraphId = "second";
        Assert.Same(value, Assert.Single(inputs));
        Assert.Same(value, Assert.Single(package.Outputs));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DescriptiveMetadataAndEmptyAttributesRoundTripWithoutReservedKeyLoss(bool logical)
    {
        if (logical)
        {
            var graph = new LogicalGraph { Id = "main" };
            graph.Operation("pending", "pending").Attributes.Add("empty", "");
            var package = new LogicalGraphPackage(graph);
            package.Metadata.Add("execution.backend", "descriptive");
            var loaded = LogicalGraphPackageXml.Deserialize(LogicalGraphPackageXml.Serialize(package));
            Assert.Equal("descriptive", loaded.Metadata["execution.backend"]);
            Assert.Equal("", Assert.IsType<OperationAction>(loaded.MainGraph!.Actions[0]).Attributes["empty"]);
        }
        else
        {
            CheckMetadata(new Graph { Id = "main" });
        }
    }

    private static void CheckMetadata(Graph graph)
    {
        graph.Operation("pending", "pending").Attributes.Add("empty", "");
        var package = new GraphPackage<Graph>(graph);
        package.Metadata.Add("execution.backend", "descriptive");
        var loaded = GraphPackageXml.Deserialize<Graph>(GraphPackageXml.Serialize(package));
        Assert.Equal("descriptive", loaded.Metadata["execution.backend"]);
        Assert.Equal("", Assert.IsType<OperationAction>(loaded.MainGraph!.Actions[0]).Attributes["empty"]);
    }

    private static void Copy(ExecutionGraph graph, string id, ResourceRef input, ResourceRef output)
    {
        var copy = graph.Operation(id, Operations.Copy);
        copy.Bind("input", input, ResourceAccess.Read);
        copy.Bind("output", output, ResourceAccess.Write);
    }

    private static float[] Values() => [1, 2, 3, 4, 5, 6, 7, 8];

    private static float[] Execute(VmProgram program)
    {
        var artifact = new CpuVmCompiler(Cpu).Compile(program);
        using var executor = artifact.CreateExecutor();
        var buffers = program.Slots.Select(slot => new byte[checked((int)slot.Tensor.ByteLength)]).ToArray();
        var input = program.Slots.ToList().FindIndex(slot => slot.Id == "input");
        var output = program.Slots.ToList().FindIndex(slot => slot.Id == "output");
        Values().AsSpan().CopyTo(MemoryMarshal.Cast<byte, float>(buffers[input].AsSpan()));
        executor.Execute("forward", buffers);
        return MemoryMarshal.Cast<byte, float>(buffers[output]).ToArray();
    }
}
