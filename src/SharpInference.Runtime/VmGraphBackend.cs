using System.Runtime.InteropServices;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Runtime;

public interface IVmGenerationScopeExecutor
{
    ValueTask<IRwkvGenerationScope> BeginGenerationAsync(CancellationToken cancellationToken);
}

public interface IVmPrefillExecutor
{
    ValueTask<float[]> PrefillAsync(ReadOnlyMemory<int> tokens, CancellationToken cancellationToken);
}

public interface IVmSessionExecutor : IProcessorSessionExecutor, IProcessorStateExecutor,
    IVmGenerationScopeExecutor, IVmPrefillExecutor
{
}

public sealed class VmCompiledPlan
{
    internal VmCompiledPlan(ExecutionGraph bindingGraph, VmProgram program,
        Func<IVmExecutable> createExecutable, Action<string>? exportArtifact = null,
        VmCompiledPlan? prefillPlan = null)
    {
        BindingGraph = bindingGraph;
        Program = program;
        CreateExecutable = createExecutable;
        ExportArtifact = exportArtifact;
        PrefillPlan = prefillPlan;
    }

    internal ExecutionGraph BindingGraph { get; }
    public VmProgram Program { get; }
    public Func<IVmExecutable> CreateExecutable { get; }
    public Action<string>? ExportArtifact { get; }
    public VmCompiledPlan? PrefillPlan { get; }
}

public sealed class VmGraphBackend : IDisposable
{
    private readonly VmTarget target;
    private readonly Func<VmProgram, Func<IVmExecutable>> compile;
    private readonly VmEngineOptions options;
    private readonly VmOptimizationOptions optimization;
    private readonly Action? disposeCompiler;
    private readonly VmProgram? suppliedProgram;
    private readonly Action<string>? exportArtifact;
    private readonly VmProgram? suppliedPrefillProgram;
    private readonly Func<VmProgram, Func<IVmExecutable>>? compilePrefill;
    private readonly Action<string>? exportPrefillArtifact;
    private readonly Func<VmSlot, IVmStorage>? allocate;
    private VmCompiledPlan? plan;
    private VmInferenceEngine? engine;
    private bool disposed;
    private readonly VmExecutionGraphGenerator? generator;

    public VmGraphBackend(VmTarget target, Func<VmProgram, Func<IVmExecutable>> compile,
        VmEngineOptions? options = null, VmOptimizationOptions? optimization = null, Action? disposeCompiler = null,
        VmProgram? suppliedProgram = null, Action<string>? exportArtifact = null,
        VmProgram? suppliedPrefillProgram = null, Func<VmProgram, Func<IVmExecutable>>? compilePrefill = null,
        Action<string>? exportPrefillArtifact = null, Func<VmSlot, IVmStorage>? allocate = null,
        string? deviceName = null, IEnumerable<IInstructionCollectionProvider>? generatorCollections = null)
    {
        this.target = target;
        this.compile = compile;
        this.options = options ?? new VmEngineOptions(16, 16);
        this.optimization = optimization ?? new VmOptimizationOptions();
        this.disposeCompiler = disposeCompiler;
        this.suppliedProgram = suppliedProgram;
        this.exportArtifact = exportArtifact;
        this.suppliedPrefillProgram = suppliedPrefillProgram;
        this.compilePrefill = compilePrefill;
        this.exportPrefillArtifact = exportPrefillArtifact;
        this.allocate = allocate;
        DeviceName = deviceName ?? target.ToString();
        var architecture = target == VmTarget.Cpu ? InstructionTarget.Cpu : InstructionTarget.Direct3D12;
        generator = generatorCollections is null ? null : new VmExecutionGraphGenerator(architecture, generatorCollections);
        KernelCatalog = new ExecutionKernelCatalog($"vm.{target.ToString().ToLowerInvariant()}",
            PrimitiveGraphOperations.CreateStandardDescriptions(false).Select(description => description.Operation)
                .Concat(PortableTensorOperationContracts.Contracts.Select(contract => contract.Operation)));
    }

    private IExecutionKernelCatalog KernelCatalog { get; }
    public VmTarget Target => target;
    public string DeviceName { get; }
    public VmProgram? Program => plan?.Program;
    public VmProgram? PrefillProgram => plan?.PrefillPlan?.Program ?? plan?.Program;

    public VmCompiledPlan Prepare(LogicalGraph logical, GraphOptimizationOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(logical);
        ObjectDisposedException.ThrowIf(disposed, this);
        if (engine is not null) throw new InvalidOperationException("An initialized backend cannot change programs.");
        var graph = new GraphOptimizer().Optimize(logical,
            options ?? new GraphOptimizationOptions(OptimizationBoundary.Unrestricted), KernelCatalog);
        var program = suppliedProgram ?? (generator ??
            throw new InvalidOperationException("Supply a graph generator IC catalog or a pre-generated execution program."))
            .Generate(logical, optimization);
        ValidateProgram(program, graph, prefill: false);
        var factory = compile(program);
        VmCompiledPlan? prefillPlan = null;
        if (suppliedPrefillProgram is { } prefillProgram)
        {
            ValidateProgram(prefillProgram, graph, prefill: true);
            prefillPlan = new VmCompiledPlan(graph, prefillProgram,
                (compilePrefill ?? compile)(prefillProgram), exportPrefillArtifact);
        }
        else ValidateProgram(program, graph, prefill: true);
        plan = new VmCompiledPlan(graph, program, factory, exportArtifact, prefillPlan);
        return plan;
    }

    private void ValidateProgram(VmProgram program, ExecutionGraph graph, bool prefill)
    {
        VmProgramValidator.Validate(program);
        if (program.Target != target || program.Abi != $"vm:{graph.Model.StateAbiId}" ||
            program.State.Schema != graph.GraphState.Schema.Name ||
            program.State.Version != 1 ||
            program.State.Entries.Count != graph.GraphState.Entries.Count)
            throw new InvalidDataException("The supplied VM program does not match this model's target/State ABI.");
        foreach (var entry in graph.GraphState.Entries)
        {
            var expected = graph.Resources.Single(resource => resource.Id == entry.Resource);
            var actualEntry = program.State.Entries.SingleOrDefault(candidate => candidate.Name == entry.Name)
                ?? throw new InvalidDataException($"The VM program omitted State entry '{entry.Name}'.");
            var actual = program.Slots.Single(slot => slot.Id == actualEntry.Slot);
            if (actual.Tensor.ElementType != VmElementType.Float32 ||
                !actual.Tensor.Dimensions.SequenceEqual(expected.Tensor.Dimensions))
                throw new InvalidDataException($"The supplied State entry '{entry.Name}' has incompatible storage.");
        }
        var input = program.Slots.SingleOrDefault(slot => slot.Id == graph.Inputs.Single().Value);
        var output = program.Slots.SingleOrDefault(slot => slot.Id == graph.Outputs.Single().Value);
        if (input is null || output is null || input.Tensor.ElementType != VmElementType.Int32 ||
            output.Tensor.ElementType != VmElementType.Float32 ||
            output.Tensor.ElementCount != (ulong)graph.Model.VocabularySize ||
            !program.Entries.Any(entry => entry.Name == (prefill ? "prefill.1" : "forward")))
            throw new InvalidDataException("The supplied VM program has an incompatible token/logits/entry contract.");
    }

    public void PrepareModelWeights(PortableGraphModel model)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (engine is not null) throw new InvalidOperationException("This backend already has a bound model.");
        var prepared = plan ?? throw new InvalidOperationException("Prepare the VM program before binding weights.");
        if (!ReferenceEquals(model.Graph, prepared.BindingGraph))
            throw new ArgumentException("The model does not match the prepared graph.", nameof(model));
        var input = prepared.BindingGraph.Inputs.Single().Value;
        var output = prepared.BindingGraph.Outputs.Single().Value;
        var prefill = prepared.PrefillPlan ?? prepared;
        engine = VmInferenceEngine.CreateAsync(new(prefill.Program, input, output, prefill.CreateExecutable),
            new(prepared.Program, input, output, prepared.CreateExecutable),
            (slot, storage) => VmModelBindings.InitializeGlobal(model.Tensors, slot, storage),
            options, allocate).AsTask().GetAwaiter().GetResult();
    }

    public IVmSessionExecutor CreateSessionExecutor(IRwkvModel model, IRwkvState state, VmCompiledPlan prepared)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (model is not PortableGraphModel portable || state is not PortableGraphState graphState ||
            !ReferenceEquals(prepared, plan) ||
            !ReferenceEquals(portable.Graph, prepared.BindingGraph))
            throw new ArgumentException("The model, state and VM program do not match.");
        var initialized = engine ?? throw new InvalidOperationException("Bind weights before creating sessions.");
        return new Session(initialized.CreateSession(), graphState, prepared.BindingGraph);
    }

    private sealed class Session : IVmSessionExecutor
    {
        private readonly VmInferenceSession session;
        private readonly PortableGraphState state;
        private readonly ExecutionGraph graph;
        private long revision;
        private bool disposed;

        public Session(VmInferenceSession session, PortableGraphState state, ExecutionGraph graph)
        {
            this.session = session;
            this.state = state;
            this.graph = graph;
            try
            {
                PublishViews();
                state.AttachDeviceSynchronizer(views =>
                {
                    var values = session.ReadStateAsync().AsTask().GetAwaiter().GetResult();
                    foreach (var view in views)
                        MemoryMarshal.Cast<byte, float>(values[view.Name].AsSpan()).CopyTo(view.Values);
                });
            }
            catch (Exception error)
            {
                try { session.DisposeAsync().AsTask().GetAwaiter().GetResult(); }
                catch (Exception cleanup) { throw new AggregateException("VM session initialization and cleanup failed.", error, cleanup); }
                throw;
            }
        }

        private void PublishViews()
        {
            var values = state.Views.ToDictionary(view => view.Name,
                view => MemoryMarshal.AsBytes(view.Values.AsSpan()).ToArray(), StringComparer.Ordinal);
            session.WriteStateAsync(values).AsTask().GetAwaiter().GetResult();
            revision = state.Revision;
        }

        private void PrepareState()
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (revision != state.Revision) PublishViews();
        }

        public void ForwardToken(int token, Span<float> logits)
        {
            PrepareState();
            session.ForwardAsync(token).AsTask().GetAwaiter().GetResult().CopyTo(logits);
            state.MarkDeviceModified();
        }

        public void ForwardTokens(ReadOnlySpan<int> tokens, Span<float> logits)
        {
            PrefillAsync(tokens.ToArray(), CancellationToken.None).AsTask().GetAwaiter().GetResult().CopyTo(logits);
        }

        public async ValueTask<float[]> PrefillAsync(ReadOnlyMemory<int> tokens, CancellationToken cancellationToken)
        {
            PrepareState();
            var logits = await session.PrefillAsync(tokens, cancellationToken).ConfigureAwait(false);
            state.MarkDeviceModified();
            return logits;
        }

        public IReadOnlyList<GraphStateValue> ReadState(GraphState schema)
        {
            PrepareState();
            if (!schema.Schema.IsCompatibleWith(graph.GraphState.Schema))
                throw new InvalidDataException("State schema does not match the VM session.");
            var values = session.ReadStateAsync().AsTask().GetAwaiter().GetResult();
            return graph.GraphState.Entries.Select(entry =>
            {
                var resource = graph.Resources.Single(resource => resource.Id == entry.Resource);
                return new GraphStateValue(entry.Name, resource.Tensor.Dimensions,
                    MemoryMarshal.Cast<byte, float>(values[entry.Name].AsSpan()).ToArray());
            }).ToArray();
        }

        public void WriteState(GraphState schema, IReadOnlyList<GraphStateValue> values)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (!schema.Schema.IsCompatibleWith(graph.GraphState.Schema))
                throw new InvalidDataException("State schema does not match the VM session.");
            GraphSessionStateAccess.Write(schema, graph.Resources, state, values);
            PublishViews();
        }

        public async ValueTask<IRwkvGenerationScope> BeginGenerationAsync(CancellationToken cancellationToken)
        {
            PrepareState();
            var lease = await session.BeginGenerationAsync(cancellationToken).ConfigureAwait(false);
            return new Scope(lease, state);
        }

        private sealed class Scope(VmGenerationLease lease, PortableGraphState state) :
            IRwkvGenerationScope, IRwkvGenerationSession
        {
            public IRwkvGenerationSession Session => this;
            public ReadOnlyMemory<float> ForwardToken(int token)
            {
                var logits = lease.ForwardToken(token);
                state.MarkDeviceModified();
                return logits;
            }
            public ReadOnlyMemory<float> Prefill(ReadOnlySpan<int> tokens) =>
                throw new InvalidOperationException("Prefill cannot run while this session owns an inference lease.");
            public ValueTask DisposeAsync() => lease.DisposeAsync();
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
            try { session.DisposeAsync().AsTask().GetAwaiter().GetResult(); }
            finally { state.DetachDeviceSynchronizer(synchronize: false); }
        }
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        List<Exception>? errors = null;
        try { engine?.DisposeAsync().AsTask().GetAwaiter().GetResult(); }
        catch (Exception error) { (errors ??= []).Add(error); }
        try { disposeCompiler?.Invoke(); }
        catch (Exception error) { (errors ??= []).Add(error); }
        if (errors is not null) throw new AggregateException("Compiled VM backend shutdown failed.", errors);
    }
}
