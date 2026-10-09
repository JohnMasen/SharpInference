using System.Runtime.InteropServices;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Runtime;

public interface IVmGenerationScopeExecutor
{
    ValueTask<ITokenGenerationScope> BeginGenerationAsync(CancellationToken cancellationToken);
}

public interface IVmPrefillExecutor
{
    ValueTask<float[]> PrefillAsync(ReadOnlyMemory<int> tokens, CancellationToken cancellationToken);
}

public interface IVmAsyncInferenceExecutor
{
    ValueTask<float[]> ForwardTokenAsync(int token, CancellationToken cancellationToken);
}

public interface IVmSessionExecutor : IProcessorSessionExecutor, IProcessorStateExecutor,
    IVmGenerationScopeExecutor, IVmPrefillExecutor, IVmAsyncInferenceExecutor
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
    private VmResourceManager? tensorResources;
    private IModelTensorCatalog? boundTensors;
    private bool modelBound;
    private bool disposed;
    private readonly VmExecutionGraphGenerator? generator;
    private readonly IGraphOperationLowerer? operationLowerer;
    private readonly HashSet<VmTensorGraphSession> graphSessions = [];

    public VmGraphBackend(VmTarget target, Func<VmProgram, Func<IVmExecutable>> compile,
        VmEngineOptions? options = null, VmOptimizationOptions? optimization = null, Action? disposeCompiler = null,
        VmProgram? suppliedProgram = null, Action<string>? exportArtifact = null,
        VmProgram? suppliedPrefillProgram = null, Func<VmProgram, Func<IVmExecutable>>? compilePrefill = null,
        Action<string>? exportPrefillArtifact = null, Func<VmSlot, IVmStorage>? allocate = null,
        string? deviceName = null, IEnumerable<IInstructionCollectionProvider>? generatorCollections = null,
        IVmBackendDiagnostics? diagnostics = null, IGraphOperationLowerer? operationLowerer = null,
        ProcessorExecutionCapabilities? executionCapabilities = null)
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
        Diagnostics = diagnostics;
        ExecutionCapabilities = executionCapabilities ?? new(1, 1, new HashSet<string>());
        this.operationLowerer = operationLowerer;
        DeviceName = deviceName ?? target.ToString();
        var architecture = target switch
        {
            VmTarget.Cpu => InstructionTarget.Cpu,
            VmTarget.Direct3D12 => InstructionTarget.Direct3D12,
            _ => throw new NotSupportedException($"No VM backend target adapter is installed for '{target}'."),
        };
        generator = generatorCollections is null ? null : new VmExecutionGraphGenerator(architecture, generatorCollections);
        KernelCatalog = new ExecutionKernelCatalog($"vm.{target.ToString().ToLowerInvariant()}",
            TierZeroOperationContracts.Contracts.Select(contract => contract.Operation)
                .Concat((generator?.InstructionCollections as IGraphInstructionProvider)?.QueryGraphInstructionBindings()
                    .Where(binding => binding.Target == architecture).Select(binding => binding.Operation) ?? []));
    }

    private IExecutionKernelCatalog KernelCatalog { get; }
    public VmTarget Target => target;
    public string DeviceName { get; }
    public TierOneOptimizationReport? OptimizationReport => generator?.LastOptimizationReport;
    public GpuMatVecOptimizationReport? MatVecOptimizationReport => generator?.LastMatVecOptimizationReport;
    public IVmBackendDiagnostics? Diagnostics { get; }
    public ProcessorExecutionCapabilities ExecutionCapabilities { get; }
    public VmProgram? Program => plan?.Program;
    public VmProgram? PrefillProgram => plan?.PrefillPlan?.Program ?? plan?.Program;
    public bool SupportsTokenSessions => plan is { } prepared &&
        prepared.BindingGraph.Model.Attributes.GetValueOrDefault("execution.token-prefill") == "true" &&
        prepared.BindingGraph.Inputs.Count == 1 && prepared.BindingGraph.Outputs.Count == 1 &&
        prepared.BindingGraph.Resources.Single(resource => resource.Id == prepared.BindingGraph.Inputs[0])
            .Tensor is { ElementType: GraphElementType.Int32, Layout: "dense" } input &&
        input.Dimensions.Aggregate(1L, (count, size) => checked(count * size)) == 1 &&
        prepared.BindingGraph.Resources.Single(resource => resource.Id == prepared.BindingGraph.Outputs[0])
            .Tensor is { ElementType: GraphElementType.Float32, Layout: "dense" } &&
        prepared.BindingGraph.Resources.Where(resource => resource.Kind == GraphResourceKind.SessionState)
            .All(resource => resource.Tensor.ElementType == GraphElementType.Float32 && resource.Tensor.Layout == "dense") &&
        (prepared.PrefillPlan ?? prepared).Program.Entries.Any(entry => entry.Name == "prefill.1");

    public VmCompiledPlan Prepare(LogicalGraph logical, GraphOptimizationOptions? options = null,
        IModelGraphModule? modelModule = null)
    {
        ArgumentNullException.ThrowIfNull(logical);
        lock (graphSessions)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (modelBound) throw new InvalidOperationException("An initialized backend cannot change programs.");
            if (graphSessions.Count != 0)
                throw new InvalidOperationException("A backend with active graph sessions cannot change programs.");
            GraphValidator.Validate(logical, modelModule?.OperationValidator);
            logical = GraphOperationLowering.Apply(logical, modelModule?.OperationLowerer);
            logical = GraphOperationLowering.Apply(logical, operationLowerer);
            var defaultOptions = logical.Nodes.Any(node => node.Resources.Any(binding => binding.View is not null))
                ? new GraphOptimizationOptions(OptimizationBoundary.Off, DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded)
                : new GraphOptimizationOptions(OptimizationBoundary.Unrestricted);
            var graph = new GraphOptimizer().Optimize(logical,
                options ?? defaultOptions, KernelCatalog);
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
    }

    private void ValidateProgram(VmProgram program, ExecutionGraph graph, bool prefill)
    {
        VmProgramValidator.Validate(program);
        VmGraphBindingValidator.Validate(program, graph, allowTokenPrefill: true);
        if (program.Target != target || program.Abi != $"vm:{graph.Model.StateAbiId}" ||
            program.State.Schema != graph.GraphState.Schema.Name ||
            program.State.Version != 1 ||
            program.State.Entries.Count != graph.GraphState.Entries.Count)
            throw new InvalidDataException("The supplied VM program does not match this model's target/State ABI.");
        if (!program.Entries.Any(entry => entry.Name == "forward"))
            throw new InvalidDataException("The supplied VM program has no graph forward entry.");
    }

    public void PrepareModelWeights(PortableGraphModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        lock (graphSessions) PrepareModelWeightsCore(model);
    }

    private void PrepareModelWeightsCore(PortableGraphModel model)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (modelBound) throw new InvalidOperationException("This backend already has a bound model.");
        var prepared = plan ?? throw new InvalidOperationException("Prepare the VM program before binding weights.");
        if (!ReferenceEquals(model.Graph, prepared.BindingGraph))
            throw new ArgumentException("The model does not match the prepared graph.", nameof(model));
        if (!SupportsTokenSessions)
        {
            var resources = new VmResourceManager((slot, storage) => VmModelBindings.InitializeGlobal(model.Tensors, slot, storage), allocate);
            try { resources.PrepareGlobals(prepared.Program); }
            catch (Exception error)
            {
                try { resources.Dispose(); }
                catch (Exception cleanup) { throw new AggregateException("Graph weight preparation and cleanup failed.", error, cleanup); }
                throw;
            }
            tensorResources = resources;
            boundTensors = model.Tensors;
            modelBound = true;
            return;
        }
        var input = prepared.BindingGraph.Inputs.Single().Value;
        var output = prepared.BindingGraph.Outputs.Single().Value;
        var prefill = prepared.PrefillPlan ?? prepared;
        engine = VmInferenceEngine.CreateAsync(new(prefill.Program, input, output, prefill.CreateExecutable),
            new(prepared.Program, input, output, prepared.CreateExecutable),
            (slot, storage) => VmModelBindings.InitializeGlobal(model.Tensors, slot, storage),
            options, allocate).AsTask().GetAwaiter().GetResult();
        boundTensors = model.Tensors;
        modelBound = true;
    }

    public IVmSessionExecutor CreateSessionExecutor(IModel model, IModelState state, VmCompiledPlan prepared)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (model is not PortableGraphModel portable || state is not PortableGraphState graphState ||
            !ReferenceEquals(prepared, plan) ||
            !ReferenceEquals(portable.Graph, prepared.BindingGraph))
            throw new ArgumentException("The model, state and VM program do not match.");
        var initialized = engine ?? throw new InvalidOperationException("Bind weights before creating sessions.");
        return new Session(initialized.CreateSession(), graphState, prepared.BindingGraph);
    }

    public VmTensorGraphSession CreateGraphSession(IModelTensorCatalog tensors, GraphTensorState? state = null)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        lock (graphSessions)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            var prepared = plan ?? throw new InvalidOperationException("Prepare the graph before creating a session.");
            if (boundTensors is not null && !ReferenceEquals(boundTensors, tensors))
                throw new ArgumentException("The tensor catalog does not belong to the bound model.", nameof(tensors));
            if (engine is not null)
                throw new NotSupportedException("This backend owns a token-session engine; use its token-session adapter.");
            var session = new VmTensorGraphSession(prepared, tensors, state ?? new GraphTensorState(prepared.BindingGraph),
                allocate, released => { lock (graphSessions) graphSessions.Remove(released); }, tensorResources);
            graphSessions.Add(session);
            return session;
        }
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
            ForwardTokenAsync(token, CancellationToken.None).AsTask().GetAwaiter().GetResult().CopyTo(logits);
        }

        public async ValueTask<float[]> ForwardTokenAsync(int token, CancellationToken cancellationToken)
        {
            PrepareState();
            var logits = await session.ForwardAsync(token, cancellationToken).ConfigureAwait(false);
            state.MarkDeviceModified();
            return logits;
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

        public async ValueTask<ITokenGenerationScope> BeginGenerationAsync(CancellationToken cancellationToken)
        {
            PrepareState();
            var lease = await session.BeginGenerationAsync(cancellationToken).ConfigureAwait(false);
            return new Scope(lease, state);
        }

        private sealed class Scope(VmGenerationLease lease, PortableGraphState state) :
            ITokenGenerationScope, IAsyncTokenGenerationSession
        {
            public ITokenGenerationSession Session => this;
            public ReadOnlyMemory<float> ForwardToken(int token) =>
                ForwardTokenAsync(token).AsTask().GetAwaiter().GetResult();

            public async ValueTask<ReadOnlyMemory<float>> ForwardTokenAsync(int token,
                CancellationToken cancellationToken = default)
            {
                var logits = await lease.ForwardTokenAsync(token, cancellationToken).ConfigureAwait(false);
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
        VmTensorGraphSession[] sessions;
        lock (graphSessions)
        {
            if (disposed) return;
            disposed = true;
            sessions = graphSessions.ToArray();
            graphSessions.Clear();
        }
        List<Exception>? errors = null;
        foreach (var session in sessions)
            try { session.Dispose(); }
            catch (Exception error) { (errors ??= []).Add(error); }
        try { engine?.DisposeAsync().AsTask().GetAwaiter().GetResult(); }
        catch (Exception error) { (errors ??= []).Add(error); }
        try { tensorResources?.Dispose(); }
        catch (Exception error) { (errors ??= []).Add(error); }
        try { disposeCompiler?.Invoke(); }
        catch (Exception error) { (errors ??= []).Add(error); }
        if (errors is not null) throw new AggregateException("Compiled VM backend shutdown failed.", errors);
    }
}
