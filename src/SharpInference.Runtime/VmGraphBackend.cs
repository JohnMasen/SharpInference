using System.Runtime.InteropServices;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Runtime;

/// <summary>Acquires exclusive VM execution scopes for token generation.</summary>
public interface IVmGenerationScopeExecutor
{
    /// <summary>Acquires an asynchronously disposable scope for exclusive token generation.</summary>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
    ValueTask<ITokenGenerationScope> BeginGenerationAsync(CancellationToken cancellationToken);
}

/// <summary>Executes asynchronous prompt prefill on a VM session.</summary>
public interface IVmPrefillExecutor
{
    /// <summary>Processes prompt tokens and returns the final token&apos;s logits.</summary>
    /// <param name="tokens">The ordered prompt token identifiers to prefill.</param>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
    ValueTask<float[]> PrefillAsync(ReadOnlyMemory<int> tokens, CancellationToken cancellationToken);
}

/// <summary>Executes asynchronous single-token inference on a VM session.</summary>
public interface IVmAsyncInferenceExecutor
{
    /// <summary>Processes one token and returns its output logits.</summary>
    /// <param name="token">The input token identifier to process.</param>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
    ValueTask<float[]> ForwardTokenAsync(int token, CancellationToken cancellationToken);
}

/// <summary>Combines VM token inference, prefill, generation scopes, state access, and session disposal.</summary>
public interface IVmSessionExecutor : IProcessorSessionExecutor, IProcessorStateExecutor,
    IVmGenerationScopeExecutor, IVmPrefillExecutor, IVmAsyncInferenceExecutor
{
}

/// <summary>Stores a VM program, executable factory, optional exports, and associated prefill and target plans.</summary>
public sealed class VmCompiledPlan
{
    internal VmCompiledPlan(LogicalGraph bindingGraph, VmProgram program,
        Func<IVmExecutable> createExecutable, Action<string>? exportArtifact = null,
        VmCompiledPlan? prefillPlan = null, TargetExecutionPlan? executionPlan = null)
    {
        BindingGraph = bindingGraph;
        Program = program;
        CreateExecutable = createExecutable;
        ExportArtifact = exportArtifact;
        PrefillPlan = prefillPlan;
        ExecutionPlan = executionPlan;
    }

    internal LogicalGraph BindingGraph { get; }
    /// <summary>Gets the compiled VM program.</summary>
    public VmProgram Program { get; }
    /// <summary>Gets the factory that creates an executable instance for this program.</summary>
    public Func<IVmExecutable> CreateExecutable { get; }
    /// <summary>Gets the optional callback that exports the compiled artifact into a directory.</summary>
    public Action<string>? ExportArtifact { get; }
    /// <summary>Gets the optional separately compiled prefill plan.</summary>
    public VmCompiledPlan? PrefillPlan { get; }
    /// <summary>Gets the authoritative target plan when compilation started from an execution plan.</summary>
    public TargetExecutionPlan? ExecutionPlan { get; }
}

/// <summary>Prepares portable graphs for a VM target and owns their compiled programs, model resources, and execution sessions.</summary>
public sealed class VmGraphBackend : IDisposable
{
    private readonly VmTarget target;
    private readonly Func<VmProgram, Func<IVmExecutable>> compile;
    private readonly VmEngineOptions options;
    private readonly VmOptimizationOptions optimization;
    private readonly Action? disposeCompiler;
    private readonly VmProgram? suppliedProgram;
    private readonly TargetExecutionPlan? suppliedExecutionPlan;
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

    /// <summary>Creates a VM graph backend with compilation, resource allocation, optimization, and optional precompiled inputs.</summary>
    /// <param name="target">The VM execution target for the backend or optimization.</param>
    /// <param name="compile">The compiler callback that returns an executable factory for a VM program.</param>
    /// <param name="options">The optional VM engine queue capacities, instance counts, and prefill token limit.</param>
    /// <param name="optimization">The optional VM optimization settings.</param>
    /// <param name="disposeCompiler">The optional callback that releases compiler-owned resources.</param>
    /// <param name="suppliedProgram">The optional pre-generated inference program, mutually exclusive with an execution plan.</param>
    /// <param name="exportArtifact">The optional callback that exports the compiled inference artifact to a directory.</param>
    /// <param name="suppliedPrefillProgram">The optional pre-generated VM program for prompt prefill.</param>
    /// <param name="compilePrefill">The optional separate compiler callback for the prefill program.</param>
    /// <param name="exportPrefillArtifact">The optional callback that exports the compiled prefill artifact to a directory.</param>
    /// <param name="allocate">The optional factory for allocating storage for VM slots.</param>
    /// <param name="deviceName">The optional display name of the execution device.</param>
    /// <param name="generatorCollections">The optional instruction providers used for target execution-plan generation.</param>
    /// <param name="diagnostics">The optional source of named backend measurements.</param>
    /// <param name="operationLowerer">The optional backend-specific graph operation lowerer.</param>
    /// <param name="executionCapabilities">The optional declared concurrency and storage-domain capabilities.</param>
    /// <param name="suppliedExecutionPlan">The optional authoritative target execution plan to use instead of generating one.</param>
    public VmGraphBackend(VmTarget target, Func<VmProgram, Func<IVmExecutable>> compile,
        VmEngineOptions? options = null, VmOptimizationOptions? optimization = null, Action? disposeCompiler = null,
        VmProgram? suppliedProgram = null, Action<string>? exportArtifact = null,
        VmProgram? suppliedPrefillProgram = null, Func<VmProgram, Func<IVmExecutable>>? compilePrefill = null,
        Action<string>? exportPrefillArtifact = null, Func<VmSlot, IVmStorage>? allocate = null,
        string? deviceName = null, IEnumerable<IInstructionCollectionProvider>? generatorCollections = null,
        IVmBackendDiagnostics? diagnostics = null, IGraphOperationLowerer? operationLowerer = null,
        ProcessorExecutionCapabilities? executionCapabilities = null, TargetExecutionPlan? suppliedExecutionPlan = null)
    {
        this.target = target;
        this.compile = compile;
        this.options = options ?? new VmEngineOptions(16, 16);
        this.optimization = optimization ?? new VmOptimizationOptions();
        this.disposeCompiler = disposeCompiler;
        this.suppliedProgram = suppliedProgram;
        if (suppliedProgram is not null && suppliedExecutionPlan is not null)
            throw new ArgumentException("Select either a VM program or a target execution plan.");
        this.suppliedExecutionPlan = suppliedExecutionPlan;
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
    }

    /// <summary>Gets the VM execution target selected for this backend.</summary>
    public VmTarget Target => target;
    /// <summary>Gets the display name of the execution device.</summary>
    public string DeviceName { get; }
    /// <summary>Gets the tier-one report when the active execution plan was generated by this backend.</summary>
    public TierOneOptimizationReport? OptimizationReport =>
        generator?.LastPlan is { } generated && ReferenceEquals(generated, plan?.ExecutionPlan) ? generator.LastOptimizationReport : null;
    /// <summary>Gets the GPU matrix-vector optimization report for the active generated plan.</summary>
    public GpuMatVecOptimizationReport? MatVecOptimizationReport =>
        generator?.LastPlan is { } generated && ReferenceEquals(generated, plan?.ExecutionPlan) ? generator.LastMatVecOptimizationReport : null;
    /// <summary>Gets the optional provider of backend counters and timings.</summary>
    public IVmBackendDiagnostics? Diagnostics { get; }
    /// <summary>Gets the backend&apos;s session concurrency and storage-domain capabilities.</summary>
    public ProcessorExecutionCapabilities ExecutionCapabilities { get; }
    /// <summary>Gets the prepared inference program, or null before preparation.</summary>
    public VmProgram? Program => plan?.Program;
    /// <summary>Gets the prepared prefill program or its inference fallback, or null before preparation.</summary>
    public VmProgram? PrefillProgram => plan?.PrefillPlan?.Program ?? plan?.Program;
    /// <summary>Gets whether the prepared graph declares compatible token input, logits output, float32 state, and prefill entry points.</summary>
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

    /// <summary>Validates and compiles a logical graph or target execution plan before model resources or sessions are bound.</summary>
    /// <param name="logical">The logical graph to validate, lower, and compile.</param>
    /// <param name="options">The optional graph optimization settings; target planning owns optimizations beyond the off boundary.</param>
    /// <param name="modelModule">The optional model module supplying operation validation and lowering.</param>
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
            if (suppliedExecutionPlan is not null) return Prepare(suppliedExecutionPlan);
            GraphValidator.Validate(logical, modelModule?.OperationValidator);
            logical = GraphOperationLowering.Apply(logical, modelModule?.OperationLowerer);
            logical = GraphOperationLowering.Apply(logical, operationLowerer);
            if (options is not null && options.Boundary != OptimizationBoundary.Off)
                throw new NotSupportedException("Graph optimization must be configured during target execution-plan generation.");
            if (suppliedProgram is not null)
            {
                var metadata = TargetExecutionPlan.CreateBindings(logical);
                return CompilePrepared(metadata, suppliedProgram, null);
            }
            var targetGenerator = generator ?? throw new InvalidOperationException(
                "Supply a graph generator IC catalog or a pre-generated execution program.");
            var il = new IlExecutionPlan(logical);
            targetGenerator.ValidateOperations(il);
            var executionPlan = targetGenerator.GeneratePlan(il, optimization);
            return Prepare(executionPlan);
        }
    }

    /// <summary>Validates and compiles a logical graph or target execution plan before model resources or sessions are bound.</summary>
    /// <param name="executionPlan">The target execution plan with authoritative input, output, and state binding metadata.</param>
    public VmCompiledPlan Prepare(TargetExecutionPlan executionPlan)
    {
        ArgumentNullException.ThrowIfNull(executionPlan);
        lock (graphSessions)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (modelBound || graphSessions.Count != 0)
                throw new InvalidOperationException("A backend with bound resources or active sessions cannot change plans.");
            executionPlan = generator?.BindPlan(executionPlan) ?? executionPlan;
            var metadata = executionPlan.Bindings ??
                throw new InvalidDataException("Runtime preparation requires execution-plan IO and state binding metadata.");
            var program = ExecutionPlanAssembler.Assemble(executionPlan);
            VmInstructionContracts.ValidateResolved(program);
            return CompilePrepared(metadata, program, executionPlan);
        }
    }

    private VmCompiledPlan CompilePrepared(LogicalGraph graph, VmProgram program, TargetExecutionPlan? executionPlan)
    {
        ValidateProgram(program, graph);
        var factory = compile(program);
        VmCompiledPlan? prefillPlan = null;
        if (suppliedPrefillProgram is { } prefillProgram)
        {
            ValidateProgram(prefillProgram, graph);
            prefillPlan = new VmCompiledPlan(graph, prefillProgram,
                (compilePrefill ?? compile)(prefillProgram), exportPrefillArtifact);
        }
        Action<string>? export = exportArtifact is null ? null : directory =>
        {
            exportArtifact(directory);
            if (executionPlan is not null)
                File.WriteAllText(Path.Combine(directory, "execution-plan.xml"), ExecutionPlanXml.Serialize(executionPlan));
        };
        plan = new VmCompiledPlan(graph, program, factory, export, prefillPlan, executionPlan);
        return plan;
    }

    private void ValidateProgram(VmProgram program, LogicalGraph graph)
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

    /// <summary>Binds model weights and initializes either shared tensor resources or the token-session engine.</summary>
    /// <param name="model">The model bound to the prepared graph and backend.</param>
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

    /// <summary>Creates a token-session executor for the model, state, and plan bound to this backend.</summary>
    /// <param name="model">The model bound to the prepared graph and backend.</param>
    /// <param name="state">The portable model state attached to the new token-session executor.</param>
    /// <param name="prepared">The compiled plan already prepared and bound on this backend.</param>
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

    /// <summary>Creates a descriptor-bound tensor session using this backend&apos;s prepared plan and optional state.</summary>
    /// <param name="tensors">The tensor catalog supplying model weights and their descriptors.</param>
    /// <param name="state">The optional existing graph state; null creates fresh state.</param>
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
        private readonly LogicalGraph graph;
        private long revision;
        private bool disposed;

        public Session(VmInferenceSession session, PortableGraphState state, LogicalGraph graph)
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

    /// <summary>Releases active graph sessions, shared resources, the inference engine, and compiler-owned resources.</summary>
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
