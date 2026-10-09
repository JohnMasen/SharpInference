using System.Runtime.InteropServices;
using System.Text;
using SharpInference.Gguf;
using SharpInference.Graphs;
using SharpInference.Vm;

namespace SharpInference.Runtime;

public sealed class GgmlModelReader : IModelReader
{
    public IModelFile Open(string path) => GgmlModelFile.Open(path);
}

public interface IArchitectureMetadataReader
{
    ModelMetadata Read(IModelTensorCatalog catalog);
}

public sealed class XmlArchitectureMetadataReader : IArchitectureMetadataReader
{
    private readonly string path;

    public XmlArchitectureMetadataReader(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        this.path = path;
    }

    public ModelMetadata Read(IModelTensorCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        var graph = GraphXml.DeserializeLogical(System.IO.File.ReadAllText(path));
        return new GraphArchitectureMetadataReader(graph).Read(catalog);
    }
}

public sealed class CatalogArchitectureMetadataReader(ModelGraphModuleRegistry modules) : IArchitectureMetadataReader
{
    public ModelMetadata Read(IModelTensorCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        ArgumentNullException.ThrowIfNull(modules);
        var module = modules.Resolve(catalog);
        var metadata = module.ReadMetadata(catalog);
        if (metadata.ArchitectureId != module.ArchitectureId)
            throw new InvalidDataException("The model module returned a different architecture identifier.");
        return metadata;
    }
}

public sealed class GraphArchitectureMetadataReader : IArchitectureMetadataReader
{
    private readonly GraphIdentity identity;
    private readonly GraphModelSignature signature;
    private readonly IReadOnlyList<GraphResource> resources;

    public GraphArchitectureMetadataReader(LogicalGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        identity = graph.Identity;
        signature = graph.Model;
        resources = graph.Resources;
    }

    public ModelMetadata Read(IModelTensorCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        foreach (var resource in resources.Where(resource => resource.Kind == GraphResourceKind.Weight))
        {
            var tensor = catalog.GetRequired(resource.BindingKey
                ?? throw new InvalidDataException($"Weight resource '{resource.Id}' has no binding key."));
            if (!resource.Tensor.Dimensions.SequenceEqual(tensor.Dimensions) ||
                resource.Tensor.ElementType != (tensor.DataType switch
                {
                    TensorDataType.Float16 => GraphElementType.Float16,
                    TensorDataType.Float32 => GraphElementType.Float32,
                    _ => throw new InvalidDataException($"Unsupported tensor type for '{tensor.Name}'."),
                }))
                throw new InvalidDataException($"Weight resource '{resource.Id}' does not match '{tensor.Name}'.");
        }

        return new ModelMetadata(identity.ArchitectureId,
            signature.Dimensions.ToDictionary(value => value.Key, value => (long)value.Value, StringComparer.Ordinal),
            signature.Attributes);
    }
}

public sealed class SuppliedLogicalGraphProvider : ILogicalGraphProvider
{
    private readonly LogicalGraph graph;

    public SuppliedLogicalGraphProvider(LogicalGraph graph) =>
        this.graph = graph ?? throw new ArgumentNullException(nameof(graph));

    public string ArchitectureId => graph.Identity.ArchitectureId;
    public LogicalGraph Build(IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        return graph;
    }
}

public sealed class ProcessorBuildContext
{
    private readonly HashSet<object> disposed = new(ReferenceEqualityComparer.Instance);

    internal ProcessorBuildContext(string path)
    {
        Path = path;
    }

    public string Path { get; }
    public IModelFile? File { get; set; }
    internal OwnedModelTensorCatalog? OwnedCatalog { get; set; }
    public IModelTensorCatalog? Catalog => (IModelTensorCatalog?)OwnedCatalog ?? File;
    public ModelMetadata? Metadata { get; set; }
    public IModelGraphModule? ModelModule { get; set; }
    public LogicalGraph? LogicalGraph { get; set; }
    public VmCompiledPlan? PreparedPlan { get; set; }
    public VmGraphBackend? Backend { get; set; }
    public IModelArchitecture? Architecture { get; set; }
    public IModel? Model { get; set; }

    public void DisposeOnce(object? resource)
    {
        if (resource is IDisposable disposable && disposed.Add(resource))
        {
            disposable.Dispose();
        }
    }
}

public interface IProcessorBuildStep
{
    string Name { get; }
    void Execute(ProcessorBuildContext context);
    void Dispose(ProcessorBuildContext context);
}

internal sealed class DelegateProcessorBuildStep(
    string name, Action<ProcessorBuildContext> execute,
    Action<ProcessorBuildContext> dispose) : IProcessorBuildStep
{
    public string Name => name;
    public void Execute(ProcessorBuildContext context) => execute(context);
    public void Dispose(ProcessorBuildContext context) => dispose(context);
}

public sealed class ProcessorPipelineBuilder
{
    private readonly List<IProcessorBuildStep> steps = [];
    private readonly HashSet<string> names = new(StringComparer.Ordinal);

    public ProcessorPipelineBuilder(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        Path = path;
    }

    public string Path { get; }
    public IReadOnlyList<string> StepNames => steps.Select(static step => step.Name).ToArray();

    internal ProcessorPipelineBuilder Append(IProcessorBuildStep step)
    {
        ArgumentNullException.ThrowIfNull(step);
        ArgumentException.ThrowIfNullOrWhiteSpace(step.Name);
        if (!names.Add(step.Name))
        {
            throw new InvalidOperationException($"Pipeline step '{step.Name}' is already registered.");
        }

        steps.Add(step);
        return this;
    }

    public Processor Build()
    {
        var context = new ProcessorBuildContext(Path);
        var started = new List<IProcessorBuildStep>();
        try
        {
            foreach (var step in steps)
            {
                started.Add(step);
                step.Execute(context);
            }

            if (context.File is null || context.Metadata is null ||
                context.Architecture is null || context.Model is null ||
                context.Backend is null || context.PreparedPlan is null)
            {
                throw new InvalidOperationException("The pipeline must load a file, read metadata, prepare a backend plan, and bind an architecture.");
            }


            if (!ProcessorPipelineExtensions.MatchesMetadata(context.Metadata, context.Model.ModelMetadata))
            {
                throw new InvalidDataException("The bound model metadata differs from the catalog metadata.");
            }

            context.OwnedCatalog?.ReleaseSource();
            var file = context.File;
            file.Dispose();
            context.File = null;
            return new Processor(context, started);
        }
        catch (Exception buildError)
        {
            var cleanupErrors = new List<Exception>();
            for (var index = started.Count - 1; index >= 0; index--)
            {
                try
                {
                    started[index].Dispose(context);
                }
                catch (Exception cleanupError)
                {
                    cleanupErrors.Add(cleanupError);
                }
            }

            if (cleanupErrors.Count != 0)
            {
                throw new AggregateException(
                    "The processor build failed and one or more steps failed to dispose.",
                    [buildError, .. cleanupErrors]);
            }

            throw;
        }
    }
}

public static class ProcessorPipelineExtensions
{
    public static ProcessorPipelineBuilder AddStep(
        this ProcessorPipelineBuilder builder, string name, Action<ProcessorBuildContext> execute) =>
        builder.AddStep(name, execute, static _ => { });

    public static ProcessorPipelineBuilder AddStep(
        this ProcessorPipelineBuilder builder, string name, Action<ProcessorBuildContext> execute,
        Action<ProcessorBuildContext> dispose)
    {
        ArgumentNullException.ThrowIfNull(builder);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(execute);
        ArgumentNullException.ThrowIfNull(dispose);
        return builder.Append(new DelegateProcessorBuildStep(name, execute, dispose));
    }

    public static ProcessorPipelineBuilder AddStep(
        this ProcessorPipelineBuilder builder, IProcessorBuildStep step) =>
        (builder ?? throw new ArgumentNullException(nameof(builder))).Append(step);

    public static ProcessorPipelineBuilder UseReader(
        this ProcessorPipelineBuilder builder, IModelReader reader,
        IArchitectureMetadataReader? metadataReader = null)
    {
        ArgumentNullException.ThrowIfNull(reader);
        return builder.AddStep(
            "reader",
            context =>
            {
                context.File = reader.Open(context.Path)
                    ?? throw new InvalidOperationException("The reader returned no model file.");
                context.Metadata = metadataReader?.Read(context.File);
            },
            context => context.DisposeOnce(context.File));
    }

    public static ProcessorPipelineBuilder UseXmlMetadata(
        this ProcessorPipelineBuilder builder, IModelReader reader, string path) =>
        builder.UseReader(reader, new XmlArchitectureMetadataReader(path));

    public static ProcessorPipelineBuilder UseProvider(
        this ProcessorPipelineBuilder builder, ILogicalGraphProvider provider) =>
        builder.UseProvider(_ => provider);

    public static ProcessorPipelineBuilder UseProvider(
        this ProcessorPipelineBuilder builder, Func<ProcessorBuildContext, ILogicalGraphProvider> factory) =>
        builder.AddStep("provider", context =>
        {
            ArgumentNullException.ThrowIfNull(factory);
            var provider = factory(context) ?? throw new InvalidOperationException("The provider factory returned no provider.");
            var catalog = context.Catalog ?? throw new InvalidOperationException("The reader must run before the provider.");
            context.ModelModule = provider as IModelGraphModule;
            if (context.Metadata is { } metadata && provider.ArchitectureId != metadata.ArchitectureId)
            {
                throw new InvalidOperationException("The graph provider does not match the model architecture.");
            }

            context.LogicalGraph = provider.Build(catalog);
            if (context.LogicalGraph.Identity.ArchitectureId != provider.ArchitectureId)
                throw new InvalidDataException("The graph provider returned a different architecture identifier.");
            context.Metadata ??= new GraphArchitectureMetadataReader(context.LogicalGraph).Read(catalog);
        });

    public static ProcessorPipelineBuilder UseXmlLogicalGraph(
        this ProcessorPipelineBuilder builder, string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return builder.AddStep("xml-logical", context =>
        {
            context.LogicalGraph = GraphXml.DeserializeLogical(System.IO.File.ReadAllText(path));
            context.Metadata ??= new GraphArchitectureMetadataReader(context.LogicalGraph).Read(context.Catalog ??
                throw new InvalidOperationException("The reader must run before the graph."));
        });
    }

    public static ProcessorPipelineBuilder UseBackend(
        this ProcessorPipelineBuilder builder, VmGraphBackend backend,
        GraphOptimizationOptions? options = null) =>
        builder.UseBackend(_ => backend, options);

    public static ProcessorPipelineBuilder UseBackend(
        this ProcessorPipelineBuilder builder,
        Func<ProcessorBuildContext, VmGraphBackend> factory,
        GraphOptimizationOptions? options = null) =>
        builder.AddStep("backend", context =>
        {
            ArgumentNullException.ThrowIfNull(factory);
            var backend = factory(context) ?? throw new InvalidOperationException("The backend factory returned no backend.");
            context.Backend = backend;
            if (context.LogicalGraph is { } logicalGraph)
            {
                ValidateGraph(logicalGraph.Identity, logicalGraph.Model, logicalGraph.Resources, context);
            }

            context.PreparedPlan = backend.Prepare(context.LogicalGraph ??
                throw new InvalidOperationException("A logical graph is required to bind the model."), options, context.ModelModule);
        }, context => context.DisposeOnce(context.Backend));

    private static void ValidateGraph(
        GraphIdentity identity, GraphModelSignature signature,
        IReadOnlyList<GraphResource> resources, ProcessorBuildContext context)
    {
        var metadata = context.Metadata ?? throw new InvalidOperationException("The reader must run before the graph.");
        var catalog = context.Catalog ?? throw new InvalidOperationException("The reader must run before the graph.");
        var graphMetadata = new ModelMetadata(identity.ArchitectureId,
            signature.Dimensions.ToDictionary(value => value.Key, value => (long)value.Value, StringComparer.Ordinal),
            signature.Attributes);
        if (!MatchesMetadata(metadata, graphMetadata))
        {
            throw new InvalidDataException("The graph model signature does not match the loaded model.");
        }

        foreach (var resource in resources.Where(static resource => resource.Kind == GraphResourceKind.Weight))
        {
            var tensor = catalog.GetRequired(resource.BindingKey
                ?? throw new InvalidDataException($"Weight resource '{resource.Id}' requires a binding key."));
            var elementType = tensor.DataType switch
            {
                TensorDataType.Float32 => GraphElementType.Float32,
                TensorDataType.Float16 => GraphElementType.Float16,
                _ => throw new InvalidDataException($"Unsupported tensor type for weight '{tensor.Name}'."),
            };
            if (resource.Tensor.ElementType != elementType ||
                !resource.Tensor.Dimensions.SequenceEqual(tensor.Dimensions))
            {
                throw new InvalidDataException(
                    $"Weight resource '{resource.Id}' does not match tensor '{tensor.Name}' shape or type.");
            }
        }
    }

    internal static bool MatchesMetadata(ModelMetadata expected, ModelMetadata actual) =>
        expected.ArchitectureId == actual.ArchitectureId &&
        expected.Dimensions.All(value => actual.Dimensions.TryGetValue(value.Key, out var dimension) && dimension == value.Value) &&
        expected.Attributes.All(value => actual.Attributes.TryGetValue(value.Key, out var attribute) && attribute == value.Value);

    private static ProcessorPipelineBuilder UseArchitecture(
        this ProcessorPipelineBuilder builder, Func<ProcessorBuildContext, IModelArchitecture> factory) =>
        builder.AddStep("architecture", context =>
        {
            ArgumentNullException.ThrowIfNull(factory);
            context.Architecture = factory(context)
                ?? throw new InvalidOperationException("The architecture factory returned no architecture.");
            if (context.Architecture.Id != context.Metadata?.ArchitectureId)
            {
                throw new InvalidOperationException("The architecture does not match the model metadata.");
            }

            if (context.Architecture is IModelWeightPreflight preflight)
            {
                preflight.PrepareWeightPlan(context.File
                    ?? throw new InvalidOperationException("The reader must run before architecture binding."));
            }

            context.OwnedCatalog = new OwnedModelTensorCatalog(
                context.File ?? throw new InvalidOperationException("The reader must run before architecture binding."),
                context.Architecture is not IModelWeightOwnershipPolicy { RequiresCpuWeightCopy: false });
            context.Model = context.Architecture.Bind(context.Catalog
                ?? throw new InvalidOperationException("The reader must run before architecture binding."));
        }, context =>
        {
            try { context.DisposeOnce(context.Model); }
            finally { context.DisposeOnce(context.Architecture); }
        });

    public static ProcessorPipelineBuilder UsePortableGraphArchitecture(this ProcessorPipelineBuilder builder) =>
        builder.UseArchitecture(context => new PortableGraphArchitecture(
            context.Backend ?? throw new InvalidOperationException("The backend must be prepared before binding the graph."),
            context.PreparedPlan ?? throw new InvalidOperationException("The graph must be prepared before binding the model.")));

}

public enum ProcessorExecutionGraphKind
{
    Inference,
    Prefill,
}

public sealed class Processor : IProcessor
{
    private readonly ProcessorBuildContext context;
    private readonly IReadOnlyList<IProcessorBuildStep> steps;
    private readonly IModelArchitecture architecture;
    private readonly IModel model;
    private readonly VmGraphBackend backend;
    private bool disposed;

    internal Processor(ProcessorBuildContext context, IReadOnlyList<IProcessorBuildStep> steps)
    {
        this.context = context;
        this.steps = steps.ToArray();
        architecture = context.Architecture!;
        model = context.Model!;
        backend = context.Backend!;
        Metadata = context.Metadata!;
        LogicalGraph = context.LogicalGraph;
        PreparedPlan = context.PreparedPlan!;
        Capabilities = new ProcessorCapabilities(
            [backend.SupportsTokenSessions ? ProcessorInputModality.Text : ProcessorInputModality.Tensor],
            ProcessorOutputModality.Tensor,
            text: backend.SupportsTokenSessions ? new ProcessorTextCapabilities(int.MaxValue) : null,
            execution: backend.ExecutionCapabilities);
    }

    public ModelMetadata Metadata { get; }
    public ProcessorCapabilities Capabilities { get; }
    public LogicalGraph? LogicalGraph { get; }
    public VmCompiledPlan PreparedPlan { get; }

    public void ExportLogicalGraph(Stream stream)
    {
        ThrowIfDisposed();
        var graph = LogicalGraph ??
            throw new InvalidOperationException("The processor has no logical graph.");
        WriteGraphText(stream, GraphJson.Serialize(graph));
    }

    public VmProgram InferenceProgram => PreparedPlan.Program;
    public VmProgram PrefillProgram => PreparedPlan.PrefillPlan?.Program ?? PreparedPlan.Program;

    public void ExportCompiledArtifact(string directory, ProcessorExecutionGraphKind kind = ProcessorExecutionGraphKind.Inference)
    {
        ThrowIfDisposed();
        if (!Enum.IsDefined(kind)) throw new ArgumentOutOfRangeException(nameof(kind));
        var compiled = PreparedPlan;
        if (kind == ProcessorExecutionGraphKind.Prefill) compiled = compiled.PrefillPlan ?? compiled;
        var export = compiled.ExportArtifact ??
            throw new NotSupportedException("The selected backend does not expose compiled VM artifacts.");
        export(directory);
    }

    public void ExportExecutionGraph(
        Stream stream, ProcessorExecutionGraphKind kind = ProcessorExecutionGraphKind.Inference)
    {
        ThrowIfDisposed();
        if (!Enum.IsDefined(kind))
            throw new ArgumentOutOfRangeException(nameof(kind));
        var compiled = PreparedPlan;
        if (kind == ProcessorExecutionGraphKind.Prefill) compiled = compiled.PrefillPlan ?? compiled;
        WriteGraphText(stream, VmProgramXml.Serialize(compiled.Program));
    }

    private static void WriteGraphText(Stream stream, string text)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanWrite)
            throw new ArgumentException("The stream must be writable.", nameof(stream));
        using var writer = new StreamWriter(stream, new UTF8Encoding(false), leaveOpen: true);
        writer.Write(text);
    }

    public static Processor Load(string path, IModelReader reader, ModelGraphModuleRegistry modules,
        VmGraphBackend backend, GraphOptimizationOptions? options = null) =>
        new ProcessorPipelineBuilder(path)
            .UseReader(reader, new CatalogArchitectureMetadataReader(modules))
            .UseProvider(context => modules.GetRequired(
                context.Metadata?.ArchitectureId ??
                throw new InvalidOperationException("The reader must run before selecting a graph provider.")))
            .UseBackend(backend, options)
            .UsePortableGraphArchitecture()
            .Build();

    public static Processor LoadGraph(
        string path, LogicalGraph graph, VmGraphBackend backend,
        GraphOptimizationOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(backend);
        return new ProcessorPipelineBuilder(path)
            .UseReader(new GgmlModelReader(), new GraphArchitectureMetadataReader(graph))
            .UseProvider(new SuppliedLogicalGraphProvider(graph))
            .UseBackend(backend, options)
            .UsePortableGraphArchitecture()
            .Build();
    }

    public static Processor LoadGraph(
        string path, ILogicalGraphProvider provider, VmGraphBackend backend,
        GraphOptimizationOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(provider);
        ArgumentNullException.ThrowIfNull(backend);
        var metadataReader = new ProviderGraphMetadataReader(provider);
        return new ProcessorPipelineBuilder(path)
            .UseReader(new GgmlModelReader(), metadataReader)
            .UseProvider(_ => new SuppliedLogicalGraphProvider(metadataReader.Graph))
            .UseBackend(backend, options)
            .UsePortableGraphArchitecture()
            .Build();
    }

    private sealed class ProviderGraphMetadataReader(ILogicalGraphProvider provider) : IArchitectureMetadataReader
    {
        private LogicalGraph? graph;

        public LogicalGraph Graph => graph ??
            throw new InvalidOperationException("The graph provider must run after the reader.");

        public ModelMetadata Read(IModelTensorCatalog catalog)
        {
            var supplied = provider.Build(catalog) ??
                throw new InvalidOperationException("The graph provider returned no graph.");
            if (supplied.Identity.ArchitectureId != provider.ArchitectureId)
                throw new InvalidDataException("The graph provider returned a different architecture.");
            var metadata = new GraphArchitectureMetadataReader(supplied).Read(catalog);
            graph = supplied;
            return metadata;
        }
    }

    internal void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);
    public ProcessorGraphSession CreateGraphSession(GraphTensorState? state = null)
    {
        ThrowIfDisposed();
        if (backend.SupportsTokenSessions)
            throw new NotSupportedException("This processor owns a token-session engine; use CreateSession.");
        var portable = model as PortableGraphModel ??
            throw new NotSupportedException("The bound model does not expose a portable tensor graph.");
        return new ProcessorGraphSession(this, backend.CreateGraphSession(portable.Tensors, state));
    }

    public ProcessorSession CreateSession()
    {
        ThrowIfDisposed();
        if (!backend.SupportsTokenSessions)
            throw new NotSupportedException("The graph does not declare the token-session contract; use CreateGraphSession.");
        return CreateSession(architecture.CreateState(model));
    }

    IProcessorSession IProcessor.CreateSession() => backend.SupportsTokenSessions ? CreateSession() : CreateGraphSession();

    internal ProcessorSession CreateSession(IModelState state)
    {
        ThrowIfDisposed();
        IVmSessionExecutor? executor = null;
        try
        {
            executor = backend.CreateSessionExecutor(model, state, PreparedPlan)
                ?? throw new InvalidOperationException("The backend returned no VM session executor.");
            return new ProcessorSession(this, model, state, executor);
        }
        catch (Exception creationError)
        {
            var cleanupErrors = new List<Exception>();
            try { executor?.Dispose(); }
            catch (Exception error) { cleanupErrors.Add(error); }
            try
            {
                if (state is IDisposable disposableState) disposableState.Dispose();
            }
            catch (Exception error) { cleanupErrors.Add(error); }
            if (cleanupErrors.Count != 0)
            {
                throw new AggregateException(
                    "The processor session could not be created and cleanup failed.",
                    [creationError, .. cleanupErrors]);
            }
            throw;
        }
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        var errors = new List<Exception>();
        for (var index = steps.Count - 1; index >= 0; index--)
        {
            try { steps[index].Dispose(context); }
            catch (Exception error) { errors.Add(error); }
        }

        if (errors.Count != 0)
            throw new AggregateException("One or more processor steps failed to dispose.", errors);
    }
}

public sealed class ProcessorSession :
    IScopedTokenGenerationSession, IAsyncTokenPrefillSession,
    IAsyncTokenGenerationSession, IProcessorSession
{
    private readonly Processor owner;
    private readonly IModelState state;
    private readonly IVmSessionExecutor executor;
    private readonly float[] logits;
    private readonly object gate = new();
    private bool disposed;
    private bool operationActive;

    IProcessor IProcessorSession.Processor => owner;

    internal ProcessorSession(
        Processor owner, IModel model, IModelState state,
        IVmSessionExecutor executor)
    {
        this.owner = owner;
        this.state = state;
        this.executor = executor;
        var graph = owner.LogicalGraph ?? throw new InvalidOperationException("A session requires a logical graph.");
        var output = graph.Resources.Single(resource => resource.Id == graph.Outputs.Single());
        logits = new float[checked(output.Tensor.Dimensions.Aggregate(1, (count, size) => checked(count * size)))];
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        owner.ThrowIfDisposed();
        if (operationActive)
            throw new InvalidOperationException("A queued operation or generation scope already owns this session.");
    }

    public ReadOnlyMemory<float> ForwardToken(int token)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            executor.ForwardToken(token, logits);
            return logits;
        }
    }

    public async ValueTask<ITokenGenerationScope> BeginGenerationAsync(CancellationToken cancellationToken = default)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            operationActive = true;
        }
        try
        {
            var scope = await executor.BeginGenerationAsync(cancellationToken).ConfigureAwait(false);
            return new ExclusiveGeneration(this, scope);
        }
        catch
        {
            EndExclusiveOperation();
            throw;
        }
    }

    public async ValueTask<ReadOnlyMemory<float>> ForwardTokenAsync(int token,
        CancellationToken cancellationToken = default)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            operationActive = true;
        }
        try
        {
            var result = await executor.ForwardTokenAsync(token, cancellationToken).ConfigureAwait(false);
            lock (gate)
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                owner.ThrowIfDisposed();
                result.CopyTo(logits);
                return result;
            }
        }
        finally { EndExclusiveOperation(); }
    }

    private void EndExclusiveOperation() { lock (gate) operationActive = false; }

    private sealed class ExclusiveGeneration(ProcessorSession owner, ITokenGenerationScope inner) :
        ITokenGenerationScope, IAsyncTokenGenerationSession
    {
        private readonly object gate = new();
        private Task? shutdown;
        public ITokenGenerationSession Session => this;
        public ReadOnlyMemory<float> ForwardToken(int token)
        {
            lock (owner.gate)
            {
                ObjectDisposedException.ThrowIf(owner.disposed, owner);
                owner.owner.ThrowIfDisposed();
                inner.Session.ForwardToken(token).Span.CopyTo(owner.logits);
                return owner.logits;
            }
        }
        public async ValueTask<ReadOnlyMemory<float>> ForwardTokenAsync(int token,
            CancellationToken cancellationToken = default)
        {
            lock (owner.gate)
            {
                ObjectDisposedException.ThrowIf(owner.disposed, owner);
                owner.owner.ThrowIfDisposed();
            }
            if (inner.Session is not IAsyncTokenGenerationSession asynchronous)
                throw new NotSupportedException("Queued generation requires asynchronous inference support.");
            var result = await asynchronous.ForwardTokenAsync(token, cancellationToken).ConfigureAwait(false);
            lock (owner.gate)
            {
                ObjectDisposedException.ThrowIf(owner.disposed, owner);
                owner.owner.ThrowIfDisposed();
                result.Span.CopyTo(owner.logits);
                return result;
            }
        }
        public ReadOnlyMemory<float> Prefill(ReadOnlySpan<int> tokens) =>
            throw new InvalidOperationException("Prefill cannot run inside a generation scope.");
        public ValueTask DisposeAsync()
        {
            lock (gate) return new ValueTask(shutdown ??= DisposeCoreAsync());
        }
        private async Task DisposeCoreAsync()
        {
            try { await inner.DisposeAsync().ConfigureAwait(false); }
            finally { owner.EndExclusiveOperation(); }
        }
    }

    public async ValueTask<ReadOnlyMemory<float>> PrefillAsync(ReadOnlyMemory<int> tokens,
        CancellationToken cancellationToken = default)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            operationActive = true;
        }
        try
        {
            var result = await executor.PrefillAsync(tokens, cancellationToken).ConfigureAwait(false);
            lock (gate)
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                owner.ThrowIfDisposed();
                result.CopyTo(logits);
                return logits;
            }
        }
        finally { EndExclusiveOperation(); }
    }

    public ReadOnlyMemory<float> Prefill(ReadOnlySpan<int> tokens)
    {
        if (tokens.IsEmpty) throw new ArgumentException("At least one token is required.", nameof(tokens));
        lock (gate)
        {
            ThrowIfDisposed();
            executor.ForwardTokens(tokens, logits);
            return logits;
        }
    }

    public ProcessorSession Fork()
    {
        lock (gate)
        {
            ThrowIfDisposed();
            return owner.CreateSession(state.Clone());
        }
    }

    public void Reset()
    {
        lock (gate)
        {
            ThrowIfDisposed();
            state.Reset();
            Array.Clear(logits);
        }
    }

    public void SaveState(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        lock (gate)
        {
            ThrowIfDisposed();
            if (!BitConverter.IsLittleEndian)
                throw new PlatformNotSupportedException("GGUF state tensors require little-endian FP32 values.");
            var graph = owner.PreparedPlan.BindingGraph;
            var tensors = executor.ReadState(graph.GraphState).Select(value =>
                new GgufStateTensor(value.Name, GgufTensorType.Float32,
                    value.Dimensions.Select(size => checked((ulong)size)),
                    MemoryMarshal.AsBytes(value.Values.AsSpan()).ToArray())).ToArray();
            GgufStateFile.Write(stream, new GgufState(graph.GraphState.Schema.Name, tensors));
        }
    }

    public void LoadState(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        lock (gate)
        {
            ThrowIfDisposed();
            if (!BitConverter.IsLittleEndian)
                throw new PlatformNotSupportedException("GGUF state tensors require little-endian FP32 values.");
            var graph = owner.PreparedPlan.BindingGraph;
            var saved = GgufStateFile.Read(stream);
            if (!string.Equals(saved.SchemaName, graph.GraphState.Schema.Name, StringComparison.Ordinal) ||
                saved.Tensors.Count != graph.GraphState.Slots.Count)
                throw new InvalidDataException("The saved state schema does not match this graph.");
            var byName = saved.Tensors.ToDictionary(tensor => tensor.Name, StringComparer.Ordinal);
            var resources = graph.Resources.ToDictionary(resource => resource.Id);
            var values = new GraphStateValue[graph.GraphState.Slots.Count];
            for (var index = 0; index < values.Length; index++)
            {
                var slot = graph.GraphState.Slots[index];
                if (!resources.TryGetValue(slot.Resource, out var resource) ||
                    !byName.TryGetValue(slot.Name, out var tensor) ||
                    resource.Tensor.ElementType != GraphElementType.Float32 ||
                    resource.Tensor.Layout != "dense" ||
                    tensor.Type != GgufTensorType.Float32 ||
                    !tensor.Dimensions.SequenceEqual(resource.Tensor.Dimensions.Select(size => checked((ulong)size))) ||
                    tensor.Data.Length != checked(resource.Tensor.Dimensions.Aggregate(
                        1L, (size, dimension) => checked(size * dimension)) * sizeof(float)))
                    throw new InvalidDataException($"State tensor '{slot.Name}' does not match GraphState.");
                var numbers = new float[tensor.Data.Length / sizeof(float)];
                MemoryMarshal.Cast<byte, float>(tensor.Data.Span).CopyTo(numbers);
                values[index] = new GraphStateValue(slot.Name, resource.Tensor.Dimensions, numbers);
            }
            executor.WriteState(graph.GraphState, values);
        }
    }

    public void Dispose()
    {
        if (Volatile.Read(ref disposed)) return;

        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            var errors = new List<Exception>();
            try { executor.Dispose(); }
            catch (Exception error) { errors.Add(error); }
            try
            {
                if (state is IDisposable disposableState) disposableState.Dispose();
            }
            catch (Exception error) { errors.Add(error); }
            if (errors.Count != 0)
                throw new AggregateException("One or more processor session resources failed to dispose.", errors);
        }
    }
}
