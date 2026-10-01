using System.Runtime.InteropServices;
using System.Text;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Gguf;
using SharpInference.Graphs;

namespace SharpInference.Runtime;

public sealed class GgmlModelReader : IModelReader
{
    public IModelFile Open(string path) => GgmlModelFile.Open(path);
}

public interface IArchitectureMetadataReader
{
    RwkvModelMetadata Read(IModelTensorCatalog catalog);
}

public enum XmlArchitectureGraphKind
{
    Logical,
    Execution,
}

public sealed class XmlArchitectureMetadataReader : IArchitectureMetadataReader
{
    private readonly string path;
    private readonly XmlArchitectureGraphKind kind;

    public XmlArchitectureMetadataReader(string path, XmlArchitectureGraphKind kind)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!Enum.IsDefined(kind)) throw new ArgumentOutOfRangeException(nameof(kind));
        this.path = path;
        this.kind = kind;
    }

    public RwkvModelMetadata Read(IModelTensorCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        var xml = System.IO.File.ReadAllText(path);
        if (kind == XmlArchitectureGraphKind.Logical)
        {
            var graph = GraphXml.DeserializeLogical(xml);
            return new GraphArchitectureMetadataReader(graph).Read(catalog);
        }

        return new GraphArchitectureMetadataReader(GraphXml.DeserializeExecution(xml)).Read(catalog);
    }
}

public sealed class CatalogArchitectureMetadataReader : IArchitectureMetadataReader
{
    public RwkvModelMetadata Read(IModelTensorCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        var architectureId = RwkvModelArchitectureDetector.Detect(catalog);
        var tensor = catalog.GetRequired(architectureId == "rwkv-6"
            ? "blocks.0.att.time_faaaa"
            : "blocks.0.att.r_k");
        var dimensions = tensor.Dimensions;
        var headCount = architectureId == "rwkv-6" && dimensions.Count == 3
            ? dimensions[2]
            : architectureId == "rwkv-7" && dimensions.Count == 2
                ? dimensions[1]
                : throw new InvalidDataException($"Invalid head tensor shape for '{architectureId}'.");
        if (headCount <= 0)
        {
            throw new InvalidDataException($"Invalid head count for '{architectureId}'.");
        }
        var headSize = architectureId == "rwkv-6" ? dimensions[1] : catalog.EmbeddingSize / headCount;
        if (headCount <= 0 || headSize <= 0 ||
            checked(headCount * headSize) != catalog.EmbeddingSize)
        {
            throw new InvalidDataException($"Invalid head tensor dimensions for '{architectureId}'.");
        }

        return new RwkvModelMetadata(
            catalog.VocabularySize, catalog.EmbeddingSize, catalog.LayerCount,
            headCount, headSize, architectureId);
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

    public GraphArchitectureMetadataReader(ExecutionGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        identity = graph.Identity;
        signature = graph.Model;
        resources = graph.Resources;
    }

    public RwkvModelMetadata Read(IModelTensorCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        if (signature.VocabularySize != catalog.VocabularySize ||
            signature.EmbeddingSize != catalog.EmbeddingSize ||
            signature.LayerCount != catalog.LayerCount ||
            signature.HeadCount * (long)signature.HeadSize != signature.EmbeddingSize)
            throw new InvalidDataException("The logical graph signature does not match the loaded model.");

        foreach (var resource in resources.Where(resource => resource.Kind == GraphResourceKind.Weight))
        {
            var tensor = catalog.GetRequired(resource.BindingKey
                ?? throw new InvalidDataException($"Weight resource '{resource.Id}' has no binding key."));
            if (!resource.Tensor.Dimensions.SequenceEqual(tensor.Dimensions) ||
                resource.Tensor.ElementType != (tensor.DataType switch
                {
                    RwkvTensorDataType.Float16 => GraphElementType.Float16,
                    RwkvTensorDataType.Float32 => GraphElementType.Float32,
                    _ => throw new InvalidDataException($"Unsupported tensor type for '{tensor.Name}'."),
                }))
                throw new InvalidDataException($"Weight resource '{resource.Id}' does not match '{tensor.Name}'.");
        }

        return new RwkvModelMetadata(signature.VocabularySize, signature.EmbeddingSize,
            signature.LayerCount, signature.HeadCount, signature.HeadSize, identity.ArchitectureId);
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

    internal ProcessorBuildContext(string path, bool debug)
    {
        Path = path;
        Debug = debug;
    }

    public string Path { get; }
    public bool Debug { get; }
    public IModelFile? File { get; set; }
    internal OwnedModelTensorCatalog? OwnedCatalog { get; set; }
    public IModelTensorCatalog? Catalog => (IModelTensorCatalog?)OwnedCatalog ?? File;
    public RwkvModelMetadata? Metadata { get; set; }
    public LogicalGraph? LogicalGraph { get; set; }
    public ExecutionGraph? ExecutionGraph { get; set; }
    public IBackendExecutablePlan? PreparedPlan { get; set; }
    public ExecutionGraph? InferenceExecutionGraph
    {
        get => ExecutionGraph;
        set => ExecutionGraph = value;
    }
    public IBackendExecutablePlan? InferencePreparedPlan
    {
        get => PreparedPlan;
        set => PreparedPlan = value;
    }
    public ExecutionGraph? PrefillExecutionGraph { get; set; }
    public IBackendExecutablePlan? PrefillPreparedPlan { get; set; }
    public IList<IPrefillGraphOptimizer> PrefillOptimizers { get; } = new List<IPrefillGraphOptimizer>();
    public IExecutionGraphBackend? Backend { get; set; }
    public IRwkvArchitecture? Architecture { get; set; }
    public IRwkvModel? Model { get; set; }

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
    public bool Debug { get; set; }
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
        var context = new ProcessorBuildContext(Path, Debug);
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

            if (context.PreparedPlan is not null &&
                !ReferenceEquals(context.PreparedPlan.Graph, context.ExecutionGraph))
            {
                throw new InvalidOperationException("The prepared plan does not match the execution graph.");
            }
            if (context.PrefillPreparedPlan is not null &&
                !ReferenceEquals(context.PrefillPreparedPlan.Graph, context.PrefillExecutionGraph))
            {
                throw new InvalidOperationException("The prefill plan does not match the prefill execution graph.");
            }

            if (context.Model.Metadata != context.Metadata)
            {
                throw new InvalidDataException("The bound model metadata differs from the catalog metadata.");
            }

            if (context.Debug &&
                (context.Backend is not IProcessorDebugBackend ||
                 context.PreparedPlan is not IDebugBackendExecutablePlan))
            {
                throw new NotSupportedException("The selected backend does not support per-session debug execution.");
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
    public static ProcessorPipelineBuilder UsePrefillOptimizers(
        this ProcessorPipelineBuilder builder, params IPrefillGraphOptimizer[] optimizers)
    {
        ArgumentNullException.ThrowIfNull(optimizers);
        if (optimizers.Length == 0 || optimizers.Any(static optimizer => optimizer is null))
            throw new ArgumentException("At least one non-null prefill optimizer is required.", nameof(optimizers));
        return builder.AddStep("prefill-optimizers",
            context =>
            {
                foreach (var optimizer in optimizers) context.PrefillOptimizers.Add(optimizer);
            });
    }

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
        metadataReader ??= new CatalogArchitectureMetadataReader();
        return builder.AddStep(
            "reader",
            context =>
            {
                context.File = reader.Open(context.Path)
                    ?? throw new InvalidOperationException("The reader returned no model file.");
                context.Metadata = metadataReader.Read(context.File);
            },
            context => context.DisposeOnce(context.File));
    }

    public static ProcessorPipelineBuilder UseXmlMetadata(
        this ProcessorPipelineBuilder builder, IModelReader reader, string path,
        XmlArchitectureGraphKind kind) =>
        builder.UseReader(reader, new XmlArchitectureMetadataReader(path, kind));

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
            if (provider.ArchitectureId != context.Metadata?.ArchitectureId)
            {
                throw new InvalidOperationException("The graph provider does not match the model architecture.");
            }

            context.LogicalGraph = provider.Build(catalog);
        });

    public static ProcessorPipelineBuilder UseExecutionGraph(
        this ProcessorPipelineBuilder builder, Func<ProcessorBuildContext, ExecutionGraph> source) =>
        builder.AddStep("execution-source", context =>
        {
            ArgumentNullException.ThrowIfNull(source);
            context.ExecutionGraph = source(context) ?? throw new InvalidOperationException("The execution source returned no graph.");
        });

    public static ProcessorPipelineBuilder UseExecutionXml(
        this ProcessorPipelineBuilder builder, string xml)
    {
        ArgumentNullException.ThrowIfNull(xml);
        return builder.UseExecutionGraph(context =>
        {
            context.LogicalGraph = null;
            return GraphXml.DeserializeExecution(xml);
        });
    }

    public static ProcessorPipelineBuilder UseXmlLogicalGraph(
        this ProcessorPipelineBuilder builder, string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return builder.AddStep("xml-logical", context =>
        {
            context.LogicalGraph = GraphXml.DeserializeLogical(System.IO.File.ReadAllText(path));
            context.ExecutionGraph = null;
            context.PrefillExecutionGraph = null;
            context.PrefillPreparedPlan = null;
        });
    }

    public static ProcessorPipelineBuilder UseXmlExecutionGraph(
        this ProcessorPipelineBuilder builder, string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return builder.AddStep("xml-execution", context =>
        {
            context.LogicalGraph = null;
            context.ExecutionGraph = GraphXml.DeserializeExecution(System.IO.File.ReadAllText(path));
            context.PrefillExecutionGraph = null;
            context.PrefillPreparedPlan = null;
        });
    }

    public static ProcessorPipelineBuilder UseBackend(
        this ProcessorPipelineBuilder builder, IExecutionGraphBackend backend,
        GraphOptimizationOptions? options = null) =>
        builder.UseBackend(_ => backend, options);

    public static ProcessorPipelineBuilder UseBackend(
        this ProcessorPipelineBuilder builder,
        Func<ProcessorBuildContext, IExecutionGraphBackend> factory,
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

            context.ExecutionGraph ??= context.LogicalGraph is { } logical
                ? new GraphOptimizer().Optimize(logical, options ?? new GraphOptimizationOptions(OptimizationBoundary.Unrestricted),
                    backend.KernelCatalog, backend as IFusedOperatorProvider)
                : throw new InvalidOperationException("A logical graph or direct execution graph is required.");
            ValidateGraph(context.ExecutionGraph.Identity, context.ExecutionGraph.Model,
                context.ExecutionGraph.Resources, context);

            context.PreparedPlan = context.Debug
                ? GraphDebugSession.PrepareDebugPlan(backend, context.ExecutionGraph)
                : backend.Prepare(context.ExecutionGraph)
                    .GetPlanOrThrow(backend.KernelCatalog.BackendId);
            if (!ReferenceEquals(context.PreparedPlan.Graph, context.ExecutionGraph))
            {
                throw new InvalidOperationException("The backend prepared a plan for a different execution graph.");
            }
            if (!context.Debug && context.PrefillOptimizers.Count > 0 &&
                backend is IProcessorPrefillBackend)
            {
                IPrefillGraphOptimizer[] candidates = context.PrefillOptimizers.ToArray();
                var prefillContext = new GraphPrefillOptimizationContext(
                    context.LogicalGraph, context.ExecutionGraph, context.ExecutionGraph.Model, context.Catalog);
                var matches = candidates.Where(candidate => candidate.CanOptimize(prefillContext)).ToArray();
                if (matches.Length > 1)
                    throw new InvalidOperationException("Multiple prefill graph optimizers matched the inference graph.");
                if (matches.Length == 1)
                {
                    context.PrefillExecutionGraph = matches[0].Optimize(prefillContext);
                    ValidateGraph(context.PrefillExecutionGraph.Identity, context.PrefillExecutionGraph.Model,
                        context.PrefillExecutionGraph.Resources, context);
                    ValidatePrefillCompatibility(context.ExecutionGraph, context.PrefillExecutionGraph);
                }
            }
        }, context => context.DisposeOnce(context.Backend));

    private static void ValidatePrefillCompatibility(ExecutionGraph inference, ExecutionGraph prefill)
    {
        if (inference.Identity.ArchitectureId != prefill.Identity.ArchitectureId ||
            inference.Model != prefill.Model ||
            inference.GraphState is not { } inferenceState ||
            prefill.GraphState is not { } prefillState ||
            inferenceState.Slots.Count == 0 ||
            !inferenceState.Schema.IsCompatibleWith(prefillState.Schema))
            throw new InvalidDataException("The prefill graph and inference graph do not share a StateSchema.");

        EnsureCompleteState(inference, inferenceState);
        EnsureCompleteState(prefill, prefillState);
        var originalState = inferenceState.Slots.Select(slot =>
            (slot.Name, Tensor: inference.Resources.Single(resource => resource.Id == slot.Resource).Tensor)).ToArray();
        var derivedState = prefillState.Slots.Select(slot =>
            (slot.Name, Tensor: prefill.Resources.Single(resource => resource.Id == slot.Resource).Tensor)).ToArray();
        if (originalState.Length != derivedState.Length ||
            originalState.Zip(derivedState).Any(pair =>
                pair.First.Name != pair.Second.Name ||
                pair.First.Tensor.ElementType != pair.Second.Tensor.ElementType ||
                pair.First.Tensor.Layout != pair.Second.Tensor.Layout ||
                !pair.First.Tensor.Dimensions.SequenceEqual(pair.Second.Tensor.Dimensions)))
            throw new InvalidDataException("The prefill graph state tensors differ from inference.");
        ValidateEndpoints(inference.Inputs, prefill.Inputs, checkShape: false, "input");
        ValidateEndpoints(inference.Outputs, prefill.Outputs, checkShape: true, "output");

        static void EnsureCompleteState(ExecutionGraph graph, GraphState state)
        {
            var declared = state.Slots.Select(slot => slot.Resource).ToHashSet();
            if (graph.Resources.Any(resource => resource.Kind == GraphResourceKind.SessionState &&
                    !declared.Contains(resource.Id)))
                throw new InvalidDataException("The prefill state schema must cover every session-state resource.");
        }

        void ValidateEndpoints(
            IReadOnlyList<ResourceId> inferenceIds, IReadOnlyList<ResourceId> prefillIds,
            bool checkShape, string direction)
        {
            if (inferenceIds.Count == 0 || inferenceIds.Count != prefillIds.Count)
                throw new InvalidDataException($"The prefill graph has incompatible {direction} resources.");
            var original = inferenceIds.Select(id => inference.Resources.Single(resource => resource.Id == id))
                .OrderBy(resource => resource.Name, StringComparer.Ordinal).ToArray();
            var derived = prefillIds.Select(id => prefill.Resources.Single(resource => resource.Id == id))
                .OrderBy(resource => resource.Name, StringComparer.Ordinal).ToArray();
            for (var index = 0; index < original.Length; index++)
            {
                if (original[index].Name != derived[index].Name ||
                    original[index].Kind != derived[index].Kind ||
                    original[index].Tensor.ElementType != derived[index].Tensor.ElementType ||
                    checkShape &&
                    (original[index].Tensor.Layout != derived[index].Tensor.Layout ||
                     !original[index].Tensor.Dimensions.SequenceEqual(derived[index].Tensor.Dimensions)))
                    throw new InvalidDataException($"The prefill graph has an incompatible {direction} resource.");
            }
        }
    }

    private static void ValidateGraph(
        GraphIdentity identity, GraphModelSignature signature,
        IReadOnlyList<GraphResource> resources, ProcessorBuildContext context)
    {
        var metadata = context.Metadata ?? throw new InvalidOperationException("The reader must run before the graph.");
        var catalog = context.Catalog ?? throw new InvalidOperationException("The reader must run before the graph.");
        if (identity.ArchitectureId != metadata.ArchitectureId ||
            signature.VocabularySize != metadata.VocabularySize ||
            signature.EmbeddingSize != metadata.EmbeddingSize ||
            signature.LayerCount != metadata.LayerCount ||
            signature.HeadCount != metadata.HeadCount ||
            signature.HeadSize != metadata.HeadSize ||
            signature.StateAbiId != $"{metadata.ArchitectureId}.state.fp32@1")
        {
            throw new InvalidDataException("The graph model signature does not match the loaded model.");
        }

        foreach (var resource in resources.Where(static resource => resource.Kind == GraphResourceKind.Weight))
        {
            var tensor = catalog.GetRequired(resource.BindingKey
                ?? throw new InvalidDataException($"Weight resource '{resource.Id}' requires a binding key."));
            var elementType = tensor.DataType switch
            {
                RwkvTensorDataType.Float32 => GraphElementType.Float32,
                RwkvTensorDataType.Float16 => GraphElementType.Float16,
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

    private static ProcessorPipelineBuilder UseArchitecture(
        this ProcessorPipelineBuilder builder, Func<ProcessorBuildContext, IRwkvArchitecture> factory) =>
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
            if (context.PrefillExecutionGraph is { } prefillGraph &&
                context.Backend is IProcessorPrefillBackend prefillBackend)
            {
                if (prefillBackend.CanPreparePrefill(context.ExecutionGraph!))
                {
                    context.PrefillPreparedPlan = prefillBackend.Prepare(prefillGraph)
                        .GetPlanOrThrow(prefillBackend.KernelCatalog.BackendId);
                    if (!ReferenceEquals(context.PrefillPreparedPlan.Graph, prefillGraph))
                        throw new InvalidOperationException("The backend prepared a plan for a different prefill graph.");
                }
                else
                {
                    context.PrefillExecutionGraph = null;
                }
            }
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

public sealed class Processor : IDisposable
{
    private readonly ProcessorBuildContext context;
    private readonly IReadOnlyList<IProcessorBuildStep> steps;
    private readonly IRwkvArchitecture architecture;
    private readonly IRwkvModel model;
    private readonly IExecutionGraphBackend? backend;
    private readonly bool debug;
    private bool disposed;

    internal Processor(ProcessorBuildContext context, IReadOnlyList<IProcessorBuildStep> steps)
    {
        this.context = context;
        this.steps = steps.ToArray();
        architecture = context.Architecture!;
        model = context.Model!;
        backend = context.Backend;
        debug = context.Debug;
        Metadata = context.Metadata!;
        LogicalGraph = context.LogicalGraph;
        ExecutionGraph = context.ExecutionGraph;
        PreparedPlan = context.PreparedPlan;
        PrefillExecutionGraph = context.PrefillExecutionGraph;
        PrefillPreparedPlan = context.PrefillPreparedPlan;
    }

    public RwkvModelMetadata Metadata { get; }
    public bool IsDebugMode => debug;
    public LogicalGraph? LogicalGraph { get; }
    public ExecutionGraph? ExecutionGraph { get; }
    public IBackendExecutablePlan? PreparedPlan { get; }
    public ExecutionGraph? InferenceExecutionGraph => ExecutionGraph;
    public IBackendExecutablePlan? InferencePreparedPlan => PreparedPlan;
    public ExecutionGraph? PrefillExecutionGraph { get; }
    public IBackendExecutablePlan? PrefillPreparedPlan { get; }

    public void ExportLogicalGraph(Stream stream)
    {
        ThrowIfDisposed();
        var graph = LogicalGraph ??
            throw new InvalidOperationException("The processor has no logical graph.");
        WriteGraphJson(stream, GraphJson.Serialize(graph));
    }

    public void ExportExecutionGraph(
        Stream stream, ProcessorExecutionGraphKind kind = ProcessorExecutionGraphKind.Inference)
    {
        ThrowIfDisposed();
        if (!Enum.IsDefined(kind))
            throw new ArgumentOutOfRangeException(nameof(kind));
        var graph = kind == ProcessorExecutionGraphKind.Inference
            ? InferenceExecutionGraph : PrefillExecutionGraph;
        if (graph is null)
            throw new InvalidOperationException($"The processor has no {kind} execution graph.");
        WriteGraphJson(stream, GraphJson.Serialize(graph));
    }

    private static void WriteGraphJson(Stream stream, string json)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanWrite)
            throw new ArgumentException("The stream must be writable.", nameof(stream));
        using var writer = new StreamWriter(stream, new UTF8Encoding(false), leaveOpen: true);
        writer.Write(json);
    }

    public static Processor Load(string path) =>
        new ProcessorPipelineBuilder(path)
            .UseReader(new GgmlModelReader())
            .UseProvider(context => RwkvRuntimeFactory.CreateGraphProvider(
                context.Metadata?.ArchitectureId ??
                throw new InvalidOperationException("The reader must run before selecting a graph provider.")))
            .UseBackend(_ => SharpInference.Backends.Cpu.CpuPrimitiveGraphBackend.Instance)
            .UsePortableGraphArchitecture()
            .Build();

    public static Processor LoadGraph(
        string path, LogicalGraph graph, IExecutionGraphBackend backend,
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
        string path, ILogicalGraphProvider provider, IExecutionGraphBackend backend,
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

        public RwkvModelMetadata Read(IModelTensorCatalog catalog)
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
    public ProcessorSession CreateSession()
    {
        ThrowIfDisposed();
        return CreateSession(architecture.CreateState(model));
    }

    internal ProcessorSession CreateSession(IRwkvState state)
    {
        ThrowIfDisposed();
        IProcessorSessionExecutor? executor = null;
        GraphDebugSession? debugSession = null;
        try
        {
            if (debug)
            {
                debugSession = new GraphDebugSession();
                executor = ((IProcessorDebugBackend)backend!).CreateDebugSession(
                    model, state, (IDebugBackendExecutablePlan)PreparedPlan!, debugSession)
                    ?? throw new InvalidOperationException("The debug backend returned no session executor.");
            }
            else if (PrefillPreparedPlan is { } prefillPlan)
            {
                executor = (backend as IProcessorPrefillBackend
                    ?? throw new InvalidOperationException("The selected backend cannot execute the prepared prefill plan."))
                    .CreateSessionExecutor(model, state, PreparedPlan!, prefillPlan)
                    ?? throw new InvalidOperationException("The backend returned no dual-graph session executor.");
            }
            else
            {
                executor = backend!.CreateSessionExecutor(model, state, PreparedPlan!)
                    ?? throw new InvalidOperationException("The backend returned no session executor.");
            }

            return new ProcessorSession(this, model, state, executor, debugSession);
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
            try { debugSession?.Dispose(); }
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

public sealed class ProcessorSession : IRwkvGenerationSession, IDisposable
{
    private readonly Processor owner;
    private readonly IRwkvState state;
    private readonly IProcessorSessionExecutor executor;
    private readonly float[] logits;
    private readonly object gate = new();
    private bool disposed;

    internal ProcessorSession(
        Processor owner, IRwkvModel model, IRwkvState state,
        IProcessorSessionExecutor executor, GraphDebugSession? debugSession)
    {
        this.owner = owner;
        this.state = state;
        this.executor = executor;
        DebugSession = debugSession;
        logits = new float[model.Metadata.VocabularySize];
    }

    public bool IsDebugMode => DebugSession is not null;
    public GraphDebugSession? DebugSession { get; }
    public Action<int, float[]>? LayerTrace
    {
        get => (executor as IProcessorLayerTraceExecutor)?.LayerTrace;
        set
        {
            ThrowIfDisposed();
            if (executor is not IProcessorLayerTraceExecutor trace)
                throw new NotSupportedException("The session backend does not support per-layer tracing.");
            trace.LayerTrace = value;
        }
    }

    public event EventHandler<GraphDebugBeforeNodeEventArgs> BeforeNodeCall
    {
        add => RequireDebugSession().BeforeNode += value;
        remove => RequireDebugSession().BeforeNode -= value;
    }

    public event EventHandler<GraphDebugAfterNodeEventArgs> AfterNodeCall
    {
        add => RequireDebugSession().AfterNode += value;
        remove => RequireDebugSession().AfterNode -= value;
    }

    public void Resume() => RequireDebugSession().Resume();
    public void Stop() => RequireDebugSession().Stop();

    private GraphDebugSession RequireDebugSession()
    {
        ThrowIfDisposed();
        return DebugSession ?? throw new NotSupportedException("This processor session is not in debug mode.");
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        owner.ThrowIfDisposed();
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
            var graph = owner.InferenceExecutionGraph ??
                throw new InvalidOperationException("The processor has no inference graph.");
            var stateExecutor = executor as IProcessorStateExecutor ??
                throw new NotSupportedException("The session backend does not support GraphState export.");
            var tensors = stateExecutor.ReadState(graph.GraphState).Select(value =>
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
            var graph = owner.InferenceExecutionGraph ??
                throw new InvalidOperationException("The processor has no inference graph.");
            var stateExecutor = executor as IProcessorStateExecutor ??
                throw new NotSupportedException("The session backend does not support GraphState import.");
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
            stateExecutor.WriteState(graph.GraphState, values);
        }
    }

    public void Dispose()
    {
        if (Volatile.Read(ref disposed)) return;

        // A paused inference holds gate; stopping it first allows the inference thread to unwind.
        if (DebugSession?.State is GraphDebugSessionState.Running or GraphDebugSessionState.Paused)
        {
            try { DebugSession.Stop(); }
            catch (InvalidOperationException) { /* The invocation finished concurrently. */ }
        }

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
            try { DebugSession?.Dispose(); }
            catch (Exception error) { errors.Add(error); }
            if (errors.Count != 0)
                throw new AggregateException("One or more processor session resources failed to dispose.", errors);
        }
    }
}
