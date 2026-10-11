using System.Runtime.InteropServices;
using System.Text;
using SharpInference.Gguf;
using SharpInference.Graphs;
using SharpInference.Vm;

namespace SharpInference.Runtime;

/// <summary>Opens GGML model files through the model-reader abstraction.</summary>
public sealed class GgmlModelReader : IModelReader
{
    /// <inheritdoc/>
    public IModelFile Open(string path) => GgmlModelFile.Open(path);
}

/// <summary>Extracts and validates architecture metadata from a model tensor catalog.</summary>
public interface IArchitectureMetadataReader
{
    /// <summary>Reads architecture metadata using the supplied tensor catalog.</summary>
    /// <param name="catalog">The model tensor catalog used for architecture resolution and weight validation.</param>
    ModelMetadata Read(IModelTensorCatalog catalog);
}

/// <summary>Reads a logical graph from XML and validates its weights against a tensor catalog.</summary>
public sealed class XmlArchitectureMetadataReader : IArchitectureMetadataReader
{
    private readonly string path;

    /// <summary>Creates a metadata reader for a logical-graph XML file.</summary>
    /// <param name="path">The path of the logical-graph XML file.</param>
    public XmlArchitectureMetadataReader(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        this.path = path;
    }

    /// <inheritdoc/>
    public ModelMetadata Read(IModelTensorCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        var graph = GraphXml.DeserializeLogical(System.IO.File.ReadAllText(path));
        return new GraphArchitectureMetadataReader(graph).Read(catalog);
    }
}

/// <summary>Resolves a registered model module and reads architecture metadata from a tensor catalog.</summary>
/// <param name="modules">The registry of supported model architecture modules.</param>
public sealed class CatalogArchitectureMetadataReader(ModelGraphModuleRegistry modules) : IArchitectureMetadataReader
{
    /// <inheritdoc/>
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

/// <summary>Derives architecture metadata from a logical graph after checking its weight tensor descriptors.</summary>
public sealed class GraphArchitectureMetadataReader : IArchitectureMetadataReader
{
    private readonly GraphIdentity identity;
    private readonly GraphModelSignature signature;
    private readonly IReadOnlyList<GraphResource> resources;

    /// <summary>Captures the logical graph&apos;s identity, model signature, and weight requirements.</summary>
    /// <param name="graph">The logical graph whose tensor and model requirements are used.</param>
    public GraphArchitectureMetadataReader(LogicalGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        identity = graph.Identity;
        signature = graph.Model;
        resources = graph.Resources;
    }

    /// <inheritdoc/>
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

/// <summary>Provides an existing logical graph without regenerating it from model tensors.</summary>
public sealed class SuppliedLogicalGraphProvider : ILogicalGraphProvider
{
    private readonly LogicalGraph graph;

    /// <summary>Creates a provider for the supplied logical graph.</summary>
    /// <param name="graph">The logical graph whose tensor and model requirements are used.</param>
    public SuppliedLogicalGraphProvider(LogicalGraph graph) =>
        this.graph = graph ?? throw new ArgumentNullException(nameof(graph));

    /// <inheritdoc/>
    public string ArchitectureId => graph.Identity.ArchitectureId;
    /// <summary>Returns the supplied graph after checking that a tensor catalog was provided.</summary>
    /// <param name="tensors">The tensor catalog supplying model weights and their descriptors.</param>
    public LogicalGraph Build(IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        return graph;
    }
}

/// <summary>Carries model resources and intermediate results through processor build and cleanup steps.</summary>
public sealed class ProcessorBuildContext
{
    private readonly HashSet<object> disposed = new(ReferenceEqualityComparer.Instance);

    internal ProcessorBuildContext(string path)
    {
        Path = path;
    }

    /// <summary>Gets the model file path supplied to the pipeline.</summary>
    public string Path { get; }
    /// <summary>Gets or sets the opened model file until the build releases its source.</summary>
    public IModelFile? File { get; set; }
    internal OwnedModelTensorCatalog? OwnedCatalog { get; set; }
    /// <summary>Gets the owned tensor catalog when available, otherwise the opened model file.</summary>
    public IModelTensorCatalog? Catalog => (IModelTensorCatalog?)OwnedCatalog ?? File;
    /// <summary>Gets or sets the architecture metadata read from the model or graph.</summary>
    public ModelMetadata? Metadata { get; set; }
    /// <summary>Gets or sets the model-specific graph module used for validation and lowering.</summary>
    public IModelGraphModule? ModelModule { get; set; }
    /// <summary>Gets or sets the logical graph selected for the model.</summary>
    public LogicalGraph? LogicalGraph { get; set; }
    /// <summary>Gets or sets the compiled backend plan prepared during the build.</summary>
    public VmCompiledPlan? PreparedPlan { get; set; }
    /// <summary>Gets or sets the VM backend that prepares and executes the model.</summary>
    public VmGraphBackend? Backend { get; set; }
    /// <summary>Gets or sets the architecture adapter used to bind the model.</summary>
    public IModelArchitecture? Architecture { get; set; }
    /// <summary>Gets or sets the bound model produced by the architecture adapter.</summary>
    public IModel? Model { get; set; }

    /// <summary>Disposes a resource at most once by object identity when it implements synchronous disposal.</summary>
    /// <param name="resource">The name of the orchestration resource.</param>
    public void DisposeOnce(object? resource)
    {
        if (resource is IDisposable disposable && disposed.Add(resource))
        {
            disposable.Dispose();
        }
    }
}

/// <summary>Defines a named processor initialization step and its matching cleanup action.</summary>
public interface IProcessorBuildStep
{
    /// <summary>Gets the unique name of this pipeline step.</summary>
    string Name { get; }
    /// <summary>Initializes this step&apos;s resources in the shared build context.</summary>
    /// <param name="context">The shared build context containing resources and intermediate results.</param>
    void Execute(ProcessorBuildContext context);
    /// <summary>Releases this step&apos;s resources from the shared build context.</summary>
    /// <param name="context">The shared build context containing resources and intermediate results.</param>
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

/// <summary>Builds a processor through ordered, uniquely named steps with reverse-order failure cleanup.</summary>
public sealed class ProcessorPipelineBuilder
{
    private readonly List<IProcessorBuildStep> steps = [];
    private readonly HashSet<string> names = new(StringComparer.Ordinal);

    /// <summary>Creates a build pipeline for a non-empty model file path.</summary>
    /// <param name="path">The model file path to open or build.</param>
    public ProcessorPipelineBuilder(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        Path = path;
    }

    /// <summary>Gets the model file path used when the pipeline runs.</summary>
    public string Path { get; }
    /// <summary>Gets the registered step names in execution order.</summary>
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

    /// <summary>Runs the pipeline, validates the bound model, releases the source file, and returns the initialized processor.</summary>
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

/// <summary>Adds model reading, graph selection, backend preparation, and architecture binding steps to processor pipelines.</summary>
public static class ProcessorPipelineExtensions
{
    /// <summary>Appends a uniquely named build step with its execution and optional cleanup actions.</summary>
    /// <param name="builder">The processor pipeline being extended.</param>
    /// <param name="name">The unique pipeline step name.</param>
    /// <param name="execute">The callback that performs the step&apos;s initialization.</param>
    public static ProcessorPipelineBuilder AddStep(
        this ProcessorPipelineBuilder builder, string name, Action<ProcessorBuildContext> execute) =>
        builder.AddStep(name, execute, static _ => { });

    /// <summary>Appends a uniquely named build step with its execution and optional cleanup actions.</summary>
    /// <param name="builder">The processor pipeline being extended.</param>
    /// <param name="name">The unique pipeline step name.</param>
    /// <param name="execute">The callback that performs the step&apos;s initialization.</param>
    /// <param name="dispose">The cleanup callback for resources initialized by the step.</param>
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

    /// <summary>Appends a uniquely named build step with its execution and optional cleanup actions.</summary>
    /// <param name="builder">The processor pipeline being extended.</param>
    /// <param name="step">The named build step with execution and cleanup behavior.</param>
    public static ProcessorPipelineBuilder AddStep(
        this ProcessorPipelineBuilder builder, IProcessorBuildStep step) =>
        (builder ?? throw new ArgumentNullException(nameof(builder))).Append(step);

    /// <summary>Adds model-file loading and optional architecture metadata reading to the pipeline.</summary>
    /// <param name="builder">The processor pipeline being extended.</param>
    /// <param name="reader">The reader used to open the model file.</param>
    /// <param name="metadataReader">The optional reader that derives architecture metadata from the opened model file.</param>
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

    /// <summary>Adds model-file loading with metadata derived from a logical-graph XML file.</summary>
    /// <param name="builder">The processor pipeline being extended.</param>
    /// <param name="reader">The reader used to open the model file.</param>
    /// <param name="path">The path of the logical-graph XML file used to derive metadata.</param>
    public static ProcessorPipelineBuilder UseXmlMetadata(
        this ProcessorPipelineBuilder builder, IModelReader reader, string path) =>
        builder.UseReader(reader, new XmlArchitectureMetadataReader(path));

    /// <summary>Adds logical-graph generation using a supplied provider or a context-dependent provider factory.</summary>
    /// <param name="builder">The processor pipeline being extended.</param>
    /// <param name="provider">The logical graph provider used to build the model graph.</param>
    public static ProcessorPipelineBuilder UseProvider(
        this ProcessorPipelineBuilder builder, ILogicalGraphProvider provider) =>
        builder.UseProvider(_ => provider);

    /// <summary>Adds logical-graph generation using a supplied provider or a context-dependent provider factory.</summary>
    /// <param name="builder">The processor pipeline being extended.</param>
    /// <param name="factory">The context-dependent factory supplying the logical graph provider.</param>
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

    /// <summary>Adds logical-graph loading from XML and reads graph metadata when it has not already been supplied.</summary>
    /// <param name="builder">The processor pipeline being extended.</param>
    /// <param name="path">The path of the logical-graph XML file to load.</param>
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

    /// <summary>Adds VM backend preparation and registers backend cleanup for the selected logical graph.</summary>
    /// <param name="builder">The processor pipeline being extended.</param>
    /// <param name="backend">The VM backend that prepares and executes the graph.</param>
    /// <param name="options">The optional graph optimization settings passed to backend preparation.</param>
    public static ProcessorPipelineBuilder UseBackend(
        this ProcessorPipelineBuilder builder, VmGraphBackend backend,
        GraphOptimizationOptions? options = null) =>
        builder.UseBackend(_ => backend, options);

    /// <summary>Adds VM backend preparation and registers backend cleanup for the selected logical graph.</summary>
    /// <param name="builder">The processor pipeline being extended.</param>
    /// <param name="factory">The context-dependent factory supplying the VM graph backend.</param>
    /// <param name="options">The optional graph optimization settings passed to backend preparation.</param>
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

    /// <summary>Adds portable graph architecture binding using the prepared backend and compiled plan.</summary>
    /// <param name="builder">The processor pipeline being extended.</param>
    public static ProcessorPipelineBuilder UsePortableGraphArchitecture(this ProcessorPipelineBuilder builder) =>
        builder.UseArchitecture(context => new PortableGraphArchitecture(
            context.Backend ?? throw new InvalidOperationException("The backend must be prepared before binding the graph."),
            context.PreparedPlan ?? throw new InvalidOperationException("The graph must be prepared before binding the model.")));

}

/// <summary>Selects the inference or prefill program when exporting processor execution artifacts.</summary>
public enum ProcessorExecutionGraphKind
{
    /// <summary>Selects the single-token inference program.</summary>
    Inference,
    /// <summary>Selects the prefill program, falling back to inference when no separate prefill plan exists.</summary>
    Prefill,
}

/// <summary>Owns a built model, compiled VM plan, backend, and the pipeline resources used to initialize them.</summary>
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

    /// <inheritdoc/>
    public ModelMetadata Metadata { get; }
    /// <inheritdoc/>
    public ProcessorCapabilities Capabilities { get; }
    /// <summary>Gets the logical graph used to build this processor.</summary>
    public LogicalGraph? LogicalGraph { get; }
    /// <summary>Gets the compiled plan that binds model resources to VM execution.</summary>
    public VmCompiledPlan PreparedPlan { get; }

    /// <summary>Writes the processor&apos;s logical graph as JSON to a writable stream without closing it.</summary>
    /// <param name="stream">The writable destination for logical-graph JSON; the stream remains open.</param>
    public void ExportLogicalGraph(Stream stream)
    {
        ThrowIfDisposed();
        var graph = LogicalGraph ??
            throw new InvalidOperationException("The processor has no logical graph.");
        WriteGraphText(stream, GraphJson.Serialize(graph));
    }

    /// <summary>Gets the VM program used for single-token inference.</summary>
    public VmProgram InferenceProgram => PreparedPlan.Program;
    /// <summary>Gets the prefill VM program, falling back to the inference program when no separate plan exists.</summary>
    public VmProgram PrefillProgram => PreparedPlan.PrefillPlan?.Program ?? PreparedPlan.Program;

    /// <summary>Exports the selected compiled backend artifact into a directory.</summary>
    /// <param name="directory">The destination directory for the compiled artifact.</param>
    /// <param name="kind">The inference or prefill program to export.</param>
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

    /// <summary>Writes the selected VM program as XML to a writable stream without closing it.</summary>
    /// <param name="stream">The writable destination for VM program XML; the stream remains open.</param>
    /// <param name="kind">The inference or prefill program to export.</param>
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

    /// <summary>Writes the authoritative target execution plan as XML without closing the destination stream.</summary>
    /// <param name="stream">The writable destination for execution-plan XML; the stream remains open.</param>
    public void ExportExecutionPlan(Stream stream)
    {
        ThrowIfDisposed();
        var executionPlan = PreparedPlan.ExecutionPlan ??
            throw new NotSupportedException("A supplied VM program does not have an authoritative target execution plan.");
        WriteGraphText(stream, ExecutionPlanXml.Serialize(executionPlan));
    }

    private static void WriteGraphText(Stream stream, string text)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanWrite)
            throw new ArgumentException("The stream must be writable.", nameof(stream));
        using var writer = new StreamWriter(stream, new UTF8Encoding(false), leaveOpen: true);
        writer.Write(text);
    }

    /// <summary>Loads a model using registered architecture modules and prepares it on the supplied backend.</summary>
    /// <param name="path">The model file path to open or build.</param>
    /// <param name="reader">The reader used to open the model file.</param>
    /// <param name="modules">The registry of supported model architecture modules.</param>
    /// <param name="backend">The VM backend that prepares and executes the graph.</param>
    /// <param name="options">The optional graph optimization settings supported by the backend.</param>
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

    /// <summary>Loads a GGML model and binds it to a supplied logical graph or graph provider on the selected backend.</summary>
    /// <param name="path">The model file path to open or build.</param>
    /// <param name="graph">The logical graph whose tensor and model requirements are used.</param>
    /// <param name="backend">The VM backend that prepares and executes the graph.</param>
    /// <param name="options">The optional graph optimization settings supported by the backend.</param>
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

    /// <summary>Loads a GGML model and binds it to a supplied logical graph or graph provider on the selected backend.</summary>
    /// <param name="path">The model file path to open or build.</param>
    /// <param name="provider">The logical graph provider used to build the model graph.</param>
    /// <param name="backend">The VM backend that prepares and executes the graph.</param>
    /// <param name="options">The optional graph optimization settings supported by the backend.</param>
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
    /// <summary>Creates a tensor graph session with supplied or new state for a processor without the token-session contract.</summary>
    /// <param name="state">The optional existing graph state; null creates fresh state.</param>
    public ProcessorGraphSession CreateGraphSession(GraphTensorState? state = null)
    {
        ThrowIfDisposed();
        if (backend.SupportsTokenSessions)
            throw new NotSupportedException("This processor owns a token-session engine; use CreateSession.");
        var portable = model as PortableGraphModel ??
            throw new NotSupportedException("The bound model does not expose a portable tensor graph.");
        return new ProcessorGraphSession(this, backend.CreateGraphSession(portable.Tensors, state));
    }

    /// <summary>Creates a token-generation session with fresh model state.</summary>
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

    /// <summary>Releases the processor&apos;s build steps in reverse order and aggregates cleanup failures.</summary>
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

/// <summary>Provides exclusive token inference, prefill, generation scopes, and serialized state for a processor-owned VM session.</summary>
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

    /// <summary>Processes one token, updates session state, and returns logits in the session&apos;s reusable buffer.</summary>
    /// <param name="token">The input token identifier to process.</param>
    public ReadOnlyMemory<float> ForwardToken(int token)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            executor.ForwardToken(token, logits);
            return logits;
        }
    }

    /// <summary>Acquires an exclusive generation scope that prevents other operations on this session until released.</summary>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
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

    /// <summary>Queues one token for inference while exclusively owning this session.</summary>
    /// <param name="token">The input token identifier to process.</param>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
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

    /// <summary>Queues prompt tokens for prefill and updates the session&apos;s reusable logits buffer.</summary>
    /// <param name="tokens">The ordered prompt token identifiers to prefill.</param>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
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

    /// <summary>Processes a non-empty prompt token sequence and returns logits from the final token.</summary>
    /// <param name="tokens">The ordered prompt token identifiers to prefill.</param>
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

    /// <summary>Creates an independent processor session with a clone of the current model state.</summary>
    public ProcessorSession Fork()
    {
        lock (gate)
        {
            ThrowIfDisposed();
            return owner.CreateSession(state.Clone());
        }
    }

    /// <summary>Resets model state and clears logits when no exclusive operation is active.</summary>
    public void Reset()
    {
        lock (gate)
        {
            ThrowIfDisposed();
            state.Reset();
            Array.Clear(logits);
        }
    }

    /// <summary>Writes session state as little-endian float32 GGUF tensors without closing the destination stream.</summary>
    /// <param name="stream">The writable destination for the GGUF state snapshot; the stream remains open.</param>
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

    /// <summary>Loads compatible little-endian float32 GGUF state tensors into the session.</summary>
    /// <param name="stream">The readable source of a compatible GGUF state snapshot; the stream remains open.</param>
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

    /// <summary>Releases the VM executor and disposable model state, aggregating cleanup failures.</summary>
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
