namespace SharpInference.Graphs.Execution;

/// <summary>An editable execution source package, independent of any backend provider.</summary>
public sealed class ExecutionGraphPackage : GraphPackage<ExecutionGraph>
{
    public const string BackendMetadataKey = "execution.backend";

    public ExecutionGraphPackage()
    {
    }
    public ExecutionGraphPackage(ExecutionGraph main, params ExecutionGraph[] definitions) : base(main, definitions) { }

    public void SetTarget(string backend)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(backend);
        Metadata[BackendMetadataKey] = backend;
    }

    public string? GetTarget() => Metadata.GetValueOrDefault(BackendMetadataKey);

    public override IReadOnlyList<GraphDiagnostic> Validate()
    {
        var diagnostics = base.Validate();
        if (diagnostics.Any(item => item.Severity == GraphDiagnosticSeverity.Error))
        {
            return diagnostics;
        }
        return diagnostics.Concat(AnalyzeDefinitions(diagnostics).Values.SelectMany(result => result.Diagnostics))
            .Distinct().ToArray();
    }

    public IReadOnlyDictionary<string, GraphAnalysis> Analyze() => AnalyzeDefinitions(base.Validate());

    public GraphAnalysis Analyze(ExecutionGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        if (!Contains(graph))
        {
            throw new ArgumentException("The graph is not a definition in this package.", nameof(graph));
        }
        return Analyze()[graph.Id];
    }

    public override ExecutionGraphPackage Snapshot() => (ExecutionGraphPackage)base.Snapshot();
    protected override GraphPackage<ExecutionGraph> CreateSnapshot() => new ExecutionGraphPackage();
}
