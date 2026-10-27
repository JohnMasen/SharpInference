namespace SharpInference.Graphs.Logical;

/// <summary>A computation definition describing model semantics and compilation intent.</summary>
public sealed class LogicalGraph : Graph
{
    public GraphCompileBoundary CompileBoundary { get; set; } = GraphCompileBoundary.Auto;

    public override IReadOnlyList<GraphDiagnostic> Validate()
    {
        var diagnostics = base.Validate().ToList();
        if (!Enum.IsDefined(CompileBoundary))
        {
            diagnostics.Add(new(GraphDiagnosticSeverity.Error, "G0003", "Unknown compilation boundary.", Graph: Id));
        }
        return diagnostics;
    }

    protected override Graph CreateSnapshot() => new LogicalGraph { CompileBoundary = CompileBoundary };
}
