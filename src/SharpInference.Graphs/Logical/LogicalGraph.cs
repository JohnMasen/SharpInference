namespace SharpInference.Graphs.Logical;

/// <summary>A computation definition describing model semantics and compilation intent.</summary>
public sealed class LogicalGraph : Graph
{
    public GraphCompileBoundary CompileBoundary { get; set; } = GraphCompileBoundary.Auto;

    protected override Graph CreateSnapshot() => new LogicalGraph { CompileBoundary = CompileBoundary };
}
