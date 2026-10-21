using System.Collections.ObjectModel;

namespace SharpInference.Graphs;

/// <summary>A collection of same-stage definitions with an explicitly selected entry graph.</summary>
public sealed class GraphPackage<TGraph> : Collection<TGraph> where TGraph : Graph
{
    public Dictionary<string, string> Metadata { get; } = new(StringComparer.Ordinal);
    public string? MainGraphId { get; set; }
    public TGraph? MainGraph => this.SingleOrDefault(graph => graph.Id == MainGraphId);
    public IEnumerable<Resource> GraphIOs
    {
        get
        {
            foreach (var resource in MainGraph?.GraphIOs ?? [])
                yield return resource;
        }
    }
    public IEnumerable<Resource> Inputs => GraphIOs.Where(resource => resource.GraphIO is GraphIO.In or GraphIO.InOut);
    public IEnumerable<Resource> Outputs => GraphIOs.Where(resource => resource.GraphIO is GraphIO.Out or GraphIO.InOut);

    public GraphPackage() { }

    public GraphPackage(TGraph main, params TGraph[] definitions)
    {
        MainGraphId = main.Id;
        Add(main);
        foreach (var graph in definitions)
            Add(graph);
    }

    public IReadOnlyList<GraphDiagnostic> Validate() =>
        GraphValidation.ValidatePackage(this, MainGraphId);

    public GraphPackage<TGraph> Snapshot()
    {
        var copy = new GraphPackage<TGraph> { MainGraphId = MainGraphId };
        foreach (var pair in Metadata)
            copy.Metadata.Add(pair.Key, pair.Value);
        foreach (var graph in this)
        {
            if (graph.Snapshot() is not TGraph snapshot)
                throw new InvalidOperationException($"Snapshot of '{graph.Id}' did not preserve {typeof(TGraph).Name}.");
            copy.Add(snapshot);
        }
        return copy;
    }
}
