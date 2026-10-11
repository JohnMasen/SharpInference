using System.Xml;
using System.Xml.Linq;

namespace SharpInference.Graphs;

public sealed class IlExecutionPlan : IExecutionPlanningGraph
{
    public IlExecutionPlan(LogicalGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        GraphValidator.Validate(graph);
        Graph = graph;
        PreservesStructure = true;
        var identities = graph.Nodes.Select((node, index) => (node.Id, Execution: new ExecutionNodeId($"e{index:D4}")))
            .ToDictionary(pair => pair.Id, pair => pair.Execution);
        Nodes = Array.AsReadOnly(graph.Nodes.Select(node => new ExecutionNode(identities[node.Id], node.Operation,
            node.Region, node.Resources, node.Dependencies.Select(dependency => identities[dependency]).ToArray(),
            node.Attributes, node.Requirements, new([node.Id], null))).ToArray());
        Resources = graph.Resources;
        InternalResourceOwners = new System.Collections.ObjectModel.ReadOnlyDictionary<ResourceId, ExecutionNodeId>(
            new Dictionary<ResourceId, ExecutionNodeId>());
    }

    private IlExecutionPlan(IlExecutionPlan source, IEnumerable<GraphResource> resources, IEnumerable<ExecutionNode> nodes)
    {
        Graph = source.Graph;
        Resources = Array.AsReadOnly(resources.ToArray());
        Nodes = Array.AsReadOnly(nodes.ToArray());
        GraphValidator.Validate(this);
        InternalResourceOwners = new System.Collections.ObjectModel.ReadOnlyDictionary<ResourceId, ExecutionNodeId>(
            Nodes.SelectMany(node => node.InternalResources.Select(binding => (binding.Resource, node.Id)))
                .ToDictionary(pair => pair.Resource, pair => pair.Id));
        var resourceIds = Resources.Select(resource => resource.Id).ToHashSet();
        var regions = Regions.Select(region => region.Id).ToHashSet();
        var preceding = new HashSet<ExecutionNodeId>();
        foreach (var node in Nodes)
        {
            if (!regions.Contains(node.Region) || node.Resources.Concat(node.InternalResources).Any(binding => !resourceIds.Contains(binding.Resource)) ||
                node.Dependencies.Any(dependency => !preceding.Contains(dependency)) || !preceding.Add(node.Id))
                throw new InvalidDataException($"Invalid IL node '{node.Id}'.");
        }
    }

    private LogicalGraph Graph { get; }
    public GraphStructure? Structure => PreservesStructure ? Graph.Structure : null;
    private bool PreservesStructure { get; }
    public GraphIdentity Identity => Graph.Identity;
    public GraphModelSignature Model => Graph.Model;
    public IReadOnlyList<GraphResource> Resources { get; }
    public IReadOnlyList<GraphRegion> Regions => Graph.Regions;
    public IReadOnlyList<ExecutionNode> Nodes { get; }
    public IReadOnlyList<ResourceId> Inputs => Graph.Inputs;
    public IReadOnlyList<ResourceId> Outputs => Graph.Outputs;
    public GraphState GraphState => Graph.GraphState;
    public IReadOnlyDictionary<ResourceId, ExecutionNodeId> InternalResourceOwners { get; }

    public IlExecutionPlan WithNodes(IEnumerable<GraphResource> resources, IEnumerable<ExecutionNode> nodes) => new(this, resources, nodes);

    public string Serialize()
    {
        var graph = ToLogicalGraph();
        if (Nodes.Any(node => node.InternalResources.Count != 0 || node.FirstWriteResources.Count != 0))
            throw new InvalidDataException("Private contracted instructions must be serialized as a target plan, not portable IL.");
        return new XElement("ExecutionPlan",
            new XAttribute("schema", "sharp-inference"), new XAttribute("version", 1),
            new XAttribute("kind", "il"), XElement.Parse(GraphXml.Serialize(graph))).ToString();
    }

    public LogicalGraph ToLogicalGraph() => PreservesStructure ? Graph : new LogicalGraph(Identity, Model, Resources, Regions,
                Nodes.Select(node => new LogicalNode(new(node.Id.Value), node.Operation, node.Region, node.Resources,
                    node.Dependencies.Select(dependency => new LogicalNodeId(dependency.Value)).ToArray(),
                    node.Attributes, node.Requirements)), Inputs, Outputs, GraphState);

    public static IlExecutionPlan Deserialize(string xml)
    {
        try
        {
            var root = ExecutionPlanXml.Read(xml);
            ExecutionPlanXml.Header(root);
            if ((string?)root.Attribute("kind") != "il" ||
                root.Attributes().Any(a => a.Name != "schema" && a.Name != "version" && a.Name != "kind") ||
                root.Elements().Count() != 1 || root.Elements().Single().Name != "Graph" ||
                root.Nodes().OfType<XText>().Any(text => !string.IsNullOrWhiteSpace(text.Value)))
                throw new InvalidDataException("Invalid IL execution plan.");
            return new(GraphXml.DeserializeLogical(root.Elements().Single().ToString()));
        }
        catch (Exception error) when (error is XmlException or ArgumentException or FormatException or OverflowException)
        {
            throw new InvalidDataException("Invalid IL execution plan XML.", error);
        }
    }
}
