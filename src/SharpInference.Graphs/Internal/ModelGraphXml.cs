using SharpInference.Instructions;
using System.Xml.Linq;
using SharpInference.Graphs.Core;
using SharpInference.Graphs.Logical;
using static SharpInference.Graphs.Internal.GraphXmlSupport;

namespace SharpInference.Graphs.Internal;

/// <summary>Encodes each resource binding once, preserving explicit edge identities and order.</summary>
public static class ModelGraphXml
{
    public static IEnumerable<XElement> WriteRelations<TNode, TEdge>(IGraphView<TNode, TEdge> graph)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge
    {
        var edges = graph.Edges.ToArray();
        var order = edges.Select((edge, index) => (edge.Id, index)).ToDictionary(pair => pair.Id, pair => pair.index);
        var reads = edges.OfType<IResourceReadEdge>()
            .ToDictionary(edge => (Node: edge.Target, edge.BindingId));
        var writes = edges.OfType<IResourceWriteEdge>()
            .ToDictionary(edge => (Node: edge.Source, edge.BindingId));
        var byOperation = edges.Select(edge => edge switch
        {
            IResourceReadEdge read => (Node: read.Target, read.BindingId),
            IResourceWriteEdge write => (Node: write.Source, write.BindingId),
            _ => default,
        }).Where(pair => pair.Node != default).Distinct().ToLookup(pair => pair.Node, pair => pair.BindingId);
        var bindings = new List<XElement>();
        foreach (var operation in graph.Nodes.OfType<IModelOperationNode>())
        {
            foreach (var bindingId in byOperation[operation.Id])
            {
                reads.TryGetValue((operation.Id, bindingId), out var read);
                writes.TryGetValue((operation.Id, bindingId), out var write);
                var resourceNode = (IModelResourceNode)graph.GetNode(read?.Source ?? write!.Target);
                bindings.Add(new XElement("Binding",
                    new XAttribute("id", bindingId),
                    new XAttribute("node", OperationName(operation.Id)),
                    new XAttribute("port", read?.Port ?? write!.Port),
                    new XAttribute("resource", resourceNode.ResourceNodeId.Resource.Value),
                    resourceNode.ResourceNodeId.Owner.IsParent ? null :
                        new XAttribute("instance", resourceNode.ResourceNodeId.Owner.Call!.Value.Value),
                    new XAttribute("access", read is not null && write is not null ? ResourceAccess.ReadWrite :
                        read is not null ? ResourceAccess.Read : ResourceAccess.Write),
                    read is null ? null : new XAttribute("readEdge", read.Id.Value),
                    read is null ? null : new XAttribute("readOrder", order[read.Id]),
                    write is null ? null : new XAttribute("writeEdge", write.Id.Value),
                    write is null ? null : new XAttribute("writeOrder", order[write.Id])));
            }
        }
        yield return new XElement("Bindings", bindings);
        yield return new XElement("Dependencies", edges.OfType<IDependencyEdge>().Select(edge =>
            new XElement("Dependency", new XAttribute("id", edge.Id.Value),
                new XAttribute("source", OperationName(edge.Source)),
                new XAttribute("target", OperationName(edge.Target)),
                new XAttribute("order", order[edge.Id]))));
    }

    public static TEdge[] ReadRelations<TEdge>(XElement graph,
        Func<GraphEdgeId, GraphNodeId, GraphNodeId, string, string, TEdge> read,
        Func<GraphEdgeId, GraphNodeId, GraphNodeId, string, string, TEdge> write,
        Func<GraphEdgeId, GraphNodeId, GraphNodeId, TEdge> dependency)
    {
        var edges = new SortedDictionary<int, TEdge>();
        var ports = new HashSet<(GraphNodeId, string)>();
        foreach (var element in Children(RequiredChild(graph, "Bindings"), "Binding"))
        {
            Check(element, "Binding", "id", "node", "port", "resource", "instance", "access",
                "readEdge", "readOrder", "writeEdge", "writeOrder");
            var id = Required(element, "id");
            var port = Required(element, "port");
            var node = new GraphNodeId($"operation:{Required(element, "node")}");
            if (!ports.Add((node, port)))
                throw new InvalidDataException($"Node '{node}' has duplicate port '{port}'.");
            var instance = (string?)element.Attribute("instance");
            var owner = instance is null ? GraphBindingOwner.Parent :
                GraphBindingOwner.Instance(new GraphDefinitionNodeId(
                    LogicalGraphDefinition.RequireId(instance, "instance")));
            var resource = new ModelResourceNodeId(owner, new(Required(element, "resource"))).GraphId;
            var access = DefinitionEnumValue<ResourceAccess>(element, "access");
            if (access != ResourceAccess.Write)
                Add(Number(element, "readOrder"),
                    read(new(Required(element, "readEdge")), resource, node, port, id));
            else if (element.Attribute("readEdge") is not null || element.Attribute("readOrder") is not null)
                throw new InvalidDataException($"Write-only binding '{id}' contains a read edge.");
            if (access != ResourceAccess.Read)
                Add(Number(element, "writeOrder"),
                    write(new(Required(element, "writeEdge")), node, resource, port, id));
            else if (element.Attribute("writeEdge") is not null || element.Attribute("writeOrder") is not null)
                throw new InvalidDataException($"Read-only binding '{id}' contains a write edge.");
        }
        foreach (var element in Children(RequiredChild(graph, "Dependencies"), "Dependency"))
        {
            Check(element, "Dependency", "id", "source", "target", "order");
            Add(Number(element, "order"), dependency(new(Required(element, "id")),
                new($"operation:{Required(element, "source")}"), new($"operation:{Required(element, "target")}")));
        }
        return edges.Values.ToArray();

        void Add(int position, TEdge edge)
        {
            if (position < 0 || !edges.TryAdd(position, edge))
                throw new InvalidDataException($"Invalid or duplicate relationship order '{position}'.");
        }
    }

    private static string OperationName(GraphNodeId id) => id.Value["operation:".Length..];
}
