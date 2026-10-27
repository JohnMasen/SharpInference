using System.Xml.Linq;

namespace SharpInference.Graphs.Logical;

/// <summary>The version-one XML entry point for logical graph packages.</summary>
public static class LogicalGraphPackageXml
{
    private static readonly Serializer Instance = new();

    public static string Serialize(LogicalGraphPackage package) => Instance.Serialize(package);
    public static LogicalGraphPackage Deserialize(string xml) => Instance.Deserialize(xml);

    private sealed class Serializer : GraphPackageXmlSerializer<LogicalGraph, LogicalGraphPackage>
    {
        protected override string Stage => "logical";

        protected override IEnumerable<string> AdditionalAttributes(string elementName) =>
            elementName == "Graph" ? ["compileBoundary"] : [];

        protected override void WriteGraphFields(LogicalGraph graph, XElement element) =>
            element.Add(new XAttribute("compileBoundary", graph.CompileBoundary));

        protected override void ReadGraphFields(LogicalGraph graph, XElement element) =>
            graph.CompileBoundary = EnumOr(element, "compileBoundary", GraphCompileBoundary.Auto);
    }
}
