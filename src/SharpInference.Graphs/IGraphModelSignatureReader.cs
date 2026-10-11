using System.Text.Json;
using System.Xml.Linq;

namespace SharpInference.Graphs;

/// <summary>Reads model-owned legacy signatures without introducing model semantics into the graph serializer.</summary>
public interface IGraphModelSignatureReader
{
    /// <summary>Reads an architecture-specific legacy model signature from a JSON model element.</summary>
    /// <param name="model">The legacy JSON model element.</param>
    GraphModelSignature ReadJson(JsonElement model);
    /// <summary>Reads an architecture-specific legacy model signature from an XML model element.</summary>
    /// <param name="model">The legacy XML model element.</param>
    GraphModelSignature ReadXml(XElement model);
}
