using System.Text.Json;
using System.Xml.Linq;

namespace SharpInference.Graphs;

/// <summary>Reads model-owned legacy signatures without introducing model semantics into the graph serializer.</summary>
public interface IGraphModelSignatureReader
{
    GraphModelSignature ReadJson(JsonElement model);
    GraphModelSignature ReadXml(XElement model);
}
