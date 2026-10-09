using System.Text.Json.Nodes;
using System.Xml.Linq;
using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class LegacyGraphSignatureTests
{
    [Theory]
    [InlineData("json")]
    [InlineData("xml")]
    public void LegacyRwkvSignaturesRequireExplicitModelReader(string format)
    {
        var signature = RwkvGraphSignatures.Create(32, 16, 2, 4, 4, "rwkv-7.state@1");
        var graph = new LogicalGraph(new GraphIdentity("rwkv-7", 1, "legacy"), signature, [],
            [new GraphRegion(new("root"), null, GraphRegionTypes.Graph, null, "Root",
                new Dictionary<string, string>())], [], [], []);
        var reader = new RwkvLegacyGraphSignatureReader();
        if (format == "json")
        {
            var root = JsonNode.Parse(GraphJson.Serialize(graph))!;
            root["Model"] = new JsonObject
            {
                ["VocabularySize"] = 32, ["EmbeddingSize"] = 16, ["LayerCount"] = 2,
                ["HeadCount"] = 4, ["HeadSize"] = 4, ["StateAbiId"] = signature.StateAbiId,
            };
            var legacy = root.ToJsonString();
            Assert.Throws<InvalidDataException>(() => GraphJson.DeserializeLogical(legacy));
            Assert.Equal(signature, GraphJson.DeserializeLogical(legacy, reader).Model);
            root["Model"]!["Unexpected"] = 1;
            Assert.Throws<InvalidDataException>(() => GraphJson.DeserializeLogical(root.ToJsonString(), reader));
        }
        else
        {
            var root = XElement.Parse(GraphXml.Serialize(graph));
            root.Element("Model")!.ReplaceWith(new XElement("Model",
                new XAttribute("vocabularySize", 32), new XAttribute("embeddingSize", 16),
                new XAttribute("layerCount", 2), new XAttribute("headCount", 4),
                new XAttribute("headSize", 4), new XAttribute("stateAbiId", signature.StateAbiId)));
            var legacy = root.ToString();
            Assert.Throws<InvalidDataException>(() => GraphXml.DeserializeLogical(legacy));
            Assert.Equal(signature, GraphXml.DeserializeLogical(legacy, reader).Model);
            root.Element("Model")!.SetAttributeValue("unexpected", 1);
            Assert.Throws<InvalidDataException>(() => GraphXml.DeserializeLogical(root.ToString(), reader));
        }
    }

    [Fact]
    public void GenericGraphSignatureHasNoModelSpecificProjections()
    {
        Assert.Equal(["ModelType", "StateAbiId", "Dimensions", "Attributes"],
            typeof(GraphModelSignature).GetProperties().Select(property => property.Name));
        Assert.All(typeof(GraphModelSignature).GetConstructors(),
            constructor => Assert.Equal(typeof(string), constructor.GetParameters()[0].ParameterType));
    }
}
