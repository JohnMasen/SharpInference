using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class GraphModelMetadataTests
{
    [Fact]
    public void GenericSignature_RoundTripsJsonAndXmlWithoutRwkvDimensions()
    {
        var signature = new GraphModelSignature(
            "multimodal-transformer",
            "multimodal.state@1",
            new Dictionary<string, int> { ["hidden"] = 3072, ["visionWidth"] = 1024 },
            new Dictionary<string, string> { ["precision"] = "float16" });
        var graph = EmptyGraph(signature);

        var json = GraphJson.DeserializeLogical(GraphJson.Serialize(graph));
        var xml = GraphXml.DeserializeLogical(GraphXml.Serialize(graph));

        Assert.Equal(signature, json.Model);
        Assert.Equal(signature, xml.Model);
        Assert.DoesNotContain("vocabulary", signature.Dimensions.Keys);
        Assert.False(RwkvGraphSignatures.IsCompatible(signature));
    }

    [Fact]
    public void RwkvMetadata_AdaptsToGenericMetadataAndLegacySignature()
    {
        var metadata = new RwkvModelMetadata(32, 16, 2, 4, 4, "rwkv-7");
        var generic = metadata.ToModelMetadata();
        var signature = RwkvGraphSignatures.Create(32, 16, 2, 4, 4, "rwkv-7.state@1");

        Assert.Equal("rwkv-7", generic.ArchitectureId);
        Assert.Equal(16, generic.Dimensions["embedding"]);
        Assert.True(RwkvGraphSignatures.IsCompatible(signature));
        Assert.Equal(32, signature.Dimensions["vocabulary"]);
    }

    private static LogicalGraph EmptyGraph(GraphModelSignature signature) =>
        new(
            new GraphIdentity("generic-test", 1, "generic"),
            signature,
            [],
            [new GraphRegion(
                new RegionId("root"), null, GraphRegionTypes.Graph, null, "root",
                new Dictionary<string, string>())],
            [],
            [],
            []);
}
