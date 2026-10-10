using SharpInference.Graphs;

namespace SharpInference.Tests;

internal static class ModelGraphNamingAssertions
{
    public static IEnumerable<GraphCall> Calls(GraphRegionBody body) =>
        body.Children.SelectMany(element => element switch
        {
            GraphCall call => new[] { call },
            GraphRegionBody region => Calls(region),
            _ => Enumerable.Empty<GraphCall>(),
        });

    public static void Validate(LogicalGraph graph, IGraphOperationValidator? validator = null)
    {
        Check(graph);
        var xml = GraphXml.Serialize(graph);
        var restored = GraphXml.DeserializeLogical(xml);
        Check(restored);
        Assert.Equal(xml, GraphXml.Serialize(restored));
        var json = GraphJson.Serialize(graph);
        var jsonRestored = GraphJson.DeserializeLogical(json);
        Check(jsonRestored);
        Assert.Equal(json, GraphJson.Serialize(jsonRestored));

        void Check(LogicalGraph candidate)
        {
            GraphValidator.Validate(candidate, validator);
            Assert.NotNull(candidate.Structure!.Root.Region.Architecture);
            Assert.Contains(candidate.Regions, region => region.Architecture is not null);
            Assert.All(candidate.Regions.Where(region => region.Type == GraphRegionTypes.Architecture),
                region => Assert.NotNull(region.Architecture));
            var resources = candidate.Resources.Select(resource => resource.Id.Value).ToHashSet(StringComparer.Ordinal);
            var nodes = candidate.Nodes.Select(node => node.Id.Value).ToHashSet(StringComparer.Ordinal);
            Assert.Empty(nodes.Intersect(resources, StringComparer.Ordinal));
            Assert.All(candidate.Nodes, node =>
            {
                Assert.All(node.Resources, binding => Assert.Contains(binding.Resource.Value, resources));
                Assert.All(node.Dependencies, dependency => Assert.Contains(dependency.Value, nodes));
            });
            Assert.All(candidate.Resources.Where(resource => resource.Kind == GraphResourceKind.Temporary), resource =>
            {
                Assert.EndsWith("_output", resource.Id.Value, StringComparison.Ordinal);
                var producer = candidate.Nodes.Last(node => node.Resources.Any(binding =>
                    binding.Resource == resource.Id && binding.Access != GraphResourceAccess.Read));
                Assert.Equal(producer.Id.Value + "_output", resource.Id.Value);
            });
        }
    }
}
