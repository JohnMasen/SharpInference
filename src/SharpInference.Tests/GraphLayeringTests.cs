using System.Reflection;
using System.Runtime.CompilerServices;
using SharpInference.Graphs.Internal;
using SharpInference.Graphs.Logical;
using SharpInference.Vm.Optimization.GraphHelper;

namespace SharpInference.Tests;

public sealed class GraphLayeringTests
{
    [Fact]
    public void GraphsDependsOnlyOnCoreAndInstructionsWithinTheRepository()
    {
        var assembly = typeof(LogicalGraphDefinition).Assembly;
        var references = assembly.GetReferencedAssemblies()
            .Select(reference => reference.Name!)
            .Where(name => name.StartsWith("SharpInference.", StringComparison.Ordinal))
            .Order(StringComparer.Ordinal);

        Assert.Equal(["SharpInference.Graphs.Core", "SharpInference.Instructions"], references);
        Assert.DoesNotContain(assembly.GetCustomAttributes<InternalsVisibleToAttribute>(),
            attribute => attribute.AssemblyName.StartsWith("SharpInference.Vm.Optimization",
                StringComparison.Ordinal));
    }

    [Fact]
    public void ExecutionTypesAreOwnedByOptimizationGraphHelper()
    {
        Type[] types =
        [
            typeof(DefaultExecutionGraphConverter), typeof(ExecutionDependencyEdge),
            typeof(ExecutionGraph), typeof(ExecutionGraphBuilder), typeof(ExecutionGraphCallNode),
            typeof(ExecutionGraphCallRelation), typeof(ExecutionGraphCompilationContext),
            typeof(ExecutionGraphCompiler), typeof(ExecutionGraphEdge), typeof(ExecutionGraphId),
            typeof(ExecutionGraphNode), typeof(ExecutionGraphPostprocessContext),
            typeof(ExecutionGraphProgram), typeof(ExecutionGraphResource), typeof(ExecutionGraphRewriter),
            typeof(ExecutionGraphSource), typeof(ExecutionGraphStructureNode), typeof(ExecutionGraphXml),
            typeof(ExecutionNodeId), typeof(ExecutionOperationNode), typeof(ExecutionResourceNode),
            typeof(ExecutionResourceReadEdge), typeof(ExecutionResourceWriteEdge),
            typeof(ExecutionSourceMap), typeof(GraphArchitecture), typeof(IExecutionGraphConverter),
            typeof(IExecutionGraphPostprocessor),
        ];

        Assert.Equal(27, types.Length);
        Assert.All(types, type =>
        {
            Assert.Equal("SharpInference.Vm.Optimization", type.Assembly.GetName().Name);
            Assert.Equal("SharpInference.Vm.Optimization.GraphHelper", type.Namespace);
            Assert.Null(typeof(LogicalGraphDefinition).Assembly.GetType(
                $"SharpInference.Graphs.Execution.{type.Name}"));
        });
    }

    [Fact]
    public void SharedHelpersArePublicWithoutExposingPrivateImplementation()
    {
        Assert.True(typeof(ModelGraphConstruction).IsPublic);
        Assert.True(typeof(ModelGraphXml).IsPublic);
        Assert.True(typeof(GraphXmlSupport).IsPublic);
        Assert.False(typeof(GraphXmlSupport).GetField("MaxCharacters",
            BindingFlags.Static | BindingFlags.NonPublic)!.IsPublic);
        Assert.False(typeof(ModelGraphXml).GetMethod("OperationName",
            BindingFlags.Static | BindingFlags.NonPublic)!.IsPublic);
    }
}
