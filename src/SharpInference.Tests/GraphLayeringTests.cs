using System.Reflection;
using System.Runtime.CompilerServices;
using SharpInference.Graphs;
using SharpInference.Graphs.Execution;
using SharpInference.Graphs.Logical;
using SharpInference.Vm.Optimization;
using SharpInference.Vm.Optimization.GraphHelper;

namespace SharpInference.Tests;

public sealed class GraphLayeringTests
{
    [Fact]
    public void GraphsDependsOnlyOnCoreAndInstructionsWithinTheRepository()
    {
        var assembly = typeof(LogicalGraph).Assembly;
        var references = assembly.GetReferencedAssemblies().Select(reference => reference.Name!)
            .Where(name => name.StartsWith("SharpInference.", StringComparison.Ordinal)).Order(StringComparer.Ordinal);
        Assert.Equal(["SharpInference.Graphs.Core", "SharpInference.Instructions"], references);
        Assert.DoesNotContain(assembly.GetCustomAttributes<InternalsVisibleToAttribute>(),
            attribute => attribute.AssemblyName.StartsWith("SharpInference.Vm.Optimization", StringComparison.Ordinal));
    }

    [Fact]
    public void EditableBusinessAndExecutionTypesBelongToGraphs()
    {
        Type[] types =
        [
            typeof(SharpInference.Graphs.Graph), typeof(GraphPackage<>), typeof(GraphAction), typeof(Resource),
            typeof(Binding), typeof(GraphFlow), typeof(GraphPackageXml), typeof(LogicalGraph),
            typeof(ExecutionGraph), typeof(ExecutionOperation), typeof(ExecutionCall),
            typeof(ExecutionStorage), typeof(ExecutionImplementation), typeof(SourceOrigin),
        ];
        Assert.All(types, type => Assert.Same(typeof(LogicalGraph).Assembly, type.Assembly));
        Assert.All(types.Where(type => type.Namespace == "SharpInference.Graphs.Execution"), type =>
            Assert.Null(typeof(ExecutionGraphCompiler).Assembly.GetType(
                $"SharpInference.Vm.Optimization.GraphHelper.{type.Name}")));
    }

    [Fact]
    public void ProcessingAndDependencyInferenceRemainOutsideTheBusinessModel()
    {
        Type[] types =
        [
            typeof(DefaultExecutionGraphConverter), typeof(ExecutionGraphCompilationContext),
            typeof(ExecutionGraphCompiler), typeof(ExecutionPassContext), typeof(ExecutionGraphRewriter),
            typeof(IExecutionGraphConverter), typeof(IExecutionGraphPostprocessor),
            typeof(GraphDependencyAnalyzer), typeof(ExecutionImplementationSelector), typeof(VmExecutionGraphEmitter),
        ];
        Assert.All(types, type => Assert.Equal("SharpInference.Vm.Optimization", type.Assembly.GetName().Name));
        Assert.Null(typeof(LogicalGraph).Assembly.GetType("SharpInference.Graphs.GraphDependencyAnalyzer"));
    }

    [Fact]
    public void XmlSafetyImplementationDoesNotExposeItsPrivateLimits()
    {
        var support = typeof(LogicalGraph).Assembly.GetType("SharpInference.Graphs.Internal.GraphXmlSupport", true)!;
        Assert.False(support.GetField("MaxCharacters", BindingFlags.Static | BindingFlags.NonPublic)!.IsPublic);
    }
}
