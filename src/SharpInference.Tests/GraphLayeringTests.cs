using System.Reflection;
using System.Runtime.CompilerServices;
using SharpInference.Graphs;
using SharpInference.Graphs.Execution;
using SharpInference.Graphs.Logical;
using SharpInference.Instructions;
using SharpInference.Vm.Optimization;
using SharpInference.Vm.Optimization.GraphHelper;

namespace SharpInference.Tests;

public sealed class GraphLayeringTests
{
    [Fact]
    public void GraphAssembliesOnlyDependOnGraphFoundations()
    {
        Assert.Equal(["SharpInference.Graphs.Core"], References(typeof(Graph).Assembly));
        string[] foundations = ["SharpInference.Graphs", "SharpInference.Graphs.Core"];
        foreach (var assembly in new[] { typeof(LogicalGraph).Assembly, typeof(ExecutionGraph).Assembly })
        {
            Assert.Contains("SharpInference.Graphs", References(assembly));
            Assert.All(References(assembly), name => Assert.Contains(name, foundations));
        }
        foreach (var assembly in new[] { typeof(Graph).Assembly, typeof(LogicalGraph).Assembly, typeof(ExecutionGraph).Assembly })
        {
            Assert.DoesNotContain(assembly.GetCustomAttributes<InternalsVisibleToAttribute>(),
                attribute => attribute.AssemblyName.StartsWith("SharpInference.", StringComparison.Ordinal));
        }
    }

    [Fact]
    public void BusinessTypesBelongToTheirOwnAssemblies()
    {
        AssertTypes(typeof(Graph).Assembly, typeof(GraphPackage<>), typeof(GraphAction), typeof(Resource),
            typeof(Binding), typeof(GraphFlow), typeof(GraphPackageXml), typeof(TensorDescriptor),
            typeof(ElementType), typeof(ResourceAccess), typeof(OperationId), typeof(PrecisionRequirement));
        AssertTypes(typeof(LogicalGraph).Assembly, typeof(LogicalGraphPackage), typeof(GraphCompileBoundary), typeof(LogicalGraphPackageXml));
        AssertTypes(typeof(ExecutionGraph).Assembly, typeof(ExecutionGraphPackage), typeof(ExecutionOperation),
            typeof(ExecutionCall), typeof(ExecutionStorage), typeof(ExecutionImplementation), typeof(SourceOrigin),
            typeof(ExecutionGraphRewriter), typeof(ExecutionPlanValidation), typeof(ExecutionGraphPackageXml));
        Assert.DoesNotContain(typeof(Graph).Assembly.GetTypes(), type =>
            type.Namespace is not null && (type.Namespace.Contains(".Logical") || type.Namespace.Contains(".Execution")));
    }

    [Fact]
    public void ConcreteCompilationRemainsOutsideTheGraphModel()
    {
        AssertTypes(typeof(ExecutionGraphCompiler).Assembly, typeof(DefaultExecutionGraphConverter),
            typeof(ExecutionGraphCompilationContext), typeof(ExecutionPassContext), typeof(IExecutionGraphConverter),
            typeof(IExecutionGraphPostprocessor), typeof(ExecutionImplementationSelector), typeof(VmExecutionGraphEmitter));
        Assert.Contains("SharpInference.Graphs", References(typeof(InstructionRegistry).Assembly));
        Assert.Contains("SharpInference.Graphs.Execution", References(typeof(InstructionRegistry).Assembly));
    }

    [Fact]
    public void XmlSafetyImplementationDoesNotExposeItsPrivateLimits()
    {
        var support = typeof(Graph).Assembly.GetType("SharpInference.Graphs.Internal.GraphXmlSupport", true)!;
        Assert.False(support.GetField("MaxCharacters", BindingFlags.Static | BindingFlags.NonPublic)!.IsPublic);
    }

    private static string[] References(Assembly assembly) =>
        assembly.GetReferencedAssemblies().Select(reference => reference.Name!)
            .Where(name => name.StartsWith("SharpInference.", StringComparison.Ordinal)).Order(StringComparer.Ordinal).ToArray();

    private static void AssertTypes(Assembly assembly, params Type[] types) =>
        Assert.All(types, type => Assert.Same(assembly, type.Assembly));
}
