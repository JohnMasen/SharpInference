using SharpInference.Graphs.Execution;
using SharpInference.Graphs.Logical;
using SharpInference.Graphs;

namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Converts a logical package into selected execution definitions.</summary>
public interface IExecutionGraphConverter
{
    ExecutionGraphPackage Convert(
        LogicalGraphPackage package, ExecutionGraphCompilationContext context);
}
