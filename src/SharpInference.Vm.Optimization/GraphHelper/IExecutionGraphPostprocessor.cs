using SharpInference.Graphs.Execution;
using SharpInference.Graphs;

namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Transforms an execution package as one ordered compilation stage.</summary>
public interface IExecutionGraphPostprocessor
{
    ExecutionGraphPackage Process(
        ExecutionGraphPackage package, ExecutionPassContext context);

    /// <summary>Gets the nonblank processor name reported in the invocation context and diagnostics.</summary>
    string Name { get; }
}
