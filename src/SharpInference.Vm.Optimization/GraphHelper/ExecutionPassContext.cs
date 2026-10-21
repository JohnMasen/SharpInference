using SharpInference.Graphs;
using SharpInference.Graphs.Logical;

namespace SharpInference.Vm.Optimization.GraphHelper;

public sealed record ExecutionPassContext(
    GraphPackage<LogicalGraph>? Source,
    ExecutionGraphCompilationContext Compilation,
    int ProcessorIndex,
    string ProcessorName)
{
    public ExecutionPassContext(ExecutionGraphCompilationContext compilation, int processorIndex, string processorName)
        : this(null, compilation, processorIndex, processorName) { }
}
