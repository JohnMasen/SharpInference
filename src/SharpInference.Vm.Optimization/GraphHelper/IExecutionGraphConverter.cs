using SharpInference.Graphs.Logical;

namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Converts a logical graph definition and its compilation settings into an execution program.</summary>
public interface IExecutionGraphConverter
{
    /// <summary>Converts the logical root and the definitions needed by the resulting execution program.</summary>
    /// <param name="graph">The source logical root definition.</param>
    /// <param name="context">The model, backend, and compilation settings.</param>
    /// <returns>A non-null execution graph program.</returns>
    ExecutionGraphProgram Convert(
        LogicalGraphDefinition graph,
        ExecutionGraphCompilationContext context);
}
