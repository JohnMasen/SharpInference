
namespace SharpInference.Graphs.Execution;

/// <summary>Transforms an execution program as one ordered stage after logical conversion.</summary>
public interface IExecutionGraphPostprocessor
{
    /// <summary>Gets the nonblank processor name reported in the invocation context and diagnostics.</summary>
    string Name { get; }
    /// <summary>Processes the current program before the compiler revalidates its graph-call relationships.</summary>
    /// <param name="program">The converter result or the preceding processor's validated result.</param>
    /// <param name="context">The original logical graph, compilation settings, and stage identity.</param>
    /// <returns>The non-null program passed to relationship validation before the next stage.</returns>
    ExecutionGraphProgram Process(
        ExecutionGraphProgram program,
        ExecutionGraphPostprocessContext context);
}
