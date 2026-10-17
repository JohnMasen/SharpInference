using SharpInference.Graphs.Logical;

namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Supplies the original logical graph and compilation settings to one postprocessor invocation.</summary>
/// <param name="SourceGraph">The logical root originally passed to the compiler.</param>
/// <param name="Compilation">The shared compilation context.</param>
/// <param name="ProcessorIndex">The zero-based index in the configured postprocessor sequence.</param>
/// <param name="ProcessorName">The name of the processor being invoked.</param>
public sealed record ExecutionGraphPostprocessContext(
    LogicalGraphDefinition SourceGraph,
    ExecutionGraphCompilationContext Compilation,
    int ProcessorIndex,
    string ProcessorName);
