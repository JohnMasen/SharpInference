using SharpInference.Graphs.Logical;

namespace SharpInference.Graphs.Execution;

/// <summary>Runs logical conversion followed by configured postprocessors in declaration order.</summary>
/// <remarks>Each postprocessor result is reconstructed as a program to validate call relationships.
/// Conversion and processing do not automatically enforce compilation-boundary policies.</remarks>
public sealed class ExecutionGraphCompiler
{
    /// <summary>The converter invoked once before any postprocessors.</summary>
    private readonly IExecutionGraphConverter converter;
    /// <summary>The snapshotted postprocessor sequence in invocation order.</summary>
    private readonly IReadOnlyList<IExecutionGraphPostprocessor> postprocessors;

    /// <summary>Creates a compiler using the default copying converter.</summary>
    /// <param name="postprocessors">Optional ordered processors; null selects no processors.</param>
    /// <exception cref="ArgumentException">A processor is null or has a blank name.</exception>
    public ExecutionGraphCompiler(
        IEnumerable<IExecutionGraphPostprocessor>? postprocessors = null)
        : this(new DefaultExecutionGraphConverter(), postprocessors) { }

    /// <summary>Creates a compiler and snapshots the ordered processor sequence.</summary>
    /// <param name="converter">The logical-to-execution converter.</param>
    /// <param name="postprocessors">Optional ordered processors; null selects no processors.</param>
    /// <exception cref="ArgumentNullException">The converter is null.</exception>
    /// <exception cref="ArgumentException">A processor is null or has a blank name.</exception>
    public ExecutionGraphCompiler(
        IExecutionGraphConverter converter,
        IEnumerable<IExecutionGraphPostprocessor>? postprocessors = null)
    {
        this.converter = converter ?? throw new ArgumentNullException(nameof(converter));
        this.postprocessors = Array.AsReadOnly(postprocessors?.ToArray() ?? []);
        if (this.postprocessors.Any(processor =>
            processor is null || string.IsNullOrWhiteSpace(processor.Name)))
        {
            throw new ArgumentException("Every postprocessor requires an instance and name.", nameof(postprocessors));
        }
    }

    /// <summary>Converts the logical root, then invokes and revalidates each processor result in order.</summary>
    /// <remarks>The converter result is used directly. After each processor, a new program is constructed
    /// from its graphs, recomputing the root and call relations without rebuilding individual graphs.
    /// Each stage receives the original logical root and the same compilation context.</remarks>
    /// <param name="graph">The original logical root definition.</param>
    /// <param name="context">The compilation settings passed to conversion and every processing stage.</param>
    /// <returns>The converter result when no processors exist, or the final revalidated program.</returns>
    /// <exception cref="ArgumentNullException">The graph or context is null.</exception>
    /// <exception cref="InvalidOperationException">The converter or a processor returns null.</exception>
    /// <exception cref="InvalidDataException">A processor result has invalid graph-call relationships.</exception>
    public ExecutionGraphProgram Compile(
        LogicalGraphDefinition graph,
        ExecutionGraphCompilationContext context)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(context);
        var program = converter.Convert(graph, context)
            ?? throw new InvalidOperationException("The execution graph converter returned null.");
        for (var index = 0; index < postprocessors.Count; index++)
        {
            var processor = postprocessors[index];
            program = processor.Process(
                program,
                new ExecutionGraphPostprocessContext(graph, context, index, processor.Name))
                ?? throw new InvalidOperationException(
                    $"Execution graph postprocessor '{processor.Name}' returned null.");
            program = new ExecutionGraphProgram(program.Graphs);
        }
        return program;
    }
}
