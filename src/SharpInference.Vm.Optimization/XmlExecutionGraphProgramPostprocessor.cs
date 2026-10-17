using SharpInference.Graphs;
using SharpInference.Vm.Optimization.GraphHelper;

namespace SharpInference.Vm.Optimization;

public sealed class XmlExecutionGraphProgramPostprocessor :
    IExecutionGraphPostprocessor
{
    private readonly string path;
    private readonly string expectedModelType;
    private readonly string expectedBackend;

    public XmlExecutionGraphProgramPostprocessor(
        string path,
        string expectedModelType,
        string expectedBackend)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentException.ThrowIfNullOrWhiteSpace(expectedModelType);
        ArgumentException.ThrowIfNullOrWhiteSpace(expectedBackend);
        this.path = Path.GetFullPath(path);
        this.expectedModelType = expectedModelType;
        this.expectedBackend = expectedBackend;
    }

    public string Name => "load-execution-graph-xml";

    public ExecutionGraphProgram Process(
        ExecutionGraphProgram program,
        ExecutionGraphPostprocessContext context)
    {
        ArgumentNullException.ThrowIfNull(program);
        ArgumentNullException.ThrowIfNull(context);
        if (!string.Equals(
                context.Compilation.ModelType,
                expectedModelType,
                StringComparison.Ordinal) ||
            !string.Equals(
                context.Compilation.Backend,
                expectedBackend,
                StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"Execution graph XML '{path}' is configured for model " +
                $"'{expectedModelType}' and backend '{expectedBackend}', not " +
                $"'{context.Compilation.ModelType}' and " +
                $"'{context.Compilation.Backend}'.");
        }

        var loaded = ExecutionGraphXml.DeserializeProgram(File.ReadAllText(path));
        var root = loaded.Graphs.Single(graph =>
            graph.Id == loaded.RootGraph);
        if (root.Source?.Definition != context.SourceGraph.Id)
        {
            throw new InvalidDataException(
                $"Execution graph XML root '{root.Id}' must identify logical " +
                $"source '{context.SourceGraph.Id}'.");
        }

        return loaded;
    }
}
