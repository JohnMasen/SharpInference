using SharpInference.Graphs;
using SharpInference.Graphs.Execution;
using SharpInference.Vm.Optimization.GraphHelper;

namespace SharpInference.Vm.Optimization;

public sealed class XmlExecutionPackagePostprocessor :
    IExecutionGraphPostprocessor
{
    private readonly string path;
    private readonly string expectedModelType;
    private readonly string expectedBackend;

    public XmlExecutionPackagePostprocessor(
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

    public ExecutionGraphPackage Process(ExecutionGraphPackage package, ExecutionPassContext context)
    {
        ArgumentNullException.ThrowIfNull(package);
        ArgumentNullException.ThrowIfNull(context);
        if (context.Compilation.ModelType != expectedModelType || context.Compilation.Backend != expectedBackend)
        {
            throw new InvalidOperationException($"Execution XML '{path}' targets '{expectedModelType}/{expectedBackend}', " +
                $"not '{context.Compilation.ModelType}/{context.Compilation.Backend}'.");
        }

        var loaded = ExecutionGraphPackageXml.Deserialize(File.ReadAllText(path));
        GraphValidation.ThrowIfErrors(loaded.Validate());
        if (loaded.MainGraphId != package.MainGraphId || loaded.GetTarget() != context.Compilation.Backend)
        {
            throw new InvalidDataException("Execution XML main graph or target does not match this compilation.");
        }

        return loaded;
    }

}
