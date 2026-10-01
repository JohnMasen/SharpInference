using System.Text;
using System.Text.RegularExpressions;
using Vortice.Dxc;

if (args.Length != 1)
{
    Console.Error.WriteLine("Usage: SharpInference.ShaderCompiler <shader-directory>");
    return 1;
}

var shaderDirectory = Path.GetFullPath(args[0]);
if (!Directory.Exists(shaderDirectory))
{
    Console.Error.WriteLine($"Shader directory '{shaderDirectory}' does not exist.");
    return 1;
}

var shaders = new List<ShaderCompilation>
{
    new("Rwkv6MatVecFp32.hlsl", "Rwkv6MatVecFp32.hlsl.dxil", DxcShaderModel.Model6_0),
    new("Rwkv6MatVecFp16Fallback.hlsl", "Rwkv6MatVecFp16Fallback.hlsl.dxil", DxcShaderModel.Model6_0),
    new("Rwkv6MatVecFp16.hlsl", "Rwkv6MatVecFp16.hlsl.dxil", DxcShaderModel.Model6_2, Enable16BitTypes: true),
};

var primitiveEntries = new[]
{
    "Copy", "Add", "Subtract", "Multiply", "Divide", "Maximum",
    "Exp", "Tanh", "Sigmoid", "ReciprocalSquareRoot", "Square", "Relu",
    "ReduceSum", "ReduceMean", "MatVec", "GatherRow",
};
foreach (var entryPoint in primitiveEntries)
{
    shaders.Add(new(
        "PrimitiveOperatorsFp32.hlsl",
        $"PrimitiveOperatorsFp32.{entryPoint}.hlsl.dxil",
        DxcShaderModel.Model6_0,
        EntryPoint: entryPoint));
    shaders.Add(new(
        "PrimitiveOperatorsFp16.hlsl",
        $"PrimitiveOperatorsFp16.{entryPoint}.hlsl.dxil",
        DxcShaderModel.Model6_2,
        Enable16BitTypes: true,
        EntryPoint: entryPoint));
}

var forwardStages = new[]
{
    "Embedding", "LayerNorm", "TimeMixPrepare", "MaaProjection", "MaaMix",
    "AttentionProjections", "DecayProjection", "DecayOutput", "WkvUpdate", "GroupNorm",
    "AttentionOutput", "ChannelMixPrepare", "ChannelReceptance", "ChannelKey",
    "ChannelValue", "HeadProjection",
};
foreach (var stage in forwardStages)
{
    var source = $"Rwkv6{stage}.hlsl";
    shaders.Add(new(source, $"Rwkv6{stage}Fp16.hlsl.dxil", DxcShaderModel.Model6_0, DefineFp16Weights: true));
    shaders.Add(new(source, $"Rwkv6{stage}Fp32.hlsl.dxil", DxcShaderModel.Model6_0));
}

var generationStages = new[]
{
    "GreedySelect", "SamplingPrepare", "SamplingRadixHistogram", "SamplingRadixScan",
    "SamplingRadixScatter", "SamplingReduce", "SamplingSelect",
    "StopPatternCheck", "StopFinalize",
};
foreach (var stage in generationStages)
    shaders.Add(new($"Rwkv6{stage}.hlsl", $"Rwkv6{stage}.hlsl.dxil", DxcShaderModel.Model6_0, DefineFp16Weights: true));

foreach (var shader in shaders)
{
    var sourcePath = Path.Combine(shaderDirectory, shader.FileName);
    var outputPath = Path.Combine(shaderDirectory, shader.OutputFileName);
    var dependencies = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    var source = ExpandIncludes(sourcePath, shaderDirectory, dependencies, new Stack<string>());
    var newestInput = dependencies.Max(File.GetLastWriteTimeUtc);
    if (File.Exists(outputPath) && File.GetLastWriteTimeUtc(outputPath) >= newestInput)
    {
        continue;
    }

    if (shader.DefineFp16Weights)
        source = "#define RWKV_FP16_WEIGHTS\n" + source;
    using var compilation = DxcCompiler.Compile(
        DxcShaderStage.Compute,
        source,
        shader.EntryPoint,
        new DxcCompilerOptions
        {
            ShaderModel = shader.ShaderModel,
            Enable16bitTypes = shader.Enable16BitTypes,
        });
    File.WriteAllBytes(outputPath, compilation.GetObjectBytecodeArray());
    Console.WriteLine($"Compiled {shader.FileName} to {shader.OutputFileName}.");
}

return 0;

static string ExpandIncludes(
    string sourcePath,
    string shaderDirectory,
    HashSet<string> dependencies,
    Stack<string> includeStack)
{
    sourcePath = Path.GetFullPath(sourcePath);
    var root = Path.TrimEndingDirectorySeparator(shaderDirectory) + Path.DirectorySeparatorChar;
    if (!sourcePath.StartsWith(root, StringComparison.OrdinalIgnoreCase))
        throw new InvalidOperationException($"Include escapes shader directory: '{sourcePath}'.");
    if (includeStack.Contains(sourcePath, StringComparer.OrdinalIgnoreCase))
        throw new InvalidOperationException($"Circular include: {string.Join(" -> ", includeStack.Reverse().Append(sourcePath))}");
    if (!File.Exists(sourcePath))
        throw new FileNotFoundException($"Shader source or include not found: '{sourcePath}'.", sourcePath);

    dependencies.Add(sourcePath);
    includeStack.Push(sourcePath);
    try
    {
        var expanded = new StringBuilder();
        foreach (var line in File.ReadLines(sourcePath))
        {
            var match = IncludeParser.Pattern().Match(line);
            if (!match.Success)
            {
                expanded.AppendLine(line);
                continue;
            }

            var includePath = Path.GetFullPath(Path.Combine(
                Path.GetDirectoryName(sourcePath)!,
                match.Groups["path"].Value));
            expanded.Append(ExpandIncludes(includePath, shaderDirectory, dependencies, includeStack));
        }
        return expanded.ToString();
    }
    finally
    {
        includeStack.Pop();
    }
}

internal static partial class IncludeParser
{
    [GeneratedRegex(@"^\s*#\s*include\s*""(?<path>[^""]+)""\s*(?://.*)?$")]
    internal static partial Regex Pattern();
}

internal sealed record ShaderCompilation(
    string FileName,
    string OutputFileName,
    DxcShaderModel ShaderModel,
    bool Enable16BitTypes = false,
    bool DefineFp16Weights = false,
    string EntryPoint = "Main");
