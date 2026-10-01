using System.Reflection;

namespace SharpInference.Tests;

internal enum TestModel
{
    Rwkv6,
    Rwkv7Fp32,
    Rwkv7Fp16,
    Rwkv7ExpectedLogits,
    Rwkv7Large,
}

internal static class TestModelLoader
{
    internal static string GetPath(TestModel model) => GetPath(
        model, Parameter(VariableName(model)), Parameter("RwkvTestModelDirectory"));

    internal static string GetPath(TestModel model, string? path, string? directory = null)
    {
        var variable = VariableName(model);
        if (path is null)
        {
            if (directory is null)
                throw new InvalidOperationException(
                    $"Set -p:{variable}=<path> or -p:RwkvTestModelDirectory=<directory> to run this external-model test.");
            if (string.IsNullOrWhiteSpace(directory) || !Path.IsPathFullyQualified(directory))
                throw new ArgumentException("RwkvTestModelDirectory must be an absolute directory path.",
                    nameof(directory));
            path = Path.Combine(directory, FileName(model));
        }
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException($"{variable} must be a non-empty file path.", variable);
        if (!Path.IsPathFullyQualified(path))
            throw new ArgumentException($"{variable} must be an absolute file path: {path}", variable);
        if (!File.Exists(path))
            throw new FileNotFoundException($"{variable} points to a missing model file: {path}", path);
        if (model != TestModel.Rwkv7ExpectedLogits)
        {
            using var catalog = GgmlModelFile.Open(path);
        }
        return path;
    }

    internal static GgmlModelFile OpenCatalog(TestModel model) => GgmlModelFile.Open(GetPath(model));

    internal static GgmlModelFile OpenCatalog(TestModel model, string? path) =>
        GgmlModelFile.Open(GetPath(model, path));

    private static string FileName(TestModel model) => model switch
    {
        TestModel.Rwkv6 => "tiny-rwkv-6v0-3m-FP32.bin",
        TestModel.Rwkv7Fp32 => "tiny-rwkv-7v0-834K-FP32.bin",
        TestModel.Rwkv7Fp16 => "tiny-rwkv-7v0-834K-FP16.bin",
        TestModel.Rwkv7ExpectedLogits => "expected-logits-7v0-834K.bin",
        TestModel.Rwkv7Large => "rwkv7-g1j-7.2b-20260831-ctx16384-FP16.bin",
        _ => throw new ArgumentOutOfRangeException(nameof(model), model, null),
    };

    private static string VariableName(TestModel model) => model switch
    {
        TestModel.Rwkv6 => "RwkvTestRwkv6Model",
        TestModel.Rwkv7Fp32 => "RwkvTestRwkv7Fp32Model",
        TestModel.Rwkv7Fp16 => "RwkvTestRwkv7Fp16Model",
        TestModel.Rwkv7ExpectedLogits => "RwkvTestRwkv7ExpectedLogits",
        TestModel.Rwkv7Large => "RwkvTestRwkv7LargeModel",
        _ => throw new ArgumentOutOfRangeException(nameof(model), model, null),
    };

    private static string? Parameter(string name) =>
        typeof(TestModelLoader).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
            .SingleOrDefault(attribute => attribute.Key == name)?.Value;
}
