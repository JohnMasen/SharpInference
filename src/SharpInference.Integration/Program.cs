using SharpInference;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Runtime;
using SharpInference.Applications;
using Microsoft.Extensions.Configuration;
using System.Text.Json;
using SharpInference.Gguf;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

if (args.Contains("--list-gpus", StringComparer.Ordinal))
{
    for (var index = 0; ; index++)
    {
        try
        {
            var (device, name) = D3D12VmDeviceFactory.Create(index);
            using (device) Console.WriteLine($"gpu-index={index} device={name}");
        }
        catch (ArgumentOutOfRangeException)
        {
            break;
        }
    }

    return;
}

var useGpu = args.Contains("--gpu", StringComparer.Ordinal);
int? gpuIndex = null;
int? benchmarkTokens = null;
string? modelPath = null;
string? configurationPath = null;
string? graphDumpDirectory = null;
for (var argumentIndex = 0; argumentIndex < args.Length; argumentIndex++)
{
    var argument = args[argumentIndex];
    if (string.Equals(argument, "--gpu", StringComparison.Ordinal))
    {
        useGpu = true;
        continue;
    }

    if (string.Equals(argument, "--runtime-precision", StringComparison.Ordinal))
    {
        throw new NotSupportedException(
            "--runtime-precision selected the removed model-specific backend. Portable graphs execute FP32 operators.");
    }

    if (string.Equals(argument, "--config", StringComparison.Ordinal))
    {
        if (++argumentIndex == args.Length)
        {
            throw new ArgumentException("--config requires a JSON configuration path.");
        }

        configurationPath = args[argumentIndex];
        continue;
    }

    if (string.Equals(argument, "--benchmark-tokens", StringComparison.Ordinal))
    {
        if (++argumentIndex == args.Length || !int.TryParse(args[argumentIndex], out var parsedBenchmarkTokens) || parsedBenchmarkTokens <= 0)
        {
            throw new ArgumentException("--benchmark-tokens requires a positive integer.");
        }

        benchmarkTokens = parsedBenchmarkTokens;
        continue;
    }

    if (string.Equals(argument, "--dump-graphs", StringComparison.Ordinal))
    {
        if (++argumentIndex == args.Length)
        {
            throw new ArgumentException("--dump-graphs requires an output directory.");
        }

        graphDumpDirectory = args[argumentIndex];
        continue;
    }

    if (string.Equals(argument, "--gpu-index", StringComparison.Ordinal))
    {
        if (++argumentIndex == args.Length || !int.TryParse(args[argumentIndex], out var parsedGpuIndex))
        {
            throw new ArgumentException("--gpu-index requires a non-negative integer.");
        }

        useGpu = true;
        gpuIndex = parsedGpuIndex;
        continue;
    }

    if (modelPath is not null)
    {
        throw new ArgumentException($"Unexpected argument '{argument}'.");
    }

    modelPath = argument;
}

modelPath ??= Environment.GetEnvironmentVariable("SHARPINFERENCE_TEST_MODEL_GGML")
              ?? @"D:\RWKVModels\RWKV-x060-World-1B6-v2.1-20240328-ctx4096-FP32.bin";

if (useGpu || gpuIndex is not null)
{
    throw new ArgumentException("Runtime-specific GPU switches are no longer supported. Configure Rwkv:Runtime in a JSON file passed with --config.");
}

var configurationBuilder = new ConfigurationBuilder().SetBasePath(Directory.GetCurrentDirectory());
if (configurationPath is not null)
{
    configurationBuilder.AddJsonFile(configurationPath, optional: false);
}

var configuration = configurationBuilder.Build();
if (configuration["Rwkv:Runtime:Vortice:WeightPrecision"] is not null)
    throw new NotSupportedException(
        "Rwkv:Runtime:Vortice:WeightPrecision is not supported by portable primitive graphs.");
using var catalog = GgmlModelFile.Open(modelPath);
if (graphDumpDirectory is not null)
{
    DumpGraphs(catalog, graphDumpDirectory);
}

var runtime = new RwkvApplicationComposition(RwkvApplicationComposition.CreateModelModules()).CreateRuntime(configuration.GetSection("Rwkv:Runtime"), catalog);
Console.WriteLine($"backend={configuration["Rwkv:Runtime:Kind"] ?? "cpu"}");

VmGraphBackend? backend = null;
using var model = new ProcessorPipelineBuilder(modelPath)
    .UseReader(new GgmlModelReader())
    .UseProvider(runtime.Provider)
    .UseBackend(_ => backend = runtime.CreateBackend())
    .UsePortableGraphArchitecture()
    .Build();
using var sha = System.Security.Cryptography.SHA256.Create();
var tracePath = Environment.GetEnvironmentVariable("SHARPINFERENCE_LOGITS_TRACE");
using var trace = tracePath is null ? null : new BinaryWriter(File.Open(tracePath, FileMode.Create, FileAccess.Write, FileShare.None));
using var session = model.CreateSession();
var tokens = new[] { 0, 10, 13, 42 };
foreach (var token in tokens)
{
    var logits = session.ForwardToken(token).Span;
    var best = 0;
    for (var index = 1; index < logits.Length; index++)
    {
        if (logits[index] > logits[best]) best = index;
    }

    var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(logits);
    trace?.Write(bytes);
    Console.WriteLine($"token={token} argmax={best} max={logits[best]:F6} logits-sha256={Convert.ToHexString(sha.ComputeHash(bytes.ToArray()))}");
}

using var snapshot = new MemoryStream();
session.SaveState(snapshot);
snapshot.Position = 0;
using var restored = model.CreateSession();
restored.LoadState(snapshot);
var expected = session.ForwardToken(99).ToArray();
var actual = restored.ForwardToken(99).ToArray();
var maximumDifference = 0f;
for (var index = 0; index < expected.Length; index++)
{
    maximumDifference = MathF.Max(maximumDifference, MathF.Abs(expected[index] - actual[index]));
}

if (maximumDifference != 0)
{
    throw new InvalidOperationException($"State snapshot round-trip diverged by {maximumDifference}.");
}

Console.WriteLine($"state-snapshot-bytes={snapshot.Length} round-trip-max-diff={maximumDifference}");

using var editableSnapshot = new MemoryStream();
session.SaveState(editableSnapshot);
editableSnapshot.Position = 0;
var editableState = GgufStateFile.Read(editableSnapshot);
var firstTensor = editableState.Tensors[0];
if (firstTensor.Type != GgufTensorType.Float32 || firstTensor.Data.Length < sizeof(float) ||
    firstTensor.Dimensions.Aggregate(1UL, (count, dimension) => checked(count * dimension)) * sizeof(float) !=
        (ulong)firstTensor.Data.Length)
    throw new InvalidDataException("Expected a nonempty FP32 state tensor.");
var editedBytes = firstTensor.Data.ToArray();
var editedValue = BitConverter.ToSingle(editedBytes) + 0.0001f;
BitConverter.GetBytes(editedValue).CopyTo(editedBytes, 0);
var editedTensors = editableState.Tensors.Select((tensor, index) => index == 0
    ? new GgufStateTensor(tensor.Name, tensor.Type, tensor.Dimensions, editedBytes) : tensor);
using var editedSnapshot = new MemoryStream();
GgufStateFile.Write(editedSnapshot, new GgufState(editableState.SchemaName, editedTensors));
using var editedStateSession = model.CreateSession();
editedSnapshot.Position = 0;
editedStateSession.LoadState(editedSnapshot);
var editedLogits = editedStateSession.ForwardToken(100).Span;
for (var index = 0; index < editedLogits.Length; index++)
{
    if (!float.IsFinite(editedLogits[index]))
    {
        throw new InvalidOperationException("An externally edited state produced non-finite logits.");
    }
}

Console.WriteLine("editable-state-import=finite-logits");

if (benchmarkTokens is int tokenCount)
{
    using var benchmarkSession = model.CreateSession();
    var benchmarkTokensSequence = new[] { 0, 10, 13, 42 };
    _ = benchmarkSession.ForwardToken(benchmarkTokensSequence[0]);
    var stopwatch = System.Diagnostics.Stopwatch.StartNew();
    for (var index = 0; index < tokenCount; index++)
    {
        _ = benchmarkSession.ForwardToken(benchmarkTokensSequence[index % benchmarkTokensSequence.Length]);
    }

    stopwatch.Stop();
    Console.WriteLine($"benchmark-tokens={tokenCount} elapsed-ms={stopwatch.Elapsed.TotalMilliseconds:F3} tokens-per-second={tokenCount / stopwatch.Elapsed.TotalSeconds:F3}");
}

Console.WriteLine($"device={backend!.DeviceName} vm-slots={model.InferenceProgram.Slots.Count}");

var prompt = Environment.GetEnvironmentVariable("SHARPINFERENCE_STREAM_PROMPT");
if (prompt is not null)
{
    var maxTokensSetting = Environment.GetEnvironmentVariable("SHARPINFERENCE_STREAM_MAX_TOKENS");
    var streamMaxTokens = 16;
    if (maxTokensSetting is not null &&
        (!int.TryParse(maxTokensSetting, out streamMaxTokens) || streamMaxTokens <= 0))
    {
        throw new ArgumentException("SHARPINFERENCE_STREAM_MAX_TOKENS must be a positive integer.");
    }

    using var generationSession = model.CreateSession();
    var generationTokenizer = Environment.GetEnvironmentVariable("SHARPINFERENCE_STREAM_BYTE_VOCAB") is null
        ? runtime.Tokenizer
        : CreateIntegrationTokenizer(RwkvModelMetadata.FromModelMetadata(model.Metadata).VocabularySize);
    var stop = Environment.GetEnvironmentVariable("SHARPINFERENCE_STREAM_STOP");
    Console.WriteLine("stream-begin");
    await foreach (var text in RwkvTextGenerator.GenerateAsync(
        generationSession,
        generationTokenizer,
        prompt,
        new RwkvGenerationOptions
        {
            MaxTokens = streamMaxTokens,
            Temperature = 0,
            Seed = 1,
            StopStrings = stop is null ? Array.Empty<string>() : [stop],
        }))
    {
        Console.Write(text);
    }

    Console.WriteLine();
    Console.WriteLine("stream-end");
}

static void DumpGraphs(IModelTensorCatalog catalog, string outputDirectory)
{
    outputDirectory = Path.GetFullPath(outputDirectory);
    Directory.CreateDirectory(outputDirectory);
    var logical = RwkvApplicationComposition.CreateModelModules().Build(catalog).Graph;
    var cpu = VmGraphOptimizer.Optimize(logical, VmTarget.Cpu);
    var gpu = VmGraphOptimizer.Optimize(logical, VmTarget.Direct3D12);
    File.WriteAllText(Path.Combine(outputDirectory, "logical.json"), GraphJson.Serialize(logical));
    File.WriteAllText(Path.Combine(outputDirectory, "cpu.vm.xml"), VmProgramXml.Serialize(cpu));
    File.WriteAllText(Path.Combine(outputDirectory, "d3d12.vm.xml"), VmProgramXml.Serialize(gpu));
    WritePlanSummary(outputDirectory, "cpu-plan.json", "cpu", cpu);
    WritePlanSummary(outputDirectory, "d3d12-plan.json", "d3d12", gpu);
    Console.WriteLine(
        $"graph-dump={outputDirectory} logical-nodes={logical.Nodes.Count} " +
        $"cpu-definitions={cpu.Definitions.Count} gpu-definitions={gpu.Definitions.Count}");
}

static void WritePlanSummary(string outputDirectory, string fileName, string backend, VmProgram program)
{
    var summary = new
    {
        Backend = backend,
        Abi = program.Abi,
        SlotCount = program.Slots.Count,
        DefinitionCount = program.Definitions.Count,
        Definitions = program.Definitions.Select(definition => new
        {
            definition.Id,
            Kind = definition.Kind.ToString(),
        }),
    };
    File.WriteAllText(
        Path.Combine(outputDirectory, fileName),
        JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true }));
}

static RwkvWorldTokenizer CreateIntegrationTokenizer(int vocabularySize)
{
    if (vocabularySize is <= 0 or > 65536)
    {
        throw new InvalidOperationException("The integration tokenizer requires a vocabulary size from 1 through 65536.");
    }

    var path = Path.GetTempFileName();
    try
    {
        File.WriteAllLines(
            path,
            Enumerable.Range(0, vocabularySize)
                .Select(token => token == 0
                    ? "0 b'a' 1"
                    : $"{token} b'\\xFF\\x{(token - 1) >> 8:X2}\\x{(token - 1) & 0xff:X2}' 3"));
        return RwkvWorldTokenizer.Load(path);
    }
    finally
    {
        File.Delete(path);
    }
}
