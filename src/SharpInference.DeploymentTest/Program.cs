using System.Diagnostics;
using System.Text;
using SharpInference;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Runtime;

try
{
    Console.OutputEncoding = new UTF8Encoding(false);
    var options = DeploymentTestOptions.Parse(args);
    var test = new DeploymentTest(options);
    await test.RunAsync();
    return 0;
}
catch (Exception exception)
{
    Console.Error.WriteLine("deployment-test: FAIL");
    Console.Error.WriteLine($"{exception.GetType().Name}: {exception.Message}");
    return 1;
}

internal sealed class DeploymentTest(DeploymentTestOptions options)
{
    public async Task RunAsync()
    {
        if (!File.Exists(options.ModelPath))
        {
            throw new FileNotFoundException("The model file does not exist.", options.ModelPath);
        }

        Console.WriteLine("deployment-test: START");
        Console.WriteLine($"model-path={options.ModelPath}");
        Console.WriteLine($"backend={options.Backend}");
        if (options.Backend == "gpu") Console.WriteLine($"adapter-index={options.AdapterIndex}");
        Console.WriteLine($"max-tokens={options.MaxTokens}");

        var loadTimer = Stopwatch.StartNew();
        VmGraphBackend? backend = null;
        using (var model = new ProcessorPipelineBuilder(options.ModelPath)
            .UseReader(new GgmlModelReader())
            .UseProvider(new PortableRwkv6GraphProvider())
            .UseBackend(_ => backend = options.Backend == "gpu"
                ? VmBackendFactory.CreateD3D12(adapterIndex: options.AdapterIndex)
                : VmBackendFactory.CreateCpu())
            .UsePortableGraphArchitecture()
            .Build())
        {
            loadTimer.Stop();

            var (tokenizer, tokenizerKind) = CreateCompatibleTokenizer(model.Metadata.VocabularySize);

            Console.WriteLine($"device={backend!.DeviceName}");
            Console.WriteLine($"model-vocabulary={model.Metadata.VocabularySize}");
            Console.WriteLine($"model-layers={model.Metadata.LayerCount}");
            Console.WriteLine($"model-embedding={model.Metadata.EmbeddingSize}");
            Console.WriteLine($"tokenizer={tokenizerKind}");
            Console.WriteLine($"model-load-ms={loadTimer.Elapsed.TotalMilliseconds:F0}");

            if (options.Backend == "gpu" && model.Metadata.VocabularySize <= 256)
            {
                VerifyAgainstCpu(options.ModelPath, model);
            }

            var generationTimer = Stopwatch.StartNew();
            var generated = new System.Text.StringBuilder();
            using var session = model.CreateSession();
            await foreach (var text in RwkvTextGenerator.GenerateAsync(
                session,
                tokenizer,
                options.Prompt,
                new RwkvGenerationOptions
                {
                    MaxTokens = options.MaxTokens,
                    Temperature = 0,
                    StopStrings = [options.StopString],
                }))
            {
                generated.Append(text);
            }

            generationTimer.Stop();
            if (options.Backend == "gpu" && model.Metadata.VocabularySize <= 256)
            {
                await VerifyTinyGenerationAgainstCpuAsync(options.ModelPath, tokenizer, options, generated.ToString(), session);
            }

            VerifyStateRoundTrip(model, session);

            var output = generated.ToString();
            if (options.OutputPath is not null)
            {
                await File.WriteAllTextAsync(options.OutputPath, output, new UTF8Encoding(false, true));
            }

            Console.WriteLine($"generation-ms={generationTimer.Elapsed.TotalMilliseconds:F0}");
            Console.WriteLine($"generation-output-chars={output.Length}");
            Console.WriteLine($"generation-output={EscapeForDisplay(output)}");
            Console.WriteLine($"generation-output-unicode={EscapeForUnicodeDiagnostics(output)}");
            if (options.OutputPath is not null)
            {
                Console.WriteLine($"generation-output-file={Path.GetFullPath(options.OutputPath)}");
            }
            Console.WriteLine("state-round-trip=PASS");
            if (output.Contains('\uFFFD'))
            {
                throw new InvalidDataException(
                    "Generated text contains U+FFFD replacement characters. Inspect the GPU logits and emitted token bytes; state round-trip alone cannot verify inference correctness.");
            }

            Console.WriteLine("deployment-test: PASS");
        }
    }

    private static Processor CreateProcessor(string modelPath, VmGraphBackend backend) =>
        Processor.LoadGraph(modelPath, new PortableRwkv6GraphProvider(), backend);

    private static void VerifyStateRoundTrip(Processor model, ProcessorSession session)
    {
        using var snapshot = new MemoryStream();
        session.SaveState(snapshot);
        snapshot.Position = 0;
        using var restored = model.CreateSession();
        restored.LoadState(snapshot);
        var expected = session.ForwardToken(0).ToArray();
        var actual = restored.ForwardToken(0).ToArray();
        if (!expected.AsSpan().SequenceEqual(actual))
        {
            throw new InvalidOperationException("State snapshot round-trip produced different logits.");
        }
    }

    private static async Task VerifyTinyGenerationAgainstCpuAsync(
        string modelPath,
        RwkvWorldTokenizer tokenizer,
        DeploymentTestOptions options,
        string gpuOutput,
        ProcessorSession gpuSession)
    {
        using var cpuModel = CreateProcessor(modelPath, VmBackendFactory.CreateCpu());
        using var cpuSession = cpuModel.CreateSession();
        var cpuOutput = new StringBuilder();
        await foreach (var text in RwkvTextGenerator.GenerateAsync(
            cpuSession, tokenizer, options.Prompt,
            new RwkvGenerationOptions
            {
                MaxTokens = options.MaxTokens,
                Temperature = 0,
                StopStrings = [options.StopString],
            }))
        {
            cpuOutput.Append(text);
        }

        if (!string.Equals(cpuOutput.ToString(), gpuOutput, StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"Tiny model CPU/GPU generation differs: CPU={EscapeForUnicodeDiagnostics(cpuOutput.ToString())}; GPU={EscapeForUnicodeDiagnostics(gpuOutput)}.");
        }

        var maximumDifference = StateComparison.MaximumDifference(cpuSession, gpuSession, "Tiny model CPU/GPU generation");

        Console.WriteLine($"cpu-gpu-generation-max-state-diff={maximumDifference:G9}");
        if (maximumDifference > 0.001f)
        {
            throw new InvalidOperationException($"Tiny model CPU/GPU generation state differs by {maximumDifference:G9} (tolerance 0.001).");
        }

        Console.WriteLine("cpu-gpu-generation=PASS");
    }

    private static void VerifyAgainstCpu(string modelPath, Processor gpuModel)
    {
        using var cpuModel = CreateProcessor(modelPath, VmBackendFactory.CreateCpu());
        using var cpu = cpuModel.CreateSession();
        using var gpu = gpuModel.CreateSession();
        var maximumLogitDifference = 0f;
        var maximumStateDifference = 0f;
        foreach (var token in new[] { 0, 10, 13, 42 })
        {
            if (token >= gpuModel.Metadata.VocabularySize)
            {
                break;
            }

            var cpuLogits = cpu.ForwardToken(token).Span;
            var gpuLogits = gpu.ForwardToken(token).Span;
            var cpuBest = 0;
            var gpuBest = 0;
            for (var index = 0; index < cpuLogits.Length; index++)
            {
                if (!float.IsFinite(cpuLogits[index]) || !float.IsFinite(gpuLogits[index]))
                {
                    throw new InvalidOperationException($"CPU/GPU oracle produced non-finite logits at token {token}, index {index}.");
                }

                maximumLogitDifference = MathF.Max(maximumLogitDifference, MathF.Abs(cpuLogits[index] - gpuLogits[index]));
                if (cpuLogits[index] > cpuLogits[cpuBest]) cpuBest = index;
                if (gpuLogits[index] > gpuLogits[gpuBest]) gpuBest = index;
            }

            maximumStateDifference = MathF.Max(maximumStateDifference,
                StateComparison.MaximumDifference(cpu, gpu, $"CPU/GPU oracle at token {token}"));

            if (cpuBest != gpuBest)
            {
                throw new InvalidOperationException($"CPU/GPU oracle argmax differs at token {token}: CPU={cpuBest}, GPU={gpuBest}.");
            }
        }

        Console.WriteLine($"cpu-gpu-max-logit-diff={maximumLogitDifference:G9}");
        Console.WriteLine($"cpu-gpu-max-state-diff={maximumStateDifference:G9}");
        if (maximumLogitDifference > 0.001f || maximumStateDifference > 0.001f)
        {
            throw new InvalidOperationException("CPU/GPU oracle exceeded the 0.001 absolute tolerance for logits or recurrent state.");
        }

        Console.WriteLine("cpu-gpu-oracle=PASS");
    }

    private static (RwkvWorldTokenizer Tokenizer, string Kind) CreateCompatibleTokenizer(int vocabularySize)
    {
        var bundled = RwkvWorldTokenizer.LoadBundled();
        if (bundled.TokenIds.Count > 0 && bundled.TokenIds[^1] < vocabularySize)
        {
            return (bundled, "world");
        }

        if (vocabularySize is <= 0 or > 256)
        {
            throw new InvalidOperationException(
                $"The bundled World tokenizer is incompatible with model vocabulary size {vocabularySize}, and no deployment-test tokenizer is available.");
        }

        var path = Path.GetTempFileName();
        try
        {
            File.WriteAllLines(
                path,
                Enumerable.Range(0, vocabularySize)
                    .Select(token => $"{token} b'\\x{token:X2}' 1"));
            return (RwkvWorldTokenizer.Load(path), "single-byte-test");
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static string EscapeForDisplay(string value) =>
        value.Replace("\\", "\\\\", StringComparison.Ordinal)
            .Replace("\r", "\\r", StringComparison.Ordinal)
            .Replace("\n", "\\n", StringComparison.Ordinal)
            .Replace("\0", "\\0", StringComparison.Ordinal);

    private static string EscapeForUnicodeDiagnostics(string value)
    {
        var escaped = new StringBuilder();
        foreach (var rune in value.EnumerateRunes())
        {
            if (rune.Value is >= 0x20 and <= 0x7e && rune.Value is not '\\' and not '?')
            {
                escaped.Append((char)rune.Value);
            }
            else
            {
                escaped.Append(rune.Value <= 0xffff
                    ? $"\\u{rune.Value:X4}"
                    : $"\\U{rune.Value:X8}");
            }
        }

        return escaped.ToString();
    }
}

internal sealed record DeploymentTestOptions(
    string ModelPath,
    string Backend,
    int AdapterIndex,
    int MaxTokens,
    string Prompt,
    string StopString,
    string? OutputPath)
{
    public static DeploymentTestOptions Parse(string[] args)
    {
        string? modelPath = null;
        var backend = "gpu";
        var adapterIndex = 0;
        var maxTokens = 16;
        var prompt = "User: Hello\nAssistant:";
        var stopString = "__sharpinference_deployment_stop__";
        string? outputPath = null;

        for (var index = 0; index < args.Length; index++)
        {
            switch (args[index])
            {
                case "--model":
                    modelPath = ReadValue(args, ref index, "--model");
                    break;
                case "--backend":
                    backend = ReadValue(args, ref index, "--backend").ToLowerInvariant();
                    if (backend is not ("cpu" or "gpu"))
                    {
                        throw new ArgumentException("--backend requires cpu or gpu.");
                    }
                    break;
                case "--adapter-index":
                    adapterIndex = ParseNonNegativeInt(ReadValue(args, ref index, "--adapter-index"), "--adapter-index");
                    break;
                case "--max-tokens":
                    maxTokens = ParsePositiveInt(ReadValue(args, ref index, "--max-tokens"), "--max-tokens");
                    break;
                case "--prompt":
                    prompt = ReadValue(args, ref index, "--prompt");
                    break;
                case "--stop":
                    stopString = ReadValue(args, ref index, "--stop");
                    break;
                case "--output":
                    outputPath = ReadValue(args, ref index, "--output");
                    break;
                case "--diagnose-layers":
                case "--diagnose-token-index":
                    throw new NotSupportedException(
                        "Per-layer tracing is unavailable in the compiled VM; use generated source or CPU/GPU oracle validation instead.");
                case "--help":
                case "-h":
                    throw new ArgumentException(Usage);
                default:
                    throw new ArgumentException($"Unknown argument '{args[index]}'.{Environment.NewLine}{Usage}");
            }
        }

        if (string.IsNullOrWhiteSpace(modelPath))
        {
            throw new ArgumentException($"--model is required.{Environment.NewLine}{Usage}");
        }

        if (string.IsNullOrEmpty(prompt) || string.IsNullOrEmpty(stopString))
        {
            throw new ArgumentException("The prompt and stop string must be non-empty.");
        }

        if (outputPath is not null && string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("--output requires a non-empty file path.");
        }

        return new DeploymentTestOptions(modelPath, backend, adapterIndex, maxTokens, prompt, stopString, outputPath);
    }

    private const string Usage =
        "Usage: SharpInference.DeploymentTest --model <path> [--backend cpu|gpu] [--adapter-index <n>] [--max-tokens <n>] [--prompt <text>] [--stop <text>] [--output <utf8-file>]";

    private static string ReadValue(string[] args, ref int index, string option)
    {
        if (++index >= args.Length)
        {
            throw new ArgumentException($"{option} requires a value.{Environment.NewLine}{Usage}");
        }

        return args[index];
    }

    private static int ParseNonNegativeInt(string value, string option) =>
        int.TryParse(value, out var result) && result >= 0
            ? result
            : throw new ArgumentException($"{option} requires a non-negative integer.");

    private static int ParsePositiveInt(string value, string option) =>
        int.TryParse(value, out var result) && result > 0
            ? result
            : throw new ArgumentException($"{option} requires a positive integer.");
}
