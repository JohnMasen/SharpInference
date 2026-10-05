#if TIER_ONE
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using SharpInference;
using SharpInference.Gguf;
using SharpInference.Runtime;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;
using SharpInference.Backends.D3D12Vm;

internal static class QuestionBenchmark
{
    private static readonly string[] Questions =
    [
        "什么是人工智能？",
        "为什么天空是蓝色的？",
        "解释一下什么是递归。",
        "光合作用是什么？",
        "如何提高学习效率？",
        "什么是数据库索引？",
        "太阳系有哪些行星？",
        "TCP 和 UDP 有什么区别？",
        "如何写一个 Python 的加法函数？",
        "运动对健康有哪些好处？",
    ];

    public static async Task Run(Arguments arguments)
    {
        if (Environment.GetEnvironmentVariable("DOTNET_TieredCompilation") != "0")
            throw new InvalidOperationException("Set DOTNET_TieredCompilation=0 before starting the historical question benchmark.");
        var path = Path.GetFullPath(arguments.Required("model"));
        var output = Path.GetFullPath(arguments.Required("output"));
        Directory.CreateDirectory(Path.GetDirectoryName(output)!);
        var reference = arguments.Optional("reference");
        var expected = reference is null ? null :
            JsonSerializer.Deserialize<Report>(File.ReadAllText(reference + ".json"))
            ?? throw new InvalidDataException("Missing question reference report.");
        if (expected is not null && expected.Status != "completed")
            throw new InvalidDataException("Question reference did not complete.");
        var configuration = new VmRuntimeConfig
        {
            GpuMatVecMode = arguments.MatVecMode,
            GpuMatVecCostProfile = arguments.Optional("matvec-profile") is { } matVecPath
                ? TierOneCostProfile.Deserialize(File.ReadAllText(matVecPath)) : null,
            TierOneCostProfile = arguments.Optional("profile") is { } profilePath
                ? TierOneCostProfile.Deserialize(File.ReadAllText(profilePath)) : null,
        };
        var initialization = Stopwatch.StartNew();
        string architecture;
        using (var catalog = GgmlModelFile.Open(path))
            architecture = RwkvModelArchitectureDetector.Detect(catalog);
        using var backend = arguments.Target == VmTarget.Cpu ? VmBackendFactory.CreateCpu(configuration) :
            VmBackendFactory.CreateD3D12(configuration);
        using var processor = Processor.LoadGraph(path, RwkvRuntimeFactory.CreateGraphProvider(architecture), backend);
        var tokenizer = RwkvTokenizerResolver.LoadForArchitecture(architecture);
        if (tokenizer.TokenIds.Any(token => token >= processor.Metadata.VocabularySize))
            throw new InvalidDataException("The model does not cover the historical World tokenizer.");
        var prompts = Questions.Select(question => tokenizer.Encode($"User: {question}\n\nAssistant:").ToArray()).ToArray();
        if (prompts.Sum(prompt => prompt.Length) != 167 ||
            prompts.Any(prompt => prompt.Any(token => token < 0 || token >= processor.Metadata.VocabularySize)))
            throw new InvalidDataException("Historical prompt-token contract changed.");
        initialization.Stop();
        var report = new Report(arguments.Required("variant"), arguments.Target.ToString(), path,
            initialization.Elapsed.TotalMilliseconds, backend.DeviceName);
        report.MatVecDecision = backend.MatVecOptimizationReport;
        report.TierOneDecision = backend.OptimizationReport;
        Save();
        var warmup = Stopwatch.StartNew();
        foreach (var index in new[] { 0, 8 })
        {
            using var session = processor.CreateSession();
            await Measure(session, index, 4, false);
        }
        warmup.Stop();
        report.WarmupMilliseconds = warmup.Elapsed.TotalMilliseconds;
        Save();
        for (var index = 0; index < Questions.Length; index++)
        {
            using var session = processor.CreateSession();
            var result = await Measure(session, index, 16, true);
            report.Results.Add(result);
            Save();
            Console.WriteLine($"Q{index + 1}: prefill {result.InputTokens / (result.PrefillMilliseconds / 1000):F3} tok/s; " +
                $"decode {16 / (result.DecodeMilliseconds / 1000):F3} tok/s; answer={JsonSerializer.Serialize(result.Answer)}");
        }
        if (report.Results.Count != 10 || report.Results.Any(result => result.OutputTokens.Length != 16))
            throw new InvalidDataException("The ten-question benchmark did not complete its fixed output lengths.");
        report.PrefillTokensPerSecond = 167 / (report.Results.Sum(result => result.PrefillMilliseconds) / 1000);
        report.DecodeTokensPerSecond = 160 / (report.Results.Sum(result => result.DecodeMilliseconds) / 1000);
        report.EndToEndOutputTokensPerSecond =
            160 / (report.Results.Sum(result => result.PrefillMilliseconds + result.DecodeMilliseconds) / 1000);
        report.Status = "completed";
        Save();
        Console.WriteLine($"{report.Variant} {architecture} {report.Target}: prefill {report.PrefillTokensPerSecond:F3}, " +
            $"decode {report.DecodeTokensPerSecond:F3}, end-to-end {report.EndToEndOutputTokensPerSecond:F3} tok/s.");

        void Save()
        {
            report.TaskStatistics = backend.TaskStatistics;
            File.WriteAllText(output + ".json", JsonSerializer.Serialize(report,
                new JsonSerializerOptions { WriteIndented = true }));
        }

        async Task<Result> Measure(ProcessorSession session, int index, int count, bool validate)
        {
            var prefill = Stopwatch.StartNew();
            var logits = await session.PrefillAsync(prompts[index]);
            prefill.Stop();
            Finite(logits.Span);
            var prefillValues = validate ? logits.ToArray() : [];
            var tracking = new TrackingSession(session);
            var text = new StringBuilder();
            var decode = Stopwatch.StartNew();
            await foreach (var fragment in RwkvTextGenerator.GenerateFromPrefilledAsync(tracking, tokenizer, logits,
                               new RwkvGenerationOptions
                               {
                                   MaxTokens = count, Temperature = 0,
                                   StopTokenIds = new HashSet<int>(), StopStrings = [],
                               }))
                text.Append(fragment);
            decode.Stop();
            Finite(tracking.LastLogits.Span);
            if (tracking.Tokens.Count != count) throw new InvalidDataException("Generation stopped before its fixed length.");
            var comparisons = new Dictionary<string, NumericComparison>();
            if (validate)
            {
                var prefix = output + $".q{index + 1}";
                File.WriteAllBytes(prefix + ".prefill.bin", MemoryMarshal.AsBytes(prefillValues.AsSpan()).ToArray());
                File.WriteAllBytes(prefix + ".logits.bin", MemoryMarshal.AsBytes(tracking.LastLogits.Span).ToArray());
                using (var state = File.Create(prefix + ".state.bin")) session.SaveState(state);
                if (expected is not null)
                {
                    var other = expected.Results.Single(result => result.QuestionIndex == index + 1);
                    if (other.Question != Questions[index] || !other.PromptTokens.SequenceEqual(prompts[index]) ||
                        !other.OutputTokens.SequenceEqual(tracking.Tokens))
                        throw new InvalidDataException($"Question {index + 1} prompt or generated token IDs differ.");
                    var referencePrefix = reference + $".q{index + 1}";
                    comparisons.Add("prefill", NumericComparison.Compare(
                        MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(referencePrefix + ".prefill.bin")), prefillValues));
                    comparisons.Add("logits", NumericComparison.Compare(
                        MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(referencePrefix + ".logits.bin")), tracking.LastLogits.Span));
                    using var expectedFile = File.OpenRead(referencePrefix + ".state.bin");
                    using var actualFile = File.OpenRead(prefix + ".state.bin");
                    var oldState = GgufStateFile.Read(expectedFile);
                    var newState = GgufStateFile.Read(actualFile);
                    if (oldState.Tensors.Count != newState.Tensors.Count) throw new InvalidDataException("State entry count differs.");
                    foreach (var tensor in oldState.Tensors)
                    {
                        var actual = newState.Tensors.Single(candidate => candidate.Name == tensor.Name);
                        if (!tensor.Dimensions.SequenceEqual(actual.Dimensions)) throw new InvalidDataException("State shape differs.");
                        comparisons.Add("state:" + tensor.Name, NumericComparison.Compare(
                            MemoryMarshal.Cast<byte, float>(tensor.Data.Span), MemoryMarshal.Cast<byte, float>(actual.Data.Span)));
                    }
                }
            }
            return new(index + 1, Questions[index], prompts[index], prompts[index].Length,
                prefill.Elapsed.TotalMilliseconds, decode.Elapsed.TotalMilliseconds,
                tracking.Tokens.ToArray(), text.ToString(), comparisons);
        }
    }

    private static void Finite(ReadOnlySpan<float> values)
    {
        if (values.IsEmpty) throw new InvalidDataException("Empty logits.");
        foreach (var value in values)
            if (!float.IsFinite(value)) throw new InvalidDataException("Nonfinite logits.");
    }

    private sealed class TrackingSession(ProcessorSession session) : IRwkvGenerationSession, IRwkvScopedGenerationSession
    {
        public List<int> Tokens { get; } = [];
        public ReadOnlyMemory<float> LastLogits { get; private set; }
        public ReadOnlyMemory<float> Prefill(ReadOnlySpan<int> tokens) => throw new InvalidOperationException("Already prefilled.");
        public ReadOnlyMemory<float> ForwardToken(int token)
        {
            Tokens.Add(token);
            return LastLogits = session.ForwardToken(token);
        }
        public async ValueTask<IRwkvGenerationScope> BeginGenerationAsync(CancellationToken cancellationToken = default) =>
            new Scope(this, await session.BeginGenerationAsync(cancellationToken));
        private sealed class Scope(TrackingSession tracking, IRwkvGenerationScope inner) : IRwkvGenerationScope, IRwkvAsyncGenerationSession
        {
            public IRwkvGenerationSession Session => this;
            public ReadOnlyMemory<float> Prefill(ReadOnlySpan<int> tokens) => throw new InvalidOperationException("Already prefilled.");
            public ReadOnlyMemory<float> ForwardToken(int token)
            {
                tracking.Tokens.Add(token);
                return tracking.LastLogits = inner.Session.ForwardToken(token);
            }
            public async ValueTask<ReadOnlyMemory<float>> ForwardTokenAsync(int token,
                CancellationToken cancellationToken = default)
            {
                tracking.Tokens.Add(token);
                tracking.LastLogits = inner.Session is IRwkvAsyncGenerationSession asynchronous
                    ? await asynchronous.ForwardTokenAsync(token, cancellationToken).ConfigureAwait(false)
                    : inner.Session.ForwardToken(token);
                return tracking.LastLogits;
            }
            public ValueTask DisposeAsync() => inner.DisposeAsync();
        }
    }

    internal sealed record Result(int QuestionIndex, string Question, int[] PromptTokens, int InputTokens,
        double PrefillMilliseconds, double DecodeMilliseconds, int[] OutputTokens, string Answer,
        Dictionary<string, NumericComparison> Comparisons);

    internal sealed record Report(string Variant, string Target, string Model, double InitializationMilliseconds, string Device)
    {
        public string Status { get; set; } = "running";
        public int FixedDecodeTokens { get; init; } = 16;
        public int PrefillInstances { get; init; } = 2;
        public int InferenceInstances { get; init; } = 2;
        public string TieredCompilation { get; init; } = "0";
        public double WarmupMilliseconds { get; set; }
        public double PrefillTokensPerSecond { get; set; }
        public double DecodeTokensPerSecond { get; set; }
        public double EndToEndOutputTokensPerSecond { get; set; }
        public GpuMatVecOptimizationReport? MatVecDecision { get; set; }
        public TierOneOptimizationReport? TierOneDecision { get; set; }
        public D3D12VmTaskStatistics? TaskStatistics { get; set; }
        public List<Result> Results { get; init; } = [];
    }
}
#endif
