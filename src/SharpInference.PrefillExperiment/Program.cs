using System.Diagnostics;
using SharpInference;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Backends.Cpu;
using SharpInference.Backends.Vortice;
using SharpInference.PrefillExperiment;
using SharpInference.Runtime;

var gpu = false;
var layerMajor = false;
var fullGpuExperiment = false;
var fullGpuResident = false;
var fullGpuNonzero = false;
var fullGpuDiagnostics = false;
var attentionPipeline = false;
var tokenCount = 1024;
var chunkSize = 64;
var headCount = 4;
var headSize = 32;
var repeats = 3;
var modelTokens = 32;
var modelRepeats = 2;
var pipelineTokens = 16;
var pipelineRepeats = 3;
var projectionWidth = 256;
string? modelPath = null;

var argumentIndex = 0;
for (; argumentIndex < args.Length; argumentIndex++)
{
    switch (args[argumentIndex])
    {
        case "--gpu":
            gpu = true;
            break;
        case "--layer-major":
            layerMajor = true;
            break;
        case "--full-gpu-experiment":
            fullGpuExperiment = true;
            break;
        case "--full-gpu-resident":
            fullGpuResident = true;
            break;
        case "--full-gpu-nonzero":
            fullGpuNonzero = true;
            break;
        case "--full-gpu-diagnostics":
            fullGpuDiagnostics = true;
            break;
        case "--main-prefill-compare":
            throw new NotSupportedException(
                "The legacy prefill comparison has been removed; use --layer-major or --full-gpu-resident for experimental comparisons.");
        case "--attention-pipeline":
            attentionPipeline = true;
            break;
        case "--model":
            modelPath = Next();
            break;
        case "--tokens":
            tokenCount = Positive(Next());
            break;
        case "--chunk":
            chunkSize = Positive(Next());
            break;
        case "--heads":
            headCount = Positive(Next());
            break;
        case "--head-size":
            headSize = Positive(Next());
            break;
        case "--repeats":
            repeats = Positive(Next());
            break;
        case "--model-tokens":
            modelTokens = Positive(Next());
            break;
        case "--model-repeats":
            modelRepeats = Positive(Next());
            break;
        case "--pipeline-tokens":
            pipelineTokens = Positive(Next());
            break;
        case "--pipeline-repeats":
            pipelineRepeats = Positive(Next());
            break;
        case "--projection-width":
            projectionWidth = Positive(Next());
            break;
        default:
            throw new ArgumentException($"Unknown argument '{args[argumentIndex]}'.");
    }

}

if (layerMajor && (gpu || modelPath is null))
    throw new ArgumentException("--layer-major requires --model and the CPU backend (omit --gpu).");
if (attentionPipeline && (!gpu || modelPath is null))
    throw new ArgumentException("--attention-pipeline requires --gpu and --model.");
if (fullGpuExperiment && (!gpu || modelPath is null))
    throw new ArgumentException("--full-gpu-experiment requires --gpu and --model.");
if (fullGpuResident && (!gpu || modelPath is null || fullGpuExperiment))
    throw new ArgumentException("--full-gpu-resident requires --gpu and --model, and excludes --full-gpu-experiment.");
if (fullGpuNonzero && !(fullGpuExperiment || fullGpuResident))
    throw new ArgumentException("--full-gpu-nonzero requires a full-model GPU experiment.");
if (fullGpuDiagnostics && !(fullGpuExperiment || fullGpuResident))
    throw new ArgumentException("--full-gpu-diagnostics requires a full-model GPU experiment.");
string Next()
{
    if (++argumentIndex == args.Length)
        throw new ArgumentException($"Missing value after '{args[argumentIndex - 1]}'.");
    return args[argumentIndex];
}

static int Positive(string value) =>
    int.TryParse(value, out var parsed) && parsed > 0
        ? parsed : throw new ArgumentException($"Expected a positive integer, got '{value}'.");

foreach (var length in new[] { 1, 2, 63, 64, 65, 127, 128, 129, 257 })
{
    CpuWkv6Benchmark.Validate(new Wkv6Inputs(2, 8, length), chunkSize);
    _ = CpuWkv6OutputsBenchmark.Measure(
        new Wkv6Sequence(new Wkv6Inputs(2, 8, length)), chunkSize, 1);
}
Console.WriteLine("RWKV-6 state and all-token outputs: 9 lengths, nonzero initial state: PASS");

var inputs = new Wkv6Inputs(headCount, headSize, tokenCount);
CpuWkv6Benchmark.Validate(inputs, chunkSize);
var (sequential, chunked) = CpuWkv6Benchmark.Measure(inputs, chunkSize, repeats);
Console.WriteLine($"CPU WKV-only: tokens={tokenCount} heads={headCount} head-size={headSize} chunk={chunkSize} " +
    $"sequential={sequential.TotalMilliseconds:F3}ms chunked={chunked.TotalMilliseconds:F3}ms " +
    $"speedup={sequential.TotalMilliseconds / chunked.TotalMilliseconds:F2}x (average of {repeats})");
var sequence = new Wkv6Sequence(inputs);
var (allSequential, allChunked) = CpuWkv6OutputsBenchmark.Measure(sequence, chunkSize, repeats);
Console.WriteLine($"CPU all WKV outputs: tokens={tokenCount} sequential={allSequential.TotalMilliseconds:F3}ms " +
    $"chunked={allChunked.TotalMilliseconds:F3}ms " +
    $"speedup={allSequential.TotalMilliseconds / allChunked.TotalMilliseconds:F2}x");
var (matVec, batchedProjection) = CpuProjectionBenchmark.Measure(tokenCount, projectionWidth, 16, repeats);
_ = CpuProjectionBenchmark.Measure(17, 31, 16, 1);
var (scalarProjection, simdProjection) =
    CpuProjectionBenchmark.MeasureVectorized(tokenCount, projectionWidth, 16, repeats);
_ = CpuProjectionBenchmark.MeasureVectorized(17, 31, 16, 1);
Console.WriteLine($"CPU synthetic projection: tokens={tokenCount} width={projectionWidth} batch=16 " +
    $"token-wise={matVec.TotalMilliseconds:F3}ms batched={batchedProjection.TotalMilliseconds:F3}ms " +
    $"speedup={matVec.TotalMilliseconds / batchedProjection.TotalMilliseconds:F2}x");
Console.WriteLine($"CPU SIMD projection: tokens={tokenCount} width={projectionWidth} batch=16 " +
    $"token-wise-Dot={scalarProjection.TotalMilliseconds:F3}ms " +
    $"batched-SIMD={simdProjection.TotalMilliseconds:F3}ms " +
    $"speedup={scalarProjection.TotalMilliseconds / simdProjection.TotalMilliseconds:F2}x");

if (gpu)
{
    foreach (var length in new[] { 1, 65 })
    {
        var boundary = new Wkv6Inputs(2, 8, length);
        var check = GpuWkv6Benchmark.Run(2, 8, length, 64,
            boundary.Keys, boundary.Values, boundary.Decays, boundary.InitialState, 1);
        CpuWkv6Benchmark.AssertClose(CpuWkv6Benchmark.Sequential(boundary, length),
            check.ChunkedState, "DirectX chunk boundary state");
        var boundarySequence = new Wkv6Sequence(boundary);
        _ = GpuWkv6OutputsBenchmark.Run(2, 8, length, 64, boundary.Keys,
            boundary.Values, boundary.Decays, boundarySequence.Receptances,
            boundarySequence.TimeFirst, boundary.InitialState, 1);
    }
    _ = GpuProjectionBenchmark.Run(17, 31, 1);
    _ = GpuFp16ProjectionBenchmark.Run(17, 31, 1);
    Console.WriteLine("DirectX kernel boundaries: lengths 1 and 65; projection 17x31: PASS");
    var result = GpuWkv6Benchmark.Run(headCount, headSize, tokenCount, chunkSize,
        inputs.Keys, inputs.Values, inputs.Decays, inputs.InitialState, repeats);
    var expected = CpuWkv6Benchmark.Sequential(inputs, tokenCount);
    CpuWkv6Benchmark.AssertClose(expected, result.SequentialState, "DirectX sequential state");
    CpuWkv6Benchmark.AssertClose(expected, result.ChunkedState, "DirectX chunked state");
    Console.WriteLine($"DirectX WKV-only: device={result.Device} sequential={result.Sequential.TotalMilliseconds:F3}ms " +
        $"chunked={result.Chunked.TotalMilliseconds:F3}ms " +
        $"speedup={result.Sequential.TotalMilliseconds / result.Chunked.TotalMilliseconds:F2}x " +
        $"(average of {repeats})");
    var allOutput = GpuWkv6OutputsBenchmark.Run(headCount, headSize, tokenCount, chunkSize,
        inputs.Keys, inputs.Values, inputs.Decays, sequence.Receptances,
        sequence.TimeFirst, inputs.InitialState, repeats);
    Console.WriteLine($"DirectX all WKV outputs: sequential={allOutput.Sequential.TotalMilliseconds:F3}ms " +
        $"chunked={allOutput.Chunked.TotalMilliseconds:F3}ms " +
        $"speedup={allOutput.Sequential.TotalMilliseconds / allOutput.Chunked.TotalMilliseconds:F2}x " +
        $"max-output-error={allOutput.ChunkedOutputMaxError:G5} max-state-error={allOutput.ChunkedStateMaxError:G5}");
    var projection = GpuProjectionBenchmark.Run(tokenCount, projectionWidth, repeats);
    Console.WriteLine($"DirectX synthetic projection: tokens={tokenCount} width={projectionWidth} " +
        $"token-wise={projection.Sequential.TotalMilliseconds:F3}ms tiled={projection.Batched.TotalMilliseconds:F3}ms " +
        $"speedup={projection.Sequential.TotalMilliseconds / projection.Batched.TotalMilliseconds:F2}x " +
        $"max-absolute-error={projection.MaxDifference:G5}");
    var fp16Projection = GpuFp16ProjectionBenchmark.Run(tokenCount, projectionWidth, repeats);
    Console.WriteLine($"DirectX FP16-weight projection: tokens={tokenCount} width={projectionWidth} " +
        $"token-wise={fp16Projection.Sequential.TotalMilliseconds:F3}ms " +
        $"tiled={fp16Projection.Batched.TotalMilliseconds:F3}ms " +
        $"speedup={fp16Projection.Sequential.TotalMilliseconds / fp16Projection.Batched.TotalMilliseconds:F2}x " +
        $"max-absolute-error={fp16Projection.MaxDifference:G5}");
}

(float[] Logits, float[] State)? gpuReference = null;
float[]? gpuInitialState = null;
TimeSpan gpuPrefill = default;
if (modelPath is not null)
{
    if (!File.Exists(modelPath))
        throw new FileNotFoundException("The GGML model is required for the full-model baseline.", modelPath);
    if (gpu)
    {
        using var processor = new ProcessorPipelineBuilder(modelPath)
            .UseReader(new GgmlModelReader())
            .UseProvider(new PortableRwkv6GraphProvider())
            .UseBackend(_ => VorticePrimitiveGraphBackend.FromConfig(new VorticeRuntimeConfig()))
            .UsePortableGraphArchitecture()
            .Build();
        var measured = MeasureModel(processor, modelTokens, modelRepeats, "DirectX portable graph");
        if (fullGpuExperiment || fullGpuResident)
        {
            gpuReference = (measured.Logits, measured.State);
            gpuPrefill = measured.Prefill;
            using var initial = processor.CreateSession();
            if (fullGpuNonzero)
            {
                _ = initial.Prefill([0, 1, 2]);
                var snapshot = ExperimentStateSnapshots.Capture(initial);
                gpuInitialState = ExperimentStateSnapshots.LayerMajor(snapshot,
                    processor.Metadata.LayerCount, processor.Metadata.EmbeddingSize, processor.Metadata.HeadSize);
                using var reference = processor.CreateSession();
                ExperimentStateSnapshots.Restore(reference, snapshot);
                var tokens = Enumerable.Range(0, modelTokens)
                    .Select(i => i % processor.Metadata.VocabularySize).ToArray();
                var watch = Stopwatch.StartNew();
                var last = reference.Prefill(tokens);
                gpuPrefill = watch.Elapsed;
                gpuReference = (last.ToArray(), ExperimentStateSnapshots.LayerMajor(reference,
                    processor.Metadata.LayerCount, processor.Metadata.EmbeddingSize, processor.Metadata.HeadSize));
            }
            else
                gpuInitialState = ExperimentStateSnapshots.LayerMajor(initial,
                    processor.Metadata.LayerCount, processor.Metadata.EmbeddingSize, processor.Metadata.HeadSize);
        }
        using var modelCatalog = GgmlModelFile.Open(modelPath);
        var keyProjection = modelCatalog.GetRequired("blocks.0.att.key.weight");
        if (keyProjection.DataType != RwkvTensorDataType.Float16)
            throw new InvalidDataException("The real-weight GPU projection experiment requires a FP16 key tensor.");
        var realProjectionTokens = Math.Min(16, modelTokens);
        var realProjection = GpuFp16ProjectionBenchmark.Run(
            realProjectionTokens, processor.Metadata.EmbeddingSize, repeats,
            keyProjection.HalfValues.ToArray());
        Console.WriteLine($"DirectX real-model FP16 key projection: tokens={realProjectionTokens} " +
            $"width={processor.Metadata.EmbeddingSize} " +
            $"token-wise={realProjection.Sequential.TotalMilliseconds:F3}ms " +
            $"tiled={realProjection.Batched.TotalMilliseconds:F3}ms " +
            $"speedup={realProjection.Sequential.TotalMilliseconds / realProjection.Batched.TotalMilliseconds:F2}x " +
            $"max-absolute-error={realProjection.MaxDifference:G5}");
        if (attentionPipeline)
        {
            var probe = Rwkv6AttentionProbeInputs.FromFirstLayer(modelCatalog, pipelineTokens);
            var pipeline = GpuAttentionPipelineBenchmark.Run(
                probe.HeadCount, probe.HeadSize, pipelineTokens, chunkSize,
                probe.KeyWeights, probe.ValueWeights, probe.ReceptanceWeights,
                probe.KeyActivations, probe.ValueActivations, probe.ReceptanceActivations,
                probe.Decays, probe.TimeFirst, probe.InitialState, pipelineRepeats);
            Console.WriteLine($"DirectX GPU-resident first-layer K/V/R -> WKV probe: tokens={pipelineTokens} " +
                $"chunk={chunkSize} runs={pipelineRepeats} sequential={pipeline.Sequential.TotalMilliseconds:F3}ms " +
                $"tiled-chunked={pipeline.Chunked.TotalMilliseconds:F3}ms " +
                $"speedup={pipeline.Sequential.TotalMilliseconds / pipeline.Chunked.TotalMilliseconds:F2}x " +
                $"max-projection-error={pipeline.ChunkedProjectionMaxError:G5} " +
                $"max-output-error={pipeline.ChunkedOutputMaxError:G5} " +
                $"max-state-error={pipeline.ChunkedStateMaxError:G5} " +
                $"allocated-bytes-estimate={pipeline.AllocatedGpuBytes}");
        }
    }
    else
    {
        using var processor = Processor.LoadGraph(modelPath,
            new PortableRwkv6GraphProvider(), CpuPrimitiveGraphBackend.Instance);
        MeasureModel(processor, modelTokens, modelRepeats, "CPU");
        if (layerMajor)
            MeasureLayerMajor(processor, modelPath, modelTokens, chunkSize, modelRepeats);
    }

    static void MeasureLayerMajor(Processor processor, string modelPath, int count, int chunkSize, int repeats)
    {
        using var catalog = GgmlModelFile.Open(modelPath);
        using var prefix = processor.CreateSession();
        _ = prefix.Prefill([0, 1, 2]);
        var initialSnapshot = ExperimentStateSnapshots.Capture(prefix);
        var initialState = ExperimentStateSnapshots.LayerMajor(initialSnapshot,
            processor.Metadata.LayerCount, processor.Metadata.EmbeddingSize, processor.Metadata.HeadSize);
        var tokens = Enumerable.Range(0, count)
            .Select(i => i % processor.Metadata.VocabularySize).ToArray();
        _ = CpuLayerMajorRwkv6Experiment.Run(catalog, tokens[..Math.Min(count, 3)], initialState, chunkSize);
        _ = CpuLayerMajorRwkv6Experiment.Run(
            catalog, tokens[..Math.Min(count, 3)], initialState, chunkSize, batchedProjections: true);
        var ordinary = TimeSpan.Zero;
        var experimental = TimeSpan.Zero;
        var serialWkv = TimeSpan.Zero;
        var batchedProjections = TimeSpan.Zero;
        float maximumLogits = 0;
        float maximumState = 0;
        for (var run = 0; run < repeats; run++)
        {
            using var reference = processor.CreateSession();
            ExperimentStateSnapshots.Restore(reference, initialSnapshot);
            (float[] Logits, float[] State) trial = default;
            (float[] Logits, float[] State) serial = default;
            (float[] Logits, float[] State) batched = default;
            float[]? expectedLogits = null;
            TimeSpan Existing()
            {
                var clock = Stopwatch.StartNew();
                var result = reference.Prefill(tokens);
                var elapsed = clock.Elapsed;
                expectedLogits = result.ToArray();
                return elapsed;
            }
            TimeSpan LayerMajor()
            {
                var clock = Stopwatch.StartNew();
                trial = CpuLayerMajorRwkv6Experiment.Run(catalog, tokens, initialState, chunkSize);
                return clock.Elapsed;
            }
            TimeSpan LayerMajorSerialWkv()
            {
                var clock = Stopwatch.StartNew();
                serial = CpuLayerMajorRwkv6Experiment.Run(
                    catalog, tokens, initialState, chunkSize, chunkedWkv: false);
                return clock.Elapsed;
            }
            TimeSpan LayerMajorBatchedProjections()
            {
                var clock = Stopwatch.StartNew();
                batched = CpuLayerMajorRwkv6Experiment.Run(
                    catalog, tokens, initialState, chunkSize, batchedProjections: true);
                return clock.Elapsed;
            }
            if (run % 2 == 0)
            {
                ordinary += Existing();
                serialWkv += LayerMajorSerialWkv();
                experimental += LayerMajor();
                batchedProjections += LayerMajorBatchedProjections();
            }
            else
            {
                batchedProjections += LayerMajorBatchedProjections();
                experimental += LayerMajor();
                serialWkv += LayerMajorSerialWkv();
                ordinary += Existing();
            }

            maximumLogits = MathF.Max(maximumLogits,
                CheckDifference(expectedLogits!, trial.Logits, "layer-major logits"));
            maximumState = MathF.Max(maximumState,
                CheckDifference(ExperimentStateSnapshots.LayerMajor(reference,
                    processor.Metadata.LayerCount, processor.Metadata.EmbeddingSize, processor.Metadata.HeadSize),
                    trial.State, "layer-major state"));
            _ = CheckDifference(expectedLogits, serial.Logits, "layer-major serial-WKV logits");
            _ = CheckDifference(ExperimentStateSnapshots.LayerMajor(reference,
                processor.Metadata.LayerCount, processor.Metadata.EmbeddingSize, processor.Metadata.HeadSize),
                serial.State, "layer-major serial-WKV state");
            _ = CheckDifference(expectedLogits, batched.Logits, "layer-major batched-projection logits");
            _ = CheckDifference(ExperimentStateSnapshots.LayerMajor(reference,
                processor.Metadata.LayerCount, processor.Metadata.EmbeddingSize, processor.Metadata.HeadSize),
                batched.State, "layer-major batched-projection state");
        }
        ordinary /= repeats;
        experimental /= repeats;
        serialWkv /= repeats;
        batchedProjections /= repeats;
        Console.WriteLine($"CPU layer-major full model: tokens={count} chunk={chunkSize} runs={repeats} " +
            $"existing-Prefill={ordinary.TotalMilliseconds:F3}ms " +
            $"layer-major-serial-WKV={serialWkv.TotalMilliseconds:F3}ms " +
            $"layer-major-chunked-WKV={experimental.TotalMilliseconds:F3}ms " +
            $"layer-major-batched-projections={batchedProjections.TotalMilliseconds:F3}ms " +
            $"speedup={ordinary.TotalMilliseconds / batchedProjections.TotalMilliseconds:F2}x " +
            $"max-logit-error={maximumLogits:G5} max-state-error={maximumState:G5}");
    }

    static float CheckDifference(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string label)
    {
        if (expected.Length != actual.Length)
            throw new InvalidOperationException($"{label}: length mismatch.");
        var maximum = 0f;
        for (var i = 0; i < expected.Length; i++)
        {
            var error = MathF.Abs(expected[i] - actual[i]);
            if (!float.IsFinite(actual[i]) || error > 0.0002f + MathF.Abs(expected[i]) * 0.001f)
                throw new InvalidOperationException(
                    $"{label}: mismatch at {i}: expected={expected[i]:G9}, actual={actual[i]:G9}.");
            maximum = MathF.Max(maximum, error);
        }
        return maximum;
    }
}

if (fullGpuExperiment)
{
    using var catalog = GgmlModelFile.Open(modelPath!);
    using var projector = new ReusableGpuProjector();
    using var wkv = new ReusableGpuWkv6Stage();
    var tokens = Enumerable.Range(0, modelTokens).Select(i => i % catalog.VocabularySize).ToArray();
    var warm = CpuLayerMajorRwkv6Experiment.Run(catalog, tokens, gpuInitialState!, chunkSize,
        gpuProjector: projector, gpuWkv: wkv);
    var logitsCheck = Compare(gpuReference!.Value.Logits, warm.Logits,
        "DirectX hybrid full-model logits", !fullGpuDiagnostics);
    var stateCheck = Compare(gpuReference.Value.State, warm.State,
        "DirectX hybrid full-model state", !fullGpuDiagnostics, 0.0005f, 0.002f);
    Console.WriteLine($"DirectX hybrid full-model correctness: tokens={modelTokens} nonzero-state={fullGpuNonzero} " +
        $"max-logits-error={logitsCheck.Maximum:G5} max-state-error={stateCheck.Maximum:G5} " +
        $"logits-outside-tolerance={logitsCheck.Mismatches} state-outside-tolerance={stateCheck.Mismatches} " +
        $"worst-state-tolerance-ratio={stateCheck.WorstRatio:G5}");
    var warmProjectionCount = projector.ProjectCount;
    var warmProjectionTime = projector.TotalElapsed;
    var elapsed = TimeSpan.Zero;
    var repeatedMismatches = 0;
    for (var run = 0; run < modelRepeats; run++)
    {
        var watch = Stopwatch.StartNew();
        var trial = CpuLayerMajorRwkv6Experiment.Run(catalog, tokens, gpuInitialState!, chunkSize,
            gpuProjector: projector, gpuWkv: wkv);
        elapsed += watch.Elapsed;
        repeatedMismatches += Compare(gpuReference.Value.Logits, trial.Logits,
            "DirectX hybrid repeated logits", !fullGpuDiagnostics).Mismatches;
        repeatedMismatches += Compare(gpuReference.Value.State, trial.State,
            "DirectX hybrid repeated state", !fullGpuDiagnostics, 0.0005f, 0.002f).Mismatches;
    }
    elapsed /= modelRepeats;
    Console.WriteLine($"DirectX hybrid full-model: tokens={modelTokens} runs={modelRepeats} " +
        $"production-Prefill={gpuPrefill.TotalMilliseconds:F3}ms " +
        $"experimental-prefill={elapsed.TotalMilliseconds:F3}ms " +
        $"speedup={gpuPrefill.TotalMilliseconds / elapsed.TotalMilliseconds:F2}x " +
        $"gpu-projections-per-run={(projector.ProjectCount - warmProjectionCount) / modelRepeats} " +
        $"gpu-projection-per-run={(projector.TotalElapsed - warmProjectionTime).TotalMilliseconds / modelRepeats:F3}ms " +
        "(warm GPU weights cached; CPU normalization/mixing and per-projection GPU readback)");
    if (logitsCheck.Mismatches != 0 || stateCheck.Mismatches != 0 || repeatedMismatches != 0)
        Environment.ExitCode = 1;
}

if (fullGpuResident)
{
    using var catalog = GgmlModelFile.Open(modelPath!);
    using var resident = new GpuResidentRwkv6Experiment(catalog, chunkSize);
    var tokens = Enumerable.Range(0, modelTokens).Select(i => i % catalog.VocabularySize).ToArray();
    var warm = resident.Run(tokens, gpuInitialState!);
    var logitsCheck = Compare(gpuReference!.Value.Logits, warm.Logits,
        "DirectX resident full-model logits", !fullGpuDiagnostics);
    var stateCheck = Compare(gpuReference.Value.State, warm.State,
        "DirectX resident full-model state", !fullGpuDiagnostics, 0.001f, 0.003f);
    var tightStateCheck = Compare(gpuReference.Value.State, warm.State,
        "DirectX resident stricter-state diagnostic", false, 0.0005f, 0.002f);
    Console.WriteLine($"DirectX resident full-model correctness: tokens={modelTokens} nonzero-state={fullGpuNonzero} " +
        $"max-logits-error={logitsCheck.Maximum:G5} max-state-error={stateCheck.Maximum:G5} " +
        $"logits-outside-tolerance={logitsCheck.Mismatches} state-outside-tolerance={stateCheck.Mismatches} " +
        $"state-outside-stricter-tolerance={tightStateCheck.Mismatches} " +
        $"worst-state-tolerance-ratio={stateCheck.WorstRatio:G5}");
    var elapsed = TimeSpan.Zero;
    var repeatedMismatches = 0;
    for (var run = 0; run < modelRepeats; run++)
    {
        var watch = Stopwatch.StartNew();
        var trial = resident.Run(tokens, gpuInitialState!);
        elapsed += watch.Elapsed;
        repeatedMismatches += Compare(gpuReference.Value.Logits, trial.Logits,
            "DirectX resident repeated logits", !fullGpuDiagnostics).Mismatches;
        repeatedMismatches += Compare(gpuReference.Value.State, trial.State,
            "DirectX resident repeated state", !fullGpuDiagnostics, 0.001f, 0.003f).Mismatches;
    }
    elapsed /= modelRepeats;
    Console.WriteLine($"DirectX resident full-model: tokens={modelTokens} runs={modelRepeats} " +
        $"production-Prefill={gpuPrefill.TotalMilliseconds:F3}ms " +
        $"experimental-prefill={elapsed.TotalMilliseconds:F3}ms " +
        $"speedup={gpuPrefill.TotalMilliseconds / elapsed.TotalMilliseconds:F2}x " +
        "(warm model weights; model load and shader compilation excluded)");
    if (logitsCheck.Mismatches != 0 || stateCheck.Mismatches != 0 || repeatedMismatches != 0)
        Environment.ExitCode = 1;
}

static (float Maximum, int Mismatches, float WorstRatio) Compare(
    ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string label, bool strict,
    float absoluteTolerance = 0.0002f, float relativeTolerance = 0.001f)
{
    if (actual.Length != expected.Length)
        throw new InvalidOperationException($"{label}: expected {expected.Length} values, received {actual.Length}.");
    float max = 0;
    float worstRatio = 0;
    var worstIndex = -1;
    var mismatches = 0;
    var firstMismatch = -1;
    for (var i = 0; i < actual.Length; i++)
    {
        var difference = MathF.Abs(actual[i] - expected[i]);
        if (!float.IsFinite(expected[i]) || !float.IsFinite(actual[i]))
            throw new InvalidOperationException($"{label} contains a non-finite value at {i}.");
        var tolerance = absoluteTolerance + relativeTolerance * MathF.Abs(expected[i]);
        var ratio = difference / tolerance;
        if (ratio > worstRatio)
        {
            worstRatio = ratio;
            worstIndex = i;
        }
        if (difference > tolerance)
        {
            if (firstMismatch < 0) firstMismatch = i;
            mismatches++;
        }
        max = MathF.Max(max, difference);
    }
    if (mismatches != 0 && strict)
        throw new InvalidOperationException(
            $"{label}: {mismatches}/{actual.Length} values exceed tolerance; first at {firstMismatch}: " +
            $"expected={expected[firstMismatch]:G9} actual={actual[firstMismatch]:G9}; maximum error={max:G9}.");
    if (mismatches != 0 && !strict)
        Console.WriteLine($"{label}: worst normalized error at {worstIndex}: " +
            $"expected={expected[worstIndex]:G9} actual={actual[worstIndex]:G9}, " +
            $"tolerance-ratio={worstRatio:G5}; first failure at {firstMismatch}: " +
            $"expected={expected[firstMismatch]:G9} actual={actual[firstMismatch]:G9}.");
    return (max, mismatches, worstRatio);
}

static (TimeSpan Prefill, float[] Logits, float[] State) MeasureModel(
    Processor processor, int count, int repeats, string name)
{
    Console.WriteLine($"{name} model: layers={processor.Metadata.LayerCount} embedding={processor.Metadata.EmbeddingSize} " +
        $"heads={processor.Metadata.HeadCount} head-size={processor.Metadata.HeadSize}");
    var vocabulary = processor.Metadata.VocabularySize;
    var tokens = Enumerable.Range(0, count).Select(i => i % vocabulary).ToArray();
    using (var warmup = processor.CreateSession())
        _ = warmup.ForwardToken(tokens[0]);

    var sequential = TimeSpan.Zero;
    var batched = TimeSpan.Zero;
    float[]? lastLogits = null;
    float[]? lastState = null;
    for (var run = 0; run < repeats; run++)
    {
        using var baseline = processor.CreateSession();
        using var prefill = processor.CreateSession();
        float[]? expectedLogits = null;
        float[]? actualLogits = null;
        TimeSpan MeasureSequential()
        {
            var watch = Stopwatch.StartNew();
            ReadOnlyMemory<float> last = default;
            foreach (var token in tokens)
                last = baseline.ForwardToken(token);
            var elapsed = watch.Elapsed;
            expectedLogits = last.ToArray();
            return elapsed;
        }
        TimeSpan MeasurePrefill()
        {
            var watch = Stopwatch.StartNew();
            var last = prefill.Prefill(tokens);
            var elapsed = watch.Elapsed;
            actualLogits = last.ToArray();
            return elapsed;
        }
        if (run % 2 == 0)
        {
            sequential += MeasureSequential();
            batched += MeasurePrefill();
        }
        else
        {
            batched += MeasurePrefill();
            sequential += MeasureSequential();
        }
        CpuWkv6Benchmark.AssertClose(expectedLogits!, actualLogits!, $"{name} prefill logits");
        var state = ExperimentStateSnapshots.LayerMajor(prefill,
            processor.Metadata.LayerCount, processor.Metadata.EmbeddingSize, processor.Metadata.HeadSize);
        _ = ExperimentStateSnapshots.Compare(ExperimentStateSnapshots.Capture(baseline),
            ExperimentStateSnapshots.Capture(prefill),
            (expected, actual, tensor) =>
            {
                CpuWkv6Benchmark.AssertClose(expected.Span, actual.Span, $"{name} prefill state {tensor}");
                return 0;
            });
        lastLogits = actualLogits;
        lastState = state;
    }
    sequential /= repeats;
    batched /= repeats;
    Console.WriteLine($"{name} full-model baseline: tokens={count} runs={repeats} ForwardToken={sequential.TotalMilliseconds:F3}ms " +
        $"Prefill={batched.TotalMilliseconds:F3}ms ratio={sequential.TotalMilliseconds / batched.TotalMilliseconds:F2}x " +
        "(portable production graph; NOT chunked full-model inference)");
    return (batched, lastLogits!, lastState!);
}
