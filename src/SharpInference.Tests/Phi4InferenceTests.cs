using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using SharpInference.Architectures.Phi4;
using SharpInference.Architectures.Phi4.D3D12;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Runtime;
using SharpInference.Vm;

namespace SharpInference.Tests;

public sealed class Phi4InferenceTests
{
    private const string Prompt =
        "<|user|>What is the answer to 1+1? Explain briefly.<|end|><|assistant|>";

    [Phi4ModelFact]
    public void Tokenizer_MatchesGoldenAndRoundTrips()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        int[] expected =
        [
            200021, 4827, 382, 290, 6052, 316, 220, 16,
            10, 16, 30, 115474, 51088, 13, 200020, 200019,
        ];

        var actual = package.Tokenizer.Encode(Prompt);

        Assert.Equal(expected, actual);
        Assert.Equal(Prompt, package.Tokenizer.Decode(actual));
    }

    [Phi4ModelFact]
    public void AudioProcessor_MatchesGoldenFeatures()
    {
        var directory = ModelDirectory();
        var (samples, sampleRate) = ReadPcm16Wave(
            Path.Combine(directory, "golden-inputs", "audio.wav"));
        var actual = new Phi4AudioProcessor().Process(samples, sampleRate);
        var expected = ReadNpyFloat32(Path.Combine(
            directory, "golden", "processor", "speech", "tensors",
            "input_audio_embeds.npy"));

        Assert.Equal(351, actual.FrameCount);
        Assert.Equal(80, actual.FeatureCount);
        Assert.Equal(44, actual.EmbedSize);
        Assert.Equal(expected.Length, actual.Values.Length);
        var maximumError = actual.Values.Zip(expected, static (left, right) => MathF.Abs(left - right)).Max();
        Assert.InRange(maximumError, 0, 2e-6f);
    }

    [Phi4ModelFact]
    public async Task Processor_PreparesCombinedInputAndSelectsVisionAdapter()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        using var processor = new Phi4Processor(package);
        using var session = processor.CreateSession();
        var pixels = Enumerable.Repeat((byte)127, 16 * 16 * 3).ToArray();
        var audio = new byte[400 * sizeof(float)];
        var input = new ProcessorInput(
        [
            new TextInputPart("<|user|>"),
            new ImageInputPart(pixels, 16, 16, 16 * 3, ProcessorPixelFormat.Rgb24),
            new AudioInputPart(audio, ProcessorAudioSampleFormat.Float32, 16000, 1),
            new TextInputPart("<|end|><|assistant|>"),
        ]);

        var prepared = await session.PrepareAsync(input);

        Assert.Single(prepared.Images);
        Assert.Single(prepared.Audios);
        Assert.Equal(Phi4Adapter.Vision, prepared.Adapter);
        Assert.Equal("vision", prepared.AudioProjectionMode);
        Assert.Equal(prepared.TokenIds.Length, prepared.AttentionMask.Length);
        Assert.Equal(prepared.Images[0].ImageTokenCount,
            prepared.TokenIds.Count(token => token == Phi4Tokenizer.ImageTokenId));
        Assert.Equal(prepared.Audios[0].EmbedSize,
            prepared.TokenIds.Count(token => token == Phi4Tokenizer.AudioTokenId));
    }

    [Phi4ModelFact]
    public void TextRuntime_ExecutesRealSingleToken()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        var session = package.CreateTextModel().CreateSession();

        var logits = session.ForwardToken(Phi4Tokenizer.UserTokenId);

        Assert.Equal(200064, logits.Length);
        Assert.All(logits, static value => Assert.True(float.IsFinite(value)));
        Assert.Equal(8256, Array.IndexOf(logits, logits.Max()));
    }

    [Phi4ModelFact]
    public void D3D12ResidentProjections_MatchCpu()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        var cpu = package.CreateTextModel().CreateSession();
        using var projector = new Phi4D3D12MatrixProjector(package);
        var gpu = package.CreateTextModel(projector).CreateSession();

        var expected = cpu.ForwardToken(Phi4Tokenizer.UserTokenId);
        var actual = gpu.ForwardToken(Phi4Tokenizer.UserTokenId);

        Assert.Equal(Array.IndexOf(expected, expected.Max()), Array.IndexOf(actual, actual.Max()));
        var errors = expected.Zip(actual, static (left, right) => MathF.Abs(left - right)).ToArray();
        Assert.InRange(errors.Average(), 0, 0.01);
        Assert.InRange(errors.Max(), 0, 0.1);
    }

    [Phi4ModelFact]
    public void D3D12ResidentTextSession_MatchesCpuDecode()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        var cpu = package.CreateTextModel().CreateSession();
        using var gpu = new Phi4D3D12TextSession(package, maximumContext: 32);
        var tokens = package.Tokenizer.Encode(Prompt);
        var firstExpected = -1;

        foreach (var (token, index) in tokens.Select((token, index) => (token, index)))
        {
            var logits = cpu.ForwardToken(token);
            var expected = Array.IndexOf(logits, logits.Max());
            if (index == 0)
                firstExpected = expected;
            var submissions = gpu.QueueSubmissionCount;

            Assert.Equal(expected, gpu.ForwardToken(token));
            Assert.Equal(submissions + 1, gpu.QueueSubmissionCount);
        }
        Assert.Equal(tokens.Length, gpu.Position);
        gpu.Reset();
        var resetSubmissions = gpu.QueueSubmissionCount;
        Assert.Equal(firstExpected, gpu.ForwardToken(tokens[0]));
        Assert.Equal(resetSubmissions + 1, gpu.QueueSubmissionCount);
        gpu.Reset();
        var embedding = package.Text.GetTensor("token_embd.weight").HalfValues
            .Slice(tokens[0] * 3072, 3072)
            .ToArray()
            .Select(static value => (float)value)
            .ToArray();
        var embeddingSubmissions = gpu.QueueSubmissionCount;
        Assert.Equal(firstExpected, gpu.ForwardEmbedding(embedding));
        Assert.Equal(embeddingSubmissions + 1, gpu.QueueSubmissionCount);
    }

    [Phi4ModelFact]
    public void D3D12ResidentTextSession_MatchesVisionAndSpeechLora()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        foreach (var adapter in new[] { Phi4Adapter.Vision, Phi4Adapter.Speech })
        {
            var cpu = package.CreateTextModel().CreateSession(adapter);
            var logits = cpu.ForwardToken(Phi4Tokenizer.UserTokenId);
            var expected = Array.IndexOf(logits, logits.Max());

            using var gpu = new Phi4D3D12TextSession(
                package, maximumContext: 4, adapter: adapter);

            Assert.Equal(expected, gpu.ForwardToken(Phi4Tokenizer.UserTokenId));
        }
    }

    [Phi4ModelFact]
    public void D3D12FusionComponent_GathersTextEmbeddingsOnGpu()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        var (device, _) = D3D12VmDeviceFactory.Create(0);
        using (device)
        using (var pool = new D3D12VmResourcePool(device))
        using (var fusion = new Phi4D3D12FusionComponent(
            package,
            pool,
            new StorageDomain(
                "d3d12",
                D3D12VmDeviceFactory.HardwareIdentity(0),
                "phi4-test")))
        {
            int[] tokens = [Phi4Tokenizer.UserTokenId, Phi4Tokenizer.AssistantTokenId];
            using var output = fusion.Fuse(tokens);
            var actual = fusion.Readback(output);
            var embeddings = package.Text.GetTensor("token_embd.weight").HalfValues;

            Assert.Equal(tokens.Length * 3072, actual.Length);
            for (var row = 0; row < tokens.Length; row++)
                for (var column = 0; column < 3072; column++)
                    Assert.Equal(
                        (float)embeddings[tokens[row] * 3072 + column],
                        actual[row * 3072 + column]);

            var cpu = package.CreateTextModel().CreateSession();
            using var decoder = new Phi4D3D12TextSession(
                package,
                pool,
                fusion,
                output,
                maximumContext: 4);
            var expected = -1;
            foreach (var token in tokens)
            {
                var logits = cpu.ForwardToken(token);
                expected = Array.IndexOf(logits, logits.Max());
            }
            var submissions = decoder.QueueSubmissionCount;
            Assert.Equal(expected, decoder.PrefillFusedEmbeddings());
            Assert.Equal(submissions + 1, decoder.QueueSubmissionCount);
            var continuation = cpu.ForwardToken(expected);
            Assert.Equal(
                Array.IndexOf(continuation, continuation.Max()),
                decoder.ForwardToken(expected));
        }
    }

    [Phi4ModelFact]
    public void D3D12MultimodalSession_OrchestratesFusionAndGeneration()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        int[] tokens = [Phi4Tokenizer.UserTokenId, Phi4Tokenizer.AssistantTokenId];
        var cpu = package.CreateTextModel().CreateSession();
        float[] logits = [];
        foreach (var token in tokens)
            logits = cpu.ForwardToken(token);
        var expected = Array.IndexOf(logits, logits.Max());
        var prepared = new Phi4PreparedInput(
            tokens,
            [true, true],
            [],
            [],
            Phi4Adapter.None,
            "speech");

        using var session = new Phi4D3D12MultimodalSession(
            package, maximumContext: 4);
        var result = session.Generate(prepared, maximumNewTokens: 1);

        Assert.Equal([expected], result.GeneratedTokenIds);
        Assert.Equal((ulong)((tokens.Length + 15) / 16 + 1), result.DecoderSubmissions);
    }

    [Phi4GoldenModelFact]
    public void D3D12MultimodalSession_MatchesGoldenSpeechDecision()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        var features = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "processor", "speech", "tensors",
            "input_audio_embeds.npy"));
        var tokens = package.Tokenizer.Encode(
            "<|user|><|endoftext11|>Transcribe the attached audio.<|end|><|assistant|>");
        var expanded = tokens.SelectMany(token =>
            token == Phi4Tokenizer.AudioTokenId
                ? Enumerable.Repeat(token, 44)
                : [token]).ToArray();
        var prepared = new Phi4PreparedInput(
            expanded,
            Enumerable.Repeat(true, expanded.Length).ToArray(),
            [],
            [new Phi4AudioFeatures(features, 351, 80, 44)],
            Phi4Adapter.Speech,
            "speech");
        var logits = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "model", "speech", "tensors",
            "prefill__logits.npy"));
        var expected = Array.IndexOf(logits, logits.Max());

        using var session = new Phi4D3D12MultimodalSession(
            package, maximumContext: 128, maximumAudioFrames: 352);
        var result = session.Generate(prepared, maximumNewTokens: 1);

        Assert.Equal([expected], result.GeneratedTokenIds);
        Assert.Equal((ulong)((expanded.Length + 15) / 16 + 1), result.DecoderSubmissions);
    }

    [Phi4GoldenVisionFact]
    public void D3D12MultimodalSession_MatchesGoldenImageAudioDecision()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        var fixture = Path.Combine(
            ModelDirectory(), "golden", "processor", "vision_speech", "tensors");
        var pixels = ReadNpyFloat32(Path.Combine(fixture, "input_image_embeds.npy"));
        var mask = ReadNpyFloat32(Path.Combine(fixture, "image_attention_mask.npy"));
        var audioFeatures = ReadNpyFloat32(Path.Combine(fixture, "input_audio_embeds.npy"));
        var tokens = ReadNpyInt64(Path.Combine(fixture, "input_ids.npy"))
            .Select(value => checked((int)value))
            .ToArray();
        var image = new Phi4ImageFeatures(pixels, mask, 7, 896, 1344, 1841);
        var prepared = new Phi4PreparedInput(
            tokens,
            Enumerable.Repeat(true, tokens.Length).ToArray(),
            [image],
            [new Phi4AudioFeatures(audioFeatures, 351, 80, 44)],
            Phi4Adapter.Vision,
            "vision");
        var logits = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "model", "vision_speech", "tensors",
            "prefill__logits.npy"));
        var expected = Array.IndexOf(logits, logits.Max());

        using var session = new Phi4D3D12MultimodalSession(
            package, maximumContext: 2048, maximumAudioFrames: 352);
        var result = session.Generate(prepared, maximumNewTokens: 1);

        Assert.Equal([expected], result.GeneratedTokenIds);
        Assert.Equal((ulong)((tokens.Length + 15) / 16 + 1), result.DecoderSubmissions);
    }

    [Phi4GoldenVisionFact]
    public void D3D12MultimodalSession_MatchesGoldenImageAudioDecisionWithCpuAudio()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        var fixture = Path.Combine(
            ModelDirectory(), "golden", "processor", "vision_speech", "tensors");
        var pixels = ReadNpyFloat32(Path.Combine(fixture, "input_image_embeds.npy"));
        var mask = ReadNpyFloat32(Path.Combine(fixture, "image_attention_mask.npy"));
        var audioFeatures = ReadNpyFloat32(Path.Combine(fixture, "input_audio_embeds.npy"));
        var tokens = ReadNpyInt64(Path.Combine(fixture, "input_ids.npy"))
            .Select(value => checked((int)value))
            .ToArray();
        var image = new Phi4ImageFeatures(pixels, mask, 7, 896, 1344, 1841);
        var prepared = new Phi4PreparedInput(
            tokens,
            Enumerable.Repeat(true, tokens.Length).ToArray(),
            [image],
            [new Phi4AudioFeatures(audioFeatures, 351, 80, 44)],
            Phi4Adapter.Vision,
            "vision");
        var logits = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "model", "vision_speech", "tensors",
            "prefill__logits.npy"));
        var expected = Array.IndexOf(logits, logits.Max());

        using var session = new Phi4D3D12MultimodalSession(
            package,
            maximumContext: 2048,
            maximumAudioFrames: 352,
            audioBackend: Phi4AudioExecutionBackend.Cpu);
        var result = session.Generate(prepared, maximumNewTokens: 1);

        Assert.Equal([expected], result.GeneratedTokenIds);
        Assert.Equal((ulong)((tokens.Length + 15) / 16 + 1), result.DecoderSubmissions);
    }

    [Phi4GoldenModelFact]
    public void TextRuntime_MatchesGoldenPrefillLogits()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        var session = package.CreateTextModel().CreateSession();
        float[] logits = [];
        foreach (var token in package.Tokenizer.Encode(Prompt))
            logits = session.ForwardToken(token);
        var expected = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "model", "text", "tensors",
            "prefill__logits.npy"));

        Assert.Equal(expected.Length, logits.Length);
        Assert.Equal(Array.IndexOf(expected, expected.Max()), Array.IndexOf(logits, logits.Max()));
        var errors = logits.Zip(expected, static (left, right) => MathF.Abs(left - right)).ToArray();
        Assert.InRange(errors.Average(), 0, 0.12);
        Assert.InRange(errors.Max(), 0, 0.8);
    }

    [Phi4GoldenModelFact]
    public async Task NativeEmbeddingProvider_MatchesGoldenSpeechFusion()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        var features = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "processor", "speech", "tensors",
            "input_audio_embeds.npy"));
        var tokens = package.Tokenizer.Encode(
            "<|user|><|endoftext11|>Transcribe the attached audio.<|end|><|assistant|>");
        var expanded = tokens.SelectMany(token =>
            token == Phi4Tokenizer.AudioTokenId
                ? Enumerable.Repeat(token, 44)
                : [token]).ToArray();
        var input = new Phi4PreparedInput(
            expanded,
            Enumerable.Repeat(true, expanded.Length).ToArray(),
            [],
            [new Phi4AudioFeatures(features, 351, 80, 44)],
            Phi4Adapter.Speech,
            "speech");

        var actual = await package.CreateEmbeddingProvider()
            .CreateFusedEmbeddingsAsync(input, CancellationToken.None);
        var expected = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "model", "speech", "tensors",
            "prefill__fused_embedding.npy"));

        AssertEmbeddingClose(actual, expected, 0.025, 0.2f);
    }

    [Phi4GoldenModelFact]
    public async Task NativeEmbeddingProvider_UsesVisionAudioProjector()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        var features = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "processor", "speech", "tensors",
            "input_audio_embeds.npy"));
        var input = new Phi4PreparedInput(
            Enumerable.Repeat(Phi4Tokenizer.AudioTokenId, 44).ToArray(),
            Enumerable.Repeat(true, 44).ToArray(),
            [],
            [new Phi4AudioFeatures(features, 351, 80, 44)],
            Phi4Adapter.Vision,
            "vision");

        var actual = await package.CreateEmbeddingProvider()
            .CreateFusedEmbeddingsAsync(input, CancellationToken.None);
        var expected = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "model", "vision_speech", "tensors",
            "prefill__audio__vision_projector_output.npy"));

        AssertEmbeddingClose(actual, expected, 0.025, 0.2f);
    }

    [Fact]
    public void D3D12AudioComponent_UsesFixedFrameBuckets()
    {
        Assert.Equal([128, 256, 352, 512, 1024, 2048, 4096],
            Phi4D3D12AudioComponent.SupportedFrameBuckets);
        Assert.Equal(128, Phi4D3D12AudioComponent.GetFrameBucket(1));
        Assert.Equal(352, Phi4D3D12AudioComponent.GetFrameBucket(351));
        Assert.Equal(4096, Phi4D3D12AudioComponent.GetFrameBucket(4096));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => Phi4D3D12AudioComponent.GetFrameBucket(0));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => Phi4D3D12AudioComponent.GetFrameBucket(4097));
    }

    [Phi4GoldenModelFact]
    public void D3D12AudioComponent_MatchesGoldenProjectors()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        var features = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "processor", "speech", "tensors",
            "input_audio_embeds.npy"));
        var input = new Phi4AudioFeatures(features, 351, 80, 44);
        using var component = new Phi4D3D12AudioComponent(package, input.FrameCount);
        var submissions = component.QueueSubmissionCount;

        var vision = component.Encode(input, Phi4AudioProjector.Vision);
        var expectedVision = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "model", "vision_speech", "tensors",
            "prefill__audio__vision_projector_output.npy"));
        AssertEmbeddingClose(ToEmbeddings(vision), expectedVision, 0.025, 0.2f);

        var speech = component.Encode(input, Phi4AudioProjector.Speech);
        var fused = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "model", "speech", "tensors",
            "prefill__fused_embedding.npy"));
        var tokens = package.Tokenizer.Encode(
            "<|user|><|endoftext11|>Transcribe the attached audio.<|end|><|assistant|>");
        var expanded = tokens.SelectMany(token =>
            token == Phi4Tokenizer.AudioTokenId
                ? Enumerable.Repeat(token, input.EmbedSize)
                : [token]).ToArray();
        var expectedSpeech = expanded.Select((token, index) => (token, index))
            .Where(value => value.token == Phi4Tokenizer.AudioTokenId)
            .SelectMany(value => fused.AsSpan(value.index * 3072, 3072).ToArray())
            .ToArray();
        AssertEmbeddingClose(ToEmbeddings(speech), expectedSpeech, 0.025, 0.2f);
        Assert.True(component.QueueSubmissionCount >= submissions + 2);
    }

    [Phi4GoldenModelFact]
    public async Task CpuAudioVm_MatchesGoldenProjectors()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        var features = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "processor", "speech", "tensors",
            "input_audio_embeds.npy"));
        var input = new Phi4AudioFeatures(features, 351, 80, 44);
        var host = new HostMemoryStorageAdapter(
            new StorageDomain("cpu", "0", "phi4-audio-test"));
        using var component = new Phi4CpuAudioComponent(package, host, input.FrameCount);

        await using (var output = component.Encode(input, Phi4AudioProjector.Vision))
        {
            var bytes = new byte[output.Handle.ByteLength];
            await host.DownloadAsync(output.Handle, bytes, CancellationToken.None);
            var actual = MemoryMarshal.Cast<byte, float>(bytes).ToArray();
            var expected = ReadNpyFloat32(Path.Combine(
                ModelDirectory(), "golden", "model", "vision_speech", "tensors",
                "prefill__audio__vision_projector_output.npy"));
            AssertEmbeddingClose(ToEmbeddings(actual[..checked(input.EmbedSize * 3072)]),
                expected, 0.025, 0.2f);
        }

        await using (var output = component.Encode(input, Phi4AudioProjector.Speech))
        {
            var bytes = new byte[output.Handle.ByteLength];
            await host.DownloadAsync(output.Handle, bytes, CancellationToken.None);
            var actual = MemoryMarshal.Cast<byte, float>(bytes).ToArray();
            var fused = ReadNpyFloat32(Path.Combine(
                ModelDirectory(), "golden", "model", "speech", "tensors",
                "prefill__fused_embedding.npy"));
            var tokens = package.Tokenizer.Encode(
                "<|user|><|endoftext11|>Transcribe the attached audio.<|end|><|assistant|>");
            var expanded = tokens.SelectMany(token =>
                token == Phi4Tokenizer.AudioTokenId
                    ? Enumerable.Repeat(token, input.EmbedSize)
                    : [token]).ToArray();
            var expected = expanded.Select((token, index) => (token, index))
                .Where(value => value.token == Phi4Tokenizer.AudioTokenId)
                .SelectMany(value => fused.AsSpan(value.index * 3072, 3072).ToArray())
                .ToArray();
            AssertEmbeddingClose(ToEmbeddings(actual[..checked(input.EmbedSize * 3072)]),
                expected, 0.025, 0.2f);
        }
    }

    [Phi4GoldenModelFact]
    public void D3D12AudioComponent_BindsResidentOutputWithinSharedPool()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        var features = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "processor", "speech", "tensors",
            "input_audio_embeds.npy"));
        var input = new Phi4AudioFeatures(features, 351, 80, 44);
        var (device, deviceName) = D3D12VmDeviceFactory.Create();
        using (device)
        using (var pool = new D3D12VmResourcePool(device))
        using (var component = new Phi4D3D12AudioComponent(
                   package,
                   pool,
                   new("d3d12", deviceName, "phi4-audio-resident-test"),
                   input.FrameCount))
        using (var resident = component.EncodeResident(input, Phi4AudioProjector.Vision))
        {
            var handle = Assert.IsAssignableFrom<IPhi4AudioEmbeddingHandle>(resident.Handle);
            Assert.Equal(input.EmbedSize, handle.ValidTokenCount);
            Assert.Equal(component.TokenBucket, handle.TokenBucket);
            Assert.Equal(Phi4AudioProjector.Vision, handle.Projector);
            Assert.Equal([component.TokenBucket, 3072], handle.Descriptor.Dimensions);
            Assert.Equal("token-major", handle.Descriptor.Layout);

            var program = CreateResidentCopyProgram(component.TokenBucket);
            var artifact = new D3D12VmCompiler(SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create())
                .Compile(program);
            using var executor = component.CreateExecutor(artifact);
            using var resources = new VmResourceManager(
                static (_, _) => throw new InvalidOperationException("No globals expected."),
                pool.Allocate);
            using var bindings = resources.CreateBindings(program);
            component.BindOutput(bindings, "input", resident);
            using var execution = bindings.BeginExecution();
            var context = execution.GetBuffers();
            var outputIndex = program.Slots.Select((slot, index) => (slot, index))
                .Single(value => value.slot.Id == "output").index;

            executor.UploadSlots(context, [outputIndex]);
            executor.Execute("copy");
            executor.ReadbackSlots(context, [outputIndex]);

            var actual = MemoryMarshal.Cast<byte, float>(context[outputIndex]).ToArray();
            var expected = ReadNpyFloat32(Path.Combine(
                ModelDirectory(), "golden", "model", "vision_speech", "tensors",
                "prefill__audio__vision_projector_output.npy"));
            AssertEmbeddingClose(ToEmbeddings(actual), expected, 0.025, 0.2f);
        }
    }

    [Phi4GoldenVisionFact]
    public async Task NativeEmbeddingProvider_MatchesGoldenVisionProjector()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        var pixels = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "processor", "vision", "tensors",
            "input_image_embeds.npy"));
        var mask = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "processor", "vision", "tensors",
            "image_attention_mask.npy"));
        var image = new Phi4ImageFeatures(pixels, mask, 7, 896, 1344, 1841);
        var input = new Phi4PreparedInput(
            Enumerable.Repeat(Phi4Tokenizer.ImageTokenId, image.ImageTokenCount).ToArray(),
            Enumerable.Repeat(true, image.ImageTokenCount).ToArray(),
            [image],
            [],
            Phi4Adapter.Vision,
            "vision");

        var actual = await package.CreateEmbeddingProvider()
            .CreateFusedEmbeddingsAsync(input, CancellationToken.None);
        var expected = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "model", "vision", "tensors",
            "prefill__vision__projector_output.npy"));

        AssertEmbeddingClose(actual, expected, 0.08, 25f);
    }

    [Fact]
    public void D3D12VisionBucket_DerivesFixedHdShape()
    {
        const int crops = 2;
        var mask = new float[crops * 32 * 32];
        mask.AsSpan().Fill(1f);
        var image = new Phi4ImageFeatures(
            new float[crops * 3 * 448 * 448],
            mask,
            crops,
            448,
            448,
            545);

        var bucket = Phi4VisionBucket.FromFeatures(image);

        Assert.Equal(crops, bucket.CropCount);
        Assert.Equal(16, bucket.UsefulHeight);
        Assert.Equal(16, bucket.UsefulWidth);
        Assert.Equal(545, bucket.ImageTokenCount);
    }

    [Phi4GoldenVisionFact]
    public void D3D12VisionComponent_MatchesGoldenVisionProjector()
    {
        using var package = Phi4ModelPackage.Open(ModelDirectory());
        var pixels = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "processor", "vision", "tensors",
            "input_image_embeds.npy"));
        var mask = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "processor", "vision", "tensors",
            "image_attention_mask.npy"));
        var image = new Phi4ImageFeatures(pixels, mask, 7, 896, 1344, 1841);
        var bucket = Phi4VisionBucket.FromFeatures(image);
        using var component = new Phi4D3D12VisionComponent(package);

        using var output = component.Encode(image);

        Assert.Equal(component.Domain, output.Handle.Domain);
        Assert.True(ComponentPortDescriptor.TensorEquals(
            component.ProjectedEmbeddingsPort(bucket).Tensor,
            output.Handle.Descriptor));
        Assert.Equal(1, component.CompiledBucketCount);
        var actual = component.Readback(output);
        var expected = ReadNpyFloat32(Path.Combine(
            ModelDirectory(), "golden", "model", "vision", "tensors",
            "prefill__vision__projector_output.npy"));
        AssertFlatClose(actual, expected, 0.08, 25f);
    }

    private static string ModelDirectory() =>
        Environment.GetEnvironmentVariable("PHI4_TEST_MODEL_DIRECTORY")
        ?? throw new InvalidOperationException("PHI4_TEST_MODEL_DIRECTORY is not configured.");

    private static (float[] Samples, int SampleRate) ReadPcm16Wave(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: false);
        if (new string(reader.ReadChars(4)) != "RIFF")
            throw new InvalidDataException("Expected a RIFF file.");
        _ = reader.ReadUInt32();
        if (new string(reader.ReadChars(4)) != "WAVE")
            throw new InvalidDataException("Expected a WAVE file.");
        int sampleRate = 0, channels = 0, bits = 0;
        byte[]? data = null;
        while (stream.Position < stream.Length)
        {
            var id = new string(reader.ReadChars(4));
            var length = reader.ReadUInt32();
            if (id == "fmt ")
            {
                if (reader.ReadUInt16() != 1)
                    throw new InvalidDataException("Expected PCM audio.");
                channels = reader.ReadUInt16();
                sampleRate = reader.ReadInt32();
                stream.Position += 6;
                bits = reader.ReadUInt16();
                stream.Position += length - 16;
            }
            else if (id == "data")
            {
                var available = stream.Length - stream.Position;
                data = reader.ReadBytes(checked((int)Math.Min(length, (ulong)available)));
                break;
            }
            else
            {
                stream.Position += length;
            }
            if ((length & 1) != 0)
                stream.Position++;
        }
        if (data is null || channels != 1 || bits != 16)
            throw new InvalidDataException("Expected mono signed 16-bit PCM.");
        var values = new float[data.Length / 2];
        for (var index = 0; index < values.Length; index++)
            values[index] = BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(index * 2)) / 32768f;
        return (values, sampleRate);
    }

    private static float[] ReadNpyFloat32(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: false);
        Assert.Equal(new byte[] { 0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59 }, reader.ReadBytes(6));
        var major = reader.ReadByte();
        _ = reader.ReadByte();
        var headerLength = major == 1 ? reader.ReadUInt16() : checked((int)reader.ReadUInt32());
        var header = Encoding.ASCII.GetString(reader.ReadBytes(headerLength));
        Assert.Contains("'descr': '<f4'", header, StringComparison.Ordinal);
        var bytes = reader.ReadBytes(checked((int)(stream.Length - stream.Position)));
        var values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static long[] ReadNpyInt64(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: false);
        Assert.Equal(new byte[] { 0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59 }, reader.ReadBytes(6));
        var major = reader.ReadByte();
        _ = reader.ReadByte();
        var headerLength = major == 1 ? reader.ReadUInt16() : checked((int)reader.ReadUInt32());
        var header = Encoding.ASCII.GetString(reader.ReadBytes(headerLength));
        Assert.Contains("'descr': '<i8'", header, StringComparison.Ordinal);
        var bytes = reader.ReadBytes(checked((int)(stream.Length - stream.Position)));
        var values = new long[bytes.Length / sizeof(long)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static void AssertEmbeddingClose(
        IReadOnlyList<ReadOnlyMemory<float>> actual,
        float[] expected,
        double maximumAverageError,
        float maximumError)
    {
        Assert.Equal(expected.Length, actual.Count * 3072);
        double totalError = 0;
        var largestError = 0f;
        var largestToken = 0;
        var largestChannel = 0;
        var largestActual = 0f;
        var largestExpected = 0f;
        for (var token = 0; token < actual.Count; token++)
        {
            var values = actual[token].Span;
            Assert.Equal(3072, values.Length);
            for (var channel = 0; channel < values.Length; channel++)
            {
                var expectedValue = expected[token * values.Length + channel];
                var error = MathF.Abs(values[channel] - expectedValue);
                totalError += error;
                if (error > largestError)
                {
                    largestError = error;
                    largestToken = token;
                    largestChannel = channel;
                    largestActual = values[channel];
                    largestExpected = expectedValue;
                }

            }
        }
        var averageError = totalError / expected.Length;
        Assert.True(
            averageError <= maximumAverageError && largestError <= maximumError,
            $"Average error {averageError}; largest error {largestError} at token {largestToken}, " +
            $"channel {largestChannel}: actual {largestActual}, expected {largestExpected}.");
    }

    private static ReadOnlyMemory<float>[] ToEmbeddings(float[] values)
    {
        Assert.Equal(0, values.Length % 3072);
        return Enumerable.Range(0, values.Length / 3072)
            .Select(index => new ReadOnlyMemory<float>(values, index * 3072, 3072))
            .ToArray();
    }

    private static VmProgram CreateResidentCopyProgram(int tokenBucket)
    {
        var tensor = new VmTensor(VmElementType.Float32, [tokenBucket, 3072]);
        VmParameter[] parameters =
        [
            new("input", VmAccess.ReadOnly, tensor),
            new("output", VmAccess.ReadWrite, tensor),
        ];
        var arguments = parameters.Select(
            parameter => new VmArgument(parameter.Name, parameter.Name)).ToArray();
        var operation = new VmOperator(
            new(GraphElementType.Float32, GraphElementType.Float32),
            InstructionCollectionIds.TierZeroFloat32,
            "core.copy",
            arguments);
        var kernel = new VmDefinition(
            "copy_kernel",
            VmDefinitionKind.Kernel,
            parameters,
            [new("body", operation)],
            new(64));
        var groups = checked((uint)((tensor.ElementCount + 63) / 64));
        var orchestration = new VmDefinition(
            "copy",
            VmDefinitionKind.Orchestration,
            parameters,
            [
                new("dispatch", new VmDispatch(
                    kernel.Id, arguments, new(groups))),
                new("barrier", new VmBarrier(["output"]), ["dispatch"]),
            ]);
        return new(
            "phi4-audio-resident-copy",
            "test",
            VmTarget.Direct3D12,
            [
                new("input", VmSlotScope.Local, VmAccess.ReadWrite, tensor),
                new("output", VmSlotScope.Local, VmAccess.ReadWrite, tensor),
            ],
            [kernel, orchestration],
            [new("copy", orchestration.Id, arguments)],
            new("none", 1, []));
    }

    private static void AssertFlatClose(
        float[] actual,
        float[] expected,
        double maximumAverageError,
        float maximumError)
    {
        Assert.Equal(expected.Length, actual.Length);
        double totalError = 0;
        var largestError = 0f;
        var largestIndex = 0;
        for (var index = 0; index < actual.Length; index++)
        {
            var error = MathF.Abs(actual[index] - expected[index]);
            totalError += error;
            if (error > largestError)
            {
                largestError = error;
                largestIndex = index;
            }
        }
        var averageError = totalError / expected.Length;
        Assert.True(
            averageError <= maximumAverageError && largestError <= maximumError,
            $"Average error {averageError}; largest error {largestError} at {largestIndex}: " +
            $"actual {actual[largestIndex]}, expected {expected[largestIndex]}.");
    }
}

public class Phi4ModelFactAttribute : FactAttribute
{
    public Phi4ModelFactAttribute()
    {
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("PHI4_TEST_MODEL_DIRECTORY")))
            Skip = "Set PHI4_TEST_MODEL_DIRECTORY to the converted Phi-4 model package.";
    }
}

public sealed class Phi4GoldenModelFactAttribute : Phi4ModelFactAttribute
{
    public Phi4GoldenModelFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("PHI4_RUN_GOLDEN_MODEL") != "1")
            Skip = "Set PHI4_RUN_GOLDEN_MODEL=1 to run the 5.6B Golden prefill comparison.";
    }
}

public sealed class Phi4GoldenVisionFactAttribute : Phi4ModelFactAttribute
{
    public Phi4GoldenVisionFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("PHI4_RUN_GOLDEN_VISION") != "1")
            Skip = "Set PHI4_RUN_GOLDEN_VISION=1 to run the native SigLIP Golden comparison.";
    }
}
