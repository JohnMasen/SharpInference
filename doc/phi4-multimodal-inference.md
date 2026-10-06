# Phi-4 multimodal inference

SharpInference supports `microsoft/Phi-4-multimodal-instruct` through the
`SharpInference.Architectures.Phi4` project. The initial backend is native CPU
execution over converted F16 GGUF components.

## Model package

`Phi4ModelPackage.Open` expects these files in one directory:

- `Phi-4-multimodal-text-f16.gguf`
- `Phi-4-multimodal-omni-f16.gguf`
- `Phi-4-multimodal-vision-lora-f16.gguf`
- `Phi-4-multimodal-speech-lora-f16.gguf`
- `tokenizer.json`

The loader validates architecture metadata, tensor counts, dimensions, and
adapter identity before exposing the package.

## Processor

`Phi4Processor` implements `IProcessor` and accepts ordered `TextInputPart`,
`ImageInputPart`, and `AudioInputPart` values. Images are decoded pixel buffers;
audio is decoded PCM. File, network, image-container, and audio-container
decoding remain outside the processor.

The processor performs:

- GPT-4o ByteLevel-BPE tokenization and Phi-4 special-token handling.
- Dynamic-HD image resize, tiling, masks, SigLIP encoding, token compression,
  HD rearrangement, and projection.
- SpeechLib-compatible 80-bin features, Conformer encoding, and projection.
- Parallel image and audio encoding when both modalities are present.
- Media embedding fusion at repeated image/audio token positions.
- Decoder prefill, KV-cache decode, and greedy text generation.

Vision input selects the vision LoRA. Speech-only input selects the speech
LoRA. Combined vision and speech selects the vision LoRA and the vision-mode
audio projector, matching the Microsoft reference implementation.

```csharp
using var package = Phi4ModelPackage.Open(modelDirectory);
using var processor = new Phi4Processor(package);
using var session = processor.CreateSession();

var result = await session.GenerateAsync(
    new ProcessorInput(
    [
        new TextInputPart("<|user|>What is 1+1?<|end|><|assistant|>"),
    ]),
    maximumNewTokens: 32);
```

## Golden validation

External Golden/model tests use:

- `PHI4_TEST_MODEL_DIRECTORY`: converted package containing `golden/` and
  `golden-inputs/`.
- `PHI4_RUN_GOLDEN_MODEL=1`: text prefill and audio/Conformer comparisons.
- `PHI4_RUN_GOLDEN_VISION=1`: the slower 27-layer SigLIP and HD-projector
  comparison.

The committed tests do not include model weights or large Golden tensors.

## Performance benchmark

Run the text decoder benchmark in Release mode:

```powershell
dotnet run --project tools\SharpInference.VmBenchmark -c Release -- phi4 `
  --model "<model-directory>" --tokens 16 --samples 3
```

The JSON result reports model setup time, cold first-token latency, warm decode
latency and throughput, prefill latency and throughput, managed allocations,
GC collection counts, and the first-token argmax. Use `--output <file>` to
write the same result to a file.

Use `--backend d3d12` to keep all 32 layers of QKV, attention-output, FFN
up/down, and the 200064-row vocabulary projection resident on the selected
D3D12 adapter. `--adapter <index>` selects a non-default adapter:

```powershell
dotnet run --project tools\SharpInference.VmBenchmark -c Release -- phi4 `
  --model "<model-directory>" --backend d3d12 --tokens 16 --samples 3
```

Use `--backend d3d12-session` for text-only GPU-resident decode:

```powershell
dotnet run --project tools\SharpInference.VmBenchmark -c Release -- phi4 `
  --model "<model-directory>" --backend d3d12-session `
  --context 4096 --tokens 16 --samples 3
```

This path keeps hidden state, scratch tensors, and FP32 KV caches on GPU. One
`decode` VM entry performs embedding lookup, all 32 decoder layers, vocabulary
projection, and argmax. A token step uploads only the token ID and position and
reads back one token ID. The control copy, complete decode schedule, and result
copy are recorded in one command list with one queue submission and one fence
wait; no intermediate activation is uploaded or read back.

The public session API is:

```csharp
using var package = Phi4ModelPackage.Open(modelDirectory);
using var session = new Phi4D3D12TextSession(package, maximumContext: 4096);
var nextToken = session.ForwardToken(Phi4Tokenizer.UserTokenId);
```

The session also supports `Phi4Adapter.Vision` and `Phi4Adapter.Speech`.
Adapter projections remain GPU-resident and execute
`W*x + 2*B*(A*x)` without activation readback. `ForwardEmbedding` accepts a
single host embedding for compatibility. The pooled constructor binds a
`Phi4D3D12FusionComponent` output and `ForwardFusedEmbedding` consumes each
prompt row directly from shared GPU storage.

### D3D12 vision component

`Phi4D3D12VisionComponent` compiles fixed crop/HD-shape buckets for the
26-layer SigLIP encoder, 2x2 compression, HD token rearrangement, and the
two-layer projector. Pixel values and attention masks can be uploaded through
their `ComponentPortDescriptor` contracts or supplied as `Phi4ImageFeatures`.
The result is an `IStorageLease` in the component's D3D12 `StorageDomain`, with
shape `[image tokens, 3072]` and semantic port
`phi4.vision.projected_embeddings@1`.

All encoder and projector intermediates remain in pooled D3D12 storage. Crop,
layer, and projector-chunk entries are separate bounded GPU submissions to
avoid Windows TDR, but they do not upload or read back intermediate values.
`Readback` is an explicit final-output operation for validation or host
consumers. `BindOutput` and `CreateExecutor` allow a same-pool downstream VM to
consume the projected embeddings without host staging.

The component reuses T0 `core.add` and the standard VM/barrier machinery. Its
architecture-local instruction collection supplies only operations not
expressible by the current general collections: batched biased linear,
LayerNorm, non-causal masked attention, patch embedding, GELU, 2x2 pooling,
and HD gather.

### D3D12 audio component

`Phi4D3D12AudioComponent` provides the convolutional subsampler, all 24
Conformer layers, relative-position attention, causal depthwise convolution,
and both Phi-4 audio projectors. Frame buckets are 128, 256, 352, 512, 1024,
2048, and 4096. `EncodeResident` returns an `IStorageLease` for the physical
`[token bucket, 3072]` tensor. `IPhi4AudioEmbeddingHandle.ValidTokenCount`
identifies the valid prefix. `Encode` remains the compatibility API and
performs an explicit final readback.

The shared-pool constructor, `BindOutput`, and `CreateExecutor` use the same
zero-copy contract as the vision component. Image+audio mode selects the
vision audio projector; audio-only mode selects the speech projector.

### D3D12 fusion and multimodal orchestration

`Phi4D3D12FusionComponent` gathers ordinary token embeddings and replaces
image/audio placeholder runs from resident encoder outputs. Only token IDs and
the replacement map are host control data. Projected media data is never
downloaded. Its resident `[sequence, 3072]` result binds directly to the
decoder's `decode_fused` entry.

`Phi4D3D12MultimodalSession` owns one device and shared resource pool, starts
vision and audio branches as separate tasks, joins them in the fusion VM,
prefills a decoder with the selected LoRA, and continues token generation with
the resident token entry. Modality selection is:

- text: no adapter
- image: vision adapter
- audio: speech adapter and speech audio projector
- image+audio: vision adapter and vision audio projector

The current pool serializes commands that use pooled resources, so branch task
scheduling and CPU preprocessing overlap, but the vision and audio GPU queues
do not yet execute concurrently. Each component remains independently
bucketed and optimizable.

A heterogeneous image+audio schedule is possible: run the vision encoder on
GPU while running the audio encoder on CPU, then upload the much smaller
projected audio tensor into the D3D12 fusion domain. This can reduce latency
when CPU audio finishes no later than GPU vision and the saved serialized GPU
audio time exceeds the final embedding upload. Running vision on CPU is
generally unattractive because the CPU SigLIP path is substantially slower.
The current multimodal session does not automatically choose a heterogeneous
schedule; it keeps both encoders on D3D12 for predictable zero-copy behavior.

Fused prefill uses true 16-token batched projections, RMSNorm, RoPE/KV writes,
GQA attention, SwiGLU, residuals, and LoRA updates. A tail chunk carries a
`valid_count`, so it does not write padding tokens into the KV cache. Each
chunk is one GPU submission and only the final selected token is read back.
Single-token decode continues from the same resident KV cache. The 1,800+
token image+audio Golden test reduced end-to-end time from approximately 97
seconds to approximately 54 seconds on the validation machine.

## Backend scope

The Phi-4 numerical implementation executes on CPU by default. The optional
`SharpInference.Architectures.Phi4.D3D12` matrix projector uses the existing T0
cooperative `core.mat-vec` instruction for all base decoder projections. One VM
executor owns the resident weights and shape-specific shared I/O slots. RMSNorm,
RoPE, KV-cache updates, attention, activations, residuals, and LoRA updates
continue to execute on CPU. The projector requires enough dedicated or shared
GPU memory for approximately 7.7 GB of F16 projection weights.

`Phi4D3D12TextSession` adds GPU-resident FP32 KV caches, approximately 1 GB at
a 4096-token capacity, and supports base, vision-LoRA, and speech-LoRA modes.
`PrefillFusedEmbeddings` processes resident prompt embeddings in 16-token
chunks using batched shared-weight matrix projections and causal attention.

The D3D12 components are independently testable. `Phi4Processor` continues to
select the CPU implementation by default; applications opt into the GPU
multimodal path by constructing `Phi4D3D12MultimodalSession`.
