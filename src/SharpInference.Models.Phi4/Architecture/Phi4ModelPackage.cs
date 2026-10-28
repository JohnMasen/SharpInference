using SharpInference.Gguf;

namespace SharpInference.Architectures.Phi4;

/// <summary>Selects the LoRA adapter applied to the Phi-4 text backbone.</summary>
public enum Phi4Adapter
{
    /// <summary>Uses the text backbone without a LoRA adapter.</summary>
    None,
    /// <summary>Applies the vision LoRA adapter.</summary>
    Vision,
    /// <summary>Applies the speech LoRA adapter.</summary>
    Speech,
}

/// <summary>Owns the text, multimodal, and adapter GGUF files and tokenizer of a Phi-4 model package.</summary>
public sealed class Phi4ModelPackage : IDisposable
{
    private bool disposed;

    private Phi4ModelPackage(
        string directory,
        GgufModelFile text,
        GgufModelFile omni,
        GgufModelFile visionAdapter,
        GgufModelFile speechAdapter,
        Phi4Tokenizer tokenizer)
    {
        Directory = directory;
        Text = text;
        Omni = omni;
        VisionAdapter = visionAdapter;
        SpeechAdapter = speechAdapter;
        Tokenizer = tokenizer;
    }

    /// <summary>Gets the absolute directory containing the model package files.</summary>
    public string Directory
    {
        get;
    }
    /// <summary>Gets the text backbone GGUF file.</summary>
    public GgufModelFile Text
    {
        get;
    }
    /// <summary>Gets the GGUF file containing the vision and audio encoder weights.</summary>
    public GgufModelFile Omni
    {
        get;
    }
    /// <summary>Gets the vision LoRA adapter GGUF file.</summary>
    public GgufModelFile VisionAdapter
    {
        get;
    }
    /// <summary>Gets the speech LoRA adapter GGUF file.</summary>
    public GgufModelFile SpeechAdapter
    {
        get;
    }
    /// <summary>Gets the tokenizer associated with the model package.</summary>
    public Phi4Tokenizer Tokenizer
    {
        get;
    }

    /// <summary>Opens and validates the GGUF components and tokenizer in the specified Phi-4 package directory.</summary>
    public static Phi4ModelPackage Open(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        directory = Path.GetFullPath(directory);
        var paths = new[]
        {
            Path.Combine(directory, "Phi-4-multimodal-text-f16.gguf"),
            Path.Combine(directory, "Phi-4-multimodal-omni-f16.gguf"),
            Path.Combine(directory, "Phi-4-multimodal-vision-lora-f16.gguf"),
            Path.Combine(directory, "Phi-4-multimodal-speech-lora-f16.gguf"),
            Path.Combine(directory, "tokenizer.json"),
        };
        foreach (var path in paths)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("Phi-4 model package component is missing.", path);
            }
        }

        GgufModelFile? text = null;
        GgufModelFile? omni = null;
        GgufModelFile? vision = null;
        GgufModelFile? speech = null;
        try
        {
            text = GgufModelFile.Open(paths[0]);
            omni = GgufModelFile.Open(paths[1]);
            vision = GgufModelFile.Open(paths[2]);
            speech = GgufModelFile.Open(paths[3]);
            Validate(text, omni, vision, speech);
            return new Phi4ModelPackage(
                directory, text, omni, vision, speech, Phi4Tokenizer.Load(paths[4]));
        }
        catch
        {
            speech?.Dispose();
            vision?.Dispose();
            omni?.Dispose();
            text?.Dispose();
            throw;
        }
    }

    /// <summary>Creates a text model over the package weights with optional matrix projection acceleration.</summary>
    public Phi4TextModel CreateTextModel(IPhi4MatrixProjector? matrixProjector = null)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return new Phi4TextModel(Text, VisionAdapter, SpeechAdapter, matrixProjector);
    }

    /// <summary>Creates a host-side embedding provider over the package text and multimodal weights.</summary>
    public Phi4NativeEmbeddingProvider CreateEmbeddingProvider()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return new Phi4NativeEmbeddingProvider(Text, Omni);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        SpeechAdapter.Dispose();
        VisionAdapter.Dispose();
        Omni.Dispose();
        Text.Dispose();
        disposed = true;
    }

    private static void Validate(
        GgufModelFile text,
        GgufModelFile omni,
        GgufModelFile vision,
        GgufModelFile speech)
    {
        if (text.GetMetadata<string>("general.architecture") != "phi3" ||
            text.GetMetadata<uint>("phi3.block_count") != 32 ||
            text.GetMetadata<uint>("phi3.embedding_length") != 3072 ||
            text.GetMetadata<uint>("phi3.attention.head_count") != 24 ||
            text.GetMetadata<uint>("phi3.attention.head_count_kv") != 8 ||
            text.Tensors.Count != 196)
        {
            throw new InvalidDataException("The text GGUF is not the expected Phi-4 multimodal backbone.");
        }

        if (omni.GetMetadata<string>("general.architecture") != "clip" ||
            omni.Tensors.Count != 1330)
        {
            throw new InvalidDataException("The omni GGUF is not the expected Phi-4 vision/audio component.");
        }

        ValidateAdapter(vision, "Vision Lora");
        ValidateAdapter(speech, "Speech Lora");

        foreach (var name in new[]
        {
            "token_embd.weight",
            "blk.0.attn_qkv.weight",
            "blk.31.ffn_down.weight",
            "output_norm.weight",
        })
        {
            _ = text.GetTensor(name);
        }
    }

    private static void ValidateAdapter(GgufModelFile file, string expectedName)
    {
        if (file.GetMetadata<string>("general.architecture") != "phi3" ||
            file.GetMetadata<string>("adapter.type") != "lora" ||
            file.GetMetadata<string>("general.name") != expectedName ||
            file.Tensors.Count != 256)
        {
            throw new InvalidDataException($"The adapter GGUF is not '{expectedName}'.");
        }
    }
}
