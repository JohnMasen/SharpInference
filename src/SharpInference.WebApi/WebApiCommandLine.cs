namespace SharpInference.WebApi;

public sealed record WebApiCommandLine(string? ConfigurationPath, IReadOnlyDictionary<string, string?> Overrides)
{
    public static WebApiCommandLine Parse(string[] args)
    {
        ArgumentNullException.ThrowIfNull(args);
        string? configurationPath = null;
        var overrides = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);

        for (var index = 0; index < args.Length; index++)
        {
            var option = args[index];
            if (option is "--help" or "-h")
            {
                throw new ArgumentException(Usage);
            }

            var value = ReadValue(args, ref index, option);
            switch (option)
            {
                case "--config":
                    configurationPath = Path.GetFullPath(value);
                    break;
                case "--model-path":
                    overrides["Rwkv:ModelPath"] = value;
                    break;
                case "--runtime-kind":
                    overrides["Rwkv:Runtime:Kind"] = value;
                    break;
                case "--enable-command-replay":
                    throw new ArgumentException("The compiled VM no longer supports '--enable-command-replay'. Command caching is backend-managed.");
                case "--prefill-instances":
                    overrides["Rwkv:Runtime:Vm:PrefillInstances"] = value;
                    break;
                case "--inference-instances":
                    overrides["Rwkv:Runtime:Vm:InferenceInstances"] = value;
                    break;
                case "--prefill-queue-capacity":
                    overrides["Rwkv:Runtime:Vm:PrefillQueueCapacity"] = value;
                    break;
                case "--inference-queue-capacity":
                    overrides["Rwkv:Runtime:Vm:InferenceQueueCapacity"] = value;
                    break;
                case "--default-max-tokens":
                    overrides["Rwkv:DefaultMaxTokens"] = value;
                    break;
                case "--context-window-tokens":
                    overrides["Rwkv:ContextWindowTokens"] = value;
                    break;
                case "--api-key":
                    overrides["Rwkv:ApiKey"] = value;
                    break;
                case "--max-resident-gpu-sessions":
                    overrides["Rwkv:GpuBatchService:MaxResidentGpuSessions"] = value;
                    break;
                case "--max-in-flight-generation-batches":
                    throw new ArgumentException("Use '--inference-instances' to configure compiled VM generation concurrency.");
                case "--state-cache-capacity":
                    overrides["Rwkv:StateManager:Capacity"] = value;
                    break;
                case "--state-cache-enabled":
                    if (!bool.TryParse(value, out var stateCacheEnabled))
                    {
                        throw new ArgumentException("The value for '--state-cache-enabled' must be true or false.");
                    }

                    overrides["Rwkv:StateManager:Enabled"] = stateCacheEnabled.ToString();
                    break;
                case "--urls":
                    overrides["Server:Urls"] = value;
                    break;
                case "--port":
                    if (!ushort.TryParse(value, out var port) || port == 0)
                    {
                        throw new ArgumentException($"The value '{value}' is not a valid TCP port.");
                    }

                    overrides["Server:Urls"] = $"http://0.0.0.0:{port}";
                    break;
                default:
                    throw new ArgumentException($"Unknown option '{option}'.{Environment.NewLine}{Usage}");
            }
        }

        if (!overrides.ContainsKey("Rwkv:ModelPath"))
        {
            throw new ArgumentException($"The required option '--model-path' was not provided.{Environment.NewLine}{Usage}");
        }

        if (configurationPath is not null && !File.Exists(configurationPath))
        {
            throw new FileNotFoundException("The specified JSON configuration file does not exist.", configurationPath);
        }

        return new WebApiCommandLine(configurationPath, overrides);
    }

    public const string Usage = """
        Usage:
          SharpInference.WebApi --model-path <ggml-path> [options]

        Required:
          --model-path <ggml-path>
              Path to the RWKV GGML model file.

        Optional:
          --config <json-path>
              Load additional JSON configuration. Command-line values override it.
          --runtime-kind <cpu|vortice|d3d12>
              Select the compiled VM backend. vortice is an alias for d3d12.
          --prefill-instances <count>
              Number of reusable prefill VMs. Default: 2.
          --inference-instances <count>
              Number of whole-generation inference VMs. Default: 2.
          --prefill-queue-capacity <count>
              Maximum queued prefill requests. Default: 16.
          --inference-queue-capacity <count>
              Maximum queued generation requests. Default: 16.
          --port <1-65535>
              Listen on http://0.0.0.0:<port>. Default: 9841.
          --urls <http-url>
              Override the complete server listen URL.
          --default-max-tokens <count>
              Default completion limit when max_tokens is omitted. Default: 4096.
          --context-window-tokens <count>
              Logical prompt plus completion context window. Default: 12800.
          --api-key <key>
              Require this API key. Empty by default.
          --max-resident-gpu-sessions <count>
              Maximum number of resident GPU sessions. Default: 64.
          --state-cache-capacity <count>
              Number of scored prompt-state entries to retain. Default: 10; 0 disables it.
          --state-cache-enabled <true|false>
              Enable or disable state cache lookup and snapshots. Default: true.
          --help, -h
              Print this help and exit.
        """;

    private static string ReadValue(string[] args, ref int index, string option)
    {
        if (index + 1 >= args.Length || args[index + 1].StartsWith("-", StringComparison.Ordinal))
        {
            throw new ArgumentException($"Option '{option}' requires a value.{Environment.NewLine}{Usage}");
        }

        return args[++index];
    }
}
