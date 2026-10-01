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
                    if (!bool.TryParse(value, out var enableCommandReplay))
                        throw new ArgumentException("The value for '--enable-command-replay' must be true or false.");
                    overrides["Rwkv:Runtime:Vortice:EnableCommandReplay"] = enableCommandReplay.ToString();
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
                    overrides["Rwkv:GpuBatchService:MaxInFlightGenerationBatches"] = value;
                    break;
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
          --runtime-kind <cpu|vortice>
              Select the model-independent graph execution backend.
          --enable-command-replay <true|false>
              Opt in to per-session D3D12 command replay for supported GPU graphs.
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
          --max-in-flight-generation-batches <count>
              Global maximum simultaneous generations. 0 uses CPU cores/2 (minimum 1) or GPU 4. Default: 0.
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
