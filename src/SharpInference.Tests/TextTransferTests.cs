using System.Text.Json;
using SharpInference.WebApi;

namespace SharpInference.Tests;

public sealed class TextTransferTests
{
    private static readonly IModelTextTransfer WorldTransfer = new Rwkv6WorldTextTransfer();

    [Fact]
    public void Render_ConvertsChatCompletionJsonToWorldChatTemplate()
    {
        const string json =
            """
            {
              "model": "rwkv-6-world",
              "messages": [
                { "role": "system", "content": " Answer accurately. " },
                { "role": "user", "content": "First line\r\n\r\nSecond line" },
                { "role": "assistant", "content": "Earlier answer." },
                { "role": "user", "content": "Next question?" }
              ]
            }
            """;
        var request = JsonSerializer.Deserialize<ChatCompletionRequest>(json);

        var prompt = WorldTransfer.Transfer(request!.Messages);

        Assert.Equal(
            "System: Answer accurately.\n\n" +
            "User: First line\nSecond line\n\n" +
            "Assistant: Earlier answer.\n\n" +
            "User: Next question?\n\n" +
            "Assistant:",
            prompt);
        Assert.DoesNotContain("Assistant: ", prompt[^11..]);
    }

    [Fact]
    public void Render_OutputEncodesToTheExpectedWorldTokens()
    {
        var content = JsonSerializer.SerializeToElement("Hello");
        ChatMessage[] messages = [new("user", content)];
        var tokenizer = RwkvWorldTokenizer.LoadBundled();
        const string expectedPrompt = "User: Hello\n\nAssistant:";

        var promptTokens = tokenizer.Encode(WorldTransfer.Transfer(messages));

        Assert.Equal(tokenizer.Encode(expectedPrompt), promptTokens);
        Assert.Equal(expectedPrompt, tokenizer.Decode(promptTokens));
    }

    [Fact]
    public void Render_RejectsUnsupportedJsonContent()
    {
        var content = JsonSerializer.SerializeToElement(new { text = "Hello" });
        ChatMessage[] messages = [new("user", content)];

        var exception = Assert.Throws<ArgumentException>(() => WorldTransfer.Transfer(messages));

        Assert.Contains("string content", exception.Message);
    }

    [Fact]
    public void Render_AcceptsTextContentPartsAndDeveloperRole()
    {
        const string json =
            """
            {
              "messages": [
                {
                  "role": "developer",
                  "content": [{ "type": "text", "text": "Answer briefly." }]
                },
                {
                  "role": "user",
                  "content": [
                    { "type": "input_text", "text": "Hello" },
                    { "type": "text", "text": " world" }
                  ]
                }
              ]
            }
            """;
        var request = JsonSerializer.Deserialize<ChatCompletionRequest>(json);

        var prompt = WorldTransfer.Transfer(request!.Messages);

        Assert.Equal(
            "System: Answer briefly.\n\nUser: Hello world\n\nAssistant:",
            prompt);
    }

    [Fact]
    public void Render_RejectsNonTextContentParts()
    {
        var content = JsonSerializer.SerializeToElement(new[]
        {
            new { type = "image_url", image_url = "https://example.invalid/image.png" },
        });
        ChatMessage[] messages = [new("user", content)];

        var exception = Assert.Throws<ArgumentException>(() => WorldTransfer.Transfer(messages));

        Assert.Contains("only text content parts", exception.Message);
    }

    [Fact]
    public void WorldTemplate_StopsBeforeAnotherConversationRole()
    {
        Assert.Equal(
            ["\n\nUser:", "\n\nSystem:", "\n\nAssistant:"],
            Rwkv6WorldChatTemplateTransfer.StopStrings);
    }

    [Fact]
    public void Resolver_SelectsTemplateForModel()
    {
        var resolver = new ModelTextTransferResolver([WorldTransfer]);
        var metadata = new RwkvModelMetadata(65536, 2048, 24, 32, 64, "rwkv-6");

        var transfer = resolver.Resolve(metadata);

        Assert.Same(WorldTransfer, transfer);
    }

    [Fact]
    public void Rwkv7G1_UsesChatRoundSeparatorsWithoutTrailingSpace()
    {
        var transfer = new Rwkv7G1TextTransfer();
        var resolver = new ModelTextTransferResolver([WorldTransfer, transfer]);
        var metadata = new RwkvModelMetadata(65536, 4096, 32, 64, 64, "rwkv-7");
        var messages = new[]
        {
            new ChatMessage("system", JsonSerializer.SerializeToElement("Be concise.")),
            new ChatMessage("user", JsonSerializer.SerializeToElement("First\r\n\r\nline")),
            new ChatMessage("assistant", JsonSerializer.SerializeToElement("Earlier answer.")),
            new ChatMessage("user", JsonSerializer.SerializeToElement("What is 2+2?")),
        };

        var prompt = resolver.Resolve(metadata).Transfer(messages);

        Assert.Equal(
            "System: Be concise.\n\nUser: First\nline\n\nAssistant: Earlier answer.\n\n" +
            "User: What is 2+2?\n\nAssistant:",
            prompt);
        Assert.False(prompt.EndsWith(' '));
        Assert.Equal(Rwkv6WorldChatTemplateTransfer.StopStrings, Rwkv7G1TextTransfer.StopStrings);
    }

    [Fact]
    public void Chain_AppliesAdditionalTransfersInOrder()
    {
        var chain = new TextTransferChain<string>(
            new AppendTextTransfer(" first"),
            [new AppendTextTransfer(" second"), new AppendTextTransfer(" third")]);

        var result = chain.Transfer("start");

        Assert.Equal("start first second third", result);
    }

    private sealed class AppendTextTransfer(string suffix) : ITextTransfer
    {
        public string Transfer(string input) => input + suffix;
    }
}
