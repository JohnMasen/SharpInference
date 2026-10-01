using SharpInference.WebApi;

namespace SharpInference.Tests;

public sealed class CompletionTokenBudgetTests
{
    [Theory]
    [InlineData(12800, 1000, 4096, 4096)]
    [InlineData(12800, 10000, 4096, 2800)]
    [InlineData(12800, 12000, 4096, 800)]
    [InlineData(12800, 12799, 256, 1)]
    public void Calculate_UsesRequestedAndRemainingMinimum(
        int contextWindow,
        int promptTokens,
        int requestedMaxTokens,
        int expected)
    {
        var actual = CompletionTokenBudget.Calculate(contextWindow, promptTokens, requestedMaxTokens);

        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData(12800)]
    [InlineData(12801)]
    public void Calculate_RejectsPromptAtOrBeyondContextWindow(int promptTokens)
    {
        var exception = Assert.Throws<ContextWindowExceededException>(
            () => CompletionTokenBudget.Calculate(12800, promptTokens, 1));

        Assert.Equal(12800, exception.ContextWindowTokens);
        Assert.Equal(promptTokens, exception.PromptTokens);
    }

    [Fact]
    public void Calculate_RejectsNonPositiveRequestedMaximum()
    {
        var exception = Assert.Throws<ArgumentException>(
            () => CompletionTokenBudget.Calculate(12800, 100, 0));

        Assert.Contains("max_tokens must be positive", exception.Message);
    }

    [Fact]
    public void CommandLine_MapsContextWindowOverride()
    {
        var commandLine = WebApiCommandLine.Parse(
            ["--model-path", "model.bin", "--context-window-tokens", "16384"]);

        Assert.Equal("16384", commandLine.Overrides["Rwkv:ContextWindowTokens"]);
    }
}
