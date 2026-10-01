using SharpInference.WebApi;

namespace SharpInference.Tests;

public sealed class PromptStateManagerTests
{
    [Fact]
    public void FindLongestPrefix_ReturnsLongestSavedInputAndIncrementsScore()
    {
        var manager = new PromptStateManager(new PromptStateManagerOptions { Capacity = 3 });
        manager.Store("System: test\n\nUser: first\n\nAssistant: answer", [1], PromptStateKind.Complete);
        manager.Store("System: test\n\nUser: first\n\nAssistant: answer\n\nUser: second\n\nAssistant: reply", [2, 3], PromptStateKind.Complete);

        var match = manager.FindLongestPrefix(
            "System: test\n\nUser: first\n\nAssistant: answer\n\nUser: second\n\nAssistant: reply\n\nUser: next\n\nAssistant:");

        Assert.NotNull(match);
        Assert.Equal("System: test\n\nUser: first\n\nAssistant: answer\n\nUser: second\n\nAssistant: reply", match.Input);
        Assert.Equal(match.Input.Length, match.StartCharacterIndex);
        Assert.Equal([2, 3], match.State.ToArray());
        Assert.Equal(1, match.Score);
        Assert.Equal(PromptStateKind.Complete, match.Kind);
    }

    [Fact]
    public void FindLongestPrefix_ReturnsNullWhenNoSavedInputIsAPrefix()
    {
        var manager = new PromptStateManager(new PromptStateManagerOptions());
        manager.Store("User: unrelated", [1], PromptStateKind.Complete);

        Assert.Null(manager.FindLongestPrefix("System: test\n\nUser: question"));
    }

    [Fact]
    public void FindLongestPrefix_IgnoresExactMatchWithoutNextTokenLogits()
    {
        var manager = new PromptStateManager(new PromptStateManagerOptions());
        manager.Store("User: question", [4], PromptStateKind.Prefill);

        Assert.Null(manager.FindLongestPrefix("User: question"));
    }

    [Fact]
    public void FindLongestPrefix_UsesShorterUsablePrefixInsteadOfExactMatch()
    {
        var manager = new PromptStateManager(new PromptStateManagerOptions());
        manager.Store("User: hello\n\nAssistant: hi", [1], PromptStateKind.Complete);
        manager.Store("User: hello\n\nAssistant: hi\n\nUser: next\n\nAssistant:", [2], PromptStateKind.Prefill);

        var match = manager.FindLongestPrefix("User: hello\n\nAssistant: hi\n\nUser: next\n\nAssistant:");
        Assert.NotNull(match);
        Assert.Equal(PromptStateKind.Complete, match.Kind);
        Assert.Equal([1], match.State.ToArray());
    }

    [Fact]
    public void Store_EvictsLowestScoreThenOldestEntry()
    {
        var manager = new PromptStateManager(new PromptStateManagerOptions { Capacity = 2 });
        manager.Store("A", [1], PromptStateKind.Complete);
        manager.Store("B", [2], PromptStateKind.Complete);
        Assert.NotNull(manager.FindLongestPrefix("A\n\nnext"));

        manager.Store("C", [3], PromptStateKind.Complete);

        Assert.NotNull(manager.FindLongestPrefix("A\n\nnext"));
        Assert.Null(manager.FindLongestPrefix("B\n\nnext"));
        Assert.NotNull(manager.FindLongestPrefix("C\n\nnext"));
    }

    [Fact]
    public void Store_UpdatesStateWithoutResettingScore()
    {
        var manager = new PromptStateManager(new PromptStateManagerOptions { Capacity = 1 });
        manager.Store("A", [1], PromptStateKind.Complete);
        Assert.Equal(1, manager.FindLongestPrefix("A\n\nnext")!.Score);

        manager.Store("A", [9], PromptStateKind.Complete);
        var match = manager.FindLongestPrefix("A\n\nnext");

        Assert.Equal([9], match!.State.ToArray());
        Assert.Equal(2, match.Score);
        Assert.Equal(1, manager.Count);
    }

    [Fact]
    public void ZeroCapacity_DisablesStorage()
    {
        var manager = new PromptStateManager(new PromptStateManagerOptions { Capacity = 0 });

        manager.Store("A", [1], PromptStateKind.Complete);

        Assert.Equal(0, manager.Count);
        Assert.Null(manager.FindLongestPrefix("A\n\nnext"));
    }

    [Fact]
    public void DisabledCache_DoesNotRetainStates()
    {
        var manager = new PromptStateManager(new PromptStateManagerOptions
        {
            Enabled = false,
            Capacity = 10,
        });

        manager.Store("User: hello\n\nAssistant: hi", [1], PromptStateKind.Complete);

        Assert.Equal(0, manager.Capacity);
        Assert.Equal(0, manager.Count);
        Assert.Null(manager.FindLongestPrefix("User: hello\n\nAssistant: hi\n\nUser: next"));
    }

    [Fact]
    public void FindLongestPrefix_MatchesOnlyCompletedAssistantMessage()
    {
        var manager = new PromptStateManager(new PromptStateManagerOptions());
        manager.Store("User: hello\n\nAssistant: hi", [1, 2], PromptStateKind.Complete);

        Assert.Null(manager.FindLongestPrefix("User: hello\n\nAssistant: high\n\nUser: next\n\nAssistant:"));
        var match = manager.FindLongestPrefix("User: hello\n\nAssistant: hi\n\nUser: next\n\nAssistant:");
        Assert.NotNull(match);
        Assert.Equal([1, 2], match.State.ToArray());
        Assert.Equal("User: hello\n\nAssistant: hi".Length, match.StartCharacterIndex);
    }

    [Fact]
    public void FindLongestPrefix_DoesNotMatchIncompleteAnswer()
    {
        var manager = new PromptStateManager(new PromptStateManagerOptions());
        manager.Store("User: hello\n\nAssistant: hi", [3], PromptStateKind.Complete);

        Assert.Null(manager.FindLongestPrefix("User: hello\n\nAssistant: hi there\n\nUser: next\n\nAssistant:"));
    }

    [Fact]
    public void FindLongestPrefix_SelectsCompleteThenFallsBackToPrefillWhenAnswerChanges()
    {
        var manager = new PromptStateManager(new PromptStateManagerOptions());
        const string prompt = "User: hello\n\nAssistant:";
        manager.Store(prompt, [1], PromptStateKind.Prefill);
        manager.Store(prompt + " hi", [2], PromptStateKind.Complete);

        var complete = manager.FindLongestPrefix(prompt + " hi\n\nUser: next\n\nAssistant:");
        Assert.NotNull(complete);
        Assert.Equal(PromptStateKind.Complete, complete.Kind);
        Assert.Equal([2], complete.State.ToArray());

        var prefill = manager.FindLongestPrefix(prompt + " hello\n\nUser: next\n\nAssistant:");
        Assert.NotNull(prefill);
        Assert.Equal(PromptStateKind.Prefill, prefill.Kind);
        Assert.Equal([1], prefill.State.ToArray());
        Assert.Equal(2, manager.Count);
    }

    [Fact]
    public void Store_KeepsDifferentKindsWithIdenticalKeysSeparate()
    {
        var manager = new PromptStateManager(new PromptStateManagerOptions());
        manager.Store("A", [1], PromptStateKind.Prefill);
        manager.Store("A", [2], PromptStateKind.Complete);

        Assert.Equal(2, manager.Count);
        Assert.Equal(PromptStateKind.Complete, manager.FindLongestPrefix("A\n\nnext")!.Kind);
        Assert.Equal(PromptStateKind.Prefill, manager.FindLongestPrefix("A next")!.Kind);
    }

    [Fact]
    public void Store_RejectsInvalidKind()
    {
        var manager = new PromptStateManager(new PromptStateManagerOptions());

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            manager.Store("A", [1], (PromptStateKind)42));
    }

    [Theory]
    [InlineData(" hi", "User: hello\n\nAssistant: hi", false)]
    [InlineData("hi", "User: hello\n\nAssistant: hi", true)]
    [InlineData(" hi ", "User: hello\n\nAssistant: hi", true)]
    [InlineData(" hi\n\nthere", "User: hello\n\nAssistant: hi\nthere", true)]
    [InlineData(" hi\r\nthere", "User: hello\n\nAssistant: hi\nthere", true)]
    [InlineData(" hi\n\nUser:", "User: hello\n\nAssistant: hi\nUser:", true)]
    public void AnswerStateKey_NormalizesAnswerToNextPrompt(string answer, string expectedKey, bool expectedReplay)
    {
        var accepted = AnswerStateCacheKey.TryCreate(
            "User: hello\n\nAssistant:", answer, out var key, out var requiresReplay);

        Assert.True(accepted);
        Assert.Equal(expectedKey, key);
        Assert.Equal(expectedReplay, requiresReplay);
    }

    [Theory]
    [InlineData("")]
    [InlineData(" \n ")]
    public void AnswerStateKey_SkipsEmptyNormalizedAnswers(string answer)
    {
        Assert.False(AnswerStateCacheKey.TryCreate(
            "User: hello\n\nAssistant:", answer, out var key, out var requiresReplay));
        Assert.Equal(string.Empty, key);
        Assert.False(requiresReplay);
    }

    [Fact]
    public void AnswerStateKey_CompleteMatchesTransferredNextTurn()
    {
        var transfer = new Rwkv6WorldTextTransfer();
        var prompt = transfer.Transfer([new ChatMessage("user", System.Text.Json.JsonSerializer.SerializeToElement("hello"))]);
        const string answer = "hi\n";
        Assert.True(AnswerStateCacheKey.TryCreate(prompt, answer, out var key, out var requiresReplay));
        Assert.True(requiresReplay);

        var nextPrompt = transfer.Transfer([
            new ChatMessage("user", System.Text.Json.JsonSerializer.SerializeToElement("hello")),
            new ChatMessage("assistant", System.Text.Json.JsonSerializer.SerializeToElement(answer)),
            new ChatMessage("user", System.Text.Json.JsonSerializer.SerializeToElement("next")),
        ]);
        var manager = new PromptStateManager(new PromptStateManagerOptions());
        manager.Store(prompt, [1], PromptStateKind.Prefill);
        manager.Store(key, [2], PromptStateKind.Complete);

        var match = manager.FindLongestPrefix(nextPrompt);
        Assert.NotNull(match);
        Assert.Equal(PromptStateKind.Complete, match.Kind);
        Assert.Equal([2], match.State.ToArray());
    }

    [Fact]
    public void CommandLine_MapsStateCacheCapacityOverride()
    {
        var commandLine = WebApiCommandLine.Parse(
            ["--model-path", "model.bin", "--state-cache-capacity", "25"]);

        Assert.Equal("25", commandLine.Overrides["Rwkv:StateManager:Capacity"]);
    }

    [Fact]
    public void CommandLine_MapsStateCacheEnabledOverride()
    {
        var enabled = WebApiCommandLine.Parse(
            ["--model-path", "model.bin", "--state-cache-enabled", "true"]);
        var disabled = WebApiCommandLine.Parse(
            ["--model-path", "model.bin", "--state-cache-enabled", "false"]);

        Assert.Equal("True", enabled.Overrides["Rwkv:StateManager:Enabled"]);
        Assert.Equal("False", disabled.Overrides["Rwkv:StateManager:Enabled"]);
        Assert.Throws<ArgumentException>(() =>
            WebApiCommandLine.Parse(["--model-path", "model.bin", "--state-cache-enabled", "invalid"]));
    }

    [Fact]
    public void Options_RejectNegativeCapacity()
    {
        var options = new PromptStateManagerOptions { Capacity = -1 };

        var exception = Assert.Throws<InvalidOperationException>(options.Validate);

        Assert.Contains("cannot be negative", exception.Message);
    }

    [Fact]
    public void CommandLine_RequiresModelPath()
    {
        var exception = Assert.Throws<ArgumentException>(
            () => WebApiCommandLine.Parse(["--port", "9841"]));

        Assert.Contains("--model-path", exception.Message);
    }

    [Fact]
    public void CommandLine_AcceptsModelPathByItself()
    {
        var commandLine = WebApiCommandLine.Parse(["--model-path", "model.bin"]);

        Assert.Equal("model.bin", commandLine.Overrides["Rwkv:ModelPath"]);
        Assert.Null(commandLine.ConfigurationPath);
    }
}
