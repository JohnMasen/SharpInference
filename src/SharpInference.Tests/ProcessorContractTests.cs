using SharpInference.Runtime;

namespace SharpInference.Tests;

public sealed class ProcessorContractTests
{
    [Fact]
    public void OrderedInput_PreservesPartsAndReportsCombinedModalities()
    {
        var text = new TextInputPart("describe ");
        var image = new ImageInputPart(new byte[2 * 2 * 3], 2, 2, 6, ProcessorPixelFormat.Rgb24);
        var audio = new AudioInputPart(new byte[16], ProcessorAudioSampleFormat.Float32, 16_000, 1);

        var input = new ProcessorInput([text, image, audio, new TextInputPart(" and summarize")]);

        Assert.Equal([text, image, audio, input.Parts[3]], input.Parts);
        Assert.Equal(
            ProcessorInputModality.Text | ProcessorInputModality.Image | ProcessorInputModality.Audio,
            input.Modalities);
        Assert.Equal(4, audio.FrameCount);
    }

    [Fact]
    public void Capabilities_RequireDescriptionsForEverySupportedModality()
    {
        Assert.Throws<ArgumentException>(() => new ProcessorCapabilities(
            [ProcessorInputModality.Text | ProcessorInputModality.Image],
            ProcessorOutputModality.Text,
            text: new ProcessorTextCapabilities(1024)));

        var capabilities = new ProcessorCapabilities(
            [ProcessorInputModality.Text, ProcessorInputModality.Text | ProcessorInputModality.Image],
            ProcessorOutputModality.Text,
            text: new ProcessorTextCapabilities(1024),
            image: new ProcessorImageCapabilities(
                new HashSet<ProcessorPixelFormat> { ProcessorPixelFormat.Rgb24 },
                2048, 2048, 4, 16));

        Assert.True(capabilities.Supports(ProcessorInputModality.Text | ProcessorInputModality.Image));
        Assert.False(capabilities.Supports(ProcessorInputModality.Image));
        Assert.True(typeof(IProcessor).IsAssignableFrom(typeof(Processor)));
        Assert.True(typeof(IProcessorSession).IsAssignableFrom(typeof(ProcessorSession)));
    }

    [Fact]
    public void MediaParts_RejectIncompleteBuffers()
    {
        Assert.Throws<ArgumentException>(() =>
            new ImageInputPart(new byte[11], 2, 2, 6, ProcessorPixelFormat.Rgb24));
        Assert.Throws<ArgumentException>(() =>
            new AudioInputPart(new byte[3], ProcessorAudioSampleFormat.Signed16, 16_000, 1));
    }
}
