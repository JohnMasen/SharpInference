using System.Diagnostics;
using System.Runtime.InteropServices;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Runtime;
using Xunit.Abstractions;

namespace SharpInference.Tests;

public sealed class CpuLargePortableResidencyTests(ITestOutputHelper output)
{
    [ExplicitLargeCpuProbeFact]
    [Trait("Category", "ExternalModel")]
    [Trait("ModelSize", "Large")]
    public Task FreshPortableOnlyG1Fp16ReportsOneTokenMemoryAndTime() => Probe([34]);

    [ExplicitLargeCpuProbeFact]
    [Trait("Category", "ExternalModel")]
    [Trait("ModelSize", "Large")]
    public Task FreshPortableOnlyG1Fp16ReportsTwoTokenWarmThroughput() => Probe([34, 105]);

    private async Task Probe(int[] tokens)
    {
        if (Environment.GetEnvironmentVariable("RWKV_CPU_7B_PROBE") != "1")
            throw new InvalidOperationException(
                "Set RWKV_CPU_7B_PROBE=1 only in an isolated, coordinated 7B test run.");
        var path = TestModelLoader.GetPath(TestModel.Rwkv7Large);
        var before = MemorySample.Capture();
        if (before.AvailablePhysicalBytes <= 10L * 1_073_741_824)
            throw new InvalidOperationException("The portable-only 7B probe requires over 10 GiB free RAM.");

        var peakGate = new object();
        var peakPrivateBytes = before.PrivateBytes;
        var peakWorkingSetBytes = before.WorkingSetBytes;
        var minimumAvailableBytes = before.AvailablePhysicalBytes;
        using var stopSampling = new CancellationTokenSource();
        var sampler = Task.Run(async () =>
        {
            try
            {
                while (true)
                {
                    var sample = MemorySample.Capture();
                    lock (peakGate)
                    {
                        peakPrivateBytes = Math.Max(peakPrivateBytes, sample.PrivateBytes);
                        peakWorkingSetBytes = Math.Max(peakWorkingSetBytes, sample.WorkingSetBytes);
                        minimumAvailableBytes = Math.Min(minimumAvailableBytes,
                            sample.AvailablePhysicalBytes);
                    }

                    if (sample.AvailablePhysicalBytes < 2L * 1_073_741_824)
                    {
                        Console.Error.WriteLine("CPU portable-only probe stopped: under 2 GiB free RAM.");
                        Process.GetCurrentProcess().Kill();
                    }
                    await Task.Delay(100, stopSampling.Token);
                }
            }
            catch (OperationCanceledException) when (stopSampling.IsCancellationRequested)
            {
            }
        });

        try
        {
            var constructTimer = Stopwatch.StartNew();
            using var processor = Processor.LoadGraph(path,
                new PortableRwkv7GraphProvider(), SharpInference.Runtime.Cpu.CpuVmBackendFactory.Create());
            constructTimer.Stop();
            var constructed = MemorySample.Capture();
            using var session = processor.CreateSession();
            output.WriteLine($"Before portable-only load: {before}");
            output.WriteLine($"Post-construct/reader-release: {constructed}");
            for (var index = 0; index < tokens.Length; index++)
            {
                var timer = Stopwatch.StartNew();
                float[] logits;
                logits = session.ForwardToken(tokens[index]).ToArray();
                timer.Stop();
                Assert.All(logits, value => Assert.True(float.IsFinite(value)));
                var afterToken = MemorySample.Capture();
                lock (peakGate)
                {
                    peakPrivateBytes = Math.Max(peakPrivateBytes, afterToken.PrivateBytes);
                    peakWorkingSetBytes = Math.Max(peakWorkingSetBytes, afterToken.WorkingSetBytes);
                    minimumAvailableBytes = Math.Min(minimumAvailableBytes,
                        afterToken.AvailablePhysicalBytes);
                }
                output.WriteLine($"After token {index + 1} ({tokens[index]}): {afterToken}; " +
                    $"elapsed {timer.Elapsed.TotalSeconds:F2} s" +
                    (index == 0 ? " (cold)" : $", {1 / timer.Elapsed.TotalSeconds:F3} warm tok/s"));
            }
            lock (peakGate)
                output.WriteLine($"Observed peaks: private {peakPrivateBytes / 1_073_741_824.0:F2} GiB, " +
                    $"working set {peakWorkingSetBytes / 1_073_741_824.0:F2} GiB; " +
                    $"minimum free physical {minimumAvailableBytes / 1_073_741_824.0:F2} GiB; " +
                    $"construct {constructTimer.Elapsed.TotalSeconds:F2} s");
        }
        finally
        {
            stopSampling.Cancel();
            await sampler;
        }
    }

    private readonly record struct MemorySample(long ManagedBytes, long WorkingSetBytes,
        long PrivateBytes, long AvailablePhysicalBytes)
    {
        public static MemorySample Capture()
        {
            using var process = Process.GetCurrentProcess();
            process.Refresh();
            var status = new MemoryStatus { Length = (uint)Marshal.SizeOf<MemoryStatus>() };
            if (!GlobalMemoryStatusEx(ref status))
                throw new InvalidOperationException("Could not query available physical memory.");
            return new MemorySample(GC.GetTotalMemory(false),
                process.WorkingSet64, process.PrivateMemorySize64, checked((long)status.AvailablePhysical));
        }

        public override string ToString() =>
            $"managed {ManagedBytes / 1_073_741_824.0:F2} GiB, " +
            $"working set {WorkingSetBytes / 1_073_741_824.0:F2} GiB, " +
            $"private {PrivateBytes / 1_073_741_824.0:F2} GiB, " +
            $"free physical {AvailablePhysicalBytes / 1_073_741_824.0:F2} GiB";
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryStatus
    {
        public uint Length;
        public uint MemoryLoad;
        public ulong TotalPhysical;
        public ulong AvailablePhysical;
        public ulong TotalPageFile;
        public ulong AvailablePageFile;
        public ulong TotalVirtual;
        public ulong AvailableVirtual;
        public ulong AvailableExtendedVirtual;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GlobalMemoryStatusEx(ref MemoryStatus status);
}

public sealed class ExplicitLargeCpuProbeFactAttribute : FactAttribute
{
    public override string? Skip
    {
        get => Environment.GetEnvironmentVariable("RWKV_CPU_7B_PROBE") == "1"
            ? base.Skip : "Set RWKV_CPU_7B_PROBE=1 only for a coordinated 7B run.";
        set => base.Skip = value;
    }
}
