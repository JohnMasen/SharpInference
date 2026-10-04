using SharpInference.Vm;

namespace SharpInference.Backends.CpuVm;

/// <summary>Immutable compiled binding metadata. Passing a frame never places tensor-sized argument lists on the stack.</summary>
public sealed class CpuVmCallFrame
{
    private readonly CpuVmParameterBinding[] parameters;
    private readonly CpuVmCallFrame[] calls;

    internal CpuVmCallFrame(CpuVmParameterBinding[] parameters, CpuVmCallFrame[] calls)
    {
        this.parameters = parameters;
        this.calls = calls;
    }

    public CpuVmCallFrame GetCall(int index) => calls[index];

    public ReadOnlySpan<byte> Read(CpuVmContext context, int parameter, int offset, int length)
    {
        var binding = View(parameter, offset, length);
        return context.Read(binding.Slot, checked(binding.Offset + offset), length);
    }

    public Span<byte> Write(CpuVmContext context, int parameter, int offset, int length)
    {
        var binding = View(parameter, offset, length);
        if (binding.Access != VmAccess.ReadWrite)
            throw new InvalidOperationException("The compiled parameter is read-only.");
        return context.Write(binding.Slot, checked(binding.Offset + offset), length);
    }

    private CpuVmParameterBinding View(int parameter, int offset, int length)
    {
        var binding = parameters[parameter];
        if (offset < 0 || length < 0 || offset > binding.Length || length > binding.Length - offset)
            throw new ArgumentOutOfRangeException(nameof(offset), "Parameter view exceeds its compiled capacity.");
        return binding;
    }
}

internal readonly record struct CpuVmParameterBinding(int Slot, int Offset, int Length, VmAccess Access);
