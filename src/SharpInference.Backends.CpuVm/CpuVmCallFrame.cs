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

    /// <summary>Gets the compiled call frame for a nested call by its position in the containing definition.</summary>
    /// <param name="index">The zero-based nested call index or table row index.</param>
    public CpuVmCallFrame GetCall(int index) => calls[index];

    /// <summary>Returns a read-only parameter view mapped to its bound execution-context slot.</summary>
    /// <param name="context">The execution context containing the bound slot buffers.</param>
    /// <param name="parameter">The zero-based parameter index in the compiled definition.</param>
    /// <param name="offset">The zero-based byte offset of the requested view.</param>
    /// <param name="length">The number of bytes in the requested view.</param>
    public ReadOnlySpan<byte> Read(CpuVmContext context, int parameter, int offset, int length)
    {
        var binding = View(parameter, offset, length);
        return context.Read(binding.Slot, checked(binding.Offset + offset), length);
    }

    /// <summary>Returns a writable parameter view after checking the compiled parameter access permission.</summary>
    /// <param name="context">The execution context containing the bound slot buffers.</param>
    /// <param name="parameter">The zero-based parameter index in the compiled definition.</param>
    /// <param name="offset">The zero-based byte offset of the requested view.</param>
    /// <param name="length">The number of bytes in the requested view.</param>
    public Span<byte> Write(CpuVmContext context, int parameter, int offset, int length)
    {
        var binding = View(parameter, offset, length);
        if (binding.Access != VmAccess.ReadWrite)
        {
            throw new InvalidOperationException("The compiled parameter is read-only.");
        }

        return context.Write(binding.Slot, checked(binding.Offset + offset), length);
    }

    private CpuVmParameterBinding View(int parameter, int offset, int length)
    {
        var binding = parameters[parameter];
        if (offset < 0 || length < 0 || offset > binding.Length || length > binding.Length - offset)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), "Parameter view exceeds its compiled capacity.");
        }

        return binding;
    }
}

internal readonly record struct CpuVmParameterBinding(int Slot, int Offset, int Length, VmAccess Access);
