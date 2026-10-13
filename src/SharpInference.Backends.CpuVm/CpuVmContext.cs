using SharpInference.Vm;

namespace SharpInference.Backends.CpuVm;

/// <summary>Caller-owned dense slot buffers, in VmProgram.Slots order. No tensor storage is allocated.</summary>
public sealed class CpuVmContext
{
    private readonly byte[][] buffers;
    private readonly VmProgram program;
    private readonly HashSet<int> initialized;
    private int executing;
    private IReadOnlyDictionary<string, CpuVmCallFrame>? frames;

    /// <summary>Binds caller-owned buffers to program slots and records explicitly initialized local slots.</summary>
    /// <param name="program">The VM program whose definitions, slots, and entries are compiled or bound.</param>
    /// <param name="buffers">Caller-owned byte arrays in program slot order, each matching its slot&apos;s exact capacity.</param>
    /// <param name="initializedLocalSlots">Names of local slots supplied with initialized contents; omitted local slots must be written before reading.</param>
    public CpuVmContext(VmProgram program, byte[][] buffers, IEnumerable<string>? initializedLocalSlots = null)
    {
        ArgumentNullException.ThrowIfNull(program);
        ArgumentNullException.ThrowIfNull(buffers);
        if (buffers.Length != program.Slots.Count)
        {
            throw new ArgumentException("Buffer count must match program slots.", nameof(buffers));
        }

        this.program = program;
        this.buffers = (byte[][])buffers.Clone();
        initialized = [];
        for (var i = 0; i < buffers.Length; i++)
        {
            if (buffers[i] is null || (ulong)buffers[i].LongLength != program.Slots[i].Tensor.ByteLength)
            {
                throw new ArgumentException($"Buffer '{program.Slots[i].Id}' has an invalid capacity.", nameof(buffers));
            }

            if (program.Slots[i].Scope != VmSlotScope.Local)
            {
                initialized.Add(i);
            }

            for (var j = 0; j < i; j++)
            {
                if (ReferenceEquals(buffers[i], buffers[j]))
                {
                    throw new ArgumentException("Distinct slots cannot alias physical byte arrays.", nameof(buffers));
                }
            }
        }
        foreach (var name in initializedLocalSlots ?? [])
        {
            var index = program.Slots.ToList().FindIndex(slot => slot.Id == name);
            if (index < 0 || program.Slots[index].Scope != VmSlotScope.Local)
            {
                throw new ArgumentException($"Unknown local slot '{name}'.", nameof(initializedLocalSlots));
            }

            initialized.Add(index);
        }
    }

    /// <summary>Returns a read-only byte view of a slot in the bound buffer array.</summary>
    /// <param name="slot">The zero-based slot index in program order.</param>
    /// <param name="offset">The zero-based byte offset of the requested view.</param>
    /// <param name="length">The number of bytes in the requested view.</param>
    public ReadOnlySpan<byte> Read(int slot, int offset, int length) => buffers[slot].AsSpan(offset, length);

    /// <summary>Returns a writable byte view of a slot, rejecting slots declared read-only.</summary>
    /// <param name="slot">The zero-based slot index in program order.</param>
    /// <param name="offset">The zero-based byte offset of the requested view.</param>
    /// <param name="length">The number of bytes in the requested view.</param>
    public Span<byte> Write(int slot, int offset, int length)
    {
        if (program.Slots[slot].Access != VmAccess.ReadWrite)
        {
            throw new InvalidOperationException($"Slot '{program.Slots[slot].Id}' is read-only.");
        }

        return buffers[slot].AsSpan(offset, length);
    }

    /// <summary>Gets the compiled root call frame for an entry during an active CPU VM invocation.</summary>
    /// <param name="entry">The name of the compiled entry to invoke or inspect.</param>
    public CpuVmCallFrame GetFrame(string entry)
    {
        var active = frames ?? throw new InvalidOperationException("Compiled frames require an active CPU VM execution.");
        return active.TryGetValue(entry, out var frame) ? frame :
            throw new ArgumentException($"Unknown compiled entry '{entry}'.", nameof(entry));
    }

    internal void Run(VmProgram expected, IReadOnlyList<CpuVmAccess> accesses,
        IReadOnlyDictionary<string, CpuVmCallFrame> compiledFrames, Action action)
    {
        if (!ReferenceEquals(program, expected) && VmProgramXml.Serialize(program) != VmProgramXml.Serialize(expected))
        {
            throw new ArgumentException("Context belongs to a different program.");
        }

        if (Interlocked.CompareExchange(ref executing, 1, 0) != 0)
        {
            throw new InvalidOperationException("The context is already executing.");
        }

        try
        {
            var ranges = initialized.ToDictionary(index => index,
                index => new List<(int Start, int End)> { (0, buffers[index].Length) });
            foreach (var access in accesses)
            {
                if (!ranges.TryGetValue(access.Slot, out var available))
                {
                    ranges.Add(access.Slot, available = []);
                }

                if (!access.Write)
                {
                    var cursor = access.Offset;
                    foreach (var range in available.OrderBy(range => range.Start))
                    {
                        if (range.Start <= cursor)
                        {
                            cursor = Math.Max(cursor, range.End);
                        }
                    }

                    if (cursor < access.Offset + access.Length)
                    {
                        throw new InvalidOperationException($"Slot '{program.Slots[access.Slot].Id}' is read before initialization.");
                    }
                }
                else
                {
                    if (program.Slots[access.Slot].Access != VmAccess.ReadWrite)
                    {
                        throw new InvalidOperationException("Execution would write a read-only slot.");
                    }

                    available.Add((access.Offset, checked(access.Offset + access.Length)));
                }
            }
            // Local scratch must be initialized within each invocation, including after a failed invocation.
            frames = compiledFrames;
            action();
        }
        finally
        {
            frames = null;
            Volatile.Write(ref executing, 0);
        }
    }
}

internal sealed record CpuVmAccess(int Slot, int Offset, int Length, bool Write);
