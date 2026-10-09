using SharpInference.Graphs;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Runtime;

/// <summary>Executes descriptor-bound graphs without imposing token, logits or float32-state semantics.</summary>
public sealed class VmTensorGraphSession : IDisposable
{
    private readonly object gate = new();
    private readonly VmCompiledPlan plan;
    private readonly GraphTensorState state;
    private readonly VmResourceManager manager;
    private readonly bool ownsManager;
    private readonly Action<VmTensorGraphSession>? onDispose;
    private VmSessionResources? session;
    private VmBindings? bindings;
    private IVmExecutable? executable;
    private bool disposed;

    internal VmTensorGraphSession(VmCompiledPlan plan, IModelTensorCatalog tensors, GraphTensorState state,
        Func<VmSlot, IVmStorage>? allocate, Action<VmTensorGraphSession>? onDispose = null,
        VmResourceManager? sharedManager = null)
    {
        this.plan = plan;
        this.state = state;
        this.onDispose = onDispose;
        VmGraphBindingValidator.Validate(plan.Program, plan.BindingGraph);
        if (state.ArchitectureId != plan.BindingGraph.Identity.ArchitectureId ||
            state.StateAbiId != plan.BindingGraph.Model.StateAbiId ||
            !state.Schema.Equals(plan.BindingGraph.GraphState.Schema) ||
            state.Buffers.Count != plan.Program.State.Entries.Count)
            throw new ArgumentException("The state does not belong to the compiled graph.", nameof(state));
        foreach (var entry in plan.BindingGraph.GraphState)
        {
            var buffer = state.Buffers.SingleOrDefault(candidate => candidate.Name == entry.Name);
            var tensor = plan.BindingGraph.Resources.Single(resource => resource.Id == entry.Resource).Tensor;
            if (buffer is null || buffer.Tensor.ElementType != tensor.ElementType ||
                buffer.Tensor.Layout != tensor.Layout || !buffer.Tensor.Dimensions.SequenceEqual(tensor.Dimensions))
                throw new ArgumentException($"State entry '{entry.Name}' has an incompatible descriptor.", nameof(state));
        }
        ownsManager = sharedManager is null;
        manager = sharedManager ?? new((slot, storage) => VmModelBindings.InitializeGlobal(tensors, slot, storage), allocate);
        try
        {
            bindings = manager.CreateBindings(plan.Program);
            session = manager.CreateSession(plan.Program);
            manager.BindSession(bindings, session);
            executable = plan.CreateExecutable();
            if (VmProgramXml.Serialize(executable.Program) != VmProgramXml.Serialize(plan.Program))
                throw new InvalidDataException("The executable does not match the compiled graph program.");
            WriteState();
        }
        catch (Exception error)
        {
            try { Dispose(); }
            catch (Exception cleanup) { throw new AggregateException("Graph session creation and cleanup failed.", error, cleanup); }
            throw;
        }
    }

    public IReadOnlyDictionary<ResourceId, byte[]> Execute(
        IReadOnlyDictionary<ResourceId, ReadOnlyMemory<byte>> inputs, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            cancellationToken.ThrowIfCancellationRequested();
            if (inputs.Count != plan.BindingGraph.Inputs.Count ||
                plan.BindingGraph.Inputs.Any(id => !inputs.ContainsKey(id)))
                throw new ArgumentException("Inputs must exactly cover the graph input resources.", nameof(inputs));
            foreach (var id in plan.BindingGraph.Inputs)
                if ((ulong)inputs[id].Length != plan.Program.Slots.Single(slot => slot.Id == id.Value).Tensor.ByteLength)
                    throw new ArgumentException($"Input '{id}' does not match its tensor descriptor.", nameof(inputs));
            WriteState();
            using var execution = bindings!.BeginExecution();
            foreach (var id in plan.BindingGraph.Inputs)
                execution.GetStorage(id.Value).Write(0, inputs[id].Span);
            try
            {
                if (executable is IVmTaskExecutable task)
                    task.ExecuteTask("forward", execution, cancellationToken);
                else
                    executable!.Execute("forward", execution.GetBuffers());
                cancellationToken.ThrowIfCancellationRequested();
                var stateValues = new Dictionary<string, byte[]>(StringComparer.Ordinal);
                foreach (var buffer in state.Buffers)
                {
                    var bytes = new byte[buffer.Data.Length];
                    execution.Read(plan.Program.State.Entries.Single(entry => entry.Name == buffer.Name).Slot,
                        0, bytes);
                    stateValues.Add(buffer.Name, bytes);
                }
                var outputs = new Dictionary<ResourceId, byte[]>();
                foreach (var id in plan.BindingGraph.Outputs)
                {
                    var slot = plan.Program.Slots.Single(slot => slot.Id == id.Value);
                    var bytes = new byte[checked((int)slot.Tensor.ByteLength)];
                    execution.Read(id.Value, 0, bytes);
                    outputs.Add(id, bytes);
                }
                cancellationToken.ThrowIfCancellationRequested();
                foreach (var buffer in state.Buffers)
                    stateValues[buffer.Name].AsSpan().CopyTo(buffer.Data.Span);
                return outputs;
            }
            catch
            {
                execution.InvalidateState();
                throw;
            }
        }
    }

    private void WriteState()
    {
        using var access = bindings!.BeginStateAccess();
        access.RestoreState(state.Buffers.ToDictionary(buffer => buffer.Name,
            buffer => (ReadOnlyMemory<byte>)buffer.Data, StringComparer.Ordinal));
    }

    public void Reset()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            state.Reset();
        }
    }

    public void Dispose()
    {
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            List<Exception>? errors = null;
            foreach (var resource in new IDisposable?[] { executable, bindings, session, ownsManager ? manager : null })
                try { resource?.Dispose(); }
                catch (Exception error) { (errors ??= []).Add(error); }
            try { onDispose?.Invoke(this); }
            catch (Exception error) { (errors ??= []).Add(error); }
            if (errors is not null) throw new AggregateException("Graph session cleanup failed.", errors);
        }
    }
}
