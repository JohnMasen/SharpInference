using System.Runtime.InteropServices;
using SharpInference.Vm;

namespace SharpInference.Vm.Optimization;

public static class VmModelBindings
{
    public static void InitializeGlobal(IModelTensorCatalog catalog, VmSlot slot, IVmStorage storage)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        ArgumentNullException.ThrowIfNull(slot);
        ArgumentNullException.ThrowIfNull(storage);
        if (slot.Scope != VmSlotScope.Global || storage.ByteLength != slot.Tensor.ByteLength)
            throw new ArgumentException("Global storage does not match the slot descriptor.", nameof(storage));
        var tensor = catalog.GetRequired(slot.BindingKey ??
            throw new InvalidDataException($"Global slot '{slot.Id}' has no model binding."));
        if (!tensor.Dimensions.SequenceEqual(slot.Tensor.Dimensions))
            throw new InvalidDataException($"Weight '{tensor.Name}' has an incompatible shape.");
        if (slot.Tensor.ElementType == VmElementType.Float32 && tensor.DataType == TensorDataType.Float32)
            storage.Write(0, MemoryMarshal.AsBytes(tensor.FloatValues));
        else if (slot.Tensor.ElementType == VmElementType.Float16 && tensor.DataType == TensorDataType.Float16)
            storage.Write(0, MemoryMarshal.AsBytes(tensor.HalfValues));
        else throw new InvalidDataException($"Weight '{tensor.Name}' has an incompatible storage type.");
    }
}
