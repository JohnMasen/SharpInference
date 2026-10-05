namespace SharpInference.Instructions.D3D12;

internal static class GpuInstructionHelpers
{
    public static InstructionHelper Load32 { get; } = new("t0.helper.0",
        "float load32(RWByteAddressBuffer b, uint o, uint i) { return asfloat(b.Load(o + i * 4)); }");
    public static InstructionHelper Maximum32 { get; } = new("t0.helper.2",
        "float maximum32(float a, float b) { if(isnan(a) || isnan(b)) return asfloat(0x7fc00000u); if(a==0.0f && b==0.0f) return asfloat((asuint(a)&asuint(b))&0x80000000u); return a>b ? a : b; }");
}
