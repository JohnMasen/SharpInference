namespace SharpInference.Instructions.D3D12;

public static class GpuInstructionHelpers
{
    public static InstructionHelper Store16
    {
        get;
    } = new("t0.store16",
        "void store16(RWByteAddressBuffer b,uint o,uint i,float value) { uint a=o+i*2u; uint shift=(a&2u)*8u; uint mask=65535u<<shift; uint bits=(f32tof16(value)&65535u)<<shift; uint old=b.Load(a&~3u); uint observed; do { uint next=(old&~mask)|bits; b.InterlockedCompareExchange(a&~3u,old,next,observed); if(observed==old) break; old=observed; } while(true); }");
    public static InstructionHelper Load32
    {
        get;
    } = new("t0.helper.0",
        "float load32(RWByteAddressBuffer b, uint o, uint i) { return asfloat(b.Load(o + i * 4)); }");
    public static InstructionHelper Maximum32
    {
        get;
    } = new("t0.helper.2",
        "float maximum32(float a, float b) { if(isnan(a) || isnan(b)) return asfloat(0x7fc00000u); if(a==0.0f && b==0.0f) return asfloat((asuint(a)&asuint(b))&0x80000000u); return a>b ? a : b; }");
}
