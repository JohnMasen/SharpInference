ByteAddressBuffer input0 : register(t0);
RWStructuredBuffer<float> output : register(u0);
cbuffer Constants : register(b0)
{
    uint elementCount;
};

[numthreads(64, 1, 1)]
void CastFp16ToFp32(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID)
{
    uint index = id.x + group.y * 65535u * 64u;
    if (index < elementCount)
    {
        uint pair = input0.Load((index / 2) * 4);
        output[index] = f16tof32((pair >> ((index & 1) * 16)) & 0xffff);
    }
}
