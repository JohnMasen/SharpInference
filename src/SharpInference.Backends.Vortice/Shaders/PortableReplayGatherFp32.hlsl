StructuredBuffer<float> table : register(t0);
StructuredBuffer<uint> tokenIndex : register(t2);
RWStructuredBuffer<float> output : register(u0);
cbuffer Constants : register(b0)
{
    uint elementCount;
    uint rowCount;
    uint columnCount;
    uint parameter;
};

[numthreads(64, 1, 1)]
void ReplayGatherRow(uint3 id : SV_DispatchThreadID)
{
    if (id.x < columnCount)
        output[id.x] = table[tokenIndex[0] * columnCount + id.x];
}
