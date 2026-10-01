StructuredBuffer<float> input0 : register(t0);
StructuredBuffer<float> input1 : register(t1);
RWStructuredBuffer<float> output : register(u0);
cbuffer Constants : register(b0)
{
    uint elementCount;
    uint rowCount;
    uint columnCount;
    uint parameter;
    uint source0;
    uint source1;
    uint source2;
    uint source3;
    uint output0;
    uint output1;
    uint output2;
    uint output3;
};

uint SourceDimension(uint axis)
{
    if (axis == 0) return source0;
    if (axis == 1) return source1;
    if (axis == 2) return source2;
    return source3;
}

uint OutputDimension(uint axis)
{
    if (axis == 0) return output0;
    if (axis == 1) return output1;
    if (axis == 2) return output2;
    return output3;
}

uint LinearIndex(uint3 id, uint3 group)
{
    return id.x + group.y * 65535u * 64u;
}

[numthreads(64, 1, 1)]
void Fill(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID)
{
    uint index = LinearIndex(id, group);
    if (index < elementCount)
        output[index] = asfloat(parameter);
}

[numthreads(64, 1, 1)]
void PortableCopy(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID)
{
    uint index = LinearIndex(id, group);
    if (index < elementCount)
        output[index] = input0[index];
}

[numthreads(64, 1, 1)]
void Slice(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID)
{
    uint index = LinearIndex(id, group);
    if (index < elementCount)
    {
        uint coordinates[4] = { 0, 0, 0, 0 };
        uint flat = index;
        for (int axis = int(columnCount) - 1; axis >= 0; axis--)
        {
            coordinates[axis] = flat % OutputDimension(uint(axis));
            flat /= OutputDimension(uint(axis));
        }
        uint sourceIndex = 0;
        for (uint axis = 0; axis < columnCount; axis++)
            sourceIndex = sourceIndex * SourceDimension(axis) +
                coordinates[axis] + (axis == rowCount ? parameter : 0);
        output[index] = input0[sourceIndex];
    }
}

[numthreads(64, 1, 1)]
void Broadcast(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID)
{
    uint index = LinearIndex(id, group);
    if (index < elementCount)
    {
        uint coordinates[4] = { 0, 0, 0, 0 };
        uint flat = index;
        for (int axis = int(columnCount) - 1; axis >= 0; axis--)
        {
            coordinates[axis] = flat % OutputDimension(uint(axis));
            flat /= OutputDimension(uint(axis));
        }
        uint sourceIndex = 0;
        uint offset = columnCount - rowCount;
        for (uint axis = offset; axis < columnCount; axis++)
        {
            uint sourceAxis = axis - offset;
            uint dimension = SourceDimension(sourceAxis);
            sourceIndex = sourceIndex * dimension +
                (dimension == 1 ? 0 : coordinates[axis]);
        }
        output[index] = input0[sourceIndex];
    }
}

[numthreads(64, 1, 1)]
void BatchedMatVec(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID)
{
    uint index = LinearIndex(id, group);
    if (index < elementCount)
    {
        uint batch = index / rowCount;
        uint row = index % rowCount;
        float sum = 0.0f;
        for (uint column = 0; column < columnCount; column++)
            sum += input0[(batch * rowCount + row) * columnCount + column] *
                   input1[batch * columnCount + column];
        output[index] = sum;
    }
}

[numthreads(64, 1, 1)]
void ReduceLastSum(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID)
{
    uint index = LinearIndex(id, group);
    if (index < elementCount)
    {
        float sum = 0.0f;
        for (uint column = 0; column < columnCount; column++)
            sum += input0[index * columnCount + column];
        output[index] = sum;
    }
}

[numthreads(64, 1, 1)]
void ReduceLastMean(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID)
{
    uint index = LinearIndex(id, group);
    if (index < elementCount)
    {
        float sum = 0.0f;
        for (uint column = 0; column < columnCount; column++)
            sum += input0[index * columnCount + column];
        output[index] = sum / float(columnCount);
    }
}

[numthreads(64, 1, 1)]
void HeadOuter(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID)
{
    uint index = LinearIndex(id, group);
    if (index < elementCount)
    {
        uint head = index / (rowCount * columnCount);
        uint row = (index / columnCount) % rowCount;
        uint column = index % columnCount;
        output[index] = input0[head * rowCount + row] *
                       input1[head * columnCount + column];
    }
}

groupshared float matVecPartial[64];

[numthreads(64, 1, 1)]
void MatVecLarge(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint row = group.y * 65535u + group.x;
    if (row >= rowCount) return;
    float sum = 0.0f;
    for (uint column = lane; column < columnCount; column += 64)
        sum += input0[row * columnCount + column] * input1[column];
    matVecPartial[lane] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32; stride > 0; stride >>= 1)
    {
        if (lane < stride) matVecPartial[lane] += matVecPartial[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0) output[row] = matVecPartial[0];
}
