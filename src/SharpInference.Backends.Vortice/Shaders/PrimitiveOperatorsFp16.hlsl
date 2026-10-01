StructuredBuffer<float16_t> input0 : register(t0);
StructuredBuffer<float16_t> input1 : register(t1);
RWStructuredBuffer<float16_t> output : register(u0);
cbuffer Constants : register(b0)
{
    uint elementCount;
    uint rowCount;
    uint columnCount;
    uint rowIndex;
};

groupshared float partial[64];

#define UNARY_KERNEL(name, expression) \
[numthreads(64, 1, 1)] \
void name(uint3 id : SV_DispatchThreadID) \
{ \
    if (id.x < elementCount) output[id.x] = float16_t(expression); \
}

#define BINARY_KERNEL(name, expression) \
[numthreads(64, 1, 1)] \
void name(uint3 id : SV_DispatchThreadID) \
{ \
    if (id.x < elementCount) output[id.x] = float16_t(expression); \
}

UNARY_KERNEL(Copy, input0[id.x])
BINARY_KERNEL(Add, input0[id.x] + input1[id.x])
BINARY_KERNEL(Subtract, input0[id.x] - input1[id.x])
BINARY_KERNEL(Multiply, input0[id.x] * input1[id.x])
BINARY_KERNEL(Divide, input0[id.x] / input1[id.x])
BINARY_KERNEL(Maximum, max(input0[id.x], input1[id.x]))
UNARY_KERNEL(Exp, exp(input0[id.x]))
UNARY_KERNEL(Tanh, tanh(input0[id.x]))
UNARY_KERNEL(Sigmoid, float16_t(1.0h) / (float16_t(1.0h) + exp(-input0[id.x])))
UNARY_KERNEL(ReciprocalSquareRoot, rsqrt(input0[id.x]))
UNARY_KERNEL(Square, input0[id.x] * input0[id.x])
UNARY_KERNEL(Relu, max(float16_t(0.0h), input0[id.x]))

void ReduceCore(uint lane, bool mean)
{
    float sum = 0.0f;
    for (uint index = lane; index < elementCount; index += 64)
        sum += float(input0[index]);
    partial[lane] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32; stride > 0; stride >>= 1)
    {
        if (lane < stride) partial[lane] += partial[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0)
        output[0] = float16_t(mean ? partial[0] / float(elementCount) : partial[0]);
}

[numthreads(64, 1, 1)]
void ReduceSum(uint lane : SV_GroupIndex)
{
    ReduceCore(lane, false);
}

[numthreads(64, 1, 1)]
void ReduceMean(uint lane : SV_GroupIndex)
{
    ReduceCore(lane, true);
}

[numthreads(64, 1, 1)]
void MatVec(uint3 groupId : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint row = groupId.x;
    float sum = 0.0f;
    for (uint column = lane; column < columnCount; column += 64)
        sum += float(input0[row * columnCount + column]) * float(input1[column]);
    partial[lane] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32; stride > 0; stride >>= 1)
    {
        if (lane < stride) partial[lane] += partial[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0) output[row] = float16_t(partial[0]);
}

[numthreads(64, 1, 1)]
void GatherRow(uint3 id : SV_DispatchThreadID)
{
    if (id.x < columnCount)
        output[id.x] = input0[rowIndex * columnCount + id.x];
}
