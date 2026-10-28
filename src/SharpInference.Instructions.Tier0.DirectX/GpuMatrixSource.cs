using SharpInference.Graphs;
using System.Globalization;

namespace SharpInference.Instructions.D3D12;

public static class GpuMatrixSource
{
    public static InstructionRecording Generate(
        InstructionParameter[] values,
        bool addBias,
        bool cooperative,
        bool tiled)
    {
        var tensors = values.OfType<InstructionTensorParameter>()
            .ToDictionary(parameter => parameter.Name, StringComparer.Ordinal);
        var attributes = values.OfType<InstructionAttributeParameter>()
            .ToDictionary(parameter => parameter.Name, parameter => parameter.Value,
                StringComparer.Ordinal);
        var left = tensors["left"];
        var right = tensors["right"];
        var bias = addBias ? tensors["bias"] : null;
        var output = tensors["output"];
        var transposeLeft = bool.Parse(attributes["transpose_left"]);
        var transposeRight = bool.Parse(attributes["transpose_right"]);
        var leftRows = left.Tensor.Dimensions[0];
        var leftColumns = left.Tensor.Dimensions[1];
        var rightRows = right.Tensor.Dimensions[0];
        var rightColumns = right.Tensor.Dimensions[1];
        var rows = transposeLeft ? leftColumns : leftRows;
        var inner = transposeLeft ? leftRows : leftColumns;
        var columns = transposeRight ? rightRows : rightColumns;
        string Store(string index) => output.Tensor.ElementType == ElementType.Float16
            ? $"store16({output.Expression},{output.OffsetExpression},{index},result);"
            : $"{output.Expression}.Store({output.OffsetExpression}+({index})*4u,asuint(result));";
        string Load(InstructionTensorParameter tensor, string index) =>
            $"load{(tensor.Tensor.ElementType == ElementType.Float16 ? 16 : 32)}" +
            $"({tensor.Expression},{tensor.OffsetExpression},{index})";
        var result = bias is null
            ? "sum"
            : bias.Tensor.ElementType == ElementType.Float16
                ? $"sum+load16({bias.Expression},{bias.OffsetExpression},column)"
                : $"sum+load32({bias.Expression},{bias.OffsetExpression},column)";
        var serialSource = $$"""
            {
                uint element=i;
                if(element<{{rows * columns}}u) {
                    uint row=element/{{columns}}u;
                    uint column=element-row*{{columns}}u;
                    precise float sum=0.0f;
                    for(uint index=0u;index<{{inner}}u;index++) {
                        uint leftIndex={{(transposeLeft
                            ? $"index*{leftColumns}u+row"
                            : $"row*{leftColumns}u+index")}};
                        uint rightIndex={{(transposeRight
                            ? $"column*{rightColumns}u+index"
                            : $"index*{rightColumns}u+column")}};
                        sum+={{Load(left, "leftIndex")}}*{{Load(right, "rightIndex")}};
                    }
                    precise float result={{result}};
                    {{Store("element")}}
                }
            }
            """;
        if (!cooperative && !tiled)
        {
            return new(serialSource,
            [
                GpuInstructionHelpers.Load32,
                GpuInstructionHelpers.Store16,
                new("t0.matrix.load16",
                    "float load16(RWByteAddressBuffer b,uint o,uint i){uint a=o+i*2u;return f16tof32((b.Load(a&~3u)>>((a&2u)*8u))&65535u);}")
            ]);
        }

        if (tiled)
        {
            var tiledResult = bias is null
                ? "sum"
                : bias.Tensor.ElementType == ElementType.Float16
                    ? $"sum+load16({bias.Expression},{bias.OffsetExpression},column)"
                    : $"sum+load32({bias.Expression},{bias.OffsetExpression},column)";
            var tiledSource = $$"""
                {
                    uint tileColumn=gpu.groupIndex%8u;
                    uint tileRow=gpu.groupIndex/8u;
                    uint row=gpu.groupId.y*8u+tileRow;
                    uint column=gpu.groupId.x*8u+tileColumn;
                    precise float sum=0.0f;
                    for(uint tile=0u;tile<{{inner}}u;tile+=8u) {
                        uint leftInner=tile+tileColumn;
                        uint rightInner=tile+tileRow;
                        precise float leftValue=0.0f;
                        precise float rightValue=0.0f;
                        if(row<{{rows}}u && leftInner<{{inner}}u) {
                            uint leftIndex={{(transposeLeft
                                ? $"leftInner*{leftColumns}u+row"
                                : $"row*{leftColumns}u+leftInner")}};
                            leftValue={{Load(left, "leftIndex")}};
                        }
                        if(column<{{columns}}u && rightInner<{{inner}}u) {
                            uint rightIndex={{(transposeRight
                                ? $"column*{rightColumns}u+rightInner"
                                : $"rightInner*{rightColumns}u+column")}};
                            rightValue={{Load(right, "rightIndex")}};
                        }
                        matrixMultiplyLeftTile[gpu.groupIndex]=leftValue;
                        matrixMultiplyRightTile[gpu.groupIndex]=rightValue;
                        GroupMemoryBarrierWithGroupSync();
                        for(uint index=0u;index<8u;index++)
                            sum+=matrixMultiplyLeftTile[tileRow*8u+index]*
                                matrixMultiplyRightTile[index*8u+tileColumn];
                        GroupMemoryBarrierWithGroupSync();
                    }
                    if(row<{{rows}}u && column<{{columns}}u) {
                        precise float result={{tiledResult}};
                        {{Store($"row*{columns}u+column")}}
                    }
                }
                """;
            return new(tiledSource,
            [
                GpuInstructionHelpers.Load32,
                GpuInstructionHelpers.Store16,
                new("t0.matrix.load16",
                    "float load16(RWByteAddressBuffer b,uint o,uint i){uint a=o+i*2u;return f16tof32((b.Load(a&~3u)>>((a&2u)*8u))&65535u);}"),
                new("t0.matrix.tiles",
                    "groupshared float matrixMultiplyLeftTile[64];groupshared float matrixMultiplyRightTile[64];")
            ], InstructionSynchronization.GroupMemoryBarrier);
        }
        var cooperativeResult = bias is null
            ? "matrixMultiplySum[0]+matrixMultiplyCorrection[0]"
            : bias.Tensor.ElementType == ElementType.Float16
                ? $"matrixMultiplySum[0]+matrixMultiplyCorrection[0]+load16({bias.Expression},{bias.OffsetExpression},column)"
                : $"matrixMultiplySum[0]+matrixMultiplyCorrection[0]+load32({bias.Expression},{bias.OffsetExpression},column)";
        var cooperativeSource = $$"""
            {
                uint element=gpu.groupId.x+gpu.groupId.y*gpu.groupCount.x+
                    gpu.groupId.z*gpu.groupCount.x*gpu.groupCount.y;
                uint lane=gpu.groupIndex;
                if(element<{{rows * columns}}u) {
                    uint row=element/{{columns}}u;
                    uint column=element-row*{{columns}}u;
                    precise float sum=0.0f, correction=0.0f;
                    for(uint index=lane;index<{{inner}}u;index+=64u) {
                        uint leftIndex={{(transposeLeft
                            ? $"index*{leftColumns}u+row"
                            : $"row*{leftColumns}u+index")}};
                        uint rightIndex={{(transposeRight
                            ? $"column*{rightColumns}u+index"
                            : $"index*{rightColumns}u+column")}};
                        precise float product={{Load(left, "leftIndex")}}*
                            {{Load(right, "rightIndex")}};
                        matrixMultiplyAdd(sum,correction,product);
                    }
                    matrixMultiplySum[lane]=sum;
                    matrixMultiplyCorrection[lane]=correction;
                    GroupMemoryBarrierWithGroupSync();
                    for(uint step=32u;step>0u;step>>=1u) {
                        if(lane<step) {
                            precise float leftSum=matrixMultiplySum[lane];
                            precise float leftCorrection=matrixMultiplyCorrection[lane];
                            matrixMultiplyAdd(
                                leftSum,leftCorrection,matrixMultiplySum[lane+step]);
                            matrixMultiplyAdd(
                                leftSum,leftCorrection,matrixMultiplyCorrection[lane+step]);
                            matrixMultiplySum[lane]=leftSum;
                            matrixMultiplyCorrection[lane]=leftCorrection;
                        }
                        GroupMemoryBarrierWithGroupSync();
                    }
                    if(lane==0u) {
                        precise float result={{cooperativeResult}};
                        {{Store("element")}}
                    }
                }
            }
            """;
        return new(cooperativeSource,
        [
            GpuInstructionHelpers.Load32,
            GpuInstructionHelpers.Store16,
            new("t0.matrix.load16",
                "float load16(RWByteAddressBuffer b,uint o,uint i){uint a=o+i*2u;return f16tof32((b.Load(a&~3u)>>((a&2u)*8u))&65535u);}"),
            new("t0.matrix.add",
                "void matrixMultiplyAdd(inout float sum,inout float correction,float value){precise float total=sum+value;if(isfinite(total)){precise float residual=abs(sum)>=abs(value)?(sum-total)+value:(value-total)+sum;correction+=residual;}else correction=0.0f;sum=total;}"),
            new("t0.matrix.shared",
                "groupshared float matrixMultiplySum[64];groupshared float matrixMultiplyCorrection[64];")
        ], InstructionSynchronization.GroupMemoryBarrier);
    }

}
