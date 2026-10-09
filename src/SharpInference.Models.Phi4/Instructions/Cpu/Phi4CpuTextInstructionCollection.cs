using System.Globalization;
using System.Text;
using SharpInference.Graphs;

namespace SharpInference.Instructions.Phi4.Cpu;

public sealed class Phi4CpuTextInstructionCollection : IInstructionCollectionProvider, IGraphInstructionProvider
{
    private readonly IReadOnlyList<GraphInstructionBinding> bindings = Phi4TextGraphOperations.Bindings(InstructionTarget.Cpu)
        .Where(binding => binding.Operation != Phi4TextGraphOperations.AdvancePosition).ToArray();
    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
        [new(Phi4InstructionCollectionIds.TextFloat32, "Phi4 FP32 decoder operations", 1, InstructionTarget.Cpu)];
    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        collectionId == Phi4InstructionCollectionIds.TextFloat32 && bindings.Any(binding => binding.InstructionName == instructionName)
            ? [new TextInstruction(instructionName)] : [];
    public IReadOnlyList<GraphInstructionBinding> QueryGraphInstructionBindings() => bindings;

    private sealed class TextInstruction(string name) : Instruction
    {
        public override Guid CollectionId => Phi4InstructionCollectionIds.TextFloat32;
        public override string Name => name;
        public override InstructionTarget Target => InstructionTarget.Cpu;
        public override IReadOnlyList<InstructionIndexBound> IndexBounds => Name switch
        {
            "transformer.rope-kv-write" or "transformer.gqa-scores" => [new("position", "key_cache", 0)],
            "transformer.gqa-values" => [new("position", "value_cache", 0)],
            "transformer.causal-softmax" => [new("position", "scores", 1)],
            _ => [],
        };
        public override IReadOnlyList<InstructionSignature> Signatures => [Signature()];
        private InstructionSignature Signature()
        {
            var precision = new KernelPrecisionProfile(GraphElementType.Float32, GraphElementType.Float32);
            return Name switch
            {
                "transformer.rms-norm" => new([R("input"), R("weight"), W("output")], ["epsilon"], precision),
                "transformer.swiglu" => new([R("gate_up"), W("output")], [], precision),
                "transformer.scaled-add" => new([R("input"), R("update"), W("output")], ["scale"], precision),
                "transformer.rope-kv-write" => new([R("qkv"), R("position", GraphElementType.Int32), R("frequencies"), W("query"),
                    new("key_cache", GraphElementType.Float32, GraphResourceAccess.ReadWrite),
                    new("value_cache", GraphElementType.Float32, GraphResourceAccess.ReadWrite)], ["head_size", "rotary_size", "rope_scale"], precision),
                "transformer.gqa-scores" => new([R("query"), R("key_cache"), R("position", GraphElementType.Int32), W("scores")], [], precision),
                "transformer.causal-softmax" => new([R("scores"), R("position", GraphElementType.Int32), W("probabilities")], [], precision),
                "transformer.gqa-values" => new([R("probabilities"), R("value_cache"), R("position", GraphElementType.Int32), W("output")], [], precision),
                _ => throw new NotSupportedException(Name),
            };
        }
        private static InstructionPort R(string name, GraphElementType type = GraphElementType.Float32) => new(name, type, GraphResourceAccess.Read);
        private static InstructionPort W(string name) => new(name, GraphElementType.Float32, GraphResourceAccess.Write);

        protected override InstructionRecording Generate(InstructionParameter[] parameters)
        {
            var tensors = parameters.OfType<InstructionTensorParameter>().ToDictionary(value => value.Name);
            var attributes = parameters.OfType<InstructionAttributeParameter>().ToDictionary(value => value.Name, value => value.Value);
            var code = new StringBuilder("{\n");
            foreach (var tensor in tensors.Values)
            {
                var type = tensor.Tensor.ElementType == GraphElementType.Int32 ? "int" : "float";
                code.AppendLine($"var {tensor.Name} = System.Runtime.InteropServices.MemoryMarshal.Cast<byte,{type}>({tensor.Expression});");
            }
            foreach (var bound in IndexBounds)
                code.AppendLine($"foreach(var index in {bound.IndexPort}) if(index < {bound.MinimumIndex} || index >= {D(bound.TensorPort, bound.Axis)}) throw new System.ArgumentOutOfRangeException(\"{bound.IndexPort}\");");
            var call = Name switch
            {
                "transformer.rms-norm" => $"RmsNorm(input,weight,output,{F("epsilon")})",
                "transformer.swiglu" => "SwiGlu(gate_up,output)",
                "transformer.scaled-add" => $"ScaledAdd(input,update,output,{F("scale")})",
                "transformer.rope-kv-write" => $"Rope(qkv,position[0],frequencies,query,key_cache,value_cache,{D("key_cache", 1)},{I("head_size")},{I("rotary_size")},{F("rope_scale")})",
                "transformer.gqa-scores" => $"Scores(query,key_cache,position[0],scores,{D("scores", 0)},{D("key_cache", 1)},{D("key_cache", 2)},{D("key_cache", 0)})",
                "transformer.causal-softmax" => $"Softmax(scores,position[0],probabilities,{D("scores", 0)},{D("scores", 1)})",
                "transformer.gqa-values" => $"Values(probabilities,value_cache,position[0],output,{D("probabilities", 0)},{D("value_cache", 1)},{D("value_cache", 2)},{D("value_cache", 0)})",
                _ => throw new NotSupportedException(Name),
            };
            code.AppendLine("SharpInference.Instructions.Phi4.Cpu.Phi4CpuTextNumerics." + call + ";");
            code.AppendLine("}");
            return new(code.ToString(), references: [typeof(Phi4CpuTextNumerics).Assembly]);
            string D(string port, int axis) => tensors[port].Tensor.Dimensions[axis].ToString(CultureInfo.InvariantCulture);
            string I(string key) => int.Parse(attributes[key], CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture);
            string F(string key) => float.Parse(attributes[key], CultureInfo.InvariantCulture).ToString("R", CultureInfo.InvariantCulture) + "f";
        }
    }
}

public static class Phi4CpuTextNumerics
{
    public static void RmsNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weight, Span<float> output, float epsilon)
    {
        var sum = 0f;
        foreach (var value in input) sum += value * value;
        var scale = 1f / MathF.Sqrt(sum / input.Length + epsilon);
        for (var i = 0; i < output.Length; i++) output[i] = input[i] * scale * weight[i];
    }
    public static void SwiGlu(ReadOnlySpan<float> gateUp, Span<float> output)
    {
        for (var i = 0; i < output.Length; i++) output[i] = gateUp[i] / (1 + MathF.Exp(-gateUp[i])) * gateUp[output.Length + i];
    }
    public static void ScaledAdd(ReadOnlySpan<float> input, ReadOnlySpan<float> update, Span<float> output, float scale)
    {
        for (var i = 0; i < output.Length; i++) output[i] = input[i] + scale * update[i];
    }
    public static void Rope(ReadOnlySpan<float> qkv, int position, ReadOnlySpan<float> frequencies, Span<float> query,
        Span<float> keys, Span<float> values, int kvHeads, int headSize, int rotary, float scale)
    {
        var kvWidth = checked(kvHeads * headSize);
        for (var i = 0; i < query.Length; i++) query[i] = Rotate(qkv, i, i % headSize, position, frequencies, rotary, scale);
        for (var i = 0; i < kvWidth; i++)
        {
            keys[position * kvWidth + i] = Rotate(qkv, query.Length + i, i % headSize, position, frequencies, rotary, scale);
            values[position * kvWidth + i] = qkv[query.Length + kvWidth + i];
        }
    }
    private static float Rotate(ReadOnlySpan<float> qkv, int index, int channel, int position, ReadOnlySpan<float> frequencies, int rotary, float scale)
    {
        if (channel >= rotary) return qkv[index];
        var half = rotary / 2;
        var left = channel < half;
        var angle = position * frequencies[left ? channel : channel - half];
        var other = qkv[left ? index + half : index - half];
        return scale * (qkv[index] * MathF.Cos(angle) + (left ? -other : other) * MathF.Sin(angle));
    }
    public static void Scores(ReadOnlySpan<float> query, ReadOnlySpan<float> keys, int position, Span<float> scores,
        int heads, int kvHeads, int headSize, int context)
    {
        scores.Fill(float.NegativeInfinity);
        for (var h = 0; h < heads; h++)
        for (var t = 0; t <= position; t++)
        {
            var sum = 0f;
            for (var d = 0; d < headSize; d++) sum += query[h * headSize + d] * keys[(t * kvHeads + h / (heads / kvHeads)) * headSize + d];
            scores[h * context + t] = sum / MathF.Sqrt(headSize);
        }
    }
    public static void Softmax(ReadOnlySpan<float> scores, int position, Span<float> probabilities, int heads, int context)
    {
        probabilities.Clear();
        for (var h = 0; h < heads; h++)
        {
            var maximum = float.NegativeInfinity;
            for (var t = 0; t <= position; t++) maximum = MathF.Max(maximum, scores[h * context + t]);
            var sum = 0f;
            for (var t = 0; t <= position; t++) sum += probabilities[h * context + t] = MathF.Exp(scores[h * context + t] - maximum);
            for (var t = 0; t <= position; t++) probabilities[h * context + t] /= sum;
        }
    }
    public static void Values(ReadOnlySpan<float> probabilities, ReadOnlySpan<float> values, int position, Span<float> output,
        int heads, int kvHeads, int headSize, int context)
    {
        for (var h = 0; h < heads; h++)
        for (var d = 0; d < headSize; d++)
        {
            var sum = 0f;
            for (var t = 0; t <= position; t++) sum += probabilities[h * context + t] * values[(t * kvHeads + h / (heads / kvHeads)) * headSize + d];
            output[h * headSize + d] = sum;
        }
    }
}
