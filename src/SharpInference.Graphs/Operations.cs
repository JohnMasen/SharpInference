namespace SharpInference.Graphs;

/// <summary>Version-one primitive operation names; OperationId accepts explicit contract versions.</summary>
public static class Operations
{
    public const string Copy = "core.copy";
    public const string Add = "core.add";
    public const string Subtract = "core.subtract";
    public const string Multiply = "core.multiply";
    public const string Divide = "core.divide";
    public const string Maximum = "core.maximum";
    public const string Exp = "core.exp";
    public const string Tanh = "core.tanh";
    public const string Sigmoid = "core.sigmoid";
    public const string ReciprocalSquareRoot = "core.rsqrt";
    public const string Square = "core.square";
    public const string ReLU = "core.relu";
    public const string ReduceSum = "core.reduce-sum";
    public const string ReduceMean = "core.reduce-mean";
    public const string MatVec = "core.mat-vec";
    public const string MatMul = "core.matrix-multiply";
    public const string Affine = "core.affine";
    public const string BiasAdd = "core.bias-add";
    public const string GatherRow = "core.gather-row";
}
