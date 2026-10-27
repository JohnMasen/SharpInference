using SharpInference.Graphs;
namespace SharpInference.Instructions;

/// <summary>Standard backend-neutral primitive operation identifiers.</summary>
public static class PrimitiveOperations
{
    /// <summary>The version-one <c>core.copy</c> operation contract identifier.</summary>
    public static readonly OperationId Copy = new("core.copy");
    /// <summary>The version-one <c>core.add</c> operation contract identifier.</summary>
    public static readonly OperationId Add = new("core.add");
    /// <summary>The version-one <c>core.subtract</c> operation contract identifier.</summary>
    public static readonly OperationId Subtract = new("core.subtract");
    /// <summary>The version-one <c>core.multiply</c> operation contract identifier.</summary>
    public static readonly OperationId Multiply = new("core.multiply");
    /// <summary>The version-one <c>core.divide</c> operation contract identifier.</summary>
    public static readonly OperationId Divide = new("core.divide");
    /// <summary>The version-one <c>core.maximum</c> operation contract identifier.</summary>
    public static readonly OperationId Maximum = new("core.maximum");
    /// <summary>The version-one <c>core.exp</c> operation contract identifier.</summary>
    public static readonly OperationId Exp = new("core.exp");
    /// <summary>The version-one <c>core.tanh</c> operation contract identifier.</summary>
    public static readonly OperationId Tanh = new("core.tanh");
    /// <summary>The version-one <c>core.sigmoid</c> operation contract identifier.</summary>
    public static readonly OperationId Sigmoid = new("core.sigmoid");
    /// <summary>The version-one <c>core.rsqrt</c> operation contract identifier.</summary>
    public static readonly OperationId ReciprocalSquareRoot = new("core.rsqrt");
    /// <summary>The version-one <c>core.square</c> operation contract identifier.</summary>
    public static readonly OperationId Square = new("core.square");
    /// <summary>The version-one <c>core.relu</c> operation contract identifier.</summary>
    public static readonly OperationId Relu = new("core.relu");
    /// <summary>The version-one <c>core.reduce-sum</c> operation contract identifier.</summary>
    public static readonly OperationId ReduceSum = new("core.reduce-sum");
    /// <summary>The version-one <c>core.reduce-mean</c> operation contract identifier.</summary>
    public static readonly OperationId ReduceMean = new("core.reduce-mean");
    /// <summary>The version-one <c>core.mat-vec</c> operation contract identifier.</summary>
    public static readonly OperationId MatVec = new("core.mat-vec");
    /// <summary>The version-one <c>core.matrix-multiply</c> operation contract identifier.</summary>
    public static readonly OperationId MatrixMultiply = new("core.matrix-multiply");
    /// <summary>The version-one <c>core.affine</c> operation contract identifier.</summary>
    public static readonly OperationId Affine = new("core.affine");
    /// <summary>The version-one <c>core.bias-add</c> operation contract identifier.</summary>
    public static readonly OperationId BiasAdd = new("core.bias-add");
    /// <summary>The version-one <c>core.gather-row</c> operation contract identifier.</summary>
    public static readonly OperationId GatherRow = new("core.gather-row");
}
