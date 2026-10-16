namespace SharpInference.Graphs;

/// <summary>Standard backend-neutral primitive operation identifiers.</summary>
public static class PrimitiveGraphOperations
{
    /// <summary>The version-one <c>core.copy</c> operation contract identifier.</summary>
    public static readonly GraphOperationId Copy = new("core.copy");
    /// <summary>The version-one <c>core.add</c> operation contract identifier.</summary>
    public static readonly GraphOperationId Add = new("core.add");
    /// <summary>The version-one <c>core.subtract</c> operation contract identifier.</summary>
    public static readonly GraphOperationId Subtract = new("core.subtract");
    /// <summary>The version-one <c>core.multiply</c> operation contract identifier.</summary>
    public static readonly GraphOperationId Multiply = new("core.multiply");
    /// <summary>The version-one <c>core.divide</c> operation contract identifier.</summary>
    public static readonly GraphOperationId Divide = new("core.divide");
    /// <summary>The version-one <c>core.maximum</c> operation contract identifier.</summary>
    public static readonly GraphOperationId Maximum = new("core.maximum");
    /// <summary>The version-one <c>core.exp</c> operation contract identifier.</summary>
    public static readonly GraphOperationId Exp = new("core.exp");
    /// <summary>The version-one <c>core.tanh</c> operation contract identifier.</summary>
    public static readonly GraphOperationId Tanh = new("core.tanh");
    /// <summary>The version-one <c>core.sigmoid</c> operation contract identifier.</summary>
    public static readonly GraphOperationId Sigmoid = new("core.sigmoid");
    /// <summary>The version-one <c>core.rsqrt</c> operation contract identifier.</summary>
    public static readonly GraphOperationId ReciprocalSquareRoot = new("core.rsqrt");
    /// <summary>The version-one <c>core.square</c> operation contract identifier.</summary>
    public static readonly GraphOperationId Square = new("core.square");
    /// <summary>The version-one <c>core.relu</c> operation contract identifier.</summary>
    public static readonly GraphOperationId Relu = new("core.relu");
    /// <summary>The version-one <c>core.reduce-sum</c> operation contract identifier.</summary>
    public static readonly GraphOperationId ReduceSum = new("core.reduce-sum");
    /// <summary>The version-one <c>core.reduce-mean</c> operation contract identifier.</summary>
    public static readonly GraphOperationId ReduceMean = new("core.reduce-mean");
    /// <summary>The version-one <c>core.mat-vec</c> operation contract identifier.</summary>
    public static readonly GraphOperationId MatVec = new("core.mat-vec");
    /// <summary>The version-one <c>core.matrix-multiply</c> operation contract identifier.</summary>
    public static readonly GraphOperationId MatrixMultiply = new("core.matrix-multiply");
    /// <summary>The version-one <c>core.affine</c> operation contract identifier.</summary>
    public static readonly GraphOperationId Affine = new("core.affine");
    /// <summary>The version-one <c>core.bias-add</c> operation contract identifier.</summary>
    public static readonly GraphOperationId BiasAdd = new("core.bias-add");
    /// <summary>The version-one <c>core.gather-row</c> operation contract identifier.</summary>
    public static readonly GraphOperationId GatherRow = new("core.gather-row");
}
