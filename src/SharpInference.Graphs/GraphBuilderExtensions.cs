namespace SharpInference.Graphs;

public static class GraphBuilderExtensions
{
    public static LogicalGraphBuilder Copy(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.Copy, input, output, regionId);

    public static LogicalGraphBuilder Square(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.Square, input, output, regionId);

    public static LogicalGraphBuilder Relu(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.Relu, input, output, regionId);

    public static LogicalGraphBuilder Sigmoid(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.Sigmoid, input, output, regionId);

    public static LogicalGraphBuilder Tanh(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.Tanh, input, output, regionId);

    public static LogicalGraphBuilder Exp(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.Exp, input, output, regionId);

    public static LogicalGraphBuilder ReciprocalSquareRoot(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.ReciprocalSquareRoot, input, output, regionId);

    public static LogicalGraphBuilder ReduceSum(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.ReduceSum, input, output, regionId);

    public static LogicalGraphBuilder ReduceMean(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.ReduceMean, input, output, regionId);

    public static LogicalGraphBuilder Add(this LogicalGraphBuilder builder, string id,
        string left, string right, string output, string? regionId = null) =>
        Binary(builder, id, PrimitiveGraphOperations.Add, left, right, output, regionId);

    public static LogicalGraphBuilder Subtract(this LogicalGraphBuilder builder, string id,
        string left, string right, string output, string? regionId = null) =>
        Binary(builder, id, PrimitiveGraphOperations.Subtract, left, right, output, regionId);

    public static LogicalGraphBuilder Multiply(this LogicalGraphBuilder builder, string id,
        string left, string right, string output, string? regionId = null) =>
        Binary(builder, id, PrimitiveGraphOperations.Multiply, left, right, output, regionId);

    public static LogicalGraphBuilder Divide(this LogicalGraphBuilder builder, string id,
        string left, string right, string output, string? regionId = null) =>
        Binary(builder, id, PrimitiveGraphOperations.Divide, left, right, output, regionId);

    public static LogicalGraphBuilder Maximum(this LogicalGraphBuilder builder, string id,
        string left, string right, string output, string? regionId = null) =>
        Binary(builder, id, PrimitiveGraphOperations.Maximum, left, right, output, regionId);

    public static LogicalGraphBuilder MatVec(this LogicalGraphBuilder builder, string id,
        string matrix, string input, string output, string? regionId = null) =>
        Emit(builder, id, PrimitiveGraphOperations.MatVec,
            [GraphBindings.Read("matrix", matrix), GraphBindings.Read("input", input),
             GraphBindings.Write("output", output)], regionId);

    public static LogicalGraphBuilder GatherRow(this LogicalGraphBuilder builder, string id,
        string table, string index, string output, string? regionId = null) =>
        Emit(builder, id, PrimitiveGraphOperations.GatherRow,
            [GraphBindings.Read("table", table), GraphBindings.Read("index", index),
             GraphBindings.Write("output", output)], regionId);

    public static LogicalGraphBuilder MatrixMultiply(this LogicalGraphBuilder builder, string id,
        string left, string right, string output, bool transposeLeft = false, bool transposeRight = false,
        string? regionId = null) =>
        Emit(builder, id, PrimitiveGraphOperations.MatrixMultiply,
            [GraphBindings.Read("left", left), GraphBindings.Read("right", right),
             GraphBindings.Write("output", output)], regionId, Transposes(transposeLeft, transposeRight));

    public static LogicalGraphBuilder Affine(this LogicalGraphBuilder builder, string id,
        string left, string right, string bias, string output,
        bool transposeLeft = false, bool transposeRight = false, string? regionId = null) =>
        Emit(builder, id, PrimitiveGraphOperations.Affine,
            [GraphBindings.Read("left", left), GraphBindings.Read("right", right), GraphBindings.Read("bias", bias),
             GraphBindings.Write("output", output)], regionId, Transposes(transposeLeft, transposeRight));

    private static LogicalGraphBuilder Unary(LogicalGraphBuilder builder, string id,
        GraphOperationId operation, string input, string output, string? regionId) =>
        Emit(builder, id, operation,
            [GraphBindings.Read("input", input), GraphBindings.Write("output", output)], regionId);

    private static LogicalGraphBuilder Binary(LogicalGraphBuilder builder, string id,
        GraphOperationId operation, string left, string right, string output, string? regionId) =>
        Emit(builder, id, operation,
            [GraphBindings.Read("left", left), GraphBindings.Read("right", right),
             GraphBindings.Write("output", output)], regionId);

    private static LogicalGraphBuilder Emit(LogicalGraphBuilder builder, string id,
        GraphOperationId operation, NodeResourceBinding[] bindings, string? regionId,
        IReadOnlyDictionary<string, string>? attributes = null)
    {
        ArgumentNullException.ThrowIfNull(builder);
        return regionId is null
            ? builder.AddNode(id, operation, bindings, attributes: attributes)
            : builder.AddNode(id, operation, regionId, bindings, attributes: attributes);
    }

    private static IReadOnlyDictionary<string, string> Transposes(bool left, bool right) =>
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["transpose_left"] = left ? "true" : "false",
            ["transpose_right"] = right ? "true" : "false",
        };
}
