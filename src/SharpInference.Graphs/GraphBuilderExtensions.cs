namespace SharpInference.Graphs;

/// <summary>Adds standard primitive operations to a logical graph builder.</summary>
public static class GraphBuilderExtensions
{
    /// <summary>Adds a graph operation that copies input elements unchanged.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="input">The input resource identifier.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    public static LogicalGraphBuilder Copy(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.Copy, input, output, regionId);

    /// <summary>Adds a graph operation that squares each input element.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="input">The input resource identifier.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    public static LogicalGraphBuilder Square(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.Square, input, output, regionId);

    /// <summary>Adds a graph operation that clamps each input element to a minimum of zero.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="input">The input resource identifier.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    public static LogicalGraphBuilder Relu(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.Relu, input, output, regionId);

    /// <summary>Adds a graph operation that computes the logistic sigmoid of each input element.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="input">The input resource identifier.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    public static LogicalGraphBuilder Sigmoid(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.Sigmoid, input, output, regionId);

    /// <summary>Adds a graph operation that computes the hyperbolic tangent of each input element.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="input">The input resource identifier.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    public static LogicalGraphBuilder Tanh(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.Tanh, input, output, regionId);

    /// <summary>Adds a graph operation that computes the exponential of each input element.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="input">The input resource identifier.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    public static LogicalGraphBuilder Exp(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.Exp, input, output, regionId);

    /// <summary>Adds a graph operation that computes the reciprocal square root of each input element.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="input">The input resource identifier.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    public static LogicalGraphBuilder ReciprocalSquareRoot(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.ReciprocalSquareRoot, input, output, regionId);

    /// <summary>Adds a graph operation that sums all input elements.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="input">The input resource identifier.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    public static LogicalGraphBuilder ReduceSum(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.ReduceSum, input, output, regionId);

    /// <summary>Adds a graph operation that computes the mean of all input elements.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="input">The input resource identifier.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    public static LogicalGraphBuilder ReduceMean(this LogicalGraphBuilder builder, string id,
        string input, string output, string? regionId = null) =>
        Unary(builder, id, PrimitiveGraphOperations.ReduceMean, input, output, regionId);

    /// <summary>Adds a graph operation that adds corresponding input elements.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="left">The left resource identifier.</param>
    /// <param name="right">The right resource identifier.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    public static LogicalGraphBuilder Add(this LogicalGraphBuilder builder, string id,
        string left, string right, string output, string? regionId = null) =>
        Binary(builder, id, PrimitiveGraphOperations.Add, left, right, output, regionId);

    /// <summary>Adds a graph operation that subtracts right-hand elements from left-hand elements.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="left">The left resource identifier.</param>
    /// <param name="right">The right resource identifier.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    public static LogicalGraphBuilder Subtract(this LogicalGraphBuilder builder, string id,
        string left, string right, string output, string? regionId = null) =>
        Binary(builder, id, PrimitiveGraphOperations.Subtract, left, right, output, regionId);

    /// <summary>Adds a graph operation that multiplies corresponding input elements.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="left">The left resource identifier.</param>
    /// <param name="right">The right resource identifier.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    public static LogicalGraphBuilder Multiply(this LogicalGraphBuilder builder, string id,
        string left, string right, string output, string? regionId = null) =>
        Binary(builder, id, PrimitiveGraphOperations.Multiply, left, right, output, regionId);

    /// <summary>Adds a graph operation that divides left-hand elements by right-hand elements.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="left">The left resource identifier.</param>
    /// <param name="right">The right resource identifier.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    public static LogicalGraphBuilder Divide(this LogicalGraphBuilder builder, string id,
        string left, string right, string output, string? regionId = null) =>
        Binary(builder, id, PrimitiveGraphOperations.Divide, left, right, output, regionId);

    /// <summary>Adds a graph operation that selects the greater corresponding input element.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="left">The left resource identifier.</param>
    /// <param name="right">The right resource identifier.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    public static LogicalGraphBuilder Maximum(this LogicalGraphBuilder builder, string id,
        string left, string right, string output, string? regionId = null) =>
        Binary(builder, id, PrimitiveGraphOperations.Maximum, left, right, output, regionId);

    /// <summary>Adds a graph operation that multiplies a matrix by an input vector.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="matrix">The matrix resource identifier.</param>
    /// <param name="input">The input resource identifier.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    public static LogicalGraphBuilder MatVec(this LogicalGraphBuilder builder, string id,
        string matrix, string input, string output, string? regionId = null) =>
        Emit(builder, id, PrimitiveGraphOperations.MatVec,
            [GraphBindings.Read("matrix", matrix), GraphBindings.Read("input", input),
             GraphBindings.Write("output", output)], regionId);

    /// <summary>Adds a graph operation that copies a selected table row.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="table">The row-major table values.</param>
    /// <param name="index">The zero-based entry index.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    public static LogicalGraphBuilder GatherRow(this LogicalGraphBuilder builder, string id,
        string table, string index, string output, string? regionId = null) =>
        Emit(builder, id, PrimitiveGraphOperations.GatherRow,
            [GraphBindings.Read("table", table), GraphBindings.Read("index", index),
             GraphBindings.Write("output", output)], regionId);

    /// <summary>Adds a graph operation that multiplies matrices with optional input transposes.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="left">The left resource identifier.</param>
    /// <param name="right">The right resource identifier.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="transposeLeft">Whether to transpose the left matrix before multiplication.</param>
    /// <param name="transposeRight">Whether to transpose the right matrix before multiplication.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    public static LogicalGraphBuilder MatrixMultiply(this LogicalGraphBuilder builder, string id,
        string left, string right, string output, bool transposeLeft = false, bool transposeRight = false,
        string? regionId = null) =>
        Emit(builder, id, PrimitiveGraphOperations.MatrixMultiply,
            [GraphBindings.Read("left", left), GraphBindings.Read("right", right),
             GraphBindings.Write("output", output)], regionId, Transposes(transposeLeft, transposeRight));

    /// <summary>Adds a graph operation that multiplies matrices with optional transposes and adds a bias.</summary>
    /// <param name="builder">The logical graph builder to extend.</param>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="left">The left resource identifier.</param>
    /// <param name="right">The right resource identifier.</param>
    /// <param name="bias">The bias resource identifier.</param>
    /// <param name="output">The output resource identifier.</param>
    /// <param name="transposeLeft">Whether to transpose the left matrix before multiplication.</param>
    /// <param name="transposeRight">Whether to transpose the right matrix before multiplication.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
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
