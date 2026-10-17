
namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Describes a call site derived from an execution graph's call node.</summary>
/// <param name="Id">The relation identity, formed from the caller and call identities.</param>
/// <param name="Source">The caller graph identity.</param>
/// <param name="Target">The callee graph identity.</param>
/// <param name="Call">The call node identity within the caller.</param>
public sealed record ExecutionGraphCallRelation(
    string Id,
    ExecutionGraphId Source,
    ExecutionGraphId Target,
    ExecutionNodeId Call);
