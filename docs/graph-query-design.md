# Graph 查询工具包设计与 20 节点查询示例

状态：基础工具包已实现并接入消费者。本文的 18 个例子已迁入正式测试；
实现与迁移结果见第 7 节。最初的 .NET 10 单文件原型用于验证接口语义，
当前另有直接引用生产项目的单文件性能比较。
XML 序列化、SQL Server 集成、图修改不属于本次范围。

## 1. 架构

统一将完整图、过滤结果和每一次模式匹配的结果表示为只读的
`IGraphView<TNode, TEdge>`。对这些对象使用同一组查询操作。
可达性索引是独立辅助对象，不实现图视图接口。

```text
IGraphView<TNode, TEdge>
    ├─ Graph                 完整图，维护 ID 和入边/出边索引
    ├─ FilteredGraphView     当前视图的筛选结果
    └─ MatchedGraphView      某一次匹配的精确节点、边集合

任意 IGraphView
    ├─ Filter(...)                 → IGraphView
    ├─ Match(pattern)              → IEnumerable<IGraphView>
    ├─ TopologicalOrder()          → 有序节点集合
    ├─ ToGraph()                   → 独立 Graph
    └─ CreateReachabilityIndex(...)→ 独立 ReachabilityIndex

IEnumerable<IGraphView>
    ├─ Any() / FirstOrDefault()    → 存在性 / 首个匹配
    └─ ToGraph()                   → 合并并物化所有匹配
```

### 1.1 共同接口与共享实现

```csharp
public interface IGraphView<TNode, TEdge>
    where TNode : IGraphNode
    where TEdge : IGraphEdge
{
    IEnumerable<TNode> Nodes { get; }
    IEnumerable<TEdge> Edges { get; }
    TNode GetNode(GraphNodeId id);
    bool ContainsNode(GraphNodeId id);
    IEnumerable<TEdge> EdgesToNode(GraphNodeId id);
    IEnumerable<TEdge> EdgesFromNode(GraphNodeId id);
}
```

这是最小只读接口；现有 Graph 的 GetEdge 等额外能力不因此删除。

`Filter`、`Match`、拓扑排序和索引创建都是针对 IGraphView 的扩展方法，
不作为接口成员，也无需共同基类。
调用方式仍为 `view.Filter(...)`、`view.Match(...)`。
Graph、GraphView、MatchedGraphView 直接实现只读接口，外部视图只需实现
底层读取能力即可复用全部查询算法。

ToGraph 有两个扩展重载：`view.ToGraph()` 将单个视图物化为独立图；
`views.ToGraph()` 针对 IEnumerable<IGraphView<TNode, TEdge>>，
消费并合并整个视图集合，尤其用于 `view.Match(pattern).ToGraph()`。
后者不是单个视图的实例方法。

图引擎源文件按一个顶层类型一个文件组织；接口、ID 类型、模式约束、
视图实现、索引构造和图重写类型均独立成文件。
节点与边直接实现 IGraphNode / IGraphEdge，不保留没有消费者的可选基类。

所有视图都必须满足：

- 可见边的两个端点均为可见节点。
- `EdgesToNode` / `EdgesFromNode` 只能返回该视图可见的边。
- 二次 Filter/Match 不得重新带回上一级视图外的节点和边。
- 元素声明顺序稳定；完整 Graph 使用构造顺序，线性匹配使用路径顺序。
- ID 唯一；Graph 负责结构校验。视图不自动增加边或节点。

#### 相邻边命名与底层索引

新接口采用直接表达方向的名称：

| 名称 | 含义 |
|---|---|
| `EdgesToNode(nodeId)` | 所有直接指向该节点的可见边 |
| `EdgesFromNode(nodeId)` | 所有直接从该节点出发的可见边 |

两者返回边而非节点，只查询一步连接，不遍历多步路径。
不存在或不可见的节点沿用明确报错行为；存在但没有对应边的节点返回空集合。

保留当前 Graph 的索引构建方式，不引入 SortedList 或支持重复键的有序容器：

```csharp
Dictionary<GraphNodeId, List<TEdge>> edgesToNode;
Dictionary<GraphNodeId, List<TEdge>> edgesFromNode;
```

构造时分别按 Target、Source 分组，最后冻结为只读字典和只读边列表。
字典中每个节点 ID 只有一个键，同一节点的多条边存放在该键对应的列表中。
例如 A -(e1)→ C、B -(e2)→ C、A -(e3)→ C：

```text
EdgesToNode:   C → [e1, e2, e3]
EdgesFromNode: A → [e1, e3]
              B → [e2]
```

e1、e3 虽然端点相同，只要边 ID 不同就都保留。列表保持边声明顺序；
无需按节点键排序。查找相关列表平均 O(1)，遍历成本取决于该节点的相关边数。
过滤视图可复用父视图的相邻边索引，再限制为自身可见内容。

Match 的结果枚举顺序不必等同于相邻边列表顺序。需要统一下游顺序时，
调用方可以复用同一份已物化的匹配结果或明确排序；不应根据当前优化器
未明确依赖顺序，就自动宣称所有现有调用都与顺序无关。

### 1.2 Filter 与 Match 的不同

```csharp
IGraphView<TNode, TEdge> Filter(
    Func<TNode, bool>? node = null,
    Func<TEdge, bool>? edge = null);

IEnumerable<IGraphView<TNode, TEdge>> Match(MatchContext pattern);
```

`Filter` 对当前视图的节点、边分别筛选，不要求找到一条完整路径。
节点被去掉时，相连的边也去掉；允许留下没有边的节点。

`Match` 按模式找到一个个完整连接，每项只包含参与该次匹配的节点和边。
生产实现应按需产生结果，每次结果使用独立、稳定的集合，不能暴露回溯时会
继续变化的工作数组。`Any()` 在首个匹配处短路。

### 1.3 模式构造与名字

```csharp
var pattern = Pattern.Create(Node("A"))
    - Edge().Where<DemoEdge>(e => e.Kind == "dep")
    - Node();
```

- `Pattern.Create` 是唯一公开创建入口，创建不可变 `MatchContext`。
- 只定义 `MatchContext - EdgePattern`、`MatchContext - NodePattern`。
- 序列为 `Node - Edge - Node ...`，沿原图边方向匹配；单节点模式也合法。
- 不自动在两个相邻 Node 之间补 Edge；不完整或不交替的序列报告错误。
- `Node("A")` 的字符串是目标节点名字，不是捕获别名；不提供 `WithName`。
- `Node(GraphNodeId)` 按 ID 匹配；`Node()` 无名字/ID 限制。
- GraphNodeId 支持从 string 隐式转换，例如 `GraphNodeId id = "A"`、
  `graph.GetNode("A")` 和 `index.IsReachable("A", "B")`。
  转换保持原始字符串，ID 合法性仍由图构造和模式入口校验。
  因为存在精确的 Node(string) 重载，`Node("A")` 仍按名称匹配；
  按 ID 匹配时使用 GraphNodeId 变量或 `Node((GraphNodeId)"A")`。
- `Edge()` 代表任意一条直接边，不代表任意长度路径。
- 节点、边的 `Where<TRequested>(Func<TRequested, bool>)` 先检查类型再执行条件。
  多个条件按 AND 合并；不匹配类型时不调用谓词。

为使名字语义可执行，本原型假设一个可选的 `INamedGraphNode : IGraphNode`
接口暴露 `Name`。`Node(string)` 只匹配实现该接口的节点。现有生产节点尚未
实现此契约，接入时需要显式适配；不能把业务名称默认为带前缀的结构 ID。
本例所有节点名字与 ID 相同，但这不是接口对生产图的要求。

本版保留普通 Func 谓词。表达式树优化、字段索引以后再评估；引擎不能从
任意 Func 自动识别 ID/Name 等条件。正式实现应利用显式 ID 和邻接索引。
只限定末端节点的模式，可在内部从末端入边开始查找，但返回顺序必须遵守
约定。当前生产实现固定从起点开始枚举，尚未实现从终点反向启动的搜索优化。

本版固定长度匹配不允许同一条匹配路径重复节点。任意长度路径、环路径枚举、
Shortest/Longest 的全局选择尚未实现；不能用一个直接 Edge 冒充可达查询。
将来的 MatchOne 可以作为结果序列的选择扩展，Any 等价 FirstOrDefault；
可变长度匹配确定后再增加最短/最长策略，避免暗中改变已讨论的需求。

### 1.4 ToGraph

```csharp
Graph<TNode, TEdge> one = singleView.ToGraph();
Graph<TNode, TEdge> combined = matches.ToGraph();
```

单视图版本物化当前可见内容；序列版本枚举一次，按 ID 合并节点和边，
只保留实际命中的内容，不补入原图中的其他边。空序列得到空图。

去重不是 Match 的职责；ToGraph 为满足图内 ID 唯一性而合并同 ID 元素。
序列版本预期消费同一来源图的结果；若调用方混合不同图，相同 ID 指向不相同
元素时正式实现须拒绝合并，不能默默覆盖。原型只验证同源合并。

调用 ToGraph 就立即消费序列并构图，不再等下游消费者出现。
合并会丢失“哪些节点原本属于哪次匹配”的分组信息，二次匹配可以横跨原来的
不同匹配。需要保留分组时，分别在每项视图上查询。

### 1.5 ReachabilityIndex

```csharp
ReachabilityIndex<TNode> CreateReachabilityIndex<TNode, TEdge>(
    this IGraphView<TNode, TEdge> view,
    IEnumerable<TNode> sources,
    IEnumerable<TNode> targets)
    where TNode : IGraphNode
    where TEdge : IGraphEdge;

bool IsReachable(GraphNodeId source, GraphNodeId target);
bool IsReachable(TNode source, TNode target);
```

- 索引独立于 view，只有该对象暴露预计算的 IsReachable。
- 两个序列各消费一次、按 ID 去重，构造端点范围的快照。
- 输入端点必须属于 view；没有索引的起点/终点查询报告错误，不返回 false。
- 节点实例重载转换为 ID；身份以 ID 为准，而不是对象引用。
- 中间节点可以是 view 的任意节点，不限于 sources/targets。
- 零步可达：`IsReachable(a, a)` 为 true。严格先后判断须排除相同 ID。
- 支持空端点集合；位集合只保存“起点数 × 终点数”的关系。
- 原型采用每个起点一次遍历，支持有环图；有环时 visited 集合保证终止。
- 正式实现可选择 DAG 反向拓扑合并以加快构建，但最终压缩矩阵大小不代表
  构造时一定只使用这么多内存。大量重复查询才适合承担构造成本。
- 索引限定于创建时的图范围。另一个过滤结果或重写结果需要自己的索引。
  若节点属性/谓词依赖可变外部状态，创建时应固定可见集合，不能让索引
  与后续 view 枚举的范围不一致。

拓扑排序独立保留：需要 DAG 顺序时直接计算，不必为它同时建立全量可达索引。

## 2. 示例图：20 个节点、35 条边

使用泛型 `Graph<DemoNode, DemoEdge>`。`DemoNode` 具有
`Id`、`Name`、`Kind`、`Enabled`；`DemoEdge` 具有
`Id`、`Source`、`Target`、`Kind`。

- 操作节点 12 个：A、B、C、D、E、F、G、H、I、J、K、L。
- 资源节点 8 个：R1、R2、R3、R4、R5、R6、R7、R8。
- 仅 H 的 Enabled 为 false；其余为 true。
- 节点声明顺序：A 到 L，然后 R1 到 R8。
- 边声明顺序：d1 到 d15，再按下面访问边表从上到下声明。

### 2.1 依赖边（Kind = dep）

```text
A ─┐          ┌→ D ─┐          ┌→ G ─┐
   ├→ C ──────┤     ├→ F ──────┤     ├→ I → J → K → L
B ─┘          └→ E ─┘          └→ H ─┘

另外：A → F 有一条直接边；
      C → D 有两条平行边。
```

| 边 ID | 起点 | 终点 |
|---|---|---|
| d1 | A | C |
| d2 | B | C |
| d3 | C | D |
| d4 | C | E |
| d5 | D | F |
| d6 | E | F |
| d7 | F | G |
| d8 | F | H |
| d9 | G | I |
| d10 | H | I |
| d11 | I | J |
| d12 | J | K |
| d13 | K | L |
| d14 | A | F |
| d15 | C | D |

### 2.2 资源访问边（write / read）

write 的方向为操作到资源；read 的方向为资源到操作。
资源边与依赖边是不同关系，不可混在一起证明操作先后。

| 资源 | 写入边 | 读取边 |
|---|---|---|
| R1 | w1: A → R1 | r1: R1 → C；r2: R1 → C；r3: R1 → D |
| R2 | w2: B → R2 | r4: R2 → C |
| R3 | w3: C → R3 | r5: R3 → D；r6: R3 → E |
| R4 | w4: D → R4 | r7: R4 → F |
| R5 | w5: E → R5 | r8: R5 → F |
| R6 | w6: F → R6 | r9: R6 → G；r10: R6 → H |
| R7 | w7: I → R7 | r11: R7 → J |
| R8 | w8: K → R8 | r12: R8 → L |

共有 15 条依赖边、8 条 write 边、12 条 read 边，合计 35 条边。

## 3. 从简单到复杂的 18 个查询

以下示例中 `Node` / `Edge` 是模式工厂，Node("A") 按名字筛选；
`graph` 为完整图。使用以下公共准备代码：

```csharp
var dep = graph.Filter(n => n.Kind == "op", e => e.Kind == "dep");
EdgePattern Kind(string kind) => Edge().Where<DemoEdge>(e => e.Kind == kind);
NodePattern Id(string id) => Node(new GraphNodeId(id));
```

### Q01：启用的操作节点

```csharp
var result = graph.Match(Pattern.Create(
    Node().Where<DemoNode>(n => n.Kind == "op" && n.Enabled)))
    .Select(m => m.Nodes.Single());
```

结果：A、B、C、D、E、F、G、I、J、K、L，共 11 个，不含 H。
每项匹配只有一个节点、没有边。

### Q02：复杂节点条件（A 或 B）

```csharp
var result = graph.Match(Pattern.Create(
    Node().Where<DemoNode>(n => n.Name == "A" || n.Name == "B")))
    .Select(m => m.Nodes.Single());
```

结果：A、B。普通 Where 能表达 OR；本版不承诺自动索引加速。

### Q03：C 的直接后继

```csharp
var result = dep.Match(Pattern.Create(Node("C")) - Edge() - Node())
    .Select(m => m.Nodes.Last());
```

结果：D、E、D，分别对应 d3、d4、d15。Match 不去重。
外围 `.DistinctBy(n => n.Id)` 后为 D、E。

### Q04：F 的直接前驱

```csharp
var result = dep.Match(Pattern.Create(Node()) - Edge() - Node("F"))
    .Select(m => m.Nodes.First());
```

结果：A、D、E，按匹配起点声明顺序。
当前实现的 EdgesToNode(F.Id) 会按边声明顺序给出 D、E、A。
集合相同、顺序不同；迁移现有调用必须明确是否依赖此顺序，不能只验证集合。

### Q05：R1 的读取者

```csharp
var result = graph.Match(Pattern.Create(Node("R1")) - Kind("read") - Node())
    .Select(m => m.Nodes.Last())
    .DistinctBy(n => n.Id);
```

原始匹配为 C、C、D；外围去重后 C、D。对应当前 Readers 的用法。

### Q06：R3 的写入者

```csharp
var result = graph.Match(Pattern.Create(Node()) - Kind("write") - Node("R3"))
    .Select(m => m.Nodes.First())
    .DistinctBy(n => n.Id);
```

结果：C。对应当前 Writers 的用法。

### Q07：找所有“写入者 → 资源 → 读取者”组合

```csharp
var result = graph.Match(Pattern.Create(Node())
    - Kind("write") - Node()
    - Kind("read") - Node());
```

结果共 12 个；A → R1 → C 因 r1、r2 两条读取边出现两次。
结果数量来自实际匹配边，不是唯一节点组合数量。

### Q08：C 到 F 的两步依赖连接

```csharp
var result = dep.Match(Pattern.Create(Node("C"))
    - Edge() - Node()
    - Edge() - Node("F"));
```

结果三项：C -(d3)→ D → F、C → E → F、C -(d15)→ D → F。
固定两步模式不是任意长度路径查询。

### Q09：将 R1 的读取匹配合并为图

```csharp
var readGraph = graph.Match(
    Pattern.Create(Node("R1")) - Kind("read") - Node()).ToGraph();
```

节点：R1、C、D；边：r1、r2、r3。
两个不同 ID 的 R1 → C 平行边都保留，不增加 C → D 依赖边。

### Q10：在 Q09 的结果图上做第二次匹配

```csharp
bool exists = readGraph.Match(
    Pattern.Create(Node("C")) - Kind("dep") - Node("D")).Any();
```

结果 false。虽然原图有 C → D，但该边没有参与第一次匹配，
结果图不能把它带回来。这是二次筛选最重要的范围约束。

### Q11：过滤掉 H

```csharp
var withoutH = dep.Filter(n => n.Name != "H");
```

结果：11 个操作节点、13 条依赖边；d8（F → H）、d10（H → I）也消失。
两条 C → D 仍保留。

### Q12：只判断直接连接是否存在

```csharp
bool aToF = dep.Match(Pattern.Create(Node("A")) - Edge() - Node("F")).Any();
bool cToL = dep.Match(Pattern.Create(Node("C")) - Edge() - Node("L")).Any();
```

结果：true、false。C 间接可到 L，但没有直接连接。
Any 只消费到首个匹配；原型另检查首项存在时只调用一次节点谓词。

### Q13：稳定拓扑顺序

```csharp
var order = dep.TopologicalOrder();
```

结果：A、B、C、D、E、F、G、H、I、J、K、L。
平行边正确计入入度；不能因为两条边连同一对节点而漏减一次入度。
这是专用图算法，不通过重复 Match 推导。

### Q14：限定起点、终点范围的可达索引

```csharp
var index = dep.CreateReachabilityIndex(
    dep.Nodes.Where(n => new[] { "A", "B", "D", "E", "G", "H" }.Contains(n.Name)),
    dep.Nodes.Where(n => new[] { "F", "I", "L" }.Contains(n.Name)));

bool aToL = index.IsReachable(new GraphNodeId("A"), new GraphNodeId("L"));
bool gToF = index.IsReachable(dep.GetNode(new("G")), dep.GetNode(new("F")));
```

结果 true、false。最终只保存 6 × 3 个关系位，构建时仍经过其他中间节点。

| 起点 | F | I | L |
|---|---|---|---|
| A | true | true | true |
| B | true | true | true |
| D | true | true | true |
| E | true | true | true |
| G | false | true | true |
| H | false | true | true |

查询 C → L 应报“起点不在索引范围内”，不能返回 false。

以下例子需要反复比较所有操作节点，创建另一份索引：

```csharp
var allIndex = dep.CreateReachabilityIndex(dep.Nodes, dep.Nodes);
```

### Q15：资源生命周期的末端访问（可能有多个）

先用 Q05、Q06 形式定义 Readers/Writers，再组合：

```csharp
DemoNode[] Terminal(string resource)
{
    var accesses = Readers(resource).Concat(Writers(resource))
        .DistinctBy(n => n.Id).ToArray();
    return accesses.Where(a => !accesses.Any(b =>
        a.Id != b.Id && allIndex.IsReachable(a, b))).ToArray();
}
```

R1 的访问为 A、C、D，末端为 D。
R6 的访问为 F、G、H，末端为 G、H；G 与 H 之间没有先后关系。
不能简单选拓扑顺序中最后一个节点代替“全部末端访问”。

### Q16：两个资源的生命周期是否已被依赖完全分开

假设 R2、R4、R5 的首次写入分别为 B、D、E：

```csharp
bool r2BeforeR4 = Terminal("R2")
    .All(last => last.Id != d.Id && allIndex.IsReachable(last, d));
bool r4BeforeR5 = Terminal("R4")
    .All(last => last.Id != e.Id && allIndex.IsReachable(last, e));
```

其中 `d`、`e` 为 D、E 节点实例。
结果 true、false：R2 最后被 C 使用，C 先于 D；R4 最后被 F 使用，
F 不先于 E。

这里只证明先后关系，不能直接宣布资源可以复用；完整写入、资源类别、
张量兼容、外部绑定等检查仍在优化器中。

### Q17：找重复或具有其他依赖路径的边

使用当前依赖约简的搜索方式，不必对每条边枚举所有替代路径：

```csharp
var seen = new HashSet<(GraphNodeId, GraphNodeId)>();
var redundant = dep.Edges.Where(edge =>
    !seen.Add((edge.Source, edge.Target)) ||
    dep.EdgesFromNode(edge.Source).Any(other =>
        other.Target != edge.Target &&
        allIndex.IsReachable(other.Target, edge.Target)));
```

结果 d14（A → F）和 d15（第二条 C → D）。
d14 可由 A → C → D/E → F 替代；d15 由保留的 d3 替代。
该组合适用于当前依赖 DAG 约简；不是通用有环图的批量删边算法。
不执行修改，仅取得候选边。保持声明顺序让重复边保留第一条。

### Q18：找没有依赖后继的节点

```csharp
var sinks = dep.Nodes.Where(n => !dep.EdgesFromNode(n.Id).Any());
```

结果 L。对应图完成节点的搜索。
这是直接邻接查询，没必要为了统一写法构造 Match 或可达性索引。

## 4. 可行性结论与需补充的边界

| 用法 | 本方案验证结果 |
|---|---|
| 单节点筛选、自定义 OR 条件 | 可行：Filter 或单节点 Match |
| Readers/Writers、固定两步连接 | 可行：Match 加外围 LINQ |
| 匹配结果变图、第二次搜索 | 可行：精确边集合，不带回原图其他边 |
| 依赖顺序、重复可达性判断 | 可行：拓扑排序与独立 ReachabilityIndex |
| 生命周期末端、候选依赖边约简 | 可组合实现，领域规则仍在外围 |
| 任意长度所有路径、最短/最长路径 | 本原型未实现，不能声称已被固定 Edge 模式覆盖 |
| 分支/汇合的完整模式、共享模式变量 | 本原型未实现；当前可以外围组合，但并非通用分支匹配语言 |
| 字段索引、Name 自动索引、表达式树分析 | 未实现；本次验证不证明性能收益 |
| 改写、安全复用决策 | 不在搜索工具职责内 |

额外通过的原型边界检查：

- `IsReachable(A, A)` 为 true；严格 Before 必须额外排除相同 ID。
- 只索引 A → L 时，中间节点仍可参与路径。
- L → A 在原 DAG 中为 false。
- 超出端点范围的查询抛异常；空端点集合可创建。
- 额外加入 L → A 后，可达索引仍能构建并正确回答 L → C、
  L 不可到 B；同一有环图的拓扑排序拒绝计算。
- Match 的 Any 查询在第一个成功节点后停止。

这些检查证明接口和搜索结果可组合，不证明查询速度接近现有索引实现。
正式接入前仍需验证：原 API 的结果顺序、异常语义、节点访问类型、
同源 ID 合并、过滤条件的稳定性、名称契约适配、生产级迭代器分配及性能。

## 5. 对现有实现的迁移

- Graph 实现 IGraphView，保留现有 ID、入边、出边索引及校验。
  接口使用 EdgesToNode/EdgesFromNode；删除旧 InEdges/OutEdges 名称，
  调用点统一更新，命名调整不改变底层存储结构。
- GraphView 演进为过滤视图；算法依赖接口而非具体图类。
- 删除旧 GraphPattern/GraphMatch，统一使用 Pattern/MatchContext/匹配视图。
  这是明确的破坏性 API 清理：当前只支持固定线性模式，不宣称等价支持
  旧匹配器的分支或断开模式。现有仓库调用使用的线性模式已迁移。
- ExecutionDependencyAnalysis 的通用位集合计算可由 ReachabilityIndex
  提供；拓扑位置表、后继表是否抽取为额外工具根据剩余真实使用决定。
- ModelGraph 的资源类型、读写边约定和业务组合保留在模型层，
  不把这些业务概念硬编码进泛型 Graph.Core。
- GraphRewriteTarget 的边界捕获与 GraphRewritePlan 保留；
  匹配视图只含命中边，不能替代“找出目标所有外部连接”的改写边界捕获。

## 6. 验证方式

使用 .NET SDK 10.0.401 的单文件应用运行 C# 参考原型：

```powershell
dotnet run --file "<GraphQueryExamples.cs 的本地路径>"
```

结果：18/18 个查询及上述边界检查通过，图为 20 节点、35 边。
验证原型位于本次会话附件中，未写入生产项目，也不依赖 Python。
原型为独立的语义验证，不是对仓库现有实现的集成测试。
最初的会话原型使用旧名称 InEdges/OutEdges；生产实现和正式测试已使用
EdgesToNode/EdgesFromNode。生产接口不再保留旧名称别名。

## 7. 实施结果与 API 边界

### 已实现

- Graph 与 GraphView 实现 IGraphView，新增 Filter、惰性 Match 与两种 ToGraph。
- Pattern.Create / MatchContext / PatternElements 支持显式 ID、可选节点名称、
  类型化 Func 条件和不可变运算符链。
- 单次匹配使用轻量只读视图，复制本次路径集合，不重建完整 Graph 索引。
- ReachabilityIndex 支持端点范围、空集合、零步、有环图及节点实例查询。
  视图构建索引时冻结可见图；稀疏起点使用遍历，较密起点的 DAG
  使用反向拓扑传播。稀疏遍历找到所有目标后即可停止。
- 拓扑排序及单次 BFS 算法接受 IGraphView。
- 模型查询可用于任意 IGraphView；资源冲突和多 writer 比较复用可达索引。
- ExecutionDependencyAnalysis 保留原类和调用接口，将位集合委托给新索引；
  缓存结构 ID，Before 的自身比较仍为 false。
- 重写器与模型邻接查询使用新名称，不改变删除、重连或领域判断规则。

不引入 GraphViewBase：Graph、过滤视图、匹配视图直接实现 IGraphView
的读取契约，查询算法通过扩展方法共享。ToGraph 同时支持单个视图物化
和视图集合合并，不扩充接口契约。源文件按顶层类型拆分，
项目的显式 Compile 项已同步更新。

### 移除的旧入口与保留的独立能力

- 删除 View、InEdges、OutEdges，分别使用 Filter、EdgesToNode、EdgesFromNode。
- 删除 GraphPattern/GraphMatch 及命名捕获、Find 入口；
  匹配结果直接是 IGraphView，按节点 ID 或路径顺序访问。
  新线性匹配不支持旧匹配器的一般分支、断开模式。
- IsReachable/ReachableFrom 的 BFS 便捷 API 保留，用于有环图或单次查询。
  可变长 Match 尚未实现，不能据此删除单次多步可达查询。
- 领域节点没有被强制加入 Name；实际消费者按原结构 ID 查询。
- Readers/Writers 保留邻接索引、声明顺序及外围去重，没有强制换成 Match。

### 正式验证

测试位于 [GraphViewQueryTests.cs](../src/SharpInference.Tests/GraphViewQueryTests.cs)，
包含 18 个例子，以及惰性/短路、ID 查找不扫描 Nodes、平行边、二次视图、
名称与 ID 分离、类型条件、空结果、冲突 ID 合并、索引范围、端点单次枚举、
可变过滤范围快照、零步、有环图及跨 64 位边界测试。
密集/稀疏、DAG/有环索引对独立 BFS 比较全部所选节点对。
既有优化器测试仍用 BFS 验证删除依赖边前后的可达关系，不与新索引共享 oracle。
最终运行：原有 66 项图相关测试（旧匹配用例已迁移），加新增 38 项查询、
扩展组合、旧入口删除、隐式 ID 转换与边界用例，
共 104 项通过、0 失败、0 跳过；测试构建覆盖图消费者相关项目。

图引擎声明和本次修改的消费者入口均包含 XML 文档，覆盖类型、属性、
构造函数、方法与运算符，并描述参数、泛型参数、返回值和关键异常。
参数说明明确区分节点 ID 与名称、可见范围、端点范围、惰性查询及快照语义。
图引擎可通过以下命令严格检查文档编译：

```powershell
dotnet build .\src\SharpInference.Graphs.Core\SharpInference.Graphs.Core.csproj --no-restore -c Release -p:GenerateDocumentationFile=true -warnaserror
```

### 性能比较

可复现脚本：[GraphQueryBenchmark.cs](../scripts/GraphQueryBenchmark.cs)。
运行：

```powershell
$env:DOTNET_TieredCompilation = '0'
dotnet run --file .\scripts\GraphQueryBenchmark.cs -c Release
```

基准使用实际 ExecutionGraphBuilder/领域节点构建的合成执行 DAG，
每张图有同数量资源和操作，含串行与跨步依赖；不是实际模型文件基准。
旧 DAG 计算为迁移前位集合算法的基准参考实现，不是生产兼容 API。
当前脚本使用直接邻接计数验证新模式结果，不再依赖已删除的旧匹配器。
每项预热 3 次，再取 7 次重复样本的中位数；时间包含指定阶段，查询 ID 预先生成以排除
领域 GraphId 字符串生成对比较的影响。普通计时有系统/GC 噪声，不设 CI 时间阈值。

以下为旧匹配器删除前保留的历史测量结果（毫秒）。
旧模式的测量行不能再通过当前脚本复现；其余场景可运行脚本重新测量。

| 比较 | 128 个操作 | 1024 个操作 |
|---|---|---|
| 旧 DAG 索引构建 | 0.628 | 5.433 |
| 新可达索引构建 | 0.806 | 5.051 |
| 旧索引 2000 次查询 | 0.157 | 0.225 |
| 新索引 2000 次查询 | 0.271 | 0.315 |
| BFS 2000 次查询 | 102.333 | 724.227 |
| 新构建加 2000 次查询 | 1.068 | 11.448 |
| 旧模式完整枚举 100 次 | 43.601 | 217.522 |
| 新显式 ID 模式完整枚举 100 次 | 0.244 | 0.245 |
| 直接 Readers 1000 次 | 1.532 | 1.085 |
| Match 再取读者/去重 1000 次 | 3.443 | 2.010 |

结果不能推广为所有模式查询的加速比例：新模式显式绑定起点 ID，
旧模式用委托过滤；对比体现的是新入口利用 ID/邻接索引避免扫描的收益。
已知终点但未知起点、复杂 Where 仍可能扫描节点。

1024 个操作时，新索引一次构建分配约 2.23 MB，旧参考约 1.68 MB；
实际最终关系位数据为 128 KiB，构建分配还包含冻结图、字典、排序等临时对象。
两种索引的 2000 次 ID 查询均无线程内分配。
新匹配完整枚举 100 次约分配 108 KB，旧模式约 130.9 MB。
Match 读者查询比直接邻接读者查询多分配，因此消费者保留原直接查询。

单次直接相邻节点的 BFS 约 0.001 ms，而创建端点索引连同冻结视图
约 0.59/2.48 ms，说明索引不适合替代所有单次查询。
本次交付不承诺所有新接口比专用旧接口快，而是统一能力并保持专用低成本路径。
