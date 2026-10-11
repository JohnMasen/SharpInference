# 执行图驱动 VM：修改计划

状态：生产迁移已实施。LogicalGraph → IL → CPU/GPU 目标图 → VM 机械装配
已经接入生成、离线编译、运行时及 artifact。旧 ExecutionGraph/GraphOptimizer
不再参与生产规划或绑定；保留的旧类型仅服务独立数值参考/调试工具，不是兼容生产路径。

## 实施进度

| 阶段 | 实施与验收结果 | 边界与说明 |
|---|---|---|
| P0 | 已运行既有基线、真实 tiny RWKV6/7 数值与状态对照 | 大型 RWKV7 权重文件本机缺失，相关测试未验证 |
| P1 | 独立 IL 执行节点/资源、共享规划接口、正式目标模型及严格 XML 已实现 | IL 序列化复用已有结构化 portable schema，不新增重复算子词汇；现有目标为 CPU/D3D12 |
| P2 | CPU 图重载、严格指令绑定、机械装配及实际数值执行通过；兼容层体归一化后去重 | 每次调用显式绑定权重、状态及 scratch；不同实现/别名模式不强行合并 |
| P3 | 连续静态 View、base 生命周期、显式 Local 规划及 CPU/GPU 实参别名/初始化校验已接入 | View/覆盖风险/不同槽绑定/偏移区间均有测试；不引入动态或 stride view |
| P4 | GPU dispatch/barrier、严格 1/64 配置、列尾部实际执行通过；目标图先拒绝缺同步、可写重叠及错误 grid | 同步按底层槽保守处理，符合现有 UAV 契约；不做额外区间同步性能优化 |
| P5 | Runtime Prepare(target)、ExecutionPlanPath、prefill/FP16/TierOne、计划 artifact 完整校验及 CLI 已接入 | 手写 VM 与独立 prefill VM 保留独立语义，不冒充图生成结果 |
| P6 | 生产规划与绑定已移除旧图依赖；IL/目标改图生效、正式查看器、导出与文档已迁移 | 旧公共类型仅保留给独立参考/调试消费者，以保存既有数值覆盖 |

已落地的生产链不再先优化一张绑定计算图，再从 LogicalGraph 生成另一套 VM。
目标图包含所有最终调用、同步、槽及 prefill 入口；装配阶段不执行图优化。
`ExecutionPlanAssembler.Bind` 直接按目标图查询指令契约，不再先构造整个 VmProgram。
生成的 artifact 带 `execution-plan.xml`，加载时比较完整装配程序，而不只比较 IO/状态 ABI。
IL 的 `WithNodes` 与保存重载均以修改后的节点为准，删除节点不从原逻辑图补回。
重复层把实际槽位归一化为形式参数，仅对完整契约、调用几何、同步及别名模式相同的
层体复用 Definition；调用处显式映射实际槽位，装配器不猜测共享。

最终验证：解决方案构建 0 警告、0 错误；合并后的图/VM/精度/视图/TierOne/runtime/
模型 lowering/依赖方向/CPU 发布/Phi4 文本视觉音频回归 509 项全部通过。
另外 GPU MatVec、Local 分配、精度与目标图契约组合 127 项通过，CPU/GPU 离线运行时、
CPU 发布及容量契约组合 58 项通过（各组有重叠，不合计为独立测试总数）。
缺少模型指令注册时保留标准 BackendPreparationException 及逐节点诊断。
真实 CPU/D3D12 目标图已重新生成并成功离线编译为 DLL/PDB 与 DXIL 包。
浏览器已验证正式 IL/CPU/GPU、CPU 层定义、GPU Kernel 和原逻辑/实验图；
没有使用结构检查代替实际数值执行测试。
当前 tiny RWKV7 FP16 正式 CPU/GPU 图均有 14 次顶层调用，归一化后共享
5 个 region 模板，保留具体槽绑定与源映射；两种目标各有 635 个静态资源视图映射。
这不是性能测量，也不代表 635 次动态 CreateView；当前没有动态视图指令。

正式产物可以通过以下命令重新生成：

```powershell
dotnet run --project .\src\SharpInference.Vm.Tool -- plan cpu weights.bin target.cpu.xml
dotnet run --project .\src\SharpInference.Vm.Tool -- plan d3d12 weights.bin target.gpu.xml
dotnet run --project .\src\SharpInference.Vm.Tool -- compile-plan target.cpu.xml cpu-artifact
dotnet run --project .\src\SharpInference.Vm.Tool -- compile-plan target.gpu.xml gpu-artifact
```

下面保留原定阶段工作与验收标准。缺失大型模型的环境项明确列为未验证，不以 tiny
模型结果替代；跨 host、动态视图及新增性能算法仍按原定范围排除。

## 1. 目标与已确定约束

唯一编译链路：

```text
LogicalGraph
  → IL ExecutionPlan
  → CPU / GPU ExecutionPlan
  → VmProgram（机械装配）
  → 现有 CPU / D3D12 编译器
  → VM 实例化与执行
```

执行图是最终执行方案的权威来源。VmProgram 保留为编译 IR，不再承担图优化或资源规划。

- 放弃旧 ExecutionGraph 模型和 XML 的兼容要求；迁移仓库内消费者，不保留生产双路径。
- 公共模型保持模型无关、后端可扩展；CPU/GPU 配置是目标扩展。
- 沿用现有 VM 的数据槽位、内部形式参数/实参绑定和统一资源管理器。
- Global/Session 由外部调用者绑定、初始化并管理生命周期；Local 按图落实。
- 调用者将方法参数写入 Inputs，将执行完成后的 Outputs 封装为返回值。
- 调用者负责跨调用并发隔离及异步完成后的回收；VM 不增加线程安全机制。
- 不允许隐式类型转换。输入输出类型、计算精度和累加精度必须满足图中明确要求。
- backend 按图构造算子；无法实现时报告具体错误，不自动改算法、精度或线程配置。
- View 是连续区间的零拷贝地址映射，不隐藏复制、转换或重排。
- 所有槽位共享/复用必须由图明确；装配器不能基于相同名字或形状自动共享。
- 当前先采用精确类型、shape 和逻辑长度；算子从参数描述获取形状，不访问底层 padding。
- 最终装配/编译不得隐式融合、添加入口、改同步或重新分配逻辑槽位。

首个可运行里程碑是静态形状、单 host、RWKV7 单 token。跨 host、动态 CreateView、
通用 stride view、跨线程组融合和新性能算法不在首期范围内。
现有 prefill、多入口、FP16 权重、原生模型指令等能力需要迁移，不因首个里程碑范围较小而永久删除。

## 2. 当前基线与主要修改位置

实施前生产链路的问题：

- `VmGraphBackend.Prepare` 产生绑定用 ExecutionGraph，却将 LogicalGraph 交给 VM 生成器。
- `VmExecutionGraphGenerator`/`VmGraphOptimizer` 再进行执行图生成、指令选择、权重视图消除、
  Local 规划、GPU 配置与同步生成，并隐式生成 prefill 入口。
- 现有绑定校验不证明计算过程与外层图相同。
- 实验 XML 已有多层定义、调用、视图、复用槽和 GPU 同步，但没有正式 C# 模型和装配器。

重要事实：生产优化器默认已有 WeightViews 消除；原逻辑图的 reshape 数量不能代表最终生产 VM 的复制量。

| 模块 | 修改职责 |
|---|---|
| [Graphs](../src/SharpInference.Graphs) | 新计划模型、XML、校验、逻辑图到 IL、源映射；迁移旧执行图公共接口 |
| [Vm.Optimization](../src/SharpInference.Vm.Optimization) | 目标 lowering/优化、槽规划、指令绑定，以及到现有 VmProgram 的机械装配 |
| [Instructions](../src/SharpInference.Instructions) | 严格装配签名与执行配置；后端实现能力声明 |
| [CPU Instructions](../src/SharpInference.Instructions.Cpu) / [GPU Instructions](../src/SharpInference.Instructions.D3D12) | 按目标要求构造实现，无隐式降级 |
| [VM](../src/SharpInference.Vm) | 沿用槽位和调用结构，补必要校验/可追溯元数据，不新增图优化 |
| [Runtime](../src/SharpInference.Runtime) / [CPU adapter](../src/SharpInference.Runtime.Cpu) / [D3D12 adapter](../src/SharpInference.Runtime.D3D12) | 唯一计划来源、绑定、工厂和 artifact 接入 |
| [CPU compiler](../src/SharpInference.Backends.CpuVm) / [D3D12 compiler](../src/SharpInference.Backends.D3D12Vm) | 忠实代码生成与计划一致性检查 |
| [VM tool](../src/SharpInference.Vm.Tool) | 离线生成/读取目标计划并编译；迁移现有命令 |
| [Tests](../src/SharpInference.Tests) | 契约、结构、数值、运行时与发布回归 |
| [GraphView](../tools/GraphView) / [实验脚本](../scripts/rwkv7_execution_plan_experiment.py) | 跟进正式 schema 和可追溯展示 |

优先使用现有项目，不新增程序集。实现类名在对应步骤定稿，避免为了计划先引入空接口或重复模型。
Graphs 不依赖 VM 或具体 backend；目标配置按现有依赖方向引入，不让公共层引用 CPU/D3D12 实现。

## 3. 正式模型需要定稿的内容

### 3.1 公共结构

- 计划身份、阶段、目标、schema 版本、来源身份。
- 入口、Inputs/Outputs、State schema/版本及状态项。
- Definition 的形式槽位、Call 的实参绑定和层级 Region。
- 算子、属性、资源访问、顺序/依赖与逻辑来源。
- 默认契约及显式例外；形状统一采用 `64,64`。

Region 只负责组织与必要的顺序语义；Definition 是可调用结构。
不将视觉分组隐式解释为函数、Kernel 或分配域。

### 3.2 资源、槽位与调用绑定

- 区分逻辑资源、规划存储槽与外部实际 backing。
- 定义内 scratch 使用形式槽位，外层调用明确绑定实际规划槽；
  可用现有 Ports/Bindings 体系表达，不另建平行参数系统。
- Definition 内的复用映射只决定该定义资源到形式 scratch 槽的关系；
  不足以决定不同调用实例是否共享 backing。
- 用外层具体槽位引用明确共享或独立，不根据局部 `s3` 名字推断。
- VM 装配按显式映射产生 Slots，并将绑定传递到内部调用。
- View 描述 base、byteOffset、类型、shape、访问权限；解析到精确连续区间。
- View 不产生独立分配或执行指令，其所有访问延长 base 的生命周期。
- 不默认支持 in-place；只读别名可共享，写入重叠须有明确合法契约。

### 3.3 指令与 GPU 配置

- 目标图指定可解析的算子身份、类型签名、计算/累加精度及执行配置。
- 指令库可按要求自动构造实现；签名检查不允许偷偷插入转换。
- `cooperativeThreads=1` 表示不做工作单元内协作，不是整个 dispatch 只有一个线程。
- MatVec 明确以输出行为协作单位；首期合法值严格为 `{1,64}`，具体规则见下节。
- 线程数、Kernel threads、dispatch grid 必须匹配；缺少实现时明确拒绝，不回退到默认配置。
- 正式化 `linear-output` 和同索引 Body 顺序，验证边界保护与安全融合。
- 图级同步显式表示；组内同步属于已选择实现的内部契约或目标图内部同步节点。
- 融合阶段负责新增/删除同步并证明合法性；最终代码生成不修补非法融合。

稳定的指令签名可以引用，不必在每个 Operation 重复展开全部约束。
不要求序列化每条机器指令，但影响调度、资源、数值要求的选择必须可追溯。

### 3.4 cooperativeThreads 的合法值与范围

该字段首期用于 GPU `core.mat-vec`，不是任意算子的通用并行度或 CPU 线程池线程数。
序列化为正整数。当前实现支持的是离散集合，不是连续范围：

| 值 | 含义 | 映射到现有实现 | Kernel/dispatch 要求 |
|---|---|---|---|
| `1` | 每个有效线程独立计算一个输出行，行内串行累加 | `serial-matvec` | 使用目标图声明的合法线程组；按线性输出索引覆盖所有行 |
| `64` | 每个线程组的 64 个线程协作计算一个输出行 | `cooperative-matvec` | Kernel threads 必须是 `64,1,1`；grid 按现有 `GpuMatVecExecution.Groups(rows)` 规则覆盖输出行 |

明确的接受/拒绝规则：

- 当前支持集合为 `{1,64}`；最小支持值为 1，最大支持值为 64。
  “1 到 64”只描述最小/最大值，不意味着中间整数合法。
- `0`、负数、非整数，以及 `2..63` 和所有大于 64 的值均拒绝。
  `32`、`96`、`128`、`256`、`512`、`1024` 即使满足部分硬件限制，也不受当前实现支持。
- GPU 线程组的硬件上限是另一项校验，不能作为支持集合的推导依据。
- 64 线程是固定协作规模；矩阵列数不要求是 64 的倍数。
  实现按 lane 步长访问列，因此列数不足 64 或不是其倍数也可以使用该配置，
  但仍须满足算子签名、非空形状和 GPU 索引/grid 限制。
- 同一个 Kernel 若包含协作 MatVec，沿用现有“隔离的单算子 Kernel”限制；
  首期不允许把其组内同步与任意其他 Body 步骤混合。
- 目标图必须显式填写该值，或引用明确含该值的配置；缺失时报错，
  不在装配阶段默认选 1 或 64。
- `1`/`64` 的映射由目标 lowering 写入图。现有优化器默认选协作方案的行为可以通过
  写出 `64` 保留，而不是改为隐式默认或静默降为 `1`。
- CPU MatVec 和其他算子首期不接受该 GPU 字段；未知/不适用配置明确报错。
  将来其他算子使用协作配置时分别声明支持集合，不直接套用 MatVec 规则。

实现依据：

- [GpuMatVecExecution](../src/SharpInference.Instructions/GpuMatVecExecution.cs) 固定 `Threads=64`。
- [GpuTierZeroInstruction](../src/SharpInference.Instructions.D3D12/GpuTierZeroInstruction.cs)
  当前协作实现使用 64 项共享数组、步长 64 和从 32 开始的归约。
- [VmGpuLaunchContracts](../src/SharpInference.Vm/VmGpuLaunchContracts.cs)
  校验隔离协作 MatVec、固定线程组和对应 grid。

未来扩展到例如 `{1,32,64,128}` 时，必须先实现对应代码生成、归约/共享内存、
launch 校验、能力声明和数值/硬件测试，再扩展允许集合。
“必须为 32 的倍数”可以成为未来的额外限制，但不能替代具体支持集合。

## 4. 分步实施

### P0：建立迁移基线与能力清单

工作：

1. 列出旧 ExecutionGraph、生成器、VM 优化器的实际消费者和入口。
2. 记录单 token、prefill、多入口、FP32/FP16 权重、模型自定义指令、typed workspace、
   CPU/GPU 配置、调优及 artifact 发布的现有行为与测试。
3. 运行相关最小测试，记录已有失败、GPU/模型可用性和数值基线。
4. 保存已有实验 XML/统计用于结构对照，不作为生产数值正确性的证据。

验收：有明确的迁移清单；已有失败与新增回归可区分；不修改外部模型数据。

### P1：正式模型、XML 与校验

工作：

1. 在公共层增加新计划模型，先支持静态资源、串行顺序、Definition/Call。
2. 定稿 IL/目标图各自允许的节点和配置，禁止未识别字段静默影响执行。
3. 实现 XML 读取/写出、默认值解析及版本校验。
4. 实现作用域、引用、访问、状态、调用非递归、顺序、形状和 View 校验。
5. 将实验 CPU/GPU scratch 改成明确的外层槽绑定，迁移三套样例和查看器。

验收：

- XML 重载语义一致，规范化指纹稳定，shape 全部逗号分隔。
- 同名局部槽不会隐式共享；显式绑定同一具体槽可表达合法复用。
- 未绑定 scratch、视图循环、越界、类型冲突和非法入口均报带位置的错误。

依赖：P0。

### P2：逻辑图到 IL 与最小 CPU 闭环

工作：

1. 实现模型无关的逻辑图到 IL 转换，保留自定义算子 lowering 和模型注册扩展。
2. 保留状态写入顺序、Definition/Call、算子属性、精度和源映射。
3. 生成不融合、不消除 reshape、不复用临时存储的 CPU 目标图；
   每个 scratch 仍有显式绑定，避免借优化掩盖装配缺陷。
4. 实现目标图到现有 VmProgram 的机械装配，明确入口和状态映射。
5. 使用现有 CPU 编译器执行小型合成图，再验证 RWKV7 单 token。

验收：

- IL/CPU XML 可以独立重载；装配器不接收 LogicalGraph。
- 保存目标图后不依赖模型图生成器仍可构造 VM（执行仍需资源与指令库）。
- 算子、调用顺序、状态项、入口和槽位与图逐项对应。
- 不支持的类型/内部精度直接失败，无隐式转换或成功形状的降级。

依赖：P1。

### P3：静态 View 与显式 Local 复用

工作：

1. 将合法 reshape 降为连续 View；保留有写入/覆盖风险的物化操作。
2. 规范化视图链和偏移，保留原始 IL 节点来源。
3. 以目标执行单元及实际 base 区间计算生命周期。
4. 生成定义内形式 scratch 映射及调用处具体槽绑定；
   第一版保守规划，未明确复用的调用不共享。
5. 校验借用资源、输入输出重叠、初始化顺序和可写区间。

验收：

- 静态 View 无独立 Slot、copy、dispatch 或 barrier。
- base 在最后一个视图消费者之前不能被覆盖/复用。
- 调用处共享/独立槽位完全按照图，VM 装配不改变该选择。
- 同 backend、相同数值实现的 View/复用开关对照结果完全一致。
- 为越界、视图写入、跨调用复用冲突增加负例。

依赖：P2。

### P4：GPU 闭环与严格执行配置

工作：

1. 将目标 GPU Unit、Invoke、threads、groups 和 Barrier 转换为现有 VM 结构。
2. 先实现独立 Kernel + 保守同步，再加入现有同索引融合。
3. 将现有 serial/cooperative MatVec 映射为正式配置；
   `cooperativeThreads` 首期严格只接受 `{1,64}`，按 3.4 节执行。
4. 校验 dispatch 覆盖、GPU ABI 容量、参数类型/精度和别名。
5. 同步检查按 base/区间判断，并覆盖复用槽 hazard。
6. 保持外部调用者的完成等待和回收职责，VM 不新增并发管理。

验收：

- 不匹配的线程配置/grid 拒绝；编译器不重新选 cooperative/serial。
- 对 `1`/`64` 增加合法装配和执行测试；对 `0`、`2`、`32`、`63`、`65`、
  `96`、`128`、`1024`、负数、非整数及缺失配置增加拒绝测试。
- 协作实现覆盖列数小于 64、等于 64 和不是 64 倍数的数值用例；
  线程组不为 `64,1,1` 或 grid 不匹配时必须拒绝。
- GPU 图与 VM 的 Kernel、dispatch、Barrier 顺序和绑定一致。
- 跨索引不安全融合拒绝，不通过代码生成临时加同步“修好”。
- CPU/GPU 数值与状态满足 P0/现有测试规定的误差阈值；
  所选算法不同不要求跨 backend 位级相等。

依赖：P3。

### P5：现有能力迁移与生产接入

工作：

1. 将 WeightViews、native-half、TierOne 选择/融合、GPU MatVec 调优及 Local 规划
   从 VM 生成阶段迁到目标图构建/优化阶段。
2. prefill 生成迁到图阶段，入口及 token 偏移在图中明确；
   最终装配阶段不得根据模型属性自动添加入口。
3. 迁移模型自定义指令、多入口、typed workspace、状态/通用 tensor session。
4. `VmGraphBackend.Prepare` 使用唯一目标图构建程序与绑定描述，
   不再独立优化一张绑定图。
5. 迁移 CPU/D3D12 工厂、预生成程序、artifact 导入/导出和 VM tool。
6. 保留手写 VmProgram 的独立使用能力；若将它绑定模型图，
   ABI 检查不能冒充图到 VM 的计算一致性证明。

验收：

- 生产 VM 生成链没有从 LogicalGraph 出发的第二次隐式重规划。
- 目标配置/优化报告从目标图产生，不从另一条 VM 路径推断。
- 既有支持能力完成迁移测试；未迁移能力不得无声丢失或默认降级。
- 外部 IO/状态契约、snapshot 恢复、取消/失败处理及 artifact 执行保持。

依赖：P4。

### P6：移除旧链路并完成交付

工作：

1. 删除旧 ExecutionGraph 生产模型/入口及隐式 VM 生成路径，迁移剩余引用。
2. 迁移调试、查看器、CLI、发布与文档，移除“仅实验 schema”的旧说明。
3. 运行相关回归及解决方案构建，核查实际调用路径。
4. 用“修改目标图后重新编译”的实验验证唯一权威来源。

验收：

- 手动改变合法的具体槽绑定或 MatVec 配置后，VM 对应字段确实变化。
- 删除图中执行步骤会改变 VM；非法图必须被拒绝，而不是从逻辑图补回。
- 图未声明的入口/融合/复用/同步不会出现在 VM 中。
- CPU/GPU 和可用模型验证实际通过；未运行环境项目明确列出，不算通过。

依赖：P5。

## 5. 验证策略与执行命令

优先运行每阶段最小测试集合；只在生产接入/删除旧模型后扩大回归。
新增测试沿用现有 xUnit 项目，测试类名称在实现时确定。

现有重点回归：

- 图：GraphXml、GraphStructure、GraphRegionArchitecture、GraphTensorView、GraphResourceAllocator。
- VM：VmProgram、VmPrecision、VmIndexBound、VmExecutionConfiguration、VmGraphBindingValidator。
- CPU：CpuVmExecution、CpuVmMixedWeights、CpuVmPrefill、CpuVmPublication。
- GPU：D3D12Vm、D3D12VmResourcePool，执行配置及真实硬件对照。
- runtime：VmTensorGraphSession、PortableGraphState、模型原生指令及模型图 provider。

命令示例（过滤条件随阶段缩小/扩展）：

```powershell
dotnet test .\src\SharpInference.Tests\SharpInference.Tests.csproj `
  --filter "FullyQualifiedName~VmProgramTests|FullyQualifiedName~GraphTensorViewTests"

dotnet build .\SharpInference.slnx
git --no-pager diff --check
```

数值验收在 P0 固定，不在实现后为通过测试放宽阈值：

- 同 backend、相同实现，仅改变 View/槽绑定：确定性样例要求完全相等。
- 不同目标算法：使用已确认的逐元素绝对/相对阈值，并校验所有状态，不只看最终 token。
- 相同 backend 的状态 snapshot 恢复：沿用现有严格检查。
- 缺少模型或硬件的检查记为未执行，不用结构测试代替真实执行结论。

不以 dispatch 数或规划容量变化宣称性能提升；性能测量与正确性验收分开。

## 6. 风险控制与阶段交付

- 大范围模型替换：阶段内可暂留开发桥接，最终生产仅一条链路，不维护旧格式兼容层。
- 指令配置失配：目标图定稿前严格装配检查；错误包含入口/定义/节点/端口位置。
- scratch 泄漏或错误共享：先不跨调用复用，再加入显式共享，分别验证。
- GPU 同步混淆：区分 dispatch 可见性、组内算法同步和外部执行完成。
- 模型无关性退化：共享模块不引入 RWKV 或特定模型分支，用自定义合成模型验证。
- 既有工作区修改：保留实验脚本、XML 和查看器已有修改，只迁移直接相关内容。

每阶段交付：实现、定向测试结果、合法 XML 示例、相关文档及实际完成范围。
按 P0 → P1 → P2 → P3 → P4 → P5 → P6 推进；未通过阶段验收不提前删除其生产依赖。

完成标准：离线目标图能够独立驱动现有 VM 编译与执行，最终程序可追溯且不再依赖另一张
隐藏执行图。动态视图、跨 host 和新增算法在该闭环稳定后另行规划。
