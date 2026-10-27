# 解决方案项目依赖图

以当前 `SharpInference.slnx` 中各项目的 `ProjectReference` 为依据。
共 24 个项目、72 条直接项目引用，无循环引用。

- 箭头 `A → B` 表示 A 引用 B。
- 节点省略统一的 `SharpInference.` 前缀。
- 保留所有直接引用，不删除与传递依赖重复的边。
- 不包括 NuGet 包、运行时调用，以及未加入解决方案的目录。

## 一、生产项目

21 个项目、53 条直接引用。保留项目分类，使用直线/折线连线。

```mermaid
%%{init: {"theme": "base", "themeVariables": {"background": "#111827", "primaryColor": "#334155", "primaryTextColor": "#f8fafc", "primaryBorderColor": "#94a3b8", "lineColor": "#e2e8f0", "textColor": "#f8fafc", "clusterBkg": "#1e293b", "clusterBorder": "#64748b", "titleColor": "#f8fafc", "edgeLabelBackground": "#111827"}, "flowchart": {"curve": "linear"}}}%%
flowchart TB
    subgraph Models[模型定义]
        MP[Models.Phi4]
        MR[Models.Rwkv]
        MG[Models.Ggml]
    end

    subgraph Compilation[优化与 VM 后端]
        OPT[Vm.Optimization]
        CPUVM[Backends.CpuVm]
        GPUVM[Backends.D3D12Vm]
        VM[Vm]
    end

    subgraph Instructions[指令体系]
        IC[Instructions.Cpu]
        ID[Instructions.D3D12]
        TZ[Instructions.TierZero]
        INS[Instructions]
        CPU[Backends.Cpu]
    end

    subgraph Graphs[图体系]
        LOG[Graphs.Logical]
        EXE[Graphs.Execution]
        G[Graphs]
        GC[Graphs.Core]
    end

    subgraph Support[其他基础项目]
        ABS[Abstractions]
        CORE[Core]
        GGUF[Gguf]
        GEN[Generation]
        SC[ShaderCompiler]
    end

    MP --> ABS & G & INS & GGUF & LOG
    MR --> ABS & G & INS & GEN & MG & LOG
    MG --> ABS

    OPT --> ABS & VM & G & INS & EXE & LOG
    CPUVM --> VM & INS & G & EXE
    GPUVM --> VM & G & INS & EXE
    VM --> G & INS & EXE

    IC --> INS & TZ & VM & CPU & G & EXE
    ID --> INS & TZ & VM & G & EXE
    TZ --> INS & G & EXE
    INS --> G & EXE
    CPU --> G

    LOG --> G & GC
    EXE --> G & GC
    G --> GC

    CORE --> ABS
    GGUF --> G

    style Models fill:#1e293b,stroke:#94a3b8,color:#f8fafc
    style Compilation fill:#1e293b,stroke:#94a3b8,color:#f8fafc
    style Instructions fill:#1e293b,stroke:#94a3b8,color:#f8fafc
    style Graphs fill:#1e293b,stroke:#94a3b8,color:#f8fafc
    style Support fill:#1e293b,stroke:#94a3b8,color:#f8fafc
    linkStyle default stroke:#e2e8f0,stroke-width:2px

    classDef foundation fill:#e8f5e9,stroke:#388e3c,color:#111
    classDef stage fill:#e3f2fd,stroke:#1976d2,color:#111
    classDef consumer fill:#fff3e0,stroke:#f57c00,color:#111
    class GC,G,ABS,GEN foundation
    class LOG,EXE stage
    class MP,MR,MG,OPT,CPUVM,GPUVM,VM,IC,ID,TZ,INS,CPU,CORE,GGUF,SC consumer
```

`ShaderCompiler` 当前没有声明项目引用，也没有被其他解决方案项目通过
`ProjectReference` 引用，因此在图中没有连线。

## 二、测试项目

3 个测试项目、19 条直接引用。以下是完整测试项目图，不是局部示例。
采用固定布局，每条引用都是单根水平直线箭头，没有曲线和折点；
目标按图体系、指令体系、优化与 VM 后端、模型与文件格式分类。
不同测试区域内出现的同名目标表示同一个项目。

![完整测试项目依赖图](images/project-test-dependencies.svg)

[打开矢量原图](images/project-test-dependencies.svg)

## 三、依赖边界

- Graphs 只引用 Graphs.Core。
- Graphs.Logical、Graphs.Execution 均只引用 Graphs、Graphs.Core，彼此不引用。
- Instructions 引用 Graphs、Graphs.Execution，不引用 Graphs.Logical。
- Models.Phi4、Models.Rwkv 直接引用 Graphs.Logical；
  同时它们引用 Instructions，因此会传递依赖 Graphs.Execution。
- Graphs.Tests 只引用图体系，不引用 Instructions、优化器、模型或 VM。
