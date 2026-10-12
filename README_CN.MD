# SharpInference

[English](./README.md) | 简体中文

SharpInference 是一个实验性的 .NET 推理编译器，提供 CPU 与 Direct3D 12 VM
后端。项目仍在持续开发中。

## 当前架构

```text
模型张量目录 + 模型图提供器
  -> LogicalGraphDefinition
  -> ExecutionGraphProgram
  -> VmExecutionGraphEmitter
  -> VmProgram
  -> CpuVmCompiler / D3D12VmCompiler
  -> CPU 程序集 / GPU DXIL
```

图层统一使用强类型通用图库：

- `Graph<TNode, TEdge>` 是唯一结构来源；
- Resource、Operation 和 GraphCall 是节点；
- Dependency、Read 和 Write 是边；
- 逻辑图与执行图共享遍历、View、模式匹配和批量重写能力。

旧图与旧运行时执行路径已经删除，不保留兼容执行路径，也不会静默回退。

## 当前验证范围

- 从兼容 rwkv.cpp 的 GGML 模型构建 RWKV-6、RWKV-7 图；
- CPU VM 编译与执行；
- Direct3D 12 VM shader 编译与执行；
- Phi-4 图构建以及 tokenizer/GGUF 组件。

测试包含 RWKV-7 tiny 模型在 CPU 和 D3D12 上的端到端 logits 对比。外部模型
文件不包含在仓库中。

## 环境要求

- .NET 10 SDK；
- 完整解决方案和 D3D12 项目需要 Windows；
- GPU 测试需要兼容的 Direct3D 12 适配器。

## 构建与测试

```powershell
dotnet build .\SharpInference.slnx
dotnet test .\SharpInference.slnx
```

运行外部 tiny RWKV 推理测试：

```powershell
dotnet test .\src\SharpInference.Tests\SharpInference.Tests.csproj `
  -p:RwkvTestModelDirectory="D:\path\to\rwkvcpp\tests"
```

指定目录应包含 `RwkvTinyExecutionPipelineTests` 使用的 tiny RWKV 模型和
expected-logits 文件。

## 项目结构

| 区域 | 职责 |
| --- | --- |
| `src/SharpInference.Graphs.Core` | 通用强类型图、View、算法、模式与重写 |
| `src/SharpInference.Graphs` | 逻辑图和执行图领域模型、XML、转换与重写 |
| `src/SharpInference.Models.Rwkv` | RWKV 逻辑图提供器 |
| `src/SharpInference.Models.Phi4` | Phi-4 图与模型组件 |
| `src/SharpInference.Instructions*` | VM 指令契约及 CPU/D3D12 实现 |
| `src/SharpInference.Vm` | VM 程序契约 |
| `src/SharpInference.Vm.Optimization` | 执行图到 VM 的发射与模型绑定 |
| `src/SharpInference.Backends.CpuVm` | CPU VM 编译器与执行器 |
| `src/SharpInference.Backends.D3D12Vm` | D3D12 VM 编译器与执行器 |
| `src/SharpInference.Tests` | 图、Emitter 与端到端推理测试 |
| `tools/GraphView` | 逻辑图和执行图 XML 查看器 |

## 许可证

项目使用 [MIT License](./LICENSE)。第三方依赖和资源见
[THIRD-PARTY-NOTICES.txt](./THIRD-PARTY-NOTICES.txt)。模型权重和外部数据集
仍受其各自许可证约束。
