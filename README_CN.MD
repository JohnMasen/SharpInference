# SharpInference

[English](./README.MD) | 简体中文

> **AI 编写说明与项目状态：** 本项目的大部分代码和文档由 AI 编写。
> 作者正在对代码和文档进行审阅、整理与完善。项目仍在持续开发中，
> 使用前请针对自己的应用场景核实实现行为及文档准确性。

SharpInference 是基于 .NET 的本地模型推理项目，提供编译式 CPU 和
Direct3D 12 执行路径，探索将模型计算逻辑、指令实现、执行程序与资源所有权分离。

## 支持的模型与范围

| 模型 | 模型文件 | 执行路径 |
| --- | --- | --- |
| RWKV-6 / RWKV-7 | 兼容 rwkv.cpp 的 GGML x060 / x070 FP32 或 FP16 权重 | 逻辑图前端与编译式 CPU / D3D12 VM，支持预填充和逐 token 解码 |
| Phi-4-multimodal-instruct | 转换后的 F16 GGUF 组件与 tokenizer | 原生 CPU 多模态处理器、专属 D3D12 组件和 GPU 常驻解码会话 |

- RWKV 会话支持递归状态、重置、会话分叉和状态快照。
- Phi-4 支持以文本、已解码图像及已解码音频为输入生成文本。
  文件和容器格式的解码由调用方负责。
- 随附 Web API 服务于 RWKV 文本模型，不是 Phi-4 多模态接口。
- FP16 权重存储不代表 FP16 运算。精度要求是显式的；RWKV 状态与当前
  T0 运算及累加使用 FP32。
- 项目不是任意 GGUF 架构或量化格式的通用加载器。后端已经注册，
  不代表所有模型规模、输入和设备上的正确性与性能都已验证。
- 当前 VM 不支持通用程序级动态分支、循环或递归；
  固定程序执行与算子内部的局部循环是不同概念。

仓库不包含模型权重和大型外部验证数据集。

## 架构概览

RWKV 的执行链路如下：

```text
模型元信息 / 张量目录 + 架构专属 GraphProvider
  -> LogicalGraph
  -> 目标优化、ExecutionGraph 与本地槽位复用规划
  -> VmProgram：固定槽位、显式 State、定义、调用和入口
  -> 生成 C# / HLSL
  -> CPU 程序集 / GPU DXIL
  -> 可复用 VM 实例
```

模型元信息提供维度和权重描述，GraphProvider 定义模型计算语义。
优化器决定逻辑资源如何映射为固定槽位，以及哪些中间资源可以安全共用存储。

运行时，`VmResourceManager` 分配实际存储、共享全局资源并绑定会话资源。
槽位分为 `Global`、`Session` 和 `Local` 三种作用域。
`LogicalVmResourceManager` 管理组件数据引用、存储域副本与传输，以及输入绑定；
D3D12 资源池提供具体 GPU 存储。
编译期临时槽位复用与运行时权重或状态共享不是同一件事。

并非所有前端都经过相同链路：Phi-4 专属 D3D12 解码器直接构造 `VmProgram`，
不经过 RWKV 的逻辑图降低路径，其 KV cache 是显式会话资源。

后端执行传入的程序，不会静默重新规划图或回退到旧执行器。
编译、产物部署、状态、队列和资源所有权的详细约定见
[当前 VM 契约](./doc/vm-execution-program.md)。

## 环境要求与构建

- .NET 10 SDK。
- 完整解决方案、Web API 和 D3D12 路径使用 Windows：多个项目的目标框架为
  `net10.0-windows10.0.19041.0`。部分平台无关库使用 `net10.0`，
  但这不代表完整应用已经支持跨平台运行。
- GPU 执行需要兼容的 D3D12 显卡和驱动，以及足以容纳权重、状态和临时资源的内存。
- 首次构建需要能够获取 NuGet 依赖。

在仓库根目录运行以下 PowerShell 命令：

```powershell
dotnet build .\SharpInference.slnx -c Release
```

源码编译与导入预编译产物的部署要求不同，详见
[VM 部署文档](./doc/vm-execution-program.md)。

## 快速开始：RWKV Web API

准备兼容的 RWKV-6 或 RWKV-7 GGML 模型，
不要用 `.pth` 文件或任意 GGUF 文件替代。

首次运行建议明确选择 CPU 后端，并仅监听本机地址：

```powershell
dotnet run --project .\src\SharpInference.WebApi -c Release -- `
  --model-path "D:\Models\your-rwkv-model.bin" `
  --runtime-kind cpu `
  --urls "http://127.0.0.1:9841"
```

将模型路径替换为自己的文件。服务启动时验证模型文件头；
完整模型加载与程序准备在首次使用模型宿主时进行，因此首次生成可能耗时较长。

在另一个 PowerShell 终端查询服务返回的模型 ID，并请求生成：

```powershell
$baseUrl = "http://127.0.0.1:9841"
Invoke-RestMethod "$baseUrl/health"
$models = Invoke-RestMethod "$baseUrl/v1/models"
$body = @{
    model = $models.data[0].id
    messages = @(@{ role = "user"; content = "Hello!" })
    max_tokens = 32
    stream = $false
} | ConvertTo-Json -Depth 5
Invoke-RestMethod "$baseUrl/v1/chat/completions" `
    -Method Post -ContentType "application/json" -Body $body
```

API 提供 OpenAI 风格的模型与聊天补全路由，支持流式响应；
这不代表完整兼容所有 OpenAI API 功能。

### 后端与服务配置

| 参数 | 用途 |
| --- | --- |
| `--model-path` | 必填，RWKV GGML 模型路径 |
| `--runtime-kind cpu` | 编译式 CPU VM |
| `--runtime-kind d3d12` | 编译式 Direct3D 12 VM，`vortice` 为别名 |
| `--config` | 额外 JSON 配置，命令行参数优先 |
| `--urls` | 完整监听地址 |
| `--prefill-instances` / `--inference-instances` | 可复用 VM 数量 |
| `--prefill-queue-capacity` / `--inference-queue-capacity` | 有界队列容量 |
| `--api-key` | 要求请求携带 API key |

仓库中的 [appsettings.json](./src/SharpInference.WebApi/appsettings.json)
选择了 `vortice`，因此 CPU 示例需要保留 `--runtime-kind cpu`。
GPU 设备索引通过 `Rwkv:Runtime:Vortice:AdapterIndex` 配置。
完整命令行参数可通过以下命令查看：

```powershell
dotnet run --project .\src\SharpInference.WebApi -c Release -- --help
```

配置认证和适当的部署环境之前，请仅在本机使用。
仓库配置的 API key 为空且监听所有网络接口，快速开始命令覆盖了监听地址。
请求日志可能包含提示词内容。

## Phi-4 快速开始

准备转换后的模型目录，包含：

```text
Phi-4-multimodal-text-f16.gguf
Phi-4-multimodal-omni-f16.gguf
Phi-4-multimodal-vision-lora-f16.gguf
Phi-4-multimodal-speech-lora-f16.gguf
tokenizer.json
```

运行文本解码器基准：

```powershell
dotnet run --project .\tools\SharpInference.VmBenchmark -c Release -- phi4 `
  --model "D:\Models\Phi4" --tokens 16 --samples 3
```

如需 GPU 常驻文本会话，增加 `--backend d3d12-session --context 4096`。
这是基准工具入口，不是 RWKV Web API。
处理器调用示例、转换工具、D3D12 组件及外部验证见
[Phi-4 多模态推理文档](./doc/phi4-multimodal-inference.md)。

## 项目结构

| 区域 | 职责 |
| --- | --- |
| `src/SharpInference.Abstractions`、`src/SharpInference.Core` | 公共契约、模型加载、会话、分词和采样 |
| `src/SharpInference.Architectures.*` | 模型专属前端与处理器 |
| `src/SharpInference.Graphs` | 图 IR、验证、优化和资源规划 |
| `src/SharpInference.Instructions*` | 指令契约及 CPU / GPU 实现 |
| `src/SharpInference.Vm*` | VM 程序契约、资源所有权、优化、模板和工具 |
| `src/SharpInference.Backends.*Vm` | 编译式 CPU 与 D3D12 执行 |
| `src/SharpInference.Runtime` | 模型与后端集成、VM 运行时配置 |
| `src/SharpInference.WebApi` | RWKV HTTP 服务 |
| `src/SharpInference.Tests`、`src/SharpInference.Gguf.Tests` | 测试 |
| `tools/SharpInference.VmBenchmark`、`scripts`、`doc` | 基准、辅助脚本和详细文档 |

## 测试与性能验证

```powershell
dotnet test .\src\SharpInference.Tests\SharpInference.Tests.csproj -c Release
dotnet test .\src\SharpInference.Gguf.Tests\SharpInference.Gguf.Tests.csproj -c Release
```

部分测试依赖 GPU、外部模型或显式启用设置。
存在跳过测试时，测试通过不代表所有大型模型和 GPU 路径均已验证。
Phi-4 外部测试使用 `PHI4_TEST_MODEL_DIRECTORY`、`PHI4_RUN_GOLDEN_MODEL`
和 `PHI4_RUN_GOLDEN_VISION`，具体配置见其详细文档。

RWKV 耗时与数值输出基准：

```powershell
dotnet run --project .\tools\SharpInference.VmBenchmark -c Release -- run `
  --target cpu --model "D:\Models\RWKV-x060-model.bin" `
  --output ".\benchmark-results\rwkv-cpu.json"
```

该命令生成报告及 logits、状态附属文件。RWKV 基准根据模型文件名选择前端，
请保留能够识别架构的名称（RWKV-6 使用 `060` / `rwkv-6`）。
GPU 测量使用 `--target gpu`。
比较性能时必须考虑硬件、模型、精度、预热及工作负载条件；
历史数据不是性能保证。

## 详细文档

- [编译式 VM 契约与部署](./doc/vm-execution-program.md)
- [VM 算子分层与精度](./doc/vm-operator-tiers.md)
- [队列化 VM 执行](./doc/queued-vm-execution.md)
- [Phi-4 多模态推理](./doc/phi4-multimodal-inference.md)
- [GPU MatVec 性能](./doc/gpu-matvec-performance.md)
- [Tier-one 性能](./doc/tier-one-performance.md)
- [架构历史](./doc/arch-design.md)：包含历史设计；
  当前运行时行为应以最新 VM 契约和源码为准。

## 许可证

本项目采用 [MIT 许可证](./LICENSE)。
随附依赖和资源的说明见 [第三方声明](./THIRD-PARTY-NOTICES.txt)。
模型权重和外部数据集受各自许可证约束。
