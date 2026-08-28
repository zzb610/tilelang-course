# 资源、版本与证据说明

> 本页记录教程使用的资料、证据等级和版本策略。最近核对日期：2026-08-28。核对时的
> TileLang `main` 提交为 `8ef052f1515d463603b76741d6760d76ec0c5917`；官网显示的稳定文档版本为
> 0.1.13。后续版本可能改变 API、后端能力和工具名称。

## 资料怎么用

不同来源解决的问题不同。本教程按下面的优先级处理冲突：

1. **官方文档与当前源码**：确认 API、安装条件、target、工具入口和示例写法；
2. **论文**：解释 TileLang 的设计目标、tile 编程模型和调度空间；
3. **官方 release、discussion 与 issue**：判断版本变化、已知限制和正在修复的问题；
4. **社区教程与文章**：补充学习路径、工程取舍和真实算子案例，但不单独作为 API 依据。

社区文章中出现的性能数字只说明作者在特定环境中的测量结果。除非能复现实验条件，本教程
不会把这些数字改写成普遍结论。Issue 也不等于当前版本一定仍有同一问题；它的价值在于提醒
读者设计反例、记录版本，并检查生成代码和边界行为。

## 版本策略

- 教程主线依赖相对稳定的概念：TIR、`T.Kernel`、`T.Parallel`、`T.copy`、
  `T.Pipelined`、`T.gemm`、正确性测试和基本计时。
- `TMA`、warpgroup、Tensor Memory、cluster copy、特定 `GemmWarpPolicy`、autotuner
  持久化格式以及 pass 名称都属于版本或架构敏感内容。使用前先检查本地 API，并与当前
  官方示例对照。
- 性能数字不是课程常数。报告必须记录 GPU、驱动、CUDA/ROCm、TileLang、PyTorch、
  dtype、输入尺寸、warmup、rep 和计时 backend。
- `T.copy` 的同步可见性、边界 guard 和越界填充值是三件不同的事。任意尺寸的
  GEMM、softmax 和 Attention 必须明确 padding 值、有效区间和输出 guard，不能只依赖
  自动 safe-access。
- `auto`、`cuda`、`hip`、`metal`、`llvm`、`webgpu`、`c` 和 `cutedsl` 的能力并不
  等价。能编译同一段源码，不代表会生成同一类指令或得到相近性能。

## TileLang 官方资料

### 入门与编程模型

- [TileLang 仓库](https://github.com/tile-ai/tilelang)：源码、release、examples 和问题追踪入口。
- [Installation Guide](https://tilelang.com/get_started/Installation.html)：PyPI、源码、Docker、
  CUDA/ROCm 与开发模式安装。
- [Overview](https://tilelang.com/get_started/overview.html)：tile 是一等对象、多级内存分块和
  数据流/调度分离的整体模型。
- [Understanding Targets](https://tilelang.com/get_started/targets.html)：`auto`、CUDA、HIP、
  Metal、LLVM、WebGPU、C 和 CuTe DSL target，以及 CUDA `arch`/`code` 的写法。
- [Language Basics](https://tilelang.com/programming_guides/language_basics.html)：
  `prim_func`、`Kernel`、循环、作用域、`T.copy` 和 JIT。
- [Type System](https://tilelang.com/programming_guides/type_system.html)：字符串、TileLang dtype
  和框架 dtype 的归一化规则。
- [Control Flow](https://tilelang.com/programming_guides/control_flow.html)：条件、四类循环、
  `Persistent`、`while`、`break` 和 `continue`。
- [Python Compatibility](https://tilelang.com/programming_guides/python_compatibility.html)：哪些
  Python 语法会进入 TIR，哪些写法需要换成 TileLang 原语。

### 指令、调度与调优

- [Instruction Guide](https://tilelang.com/programming_guides/instructions.html)：内存分配、
  `T.copy`、GEMM、归约、同步和调试原语；适合与第 02、03、07 章并读。
- [Software Pipeline Guide](https://tilelang.com/programming_guides/software_pipeline.html)：
  `T.Pipelined`、`stage/order`、生产者—消费者关系和 replayable scalar Bind。
- [Autotuning Guide](https://tilelang.com/programming_guides/autotuning.html)：装饰器与编程式
  autotune、输入供应、正确性、并行编译、缓存和结果固化。
- [官方 GEMM 示例](https://github.com/tile-ai/tilelang/tree/main/examples/gemm)：基础、autotune、
  persistent 和高级调优变体。
- [官方 FlashAttention 示例](https://github.com/tile-ai/tilelang/tree/main/examples/flash_attention)：
  MHA/GQA、前向/反向、BSHD/BHSD 和 varlen 变体。
- [Grouped GEMM 示例](https://github.com/tile-ai/tilelang/tree/main/examples/grouped_gemm)：前向、
  反向和 pointer-style 实现。
- [Fused-MoE 示例](https://github.com/tile-ai/tilelang/blob/main/examples/fusedmoe/example_fusedmoe_tilelang.py)：
  token dispatch、分组元数据、grouped GEMM 和 scatter 的端到端组合。
- [完整 examples 目录](https://github.com/tile-ai/tilelang/tree/main/examples)：还包括 GEMV、卷积、
  split-K/Stream-K、稀疏 Attention、量化 GEMM 和面向特定架构的实现。

### 调试与分析工具

- [Tools Overview](https://tilelang.com/tools/index.html)：按问题选择 compile-only、Analyzer、
  layout visualization、AutoDD、IR Lower Trace 和 IKET。
- [Compile-Only](https://tilelang.com/tools/compile_only.html)：无 GPU 时把第一个 JIT kernel 或
  `PrimFunc` lower 成目标源码；它不证明设备运行正确。
- [Performance Analyzer](https://tilelang.com/tools/analyzer.html)：从 TIR 估算已识别
  `T.gemm` 的 FLOPs 和跨全局边界的 `T.copy` 字节数；不能替代真实 profiler。
- [Layout Visualization](https://tilelang.com/tools/layout_visualization.html)：查看显式或推断出的
  `T.Layout`/`T.Fragment` 线程映射。
- [IR Lower Trace](https://tilelang.com/tools/lower_trace.html)：追踪完整 lowering 流程。旧的
  Pass Diff 仍可用，但官方已经建议优先使用 IR Lower Trace。
- [AutoDD](https://tilelang.com/tools/autodd.html)：把稳定失败的程序缩成更小的复现。

## 论文、教程与讨论

- [TileLang: A Composable Tiled Programming Model for AI Systems](https://arxiv.org/abs/2504.17577)：
  论文版设计说明；重点看数据流与调度空间如何解耦，以及 layout、tensorize、pipeline
  为什么作为可定制调度维度。
- [ICLR 2026 论文页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/76fb92288bf90360c527efb0d1c2aba6-Abstract-Conference.html)：
  经过会议发表的版本和引用信息。
- [Writing High-Performance Kernels in TileLang, from GEMM to MLA](https://huggingface.co/blog/AtlasCloud-AI/writing-high-performance-kernels-in-tilelang)：
  社区工程文章。适合比较 Triton、TileLang 与 CUTLASS/CuTe 的控制粒度，并观察 GEMM、
  MLA decode 和 RMSNorm 的取舍；其中性能结果只适用于作者公开的测试环境。
- [Correct but Slow: An Empirical Study of the GPU Kernel Evaluation Gap](https://arxiv.org/abs/2607.04454)：
  第三方经验研究，说明只做正确性抽样可能放过性能很差的 DSL kernel。它用于设计“库相对
  效率”和 roofline 筛查，不作为 TileLang 当前版本整体性能的结论。
- [TileLang Discussions](https://github.com/tile-ai/tilelang/discussions)：release 说明、RFC 和
  使用交流。阅读时记录讨论日期、版本与硬件。
- [TileLang Issues](https://github.com/tile-ai/tilelang/issues)：查回归、后端限制和错误案例。
  例如，越界 tile copy 的 safe value、流水线依赖或特定线程数下的 copy 问题都说明：
  “编译成功”不能替代边界测试和生成代码检查。

## CUDA 与性能分析资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)：执行模型、
  global/device memory、缓存、共享内存、同步和架构语义。
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)：
  合并访问、共享内存、测量和优化方法。
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)：越界、
  竞态和初始化问题排查。
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/)：时间线、CPU/GPU 并发和传输重叠。
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/)：单 kernel 的吞吐、stall、内存和
  occupancy 指标。

## 按章节阅读

| 教程阶段 | 先读 | 遇到问题再读 |
|---|---|---|
| 01～02：跑通与语法 | Installation、Overview、Language Basics | Targets、Type System、Python Compatibility |
| 03～04：搬运与流水线 | Instruction Guide、Software Pipeline Guide | CUDA Programming/Best Practices |
| 05：GEMM | 官方 GEMM 示例 | Analyzer、layout visualization、split-K/Stream-K 示例 |
| 06：Attention | 官方 FlashAttention 示例 | varlen/GQA 示例、论文和社区 MLA 文章 |
| 07：布局与架构原语 | Instruction Guide、layout visualization | 对应架构示例与 CUDA Guide |
| 08：调优 | Autotuning Guide | release/discussion 中的调优变更 |
| 09：调试 | Tools Overview、IR Lower Trace | Compile-Only、AutoDD、Compute Sanitizer、Nsight |
| 13：Grouped GEMM | grouped GEMM、fused-MoE 示例 | 量化 grouped GEMM 与目标架构实现 |
| 14：Capstone | 官方 examples 中目标算子 | issue/discussion、社区工程文章与 vendor 基线 |

阅读资料不是课程产出。读完后至少要留下一个可验证对象：最小代码、边界反例、生成代码片段、
profile 记录或对照表。
