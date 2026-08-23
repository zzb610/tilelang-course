# 资源与版本说明

> 本文件记录课程依赖的权威入口和版本策略。访问日期：2026-08-23。

## 版本策略

- 教程主线只依赖相对稳定的概念：TIR、`T.Kernel`、`T.Parallel`、`T.copy`、`T.Pipelined`、`T.gemm`、正确性测试和基本计时。
- `TMA`、warpgroup、Tensor Memory、特定 `GemmWarpPolicy`、autotuner 持久化格式以及 pass 名称都属于版本/架构敏感内容。使用前先运行本地 API 检查，并以当前官方示例为准。
- 性能数字不是课程常数。报告中必须记录 GPU、驱动、CUDA/ROCm、TileLang、PyTorch、dtype、输入尺寸、warmup、rep 和计时 backend。
- 课程示例优先保证“能解释、能验证”。对任意尺寸的 GEMM/Attention，不能只依赖自动 safe-access；越界 tile 的填充值、输出 guard 和 K 维 padding 必须明确设计。

## TileLang 官方资源

- [TileLang 仓库](https://github.com/tile-ai/tilelang)：安装、发行说明、源码和完整 examples。
- [Getting Started / Overview](https://github.com/tile-ai/tilelang/blob/main/docs/get_started/overview.md)：tile 编程模型和 GEMM 入门。
- [Language Basics](https://github.com/tile-ai/tilelang/blob/main/docs/programming_guides/language_basics.md)：作用域、`T.copy`、边界和基本语法。
- [Instruction Guide](https://github.com/tile-ai/tilelang/blob/main/docs/programming_guides/instructions.md)：GEMM、归约、同步和调试原语。
- [Software Pipeline Guide](https://github.com/tile-ai/tilelang/blob/main/docs/programming_guides/software_pipeline.md)：`T.Pipelined`、`stage/order` 和 replayable Bind。
- [官方 GEMM 示例](https://github.com/tile-ai/tilelang/blob/main/examples/gemm/example_gemm.py)：建议与第 05 章并排阅读。
- [官方 examples 目录](https://github.com/tile-ai/tilelang/tree/main/examples)：按 GEMM、attention、归约、量化和架构筛选实验。

## CUDA 相关资源

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)：执行模型、内存、同步和架构语义。
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)：合并访问、共享内存、测量和优化方法。
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)：越界、竞态和初始化问题排查。
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/)：时间线、CPU/GPU 并发和传输重叠。
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/)：单 kernel 的吞吐、stall、内存和 occupancy 指标。

## 推荐阅读顺序

1. TileLang Overview → Language Basics；
2. 本教程第 01～04 章；
3. 官方 GEMM 示例 → Software Pipeline Guide；
4. 本教程第 05～09 章；
5. 目标 GPU 对应的 CUDA Guide、Best Practices 和 Nsight 文档。
