# TileLang 从入门到专家：一套面向面试的完整教程

> 从「TileLang 小白」到「能写高性能内核、能应对面试」的 13 章系统课程。
> 全部内容基于 TileLang 官方文档与官方 examples（`tile-ai/tilelang`），API 以官方 repo 为准。

## 这套教程是什么

TileLang 是一个用于编写高性能 GPU 内核的领域特定语言（DSL）。它站在 TVM TIR 之上，
提供三档抽象：

- **L1（最省心）**：只写计算逻辑，调度交给编译器；
- **L2（推荐，多数生产内核用这档）**：控制 tiling / 共享内存 / 线程块，但布局推断、
  软件流水线由编译器完成——类似 Triton 的编程模型，但更贴近硬件；
- **L3（专家）**：直接操控线程、PTX 内联汇编、warp 原语，几乎等价于手写 CUDA/HIP。

它被 DeepSeek 等团队用于生产环境（FlashMLA、DeepSeek 系列推理内核），性能可与
cuBLAS、vendor 库持平或反超，同时具备 Triton 所不具备的精细控制能力。
**这也正是面试官看重它的原因**：会 TileLang 的人，通常同时懂 GPU 架构、内核优化
和 AI 编译器，是"全栈"内核工程师。

## 课程结构（13 章）

| 章节 | 主题 | 读完收获 | 难度 |
|---|---|---|---|
| [01](01-环境搭建与第一个内核.md) | 环境搭建与第一个内核 | 装好环境，跑通 vector add | ★☆☆ |
| [02](02-核心语法.md) | 核心语法：prim_func、Kernel、循环 | 能读能写任何 TileLang 内核骨架 | ★☆☆ |
| [03](03-内存层次与数据搬运.md) | 内存层次与数据搬运 | 理解 global/shared/fragment 与 T.copy | ★★☆ |
| [04](04-软件流水线与控制流.md) | 软件流水线与控制流 | 理解 num_stages、async copy、stage/order | ★★★ |
| [05](05-GEMM实战.md) | GEMM 实战：从朴素到接近峰值 | 完整走通 GEMM 优化全流程 | ★★★ |
| [06](06-FlashAttention手写实现.md) | FlashAttention 手写实现 | 在线 softmax 推导 + 完整实现 | ★★★★ |
| [07](07-高级指令与布局.md) | 高级指令与布局：mma、TMA、warp 原语 | 银行冲突、swizzle、底层指令 | ★★★★ |
| [08](08-自动调优与工程化实践.md) | 自动调优与工程化 | autotune、profiler、缓存、多后端 | ★★★ |
| [09](09-编译流程底层与调试工具.md) | 编译流程底层与调试工具 | 看懂编译流水，会系统排查 bug | ★★★★ |
| [10](10-面试冲刺题库与参考答案.md) | **面试冲刺：高频考点与参考答案** | 概念/设计/手撕题全覆盖 | ★★★★★ |
| [11](11-练习题库与答案解析.md) | 练习题库与答案解析 | 动手巩固 + 模拟面试 | — |
| [12](12-Cheatsheet速查表.md) | Cheatsheet 速查表 | 面试前 30 分钟速览 | — |

## 三条学习路线

**路线 A：2 周速成（应付面试最低配）**
01 → 02 → 03 → 05 → 06 → 10 → 12。每天 2-3 小时，重点是把 GEMM 和
FlashAttention 的代码亲手敲一遍，再背熟第 10 章概念题。

**路线 B：6 周系统课（推荐，真正变专家）**
01 → 12 顺序学习，每章完成对应练习（第 11 章），第 5 周开始每天做一道第 10 章
综合题，第 6 周做模拟面试 + 用 Nsight Compute 分析自己内核的性能。

**路线 C：专家路线（面向内核岗/编译岗面试）**
在路线 B 基础上，精读官方 repo 的 `src/` 与 `docs/compiler_internals/`，
把第 07、09 章的 pass 级内容吃透，并自己动手给官方 examples 里的 GEMM 加一个
epilogue（如 GELU、分组统计），提交 PR 或写博客。

## 学习建议（老鸟心得）

1. **代码必须亲手敲，不许复制粘贴**。面试手撕题靠的是肌肉记忆。
2. **每个示例都跑 profiler**：`kernel.get_profiler().do_bench()`，用 TFLOPS 与自己
   手写的 CUDA 版本对比，理解"为什么 TileLang 快"。
3. **对照生成代码学**：`kernel.get_kernel_source()` 把 TileLang 一行代码展开成什么
   CUDA 代码，这是理解抽象层的最佳途径，也是面试亮点。
4. **建立心智模型**：任何内核优化问题，先回答三件事——数据放哪（内存层次）、
   怎么并行（grid/block/thread）、怎么隐藏延迟（流水线/占用率）。
5. **面试前必看**：第 10 章 + 第 12 章 + 第 11 章模拟题。

## 环境要求

- Python ≥ 3.10，机器有 NVIDIA GPU（或 AMD GPU 走 ROCm / macOS 走 Metal 体验 DSL）
- `pip install tilelang`（详见第 01 章）
- 熟悉 PyTorch 基本用法（用于正确性验证与基准对比）

## 本机（macOS Apple Silicon）已安装的运行环境

本仓库已自带一个可用的虚拟环境（2026-08 实机验证，`tilelang 0.1.13` + `torch 2.13`）：

```bash
# 激活后即可使用 tilelang
source .venv/bin/activate

# 端到端验证（vector add + 分块 softmax + Metal 源码查看）
python examples/metal_verify.py
```

### macOS / Metal 后端实测结论（tilelang 0.1.13）

| 能力 | 状态 | 说明 |
|---|---|---|
| import / JIT 编译 / `get_kernel_source()` | ✅ | 自动检出 `metal` target，输出真实 Metal shader |
| `T.Parallel` 类内核（vector add、2D elementwise） | ✅ | 运行结果与 torch 一致 |
| 共享内存 + 串行归约（第 03 章 softmax） | ✅ | 需 `threads ≤ 32`（见下） |
| 性能分析 `do_bench()` | ⚠️ | CUDA event 路径不可用，改用 torch 计时 |
| `T.gemm`（Tensor Core 风格） | ❌ | 0.1.13 Metal codegen 限制（shared 指针地址空间符），需 CUDA GPU 或更新版本 |
| `T.infinity` | ❌ | 用大负数字面量代替 |
| 多 simdgroup 复制循环（threads > 32） | ⚠️ | 0.1.13 后端缺陷：串行复制循环结果错，用 `threads ≤ 32` |

要点：**本机适合学习语法与跑 elementwise/reduce/softmax 类内核**；第 05/06 章的
GEMM/FlashAttention（依赖 `T.gemm`）请在 NVIDIA GPU（本地或云）上运行，代码无需
修改——课程示例以 CUDA 为准。

## 参考资源

- 官方仓库：<https://github.com/tile-ai/tilelang>
- 官方文档：<https://tilelang.ai>
- 本课程 API 全部核对自官方 `docs/` 与 `examples/`，如与最新版有出入，以官方为准。

---

*开始吧：从 [第 01 章：环境搭建与第一个内核](01-环境搭建与第一个内核.md) 起步。*