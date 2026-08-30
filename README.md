# TileLang 教程：从第一个内核到性能工程

这是一部面向 GPU 内核、AI 编译器和性能工程学习者的中文 TileLang 教材。它从一个向量加法
开始，沿着“程序如何成为内核、数据为何影响性能、抽象怎样变成算子、如何解释性能差异”
这条线，最后走到一个真实算子的交付。

```text
写最小正确基线 → 做 edge-shape 测试 → 测量 → 看生成代码/Profiler
→ 一次只改一个变量 → 解释收益与代价 → 固化配置
```

代码和实验在书中承担的是证据，而不是步骤本身。我们会先提出问题、建立可推演的模型，再用
最小代码观察模型如何落到硬件；章末实验用于检验理解。性能结论必须和硬件、版本、输入及
计时方法一起记录。

课程默认以 NVIDIA CUDA GPU 为目标。没有 CUDA GPU 也可以学习语法、编译模型和
compile-only 流程。当前 TileLang 已提供 Metal、HIP、LLVM、WebGPU、C 和 CuTe DSL 等
target，Metal 也开始支持部分矩阵乘路径；但各后端的指令、算子覆盖和性能成熟度不同，
不能把某一后端的结果外推到另一后端。

## 怎样使用这本书

第一次阅读可以从第 00 章直接进入主线。下面几份材料承担不同作用，不必在开始前一次读完：

- [课程导读与实验规范](00-课程导读与实验规范.md) 是正文的入口；
- [课程大纲](OUTLINE.md) 用来查看全书结构和学习时间；
- [教材写作约定](WRITING_GUIDE.md) 供作者与贡献者维护全书风格；
- [资源与版本说明](resources.md) 在遇到 API 或后端差异时查阅。

## TileLang 是什么

TileLang 是构建在 TVM/TIR 之上的 GPU/加速器内核 DSL。它的核心设计（见
[TileLang 论文](https://arxiv.org/abs/2504.17577)）是**把数据流与调度空间解耦**：
用户用 tile 算子（`T.copy`、`T.gemm`、`T.reduce` 等）描述「算什么、数据怎么流」，
编译器自动完成线程绑定、内存布局、指令选择（tensorize）和软件流水线这四类调度；
编译器默认方案不够好时，用户再用 `T.Pipelined`、`T.annotate_layout`、`T.use_swizzle`、
`T.ptx` 等注解与原语逐个接管。整个课程就是反复练习这条「先信任推断、再用证据接管」的路线。

可以先用三层心智模型理解它：

- **高层（数据流）**：用 tile 算子表达计算和 tile 关系，不关心线程与布局；
- **中层（调度）**：用 grid、shared memory、fragment、流水线和布局注解控制「谁在何时做」；
- **底层（指令）**：必要时观察生成代码，使用 warp、PTX、TMA 或架构相关原语。

这里的「层」是学习视角，不是保证每个版本都有固定的 L1/L2/L3 API。具体接口和支持范围以当前官方文档/示例为准。
这套抽象与 Triton 的关键差别在第 02 章的「数据流算子与调度原语」表中展开。

## 课程结构

第 00～09 章是递进的主线，第 10～12 章分别承担面试表达、迁移练习和查表任务，第 13 章是
建立在 GEMM/MoE 经验之上的专项扩展，第 14 章负责真实算子交付。不要跳过前面的正确性
与测量训练直接背模板。

下表给出全书十五个章节的主题、完成标志和难度，你可以据此判断当前进度与下一步该去哪：

| 章节 | 主题 | 完成标志 | 难度 |
|---|---|---|---|
| 01 | 环境搭建与第一个内核 | 跑通 vector add，完成正确性与源码查看 | ★☆☆ |
| 02 | 核心语法 | 能读写 `prim_func`、`Kernel`、循环和作用域 | ★☆☆ |
| 03 | 内存层次与数据搬运 | 能画 global → shared/fragment → global 数据流 | ★★☆ |
| 04 | 软件流水线与控制流 | 能解释 `T.Pipelined` 的生产者/消费者关系 | ★★★ |
| 05 | GEMM 实战 | 从基线迭代到分块、流水线和调优 | ★★★ |
| 06 | FlashAttention 手写实现 | 推导 online softmax，完成小尺寸校验 | ★★★★ |
| 07 | 高级指令与布局 | 区分 bank-conflict swizzle 和 block rasterization | ★★★★ |
| 08 | 自动调优与工程化 | 固定输入、校验配置、记录和复用最佳配置 | ★★★ |
| 09 | 编译流程底层与调试工具 | 能按生成期/正确性/性能分类排障 | ★★★★ |
| 10 | 面试冲刺 | 能把实现、测量和归因讲成完整答案 | ★★★★★ |
| 11 | 练习题库与答案解析 | 用题目验证迁移，而不是只背概念 | — |
| 12 | Cheatsheet 速查表 | 查 API、公式和排错关键词 | — |
| 13 | 分组 GEMM 实战 | 能处理不规则 group、metadata 和 MoE 风格调度 | ★★★★★ |
| 14 | 收官项目：交付一个真实算子 | 完成需求、基线、边界、优化、集成和报告 | ★★★★★ |

## 三条学习路线

三条路线对应三种投入，你可以先按自己的时间选择。先用两周最小路径完成一次完整实验，
建立信心，再按六到八周系统路径补齐主线，最后沿研究/工程路径把能力收敛到一次
真实交付：

**两周最小路径**

00 → 01 → 02 → 03 → 05 → 06 → 11 中对应练习 → 10 → 12。重点是亲手完成一个 elementwise 内核和一个小尺寸 GEMM/Attention 正确性实验。

**六到八周系统路径（推荐）**

01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09；每章完成正文实验和第 11 章的对应练习，最后做一个小型收官项目。

**研究/工程路径**

完成系统路径后，先做 [第 14 章：收官项目](14-Capstone真实算子交付.md)。从
[官方 examples](https://github.com/tile-ai/tilelang/tree/main/examples) 选择一个与你的硬件和
工作负载匹配的算子，提交基线、边界测试、profile 报告、库对比和版本/硬件限制。若目标是
MoE 或不规则批量矩阵乘，先完成 [第 13 章：分组 GEMM 实战](13-分组GEMM实战.md)。

## 环境检查

建议在独立虚拟环境中安装，并先确认设备是否可用：

```bash
python -m venv .venv
source .venv/bin/activate       # Windows 请使用对应的 Scripts/activate
python -m pip install -U pip
python -m pip install tilelang torch

python - <<'PY'
import sys
import tilelang
import torch

print("python:", sys.version.split()[0])
print("tilelang:", getattr(tilelang, "__version__", "unknown"))
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("mps:", bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available())
PY
```

安装方式、nightly 和源码构建选项会随版本变化，优先参考 [TileLang 官方仓库](https://github.com/tile-ai/tilelang) 和 [资源与版本说明](resources.md)。不要把 CUDA Toolkit 的某个旧版本下限写成所有机器都适用的硬性要求：应以驱动、PyTorch、TileLang wheel 和目标 GPU 的兼容组合为准。

## macOS / Metal 说明

仓库提供了 [examples/metal_verify.py](examples/metal_verify.py) 作为可选 smoke test。它会先检查 MPS 是否可用；没有可用 MPS 时会给出提示并退出，不会把「没有设备」误报成代码错误。

Metal 适合验证：

- `@T.prim_func`、`T.Kernel`、`T.Parallel` 等基础语法；
- vector add、部分 elementwise/reduce，以及当前版本支持的矩阵乘路径；
- 生成的 Metal 源码查看。

Metal 不适合作为本教程 CUDA 路径的性能基线。即使同一个 `T.gemm` 在 Metal 与 CUDA
都能编译，两者也可能使用不同矩阵指令、tile 约束和工具链。`cp.async`、TMA 和 Nsight
仍属于 CUDA 路径；Metal 实验应单独记录设备、系统、TileLang 版本和生成源码。

## 如何判断自己真的学会了

每完成一章，至少留下：

- 一个可复现的正确性结果；
- 一个整除尺寸和一个尾部尺寸的测试；
- 若涉及性能，一份包含硬件、版本、输入、warmup/rep、计时 backend 的记录；
- 一句话说明「我改了什么、观察到什么、为什么相信瓶颈在那里」。

性能数字必须带条件。不要只写「快了 1.2 倍」，要写清基线、输入形状、dtype、测量方法和正确性容差。

## 参考入口

- [TileLang 官方仓库](https://github.com/tile-ai/tilelang)
- [TileLang Getting Started](https://github.com/tile-ai/tilelang/blob/main/docs/get_started/overview.md)
- [TileLang Language Basics](https://github.com/tile-ai/tilelang/blob/main/docs/programming_guides/language_basics.md)
- [TileLang 论文（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/76fb92288bf90360c527efb0d1c2aba6-Abstract-Conference.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

教程 API 和后端都在演进；若本地版本与正文不同，请优先看官方示例，并在实验记录中注明版本。
