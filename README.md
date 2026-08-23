# TileLang 从入门到专家：以实验为中心的内核教程

这是一套面向 GPU 内核、AI 编译器和性能工程学习者的中文 TileLang 教程。主线目标不是记住一串 API，而是建立一条可复用的工作流：

```text
写最小正确基线 → 做 edge-shape 测试 → 测量 → 看生成代码/Profiler
→ 一次只改一个变量 → 解释收益与代价 → 固化配置
```

课程默认以 NVIDIA CUDA GPU 为目标。没有 CUDA GPU 也可以学习语法和编译模型；macOS/Metal 只用于有限的 smoke test，不能用来替代 CUDA 的 GEMM、FlashAttention 或性能实验。

## 先读这里

1. [课程导读与实验规范](00-课程导读与实验规范.md)：学习目标、代码块等级、测试和性能报告模板；
2. [课程大纲](OUTLINE.md)：每章的产出、前置知识和建议时长；
3. [写作与阅读约定](WRITING_GUIDE.md)：章节结构、中文术语、代码块和实验记录规范；
4. [资源与版本说明](resources.md)：官方文档、示例和版本敏感点；
5. 然后从 [第 01 章：环境搭建与第一个内核](01-环境搭建与第一个内核.md) 开始。

## TileLang 是什么

TileLang 是构建在 TVM/TIR 之上的 GPU/加速器内核 DSL。它把 tile、内存作用域、线程块、数据搬运和计算原语作为一等概念，让学习者在 Python 风格语法中表达接近硬件的程序结构。

可以先用三层心智模型理解它：

- **高层**：表达计算和 tile 关系；
- **中层**：控制 grid、shared memory、fragment、流水线和布局；
- **底层**：必要时观察生成代码，使用 warp、PTX、TMA 或架构相关原语。

这里的“层”是学习视角，不是保证每个版本都有固定的 L1/L2/L3 API。具体接口和支持范围以当前官方文档/示例为准。

## 课程结构（13 章）

前 09 章是递进的主线，第 10～12 章分别承担面试表达、迁移练习和查表任务，第 13 章是
建立在 GEMM/MoE 经验之上的专项扩展。不要跳过前面的正确性与测量训练直接背模板。

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

## 三条学习路线

**两周最小路径**

00 → 01 → 02 → 03 → 05 → 06 → 11 中对应练习 → 10 → 12。重点是亲手完成一个 elementwise 内核和一个小尺寸 GEMM/Attention 正确性实验。

**六到八周系统路径（推荐）**

01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09；每章完成正文实验和第 11 章对应练习，最后做一个小型 capstone。

**研究/工程路径**

完成系统路径后，阅读 [官方 examples](https://github.com/tile-ai/tilelang/tree/main/examples)，选择一个算子，提交基线、边界测试、profile 报告、库对比和版本/硬件 caveat。若目标是 MoE 或不规则批量矩阵乘，再进入 [第 13 章：分组 GEMM 实战](13-分组GEMM实战.md)。

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

仓库提供了 [examples/metal_verify.py](examples/metal_verify.py) 作为可选 smoke test。它会先检查 MPS 是否可用；没有可用 MPS 时会给出提示并退出，不会把“没有设备”误报成代码错误。

Metal 适合验证：

- `@T.prim_func`、`T.Kernel`、`T.Parallel` 等基础语法；
- vector add、部分 elementwise/reduce 示例；
- 生成的 Metal 源码查看。

Metal 不适合作为本教程 CUDA 路径的性能基线。第 05/06 章的 `T.gemm`、Tensor Core、cp.async/TMA 和 Nsight 实验需要支持它们的目标 GPU/工具链。

## 如何判断自己真的学会了

每完成一章，至少留下：

- 一个可复现的正确性结果；
- 一个整除尺寸和一个尾部尺寸的测试；
- 若涉及性能，一份包含硬件、版本、输入、warmup/rep、计时 backend 的记录；
- 一句话说明“我改了什么、观察到什么、为什么相信瓶颈在那里”。

性能数字必须带条件。不要只写“快了 1.2 倍”，要写清基线、输入形状、dtype、测量方法和正确性容差。

## 参考入口

- [TileLang 官方仓库](https://github.com/tile-ai/tilelang)
- [TileLang Getting Started](https://github.com/tile-ai/tilelang/blob/main/docs/get_started/overview.md)
- [TileLang Language Basics](https://github.com/tile-ai/tilelang/blob/main/docs/programming_guides/language_basics.md)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

教程 API 和后端都在演进；若本地版本与正文不同，请优先看官方示例，并在实验记录中注明版本。
