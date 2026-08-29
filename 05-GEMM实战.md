# 第 05 章 GEMM 实战：从朴素实现到可测优化

第一次写 GPU 矩阵乘的人，多半会先经历一次认知错位：矩阵乘而已，`torch.matmul` 一行就
写完了，为什么到了 GPU 上要写几十行？答案在第 02～04 章里已经埋好了——一个输出元素要
沿 K 维不断累加，一块输入会被许多输出重复使用，数据复用、片上容量、流水线和矩阵指令
同时挤进同一个设计。这一章就把这些概念第一次真正汇合起来，用五个版本的 GEMM 让你看见
它们逐个进场。

我们会从一个「数学正确、访存糟糕」的版本出发，逐版加入共享内存分块、软件流水线、边界
处理、布局与调度，最后把已知的参数空间交给自动调优。每一版只改变一个主要因素，每一步
都用正确性、生成代码和 profiler 说明收益来自哪里。学完这一章，你留下的不是一段最快的
代码，而是一条随时可以重走的优化路径。

> **本章导航** 中高级难度，预计 5–8 小时；前置是第 01～04 章，理解 shared、fragment、
> pipeline 和边界处理。完整版本需要支持 `T.gemm` 的目标 GPU；本章性能数字只在记录的
> 硬件/版本上成立。学完你会留下一个整除尺寸基线、一个 edge-shape 设计、一张版本对比表
> 和一份 TFLOPS 报告。写作参考：[官方 GEMM 示例](https://github.com/tile-ai/tilelang/blob/main/examples/gemm/example_gemm.py)、
> [TileLang Overview](https://github.com/tile-ai/tilelang/blob/main/docs/get_started/overview.md)。

## 5.1 问题与理论峰值

先钉死问题和度量单位。计算 `C[M,N] = A[M,K] × B[K,N]`（行主序）。计算量是 `2·M·N·K`
FLOPs；拿 `M=N=K=4096` 心算一下，大约 137 GFLOPs——在一张 fp16 峰值 330 TFLOPS 的卡上，
理论下限只有 0.4 毫秒左右。这个数值得先记下来：它告诉我们「快」的物理边界在哪，也告诉
我们离边界还有多远。

- 理论峰值取决于 GPU 型号、dtype、稀疏模式、时钟和库路径；请把目标 GPU 的官方规格
  填入实验记录，不要把某张卡的数字当作课程常数；
- 目标是在明确硬件和测量方法的前提下找到证据，单求一个快的数字不够。

衡量标准：**TFLOPS = 2·M·N·K / 延迟(ms) / 1e9**。与 cuBLAS/torch 的比较必须使用相同
输入、dtype、warmup、rep 和同步方式；不要把历史 benchmark 数字直接套到自己的机器。

这里的公式把延迟单位写成毫秒；等价地，也可以先把延迟换算成秒，再除以 `1e12`。如果只
统计 A/B 的读取或使用了不同的矩阵乘定义，应在报告中明确说明 FLOPs 口径。

### 5.1.1 先钉死精度语义：fp32 的 `T.gemm` 可能悄悄变成 TF32

在写任何优化之前，先处理一个会让「正确性」结论失效的陷阱——它值得放在最前面，因为一旦
中招，你后面测的所有 TFLOPS 都是在一个错误的前提上得出的。"Correct but Slow" 研究
（[arxiv 2607.04454](https://arxiv.org/abs/2607.04454)，A100 实测）报告过：在 SM80 上，
fp32 输入的 `T.gemm` 会**静默降级为 TF32**（TensorFloat-32，尾数 10 位），导致 31.8% 的
输出无法通过 `rtol=1e-5` 的 fp32 检查，而同样形状的手工 FMA 循环却能通过。报告把它归为
「精度降级伪影」而非内核 bug。

教程的处理约定：

1. 本章示例统一用 fp16 输入 + fp32 累加，避开这条路径；
2. 任何声称「fp32 GEMM」的报告，都要写清实际执行的精度路径（fp32 还是 TF32），
   并用严格的 `rtol`（如 `1e-5`）与参考实现对照后再下结论；
3. 如果你的 TileLang 版本在目标架构上对 fp32 `T.gemm` 有显式开关或文档说明，
   以当前版本为准；否则默认把它当作「TF32 精度」来陈述。

## 5.2 版本 0：不碰共享内存（教学用）

按照课程的老规矩，先写一个**肯定正确、肯定不快**的版本。每个 block 算一个 `BM×BN` 输出
tile，直接从全局内存累加（K 维每步都读全局）。你可能会问：明知道慢，写它干什么？因为
它把「复用很差」这件事做成了可见的对照——后面每个版本的提速，都能在它身上找到原因。
为突出访存差异，下面的 v0 只用于整除尺寸；尾部处理留到 5.4：

```python
import torch
import tilelang
import tilelang.language as T
from tilelang import jit

@jit
def gemm_naive(M: int, N: int, K: int, BM: int = 64, BN: int = 64,
               dtype: str = 'float16', accum_dtype: str = 'float32'):
    @T.prim_func
    def kern(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            C_f = T.alloc_fragment((BM, BN), accum_dtype)
            T.clear(C_f)
            for i, j in T.Parallel(BM, BN):
                for k in T.serial(K):                       # K 维串行累加（每步读全局！）
                    C_f[i, j] += A[by * BM + i, k] * B[k, bx * BN + j]
            T.copy(C_f, C[by * BM, bx * BN])
    return kern
```

**为什么慢**：每个输出元素每步都要从 global memory 读 `A[i,k]` 和 `B[k,j]`，复用很差 →
访存量接近 O(MNK)。先用小尺寸验证它，再进入 shared/GEMM 路径，不要把它当作生产实现。

## 5.3 版本 1：分块 + 共享内存 + 流水线

核心改动：K 维切成 `BK` 的块；每轮的 `A/B` tile 先 `T.copy` 进共享内存，再用 `T.gemm`
一句完成 tile 级矩阵乘。具体会落到哪类 Tensor Core/后端指令由目标 GPU、dtype 和
TileLang 版本决定，必须用生成源码确认。

```python
@jit
def gemm_v1(M: int, N: int, K: int, BM: int = 128, BN: int = 128, BK: int = 32,
            num_stages: int = 3, threads: int = 128,
            dtype: str = 'float16', accum_dtype: str = 'float32'):
    @T.prim_func
    def kern(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
            A_s = T.alloc_shared((BM, BK), dtype)      # A 的 tile：BM×BK
            B_s = T.alloc_shared((BK, BN), dtype)      # B 的 tile：BK×BN
            C_f = T.alloc_fragment((BM, BN), accum_dtype)  # 寄存器累加器
            T.clear(C_f)

            for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=num_stages):
                T.copy(A[by * BM, ko * BK], A_s)       # 生产者：global → shared
                T.copy(B[ko * BK, bx * BN], B_s)
                T.gemm(A_s, B_s, C_f)                  # 消费者：tile 级 GEMM

            T.copy(C_f, C[by * BM, bx * BN])           # shared(寄存器) → global
    return kern
```

逐行解读：

1. 网格：`(N/BN) × (M/BM)` 个 block，每个 block 负责一个 `BM×BN` 的输出 tile；
2. **两级缓冲**：数据从 global 进 shared（`A_s/B_s`），再从 shared 进寄存器累加器
   （`C_f`，T.gemm 内部）。共享内存解决复用，寄存器解决 Tensor Core 数据格式；
3. **流水线**：`T.Pipelined(..., num_stages=3)` 让「拷下一轮 tile」与「算本轮 tile」
   重叠——让部分数据搬运等待隐藏在计算后面；能隐藏多少要由生成代码和测量确认；
4. `T.gemm` 一个调用隐含：读共享内存 → 按 MMA 布局分发到线程 → 调 Tensor Core；
5. 累加器 `C_f` 用 fp32：精度 + 匹配 Tensor Core 累加要求。

运行与验证：

```python
M, N, K = 4096, 4096, 4096
A = torch.randn(M, K, device='cuda', dtype=torch.float16)   # shape: [M, K]
B = torch.randn(K, N, device='cuda', dtype=torch.float16)   # shape: [K, N]
C = torch.empty(M, N, device='cuda', dtype=torch.float16)   # shape: [M, N]

kernel = gemm_v1(M, N, K)
kernel(A, B, C)
ref = A.to(torch.float32) @ B.to(torch.float32)             # shape: [M, N]
torch.testing.assert_close(C.float(), ref, rtol=1e-2, atol=1e-2)

lat = kernel.get_profiler().do_bench()
print(f"latency={lat:.3f} ms, {2*M*N*K/lat/1e9:.1f} TFLOPS")
```

> 本示例 M/N/K 都是 tile 的整数倍，无边界问题；非整除见 5.4。

## 5.4 版本 2：处理任意尺寸（边界/残差）

真实的矩阵尺寸几乎从不是 4096 这种漂亮的数——`M=1003`、`N=517`、`K=70` 才是常态。M、N、
K 不整除时，要分别处理三件事：

1. grid 用 `ceildiv` 覆盖尾部输出 tile；
2. epilogue 只写 `M×N` 有效区域；
3. K 维尾部 tile 必须写入确定的 0，不能因为 safe-access 跳过写入就把 shared 中的旧值
   当成 padding。`T.clear(C_f)` 只能清累加器，不能清 `A_s/B_s`。

教学阶段推荐「显式 guarded copy」；工程阶段也可以在 host 侧把 A/B pad 到 tile 整数倍，
再用无分支的主 kernel。下面是 guarded copy 的核心形状（完整代码中保留其余 v1 结构）：

```python
# v0 教学基线：每个 block 负责 BM×BN 输出 tile，直接从 global 读取 K 维数据。
for i, k in T.Parallel(BM, BK):
    # A_s shape=[BM, BK]；只把合法的 A 元素搬入 shared。
    gi = by * BM + i
    gk = ko * BK + k
    if T.all_of(gi < M, gk < K):
        A_s[i, k] = A[gi, gk]
    else:
        A_s[i, k] = 0

for k, j in T.Parallel(BK, BN):
    # B_s shape=[BK, BN]；无效元素必须明确填 0。
    gk = ko * BK + k
    gj = bx * BN + j
    if T.all_of(gk < K, gj < N):
        B_s[k, j] = B[gk, gj]
    else:
        B_s[k, j] = 0

T.gemm(A_s, B_s, C_f)

for i, j in T.Parallel(BM, BN):
    # C_f shape=[BM, BN]；写回前保护 M/N 尾部。
    gi, gj = by * BM + i, bx * BN + j
    if T.all_of(gi < M, gj < N):
        C[gi, gj] = C_f[i, j]
```

> 这里的关键是把每一个可能的无效元素定义成 0，并单独保护输出；写多少 `if` 本身不重要。
> 不同版本的 safe-access pass 可能自动插入 guard，但不会替你推断 GEMM padding
> 的数值语义。先用 `M=1003, N=517, K=70` 这类尺寸做正确性测试，再测大尺寸。

## 5.5 版本 3：布局与光栅化（swizzle）

到了这一步，你已经发现调参的节奏：改一个开关 → 重编译 → 测一次 → 记下来。为了让这个
循环跑得快，把开关做成函数参数，这就是 v3 的写法。它同时给出「从会写到会调」的临门
一脚——两个都叫 swizzle、但作用完全不同的开关：

```python
from tilelang.cuda.intrinsics import make_mma_swizzle_layout

@jit
def gemm_v3(M: int, N: int, K: int, BM: int = 128, BN: int = 128, BK: int = 32,
            num_stages: int = 3, threads: int = 128,
            dtype: str = 'float16', accum_dtype: str = 'float32',
            swizzle: bool = True, rasterize: bool = True):
    @T.prim_func
    def kern(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
            T.use_swizzle(panel_size=10, enable=rasterize)
            A_s = T.alloc_shared((BM, BK), dtype)
            B_s = T.alloc_shared((BK, BN), dtype)
            C_f = T.alloc_fragment((BM, BN), accum_dtype)
            T.clear(C_f)

            if swizzle:
                # 共享内存布局做 swizzle：改变 bank 映射，是否减少冲突要实测（第 07 章）
                T.annotate_layout({
                    A_s: make_mma_swizzle_layout(A_s),
                    B_s: make_mma_swizzle_layout(B_s),
                })

            for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=num_stages):
                T.copy(A[by * BM, ko * BK], A_s)
                T.copy(B[ko * BK, bx * BN], B_s)
                T.gemm(A_s, B_s, C_f)
            T.copy(C_f, C[by * BM, bx * BN])
    return kern
```

`T.use_swizzle(panel_size=10, enable=True)` 是另一个层面的「swizzle」——**光栅化
（rasterization）**：改变 block 的调度顺序（对角线优先而不是按行扫描），提高 L2 命中率。
放在 `T.Kernel` 上下文里即可：

```python
with T.Kernel(...) as (bx, by):
    T.use_swizzle(panel_size=10, enable=True)   # 可选：光栅化调度
    # 其余 tile 计算省略
```

交互实验建议：`swizzle=True/False`、`rasterize=True/False`、`num_stages=2/3/4`、
`threads=128/256`，各跑一次 `do_bench`，记录 TFLOPS 表格——**自己测出来的数据比任何
教程都有说服力**。如果你测下来某个开关反而变慢，别删掉那行记录：失败实验同样有价值，
它告诉你这个访问模式不吃这套优化。

## 5.6 版本 4：交给 autotune（生产做法）

手动把上面的组合矩阵扫完，是 2×2×3×2 = 24 次编译测量。这种重复劳动正是 autotune 的
主场——但请记住，它只在你定义的参数空间里搜索，不会替你发明新的维度：

```python
@tilelang.autotune(configs=lambda: [...], warmup=25, rep=100, timeout=60)
@tilelang.jit(out_idx=[-1])
def gemm_v4(M: int, N: int, K: int, block_M: int = 128, block_N: int = 128,
            block_K: int = 32, threads: int = 128, num_stages: int = 3,
            dtype: str = 'float16', accum_dtype: str = 'float32'):
    @T.prim_func
    def kern(A, B, C):
        ...   # 与 v1 相同的主体内核
    return kern
```

配置空间、正确性校验、缓存与工程化细节在第 08 章展开，这里只留一句：**先有一个正确
基线，再让 autotune 替你在 tile 尺寸/流水线深度/线程数上扫最优解**。

## 5.7 分析工具箱（每个版本都要用）

每个版本跑完，固定用同一套工具收尾，养成习惯后排查会快很多：

```python
# 1) 正确性
profiler = kernel.get_profiler()
profiler.assert_allclose(lambda A, B: (A.float() @ B.float()), rtol=1e-2, atol=1e-2)

# 2) 延迟
lat = profiler.do_bench(warmup=25, rep=100)

# 3) 对比参考（cuBLAS 或 torch）
import torch
def torch_gemm(A, B):
    return A.float() @ B.float()
lat_ref = profiler.do_bench(torch_gemm, warmup=25, rep=100)

# 4) 生成代码复盘
print(kernel.get_kernel_source()[:3000])
```

看生成代码时的观察点：`T.gemm` 变成了什么（`mma.sync` / `wmma` / `wgmma` 等）；`T.copy`
是否变成 `cp.async` + 屏障；K 循环是否出现 prologue/steady/epilogue 结构；`T.Parallel`
是否被向量化成 `float4`。

### 5.7.1 大 GEMM 的两个结构性陷阱：调优网格与 L2 驻留

如果你已经把上面所有开关扫过一遍、TFLOPS 却仍然停在低位，先别急着改代码——"Correct
but Slow" 研究在 A100/GH200 上给大尺寸 GEMM 归因出了两个值得提前知道的坑：

1. **默认调优网格缺配置（RC2a）**。它测的 16384² GEMM 用教程默认的
   `(BLOCK_M, BLOCK_N, BLOCK_K)` 网格只到 cuBLAS 的 59.7%；把 `GROUP_SIZE_M`
   （L2 swizzle 参数）、`num_warps`、`num_stages` 一并纳入扫描后恢复到 81.9%。
   教训：**autotune 的空间由你定义，缺的维度不会自动出现**——第 08 章的 config
   空间至少要覆盖 tile、stages、threads/warps 与调度开关。
2. **L2 驻留（RC2b）**。16384² 的 A+B+C 工作集约 1.6 GB，远超 A100 的 40 MB L2。
   cuBLAS 通过 cuBLASLt 的 plan 选择器做了缓存分块（L2 命中 80.5%），通用 DSL 编译
   缺少这类启发式时只有 49.4% 命中、呈 DRAM-bound。教训：大尺寸下
   `T.use_swizzle`（光栅化调度）改善 L2 局部性的意义远超小尺寸，值得单独开关实验。

这两项都属于「结构性残留」：改作者代码救不回来，要靠调度空间和缓存策略。第 09 章给出
对应的 Nsight 判据（L2 命中率、DRAM 吞吐、TFLOPS 对扫参曲线）。

## 5.8 进阶方向（先知道它们解决什么问题）

| 方向 | 解决什么 | 官方示例 |
|---|---|---|
| Split-K | 单个输出 tile 串行 K 过长时，把 K 拆给多个 block | `examples/gemm_splitk/` |
| Persistent / Stream-K | 缓解波次尾部和 K 维工作分配不均 | `examples/gemm/example_gemm_persistent.py`、`examples/gemm_streamk/` |
| FP8 / INT4 量化 GEMM | 低精度推理吞吐 | `examples/gemm_fp8/`、`gemm_int4/` |
| 2:4 稀疏（T.gemm_sp） | 利用稀疏性翻倍算力 | `examples/gemm_sp/` |
| 分组 GEMM | MoE 专家计算和不规则矩阵批次 | `examples/grouped_gemm/` |

选择进阶路线前先看问题形状。Split-K 增加局部结果合并，persistent kernel 可能降低并发
灵活性，Stream-K 也会改变工作分配与归约成本。这三者各解决一类问题，不存在固定的升级顺序。

### 5.8.1 “结果正确”和“值得替换库”是两道门槛

社区教程常展示某个特定形状上接近或超过库实现的结果，但这不等于交付完成。一项面向 GPU
DSL 的[经验研究](https://arxiv.org/abs/2607.04454) 专门量化了「结果正确但远慢于库」这一
评估缺口。它的样本和设备有限，不能推广成「所有 TileLang 内核都慢」；可引出的正确结论是，
真实交付至少要回答两组问题：

1. **正确性覆盖**：整除、尾部、小尺寸、不同 dtype、转置和数值极端输入是否通过？
2. **替换价值**：目标 shape 分布上的延迟、吞吐、编译成本和端到端收益，是否优于现有路径？

一个内核通过 `assert_close`，只说明抽样输入上的数值结果符合容差。它可能仍然因为串行归约、
不必要的类型转换、错误 tile 或调优空间太窄而非常慢。报告中应增加“相对库效率”一列：

```text
relative_efficiency = reference_library_latency / tilelang_latency
```

这里使用的是延迟比；数值大于 1 表示 TileLang 内核更快。必须保证两边输出语义、输入、精度、
warmup、rep 和计时范围一致。若内核只在一个 shape 上占优，就把 dispatch 范围限制在该 shape
bucket，而不是替换所有 GEMM。

## 5.9 本章回顾

这一章把前三章的概念凝成了一个可迭代的 GEMM。朴素版的主要问题是**缺少数据复用**；
分块、shared 和流水线构成常见的优化路径。优化顺序可以记为：正确基线 → `T.Pipelined`
→ swizzle 布局 → 光栅化 → autotune → 逐项用 TFLOPS 验证。`T.gemm` 是 tile 级原语：
内部完成布局分发与 Tensor Core 指令选择。一句话总结：**GEMM 的优化通常围绕内存复用、
流水线重叠和目标指令利用展开**。

## 5.10 动手任务

以下实验的重点是保留版本差异。只有前后版本都可复现，性能变化才有解释对象。完成下面任务再进入第 6 章：

1. 先只运行 v0 和 v1，给出正确性结果和延迟；
2. 用一个 M/N/K 都不整除的尺寸验证 guarded copy 或 host padding；
3. 每次只打开一个开关：`num_stages`、布局 swizzle、rasterization、tile 尺寸；
4. 以表格记录 latency、TFLOPS、shared memory/寄存器线索和生成代码观察；
5. 解释为什么某个配置更快，不能只写「autotune 选中了它」。

## 5.11 自问自答

下面这些问题用来检验你是否能把这一章的知识讲出来（详答见第 10 章）：

1. **口述分块 GEMM 的数据流**（global→shared→fragment→global，K 维流水线）。
2. **为什么 TileLang 能 20 行写完 CUDA 要 200 行的 GEMM？**（布局推断 + 流水线自动注入 + T.gemm 直通 Tensor Core）。
3. **num_stages 为什么不能无限大？**（共享内存上限、寄存器、同步开销）。
4. **swizzle 的两种含义？**（共享内存 XOR 布局 vs 光栅化调度；见第 07 章）。
5. **如何测量与报告 GEMM 性能？**（TFLOPS、与 cuBLAS 对比、解释差距来源）。
6. **fp16 输入为什么 fp32 累加？**（数值稳定性 + Tensor Core 要求）。
7. **非整除尺寸怎么处理？**（pad 或 predicate + 自动安全访问）。
