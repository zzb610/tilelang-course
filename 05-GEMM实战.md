# 第 05 章 GEMM 实战：从朴素实现到可测优化

> **本章目标**：围绕同一个 FP16 GEMM，按“正确基线 → 数据复用 → 流水线 → 布局/调度
> → 自动调优”的顺序做实验。每一版只改变一个主要因素，并用正确性、生成代码和
> profiler 说明收益来自哪里；本章不承诺任何固定的峰值比例。

**学习信息**

- 难度：中高级；预计用时：5–8 小时；
- 前置：第 01～04 章，理解 shared、fragment、pipeline 和边界处理；
- 运行范围：完整版本需要支持 `T.gemm` 的目标 GPU；本章性能数字只在记录的硬件/版本上成立；
- 本章产出：一个整除尺寸基线、一个 edge-shape 设计、一张版本对比表和一份 TFLOPS 报告。
- 参考：[官方 GEMM 示例](https://github.com/tile-ai/tilelang/blob/main/examples/gemm/example_gemm.py)、[TileLang Overview](https://github.com/tile-ai/tilelang/blob/main/docs/get_started/overview.md)。

**阅读路线：**先用 v0 建立“能算对但访存重复”的对照，再只引入 shared tile，随后引入
流水线。v2 专门处理任意尺寸，v3 只做布局和 block 调度实验，v4 才把已知的参数空间
交给 autotune。每次实验都保留上一版，避免把多个变化混成一个“黑盒加速”。

## 5.1 问题与理论峰值

计算 `C[M,N] = A[M,K] × B[K,N]`（行主序）。

- 计算量：`2·M·N·K` FLOPs；
- 理论峰值取决于 GPU 型号、dtype、稀疏模式、时钟和库路径；请把目标 GPU 的官方规格
  填入实验记录，不要把某张卡的数字当作课程常数；
- 目标不是只追求一个快的数字，而是在明确硬件和测量方法的前提下找到证据。

衡量标准：**TFLOPS = 2·M·N·K / 延迟(ms) / 1e9**。与 cuBLAS/torch 的比较必须
使用相同输入、dtype、warmup、rep 和同步方式；不要把历史 benchmark 数字直接套到自己的机器。

这里的公式把延迟单位写成毫秒；等价地，也可以先把延迟换算成秒，再除以 `1e12`。
如果只统计 A/B 的读取或使用了不同的矩阵乘定义，应在报告中明确说明 FLOPs 口径。

## 5.2 版本 0：不碰共享内存（教学用）

每个 block 算一个 `BM×BN` 输出 tile，直接从全局内存累加（K 维每步都读全局）。为
突出访存差异，下面的 v0 只用于整除尺寸；尾部处理留到 5.4：

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

**为什么慢**：每个输出元素每步都要从 global memory 读 `A[i,k]` 和 `B[k,j]`，复用
很差 → 访存量接近 O(MNK)。这是用于建立对照的教学基线；先用小尺寸验证它，再进入
shared/GEMM 路径，不要把它当作生产实现。

## 5.3 版本 1：分块 + 共享内存 + 流水线（官方标准骨架）

核心改动：K 维切成 `BK` 的块；每轮的 `A/B` tile 先 `T.copy` 进共享内存，再用
`T.gemm` 一句完成 tile 级矩阵乘。具体会落到哪类 Tensor Core/后端指令由目标 GPU、
dtype 和 TileLang 版本决定，必须用生成源码确认。

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
3. **流水线**：`T.Pipelined(..., num_stages=3)` 让"拷下一轮 tile"与"算本轮 tile"
   重叠——拷贝的数百周期延迟被藏进计算里；
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

M、N、K 不整除时，要分别处理三件事：

1. grid 用 `ceildiv` 覆盖尾部输出 tile；
2. epilogue 只写 `M×N` 有效区域；
3. K 维尾部 tile 必须写入确定的 0，不能因为 safe-access 跳过写入就把 shared 中的旧值
   当成 padding。`T.clear(C_f)` 只能清累加器，不能清 `A_s/B_s`。

教学阶段推荐“显式 guarded copy”；工程阶段也可以在 host 侧把 A/B pad 到 tile 整数倍，
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

> 这里的关键不是“写了几个 if”，而是把每一个可能的无效元素定义成 0，并单独保护
> 输出。不同版本的 safe-access pass 可能自动插入 guard，但不会替你推断 GEMM padding
> 的数值语义。先用 `M=1003, N=517, K=70` 这类尺寸做正确性测试，再测大尺寸。

## 5.5 版本 3：布局与光栅化（swizzle）

这是"从会写到会调"的临门一脚：

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
                # 共享内存布局做 XOR swizzle：消除 bank conflict（详见第 07 章）
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

`T.use_swizzle(panel_size=10, enable=True)` 是另一个层面的"swizzle"——**光栅化
（rasterization）**：改变 block 的调度顺序（对角线优先而不是按行扫描），提高 L2
命中率。放在 `T.Kernel` 上下文里即可：

```python
with T.Kernel(...) as (bx, by):
    T.use_swizzle(panel_size=10, enable=True)   # 可选：光栅化调度
    # 其余 tile 计算省略
```

交互实验建议：`swizzle=True/False`、`rasterize=True/False`、`num_stages=2/3/4`、
`threads=128/256`，各跑一次 `do_bench`，记录 TFLOPS 表格——**自己测出来的数据
比任何教程都有说服力**。

## 5.6 版本 4：交给 autotune（生产做法）

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

配置空间、正确性校验、缓存与工程化细节在第 08 章展开，这里只留一句：**先有一个
正确基线，再让 autotune 替你在 tile 尺寸/流水线深度/线程数上扫最优解**。

## 5.7 分析工具箱（每个版本都要用）

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

看生成代码时的观察点：`T.gemm` 变成了什么（`mma.sync` / `wmma` /
`wgmma` 等）；`T.copy` 是否变成 `cp.async` + 屏障；K 循环是否出现 prologue/
steady/epilogue 结构；`T.Parallel` 是否被向量化成 `float4`。

## 5.8 进阶方向（先知道它们解决什么问题）

| 方向 | 解决什么 | 官方示例 |
|---|---|---|
| Split-K | 单个输出 tile 串行 K 过长时，把 K 拆给多个 block | `examples/gemm_splitk/` |
| Persistent / Stream-K | 减少 block 启动/负载不均，波次效应（tail effect） | `examples/gemm_persistent/`、`gemm_streamk/` |
| FP8 / INT4 量化 GEMM | 低精度推理吞吐 | `examples/gemm_fp8/`、`gemm_int4/` |
| 2:4 稀疏（T.gemm_sp） | 利用稀疏性翻倍算力 | `examples/gemm_sp/` |
| 分组 GEMM | MoE 注意力场景 | `examples/grouped_gemm/` |

## 5.9 本章小结

- 朴素版的主要问题是**缺少数据复用**；分块、shared 和流水线构成常见的优化路径。
- 优化顺序：正确基线 → `T.Pipelined` → swizzle 布局 → 光栅化 → autotune → 逐项
  用 TFLOPS 验证。
- `T.gemm` 是 tile 级原语：内部完成布局分发与 Tensor Core 指令选择。
- 一句话总结：**GEMM 的优化通常围绕内存复用、流水线重叠和目标指令利用展开**。

## 5.10 Checkpoint

1. 先只运行 v0 和 v1，给出正确性结果和延迟；
2. 用一个 M/N/K 都不整除的尺寸验证 guarded copy 或 host padding；
3. 每次只打开一个开关：`num_stages`、布局 swizzle、rasterization、tile 尺寸；
4. 以表格记录 latency、TFLOPS、shared memory/寄存器线索和生成代码观察；
5. 解释为什么某个配置更快，不能只写“autotune 选中了它”。

## 口述自测（详答见第 10 章）

1. **口述分块 GEMM 的数据流**（global→shared→fragment→global，K 维流水线）。
2. **为什么 TileLang 能 20 行写完 CUDA 要 200 行的 GEMM？**（布局推断 + 流水线自动注入 + T.gemm 直通 Tensor Core）。
3. **num_stages 为什么不能无限大？**（共享内存上限、寄存器、同步开销）。
4. **swizzle 的两种含义？**（共享内存 XOR 布局 vs 光栅化调度；见第 07 章）。
5. **如何测量与报告 GEMM 性能？**（TFLOPS、与 cuBLAS 对比、解释差距来源）。
6. **fp16 输入为什么 fp32 累加？**（数值稳定性 + Tensor Core 要求）。
7. **非整除尺寸怎么处理？**（pad 或 predicate + 自动安全访问）。
