# 第 05 章 GEMM 实战：从朴素实现到接近峰值

> **本章目标**：亲手写出并优化一个 FP16 GEMM，走完「朴素 → 分块 → 流水线 →
> swizzle → 自动调优」全流程，并用 profiler 量化每一步的收益。GEMM 是 GPU 内核
> 面试的"hello world"，本章结束后你要能**闭着眼画出它的数据流并口述优化顺序**。

## 5.1 问题与理论峰值

计算 `C[M,N] = A[M,K] × B[K,N]`（行主序）。

- 计算量：`2·M·N·K` FLOPs；
- 理论峰值（示例，近似值）：A100 fp16 tensor core ≈ 312 TFLOPS（含 sparsity 翻倍），
  H100 ≈ 990 TFLOPS，RTX 4090 ≈ 330 TFLOPS；HBM 带宽 A100 ≈ 2 TB/s，H100 ≈ 3.35 TB/s；
- 目标不是"跑得快"，是**逼近硬件峰值的前提下找证据**（面试官最恨"我感觉很快"）。

衡量标准：**TFLOPS = 2·M·N·K / 延迟(ms) / 1e9**。官方基准：调优后的 TileLang
GEMM 在 4090 上约为 cuBLAS 的 1.1 倍、A100 上 ~0.97 倍、H100 上 ~1.0 倍（官方
matmul.md 数据，可作为面试谈资）。

## 5.2 版本 0：不碰共享内存（教学用）

每个 block 算一个 `BM×BN` 输出 tile，直接从全局内存累加（K 维每步都读全局）：

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

**为什么慢**：每个输出元素每步都要从 HBM 读 `A[i,k]` 和 `B[k,j]`，没有复用 → 访存
量 O(MNK) → 被显存带宽死死摁住。多数新手的第一版 GEMM 都死在这。它是对的，但
不是性能起点。

## 5.3 版本 1：分块 + 共享内存 + 流水线（官方标准骨架）

核心改动：K 维切成 `BK` 的块；每轮的 `A/B` tile 先 `T.copy` 进共享内存，再用
`T.gemm` 一句完成 tile 级矩阵乘（编译器把它翻译成 Tensor Core 指令，如 `mma`）。

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

逐行解读（面试口述版）：

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
A = torch.randn(M, K, device='cuda', dtype=torch.float16)
B = torch.randn(K, N, device='cuda', dtype=torch.float16)
C = torch.empty(M, N, device='cuda', dtype=torch.float16)

kernel = gemm_v1(M, N, K)
kernel(A, B, C)
ref = A.to(torch.float32) @ B.to(torch.float32)
torch.testing.assert_close(C.float(), ref, rtol=1e-2, atol=1e-2)

lat = kernel.get_profiler().do_bench()
print(f"latency={lat:.3f} ms, {2*M*N*K/lat/1e6:.1f} TFLOPS")
```

> 本示例 M/N/K 都是 tile 的整数倍，无边界问题；非整除见 5.4。

## 5.4 版本 2：处理任意尺寸（边界/残差）

M、N、K 不整除时，`T.copy` 与 `T.gemm` 都会被自动保护，但最好**先把全局 tile
范围写对**。两种方案：

- **方案 A（软件边界）**：拷贝时用 `T.min` 截断范围，计算时用 predicate 掩码——
  对 GEMM 通常先做 `T.copy` 然后让 auto-safe pass 处理，多数官方示例依赖自动
  保护，仅在 epilogue 写回前判断：
- **方案 B（pad 到整数倍）**：主机侧把 A/B 零填充（或拷贝时 `if` 保护）。性能最好，
  工程上常用。

```python
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
        ...
        for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=num_stages):
            # 用 T.min 限制范围，避免读越界（自动安全 pass 也会兜底）
            T.copy(A[by * BM, ko * BK], A_s)
            T.copy(B[ko * BK, bx * BN], B_s)
            T.gemm(A_s, B_s, C_f)
        for i, j in T.Parallel(BM, BN):
            if T.all_of(by * BM + i < M, bx * BN + j < N):
                C[by * BM + i, bx * BN + j] = C_f[i, j]   # 只写合法区域
```

（更稳的工程做法：K 维也按 `T.ceildiv(K, BK)` 迭代，越界 tile 由
`LegalizeSafeMemoryAccess` 自动屏蔽，前提是**累加器初始值正确**——`T.clear` 保证
整块清零，即使某些 k 是垃圾数据也不影响测试。面试时把这两个坑说清楚很加分。）

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
        ...
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
    def kern(...):
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

看生成代码时的观察点（面试可聊）：`T.gemm` 变成了什么（`mma.sync` / `wmma` /
`wgmma` 等）；`T.copy` 是否变成 `cp.async` + 屏障；K 循环是否出现 prologue/
steady/epilogue 结构；`T.Parallel` 是否被向量化成 `float4`。

## 5.8 进阶方向（一句话认识，面试不虚）

| 方向 | 解决什么 | 官方示例 |
|---|---|---|
| Split-K | 单个输出 tile 串行 K 过长时，把 K 拆给多个 block | `examples/gemm_splitk/` |
| Persistent / Stream-K | 减少 block 启动/负载不均，波次效应（tail effect） | `examples/gemm_persistent/`、`gemm_streamk/` |
| FP8 / INT4 量化 GEMM | 低精度推理吞吐 | `examples/gemm_fp8/`、`gemm_int4/` |
| 2:4 稀疏（T.gemm_sp） | 利用稀疏性翻倍算力 | `examples/gemm_sp/` |
| 分组 GEMM | MoE 注意力场景 | `examples/grouped_gemm/` |

## 5.9 本章小结

- 朴素版死因是**无数据复用**；分块 + shared + 流水线是 GEMM 的标准答案。
- 优化顺序：正确基线 → `T.Pipelined` → swizzle 布局 → 光栅化 → autotune → 逐项
  用 TFLOPS 验证。
- `T.gemm` 是 tile 级原语：内部完成布局分发与 Tensor Core 指令选择。
- 面试黄金句：**"GEMM 的优化本质是内存复用 + 流水线重叠 + 指令集利用"**。

## 面试考点（本章相关）

1. **口述分块 GEMM 的数据流**（global→shared→fragment→global，K 维流水线）。
2. **为什么 TileLang 能 20 行写完 CUDA 要 200 行的 GEMM？**（布局推断 + 流水线自动注入 + T.gemm 直通 Tensor Core）。
3. **num_stages 为什么不能无限大？**（共享内存上限、寄存器、同步开销）。
4. **swizzle 的两种含义？**（共享内存 XOR 布局 vs 光栅化调度；见第 07 章）。
5. **如何测量与报告 GEMM 性能？**（TFLOPS、与 cuBLAS 对比、解释差距来源）。
6. **fp16 输入为什么 fp32 累加？**（数值稳定性 + Tensor Core 要求）。
7. **非整除尺寸怎么处理？**（pad 或 predicate + 自动安全访问）。