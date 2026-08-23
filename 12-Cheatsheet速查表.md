# 第 12 章 Cheatsheet 速查表

> 面试前 30 分钟 / 写内核时随手翻的一页式参考。这里收录常用 API；
> 硬件数字为**近似值**，型号不同会有出入，面试时说明"我记得大概是"。

> 这是查表，不是跨版本 API 契约。`TMA`、warpgroup、TMEM、执行后端、autotune
> 持久化和部分 layout helper 可能随 TileLang 版本/目标架构变化；复制前先查
> `resources.md` 和当前官方 examples。

## 0. 标准开头与最小模板

```python
import tilelang
import tilelang.language as T
from tilelang import jit

@jit
def add(N: int, block: int = 256, dtype: str = 'float32'):
    @T.prim_func
    def kern(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype),
             C: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                gi = bx * block + i
                C[gi] = A[gi] + B[gi]
    return kern
# kernel = add(1 << 20); kernel(A, B, C); kernel.get_profiler().do_bench()
```

## 1. 核心 API 速查

### 内核与循环

| API | 作用 | 备注 |
|---|---|---|
| `@T.prim_func` | 函数 → TIR | 参数用 `T.Tensor((shp,), dtype)` |
| `with T.Kernel(gx, gy, gz, threads=t) as (bx,by,bz)` | grid/block 配置 | bx↔blockIdx.x；线程映射自动 |
| `T.serial(a, b, s)` | 顺序循环 | 有依赖/归约用 |
| `T.unroll(n)` | 展开 | 小循环 |
| `T.Parallel(e0, e1, ...)` | 并行循环 | elementwise/拷贝；给编译器并行化/向量化机会 |
| `T.Pipelined(n, num_stages=k)` | 软件流水线 | GEMM/FA 主循环标配 |
| `T.Persistent(...)` | 持久化 block | 高级模板 |
| `T.while` / `break` / `continue` | 控制流 | 条件须为 TIR 表达式 |
| `T.all_of / T.any_of` | 多条件 | 边界判断用 |

### 内存分配

| API | 作用域 | 用途 |
|---|---|---|
| `T.alloc_shared(shape, dtype)` | shared | block 共享，数据中转 |
| `T.alloc_fragment(shape, dtype)` | 寄存器 | 累加器/局部结果（布局推断） |
| `T.alloc_var(dtype, init=...)` | 寄存器 | 标量变量 |
| `T.alloc_barrier(arrive_count)` | shared | mbarrier（TMA 协作） |
| `T.alloc_tmem(shape, dtype)` | TMEM | Blackwell+ |
| `T.empty(shape, dtype)` | global | 声明输出张量 |

### 数据搬运

| API | 说明 |
|---|---|
| `T.copy(src, dst, coalesced_width=None, disable_tma=False, eviction_policy=None, loop_layout=None)` | 同步语义；自动合并/向量化/选择机制 |
| `T.async_copy(src, dst)` | 显式异步；消费前按目标版本确认 wait/sync |
| `T.tma_copy(src, dst)` | cp.async.bulk（TMA） |
| `T.transpose(src, dst)` | 共享内存转置 |
| `T.c2d_im2col(img, col, ...)` | 卷积 im2col |

### 计算原语

| API | 说明 |
|---|---|
| `T.gemm(A_s, B_s, C_f, transpose_A/B=False, policy=...)` | tile GEMM → Tensor Core；A/B 需 shared，C 需 fragment |
| `T.gemm_sp(...)` | 2:4 稀疏 |
| `T.reduce_sum/max/min(acc_s, dst, dim=1, clear=False)` | 片段归约到行/列 |
| `T.cumsum / T.cummax` | 扫描 |
| `T.warp_reduce_sum/max/min(...)` | warp 内归约（shuffle） |
| `T.clear(buf) / T.fill(buf, v)` | 清零/填充 |
| `T.exp / T.log / T.rsqrt / T.max / T.min / T.exp2` | 数学（TIR） |
| `T.if_then_else(c, a, b)` | 三目 |
| `T.ceildiv(a, b)` | 向上取整除法 |
| `T.reshape / T.view` | 零拷贝视图 |

### 同步与原语

| API | 说明 |
|---|---|
| `T.sync_threads() / T.sync_warp() / T.sync_grid()` | 屏障 |
| `T.ptx_wait_group(0)` / `T.ptx_commit_group()` | cp.async 完成控制 |
| `T.get_lane_idx() / T.get_warp_idx()` | 索引 |
| `T.shfl_sync / shfl_down / shfl_up / shfl_xor` | warp 数据交换 |
| `T.ballot_sync / T.activemask / T.any_sync / T.all_sync` | 投票 |
| `T.match_any_sync(value)` | 值匹配掩码（sm_70+，HIP 无） |
| `T.syncthreads_count/and/or(pred)` | 同步 + 统计 |
| `T.atomic_add/max/min/load/store(...)` | 原子（memory_order/return_prev） |
| `T.dp4a(A, B, C)` | int8 点积累加 |
| `T.fence_proxy_async()` | async proxy fence（TMA 前） |
| `T.set_max_nreg / inc / dec` | 寄存器压力控制 |

### 调试与注解

| API | 说明 |
|---|---|
| `T.print(obj, msg='...')` | 单线程打印 buffer/标量 |
| `T.device_assert(cond, msg)` | 设备端断言 |
| `T.use_swizzle(panel_size=10, enable=True)` | 光栅化调度（L2 局部性） |
| `T.annotate_layout({buf: layout})` | 显式布局（如 `make_mma_swizzle_layout`） |
| `T.annotate_l2_hit_ratio(buf, ratio)` | 缓存行为提示 |

## 2. Tiled GEMM 模板（默写）

```python
@T.prim_func
def gemm(A: T.Tensor((M, K), 'float16'), B: T.Tensor((K, N), 'float16'),
         C: T.Tensor((M, N), 'float16')):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        A_s = T.alloc_shared((BM, BK), 'float16')
        B_s = T.alloc_shared((BK, BN), 'float16')
        C_f = T.alloc_fragment((BM, BN), 'float32')   # 累加用 fp32！
        T.clear(C_f)
        for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
            T.copy(A[by * BM, ko * BK], A_s)
            T.copy(B[ko * BK, bx * BN], B_s)
            T.gemm(A_s, B_s, C_f)
        T.copy(C_f, C[by * BM, bx * BN])
```

## 3. 编译与运行速查

```python
kernel = tilelang.compile(func, out_idx=[2], target="cuda")   # 显式编译
kernel = factory(M, N, K)                                      # @jit 风格
kernel(A, B)                                                   # 调用
kernel.get_kernel_source()                                     # 生成的 CUDA 源码
p = kernel.get_profiler(); p.do_bench(warmup=25, rep=100)      # 延迟 ms
p.assert_allclose(ref_fn, rtol=1e-2, atol=1e-2)                # 校验
# TFLOPS = 2·M·N·K / latency_ms / 1e9
```

| 参数 | 取值 |
|---|---|
| `target` | `'auto' \| 'cuda' \| 'hip' \| 'metal'` |
| `execution_backend` | `'auto' \| 'tvm_ffi' \| 'cython' \| 'nvrtc' \| 'torch'` |
| `out_idx` | 输出参数下标；`[-1]` 最后一个；多输出返回元组 |
| `pass_configs` | e.g. `{tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}` |

### Autotune 最小模板

```python
@tilelang.autotune(configs=configs_fn, warmup=25, rep=100, timeout=60)
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M=128, block_N=128, block_K=32,
           threads=128, num_stages=3, dtype='float16', accum_dtype='float32'):
    @T.prim_func
    def kern(A, B, C):
        ...
    return kern

from tilelang.autotuner import set_autotune_inputs
with set_autotune_inputs(A, B, C):
    best = matmul(M, N, K)
```

## 4. dtype 速查

| 类别 | 值 | 说明 |
|---|---|---|
| 半精度 | `'float16'`（e5m10）| GEMM 输入主力 |
| BF16 | `'bfloat16'` | 范围同 fp32，尾数 8 位 |
| 单/双 | `'float32'` / `'float64'` | 累加常用 fp32 |
| 整型 | `'int8/16/32/64'`、`'uint*'` | |
| FP8 | `'float8_e4m3fn'`、`'float8_e5m2'` 等 | 依目标 GPU、编译器和 PyTorch 支持；不要直接承诺吞吐翻倍 |
| 向量 | `'float32x4'` 等 x2/x4/x8/... | SIMD pack |

指定方式三种等价：字符串 / `T.float32` / `torch.float32`。
**铁律：fp16/bf16 输入 + fp32 累加。**

## 5. 环境变量速查

| 变量 | 作用 |
|---|---|
| `TILELANG_CACHE_DIR` | 缓存根（默认 `~/.tilelang/cache`） |
| `TILELANG_TMP_DIR` | 临时目录 |
| `TILELANG_DISABLE_CACHE=1` | 关全部内核缓存 |
| `TILELANG_AUTO_TUNING_DISABLE_CACHE=1` | 只关 autotune 磁盘缓存 |
| `TL_LOWER_TRACE=terminal\|html\|both` | 每 pass 的 IR 变化追踪 |
| `TILELANG_AUTO_TUNING_CPU_COUNTS` 等 | autotune 并行度控制 |
| `TVM_ROOT` / `WITH_PIP_CUDA_TOOLCHAIN` | 构建期 |

## 6. GPU 硬件常数参考（近似值，随型号变化）

| 项目 | A100 | H100 SXM | RTX 4090 |
|---|---|---|---|
| HBM 带宽 | ~2.0 TB/s | ~3.35 TB/s | ~1.0 TB/s |
| FP16 Tensor 峰值 | ~312 TFLOPS | ~990 TFLOPS | ~330 TFLOPS |
| FP8 Tensor 峰值 | — | ~1979 TFLOPS | — |
| 共享内存/block | 48KB 默认，可上探 ~164KB | 类似（Hopper 227KB 动态） | 48KB 默认，~100KB 动态 |
| warp 大小 | 32 线程 | 32 | 32 |
| SM bank 数 | 32（4B/bank） | 32 | 32 |
| 最大线程/block | 1024 | 1024 | 1024 |

> 面试说"近似、以官方规格为准"即可；重点是用它们做**理论峰值对比**。

## 7. 常见报错与解决

| 报错/症状 | 原因与解法 |
|---|---|
| `Ramp of more than 4 lanes is not allowed` | 向量化宽度过宽（8 lane）；检查 T.copy/T.Parallel 的宽度提示或改布局 |
| shared memory 超限 | tile×stages 太大；降 BK/stages 或 tile |
| `T.gemm` 布局不匹配报错 | A_s/B_s/C_f 布局冲突；`annotate_layout` 显式指定 |
| 结果错在边界 tile | 越界读写或尾部 tile 未零填充；输出用 guard，GEMM 输入明确 padding |
| `async_copy` 结果错 | 忘了 `T.ptx_wait_group`；或屏障时机不对 |
| autotune "No configurations" | configs 为空/过滤过头；检查生成器 |
| 动态形状 autotune 报错 | 内置生成器需静态形状；改用 `set_autotune_inputs` |

## 8. 面试 30 秒自我介绍稿（模板）

"我使用 TileLang（基于 TVM TIR 的 GPU 内核 DSL）做过 [算子名]：从朴素版本到
分块 + 软件流水线 + swizzle 布局 + 自动调优，最终 [X] TFLOPS，达到硬件峰值的
[Y]%，与 cuBLAS 对比 [结论]。我的方法：先判断 memory/compute bound，再画数据流，
逐项优化并用 profiler 验证，最后用 Nsight 归因剩余差距。"

---

*全课程到这里结束。祝你面试顺利——记住：数据、对照、归因，永远是内核工程师的
三个关键词。*
