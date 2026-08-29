# 第 12 章 Cheatsheet 速查表

走到这里，你已经读完了主线、练过了表达、做过了题。这一章要回答一个更实际的问题：
当你在写内核、调试或在面试前临时查一个 API 时，应该去哪一眼找到答案。下面这张表就是
干这个用的——它是「查表」，不是「学习顺序」，只适合在完成主线之后用来快速确认 API、
公式、排错入口和报告字段。它回答的是"这个 API 怎么调""这个报错大概是什么"这类即时
问题；一旦某个概念让你觉得不确定，就应当回到第 01～09 章去补，而不是停在这里的单行
结论上。让我们先把这个边界说清楚，再进入表格。

> 这是查表，不是跨版本 API 契约。`TMA`、warpgroup、TMEM、执行后端、autotune
> 持久化和部分 layout helper 可能随 TileLang 版本/目标架构变化；复制前先查
> `resources.md` 和当前官方 examples。

## 0. 标准开头与最小模板

动手之前的第一个问题通常是"一个最简内核长什么样"。下面的最小模板集成了 import、
`@jit` 工厂、`@T.prim_func` 和一次边界保护，你可以把它当作所有实验的固定起点：

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
                if gi < N:
                    C[gi] = A[gi] + B[gi]
    return kern
# kernel = add(1 << 20); kernel(A, B, C); kernel.get_profiler().do_bench()
```

## 1. 核心 API 速查

这一节按用途把最常用的 API 分组列出。查的时候先想清楚你要做的是哪一类操作——是声明
并行结构、分配内存、搬运数据，还是做归约/同步——再定位到对应的小节；每张表的第一列
是 API 名，后两列分别是它的作用和需要注意的地方。

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
| Python `while` / `break` / `continue` | 控制流 | 条件须能转成 TIR 表达式 |
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
| `T.gemm(A_s, B_s, C_f, transpose_A/B=False, policy=...)` | tile GEMM；具体矩阵指令、scope 和 shape 约束取决于 target。**fp32 输入在 SM80 上可能静默降为 TF32**，严格精度报告需确认实际执行精度 |
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

API 查得再多，还是要有一份可以脱口而出的标准结构。下面这个 tiled GEMM 模板应当成为
你写任何矩阵乘内核时的肌肉记忆——注意累加器用 fp32，这是精度语义里最容易漏的一点：

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

写完内核之后的问题，是"怎么编译、怎么跑、怎么测"。下面的代码块把编译、调用、取源码、
计时和校验串成一条线，随后那张表补充了几个关键参数的可选取值：

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
| `target` | `'auto' \| 'cuda' \| 'hip' \| 'metal' \| 'llvm' \| 'webgpu' \| 'c' \| 'cutedsl'` |
| `execution_backend` | `'auto' \| 'tvm_ffi' \| 'cython' \| 'nvrtc' \| 'torch'` |
| `out_idx` | 输出参数下标；`[-1]` 最后一个；多输出返回元组 |
| `pass_configs` | e.g. `{tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}` |

### Autotune 最小模板

当调参空间太大、手动扫不过来时，就轮到 autotune 出场。下面是最小模板，记住所有候选
配置必须喂进同一批输入，这是让最终配置可信的前提：

```python
# 速查模板：A/B/C 都是二维 tile GEMM 的输入输出，具体 shape 在 prim_func 中声明。
@tilelang.autotune(configs=configs_fn, warmup=25, rep=100, timeout=60,
                   early_stop=False)
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M=128, block_N=128, block_K=32,
           threads=128, num_stages=3, dtype='float16', accum_dtype='float32'):
    # 这里是 autotune 外壳；真实 kernel 内应把 A/B/C 声明为 [M,K]/[K,N]/[M,N]。
    @T.prim_func
    def kern(A, B, C):
        # A/B/C 的 shape 见上一行说明；此处省略 tiled GEMM 主体。
        ...
    return kern

from tilelang.autotuner import set_autotune_inputs
# A=[M,K]、B=[K,N]、C=[M,N]；所有候选使用同一批输入。
with set_autotune_inputs(A, B, C):
    best = matmul(M, N, K)
```

## 4. dtype 速查

精度选择直接影响吞吐与正确性，是很多坑的源头。这张表列出常见 dtype 的值、命名和
用途，帮你在一秒内确认"应该用哪个"：

| 类别 | 值 | 说明 |
|---|---|---|
| 半精度 | `'float16'`（e5m10）| GEMM 输入主力 |
| BF16 | `'bfloat16'` | 范围同 fp32，尾数 8 位 |
| 单/双 | `'float32'` / `'float64'` | 累加常用 fp32 |
| 整型 | `'int8/16/32/64'`、`'uint*'` | |
| FP8 | `'float8_e4m3fn'`、`'float8_e5m2'` 等 | 依目标 GPU、编译器和 PyTorch 支持；不要直接承诺吞吐翻倍 |
| 向量 | `'float32x4'` 等 x2/x4/x8/... | SIMD pack |

指定 dtype 的方式通常允许字符串、`T.float32` 或框架 dtype，但可接受范围仍以当前版本和后端
为准。常见 GEMM 做法是 fp16/bf16 输入、fp32 累加；这不是所有算子和 dtype 的硬性规则。

## 5. 环境变量速查

这类变量大多在排查缓存或调试 lowering 时才会用到。表里列出变量名与作用，调试时按需
翻阅即可，无需死记：

| 变量 | 作用 |
|---|---|
| `TILELANG_CACHE_DIR` | 缓存根（默认 `~/.tilelang/cache`） |
| `TILELANG_TMP_DIR` | 临时目录 |
| `TILELANG_DISABLE_CACHE=1` | 关全部内核缓存 |
| `TILELANG_AUTO_TUNING_DISABLE_CACHE=1` | 只关 autotune 磁盘缓存 |
| `TL_LOWER_TRACE=terminal\|html\|both` | 每 pass 的 IR 变化追踪 |
| `TILELANG_AUTO_TUNING_CPU_COUNTS` 等 | autotune 并行度控制 |
| `TVM_ROOT` / `WITH_PIP_CUDA_TOOLCHAIN` | 构建期 |

## 6. 目标硬件记录项

硬件数字会随型号、形态、功耗配置和精度口径变化，速查表不保存一份容易过期的峰值副本。
每次实验从目标 GPU 的官方规格和运行时查询下列字段：

| 类别 | 要记录什么 |
|---|---|
| 身份 | GPU 型号、compute capability/target arch、设备数量 |
| 软件 | 驱动、CUDA/ROCm、TileLang、PyTorch、编译后端 |
| 计算 | 目标 dtype 下的 dense/sparse 峰值口径，是否含 sparsity |
| 内存 | 显存类型与带宽、L2 容量、每 SM/block 可用 shared memory |
| 执行 | warp/wavefront 宽度、最大线程/block、寄存器和 occupancy 限制 |
| 测量 | 实际时钟/功耗状态、warmup、rep、计时 backend |

这里顺带澄清一个常见混淆：HBM/GDDR 是显存的物理实现，不是 CUDA 代码中的独立通用作用域。
代码和数据流图使用 global/device memory，性能报告再注明物理显存类型。

## 7. 常见报错与解决

报错往往比文档更能定位问题。下面的表按"报错/症状 → 原因与解法"组织，命中相似报错时
先看左列，再到右列找第一步该做的排查动作：

| 报错/症状 | 原因与解法 |
|---|---|
| `Ramp of more than 4 lanes is not allowed` | 向量化宽度过宽（8 lane）；检查 T.copy/T.Parallel 的宽度提示或改布局 |
| shared memory 超限 | tile×stages 太大；降 BK/stages 或 tile |
| `T.gemm` 布局不匹配报错 | A_s/B_s/C_f 布局冲突；`annotate_layout` 显式指定 |
| 结果错在边界 tile | 越界读写或尾部 tile 未零填充；输出用 guard，GEMM 输入明确 padding。若依赖哨兵值填充：直接打印边界 tile 内容，并 grep 生成源码里是否出现该常量（历史 issue #2543：whole-tile `T.copy` 曾静默忽略 `annotate_safe_value`） |
| `async_copy` 结果错 | 忘了 `T.ptx_wait_group`；或屏障时机不对 |
| autotune "No configurations" | configs 为空/过滤过头；检查生成器 |
| 动态形状 autotune 报错 | 内置生成器需静态形状；改用 `set_autotune_inputs` |
| 归约类算子正确但极慢 | 很可能用了 `T.serial` 归约：换 `T.reduce`，再用 Nsight 看 `warp_stall_long_scoreboard` 是否归零 |

### 工具选择

当报错无法一眼定位时，按目标选对工具往往事半功倍。这张表给出每种目标的第一工具：

| 目标 | 第一工具 |
|---|---|
| 无 GPU 检查生成路径 | `python -m tilelang.tools.compile_only` |
| 查看 pass 首次改变 IR 的位置 | `TL_LOWER_TRACE=html` |
| 缩小稳定失败程序 | `python -m tilelang.autodd` |
| 查看线程—元素布局 | `tilelang.tools.plot_layout` |
| 估算已识别 GEMM FLOPs/全局 copy 字节 | `tilelang.tools.Analyzer` |
| CUDA 越界/竞态 | Compute Sanitizer |
| 时间线/单 kernel 指标 | Nsight Systems / Nsight Compute |

## 8. 面试 30 秒自我介绍稿（模板）

最后这一节服务于面试前的临场速记。下面是一段 30 秒自我介绍的模板，把「算子 → 方法 →
数据 → 验证」压缩成一段话，填空即可：

「我使用 TileLang（基于 TVM TIR 的 GPU 内核 DSL）做过 [算子名]：从朴素版本到
分块 + 软件流水线 + swizzle 布局 + 自动调优，最终 [X] TFLOPS，达到硬件峰值的
[Y]%，与 cuBLAS 对比 [结论]。我的方法：先判断 memory/compute bound，再画数据流，
逐项优化并用 profiler 验证，最后用 Nsight 归因剩余差距。」

---

到这里，速查表就用完了。请记住它的边界：查表解决的是"此刻怎么查"，而不是"有没有学会"。
用完这张表之后回到第 14 章完成 Capstone——只有代码、测试、测量、归因和适用范围同时齐全，
这个内核才算达到课程验收标准。
