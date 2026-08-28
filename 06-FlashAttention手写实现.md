# 第 06 章 FlashAttention 手写实现

> **本章目标**：先从朴素 Attention 的存储问题出发，推导在线 softmax，再把公式映射到
> 两个 tile GEMM 和一个分块循环。学习重点是「为什么这样更新仍然等价」，而不是先背
> 一份长代码；代码验证通过后，再讨论 IO、反向重计算和变体。

**学习信息**

- 难度：高级；预计用时：5–8 小时；
- 前置：第 03～05 章，理解分块 GEMM、归约、mask 和 fp16/fp32 混合精度；
- 运行范围：完整前向示例需要支持 `T.gemm`/fragment 的 GPU；先用小尺寸做正确性，不要直接运行大尺寸参考实现；
- 本章产出：online softmax 推导、一份小尺寸 causal/non-causal 校验，以及 IO 复杂度解释。
- 参考：[官方 FlashAttention examples](https://github.com/tile-ai/tilelang/tree/main/examples/flash_attention)、
  [DeepSeek MLA 文档](https://tilelang.com/deeplearning_operators/deepseek_mla.html) 和
  [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)。

**阅读路线：**先把 `S → P → O` 的数据形状写清楚；再只学习 `(m, l, O)` 三个在线状态；
随后逐段阅读实现；最后分开做「整除尺寸正确性」和「尾部尺寸设计」。不要把 dense reference
能否运行、kernel 是否正确、kernel 是否高效混成一个问题。

## 6.1 朴素 Attention 为什么慢

```text
S = Q @ K^T / sqrt(d)      # [B,H,N,N]，N² 张量
P = softmax(S, dim=-1)     # 需要整行才能归一化！
O = P @ V                  # [B,H,N,d]
```

两个致命问题：

1. **显存爆炸**：`S` 和 `P` 都是 `N×N`，序列 32K 时单头仅一个 fp16 中间张量就约
   2 GiB；多头/多 batch 更快放大；
2. **IO 浪费**：softmax 的归一化分母需要看到整行，所以 `Q@K^T` 的结果必须先写回
   HBM，`P@V` 时再读回来——**同一份中间数据被存取两遍**。

朴素实现会把 O(N²) 的中间结果写回/读回 HBM，而计算量是 O(N²·d)。FlashAttention
的解法是**分块 + 在线 softmax**：把 N 维切成块，在片上滚动维护「部分归一化」，不把
完整 S/P 落盘。它不是把所有 Attention 的 HBM IO 变成 O(N)；更准确的 IO 上界依赖
片上存储容量，常见表达为 O(N²·d²/M)，并额外包含 Q/K/V/O 的线性项。

核心结论是：FlashAttention 减少了 HBM 往返，并避免把完整的 `S/P` 中间矩阵落盘；它
是否更快、是否受访存限制，仍取决于序列长度、head_dim、GPU、实现和 profiler 结果。

## 6.2 在线 softmax：数学推导（必会）

对一行 logits `s_1..s_N`，softmax 需要 `m = max s_j` 和 `l = Σ exp(s_j − m)`。

分块处理：假设已处理前 `t` 块，维护：

```text
m_prev = 当前 max（前 t 块）
l_prev = 当前 exp 和（已用 m_prev 归一）
```

来了新块，局部统计 `m_local`、`l_local`。合并公式：

```text
m_new   = max(m_prev, m_local)
l_new   = l_prev · exp(m_prev − m_new) + l_local · exp(m_local − m_new)
```

输出也要重缩放：老块的每个输出行向量 `O_prev` 乘 `exp(m_prev − m_new)`。
推导一句话：**归一化变了，分母换了，旧的分子乘上「换分母的修正因子」就行**——不用
重读任何东西。

TileLang 的实现里（官方 `example_mha_fwd_bshd.py`），每轮做：

```python
# 先保存旧 max，再算新 max
T.copy(scores_max, scores_max_prev)
T.fill(scores_max, -T.infinity(accum_dtype))
T.reduce_max(acc_s, scores_max, dim=1, clear=False)
for i in T.Parallel(block_M):
    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

# 缩放因子 scale = 旧 max 调整 → 新 max 调整（融合 log2e 进来）
for i in T.Parallel(block_M):
    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

# scores → softmax 概率（按新 max 归一）
for i, j in T.Parallel(block_M, block_N):
    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

# 更新分母 logsum = l_prev·scale + l_local
T.reduce_sum(acc_s, scores_sum, dim=1)
for i in T.Parallel(block_M):
    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

# 老输出重缩放
for i, j in T.Parallel(block_M, dim):
    acc_o[i, j] *= scores_scale[i]
```

> 细节：`scale = (1/√d)·log2(e)`，配合 `T.exp2` 把 `/√d` 和 `exp` 都变成
> `exp2(x·scale)` 一次计算——把缩放和指数变换放到同一表达式中。它是一个值得通过
> 生成代码和数值测试确认的实现细节，不应脱离目标后端夸大收益。

## 6.3 TileLang 实现：完整代码

以下代码精简自官方示例（去掉 autotune 装饰器，便于学习；`@tilelang.jit` 保留）。为了
让主线先聚焦算法，代码先以整除 tile 的小尺寸为验证目标；尾部 query/KV tile 的 guard、
padding 和输出写回需要在通过主线后单独补齐。

```python
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang import jit

@jit(out_idx=[3])   # 第 4 个参数 Output 由宿主自动分配并返回
def flashattn(batch: int, heads: int, seq_len: int, dim: int,
              is_causal: bool = False,
              block_M: int = 128, block_N: int = 128,
              num_stages: int = 1, threads: int = 128):
    scale = (1.0 / dim) ** 0.5 * 1.44269504      # 1/sqrt(d) * log2(e)
    shape = (batch, seq_len, heads, dim)         # Q/K/V/Output shape: [B, N, H, D]，bshd 布局
    dtype = 'float16'
    accum_dtype = 'float32'

    @T.prim_func
    def main(
        Q: T.Tensor((batch, seq_len, heads, dim), dtype),      # shape: [B, N, H, D]
        K: T.Tensor((batch, seq_len, heads, dim), dtype),      # shape: [B, N, H, D]
        V: T.Tensor((batch, seq_len, heads, dim), dtype),      # shape: [B, N, H, D]
        Output: T.Tensor((batch, seq_len, heads, dim), dtype), # shape: [B, N, H, D]
    ):
        # 网格：(seq/block_M, heads, batch)；一个 block 算一行 (block_M 个 query)
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch,
                      threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared((block_M, dim), dtype)       # shape: [BM, D]
            K_shared = T.alloc_shared((block_N, dim), dtype)       # shape: [BN, D]
            V_shared = T.alloc_shared((block_N, dim), dtype)       # shape: [BN, D]
            O_shared = T.alloc_shared((block_M, dim), dtype)       # shape: [BM, D]

            acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)  # shape: [BM, BN]，S 块
            acc_s_cast = T.alloc_fragment((block_M, block_N), dtype)   # shape: [BM, BN]，喂 GEMM 要 fp16
            acc_o = T.alloc_fragment((block_M, dim), accum_dtype)      # shape: [BM, D]，P@V 累加
            scores_max = T.alloc_fragment((block_M,), accum_dtype)     # shape: [BM]
            scores_max_prev = T.alloc_fragment((block_M,), accum_dtype) # shape: [BM]
            scores_scale = T.alloc_fragment((block_M,), accum_dtype)    # shape: [BM]
            scores_sum = T.alloc_fragment((block_M,), accum_dtype)      # shape: [BM]
            logsum = T.alloc_fragment((block_M,), accum_dtype)          # shape: [BM]

            # 把本 block 的 Q tile 一次拷进共享内存（复用 block_N 轮）
            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # causal 时 kv 只需要扫到 (bx+1)*block_M
            loop_range = (
                T.min(T.ceildiv(seq_len, block_N),
                      T.ceildiv((bx + 1) * block_M, block_N))
                if is_causal else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)

                # 初始化 S 块：causal 下三角掩码或尾部 padding
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j, 0,
                            -T.infinity(acc_s.dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            k * block_N + j >= seq_len,
                            -T.infinity(acc_s.dtype), 0)

                # S 块 = Q_shared @ K_shared^T（转置 B）
                T.gemm(Q_shared, K_shared, acc_s,
                       transpose_B=True, clear_accum=False,
                       policy=T.GemmWarpPolicy.FullRow)

                # ---- 在线 softmax 更新（见 6.2）----
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)          # fp32 → fp16 再喂 GEMM

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]   # 老输出重缩放

                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # 收尾：除以 logsum → 写回（shared 中转再拷到全局）
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main
```

### 6.3.1 逐段解读

| 片段 | 作用 | 关键点 |
|---|---|---|
| 网格 `(seq/block_M, heads, batch)` | 一个 block 处理 `block_M` 个 query | `bz` 是 batch、`by` 是头 |
| `Q_shared` 一次拷入 | Q 每轮复用 | 比每轮重读省 block_N 倍 Q 访存 |
| 掩码初始化 `acc_s` | causal / padding | `−∞` 在 exp2 后变 0 |
| `T.gemm(..., transpose_B=True)` | S = Q·Kᵀ | K 是 `[block_N, dim]`，转置后 `[block_M,block_N]` |
| max/scale/exp2/sum 序列 | 在线归一 | `exp2(x·scale)` 融合 `1/√d` |
| `acc_o *= scores_scale` | 老输出换分母 | **这就是不重读 O 的关键** |
| 收尾 `/= logsum` | 最后除一次 | 全程只有一次除法 |

### 6.3.2 运行与验证

```python
# 先用整除 tile 的小尺寸做正确性；dense reference 会显式创建 N×N scores，不能任意放大 N。
B, H, N, D = 1, 2, 256, 64
Q = torch.randn(B, N, H, D, device='cuda', dtype=torch.float16)   # shape: [B, N, H, D]
K = torch.randn(B, N, H, D, device='cuda', dtype=torch.float16)   # shape: [B, N, H, D]
V = torch.randn(B, N, H, D, device='cuda', dtype=torch.float16)   # shape: [B, N, H, D]

kernel = flashattn(B, H, N, D, is_causal=True)
O = kernel(Q, K, V)                      # shape: [B, N, H, D]；out_idx=[3]，返回输出

# 参考实现（einsum + 因果掩码 + softmax）
scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) / (D ** 0.5)  # shape: [B, H, N, N]
mask = torch.tril(torch.ones(N, N, device='cuda')).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, N, N]
scores = scores.masked_fill(mask == 0, float('-inf'))       # shape: [B, H, N, N]
P = F.softmax(scores, dim=-1)                               # shape: [B, H, N, N]
ref = torch.einsum('bhqk,bkhd->bqhd', P, V)                 # shape: [B, N, H, D]
torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)

lat = kernel.get_profiler().do_bench()
print(f"{2 * 2 * B * H * N * N * D / lat / 1e9:.1f} TFLOPS")

# 性能实验时，再单独选择显存能承受的 N，并把 reference 与 kernel 的计时分开。
```

## 6.4 反向传播：为什么「重计算」

反向需要 `P` 和 `S`，但前向从不落盘。FlashAttention 的做法（官方 bwd 示例
`example_mha_bwd_bshd.py` 等）：**反向时用同样的分块循环重算 S/P**（用保存的
`logsum` 或 `lse`），再算 `dV、dK、dQ`。代价是额外重计算，具体 FLOPs 比例取决于
实现和融合方式，换来 O(N²) 中间张量的显存节省与更少的 HBM 往返——在长序列上通常值得。
解释反向时，应同时说明重计算、显存占用和实际运行时间之间的权衡。

## 6.5 变长序列（varlen）：从 padding 到 packed layout

前面的实现假设一个 batch 中所有样本都有相同的 `seq_len`。真实的训练和推理通常不是这样：
同一个 batch 里的句子长度可能分别是 4096、1732、287，若仍然把它们补齐到 4096，
Attention 会对大量 padding token 做无效计算。

varlen 的核心是改变输入的**存储布局和索引方式**，Attention 公式本身不变：

1. 去掉每条序列末尾的 padding，把有效 token 沿序列维拼接成一块；
2. 用 `cu_seqlens` 记录每条序列在这块连续内存中的起止偏移；
3. kernel 仍然按「一个 batch 样本 + 一个 query tile」计算，但通过偏移量找到该样本自己的 Q/K/V；
4. 最终输出仍按 packed 顺序写回，必要时再由宿主函数恢复成 padded layout。

官方示例可参考 [example_mha_fwd_varlen.py](https://github.com/tile-ai/tilelang/blob/main/examples/flash_attention/example_mha_fwd_varlen.py)
和 [varlen_utils.py](https://github.com/tile-ai/tilelang/blob/main/examples/flash_attention/varlen_utils.py)。
下面先建立数据模型，再看索引、mask、TileLang 内核和验证方法。这样读者不会把
`cu_seqlens`、实际长度和 `max_seqlen` 混为一谈。

### 6.5.1 为什么要从 dense 改成 packed

等长版本的张量通常是：

```text
Q: [B, Nq, H, D]       # 所有样本都按同一个 Nq 分配
K: [B, Nk, Hkv, D]     # Hkv 可以等于 H，也可以用于 GQA/MQA
V: [B, Nk, Hkv, D]
O: [B, Nq, H, D]
```

如果第 `b` 条样本的有效 query 长度是 `Lq[b]`，有效 key/value 长度是 `Lk[b]`，
varlen 会把它们改成：

```text
Q_unpad: [total_q, H, D]       # total_q = Σ_b Lq[b]
K_unpad: [total_k, Hkv, D]     # total_k = Σ_b Lk[b]
V_unpad: [total_k, Hkv, D]
O_unpad: [total_q, H, D]
```

每条序列的边界保存在两个前缀和数组中：

```text
cu_seqlens_q: [B + 1]，int32  # query 的前缀和
cu_seqlens_k: [B + 1]，int32  # key/value 的前缀和
```

对样本 `b`：

```text
q_start = cu_seqlens_q[b]
q_end   = cu_seqlens_q[b + 1]
Lq      = q_end - q_start

k_start = cu_seqlens_k[b]
k_end   = cu_seqlens_k[b + 1]
Lk      = k_end - k_start
```

因此，原来 dense 布局中的 `Q[b, i, h, d]`，在 packed 布局中变成
`Q_unpad[q_start + i, h, d]`；`K` 和 `V` 同理。注意：`cu_seqlens` 存的是**边界偏移**，
不是每条序列的长度；长度必须通过相邻元素相减得到。

### 6.5.2 用一个小例子读懂 `cu_seqlens`

设 batch 中有两条样本：

```text
Lq = [5, 3]                  # query 长度
Lk = [5, 4]                  # key/value 长度
cu_seqlens_q = [0, 5, 8]     # 第 0 条占 [0, 5)，第 1 条占 [5, 8)
cu_seqlens_k = [0, 5, 9]     # 第 0 条占 [0, 5)，第 1 条占 [5, 9)
```

于是：

| batch `b` | `q_start:q_end` | `Lq` | `k_start:k_end` | `Lk` |
|---:|---:|---:|---:|---:|
| 0 | `0:5` | 5 | `0:5` | 5 |
| 1 | `5:8` | 3 | `5:9` | 4 |

第二条样本的第 2 个 query 是 `Q_unpad[5 + 2]`，不是
`Q_unpad[1 * max_seqlen + 2]`。这正是 varlen 内核和普通 padded 内核最容易混淆的地方。

宿主侧构造前缀和的最小实现如下：

```python
import torch


def make_cu_seqlens(lengths, device):
    # lengths: [B]，每条样本的有效 token 数
    lengths = torch.as_tensor(lengths, dtype=torch.int32, device=device)  # shape: [B]
    cu_seqlens = torch.zeros(lengths.numel() + 1, dtype=torch.int32, device=device)  # shape: [B + 1]
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)  # shape: [B]，写入前缀和
    return cu_seqlens  # shape: [B + 1]


cu_q = make_cu_seqlens([5, 3], device="cuda")  # shape: [3]，值为 [0, 5, 8]
cu_k = make_cu_seqlens([5, 4], device="cuda")  # shape: [3]，值为 [0, 5, 9]
```

需要同时满足：`cu_seqlens[0] = 0`、单调不减、最后一个元素等于 packed token 总数，
并且通常使用 `int32` 且放在和 Q/K/V 相同的设备上。长度为 0 的样本可以存在，
但必须提前决定它的输出语义；教学实现一般先用正长度样本验证，再补零长度边界。

### 6.5.3 从 dense 坐标到 packed 坐标

把索引关系写成公式，内核就容易读懂了。对 batch 样本 `b`、局部 query 行 `i`、局部 key 行 `j`：

```text
q_pack = cu_seqlens_q[b] + i
k_pack = cu_seqlens_k[b] + j

Q_unpad[q_pack, h, d] = 第 b 条序列的 Q[i, h, d]
K_unpad[k_pack, h, d] = 第 b 条序列的 K[j, h, d]
V_unpad[k_pack, h, d] = 第 b 条序列的 V[j, h, d]
O_unpad[q_pack, h, d] = 第 b 条序列的输出 O[i, h, d]
```

一个 varlen query block 通常由三类坐标共同确定：

| 坐标 | 含义 | 用途 |
|---|---|---|
| `bz` | batch 样本 `b` | 读取 `cu_seqlens_q[b]` 和 `cu_seqlens_k[b]` |
| `bx` | 局部 query tile 编号 | `q_local = bx * block_M + i` |
| `by` | head 编号 | 读取对应的 Q/K/V head |

在得到 `q_start`、`k_start` 后，普通 FlashAttention 的两层循环仍然保留，只是全局
切片从 `bx * block_M` 改成 `q_start + bx * block_M`，从 `k * block_N` 改成
`k_start + k * block_N`。网格的 query 方向使用 `max_seqlen_q` 作为上界，而不是
把所有样本都错误地当成 `max_seqlen_q` 长度。

### 6.5.4 TileLang 内核骨架：偏移量放在哪里

下面是根据官方 varlen 示例整理的教学骨架。它保留了本章前面已经讲过的
`QKᵀ → online softmax → PV` 主线，新增的只有四件事：读取前缀和、使用局部长度、
处理 query/key 尾部、把输出写回 packed 偏移处。

这里的 `UQ`、`UKV` 表示本次编译实例的 packed 总长度；`max_seqlen_q` 和
`max_seqlen_k` 是长度上界。TileLang 不同版本对动态 shape、边界 copy 和 pipeline
的支持可能有差异，因此请把这段当成**内核结构示意**，实际运行时以当前仓库官方示例
为准，不要只复制接口名称。

```python
import tilelang
import tilelang.language as T
from tilelang import jit


@jit(out_idx=[7])  # Output_unpad 是第 8 个参数，由宿主侧自动分配并返回
def flashattn_varlen(batch_size: int, UQ: int, UKV: int, heads: int, dim: int,
                     is_causal: bool, block_M: int = 64, block_N: int = 64,
                     num_stages: int = 1, threads: int = 128):
    # UQ/UKV 是 packed 后的总 token 数；它们不是 batch 内的最大序列长度。
    q_shape = (UQ, heads, dim)       # Q_unpad shape: [total_q, H, D]
    k_shape = (UKV, heads, dim)      # K_unpad shape: [total_k, H, D]
    v_shape = (UKV, heads, dim)      # V_unpad shape: [total_k, H, D]
    o_shape = (UQ, heads, dim)       # Output_unpad shape: [total_q, H, D]
    dtype = T.float16
    accum_dtype = T.float32
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # 1/sqrt(D) * log2(e)

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),                     # shape: [total_q, H, D]
        K_unpad: T.Tensor(k_shape, dtype),                     # shape: [total_k, H, D]
        V_unpad: T.Tensor(v_shape, dtype),                     # shape: [total_k, H, D]
        cu_seqlens_q: T.Tensor((batch_size + 1,), T.int32),     # shape: [B + 1]
        cu_seqlens_k: T.Tensor((batch_size + 1,), T.int32),     # shape: [B + 1]
        max_seqlen_q: T.int32,                                 # scalar：所有 q 长度的最大值
        max_seqlen_k: T.int32,                                 # scalar：所有 k 长度的最大值
        Output_unpad: T.Tensor(o_shape, dtype),                 # shape: [total_q, H, D]
    ):
        # bx 是局部 query tile，by 是 head，bz 是 batch 样本。
        with T.Kernel(T.ceildiv(max_seqlen_q, block_M), heads, batch_size,
                      threads=threads) as (bx, by, bz):
            # 先查边界；所有后续地址都基于当前样本自己的 start 偏移。
            q_start = cu_seqlens_q[bz]       # scalar：当前样本 Q 的 packed 起点
            q_end = cu_seqlens_q[bz + 1]     # scalar：当前样本 Q 的 packed 终点
            k_start = cu_seqlens_k[bz]       # scalar：当前样本 K/V 的 packed 起点
            k_end = cu_seqlens_k[bz + 1]     # scalar：当前样本 K/V 的 packed 终点
            q_len = q_end - q_start          # scalar：当前样本实际 query 长度
            k_len = k_end - k_start          # scalar：当前样本实际 key/value 长度

            Q_shared = T.alloc_shared((block_M, dim), dtype)      # shape: [BM, D]
            K_shared = T.alloc_shared((block_N, dim), dtype)      # shape: [BN, D]
            V_shared = T.alloc_shared((block_N, dim), dtype)      # shape: [BN, D]
            O_shared = T.alloc_shared((block_M, dim), dtype)      # shape: [BM, D]
            acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)       # shape: [BM, BN]
            acc_s_cast = T.alloc_fragment((block_M, block_N), dtype)        # shape: [BM, BN]
            acc_o = T.alloc_fragment((block_M, dim), accum_dtype)            # shape: [BM, D]
            scores_max = T.alloc_fragment((block_M,), accum_dtype)           # shape: [BM]
            scores_max_prev = T.alloc_fragment((block_M,), accum_dtype)      # shape: [BM]
            scores_scale = T.alloc_fragment((block_M,), accum_dtype)         # shape: [BM]
            scores_sum = T.alloc_fragment((block_M,), accum_dtype)            # shape: [BM]
            logsum = T.alloc_fragment((block_M,), accum_dtype)               # shape: [BM]

            # q_base 是本 tile 在 packed Q 中的起点；q_local 仍是当前样本内的局部坐标。
            q_base = q_start + bx * block_M
            T.copy(Q_unpad[q_base : q_base + block_M, by, :], Q_shared)  # 越界行由下方 guard 处理
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # 右对齐 causal：q_local + (k_len - q_len) >= k_local 才允许访问。
            offset = k_len - q_len
            loop_range = (
                T.min(T.ceildiv(T.max(0, offset + (bx + 1) * block_M), block_N),
                      T.ceildiv(k_len, block_N))
                if is_causal else T.ceildiv(k_len, block_N)
            )

            for k_block in T.Pipelined(loop_range, num_stages=num_stages):
                k_base = k_start + k_block * block_N
                T.copy(K_unpad[k_base : k_base + block_N, by, :], K_shared)  # shape: [BN, D]

                # 先把无效 q/k 位置设成大负数，再做 QK^T；无效位置的权重最终为 0。
                for i, j in T.Parallel(block_M, block_N):
                    q_local = bx * block_M + i
                    k_local = k_block * block_N + j
                    invalid = (q_local >= q_len or k_local >= k_len)
                    if is_causal:
                        invalid = invalid or (q_local + offset < k_local)
                    acc_s[i, j] = T.if_then_else(invalid, -1e9, 0)

                # S = Q_shared @ K_shared^T；acc_s 保留上面初始化的 mask 偏置。
                T.gemm(Q_shared, K_shared, acc_s,
                       transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # 下面与等长版本完全相同：在线 softmax 合并当前 K tile。
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V_unpad[k_base : k_base + block_N, by, :], V_shared)  # shape: [BN, D]
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # 只对有效 query 行归一化并写回，避免把 padding 行写进 Output_unpad。
            for i, d in T.Parallel(block_M, dim):
                q_local = bx * block_M + i
                if q_local < q_len:
                    # Lq > Lk 时，右对齐 causal 可能让最前面的 query 没有可见 key。
                    acc_o[i, d] = (0 if is_causal and q_local + offset < 0
                                   else acc_o[i, d] / logsum[i])
            T.copy(acc_o, O_shared)
            for i, d in T.Parallel(block_M, dim):
                q_local = bx * block_M + i
                if q_local < q_len:
                    Output_unpad[q_start + q_local, by, d] = O_shared[i, d]

    return main
```

读这段代码时，优先检查四个地址是否正确，而不是先纠结 autotune：

| 位置 | 等长版本 | varlen 版本 |
|---|---|---|
| Q 起点 | `bx * block_M` | `q_start + bx * block_M` |
| K 起点 | `k_block * block_N` | `k_start + k_block * block_N` |
| query 有效长度 | 统一的 `seq_len` | 当前样本的 `q_len` |
| 输出地址 | `Output[b, row, h, d]` | `Output_unpad[q_start + row, h, d]` |

在线 softmax 本身没有变：varlen 只是让每个 batch 样本拥有自己的 `q_len`、`k_len` 和
packed 起点。也就是说，**变的是数据访问边界，不是 softmax 的数学状态**。

注意，示意代码中的整块 `T.copy` 依赖当前 TileLang 版本和目标后端对越界 slice 的安全处理。
如果你的版本不会自动做 predicated copy，就必须改成带 mask 的 copy、先把输入 pad 到 tile
边界，或用显式循环保护每一个全局地址；仅仅在后面的 softmax mask 中标记无效位置，不能
阻止一次已经发生的越界加载。

### 6.5.5 causal mask：长度不同时必须先选对齐语义

当 `Lq = Lk` 时，最直观的 causal 条件是：

```text
q_local >= k_local
```

但在推理 decode 中，query 可能只有最近的 `Lq` 个 token，而 K/V 已经包含 `Lk` 个历史
token，此时直接比较局部坐标会把 query 错误地当成从序列开头开始。常见的右对齐语义是：

```text
offset = Lk - Lq
允许访问 ⇔ q_local + offset >= k_local
```

例如 `Lq=2、Lk=5` 时，两个 query 对应完整序列中的位置 3、4，最后一个 query 才能
看到全部 0～4 的 key。TileLang 官方 varlen 示例也采用了这个右对齐条件。

这里有三个边界必须单独处理：

1. **query 尾部**：`q_local >= q_len` 的行只是在 tile 中占位，不能参与 softmax，也不能写回；
2. **key 尾部**：`k_local >= k_len` 的列必须是无效列，权重应为 0，K/V 的越界加载不能影响结果；
3. **没有可见 key 的 query**：当 `Lq > Lk` 且使用右对齐 causal 时，最前面的 query
   可能一行都看不到。此时不能直接做 `0 / logsum`，应按约定把输出置 0，或在参考实现中
   显式处理这一类行。

`q_len` 和 `k_len` 不同并不自动意味着某一种 mask 语义。训练、prefill、decode 以及
带 KV cache 的实现可能采用不同的对齐约定；教程中的公式必须和验证参考、实际框架保持一致。

### 6.5.6 如何验证：先按序列比较，再比较 packed 输出

不要把所有 packed token 直接和一个 padded Attention 结果比较。正确做法是：

1. 为每条样本分别生成 Q/K/V 和 dense reference；
2. 在每条样本内部完成 causal/non-causal Attention；
3. 按相同顺序把各条输出拼接成 `ref_unpad`；
4. 用 `torch.testing.assert_close` 比较 kernel 的 `out_unpad` 和 `ref_unpad`；
5. 如果还需要 padded 输出，再单独执行 unpack，并检查每条样本的有效区间。

下面的参考实现覆盖了不同的 `Lq/Lk`，并显式处理「没有可见 key」的 query。所有中间 tensor
的形状都写在注释中，便于对照 kernel 的 `[BM, BN]` 和 `[BM, D]` fragment。

```python
import torch
import torch.nn.functional as F


def pack_sequences(sequences):
    # sequences 是若干个 [L_b, H, D] tensor；每条样本的 L_b 可以不同。
    packed = torch.cat(sequences, dim=0)  # shape: [Σ_b L_b, H, D]
    return packed


def varlen_reference(q_list, k_list, v_list, causal=False):
    # 返回 packed 输出；q_list[b] shape: [Lq_b, H, D]
    # k_list[b] / v_list[b] shape: [Lk_b, H, D]
    outputs = []
    for q, k, v in zip(q_list, k_list, v_list):
        q_len, heads, dim = q.shape
        k_len = k.shape[0]
        scores = torch.einsum("qhd,khd->hqk", q, k) / (dim ** 0.5)  # shape: [H, Lq, Lk]

        if causal:
            # 右对齐：q_local + (Lk - Lq) >= k_local。
            q_pos = torch.arange(q_len, device=q.device)[:, None]  # shape: [Lq, 1]
            k_pos = torch.arange(k_len, device=k.device)[None, :]  # shape: [1, Lk]
            valid = q_pos + (k_len - q_len) >= k_pos  # shape: [Lq, Lk]
            scores = scores.masked_fill(~valid[None, :, :], float("-inf"))  # shape: [H, Lq, Lk]

            # 对完全没有可见 key 的 query，先用安全 logits 做 softmax，最后把输出清零。
            has_key = valid.any(dim=-1)  # shape: [Lq]
            safe_scores = torch.where(has_key[None, :, None], scores, torch.zeros_like(scores))
            probs = F.softmax(safe_scores, dim=-1)  # shape: [H, Lq, Lk]
            probs = probs * has_key[None, :, None]  # shape: [H, Lq, Lk]
        else:
            probs = F.softmax(scores, dim=-1)  # shape: [H, Lq, Lk]

        output = torch.einsum("hqk,khd->qhd", probs, v)  # shape: [Lq, H, D]
        outputs.append(output)

    return torch.cat(outputs, dim=0)  # shape: [total_q, H, D]


# 两条样本的 query/key 长度不同，专门覆盖 cu_seqlens 和 causal 对齐边界。
device = "cuda"
heads, dim = 2, 16
lengths_q = [5, 3]  # shape: [B] 的宿主侧长度列表
lengths_k = [5, 4]  # shape: [B] 的宿主侧长度列表
q_list = [torch.randn(length, heads, dim, device=device, dtype=torch.float16) for length in lengths_q]
k_list = [torch.randn(length, heads, dim, device=device, dtype=torch.float16) for length in lengths_k]
v_list = [torch.randn(length, heads, dim, device=device, dtype=torch.float16) for length in lengths_k]
q_unpad = pack_sequences(q_list)  # shape: [8, H, D]
k_unpad = pack_sequences(k_list)  # shape: [9, H, D]
v_unpad = pack_sequences(v_list)  # shape: [9, H, D]
cu_q = torch.tensor([0, 5, 8], device=device, dtype=torch.int32)  # shape: [B + 1]
cu_k = torch.tensor([0, 5, 9], device=device, dtype=torch.int32)  # shape: [B + 1]

# kernel 的返回值：Output_unpad shape: [8, H, D]
kernel = flashattn_varlen(2, 8, 9, heads, dim, is_causal=True)
out_unpad = kernel(q_unpad, k_unpad, v_unpad, cu_q, cu_k, 5, 5)  # shape: [8, H, D]
ref_unpad = varlen_reference(q_list, k_list, v_list, causal=True)  # shape: [8, H, D]
torch.testing.assert_close(out_unpad, ref_unpad, rtol=1e-2, atol=1e-2)
```

第一次调试建议把 `block_M=2、block_N=2、heads=1、dim=4`，并打印每条序列的
`q_start/q_end/k_start/k_end`。如果 packed 结果整体错位，优先检查 offset；如果只有 tile
尾部错，检查 `q_local/k_local` 的 guard；如果 causal 只在 `Lq != Lk` 时出错，检查右对齐
公式，而不是先怀疑 online softmax。

### 6.5.7 varlen 的性能收益与测量方式

varlen 主要减少 padding 带来的无效 token 和无效 query-key 配对，但它不保证任何输入分布下
都一定更快。至少要同时报告三类量：

- **token 利用率**：`total_q / (B · max_seqlen_q)`，以及 `total_k / (B · max_seqlen_k)`；
- **Attention 配对利用率**：`Σ_b (Lq[b] · Lk[b]) / (B · max_seqlen_q · max_seqlen_k)`；
- **实际时间**：分别测 kernel-only，以及包含 pack/unpack、`cu_seqlens` 构造的端到端时间。

以 `Lq=[5,3]、Lk=[5,4]` 为例：

```text
total_q = 8，B · max(Lq) = 2 · 5 = 10
token 利用率 = 8 / 10 = 80%

实际 query-key 对 = 5·5 + 3·4 = 37
等长 padded 对 = 2·5·5 = 50
Attention 配对利用率 = 37 / 50 = 74%
```

但 packed kernel 的网格仍按 `ceildiv(max_seqlen_q, block_M)` 发射。若长度分布非常离散，
短序列所在 block 的尾部仍会有浪费；如果 `max_seqlen` 极大而平均长度很短，还要考虑按长度
分桶、选择更小的 tile，或采用专门的调度策略。另一个现实权衡是：packed 布局减少了计算和
显存，但 pack/unpack 本身也需要 kernel 和带宽；在短序列、小 batch 上，这部分开销可能抵消
收益。

### 6.5.8 其他变体：放在 varlen 之后理解

- **GQA/MQA**：多个 query head 共享 K/V head。它改变的是 head 映射和 KV 读取，不能替代
  `cu_seqlens`；两者可以组合成 GQA-varlen。
- **Flash Decoding**：decode 阶段一个 query 要访问很长的历史 K/V，可把 K/V 维度切给
  多个 block，最后用 lse 合并 partial result；核心仍然是在线 softmax 的可结合更新。
- **反向 varlen**：除了前向的 `cu_seqlens_q/k`，还要保证梯度输出与 packed Q/K/V 的
  顺序一致；重计算的每个 tile 必须复用同一套边界和 causal 对齐规则。
- **MLA（Multi-head Latent Attention，DeepSeek）**：见下面的专节——它是本章所有概念
  （在线 softmax、fragment、`GemmWarpPolicy`、warp specialization）的汇合点。

### 6.5.8.1 MLA decode：寄存器压力如何决定 warp 分工

一篇社区工程文章（[Writing High-Performance Kernels in TileLang, from GEMM to MLA](https://huggingface.co/blog/AtlasCloud-AI/writing-high-performance-kernels-in-tilelang)）
把 MLA 讲成了本章概念的总复习，核心难点不是数学而是**寄存器预算**：

1. **形状**：MLA 的 query/key 宽 576（512 的 no-pe 部分 + 64 的 rope 部分）、value 宽 512，
  所以输出累加器 `acc_o = [block_M, 512]` 必须在整个 KV 循环期间常驻寄存器。
2. **硬件约束**：Hopper 的 `wgmma.mma_async` 把 4 个 warp（128 线程）绑成一个 warpgroup，
  且要求 M ≥ 64。于是一个 warpgroup 至少要持有 64×512 的累加器——放不下，寄存器溢出后性能断崖。
3. **解法**：`T.gemm` 的 `policy=T.GemmWarpPolicy.FullCol` 把输出沿 dim 维切给两个
   warpgroup（各持 64×256），同时该 policy 把约束**反向传播**：`P@V` 的每个 warpgroup
   需要完整的 `acc_s`，于是 staging buffer `S_shared` 也必须是 `[BM, BN]`，而 `Q@K` 阶段
   每个 warpgroup 只算一半的 score slab。两个 warpgroup 各自写出一半 `acc_s`，再经
   `S_shared` 交换补齐——**这一段数据交换由布局推断自动生成，你只需要选 policy、写数学**。
4. **对照**：同样的设计在 CuTe 里要手写布局、swizzle、Tensor Core 对齐和
   producer/consumer 同步；在 TileLang 里约 80 行，官方参考实现在论文评测中
   （H100，fp16，batch 64/128）达到 FlashMLA 约 98% 的性能。

论文 [TileLang 论文](https://arxiv.org/abs/2504.17577) 还记录了 MLA 对 PyTorch 约
1075.9× 的加速（H100，相同评测设置）。这些数字只对论文/文章公开的环境成立；你的设备上
应重跑官方 [FlashMLA 教程](https://tilelang.com/deeplearning_operators/deepseek_mla.html)，
并单独记录 tile、policy、warpgroup 数与生成源码。

这节想传递的结论是：**当寄存器不够时，改的不是数学，而是 warp policy 与 staging**；
第 07 章的布局推断在这里兑现为「选一个 policy，编译器替你完成数据交换」。

### 6.5.9 先按语义选示例，不按文件名复制

官方目录同时包含 MHA/GQA、前向/反向、BSHD/BHSD、固定长度/varlen 等实现。复制代码前
先列出自己的六个条件：

| 条件 | 需要确认的内容 |
|---|---|
| 训练还是推理 | 是否需要反向、dropout、保存或重算 LSE |
| MHA/GQA/MQA | query head 到 KV head 的映射 |
| 张量布局 | BSHD、BHSD 或 packed `[total_tokens, H, D]` |
| mask 语义 | non-causal、causal、局部窗口或 block-causal |
| 长度模型 | 固定长度、varlen、paged KV cache |
| 目标架构 | 可用的矩阵指令、TMA、warp specialization 和 tile 限制 |

社区的 MLA 教程适合观察寄存器压力、warp 分工和真实 decode 工作负载，但不能把其中的
tile、warp policy 或性能数字直接套到训练态 MHA。先从语义最接近的官方测试文件出发，再
保留自己的 reference 和 edge cases。

## 6.6 本章小结

- 朴素 Attention 慢在 **N² 中间张量的 HBM 往返**；FA 用**分块 + 在线 softmax**
  避免完整 S/P 落盘，实际 IO 仍取决于片上容量和 tile 设计。
- 公式三件套：`m_new=max(m_old,m_local)`，`l_new=l_old·e^(m_old−m_new)+l_local·e^(m_local−m_new)`，`O_old *= e^(m_old−m_new)`。
- TileLang 实现的主干是 2 个 `T.gemm`（QKᵀ、PV）加 fragment 中的 max/exp2/sum 与
  掩码；实际代码长度取决于布局、边界、变长、反向和目标架构。
- varlen 不改变 Attention 公式，而是将 Q/K/V 打包成 `[total_tokens, H, D]`，用
  `cu_seqlens` 找到每条样本的起止偏移，并在每个 tile 中使用实际 `q_len/k_len`。
- varlen 的正确性关键是三件事：packed 地址、query/key 尾部 guard、`Lq != Lk` 时的
  causal 对齐；性能评估还要把 pack/unpack 开销和 kernel-only 时间分开。
- 反向靠重计算；`exp2(x·scale)` 融合 `1/√d` 是需要理解的实现细节，因为它同时影响数值表达和指令路径。

## 6.7 Checkpoint

1. 用 `[1, 2, 3, 4]` 分两块手算 `m_new`、`l_new` 和旧输出缩放因子；
2. 以 `N=17`、`D=32` 为边界设计题：列出 Q/K/V tile 的 padding、mask 和输出 guard，
   再把这些处理补进代码后验证 causal 和 non-causal；
3. 解释为什么 masked logits 要使用 `clear_accum=False` 保留 `-inf` 的语义；
4. 写清 dense reference 为什么不能用于任意大序列的默认测试，并记录你的显存预算。
5. 给 `Lq=[5,3]、Lk=[5,4]` 构造 `cu_seqlens_q/k`，写出第二条样本第 2 个 query
   和第 3 个 key 在 packed tensor 中的线性下标；再解释为什么 causal mask 要使用
   `q_local + (Lk-Lq) >= k_local`。

## 口述自测（详答见第 10 章）

1. **默写在线 softmax 三公式并解释每项**（尤其 rescaling 因子）。
2. **FA 的 IO 复杂度推导与「为什么 memory-bound」**（O(N²)→O(N²·d²/M)）。
3. **为什么 P 不落盘、反向如何工作？**（重计算 + 存 lse）。
4. **causal 掩码为什么放 GEMM 前初始化 acc_s 而不是 GEMM 后？**（避免把 −∞ 参与乘加，数值安全；`if_then_else` 直接写 fragment）。
5. **`transpose_B=True` 是干嘛的？**（K 是 `[block_N, dim]`，S=Q·Kᵀ 需要转置）。
6. **`acc_s_cast` 为什么转 fp16？**（第二个 GEMM 的输入类型；fp32 累加只保留在 acc_o）。
7. **num_stages 对 FA 有用吗？**（K/V 拷贝与计算重叠，但 FA 通常少量 stage 即可，block_N 大时收益有限——官方默认 1；可展开讲为什么和 GEMM 不同）。
8. **varlen 中 `cu_seqlens`、`q_len/k_len` 和 `max_seqlen` 分别是什么？**（前缀和边界、
   当前样本实际长度、发射网格/动态 shape 的上界；三者不能互相替代）。
9. **为什么 `Lq != Lk` 时不能直接使用 `q_local >= k_local`？**（decode 常把 query 右对齐
   到 KV 尾部，应先加 `offset = Lk-Lq`，并处理没有可见 key 的行）。
