# 第 06 章 FlashAttention 手写实现

> **本章目标**：从数学到代码完整实现 FlashAttention 前向（在线 softmax + 分块），
> 并讲清为什么它把 Attention 的 IO 复杂度从 O(N²) 降到接近 O(N)。这是大模型
> 内核面试的"压轴题"，本章内容要能**脱稿推导 + 脱稿写码**。

## 6.1 朴素 Attention 为什么慢

```text
S = Q @ K^T / sqrt(d)      # [B,H,N,N]，N² 张量
P = softmax(S, dim=-1)     # 需要整行才能归一化！
O = P @ V                  # [B,H,N,d]
```

两个致命问题：

1. **显存爆炸**：`S` 和 `P` 都是 `N×N`，序列 32K 时单头就是 1.1GB（fp16 bf16），
   多卡训练根本放不下；
2. **IO 浪费**：softmax 的归一化分母需要看到整行，所以 `Q@K^T` 的结果必须先写回
   HBM，`P@V` 时再读回来——**同一份中间数据被存取两遍**。

总 HBM 访存 O(N²)，而计算只有 O(N²·d)。当 d=128 时，理想实现应该把 N² 的中间
张量永远留在片上。FlashAttention 的解法：**分块 + 在线 softmax**——把 N 维切成
块，在寄存器里滚动维护"部分归一化"，绝不落盘 S/P。

面试点：**FlashAttention 的核心贡献是 IO 复杂度**：从 O(N²) HBM 访存降到
O(N²·d²/M)（M 为片上内存），实际训练/推理里 Attention 从 memory-bound 变成了
接近 compute-bound。论文标题里的 "Fast and Memory-Efficient" 各指其一。

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
推导一句话：**归一化变了，分母换了，旧的分子乘上"换分母的修正因子"就行**——不用
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
> `exp2(x·scale)` 一次搞定——少一次除法、少一次乘，还更稳定。这是个值得在面试
> 里主动提起的实现细节。

## 6.3 TileLang 实现：完整代码

以下代码精简自官方示例（去掉 autotune 装饰器，便于学习；`@tilelang.jit` 保留）：

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
    shape = [batch, seq_len, heads, dim]         # bshd 布局
    dtype = 'float16'
    accum_dtype = 'float32'

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        # 网格：(seq/block_M, heads, batch)；一个 block 算一行 (block_M 个 query)
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch,
                      threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)

            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)  # S 块
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)   # 喂 GEMM 要 fp16
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)      # P@V 累加
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

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
                       transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

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

### 6.3.1 逐段解读（面试可背诵版）

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
B, H, N, D = 8, 32, 4096, 128
Q = torch.randn(B, N, H, D, device='cuda', dtype=torch.float16)
K = torch.randn(B, N, H, D, device='cuda', dtype=torch.float16)
V = torch.randn(B, N, H, D, device='cuda', dtype=torch.float16)

kernel = flashattn(B, H, N, D, is_causal=True)
O = kernel(Q, K, V)                      # out_idx=[3]，返回输出

# 参考实现（einsum + 因果掩码 + softmax）
scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) / (D ** 0.5)
mask = torch.tril(torch.ones(N, N, device='cuda')).unsqueeze(0).unsqueeze(0)
scores = scores.masked_fill(mask == 0, float('-inf'))
P = F.softmax(scores, dim=-1)
ref = torch.einsum('bhqk,bkhd->bqhd', P, V)
torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)

total_flops = 4 * B * H * N * N * D      # QK^T + PV 两个 matmul（各 2·B·H·N²·D）
lat = kernel.get_profiler().do_bench()
print(f"{2 * 2 * B * H * N * N * D / lat / 1e9:.1f} TFLOPS")
```

## 6.4 反向传播：为什么"重计算"

反向需要 `P` 和 `S`，但前向从不落盘。FlashAttention 的做法（官方 bwd 示例
`example_mha_bwd_bshd.py` 等）：**反向时用同样的分块循环重算 S/P**（用保存的
`logsum` 或 `lse`），再算 `dV、dK、dQ`。代价是反向多一遍 QKᵀ/PV 计算（约 +33%
FLOPs），换来 O(N²) 显存节省与 IO 减少——在长序列上稳赚。面试答"为什么反向也快"
时，说清楚这个权衡即可。

## 6.5 变体速览（了解即可）

- **GQA/MQA**：多 query 共享 K/V，减少 KV 缓存；
- **varlen（变长序列）**：用 `cu_seqlens` 偏移量把不同长度序列打包成一块，官方
  `example_mha_fwd_varlen.py`；
- **Flash Decoding**：decode 阶段每个 query 对应所有历史 K/V，把 K/V 按块分给
  不同 block，最后用 lse 合并——本质还是在线 softmax 思想。

## 6.6 本章小结

- 朴素 Attention 慢在 **O(N²) HBM 访存**（S/P 落盘两次）；FA 用**分块 + 在线
  softmax** 把它留在片上，IO 复杂度降到近 O(N)。
- 公式三件套：`m_new=max(m_old,m_local)`，`l_new=l_old·e^(m_old−m_new)+l_local·e^(m_local−m_new)`，`O_old *= e^(m_old−m_new)`。
- TileLang 实现 = 2 个 `T.gemm`（QKᵀ、PV）+ fragment 里的 max/exp2/sum + 掩码，
  约 90 行。
- 反向靠重计算；`exp2(x·scale)` 融合 `1/√d` 是值得炫耀的实现细节。

## 面试考点（本章相关）

1. **默写在线 softmax 三公式并解释每项**（尤其 rescaling 因子）。
2. **FA 的 IO 复杂度推导与"为什么 memory-bound"**（O(N²)→O(N²·d²/M)）。
3. **为什么 P 不落盘、反向如何工作？**（重计算 + 存 lse）。
4. **causal 掩码为什么放 GEMM 前初始化 acc_s 而不是 GEMM 后？**（避免把 −∞ 参与乘加，数值安全；`if_then_else` 直接写 fragment）。
5. **`transpose_B=True` 是干嘛的？**（K 是 `[block_N, dim]`，S=Q·Kᵀ 需要转置）。
6. **`acc_s_cast` 为什么转 fp16？**（第二个 GEMM 的输入类型；fp32 累加只保留在 acc_o）。
7. **num_stages 对 FA 有用吗？**（K/V 拷贝与计算重叠，但 FA 通常少量 stage 即可，block_N 大时收益有限——官方默认 1；可展开讲为什么和 GEMM 不同）。