# 第 13 章 分组 GEMM：从多个小矩阵到 MoE 内核

普通 GEMM 的规则性给了我们一个舒适前提：每个 block 都处理同样大小的输出 tile。MoE
打破了这个前提。token 被路由到不同 expert 后，各组的行数随输入变化；有的组很大，有的组
很小，甚至为空。乘法公式没有改变，工作量的分布却不再规则。

这使 Grouped GEMM 的主要困难从“怎样做矩阵乘”转移到“怎样表示和调度一组不同大小的矩阵
乘”。packed layout 决定数据如何连续存放，前缀和描述每组的边界，block 映射把线性工作编号
还原为 group 与 tile 坐标。第 05 章的 shared、fragment、流水线和 `T.gemm` 仍在内层发挥
作用，只是外面多了一层不规则映射。

本章通过一个具体的小组分布逐步建立这层映射，再讨论 padding、空组、分桶和端到端 MoE
开销。文中的 `group_idx_for_block` 是这种映射的一份元数据，不是 TileLang 保证存在的固定
API。这个区别也提醒我们：工程设计首先来自工作负载的结构，而不是来自可调用的函数名。

> **本章导航** 高级专项难度，预计 4–6 小时。前置是第 05 章 GEMM、第 07 章布局与高级
> 指令、第 08 章自动调优。运行范围以 CUDA GPU 为主线，完整性能实验需要支持 `T.gemm`
> 的 GPU 和匹配的 TileLang 版本。本章产出一份 grouped GEMM 正确性测试、一张分组元数据
> 表、一份 kernel-only 与 end-to-end 性能对比。官方参考：[Grouped GEMM examples](https://github.com/tile-ai/tilelang/tree/main/examples/grouped_gemm)、
> [TileLang fused-MoE 示例](https://github.com/tile-ai/tilelang/blob/main/examples/fusedmoe/example_fusedmoe_tilelang.py)。

## 13.1 为什么需要分组 GEMM

先退回到普通 GEMM，它只有一组矩阵：

$$
C = A \times B,\qquad A\in\mathbb{R}^{M\times K},
B\in\mathbb{R}^{K\times N}, C\in\mathbb{R}^{M\times N}.
$$

分组 GEMM 则同时计算 `G` 组矩阵：

$$
C_g = A_g \times B_g,
\qquad A_g\in\mathbb{R}^{M_g\times K},
B_g\in\mathbb{R}^{K\times N},
C_g\in\mathbb{R}^{M_g\times N}.
$$

本章先假设所有组共享 `K/N`，但每组的 `M_g` 可以不同。这正是 token 被路由到不同 expert 后常见的形状：每个 expert 收到的 token 数不同，但隐藏维度和 expert 中间维度通常相同。

例如有 4 个 expert：

```text
expert 0: A0[32, 4096]  × B0[4096, 11008] → C0[32, 11008]
expert 1: A1[128,4096]  × B1[4096, 11008] → C1[128, 11008]
expert 2: A2[7, 4096]   × B2[4096, 11008] → C2[7, 11008]
expert 3: A3[64, 4096]  × B3[4096, 11008] → C3[64, 11008]
```

如果逐组调用 GEMM，就会产生 4 次 kernel 调度；如果每组很小，矩阵乘本身的并行度不足，调度开销和尾部浪费会更加明显。分组 GEMM 的目标是：**保留每组独立的矩阵和结果语义，同时让多个组共享一套 kernel 调度和 tile 逻辑。**

### 13.1.1 它和 batched GEMM 不一样

三种问题不要混在一起：

| 形式 | 典型形状 | 主要特点 | 常见入口 |
|---|---|---|---|
| 普通 GEMM | 一个 `M×K` 乘一个 `K×N` | 只有一组矩阵 | `T.gemm`、cuBLAS GEMM |
| Batched GEMM | `G` 组相同的 `M/N/K` | 组间形状相同，地址通常有固定 stride | `torch.bmm`、batched library |
| Grouped GEMM | 第 `g` 组有 `M_g/N_g/K_g`，至少一维可以不同 | 需要额外的组元数据和调度映射 | grouped GEMM kernel |

如果所有组的形状完全相同，优先先考虑 batched GEMM；只有当形状不规则、需要不同指针或需要和路由后的 token 排布结合时，grouped GEMM 才真正体现价值。

### 13.1.2 分组 GEMM 在 MoE 中的位置

分组 GEMM 不是完整的 MoE，而是 MoE 中间的一段计算：

```text
router 得到 expert id
        ↓
dispatch：按 expert 重排 token，形成 A0、A1、…、A(G-1)
        ↓
grouped GEMM：每个 expert 用自己的权重处理自己的 token
        ↓
combine/scatter：把结果还原到原始 token 顺序并做权重合并
```

因此，不能只报告 grouped GEMM kernel 的延迟就声称整个 MoE 加速了。端到端报告还要说明 token 重排、元数据生成、结果 scatter/reduce 是否计入计时。

## 13.2 先选一种数据表示

Grouped GEMM 的第一个工程决策不是 tile 大小，而是「矩阵组如何放在内存里」。下面先采用一种适合教学和 MoE 的 packed layout。

### 13.2.1 Packed layout：把 A 和 C 按行拼接

把每组输入矩阵沿 `M` 维首尾相接：

```text
A_pack = [ A0 的 M0 行 ][ A1 的 M1 行 ][ A2 的 M2 行 ] ...
C_pack = [ C0 的 M0 行 ][ C1 的 M1 行 ][ C2 的 M2 行 ] ...

B_stack[g] = 第 g 组的 B_g，形状为 [K, N]
```

需要维护两类 offset：

```text
group_sizes = [M0, M1, M2, ...]
row_offsets = [0, M0, M0+M1, M0+M1+M2, ...]
```

对第 `g` 组：

$$
A_g = A_{pack}[row\_offsets[g]:row\_offsets[g+1], :],
$$

`C_g` 使用同样的行区间。这样，输入和输出可以保持连续存储，只有权重 `B` 需要通过组编号索引。

### 13.2.2 指针式 layout：每组保存独立地址

另一种表示是保存：

```text
A_ptrs[g] → A_g
B_ptrs[g] → B_g
C_ptrs[g] → C_g
```

它适合原始矩阵已经分散在不同内存区域的场景，避免重新 pack；代价是 kernel 内需要做指针加载和地址计算，地址访问也更难统一。TileLang 官方 `examples/grouped_gemm/` 同时提供了常规和 pointer 风格的示例，实际选型要结合上游数据是否已经连续、是否需要额外重排来测量。

### 13.2.3 主线采用的假设

为了先把调度问题讲清楚，本章主线使用下面的约束：

1. `A_pack` 和 `C_pack` 沿行拼接；
2. 所有组共享 `K/N`，只有 `M_g` 不同；
3. `B_stack` 的形状是 `[G, K, N]`；
4. 第一版把每组的 `M_g` 向上补齐到 `BM`，补齐行写成 0；
5. `K` 和 `N` 先使用能整除 `BK/BN` 的尺寸，尾部处理放在 13.7。

这只是便于验证索引和数据流的一种教学基线，grouped GEMM 还有其它表示形式。

## 13.3 参考实现：先保证每组结果正确

在写 GPU kernel 之前，先用 host 侧循环建立参考实现。这个版本故意不追求速度，只负责定义结果语义。

```python
import torch


def grouped_gemm_reference(A_pack, B_stack, row_offsets):
    """A_pack: [sum(M_g), K], B_stack: [G, K, N]."""
    outputs = []  # 每个元素 shape: [M_g, N]
    group_count = B_stack.shape[0]

    for g in range(group_count):
        row_start = int(row_offsets[g])
        row_end = int(row_offsets[g + 1])
        outputs.append(A_pack[row_start:row_end] @ B_stack[g])

    if not outputs:
        return A_pack.new_empty((0, B_stack.shape[-1]))  # shape: [0, N]
    result = torch.cat(outputs, dim=0)  # shape: [sum(M_g), N]
    return result
```

这里的 `row_offsets` 可以是 Python 列表，也可以是 CPU 上的整数 tensor。参考实现要明确三件事：

- 第 `g` 组只使用 `A_pack[row_offsets[g]:row_offsets[g+1]]`；
- 第 `g` 组只使用 `B_stack[g]`；
- 输出顺序仍按 group id 排列，不是按 token 原始 id 排列。若上游做过 token dispatch，原始顺序的恢复属于后续 scatter/combine 阶段。

### 13.3.1 逐组调用是正确但低效的基线

```python
def grouped_gemm_separate_launches(A_pack, B_stack, row_offsets):
    outputs = []  # 每个元素 shape: [M_g, N]
    for g in range(B_stack.shape[0]):
        row_start = int(row_offsets[g])
        row_end = int(row_offsets[g + 1])
        outputs.append(A_pack[row_start:row_end] @ B_stack[g])
    result = torch.cat(outputs, dim=0)  # shape: [sum(M_g), N]
    return result
```

这个基线可以用来回答第一个性能问题：**把多个 GEMM 合成一个 grouped kernel，是否只是减少了 kernel launch？** 不一定。除了 launch 次数，还可能改变：

- 小矩阵的线程块利用率；
- `B_g` 的加载和缓存行为；
- `M_g` 尾部 tile 的 padding 浪费；
- 元数据读取和 block 映射开销。

所以后续报告至少要同时比较 kernel-only 和 end-to-end 两种口径。

## 13.4 最关键的难点：把 block 映射回 group

普通 GEMM 中，`blockIdx.x/y` 可以直接解释为输出 tile 的列/行坐标。Grouped GEMM 中，一个线性 block id 还需要知道它属于哪一组。

### 13.4.1 用每组 tile 数建立前缀和

设 `BM/BN` 是输出 tile 大小，并假设 `N` 共享，则第 `g` 组的 tile 数为：

$$
T_g = \left\lceil\frac{M_g}{BM}\right\rceil
      \times\left\lceil\frac{N}{BN}\right\rceil.
$$

再定义 tile 前缀和：

$$
P_0=0,\qquad P_{g+1}=P_g+T_g.
$$

于是，线性 block `b` 属于满足下面条件的组：

$$
P_g \le b < P_{g+1}.
$$

组内 tile 编号是：

$$
local\_tile=b-P_g.
$$

再把它拆成二维坐标：

$$
by=\left\lfloor\frac{local\_tile}{\lceil N/BN\rceil}\right\rfloor,
\qquad
bx=local\_tile\bmod\lceil N/BN\rceil.
$$

### 13.4.2 具体例子

取 `BM=128`、`BN=128`、`N=256`，因此每组有 2 个 N 方向 tile。若：

```text
M = [130, 32, 256]
M 方向 tile 数 = [2, 1, 2]
每组 tile 数   = [4, 2, 4]
tile_offsets    = [0, 4, 6, 10]
```

映射关系如下：

| 线性 block id | group | 组内 tile | `by` | `bx` |
|---:|---:|---:|---:|---:|
| 0～3 | 0 | 0～3 | 0～1 | 0～1 |
| 4～5 | 1 | 0～1 | 0 | 0～1 |
| 6～9 | 2 | 0～3 | 0～1 | 0～1 |

这个表说明了 grouped GEMM 的核心：**矩阵乘本身仍然是规则 tile；不规则性集中在 block 到 group 的映射和每组尾部尺寸上。**

### 13.4.3 三种实现映射的方式

| 方法 | 做法 | 优点 | 代价 |
|---|---|---|---|
| 预计算 `group_idx_for_block` | host 侧生成每个 block 对应的 group id | kernel 内只读一次，简单且快 | 路由变化时要重建元数据 |
| kernel 内二分查找 `tile_offsets` | 每个 block 自己找 `g` | 元数据更小 | 每个 block 多做查找，分支和负载不规则 |
| padded row mapping | 按 M 方向补齐，记录 `group_padded_offsets` 和 block→group 映射 | 适合 MoE 的 packed token，和行 tile 对齐 | 需要处理 padding 和空组 |

教学和工程实践通常先选第一种。官方 fused-MoE 示例采用了 `group_sizes`、`group_offsets`、`group_padded_offsets` 和 `group_idx_for_bx` 这类元数据，把 M 方向的 block 映射和实际有效行数分开处理。

## 13.5 准备分组元数据

下面的代码只负责在 host 侧计算行 offset、padding offset 和 M 方向 block 的 group id。它不是 GPU kernel，适合先单独测试。

```python
import torch


def build_group_metadata(group_sizes, block_m, device="cuda"):
    sizes = torch.as_tensor(group_sizes, dtype=torch.int32)  # shape: [G]
    group_count = sizes.numel()

    row_offsets = torch.zeros(group_count + 1, dtype=torch.int32)  # shape: [G+1]
    row_offsets[1:] = torch.cumsum(sizes, dim=0)

    # 空组不产生有效输出；这里给它 0 个工作块，避免无意义计算。
    m_blocks = (sizes + block_m - 1) // block_m  # shape: [G]
    padded_sizes = m_blocks * block_m              # shape: [G]

    padded_offsets = torch.zeros(group_count + 1, dtype=torch.int32)  # shape: [G+1]
    padded_offsets[1:] = torch.cumsum(padded_sizes, dim=0)

    group_ids = torch.arange(group_count, dtype=torch.int32)  # shape: [G]
    group_idx_for_block = torch.repeat_interleave(group_ids, m_blocks)  # shape: [num_m_blocks]
    num_m_blocks = int(group_idx_for_block.numel())
    padded_total_m = int(padded_offsets[-1].item())
    total_m = int(row_offsets[-1].item())

    metadata = {
        "group_sizes": sizes.to(device),                    # shape: [G]
        "row_offsets": row_offsets.to(device),              # shape: [G+1]
        "padded_offsets": padded_offsets.to(device),        # shape: [G+1]
        "group_idx_for_block": group_idx_for_block.to(device),  # shape: [num_m_blocks]
    }
    return metadata, total_m, padded_total_m, num_m_blocks
```

这个函数有几个必须验证的性质：

```text
row_offsets[0] == 0
row_offsets[g+1] - row_offsets[g] == group_sizes[g]
padded_offsets[g+1] - padded_offsets[g] == ceildiv(group_sizes[g], BM) * BM
group_idx_for_block 的长度 == sum(ceildiv(group_sizes[g], BM))
```

当 `group_sizes[g] == 0` 时，`m_blocks[g] == 0`，因此不会生成对应 block。这样做的好处是没有空计算；代价是 kernel 的 group 映射只覆盖非空组。若工程接口要求每个 group 都保留一个工作项，可以给空组分配一个 dummy block，但必须让它在进入 `T.copy`/`T.gemm` 前直接退出或跳过写回，不能对无效地址做读取。

### 13.5.1 A 的 padding 做法

教学骨架使用 `A_padded`，它的每组行数已经向上补齐到 `BM`，并且补齐行写成 0：

```text
A_padded = [ pad(A0, ceildiv(M0,BM)*BM )
             pad(A1, ceildiv(M1,BM)*BM )
             ... ]
```

这样，最后一个 M tile 仍然可以用规则的 `T.copy` 搬进 shared memory；输出时再用 `group_sizes[g]` 保护真实行。补零的原因是：无效行参与 GEMM 时不会污染有效行的结果。

生产实现也可以不提前 materialize `A_padded`，而是在最后一个 M tile 中用显式掩码加载有效行、把无效行填 0。这样能减少 padding 的存储，但代码和分支更复杂，必须单独测量。

## 13.6 TileLang 内核骨架：规则 tile + 分组映射

下面是一个教学用内核骨架。它展示了分组 GEMM 的关键数据流，不把所有 host 包装代码和不同后端的边界分支塞进一个例子。

**代码前提：**

- `A_padded` 已按 `BM` 对每组补齐，补齐行是 0；
- `K` 可以被 `BK` 整除，`N` 可以被 `BN` 整除；
- `B_stack` 形状为 `[G, K, N]`；
- `group_idx_for_block[bm_id]` 给出当前 M 方向 block 属于哪一组；
- `group_sizes` 和 `row_offsets` 描述未 padding 的逻辑输出。

```python
import tilelang
import tilelang.language as T


@tilelang.jit
def grouped_gemm_padded(
    total_m: int,
    padded_total_m: int,
    group_count: int,
    num_m_blocks: int,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 32,
    threads: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    M, N, K, G = T.const("M, N, K, G")

    @T.prim_func
    def kernel(
        A_padded: T.Tensor((padded_total_m, K), dtype),
        B_stack: T.Tensor((G, K, N), dtype),
        C_pack: T.Tensor((total_m, N), dtype),
        group_sizes: T.Tensor((G,), T.int32),
        row_offsets: T.Tensor((G + 1,), T.int32),
        padded_offsets: T.Tensor((G + 1,), T.int32),
        group_idx_for_block: T.Tensor((num_m_blocks,), T.int32),
    ):
        with T.Kernel(num_m_blocks, T.ceildiv(N, block_n), threads=threads) as (bm_id, bn_id):
            group_id = group_idx_for_block[bm_id]

            # bm_id*block_m 是 A_padded 中的物理行起点。
            # 减去该组之前的 padded 行数，得到组内逻辑行起点。
            group_row0 = bm_id * block_m - padded_offsets[group_id]
            output_row0 = row_offsets[group_id] + group_row0
            valid_m = T.min(block_m, group_sizes[group_id] - group_row0)

            A_shared = T.alloc_shared((block_m, block_k), dtype)
            B_shared = T.alloc_shared((block_k, block_n), dtype)
            C_frag = T.alloc_fragment((block_m, block_n), accum_dtype)
            T.clear(C_frag)

            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=2):
                T.copy(A_padded[bm_id * block_m, ko * block_k], A_shared)
                T.copy(B_stack[group_id, ko * block_k, bn_id * block_n], B_shared)
                T.gemm(A_shared, B_shared, C_frag)

            for i, j in T.Parallel(block_m, block_n):
                if i < valid_m and bn_id * block_n + j < N:
                    C_pack[output_row0 + i, bn_id * block_n + j] = C_frag[i, j]

    return kernel
```

### 13.6.1 逐段理解这段代码

1. **block 到 group**：`bm_id` 先查 `group_idx_for_block`，得到 `group_id`。这是 grouped GEMM 相对于普通 GEMM 新增的第一层索引。
2. **物理行到逻辑行**：`A_padded` 按补齐后的行排列，`C_pack` 按真实行排列，所以需要用 `padded_offsets` 和 `row_offsets` 做一次坐标转换。
3. **权重选择**：`B_stack[group_id, ...]` 选择当前 expert 的权重。一个 block 不会访问其他 group 的 `B`。
4. **K 循环**：组内 GEMM 的 K 维仍然使用第 05 章的 `T.Pipelined`、shared tile、fragment 累加器和 `T.gemm`。
5. **输出保护**：最后一个 M tile 可能只有一部分真实行，`valid_m` 防止把 padding 结果写回 `C_pack`。

这说明一个重要事实：**Grouped GEMM 不是把 `T.gemm` 换成另一种乘法，而是在普通 tiled GEMM 外面增加「组选择、地址偏移和尾部管理」。**

### 13.6.2 不在内层循环反复查 group 的原因

`group_id` 在一个 M tile 内是常量。正确做法是每个 block 只读取一次组编号，然后把它用于：

- 计算 A/C 的行偏移；
- 选择 B 的 batch 维；
- 计算有效行数。

不要在 `T.Parallel(block_m, block_n)` 的每个元素里再次搜索 `group_id`。那会把本应是 block 级的元数据开销放大到 element 级，并且使访问和分支更难优化。

### 13.6.3 这个骨架不直接承诺可运行的原因

Grouped GEMM 的实际签名会受以下因素影响：

- 当前 TileLang 版本对动态 `T.Tensor` 维度和 3D `T.copy` 的支持；
- 目标后端对 `T.gemm` 输入布局和 `transpose_B` 的要求；
- grid 维度是按当前输入编译，还是采用上限网格加 active guard；
- 是否启用 warp specialization、TMA 或特定的 layout policy。

因此，阅读这段代码时先检查索引和数据流，再对照当前版本的 [官方 grouped GEMM 示例](https://github.com/tile-ai/tilelang/tree/main/examples/grouped_gemm) 调整 API。能通过 AST 语法检查不等于能在任意 GPU、任意 TileLang wheel 上编译。

## 13.7 边界条件：正确性比吞吐更优先

Grouped GEMM 最容易出现的错误不是矩阵乘公式写错，而是组边界和 padding 处理错。

### 13.7.1 M 方向尾部

若 `M_g` 不是 `BM` 的倍数：

- A 的 padding 行必须是 0，或者加载时显式填 0；
- C 的最后一个 tile 只能写 `valid_m` 行；
- `group_row0` 必须基于当前 group 的 padded offset，而不是直接使用全局 `bm_id*BM`。

### 13.7.2 K 方向尾部

若 `K` 不是 `BK` 的倍数，最后一个 K tile 需要把无效列填 0。否则，A/B 的无效元素会参与乘加，结果会被污染。可选方案：

1. host 侧把 K padding 到 `BK` 的倍数；
2. 每轮先 `T.clear(A_shared/B_shared)`，再用 guarded load 写有效元素；
3. 为整除尺寸和尾部尺寸分别选择不同的专用 kernel。

不要因为 `T.copy` 可能插入 safe-access guard，就默认越界输入会自动变成 0。边界保护和无效元素的数值语义是两件事。

### 13.7.3 空组

`M_g=0` 的组不应该发射有效 GEMM。常见处理有两种：

- 在 metadata 阶段过滤空组，但保留原始 `group_id`；
- 为每个空组保留一个 dummy 工作项，在 kernel 中直接跳过读写。

第一种通常减少无效工作，第二种更容易保持固定的组编号表。无论采用哪种方式，都要测试「全部为空」「中间为空」和「最后为空」。

### 13.7.4 N/K 不同的组

本章主线假设各组共享 `N/K`。如果不同组连 `N/K` 也不同，单个规则 `T.gemm` 骨架就不再适用。工程上通常先按 `(N, K, dtype, transpose)` 分桶，再对每个 bucket 调用专用 grouped kernel；形状极少时也可以退回逐组调用库。

## 13.8 性能分析：不要只看 kernel 延迟

Grouped GEMM 的性能报告至少要有两种口径：

### 13.8.1 Kernel-only

把 pack、metadata、dispatch 和输出 scatter 都放在计时区间外，只测 grouped kernel 与逐组 GEMM 的 GPU 执行时间。它回答的是：**内核调度和 tile 组织本身是否有效？**

### 13.8.2 End-to-end

把上游 token 重排、metadata 构造、kernel、结果恢复都计入。它回答的是：**在真实模型路径里，这个设计是否值得？**

两者差距很大时，继续调 `BM/BN` 往往收效有限；更应先减少重复 pack、缓存稳定 metadata，或把 dispatch/grouped GEMM/combine 做更紧的融合。

### 13.8.3 有效 FLOPs 和 padding FLOPs

有用的数学量是：

$$
FLOPs_{useful}=2\sum_g M_gKN.
$$

如果 M 方向向上 padding，实际执行的近似计算量是：

$$
FLOPs_{padded}=2\sum_g
\left\lceil\frac{M_g}{BM}\right\rceil BM\cdot K\cdot N.
$$

可以报告 padding 利用率：

$$
\rho_M=\frac{\sum_g M_g}
{\sum_g \lceil M_g/BM\rceil BM}.
$$

例如 `M=[32,128,7,64]`、`BM=128` 时：

```text
真实行数       = 32 + 128 + 7 + 64 = 231
执行行数       = 128 + 128 + 128 + 128 = 512
M 方向利用率   = 231 / 512 ≈ 45.1%
```

这只是 M 方向的利用率，不等于最终 TFLOPS；但它能帮助解释「为什么一个看似减少 launch 的 grouped kernel 仍然不快」。

### 13.8.4 TFLOPS 口径

若延迟单位是毫秒：

```text
TFLOPS = 2 * sum(M_g * K * N) / latency_ms / 1e9
```

对于 padding kernel，建议同时报告 useful TFLOPS 和 padded TFLOPS，避免用「包含无效行的计算量」掩盖真实利用率。

## 13.9 优化路线：一次只改变一个变量

建议按照下面的版本顺序推进：

| 版本 | 主要变化 | 要回答的问题 |
|---|---|---|
| v0 | 逐组调用 `torch.mm` 或库 GEMM | launch 和小矩阵利用率的基线是多少？ |
| v1 | packed A/C + 一次 grouped kernel + M padding | 合并调度后是否降低 kernel-only 延迟？ |
| v2 | 保留真实 `M_g`，补齐行 guarded write/load | padding 浪费占多少？正确性是否覆盖空组和尾部？ |
| v3 | 按 `(M_bucket, N, K, dtype)` 分桶 | 是否能减少形状差异带来的 tile 浪费？ |
| v4 | pipeline、layout、rasterization 和 autotune | 瓶颈是搬运、Tensor Core、负载不均还是 metadata？ |
| v5 | persistent/task-based 调度或融合 dispatch/combine | 端到端开销是否已经成为主要瓶颈？ |

每一版都保留：

- 同一组输入和同一份参考实现；
- 相同的 warmup、rep、同步和计时 backend；
- 同时覆盖均匀分组和高度倾斜分组；
- 生成源码或 profiler 证据；
- 一句明确的结论：改了什么、观察到什么、为什么相信收益来自这个变量。

### 13.9.1 分桶为什么常常比扩大 tile 更有效

若一批 group 的 `M_g` 差异很大，一个统一的 `BM` 很难同时照顾大组和小组：

- `BM` 太大：小组 padding 严重；
- `BM` 太小：大组的 tile 数增多，Tensor Core 和数据搬运效率可能下降；
- `stages` 太多：大组可能受益，小组却可能被 shared/register 开销拖慢。

分桶的思路是先把形状相近的 group 放到一起，再在每个 bucket 内调参。它增加了调度和 metadata 管理，但通常比让一个 kernel 适配所有极端形状更可控。

## 13.10 调试清单

### 13.10.1 先在 host 侧检查元数据

```python
import torch


def check_group_metadata(group_sizes, row_offsets, padded_offsets, group_idx_for_block, block_m):
    sizes = torch.as_tensor(group_sizes, dtype=torch.int32, device="cpu")  # shape: [G]
    rows = torch.as_tensor(row_offsets, dtype=torch.int32, device="cpu")   # shape: [G+1]
    padded = torch.as_tensor(padded_offsets, dtype=torch.int32, device="cpu")  # shape: [G+1]
    block_groups = torch.as_tensor(group_idx_for_block, dtype=torch.int32, device="cpu")  # shape: [num_m_blocks]

    assert rows[0].item() == 0
    assert padded[0].item() == 0
    assert torch.equal(rows[1:] - rows[:-1], sizes)
    assert torch.all(rows[1:] >= rows[:-1])
    assert torch.all(padded[1:] >= padded[:-1])
    assert torch.all((block_groups >= 0) & (block_groups < sizes.numel()))

    expected_blocks = ((sizes + block_m - 1) // block_m).sum()
    assert block_groups.numel() == expected_blocks.item()
```

这一步能先排除大量 GPU kernel 问题：offset 长度不对、block 数不对、group id 越界、空组处理不一致，都应该在 host 侧直接失败。

### 13.10.2 再检查三个最小 case

1. `G=1`：应退化为普通 tiled GEMM；
2. `G=2` 且 `M=[BM, 1]`：专门检查第二组尾部；
3. `G=3` 且 `M=[0, 7, 2*BM+1]`：检查空组、中间组和大组同时出现。

如果 `G=1` 都不正确，先不要调 grouped mapping；如果 `G=1` 正确而 `G>1` 错误，优先打印 `group_id`、`group_row0`、`output_row0` 和 `valid_m`。

### 13.10.3 `T.print` 的打印对象

调试输入缩小后，每个 M block 只打印一次：

```text
bm_id, group_id, group_row0, output_row0, valid_m
```

不要打印整个 fragment 或整个 tile。设备端打印会改变时序，也不能作为性能测量工具。定位完成后删除打印，再重新运行正确性和 benchmark。

## 13.11 练习与动手任务

### 练习 1：回忆

用自己的话区分普通 GEMM、batched GEMM 和 grouped GEMM，并说明为什么 MoE 更接近 grouped GEMM。

### 练习 2：跟踪

给定：

```text
M=[130,32,256], N=256, BM=BN=128
```

写出每组 tile 数、前缀和，并判断线性 block 5 所属的 group 和组内坐标。

### 练习 3：修改

修改 `build_group_metadata`，让它同时返回 `tile_offsets` 和 `group_idx_for_tile`。比较「精确 tile 映射」和「按 M 方向 padded mapping」的 metadata 大小与 kernel 逻辑。

### 练习 4：实现

在 `A_padded` 前提下补齐第 13.6 节的 host 包装：生成 `A_padded`、`B_stack`、`C_pack` 和 metadata，并用第 13.3 节的参考实现校验三个 edge case。

### 练习 5：优化

固定 `K=N=4096`，比较以下两组分布：

```text
均匀：M=[128,128,128,128]
倾斜：M=[512,8,8,8]
```

分别测试逐组 GEMM、统一 `BM=128` 的 grouped GEMM，以及按 M 分桶后的版本。报告 kernel-only、end-to-end、useful TFLOPS 和 M padding 利用率。

### 练习 6：解释

写一页实验记录，回答：

1. 你的瓶颈是 launch、padding、内存、Tensor Core 利用率，还是 metadata/dispatch？
2. 哪个 profiler 指标或生成代码片段支持这个判断？
3. 如果 `N/K` 也不相同，你会选择分桶、指针式 kernel，还是退回库调用？为什么？

### 动手任务

完成下面任务再进入第 14 章：

本章的完成标准不是「能背出 Grouped GEMM 的定义」，而是留下以下产物：

1. 一份 host 侧参考实现和至少 3 组边界测试；
2. 一张包含 `group_sizes`、`row_offsets`、`padded_offsets`、`group_idx_for_block` 的 metadata 表；
3. 一个能解释 block→group→tile→输出行映射的 TileLang 内核骨架；
4. 一份同时包含 kernel-only 和 end-to-end 的性能报告；
5. 一段说明 grouped GEMM 何时应该退回 batched GEMM、分桶 kernel 或现成库的工程判断。
6. 若把本章作为收官项目，补充 group-size 分布、路由/pack/scatter 成本和 dispatch 范围，
   不能只提交矩阵乘 kernel 的最快数字。

## 13.12 本章回顾

这一章把第 05 章的 GEMM 经验扩展到了形状不规则的场景，而它的核心洞见其实很集中：
Grouped GEMM 的难点不在新的乘法公式，而在**不规则 group 的表示与调度映射**。落到具体
数据上，`group_sizes` 描述真实工作量，`row_offsets` 描述 packed 逻辑位置，
`padded_offsets` 描述 tile 对齐后的物理位置，`group_idx_for_block` 则把线性 block 映射
回 group。需要记住的是，普通 GEMM 的 shared、fragment、`T.Pipelined` 和 `T.gemm`
仍然适用，新增的只是组选择、地址偏移和尾部保护这层外壳。至于性能，padding、空组、
K/N 尾部、分组倾斜和端到端 dispatch 都可能决定最终结果，所以要先建立逐组参考和正确
baseline，再做 packed、分桶、pipeline、layout 和 persistent 调度实验；具体的 API、布局
约束和后端支持，始终以当前 TileLang 官方 grouped GEMM 示例为准。

下一步进入 [第 14 章](14-Capstone真实算子交付.md)，把本章的 grouped GEMM 放回完整
MoE 路径，或选择另一个真实算子完成契约、测试、优化、库对比和 fallback。

## 13.13 自问自答

这些问题用于检验你是否能把本章的知识讲出来。作答时尽量把"表示—映射—边界"这条线串
完整，而不只是复述单条结论：

1. 为什么 grouped GEMM 不能简单等同于 batched GEMM？
2. `group_idx_for_block` 解决了什么问题？它为什么适合在 host 侧预计算？
3. `row_offsets` 和 `padded_offsets` 分别描述什么？
4. 为什么 M 尾部必须填 0 或做 guarded load，而不能只 guard 输出？
5. 如何同时报告 useful TFLOPS、padded TFLOPS 和 end-to-end 延迟？
6. 如果 90% 的 group 都很小、只有一个 group 很大，你会如何选择 BM、分桶和调度策略？

## 13.14 参考资料

- [TileLang grouped GEMM examples](https://github.com/tile-ai/tilelang/tree/main/examples/grouped_gemm)：前向、反向和 pointer-style grouped GEMM 示例。
- [TileLang fused-MoE example](https://github.com/tile-ai/tilelang/blob/main/examples/fusedmoe/example_fusedmoe_tilelang.py)：展示 token dispatch、group metadata、分组 GEMM 和结果 scatter 的组合路径。
- [TileLang Language Basics](https://github.com/tile-ai/tilelang/blob/main/docs/programming_guides/language_basics.md)：`T.Kernel`、作用域、`T.copy` 和 tiled GEMM 基础。
- [TileLang Instruction Guide](https://github.com/tile-ai/tilelang/blob/main/docs/programming_guides/instructions.md)：`T.gemm`、`T.copy`、异步搬运和同步语义。
- [TileLang Software Pipeline Guide](https://github.com/tile-ai/tilelang/blob/main/docs/programming_guides/software_pipeline.md)：`T.Pipelined`、stage/order 和生产者—消费者关系。
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)：CUDA 执行模型、内存和同步语义。
