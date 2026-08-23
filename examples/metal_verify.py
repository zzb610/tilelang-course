"""TileLang 的基础 smoke test。

运行: .venv/bin/python examples/metal_verify.py

脚本优先使用 CUDA，其次使用 Apple MPS。没有可用 GPU 时会打印环境提示并正常退出；
这让它适合放进新环境的第一轮检查，而不是把“没有设备”误报成内核错误。

注意（2026-08，tilelang 0.1.13 macOS arm64 wheel）：
- T.Parallel / T.copy / alloc_shared / 串行归约（threads<=32）在 Metal 上验证可用；
- Metal 后端暂不支持 T.infinity（下方用大负数字面量代替；
   GPU/CUDA 标准写法：T.infinity(dtype)）；
- threads>32 时多 simdgroup 的“全线程复制串行循环”结果错误（后端缺陷，用 threads<=32；
   GPU/CUDA 标准写法：threads=128/256 等任意取值均正常）；
- T.gemm 在 Metal 后端 0.1.13 存在 codegen 限制（shared 指针地址空间限定符），
  请在 CUDA GPU 上运行第 05/06 章的 GEMM/FlashAttention 示例。
"""
import torch
import tilelang
import tilelang.language as T
from tilelang import jit


@jit
def softmax_rows(N: int, threads: int = 32, dtype: str = 'float32'):
    """第 03 章：分块 softmax（每 block 一行，shared 中转）。
    # GPU/CUDA 标准写法：threads 取 128/256 等常规值；Mac/Metal 0.1.13 需 <=32
    # （后端对多 simdgroup 的复制串行循环有缺陷）。
    """
    @T.prim_func
    def kern(A: T.Tensor((N, N), dtype), O: T.Tensor((N, N), dtype)):
        with T.Kernel(N, threads=threads) as bx:
            row_s = T.alloc_shared((N,), dtype)  # shape: [N]
            row_max = T.alloc_var(dtype)          # shape: []，标量
            row_sum = T.alloc_var(dtype)          # shape: []，标量
            T.copy(A[bx, 0], row_s)
            # GPU/CUDA 标准写法：row_max = -T.infinity(dtype)
            # Mac/Metal 0.1.13 暂不支持 T.infinity，故用大负数字面量
            row_max = -1e30
            for j in T.serial(N):
                row_max = T.max(row_max, row_s[j])
            row_sum = 0
            for j in T.serial(N):
                row_s[j] = T.exp(row_s[j] - row_max)
                row_sum += row_s[j]
            for j in T.serial(N):
                row_s[j] = row_s[j] / row_sum
            T.copy(row_s, O[bx, 0])
    return kern


@jit
def vector_add(N: int, block: int = 256, dtype: str = 'float32'):
    """第 01 章：向量加法（Metal 验证通过；与 GPU/CUDA 写法完全一致，无改动）"""
    @T.prim_func
    def kern(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype),
             C: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                gi = bx * block + i
                C[gi] = A[gi] + B[gi]
    return kern


if __name__ == "__main__":
    if torch.cuda.is_available():
        # NVIDIA GPU：课程主路径
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # Apple Silicon：基础语法/有限后端 smoke test
        dev = "mps"
    else:
        print("SKIP: 没有可用的 CUDA 或 MPS 设备；请在 GPU 环境运行本 smoke test。")
        raise SystemExit(0)

    N = 1 << 20
    a = torch.randn(N, device=dev)       # shape: [N]
    b = torch.randn(N, device=dev)       # shape: [N]
    k1 = vector_add(N)
    c = torch.empty(N, device=dev)       # shape: [N]
    k1(a, b, c)
    torch.testing.assert_close(c, a + b)
    print("✓ vector add")

    N = 256
    A = torch.randn(N, N, device=dev)    # shape: [N, N]
    O = torch.empty(N, N, device=dev)    # shape: [N, N]
    # GPU/CUDA 标准写法可以使用 128/256 个线程；32 对 Metal 兼容性更保守。
    k2 = softmax_rows(N, threads=32)   # Metal 兼容：threads<=32
    k2(A, O)
    torch.testing.assert_close(O, torch.softmax(A, dim=-1), rtol=1e-4, atol=1e-6)
    print("✓ 分块 softmax（threads=32）")

    src = k2.get_kernel_source()
    print("✓ 生成 Metal 源码可查看（长度", len(src), "字符）")
