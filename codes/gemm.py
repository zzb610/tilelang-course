import torch
import tilelang.language as T
from tilelang import jit

# CUDA target for NVIDIA GPU execution.
@jit(target="cuda")
def gemm(M: int, N: int, K: int, block_m: int, block_n: int, block_k: int, block_threads: int = 128, dtype=torch.float32):
    @T.prim_func
    def gemm_kernel(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):  # pyright: ignore[reportInvalidTypeForm]
        with T.Kernel(T.cdiv(M, block_m), T.cdiv(N, block_n), threads=block_threads) as (bm, bn):
            A_shared = T.alloc_shared((block_m, block_k), dtype)
            B_shared = T.alloc_shared((block_k, block_n), dtype)
            C_acc = T.alloc_fragment((block_m, block_n), T.float32)
            T.clear(C_acc)
            for k in T.Pipelined(T.cdiv(K, block_k)):
                T.copy(A[bm * block_m, k * block_k], A_shared)
                T.copy(B[k * block_k, bn * block_n], B_shared)
                T.gemm(A_shared, B_shared, C_acc)
            T.copy(C[bm * block_m, bn * block_n], C_acc)
    return gemm_kernel

M, N, K = 4096, 1024, 4096
BLOCK_M, BLOCK_N, K_TILE = 64, 32, 64
DEV = 'cuda'
THREADS = 512

A = torch.randn((M, K), device=DEV)
B = torch.randn((K, N), device=DEV)
C = torch.empty((M, N), device=DEV)

ref_C = torch.matmul(A, B)

my_gemm = gemm(M, N, K, BLOCK_M, BLOCK_N, K_TILE, THREADS)
my_gemm(A, B, C)

torch.testing.assert_close(C, ref_C)

print('C=', C)
print('ref_C=', ref_C)
