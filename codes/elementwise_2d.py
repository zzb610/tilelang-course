import torch
import tilelang.language as T
from tilelang import jit

@jit
def add_2d(M: int, N: int, block_size: int = 64, dtype = torch.float32):
    
    @T.prim_func
    def add_2d_kernel(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)): # type: ignore
        with T.Kernel(T.cdiv(M, block_size), T.cdiv(N, block_size), threads=1024) as (bx, by):
            for i, j in T.Parallel(block_size, block_size):
                idx = bx * block_size + i
                idy = by * block_size + j
                if idx < M and idy < N: 
                    C[idx, idy] = A[idx, idy] + B[idx, idy]
    
    return add_2d_kernel

M, N = 1000, 1000
DEV = 'mps'
A = torch.randn(M, N, device=DEV)
B = torch.randn(M, N, device=DEV)
C = torch.empty_like(A)

my_add_2d = add_2d(M, N)
ref_C = A + B
my_add_2d(A, B, C)

torch.testing.assert_close(C, ref_C)
print('C=',C)
print('ref_C=', ref_C)