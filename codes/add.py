import torch
import tilelang.language as T
from tilelang import jit

@jit
def add(N: int, block_size: int = 128, dtype=torch.float32):
    @T.prim_func
    def add_kernel(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)): # pyright: ignore[reportInvalidTypeForm]
        with T.Kernel(T.cdiv(N, block_size), threads=block_size) as bx:
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < N:
                    C[idx] = A[idx] + B[idx]
    return add_kernel

DEVICE='mps'
N =270
a = torch.randn((N), device=DEVICE)
b = torch.randn((N,), device=DEVICE)
c = torch.empty((N,), device=DEVICE)

my_add = add(N)
my_add(a, b, c)

ref_c = a + b

torch.testing.assert_close(c, ref_c)
print('ref_c=', ref_c)
print('c=', c)

print(my_add.get_kernel_source()) # type: ignore

# profiler = my_add.get_profiler() # type: ignore
# print(profiler.do_bench())