import os

import torch
import torch.utils.cpp_extension

os.environ["CC"] = "gcc-11"
os.environ["CXX"] = "g++-11"

cuda_begin = r'''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
'''

cuda_src = cuda_begin + r'''
const int TILE_SIZE = 16;

__global__ void tiled_matmul_kernel(
    const float* a,
    const float* b,
    float* out,
    int n
) {
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;
    const int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tiled_col_a = tile * TILE_SIZE + threadIdx.x;
        const int tiled_row_b = tile * TILE_SIZE + threadIdx.y;

        if (row < n && tiled_col_a < n) {
        a_tile[threadIdx.y][threadIdx.x] = a[row * n + tiled_col_a];
        } else {
        a_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (tiled_row_b < n && col < n) {
        b_tile[threadIdx.y][threadIdx.x] = b[tiled_row_b * n + col];             
        }else{
        b_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int kk = 0; kk < TILE_SIZE; ++kk) {
            acc += a_tile[threadIdx.y][kk] * b_tile[kk][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        out[row * n + col] = acc;
    }
}

torch::Tensor tiled_matmul_out(const torch::Tensor& a, const torch::Tensor& b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    const int64_t n = a.size(0);
    
    auto output = torch::empty({a.size(0), a.size(0)}, a.options());

    const dim3 threads(TILE_SIZE, TILE_SIZE);
    const dim3 blocks(cdiv(static_cast<int>(n), TILE_SIZE), cdiv(static_cast<int>(n), TILE_SIZE));

    tiled_matmul_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(n)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
'''

cpp_src = """
torch::Tensor tiled_matmul_out(const torch::Tensor& a, const torch::Tensor& b);
"""

print("Compiling tiled matmul extension...")
tiled_matmul_module = torch.utils.cpp_extension.load_inline(
    "tiled_matmul_ext",
    cpp_src,
    cuda_src,
    functions=["tiled_matmul_out"],
    extra_cuda_cflags=["--ptxas-options=-v", "-v"],
    verbose=True,
)

print("Loaded tiled matmul extension.")


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return tiled_matmul_module.tiled_matmul_out(a.contiguous(), b.contiguous())

n = 512
a = torch.randn(n, n, device="cuda", dtype=torch.float32)
b = torch.randn(n, n, device="cuda", dtype=torch.float32)

out = matmul(a, b)
print(torch.allclose(out, a @ b, atol=1e-3, rtol=1e-3))
