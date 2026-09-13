import torch
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load_inline
import re
import subprocess

def show_img(x, figsize=(4,3), **kwargs):
    "Display HW or CHW format image `x`"                    
    plt.figure(figsize=figsize)
    plt.axis('off')
    if len(x.shape)==3: x = x.permute(1,2,0)  # CHW -> HWC
    plt.imshow(x.cpu(), **kwargs)

def load_cuda(cuda_src, cpp_src, funcs, opt=True, verbose=False, name=None):
    "Simple wrapper for torch.utils.cpp_extension.load_inline"
    if name is None: name = funcs[0]
    flags = "-O3 -Xptxas -O3 -Xcompiler -O3" if opt else "-O0 -Xptxas -O0 -Xcompiler -O0"
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=[flags], verbose=verbose, name=name)

def cdiv(a,b):
    "Int ceiling division of `a` over `b`"
    return (a+b-1)//b

def get_sig(fname, src):
    res = re.findall(rf'^(.+\s+{fname}\(.*?\))\s*{{?\s*$', src, re.MULTILINE)
    return res[0]+';' if res else None
    
cuda = r'''

#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CUDA_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a+b-1)/b;}

__global__ void matmul_kernel(const float* m, const float* n, float* output, int h, int w, int k)
{
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.x*blockDim.x + threadIdx.x;

    if (r >= h || c >= w) return;
    float o = 0.0;
    for (int i = 0; i < k; i++){
        o += m[r * k + i] * n[i * w + c];
    }
    output[r * w + c] = o;
}
torch::Tensor matmul_bk(const torch::Tensor& m, const torch::Tensor& n)
{
    TORCH_CHECK(m.scalar_type() == torch::kFloat, "m must be float32");
    TORCH_CHECK(n.scalar_type() == torch::kFloat, "n must be float32");
    
    CHECK_INPUT(m); 
    CHECK_INPUT(n);

    auto h = m.size(0);
    auto w = n.size(1);
    auto k2 = n.size(0);
    auto k = m.size(1);

    TORCH_CHECK(k==k2, "Size mismatch!");

    torch::Tensor output = torch::zeros({m.size(0), n.size(1)}, m.options());
    dim3 tpb(16, 16);   //threads per block
    dim3 blocks(cdiv(w, 16), cdiv(h, 16));  //number of blocks, we need to cover the whole output matrix, so we divide its dimensions by the block size and round up
    
    matmul_kernel<<<blocks,tpb>>>(
        m.data_ptr<float>(),
        n.data_ptr<float>(),
        output.data_ptr<float>(),
        m.size(0), n.size(1), m.size(1)
    );    
    CUDA_ERR(cudaGetLastError());
    CUDA_ERR(cudaDeviceSynchronize());

    return output;
}
'''



cpp_src = """
torch::Tensor matmul_bk(
    const torch::Tensor& m,
    const torch::Tensor& n
);
"""


matmul_ext = load_cuda(
    cuda_src=cuda,
    cpp_src=cpp_src,
    funcs=["matmul_bk"],
    verbose=True
)
a = torch.randn(
    128,
    64,
    device="cuda",
    dtype=torch.float32
)

b = torch.randn(
    64,
    256,
    device="cuda",
    dtype=torch.float32
)
c_custom = matmul_ext.matmul_bk(a, b)
c_torch = a @ b

print("allclose:",
      torch.allclose(
          c_custom,
          c_torch,
          atol=1e-4,
          rtol=1e-4
      ))

print(
    "max error:",
    (c_custom - c_torch)
    .abs()
    .max()
)