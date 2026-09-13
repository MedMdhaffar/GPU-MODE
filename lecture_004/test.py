import torch
import torch.utils.cpp_extension
import os
os.environ["CC"] = "gcc-12"
os.environ["CXX"] = "g++-12"

# based on Jeremy's Lecture 3 notebook
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
__global__ void my_gelu_kernel(float* out, float* inp, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = inp[i];
    out[i] = 0.5f * x * (1.0f+ tanhf(sqrtf(2.0f/3.141592653589793f) * (x+0.044715f * (x*x*x))));
}

torch::Tensor my_gelu_out(torch::Tensor output, const torch::Tensor& inp) {
    CHECK_INPUT(inp);
    int n = inp.numel();
    TORCH_CHECK((output.sizes() == inp.sizes())  || (output.device() == inp.device())
                || (output.scalar_type() == inp.scalar_type()));
    int threads = 256;
    my_gelu_kernel<<<cdiv(n, threads), threads>>>(
        output.data_ptr<float>(), inp.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor my_gelu(const torch::Tensor& inp) {
    CHECK_INPUT(inp);
    auto output = torch::empty_like(inp);
    my_gelu_out(output, inp);
    return output;
}
'''

cpp_src = """
torch::Tensor my_gelu(const torch::Tensor& inp);
torch::Tensor my_gelu_out(torch::Tensor output, const torch::Tensor& inp);
"""

print("Compiling...")
gelu_module_opt = torch.utils.cpp_extension.load_inline(
    "test_ext_gelu", cpp_src, cuda_src, 
    functions=['my_gelu', 'my_gelu_out'], extra_cuda_cflags=['--ptxas-options=-v'], verbose=True)

print("Loaded!")