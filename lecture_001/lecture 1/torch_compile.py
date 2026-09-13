# TORCH_LOGS='OUTPUT_CODE' python torch_compile.py


import torch


def square(x):
    return x * x

opt_square = torch.compile(square)
a = torch.randn(10000, 10000).cuda()
print(opt_square(a))