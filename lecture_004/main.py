import torch
from tiled_matmul import matmul

n = 512
a = torch.randn(n, n, device="cuda", dtype=torch.float32)
b = torch.randn(n, n, device="cuda", dtype=torch.float32)

out = matmul(a, b)
print(torch.allclose(out, a @ b, atol=1e-3, rtol=1e-3))
