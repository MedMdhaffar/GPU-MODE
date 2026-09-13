import torch, importlib.util

# Load precompiled .so directly, no compilation
import glob
so_path = glob.glob('./load_inline_cuda/*.so')[0]
spec = importlib.util.spec_from_file_location('square_matrix_extension', so_path)
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)

# Use a large matrix — small ones give the "0.4 waves" warning
a = torch.randn(4096, 4096, device='cuda')
result = ext.square_matrix(a)
print(result)
