import torch 

a = torch.tensor([1, 2, 3])


print(torch.square(a))

print(a ** 2)

print(a * a)

def time_pytorch_function(func, input):
    #CUDA is asynchronous, so we need to use events to measure the time correctly
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    #warm up 
    for _ in range(5):
        func(input)

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)    


def square2(x):
    return x ** 2

def square3(x):
    return x * x

b = torch.randn(10000, 10000).cuda()

print(time_pytorch_function(square2, b))
print(time_pytorch_function(square3, b))
print(time_pytorch_function(torch.square, b))


print('=================')
print("profiling torch.square")
print('=================')


# Now profile each function using pytorch profiler
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    torch.square(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a * a")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square2(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a ** 2")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square3(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

