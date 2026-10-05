#include <cuda_runtime.h>
#include <cstdio>

int main()
{
    cudaDeviceProp prop;

    cudaError_t status = cudaGetDeviceProperties(&prop, 0);
    if (status != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(status));
        return 1;
    }

    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Asynchronous copy engines: %d\n", prop.asyncEngineCount);

    printf("Concurrent copy and kernel execution: %s\n",
           prop.asyncEngineCount > 0 ? "Supported" : "Not supported");

    printf("Concurrent kernels: %s\n",
           prop.concurrentKernels ? "Supported" : "Not supported");

    return 0;
}