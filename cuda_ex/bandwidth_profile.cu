#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void performCopies(float *h_a, float *h_b, float *d,
                   size_t n, const char *description)
{
    const size_t bytes = n * sizeof(float);

    printf("\n%s transfers\n", description);

    // CPU → GPU
    cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice);

    // GPU → CPU
    cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost);

    // Verify the round trip.
    for (size_t i = 0; i < n; ++i) {
        if (h_a[i] != h_b[i]) {
            printf("*** %s transfers failed ***\n", description);
            return;
        }
    }

    printf("Verification passed.\n");
}

int main()
{
    const size_t nElements = 4 * 1024 * 1024;
    const size_t bytes = nElements * sizeof(float);

    // Ordinary CPU memory.
    float *h_aPageable = static_cast<float *>(malloc(bytes));
    float *h_bPageable = static_cast<float *>(malloc(bytes));

    // Pinned CPU memory.
    float *h_aPinned;
    float *h_bPinned;
    cudaMallocHost(reinterpret_cast<void **>(&h_aPinned), bytes);
    cudaMallocHost(reinterpret_cast<void **>(&h_bPinned), bytes);

    // GPU memory.
    float *d_a;
    cudaMalloc(reinterpret_cast<void **>(&d_a), bytes);

    // Initialize the input arrays.
    for (size_t i = 0; i < nElements; ++i) {
        h_aPageable[i] = static_cast<float>(i);
    }

    memcpy(h_aPinned, h_aPageable, bytes);
    memset(h_bPageable, 0, bytes);
    memset(h_bPinned, 0, bytes);

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);

    printf("Device: %s\n", properties.name);
    printf("Transfer size: %zu MiB\n", bytes / (1024 * 1024));

    performCopies(h_aPageable, h_bPageable, d_a,
                  nElements, "Pageable");

    performCopies(h_aPinned, h_bPinned, d_a,
                  nElements, "Pinned");

    cudaFree(d_a);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    free(h_aPageable);
    free(h_bPageable);

    return 0;
}