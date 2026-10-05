int main()
{
    const unsigned int N = 1048576;
    const unsigned int bytes = N * sizeof(int);

    int *h_a = (int*)malloc(bytes);

    int *d_a;
    cudaMalloc((void**)&d_a, bytes);

    memset(h_a, 0, bytes);

    // CPU → GPU, then wait for completion
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // GPU → CPU, then wait for completion
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_a);
    free(h_a);

    return 0;
}
