int main()
{
    const unsigned int N = 1048576;  //or 2^20
    const unsigned int bytes = N * sizeof(int);   // 4MB array

    int *h_a = (int*)malloc(bytes);  //allocate a 4MB array on the host with h_a is pointer to int

    int *d_a;
    cudaMalloc((int**)&d_a, bytes);     //allocate a 4MB array on the device andd_a is pointer to int

    memset(h_a, 0, bytes);   //sets all bytes of the allocated memory to zero

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);  //Copy CPU → GPU
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);   //Copy GPU → CPU


    cudaFree(d_a);  // Release GPU memory
    free(h_a);     // Release CPU memory
    
    
    return 0;
}