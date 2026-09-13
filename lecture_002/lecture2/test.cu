//Compute C = A + B


#include <stdio.h>

__global__ void Test() {

    float *A_d;
    size__t size = n * sizeof(float);
    cudaMalloc((void**)&A_d,  size);   
    
    cudaFree(A_d);
//
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);


    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

}



int main() {
    Test<<<1,1>>>();
    return 0;
}