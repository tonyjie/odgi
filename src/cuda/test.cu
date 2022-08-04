#include <stdio.h>
#include <cuda.h>

__global__ void cuda_hello_device() {
    printf("Hello World from CUDA device\n");
}

void cuda_hello_host() {
    printf("Hello World from CUDA host\n");
    cuda_hello_device<<<1,10>>>();
    cudaDeviceSynchronize();
    return;
}
