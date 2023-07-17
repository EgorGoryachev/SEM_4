#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <stdlib.h>
#define N (33 * 1024)


__global__ void add(int* a, int* b, int* c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

int main(void)
{
    int* a, * b, * c;
    int* dev_a, * dev_b, * dev_c;
    a = (int*)malloc(N * sizeof(int));
    b = (int*)malloc(N * sizeof(int));
    c = (int*)malloc(N * sizeof(int));
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = 2 * i;
    }
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    add << < (N + 255) / 256, 256) >> > (dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    bool success = true;
    for (int i = 0; i < N; i++)
    {
        if ((a[i] + b[i]) != c[i])
        {
            printf("Error:  %d + %d != %d\n", a[i], b[i], c[i]);
            success = false;
        }
    }
    if (success) {
        printf("The work is completed\n");
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    free(a);
    free(b);
    free(c);
    return 0;
}