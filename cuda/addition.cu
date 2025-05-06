// nvcc -o matrix_mul matrix_mul.cu
// ./matrix_mul
// .cu



#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// CUDA kernel for vector addition
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1000000;
    int size = n * sizeof(int);

    // Allocate host memory using new
    int* a = new int[n];
    int* b = new int[n];
    int* c = new int[n];

    // Initialize vectors
    for (int i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = i;
    }

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy host data to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify result
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (c[i] != 2 * i) {
            cout<< "Error at index " << i << ": " << c[i] << endl;
            success = false;
            break;
        }
    }

    if (success) {
        cout << "Vector addition successful!" << endl;
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
