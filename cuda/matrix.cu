// nvcc -o matrix_mul)file name) matrix_mul.cu (with extenstion)
// ./matrix_mul
// .cu

#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// CUDA kernel for matrix multiplication
__global__ void matrix_multiply(float *a, float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < n && col < n) {
        for (int i = 0; i < n; ++i) {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    int n = 1024;
    size_t size = n * n * sizeof(float);

    // Allocate host memory
    float* a = new float[n * n];
    float* b = new float[n * n];
    float* c = new float[n * n];

    // Initialize matrices
    for (int i = 0; i < n * n; ++i) {
        a[i] = static_cast<float>(i % n);
        b[i] = static_cast<float>(i % n);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch and time the kernel
    cudaEventRecord(start);
    matrix_multiply<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "Elapsed time for matrix multiplication: " << elapsed_time << " ms" << std::endl;

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}



