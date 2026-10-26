// Matrix multiplication in CUDA C (SJSU DATA266 coursework)
// SEED = 2096
//
// C = A * B, all matrices are N x N, stored row-major as flat float arrays.
//
// Threading model: we launch a 2D grid of 2D thread blocks. Each thread
// computes exactly one element C[row][col] by looping over the shared
// dimension. This keeps the block/thread mapping simple to explain:
//   - blockDim = 16 x 16 threads per block
//   - gridDim  = enough blocks in x and y to cover the whole N x N matrix
//   - a thread's (row, col) in C is derived from its block index and
//     thread index within the block

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define BLOCK_SIZE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matmul_cpu(const float* A, const float* B, float* C, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

void fill_random(float* mat, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n * n; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

float max_abs_diff(const float* a, const float* b, int n) {
    float worst = 0.0f;
    for (int i = 0; i < n * n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > worst) worst = diff;
    }
    return worst;
}

int main(int argc, char** argv) {
    int n = 256;
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);       // GPU result, copied back
    float* h_C_ref = (float*)malloc(bytes);   // CPU reference result

    fill_random(h_A, n, 2096);
    fill_random(h_B, n, 2097);

    // ---- CPU baseline timing ----
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A, h_B, h_C_ref, n);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // ---- GPU setup ----
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaEvent_t e_h2d_start, e_h2d_end, e_kernel_start, e_kernel_end, e_d2h_start, e_d2h_end;
    cudaEventCreate(&e_h2d_start);
    cudaEventCreate(&e_h2d_end);
    cudaEventCreate(&e_kernel_start);
    cudaEventCreate(&e_kernel_end);
    cudaEventCreate(&e_d2h_start);
    cudaEventCreate(&e_d2h_end);

    // ---- Warm-up (untimed): forces CUDA context creation and kernel JIT
    // compilation to happen here instead of contaminating the timed run below.
    matmul_kernel<<<1, 1>>>(d_A, d_B, d_C, 0);
    cudaDeviceSynchronize();

    // ---- Host to device transfer ----
    cudaEventRecord(e_h2d_start);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(e_h2d_end);

    // ---- Kernel launch ----
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEventRecord(e_kernel_start);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaEventRecord(e_kernel_end);

    // ---- Device to host transfer ----
    cudaEventRecord(e_d2h_start);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(e_d2h_end);

    cudaEventSynchronize(e_d2h_end);

    float h2d_ms, kernel_ms, d2h_ms;
    cudaEventElapsedTime(&h2d_ms, e_h2d_start, e_h2d_end);
    cudaEventElapsedTime(&kernel_ms, e_kernel_start, e_kernel_end);
    cudaEventElapsedTime(&d2h_ms, e_d2h_start, e_d2h_end);

    float transfer_ms = h2d_ms + d2h_ms;
    double gpu_end_to_end_ms = kernel_ms + transfer_ms;
    double speedup = cpu_ms / gpu_end_to_end_ms;

    float err = max_abs_diff(h_C, h_C_ref, n);

    printf("N=%d\n", n);
    printf("CPU time (ms): %.4f\n", cpu_ms);
    printf("GPU kernel time (ms): %.4f\n", kernel_ms);
    printf("H2D+D2H transfer time (ms): %.4f\n", transfer_ms);
    printf("GPU end-to-end time (ms): %.4f\n", gpu_end_to_end_ms);
    printf("Speedup (CPU / GPU end-to-end): %.4f\n", speedup);
    printf("Max abs diff vs CPU reference: %e\n", err);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}
