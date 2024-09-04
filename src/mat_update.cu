#include "../include/mat_update.cuh"

__global__ void mat_update(float* A, float* B, float* C, float* CC, int m, int n, int r) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate global row and column
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    float sum = 0;

    if (row < m && col < n) {
        for (int i=0; i < r; i++) {
            sum += A[row*r+i] * B[i*n+col];
        }
        CC[row*n + col] = C[row*n + col] + sum;
    }
}

__host__ void host_mat_update(float* A, float* B, float* C, float* CC, int m, int n, int r) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            CC[i*n+j]=0;
            for (int k = 0; k < r; k++) {
                CC[i*n+j] += C[i*n+j] + A[i*r+k] * B[k*n+j];
            }
        }
    }
}

__global__ void tiled_mat_update(float* A, float* B, float* C, float* CC, int m, int n, int r) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    // Calculate global row and column
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    float sum = 0;

    if (row < m && col < n) {

        // Sweep across matrix by block
        for (int i = 0; i < (r+BLOCK_SIZE-1)/BLOCK_SIZE; i++) {
            
            // Load in elements for this block into shared memory
            if (i*BLOCK_SIZE+tx < r) {
                shared_A[ty][tx] = A[row*r + i*BLOCK_SIZE+tx];
            } else {
                shared_A[ty][tx] = 0;
            }

            if (i*BLOCK_SIZE+ty < r) {
                shared_B[ty][tx] = B[(i*BLOCK_SIZE+ty)*n+col];
            } else {
                shared_B[ty][tx] = 0;
            }

            __syncthreads();

            // Do matrix multiplication on the small matrix
            for (int j = 0; j < BLOCK_SIZE; j++) {
                sum += (shared_A[ty][j] * shared_B[j][tx]);
            }

            __syncthreads();
        }

        // Write back results
        CC[row*n + col] = C[row*n + col] + sum;
    }
}

