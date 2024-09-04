#ifndef MAT_MULT_H
#define MAT_MULT_H 1
#include "cuda_includes.cuh"
__global__ void mat_update(float* A, float* B, float* C, float* CC, int m, int n, int r);
__host__ void host_mat_update(float* A, float* B, float* C, float* CC, int m, int n, int r);
__global__ void tiled_mat_update(float* A, float* B, float* C, float* CC, int m, int n, int r);
#endif