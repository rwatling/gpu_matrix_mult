#ifndef CUDA_INCLUDES_H
#define CUDA_INCLUDES_H 1
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <nvml.h>
#include <cuda_profiler_api.h>
#define BLOCK_SIZE 16
#define SHMEM_SIZE BLOCK_SIZE*BLOCK_SIZE*sizeof(float)
#endif