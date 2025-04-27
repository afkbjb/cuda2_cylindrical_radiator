#ifndef RADIATOR_GPU_H
#define RADIATOR_GPU_H

// propagate 单步核：16×16 block
__global__ void propagate_kernel(
    const float* oldM, float* newM,
    int n, int m);

// average 核：每行一个 block
__global__ void average_kernel(
    const float* mat, float* avgs,
    int n, int m);

#endif // RADIATOR_GPU_H
