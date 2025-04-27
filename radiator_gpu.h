#ifndef RADIATOR_GPU_H
#define RADIATOR_GPU_H

// propagates matrix with 16x16 blocks
__global__ void propagate_kernel(
    const float* oldM, float* newM,
    int n, int m);

// calculates row averages
__global__ void average_kernel(
    const float* mat, float* avgs,
    int n, int m);

#endif // RADIATOR_GPU_H
