#ifndef RADIATOR_GPU_DP_H
#define RADIATOR_GPU_DP_H

__global__ void propagate_kernel_dp(
    const double* oldM,
    double*       newM,
    int           n,
    int           m);

__global__ void average_kernel_dp(
    const double* mat,
    double*       avgs,
    int           n,
    int           m);

#endif // RADIATOR_GPU_DP_H
