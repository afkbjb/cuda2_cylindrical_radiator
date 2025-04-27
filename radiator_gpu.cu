#include "radiator_gpu.h"

#define IDX(i,j,m) ((i)*(m) + (j))

__global__ void propagate_kernel(
    const float* oldM,
    float*       newM,
    int          n,
    int          m)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= m) return;

    if (col == 0) {
        // Boundary remains unchanged
        newM[IDX(row,0,m)] = oldM[IDX(row,0,m)];
    } else {
        // Wraparound index
        int jm2 = (col - 2 + m) % m;
        int jm1 = (col - 1 + m) % m;
        int jp1 = (col + 1) % m;
        int jp2 = (col + 2) % m;
        float sum =
            1.60f * oldM[IDX(row,jm2,m)] +
            1.55f * oldM[IDX(row,jm1,m)] +
                 1.0f * oldM[IDX(row, col,   m)] +
            0.60f * oldM[IDX(row,jp1,m)] +
            0.25f * oldM[IDX(row,jp2,m)];
        newM[IDX(row,col,m)] = sum * 0.2f; // Equivalent to /5.0f
    }
}

__global__ void average_kernel(
    const float* mat,
    float*       avgs,
    int          n,
    int          m)
{
    // Each block corresponds to one row, threadIdx.x is not used
    int row = blockIdx.x;
    if (row >= n) return;

    float sum = 0.0f;
    // Single-threaded summation
    for (int j = 0; j < m; ++j) {
        sum += mat[IDX(row,j,m)];
    }
    avgs[row] = sum / m;
}
