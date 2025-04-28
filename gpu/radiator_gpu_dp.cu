#include "radiator_gpu_dp.h"
#define IDX(i,j,m) ((i)*(m)+(j))

__global__ void propagate_kernel_dp(
    const double* oldM,
    double*       newM,
    int           n,
    int           m)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row>=n||col>=m) return;

    if (col==0) {
        newM[IDX(row,0,m)] = oldM[IDX(row,0,m)];
    } else {
        int jm2 = (col-2 + m)%m;
        int jm1 = (col-1 + m)%m;
        int jp1 = (col+1)%m;
        int jp2 = (col+2)%m;
        double sum =
            1.60  * oldM[IDX(row,jm2,m)] +
            1.55  * oldM[IDX(row,jm1,m)] +
                  oldM[IDX(row,col,m)] +
            0.60  * oldM[IDX(row,jp1,m)] +
            0.25  * oldM[IDX(row,jp2,m)];
        newM[IDX(row,col,m)] = sum * 0.2;
    }
}

__global__ void average_kernel_dp(
    const double* mat,
    double*       avgs,
    int           n,
    int           m)
{
    int row = blockIdx.x;
    if (row >= n) return;
    double sum = 0.0;
    for (int j = 0; j < m; ++j) sum += mat[IDX(row,j,m)];
    avgs[row] = sum / m;
}
