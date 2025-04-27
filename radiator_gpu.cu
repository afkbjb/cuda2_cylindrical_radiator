#include "radiator_gpu.h"
#define IDX(i,j,m) ((i)*(m)+(j))
const int BX = 16, BY = 16;

__global__ void propagate_kernel(
    const float* oldM, float* newM,
    int n, int m)
{
    int i = blockIdx.y*BY + threadIdx.y;
    int j = blockIdx.x*BX + threadIdx.x;
    if (i>=n||j>=m) return;

    if (j==0) {
        newM[IDX(i,0,m)] = oldM[IDX(i,0,m)];
    } else {
        int jm2 = (j-2 + m)%m;
        int jm1 = (j-1 + m)%m;
        int jp1 = (j+1)%m;
        int jp2 = (j+2)%m;
        float s = 1.60f*oldM[IDX(i,jm2,m)]
                + 1.55f*oldM[IDX(i,jm1,m)]
                +       oldM[IDX(i,j ,m)]
                + 0.60f*oldM[IDX(i,jp1,m)]
                + 0.25f*oldM[IDX(i,jp2,m)];
        newM[IDX(i,j,m)] = s * 0.2f;
    }
}

__global__ void average_kernel(
    const float* mat, float* avgs,
    int n, int m)
{
    int i = blockIdx.x;
    if (i>=n) return;
    float sum = 0.0f;
    for (int j = 0; j < m; ++j) sum += mat[IDX(i,j,m)];
    avgs[i] = sum / m;
}
