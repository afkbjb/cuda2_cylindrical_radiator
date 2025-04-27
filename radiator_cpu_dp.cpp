#include "radiator_cpu_dp.h"
#include <cstring>

#define IDX(i,j,m) ((i)*(m)+(j))

void initialize_dp(double* h_init, double* h_next0, int n, int m) {
    // boundary column 0
    for (int i = 0; i < n; ++i) {
        h_init[IDX(i,0,m)]  = 0.98 * double((i+1)*(i+1)) / double(n*n);
        h_next0[IDX(i,0,m)] = h_init[IDX(i,0,m)];
    }
    // interior init
    for (int i = 0; i < n; ++i) {
        for (int j = 1; j < m; ++j) {
            double v = h_init[IDX(i,0,m)];
            h_init[IDX(i,j,m)]  = v * double((m-j)*(m-j)) / double(m*m);
            h_next0[IDX(i,j,m)] = h_init[IDX(i,j,m)];
        }
    }
}

void propagate_step_dp(double* mat, double* next, int n, int m) {
    for (int i = 0; i < n; ++i) {
        // column 0 fixed
        next[IDX(i,0,m)] = mat[IDX(i,0,m)];
        for (int j = 1; j < m; ++j) {
            int jm2 = (j-2 + m) % m;
            int jm1 = (j-1 + m) % m;
            int jp1 = (j+1) % m;
            int jp2 = (j+2) % m;
            double sum =
                1.60  * mat[IDX(i,jm2,m)] +
                1.55  * mat[IDX(i,jm1,m)] +
                       mat[IDX(i, j,  m)] +
                0.60  * mat[IDX(i,jp1,m)] +
                0.25  * mat[IDX(i,jp2,m)];
            next[IDX(i,j,m)] = sum * 0.2;
        }
    }
}

void compute_averages_dp(const double* mat, double* avgs, int n, int m) {
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < m; ++j) {
            sum += mat[IDX(i,j,m)];
        }
        avgs[i] = sum / m;
    }
}
