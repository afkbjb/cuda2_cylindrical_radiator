#include "radiator_cpu.h"

// Fast index calculation
inline int idx(int i, int j, int m) { return i * m + j; }

void initialize(float *mat, float *next, int n, int m) {
    // Boundary column j=0
    for (int i = 0; i < n; ++i) {
        float b = 0.98f * (float)((i + 1) * (i + 1)) / (float)(n * n);
        mat[idx(i, 0, m)]  = b;
        next[idx(i, 0, m)] = b;
    }
    for (int i = 0; i < n; ++i) {
        float b = mat[idx(i, 0, m)];
        for (int j = 1; j < m; ++j) {
            float factor = (float)((m - j) * (m - j)) / (float)(m * m);
            float v = b * factor;
            mat[idx(i, j, m)]  = v;
            next[idx(i, j, m)] = v;
        }
    }
}

void propagate_step(const float *oldM, float *newM, int n, int m) {
    for (int i = 0; i < n; ++i) {
        // Keep boundary column unchanged
        newM[idx(i, 0, m)] = oldM[idx(i, 0, m)];
        for (int j = 1; j < m; ++j) {
            // Wrap-around indexing
            int jm2 = (j - 2 + m) % m;
            int jm1 = (j - 1 + m) % m;
            int jp1 = (j + 1) % m;
            int jp2 = (j + 2) % m;
            float sum =
                1.60f * oldM[idx(i, jm2, m)] +
                1.55f * oldM[idx(i, jm1, m)] +
                     1.0f * oldM[idx(i, j,   m)] +
                0.60f * oldM[idx(i, jp1, m)] +
                0.25f * oldM[idx(i, jp2, m)];
            newM[idx(i, j, m)] = sum / 5.0f;
        }
    }
}

void compute_averages(const float *mat, float *avgs, int n, int m) {
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < m; ++j)
            sum += mat[idx(i, j, m)];
        avgs[i] = sum / m;
    }
}
