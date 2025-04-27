#ifndef RADIATOR_CPU_H
#define RADIATOR_CPU_H

void initialize(float *mat, float *next, int n, int m);

void propagate_step(const float *oldM, float *newM, int n, int m);

void compute_averages(const float *mat, float *avgs, int n, int m);

#endif // RADIATOR_CPU_H
