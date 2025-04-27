#ifndef RADIATOR_CPU_DP_H
#define RADIATOR_CPU_DP_H

void initialize_dp(double* h_init, double* h_next0, int n, int m);

void propagate_step_dp(double* mat, double* next, int n, int m);

void compute_averages_dp(const double* mat, double* avgs, int n, int m);

#endif // RADIATOR_CPU_DP_H
