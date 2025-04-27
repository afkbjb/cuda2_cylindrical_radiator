#ifndef RADIATOR_CPU_H
#define RADIATOR_CPU_H

// 初始化矩阵：边界列和初始值，next 也赋相同值
void initialize(float *mat, float *next, int n, int m);

// 在 old → new 上做一次传播迭代
void propagate_step(const float *oldM, float *newM, int n, int m);

// 计算每行平均，放入 avgs[n]
void compute_averages(const float *mat, float *avgs, int n, int m);

#endif // RADIATOR_CPU_H
