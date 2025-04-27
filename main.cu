#include <iostream>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include "radiator_cpu.h"     
#include "radiator_gpu.h"    

static void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

static const int BX = 16;
static const int BY = 16;

int main(int argc, char** argv) {
    // Default parameters
    int n = 32, m = 32, p = 10;
    bool do_avg = false, skip_cpu = false, timing = false;

    // Command line argument parsing
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i+1 < argc) {
            n = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "-m") && i+1 < argc) {
            m = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "-p") && i+1 < argc) {
            p = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "-a")) {
            do_avg = true;
        } else if (!std::strcmp(argv[i], "-c")) {
            skip_cpu = true;
        } else if (!std::strcmp(argv[i], "-t")) {
            timing = true;
        } else {
            std::cerr << "Unknown option " << argv[i] << "\n";
            return 1;
        }
    }

    // Check if block size divides matrix dimensions
    if (m % BX != 0 || n % BY != 0) {
        std::cerr << "Error: block (" << BX << "x" << BY
                  << ") must divide matrix (" << n << "x" << m << ")\n";
        return 1;
    }
    dim3 block(BX, BY), grid(m/BX, n/BY);

    // Allocate and initialize the "initial" matrix
    float* h_init  = new float[n*m];
    float* h_next0 = new float[n*m];
    initialize(h_init, h_next0, n, m);

    // CPU computation
    float *h_mat_cpu = nullptr, *h_next_cpu = nullptr;
    double cpuTime = 0.0;
    if (!skip_cpu) {
        // Copy initial data to CPU buffer
        h_mat_cpu  = new float[n*m];
        h_next_cpu = new float[n*m];
        std::memcpy(h_mat_cpu,  h_init,  n*m * sizeof(float));
        std::memcpy(h_next_cpu, h_next0, n*m * sizeof(float));

        // Timing and iteration
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < p; ++it) {
            propagate_step(h_mat_cpu, h_next_cpu, n, m);
            std::swap(h_mat_cpu, h_next_cpu);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        cpuTime = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (timing) {
            std::cout << "CPU compute: " << cpuTime << " ms\n";
        }
    }

    // GPU memory allocation and timing
    size_t matBytes = n * m * sizeof(float);
    size_t avgBytes = n * sizeof(float);
    float *d1, *d2, *dav;
    cudaEvent_t e0, e1;
    check(cudaEventCreate(&e0));
    check(cudaEventCreate(&e1));
    cudaEventRecord(e0);
    check(cudaMalloc(&d1, matBytes));
    check(cudaMalloc(&d2, matBytes));
    check(cudaMalloc(&dav, avgBytes));
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float tAlloc = 0.0f;
    cudaEventElapsedTime(&tAlloc, e0, e1);
    if (timing) {
        std::cout << "GPU alloc: " << tAlloc << " ms\n";
    }

    // Host to device copy and timing
    cudaEventRecord(e0);
    check(cudaMemcpy(d1, h_init,  matBytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d2, h_next0, matBytes, cudaMemcpyHostToDevice));
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float tH2D = 0.0f;
    cudaEventElapsedTime(&tH2D, e0, e1);
    if (timing) {
        std::cout << "GPU H2D: " << tH2D << " ms\n";
    }

    // GPU propagate iteration and timing
    cudaEventRecord(e0);
    for (int it = 0; it < p; ++it) {
        propagate_kernel<<<grid, block>>>(d1, d2, n, m);
        cudaDeviceSynchronize();
        std::swap(d1, d2);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float tProp = 0.0f;
    cudaEventElapsedTime(&tProp, e0, e1);
    if (timing) {
        std::cout << "GPU propagate: " << tProp << " ms\n";
    }

    // GPU average kernel and timing
    cudaEventRecord(e0);
    average_kernel<<<n, 1>>>(d1, dav, n, m);
    cudaDeviceSynchronize();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float tAvg = 0.0f;
    cudaEventElapsedTime(&tAvg, e0, e1);
    if (timing) {
        std::cout << "GPU average: " << tAvg << " ms\n";
    }

    // Device to host copy and timing
    float* h_res = new float[n*m];
    float* h_avr = new float[n];
    cudaEventRecord(e0);
    check(cudaMemcpy(h_res, d1,  matBytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(h_avr, dav, avgBytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float tDtoH = 0.0f;
    cudaEventElapsedTime(&tDtoH, e0, e1);
    if (timing) {
        std::cout << "GPU D2H: " << tDtoH << " ms\n";
    }

    // Compare results
    if (!skip_cpu) {
        int cntM = 0, cntA = 0;
        float maxM = 0.0f, maxA = 0.0f;
        // Matrix comparison
        for (int i = 0; i < n*m; ++i) {
            float d = std::fabs(h_mat_cpu[i] - h_res[i]);
            if (d > 1e-4f) ++cntM;
            if (d > maxM) maxM = d;
        }
        // Average comparison
        float* h_avc = new float[n];
        compute_averages(h_mat_cpu, h_avc, n, m);
        for (int i = 0; i < n; ++i) {
            float d = std::fabs(h_avc[i] - h_avr[i]);
            if (d > 1e-4f) ++cntA;
            if (d > maxA) maxA = d;
        }
        std::cout << "Matrix mismatches (>1e-4): " << cntM
                  << ", max diff: " << maxM << "\n";
        std::cout << "Average mismatches (>1e-4): " << cntA
                  << ", max diff: " << maxA << "\n";
        delete[] h_avc;

        if (timing) {
            double gpuComp = tProp + tAvg;
            std::cout << "Speedup (CPU/GPU compute): "
                      << (cpuTime / gpuComp) << "\n";
        }
    }

    // Output row averages if -a is set
    if (do_avg) {
        std::cout << "Row averages (GPU):\n";
        for (int i = 0; i < n; ++i) {
            std::cout << "  row " << i << ": " << h_avr[i] << "\n";
        }
    }

    delete[] h_init;
    delete[] h_next0;
    delete[] h_mat_cpu;
    delete[] h_next_cpu;
    delete[] h_res;
    delete[] h_avr;
    cudaFree(d1);
    cudaFree(d2);
    cudaFree(dav);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    return 0;
}
