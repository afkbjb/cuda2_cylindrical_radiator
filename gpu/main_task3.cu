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
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int n = 32, m = 32, p = 10;
    int bx = 16, by = 16;
    bool do_avg = false, skip_cpu = false, timing = false;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n")  && i+1<argc) n  = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-m") && i+1<argc) m  = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-p") && i+1<argc) p  = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-bx")&& i+1<argc) bx = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-by")&& i+1<argc) by = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-a"))  do_avg   = true;
        else if (!std::strcmp(argv[i], "-c"))  skip_cpu = true;
        else if (!std::strcmp(argv[i], "-t"))  timing   = true;
        else {
            std::cerr << "Unknown option " << argv[i] << "\n";
            return EXIT_FAILURE;
        }
    }

    // Check if the blocksize is divisible
    if (m % bx != 0 || n % by != 0) {
        std::cerr << "Error: block (" << bx << "×" << by
                  << ") must divide matrix (" << n << "×" << m << ")\n";
        return EXIT_FAILURE;
    }
    dim3 block(bx, by), grid(m/bx, n/by);

    // Allocate & init host
    float *h_init  = new float[n*m];
    float *h_next0 = new float[n*m];
    initialize(h_init, h_next0, n, m);

    // CPU compute (optional)
    float *h_cpu = nullptr, *h_next_cpu = nullptr;
    double cpuTime = 0.0;
    if (!skip_cpu) {
        h_cpu      = new float[n*m];
        h_next_cpu = new float[n*m];
        std::memcpy(h_cpu,      h_init,  n*m*sizeof(float));
        std::memcpy(h_next_cpu, h_next0, n*m*sizeof(float));

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < p; ++it) {
            propagate_step(h_cpu, h_next_cpu, n, m);
            std::swap(h_cpu, h_next_cpu);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        cpuTime = std::chrono::duration<double,std::milli>(t1-t0).count();
        if (timing) std::cout << "CPU compute: " << cpuTime << " ms\n";
    }

    // GPU memory alloc & timing
    size_t matB = size_t(n)*m*sizeof(float);
    size_t avgB = size_t(n)*sizeof(float);
    float *d1, *d2, *davg;
    cudaEvent_t e0, e1;
    check(cudaEventCreate(&e0));
    check(cudaEventCreate(&e1));
    cudaEventRecord(e0);
    check(cudaMalloc(&d1, matB));
    check(cudaMalloc(&d2, matB));
    check(cudaMalloc(&davg, avgB));
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float tAlloc = 0.0f; cudaEventElapsedTime(&tAlloc,e0,e1);
    if (timing) std::cout << "GPU alloc: " << tAlloc << " ms\n";

    // H2D
    cudaEventRecord(e0);
    check(cudaMemcpy(d1, h_init,  matB, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d2, h_next0, matB, cudaMemcpyHostToDevice));
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float tH2D=0.0f; cudaEventElapsedTime(&tH2D,e0,e1);
    if (timing) std::cout << "GPU H2D: " << tH2D << " ms\n";

    // GPU propagate
    cudaEventRecord(e0);
    for (int it = 0; it < p; ++it) {
        propagate_kernel<<<grid,block>>>(d1,d2,n,m);
        cudaDeviceSynchronize();
        std::swap(d1,d2);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float tProp=0.0f; cudaEventElapsedTime(&tProp,e0,e1);
    if (timing) std::cout << "GPU propagate: " << tProp << " ms\n";

    // GPU average
    cudaEventRecord(e0);
    average_kernel<<<n,1>>>(d1,davg,n,m);
    cudaDeviceSynchronize();
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float tAvg=0.0f; cudaEventElapsedTime(&tAvg,e0,e1);
    if (timing) std::cout << "GPU average: " << tAvg << " ms\n";

    // D2H
    float *h_res = new float[n*m];
    float *h_avr = new float[n];
    cudaEventRecord(e0);
    check(cudaMemcpy(h_res, d1,   matB, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(h_avr, davg, avgB, cudaMemcpyDeviceToHost));
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float tDtoH=0.0f; cudaEventElapsedTime(&tDtoH,e0,e1);
    if (timing) std::cout << "GPU D2H: " << tDtoH << " ms\n";

    // Compare & report
    if (!skip_cpu) {
        int cntM = 0, cntA = 0;
        float maxM = 0.0f, maxA = 0.0f;
        for (int i = 0; i < n*m; ++i) {
            float d = std::fabs(h_cpu[i] - h_res[i]);
            if (d > 1e-4f) ++cntM;
            maxM = std::max(maxM, d);
        }
        float* h_avc = new float[n];
        compute_averages(h_cpu, h_avc, n, m);
        for (int i = 0; i < n; ++i) {
            float d = std::fabs(h_avc[i] - h_avr[i]);
            if (d > 1e-4f) ++cntA;
            maxA = std::max(maxA, d);
        }
        std::cout << "Matrix mismatches (>1e-4): " << cntM
                  << ", max diff: " << maxM << "\n";
        std::cout << "Average mismatches (>1e-4): " << cntA
                  << ", max diff: " << maxA << "\n";
        if (timing) {
            std::cout << "Speedup: " << (cpuTime / (tProp + tAvg))
                      << "\n";
        }
        delete[] h_avc;
    }

    // -a output per-row averages
    if (do_avg) {
        std::cout << "Row averages (GPU):\n";
        for (int i = 0; i < n; ++i) {
            std::cout << "  row " << i << ": " << h_avr[i] << "\n";
        }
    }

    delete[] h_init;   delete[] h_next0;
    delete[] h_cpu;    delete[] h_next_cpu;
    delete[] h_res;    delete[] h_avr;
    cudaFree(d1);      cudaFree(d2);      cudaFree(davg);
    cudaEventDestroy(e0); cudaEventDestroy(e1);

    return 0;
}
