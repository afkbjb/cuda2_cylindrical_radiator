#include <iostream>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include "radiator_cpu_dp.h"
#include "radiator_gpu_dp.h"

static void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<"\n";
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int n=32, m=32, p=10, bx=16, by=16;
    bool do_avg=false, skip_cpu=false, timing=false;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-n") && i+1<argc) n=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-m") && i+1<argc) m=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-p") && i+1<argc) p=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-bx")&& i+1<argc) bx=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-by")&& i+1<argc) by=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-a")) do_avg=true;
        else if(!strcmp(argv[i],"-c")) skip_cpu=true;
        else if(!strcmp(argv[i],"-t")) timing=true;
        else { std::cerr<<"Unknown "<<argv[i]<<"\n"; return 1; }
    }
    if(m%bx||n%by){
        std::cerr<<"Error: block ("<<bx<<"x"<<by
                 <<") must divide matrix ("<<n<<"x"<<m<<")\n";
        return 1;
    }
    dim3 block(bx,by), grid(m/bx,n/by);

    // init host
    double *h_init=new double[n*m], *h_next0=new double[n*m];
    initialize_dp(h_init,h_next0,n,m);

    // CPU
    double *h_cpu=nullptr, *h_next_cpu=nullptr;
    double cpuTime=0;
    if(!skip_cpu){
        h_cpu     = new double[n*m];
        h_next_cpu= new double[n*m];
        memcpy(h_cpu, h_init,   n*m*sizeof(double));
        memcpy(h_next_cpu,h_next0,n*m*sizeof(double));
        auto t0=std::chrono::high_resolution_clock::now();
        for(int it=0;it<p;it++){
            propagate_step_dp(h_cpu,h_next_cpu,n,m);
            std::swap(h_cpu,h_next_cpu);
        }
        auto t1=std::chrono::high_resolution_clock::now();
        cpuTime=std::chrono::duration<double,std::milli>(t1-t0).count();
        if(timing) std::cout<<"CPU compute: "<<cpuTime<<" ms\n";
    }

    // GPU alloc
    size_t matB=n*m*sizeof(double), avgB=n*sizeof(double);
    double *d1,*d2,*dav;
    cudaEvent_t e0,e1;
    check(cudaEventCreate(&e0)); check(cudaEventCreate(&e1));
    cudaEventRecord(e0);
    check(cudaMalloc(&d1,matB));
    check(cudaMalloc(&d2,matB));
    check(cudaMalloc(&dav,avgB));
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float tAlloc; cudaEventElapsedTime(&tAlloc,e0,e1);
    if(timing) std::cout<<"GPU alloc: "<<tAlloc<<" ms\n";

    // H2D
    cudaEventRecord(e0);
    check(cudaMemcpy(d1,h_init,matB,cudaMemcpyHostToDevice));
    check(cudaMemcpy(d2,h_next0,matB,cudaMemcpyHostToDevice));
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float tH2D; cudaEventElapsedTime(&tH2D,e0,e1);
    if(timing) std::cout<<"GPU H2D: "<<tH2D<<" ms\n";

    // propagate
    cudaEventRecord(e0);
    for(int it=0;it<p;it++){
        propagate_kernel_dp<<<grid,block>>>(d1,d2,n,m);
        cudaDeviceSynchronize();
        std::swap(d1,d2);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float tProp; cudaEventElapsedTime(&tProp,e0,e1);
    if(timing) std::cout<<"GPU propagate: "<<tProp<<" ms\n";

    // average
    cudaEventRecord(e0);
    average_kernel_dp<<<n,1>>>(d1,dav,n,m);
    cudaDeviceSynchronize();
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float tAvg; cudaEventElapsedTime(&tAvg,e0,e1);
    if(timing) std::cout<<"GPU average: "<<tAvg<<" ms\n";

    // D2H
    double *h_res=new double[n*m], *h_avr=new double[n];
    cudaEventRecord(e0);
    check(cudaMemcpy(h_res,d1,matB,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(h_avr,dav,avgB,cudaMemcpyDeviceToHost));
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float tDtoH; cudaEventElapsedTime(&tDtoH,e0,e1);
    if(timing) std::cout<<"GPU D2H: "<<tDtoH<<" ms\n";

    // compare
    if(!skip_cpu){
        int cntM=0,cntA=0; double maxM=0, maxA=0;
        for(int i=0;i<n*m;i++){
            double d=fabs(h_cpu[i]-h_res[i]);
            if(d>1e-8) cntM++;
            maxM = std::max(maxM,d);
        }
        double *h_avc=new double[n];
        compute_averages_dp(h_cpu,h_avc,n,m);
        for(int i=0;i<n;i++){
            double d=fabs(h_avc[i]-h_avr[i]);
            if(d>1e-8) cntA++;
            maxA = std::max(maxA,d);
        }
        std::cout<<"Matrix mismatches (>1e-8): "<<cntM
                 <<", max diff: "<<maxM<<"\n";
        std::cout<<"Average mismatches (>1e-8): "<<cntA
                 <<", max diff: "<<maxA<<"\n";
        if(timing){
            std::cout<<"Speedup: "<<(cpuTime/(tProp+tAvg))<<"\n";
        }
        delete[] h_avc;
    }

    if(do_avg){
        std::cout<<"Row avgs (GPU):\n";
        for(int i=0;i<n;i++)
            std::cout<<"  row "<<i<<": "<<h_avr[i]<<"\n";
    }

    // cleanup
    delete[] h_init; delete[] h_next0;
    delete[] h_cpu;  delete[] h_next_cpu;
    delete[] h_res;  delete[] h_avr;
    cudaFree(d1);    cudaFree(d2);    cudaFree(dav);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 0;
}
