#include <iostream>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include "radiator_cpu.h"     // CPU initialize/propagate/avg
#include "radiator_gpu.h"

static void check(cudaError_t e){
  if(e!=cudaSuccess){
    std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<"\n";
    std::exit(1);
  }
}

static const int BX = 16;
static const int BY = 16;

int main(int argc,char**argv){
  int n=32,m=32,p=10;
  bool do_avg=false, skip_cpu=false, timing=false;

  // 解析
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-n")&&i+1<argc) n=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-m")&&i+1<argc) m=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-p")&&i+1<argc) p=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-a")) do_avg=true;
    else if(!strcmp(argv[i],"-c")) skip_cpu=true;
    else if(!strcmp(argv[i],"-t")) timing=true;
    else { std::cerr<<"Unknown "<<argv[i]<<"\n"; return 1; }
  }

  // 检查整除
  if(m%BX||n%BY){
    std::cerr<<"Error: "<<BX<<"x"<<BY<<" block must divide "<<n<<"x"<<m<<"\n";
    return 1;
  }
  dim3 block(BX,BY), grid(m/BX,n/BY);

  float *h_mat=nullptr,*h_next=nullptr;
  double cpu_t=0;
  if(!skip_cpu){
    h_mat  = new float[n*m];
    h_next = new float[n*m];
    initialize(h_mat,h_next,n,m);
    auto t0=std::chrono::high_resolution_clock::now();
    for(int it=0;it<p;it++){
      propagate_step(h_mat,h_next,n,m);
      std::swap(h_mat,h_next);
    }
    auto t1=std::chrono::high_resolution_clock::now();
    cpu_t = std::chrono::duration<double, std::milli>(t1-t0).count();
    if(timing) std::cout<<"CPU compute: "<<cpu_t<<" ms\n";
  }

  // GPU malloc
  size_t Mbytes = n*m*sizeof(float), Abytes=n*sizeof(float);
  float *d1,*d2,*dav;
  cudaEvent_t e0,e1; check(cudaEventCreate(&e0)); check(cudaEventCreate(&e1));
  cudaEventRecord(e0);
  check(cudaMalloc(&d1,Mbytes));
  check(cudaMalloc(&d2,Mbytes));
  check(cudaMalloc(&dav,Abytes));
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float tAlloc; cudaEventElapsedTime(&tAlloc,e0,e1);
  if(timing) std::cout<<"GPU alloc: "<<tAlloc<<" ms\n";

  // 如果 skip_cpu，自行初始化 h_mat/h_next
  if(skip_cpu){
    h_mat  = new float[n*m];
    h_next = new float[n*m];
    initialize(h_mat,h_next,n,m);
  }

  // H2D
  cudaEventRecord(e0);
  check(cudaMemcpy(d1,h_mat,Mbytes,cudaMemcpyHostToDevice));
  check(cudaMemcpy(d2,h_next,Mbytes,cudaMemcpyHostToDevice));
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float tH2D; cudaEventElapsedTime(&tH2D,e0,e1);
  if(timing) std::cout<<"GPU H2D: "<<tH2D<<" ms\n";

  // GPU propagate p 步
  cudaEventRecord(e0);
  for(int it=0;it<p;it++){
    propagate_kernel<<<grid,block>>>(d1,d2,n,m);
    cudaDeviceSynchronize();
    std::swap(d1,d2);
  }
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float tProp; cudaEventElapsedTime(&tProp,e0,e1);
  if(timing) std::cout<<"GPU propagate: "<<tProp<<" ms\n";

  // GPU average
  cudaEventRecord(e0);
  average_kernel<<<n,1>>>(d1,dav,n,m);
  cudaDeviceSynchronize();
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float tAvg; cudaEventElapsedTime(&tAvg,e0,e1);
  if(timing) std::cout<<"GPU average: "<<tAvg<<" ms\n";

  // D2H
  float *h_res=new float[n*m], *h_avr=new float[n];
  cudaEventRecord(e0);
  check(cudaMemcpy(h_res,d1,Mbytes,cudaMemcpyDeviceToHost));
  check(cudaMemcpy(h_avr,dav,Abytes,cudaMemcpyDeviceToHost));
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float tDtoH; cudaEventElapsedTime(&tDtoH,e0,e1);
  if(timing) std::cout<<"GPU D2H: "<<tDtoH<<" ms\n";

  // 比对
  if(!skip_cpu){
    int cntM=0,cntA=0; float maxM=0,maxA=0;
    for(int i=0;i<n*m;i++){
      float d=fabs(h_mat[i]-h_res[i]);
      if(d>1e-4) cntM++;
      maxM=fmax(maxM,d);
    }
    float *h_avc=new float[n];
    compute_averages(h_mat,h_avc,n,m);
    for(int i=0;i<n;i++){
      float d=fabs(h_avc[i]-h_avr[i]);
      if(d>1e-4) cntA++;
      maxA=fmax(maxA,d);
    }
    std::cout<<"Matrix mismatches: "<<cntM<<" max="<<maxM<<"\n";
    std::cout<<"Avg mismatches:    "<<cntA<<" max="<<maxA<<"\n";
    if(timing){
      double gpuComp = tProp + tAvg;
      std::cout<<"Speedup: "<<(cpu_t/gpuComp)<<"\n";
    }
    delete[] h_avc;
  }

  // -a 时打印行平均
  if(do_avg){
    std::cout<<"Row avgs (GPU):\n";
    for(int i=0;i<n;i++)
      std::cout<<"  "<<i<<": "<<h_avr[i]<<"\n";
  }

  // 释放
  delete[] h_mat; delete[] h_next;
  delete[] h_res; delete[] h_avr;
  cudaFree(d1); cudaFree(d2); cudaFree(dav);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return 0;
}
