# Makefile - 支持 Task 1 (CPU版) 和 Task 2 (CUDA版)，适合所有文件在同一目录的情况

# ------ Task 1 (CPU版) 配置 ------
CPU_TARGET = radiator_cpu
CPU_SRC = main.cpp radiator_cpu.cpp
CPU_CXX = g++
CPU_CXXFLAGS = -std=c++11 -O2

# ------ Task 2 (CUDA版) 配置 ------
CUDA_TARGET = radiator_gpu
CUDA_SRC = main.cu radiator_gpu.cu radiator_cpu.cpp
CUDA_NVCC = nvcc
CUDA_NVCCFLAGS = -std=c++11 -O2

# ------ 默认目标 ------
all: cpu cuda

# ------ 编译 CPU版 ------
cpu: $(CPU_SRC)
	$(CPU_CXX) $(CPU_CXXFLAGS) $(CPU_SRC) -o $(CPU_TARGET)

# ------ 编译 CUDA版 ------
cuda: $(CUDA_SRC)
	$(CUDA_NVCC) $(CUDA_NVCCFLAGS) $(CUDA_SRC) -o $(CUDA_TARGET)

# ------ 清理 ------
clean:
	rm -f $(CPU_TARGET) $(CUDA_TARGET)
