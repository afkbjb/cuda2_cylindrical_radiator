CPU_TARGET = radiator_cpu
CPU_SRC    = main.cpp radiator_cpu.cpp

GPU_TARGET = radiator_gpu
GPU_SRC    = main.cu radiator_gpu.cu radiator_cpu.cpp

GPU_TASK3_TARGET = radiator_gpu_task3
GPU_TASK3_SRC    = main_task3.cu radiator_gpu.cu radiator_cpu.cpp

DP_TARGET = radiator_gpu_dp
DP_SRC    = main_task4_dp.cu radiator_gpu_dp.cu radiator_cpu_dp.cpp

CXX  = g++
NVCC = nvcc

CXXFLAGS  = -std=c++11 -O2 -Wall -Wextra
NVCCFLAGS = -std=c++11 -O2 -I. -Xcompiler "-Wall -Wextra"

all: $(CPU_TARGET) $(GPU_TARGET) $(GPU_TASK3_TARGET) $(DP_TARGET)

$(CPU_TARGET): $(CPU_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(GPU_TARGET): $(GPU_SRC)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(GPU_TASK3_TARGET): $(GPU_TASK3_SRC)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(DP_TARGET): $(DP_SRC)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

clean:
	rm -f $(CPU_TARGET) $(GPU_TARGET) $(GPU_TASK3_TARGET) $(DP_TARGET)

.PHONY: all clean
