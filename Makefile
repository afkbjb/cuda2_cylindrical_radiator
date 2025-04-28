CPU_DIR := cpu
GPU_DIR := gpu

CXX  := g++
NVCC := nvcc

CXXFLAGS  := -O2 -Wall -Wextra -std=c++11 -I$(CPU_DIR) -I$(GPU_DIR)
NVCCFLAGS := -std=c++11 -O2 -Xcompiler "-Wall -Wextra" -I$(CPU_DIR) -I$(GPU_DIR)

CPU_OUT        := radiator_cpu
GPU_OUT        := radiator_gpu
GPU_TASK3_OUT  := radiator_gpu_task3
GPU_DP_OUT     := radiator_gpu_dp

CPU_OBJ        := radiator_cpu.o
CPU_DP_OBJ     := radiator_cpu_dp.o

.PHONY: all clean

all: $(CPU_OUT) $(CPU_OBJ) $(CPU_DP_OBJ) \
     $(GPU_OUT) $(GPU_TASK3_OUT) $(GPU_DP_OUT)

$(CPU_OUT): \
    $(CPU_DIR)/main.cpp \
    $(CPU_DIR)/radiator_cpu.cpp \
    $(CPU_DIR)/radiator_cpu.h
	$(CXX) $(CXXFLAGS) \
	    $(CPU_DIR)/main.cpp \
	    $(CPU_DIR)/radiator_cpu.cpp \
	    -o $@

$(CPU_OBJ): \
    $(CPU_DIR)/radiator_cpu.cpp \
    $(CPU_DIR)/radiator_cpu.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CPU_DP_OBJ): \
    $(CPU_DIR)/radiator_cpu_dp.cpp \
    $(CPU_DIR)/radiator_cpu_dp.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(GPU_OUT): \
    $(GPU_DIR)/main.cu \
    $(GPU_DIR)/radiator_gpu.cu \
    $(GPU_DIR)/radiator_gpu.h \
    $(CPU_OBJ)
	$(NVCC) $(NVCCFLAGS) \
	    $(GPU_DIR)/main.cu \
	    $(GPU_DIR)/radiator_gpu.cu \
	    $(CPU_OBJ) \
	    -o $@

$(GPU_TASK3_OUT): \
    $(GPU_DIR)/main_task3.cu \
    $(GPU_DIR)/radiator_gpu.cu \
    $(GPU_DIR)/radiator_gpu.h \
    $(CPU_OBJ)
	$(NVCC) $(NVCCFLAGS) \
	    $(GPU_DIR)/main_task3.cu \
	    $(GPU_DIR)/radiator_gpu.cu \
	    $(CPU_OBJ) \
	    -o $@

$(GPU_DP_OUT): \
    $(GPU_DIR)/main_task4_dp.cu \
    $(GPU_DIR)/radiator_gpu_dp.cu \
    $(GPU_DIR)/radiator_gpu_dp.h \
    $(CPU_DP_OBJ)
	$(NVCC) $(NVCCFLAGS) \
	    $(GPU_DIR)/main_task4_dp.cu \
	    $(GPU_DIR)/radiator_gpu_dp.cu \
	    $(CPU_DP_OBJ) \
	    -o $@

clean:
	rm -f \
	    $(CPU_OUT) \
	    $(CPU_OBJ) \
	    $(CPU_DP_OBJ) \
	    $(GPU_OUT) \
	    $(GPU_TASK3_OUT) \
	    $(GPU_DP_OUT)
