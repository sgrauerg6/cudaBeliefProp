CUDA_DIR = /usr/local/cuda/
CUDA_SDK_ROOT := 

TOP_DIR = ..
BP_PATH = ${TOP_DIR}/BeliefProp

L_FLAGS = -L $(CUDA_DIR)/bin -L $(CUDA_DIR)/lib64 -lcudart
INCLUDES_CUDA = -I$(CUDA_DIR)/include

LIBDIR     := $(CUDA_SDK_ROOT)/C/lib
COMMONDIR  := $(CUDA_SDK_ROOT)/C/common
SHAREDDIR  := $(CUDA_SDK_ROOT)/shared
OSLOWER := $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# contains the directories to include...
INCLUDE_DIRS := -I. -I./ -I./.. -I./../BeliefProp -I$(CUDA_DIR)/include -I$(COMMONDIR)/inc -I$(SHAREDDIR)/inc

# contains the library files needed for linking
# need -lstdc++fs for c++17 filesystem library to work
LIB := -L$(CUDA_DIR)/lib64 -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -lcudart -lstdc++fs

# define the path for the nvcc cuda compiler
NVCC := $(CUDA_DIR)/bin/nvcc
CC    = g++

# uncomment to use bfloat16 data type rather than default half data type for CUDA 16-bit float processing
BFLOAT_DATA_TYPE_SETTING =
#-DUSE_BFLOAT16_FOR_HALF_PRECISION

# set to use only smaller data sets
USE_ONLY_SMALLER_DATA_SETTING =
#-DSMALLER_SETS_ONLY

# set to use limited set of test parameters and fewer runs for faster results for testing
USE_LIMITED_TEST_PARAMS_FEWER_RUNS = -DLIMITED_TEST_PARAMS_FEWER_RUNS

# defines to add for CUDA processing
CUDA_RUN_DEFINES = -DOPTIMIZED_CUDA_RUN ${BFLOAT_DATA_TYPE_SETTING} ${USE_ONLY_SMALLER_DATA_SETTING} ${USE_LIMITED_TEST_PARAMS_FEWER_RUNS}

# have include directory as flag
COMPILE_FLAGS += $(INCLUDE_DIRS) -DUNIX ${CUDA_RUN_DEFINES}

# include the optimization level
COMPILE_FLAGS += -O3 -std=c++20 -Wall
CUDA_COMPILE_FLAGS = $(INCLUDE_DIRS) -DUNIX ${CUDA_RUN_DEFINES} -O3 -std=c++20
ARCHITECTURE_COMPILE_FLAG = -O3 -march=native -std=c++20

# may need to adjust to allow support for target GPU (note that can't use half precision before compute capability 5.3)
# use native GPU by default (native flag requires CUDA 11.6 or later)
ARCHITECTURES_GENCODE = -arch=native

DRIVER_BENCHMARK_CUDA = driverCudaBp
DRIVER_BENCHMARK_CUDA_OBJECT = driverCudaBp.o
DRIVER_BENCHMARK_CUDA_CUSTOM = driverCudaBpCustom
DRIVER_BENCHMARK_CUDA_OBJECT_CUSTOM = driverBpStereoCUDA_customRun.o
OBJECTS = EvaluateAcrossRuns.o InputSignature.o RunResultsSpeedups.o EvaluateBPImpResults.o EvaluateImpResults.o \
RunEvalImpMultSettings.o RunData.o BpParallelParams.o BpLevel.o ParallelParams.o RunEvalBpImp.o RunBenchmarkImp.o \
BpFileHandling.o DisparityMap.o BpEvaluationResults.o BpEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o \
SmoothImageCUDA.o SmoothImage.o DetailedTimings.o ProcessBPOnTargetDevice.o ProcessCUDABP.o

all: impDriverCUDA impDriverCUDACustom

%.o: ${TOP_DIR}/*/%.cpp
	$(CC) $< -c $(INCLUDE_DIRS) $(COMPILE_FLAGS) -o $@

%.o: ${BP_PATH}/*/%.cpp
	$(CC) $< -c $(INCLUDE_DIRS) $(COMPILE_FLAGS) -o $@

ProcessCUDABP.o:
	$(NVCC) -x cu -c ${BP_PATH}/BpOptimizeCUDA/ProcessCUDABP.cpp $(ARCHITECTURES_GENCODE) -Xptxas -v -o ProcessCUDABP.o $(INCLUDE_DIRS) $(CUDA_COMPILE_FLAGS)

SmoothImageCUDA.o:
	$(NVCC) -x cu -c ${BP_PATH}/BpOptimizeCUDA/SmoothImageCUDA.cpp $(ARCHITECTURES_GENCODE) -o SmoothImageCUDA.o $(CUDA_COMPILE_FLAGS)

impDriverCUDA: ${OBJECTS} ${DRIVER_BENCHMARK_CUDA_OBJECT}
	$(CC) ${DRIVER_BENCHMARK_CUDA_OBJECT} ${OBJECTS} $(LIB) -fopenmp $(ARCHITECTURE_COMPILE_FLAG) -o ${DRIVER_BENCHMARK_CUDA} -O

impDriverCUDACustom: ${OBJECTS} ${DRIVER_BENCHMARK_CUDA_OBJECT_CUSTOM}
	$(CC) ${DRIVER_BENCHMARK_CUDA_OBJECT_CUSTOM} ${OBJECTS} $(LIB) -fopenmp $(ARCHITECTURE_COMPILE_FLAG) -o ${DRIVER_BENCHMARK_CUDA_CUSTOM} -O

make clean:
	rm *.o driverCudaBp driverCPUBp driverCudaBpCustom
