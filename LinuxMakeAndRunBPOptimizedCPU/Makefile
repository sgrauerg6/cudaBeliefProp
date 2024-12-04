OSLOWER := $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

TOP_DIR = ..
BP_PATH = ${TOP_DIR}/BeliefProp

# contains the directories to include...
INCLUDE_DIRS := -I. -I./ -I./.. -I./../BeliefProp -I$(COMMONDIR)/inc -I$(SHAREDDIR)/inc

# contains the library files needed for linking
# need -lstdc++fs for c++17 filesystem library to work
LIB_CPU := -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -lstdc++fs -lpthread -fopenmp

# set OPENMP_LIB if using OpenMP
OPENMP_LIB := -fopenmp

# define the path for the compiler
CC := g++

# setting for CPU vectorization
# switch to -DAVX_256_VECTORIZATION if AVX 512 not supported on target CPU
CPU_VECTORIZATION_DEFINE = -DAVX_512_VECTORIZATION

# set to use only smaller data sets
USE_ONLY_SMALLER_DATA_SETTING =
#-DSMALLER_SETS_ONLY

# set to use limited set of test parameters and fewer runs for faster results for testing
USE_LIMITED_TEST_PARAMS_FEWER_RUNS = -DLIMITED_TEST_PARAMS_FEWER_RUNS

# set compile flags and debug setting
DBG_SETTING    =
#-g
COMPILE_FLAGS = $(INCLUDE_DIRS) -DUNIX -DOPTIMIZED_CPU_RUN ${CPU_VECTORIZATION_DEFINE} ${USE_ONLY_SMALLER_DATA_SETTING} \
                ${USE_LIMITED_TEST_PARAMS_FEWER_RUNS} ${DBG_SETTING} $(OPENMP_LIB) -O3 -std=c++20 -Wall -march=native

DRIVER_BENCHMARK_CPU = driverCPUBp
DRIVER_BENCHMARK_CPU_OBJECT = driverBpStereoCPU.o
DRIVER_BENCHMARK_CPU_CUSTOM = driverCPUBpCustom
DRIVER_BENCHMARK_CPU_OBJECT_CUSTOM = driverBpStereoCPU_customRun.o
OBJECTS = EvaluateAcrossRuns.o InputSignature.o RunResultsSpeedups.o EvaluateBPImpResults.o EvaluateImpResults.o RunEvalImpMultSettings.o RunData.o \
BpParallelParams.o BpLevel.o ParallelParams.o RunEvalBpImp.o RunBenchmarkImp.o BpFileHandling.o DisparityMap.o BpEvaluationResults.o \
BpEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o SmoothImage.o SmoothImageCPU.o ProcessBPOnTargetDevice.o DetailedTimings.o \
RunBpStereoOptimizedCPU.o ProcessOptimizedCPUBP.o

all: impDriverCPU impDriverCPUCustom

%.o: ${TOP_DIR}/*/%.cpp
	$(CC) $< -c $(INCLUDE_DIRS) $(COMPILE_FLAGS) -o $@

%.o: ${BP_PATH}/*/%.cpp
	$(CC) $< -c $(INCLUDE_DIRS) $(COMPILE_FLAGS) -o $@

impDriverCPU: ${DRIVER_BENCHMARK_CPU_OBJECT} ${OBJECTS} 
	$(CC) ${DRIVER_BENCHMARK_CPU_OBJECT} ${OBJECTS}  $(COMPILE_FLAGS) $(LIB_CPU) $(OPENMP_LIB) -o ${DRIVER_BENCHMARK_CPU} -O

impDriverCPUCustom: ${DRIVER_BENCHMARK_CPU_OBJECT_CUSTOM} ${OBJECTS} 
	$(CC) ${DRIVER_BENCHMARK_CPU_OBJECT_CUSTOM} ${OBJECTS}  $(COMPILE_FLAGS) $(LIB_CPU) $(OPENMP_LIB) -o ${DRIVER_BENCHMARK_CPU_CUSTOM} -O

make clean:
	rm *.o driverCPUBp driverCPUBpCustom
