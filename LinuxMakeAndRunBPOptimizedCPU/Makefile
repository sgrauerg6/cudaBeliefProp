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

# define directory for compiled objects and object files to generate during compilation
OBJ_DIR = obj
OBJECTS = ${OBJ_DIR}/EvaluateAcrossRuns.o ${OBJ_DIR}/InputSignature.o ${OBJ_DIR}/RunResultsSpeedups.o \
${OBJ_DIR}/EvaluateBPImpResults.o ${OBJ_DIR}/EvaluateImpResults.o ${OBJ_DIR}/RunEvalImpMultSettings.o \
${OBJ_DIR}/RunData.o ${OBJ_DIR}/BpParallelParams.o ${OBJ_DIR}/BpLevel.o ${OBJ_DIR}/ParallelParams.o \
${OBJ_DIR}/RunEvalBpImp.o ${OBJ_DIR}/RunBenchmarkImp.o ${OBJ_DIR}/BpFileHandling.o ${OBJ_DIR}/DisparityMap.o \
${OBJ_DIR}/BpEvaluationResults.o ${OBJ_DIR}/BpEvaluationParameters.o ${OBJ_DIR}/stereo.o \
${OBJ_DIR}/RunBpStereoSet.o ${OBJ_DIR}/BpImage.o ${OBJ_DIR}/SmoothImage.o ${OBJ_DIR}/SmoothImageCPU.o \
${OBJ_DIR}/ProcessBPOnTargetDevice.o ${OBJ_DIR}/DetailedTimings.o ${OBJ_DIR}/RunBpStereoOptimizedCPU.o \
${OBJ_DIR}/ProcessOptimizedCPUBP.o
DRIVER_BENCHMARK_CPU_OBJECT = ${OBJ_DIR}/driverBpStereoCPU.o
DRIVER_BENCHMARK_CPU_OBJECT_CUSTOM = ${OBJ_DIR}/driverBpStereoCPU_customRun.o

# executables for belief propagation CPU implementations
DRIVER_BENCHMARK_CPU = driverCPUBp
DRIVER_BENCHMARK_CPU_CUSTOM = driverCPUBpCustom

all: impDriverCPU impDriverCPUCustom

${OBJ_DIR}/%.o: ${TOP_DIR}/*/%.cpp
	$(CC) $< -c $(INCLUDE_DIRS) $(COMPILE_FLAGS) -o $@

${OBJ_DIR}/%.o: ${BP_PATH}/*/%.cpp
	$(CC) $< -c $(INCLUDE_DIRS) $(COMPILE_FLAGS) -o $@

impDriverCPU: ${DRIVER_BENCHMARK_CPU_OBJECT} ${OBJECTS} 
	$(CC) ${DRIVER_BENCHMARK_CPU_OBJECT} ${OBJECTS}  $(COMPILE_FLAGS) $(LIB_CPU) $(OPENMP_LIB) -o ${DRIVER_BENCHMARK_CPU} -O

impDriverCPUCustom: ${DRIVER_BENCHMARK_CPU_OBJECT_CUSTOM} ${OBJECTS} 
	$(CC) ${DRIVER_BENCHMARK_CPU_OBJECT_CUSTOM} ${OBJECTS}  $(COMPILE_FLAGS) $(LIB_CPU) $(OPENMP_LIB) -o ${DRIVER_BENCHMARK_CPU_CUSTOM} -O

make clean:
	rm *.o ${OBJ_DIR}/*.o driverCPUBp driverCPUBpCustom
