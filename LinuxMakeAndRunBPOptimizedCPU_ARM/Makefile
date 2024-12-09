OSLOWER := $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

SRC_DIR = ../src
SRC_DIR_BP = ${SRC_DIR}/beliefprop

# contains the directories to include...
INCLUDE_DIRS := -I. -I./ -I${SRC_DIR} -I${SRC_DIR_BP} -I$(COMMONDIR)/inc -I$(SHAREDDIR)/inc

# contains the library files needed for linking
# need -lstdc++fs for c++17 filesystem library to work
LIB_CPU := -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -lstdc++fs -lpthread

# set OPENMP_LIB if using OpenMP
OPENMP_LIB := -fopenmp

# define the path for the compiler
CC := g++

# setting for CPU vectorization
# NEON vectorization is used by default on ARM so doesn't need to be defined
CPU_VECTORIZATION_DEFINE =

# set to use only smaller data sets
USE_ONLY_SMALLER_DATA_SETTING = 
#-DSMALLER_SETS_ONLY

# set to use limited set of test parameters and fewer runs for faster results for testing
USE_LIMITED_TEST_PARAMS_FEWER_RUNS = -DLIMITED_TEST_PARAMS_FEWER_RUNS

# set compile flags and debug setting
DBG_SETTING    =
#-g
COMPILE_FLAGS = $(INCLUDE_DIRS) -DUNIX -DOPTIMIZED_CPU_RUN -DCOMPILING_FOR_ARM ${CPU_VECTORIZATION_DEFINE} ${USE_ONLY_SMALLER_DATA_SETTING} \
                ${USE_LIMITED_TEST_PARAMS_FEWER_RUNS} ${DBG_SETTING} $(OPENMP_LIB) -O3 -std=c++20 -Wall -march=native

# define directory for compiled objects and object files to generate during compilation
OBJ_DIR = obj
OBJECTS = ${OBJ_DIR}/EvaluateAcrossRuns.o ${OBJ_DIR}/InputSignature.o ${OBJ_DIR}/RunResultsSpeedups.o \
${OBJ_DIR}/EvaluateBpImpResults.o ${OBJ_DIR}/EvaluateImpResults.o ${OBJ_DIR}/RunImpMultTypesAccels.o \
${OBJ_DIR}/RunData.o ${OBJ_DIR}/BpParallelParams.o ${OBJ_DIR}/BpLevel.o ${OBJ_DIR}/ParallelParams.o \
${OBJ_DIR}/RunBpImpMultInputs.o ${OBJ_DIR}/RunImpMultInputs.o ${OBJ_DIR}/BpFileHandling.o ${OBJ_DIR}/DisparityMap.o \
${OBJ_DIR}/DisparityMapEvaluation.o ${OBJ_DIR}/stereo.o ${OBJ_DIR}/RunBpOnStereoSet.o ${OBJ_DIR}/BpImage.o \
${OBJ_DIR}/SmoothImage.o ${OBJ_DIR}/SmoothImageCPU.o ${OBJ_DIR}/ProcessBp.o ${OBJ_DIR}/DetailedTimings.o \
${OBJ_DIR}/RunBpOnStereoSetOptimizedCPU.o ${OBJ_DIR}/ProcessBpOptimizedCPU.o
DRIVER_BENCHMARK_CPU_OBJECT = ${OBJ_DIR}/DriverBpStereoCPU.o
DRIVER_BENCHMARK_CPU_OBJECT_CUSTOM = ${OBJ_DIR}/DriverBpStereoCPU_customRun.o

# executables for belief propagation CPU implementations
DRIVER_BENCHMARK_CPU = driverCPUBp
DRIVER_BENCHMARK_CPU_CUSTOM = driverCPUBpCustom

all: impDriverCPU impDriverCPUCustom

${OBJ_DIR}/%.o: ${SRC_DIR}/*/%.cpp
	$(CC) $< -c $(INCLUDE_DIRS) $(COMPILE_FLAGS) -o $@

${OBJ_DIR}/%.o: ${SRC_DIR_BP}/*/%.cpp
	$(CC) $< -c $(INCLUDE_DIRS) $(COMPILE_FLAGS) -o $@

impDriverCPU: ${DRIVER_BENCHMARK_CPU_OBJECT} ${OBJECTS} 
	$(CC) ${DRIVER_BENCHMARK_CPU_OBJECT} ${OBJECTS}  $(COMPILE_FLAGS) $(LIB_CPU) $(OPENMP_LIB) -o ${DRIVER_BENCHMARK_CPU}

impDriverCPUCustom: ${DRIVER_BENCHMARK_CPU_OBJECT_CUSTOM} ${OBJECTS} 
	$(CC) ${DRIVER_BENCHMARK_CPU_OBJECT_CUSTOM} ${OBJECTS}  $(COMPILE_FLAGS) $(LIB_CPU) $(OPENMP_LIB) -o ${DRIVER_BENCHMARK_CPU_CUSTOM}

make clean:
	rm *.o ${OBJ_DIR}/*.o driverCPUBp driverCPUBpCustom
