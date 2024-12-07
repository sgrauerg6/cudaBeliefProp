CUDA_DIR = /usr/local/cuda/
CUDA_SDK_ROOT := 

L_FLAGS = -L $(CUDA_DIR)/bin -L $(CUDA_DIR)/lib64 -lcudart
INCLUDES_CUDA = -I$(CUDA_DIR)/include

LIBDIR     := $(CUDA_SDK_ROOT)/C/lib
COMMONDIR  := $(CUDA_SDK_ROOT)/C/common
SHAREDDIR  := $(CUDA_SDK_ROOT)/shared
OSLOWER := $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

SRC_DIR = ../src
SRC_DIR_BP = ${SRC_DIR}/beliefprop

# contains the directories to include...
INCLUDE_DIRS := -I. -I./ -I${SRC_DIR} -I${SRC_DIR_BP} -I$(CUDA_DIR)/include -I$(COMMONDIR)/inc -I$(SHAREDDIR)/inc

# contains the library files needed for linking
# need -lstdc++fs for c++17 filesystem library to work
LIB := -L$(CUDA_DIR)/lib64 -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -lcudart -lstdc++fs

# define the path for the nvcc cuda compiler
NVCC := $(CUDA_DIR)/bin/nvcc
CC    = g++

# use bfloat16 data type rather than default half data type for CUDA 16-bit float processing
# remove if compute capability of CUDA device is under 8.0 since bfloat16 not supported until
# compute capability 8.0
BFLOAT_DATA_TYPE_SETTING = -DUSE_BFLOAT16_FOR_HALF_PRECISION

# set to use only smaller data sets
USE_ONLY_SMALLER_DATA_SETTING =
#-DSMALLER_SETS_ONLY

# set to use limited set of test parameters and fewer runs for faster results for testing
USE_LIMITED_TEST_PARAMS_FEWER_RUNS = -DLIMITED_TEST_PARAMS_FEWER_RUNS

# defines to add for CUDA processing
CUDA_RUN_DEFINES = -DOPTIMIZED_CUDA_RUN ${BFLOAT_DATA_TYPE_SETTING} ${USE_ONLY_SMALLER_DATA_SETTING} ${USE_LIMITED_TEST_PARAMS_FEWER_RUNS}

# have include directory as flag
COMPILE_FLAGS = $(INCLUDE_DIRS) -DUNIX ${CUDA_RUN_DEFINES}

# add optimization level and C++ version to compile flags
COMPILE_FLAGS += -O3 -std=c++20 -Wall

# set compile flags for CUDA compiling using NVCC
CUDA_COMPILE_FLAGS = $(INCLUDE_DIRS) -DUNIX ${CUDA_RUN_DEFINES} -O3 -std=c++20
ARCHITECTURE_COMPILE_FLAG = -O3 -march=native -std=c++20

# may need to adjust to allow support for target GPU
# use native GPU by default (native flag requires CUDA 11.6 or later)
ARCHITECTURES_GENCODE = -arch=native

# define directory for compiled objects and object files to generate during compilation
OBJ_DIR = obj
OBJECTS = ${OBJ_DIR}/EvaluateAcrossRuns.o ${OBJ_DIR}/InputSignature.o ${OBJ_DIR}/RunResultsSpeedups.o \
${OBJ_DIR}/EvaluateBpImpResults.o ${OBJ_DIR}/EvaluateImpResults.o ${OBJ_DIR}/RunImpMultTypesAccels.o \
${OBJ_DIR}/RunData.o ${OBJ_DIR}/BpParallelParams.o ${OBJ_DIR}/BpLevel.o ${OBJ_DIR}/ParallelParams.o \
${OBJ_DIR}/RunBpImpMultInputs.o ${OBJ_DIR}/RunImpMultInputs.o ${OBJ_DIR}/BpFileHandling.o ${OBJ_DIR}/DisparityMap.o \
${OBJ_DIR}/DisparityMapEvaluation.o ${OBJ_DIR}/DisparityMapEvaluationParams.o ${OBJ_DIR}/stereo.o \
${OBJ_DIR}/RunBpOnStereoSet.o ${OBJ_DIR}/BpImage.o ${OBJ_DIR}/SmoothImageCUDA.o ${OBJ_DIR}/SmoothImage.o \
${OBJ_DIR}/DetailedTimings.o ${OBJ_DIR}/ProcessBp.o ${OBJ_DIR}/ProcessBpCUDA.o
DRIVER_BENCHMARK_CUDA_OBJECT = ${OBJ_DIR}/driverCudaBp.o
DRIVER_BENCHMARK_CUDA_OBJECT_CUSTOM = ${OBJ_DIR}/driverBpStereoCUDA_customRun.o

# executables for belief propagation CUDA implementations
DRIVER_BENCHMARK_CUDA = driverCudaBp
DRIVER_BENCHMARK_CUDA_CUSTOM = driverCudaBpCustom

all: impDriverCUDA impDriverCUDACustom

${OBJ_DIR}/%.o: ${SRC_DIR}/*/%.cpp
	$(CC) $< -c $(INCLUDE_DIRS) $(COMPILE_FLAGS) -o $@

${OBJ_DIR}/%.o: ${SRC_DIR_BP}/*/%.cpp
	$(CC) $< -c $(INCLUDE_DIRS) $(COMPILE_FLAGS) -o $@

# specify compile commands using NVCC compiler for objects with CUDA for processing on GPU
${OBJ_DIR}/ProcessBpCUDA.o:
	$(NVCC) -x cu -c ${SRC_DIR_BP}/BpOptimizeCUDA/ProcessBpCUDA.cpp $(ARCHITECTURES_GENCODE) -Xptxas -v -o $@ $(INCLUDE_DIRS) $(CUDA_COMPILE_FLAGS)

${OBJ_DIR}/SmoothImageCUDA.o:
	$(NVCC) -x cu -c ${SRC_DIR_BP}/BpOptimizeCUDA/SmoothImageCUDA.cpp $(ARCHITECTURES_GENCODE) -o $@ $(CUDA_COMPILE_FLAGS)

impDriverCUDA: ${OBJECTS} ${DRIVER_BENCHMARK_CUDA_OBJECT}
	$(CC) ${DRIVER_BENCHMARK_CUDA_OBJECT} ${OBJECTS} $(LIB) -fopenmp $(ARCHITECTURE_COMPILE_FLAG) -o ${DRIVER_BENCHMARK_CUDA} -O

impDriverCUDACustom: ${OBJECTS} ${DRIVER_BENCHMARK_CUDA_OBJECT_CUSTOM}
	$(CC) ${DRIVER_BENCHMARK_CUDA_OBJECT_CUSTOM} ${OBJECTS} $(LIB) -fopenmp $(ARCHITECTURE_COMPILE_FLAG) -o ${DRIVER_BENCHMARK_CUDA_CUSTOM} -O

make clean:
	rm *.o ${OBJ_DIR}/*.o driverCudaBp driverCPUBp driverCudaBpCustom
