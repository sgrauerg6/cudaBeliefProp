include ../common.mk

# define the path to CUDA on target compile device
CUDA_DIR = /usr/local/cuda

# define the path for the nvcc cuda compiler
NVCC := $(CUDA_DIR)/bin/nvcc

# define path to belief propatation source code
SRC_DIR_BP := ../${SRC_DIR}/beliefprop

# define the directories to include when compiling CUDA implementation
CUDA_IMP_INCLUDES := -I../${SRC_DIR} -I${SRC_DIR_BP} -I${CUDA_DIR}/include

# contains the library files needed for linking
# need -lstdc++fs for c++17 filesystem library to work
CUDA_IMP_LIBS := -L$(CUDA_DIR)/lib64 -lcudart -lstdc++fs

# use bfloat16 data type rather than default half data type for CUDA 16-bit float processing
# remove if compute capability of CUDA device is under 8.0 since bfloat16 not supported until
# compute capability 8.0
BFLOAT_DATA_TYPE_SETTING = -DUSE_BFLOAT16_FOR_HALF_PRECISION

# defines to add for CUDA processing
CUDA_IMP_RUN_DEFINES = -DOPTIMIZED_CUDA_RUN ${BFLOAT_DATA_TYPE_SETTING} ${COMMON_IMP_DEFINES}

# set compile flags for CUDA implementation for gcc and nvcc
CUDA_IMP_COMPILE_FLAGS_GCC = -DUNIX -O3 -std=c++20 -march=native -Wall ${COMPILE_FLAGS} ${CUDA_IMP_RUN_DEFINES} 
CUDA_IMP_COMPILE_FLAGS_NVCC = -DUNIX -O3 -std=c++20 -arch=native ${COMPILE_FLAGS} ${CUDA_IMP_RUN_DEFINES} 
CUDA_LINK_FLAGS = -O3 -march=native -std=c++20

# define directory for compiled objects and object files to generate during compilation
OBJ_DIR = obj
OBJECTS = ${OBJ_DIR}/EvaluateAcrossRuns.o ${OBJ_DIR}/InputSignature.o ${OBJ_DIR}/RunResultsSpeedups.o \
${OBJ_DIR}/EvaluateImpResultsBp.o ${OBJ_DIR}/EvaluateImpResults.o ${OBJ_DIR}/RunImpMultTypesAccels.o \
${OBJ_DIR}/RunData.o ${OBJ_DIR}/ParallelParamsBp.o ${OBJ_DIR}/BpLevel.o ${OBJ_DIR}/ParallelParams.o \
${OBJ_DIR}/RunImpMultInputsBp.o ${OBJ_DIR}/RunImpMultInputs.o ${OBJ_DIR}/BpFileHandling.o ${OBJ_DIR}/DisparityMap.o \
${OBJ_DIR}/DisparityMapEvaluation.o ${OBJ_DIR}/stereo.o ${OBJ_DIR}/RunBpOnStereoSet.o ${OBJ_DIR}/BpImage.o \
${OBJ_DIR}/SmoothImageCUDA.o ${OBJ_DIR}/SmoothImage.o ${OBJ_DIR}/DetailedTimings.o ${OBJ_DIR}/ProcessBp.o \
${OBJ_DIR}/ProcessBpCUDA.o
DRIVER_BENCHMARK_CUDA_OBJECT = ${OBJ_DIR}/DriverCudaBp.o
DRIVER_BENCHMARK_CUDA_OBJECT_CUSTOM = ${OBJ_DIR}/DriverBpStereoCUDA_customRun.o

# executables for belief propagation CUDA implementations
DRIVER_BENCHMARK_CUDA = driverCudaBp
DRIVER_BENCHMARK_CUDA_CUSTOM = driverCudaBpCustom

all: impDriverCUDA impDriverCUDACustom

${OBJ_DIR}/%.o: ../${SRC_DIR}/*/%.cpp
	$(CC) $< -c $(CUDA_IMP_INCLUDES) $(CUDA_IMP_COMPILE_FLAGS_GCC) -o $@

${OBJ_DIR}/%.o: ${SRC_DIR_BP}/*/%.cpp
	$(CC) $< -c $(CUDA_IMP_INCLUDES) $(CUDA_IMP_COMPILE_FLAGS_GCC) -o $@

# specify compile commands using NVCC compiler for objects with CUDA for processing on GPU
${OBJ_DIR}/ProcessBpCUDA.o:
	$(NVCC) -x cu -c ${SRC_DIR_BP}/BpOptimizeCUDA/ProcessBpCUDA.cpp -Xptxas -v -o $@ $(CUDA_IMP_INCLUDES) $(CUDA_IMP_COMPILE_FLAGS_NVCC)

${OBJ_DIR}/SmoothImageCUDA.o:
	$(NVCC) -x cu -c ${SRC_DIR_BP}/BpOptimizeCUDA/SmoothImageCUDA.cpp -o $@ $(CUDA_IMP_INCLUDES) $(CUDA_IMP_COMPILE_FLAGS_NVCC)

impDriverCUDA: ${OBJECTS} ${DRIVER_BENCHMARK_CUDA_OBJECT}
	$(CC) ${DRIVER_BENCHMARK_CUDA_OBJECT} ${OBJECTS} $(CUDA_IMP_LIBS) -fopenmp $(CUDA_LINK_FLAGS) -o ${DRIVER_BENCHMARK_CUDA} -O

impDriverCUDACustom: ${OBJECTS} ${DRIVER_BENCHMARK_CUDA_OBJECT_CUSTOM}
	$(CC) ${DRIVER_BENCHMARK_CUDA_OBJECT_CUSTOM} ${OBJECTS} $(CUDA_IMP_LIBS) -fopenmp $(CUDA_LINK_FLAGS) -o ${DRIVER_BENCHMARK_CUDA_CUSTOM} -O

make clean:
	rm *.o ${OBJ_DIR}/*.o driverCudaBp driverCPUBp driverCudaBpCustom
