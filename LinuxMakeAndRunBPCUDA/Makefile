CUDA_DIR = /usr/local/cuda/
CUDA_SDK_ROOT := 

TOP_DIR = ..
BP_PATH = ${TOP_DIR}/BeliefProp

CU_FILE = driverCudaBp.cu
CU_OBJ = driverCudaBp.o
FILE_DEPENDENCIES = BpConstsAndParams/bpStereoCudaParameters.h BpConstsAndParams/bpStereoParameters.h BpConstsAndParams/RunSettings.h BpConstsAndParams/bpStructsAndEnums.h

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
BFLOAT_DATA_TYPE_SETTING = #-DUSE_BFLOAT16_FOR_HALF_PRECISION

# set to use only smaller data sets
USE_ONLY_SMALLER_DATA_SETTING = -DSMALLER_SETS_ONLY

# defines to add for CUDA processing
CUDA_RUN_DEFINES = -DOPTIMIZED_CUDA_RUN ${BFLOAT_DATA_TYPE_SETTING} ${USE_ONLY_SMALLER_DATA_SETTING}

# have include directory as flag
COMPILE_FLAGS += $(INCLUDE_DIRS) -DUNIX ${CUDA_RUN_DEFINES}

# include the optimization level
COMPILE_FLAGS += -O3 -std=c++20 -Wall
CUDA_COMPILE_FLAGS = $(INCLUDE_DIRS) -DUNIX ${CUDA_RUN_DEFINES} -O3 -std=c++20
ARCHITECTURE_COMPILE_FLAG = -O3 -march=native -std=c++20

# may need to adjust to allow support for target GPU (note that can't use half precision before compute capability 5.3)
# use native GPU by default (native flag requires CUDA 11.6 or later)
ARCHITECTURES_GENCODE = -arch=native

all: impDriverCUDA

impDriverCUDA: driverCudaBp.o BpParallelParams.o ParallelParams.o RunEvalBpImp.o RunBenchmarkImp.o BpFileHandling.o DisparityMap.o OutputEvaluationResults.o OutputEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o SmoothImageCUDA.o SmoothImage.o DetailedTimings.o ProcessBPOnTargetDevice.o ProcessCUDABP.o
	$(CC) driverCudaBp.o BpParallelParams.o ParallelParams.o RunEvalBpImp.o RunBenchmarkImp.o BpFileHandling.o DisparityMap.o OutputEvaluationResults.o OutputEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o SmoothImageCUDA.o SmoothImage.o DetailedTimings.o ProcessBPOnTargetDevice.o ProcessCUDABP.o $(LIB) -fopenmp $(ARCHITECTURE_COMPILE_FLAG) -o driverCudaBp -O

RunEvalBpImp.o:
	$(CC) ${BP_PATH}/BpRunImp/RunEvalBpImp.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

BpParallelParams.o:
	$(CC) ${BP_PATH}/BpRunImp/BpParallelParams.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ParallelParams.o:
	$(CC) ${TOP_DIR}/RunImp/ParallelParams.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBenchmarkImp.o:
	$(CC) ${TOP_DIR}/RunImp/RunBenchmarkImp.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

driverCudaBp.o: ${TOP_DIR}/MainDriverFiles/driverCudaBp.cpp ${TOP_DIR}/MainDriverFiles/RunEvalImp.h ${BP_PATH}/BpConstsAndParams/bpStereoCudaParameters.h ${BP_PATH}/BpConstsAndParams/bpStereoParameters.h ${BP_PATH}/BpConstsAndParams/bpStructsAndEnums.h RunBpStereoSetOnGPUWithCUDA.o stereo.o DisparityMap.o
	$(CC) ${TOP_DIR}/MainDriverFiles/driverCudaBp.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

BpFileHandling.o: ${BP_PATH}/BpFileProcessing/BpFileHandling.h ${BP_PATH}/BpFileProcessing/BpFileHandling.cpp ${BP_PATH}/BpFileProcessing/BpFileHandlingConsts.h
	$(CC) ${BP_PATH}/BpFileProcessing/BpFileHandling.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

DetailedTimings.o: ${TOP_DIR}/RuntimeTiming/DetailedTimings.cpp ${TOP_DIR}/RuntimeTiming/DetailedTimings.h
	$(CC) ${TOP_DIR}/RuntimeTiming/DetailedTimings.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessBPOnTargetDevice.o : ${BP_PATH}/BpRunProcessing/ProcessBPOnTargetDevice.cpp ${BP_PATH}/BpRunProcessing/ProcessBPOnTargetDevice.h DetailedTimings.o ${BP_PATH}/BpConstsAndParams/DetailedTimingBPConsts.h ${TOP_DIR}/RunSettingsEval/RunSettings.h ${BP_PATH}/BpConstsAndParams/bpStereoParameters.h ${BP_PATH}/BpConstsAndParams/bpStructsAndEnums.h
	$(CC) ${BP_PATH}/BpRunProcessing/ProcessBPOnTargetDevice.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessOptimizedCPUBP.o : ${BP_PATH}/BpOptimizeCPU/ProcessOptimizedCPUBP.cpp ${BP_PATH}/BpOptimizeCPU/ProcessOptimizedCPUBP.h ProcessBPOnTargetDevice.o ${BP_PATH}/BpOptimizeCPU/KernelBpStereoCPU.h ${BP_PATH}/BpOptimizeCPU/KernelBpStereoCPU.cpp
	$(CC) ${BP_PATH}/BpOptimizeCPU/ProcessOptimizedCPUBP.cpp -c -fopenmp $(ARCHITECTURE_COMPILE_FLAG) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoSet.o: ${BP_PATH}/BpRunProcessing/RunBpStereoSet.cpp ${BP_PATH}/BpRunProcessing/RunBpStereoSet.h ${BP_PATH}/BpConstsAndParams/bpStereoParameters.h ${TOP_DIR}/RunSettingsEval/RunSettings.h ${BP_PATH}/BpConstsAndParams/bpStructsAndEnums.h SmoothImage.o ProcessBPOnTargetDevice.o BpImage.o DetailedTimings.o ${BP_PATH}/BpConstsAndParams/DetailedTimingBPConsts.h
	$(CC) ${BP_PATH}/BpRunProcessing/RunBpStereoSet.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

BpImage.o: ${BP_PATH}/BpImageProcessing/BpImage.cpp ${BP_PATH}/BpImageProcessing/BpImage.h
	$(CC) ${BP_PATH}/BpImageProcessing/BpImage.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

OutputEvaluationParameters.o: ${BP_PATH}/BpOutputEvaluation/OutputEvaluationParameters.cpp ${BP_PATH}/BpOutputEvaluation/OutputEvaluationParameters.h
	$(CC) ${BP_PATH}/BpOutputEvaluation/OutputEvaluationParameters.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

OutputEvaluationResults.o: ${BP_PATH}/BpOutputEvaluation/OutputEvaluationResults.cpp ${BP_PATH}/BpOutputEvaluation/OutputEvaluationResults.h
	$(CC) ${BP_PATH}/BpOutputEvaluation/OutputEvaluationResults.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

DisparityMap.o: ${BP_PATH}/BpImageProcessing/DisparityMap.cpp ${BP_PATH}/BpImageProcessing/DisparityMap.h OutputEvaluationParameters.o OutputEvaluationResults.o
	$(CC) ${BP_PATH}/BpImageProcessing/DisparityMap.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

stereo.o: ${BP_PATH}/BpSingleThreadCPU/stereo.cpp ${BP_PATH}/BpSingleThreadCPU/stereo.h ${BP_PATH}/BpConstsAndParams/bpStereoCudaParameters.h ${BP_PATH}/BpConstsAndParams/bpStructsAndEnums.h ProcessBPOnTargetDevice.o RunBpStereoSet.o SmoothImage.o
	$(CC) ${BP_PATH}/BpSingleThreadCPU/stereo.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

SmoothImage.o: ${BP_PATH}/BpImageProcessing/SmoothImage.cpp ${BP_PATH}/BpImageProcessing/SmoothImage.h 
	$(CC) ${BP_PATH}/BpImageProcessing/SmoothImage.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoSetOnGPUWithCUDA.o: ${BP_PATH}/BpOptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.cpp ${BP_PATH}/BpOptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.h ProcessCUDABP.o SmoothImageCUDA.o RunBpStereoSet.o ProcessBPOnTargetDevice.o
	$(CC) ${BP_PATH}/BpOptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessCUDABP.o: ${BP_PATH}/BpOptimizeCUDA/ProcessCUDABP.cpp ${BP_PATH}/BpOptimizeCUDA/ProcessCUDABP.h
	$(NVCC) -x cu -c ${BP_PATH}/BpOptimizeCUDA/ProcessCUDABP.cpp $(ARCHITECTURES_GENCODE) -Xptxas -v -o ProcessCUDABP.o $(INCLUDE_DIRS) $(CUDA_COMPILE_FLAGS)

SmoothImageCUDA.o: ${BP_PATH}/BpOptimizeCUDA/SmoothImageCUDA.cpp ${BP_PATH}/BpOptimizeCUDA/SmoothImageCUDA.h ${BP_PATH}/BpOptimizeCUDA/kernelFilter.cu ${BP_PATH}/BpOptimizeCUDA/kernelFilterHeader.cuh
	$(NVCC) -x cu -c ${BP_PATH}/BpOptimizeCUDA/SmoothImageCUDA.cpp $(ARCHITECTURES_GENCODE) -o SmoothImageCUDA.o $(CUDA_COMPILE_FLAGS)

make clean:
	rm *.o driverCudaBp driverCPUBp
