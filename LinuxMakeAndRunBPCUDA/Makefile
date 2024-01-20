CUDA_DIR = /usr/local/cuda/
CUDA_SDK_ROOT := 

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
INCLUDE_DIRS := -I. -I./ -I./.. -I$(CUDA_DIR)/include -I$(COMMONDIR)/inc -I$(SHAREDDIR)/inc

# contains the library files needed for linking
# need -lstdc++fs for c++17 filesystem library to work
LIB := -L$(CUDA_DIR)/lib64 -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -lcudart -lstdc++fs

# define the path for the nvcc cuda compiler
NVCC := $(CUDA_DIR)/bin/nvcc
CC    = g++

# uncomment to use bfloat16 data type rather than default half data type for CUDA 16-bit float processing
BFLOAT_DATA_TYPE_SETTING = #-DUSE_BFLOAT16_FOR_HALF_PRECISION

# defines to add for CUDA processing
CUDA_RUN_DEFINES = -DOPTIMIZED_CUDA_RUN ${BFLOAT_DATA_TYPE_SETTING}

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

impDriverCUDA: driverCudaBp.o BpFileHandling.o DisparityMap.o OutputEvaluationResults.o OutputEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o SmoothImageCUDA.o SmoothImage.o DetailedTimings.o ProcessBPOnTargetDevice.o ProcessCUDABP.o
	$(CC) driverCudaBp.o BpFileHandling.o DisparityMap.o OutputEvaluationResults.o OutputEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o SmoothImageCUDA.o SmoothImage.o DetailedTimings.o ProcessBPOnTargetDevice.o ProcessCUDABP.o $(LIB) -fopenmp $(ARCHITECTURE_COMPILE_FLAG) -o driverCudaBp -O

driverCudaBp.o: ../MainDriverFiles/driverCudaBp.cpp ../MainDriverFiles/RunAndEvaluateBpResults.h ../BpConstsAndParams/bpStereoCudaParameters.h ../BpConstsAndParams/bpStereoParameters.h ../BpConstsAndParams/bpStructsAndEnums.h RunBpStereoSetOnGPUWithCUDA.o stereo.o DisparityMap.o
	$(CC) ../MainDriverFiles/driverCudaBp.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

BpFileHandling.o: ../BpFileProcessing/BpFileHandling.h ../BpFileProcessing/BpFileHandling.cpp ../BpFileProcessing/BpFileHandlingConsts.h
	$(CC) ../BpFileProcessing/BpFileHandling.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

DetailedTimings.o: ../RuntimeTiming/DetailedTimings.cpp ../RuntimeTiming/DetailedTimings.h
	$(CC) ../RuntimeTiming/DetailedTimings.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessBPOnTargetDevice.o : ../BpAndSmoothProcessing/ProcessBPOnTargetDevice.cpp ../BpAndSmoothProcessing/ProcessBPOnTargetDevice.h DetailedTimings.o ../BpConstsAndParams/DetailedTimingBPConsts.h ../RunSettingsEval/RunSettings.h ../BpConstsAndParams/bpStereoParameters.h ../BpConstsAndParams/bpStructsAndEnums.h
	$(CC) ../BpAndSmoothProcessing/ProcessBPOnTargetDevice.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessOptimizedCPUBP.o : ../OptimizeCPU/ProcessOptimizedCPUBP.cpp ../OptimizeCPU/ProcessOptimizedCPUBP.h ProcessBPOnTargetDevice.o ../OptimizeCPU/KernelBpStereoCPU.h ../OptimizeCPU/KernelBpStereoCPU.cpp
	$(CC) ../OptimizeCPU/ProcessOptimizedCPUBP.cpp -c -fopenmp $(ARCHITECTURE_COMPILE_FLAG) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoSet.o: ../BpAndSmoothProcessing/RunBpStereoSet.cpp ../BpAndSmoothProcessing/RunBpStereoSet.h ../BpConstsAndParams/bpStereoParameters.h ../RunSettingsEval/RunSettings.h ../BpConstsAndParams/bpStructsAndEnums.h SmoothImage.o ProcessBPOnTargetDevice.o BpImage.o DetailedTimings.o ../BpConstsAndParams/DetailedTimingBPConsts.h
	$(CC) ../BpAndSmoothProcessing/RunBpStereoSet.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

BpImage.o: ../BpImageDataAndProcessing/BpImage.cpp ../BpImageDataAndProcessing/BpImage.h
	$(CC) ../BpImageDataAndProcessing/BpImage.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

OutputEvaluationParameters.o: ../BpOutputEvaluation/OutputEvaluationParameters.cpp ../BpOutputEvaluation/OutputEvaluationParameters.h
	$(CC) ../BpOutputEvaluation/OutputEvaluationParameters.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

OutputEvaluationResults.o: ../BpOutputEvaluation/OutputEvaluationResults.cpp ../BpOutputEvaluation/OutputEvaluationResults.h
	$(CC) ../BpOutputEvaluation/OutputEvaluationResults.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

DisparityMap.o: ../BpOutputEvaluation/DisparityMap.cpp ../BpOutputEvaluation/DisparityMap.h OutputEvaluationParameters.o OutputEvaluationResults.o
	$(CC) ../BpOutputEvaluation/DisparityMap.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

stereo.o: ../SingleThreadCPU/stereo.cpp ../SingleThreadCPU/stereo.h ../BpConstsAndParams/bpStereoCudaParameters.h ../BpConstsAndParams/bpStructsAndEnums.h ProcessBPOnTargetDevice.o RunBpStereoSet.o SmoothImage.o
	$(CC) ../SingleThreadCPU/stereo.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

SmoothImage.o: ../BpAndSmoothProcessing/SmoothImage.cpp ../BpAndSmoothProcessing/SmoothImage.h 
	$(CC) ../BpAndSmoothProcessing/SmoothImage.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoSetOnGPUWithCUDA.o: ../OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.cpp ../OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.h ProcessCUDABP.o SmoothImageCUDA.o RunBpStereoSet.o ProcessBPOnTargetDevice.o
	$(CC) ../OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessCUDABP.o: ../OptimizeCUDA/ProcessCUDABP.cpp ../OptimizeCUDA/ProcessCUDABP.h
	$(NVCC) -x cu -c ../OptimizeCUDA/ProcessCUDABP.cpp $(ARCHITECTURES_GENCODE) -Xptxas -v -o ProcessCUDABP.o $(INCLUDE_DIRS) $(CUDA_COMPILE_FLAGS)

SmoothImageCUDA.o: ../OptimizeCUDA/SmoothImageCUDA.cpp ../OptimizeCUDA/SmoothImageCUDA.h ../OptimizeCUDA/kernelFilter.cu ../OptimizeCUDA/kernelFilterHeader.cuh
	$(NVCC) -x cu -c ../OptimizeCUDA/SmoothImageCUDA.cpp $(ARCHITECTURES_GENCODE) -o SmoothImageCUDA.o $(CUDA_COMPILE_FLAGS)

make clean:
	rm *.o driverCudaBp driverCPUBp
