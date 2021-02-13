CUDA_DIR = /usr/local/cuda/
CUDA_SDK_ROOT := 

CU_FILE = driverCudaBp.cu
CU_OBJ = driverCudaBp.o
FILE_DEPENDENCIES = ParameterFiles/bpStereoCudaParameters.h ParameterFiles/bpParametersFromPython.h ParameterFiles/bpStereoParameters.h ParameterFiles/bpRunSettings.h ParameterFiles/bpStructsAndEnums.h

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

# have include directory as flag
COMPILE_FLAGS += $(INCLUDE_DIRS) -DUNIX

# include the optimization level
COMPILE_FLAGS += -O3 -std=c++17 -Wall
CUDA_COMPILE_FLAGS = $(INCLUDE_DIRS) -DUNIX -O3 
#-std=c++17
ARCHITECTURE_COMPILE_FLAG = -O3 -march=native -std=c++17

# may need to adjust to allow support for target GPU (note that can't use half precision before compute capability 5.3)
# by default including "server GPU" compute capabilities (P100, V100, A100)
ARCHITECTURES_GENCODE = -gencode arch=compute_35,code=sm_35
#-gencode arch=compute_70,code=sm_70 -gencode arch=compute_60,code=sm_60
#ARCHITECTURES_GENCODE = -gencode arch=compute_86,code=sm_86 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_60,code=sm_60
#ARCHITECTURES_GENCODE = -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_60,code=sm_60 #-gencode arch=compute_35,code=sm_35

all: impDriverCUDA

impDriverCUDA: driverCudaBp.o BpFileHandling.o RunAndEvaluateBpResults.o DisparityMap.o OutputEvaluationResults.o OutputEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o SmoothImageCUDA.o SmoothImage.o DetailedTimings.o ProcessBPOnTargetDevice.o ProcessCUDABP.o
	$(CC) driverCudaBp.o BpFileHandling.o RunAndEvaluateBpResults.o DisparityMap.o OutputEvaluationResults.o OutputEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o SmoothImageCUDA.o SmoothImage.o DetailedTimings.o ProcessBPOnTargetDevice.o ProcessCUDABP.o $(LIB) -fopenmp $(ARCHITECTURE_COMPILE_FLAG) -o driverCudaBp -O

driverCudaBp.o: ../MainDriverFiles/driverCudaBp.cpp ../ParameterFiles/bpStereoCudaParameters.h ../ParameterFiles/bpStereoParameters.h ../ParameterFiles/bpParametersFromPython.h ../ParameterFiles/bpStructsAndEnums.h RunBpStereoSetOnGPUWithCUDA.o stereo.o DisparityMap.o RunAndEvaluateBpResults.o
	$(CC) ../MainDriverFiles/driverCudaBp.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunAndEvaluateBpResults.o: ../MainDriverFiles/RunAndEvaluateBpResults.h ../MainDriverFiles/RunAndEvaluateBpResults.cpp
	$(CC) ../MainDriverFiles/RunAndEvaluateBpResults.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

BpFileHandling.o: ../FileProcessing/BpFileHandling.h ../FileProcessing/BpFileHandling.cpp ../FileProcessing/BpFileHandlingConsts.h
	$(CC) ../FileProcessing/BpFileHandling.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

DetailedTimings.o: ../RuntimeTiming/DetailedTimings.cpp ../RuntimeTiming/DetailedTimings.h
	$(CC) ../RuntimeTiming/DetailedTimings.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessBPOnTargetDevice.o : ../BpAndSmoothProcessing/ProcessBPOnTargetDevice.cpp ../BpAndSmoothProcessing/ProcessBPOnTargetDevice.h DetailedTimings.o ../RuntimeTiming/DetailedTimingBPConsts.h ../ParameterFiles/bpRunSettings.h ../ParameterFiles/bpStereoParameters.h ../ParameterFiles/bpStructsAndEnums.h
	$(CC) ../BpAndSmoothProcessing/ProcessBPOnTargetDevice.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessOptimizedCPUBP.o : ../OptimizeCPU/ProcessOptimizedCPUBP.cpp ../OptimizeCPU/ProcessOptimizedCPUBP.h ProcessBPOnTargetDevice.o ../OptimizeCPU/KernelBpStereoCPU.h ../OptimizeCPU/KernelBpStereoCPU.cpp
	$(CC) ../OptimizeCPU/ProcessOptimizedCPUBP.cpp -c -fopenmp $(ARCHITECTURE_COMPILE_FLAG) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoSet.o: ../BpAndSmoothProcessing/RunBpStereoSet.cpp ../BpAndSmoothProcessing/RunBpStereoSet.h ../ParameterFiles/bpStereoParameters.h ../ParameterFiles/bpRunSettings.h ../ParameterFiles/bpStructsAndEnums.h SmoothImage.o ProcessBPOnTargetDevice.o BpImage.o DetailedTimings.o ../RuntimeTiming/DetailedTimingBPConsts.h
	$(CC) ../BpAndSmoothProcessing/RunBpStereoSet.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

BpImage.o: ../ImageDataAndProcessing/BpImage.cpp ../ImageDataAndProcessing/BpImage.h
	$(CC) ../ImageDataAndProcessing/BpImage.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

OutputEvaluationParameters.o: ../OutputEvaluation/OutputEvaluationParameters.cpp ../OutputEvaluation/OutputEvaluationParameters.h
	$(CC) ../OutputEvaluation/OutputEvaluationParameters.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

OutputEvaluationResults.o: ../OutputEvaluation/OutputEvaluationResults.cpp ../OutputEvaluation/OutputEvaluationResults.h
	$(CC) ../OutputEvaluation/OutputEvaluationResults.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

DisparityMap.o: ../OutputEvaluation/DisparityMap.cpp ../OutputEvaluation/DisparityMap.h OutputEvaluationParameters.o OutputEvaluationResults.o
	$(CC) ../OutputEvaluation/DisparityMap.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

stereo.o: ../SingleThreadCPU/stereo.cpp ../SingleThreadCPU/stereo.h ../ParameterFiles/bpStereoCudaParameters.h ../ParameterFiles/bpParametersFromPython.h ../ParameterFiles/bpStructsAndEnums.h ProcessBPOnTargetDevice.o RunBpStereoSet.o SmoothImage.o
	$(CC) ../SingleThreadCPU/stereo.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

SmoothImage.o: ../BpAndSmoothProcessing/SmoothImage.cpp ../BpAndSmoothProcessing/SmoothImage.h 
	$(CC) ../BpAndSmoothProcessing/SmoothImage.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoSetOnGPUWithCUDA.o: ../OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.cpp ../OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.h ProcessCUDABP.o SmoothImageCUDA.o RunBpStereoSet.o ProcessBPOnTargetDevice.o
	$(CC) ../OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessCUDABP.o: ../OptimizeCUDA/ProcessCUDABP.cpp ../OptimizeCUDA/ProcessCUDABP.h
	$(NVCC) -x cu -c ../OptimizeCUDA/ProcessCUDABP.cpp $(ARCHITECTURES_GENCODE) -Xptxas -v -o ProcessCUDABP.o $(INCLUDE_DIRS) $(CUDA_COMPILE_FLAGS)

SmoothImageCUDA.o: ../OptimizeCUDA/SmoothImageCUDA.cpp ../OptimizeCUDA/SmoothImageCUDA.h ../OptimizeCUDA/kernalFilter.cu ../OptimizeCUDA/kernalFilterHeader.cuh
	$(NVCC) -x cu -c ../OptimizeCUDA/SmoothImageCUDA.cpp $(ARCHITECTURES_GENCODE) -o SmoothImageCUDA.o $(CUDA_COMPILE_FLAGS)

make clean:
	rm *.o driverCudaBp driverCPUBp
