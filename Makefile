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
INCLUDE_DIRS := -I. -I./ -I$(CUDA_DIR)/include -I$(COMMONDIR)/inc -I$(SHAREDDIR)/inc

# contains the library files needed for linking
# need -lstdc++fs for c++17 filesystem library to work
LIB := -L$(CUDA_DIR)/lib64 -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -lcudart -lstdc++fs
LIB_CPU := -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -lstdc++fs

# define the path for the nvcc cuda compiler
NVCC := $(CUDA_DIR)/bin/nvcc

# have include directory as flag
COMPILE_FLAGS += $(INCLUDE_DIRS) -DUNIX

# include the optimization level
COMPILE_FLAGS += -O3 -std=c++17 -Wall
CUDA_COMPILE_FLAGS = $(INCLUDE_DIRS) -DUNIX -O3
#ARCHITECTURE_COMPILE_FLAG = -march=native
ARCHITECTURE_COMPILE_FLAG = -O3 -march=native -std=c++17
#ARCHITECTURE_COMPILE_FLAG = -march=znver1

# need to adjust to allow support for compute capability under 6.0 (note that can't use half precision before compute capability 5.3)
ARCHITECTURES_GENCODE = -gencode arch=compute_75,code=sm_75 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_60,code=sm_60

INCDIR = -I.
DBG    = -g
OPT    = -O3
CPP    = g++
CFLAGS = $(DBG) $(OPT) $(INCDIR)
LINK   = -lm

.cpp.o:
	$(CPP) $(CFLAGS) -c $< -o $@

all: impDriverCUDA impDriverCPU

impDriverCUDA: driverCudaBp.o BpFileHandling.o RunAndEvaluateBpResults.o DisparityMap.o OutputEvaluationResults.o OutputEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o SmoothImageCUDA.o SmoothImage.o DetailedTimings.o ProcessBPOnTargetDevice.o ProcessCUDABP.o
	g++ driverCudaBp.o BpFileHandling.o RunAndEvaluateBpResults.o DisparityMap.o OutputEvaluationResults.o OutputEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o SmoothImageCUDA.o SmoothImage.o DetailedTimings.o ProcessBPOnTargetDevice.o ProcessCUDABP.o $(LIB) -fopenmp $(ARCHITECTURE_COMPILE_FLAG) -o driverCudaBp -O

impDriverCPU: driverBpStereoCPU.o RunAndEvaluateBpResults.o BpFileHandling.o DisparityMap.o OutputEvaluationResults.o OutputEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o SmoothImage.o SmoothImageCPU.o ProcessBPOnTargetDevice.o DetailedTimings.o RunBpStereoOptimizedCPU.o ProcessOptimizedCPUBP.o
	g++ driverBpStereoCPU.o BpFileHandling.o RunAndEvaluateBpResults.o DisparityMap.o OutputEvaluationResults.o OutputEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o SmoothImage.o SmoothImageCPU.o ProcessBPOnTargetDevice.o DetailedTimings.o RunBpStereoOptimizedCPU.o ProcessOptimizedCPUBP.o $(ARCHITECTURE_COMPILE_FLAG) $(LIB_CPU) -fopenmp $(ARCHITECTURE_COMPILE_FLAG) -o driverCPUBp -O

RunAndEvaluateBpResults.o: MainDriverFiles/RunAndEvaluateBpResults.h MainDriverFiles/RunAndEvaluateBpResults.cpp
	g++ MainDriverFiles/RunAndEvaluateBpResults.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)
	
BpFileHandling.o: FileProcessing/BpFileHandling.h FileProcessing/BpFileHandling.cpp FileProcessing/BpFileHandlingConsts.h
	g++ FileProcessing/BpFileHandling.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)
	
DetailedTimings.o: RuntimeTiming/DetailedTimings.cpp RuntimeTiming/DetailedTimings.h
	g++ RuntimeTiming/DetailedTimings.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

driverBpStereoCPU.o: MainDriverFiles/driverBpStereoCPU.cpp ParameterFiles/bpStereoParameters.h ParameterFiles/bpParametersFromPython.h ParameterFiles/bpStructsAndEnums.h BpImage.o stereo.o RunBpStereoOptimizedCPU.o DisparityMap.o RunAndEvaluateBpResults.o
	g++ MainDriverFiles/driverBpStereoCPU.cpp -c -fopenmp $(ARCHITECTURE_COMPILE_FLAG) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessBPOnTargetDevice.o : BpAndSmoothProcessing/ProcessBPOnTargetDevice.cpp BpAndSmoothProcessing/ProcessBPOnTargetDevice.h DetailedTimings.o RuntimeTiming/DetailedTimingBPConsts.h ParameterFiles/bpRunSettings.h ParameterFiles/bpStereoParameters.h ParameterFiles/bpStructsAndEnums.h
	g++ BpAndSmoothProcessing/ProcessBPOnTargetDevice.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessOptimizedCPUBP.o : OptimizeCPU/ProcessOptimizedCPUBP.cpp OptimizeCPU/ProcessOptimizedCPUBP.h ProcessBPOnTargetDevice.o OptimizeCPU/KernelBpStereoCPU.h OptimizeCPU/KernelBpStereoCPU.cpp
	g++ OptimizeCPU/ProcessOptimizedCPUBP.cpp -c -fopenmp $(ARCHITECTURE_COMPILE_FLAG) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoSet.o: BpAndSmoothProcessing/RunBpStereoSet.cpp BpAndSmoothProcessing/RunBpStereoSet.h ParameterFiles/bpStereoParameters.h ParameterFiles/bpRunSettings.h ParameterFiles/bpStructsAndEnums.h SmoothImage.o ProcessBPOnTargetDevice.o BpImage.o DetailedTimings.o RuntimeTiming/DetailedTimingBPConsts.h
	g++ BpAndSmoothProcessing/RunBpStereoSet.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

BpImage.o: ImageDataAndProcessing/BpImage.cpp ImageDataAndProcessing/BpImage.h
	g++ ImageDataAndProcessing/BpImage.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

OutputEvaluationParameters.o: OutputEvaluation/OutputEvaluationParameters.cpp OutputEvaluation/OutputEvaluationParameters.h
	g++ OutputEvaluation/OutputEvaluationParameters.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

OutputEvaluationResults.o: OutputEvaluation/OutputEvaluationResults.cpp OutputEvaluation/OutputEvaluationResults.h
	g++ OutputEvaluation/OutputEvaluationResults.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)
		
DisparityMap.o: OutputEvaluation/DisparityMap.cpp OutputEvaluation/DisparityMap.h OutputEvaluationParameters.o OutputEvaluationResults.o
	g++ OutputEvaluation/DisparityMap.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

stereo.o: SingleThreadCPU/stereo.cpp SingleThreadCPU/stereo.h ParameterFiles/bpStereoCudaParameters.h ParameterFiles/bpParametersFromPython.h ParameterFiles/bpStructsAndEnums.h ProcessBPOnTargetDevice.o RunBpStereoSet.o SmoothImage.o
	g++ SingleThreadCPU/stereo.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

SmoothImage.o: BpAndSmoothProcessing/SmoothImage.cpp BpAndSmoothProcessing/SmoothImage.h 
	g++ BpAndSmoothProcessing/SmoothImage.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

SmoothImageCPU.o: OptimizeCPU/SmoothImageCPU.cpp OptimizeCPU/SmoothImageCPU.h SmoothImage.o
	g++ OptimizeCPU/SmoothImageCPU.cpp -c -fopenmp $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoOptimizedCPU.o: OptimizeCPU/RunBpStereoOptimizedCPU.cpp OptimizeCPU/RunBpStereoOptimizedCPU.h SmoothImageCPU.o RunBpStereoSet.o ProcessBPOnTargetDevice.o ProcessOptimizedCPUBP.o
	g++ OptimizeCPU/RunBpStereoOptimizedCPU.cpp -c -fopenmp $(ARCHITECTURE_COMPILE_FLAG) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoSetOnGPUWithCUDA.o: OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.cpp OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.h ProcessCUDABP.o SmoothImageCUDA.o RunBpStereoSet.o ProcessBPOnTargetDevice.o
	g++ OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessCUDABP.o: OptimizeCUDA/ProcessCUDABP.cpp OptimizeCUDA/ProcessCUDABP.h
	$(NVCC) -x cu -c OptimizeCUDA/ProcessCUDABP.cpp $(ARCHITECTURES_GENCODE) -Xptxas -v -o ProcessCUDABP.o $(INCLUDE_DIRS) $(CUDA_COMPILE_FLAGS)

SmoothImageCUDA.o: OptimizeCUDA/SmoothImageCUDA.cpp OptimizeCUDA/SmoothImageCUDA.h OptimizeCUDA/kernalFilter.cu OptimizeCUDA/kernalFilterHeader.cuh
	$(NVCC) -x cu -c OptimizeCUDA/SmoothImageCUDA.cpp $(ARCHITECTURES_GENCODE) -o SmoothImageCUDA.o $(CUDA_COMPILE_FLAGS)

driverCudaBp.o: MainDriverFiles/driverCudaBp.cpp ParameterFiles/bpStereoCudaParameters.h ParameterFiles/bpStereoParameters.h ParameterFiles/bpParametersFromPython.h ParameterFiles/bpStructsAndEnums.h RunBpStereoSetOnGPUWithCUDA.o stereo.o DisparityMap.o RunAndEvaluateBpResults.o
	g++ MainDriverFiles/driverCudaBp.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)
	
make clean:
	rm *.o driverCudaBp driverCPUBp
