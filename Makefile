CUDA_DIR = /usr/local/cuda/
CUDA_SDK_ROOT := 

CU_FILE = driverCudaBp.cu
CU_OBJ = driverCudaBp.o
FILE_DEPENDENCIES = bpStereoCudaParameters.h bpParametersFromPython.h bpStereoParameters.h

L_FLAGS = -L $(CUDA_DIR)/bin -L $(CUDA_DIR)/lib64 -lcudart
INCLUDES_CUDA = -I$(CUDA_DIR)/include

LIBDIR     := $(CUDA_SDK_ROOT)/C/lib
COMMONDIR  := $(CUDA_SDK_ROOT)/C/common
SHAREDDIR  := $(CUDA_SDK_ROOT)/shared
OSLOWER := $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# contains the directories to include...
INCLUDE_DIRS := -I. -I./ -I$(CUDA_DIR)/include -I$(COMMONDIR)/inc -I$(SHAREDDIR)/inc

# contains the library files needed for linking
LIB := -L$(CUDA_DIR)/lib64 -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -lcudart
LIB_CPU := -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER)

# define the path for the nvcc cuda compiler
NVCC := $(CUDA_DIR)/bin/nvcc

# have include directory as flag
COMPILE_FLAGS += $(INCLUDE_DIRS) -DUNIX 

# include the optimization level
COMPILE_FLAGS += -O2 -std=c++11
ARCHITECTURE_COMPILE_FLAG = -march=native
#ARCHITECTURE_COMPILE_FLAG = -march=skylake-avx512
#ARCHITECTURE_COMPILE_FLAG = -march=znver1

# need to adjust to allow support for compute capability under 6.0 (note that can't use half precision before compute capability 5.3)
ARCHITECTURES_GENCODE = -gencode arch=compute_75,code=sm_75 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_60,code=sm_60

INCDIR = -I.
DBG    = -g
OPT    = -O2
CPP    = g++
CFLAGS = $(DBG) $(OPT) $(INCDIR)
LINK   = -lm

.cpp.o:
	$(CPP) $(CFLAGS) -c $< -o $@

all: impDriver 

impDriver: driverCudaBp.o stereo.o RunBpStereoSet.o imageHelpers.o stereoResultsEval.o SmoothImageCUDA.o RunBpStereoSetOnGPUWithCUDA.o SmoothImage.o RunBpStereoOptimizedCPU.o SmoothImageCPU.o DetailedTimings.o ProcessBPOnTargetDevice.o ProcessCUDABP.o ProcessOptimizedCPUBP.o
	g++ driverCudaBp.o stereo.o RunBpStereoSet.o imageHelpers.o stereoResultsEval.o SmoothImageCUDA.o SmoothImage.o RunBpStereoSetOnGPUWithCUDA.o RunBpStereoOptimizedCPU.o SmoothImageCPU.o DetailedTimings.o ProcessBPOnTargetDevice.o ProcessCUDABP.o ProcessOptimizedCPUBP.o $(LIB) -fopenmp $(ARCHITECTURE_COMPILE_FLAG) -o driverCudaBp -O -m64

impDriveCPU: driverBpStereoCPU.o stereo.o RunBpStereoSet.o imageHelpers.o stereoResultsEval.o SmoothImage.o SmoothImageCPU.o ProcessBPOnTargetDevice.o DetailedTimings.o RunBpStereoOptimizedCPU.o ProcessOptimizedCPUBP.o
	g++ driverBpStereoCPU.o stereo.o RunBpStereoSet.o imageHelpers.o stereoResultsEval.o SmoothImage.o SmoothImageCPU.o ProcessBPOnTargetDevice.o DetailedTimings.o RunBpStereoOptimizedCPU.o ProcessOptimizedCPUBP.o $(ARCHITECTURE_COMPILE_FLAG) $(LIB_CPU) -fopenmp $(ARCHITECTURE_COMPILE_FLAG) -o driverCPUBp -O -m64

DetailedTimings.o: DetailedTimings.cpp DetailedTimings.h
	g++ DetailedTimings.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

#KernelBpStereoCPU.o: OptimizeCPU/KernelBpStereoCPU.cpp OptimizeCPU/KernelBpStereoCPU.h
#	g++ OptimizeCPU/KernelBpStereoCPU.cpp -c -fopenmp $(ARCHITECTURE_COMPILE_FLAG) $(INCLUDE_DIRS) $(COMPILE_FLAGS)
driverBpStereoCPU.o: driverBpStereoCPU.cpp bpStereoParameters.h bpParametersFromPython.h imageHelpers.o stereoResultsEval.o stereo.o RunBpStereoOptimizedCPU.o
	g++ driverBpStereoCPU.cpp -c -fopenmp $(ARCHITECTURE_COMPILE_FLAG) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessBPOnTargetDevice.o : ProcessBPOnTargetDevice.cpp ProcessBPOnTargetDevice.h DetailedTimings.o
	g++ ProcessBPOnTargetDevice.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessOptimizedCPUBP.o : OptimizeCPU/ProcessOptimizedCPUBP.cpp OptimizeCPU/ProcessOptimizedCPUBP.h ProcessBPOnTargetDevice.o bpStereoParameters.h OptimizeCPU/KernelBpStereoCPU.h OptimizeCPU/KernelBpStereoCPU.cpp
	g++ OptimizeCPU/ProcessOptimizedCPUBP.cpp -c -fopenmp $(ARCHITECTURE_COMPILE_FLAG) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoSet.o: RunBpStereoSet.cpp RunBpStereoSet.h bpStereoParameters.h SmoothImage.o ProcessBPOnTargetDevice.o imageHelpers.o DetailedTimings.o
	g++ RunBpStereoSet.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

imageHelpers.o: imageHelpers.cpp imageHelpers.h
	g++ imageHelpers.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

stereoResultsEval.o: stereoResultsEval.cpp stereoResultsEval.h stereoResultsEvalParameters.h
	g++ stereoResultsEval.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

stereo.o: SingleThreadCPU/stereo.cpp SingleThreadCPU/stereo.h bpStereoCudaParameters.h bpParametersFromPython.h ProcessBPOnTargetDevice.o RunBpStereoSet.o SmoothImage.o
	g++ SingleThreadCPU/stereo.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

SmoothImage.o: SmoothImage.cpp SmoothImage.h 
	g++ SmoothImage.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

SmoothImageCPU.o: OptimizeCPU/SmoothImageCPU.cpp OptimizeCPU/SmoothImageCPU.h SmoothImage.o
	g++ OptimizeCPU/SmoothImageCPU.cpp -c -fopenmp $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoOptimizedCPU.o: OptimizeCPU/RunBpStereoOptimizedCPU.cpp OptimizeCPU/RunBpStereoOptimizedCPU.h SmoothImageCPU.o RunBpStereoSet.o ProcessBPOnTargetDevice.o ProcessOptimizedCPUBP.o
	g++ OptimizeCPU/RunBpStereoOptimizedCPU.cpp -c -fopenmp $(ARCHITECTURE_COMPILE_FLAG) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoSetOnGPUWithCUDA.o: OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.cpp OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.h ProcessCUDABP.o SmoothImageCUDA.o RunBpStereoSet.o ProcessBPOnTargetDevice.o
	g++ OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessCUDABP.o: OptimizeCUDA/ProcessCUDABP.cpp OptimizeCUDA/ProcessCUDABP.h
	$(NVCC) -x cu -c OptimizeCUDA/ProcessCUDABP.cpp $(ARCHITECTURES_GENCODE) -o ProcessCUDABP.o $(INCLUDE_DIRS) $(COMPILE_FLAGS)

SmoothImageCUDA.o: OptimizeCUDA/SmoothImageCUDA.cpp OptimizeCUDA/SmoothImageCUDA.h OptimizeCUDA/kernalFilter.cu OptimizeCUDA/kernalFilterHeader.cuh
	$(NVCC) -x cu -c OptimizeCUDA/SmoothImageCUDA.cpp $(ARCHITECTURES_GENCODE) -o SmoothImageCUDA.o $(COMPILE_FLAGS)

driverCudaBp.o: driverCudaBp.cu bpStereoCudaParameters.h bpStereoParameters.h bpParametersFromPython.h RunBpStereoSetOnGPUWithCUDA.o RunBpStereoOptimizedCPU.o stereoResultsEval.o stereo.o
	# need to adjust ARCHITECTURES_GENCODE to allow support for compute capability under 6.0 (can't use half precision before compute capability 5.3)
	$(NVCC) -c driverCudaBp.cu $(ARCHITECTURES_GENCODE) -o driverCudaBp.o $(COMPILE_FLAGS) 
make clean:
	rm *.o driverCudaBp driverCPUBp
