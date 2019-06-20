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
ARCHITECTURE_COMPILE_FLAG = -march=skylake
# ARCHITECTURE_COMPILE_FLAG = -march=skylake-avx512
#ARCHITECTURE_COMPILE_FLAG = -march=skylake-avx512

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

impDriver: $(CU_OBJ) stereo.o RunBpStereoSet.o imageHelpers.o stereoResultsEval.o SmoothImageCUDA.o RunBpStereoSetOnGPUWithCUDA.o SmoothImage.o RunBpStereoOptimizedCPU.o SmoothImageCPU.o DetailedTimings.o DetailedTimingsCUDA.o ProcessBPOnTargetDevice.o ProcessCUDABP.o ProcessOptimizedCPUBP.o
	g++ $(CU_OBJ) stereo.o RunBpStereoSet.o imageHelpers.o stereoResultsEval.o SmoothImageCUDA.o SmoothImage.o RunBpStereoSetOnGPUWithCUDA.o RunBpStereoOptimizedCPU.o SmoothImageCPU.o DetailedTimings.o DetailedTimingsCUDA.o ProcessBPOnTargetDevice.o ProcessCUDABP.o ProcessOptimizedCPUBP.o $(LIB) -fopenmp -o driverCudaBp -O -m64
	#g++ $(CU_OBJ) stereo.o RunBpStereoSet.o imageHelpers.o stereoResultsEval.o SmoothImageCUDA.o SmoothImage.o RunBpStereoSetOnGPUWithCUDA.o RunBpStereoOptimizedCPU.o SmoothImageCPU.o $(LIB) -o driverCudaBp -O -m64

impDriveCPU: driverBpStereoCPU.o stereo.o RunBpStereoSet.o imageHelpers.o stereoResultsEval.o SmoothImage.o RunBpStereoOptimizedCPU.o SmoothImageCPU.o ProcessBPOnTargetDevice.o DetailedTimings.o ProcessOptimizedCPUBP.o
	g++ driverBpStereoCPU.o stereo.o RunBpStereoSet.o imageHelpers.o stereoResultsEval.o SmoothImage.o RunBpStereoOptimizedCPU.o SmoothImageCPU.o ProcessBPOnTargetDevice.o DetailedTimings.o ProcessOptimizedCPUBP.o $(ARCHITECTURE_COMPILE_FLAG) $(LIB_CPU) -fopenmp -o driverCPUBp -O -m64

DetailedTimingsCUDA.o: OptimizeCUDA/DetailedTimingsCUDA.cpp OptimizeCUDA/DetailedTimingsCUDA.h
	g++ OptimizeCUDA/DetailedTimingsCUDA.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

#runBpStereoHost.o: OptimizeCUDA/runBpStereoHost.cpp OptimizeCUDA/runBpStereoHost.h
#	$(NVCC) -x cu -c OptimizeCUDA/runBpStereoHost.cpp $(ARCHITECTURES_GENCODE) -o runBpStereoHost.o $(INCLUDE_DIRS) $(COMPILE_FLAGS)

DetailedTimings.o: DetailedTimings.cpp DetailedTimings.h
	g++ DetailedTimings.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)
		
driverBpStereoCPU.o: driverBpStereoCPU.cpp bpStereoParameters.h bpParametersFromPython.h
	g++ driverBpStereoCPU.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)
	
ProcessBPOnTargetDevice.o : ProcessBPOnTargetDevice.cpp ProcessBPOnTargetDevice.h
	g++ ProcessBPOnTargetDevice.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)
	
ProcessOptimizedCPUBP.o : OptimizeCPU/ProcessOptimizedCPUBP.cpp OptimizeCPU/ProcessOptimizedCPUBP.h
	g++ OptimizeCPU/ProcessOptimizedCPUBP.cpp -c -fopenmp $(ARCHITECTURE_COMPILE_FLAG) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoSet.o: RunBpStereoSet.cpp RunBpStereoSet.h
	g++ RunBpStereoSet.cpp -x cu -c $(INCLUdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpdriverCPUBpDE_DIRS) $(COMPILE_FLAGS)

imageHelpers.o: imageHelpers.cpp imageHelpers.h
	g++ imageHelpers.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

stereoResultsEval.o: stereoResultsEval.cpp stereoResultsEval.h stereoResultsEvalParameters.h
	g++ stereoResultsEval.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

stereo.o: SingleThreadCPU/stereo.cpp SingleThreadCPU/stereo.h bpStereoCudaParameters.h bpParametersFromPython.h bpStereoParameters.h
	g++ SingleThreadCPU/stereo.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)
	
SmoothImage.o: SmoothImage.cpp SmoothImage.h
	g++ SmoothImage.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)
	
SmoothImageCPU.o: OptimizeCPU/SmoothImageCPU.cpp OptimizeCPU/SmoothImageCPU.h
	g++ OptimizeCPU/SmoothImageCPU.cpp -c -fopenmp $(INCLUDE_DIRS) $(COMPILE_FLAGS)
	#g++ OptimizeCPU/SmoothImageCPU.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)
	
RunBpStereoOptimizedCPU.o: OptimizeCPU/RunBpStereoOptimizedCPU.cpp OptimizeCPU/RunBpStereoOptimizedCPU.h bpParametersFromPython.h bpStereoParameters.h OptimizeCPU/KernelBpStereoCPU.cpp OptimizeCPU/KernelBpStereoCPU.h OptimizeCPU/ProcessOptimizedCPUBP.cpp OptimizeCPU/ProcessOptimizedCPUBP.h OptimizeCPU/SmoothImageCPU.cpp OptimizeCPU/SmoothImageCPU.h
	g++ OptimizeCPU/RunBpStereoOptimizedCPU.cpp -c -fopenmp $(ARCHITECTURE_COMPILE_FLAG) $(INCLUDE_DIRS) $(COMPILE_FLAGS)
	#g++ OptimizeCPU/RunBpStereoOptimizedCPU.cpp -c -mavx2 $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessCUDABP.o: OptimizeCUDA/ProcessCUDABP.cpp OptimizeCUDA/ProcessCUDABP.h
	$(NVCC) -x cu -c OptimizeCUDA/ProcessCUDABP.cpp $(ARCHITECTURES_GENCODE) -o ProcessCUDABP.o $(INCLUDE_DIRS) $(COMPILE_FLAGS)
		
SmoothImageCUDA.o: OptimizeCUDA/SmoothImageCUDA.cpp OptimizeCUDA/SmoothImageCUDA.h OptimizeCUDA/kernalFilter.cu OptimizeCUDA/kernalFilterHeader.cuh
	$(NVCC) -x cu -c OptimizeCUDA/SmoothImageCUDA.cpp $(ARCHITECTURES_GENCODE) -o SmoothImageCUDA.o $(COMPILE_FLAGS)
	
RunBpStereoSetOnGPUWithCUDA.o: OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.cpp OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.h OptimizeCUDA/kernalBpStereo.cu OptimizeCUDA/kernalBpStereoHeader.cuh bpStereoParameters.h bpStereoCudaParameters.h bpParametersFromPython.h OptimizeCUDA/DetailedTimingsCUDA.h
	$(NVCC) -x cu -c OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.cpp $(ARCHITECTURES_GENCODE) -o RunBpStereoSetOnGPUWithCUDA.o $(COMPILE_FLAGS)

$(CU_OBJ): $(CU_FILE) $(CU_HEADER) $(FILE_DEPENDENCIES)
	# need to adjust ARCHITECTURES_GENCODE to allow support for compute capability under 6.0 (can't use half precision before compute capability 5.3)
	$(NVCC) -c $(CU_FILE) $(ARCHITECTURES_GENCODE) -o $(CU_OBJ) $(COMPILE_FLAGS) 
make clean:
	rm *.o driverCudaBp driverCPUBp
