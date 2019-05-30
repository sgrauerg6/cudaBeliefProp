CUDA_DIR = /usr/local/cuda/
CUDA_SDK_ROOT := 

CU_FILE = driverCudaBp.cu
CU_OBJ = driverCudaBp.o
FILE_DEPENDENCIES = bpStereoCudaParameters.cuh imageHelpersHost.cu imageHelpersHostHeader.cuh kernalBpStereo.cu kernalBpStereoHeader.cuh  kernalFilter.cu kernalFilterHeader.cuh runBpStereoDivideImage.cu runBpStereoDivideImageHeader.cuh runBpStereoDivideImageParameters.cuh runBpStereoHost.cu runBpStereoHostHeader.cuh runBpStereoImageSeries.cu runBpStereoImageSeriesHeader.cuh saveResultingDisparityMap.cu saveResultingDisparityMapHeader.cuh smoothImageHost.cu smoothImageHostHeader.cuh stereoResultsEvalHost.cu stereoResultsEvalHostHeader.cuh stereoResultsEvalParameters.cuh utilityFunctsForEval.cu utilityFunctsForEvalHeader.cuh

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

# define the path for the nvcc cuda compiler
NVCC := $(CUDA_DIR)/bin/nvcc

# have include directory as flag
COMPILE_FLAGS += $(INCLUDE_DIRS) -DUNIX 

# include the optimization level
COMPILE_FLAGS += -O2 -std=c++11

INCDIR = -I.
DBG    = -g
OPT    = -O2
CPP    = g++
CFLAGS = $(DBG) $(OPT) $(INCDIR)
LINK   = -lm

.cpp.o:
	$(CPP) $(CFLAGS) -c $< -o $@

all: impDriver

impDriver: $(CU_OBJ) stereo.o
	g++ $(CU_OBJ) stereo.o $(LIB) -o driverCudaBp -O -m64
	
stereo.o: stereo.cpp bpStereoCudaParameters.cuh
	g++ stereo.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS) 

$(CU_OBJ): $(CU_FILE) $(CU_HEADER) $(FILE_DEPENDENCIES)
	#adjust to allow support for compute capability under 6.0 (can't use half precision before compute capability 5.3)
	#$(NVCC) -c $(CU_FILE) -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_30,code=sm_30 -o $(CU_OBJ) $(COMPILE_FLAGS) 
	$(NVCC) -c $(CU_FILE) -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_60,code=sm_60 -o $(CU_OBJ) $(COMPILE_FLAGS) 
make clean:
	rm *.o driverCudaBp
