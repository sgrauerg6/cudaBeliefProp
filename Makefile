CUDA_DIR = /usr/local/cuda/
CUDA_SDK_ROOT := 

CU_FILE = driverCudaBp.cu
CU_OBJ = driverCudaBp.o 

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
NVCC    := $(CUDA_DIR)/bin/nvcc

# have include directory as flag
COMPILE_FLAGS += $(INCLUDE_DIRS) -DUNIX 

# include the optimization level
COMPILE_FLAGS += -O2

all: impDriver

impDriver: $(CU_OBJ)
	g++ $(CU_OBJ) $(LIB) -o driverCudaBp -O -m64

$(CU_OBJ): $(CU_FILE) $(CU_HEADER)
	$(NVCC) -c $(CU_FILE) -gencode arch=compute_61,code=sm_61 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_20,code=sm_20 -o $(CU_OBJ) $(COMPILE_FLAGS) 
make clean:
	rm *.o driverCudaBp
