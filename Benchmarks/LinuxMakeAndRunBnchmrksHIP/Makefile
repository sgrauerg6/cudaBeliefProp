include ../../common.mk

# define the path to HIP on target compile device
HIP_DIR = /opt/rocm
HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# define path, flags, and includes for HIP compilation
HIPCC = hipcc
HIPFLAGS = -O3 --offload-arch=gfx906
HIP_INCLUDE = -I$(ROCM_PATH)/hip/include

# define path to benchmarks source code
SRC_DIR_BNCHMRKS = ../../${SRC_DIR}/benchmarks

# define the directories to include when compiling CUDA implementation
HIP_IMP_INCLUDES := -I../../${SRC_DIR} -I${SRC_DIR_BNCHMRKS} -I${HIP_DIR}/include

# contains the library files needed for linking
# need -lstdc++fs for c++17 filesystem library to work
HIP_IMP_LIBS := -L$(HIP_DIR)/lib64 -lcudart -lstdc++fs

# use bfloat16 data type rather than default half data type for HIP 16-bit float processing
BFLOAT_DATA_TYPE_SETTING = -DUSE_BFLOAT16_FOR_HALF_PRECISION

USE_FUSED_MULT_ADD_DEFINE = -DUSE_FUSED_MULTIPLY_ADD
#-DUSE_FUSED_MULTIPLY_ADD

# defines to add for CUDA processing
HIP_IMP_RUN_DEFINES = -DOPTIMIZED_HIP_RUN ${BFLOAT_DATA_TYPE_SETTING} ${COMMON_IMP_DEFINES}

# set compile flags for CUDA implementation for gcc and nvcc
HIP_IMP_COMPILE_FLAGS_GCC = -DUNIX -O3 -std=c++20 -march=native -Wall ${COMPILE_FLAGS} ${HIP_IMP_RUN_DEFINES} -MMD -MP
HIP_IMP_COMPILE_FLAGS_HIP = -DUNIX -O3 -std=c++20 -arch=native ${COMPILE_FLAGS} ${HIP_IMP_RUN_DEFINES} -MMD -MP
HIP_LINK_FLAGS = -O3 -march=native -std=c++20

# define directory for compiled objects and object files to generate during compilation
OBJ_DIR = obj
OBJECTS = ${OBJ_DIR}/EvalAcrossRuns.o ${OBJ_DIR}/InputSignature.o ${OBJ_DIR}/RunResultsSpeedups.o \
${OBJ_DIR}/EvalImpResultsBnchmrks.o ${OBJ_DIR}/EvalImpResults.o ${OBJ_DIR}/RunImpMultTypesAccels.o \
${OBJ_DIR}/RunData.o ${OBJ_DIR}/ParallelParamsBnchmrks.o ${OBJ_DIR}/ParallelParams.o \
${OBJ_DIR}/RunImpMultInputsBnchmrks.o ${OBJ_DIR}/RunImpMultInputs.o \
${OBJ_DIR}/RunBnchmrksSingThreadCPU.o ${OBJ_DIR}/RunBnchmrks.o \
${OBJ_DIR}/DetailedTimings.o ${OBJ_DIR}/ProcessBnchmrksDevice.o \
${OBJ_DIR}/ProcessBnchmrksHIP.o
DRIVER_BENCHMARK_HIP_OBJECT = ${OBJ_DIR}/DriverBenchmarksHIP.o

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -I $(COMMON_INCLUDE_DIR)
ILDFLAGS  :=
ILDLIBS   :=

ifeq ($(GPU_RUNTIME), CUDA)
	ICXXFLAGS += -x cu
	ICPPFLAGS += -isystem $(HIP_INCLUDE_DIR)
else ifeq ($(GPU_RUNTIME), HIP)
	CXXFLAGS ?= -Wall -Wextra
else
	$(error GPU_RUNTIME is set to "$(GPU_RUNTIME)". GPU_RUNTIME must be either CUDA or HIP)
endif

# dependency files corresponding to each object
DEPS := $(OBJECTS:.o=.d)

# executables for benchmark HIP implementations
DRIVER_BENCHMARK_HIP = driverHIPBnchmrks

all: impDriverHIP

${OBJ_DIR}/%.o: ../../${SRC_DIR}/*/%.cpp
	$(CC) $< -c $(HIP_IMP_INCLUDES) $(HIP_IMP_COMPILE_FLAGS_GCC) -o $@

${OBJ_DIR}/%.o: ${SRC_DIR_BNCHMRKS}/*/%.cpp
	$(CC) $< -c $(HIP_IMP_INCLUDES) $(HIP_IMP_COMPILE_FLAGS_GCC) -o $@

# specify compile commands using HIP compiler for objects with HIP for processing on GPU
${OBJ_DIR}/ProcessBnchmrksHIP.o:
	$(HIPCXX) ${ICXXFLAGS} -c ${SRC_DIR_BNCHMRKS}/benchmarksHIP/ProcessBnchmrksHIP.cpp -Xptxas -v -o $@ $(HIP_IMP_INCLUDES) $(HIP_IMP_COMPILE_FLAGS_HIP)

impDriverHIP: ${OBJECTS} ${DRIVER_BENCHMARK_HIP_OBJECT}
	$(CC) ${DRIVER_BENCHMARK_HIP_OBJECT} ${OBJECTS} $(HIP_IMP_LIBS) -fopenmp $(HIP_LINK_FLAGS) -o ${DRIVER_BENCHMARK_HIP} -O

make clean:
	rm *.o ${OBJ_DIR}/*.o ${OBJ_DIR}/*.d driverHIPBnchmrks

# include for dependency files
-include $(DEPS)
