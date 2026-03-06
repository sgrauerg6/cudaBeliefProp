include ../../common.mk

SRC_DIR_BNCHMRKS = ../../${SRC_DIR}/benchmarks

INCLUDE_DIRS = -I../../${SRC_DIR} -I${SRC_DIR_BNCHMRKS}

# contains the library files needed for linking
# need -lstdc++fs for c++17 filesystem library to work
LIB_CPU := -lpthread 
#-F/System/Library/Frameworks
#-ldispatch
#-lstdc++fs 

# set OPENMP_LIB if using OpenMP
OPENMP_LIB :=
#-fopenmp

# setting for CPU vectorization
# no CPU vectorization define needed for ARM since NEON is
# only option so that is used
CPU_VECTORIZATION_DEFINE =

USE_FUSED_MULT_ADD_DEFINE = -DUSE_FUSED_MULTIPLY_ADD
#-DUSE_FUSED_MULTIPLY_ADD

# set compile flags and debug setting
DBG_SETTING    =
COMPILE_FLAGS = -DUNIX -DOPTIMIZED_METAL_RUN -DCOMPILING_FOR_ARM ${CPU_VECTORIZATION_DEFINE} ${USE_FUSED_MULT_ADD_DEFINE} ${COMMON_IMP_DEFINES} \
				${DBG_SETTING} $(OPENMP_LIB) -O3 -std=c++20 -Wall -march=native -MMD -MP -I/Users/scottgg/Downloads/PerformingCalculationsOnAGPU/metal-cpp/ -F/Users/scottgg/Downloads/PerformingCalculationsOnAGPU/metal-cpp/

# Location of file with Metal kernel(s)
METAL_SRC := ${SRC_DIR_BNCHMRKS}/benchmarksMetal/KernelBnchmrksMetal.metal

# Frameworks required for Metal, CoreGraphics, and Foundation
FRAMEWORKS := -framework Metal -framework Cocoa -framework QuartzCore
# Compiler flags: C++17, warnings, include path for metal-cpp headers
CXXFLAGS := -std=c++17 -Wall -Wextra -I/Users/scottgg/Downloads/PerformingCalculationsOnAGPU/metal-cpp/ -F/Users/scottgg/Downloads/PerformingCalculationsOnAGPU/metal-cpp/

# define directory for compiled objects and object files to generate during compilation
OBJ_DIR = obj
OBJECTS = ${OBJ_DIR}/EvalAcrossRuns.o ${OBJ_DIR}/InputSignature.o ${OBJ_DIR}/RunResultsSpeedups.o \
${OBJ_DIR}/EvalImpResultsBnchmrks.o ${OBJ_DIR}/EvalImpResults.o ${OBJ_DIR}/RunImpMultTypesAccels.o \
${OBJ_DIR}/RunData.o ${OBJ_DIR}/ParallelParamsBnchmrks.o ${OBJ_DIR}/ParallelParams.o \
${OBJ_DIR}/RunImpMultInputsBnchmrks.o ${OBJ_DIR}/RunImpMultInputs.o \
${OBJ_DIR}/RunBnchmrksSingThreadCPU.o ${OBJ_DIR}/RunImpOnInputBnchmrks.o \
${OBJ_DIR}/DetailedTimings.o ${OBJ_DIR}/RunBnchmrksMetal.o ${OBJ_DIR}/RunBnchmrks.o
DRIVER_BENCHMARK_CPU_OBJECT = ${OBJ_DIR}/DriverBenchmarksCPU.o

# dependency files corresponding to each object
DEPS := $(OBJECTS:.o=.d)

# executables for benchmarks Metal implementation
DRIVER_BENCHMARK_CPU = driverCPUBnchmrks

all: impDriverCPU

# Rule to compile the Metal Shading Language source into a .metallib
$(OBJ_DIR)/default.metallib: $(METAL_SRC) | $(OBJ_DIR)
	# 1. Compile the Metal source to an intermediate archive (.air)
	xcrun metal -c $(METAL_SRC) -o $(OBJ_DIR)/$(notdir $(METAL_SRC:.metal=.air))
	# 2. Link the archive into a redistributable Metal library (.metallib)
	xcrun metallib $(OBJ_DIR)/$(notdir $(METAL_SRC:.metal=.air)) -o $(OBJ_DIR)/default.metallib

${OBJ_DIR}/%.o: ../../${SRC_DIR}/*/%.cpp
	$(CC) $< -c $(INCLUDE_DIRS) $(COMPILE_FLAGS) -o $@

${OBJ_DIR}/%.o: ${SRC_DIR_BNCHMRKS}/*/%.cpp
	$(CC) $< -c $(INCLUDE_DIRS) $(COMPILE_FLAGS) -o $@

impDriverCPU: ${DRIVER_BENCHMARK_CPU_OBJECT} ${OBJECTS} $(OBJ_DIR)/default.metallib
	$(CC) ${DRIVER_BENCHMARK_CPU_OBJECT} ${OBJECTS} $(COMPILE_FLAGS) $(LIB_CPU) $(OPENMP_LIB) -o ${DRIVER_BENCHMARK_CPU} $(FRAMEWORKS)

make clean:
	rm *.o ${OBJ_DIR}/*.o ${OBJ_DIR}/*.d driverCPUBp driverCPUBpCustom

# include for dependency files
-include $(DEPS)
