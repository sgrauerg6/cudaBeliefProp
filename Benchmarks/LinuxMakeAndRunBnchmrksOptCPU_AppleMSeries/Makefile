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

# set compile flags and debug setting
DBG_SETTING    =
COMPILE_FLAGS = -DUNIX -DOPTIMIZED_CPU_RUN -DCOMPILING_FOR_ARM ${CPU_VECTORIZATION_DEFINE} ${COMMON_IMP_DEFINES} \
				${DBG_SETTING} $(OPENMP_LIB) -O3 -std=c++20 -Wall -march=native -MMD -MP

# define directory for compiled objects and object files to generate during compilation
OBJ_DIR = obj
OBJECTS = ${OBJ_DIR}/EvaluateAcrossRuns.o ${OBJ_DIR}/InputSignature.o ${OBJ_DIR}/RunResultsSpeedups.o \
${OBJ_DIR}/EvalBenchmarkResults.o ${OBJ_DIR}/EvaluateImpResults.o ${OBJ_DIR}/RunImpMultTypesAccels.o \
${OBJ_DIR}/RunData.o ${OBJ_DIR}/ParallelParamsBnchmrks.o ${OBJ_DIR}/ParallelParams.o \
${OBJ_DIR}/RunBenchmarksMultInputs.o ${OBJ_DIR}/RunImpMultInputs.o \
${OBJ_DIR}/DisparityMapEvaluation.o ${OBJ_DIR}/RunBpImpSingThreadCPU.o ${OBJ_DIR}/RunBenchmarksOnInput.o \
${OBJ_DIR}/DetailedTimings.o ${OBJ_DIR}/RunBenchmarksOptCPU.o ${OBJ_DIR}/RunBenchmarks.o
DRIVER_BENCHMARK_CPU_OBJECT = ${OBJ_DIR}/DriverBpStereoCPU.o
DRIVER_BENCHMARK_CPU_OBJECT_CUSTOM = ${OBJ_DIR}/DriverBpStereoCPU_customRun.o

# dependency files corresponding to each object
DEPS := $(OBJECTS:.o=.d)

# executables for belief propagation CPU implementations
DRIVER_BENCHMARK_CPU = driverCPUBnchmrks
DRIVER_BENCHMARK_CPU_CUSTOM = driverCPUBnchmrksCustom

all: impDriverCPU impDriverCPUCustom

${OBJ_DIR}/%.o: ../../${SRC_DIR}/*/%.cpp
	$(CC) $< -c $(INCLUDE_DIRS) $(COMPILE_FLAGS) -o $@

${OBJ_DIR}/%.o: ${SRC_DIR_BNCHMRKS}/*/%.cpp
	$(CC) $< -c $(INCLUDE_DIRS) $(COMPILE_FLAGS) -o $@

impDriverCPU: ${DRIVER_BENCHMARK_CPU_OBJECT} ${OBJECTS} 
	$(CC) ${DRIVER_BENCHMARK_CPU_OBJECT} ${OBJECTS} $(COMPILE_FLAGS) $(LIB_CPU) $(OPENMP_LIB) -o ${DRIVER_BENCHMARK_CPU}

impDriverCPUCustom: ${DRIVER_BENCHMARK_CPU_OBJECT_CUSTOM} ${OBJECTS} 
	$(CC) ${DRIVER_BENCHMARK_CPU_OBJECT_CUSTOM} ${OBJECTS} $(COMPILE_FLAGS) $(LIB_CPU) $(OPENMP_LIB) -o ${DRIVER_BENCHMARK_CPU_CUSTOM}

make clean:
	rm *.o ${OBJ_DIR}/*.o driverCPUBp driverCPUBpCustom

# include for dependency files
-include $(DEPS)
