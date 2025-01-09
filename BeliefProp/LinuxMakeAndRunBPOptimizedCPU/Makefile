include ../../common.mk

SRC_DIR_BP = ../../${SRC_DIR}/beliefprop

INCLUDE_DIRS = -I../../${SRC_DIR} -I${SRC_DIR_BP}

# contains the library files needed for linking
# need -lstdc++fs for c++17 filesystem library to work
LIB_CPU := -lstdc++fs -lpthread

# set OPENMP_LIB if using OpenMP
OPENMP_LIB := -fopenmp

# setting for CPU vectorization
# switch to -DAVX_256_VECTORIZATION if AVX 512 not supported on target CPU
# switch to -DAVX_512_F16_VECTORIZATION if half-precision vectorization supported on target CPU
CPU_VECTORIZATION_DEFINE = -DAVX_512_VECTORIZATION

# setting to enable float16 vectorization
# needs to be supported on architecture to work (such as support for AVX512_FP16)
FLOAT16_VECTORIZATION_DEFINE = 
#-DFLOAT16_VECTORIZATION

# set compile flags and debug setting
DBG_SETTING    =
COMPILE_FLAGS = -DUNIX -DOPTIMIZED_CPU_RUN ${CPU_VECTORIZATION_DEFINE} ${FLOAT16_VECTORIZATION_DEFINE} ${COMMON_IMP_DEFINES} \
				${DBG_SETTING} $(OPENMP_LIB) -O3 -std=c++20 -Wall -march=native -MMD -MP

# define directory for compiled objects and object files to generate during compilation
OBJ_DIR = obj
OBJECTS = ${OBJ_DIR}/EvaluateAcrossRuns.o ${OBJ_DIR}/InputSignature.o ${OBJ_DIR}/RunResultsSpeedups.o \
${OBJ_DIR}/EvaluateImpResultsBp.o ${OBJ_DIR}/EvaluateImpResults.o ${OBJ_DIR}/RunImpMultTypesAccels.o \
${OBJ_DIR}/RunData.o ${OBJ_DIR}/ParallelParamsBp.o ${OBJ_DIR}/BpLevel.o ${OBJ_DIR}/ParallelParams.o \
${OBJ_DIR}/RunImpMultInputsBp.o ${OBJ_DIR}/RunImpMultInputs.o ${OBJ_DIR}/BpFileHandling.o ${OBJ_DIR}/DisparityMap.o \
${OBJ_DIR}/DisparityMapEvaluation.o ${OBJ_DIR}/stereo.o ${OBJ_DIR}/RunBpOnStereoSet.o ${OBJ_DIR}/BpImage.o \
${OBJ_DIR}/SmoothImage.o ${OBJ_DIR}/SmoothImageCPU.o ${OBJ_DIR}/ProcessBp.o ${OBJ_DIR}/DetailedTimings.o \
${OBJ_DIR}/RunBpOnStereoSetOptimizedCPU.o ${OBJ_DIR}/ProcessBpOptimizedCPU.o
DRIVER_BENCHMARK_CPU_OBJECT = ${OBJ_DIR}/DriverBpStereoCPU.o
DRIVER_BENCHMARK_CPU_OBJECT_CUSTOM = ${OBJ_DIR}/DriverBpStereoCPU_customRun.o

# dependency files corresponding to each object
DEPS := $(OBJECTS:.o=.d)

# executables for belief propagation CPU implementations
DRIVER_BENCHMARK_CPU = driverCPUBp
DRIVER_BENCHMARK_CPU_CUSTOM = driverCPUBpCustom

all: impDriverCPU impDriverCPUCustom

${OBJ_DIR}/%.o: ../../${SRC_DIR}/*/%.cpp
	$(CC) $< -c $(INCLUDE_DIRS) $(COMPILE_FLAGS) -o $@

${OBJ_DIR}/%.o: ${SRC_DIR_BP}/*/%.cpp
	$(CC) $< -c $(INCLUDE_DIRS) $(COMPILE_FLAGS) -o $@

impDriverCPU: ${DRIVER_BENCHMARK_CPU_OBJECT} ${OBJECTS} 
	$(CC) ${DRIVER_BENCHMARK_CPU_OBJECT} ${OBJECTS} $(COMPILE_FLAGS) $(LIB_CPU) $(OPENMP_LIB) -o ${DRIVER_BENCHMARK_CPU}

impDriverCPUCustom: ${DRIVER_BENCHMARK_CPU_OBJECT_CUSTOM} ${OBJECTS} 
	$(CC) ${DRIVER_BENCHMARK_CPU_OBJECT_CUSTOM} ${OBJECTS} $(COMPILE_FLAGS) $(LIB_CPU) $(OPENMP_LIB) -o ${DRIVER_BENCHMARK_CPU_CUSTOM}

Float16Test: ${OBJ_DIR}/Float16Test.o 
	$(CC) ${OBJ_DIR}/Float16Test.o $(COMPILE_FLAGS) $(LIB_CPU) $(OPENMP_LIB) -o float16Test

make clean:
	rm *.o ${OBJ_DIR}/*.o driverCPUBp driverCPUBpCustom float16Test

# include for dependency files
-include $(DEPS)
