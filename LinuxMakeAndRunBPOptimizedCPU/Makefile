OSLOWER := $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

TOP_DIR = ..
BP_PATH = ${TOP_DIR}/BeliefProp

# contains the directories to include...
INCLUDE_DIRS := -I. -I./ -I./.. -I./../BeliefProp -I$(COMMONDIR)/inc -I$(SHAREDDIR)/inc

# contains the library files needed for linking
# need -lstdc++fs for c++17 filesystem library to work
LIB_CPU := -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -lstdc++fs -lpthread

# set OPENMP_LIB if using OpenMP
OPENMP_LIB := -fopenmp

# define the path for the compiler
CC := g++

# setting for CPU vectorization
# switch to -DAVX_256_VECTORIZATION if AVX 512 not supported on target CPU
CPU_VECTORIZATION_DEFINE = -DAVX_512_VECTORIZATION

# set to use only smaller data sets
USE_ONLY_SMALLER_DATA_SETTING =
#-DSMALLER_SETS_ONLY

# set to use limited set of test parameters and fewer runs for faster results for testing
USE_LIMITED_TEST_PARAMS_FEWER_RUNS =
#-DLIMITED_TEST_PARAMS_FEWER_RUNS

# have include directory as flag
DBG_SETTING    =
#-g
COMPILE_FLAGS += $(INCLUDE_DIRS) -DUNIX -DOPTIMIZED_CPU_RUN ${CPU_VECTORIZATION_DEFINE} ${USE_ONLY_SMALLER_DATA_SETTING} ${USE_LIMITED_TEST_PARAMS_FEWER_RUNS} ${DBG_SETTING} -O3 -std=c++20 -Wall -march=native

all: impDriverCPU impDriverCPUCustom

impDriverCPU: driverBpStereoCPU.o EvaluateAcrossRuns.o EvaluateBPImpResults.o EvaluateImpResults.o RunEvalImpMultSettings.o RunData.o BpParallelParams.o ParallelParams.o RunEvalBpImp.o RunBenchmarkImp.o BpFileHandling.o DisparityMap.o BpEvaluationResults.o BpEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o SmoothImage.o SmoothImageCPU.o ProcessBPOnTargetDevice.o DetailedTimings.o RunBpStereoOptimizedCPU.o ProcessOptimizedCPUBP.o
	$(CC) driverBpStereoCPU.o EvaluateAcrossRuns.o EvaluateBPImpResults.o EvaluateImpResults.o RunEvalImpMultSettings.o RunData.o BpParallelParams.o ParallelParams.o RunEvalBpImp.o RunBenchmarkImp.o BpFileHandling.o DisparityMap.o BpEvaluationResults.o BpEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o SmoothImage.o SmoothImageCPU.o ProcessBPOnTargetDevice.o DetailedTimings.o RunBpStereoOptimizedCPU.o ProcessOptimizedCPUBP.o $(COMPILE_FLAGS) $(LIB_CPU) $(OPENMP_LIB) -o driverCPUBp -O

impDriverCPUCustom: driverBpStereoCPU_customRun.o EvaluateAcrossRuns.o EvaluateBPImpResults.o EvaluateImpResults.o RunEvalImpMultSettings.o RunData.o BpParallelParams.o ParallelParams.o RunEvalBpImp.o RunBenchmarkImp.o BpFileHandling.o DisparityMap.o BpEvaluationResults.o BpEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o SmoothImage.o SmoothImageCPU.o ProcessBPOnTargetDevice.o DetailedTimings.o RunBpStereoOptimizedCPU.o ProcessOptimizedCPUBP.o
	$(CC) driverBpStereoCPU_customRun.o EvaluateAcrossRuns.o EvaluateBPImpResults.o EvaluateImpResults.o RunEvalImpMultSettings.o RunData.o BpParallelParams.o ParallelParams.o RunEvalBpImp.o RunBenchmarkImp.o BpFileHandling.o DisparityMap.o BpEvaluationResults.o BpEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o SmoothImage.o SmoothImageCPU.o ProcessBPOnTargetDevice.o DetailedTimings.o RunBpStereoOptimizedCPU.o ProcessOptimizedCPUBP.o $(COMPILE_FLAGS) $(LIB_CPU) $(OPENMP_LIB) -o driverCPUBpCustom -O

EvaluateAcrossRuns.o :
	$(CC) ${TOP_DIR}/RunSettingsEval/EvaluateAcrossRuns.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

EvaluateBPImpResults.o : 
	$(CC) ${BP_PATH}/BpResultsEvaluation/EvaluateBPImpResults.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

EvaluateImpResults.o :
	$(CC) ${TOP_DIR}/RunSettingsEval/EvaluateImpResults.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunEvalImpMultSettings.o:
	$(CC) ${TOP_DIR}/RunImp/RunEvalImpMultSettings.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunData.o:
	$(CC) ${TOP_DIR}/RunSettingsEval/RunData.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunEvalBpImp.o:
	$(CC) ${BP_PATH}/BpRunImp/RunEvalBpImp.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

BpParallelParams.o:
	$(CC) ${BP_PATH}/BpRunImp/BpParallelParams.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ParallelParams.o:
	$(CC) ${TOP_DIR}/RunImp/ParallelParams.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBenchmarkImp.o:
	$(CC) ${TOP_DIR}/RunImp/RunBenchmarkImp.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

BpFileHandling.o:
	$(CC) ${BP_PATH}/BpFileProcessing/BpFileHandling.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

DetailedTimings.o:
	$(CC) ${TOP_DIR}/RuntimeTiming/DetailedTimings.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

driverBpStereoCPU.o: BpImage.o stereo.o RunBpStereoOptimizedCPU.o DisparityMap.o
	$(CC) ${TOP_DIR}/MainDriverFiles/driverBpStereoCPU.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

driverBpStereoCPU_customRun.o: BpImage.o stereo.o RunBpStereoOptimizedCPU.o DisparityMap.o
	$(CC) ${TOP_DIR}/MainDriverFiles/driverBpStereoCPU_customRun.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessBPOnTargetDevice.o : DetailedTimings.o
	$(CC) ${BP_PATH}/BpRunProcessing/ProcessBPOnTargetDevice.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessOptimizedCPUBP.o : ProcessBPOnTargetDevice.o
	$(CC) ${BP_PATH}/BpOptimizeCPU/ProcessOptimizedCPUBP.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoSet.o: SmoothImage.o ProcessBPOnTargetDevice.o BpImage.o DetailedTimings.o
	$(CC) ${BP_PATH}/BpRunProcessing/RunBpStereoSet.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

BpImage.o:
	$(CC) ${BP_PATH}/BpImageProcessing/BpImage.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

BpEvaluationParameters.o:
	$(CC) ${BP_PATH}/BpResultsEvaluation/BpEvaluationParameters.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

BpEvaluationResults.o:
	$(CC) ${BP_PATH}/BpResultsEvaluation/BpEvaluationResults.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

DisparityMap.o: BpEvaluationParameters.o BpEvaluationResults.o
	$(CC) ${BP_PATH}/BpImageProcessing/DisparityMap.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

stereo.o: ProcessBPOnTargetDevice.o RunBpStereoSet.o SmoothImage.o
	$(CC) ${BP_PATH}/BpSingleThreadCPU/stereo.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

SmoothImage.o:
	$(CC) ${BP_PATH}/BpImageProcessing/SmoothImage.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

SmoothImageCPU.o: SmoothImage.o
	$(CC) ${BP_PATH}/BpOptimizeCPU/SmoothImageCPU.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoOptimizedCPU.o: SmoothImageCPU.o RunBpStereoSet.o ProcessBPOnTargetDevice.o ProcessOptimizedCPUBP.o
	$(CC) ${BP_PATH}/BpOptimizeCPU/RunBpStereoOptimizedCPU.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

make clean:
	rm *.o driverCPUBp driverCPUBpCustom
