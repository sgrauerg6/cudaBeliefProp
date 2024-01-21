OSLOWER := $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# contains the directories to include...
INCLUDE_DIRS := -I. -I./ -I./.. -I./../BeliefProp -I$(COMMONDIR)/inc -I$(SHAREDDIR)/inc

# contains the library files needed for linking
# need -lstdc++fs for c++17 filesystem library to work
LIB_CPU := -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -lstdc++fs -lpthread

# set OPENMP_LIB if using OpenMP
OPENMP_LIB := -fopenmp

# define the path for the compiler
CC := g++

# have include directory as flag
COMPILE_FLAGS += $(INCLUDE_DIRS) -DUNIX -DOPTIMIZED_CPU_RUN -DCOMPILING_FOR_ARM -O3 -std=c++20 -Wall -march=native
DBG    = -g

all: impDriverCPU

impDriverCPU: driverBpStereoCPU.o BpFileHandling.o DisparityMap.o OutputEvaluationResults.o OutputEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o SmoothImage.o SmoothImageCPU.o ProcessBPOnTargetDevice.o DetailedTimings.o RunBpStereoOptimizedCPU.o ProcessOptimizedCPUBP.o
	$(CC) driverBpStereoCPU.o BpFileHandling.o DisparityMap.o OutputEvaluationResults.o OutputEvaluationParameters.o stereo.o RunBpStereoSet.o BpImage.o SmoothImage.o SmoothImageCPU.o ProcessBPOnTargetDevice.o DetailedTimings.o RunBpStereoOptimizedCPU.o ProcessOptimizedCPUBP.o $(COMPILE_FLAGS) $(LIB_CPU) $(OPENMP_LIB) -o driverCPUBp -O

BpFileHandling.o: BpFileProcessing/BpFileHandling.h BpFileProcessing/BpFileHandling.cpp BpFileProcessing/BpFileHandlingConsts.h
	$(CC) BpFileProcessing/BpFileHandling.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)
	
DetailedTimings.o: RuntimeTiming/DetailedTimings.cpp RuntimeTiming/DetailedTimings.h
	$(CC) RuntimeTiming/DetailedTimings.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

driverBpStereoCPU.o: MainDriverFiles/driverBpStereoCPU.cpp MainDriverFiles/RunAndEvaluateBpResults.h BpConstsAndParams/bpStereoParameters.h BpConstsAndParams/bpStructsAndEnums.h BpImage.o stereo.o RunBpStereoOptimizedCPU.o DisparityMap.o
	$(CC) MainDriverFiles/driverBpStereoCPU.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessBPOnTargetDevice.o : BpRunProcessing/ProcessBPOnTargetDevice.cpp BpRunProcessing/ProcessBPOnTargetDevice.h BpConstsAndParams/DetailedTimingBPConsts.h RunSettingsEval/RunSettings.h BpConstsAndParams/bpStereoParameters.h BpConstsAndParams/bpStructsAndEnums.h DetailedTimings.o
	$(CC) BpRunProcessing/ProcessBPOnTargetDevice.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessOptimizedCPUBP.o : BpOptimizeCPU/ProcessOptimizedCPUBP.cpp BpOptimizeCPU/ProcessOptimizedCPUBP.h BpOptimizeCPU/KernelBpStereoCPU.h BpOptimizeCPU/KernelBpStereoCPU.cpp ProcessBPOnTargetDevice.o
	$(CC) BpOptimizeCPU/ProcessOptimizedCPUBP.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoSet.o: BpRunProcessing/RunBpStereoSet.cpp BpRunProcessing/RunBpStereoSet.h BpConstsAndParams/bpStereoParameters.h RunSettingsEval/RunSettings.h BpConstsAndParams/bpStructsAndEnums.h BpConstsAndParams/DetailedTimingBPConsts.h SmoothImage.o ProcessBPOnTargetDevice.o BpImage.o DetailedTimings.o
	$(CC) BpRunProcessing/RunBpStereoSet.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

BpImage.o: BpImageProcessing/BpImage.cpp BpImageProcessing/BpImage.h
	$(CC) BpImageProcessing/BpImage.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

OutputEvaluationParameters.o: BpOutputEvaluation/OutputEvaluationParameters.cpp BpOutputEvaluation/OutputEvaluationParameters.h
	$(CC) BpOutputEvaluation/OutputEvaluationParameters.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

OutputEvaluationResults.o: BpOutputEvaluation/OutputEvaluationResults.cpp BpOutputEvaluation/OutputEvaluationResults.h
	$(CC) BpOutputEvaluation/OutputEvaluationResults.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)
		
DisparityMap.o: BpImageProcessing/DisparityMap.cpp BpImageProcessing/DisparityMap.h OutputEvaluationParameters.o OutputEvaluationResults.o
	$(CC) BpImageProcessing/DisparityMap.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

stereo.o: BpSingleThreadCPU/stereo.cpp BpSingleThreadCPU/stereo.h BpConstsAndParams/bpStereoCudaParameters.h BpConstsAndParams/bpStructsAndEnums.h ProcessBPOnTargetDevice.o RunBpStereoSet.o SmoothImage.o
	$(CC) BpSingleThreadCPU/stereo.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

SmoothImage.o: BpImageProcessing/SmoothImage.cpp BpImageProcessing/SmoothImage.h 
	$(CC) BpImageProcessing/SmoothImage.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

SmoothImageCPU.o: BpOptimizeCPU/SmoothImageCPU.cpp BpOptimizeCPU/SmoothImageCPU.h SmoothImage.o
	$(CC) BpOptimizeCPU/SmoothImageCPU.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoOptimizedCPU.o: BpOptimizeCPU/RunBpStereoOptimizedCPU.cpp BpOptimizeCPU/RunBpStereoOptimizedCPU.h SmoothImageCPU.o RunBpStereoSet.o ProcessBPOnTargetDevice.o ProcessOptimizedCPUBP.o
	$(CC) BpOptimizeCPU/RunBpStereoOptimizedCPU.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

make clean:
	rm *.o driverCPUBp
