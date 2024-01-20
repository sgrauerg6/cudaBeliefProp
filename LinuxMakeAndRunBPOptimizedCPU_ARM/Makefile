OSLOWER := $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# contains the directories to include...
INCLUDE_DIRS := -I. -I./ -I./.. -I$(COMMONDIR)/inc -I$(SHAREDDIR)/inc

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

BpFileHandling.o: ../FileProcessing/BpFileHandling.h ../FileProcessing/BpFileHandling.cpp ../FileProcessing/BpFileHandlingConsts.h
	$(CC) ../FileProcessing/BpFileHandling.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)
	
DetailedTimings.o: ../RuntimeTiming/DetailedTimings.cpp ../RuntimeTiming/DetailedTimings.h
	$(CC) ../RuntimeTiming/DetailedTimings.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

driverBpStereoCPU.o: ../MainDriverFiles/driverBpStereoCPU.cpp ../MainDriverFiles/RunAndEvaluateBpResults.h ../ParameterFiles/bpStereoParameters.h ../ParameterFiles/bpStructsAndEnums.h BpImage.o stereo.o RunBpStereoOptimizedCPU.o DisparityMap.o
	$(CC) ../MainDriverFiles/driverBpStereoCPU.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessBPOnTargetDevice.o : ../BpAndSmoothProcessing/ProcessBPOnTargetDevice.cpp ../BpAndSmoothProcessing/ProcessBPOnTargetDevice.h ../RuntimeTiming/DetailedTimingBPConsts.h ../ParameterFiles/RunSettings.h ../ParameterFiles/bpStereoParameters.h ../ParameterFiles/bpStructsAndEnums.h DetailedTimings.o
	$(CC) ../BpAndSmoothProcessing/ProcessBPOnTargetDevice.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

ProcessOptimizedCPUBP.o : ../OptimizeCPU/ProcessOptimizedCPUBP.cpp ../OptimizeCPU/ProcessOptimizedCPUBP.h ../OptimizeCPU/KernelBpStereoCPU.h ../OptimizeCPU/KernelBpStereoCPU.cpp ProcessBPOnTargetDevice.o
	$(CC) ../OptimizeCPU/ProcessOptimizedCPUBP.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoSet.o: ../BpAndSmoothProcessing/RunBpStereoSet.cpp ../BpAndSmoothProcessing/RunBpStereoSet.h ../ParameterFiles/bpStereoParameters.h ../ParameterFiles/RunSettings.h ../ParameterFiles/bpStructsAndEnums.h ../RuntimeTiming/DetailedTimingBPConsts.h SmoothImage.o ProcessBPOnTargetDevice.o BpImage.o DetailedTimings.o
	$(CC) ../BpAndSmoothProcessing/RunBpStereoSet.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

BpImage.o: ../ImageDataAndProcessing/BpImage.cpp ../ImageDataAndProcessing/BpImage.h
	$(CC) ../ImageDataAndProcessing/BpImage.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

OutputEvaluationParameters.o: ../OutputEvaluation/OutputEvaluationParameters.cpp ../OutputEvaluation/OutputEvaluationParameters.h
	$(CC) ../OutputEvaluation/OutputEvaluationParameters.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

OutputEvaluationResults.o: ../OutputEvaluation/OutputEvaluationResults.cpp ../OutputEvaluation/OutputEvaluationResults.h
	$(CC) ../OutputEvaluation/OutputEvaluationResults.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)
		
DisparityMap.o: ../OutputEvaluation/DisparityMap.cpp ../OutputEvaluation/DisparityMap.h OutputEvaluationParameters.o OutputEvaluationResults.o
	$(CC) ../OutputEvaluation/DisparityMap.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

stereo.o: ../SingleThreadCPU/stereo.cpp ../SingleThreadCPU/stereo.h ../ParameterFiles/bpStereoCudaParameters.h ../ParameterFiles/bpStructsAndEnums.h ProcessBPOnTargetDevice.o RunBpStereoSet.o SmoothImage.o
	$(CC) ../SingleThreadCPU/stereo.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

SmoothImage.o: ../BpAndSmoothProcessing/SmoothImage.cpp ../BpAndSmoothProcessing/SmoothImage.h 
	$(CC) ../BpAndSmoothProcessing/SmoothImage.cpp -c $(INCLUDE_DIRS) $(COMPILE_FLAGS)

SmoothImageCPU.o: ../OptimizeCPU/SmoothImageCPU.cpp ../OptimizeCPU/SmoothImageCPU.h SmoothImage.o
	$(CC) ../OptimizeCPU/SmoothImageCPU.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

RunBpStereoOptimizedCPU.o: ../OptimizeCPU/RunBpStereoOptimizedCPU.cpp ../OptimizeCPU/RunBpStereoOptimizedCPU.h SmoothImageCPU.o RunBpStereoSet.o ProcessBPOnTargetDevice.o ProcessOptimizedCPUBP.o
	$(CC) ../OptimizeCPU/RunBpStereoOptimizedCPU.cpp -c $(OPENMP_LIB) $(INCLUDE_DIRS) $(COMPILE_FLAGS)

make clean:
	rm *.o driverCPUBp
