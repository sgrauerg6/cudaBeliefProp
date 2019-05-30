/*
Copyright (C) 2009 Scott Grauer-Gray, Chandra Kambhamettu, and Kannappan Palaniappan

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

//This function declares the host functions to run the CUDA implementation of Stereo estimation using BP

#ifndef RUN_BP_STEREO_HOST_HEADER_CUH
#define RUN_BP_STEREO_HOST_HEADER_CUH

#include "bpStereoCudaParameters.cuh"

//include for the kernal functions to be run on the GPU
#include "kernalBpStereo.cu"
#include <vector>
#include <algorithm>

struct DetailedTimings
{
	std::vector<double> totalTimeInitSettingsConstMem;
	std::vector<double> totalTimeInitSettingsMallocStart;
	std::vector<double> totalTimeGetDataCostsBottomLevel;
	std::vector<double> totalTimeGetDataCostsHigherLevels;
	std::vector<double> totalTimeInitMessageVals;
	std::vector<double> totalTimeBpIters;
	std::vector<double> timeBpItersKernelTotalTime;
	std::vector<double> totalTimeCopyData;
	std::vector<double> timeCopyDataKernelTotalTime;
	std::vector<double> totalTimeGetOutputDisparity;
	std::vector<double> totalTimeFinalUnbindFree;
	std::vector<double> totalTimeFinalUnbind;
	std::vector<double> totalTimeFinalFree;
	std::vector<double> totalTimed;
	std::vector<double> totalTimeInitMessageValuesKernelTime;
	std::vector<double> totalMemoryProcessingTime;
	std::vector<double> totalComputationProcessing;
	int totNumTimings = 0;

	void SortTimings()
	{
		std::sort(totalTimeInitSettingsConstMem.begin(), totalTimeInitSettingsConstMem.end());
		std::sort(totalTimeInitSettingsMallocStart.begin(), totalTimeInitSettingsMallocStart.end());
		std::sort(totalTimeGetDataCostsBottomLevel.begin(), totalTimeGetDataCostsBottomLevel.end());
		std::sort(totalTimeGetDataCostsHigherLevels.begin(), totalTimeGetDataCostsHigherLevels.end());
		std::sort(totalTimeInitMessageVals.begin(), totalTimeInitMessageVals.end());
		std::sort(totalTimeBpIters.begin(), totalTimeBpIters.end());
		std::sort(timeBpItersKernelTotalTime.begin(), timeBpItersKernelTotalTime.end());
		std::sort(totalTimeCopyData.begin(), totalTimeCopyData.end());
		std::sort(timeCopyDataKernelTotalTime.begin(), timeCopyDataKernelTotalTime.end());
		std::sort(totalTimeGetOutputDisparity.begin(), totalTimeGetOutputDisparity.end());
		std::sort(totalTimeFinalUnbindFree.begin(), totalTimeFinalUnbindFree.end());
		std::sort(totalTimeFinalUnbind.begin(), totalTimeFinalUnbind.end());
		std::sort(totalTimeFinalFree.begin(), totalTimeFinalFree.end());
		std::sort(totalTimed.begin(), totalTimed.end());
		std::sort(totalTimeInitMessageValuesKernelTime.begin(), totalTimeInitMessageValuesKernelTime.end());
		std::sort(totalMemoryProcessingTime.begin(), totalMemoryProcessingTime.end());
		std::sort(totalComputationProcessing.begin(), totalComputationProcessing.end());
	}

	void PrintMedianTimings()
	{
		SortTimings();
		printf("Median Timings\n");
		printf("Time const mem in init settings: %f\n", totalTimeInitSettingsConstMem.at(totNumTimings/2));
		printf("Time init settings malloc: %f\n", totalTimeInitSettingsMallocStart.at(totNumTimings/2));
		printf("Time get data costs bottom level: %f\n", totalTimeGetDataCostsBottomLevel.at(totNumTimings/2));
		printf("Time get data costs higher levels: %f\n", totalTimeGetDataCostsHigherLevels.at(totNumTimings/2));
		printf("Time to init message values: %f\n", totalTimeInitMessageVals.at(totNumTimings/2));
		printf("Time to init message values (kernel portion only): %f\n", totalTimeInitMessageValuesKernelTime.at(totNumTimings/2));
		printf("Total time BP Iters: %f\n", totalTimeBpIters.at(totNumTimings/2));
		printf("Total time BP Iters (kernel portion only): %f\n", timeBpItersKernelTotalTime.at(totNumTimings/2));
		printf("Total time Copy Data: %f\n", totalTimeCopyData.at(totNumTimings/2));
		printf("Total time Copy Data (kernel portion only): %f\n", timeCopyDataKernelTotalTime.at(totNumTimings/2));
		printf("Time get output disparity: %f\n", totalTimeGetOutputDisparity.at(totNumTimings/2));
		printf("Time final unbind free: %f\n", totalTimeFinalUnbindFree.at(totNumTimings/2));
		printf("Time final unbind: %f\n", totalTimeFinalUnbind.at(totNumTimings/2));
		printf("Time final free: %f\n", totalTimeFinalFree.at(totNumTimings/2));
		printf("Total timed: %f\n", totalTimed.at(totNumTimings/2));
		printf("Total memory processing time: %f\n", totalMemoryProcessingTime.at(totNumTimings/2));
		printf("Total computation processing time: %f\n", totalComputationProcessing.at(totNumTimings/2));
	}

	void PrintMedianTimingsToFile(FILE* pFile) {
		SortTimings();
		fprintf(pFile, "Median Timings\n");
		fprintf(pFile, "Time const mem in init settings: %f\n",
				totalTimeInitSettingsConstMem.at(totNumTimings / 2));
		fprintf(pFile, "Time init settings malloc: %f\n",
				totalTimeInitSettingsMallocStart.at(totNumTimings / 2));
		fprintf(pFile, "Time get data costs bottom level: %f\n",
				totalTimeGetDataCostsBottomLevel.at(totNumTimings / 2));
		fprintf(pFile, "Time get data costs higher levels: %f\n",
				totalTimeGetDataCostsHigherLevels.at(totNumTimings / 2));
		fprintf(pFile, "Time to init message values: %f\n",
				totalTimeInitMessageVals.at(totNumTimings / 2));
		fprintf(pFile,
				"Time to init message values (kernel portion only): %f\n",
				totalTimeInitMessageValuesKernelTime.at(totNumTimings / 2));
		fprintf(pFile, "Total time BP Iters: %f\n",
				totalTimeBpIters.at(totNumTimings / 2));
		fprintf(pFile, "Total time BP Iters (kernel portion only): %f\n",
				timeBpItersKernelTotalTime.at(totNumTimings / 2));
		fprintf(pFile, "Total time Copy Data: %f\n",
				totalTimeCopyData.at(totNumTimings / 2));
		fprintf(pFile, "Total time Copy Data (kernel portion only): %f\n",
				timeCopyDataKernelTotalTime.at(totNumTimings / 2));
		fprintf(pFile, "Time get output disparity: %f\n",
				totalTimeGetOutputDisparity.at(totNumTimings / 2));
		fprintf(pFile, "Time final unbind free: %f\n",
				totalTimeFinalUnbindFree.at(totNumTimings / 2));
		fprintf(pFile, "Time final unbind: %f\n",
				totalTimeFinalUnbind.at(totNumTimings / 2));
		fprintf(pFile, "Time final free: %f\n",
				totalTimeFinalFree.at(totNumTimings / 2));
		fprintf(pFile, "Total timed: %f\n", totalTimed.at(totNumTimings / 2));
		fprintf(pFile, "Total memory processing time: %f\n",
				totalMemoryProcessingTime.at(totNumTimings / 2));
		fprintf(pFile, "Total computation processing time: %f\n",
				totalComputationProcessing.at(totNumTimings / 2));
	}
};


texture<float, 1, cudaReadModeElementType> dataCostTexStereoCheckerboard1NotDecInKern;
texture<float, 1, cudaReadModeElementType> dataCostTexStereoCheckerboard2NotDecInKern;

texture<float, 1, cudaReadModeElementType> messageUPrevTexStereoCheckerboard1NotDecInKern;
texture<float, 1, cudaReadModeElementType> messageDPrevTexStereoCheckerboard1NotDecInKern;
texture<float, 1, cudaReadModeElementType> messageLPrevTexStereoCheckerboard1NotDecInKern;
texture<float, 1, cudaReadModeElementType> messageRPrevTexStereoCheckerboard1NotDecInKern;

//functions directed related to running BP to retrieve the movement between the images

//set the current BP settings in the host in constant memory on the device
__host__ void setBPSettingInConstMem(BPsettings& currentBPSettings);


//run the given number of iterations of BP at the current level using the given message values in global device memory
template<typename T>
__host__ void runBPAtCurrentLevel(int& numIterationsAtLevel, int& widthLevelActualIntegerSize, int& heightLevelActualIntegerSize,
	T*& messageUDeviceCheckerboard1, T*& messageDDeviceCheckerboard1, T*& messageLDeviceCheckerboard1,
	T*& messageRDeviceCheckerboard1, T*& messageUDeviceCheckerboard2, T*& messageDDeviceCheckerboard2, T*& messageLDeviceCheckerboard2,
	T*& messageRDeviceCheckerboard2, T* dataCostDeviceCheckerboard1,
	T* dataCostDeviceCheckerboard2);

//run the given number of iterations of BP at the current level using the given message values in global device memory without using textures
__host__ void runBPAtCurrentLevelNoTextures(int& numIterationsAtLevel, int& widthLevelActualIntegerSize, int& heightLevelActualIntegerSize, 
	float*& messageUDeviceCheckerboard1, float*& messageDDeviceCheckerboard1, float*& messageLDeviceCheckerboard1, 
	float*& messageRDeviceCheckerboard1, float*& messageUDeviceCheckerboard2, float*& messageDDeviceCheckerboard2, float*& messageLDeviceCheckerboard2, 
	float*& messageRDeviceCheckerboard2, dim3& grid, dim3& threads, int& numBytesDataAndMessageSetInCheckerboardAtLevel,
	float*& dataCostDeviceCheckerboard1, float*& dataCostDeviceCheckerboard2, int& offsetDataLevel); 

//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
template<typename T>
__host__ void copyMessageValuesToNextLevelDown(int& widthLevelActualIntegerSizePrevLevel, int& heightLevelActualIntegerSizePrevLevel,
	int& widthLevelActualIntegerSizeNextLevel, int& heightLevelActualIntegerSizeNextLevel,
	T*& messageUDeviceCheckerboard1CopyFrom, T*& messageDDeviceCheckerboard1CopyFrom, T*& messageLDeviceCheckerboard1CopyFrom,
	T*& messageRDeviceCheckerboard1CopyFrom, T*& messageUDeviceCheckerboard2CopyFrom, T*& messageDDeviceCheckerboard2CopyFrom,
	T*& messageLDeviceCheckerboard2CopyFrom, T*& messageRDeviceCheckerboard2CopyFrom, T*& messageUDeviceCheckerboard1CopyTo,
	T*& messageDDeviceCheckerboard1CopyTo, T*& messageLDeviceCheckerboard1CopyTo, T*& messageRDeviceCheckerboard1CopyTo,
	T*& messageUDeviceCheckerboard2CopyTo, T*& messageDDeviceCheckerboard2CopyTo, T*& messageLDeviceCheckerboard2CopyTo,
	T*& messageRDeviceCheckerboard2CopyTo);

//initialize the data cost at each pixel for each disparity value
__host__ void initializeDataCosts(float*& image1PixelsDevice, float*& image2PixelsDevice, float*& dataCostDeviceCheckerboard1, float*& dataCostDeviceCheckerboard2, BPsettings& algSettings);


//initialize the message values for every pixel at every disparity to DEFAULT_INITIAL_MESSAGE_VAL (value is 0.0f unless changed)
__host__ void initializeMessageValsToDefault(float*& messageUDeviceSet0Checkerboard1, float*& messageDDeviceSet0Checkerboard1, float*& messageLDeviceSet0Checkerboard1, float*& messageRDeviceSet0Checkerboard1,
												float*& messageUDeviceSet0Checkerboard2, float*& messageDDeviceSet0Checkerboard2, float*& messageLDeviceSet0Checkerboard2, float*& messageRDeviceSet0Checkerboard2,
												int widthOfCheckerboard, int heightOfCheckerboard, int numPossibleMovements);

template<typename T>
__host__ void initializeDataCurrentLevel(T* dataCostDeviceCheckerboard1,
		T* dataCostDeviceCheckerboard2, int prev_level_offset_level,
		int offsetLevel, int widthLevelActualIntegerSize,
		int heightLevelActualIntegerSize, int prevWidthLevelActualIntegerSize,
		int prevHeightLevelActualIntegerSize);

template<typename T>
__host__ void retrieveOutputDisparity(T* dataCostDeviceCurrentLevelCheckerboard1, T* dataCostDeviceCurrentLevelCheckerboard2,
		T* messageUDeviceSet0Checkerboard1, T* messageDDeviceSet0Checkerboard1, T* messageLDeviceSet0Checkerboard1, T* messageRDeviceSet0Checkerboard1,
		T* messageUDeviceSet0Checkerboard2, T* messageDDeviceSet0Checkerboard2, T* messageLDeviceSet0Checkerboard2, T* messageRDeviceSet0Checkerboard2,
		float* resultingDisparityMapDevice, int widthLevel, int heightLevel, int currentCheckerboardSet);

//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
//the input images image1PixelsDevice and image2PixelsDevice are stored in the global memory of the GPU
//the output movements resultingDisparityMapDevice is stored in the global memory of the GPU
__host__ void runBeliefPropStereoCUDA(float*& image1PixelsDevice, float*& image2PixelsDevice, float*& resultingDisparityMapDevice, BPsettings& algSettings, DetailedTimings& timings);

#endif //RUN_BP_STEREO_HOST_HEADER_CUH
