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

#include "bpStereoCudaParameters.h"

//include for the kernal functions to be run on the GPU
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "DetailedTimingsCUDA.h"
#include <chrono>
#include <cuda_fp16.h>
#include "ProcessBPOnTarget.h"

class ProcessCUDABPHelperFuncts
{
public:
	//run the given number of iterations of BP at the current level using the given message values in global device memory
	template<typename T>
	static void runBPAtCurrentLevel(BPsettings& algSettings, int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
			T* messageUDeviceCheckerboard1, T* messageDDeviceCheckerboard1, T* messageLDeviceCheckerboard1,
			T* messageRDeviceCheckerboard1, T* messageUDeviceCheckerboard2, T* messageDDeviceCheckerboard2, T* messageLDeviceCheckerboard2,
			T* messageRDeviceCheckerboard2, T* dataCostDeviceCheckerboard1,
			T* dataCostDeviceCheckerboard2);

	//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
	//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
	//in the next level down
	//need two different "sets" of message values to avoid read-write conflicts
	template<typename T>
	static void copyMessageValuesToNextLevelDown(int widthLevelActualIntegerSizePrevLevel, int heightLevelActualIntegerSizePrevLevel,
		int widthLevelActualIntegerSizeNextLevel, int heightLevelActualIntegerSizeNextLevel,
		T* messageUDeviceCheckerboard1CopyFrom, T* messageDDeviceCheckerboard1CopyFrom, T* messageLDeviceCheckerboard1CopyFrom,
		T* messageRDeviceCheckerboard1CopyFrom, T* messageUDeviceCheckerboard2CopyFrom, T* messageDDeviceCheckerboard2CopyFrom,
		T* messageLDeviceCheckerboard2CopyFrom, T* messageRDeviceCheckerboard2CopyFrom, T** messageUDeviceCheckerboard1CopyTo,
		T** messageDDeviceCheckerboard1CopyTo, T** messageLDeviceCheckerboard1CopyTo, T** messageRDeviceCheckerboard1CopyTo,
		T** messageUDeviceCheckerboard2CopyTo, T** messageDDeviceCheckerboard2CopyTo, T** messageLDeviceCheckerboard2CopyTo,
		T** messageRDeviceCheckerboard2CopyTo);

	//initialize the data cost at each pixel for each disparity value
	template<typename T>
	static void initializeDataCosts(float*& image1PixelsDevice, float*& image2PixelsDevice, T* dataCostDeviceCheckerboard1, T* dataCostDeviceCheckerboard2, BPsettings& algSettings);


	//initialize the message values for every pixel at every disparity to DEFAULT_INITIAL_MESSAGE_VAL (value is 0.0f unless changed)
	template<typename T>
	static void initializeMessageValsToDefault(T* messageUDeviceSet0Checkerboard1, T* messageDDeviceSet0Checkerboard1, T* messageLDeviceSet0Checkerboard1, T* messageRDeviceSet0Checkerboard1,
													  T* messageUDeviceSet0Checkerboard2, T* messageDDeviceSet0Checkerboard2, T* messageLDeviceSet0Checkerboard2, T* messageRDeviceSet0Checkerboard2,
													  int widthLevel, int heightLevel, int numPossibleMovements);

	template<typename T>
	static void initializeDataCurrentLevel(T* dataCostStereoCheckerboard1,
			T* dataCostStereoCheckerboard2, T* dataCostDeviceToWriteToCheckerboard1,
			T* dataCostDeviceToWriteToCheckerboard2,
			int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
			int prevWidthLevelActualIntegerSize,
			int prevHeightLevelActualIntegerSize);


	template<typename T>
	static void retrieveOutputDisparity(T* dataCostDeviceCurrentLevelCheckerboard1, T* dataCostDeviceCurrentLevelCheckerboard2,
			T* messageUDeviceSet0Checkerboard1, T* messageDDeviceSet0Checkerboard1, T* messageLDeviceSet0Checkerboard1, T* messageRDeviceSet0Checkerboard1,
			T* messageUDeviceSet0Checkerboard2, T* messageDDeviceSet0Checkerboard2, T* messageLDeviceSet0Checkerboard2, T* messageRDeviceSet0Checkerboard2,
			T* messageUDeviceSet1Checkerboard1, T* messageDDeviceSet1Checkerboard1, T* messageLDeviceSet1Checkerboard1, T* messageRDeviceSet1Checkerboard1,
			T* messageUDeviceSet1Checkerboard2, T* messageDDeviceSet1Checkerboard2, T* messageLDeviceSet1Checkerboard2, T* messageRDeviceSet1Checkerboard2,
			float* resultingDisparityMapDevice, int widthLevel, int heightLevel, int currentCheckerboardSet);

	template<typename T>
	void printDataAndMessageValsToPoint(int xVal, int yVal,
			T* dataCostDeviceCurrentLevelCheckerboard1,
			T* dataCostDeviceCurrentLevelCheckerboard2,
			T* messageUDeviceSet0Checkerboard1,
			T* messageDDeviceSet0Checkerboard1,
			T* messageLDeviceSet0Checkerboard1,
			T* messageRDeviceSet0Checkerboard1,
			T* messageUDeviceSet0Checkerboard2,
			T* messageDDeviceSet0Checkerboard2,
			T* messageLDeviceSet0Checkerboard2,
			T* messageRDeviceSet0Checkerboard2,
			T* messageUDeviceSet1Checkerboard1,
			T* messageDDeviceSet1Checkerboard1,
			T* messageLDeviceSet1Checkerboard1,
			T* messageRDeviceSet1Checkerboard1,
			T* messageUDeviceSet1Checkerboard2,
			T* messageDDeviceSet1Checkerboard2,
			T* messageLDeviceSet1Checkerboard2,
			T* messageRDeviceSet1Checkerboard2, int widthCheckerboard,
			int heightLevel, int currentCheckerboardSet);

	template<typename T>
	void printDataAndMessageValsAtPoint(int xVal, int yVal,
			T* dataCostDeviceCurrentLevelCheckerboard1,
			T* dataCostDeviceCurrentLevelCheckerboard2,
			T* messageUDeviceSet0Checkerboard1,
			T* messageDDeviceSet0Checkerboard1,
			T* messageLDeviceSet0Checkerboard1,
			T* messageRDeviceSet0Checkerboard1,
			T* messageUDeviceSet0Checkerboard2,
			T* messageDDeviceSet0Checkerboard2,
			T* messageLDeviceSet0Checkerboard2,
			T* messageRDeviceSet0Checkerboard2,
			T* messageUDeviceSet1Checkerboard1,
			T* messageDDeviceSet1Checkerboard1,
			T* messageLDeviceSet1Checkerboard1,
			T* messageRDeviceSet1Checkerboard1,
			T* messageUDeviceSet1Checkerboard2,
			T* messageDDeviceSet1Checkerboard2,
			T* messageLDeviceSet1Checkerboard2,
			T* messageRDeviceSet1Checkerboard2, int widthCheckerboard,
			int heightLevel, int currentCheckerboardSet);
};

template<typename T>
class ProcessCUDABP : public ProcessBPOnTarget<beliefPropProcessingDataType>
{
public:
	//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
	//the input images image1PixelsDevice and image2PixelsDevice are stored in the global memory of the GPU
	//the output movements resultingDisparityMapDevice is stored in the global memory of the GPU
	//Return detailed timings of processing (or null if data not collected)
	DetailedTimings* operator()(float* image1PixelsCompDevice, float* image2PixelsCompDevice, float* resultingDisparityMapCompDevice, BPsettings& algSettings);
};

template<>
class ProcessCUDABP<short> : public ProcessBPOnTarget<short>
{
public:
	//if type is specified as short, process as half on GPU
	//note that half is considered a data type for 16-bit floats in CUDA
	DetailedTimings* operator()(float* image1PixelsCompDevice, float* image2PixelsCompDevice, float* resultingDisparityMapCompDevice, BPsettings& algSettings)
	{

#if CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF

		//printf("Processing as half on GPU\n");
		ProcessCUDABP<half> processCUDABPHalfType;
		return processCUDABPHalfType(image1PixelsCompDevice, image2PixelsCompDevice, resultingDisparityMapCompDevice, algSettings);

#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO

		//printf("Processing as half2 on GPU\n");
		ProcessCUDABP<half2> processCUDABPHalfTwoType;
		return processCUDABPHalfTwoType(image1PixelsCompDevice, image2PixelsCompDevice, resultingDisparityMapCompDevice, algSettings);

#else

		printf("ERROR IN DATA TYPE\n");

#endif

	}
};

#endif //RUN_BP_STEREO_HOST_HEADER_CUH
