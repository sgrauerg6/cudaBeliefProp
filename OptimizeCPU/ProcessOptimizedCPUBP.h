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

#ifndef BP_STEREO_PROCESSING_OPTIMIZED_CPU_H
#define BP_STEREO_PROCESSING_OPTIMIZED_CPU_H

#include "bpStereoParameters.h"

//include for the kernal functions to be run on the GPU
#include "KernelBpStereoCPU.h"
#include <vector>
#include <algorithm>
#include <chrono>
#include "ProcessBPOnTargetDevice.h"

template<typename T>
class ProcessOptimizedCPUBP : public ProcessBPOnTargetDevice<T>
{
public:
		int getCheckerboardWidthTargetDevice(int widthLevelActualIntegerSize)
		{
			return (int)ceil(((float)widthLevelActualIntegerSize) / 2.0);
		}

		void allocateMemoryOnTargetDevice(void** arrayToAllocate, int numBytesAllocate)
		{
			*arrayToAllocate = malloc(numBytesAllocate);
		}

		void freeMemoryOnTargetDevice(void* arrayToFree)
		{
			free(arrayToFree);
		}

	//initialize the data cost at each pixel for each disparity value
		void initializeDataCosts(float* image1PixelsDevice, float* image2PixelsDevice, T* dataCostDeviceCheckerboard1, T* dataCostDeviceCheckerboard2, BPsettings& algSettings);

		void initializeDataCurrentLevel(T* dataCostStereoCheckerboard1,
				T* dataCostStereoCheckerboard2, T* dataCostDeviceToWriteToCheckerboard1,
				T* dataCostDeviceToWriteToCheckerboard2,
				int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
				int prevWidthLevelActualIntegerSize,
				int prevHeightLevelActualIntegerSize);

		//initialize the message values for every pixel at every disparity to DEFAULT_INITIAL_MESSAGE_VAL (value is 0.0f unless changed)
		void initializeMessageValsToDefault(T* messageUDeviceSet0Checkerboard1, T* messageDDeviceSet0Checkerboard1, T* messageLDeviceSet0Checkerboard1, T* messageRDeviceSet0Checkerboard1,
														  T* messageUDeviceSet0Checkerboard2, T* messageDDeviceSet0Checkerboard2, T* messageLDeviceSet0Checkerboard2, T* messageRDeviceSet0Checkerboard2,
														  int widthLevel, int heightLevel, int numPossibleMovements);

		//run the given number of iterations of BP at the current level using the given message values in global device memory
		void runBPAtCurrentLevel(BPsettings& algSettings, int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
				T* messageUDeviceCheckerboard1, T* messageDDeviceCheckerboard1, T* messageLDeviceCheckerboard1,
				T* messageRDeviceCheckerboard1, T* messageUDeviceCheckerboard2, T* messageDDeviceCheckerboard2, T* messageLDeviceCheckerboard2,
				T* messageRDeviceCheckerboard2, T* dataCostDeviceCheckerboard1,
				T* dataCostDeviceCheckerboard2);

		//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
		//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
		//in the next level down
		//need two different "sets" of message values to avoid read-write conflicts
		void copyMessageValuesToNextLevelDown(int widthLevelActualIntegerSizePrevLevel, int heightLevelActualIntegerSizePrevLevel,
			int widthLevelActualIntegerSizeNextLevel, int heightLevelActualIntegerSizeNextLevel,
			T* messageUDeviceCheckerboard1CopyFrom, T* messageDDeviceCheckerboard1CopyFrom, T* messageLDeviceCheckerboard1CopyFrom,
			T* messageRDeviceCheckerboard1CopyFrom, T* messageUDeviceCheckerboard2CopyFrom, T* messageDDeviceCheckerboard2CopyFrom,
			T* messageLDeviceCheckerboard2CopyFrom, T* messageRDeviceCheckerboard2CopyFrom, T** messageUDeviceCheckerboard1CopyTo,
			T** messageDDeviceCheckerboard1CopyTo, T** messageLDeviceCheckerboard1CopyTo, T** messageRDeviceCheckerboard1CopyTo,
			T** messageUDeviceCheckerboard2CopyTo, T** messageDDeviceCheckerboard2CopyTo, T** messageLDeviceCheckerboard2CopyTo,
			T** messageRDeviceCheckerboard2CopyTo);

		void retrieveOutputDisparity(T* dataCostDeviceCurrentLevelCheckerboard1, T* dataCostDeviceCurrentLevelCheckerboard2,
				T* messageUDeviceSet0Checkerboard1, T* messageDDeviceSet0Checkerboard1, T* messageLDeviceSet0Checkerboard1, T* messageRDeviceSet0Checkerboard1,
				T* messageUDeviceSet0Checkerboard2, T* messageDDeviceSet0Checkerboard2, T* messageLDeviceSet0Checkerboard2, T* messageRDeviceSet0Checkerboard2,
				T* messageUDeviceSet1Checkerboard1, T* messageDDeviceSet1Checkerboard1, T* messageLDeviceSet1Checkerboard1, T* messageRDeviceSet1Checkerboard1,
				T* messageUDeviceSet1Checkerboard2, T* messageDDeviceSet1Checkerboard2, T* messageLDeviceSet1Checkerboard2, T* messageRDeviceSet1Checkerboard2,
				float* resultingDisparityMapDevice, int widthLevel, int heightLevel, int currentCheckerboardSet);

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

	//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
	//input is images image1Pixels and image1Pixels
	//output is resultingDisparityMap
	//DetailedTimings* operator()(float* image1PixelsCompDevice, float* image2PixelsCompDevice, float* resultingDisparityMapCompDevice, BPsettings& algSettings);
};

//if not using AVX-256 or AVX-512, process using float if short data type used
//Processing with short data type has been implemented when using AVX-256
//NOTE: THIS HAS BEEN REMOVED AND JUST PRINTS AN ERROR THAT NOT SUPPORTED
#if (CPU_OPTIMIZATION_SETTING != USE_AVX_256) && (CPU_OPTIMIZATION_SETTING != USE_AVX_512)

#endif


#endif //RUN_BP_STEREO_HOST_HEADER_CUH
