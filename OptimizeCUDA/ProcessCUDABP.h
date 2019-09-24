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

#include "ParameterFiles/bpStereoCudaParameters.h"

//include for the kernal functions to be run on the GPU
#include <cuda_runtime.h>
#include "../BpAndSmoothProcessing/ProcessBPOnTargetDevice.h"

#if ((CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF) || (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO))
#include <cuda_fp16.h>
#endif

template<typename T>
class ProcessCUDABP : public ProcessBPOnTargetDevice<T>
{
public:

	void allocateMemoryOnTargetDevice(void** arrayToAllocate,
			unsigned long numBytesAllocate) {
		cudaMalloc(arrayToAllocate, numBytesAllocate);
	}

	void freeMemoryOnTargetDevice(void* arrayToFree) {
		cudaFree(arrayToFree);
	}

	//initialize the data cost at each pixel for each disparity value
	void initializeDataCosts(const BPsettings& algSettings, const levelProperties& currentLevelProperties,
			float* image1PixelsCompDevice, float* image2PixelsCompDevice, T* dataCostDeviceCheckerboard1,
			T* dataCostDeviceCheckerboard2);

	void initializeDataCurrentLevel(const levelProperties& currentLevelProperties,
			const levelProperties& prevLevelProperties,
			T* dataCostStereoCheckerboard1,
			T* dataCostStereoCheckerboard2,
			T* dataCostDeviceToWriteToCheckerboard1,
			T* dataCostDeviceToWriteToCheckerboard2);

	//initialize the message values for every pixel at every disparity to DEFAULT_INITIAL_MESSAGE_VAL (value is 0.0f unless changed)
	void initializeMessageValsToDefault(
			const levelProperties& currentLevelProperties,
			const checkerboardMessages<T>& messagesDeviceCheckerboard0,
			const checkerboardMessages<T>& messagesDeviceCheckerboard1);

	//run the given number of iterations of BP at the current level using the given message values in global device memory
	void runBPAtCurrentLevel(const BPsettings& algSettings,
			const levelProperties& currentLevelProperties,
			T* dataCostDeviceCurrentLevelCheckerboard1,
			T* dataCostDeviceCurrentLevelCheckerboard2,
			const checkerboardMessages<T>& messagesDeviceCheckerboard0,
			const checkerboardMessages<T>& messagesDeviceCheckerboard1);

	//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
	//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
	//in the next level down
	//need two different "sets" of message values to avoid read-write conflicts
	void copyMessageValuesToNextLevelDown(
			const levelProperties& currentLevelProperties,
			const levelProperties& nextlevelProperties,
			const checkerboardMessages<T>& messagesDeviceCheckerboard0CopyFrom,
			const checkerboardMessages<T>& messagesDeviceCheckerboard1CopyFrom,
			const checkerboardMessages<T>& messagesDeviceCheckerboard0CopyTo,
			const checkerboardMessages<T>& messagesDeviceCheckerboard1CopyTo);

	void retrieveOutputDisparity(
			const Checkerboard_Parts currentCheckerboardSet,
			const levelProperties& currentLevelProperties,
			T* dataCostDeviceCurrentLevelCheckerboard1,
			T* dataCostDeviceCurrentLevelCheckerboard2,
			const checkerboardMessages<T>& messagesDeviceSet0Checkerboard0,
			const checkerboardMessages<T>& messagesDeviceSet0Checkerboard1,
			const checkerboardMessages<T>& messagesDeviceSet1Checkerboard0,
			const checkerboardMessages<T>& messagesDeviceSet1Checkerboard1,
			float* resultingDisparityMapCompDevice);

	void printDataAndMessageValsToPoint(int xVal, int yVal,
			const levelProperties& currentLevelProperties,
			T* dataCostDeviceCurrentLevelCheckerboard1,
			T* dataCostDeviceCurrentLevelCheckerboard2,
			const checkerboardMessages<T>& messagesDeviceSet0Checkerboard0,
			const checkerboardMessages<T>& messagesDeviceSet0Checkerboard1,
			const checkerboardMessages<T>& messagesDeviceSet1Checkerboard0,
			const checkerboardMessages<T>& messagesDeviceSet1Checkerboard1,
			const Checkerboard_Parts currentCheckerboardSet);

	void printDataAndMessageValsAtPoint(int xVal, int yVal,
			const levelProperties& levelProperties,
			T* dataCostDeviceCurrentLevelCheckerboard1,
			T* dataCostDeviceCurrentLevelCheckerboard2,
			const checkerboardMessages<T>& messagesDeviceSet0Checkerboard0,
			const checkerboardMessages<T>& messagesDeviceSet0Checkerboard1,
			const checkerboardMessages<T>& messagesDeviceSet1Checkerboard0,
			const checkerboardMessages<T>& messagesDeviceSet1Checkerboard1,
			const Checkerboard_Parts currentCheckerboardSet);

};


#endif //RUN_BP_STEREO_HOST_HEADER_CUH
