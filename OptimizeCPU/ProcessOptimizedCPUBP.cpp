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

//Defines the functions to run the CUDA implementation of 2-D Stereo estimation using BP

#include "ProcessOptimizedCPUBP.h"
#include "KernelBpStereoCPU.cpp"


template<typename T, typename U>
void ProcessOptimizedCPUBP<T, U>::printDataAndMessageValsAtPoint(int xVal, int yVal, const levelProperties& currentLevelProperties,
		const dataCostData<U>& dataCostDeviceCheckerboard,
		const checkerboardMessages<U>& messagesDeviceSet0Checkerboard0,
		const checkerboardMessages<U>& messagesDeviceSet0Checkerboard1,
		const checkerboardMessages<U>& messagesDeviceSet1Checkerboard0,
		const checkerboardMessages<U>& messagesDeviceSet1Checkerboard1, const Checkerboard_Parts currentCheckerboardSet)
{
	if (currentCheckerboardSet == Checkerboard_Parts::CHECKERBOARD_PART_0) {
		KernelBpStereoCPU::printDataAndMessageValsAtPointKernelCPU<T>(xVal, yVal, currentLevelProperties,
				dataCostDeviceCheckerboard.dataCostCheckerboard0,
				dataCostDeviceCheckerboard.dataCostCheckerboard1,
				messagesDeviceSet0Checkerboard0.messagesU,
				messagesDeviceSet0Checkerboard0.messagesD,
				messagesDeviceSet0Checkerboard0.messagesL,
				messagesDeviceSet0Checkerboard0.messagesR,
				messagesDeviceSet0Checkerboard1.messagesU,
				messagesDeviceSet0Checkerboard1.messagesD,
				messagesDeviceSet0Checkerboard1.messagesL,
				messagesDeviceSet0Checkerboard1.messagesR);
	} else {
		KernelBpStereoCPU::printDataAndMessageValsAtPointKernelCPU<T>(xVal, yVal, currentLevelProperties,
				dataCostDeviceCheckerboard.dataCostCheckerboard0,
				dataCostDeviceCheckerboard.dataCostCheckerboard1,
				messagesDeviceSet1Checkerboard0.messagesU,
				messagesDeviceSet1Checkerboard0.messagesD,
				messagesDeviceSet1Checkerboard0.messagesL,
				messagesDeviceSet1Checkerboard0.messagesR,
				messagesDeviceSet1Checkerboard1.messagesU,
				messagesDeviceSet1Checkerboard1.messagesD,
				messagesDeviceSet1Checkerboard1.messagesL,
				messagesDeviceSet1Checkerboard1.messagesR);
	}
}


template<typename T, typename U>
void ProcessOptimizedCPUBP<T, U>::printDataAndMessageValsToPoint(int xVal, int yVal, const levelProperties& currentLevelProperties,
		const dataCostData<U>& dataCostDeviceCheckerboard,
		const checkerboardMessages<U>& messagesDeviceSet0Checkerboard0,
		const checkerboardMessages<U>& messagesDeviceSet0Checkerboard1,
		const checkerboardMessages<U>& messagesDeviceSet1Checkerboard0,
		const checkerboardMessages<U>& messagesDeviceSet1Checkerboard1,
		const Checkerboard_Parts currentCheckerboardSet)
{
	if (currentCheckerboardSet == Checkerboard_Parts::CHECKERBOARD_PART_0) {
		KernelBpStereoCPU::printDataAndMessageValsToPointKernelCPU<T>(xVal, yVal, currentLevelProperties,
				dataCostDeviceCheckerboard.dataCostCheckerboard0,
				dataCostDeviceCheckerboard.dataCostCheckerboard1,
				messagesDeviceSet0Checkerboard0.messagesU,
				messagesDeviceSet0Checkerboard0.messagesD,
				messagesDeviceSet0Checkerboard0.messagesL,
				messagesDeviceSet0Checkerboard0.messagesR,
				messagesDeviceSet0Checkerboard1.messagesU,
				messagesDeviceSet0Checkerboard1.messagesD,
				messagesDeviceSet0Checkerboard1.messagesL,
				messagesDeviceSet0Checkerboard1.messagesR);
	} else {
		KernelBpStereoCPU::printDataAndMessageValsToPointKernelCPU<T>(xVal, yVal, currentLevelProperties,
				dataCostDeviceCheckerboard.dataCostCheckerboard0,
				dataCostDeviceCheckerboard.dataCostCheckerboard1,
				messagesDeviceSet1Checkerboard0.messagesU,
				messagesDeviceSet1Checkerboard0.messagesD,
				messagesDeviceSet1Checkerboard0.messagesL,
				messagesDeviceSet1Checkerboard0.messagesR,
				messagesDeviceSet1Checkerboard1.messagesU,
				messagesDeviceSet1Checkerboard1.messagesD,
				messagesDeviceSet1Checkerboard1.messagesL,
				messagesDeviceSet1Checkerboard1.messagesR);
	}
}

//functions directed related to running BP to retrieve the movement between the images

//run the given number of iterations of BP at the current level using the given message values in global device memory
template<typename T, typename U>
void ProcessOptimizedCPUBP<T, U>::runBPAtCurrentLevel(const BPsettings& algSettings,
		const levelProperties& currentLevelProperties,
		const dataCostData<U>& dataCostDeviceCheckerboard,
		const checkerboardMessages<U>& messagesDeviceCheckerboard0,
		const checkerboardMessages<U>& messagesDeviceCheckerboard1)
{
	//at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
	for (int iterationNum = 0; iterationNum < algSettings.numIterations; iterationNum++)
	{
		Checkerboard_Parts checkboardPartUpdate = CHECKERBOARD_PART_1;

		if ((iterationNum % 2) == 0)
		{
			checkboardPartUpdate = CHECKERBOARD_PART_1;
		}
		else
		{
			checkboardPartUpdate = CHECKERBOARD_PART_0;
		}

		KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPU<T>(checkboardPartUpdate, currentLevelProperties,
				dataCostDeviceCheckerboard.dataCostCheckerboard0, dataCostDeviceCheckerboard.dataCostCheckerboard1,
				messagesDeviceCheckerboard0.messagesU, messagesDeviceCheckerboard0.messagesD,
				messagesDeviceCheckerboard0.messagesL, messagesDeviceCheckerboard0.messagesR,
				messagesDeviceCheckerboard1.messagesU, messagesDeviceCheckerboard1.messagesD,
				messagesDeviceCheckerboard1.messagesL, messagesDeviceCheckerboard1.messagesR,
						algSettings.disc_k_bp);
	}
}

//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
template<typename T, typename U>
void ProcessOptimizedCPUBP<T, U>::copyMessageValuesToNextLevelDown(
		const levelProperties& currentLevelProperties,
		const levelProperties& nextlevelProperties,
		const checkerboardMessages<U>& messagesDeviceCheckerboard0CopyFrom,
		const checkerboardMessages<U>& messagesDeviceCheckerboard1CopyFrom,
		const checkerboardMessages<U>& messagesDeviceCheckerboard0CopyTo,
		const checkerboardMessages<U>& messagesDeviceCheckerboard1CopyTo)
{
	//call the kernal to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
	//storing the current message values
	KernelBpStereoCPU::copyPrevLevelToNextLevelBPCheckerboardStereoCPU<
			T>(CHECKERBOARD_PART_0, currentLevelProperties, nextlevelProperties,
			messagesDeviceCheckerboard0CopyFrom.messagesU,
			messagesDeviceCheckerboard0CopyFrom.messagesD,
			messagesDeviceCheckerboard0CopyFrom.messagesL,
			messagesDeviceCheckerboard0CopyFrom.messagesR,
			messagesDeviceCheckerboard1CopyFrom.messagesU,
			messagesDeviceCheckerboard1CopyFrom.messagesD,
			messagesDeviceCheckerboard1CopyFrom.messagesL,
			messagesDeviceCheckerboard1CopyFrom.messagesR,
			messagesDeviceCheckerboard0CopyTo.messagesU,
			messagesDeviceCheckerboard0CopyTo.messagesD,
			messagesDeviceCheckerboard0CopyTo.messagesL,
			messagesDeviceCheckerboard0CopyTo.messagesR,
			messagesDeviceCheckerboard1CopyTo.messagesU,
			messagesDeviceCheckerboard1CopyTo.messagesD,
			messagesDeviceCheckerboard1CopyTo.messagesL,
			messagesDeviceCheckerboard1CopyTo.messagesR);

	KernelBpStereoCPU::copyPrevLevelToNextLevelBPCheckerboardStereoCPU<T>(
			CHECKERBOARD_PART_1, currentLevelProperties, nextlevelProperties,
			messagesDeviceCheckerboard0CopyFrom.messagesU,
			messagesDeviceCheckerboard0CopyFrom.messagesD,
			messagesDeviceCheckerboard0CopyFrom.messagesL,
			messagesDeviceCheckerboard0CopyFrom.messagesR,
			messagesDeviceCheckerboard1CopyFrom.messagesU,
			messagesDeviceCheckerboard1CopyFrom.messagesD,
			messagesDeviceCheckerboard1CopyFrom.messagesL,
			messagesDeviceCheckerboard1CopyFrom.messagesR,
			messagesDeviceCheckerboard0CopyTo.messagesU,
			messagesDeviceCheckerboard0CopyTo.messagesD,
			messagesDeviceCheckerboard0CopyTo.messagesL,
			messagesDeviceCheckerboard0CopyTo.messagesR,
			messagesDeviceCheckerboard1CopyTo.messagesU,
			messagesDeviceCheckerboard1CopyTo.messagesD,
			messagesDeviceCheckerboard1CopyTo.messagesL,
			messagesDeviceCheckerboard1CopyTo.messagesR);
}

//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
template<typename T, typename U>
void ProcessOptimizedCPUBP<T, U>::initializeDataCosts(const BPsettings& algSettings, const levelProperties& currentLevelProperties, float* image1PixelsCompDevice,
		float* image2PixelsCompDevice, const dataCostData<U>& dataCostDeviceCheckerboard)
{
	//initialize the data the the "bottom" of the image pyramid
	KernelBpStereoCPU::initializeBottomLevelDataStereoCPU<T>(currentLevelProperties, image1PixelsCompDevice,
			image2PixelsCompDevice, dataCostDeviceCheckerboard.dataCostCheckerboard0,
			dataCostDeviceCheckerboard.dataCostCheckerboard1, algSettings.lambda_bp, algSettings.data_k_bp);
}

//initialize the message values with no previous message values...all message values are set to 0
template<typename T, typename U>
void ProcessOptimizedCPUBP<T, U>::initializeMessageValsToDefault(
		const levelProperties& currentLevelProperties,
		const checkerboardMessages<U>& messagesDeviceCheckerboard0,
		const checkerboardMessages<U>& messagesDeviceCheckerboard1)
{
	//initialize all the message values for each pixel at each possible movement to the default value in the kernal
	KernelBpStereoCPU::initializeMessageValsToDefaultKernelCPU<T>(currentLevelProperties, messagesDeviceCheckerboard0.messagesU, messagesDeviceCheckerboard0.messagesD, messagesDeviceCheckerboard0.messagesL,
			messagesDeviceCheckerboard0.messagesR, messagesDeviceCheckerboard1.messagesU, messagesDeviceCheckerboard1.messagesD,
			messagesDeviceCheckerboard1.messagesL, messagesDeviceCheckerboard1.messagesR);
}


template<typename T, typename U>
void ProcessOptimizedCPUBP<T, U>::initializeDataCurrentLevel(const levelProperties& currentLevelProperties,
		const levelProperties& prevLevelProperties,
		const dataCostData<U>& dataCostDeviceCheckerboard,
		const dataCostData<U>& dataCostDeviceCheckerboardWriteTo)
{
	size_t offsetNum = 0;

	KernelBpStereoCPU::initializeCurrentLevelDataStereoCPU<T>(
			CHECKERBOARD_PART_0, currentLevelProperties, prevLevelProperties,
			dataCostDeviceCheckerboard.dataCostCheckerboard0,
			dataCostDeviceCheckerboard.dataCostCheckerboard1,
			dataCostDeviceCheckerboardWriteTo.dataCostCheckerboard0,
			((int) offsetNum / sizeof(float)));

	KernelBpStereoCPU::initializeCurrentLevelDataStereoCPU<T>(
			CHECKERBOARD_PART_1, currentLevelProperties, prevLevelProperties,
			dataCostDeviceCheckerboard.dataCostCheckerboard0,
			dataCostDeviceCheckerboard.dataCostCheckerboard1,
			dataCostDeviceCheckerboardWriteTo.dataCostCheckerboard1,
			((int) offsetNum / sizeof(float)));
}

template<typename T, typename U>
void ProcessOptimizedCPUBP<T, U>::retrieveOutputDisparity(
		const Checkerboard_Parts currentCheckerboardSet,
		const levelProperties& levelProperties,
		const dataCostData<U>& dataCostDeviceCheckerboard,
		const checkerboardMessages<U>& messagesDeviceSet0Checkerboard0,
		const checkerboardMessages<U>& messagesDeviceSet0Checkerboard1,
		const checkerboardMessages<U>& messagesDeviceSet1Checkerboard0,
		const checkerboardMessages<U>& messagesDeviceSet1Checkerboard1,
		float* resultingDisparityMapCompDevice)
{
	if (currentCheckerboardSet == 0)
	{
		KernelBpStereoCPU::retrieveOutputDisparityCheckerboardStereoOptimizedCPU<
				T>(levelProperties,
				dataCostDeviceCheckerboard.dataCostCheckerboard0,
				dataCostDeviceCheckerboard.dataCostCheckerboard1,
				messagesDeviceSet0Checkerboard0.messagesU,
				messagesDeviceSet0Checkerboard0.messagesD,
				messagesDeviceSet0Checkerboard0.messagesL,
				messagesDeviceSet0Checkerboard0.messagesR,
				messagesDeviceSet0Checkerboard1.messagesU,
				messagesDeviceSet0Checkerboard1.messagesD,
				messagesDeviceSet0Checkerboard1.messagesL,
				messagesDeviceSet0Checkerboard1.messagesR,
				resultingDisparityMapCompDevice);
	} else {
		KernelBpStereoCPU::retrieveOutputDisparityCheckerboardStereoOptimizedCPU<
				T>(levelProperties,
				dataCostDeviceCheckerboard.dataCostCheckerboard0,
				dataCostDeviceCheckerboard.dataCostCheckerboard1,
				messagesDeviceSet1Checkerboard0.messagesU,
				messagesDeviceSet1Checkerboard0.messagesD,
				messagesDeviceSet1Checkerboard0.messagesL,
				messagesDeviceSet1Checkerboard0.messagesR,
				messagesDeviceSet1Checkerboard1.messagesU,
				messagesDeviceSet1Checkerboard1.messagesD,
				messagesDeviceSet1Checkerboard1.messagesL,
				messagesDeviceSet1Checkerboard1.messagesR,
				resultingDisparityMapCompDevice);
	}
}

#if (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT)

template class ProcessOptimizedCPUBP<float, float*>;

#elif (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_DOUBLE)

template class ProcessOptimizedCPUBP<double, double*>;

#elif (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF)

#ifdef COMPILING_FOR_ARM
template class ProcessOptimizedCPUBP<float16_t, float16_t*>;
#else
template class ProcessOptimizedCPUBP<short, short*>;
#endif //COMPILING_FOR_ARM

#endif //(CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT)
