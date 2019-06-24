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


template<typename T>
void ProcessOptimizedCPUBP<T>::printDataAndMessageValsAtPoint(int xVal, int yVal, levelProperties& levelPropertes,
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
		T* messageRDeviceSet1Checkerboard2, int currentCheckerboardSet)
{
	if (currentCheckerboardSet == 0) {
		KernelBpStereoCPU::printDataAndMessageValsAtPointKernelCPU<T>(xVal, yVal, levelPropertes,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messageUDeviceSet0Checkerboard1,
				messageDDeviceSet0Checkerboard1,
				messageLDeviceSet0Checkerboard1,
				messageRDeviceSet0Checkerboard1,
				messageUDeviceSet0Checkerboard2,
				messageDDeviceSet0Checkerboard2,
				messageLDeviceSet0Checkerboard2,
				messageRDeviceSet0Checkerboard2);
	} else {
		KernelBpStereoCPU::printDataAndMessageValsAtPointKernelCPU<T>(xVal, yVal, levelPropertes,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messageUDeviceSet1Checkerboard1,
				messageDDeviceSet1Checkerboard1,
				messageLDeviceSet1Checkerboard1,
				messageRDeviceSet1Checkerboard1,
				messageUDeviceSet1Checkerboard2,
				messageDDeviceSet1Checkerboard2,
				messageLDeviceSet1Checkerboard2,
				messageRDeviceSet1Checkerboard2);
	}
}


template<typename T>
void ProcessOptimizedCPUBP<T>::printDataAndMessageValsToPoint(int xVal, int yVal, levelProperties& levelPropertes,
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
		T* messageRDeviceSet1Checkerboard2, int currentCheckerboardSet)
{
	if (currentCheckerboardSet == 0) {
		KernelBpStereoCPU::printDataAndMessageValsToPointKernelCPU<T>(xVal, yVal, levelPropertes,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messageUDeviceSet0Checkerboard1,
				messageDDeviceSet0Checkerboard1,
				messageLDeviceSet0Checkerboard1,
				messageRDeviceSet0Checkerboard1,
				messageUDeviceSet0Checkerboard2,
				messageDDeviceSet0Checkerboard2,
				messageLDeviceSet0Checkerboard2,
				messageRDeviceSet0Checkerboard2);
	} else {
		KernelBpStereoCPU::printDataAndMessageValsToPointKernelCPU<T>(xVal, yVal, levelPropertes,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messageUDeviceSet1Checkerboard1,
				messageDDeviceSet1Checkerboard1,
				messageLDeviceSet1Checkerboard1,
				messageRDeviceSet1Checkerboard1,
				messageUDeviceSet1Checkerboard2,
				messageDDeviceSet1Checkerboard2,
				messageLDeviceSet1Checkerboard2,
				messageRDeviceSet1Checkerboard2);
	}
}

//functions directed related to running BP to retrieve the movement between the images

//run the given number of iterations of BP at the current level using the given message values in global device memory
template<typename T>
void ProcessOptimizedCPUBP<T>::runBPAtCurrentLevel(BPsettings& algSettings,
		levelProperties& currentLevelProperties,
		T* dataCostDeviceCurrentLevelCheckerboard1,
		T* dataCostDeviceCurrentLevelCheckerboard2,
		T* messageUDeviceCheckerboard1,
		T* messageDDeviceCheckerboard1,
		T* messageLDeviceCheckerboard1,
		T* messageRDeviceCheckerboard1,
		T* messageUDeviceCheckerboard2,
		T* messageDDeviceCheckerboard2,
		T* messageLDeviceCheckerboard2,
		T* messageRDeviceCheckerboard2)
{
	//at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
	for (int iterationNum = 0; iterationNum < algSettings.numIterations; iterationNum++)
	{
		int checkboardPartUpdate = CHECKERBOARD_PART_2;

		if ((iterationNum % 2) == 0)
		{
			checkboardPartUpdate = CHECKERBOARD_PART_2;
		}
		else
		{
			checkboardPartUpdate = CHECKERBOARD_PART_1;
		}

		KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPU<T>(checkboardPartUpdate, currentLevelProperties,
				dataCostDeviceCurrentLevelCheckerboard1, dataCostDeviceCurrentLevelCheckerboard2,
						messageUDeviceCheckerboard1, messageDDeviceCheckerboard1,
						messageLDeviceCheckerboard1, messageRDeviceCheckerboard1,
						messageUDeviceCheckerboard2, messageDDeviceCheckerboard2,
						messageLDeviceCheckerboard2, messageRDeviceCheckerboard2,
						algSettings.disc_k_bp);
	}
}

//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
template<typename T>
void ProcessOptimizedCPUBP<T>::copyMessageValuesToNextLevelDown(
		levelProperties& currentLevelPropertes,
		levelProperties& nextLevelPropertes,
		T* messageUDeviceCheckerboard1CopyFrom,
		T* messageDDeviceCheckerboard1CopyFrom,
		T* messageLDeviceCheckerboard1CopyFrom,
		T* messageRDeviceCheckerboard1CopyFrom,
		T* messageUDeviceCheckerboard2CopyFrom,
		T* messageDDeviceCheckerboard2CopyFrom,
		T* messageLDeviceCheckerboard2CopyFrom,
		T* messageRDeviceCheckerboard2CopyFrom,
		T* messageUDeviceCheckerboard1CopyTo,
		T* messageDDeviceCheckerboard1CopyTo,
		T* messageLDeviceCheckerboard1CopyTo,
		T* messageRDeviceCheckerboard1CopyTo,
		T* messageUDeviceCheckerboard2CopyTo,
		T* messageDDeviceCheckerboard2CopyTo,
		T* messageLDeviceCheckerboard2CopyTo,
		T* messageRDeviceCheckerboard2CopyTo)
{
	//call the kernal to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
	//storing the current message values
	KernelBpStereoCPU::copyPrevLevelToNextLevelBPCheckerboardStereoNoTexturesCPU<
			T>(CHECKERBOARD_PART_1, currentLevelPropertes, nextLevelPropertes,
			messageUDeviceCheckerboard1CopyFrom,
			messageDDeviceCheckerboard1CopyFrom,
			messageLDeviceCheckerboard1CopyFrom,
			messageRDeviceCheckerboard1CopyFrom,
			messageUDeviceCheckerboard2CopyFrom,
			messageDDeviceCheckerboard2CopyFrom,
			messageLDeviceCheckerboard2CopyFrom,
			messageRDeviceCheckerboard2CopyFrom,
			messageUDeviceCheckerboard1CopyTo,
			messageDDeviceCheckerboard1CopyTo,
			messageLDeviceCheckerboard1CopyTo,
			messageRDeviceCheckerboard1CopyTo,
			messageUDeviceCheckerboard2CopyTo,
			messageDDeviceCheckerboard2CopyTo,
			messageLDeviceCheckerboard2CopyTo,
			messageRDeviceCheckerboard2CopyTo);

	KernelBpStereoCPU::copyPrevLevelToNextLevelBPCheckerboardStereoNoTexturesCPU<
			T>(CHECKERBOARD_PART_2, currentLevelPropertes, nextLevelPropertes,
			messageUDeviceCheckerboard1CopyFrom,
			messageDDeviceCheckerboard1CopyFrom,
			messageLDeviceCheckerboard1CopyFrom,
			messageRDeviceCheckerboard1CopyFrom,
			messageUDeviceCheckerboard2CopyFrom,
			messageDDeviceCheckerboard2CopyFrom,
			messageLDeviceCheckerboard2CopyFrom,
			messageRDeviceCheckerboard2CopyFrom,
			messageUDeviceCheckerboard1CopyTo,
			messageDDeviceCheckerboard1CopyTo,
			messageLDeviceCheckerboard1CopyTo,
			messageRDeviceCheckerboard1CopyTo,
			messageUDeviceCheckerboard2CopyTo,
			messageDDeviceCheckerboard2CopyTo,
			messageLDeviceCheckerboard2CopyTo,
			messageRDeviceCheckerboard2CopyTo);
}

//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
template<typename T>
void ProcessOptimizedCPUBP<T>::initializeDataCosts(BPsettings& algSettings, levelProperties& currentLevelProperties, float* image1PixelsCompDevice,
		float* image2PixelsCompDevice, T* dataCostDeviceCheckerboard1,
		T* dataCostDeviceCheckerboard2)
{
	//initialize the data the the "bottom" of the image pyramid
	KernelBpStereoCPU::initializeBottomLevelDataStereoCPU<T>(currentLevelProperties, image1PixelsCompDevice,
			image2PixelsCompDevice, dataCostDeviceCheckerboard1,
			dataCostDeviceCheckerboard2, algSettings.lambda_bp, algSettings.data_k_bp);
}

//initialize the message values with no previous message values...all message values are set to 0
template<typename T>
void ProcessOptimizedCPUBP<T>::initializeMessageValsToDefault(
		levelProperties& currentLevelPropertes,
		T* messageUDeviceCheckerboard1,
		T* messageDDeviceCheckerboard1,
		T* messageLDeviceCheckerboard1,
		T* messageRDeviceCheckerboard1,
		T* messageUDeviceCheckerboard2,
		T* messageDDeviceCheckerboard2,
		T* messageLDeviceCheckerboard2,
		T* messageRDeviceCheckerboard2)
{
	//initialize all the message values for each pixel at each possible movement to the default value in the kernal
	KernelBpStereoCPU::initializeMessageValsToDefaultKernelCPU<T>(currentLevelPropertes, messageUDeviceCheckerboard1, messageDDeviceCheckerboard1, messageLDeviceCheckerboard1,
												messageRDeviceCheckerboard1, messageUDeviceCheckerboard2, messageDDeviceCheckerboard2,
												messageLDeviceCheckerboard2, messageRDeviceCheckerboard2);
}


template<typename T>
void ProcessOptimizedCPUBP<T>::initializeDataCurrentLevel(levelProperties& currentLevelPropertes,
		levelProperties& prevLevelProperties,
		T* dataCostStereoCheckerboard1,
		T* dataCostStereoCheckerboard2,
		T* dataCostDeviceToWriteToCheckerboard1,
		T* dataCostDeviceToWriteToCheckerboard2)
{
	size_t offsetNum = 0;

	KernelBpStereoCPU::initializeCurrentLevelDataStereoNoTexturesCPU<T>(CHECKERBOARD_PART_1, currentLevelPropertes, prevLevelProperties,
			dataCostStereoCheckerboard1,
			dataCostStereoCheckerboard2,
			dataCostDeviceToWriteToCheckerboard1,
			((int) offsetNum / sizeof(float)));

	KernelBpStereoCPU::initializeCurrentLevelDataStereoNoTexturesCPU<T>(CHECKERBOARD_PART_2, currentLevelPropertes, prevLevelProperties,
			dataCostStereoCheckerboard1,
			dataCostStereoCheckerboard2,
			dataCostDeviceToWriteToCheckerboard2,
			((int) offsetNum / sizeof(float)));
}

template<typename T>
void ProcessOptimizedCPUBP<T>::retrieveOutputDisparity(
		int currentCheckerboardSet,
		levelProperties& levelPropertes,
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
		T* messageRDeviceSet1Checkerboard2,
		float* resultingDisparityMapCompDevice)
{
	if (currentCheckerboardSet == 0)
	{
		KernelBpStereoCPU::retrieveOutputDisparityCheckerboardStereoOptimizedCPU<T>(levelPropertes,
						dataCostDeviceCurrentLevelCheckerboard1,
						dataCostDeviceCurrentLevelCheckerboard2,
						messageUDeviceSet0Checkerboard1,
						messageDDeviceSet0Checkerboard1,
						messageLDeviceSet0Checkerboard1,
						messageRDeviceSet0Checkerboard1,
						messageUDeviceSet0Checkerboard2,
						messageDDeviceSet0Checkerboard2,
						messageLDeviceSet0Checkerboard2,
						messageRDeviceSet0Checkerboard2, resultingDisparityMapCompDevice);
	}
	else
	{
		KernelBpStereoCPU::retrieveOutputDisparityCheckerboardStereoOptimizedCPU<T>(levelPropertes,
						dataCostDeviceCurrentLevelCheckerboard1,
						dataCostDeviceCurrentLevelCheckerboard2,
						messageUDeviceSet1Checkerboard1,
						messageDDeviceSet1Checkerboard1,
						messageLDeviceSet1Checkerboard1,
						messageRDeviceSet1Checkerboard1,
						messageUDeviceSet1Checkerboard2,
						messageDDeviceSet1Checkerboard2,
						messageLDeviceSet1Checkerboard2,
						messageRDeviceSet1Checkerboard2, resultingDisparityMapCompDevice);
	}
}

#if (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT)

template class ProcessOptimizedCPUBP<float>;

#elif (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_DOUBLE)

template class ProcessOptimizedCPUBP<double>;

#elif (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF)

#ifdef COMPILING_FOR_ARM
template class ProcessOptimizedCPUBP<float16_t>;
#else
template class ProcessOptimizedCPUBP<short>;
#endif //COMPILING_FOR_ARM

#endif //(CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT)
