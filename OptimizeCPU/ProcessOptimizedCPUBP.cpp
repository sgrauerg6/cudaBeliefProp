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

//functions directed related to running BP to retrieve the movement between the images

//run the given number of iterations of BP at the current level using the given message values in global device memory
template<typename T, typename U>
void ProcessOptimizedCPUBP<T, U>::runBPAtCurrentLevel(const BPsettings& algSettings,
		const levelProperties& currentLevelProperties,
		const dataCostData<U>& dataCostDeviceCheckerboard,
		const checkerboardMessages<U>& messagesDevice)
{
	//at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
	for (int iterationNum = 0; iterationNum < algSettings.numIterations; iterationNum++)
	{
		Checkerboard_Parts checkboardPartUpdate = ((iterationNum % 2) == 0) ? CHECKERBOARD_PART_1 : CHECKERBOARD_PART_0;

		KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPU<T>(
				checkboardPartUpdate, currentLevelProperties,
				dataCostDeviceCheckerboard.dataCostCheckerboard0,
				dataCostDeviceCheckerboard.dataCostCheckerboard1,
				messagesDevice.messagesU_Checkerboard0,
				messagesDevice.messagesD_Checkerboard0,
				messagesDevice.messagesL_Checkerboard0,
				messagesDevice.messagesR_Checkerboard0,
				messagesDevice.messagesU_Checkerboard1,
				messagesDevice.messagesD_Checkerboard1,
				messagesDevice.messagesL_Checkerboard1,
				messagesDevice.messagesR_Checkerboard1, algSettings.disc_k_bp);
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
		const checkerboardMessages<U>& messagesDeviceCopyFrom,
		const checkerboardMessages<U>& messagesDeviceCopyTo)
{
	for (const auto& checkerboard_part : {CHECKERBOARD_PART_0, CHECKERBOARD_PART_1})
	{
		//call the kernal to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
		//storing the current message values
		KernelBpStereoCPU::copyPrevLevelToNextLevelBPCheckerboardStereoCPU<T>(
				checkerboard_part, currentLevelProperties, nextlevelProperties,
				messagesDeviceCopyFrom.messagesU_Checkerboard0,
				messagesDeviceCopyFrom.messagesD_Checkerboard0,
				messagesDeviceCopyFrom.messagesL_Checkerboard0,
				messagesDeviceCopyFrom.messagesR_Checkerboard0,
				messagesDeviceCopyFrom.messagesU_Checkerboard1,
				messagesDeviceCopyFrom.messagesD_Checkerboard1,
				messagesDeviceCopyFrom.messagesL_Checkerboard1,
				messagesDeviceCopyFrom.messagesR_Checkerboard1,
				messagesDeviceCopyTo.messagesU_Checkerboard0,
				messagesDeviceCopyTo.messagesD_Checkerboard0,
				messagesDeviceCopyTo.messagesL_Checkerboard0,
				messagesDeviceCopyTo.messagesR_Checkerboard0,
				messagesDeviceCopyTo.messagesU_Checkerboard1,
				messagesDeviceCopyTo.messagesD_Checkerboard1,
				messagesDeviceCopyTo.messagesL_Checkerboard1,
				messagesDeviceCopyTo.messagesR_Checkerboard1);
	}
}

//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
template<typename T, typename U>
void ProcessOptimizedCPUBP<T, U>::initializeDataCosts(const BPsettings& algSettings, const levelProperties& currentLevelProperties,
		const std::array<float*, 2>& imagesOnTargetDevice, const dataCostData<U>& dataCostDeviceCheckerboard)
{
	//initialize the data the the "bottom" of the image pyramid
	KernelBpStereoCPU::initializeBottomLevelDataStereoCPU<T>(currentLevelProperties, imagesOnTargetDevice[0],
			imagesOnTargetDevice[1], dataCostDeviceCheckerboard.dataCostCheckerboard0,
			dataCostDeviceCheckerboard.dataCostCheckerboard1, algSettings.lambda_bp, algSettings.data_k_bp);
}

//initialize the message values with no previous message values...all message values are set to 0
template<typename T, typename U>
void ProcessOptimizedCPUBP<T, U>::initializeMessageValsToDefault(
		const levelProperties& currentLevelProperties,
		const checkerboardMessages<U>& messagesDevice)
{
	//initialize all the message values for each pixel at each possible movement to the default value in the kernal
	KernelBpStereoCPU::initializeMessageValsToDefaultKernelCPU<T>(
			currentLevelProperties,
			messagesDevice.messagesU_Checkerboard0,
			messagesDevice.messagesD_Checkerboard0,
			messagesDevice.messagesL_Checkerboard0,
			messagesDevice.messagesR_Checkerboard0,
			messagesDevice.messagesU_Checkerboard1,
			messagesDevice.messagesD_Checkerboard1,
			messagesDevice.messagesL_Checkerboard1,
			messagesDevice.messagesR_Checkerboard1);
}


template<typename T, typename U>
void ProcessOptimizedCPUBP<T, U>::initializeDataCurrentLevel(const levelProperties& currentLevelProperties,
		const levelProperties& prevLevelProperties,
		const dataCostData<U>& dataCostDeviceCheckerboard,
		const dataCostData<U>& dataCostDeviceCheckerboardWriteTo)
{
	size_t offsetNum = 0;

	for (const auto& checkerboardAndDataCost : { std::make_pair(
			CHECKERBOARD_PART_0,
			dataCostDeviceCheckerboardWriteTo.dataCostCheckerboard0),
			std::make_pair(CHECKERBOARD_PART_1,
					dataCostDeviceCheckerboardWriteTo.dataCostCheckerboard1) })
	{
		KernelBpStereoCPU::initializeCurrentLevelDataStereoCPU<T>(
				checkerboardAndDataCost.first, currentLevelProperties, prevLevelProperties,
					dataCostDeviceCheckerboard.dataCostCheckerboard0,
					dataCostDeviceCheckerboard.dataCostCheckerboard1,
					checkerboardAndDataCost.second,
					((int) offsetNum / sizeof(float)));
	}
}

template<typename T, typename U>
float* ProcessOptimizedCPUBP<T, U>::retrieveOutputDisparity(
		const levelProperties& currentLevelProperties,
		const dataCostData<U>& dataCostDeviceCheckerboard,
		const checkerboardMessages<U>& messagesDevice)
{
	float* resultingDisparityMapCompDevice = new float[currentLevelProperties.widthLevel * currentLevelProperties.heightLevel];

	KernelBpStereoCPU::retrieveOutputDisparityCheckerboardStereoOptimizedCPU<T>(
			currentLevelProperties,
			dataCostDeviceCheckerboard.dataCostCheckerboard0,
			dataCostDeviceCheckerboard.dataCostCheckerboard1,
			messagesDevice.messagesU_Checkerboard0,
			messagesDevice.messagesD_Checkerboard0,
			messagesDevice.messagesL_Checkerboard0,
			messagesDevice.messagesR_Checkerboard0,
			messagesDevice.messagesU_Checkerboard1,
			messagesDevice.messagesD_Checkerboard1,
			messagesDevice.messagesL_Checkerboard1,
			messagesDevice.messagesR_Checkerboard1,
			resultingDisparityMapCompDevice);

	return resultingDisparityMapCompDevice;
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
