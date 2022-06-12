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

#include <malloc.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <stdlib.h>
#include "../BpAndSmoothProcessing/ProcessBPOnTargetDevice.h"
#include "../ParameterFiles/bpStereoParameters.h"
#include "../ParameterFiles/bpStructsAndEnums.h"
#include "../ParameterFiles/bpRunSettings.h"

//include for the "kernel" functions to be run on the CPU
#include "KernelBpStereoCPU.cpp"
#include "KernelBpStereoCPU.h"

template<typename T, typename U, unsigned int DISP_VALS>
class ProcessOptimizedCPUBP : public ProcessBPOnTargetDevice<T, U, DISP_VALS>
{
public:
		ProcessOptimizedCPUBP(const beliefprop::ParallelParameters& optCPUParams) : optCPUParams_(optCPUParams) { }

		void allocateRawMemoryOnTargetDevice(void** arrayToAllocate, const unsigned long numBytesAllocate) override
		{
			//std::cout << "RUN ALLOC: " << numBytesAllocate << "\n";
			//*arrayToAllocate = malloc(numBytesAllocate);
			//necessary to align for aligned avx load instructions to work as expected
#ifdef _WIN32
			*arrayToAllocate = _aligned_malloc(numBytesAllocate, beliefprop::NUM_DATA_ALIGN_WIDTH * sizeof(T));
#else
			*arrayToAllocate = aligned_alloc(beliefprop::NUM_DATA_ALIGN_WIDTH * sizeof(T), numBytesAllocate);
#endif
		}

		void freeRawMemoryOnTargetDevice(void* arrayToFree) override
		{
#ifdef _WIN32
			_aligned_free(arrayToFree);
#else
			free(arrayToFree);
#endif

		}

		U allocateMemoryOnTargetDevice(const unsigned long numData) override
		{
#ifdef _WIN32
			U memoryData = static_cast<U>(_aligned_malloc(numData * sizeof(T), beliefprop::NUM_DATA_ALIGN_WIDTH * sizeof(T)));
			return memoryData;
#else
			U memoryData = static_cast<U>(std::aligned_alloc(beliefprop::NUM_DATA_ALIGN_WIDTH * sizeof(T), numData * sizeof(T)));
			return memoryData;
#endif
		}

		void freeMemoryOnTargetDevice(U memoryToFree) override
		{
#ifdef _WIN32
			_aligned_free(memoryToFree);
#else
			free(memoryToFree);
#endif
		}

		void initializeDataCosts(const beliefprop::BPsettings& algSettings, const beliefprop::levelProperties& currentLevelProperties,
				const std::array<float*, 2>& imagesOnTargetDevice, const beliefprop::dataCostData<U>& dataCostDeviceCheckerboard) override;

		void initializeDataCurrentLevel(const beliefprop::levelProperties& currentLevelProperties,
				const beliefprop::levelProperties& prevLevelProperties,
				const beliefprop::dataCostData<U>& dataCostDeviceCheckerboard,
				const beliefprop::dataCostData<U>& dataCostDeviceCheckerboardWriteTo,
				const unsigned int bpSettingsNumDispVals) override;

		void initializeMessageValsToDefault(
				const beliefprop::levelProperties& currentLevelProperties,
				const beliefprop::checkerboardMessages<U>& messagesDevice,
				const unsigned int bpSettingsNumDispVals) override;

		void runBPAtCurrentLevel(const beliefprop::BPsettings& algSettings,
				const beliefprop::levelProperties& currentLevelProperties,
				const beliefprop::dataCostData<U>& dataCostDeviceCheckerboard,
				const beliefprop::checkerboardMessages<U>& messagesDevice,
				void* allocatedMemForProcessing) override;

		void copyMessageValuesToNextLevelDown(
				const beliefprop::levelProperties& currentLevelProperties,
				const beliefprop::levelProperties& nextlevelProperties,
				const beliefprop::checkerboardMessages<U>& messagesDeviceCopyFrom,
				const beliefprop::checkerboardMessages<U>& messagesDeviceCopyTo,
				const unsigned int bpSettingsNumDispVals) override;

		float* retrieveOutputDisparity(
				const beliefprop::levelProperties& currentLevelProperties,
				const beliefprop::dataCostData<U>& dataCostDeviceCheckerboard,
				const beliefprop::checkerboardMessages<U>& messagesDevice,
				const unsigned int bpSettingsNumDispVals) override;

private:
	const beliefprop::ParallelParameters& optCPUParams_;
};

//functions definitions related to running BP to retrieve the movement between the images

//run the given number of iterations of BP at the current level using the given message values in global device memory
template<typename T, typename U, unsigned int DISP_VALS>
inline void ProcessOptimizedCPUBP<T, U, DISP_VALS>::runBPAtCurrentLevel(const beliefprop::BPsettings& algSettings,
		const beliefprop::levelProperties& currentLevelProperties,
		const beliefprop::dataCostData<U>& dataCostDeviceCheckerboard,
		const beliefprop::checkerboardMessages<U>& messagesDevice,
		void* allocatedMemForProcessing)
{
	//at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
	for (unsigned int iterationNum = 0; iterationNum < algSettings.numIterations_; iterationNum++)
	{
		beliefprop::Checkerboard_Parts checkboardPartUpdate = ((iterationNum % 2) == 0) ? beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1 : beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0;

		KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPU<T, DISP_VALS>(
				checkboardPartUpdate, currentLevelProperties,
				dataCostDeviceCheckerboard.dataCostCheckerboard0_,
				dataCostDeviceCheckerboard.dataCostCheckerboard1_,
				messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_0], messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_0],
				messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_0], messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_0],
				messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_1], messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_1],
				messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_1], messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_1],
				algSettings.disc_k_bp_, algSettings.numDispVals_, optCPUParams_);
	}
}

//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
template<typename T, typename U, unsigned int DISP_VALS>
inline void ProcessOptimizedCPUBP<T, U, DISP_VALS>::copyMessageValuesToNextLevelDown(
		const beliefprop::levelProperties& currentLevelProperties,
		const beliefprop::levelProperties& nextlevelProperties,
		const beliefprop::checkerboardMessages<U>& messagesDeviceCopyFrom,
		const beliefprop::checkerboardMessages<U>& messagesDeviceCopyTo,
		const unsigned int bpSettingsNumDispVals)
{
	for (const auto& checkerboard_part : {beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0, beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1})
	{
		//call the kernel to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
		//storing the current message values
		KernelBpStereoCPU::copyPrevLevelToNextLevelBPCheckerboardStereoCPU<T, DISP_VALS>(
				checkerboard_part, currentLevelProperties, nextlevelProperties,
				messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_0], messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_0],
				messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_0], messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_0],
				messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_1], messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_1],
				messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_1], messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_1],
				messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_0], messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_0],
				messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_0], messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_0],
				messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_1], messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_1],
				messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_1], messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_1],
				bpSettingsNumDispVals, optCPUParams_);
	}
}

//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
template<typename T, typename U, unsigned int DISP_VALS>
inline void ProcessOptimizedCPUBP<T, U, DISP_VALS>::initializeDataCosts(const beliefprop::BPsettings& algSettings, const beliefprop::levelProperties& currentLevelProperties,
		const std::array<float*, 2>& imagesOnTargetDevice, const beliefprop::dataCostData<U>& dataCostDeviceCheckerboard)
{
	//initialize the data the the "bottom" of the image pyramid
	KernelBpStereoCPU::initializeBottomLevelDataStereoCPU<T, DISP_VALS>(currentLevelProperties, imagesOnTargetDevice[0],
			imagesOnTargetDevice[1], dataCostDeviceCheckerboard.dataCostCheckerboard0_,
			dataCostDeviceCheckerboard.dataCostCheckerboard1_, algSettings.lambda_bp_, algSettings.data_k_bp_,
			algSettings.numDispVals_, optCPUParams_);
}

//initialize the message values with no previous message values...all message values are set to 0
template<typename T, typename U, unsigned int DISP_VALS>
void ProcessOptimizedCPUBP<T, U, DISP_VALS>::initializeMessageValsToDefault(
		const beliefprop::levelProperties& currentLevelProperties,
		const beliefprop::checkerboardMessages<U>& messagesDevice,
		const unsigned int bpSettingsNumDispVals)
{
	//initialize all the message values for each pixel at each possible movement to the default value in the kernel
	KernelBpStereoCPU::initializeMessageValsToDefaultKernelCPU<T, DISP_VALS>(
			currentLevelProperties,
			messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_0], messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_0],
			messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_0], messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_0],
			messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_1], messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_1],
			messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_1], messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_1],
			bpSettingsNumDispVals, optCPUParams_);
}


template<typename T, typename U, unsigned int DISP_VALS>
inline void ProcessOptimizedCPUBP<T, U, DISP_VALS>::initializeDataCurrentLevel(const beliefprop::levelProperties& currentLevelProperties,
		const beliefprop::levelProperties& prevLevelProperties,
		const beliefprop::dataCostData<U>& dataCostDeviceCheckerboard,
		const beliefprop::dataCostData<U>& dataCostDeviceCheckerboardWriteTo,
		const unsigned int bpSettingsNumDispVals)
{
	size_t offsetNum = 0;

	for (const auto& checkerboardAndDataCost : { std::make_pair(
			beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0,
			dataCostDeviceCheckerboardWriteTo.dataCostCheckerboard0_),
			std::make_pair(beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1,
					dataCostDeviceCheckerboardWriteTo.dataCostCheckerboard1_) })
	{
		KernelBpStereoCPU::initializeCurrentLevelDataStereoCPU<T, DISP_VALS>(
				checkerboardAndDataCost.first, currentLevelProperties, prevLevelProperties,
					dataCostDeviceCheckerboard.dataCostCheckerboard0_,
					dataCostDeviceCheckerboard.dataCostCheckerboard1_,
					checkerboardAndDataCost.second,
					((int) offsetNum / sizeof(float)),
					bpSettingsNumDispVals, optCPUParams_);
	}
}

template<typename T, typename U, unsigned int DISP_VALS>
inline float* ProcessOptimizedCPUBP<T, U, DISP_VALS>::retrieveOutputDisparity(
		const beliefprop::levelProperties& currentLevelProperties,
		const beliefprop::dataCostData<U>& dataCostDeviceCheckerboard,
		const beliefprop::checkerboardMessages<U>& messagesDevice,
		const unsigned int bpSettingsNumDispVals)
{
	float* resultingDisparityMapCompDevice = new float[currentLevelProperties.widthLevel_ * currentLevelProperties.heightLevel_];

	KernelBpStereoCPU::retrieveOutputDisparityCheckerboardStereoOptimizedCPU<T, DISP_VALS>(
			currentLevelProperties,
			dataCostDeviceCheckerboard.dataCostCheckerboard0_,
			dataCostDeviceCheckerboard.dataCostCheckerboard1_,
			messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_0], messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_0],
			messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_0], messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_0],
			messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_1], messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_1],
			messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_1], messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_1],
			resultingDisparityMapCompDevice, bpSettingsNumDispVals, optCPUParams_);

	return resultingDisparityMapCompDevice;
}

#endif //RUN_BP_STEREO_HOST_HEADER_CUH
