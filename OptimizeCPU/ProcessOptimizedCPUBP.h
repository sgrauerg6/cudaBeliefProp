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

//include for the "kernal" functions to be run on the CPU
#include "KernelBpStereoCPU.h"

template<typename T, typename U, unsigned int DISP_VALS>
class ProcessOptimizedCPUBP : public ProcessBPOnTargetDevice<T, U, DISP_VALS>
{
public:
		void allocateRawMemoryOnTargetDevice(void** arrayToAllocate, const unsigned long numBytesAllocate) override
		{
			//std::cout << "RUN ALLOC: " << numBytesAllocate << "\n";
			//*arrayToAllocate = malloc(numBytesAllocate);
			//necessary to align for aligned avx load instructions to work as expected
#ifdef _WIN32
			*arrayToAllocate = _aligned_malloc(numBytesAllocate, bp_params::NUM_DATA_ALIGN_WIDTH * sizeof(T));
#else
			*arrayToAllocate = aligned_alloc(bp_params::NUM_DATA_ALIGN_WIDTH * sizeof(T), numBytesAllocate);
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
			U memoryData = static_cast<U>(_aligned_malloc(numData * sizeof(T), bp_params::NUM_DATA_ALIGN_WIDTH * sizeof(T)));
			return memoryData;
#else
			U memoryData = static_cast<U>(std::aligned_alloc(bp_params::NUM_DATA_ALIGN_WIDTH * sizeof(T), numData * sizeof(T)));
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

		void initializeDataCosts(const BPsettings& algSettings, const levelProperties& currentLevelProperties,
				const std::array<float*, 2>& imagesOnTargetDevice, const dataCostData<U>& dataCostDeviceCheckerboard) override;

		void initializeDataCurrentLevel(const levelProperties& currentLevelProperties,
				const levelProperties& prevLevelProperties,
				const dataCostData<U>& dataCostDeviceCheckerboard,
				const dataCostData<U>& dataCostDeviceCheckerboardWriteTo) override;

		void initializeMessageValsToDefault(
				const levelProperties& currentLevelProperties,
				const checkerboardMessages<U>& messagesDevice) override;

		void runBPAtCurrentLevel(const BPsettings& algSettings,
				const levelProperties& currentLevelProperties,
				const dataCostData<U>& dataCostDeviceCheckerboard,
				const checkerboardMessages<U>& messagesDevice) override;

		void copyMessageValuesToNextLevelDown(
				const levelProperties& currentLevelProperties,
				const levelProperties& nextlevelProperties,
				const checkerboardMessages<U>& messagesDeviceCopyFrom,
				const checkerboardMessages<U>& messagesDeviceCopyTo) override;

		float* retrieveOutputDisparity(
				const levelProperties& currentLevelProperties,
				const dataCostData<U>& dataCostDeviceCheckerboard,
				const checkerboardMessages<U>& messagesDevice) override;
};

#endif //RUN_BP_STEREO_HOST_HEADER_CUH
