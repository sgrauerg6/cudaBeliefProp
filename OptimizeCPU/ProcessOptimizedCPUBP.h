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

#include "ParameterFiles/bpStereoParameters.h"
#include "ParameterFiles/bpStructsAndEnums.h"

//include for the kernal functions to be run on the GPU
#include "KernelBpStereoCPU.h"
#include <vector>
#include <algorithm>
#include <chrono>
#include "../BpAndSmoothProcessing/ProcessBPOnTargetDevice.h"
#include <stdlib.h>

template<typename T, typename U>
class ProcessOptimizedCPUBP : public ProcessBPOnTargetDevice<T, U>
{
public:
		void allocateRawMemoryOnTargetDevice(void** arrayToAllocate, unsigned long numBytesAllocate)
		{
			//std::cout << "RUN ALLOC: " << numBytesAllocate << "\n";
			//*arrayToAllocate = malloc(numBytesAllocate);
			//necessary to align for aligned avx load instructions to work as expected
			*arrayToAllocate = aligned_alloc(NUM_DATA_ALIGN_WIDTH * sizeof(T), numBytesAllocate);
		}

		void freeRawMemoryOnTargetDevice(void* arrayToFree)
		{
			free(arrayToFree);
		}

		U allocateMemoryOnTargetDevice(unsigned long numData)
		{
			U memoryData = static_cast<U>(std::aligned_alloc(NUM_DATA_ALIGN_WIDTH * sizeof(T), numData * sizeof(T)));
			return memoryData;
		}

		void freeMemoryOnTargetDevice(U memoryToFree)
		{
			free(memoryToFree);
		}

		void initializeDataCosts(const BPsettings& algSettings, const levelProperties& currentLevelProperties,
				const std::array<float*, 2>& imagesOnTargetDevice, const dataCostData<U>& dataCostDeviceCheckerboard);

		void initializeDataCurrentLevel(const levelProperties& currentLevelProperties,
					const levelProperties& prevLevelProperties,
					const dataCostData<U>& dataCostDeviceCheckerboard,
					const dataCostData<U>& dataCostDeviceCheckerboardWriteTo);

		void initializeMessageValsToDefault(
					const levelProperties& currentLevelProperties,
					const checkerboardMessages<U>& messagesDevice);

		void runBPAtCurrentLevel(const BPsettings& algSettings,
					const levelProperties& currentLevelProperties,
					const dataCostData<U>& dataCostDeviceCheckerboard,
					const checkerboardMessages<U>& messagesDevice);

		void copyMessageValuesToNextLevelDown(
					const levelProperties& currentLevelProperties,
					const levelProperties& nextlevelProperties,
					const checkerboardMessages<U>& messagesDeviceCopyFrom,
					const checkerboardMessages<U>& messagesDeviceCopyTo);

		float* retrieveOutputDisparity(
					const levelProperties& currentLevelProperties,
					const dataCostData<U>& dataCostDeviceCheckerboard,
					const checkerboardMessages<U>& messagesDevice);
};

//if not using AVX-256 or AVX-512, process using float if short data type used
//Processing with short data type has been implemented when using AVX-256
//NOTE: THIS HAS BEEN REMOVED AND JUST PRINTS AN ERROR THAT NOT SUPPORTED
#if (CPU_OPTIMIZATION_SETTING != USE_AVX_256) && (CPU_OPTIMIZATION_SETTING != USE_AVX_512)

#endif


#endif //RUN_BP_STEREO_HOST_HEADER_CUH
