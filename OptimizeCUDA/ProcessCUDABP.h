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

#include "../ParameterFiles/bpStereoCudaParameters.h"

//include for the kernel functions to be run on the GPU
#include <cuda_runtime.h>
#include "../BpAndSmoothProcessing/ProcessBPOnTargetDevice.h"
#include <cuda_fp16.h>

template<typename T, typename U, unsigned int DISP_VALS>
class ProcessCUDABP : public ProcessBPOnTargetDevice<T, U, DISP_VALS>
{
public:
	ProcessCUDABP(const beliefprop::ParallelParameters& cudaParams) : cudaParams_(cudaParams) { }

	U allocateMemoryOnTargetDevice(const unsigned long numData) override
	{
		U arrayToAllocate;
		cudaMalloc((void**)&arrayToAllocate, numData*sizeof(T));
		return arrayToAllocate;
	}

	void freeMemoryOnTargetDevice(U memoryToFree) override {
		cudaFree(memoryToFree);
	}

	//initialize the data cost at each pixel for each disparity value
	void initializeDataCosts(const beliefprop::BPsettings& algSettings, const beliefprop::levelProperties& currentLevelProperties,
			const std::array<float*, 2>& imagesOnTargetDevice, const beliefprop::dataCostData<U>& dataCostDeviceCheckerboard) override;

	void initializeDataCurrentLevel(const beliefprop::levelProperties& currentLevelProperties,
			const beliefprop::levelProperties& prevLevelProperties,
			const beliefprop::dataCostData<U>& dataCostDeviceCheckerboard,
			const beliefprop::dataCostData<U>& dataCostDeviceCheckerboardWriteTo,
			const unsigned int bpSettingsNumDispVals) override;

	//initialize the message values for every pixel at every disparity to DEFAULT_INITIAL_MESSAGE_VAL (value is 0.0f unless changed)
	void initializeMessageValsToDefault(
			const beliefprop::levelProperties& currentLevelProperties,
			const beliefprop::checkerboardMessages<U>& messagesDevice,
			const unsigned int bpSettingsNumDispVals) override;

	//run the given number of iterations of BP at the current level using the given message values in global device memory
	void runBPAtCurrentLevel(const beliefprop::BPsettings& algSettings,
			const beliefprop::levelProperties& currentLevelProperties,
			const beliefprop::dataCostData<U>& dataCostDeviceCheckerboard,
			const beliefprop::checkerboardMessages<U>& messagesDevice,
			void* allocatedMemForProcessing) override;

	//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
	//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
	//in the next level down
	//need two different "sets" of message values to avoid read-write conflicts
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
	const beliefprop::ParallelParameters& cudaParams_;
};

#endif //RUN_BP_STEREO_HOST_HEADER_CUH
