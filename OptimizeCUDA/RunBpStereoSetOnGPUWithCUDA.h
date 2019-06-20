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

//Declares the methods to run Stereo BP on a series of images

#ifndef RUN_BP_STEREO_STEREO_SET_ON_GPU_WITH_CUDA_H
#define RUN_BP_STEREO_STEREO_SET_ON_GPU_WITH_CUDA_H

#include "bpStereoCudaParameters.h"
#include "RunBpStereoSet.h"
#include "SmoothImageCUDA.h"
#include <cuda_runtime.h>

//#include "runBpStereoHost.h"

class RunBpStereoSetOnGPUWithCUDA : public RunBpStereoSet
{
public:

	void allocateDataOnCompDevice(void** arrayToAllocate, int numBytes)
	{
		printf("ALLOC_GPU\n");
		//allocate the space for the disparity map estimation
		cudaMalloc((void **) arrayToAllocate, numBytes);
	}

	void freeDataOnCompDevice(void** arrayToFree)
	{
		printf("FREE_GPU\n");
		cudaFree(*arrayToFree);
	}

	void transferDataFromCompDeviceToHost(void* destArray, void* inArray, int numBytesTransfer)
	{
		printf("TRANSFER_GPU\n");
		cudaMemcpy(destArray,
				inArray,
				numBytesTransfer,
				cudaMemcpyDeviceToHost);
	}

	float operator()(const char* refImagePath, const char* testImagePath,
				BPsettings algSettings,	const char* saveDisparityMapImagePath, FILE* resultsFile, SmoothImage* smoothImage = nullptr, ProcessBPOnTarget<beliefPropProcessingDataType>* runBpStereo = nullptr);
/*	{
		SmoothImageCUDA smoothImageCUDA;
		ProcessCUDABP<beliefPropProcessingDataType> processImageCUDA;
		return RunBpStereoSet::operator ()(refImagePath, testImagePath, algSettings, saveDisparityMapImagePath, resultsFile, &smoothImageCUDA, &processImageCUDA);
	}*/

	//run the disparity map estimation BP on a series of stereo images and save the results between each set of images if desired
	//float operator()(const char* refImagePath, const char* testImagePath,
	//		BPsettings algSettings,	const char* saveDisparityMapImagePath, FILE* resultsFile, SmoothImage* smoothImage = nullptr, ProcessBPOnTarget<beliefPropProcessingDataType>* runBpStereo = nullptr);
};
#endif //RUN_BP_STEREO_IMAGE_SERIES_HEADER_CUH
