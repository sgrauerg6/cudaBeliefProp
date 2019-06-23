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
#include "ProcessBPOnTargetDevice.h"
#include "ProcessCUDABP.h"
#include <cuda_runtime.h>

#if ((CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF) || (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO))
#include <cuda_fp16.h>
#endif

class RunBpStereoSetCUDMemoryManagement : public RunBpStereoSetMemoryManagement
{
public:

	void allocateDataOnCompDevice(void** arrayToAllocate, int numBytes)
	{
		//printf("ALLOC_GPU\n");
		//allocate the space for the disparity map estimation
		cudaMalloc((void **) arrayToAllocate, numBytes);
	}

	void freeDataOnCompDevice(void** arrayToFree)
	{
		//printf("FREE_GPU\n");
		cudaFree(*arrayToFree);
	}

	void transferDataFromCompDeviceToHost(void* destArray, void* inArray, int numBytesTransfer)
	{
		//printf("TRANSFER_GPU\n");
		cudaMemcpy(destArray,
				inArray,
				numBytesTransfer,
				cudaMemcpyDeviceToHost);
	}

	void transferDataFromCompHostToDevice(void* destArray, void* inArray, int numBytesTransfer)
	{
		//printf("TRANSFER_GPU\n");
			cudaMemcpy(destArray,
					inArray,
					numBytesTransfer,
					cudaMemcpyHostToDevice);
	}
};

template <typename T>
class RunBpStereoSetOnGPUWithCUDA : public RunBpStereoSet<T>
{
public:
	//run the disparity map estimation BP on a set of stereo images and save the results between each set of images
	float operator()(const char* refImagePath, const char* testImagePath,
				BPsettings algSettings,	const char* saveDisparityMapImagePath, FILE* resultsFile, SmoothImage* smoothImage = nullptr, ProcessBPOnTargetDevice<T>* runBpStereo = nullptr, RunBpStereoSetMemoryManagement* runBPMemoryMangement = nullptr)
	{
		SmoothImageCUDA smoothImageCUDA;
		ProcessCUDABP<T> processImageCUDA;
		RunBpStereoSet<T> runBPCUDA;
		RunBpStereoSetCUDMemoryManagement runBPCUDAMemoryManagement;
		fprintf(resultsFile, "CURRENT RUN: GPU WITH CUDA\n");
		return runBPCUDA(refImagePath, testImagePath, algSettings, saveDisparityMapImagePath, resultsFile, &smoothImageCUDA, &processImageCUDA, &runBPCUDAMemoryManagement);
	}
};

template<>
class RunBpStereoSetOnGPUWithCUDA<short> : public RunBpStereoSet<short>
{
public:

	void allocateDataOnCompDevice(void** arrayToAllocate, int numBytes)
	{
		//printf("ALLOC_GPU\n");
		//allocate the space for the disparity map estimation
		cudaMalloc((void **) arrayToAllocate, numBytes);
	}

	void freeDataOnCompDevice(void** arrayToFree)
	{
		//printf("FREE_GPU\n");
		cudaFree(*arrayToFree);
	}

	void transferDataFromCompDeviceToHost(void* destArray, void* inArray, int numBytesTransfer)
	{
		//printf("TRANSFER_GPU\n");
		cudaMemcpy(destArray,
					inArray,
					numBytesTransfer,
					cudaMemcpyDeviceToHost);
	}

	void transferDataFromCompHostToDevice(void* destArray, void* inArray, int numBytesTransfer)
	{
		printf("TRANSFER_GPU\n");
		cudaMemcpy(destArray,
					inArray,
					numBytesTransfer,
					cudaMemcpyHostToDevice);
	}

	//if type is specified as short, process as half on GPU
	//note that half is considered a data type for 16-bit floats in CUDA
	float operator()(const char* refImagePath, const char* testImagePath,
			BPsettings algSettings,	const char* saveDisparityMapImagePath, FILE* resultsFile, SmoothImage* smoothImage = nullptr, ProcessBPOnTargetDevice<short>* runBpStereo = nullptr, RunBpStereoSetMemoryManagement* runBPMemoryMangement = nullptr)
	{

#if CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF

		//printf("Processing as half on GPU\n");
		RunBpStereoSetOnGPUWithCUDA<half> runCUDABpStereoSet;
		ProcessCUDABP<half> runCUDABPHalfPrecision;
		return runCUDABpStereoSet(refImagePath,
				testImagePath,
				algSettings,
				saveDisparityMapImagePath,
				resultsFile,
				smoothImage,
				&runCUDABPHalfPrecision,
				runBPMemoryMangement);

#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO

		//printf("Processing as half2 on GPU\n");
		RunBpStereoSetOnGPUWithCUDA<half2> runCUDABpStereoSet;
		ProcessCUDABP<half2> runCUDABPHalfTwoDataType;
		return runCUDABpStereoSet(refImagePath, testImagePath, algSettings, saveDisparityMapImagePath, resultsFile, smoothImage, &runCUDABPHalfTwoDataType, runBPMemoryMangement);

#else

		printf("ERROR IN DATA TYPE\n");
		return 0.0;

#endif

	}
};

//float16_t data type used for arm (rather than float)
#ifdef COMPILING_FOR_ARM

template<>
class RunBpStereoSetOnGPUWithCUDA<float16_t> : public RunBpStereoSet<float16_t>
{
public:

	void allocateDataOnCompDevice(void** arrayToAllocate, int numBytes)
	{
		//printf("ALLOC_GPU\n");
		//allocate the space for the disparity map estimation
		cudaMalloc((void **) arrayToAllocate, numBytes);
	}

	void freeDataOnCompDevice(void** arrayToFree)
	{
		//printf("FREE_GPU\n");
		cudaFree(*arrayToFree);
	}

	void transferDataFromCompDeviceToHost(void* destArray, void* inArray, int numBytesTransfer)
	{
		//printf("TRANSFER_GPU\n");
		cudaMemcpy(destArray,
					inArray,
					numBytesTransfer,
					cudaMemcpyDeviceToHost);
	}

	void transferDataFromCompHostToDevice(void* destArray, void* inArray, int numBytesTransfer)
	{
		printf("TRANSFER_GPU\n");
		cudaMemcpy(destArray,
					inArray,
					numBytesTransfer,
					cudaMemcpyHostToDevice);
	}

	//if type is specified as short, process as half on GPU
	//note that half is considered a data type for 16-bit floats in CUDA
	float operator()(const char* refImagePath, const char* testImagePath,
			BPsettings algSettings,	const char* saveDisparityMapImagePath, FILE* resultsFile, SmoothImage* smoothImage = nullptr, ProcessBPOnTargetDevice<short>* runBpStereo = nullptr, RunBpStereoSetMemoryManagement* runBPMemoryMangement = nullptr)
	{

#if CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF

		//printf("Processing as half on GPU\n");
		RunBpStereoSetOnGPUWithCUDA<half> runCUDABpStereoSet;
		ProcessCUDABP<half> runCUDABPHalfPrecision;
		return runCUDABpStereoSet(refImagePath,
				testImagePath,
				algSettings,
				saveDisparityMapImagePath,
				resultsFile,
				smoothImage,
				&runCUDABPHalfPrecision,
				runBPMemoryMangement);

#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO

		//printf("Processing as half2 on GPU\n");
		RunBpStereoSetOnGPUWithCUDA<half2> runCUDABpStereoSet;
		ProcessCUDABP<half2> runCUDABPHalfTwoDataType;
		return runCUDABpStereoSet(refImagePath, testImagePath, algSettings, saveDisparityMapImagePath, resultsFile, smoothImage, &runCUDABPHalfTwoDataType, runBPMemoryMangement);

#else

		printf("ERROR IN DATA TYPE\n");
		return 0.0;

#endif

	}
};

#endif

#endif //RUN_BP_STEREO_IMAGE_SERIES_HEADER_CUH
