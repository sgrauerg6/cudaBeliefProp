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

#if ((CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF) || (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO))
#include <cuda_fp16.h>
#endif

#include <cuda_runtime.h>
#include "ProcessCUDABP.h"
#include <iostream>
#include <memory>
#include "ParameterFiles/bpStereoCudaParameters.h"
#include "../BpAndSmoothProcessing/RunBpStereoSet.h"
#include "SmoothImageCUDA.h"
#include "../BpAndSmoothProcessing/ProcessBPOnTargetDevice.h"
#include "RunBpStereoSetCUDAMemoryManagement.h"

template <typename T = float>
class RunBpStereoSetOnGPUWithCUDA : public RunBpStereoSet<T>
{
public:

	std::string getBpRunDescription() { return "CUDA"; }

	//run the disparity map estimation BP on a set of stereo images and save the results between each set of images
	ProcessStereoSetOutput operator()(const std::string& refImagePath, const std::string& testImagePath,
				const BPsettings& algSettings, std::ostream& resultsStream)
	{
		//using SmoothImageCUDA::SmoothImage;
		resultsStream << "CURRENT RUN: GPU WITH CUDA\n";
		std::unique_ptr<SmoothImage<>> smoothImageCUDA = std::make_unique<SmoothImageCUDA<>>();
		std::unique_ptr<ProcessBPOnTargetDevice<T, T*>> processImageCUDA = std::make_unique<ProcessCUDABP<T, T*>>();
		std::unique_ptr<RunBpStereoSetMemoryManagement<>> runBPCUDAMemoryManagement = std::make_unique<RunBpStereoSetCUDAMemoryManagement<>>();
		return this->processStereoSet(refImagePath, testImagePath, algSettings, resultsStream, smoothImageCUDA, processImageCUDA, runBPCUDAMemoryManagement);
	}
};

template<>
class RunBpStereoSetOnGPUWithCUDA<short> : public RunBpStereoSet<short>
{
public:

	std::string getBpRunDescription() { return "CUDA"; }

	//if type is specified as short, process as half on GPU
	//note that half is considered a data type for 16-bit floats in CUDA
	ProcessStereoSetOutput operator()(const std::string& refImagePath, const std::string& testImagePath,
					const BPsettings& algSettings, std::ostream& resultsStream)
	{

#if CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF

		std::cout << "Processing as half on GPU\n";
		std::unique_ptr<SmoothImage<>> smoothImageCUDA = std::make_unique<SmoothImageCUDA<>>();
		std::unique_ptr<ProcessBPOnTargetDevice<half, half*>> processImageCUDA = std::make_unique<ProcessCUDABP<T, T*>>();
		std::unique_ptr<RunBpStereoSetMemoryManagement<>> runBPCUDAMemoryManagement = std::make_unique<RunBpStereoSetCUDAMemoryManagement<>>();
		return this->processStereoSet(refImagePath, testImagePath, algSettings, resultsStream, smoothImageCUDA, processImageCUDA, runBPCUDAMemoryManagement);

#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO

		std::cout << "Processing as half2 on GPU\n";
		std::unique_ptr<SmoothImage<>> smoothImageCUDA = std::make_unique<SmoothImageCUDA<>>();
		std::unique_ptr<ProcessBPOnTargetDevice<half2, half2*>> processImageCUDA = std::make_unique<ProcessCUDABP<T, T*>>();
		std::unique_ptr<RunBpStereoSetMemoryManagement<>> runBPCUDAMemoryManagement = std::make_unique<RunBpStereoSetCUDAMemoryManagement<>>();
		return this->processStereoSet(refImagePath, testImagePath, algSettings, resultsStream, smoothImageCUDA, processImageCUDA, runBPCUDAMemoryManagement);

#else

		std::cout << "ERROR IN DATA TYPE\n";
		return ProcessStereoSetOutput();

#endif

	}
};

//float16_t data type used for arm (rather than short)
//TODO: needs to be updated with other code changes
#ifdef COMPILING_FOR_ARM

template<>
class RunBpStereoSetOnGPUWithCUDA<float16_t, float16_t*> : public RunBpStereoSet<float16_t, float16_t*>
{
public:

	std::string getBpRunDescription() { return "CUDA"; }

	//if type is specified as short, process as half on GPU
	//note that half is considered a data type for 16-bit floats in CUDA
	ProcessStereoSetOutput operator()(const std::string& refImagePath, const std::string& testImagePath,
			const BPsettings& algSettings, std::ostream& resultsStream, SmoothImage* smoothImage = nullptr, ProcessBPOnTargetDevice<short>* runBpStereo = nullptr, RunBpStereoSetMemoryManagement* runBPMemoryMangement = nullptr)
	{

#if CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF

		//std::cout << "Processing as half on GPU\n";
		RunBpStereoSetOnGPUWithCUDA<half> runCUDABpStereoSet;
		ProcessCUDABP<half> runCUDABPHalfPrecision;
		return runCUDABpStereoSet(refImagePath,
				testImagePath,
				algSettings,
				saveDisparityMapImagePath,
				resultsStream,
				smoothImage,
				&runCUDABPHalfPrecision,
				runBPMemoryMangement);

#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO

		//std::cout << "Processing as half2 on GPU\n";
		RunBpStereoSetOnGPUWithCUDA<half2> runCUDABpStereoSet;
		ProcessCUDABP<half2> runCUDABPHalfTwoDataType;
		return runCUDABpStereoSet(refImagePath, testImagePath, algSettings, saveDisparityMapImagePath, resultsStream, smoothImage, &runCUDABPHalfTwoDataType, runBPMemoryMangement);

#else

		std::cout << "ERROR IN DATA TYPE\n";
		return ProcessStereoSetOutput();

#endif

	}
};

#endif

#endif //RUN_BP_STEREO_IMAGE_SERIES_HEADER_CUH
