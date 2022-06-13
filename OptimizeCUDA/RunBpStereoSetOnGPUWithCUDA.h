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
#include "../ParameterFiles/bpStereoCudaParameters.h"
#include "../BpAndSmoothProcessing/RunBpStereoSet.h"
#include "SmoothImageCUDA.h"
#include "../BpAndSmoothProcessing/ProcessBPOnTargetDevice.h"
#include "RunBpStereoSetCUDAMemoryManagement.h"

namespace bp_cuda_device
{
	void retrieveDeviceProperties(int numDevice, std::ostream& resultsStream)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, numDevice);
		int cudaDriverVersion;
		cudaDriverGetVersion(&cudaDriverVersion);

		resultsStream << "Device " << numDevice << ": " << prop.name << " with " << prop.multiProcessorCount << " multiprocessors\n";
		resultsStream << "Cuda version: " << cudaDriverVersion << "\n";
		int cudaRuntimeVersion;
		cudaRuntimeGetVersion(&cudaRuntimeVersion);
		resultsStream << "Cuda Runtime Version: " << cudaRuntimeVersion << "\n";
	}
};

template <typename T, unsigned int DISP_VALS>
class RunBpStereoSetOnGPUWithCUDA : public RunBpStereoSet<T, DISP_VALS>
{
public:
    RunBpStereoSetOnGPUWithCUDA() {}

	std::string getBpRunDescription() override { return "CUDA"; }

	//run the disparity map estimation BP on a set of stereo images and save the results between each set of images
	ProcessStereoSetOutput operator()(const std::array<std::string, 2>& refTestImagePath,
				const beliefprop::BPsettings& algSettings, 
				const beliefprop::ParallelParameters& parallelParams,
				std::ostream& resultsStream) override
	{
		//using SmoothImageCUDA::SmoothImage;
		resultsStream << "CURRENT RUN: GPU WITH CUDA\n";
		bp_cuda_device::retrieveDeviceProperties(0, resultsStream);

		//generate struct with pointers to objects for running CUDA implementation and call
		//function to run CUDA implementation
		return this->processStereoSet(refTestImagePath, algSettings,
			BpOnDevice<T, T*, DISP_VALS>{std::make_unique<SmoothImageCUDA>(parallelParams),
										 std::make_unique<ProcessCUDABP<T, T*, DISP_VALS>>(parallelParams),
										 std::make_unique<RunBpStereoSetCUDAMemoryManagement<>>(),
										 std::make_unique<RunBpStereoSetCUDAMemoryManagement<T>>()},
			resultsStream);
	}
};

//float16_t data type used for arm (rather than short)
//TODO: needs to be updated with other code changes
#ifdef COMPILING_FOR_ARM

#ifdef HALF_PRECISION_SUPPORTED

template<>
class RunBpStereoSetOnGPUWithCUDA<float16_t, float16_t*> : public RunBpStereoSet<float16_t, float16_t*>
{
public:
    RunBpStereoSetOnGPUWithCUDA() {}

	std::string getBpRunDescription() override  { return "CUDA"; }

	//if type is specified as short, process as half on GPU
	//note that half is considered a data type for 16-bit floats in CUDA
	ProcessStereoSetOutput operator() (const std::string& refImagePath, const std::string& testImagePath,
			const beliefprop::BPsettings& algSettings, std::ostream& resultsStream, SmoothImage* smoothImage = nullptr, ProcessBPOnTargetDevice<short>* runBpStereo = nullptr, RunBpStereoSetMemoryManagement* memManagementImages = nullptr) override
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
				memManagementImages);

#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO

		//std::cout << "Processing as half2 on GPU\n";
		RunBpStereoSetOnGPUWithCUDA<half2> runCUDABpStereoSet;
		ProcessCUDABP<half2> runCUDABPHalfTwoDataType;
		return runCUDABpStereoSet(refImagePath, testImagePath, algSettings, saveDisparityMapImagePath, resultsStream, smoothImage, &runCUDABPHalfTwoDataType, memManagementImages);

#else

		std::cout << "ERROR IN DATA TYPE\n";
		return ProcessStereoSetOutput();

#endif //CURRENT_DATA_TYPE_PROCESSING

	}
};

#endif //HALF_PRECISION_SUPPORTED

#endif //COMPILING_FOR_ARM

#ifdef _WIN32

extern "C" __declspec(dllexport) RunBpStereoSet<float, 0> * __cdecl createRunBpStereoSetOnGPUWithCUDAFloat();
extern "C" __declspec(dllexport) RunBpStereoSet<double, 0> * __cdecl createRunBpStereoSetOnGPUWithCUDADouble();
extern "C" __declspec(dllexport) RunBpStereoSet<short, 0> * __cdecl createRunBpStereoSetOnGPUWithCUDAShort();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]> * __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp0();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]> * __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp0();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]> * __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp0();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]> * __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp1();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]> * __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp1();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]> * __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp1();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]> * __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp2();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]> * __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp2();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]> * __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp2();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]> * __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp3();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]> * __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp3();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]> * __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp3();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]> * __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp4();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]> * __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp4();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]> * __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp4();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]> * __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp5();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]> * __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp5();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]> * __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp5();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]> * __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp6();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]> * __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp6();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]> * __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp6();

#endif //_WIN32

#endif //RUN_BP_STEREO_IMAGE_SERIES_HEADER_CUH
