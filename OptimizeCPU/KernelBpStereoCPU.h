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

//This header declares the kernal functions and constant/texture storage to run belief propagation on CUDA

#ifndef KERNAL_BP_STEREO_CPU_H
#define KERNAL_BP_STEREO_CPU_H

#include "../ParameterFiles/bpStereoParameters.h"
#include "../ParameterFiles/bpStructsAndEnums.h"
#include <math.h>
#include <omp.h>
#include <algorithm>
#include "KernelBpStereoCPU_AVX256_SIMDFuncts.h"

class KernelBpStereoCPU
{
public:

	//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
	//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
	template<typename T, unsigned int DISP_VALS>
	static void initializeBottomLevelDataStereoCPU(const levelProperties& currentLevelProperties,
			float* image1PixelsDevice, float* image2PixelsDevice,
			T* dataCostDeviceStereoCheckerboard0, T* dataCostDeviceStereoCheckerboard1,
			const float lambda_bp, const float data_k_bp);

	template<typename T, unsigned int DISP_VALS>
	static void initializeCurrentLevelDataStereoCPU(const Checkerboard_Parts checkerboardPart,
			const levelProperties& currentLevelProperties, const levelProperties& prevLevelProperties,
			T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
			T* dataCostDeviceToWriteTo, const unsigned int offsetNum);

	//initialize the message values at each pixel of the current level to the default value
	template<typename T, unsigned int DISP_VALS>
	static void initializeMessageValsToDefaultKernelCPU(const levelProperties& currentLevelProperties,
			T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
			T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
			T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1);

	//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
	//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
	template<typename T, unsigned int DISP_VALS>
	static void runBPIterationUsingCheckerboardUpdatesCPU(const Checkerboard_Parts checkerboardToUpdate,
			const levelProperties& currentLevelProperties,
			T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
			T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
			T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
			T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
			const float disc_k_bp);

	template<typename T, unsigned int DISP_VALS>
	static void runBPIterationUsingCheckerboardUpdatesCPUNoPackedInstructions(
			const Checkerboard_Parts checkerboardPartUpdate, const levelProperties& currentLevelProperties,
			T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
			T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
			T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
			T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
			const float disc_k_bp);

	//kernal to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
	//the kernal works from the point of view of the pixel at the prev level that is being copied to four different places
	template<typename T, unsigned int DISP_VALS>
	static void copyPrevLevelToNextLevelBPCheckerboardStereoCPU(const Checkerboard_Parts checkerboardPart,
			const levelProperties& currentLevelProperties, const levelProperties& nextLevelProperties,
			T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
			T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
			T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
			T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
			T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
			T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
			T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1);

	//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
	template<typename T, unsigned int DISP_VALS>
	static void retrieveOutputDisparityCheckerboardStereoOptimizedCPU(const levelProperties& currentLevelProperties,
			T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
			T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
			T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
			T* messageUPrevStereoCheckerboard2, T* messageDPrevStereoCheckerboard2,
			T* messageLPrevStereoCheckerboard2, T* messageRPrevStereoCheckerboard2,
			float* disparityBetweenImagesDevice);

	template<typename T, unsigned int DISP_VALS>
	static void printDataAndMessageValsAtPointKernelCPU(const unsigned int xVal, const unsigned int yVal,
			const levelProperties& currentLevelProperties,
			T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
			T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
			T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
			T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1);

	template<typename T, unsigned int DISP_VALS>
	static void printDataAndMessageValsToPointKernelCPU(
			const unsigned int xVal, const unsigned int yVal,
			const levelProperties& currentLevelProperties,
			T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
			T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
			T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
			T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1);

	//device portion of the kernel function to run the current iteration of belief propagation where the input messages and data costs come in as array in local memory
	//and the output message values are save to output message arrays
	template<typename T, typename U, unsigned int DISP_VALS>
	static void runBPIterationInOutDataInLocalMemCPUUseSIMDVectors(const unsigned int xValStartProcessing,
			const unsigned int yVal, const levelProperties& currentLevelProperties,
			U prevUMessage[DISP_VALS], U prevDMessage[DISP_VALS],
			U prevLMessage[DISP_VALS], U prevRMessage[DISP_VALS],
			U dataMessage[DISP_VALS],
			T* currentUMessageArray, T* currentDMessageArray,
			T* currentLMessageArray, T* currentRMessageArray,
			const U disc_k_bp_vector, const bool dataAlignedAtxValStartProcessing);

	template<unsigned int DISP_VALS>
	static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors(
			const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
			float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
			float* messageUDeviceCurrentCheckerboard0, float* messageDDeviceCurrentCheckerboard0,
			float* messageLDeviceCurrentCheckerboard0, float* messageRDeviceCurrentCheckerboard0,
			float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1,
			float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
			const float disc_k_bp);

	template<unsigned int DISP_VALS>
	static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors(
			const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
			short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
			short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
			short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
			short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
			short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
			const float disc_k_bp);

	template<unsigned int DISP_VALS>
	static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors(
			const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
			double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
			double* messageUDeviceCurrentCheckerboard0, double* messageDDeviceCurrentCheckerboard0,
			double* messageLDeviceCurrentCheckerboard0, double* messageRDeviceCurrentCheckerboard0,
			double* messageUDeviceCurrentCheckerboard1, double* messageDDeviceCurrentCheckerboard1,
			double* messageLDeviceCurrentCheckerboard1, double* messageRDeviceCurrentCheckerboard1,
			const float disc_k_bp);

	template<typename T, typename U, unsigned int DISP_VALS>
	static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess(
			const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
			T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
			T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
			T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
			T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
			const float disc_k_bp, const unsigned int numDataInSIMDVector);

	template<typename T, typename U>
	static U loadPackedDataAligned(const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
			const levelProperties& currentLevelProperties, const unsigned int numDispVals, T* inData)
	{
		printf("Data type not supported for loading aligned data\n");
	}

	template<typename T, typename U>
	static U loadPackedDataUnaligned(const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
			const levelProperties& currentLevelProperties, const unsigned int numDispVals, T* inData)
	{
		printf("Data type not supported for loading unaligned data\n");
	}

	template<typename T>
	static T createSIMDVectorSameData(const float data)
	{
		printf("Data type not supported for creating simd vector\n");
	}
};

#ifdef COMPILING_FOR_ARM

#include "KernelBpStereoCPU_ARMTemplateSpFuncts.h"

#if CPU_OPTIMIZATION_SETTING == USE_NEON

#include "KernelBpStereoCPU_NEON.h"

#endif //CPU_OPTIMIZATION_SETTING == USE_NEON

#else

//needed so that template specializations are used when available
#include "KernelBpStereoCPU_TemplateSpFuncts.h"
#include "KernelBpStereoCPU_AVX256TemplateSpFuncts.h"
//uncomment to enable AVX512 (need to then comment out AVX256)
//#include "KernelBpStereoCPU_AVX512TemplateSpFuncts.h"

//do nothing

#endif //COMPILING_FOR_ARM


#endif //KERNAL_BP_STEREO_CPU_H
