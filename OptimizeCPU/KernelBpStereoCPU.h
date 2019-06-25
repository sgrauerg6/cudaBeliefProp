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

#include "bpStereoParameters.h"
#include <math.h>
#include <omp.h>

class KernelBpStereoCPU
{
public:

	//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
	//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
	template<typename T>
	static void initializeBottomLevelDataStereoCPU(levelProperties& currentLevelProperties, float* image1PixelsDevice, float* image2PixelsDevice, T* dataCostDeviceStereoCheckerboard1, T* dataCostDeviceStereoCheckerboard2, float lambda_bp, float data_k_bp);

	template<typename T>
	static void initializeCurrentLevelDataStereoNoTexturesCPU(int checkerboardPart, levelProperties& currentLevelProperties, levelProperties& prevLevelProperties, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2, T* dataCostDeviceToWriteTo, int offsetNum);

	//initialize the message values at each pixel of the current level to the default value
	template<typename T>
	static void initializeMessageValsToDefaultKernelCPU(levelProperties& currentLevelProperties, T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1, T* messageLDeviceCurrentCheckerboard1,
														T* messageRDeviceCurrentCheckerboard1, T* messageUDeviceCurrentCheckerboard2, T* messageDDeviceCurrentCheckerboard2,
														T* messageLDeviceCurrentCheckerboard2, T* messageRDeviceCurrentCheckerboard2);

	//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	template<typename T>
	static void dtStereoSIMD(T f[NUM_POSSIBLE_DISPARITY_VALUES]);

	// compute current message
	template<typename T>
	static void msgStereoSIMD(T messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], T messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		T messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], T dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		T dst[NUM_POSSIBLE_DISPARITY_VALUES], T disc_k_bp);

	//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
	//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
	template<typename T>
	static void runBPIterationUsingCheckerboardUpdatesNoTexturesCPU(int checkerboardToUpdate, levelProperties& currentLevelProperties, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
									T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1, T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
									T* messageUDeviceCurrentCheckerboard2, T* messageDDeviceCurrentCheckerboard2, T* messageLDeviceCurrentCheckerboard2,
									T* messageRDeviceCurrentCheckerboard2, float disc_k_bp);

	template<typename T>
	static void runBPIterationUsingCheckerboardUpdatesNoTexturesCPUNoPackedInstructions(
			int checkerboardPartUpdate, levelProperties& currentLevelProperties,
			T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
			T* messageUDeviceCurrentCheckerboard1,
			T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1,
			T* messageRDeviceCurrentCheckerboard1,
			T* messageUDeviceCurrentCheckerboard2,
			T* messageDDeviceCurrentCheckerboard2,
			T* messageLDeviceCurrentCheckerboard2,
			T* messageRDeviceCurrentCheckerboard2, float disc_k_bp);

	//kernal to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
	//the kernal works from the point of view of the pixel at the prev level that is being copied to four different places
	template<typename T>
		static void copyPrevLevelToNextLevelBPCheckerboardStereoNoTexturesCPU(
				int checkerboardPart,
				levelProperties& currentLevelProperties,
				levelProperties& nextLevelProperties,
				T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
				T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
				T* messageUPrevStereoCheckerboard2, T* messageDPrevStereoCheckerboard2,
				T* messageLPrevStereoCheckerboard2, T* messageRPrevStereoCheckerboard2,
				T* messageUDeviceCurrentCheckerboard1,
				T* messageDDeviceCurrentCheckerboard1,
				T* messageLDeviceCurrentCheckerboard1,
				T* messageRDeviceCurrentCheckerboard1,
				T* messageUDeviceCurrentCheckerboard2,
				T* messageDDeviceCurrentCheckerboard2,
				T* messageLDeviceCurrentCheckerboard2,
				T* messageRDeviceCurrentCheckerboard2);

	//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
	template<typename T>
	static void retrieveOutputDisparityCheckerboardStereoOptimizedCPU(levelProperties& currentLevelProperties, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2, T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1, T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1, T* messageUPrevStereoCheckerboard2, T* messageDPrevStereoCheckerboard2, T* messageLPrevStereoCheckerboard2, T* messageRPrevStereoCheckerboard2, float* disparityBetweenImagesDevice);

	template<typename T>
	static void printDataAndMessageValsAtPointKernelCPU(int xVal, int yVal, levelProperties& currentLevelProperties, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
			T* messageUDeviceCurrentCheckerboard1,
			T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1,
			T* messageRDeviceCurrentCheckerboard1,
			T* messageUDeviceCurrentCheckerboard2,
			T* messageDDeviceCurrentCheckerboard2,
			T* messageLDeviceCurrentCheckerboard2,
			T* messageRDeviceCurrentCheckerboard2);

	template<typename T>
	static void printDataAndMessageValsToPointKernelCPU(int xVal, int yVal, levelProperties& currentLevelProperties, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
			T* messageUDeviceCurrentCheckerboard1,
			T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1,
			T* messageRDeviceCurrentCheckerboard1,
			T* messageUDeviceCurrentCheckerboard2,
			T* messageDDeviceCurrentCheckerboard2,
			T* messageLDeviceCurrentCheckerboard2,
			T* messageRDeviceCurrentCheckerboard2);

#define DIVISOR_FOR_PADDED_CHECKERBOARD_WIDTH_FOR_ALIGNMENT 16

	//inline function to check if data is aligned at xValDataStart for avx loads/stores that require alignment
	static bool MemoryAlignedAtDataStart(int xValDataStart, int numDataInAVXVector)
	{
		//assuming that the padded checkerboard width divides evenly by NUM_DATA_ALIGN_WIDTH_FROM_PYTHON (if that's not the case it's a bug)
		return (((xValDataStart % numDataInAVXVector) == 0) && ((NUM_DATA_ALIGN_WIDTH_FROM_PYTHON % DIVISOR_FOR_PADDED_CHECKERBOARD_WIDTH_FOR_ALIGNMENT) == 0));
	}

	template<typename T>
		static void runBPIterationUsingCheckerboardUpdatesNoTexturesCPUUseSIMDVectors(
				int checkerboardToUpdate, levelProperties& currentLevelProperties,
				T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
				T* messageUDeviceCurrentCheckerboard1,
				T* messageDDeviceCurrentCheckerboard1,
				T* messageLDeviceCurrentCheckerboard1,
				T* messageRDeviceCurrentCheckerboard1,
				T* messageUDeviceCurrentCheckerboard2,
				T* messageDDeviceCurrentCheckerboard2,
				T* messageLDeviceCurrentCheckerboard2,
				T* messageRDeviceCurrentCheckerboard2, float disc_k_bp)
	{
		printf("Data type not supported\n");
	}

	template<typename T, typename U>
	static void runBPIterationUsingCheckerboardUpdatesNoTexturesCPUUseSIMDVectorsProcess(
			int checkerboardToUpdate, levelProperties& currentLevelProperties,
			T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
			T* messageUDeviceCurrentCheckerboard1,
			T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1,
			T* messageRDeviceCurrentCheckerboard1,
			T* messageUDeviceCurrentCheckerboard2,
			T* messageDDeviceCurrentCheckerboard2,
			T* messageLDeviceCurrentCheckerboard2,
			T* messageRDeviceCurrentCheckerboard2, float disc_k_bp,
			int numDataInSIMDVector);

	template<typename T, typename U>
	static U loadPackedDataAligned(int x, int y, int currentDisparity, levelProperties& currentLevelProperties, T* inData)
	{
		printf("Data type not supported for loading aligned data\n");
	}

	template<typename T, typename U>
	static U loadPackedDataUnaligned(int x, int y, int currentDisparity, levelProperties& currentLevelProperties, T* inData)
	{
		printf("Data type not supported for loading unaligned data\n");
	}

	template<typename T, typename U>
	static void storePackedDataAligned(
			int indexDataStore, T* locationDataStore, U dataToStore)
	{
		printf("Data type not supported for storing aligned data\n");
	}

	template<typename T, typename U>
	static void storePackedDataUnaligned(
			int indexDataStore, T* locationDataStore, U dataToStore)
	{
		printf("Data type not supported for storing unaligned data\n");
	}

	template<typename T>
	static T createSIMDVectorSameData(float data)
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

#if CPU_OPTIMIZATION_SETTING == USE_AVX_256

#include "KernelBpStereoCPU_AVX256TemplateSpFuncts.h"

#elif CPU_OPTIMIZATION_SETTING == USE_AVX_512

#include "KernelBpStereoCPU_AVX512TemplateSpFuncts.h"

#else

//do nothing

#endif //CPU_OPTIMIZATION_SETTING == USE_AVX_256

#endif //COMPILING_FOR_ARM


#endif //KERNAL_BP_STEREO_CPU_H
