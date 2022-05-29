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

//This header declares the kernel functions and constant/texture storage to run belief propagation on CUDA

#ifndef KERNEL_BP_STEREO_CPU_H
#define KERNEL_BP_STEREO_CPU_H

#include "../ParameterFiles/bpStereoParameters.h"
#include "../ParameterFiles/bpStructsAndEnums.h"
#include "../ParameterFiles/bpRunSettings.h"
#include <math.h>
#if ((CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_CHUNKS) || (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_DISTRIBUTED))
#include "../ThreadPool/thread_pool.hpp"
#else //(CPU_PARALLELIZATION_METHOD == USE_OPENMP)
#include <omp.h>
#endif //CPU_PARALLELIZATION_METHOD
#include <algorithm>

//headers to include differ depending on architecture and CPU vectorization setting
#ifdef COMPILING_FOR_ARM
#include <arm_neon.h>
#endif //COMPILING_FOR_ARM

class KernelBpStereoCPU
{
public:
#if ((CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_CHUNKS) || (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_DISTRIBUTED))
    //initialize thread pool with default number of threads
	inline static thread_pool tPool;
#endif //CPU_PARALLELIZATION_METHOD

	//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
	//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
	template<typename T, unsigned int DISP_VALS>
	static void initializeBottomLevelDataStereoCPU(const levelProperties& currentLevelProperties,
			float* image1PixelsDevice, float* image2PixelsDevice,
			T* dataCostDeviceStereoCheckerboard0, T* dataCostDeviceStereoCheckerboard1,
			const float lambda_bp, const float data_k_bp, const unsigned int bpSettingsDispVals);

	template<typename T, unsigned int DISP_VALS>
	static void initializeCurrentLevelDataStereoCPU(const Checkerboard_Parts checkerboardPart,
			const levelProperties& currentLevelProperties, const levelProperties& prevLevelProperties,
			T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
			T* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals);

	//initialize the message values at each pixel of the current level to the default value
	template<typename T, unsigned int DISP_VALS>
	static void initializeMessageValsToDefaultKernelCPU(const levelProperties& currentLevelProperties,
			T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
			T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
			T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
			const unsigned int bpSettingsDispVals);

	//kernel function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
	//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
	template<typename T, unsigned int DISP_VALS>
	static void runBPIterationUsingCheckerboardUpdatesCPU(const Checkerboard_Parts checkerboardToUpdate,
			const levelProperties& currentLevelProperties,
			T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
			T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
			T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
			T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
			const float disc_k_bp, const unsigned int bpSettingsNumDispVals);

	template<typename T, unsigned int DISP_VALS>
	static void runBPIterationUsingCheckerboardUpdatesCPUNoPackedInstructions(
			const Checkerboard_Parts checkerboardPartUpdate, const levelProperties& currentLevelProperties,
			T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
			T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
			T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
			T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
			const float disc_k_bp, const unsigned int bpSettingsDispVals);

	//kernel to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
	//the kernel works from the point of view of the pixel at the prev level that is being copied to four different places
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
			T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
			const unsigned int bpSettingsDispVals);

	//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
	template<typename T, unsigned int DISP_VALS>
	static void retrieveOutputDisparityCheckerboardStereoOptimizedCPU(const levelProperties& currentLevelProperties,
			T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
			T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
			T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
			T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
			T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
			float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals);

	//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel using SIMD vectors
    template<typename T, typename U, typename V, typename W, unsigned int DISP_VALS>
	static void retrieveOutDispOptimizedCPUUseSIMDVectorsProcess(const levelProperties& currentLevelProperties,
			T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
			T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
			T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
			T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
			T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
			float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
			const unsigned int numDataInSIMDVector);

	template<unsigned int DISP_VALS>
    static void retrieveOutputDisparityCheckerboardStereoOptimizedCPUUseSIMDVectors(
		const levelProperties& currentLevelProperties,
		float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
		float* messageUPrevStereoCheckerboard0, float* messageDPrevStereoCheckerboard0,
		float* messageLPrevStereoCheckerboard0, float* messageRPrevStereoCheckerboard0,
		float* messageUPrevStereoCheckerboard1, float* messageDPrevStereoCheckerboard1,
		float* messageLPrevStereoCheckerboard1, float* messageRPrevStereoCheckerboard1,
		float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals);

	template<unsigned int DISP_VALS>
    static void retrieveOutputDisparityCheckerboardStereoOptimizedCPUUseSIMDVectors(
		const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
		short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
		short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0,
		short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
		short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1,
		float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals);

	template<unsigned int DISP_VALS>
    static void retrieveOutputDisparityCheckerboardStereoOptimizedCPUUseSIMDVectors(
		const levelProperties& currentLevelProperties,
		double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
		double* messageUPrevStereoCheckerboard0, double* messageDPrevStereoCheckerboard0,
		double* messageLPrevStereoCheckerboard0, double* messageRPrevStereoCheckerboard0,
		double* messageUPrevStereoCheckerboard1, double* messageDPrevStereoCheckerboard1,
		double* messageLPrevStereoCheckerboard1, double* messageRPrevStereoCheckerboard1,
		float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals);

#ifdef COMPILING_FOR_ARM
	template<unsigned int DISP_VALS>
    static void retrieveOutputDisparityCheckerboardStereoOptimizedCPUUseSIMDVectors(
		const levelProperties& currentLevelProperties,
		float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
		float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
		float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
		float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
		float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
		float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals);
#endif //COMPILING_FOR_ARM

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

	template<typename T, typename U>
	static void runBPIterationInOutDataInLocalMemCPUUseSIMDVectors(
			const unsigned int xValStartProcessing, const unsigned int yVal,
			const levelProperties& currentLevelProperties,
			U* prevUMessage, U* prevDMessage,
			U* prevLMessage, U* prevRMessage,
			U* dataMessage,
			T* currentUMessageArray, T* currentDMessageArray,
			T* currentLMessageArray, T* currentRMessageArray,
			const U disc_k_bp_vector, const bool dataAlignedAtxValStartProcessing,
			const unsigned int bpSettingsDispVals);

	template<unsigned int DISP_VALS>
	static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors(
			const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
			float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
			float* messageUDeviceCurrentCheckerboard0, float* messageDDeviceCurrentCheckerboard0,
			float* messageLDeviceCurrentCheckerboard0, float* messageRDeviceCurrentCheckerboard0,
			float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1,
			float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
			const float disc_k_bp, const unsigned int bpSettingsDispVals = 0);

	template<unsigned int DISP_VALS>
	static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors(
			const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
			short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
			short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
			short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
			short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
			short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
			const float disc_k_bp, const unsigned int bpSettingsDispVals = 0);

	template<unsigned int DISP_VALS>
	static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors(
			const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
			double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
			double* messageUDeviceCurrentCheckerboard0, double* messageDDeviceCurrentCheckerboard0,
			double* messageLDeviceCurrentCheckerboard0, double* messageRDeviceCurrentCheckerboard0,
			double* messageUDeviceCurrentCheckerboard1, double* messageDDeviceCurrentCheckerboard1,
			double* messageLDeviceCurrentCheckerboard1, double* messageRDeviceCurrentCheckerboard1,
			const float disc_k_bp, const unsigned int bpSettingsDispVals = 0);

#ifdef COMPILING_FOR_ARM
	template<unsigned int DISP_VALS>
	static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors(
			const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
			float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
			float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
			float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
			float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
			float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
			const float disc_k_bp, const unsigned int bpSettingsDispVals = 0);
#endif //COMPILING_FOR_ARM

	template<typename T, typename U, unsigned int DISP_VALS>
	static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess(
			const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
			T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
			T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
			T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
			T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
			const float disc_k_bp, const unsigned int numDataInSIMDVector,
			const unsigned int bpSettingsDispVals);

	// compute current message
	template<typename T, typename U, unsigned int DISP_VALS>
	static void msgStereoSIMD(const unsigned int xVal, const unsigned int yVal,
			const levelProperties& currentLevelProperties,
			U messageValsNeighbor1[DISP_VALS], U messageValsNeighbor2[DISP_VALS],
			U messageValsNeighbor3[DISP_VALS], U dataCosts[DISP_VALS],
			T* dstMessageArray, const U& disc_k_bp, const bool dataAligned);

	// compute current message
	template<typename T, typename U>
	static void msgStereoSIMD(const unsigned int xVal, const unsigned int yVal,
			const levelProperties& currentLevelProperties,
			U* messageValsNeighbor1, U* messageValsNeighbor2,
			U* messageValsNeighbor3, U* dataCosts,
			T* dstMessageArray, const U& disc_k_bp, const bool dataAligned,
			const unsigned int bpSettingsDispVals);

	// compute current message
	template<typename T, typename U, typename V, typename W>
	static void msgStereoSIMDProcessing(const unsigned int xVal, const unsigned int yVal,
			const levelProperties& currentLevelProperties,
			U* messageValsNeighbor1, U* messageValsNeighbor2,
			U* messageValsNeighbor3, U* dataCosts,
			T* dstMessageArray, const U& disc_k_bp, const bool dataAligned,
			const unsigned int bpSettingsDispVals);

	// compute current message
	template<typename T, typename U, typename V, typename W, unsigned int DISP_VALS>
	static void msgStereoSIMDProcessing(const unsigned int xVal, const unsigned int yVal,
			const levelProperties& currentLevelProperties,
			U messageValsNeighbor1[DISP_VALS], U messageValsNeighbor2[DISP_VALS],
			U messageValsNeighbor3[DISP_VALS], U dataCosts[DISP_VALS],
			T* dstMessageArray, const U& disc_k_bp, const bool dataAligned);

	//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	template<typename T, typename U, unsigned int DISP_VALS>
	static void dtStereoSIMD(U f[DISP_VALS]);

	//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	//TODO: look into defining function in .cpp file so don't need to declare inline
	template<typename T, typename U>
	static void dtStereoSIMD(U* f, const unsigned int bpSettingsDispVals);

	template<typename T>
	static void updateBestDispBestVals(T& bestDisparities, T& bestVals, const T& currentDisparity, const T& valAtDisp) {
		printf("Data type not supported for updating best disparities and values\n");
	}

	template<typename T, typename U>
	static U loadPackedDataAligned(const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
			const levelProperties& currentLevelProperties, const unsigned int numDispVals, T* inData) {
		printf("Data type not supported for loading aligned data\n");
	}

	template<typename T, typename U>
	static U loadPackedDataUnaligned(const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
			const levelProperties& currentLevelProperties, const unsigned int numDispVals, T* inData) {
		printf("Data type not supported for loading unaligned data\n");
	}

	template<typename T>
	static T createSIMDVectorSameData(const float data) {
		printf("Data type not supported for creating simd vector\n");
	}

	template<typename T, typename U, typename V>
	static V addVals(const T& val1, const U& val2) { return (val1 + val2); }

	template<typename T, typename U, typename V>
	static V subtractVals(const T& val1, const U& val2) { return (val1 - val2); }

	template<typename T, typename U, typename V>
	static V divideVals(const T& val1, const U& val2) { return (val1 / val2); }

	template<typename T, typename V>
	static T convertValToDatatype(const V val) { return (T)val; }

	template<typename T>
	static T getMinByElement(const T& val1, const T& val2) { return std::min(val1, val2); }

	template<typename T, typename U>
	static void storePackedDataAligned(const unsigned int indexDataStore, T* locationDataStore, const U& dataToStore) {
		locationDataStore[indexDataStore] = dataToStore;
	}

	template<typename T, typename U>
	static void storePackedDataUnaligned(const unsigned int indexDataStore, T* locationDataStore, const U& dataToStore) {
		locationDataStore[indexDataStore] = dataToStore;
	}

	template<typename T, unsigned int DISP_VALS>
	static void printDataAndMessageValsAtPointKernelCPU(const unsigned int xVal, const unsigned int yVal,
			const levelProperties& currentLevelProperties,
			T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
			T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
			T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
			T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1);

	template<typename T, unsigned int DISP_VALS>
	static void printDataAndMessageValsToPointKernelCPU(const unsigned int xVal, const unsigned int yVal,
			const levelProperties& currentLevelProperties,
			T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
			T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
			T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
			T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1);
};

//headers to include differ depending on architecture and CPU vectorization setting
#ifdef COMPILING_FOR_ARM
#include "KernelBpStereoCPU_ARMTemplateSpFuncts.h"

#if (CPU_VECTORIZATION_SETTING == NEON)
#include "KernelBpStereoCPU_NEON.h"
#endif //CPU_OPTIMIZATION_SETTING == USE_NEON

#else
//needed so that template specializations are used when available
#include "KernelBpStereoCPU_TemplateSpFuncts.h"

#if (CPU_VECTORIZATION_SETTING == AVX_256)
#include "KernelBpStereoCPU_AVX256TemplateSpFuncts.h"
#elif (CPU_VECTORIZATION_SETTING == AVX_512)
#include "KernelBpStereoCPU_AVX512TemplateSpFuncts.h"
#endif

#endif //COMPILING_FOR_ARM

#endif //KERNEL_BP_STEREO_CPU_H
