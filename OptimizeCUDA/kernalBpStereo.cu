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

//This file defines the methods to perform belief propagation for disparity map estimation from stereo images on CUDA


//#include "kernalBpStereoHeader.cuh"
#define PROCESSING_ON_GPU
#include "../SharedFuncts/SharedBPProcessingFuncts.h"
#undef PROCESSING_ON_GPU

#include "../ParameterFiles/bpStereoCudaParameters.h"


#if ((USE_SHARED_MEMORY == 1) && (DISP_INDEX_START_REG_LOCAL_MEM > 0))
#include "SharedMemoryKernels/KernalBpStereoUseSharedMemory.cu"
#elif ((USE_SHARED_MEMORY == 2) && (DISP_INDEX_START_REG_LOCAL_MEM > 0))
#include "SharedMemoryKernels/KernalBpStereoUseSharedMemoryActuallyDuplicateRegMem.cu"
#elif ((USE_SHARED_MEMORY == 3) && (DISP_INDEX_START_REG_LOCAL_MEM > 0))
#include "SharedMemoryKernels/KernelBpStereoUseDynamicSharedMemory.cu"
#elif ((USE_SHARED_MEMORY == 4) && (DISP_INDEX_START_REG_LOCAL_MEM > 0))
#include "SharedMemoryKernels/KernelBpStereoDataAndMessageInDynamicSharedMemory.cu"
#else

#if ((CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF) || (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO))

//template specialization for processing messages with half-precision; has safeguard to check if valToNormalize goes to infinity and set output
//for every disparity at point to be 0.0 if that's the case; this has only been observed when using more than 5 computation levels with half-precision
template<>
__device__ void msgStereo<half, half>(int xVal, int yVal,
		levelProperties& currentLevelProperties,
		half messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES],
		half messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		half messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES],
		half dataCosts[NUM_POSSIBLE_DISPARITY_VALUES], half* dstMessageArray,
		half disc_k_bp, bool dataAligned)
{
	// aggregate and find min
	half minimum = INF_BP;

	half dst[NUM_POSSIBLE_DISPARITY_VALUES];

	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		dst[currentDisparity] = messageValsNeighbor1[currentDisparity]
				+ messageValsNeighbor2[currentDisparity]
				+ messageValsNeighbor3[currentDisparity]
				+ dataCosts[currentDisparity];
		if (dst[currentDisparity] < minimum)
			minimum = dst[currentDisparity];
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereo < half > (dst);

	// truncate
	minimum += disc_k_bp;

	// normalize
	half valToNormalize = 0;

	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++)
	{
		if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}

		valToNormalize += dst[currentDisparity];
	}

	//if valToNormalize is infinite or NaN (observed when using more than 5 computation levels with half-precision),
	//set destination vector to 0 for all disparities
	//note that may cause results to differ a little from ideal
	if (__hisnan(valToNormalize) || ((__hisinf(valToNormalize)) != 0)) {
		int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, 0,
				NUM_POSSIBLE_DISPARITY_VALUES);

		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			dstMessageArray[destMessageArrayIndex] = (half) 0.0;
#if OPTIMIZED_INDEXING_SETTING == 1
			destMessageArrayIndex +=
					currentLevelProperties.paddedWidthCheckerboardLevel;
#else
			destMessageArrayIndex++;
#endif //OPTIMIZED_INDEXING_SETTING == 1
		}
	}
	else
	{
		valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;

		int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, 0,
				NUM_POSSIBLE_DISPARITY_VALUES);

		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++)
		{
			dst[currentDisparity] -= valToNormalize;
			dstMessageArray[destMessageArrayIndex] = dst[currentDisparity];
#if OPTIMIZED_INDEXING_SETTING == 1
			destMessageArrayIndex +=
					currentLevelProperties.paddedWidthCheckerboardLevel;
#else
			destMessageArrayIndex++;
#endif //OPTIMIZED_INDEXING_SETTING == 1
		}
	}
}

#endif //((CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF) || (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO))

#endif //#if ((USE_SHARED_MEMORY == 1) && (DISP_INDEX_START_REG_LOCAL_MEM > 0))


//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
template<typename T>
__global__ void initializeBottomLevelDataStereo(
		levelProperties currentLevelProperties, float* image1PixelsDevice,
		float* image2PixelsDevice, T* dataCostDeviceStereoCheckerboard0,
		T* dataCostDeviceStereoCheckerboard1, float lambda_bp,
		float data_k_bp)
{
	// Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

	int xVal = bx * bp_cuda_params::BLOCK_SIZE_WIDTH_BP + tx;
	int yVal = by * bp_cuda_params::BLOCK_SIZE_HEIGHT_BP + ty;

	int xInCheckerboard = xVal / 2;

	if (withinImageBounds(xInCheckerboard, yVal, currentLevelProperties.widthLevel, currentLevelProperties.heightLevel))
	{
		initializeBottomLevelDataStereoPixel<T, T>(xVal, yVal,
				currentLevelProperties, image1PixelsDevice,
				image2PixelsDevice, dataCostDeviceStereoCheckerboard0,
				dataCostDeviceStereoCheckerboard1, lambda_bp,
				data_k_bp);
	}
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<typename T>
__global__ void initializeCurrentLevelDataStereo(
		Checkerboard_Parts checkerboardPart,
		levelProperties currentLevelProperties,
		levelProperties prevLevelProperties, T* dataCostStereoCheckerboard0,
		T* dataCostStereoCheckerboard1, T* dataCostDeviceToWriteTo,
		int offsetNum)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int xVal = bx * bp_cuda_params::BLOCK_SIZE_WIDTH_BP + tx;
	int yVal = by * bp_cuda_params::BLOCK_SIZE_HEIGHT_BP + ty;

	if (withinImageBounds(xVal, yVal, currentLevelProperties.widthCheckerboardLevel, currentLevelProperties.heightLevel))
	{
		initializeCurrentLevelDataStereoPixel<T, T>(
				xVal, yVal, checkerboardPart,
				currentLevelProperties,
				prevLevelProperties, dataCostStereoCheckerboard0,
				dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
				offsetNum);
	}
}


//initialize the message values at each pixel of the current level to the default value
template<typename T>
__global__ void initializeMessageValsToDefaultKernel(
		levelProperties currentLevelProperties,
		T* messageUDeviceCurrentCheckerboard0,
		T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0,
		T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int xValInCheckerboard = bx * bp_cuda_params::BLOCK_SIZE_WIDTH_BP + tx;
	int yVal = by * bp_cuda_params::BLOCK_SIZE_HEIGHT_BP + ty;

	if (withinImageBounds(xValInCheckerboard, yVal, currentLevelProperties.widthCheckerboardLevel, currentLevelProperties.heightLevel))
	{
		//initialize message values in both checkerboards

		initializeMessageValsToDefaultKernelPixel<T>(xValInCheckerboard,  yVal, currentLevelProperties, messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0, messageLDeviceCurrentCheckerboard0,
				messageRDeviceCurrentCheckerboard0, messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
				messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1);
	}
}


//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
template<typename T>
__global__ void runBPIterationUsingCheckerboardUpdates(
		Checkerboard_Parts checkerboardToUpdate,
		levelProperties currentLevelProperties, T* dataCostStereoCheckerboard0,
		T* dataCostStereoCheckerboard1, T* messageUDeviceCurrentCheckerboard0,
		T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0,
		T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1, float disc_k_bp,
		bool dataAligned)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int xVal = bx * bp_cuda_params::BLOCK_SIZE_WIDTH_BP + tx;
	int yVal = by * bp_cuda_params::BLOCK_SIZE_HEIGHT_BP + ty;

	if (withinImageBounds(xVal, yVal, currentLevelProperties.widthLevel/2, currentLevelProperties.heightLevel))
	{
		runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<T, T>(
				xVal, yVal, checkerboardToUpdate,
				currentLevelProperties,
				dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
				messageUDeviceCurrentCheckerboard0,
				messageDDeviceCurrentCheckerboard0,
				messageLDeviceCurrentCheckerboard0,
				messageRDeviceCurrentCheckerboard0,
				messageUDeviceCurrentCheckerboard1,
				messageDDeviceCurrentCheckerboard1,
				messageLDeviceCurrentCheckerboard1,
				messageRDeviceCurrentCheckerboard1, disc_k_bp,
				0, dataAligned);
	}
}


//kernal to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernal works from the point of view of the pixel at the prev level that is being copied to four different places
template<typename T>
__global__ void copyPrevLevelToNextLevelBPCheckerboardStereo(
		Checkerboard_Parts checkerboardPart,
		levelProperties currentLevelProperties,
		levelProperties nextLevelProperties,
		T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
		T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
		T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
		T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0,
		T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0,
		T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int xVal = bx * bp_cuda_params::BLOCK_SIZE_WIDTH_BP + tx;
	int yVal = by * bp_cuda_params::BLOCK_SIZE_HEIGHT_BP + ty;

	if (withinImageBounds(xVal, yVal, currentLevelProperties.widthCheckerboardLevel, currentLevelProperties.heightLevel))
	{
		copyPrevLevelToNextLevelBPCheckerboardStereoPixel<T>(xVal, yVal,
				checkerboardPart, currentLevelProperties,
				nextLevelProperties,
				messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
				messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
				messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
				messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
				messageUDeviceCurrentCheckerboard0,
				messageDDeviceCurrentCheckerboard0,
				messageLDeviceCurrentCheckerboard0,
				messageRDeviceCurrentCheckerboard0,
				messageUDeviceCurrentCheckerboard1,
				messageDDeviceCurrentCheckerboard1,
				messageLDeviceCurrentCheckerboard1,
				messageRDeviceCurrentCheckerboard1);
	}
}


//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
template<typename T>
__global__ void retrieveOutputDisparityCheckerboardStereoOptimized(
		levelProperties currentLevelProperties, T* dataCostStereoCheckerboard0,
		T* dataCostStereoCheckerboard1, T* messageUPrevStereoCheckerboard0,
		T* messageDPrevStereoCheckerboard0, T* messageLPrevStereoCheckerboard0,
		T* messageRPrevStereoCheckerboard0, T* messageUPrevStereoCheckerboard1,
		T* messageDPrevStereoCheckerboard1, T* messageLPrevStereoCheckerboard1,
		T* messageRPrevStereoCheckerboard1,
		float* disparityBetweenImagesDevice)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int xVal = bx * bp_cuda_params::BLOCK_SIZE_WIDTH_BP + tx;
	int yVal = by * bp_cuda_params::BLOCK_SIZE_HEIGHT_BP + ty;

	if (withinImageBounds(xVal, yVal, currentLevelProperties.widthCheckerboardLevel, currentLevelProperties.heightLevel))
	{
		retrieveOutputDisparityCheckerboardStereoOptimizedPixel<T, T>(xVal, yVal,
				currentLevelProperties, dataCostStereoCheckerboard0,
				dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
				messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
				messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1,
				messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
				messageRPrevStereoCheckerboard1,
				disparityBetweenImagesDevice);
	}
}

template<typename T>
__global__ void printDataAndMessageValsAtPointKernel(int xVal, int yVal, T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0,
		T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0,
		T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1, int widthLevelCheckerboardPart,
		int heightLevel)
{
	if (((xVal + yVal) % 2) == 0) {
		printf("xVal: %d\n", xVal);
		printf("yVal: %d\n", yVal);
		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			printf("DISP: %d\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
		}
	} else {
		printf("xVal: %d\n", xVal);
		printf("yVal: %d\n", yVal);
		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			printf("DISP: %d\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
		}
	}
}


template<typename T>
__device__ void printDataAndMessageValsAtPointDevice(int xVal, int yVal, T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0,
		T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0,
		T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1, int widthLevelCheckerboardPart,
		int heightLevel)
{
	if (((xVal + yVal) % 2) == 0) {
		printf("xVal: %d\n", xVal);
		printf("yVal: %d\n", yVal);
		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			printf("DISP: %d\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
		}
	} else {
		printf("xVal: %d\n", xVal);
		printf("yVal: %d\n", yVal);
		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			printf("DISP: %d\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
		}
	}
}


template<typename T>
__global__ void printDataAndMessageValsToPointKernel(int xVal, int yVal, T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0,
		T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0,
		T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1, int widthLevelCheckerboardPart,
		int heightLevel)
{
	int checkerboardAdjustment;
	if (((xVal + yVal) % 2) == 0)
		{
			checkerboardAdjustment = ((yVal)%2);
		}
		else //checkerboardToUpdate == CHECKERBOARD_PART_1
		{
			checkerboardAdjustment = ((yVal+1)%2);
		}
	if (((xVal + yVal) % 2) == 0) {
			printf("xVal: %d\n", xVal);
			printf("yVal: %d\n", yVal);
			for (int currentDisparity = 0;
					currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
					currentDisparity++) {
				printf("DISP: %d\n", currentDisparity);
				printf("messageUPrevStereoCheckerboard: %f \n",
						(float) messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
								xVal / 2, yVal + 1, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageDPrevStereoCheckerboard: %f \n",
						(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
								xVal / 2, yVal - 1, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageLPrevStereoCheckerboard: %f \n",
						(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
								xVal / 2 + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageRPrevStereoCheckerboard: %f \n",
						(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
								(xVal / 2 - 1) + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("dataCostStereoCheckerboard: %f \n",
						(float) dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
								xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
		} else {
			printf("xVal: %d\n", xVal);
			printf("yVal: %d\n", yVal);
			for (int currentDisparity = 0;
					currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
					currentDisparity++) {
				printf("DISP: %d\n", currentDisparity);
				printf("messageUPrevStereoCheckerboard: %f \n",
						(float) messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
								xVal / 2, yVal + 1, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageDPrevStereoCheckerboard: %f \n",
						(float) messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
								xVal / 2, yVal - 1, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageLPrevStereoCheckerboard: %f \n",
						(float) messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
								xVal / 2 + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageRPrevStereoCheckerboard: %f \n",
						(float) messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
								(xVal / 2 - 1) + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("dataCostStereoCheckerboard: %f \n",
						(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
								xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
		}
}


template<typename T>
__device__ void printDataAndMessageValsToPointDevice(int xVal, int yVal, T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0,
		T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0,
		T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1, int widthLevelCheckerboardPart,
		int heightLevel)
{
	int checkerboardAdjustment;
	if (((xVal + yVal) % 2) == 0)
		{
			checkerboardAdjustment = ((yVal)%2);
		}
		else //checkerboardToUpdate == CHECKERBOARD_PART_1
		{
			checkerboardAdjustment = ((yVal+1)%2);
		}

	if (((xVal + yVal) % 2) == 0) {
		printf("xVal: %d\n", xVal);
		printf("yVal: %d\n", yVal);
		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			printf("DISP: %d\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal + 1, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal - 1, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2 + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							(xVal / 2 - 1) + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
		}
	} else {
		printf("xVal: %d\n", xVal);
		printf("yVal: %d\n", yVal);
		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			printf("DISP: %d\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal + 1, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal - 1, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2 + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							(xVal / 2 - 1) + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
		}
	}
}


/*template<>
__device__ half2 getZeroVal<half2>()
{
	return __floats2half2_rn (0.0, 0.0);
}


__device__ half2 getMinBothPartsHalf2(half2 val1, half2 val2)
{
	half2 val1Less = __hlt2(val1, val2);
	half2 val2LessOrEqual = __hle2(val2, val1);
	return __hadd2(__hmul2(val1Less, val1), __hmul2(val2LessOrEqual, val2));
}

template<>
__device__ void dtStereo<half2>(half2 f[NUM_POSSIBLE_DISPARITY_VALUES])
{
	half2 prev;
	for (int currentDisparity = 1; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		prev = __hadd2(f[currentDisparity-1], __float2half2_rn(1.0f));
		f[currentDisparity] = getMinBothPartsHalf2(prev, f[currentDisparity]);
	}

	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		prev = __hadd2(f[currentDisparity+1], __float2half2_rn(1.0f));
		f[currentDisparity] = getMinBothPartsHalf2(prev, f[currentDisparity]);
	}
}*/


/*template<>
__device__ void msgStereo<half2>(half2 messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], half2 messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		half2 messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], half2 dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		half2 dst[NUM_POSSIBLE_DISPARITY_VALUES], half2 disc_k_bp)
{
	// aggregate and find min
	half2 minimum = __float2half2_rn(INF_BP);

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dst[currentDisparity] = __hadd2(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
		dst[currentDisparity] = __hadd2(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
		dst[currentDisparity] = __hadd2(dst[currentDisparity], dataCosts[currentDisparity]);

		minimum = getMinBothPartsHalf2(dst[currentDisparity], minimum);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereo<half2>(dst);

	// truncate
	minimum = __hadd2(minimum, disc_k_bp);

	// normalize
	half2 valToNormalize = __float2half2_rn(0.0f);

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dst[currentDisparity] = getMinBothPartsHalf2(minimum, dst[currentDisparity]);
		valToNormalize = __hadd2(valToNormalize, dst[currentDisparity]);
	}

	//if either valToNormalize in half2 is infinite or NaN, set destination vector to 0 for all disparities
	//note that may cause results to differ a little from ideal
	if (((__hisnan(__low2half(valToNormalize)))
			|| ((__hisinf(__low2half(valToNormalize)) != 0)))
			|| ((__hisnan(__high2half(valToNormalize)))
					|| ((__hisinf(__high2half(valToNormalize)) != 0))))
	{
		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++)
		{
			dst[currentDisparity] = __floats2half2_rn(0.0f, 0.0f);
		}
	}
	else
	{
		valToNormalize = __h2div(valToNormalize,
				__float2half2_rn((float) NUM_POSSIBLE_DISPARITY_VALUES));

		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++)
		{
			dst[currentDisparity] = __hsub2(dst[currentDisparity],
					valToNormalize);
		}
	}
	//check if both values in half2 are inf or nan
	/*if (((__hisnan(__low2half(valToNormalize)))
			|| ((__hisinf(__low2half(valToNormalize)) != 0)))
			&& ((__hisnan(__high2half(valToNormalize)))
					|| ((__hisinf(__high2half(valToNormalize)) != 0))))
	{
		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++)
		{
			dst[currentDisparity] = __floats2half2_rn(0.0f, 0.0f);
		}
	}
	else if (((__hisnan(__low2half(valToNormalize)))
			|| ((__hisinf(__low2half(valToNormalize)) != 0))))
	{
		//lower half of half2 is inf or nan
		valToNormalize = __h2div(valToNormalize,
				__float2half2_rn((float) NUM_POSSIBLE_DISPARITY_VALUES));

		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++)
		{
			dst[currentDisparity] = __hsub2(dst[currentDisparity],
					valToNormalize);
		}

		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++)
		{
			dst[currentDisparity] = __halves2half2((half)0.0f,
					__high2half(dst[currentDisparity]));
		}
	}
	else if ((__hisnan(__high2half(valToNormalize)))
			|| ((__hisinf(__high2half(valToNormalize)) != 0)))
	{
		//higher half of half2 is inf or nan
		valToNormalize = __h2div(valToNormalize,
				__float2half2_rn((float) NUM_POSSIBLE_DISPARITY_VALUES));

		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++)
		{
			dst[currentDisparity] = __hsub2(dst[currentDisparity],
					valToNormalize);
		}

		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++)
		{
			dst[currentDisparity] = __halves2half2(
					__low2half(dst[currentDisparity]), (half)0.0f);
		}
	}
}*/


//device portion of the kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the
//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//this function uses local memory to store the message and data values at each disparity in the intermediate step of current message computation
//this function uses linear memory bound to textures to access the current data and message values
/*template<>
__device__ void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMem<half2>(int xVal, int yVal,
		int checkerboardToUpdate, levelProperties& currentLevelProperties, half2* dataCostStereoCheckerboard1, half2* dataCostStereoCheckerboard2,
		half2* messageUDeviceCurrentCheckerboard1,
		half2* messageDDeviceCurrentCheckerboard1,
		half2* messageLDeviceCurrentCheckerboard1,
		half2* messageRDeviceCurrentCheckerboard1,
		half2* messageUDeviceCurrentCheckerboard2,
		half2* messageDDeviceCurrentCheckerboard2,
		half2* messageLDeviceCurrentCheckerboard2,
		half2* messageRDeviceCurrentCheckerboard2,
		float disc_k_bp, int offsetData)
{
}

	int indexWriteTo;
	int checkerboardAdjustment;

	//checkerboardAdjustment used for indexing into current checkerboard to update
	if (checkerboardToUpdate == CHECKERBOARD_PART_0)
	{
		checkerboardAdjustment = ((yVal)%2);
	}
	else //checkerboardToUpdate == CHECKERBOARD_PART_1
	{
		checkerboardAdjustment = ((yVal+1)%2);
	}

	//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
	//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
	//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
	if ((xVal >= (1/*switch to 0 if trying to match half results exactly*//* - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - checkerboardAdjustment)) && (yVal > 0) && (yVal < (heightLevel - 1)))
	{
		half2 prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		half2 prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		half2 prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		half2 prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		half2 dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		if (checkerboardToUpdate == CHECKERBOARD_PART_0)
		{
			half* messageLDeviceCurrentCheckerboard2Half = (half*)messageLDeviceCurrentCheckerboard2;
			half* messageRDeviceCurrentCheckerboard2Half = (half*)messageRDeviceCurrentCheckerboard2;

			for (int currentDisparity = 0;
					currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
					currentDisparity++)
			{
				dataMessage[currentDisparity] =
						dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
								xVal, yVal, widthLevelCheckerboardPart,
								heightLevel, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES, offsetData)];
				prevUMessage[currentDisparity] =
						messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
								xVal, (yVal + 1), widthLevelCheckerboardPart,
								heightLevel, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES)];
				prevDMessage[currentDisparity] =
						messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
								xVal, (yVal - 1), widthLevelCheckerboardPart,
								heightLevel, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES)];
				prevLMessage[currentDisparity] =
						__halves2half2(
								messageLDeviceCurrentCheckerboard2Half[retrieveIndexInDataAndMessage(
										((xVal * 2) + checkerboardAdjustment),
										yVal, widthLevelCheckerboardPart * 2,
										heightLevel, currentDisparity,
										NUM_POSSIBLE_DISPARITY_VALUES)],
								messageLDeviceCurrentCheckerboard2Half[retrieveIndexInDataAndMessage(
										((xVal * 2 + 1) + checkerboardAdjustment),
										yVal, widthLevelCheckerboardPart * 2,
										heightLevel, currentDisparity,
										NUM_POSSIBLE_DISPARITY_VALUES)]);

				//if ((((xVal * 2) - 1) + checkerboardAdjustment) >= 0)
				{
					prevRMessage[currentDisparity] =
							__halves2half2(
									messageRDeviceCurrentCheckerboard2Half[retrieveIndexInDataAndMessage(
											(((xVal * 2) - 1)
													+ checkerboardAdjustment),
											yVal,
											widthLevelCheckerboardPart * 2,
											heightLevel, currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)],
									messageRDeviceCurrentCheckerboard2Half[retrieveIndexInDataAndMessage(
											(((xVal * 2 + 1) - 1)
													+ checkerboardAdjustment),
											yVal,
											widthLevelCheckerboardPart * 2,
											heightLevel, currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)]);
				}
				/*else
				{
					prevRMessage[currentDisparity] =
							__halves2half2((half)0.0f,
									messageRDeviceCurrentCheckerboard2Half[retrieveIndexInDataAndMessage(
											(((xVal * 2 + 1) - 1)
													+ checkerboardAdjustment),
											yVal,
											widthLevelCheckerboardPart * 2,
											heightLevel, currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)]);
				}*//*
			}
		}
		else //checkerboardToUpdate == CHECKERBOARD_PART_1
		{
			half* messageLDeviceCurrentCheckerboard1Half = (half*)messageLDeviceCurrentCheckerboard1;
			half* messageRDeviceCurrentCheckerboard1Half = (half*)messageRDeviceCurrentCheckerboard1;

			for (int currentDisparity = 0;
					currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
					currentDisparity++)
			{
				dataMessage[currentDisparity] =
						dataCostStereoCheckerboard2[retrieveIndexInDataAndMessage(
								xVal, yVal, widthLevelCheckerboardPart,
								heightLevel, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES, offsetData)];
				prevUMessage[currentDisparity] =
						messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
								xVal, (yVal + 1), widthLevelCheckerboardPart,
								heightLevel, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES)];
				prevDMessage[currentDisparity] =
						messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
								xVal, (yVal - 1), widthLevelCheckerboardPart,
								heightLevel, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES)];
				prevLMessage[currentDisparity] =
						__halves2half2(
								messageLDeviceCurrentCheckerboard1Half[retrieveIndexInDataAndMessage(
										((xVal * 2)
												+ checkerboardAdjustment),
										yVal, widthLevelCheckerboardPart * 2,
										heightLevel, currentDisparity,
										NUM_POSSIBLE_DISPARITY_VALUES)],
								messageLDeviceCurrentCheckerboard1Half[retrieveIndexInDataAndMessage(
										((xVal * 2 + 1)
												+ checkerboardAdjustment),
										yVal, widthLevelCheckerboardPart * 2,
										heightLevel, currentDisparity,
										NUM_POSSIBLE_DISPARITY_VALUES)]);

				//if ((((xVal * 2) - 1) + checkerboardAdjustment) >= 0)
				{
					prevRMessage[currentDisparity] =
							__halves2half2(
									messageRDeviceCurrentCheckerboard1Half[retrieveIndexInDataAndMessage(
											(((xVal * 2) - 1)
													+ checkerboardAdjustment),
											yVal,
											widthLevelCheckerboardPart * 2,
											heightLevel, currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)],
									messageRDeviceCurrentCheckerboard1Half[retrieveIndexInDataAndMessage(
											(((xVal * 2 + 1) - 1)
													+ checkerboardAdjustment),
											yVal,
											widthLevelCheckerboardPart * 2,
											heightLevel, currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)]);
				}
				/*else
				{
					prevRMessage[currentDisparity] =
							__halves2half2((half) 0.0,
									messageRDeviceCurrentCheckerboard1Half[retrieveIndexInDataAndMessage(
											(((xVal * 2 + 1) - 1)
													+ checkerboardAdjustment),
											yVal,
											widthLevelCheckerboardPart * 2,
											heightLevel, currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)]);
				}*//*
			}
		}

		half2 currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		half2 currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		half2 currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		half2 currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		//uses the previous message values and data cost to calculate the current message values and store the results
		runBPIterationInOutDataInLocalMem<half2>(prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							currentUMessage, currentDMessage, currentLMessage, currentRMessage, __float2half2_rn(disc_k_bp));

		//write the calculated message values to global memory
		for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		{
			indexWriteTo = retrieveIndexInDataAndMessage(xVal, yVal, widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
			if (checkerboardToUpdate == CHECKERBOARD_PART_0)
			{
				messageUDeviceCurrentCheckerboard1[indexWriteTo] = currentUMessage[currentDisparity];
				messageDDeviceCurrentCheckerboard1[indexWriteTo] = currentDMessage[currentDisparity];
				messageLDeviceCurrentCheckerboard1[indexWriteTo] = currentLMessage[currentDisparity];
				messageRDeviceCurrentCheckerboard1[indexWriteTo] = currentRMessage[currentDisparity];
			}
			else //checkerboardToUpdate == CHECKERBOARD_PART_1
			{
				messageUDeviceCurrentCheckerboard2[indexWriteTo] = currentUMessage[currentDisparity];
				messageDDeviceCurrentCheckerboard2[indexWriteTo] = currentDMessage[currentDisparity];
				messageLDeviceCurrentCheckerboard2[indexWriteTo] = currentLMessage[currentDisparity];
				messageRDeviceCurrentCheckerboard2[indexWriteTo] = currentRMessage[currentDisparity];
			}
		}
	}
}
*/

//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
/*template<>
__global__ void retrieveOutputDisparityCheckerboardStereoOptimized<half2>(levelProperties currentLevelProperties, half2* dataCostStereoCheckerboard1, half2* dataCostStereoCheckerboard2, half2* messageUPrevStereoCheckerboard1, half2* messageDPrevStereoCheckerboard1, half2* messageLPrevStereoCheckerboard1, half2* messageRPrevStereoCheckerboard1, half2* messageUPrevStereoCheckerboard2, half2* messageDPrevStereoCheckerboard2, half2* messageLPrevStereoCheckerboard2, half2* messageRPrevStereoCheckerboard2, float* disparityBetweenImagesDevice)
{

}*/

//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
/*template<typename T>
__global__ void retrieveOutputDisparityCheckerboardStereo(T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2, T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1, T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1, T* messageUPrevStereoCheckerboard2, T* messageDPrevStereoCheckerboard2, T* messageLPrevStereoCheckerboard2, T* messageRPrevStereoCheckerboard2, float* disparityBetweenImagesDevice, int widthLevel, int heightLevel)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int xVal = bx * BLOCK_SIZE_WIDTH_BP + tx;
	int yVal = by * BLOCK_SIZE_HEIGHT_BP + ty;

	if (withinImageBounds(xVal, yVal, widthLevel, heightLevel))
	{
		int widthCheckerboard = getCheckerboardWidth<T>(widthLevel);
		int xValInCheckerboardPart = xVal/2;

		if (((yVal+xVal) % 2) == 0) //if true, then pixel is from part 1 of the checkerboard; otherwise, it's from part 2
		{
			int	checkerboardPartAdjustment = (yVal%2);

			if ((xVal >= 1) && (xVal < (widthLevel - 1)) && (yVal >= 1) && (yVal < (heightLevel - 1)))
			{
				// keep track of "best" disparity for current pixel
				int bestDisparity = 0;
				T best_val = INF_BP;
				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					T val = messageUPrevStereoCheckerboard2[retrieveIndexInDataAndMessage(xValInCheckerboardPart, (yVal + 1), widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)] +
						 messageDPrevStereoCheckerboard2[retrieveIndexInDataAndMessage(xValInCheckerboardPart, (yVal - 1), widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)] +
						 messageLPrevStereoCheckerboard2[retrieveIndexInDataAndMessage((xValInCheckerboardPart + checkerboardPartAdjustment), yVal, widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)] +
						 messageRPrevStereoCheckerboard2[retrieveIndexInDataAndMessage((xValInCheckerboardPart - 1 + checkerboardPartAdjustment), yVal, widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)] +
						 dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboardPart, yVal, widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];

					if (val < (best_val)) {
						best_val = val;
						bestDisparity = currentDisparity;
					}
				}
				disparityBetweenImagesDevice[yVal*widthLevel + xVal] = bestDisparity;
			}
			else
			{
				disparityBetweenImagesDevice[yVal*widthLevel + xVal] = 0;
			}
		}
		else //pixel from part 2 of checkerboard
		{
			int	checkerboardPartAdjustment = ((yVal + 1) % 2);

			if ((xVal >= 1) && (xVal < (widthLevel - 1)) && (yVal >= 1) && (yVal < (heightLevel - 1)))
			{


				// keep track of "best" disparity for current pixel
				int bestDisparity = 0;
				T best_val = INF_BP;
				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					T val = messageUPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboardPart, (yVal + 1), widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)] +
						messageDPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboardPart, (yVal - 1), widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)] +
						messageLPrevStereoCheckerboard1[retrieveIndexInDataAndMessage((xValInCheckerboardPart + checkerboardPartAdjustment), yVal, widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)] +
						messageRPrevStereoCheckerboard1[retrieveIndexInDataAndMessage((xValInCheckerboardPart - 1 + checkerboardPartAdjustment), yVal, widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)] +
						dataCostStereoCheckerboard2[retrieveIndexInDataAndMessage(xValInCheckerboardPart, yVal, widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];

					if (val < (best_val))
					{
						best_val = val;
						bestDisparity = currentDisparity;
					}
				}
				disparityBetweenImagesDevice[yVal*widthLevel + xVal] = bestDisparity;
			}
			else
			{
				disparityBetweenImagesDevice[yVal*widthLevel + xVal] = 0;
			}
		}
	}
}

//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
template<>
__global__ void retrieveOutputDisparityCheckerboardStereo<half2>(half2* dataCostStereoCheckerboard1, half2* dataCostStereoCheckerboard2, half2* messageUPrevStereoCheckerboard1, half2* messageDPrevStereoCheckerboard1, half2* messageLPrevStereoCheckerboard1, half2* messageRPrevStereoCheckerboard1, half2* messageUPrevStereoCheckerboard2, half2* messageDPrevStereoCheckerboard2, half2* messageLPrevStereoCheckerboard2, half2* messageRPrevStereoCheckerboard2, float* disparityBetweenImagesDevice, int widthLevel, int heightLevel)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int xVal = bx * BLOCK_SIZE_WIDTH_BP + tx;
	int yVal = by * BLOCK_SIZE_HEIGHT_BP + ty;

	if (withinImageBounds(xVal*2, yVal, widthLevel, heightLevel))
	{
		int widthCheckerboard = getCheckerboardWidth<half2>(widthLevel);
		int xValInCheckerboardPart = xVal/2;

		if (((yVal+xVal) % 2) == 0) //if true, then pixel is from part 1 of the checkerboard; otherwise, it's from part 2
		{
			int	checkerboardPartAdjustment = (yVal%2);

			half* messageLPrevStereoCheckerboard2Half = (half*)messageLPrevStereoCheckerboard2;
			half* messageRPrevStereoCheckerboard2Half = (half*)messageRPrevStereoCheckerboard2;

			if ((xVal >= 1) && (xVal < (widthLevel - 1)) && (yVal >= 1) && (yVal < (heightLevel - 1)))
			{
				// keep track of "best" disparity for current pixel
				int bestDisparity1 = 0;
				int bestDisparity2 = 0;
				float best_val1 = INF_BP;
				float best_val2 = INF_BP;
				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					half2 val = __hadd2(messageUPrevStereoCheckerboard2[retrieveIndexInDataAndMessage(xValInCheckerboardPart, (yVal + 1), widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)],
											 messageDPrevStereoCheckerboard2[retrieveIndexInDataAndMessage(xValInCheckerboardPart, (yVal - 1), widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					val =
							__hadd2(val,
									__halves2half2(
											messageLPrevStereoCheckerboard2Half[retrieveIndexInDataAndMessage(
													((xValInCheckerboardPart * 2)
															+ checkerboardPartAdjustment),
													yVal, widthCheckerboard * 2,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)],
									messageLPrevStereoCheckerboard2Half[retrieveIndexInDataAndMessage(
											((xValInCheckerboardPart * 2 + 1)
													+ checkerboardPartAdjustment),
											yVal, widthCheckerboard * 2,
											heightLevel, currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)]));
					val =
							__hadd2(val,
									__halves2half2(
											messageRPrevStereoCheckerboard2Half[retrieveIndexInDataAndMessage(
													((xValInCheckerboardPart * 2)
															- 1
															+ checkerboardPartAdjustment),
													yVal, widthCheckerboard * 2,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)],
									messageRPrevStereoCheckerboard2Half[retrieveIndexInDataAndMessage(
											((xValInCheckerboardPart * 2 + 1)
													- 1
													+ checkerboardPartAdjustment),
											yVal, widthCheckerboard * 2,
											heightLevel, currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)]));
					val = __hadd2(val, dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboardPart, yVal, widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);

					float valLow = __low2float ( val);
					float valHigh = __high2float ( val);
					if (valLow < best_val1)
					{
						best_val1 = valLow;
						bestDisparity1 = currentDisparity;
					}
					if (valHigh < best_val2)
					{
						best_val2 = valHigh;
						bestDisparity2 = currentDisparity;
					}
				}
				disparityBetweenImagesDevice[yVal*widthLevel + (xVal*2 - checkerboardPartAdjustment)] = bestDisparity1;
				if (((xVal*2) + 2) < widthLevel)
				{
					disparityBetweenImagesDevice[yVal*widthLevel + (xVal*2 - checkerboardPartAdjustment) + 2] = bestDisparity2;
				}
			}
			else
			{
				disparityBetweenImagesDevice[yVal * widthLevel + (xVal * 2 - checkerboardPartAdjustment)] =
						0;
				if (((xVal * 2) + 2) < widthLevel)
				{
					disparityBetweenImagesDevice[yVal * widthLevel + (xVal * 2 - checkerboardPartAdjustment)
							+ 2] = 0;
				}
			}
		}
		else //pixel from part 2 of checkerboard
		{
			int	checkerboardPartAdjustment = ((yVal + 1) % 2);
			half* messageLPrevStereoCheckerboard1Half = (half*)messageLPrevStereoCheckerboard1;
			half* messageRPrevStereoCheckerboard1Half = (half*)messageRPrevStereoCheckerboard1;

			if ((xVal >= 1) && (xVal < (widthLevel - 1)) && (yVal >= 1) && (yVal < (heightLevel - 1)))
			{
				// keep track of "best" disparity for current pixel
				int bestDisparity1 = 0;
				int bestDisparity2 = 0;
				float best_val1 = INF_BP;
				float best_val2 = INF_BP;
				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					half2 val =
							__hadd2(
									messageUPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
											xValInCheckerboardPart, (yVal + 1),
											widthCheckerboard, heightLevel,
											currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)],
											messageDPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
													xValInCheckerboardPart,
													(yVal - 1),
													widthCheckerboard,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)]);
					val =
							__hadd2(val,
									__halves2half2(
											messageLPrevStereoCheckerboard1Half[retrieveIndexInDataAndMessage(
													((xValInCheckerboardPart * 2)
															+ checkerboardPartAdjustment),
													yVal, widthCheckerboard * 2,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)],
									messageLPrevStereoCheckerboard1Half[retrieveIndexInDataAndMessage(
											((xValInCheckerboardPart * 2 + 1)
													+ checkerboardPartAdjustment),
											yVal, widthCheckerboard * 2,
											heightLevel, currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)]));
					val =
							__hadd2(val,
									__halves2half2(
											messageRPrevStereoCheckerboard1Half[retrieveIndexInDataAndMessage(
													((xValInCheckerboardPart * 2)
															- 1
															+ checkerboardPartAdjustment),
													yVal, widthCheckerboard * 2,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)],
									messageRPrevStereoCheckerboard1Half[retrieveIndexInDataAndMessage(
											((xValInCheckerboardPart * 2 + 1)
													- 1
													+ checkerboardPartAdjustment),
											yVal, widthCheckerboard * 2,
											heightLevel, currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)]));

					val =
							__hadd2(val,
									dataCostStereoCheckerboard2[retrieveIndexInDataAndMessage(
											xValInCheckerboardPart, yVal,
											widthCheckerboard, heightLevel,
											currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)]);

					float val1 = __low2float(val);
					float val2 = __high2float(val);
					if (val1 < best_val1) {
						best_val1 = val1;
						bestDisparity1 = currentDisparity;
					}
					if (val2 < best_val2) {
						best_val2 = val2;
						bestDisparity2 = currentDisparity;
					}
				}

				disparityBetweenImagesDevice[yVal * widthLevel + (xVal * 2 - checkerboardPartAdjustment)] =
						bestDisparity1;
				if (((xVal * 2) + 2) < widthLevel) {
					disparityBetweenImagesDevice[yVal * widthLevel + (xVal * 2 - checkerboardPartAdjustment)
							+ 2] = bestDisparity2;
				}
			}
			else
			{
				disparityBetweenImagesDevice[yVal * widthLevel + (xVal * 2 - checkerboardPartAdjustment)] =
						0;
				if (((xVal * 2) + 2) < widthLevel) {
					disparityBetweenImagesDevice[yVal * widthLevel + (xVal * 2 - checkerboardPartAdjustment)
							+ 2] = 0;
				}
			}
		}
	}
}
*/

/*template<>
__global__ void initializeBottomLevelDataStereo<half2>(levelProperties currentLevelProperties, float* image1PixelsDevice, float* image2PixelsDevice, half2* dataCostDeviceStereoCheckerboard1, half2* dataCostDeviceStereoCheckerboard2, float lambda_bp, float data_k_bp)
{
	// Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

	int xVal = bx * BLOCK_SIZE_WIDTH_BP + tx;
	int yVal = by * BLOCK_SIZE_HEIGHT_BP + ty;

	int indexVal;
	int imageCheckerboardWidth = getCheckerboardWidth<half2>(widthImages);
	int xInCheckerboard = xVal / 2;

	if (withinImageBounds(xInCheckerboard, yVal, imageCheckerboardWidth, heightImages))
	{
		int imageXPixelIndexStart = 0;
		int checkerboardNum = 1;

		//check which checkerboard data values for and make necessary adjustment to start
		if (((yVal) % 2) == 0) {
			if (((xVal) % 2) == 0) {
				checkerboardNum = 1;
			} else {
				checkerboardNum = 2;
			}
		} else {
			if (((xVal) % 2) == 0) {
				checkerboardNum = 2;
			} else {
				checkerboardNum = 1;
			}
		}

		imageXPixelIndexStart = xVal*2;
		if ((((yVal) % 2) == 0) && (checkerboardNum == 2)) {
			imageXPixelIndexStart -= 1;
		}
		if ((((yVal) % 2) == 1) && (checkerboardNum == 1)) {
			imageXPixelIndexStart -= 1;
		}

		//make sure that it is possible to check every disparity value
		if ((((imageXPixelIndexStart + 2) - (NUM_POSSIBLE_DISPARITY_VALUES-1)) >= 0))
		{
			for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
			{
				float currentPixelImage1_low = 0.0;
				float currentPixelImage2_low = 0.0;

				if ((((imageXPixelIndexStart) - (NUM_POSSIBLE_DISPARITY_VALUES-1)) >= 0))
				{
					if (withinImageBounds(imageXPixelIndexStart, yVal, widthImages,
							heightImages)) {
						currentPixelImage1_low = image1PixelsDevice[yVal
								* widthImages + imageXPixelIndexStart];
						currentPixelImage2_low = image2PixelsDevice[yVal
								* widthImages + (imageXPixelIndexStart - currentDisparity)];
					}
				}

				float currentPixelImage1_high = 0.0;
				float currentPixelImage2_high = 0.0;

				if (withinImageBounds(imageXPixelIndexStart + 2, yVal, widthImages,
						heightImages))
				{
					currentPixelImage1_high = image1PixelsDevice[yVal * widthImages
							+ (imageXPixelIndexStart + 2)];
					currentPixelImage2_high = image2PixelsDevice[yVal * widthImages
							+ ((imageXPixelIndexStart + 2) - currentDisparity)];
				}

				indexVal = retrieveIndexInDataAndMessage(xInCheckerboard, yVal, imageCheckerboardWidth, heightImages, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);

				half lowVal = (half)(lambda_bp * min(abs(currentPixelImage1_low - currentPixelImage2_low), data_k_bp));
				half highVal = (half)(lambda_bp * min(abs(currentPixelImage1_high - currentPixelImage2_high), data_k_bp));

				//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
				if (checkerboardNum == 1)
				{
					dataCostDeviceStereoCheckerboard1[indexVal] = __halves2half2(lowVal, highVal);
				}
				else
				{
					dataCostDeviceStereoCheckerboard2[indexVal] = __halves2half2(lowVal, highVal);
				}
			}
		}
		else
		{
			for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
			{
				indexVal = retrieveIndexInDataAndMessage(xInCheckerboard, yVal, imageCheckerboardWidth, heightImages, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);

				//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
				if (((xVal + yVal) % 2) == 0)
				{
					dataCostDeviceStereoCheckerboard1[indexVal] = getZeroVal<half2>();
				}
				else
				{
					dataCostDeviceStereoCheckerboard2[indexVal] = getZeroVal<half2>();
				}
			}
		}
	}
}*/
