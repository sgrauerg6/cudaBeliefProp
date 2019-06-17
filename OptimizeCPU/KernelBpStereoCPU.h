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

#include "bpStereoCudaParameters.cuh"

//indexing is performed in such a way so that the memory accesses as coalesced as much as possible
#if OPTIMIZED_INDEXING_SETTING == 1
#define RETRIEVE_INDEX_IN_DATA_OR_MESSAGE_ARRAY_EQUATION_CPU (yVal*width*totalNumDispVals + width*currentDisparity + xVal)
#else
#define RETRIEVE_INDEX_IN_DATA_OR_MESSAGE_ARRAY_EQUATION_CPU ((yVal*width + xVal)*totalNumDispVals + currentDisparity)
#endif
//#define RETRIEVE_INDEX_IN_DATA_OR_MESSAGE_ARRAY_EQUATION (height*width*currentDisparity + width*yVal + xVal)
//#define RETRIEVE_INDEX_IN_DATA_OR_MESSAGE_ARRAY_EQUATION ((yVal*width + xVal)*totalNumDispVals + currentDisparity)

class KernelBpStereoCPU
{
public:
	template<typename T>
	void printDataAndMessageValsAtPointKernelCPU(int xVal, int yVal, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
			T* messageUDeviceCurrentCheckerboard1,
			T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1,
			T* messageRDeviceCurrentCheckerboard1,
			T* messageUDeviceCurrentCheckerboard2,
			T* messageDDeviceCurrentCheckerboard2,
			T* messageLDeviceCurrentCheckerboard2,
			T* messageRDeviceCurrentCheckerboard2, int widthLevelCheckerboardPart,
			int heightLevel);

	template<typename T>
	void printDataAndMessageValsToPointKernelCPU(int xVal, int yVal, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
			T* messageUDeviceCurrentCheckerboard1,
			T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1,
			T* messageRDeviceCurrentCheckerboard1,
			T* messageUDeviceCurrentCheckerboard2,
			T* messageDDeviceCurrentCheckerboard2,
			T* messageLDeviceCurrentCheckerboard2,
			T* messageRDeviceCurrentCheckerboard2, int widthLevelCheckerboardPart,
			int heightLevel);

	//checks if the current point is within the image bounds
	static bool withinImageBoundsCPU(int xVal, int yVal, int width, int height);

	//retrieve the current 1-D index value of the given point at the given disparity in the data cost and message data
	static int retrieveIndexInDataAndMessageCPU(int xVal, int yVal, int width, int height, int currentDisparity, int totalNumDispVals, int offsetData = 0);

	template<typename T>
	static int getCheckerboardWidthCPU(int imageWidth);

	template<typename T>
	static T getZeroValCPU();

	//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	template<typename T>
	static void dtStereoCPU(T f[NUM_POSSIBLE_DISPARITY_VALUES]);

	// compute current message
	template<typename T>
	static void msgStereoCPU(T messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], T messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		T messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], T dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		T dst[NUM_POSSIBLE_DISPARITY_VALUES], T disc_k_bp);

	//device portion of the kernel function to run the current iteration of belief propagation where the input messages and data costs come in as array in local memory
	//and the output message values are stored in local memory
	template<typename T>
	static void runBPIterationInOutDataInLocalMemCPU(T prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES], T prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES], T prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES], T prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES], T dataMessage[NUM_POSSIBLE_DISPARITY_VALUES],
									T currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES], T currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES], T currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES], T currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES], T disc_k_bp);

	//device portion of the kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the
	//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
	//this function uses local memory to store the message and data values at each disparity in the intermediate step of current message computation
	//this function uses linear memory bound to textures to access the current data and message values
	template<typename T>
	static void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU(
			T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
			T* messageUDeviceCurrentCheckerboard1,
			T* messageDDeviceCurrentCheckerboard1,
			T* messageLDeviceCurrentCheckerboard1,
			T* messageRDeviceCurrentCheckerboard1,
			T* messageUDeviceCurrentCheckerboard2,
			T* messageDDeviceCurrentCheckerboard2,
			T* messageLDeviceCurrentCheckerboard2,
			T* messageRDeviceCurrentCheckerboard2,
			int widthLevelCheckerboardPart, int heightLevel,
			int checkerboardToUpdate, int xVal, int yVal, int offsetData, float disc_k_bp);

	//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
	//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
	template<typename T>
	static void initializeBottomLevelDataStereoCPU(float* image1PixelsDevice, float* image2PixelsDevice, T* dataCostDeviceStereoCheckerboard1, T* dataCostDeviceStereoCheckerboard2, int widthImages, int heightImages, float lambda_bp, float data_k_bp);


	//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
	template<typename T>
	static void initializeCurrentLevelDataStereoNoTexturesCPU(T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2, T* dataCostDeviceToWriteTo, int widthCurrentLevel, int heightCurrentLevel, int widthPrevLevel, int heightPrevLevel, int checkerboardPart, int offsetNum);

	//initialize the message values at each pixel of the current level to the default value
	template<typename T>
	static void initializeMessageValsToDefaultKernelCPU(T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1, T* messageLDeviceCurrentCheckerboard1,
													T* messageRDeviceCurrentCheckerboard1, T* messageUDeviceCurrentCheckerboard2, T* messageDDeviceCurrentCheckerboard2,
													T* messageLDeviceCurrentCheckerboard2, T* messageRDeviceCurrentCheckerboard2, int widthCheckerboardAtLevel, int heightLevel);

	//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
	//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
	template<typename T>
	static void runBPIterationUsingCheckerboardUpdatesNoTexturesCPU(T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
									T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1, T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
									T* messageUDeviceCurrentCheckerboard2, T* messageDDeviceCurrentCheckerboard2, T* messageLDeviceCurrentCheckerboard2,
									T* messageRDeviceCurrentCheckerboard2, int widthLevel, int heightLevel, int checkerboardPartUpdate, float disc_k_bp);

	//kernal to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
	//the kernal works from the point of view of the pixel at the prev level that is being copied to four different places
	template<typename T>
	static void copyPrevLevelToNextLevelBPCheckerboardStereoNoTexturesCPU(T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1, T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1, T* messageUPrevStereoCheckerboard2, T* messageDPrevStereoCheckerboard2, T* messageLPrevStereoCheckerboard2, T* messageRPrevStereoCheckerboard2, T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1, T* messageLDeviceCurrentCheckerboard1,
																T* messageRDeviceCurrentCheckerboard1, T* messageUDeviceCurrentCheckerboard2, T* messageDDeviceCurrentCheckerboard2,
																T* messageLDeviceCurrentCheckerboard2, T* messageRDeviceCurrentCheckerboard2, int widthCheckerboardPrevLevel, int heightLevelPrev, int widthCheckerboardNextLevel, int heightLevelNext,
																int checkerboardPart);

	//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
	template<typename T>
	static void retrieveOutputDisparityCheckerboardStereoNoTexturesCPU(T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2, T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1, T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1, T* messageUPrevStereoCheckerboard2, T* messageDPrevStereoCheckerboard2, T* messageLPrevStereoCheckerboard2, T* messageRPrevStereoCheckerboard2, float* disparityBetweenImagesDevice, int widthLevel, int heightLevel);
};

#endif //KERNAL_BP_STEREO_CPU_H
