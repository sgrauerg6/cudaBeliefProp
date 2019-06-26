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


#include "KernelBpStereoCPU.h"
#include "../SharedFuncts/SharedBPProcessingFuncts.h"


//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
template<typename T>
void KernelBpStereoCPU::initializeBottomLevelDataStereoCPU(levelProperties& currentLevelProperties, float* image1PixelsDevice, float* image2PixelsDevice, T* dataCostDeviceStereoCheckerboard1, T* dataCostDeviceStereoCheckerboard2, float lambda_bp, float data_k_bp)
{
	#pragma omp parallel for
	for (int val = 0; val < (currentLevelProperties.widthLevel*currentLevelProperties.heightLevel); val++)
	{
		int yVal = val / currentLevelProperties.widthLevel;
		int xVal = val % currentLevelProperties.widthLevel;

		initializeBottomLevelDataStereoPixel<T, T>(xVal, yVal,
				currentLevelProperties, image1PixelsDevice,
				image2PixelsDevice, dataCostDeviceStereoCheckerboard1,
				dataCostDeviceStereoCheckerboard2, lambda_bp,
				data_k_bp);
	}
}


//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<typename T>
void KernelBpStereoCPU::initializeCurrentLevelDataStereoCPU(int checkerboardPart, levelProperties& currentLevelProperties, levelProperties& prevLevelProperties, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2, T* dataCostDeviceToWriteTo, int offsetNum)
{
	#pragma omp parallel for
	for (int val = 0; val < (currentLevelProperties.widthCheckerboardLevel*currentLevelProperties.heightLevel); val++)
	{
		int yVal = val / currentLevelProperties.widthCheckerboardLevel;
		int xVal = val % currentLevelProperties.widthCheckerboardLevel;

		initializeCurrentLevelDataStereoPixel<T, T>(
				xVal, yVal, checkerboardPart,
				currentLevelProperties,
				prevLevelProperties, dataCostStereoCheckerboard1,
				dataCostStereoCheckerboard2, dataCostDeviceToWriteTo,
				offsetNum);
	}
}


//initialize the message values at each pixel of the current level to the default value
template<typename T>
void KernelBpStereoCPU::initializeMessageValsToDefaultKernelCPU(levelProperties& currentLevelProperties, T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1, T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1, T* messageUDeviceCurrentCheckerboard2, T* messageDDeviceCurrentCheckerboard2,
		T* messageLDeviceCurrentCheckerboard2, T* messageRDeviceCurrentCheckerboard2)
{
	#pragma omp parallel for
	for (int val = 0; val < (currentLevelProperties.widthCheckerboardLevel*currentLevelProperties.heightLevel); val++)
	{
		int yVal = val / currentLevelProperties.widthCheckerboardLevel;
		int xValInCheckerboard = val % currentLevelProperties.widthCheckerboardLevel;

		initializeMessageValsToDefaultKernelPixel(xValInCheckerboard,  yVal, currentLevelProperties, messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1,
				messageRDeviceCurrentCheckerboard1, messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2,
				messageLDeviceCurrentCheckerboard2, messageRDeviceCurrentCheckerboard2);
	}
}


template<typename T>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUNoPackedInstructions(
		int checkerboardPartUpdate, levelProperties& currentLevelProperties,
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
	int widthCheckerboardRunProcessing = currentLevelProperties.widthLevel / 2;

	//in cuda kernel storing data one at a time (though it is coalesced), so numDataInSIMDVector not relevant here and set to 1
	//still is a check if start of row is aligned
	bool dataAligned = MemoryAlignedAtDataStart(0, 1);

	#pragma omp parallel for
	for (int val = 0; val < (widthCheckerboardRunProcessing * currentLevelProperties.heightLevel);
			val++)
	{
		int yVal = val / widthCheckerboardRunProcessing;
		int xVal = val % widthCheckerboardRunProcessing;

		runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<T, T>(
				xVal, yVal, checkerboardPartUpdate, currentLevelProperties,
				dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
				messageUDeviceCurrentCheckerboard1,
				messageDDeviceCurrentCheckerboard1,
				messageLDeviceCurrentCheckerboard1,
				messageRDeviceCurrentCheckerboard1,
				messageUDeviceCurrentCheckerboard2,
				messageDDeviceCurrentCheckerboard2,
				messageLDeviceCurrentCheckerboard2,
				messageRDeviceCurrentCheckerboard2, disc_k_bp, 0, dataAligned);
	}
}

template<typename T, typename U>
void KernelBpStereoCPU::runBPIterationInOutDataInLocalMemCPUUseSIMDVectors(int xValStartProcessing,
		int yVal, levelProperties& currentLevelProperties,
		U prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES],
		U prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES],
		U prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES],
		U prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES],
		U dataMessage[NUM_POSSIBLE_DISPARITY_VALUES], T* currentUMessageArray,
		T* currentDMessageArray, T* currentLMessageArray,
		T* currentRMessageArray, U disc_k_bp_vector, bool dataAlignedAtxValStartProcessing)
{
	msgStereoSIMD<T, U>(xValStartProcessing, yVal, currentLevelProperties, prevUMessage,
			prevLMessage, prevRMessage, dataMessage, currentUMessageArray,
			disc_k_bp_vector, dataAlignedAtxValStartProcessing);

	msgStereoSIMD<T, U>(xValStartProcessing, yVal, currentLevelProperties, prevDMessage,
			prevLMessage, prevRMessage, dataMessage, currentDMessageArray,
			disc_k_bp_vector, dataAlignedAtxValStartProcessing);

	msgStereoSIMD<T, U>(xValStartProcessing, yVal, currentLevelProperties, prevUMessage,
			prevDMessage, prevRMessage, dataMessage, currentRMessageArray,
			disc_k_bp_vector, dataAlignedAtxValStartProcessing);

	msgStereoSIMD<T, U>(xValStartProcessing, yVal, currentLevelProperties, prevUMessage,
			prevDMessage, prevLMessage, dataMessage, currentLMessageArray,
			disc_k_bp_vector, dataAlignedAtxValStartProcessing);
}

template<typename T, typename U>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess(
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
		int numDataInSIMDVector)
{
	int widthCheckerboardRunProcessing = currentLevelProperties.widthLevel / 2;
	U disc_k_bp_vector = createSIMDVectorSameData<U>(disc_k_bp);

#pragma omp parallel for
	for (int yVal = 1; yVal < currentLevelProperties.heightLevel - 1; yVal++) {
		//checkerboardAdjustment used for indexing into current checkerboard to update
		int checkerboardAdjustment;
		if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
			checkerboardAdjustment = ((yVal) % 2);
		} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
		{
			checkerboardAdjustment = ((yVal + 1) % 2);
		}

		int startX = (checkerboardAdjustment == 1) ? 0 : 1;
		int endFinal = std::min(
				currentLevelProperties.widthCheckerboardLevel
						- checkerboardAdjustment,
				widthCheckerboardRunProcessing);
		int endXSIMDVectorStart = (endFinal / numDataInSIMDVector)
				* numDataInSIMDVector - numDataInSIMDVector;

		for (int xVal = 0; xVal < endFinal; xVal += numDataInSIMDVector) {
			int xValProcess = xVal;

			//need this check first for case where endXAvxStart is 0 and startX is 1
			//if past the last AVX start (since the next one would go beyond the row), set to numDataInSIMDVector from the final pixel so processing the last numDataInAvxVector in avx
			//may be a few pixels that are computed twice but that's OK
			if (xValProcess > endXSIMDVectorStart) {
				xValProcess = endFinal - numDataInSIMDVector;
			}

			//not processing at x=0 if startX is 1 (this will cause this processing to be less aligned than ideal for this iteration)
			xValProcess = std::max(startX, xValProcess);

			//check if the memory is aligned for AVX instructions at xValProcess location
			bool dataAlignedAtXValProcess = MemoryAlignedAtDataStart(
					xValProcess, numDataInSIMDVector);

			//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
			//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
			//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
			U dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			U prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			U prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			U prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			U prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

			//load using aligned instructions when possible
			if (dataAlignedAtXValProcess) {
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						dataMessage[currentDisparity] = loadPackedDataAligned<T,
								U>(xValProcess, yVal, currentDisparity,
								currentLevelProperties,
								dataCostStereoCheckerboard1);
						prevUMessage[currentDisparity] = loadPackedDataAligned<
								T, U>(xValProcess, yVal + 1, currentDisparity,
								currentLevelProperties,
								messageUDeviceCurrentCheckerboard2);
						prevDMessage[currentDisparity] = loadPackedDataAligned<
								T, U>(xValProcess, yVal - 1, currentDisparity,
								currentLevelProperties,
								messageDDeviceCurrentCheckerboard2);
						prevLMessage[currentDisparity] =
								loadPackedDataUnaligned<T, U>(
										xValProcess + checkerboardAdjustment,
										yVal, currentDisparity,
										currentLevelProperties,
										messageLDeviceCurrentCheckerboard2);
						prevRMessage[currentDisparity] =
								loadPackedDataUnaligned<T, U>(
										(xValProcess - 1)
												+ checkerboardAdjustment, yVal,
										currentDisparity,
										currentLevelProperties,
										messageRDeviceCurrentCheckerboard2);
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] = loadPackedDataAligned<T,
								U>(xValProcess, yVal, currentDisparity,
								currentLevelProperties,
								dataCostStereoCheckerboard2);
						prevUMessage[currentDisparity] = loadPackedDataAligned<
								T, U>(xValProcess, yVal + 1, currentDisparity,
								currentLevelProperties,
								messageUDeviceCurrentCheckerboard1);
						prevDMessage[currentDisparity] = loadPackedDataAligned<
								T, U>(xValProcess, yVal - 1, currentDisparity,
								currentLevelProperties,
								messageDDeviceCurrentCheckerboard1);
						prevLMessage[currentDisparity] =
								loadPackedDataUnaligned<T, U>(
										xValProcess + checkerboardAdjustment,
										yVal, currentDisparity,
										currentLevelProperties,
										messageLDeviceCurrentCheckerboard1);
						prevRMessage[currentDisparity] =
								loadPackedDataUnaligned<T, U>(
										(xValProcess - 1)
												+ checkerboardAdjustment, yVal,
										currentDisparity,
										currentLevelProperties,
										messageRDeviceCurrentCheckerboard1);
					}
				}
			} else {
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						dataMessage[currentDisparity] = loadPackedDataUnaligned<
								T, U>(xValProcess, yVal, currentDisparity,
								currentLevelProperties,
								dataCostStereoCheckerboard1);
						prevUMessage[currentDisparity] =
								loadPackedDataUnaligned<T, U>(xValProcess,
										yVal + 1, currentDisparity,
										currentLevelProperties,
										messageUDeviceCurrentCheckerboard2);
						prevDMessage[currentDisparity] =
								loadPackedDataUnaligned<T, U>(xValProcess,
										yVal - 1, currentDisparity,
										currentLevelProperties,
										messageDDeviceCurrentCheckerboard2);
						prevLMessage[currentDisparity] =
								loadPackedDataUnaligned<T, U>(
										xValProcess + checkerboardAdjustment,
										yVal, currentDisparity,
										currentLevelProperties,
										messageLDeviceCurrentCheckerboard2);
						prevRMessage[currentDisparity] =
								loadPackedDataUnaligned<T, U>(
										(xValProcess - 1)
												+ checkerboardAdjustment, yVal,
										currentDisparity,
										currentLevelProperties,
										messageRDeviceCurrentCheckerboard2);
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] = loadPackedDataUnaligned<
								T, U>(xValProcess, yVal, currentDisparity,
								currentLevelProperties,
								dataCostStereoCheckerboard2);
						prevUMessage[currentDisparity] =
								loadPackedDataUnaligned<T, U>(xValProcess,
										yVal + 1, currentDisparity,
										currentLevelProperties,
										messageUDeviceCurrentCheckerboard1);
						prevDMessage[currentDisparity] =
								loadPackedDataUnaligned<T, U>(xValProcess,
										yVal - 1, currentDisparity,
										currentLevelProperties,
										messageDDeviceCurrentCheckerboard1);
						prevLMessage[currentDisparity] =
								loadPackedDataUnaligned<T, U>(
										xValProcess + checkerboardAdjustment,
										yVal, currentDisparity,
										currentLevelProperties,
										messageLDeviceCurrentCheckerboard1);
						prevRMessage[currentDisparity] =
								loadPackedDataUnaligned<T, U>(
										(xValProcess - 1)
												+ checkerboardAdjustment, yVal,
										currentDisparity,
										currentLevelProperties,
										messageRDeviceCurrentCheckerboard1);
					}
				}
			}

			if (checkerboardToUpdate == CHECKERBOARD_PART_1)
			{
				runBPIterationInOutDataInLocalMemCPUUseSIMDVectors<T, U>(xValProcess,
						yVal, currentLevelProperties, prevUMessage,
						prevDMessage, prevLMessage, prevRMessage, dataMessage,
						messageUDeviceCurrentCheckerboard1,
						messageDDeviceCurrentCheckerboard1,
						messageLDeviceCurrentCheckerboard1,
						messageRDeviceCurrentCheckerboard1, disc_k_bp_vector,
						dataAlignedAtXValProcess);
			}
			else
			{
				runBPIterationInOutDataInLocalMemCPUUseSIMDVectors<T, U>(xValProcess,
						yVal, currentLevelProperties, prevUMessage,
						prevDMessage, prevLMessage, prevRMessage, dataMessage,
						messageUDeviceCurrentCheckerboard2,
						messageDDeviceCurrentCheckerboard2,
						messageLDeviceCurrentCheckerboard2,
						messageRDeviceCurrentCheckerboard2, disc_k_bp_vector,
						dataAlignedAtXValProcess);
			}
		}
	}
}

//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
template<typename T>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPU(int checkerboardToUpdate, levelProperties& currentLevelProperties, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
		T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1, T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
		T* messageUDeviceCurrentCheckerboard2, T* messageDDeviceCurrentCheckerboard2, T* messageLDeviceCurrentCheckerboard2,
		T* messageRDeviceCurrentCheckerboard2, float disc_k_bp)
{

#if CPU_OPTIMIZATION_SETTING == USE_AVX_256

	//only use AVX-256 if width of processing checkerboard is over 20
	if (currentLevelProperties.widthCheckerboardLevel > 10)
	{
		runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors<T>(checkerboardToUpdate, currentLevelProperties, dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
				messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
				messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
				messageRDeviceCurrentCheckerboard2, disc_k_bp);
	}
	else
	{
		runBPIterationUsingCheckerboardUpdatesCPUNoPackedInstructions<T>(checkerboardToUpdate, currentLevelProperties, dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
							messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
							messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
							messageRDeviceCurrentCheckerboard2, disc_k_bp);
	}

#elif CPU_OPTIMIZATION_SETTING == USE_AVX_512

	//only use AVX-512 if width of processing checkerboard is over 20
	if (currentLevelProperties.widthCheckerboardLevel > 20)
	{
		runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors<T>(checkerboardToUpdate, currentLevelProperties, dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
						messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
						messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
						messageRDeviceCurrentCheckerboard2, disc_k_bp);
	}
	else
	{
		runBPIterationUsingCheckerboardUpdatesCPUNoPackedInstructions<T>(checkerboardToUpdate, currentLevelProperties, dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
					messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
					messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
					messageRDeviceCurrentCheckerboard2, disc_k_bp);
	}

#elif CPU_OPTIMIZATION_SETTING == USE_NEON

	if (currentLevelProperties.widthCheckerboardLevel > 5)
	{
		runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors<T>(checkerboardToUpdate, currentLevelProperties, dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
						messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
						messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
						messageRDeviceCurrentCheckerboard2, disc_k_bp);
	}
	else
	{
		runBPIterationUsingCheckerboardUpdatesCPUNoPackedInstructions<T>(checkerboardToUpdate, currentLevelProperties, dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
								messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
								messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
								messageRDeviceCurrentCheckerboard2, disc_k_bp);
	}

#else

	runBPIterationUsingCheckerboardUpdatesCPUNoPackedInstructions<T>(checkerboardToUpdate, currentLevelProperties, dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
			messageRDeviceCurrentCheckerboard2, disc_k_bp);

#endif
}


//kernal to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernal works from the point of view of the pixel at the prev level that is being copied to four different places
template<typename T>
void KernelBpStereoCPU::copyPrevLevelToNextLevelBPCheckerboardStereoCPU(
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
		T* messageRDeviceCurrentCheckerboard2)
{
	#pragma omp parallel for
	for (int val = 0; val < (currentLevelProperties.widthCheckerboardLevel*currentLevelProperties.heightLevel); val++)
	{
		int yVal = val / currentLevelProperties.widthCheckerboardLevel;
		int xVal = val % currentLevelProperties.widthCheckerboardLevel;

		copyPrevLevelToNextLevelBPCheckerboardStereoPixel<T>(xVal, yVal,
				checkerboardPart, currentLevelProperties,
				nextLevelProperties,
				messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
				messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
				messageUPrevStereoCheckerboard2, messageDPrevStereoCheckerboard2,
				messageLPrevStereoCheckerboard2, messageRPrevStereoCheckerboard2,
				messageUDeviceCurrentCheckerboard1,
				messageDDeviceCurrentCheckerboard1,
				messageLDeviceCurrentCheckerboard1,
				messageRDeviceCurrentCheckerboard1,
				messageUDeviceCurrentCheckerboard2,
				messageDDeviceCurrentCheckerboard2,
				messageLDeviceCurrentCheckerboard2,
				messageRDeviceCurrentCheckerboard2);
	}
}

template<typename T>
void KernelBpStereoCPU::retrieveOutputDisparityCheckerboardStereoOptimizedCPU(levelProperties& currentLevelProperties, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2, T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1, T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1, T* messageUPrevStereoCheckerboard2, T* messageDPrevStereoCheckerboard2, T* messageLPrevStereoCheckerboard2, T* messageRPrevStereoCheckerboard2, float* disparityBetweenImagesDevice)
{
	#pragma omp parallel for
	for (int val = 0; val < (currentLevelProperties.widthCheckerboardLevel*currentLevelProperties.heightLevel); val++)
	{
		int yVal = val / currentLevelProperties.widthCheckerboardLevel;
		int xVal = val % currentLevelProperties.widthCheckerboardLevel;

		retrieveOutputDisparityCheckerboardStereoOptimizedPixel<T, T>(xVal, yVal,
				currentLevelProperties, dataCostStereoCheckerboard1,
				dataCostStereoCheckerboard2, messageUPrevStereoCheckerboard1,
				messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
				messageRPrevStereoCheckerboard1, messageUPrevStereoCheckerboard2,
				messageDPrevStereoCheckerboard2, messageLPrevStereoCheckerboard2,
				messageRPrevStereoCheckerboard2,
				disparityBetweenImagesDevice);
	}
}


template<typename T>
void KernelBpStereoCPU::printDataAndMessageValsAtPointKernelCPU(int xVal, int yVal, levelProperties& currentLevelProperties, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1,
		T* messageUDeviceCurrentCheckerboard2,
		T* messageDDeviceCurrentCheckerboard2,
		T* messageLDeviceCurrentCheckerboard2,
		T* messageRDeviceCurrentCheckerboard2)
{
	if (((xVal + yVal) % 2) == 0) {
		printf("xVal: %d\n", xVal);
		printf("yVal: %d\n", yVal);
		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			printf("DISP: %d\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
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
					(float) messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard2[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
		}
	}
}

template<typename T>
void KernelBpStereoCPU::printDataAndMessageValsToPointKernelCPU(int xVal, int yVal, levelProperties& currentLevelProperties, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1,
		T* messageUDeviceCurrentCheckerboard2,
		T* messageDDeviceCurrentCheckerboard2,
		T* messageLDeviceCurrentCheckerboard2,
		T* messageRDeviceCurrentCheckerboard2)
{
	int checkerboardAdjustment;
	if (((xVal + yVal) % 2) == 0)
		{
			checkerboardAdjustment = ((yVal)%2);
		}
		else //checkerboardToUpdate == CHECKERBOARD_PART_2
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
						(float) messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
								xVal / 2, yVal + 1, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageDPrevStereoCheckerboard: %f \n",
						(float) messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
								xVal / 2, yVal - 1, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageLPrevStereoCheckerboard: %f \n",
						(float) messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
								xVal / 2 + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageRPrevStereoCheckerboard: %f \n",
						(float) messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
								(xVal / 2 - 1) + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("dataCostStereoCheckerboard: %f \n",
						(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
								xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
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
								xVal / 2, yVal + 1, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageDPrevStereoCheckerboard: %f \n",
						(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
								xVal / 2, yVal - 1, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageLPrevStereoCheckerboard: %f \n",
						(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
								xVal / 2 + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageRPrevStereoCheckerboard: %f \n",
						(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
								(xVal / 2 - 1) + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("dataCostStereoCheckerboard: %f \n",
						(float) dataCostStereoCheckerboard2[retrieveIndexInDataAndMessage(
								xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
		}
}

