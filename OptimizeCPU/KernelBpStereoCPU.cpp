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


#include "bpStereoCudaParameters.cuh"
#include "KernelBpStereoCPU.h"
#include <math.h>
#include <omp.h>

//checks if the current point is within the image bounds
bool KernelBpStereoCPU::withinImageBoundsCPU(int xVal, int yVal, int width, int height)
{
	return ((xVal >= 0) && (xVal < width) && (yVal >= 0) && (yVal < height));
}

//retrieve the current 1-D index value of the given point at the given disparity in the data cost and message data
int KernelBpStereoCPU::retrieveIndexInDataAndMessageCPU(int xVal, int yVal, int width, int height, int currentDisparity, int totalNumDispVals, int offsetData)
{
	return RETRIEVE_INDEX_IN_DATA_OR_MESSAGE_ARRAY_EQUATION_CPU + offsetData;
}

template<typename T>
int KernelBpStereoCPU::getCheckerboardWidthCPU(int imageWidth)
{
	return (int)ceil(((float)imageWidth) / 2.0);
}

template<typename T>
T KernelBpStereoCPU::getZeroValCPU()
{
	return (T)0.0;
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<typename T>
void KernelBpStereoCPU::dtStereoCPU(T f[NUM_POSSIBLE_DISPARITY_VALUES])
{
	T prev;
	for (int currentDisparity = 1; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		prev = f[currentDisparity-1] + (T)1.0;
		if (prev < f[currentDisparity])
			f[currentDisparity] = prev;
	}

	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		prev = f[currentDisparity+1] + (T)1.0;
		if (prev < f[currentDisparity])
			f[currentDisparity] = prev;
	}
}

// compute current message
template<typename T>
void KernelBpStereoCPU::msgStereoCPU(T messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], T messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
	T messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], T dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
	T dst[NUM_POSSIBLE_DISPARITY_VALUES], T disc_k_bp)
{
	// aggregate and find min
	T minimum = INF_BP;

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		if (dst[currentDisparity] < minimum)
			minimum = dst[currentDisparity];
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoCPU<T>(dst);

	// truncate 
	minimum += disc_k_bp;

	// normalize
	T valToNormalize = 0;

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}

		valToNormalize += dst[currentDisparity];
	}
	
	valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++) 
		dst[currentDisparity] -= valToNormalize;
}


//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
template<typename T>
void KernelBpStereoCPU::initializeBottomLevelDataStereoCPU(float* image1PixelsDevice, float* image2PixelsDevice, T* dataCostDeviceStereoCheckerboard1, T* dataCostDeviceStereoCheckerboard2, int widthImages, int heightImages, float lambda_bp, float data_k_bp)
{
	int imageCheckerboardWidth = getCheckerboardWidthCPU<T>(widthImages);

	#pragma omp parallel for
	for (int val = 0; val < widthImages*heightImages; val++)
	{
		int yVal = val / widthImages;
		int xVal = val % widthImages;
	/*for (int yVal = 0; yVal < heightImages; yVal++)
	{
		#pragma omp parallel for
		for (int xVal = 0; xVal < widthImages; xVal++)
		{*/
			int indexVal;
			int xInCheckerboard = xVal / 2;

			if (withinImageBoundsCPU(xInCheckerboard, yVal, imageCheckerboardWidth, heightImages))
			{
				//make sure that it is possible to check every disparity value
				if ((xVal - (NUM_POSSIBLE_DISPARITY_VALUES-1)) >= 0)
				{
					for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
					{
						float currentPixelImage1 = 0.0f;
						float currentPixelImage2 = 0.0f;

						if (withinImageBoundsCPU(xVal, yVal, widthImages, heightImages))
						{
							currentPixelImage1 = image1PixelsDevice[yVal * widthImages
									+ xVal];
							currentPixelImage2 = image2PixelsDevice[yVal * widthImages
									+ (xVal - currentDisparity)];
						}

						indexVal = retrieveIndexInDataAndMessageCPU(xInCheckerboard, yVal, imageCheckerboardWidth, heightImages, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);

						//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
						if (((xVal + yVal) % 2) == 0)
						{
							dataCostDeviceStereoCheckerboard1[indexVal] = (T)(lambda_bp * std::min(((T)abs(currentPixelImage1 - currentPixelImage2)), (T)data_k_bp));
						}
						else
						{
							dataCostDeviceStereoCheckerboard2[indexVal] = (T)(lambda_bp * std::min(((T)abs(currentPixelImage1 - currentPixelImage2)), (T)data_k_bp));
						}
					}
				}
				else
				{
					for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
					{
						indexVal = retrieveIndexInDataAndMessageCPU(xInCheckerboard, yVal, imageCheckerboardWidth, heightImages, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);

						//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
						if (((xVal + yVal) % 2) == 0)
						{
							dataCostDeviceStereoCheckerboard1[indexVal] = getZeroValCPU<T>();
						}
						else
						{
							dataCostDeviceStereoCheckerboard2[indexVal] = getZeroValCPU<T>();
						}
					}
				}
			}
		//}
	}
}


template<typename T>
void KernelBpStereoCPU::printDataAndMessageValsAtPointKernelCPU(int xVal, int yVal, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1,
		T* messageUDeviceCurrentCheckerboard2,
		T* messageDDeviceCurrentCheckerboard2,
		T* messageLDeviceCurrentCheckerboard2,
		T* messageRDeviceCurrentCheckerboard2, int widthLevelCheckerboardPart,
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
					(float) messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
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
					(float) messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
		}
	}
}


template<typename T>
void KernelBpStereoCPU::printDataAndMessageValsToPointKernelCPU(int xVal, int yVal, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1,
		T* messageUDeviceCurrentCheckerboard2,
		T* messageDDeviceCurrentCheckerboard2,
		T* messageLDeviceCurrentCheckerboard2,
		T* messageRDeviceCurrentCheckerboard2, int widthLevelCheckerboardPart,
		int heightLevel)
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
						(float) messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
								xVal / 2, yVal + 1, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageDPrevStereoCheckerboard: %f \n",
						(float) messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
								xVal / 2, yVal - 1, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageLPrevStereoCheckerboard: %f \n",
						(float) messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
								xVal / 2 + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageRPrevStereoCheckerboard: %f \n",
						(float) messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
								(xVal / 2 - 1) + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("dataCostStereoCheckerboard: %f \n",
						(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
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
						(float) messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
								xVal / 2, yVal + 1, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageDPrevStereoCheckerboard: %f \n",
						(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
								xVal / 2, yVal - 1, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageLPrevStereoCheckerboard: %f \n",
						(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
								xVal / 2 + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageRPrevStereoCheckerboard: %f \n",
						(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
								(xVal / 2 - 1) + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("dataCostStereoCheckerboard: %f \n",
						(float) dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
								xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
		}
}


//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<typename T>
void KernelBpStereoCPU::initializeCurrentLevelDataStereoNoTexturesCPU(T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2, T* dataCostDeviceToWriteTo, int widthCurrentLevel, int heightCurrentLevel, int widthPrevLevel, int heightPrevLevel, int checkerboardPart, int offsetNum)
{
	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<T>(widthCurrentLevel);
	int widthCheckerboardPrevLevel = getCheckerboardWidthCPU<T>(widthPrevLevel);

	#pragma omp parallel for
	for (int val = 0; val < widthCheckerboardCurrentLevel*heightCurrentLevel; val++)
	{
		int yVal = val / widthCheckerboardCurrentLevel;
		int xVal = val % widthCheckerboardCurrentLevel;

	/*for (int yVal = 0; yVal < heightCurrentLevel; yVal++)
	{
		#pragma omp parallel for
		for (int xVal = 0; xVal < widthCheckerboardCurrentLevel; xVal++)
		{*/
			//if (withinImageBoundsCPU(xVal, yVal, widthCheckerboardCurrentLevel,
			//		heightCurrentLevel))
			{
				//add 1 or 0 to the x-value depending on checkerboard part and row adding to; CHECKERBOARD_PART_1 with slot at (0, 0) has adjustment of 0 in row 0,
				//while CHECKERBOARD_PART_2 with slot at (0, 1) has adjustment of 1 in row 0
				int checkerboardPartAdjustment = 0;

				if (checkerboardPart == CHECKERBOARD_PART_1) {
					checkerboardPartAdjustment = (yVal % 2);
				} else if (checkerboardPart == CHECKERBOARD_PART_2) {
					checkerboardPartAdjustment = ((yVal + 1) % 2);
				}

				//the corresponding x-values at the "lower" level depends on which checkerboard the pixel is in
				int xValPrev = xVal * 2 + checkerboardPartAdjustment;

				if (withinImageBoundsCPU(xValPrev, (yVal * 2 + 1),
						widthCheckerboardPrevLevel, heightPrevLevel)) {
					for (int currentDisparity = 0;
							currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
							currentDisparity++) {
						dataCostDeviceToWriteTo[retrieveIndexInDataAndMessageCPU(xVal,
								yVal, widthCheckerboardCurrentLevel,
								heightCurrentLevel, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES)] =
								(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
										xValPrev, (yVal * 2),
										widthCheckerboardPrevLevel,
										heightPrevLevel, currentDisparity,
										NUM_POSSIBLE_DISPARITY_VALUES,
										offsetNum)]
										+ dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValPrev, (yVal * 2),
												widthCheckerboardPrevLevel,
												heightPrevLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES,
												offsetNum)]
										+ dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValPrev, (yVal * 2 + 1),
												widthCheckerboardPrevLevel,
												heightPrevLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES,
												offsetNum)]
										+ dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValPrev, (yVal * 2 + 1),
												widthCheckerboardPrevLevel,
												heightPrevLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES,
												offsetNum)]);
					}
				}
			}
		//}
	}
}


//initialize the message values at each pixel of the current level to the default value
template<typename T>
void KernelBpStereoCPU::initializeMessageValsToDefaultKernelCPU(T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1, T* messageLDeviceCurrentCheckerboard1,
												T* messageRDeviceCurrentCheckerboard1, T* messageUDeviceCurrentCheckerboard2, T* messageDDeviceCurrentCheckerboard2,
												T* messageLDeviceCurrentCheckerboard2, T* messageRDeviceCurrentCheckerboard2, int widthCheckerboardAtLevel, int heightLevel)
{
	#pragma omp parallel for
	for (int val = 0; val < widthCheckerboardAtLevel*heightLevel; val++)
	{
		int yVal = val / widthCheckerboardAtLevel;
		int xValInCheckerboard = val % widthCheckerboardAtLevel;
	/*for (int yVal = 0; yVal < heightLevel; yVal++)
	{
		#pragma omp parallel for
		for (int xValInCheckerboard = 0;
				xValInCheckerboard < widthCheckerboardAtLevel;
				xValInCheckerboard++)
		{*/
			//if (withinImageBoundsCPU(xValInCheckerboard, yVal,
			//		widthCheckerboardAtLevel, heightLevel))
			{
				//initialize message values in both checkerboards

				//set the message value at each pixel for each disparity to 0
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, widthCheckerboardAtLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroValCPU<T>();
					messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, widthCheckerboardAtLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroValCPU<T>();
					messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, widthCheckerboardAtLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroValCPU<T>();
					messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, widthCheckerboardAtLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroValCPU<T>();
				}

				//retrieve the previous message value at each movement at each pixel
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, widthCheckerboardAtLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroValCPU<T>();
					messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, widthCheckerboardAtLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroValCPU<T>();
					messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, widthCheckerboardAtLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroValCPU<T>();
					messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, widthCheckerboardAtLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroValCPU<T>();
				}
			}
		//}
	}
}


//device portion of the kernel function to run the current iteration of belief propagation where the input messages and data costs come in as array in local memory
//and the output message values are stored in local memory
template<typename T>
void KernelBpStereoCPU::runBPIterationInOutDataInLocalMemCPU(T prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES], T prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES], T prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES], T prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES], T dataMessage[NUM_POSSIBLE_DISPARITY_VALUES],
								T currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES], T currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES], T currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES], T currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES], T disc_k_bp)
{
	msgStereoCPU<T>(prevUMessage, prevLMessage, prevRMessage, dataMessage,
			currentUMessage, disc_k_bp);

	msgStereoCPU<T>(prevDMessage, prevLMessage, prevRMessage, dataMessage,
			currentDMessage, disc_k_bp);

	msgStereoCPU<T>(prevUMessage, prevDMessage, prevRMessage, dataMessage,
			currentRMessage, disc_k_bp);

	msgStereoCPU<T>(prevUMessage, prevDMessage, prevLMessage, dataMessage,
			currentLMessage, disc_k_bp);
}


//device portion of the kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the
//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//this function uses local memory to store the message and data values at each disparity in the intermediate step of current message computation
//this function uses linear memory bound to textures to access the current data and message values
template<typename T>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU(
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
		int checkerboardToUpdate, int xVal, int yVal, int offsetData, float disc_k_bp)
{
	int indexWriteTo;
	int checkerboardAdjustment;

	//checkerboardAdjustment used for indexing into current checkerboard to update
	if (checkerboardToUpdate == CHECKERBOARD_PART_1)
	{
		checkerboardAdjustment = ((yVal)%2);
	}
	else //checkerboardToUpdate == CHECKERBOARD_PART_2
	{
		checkerboardAdjustment = ((yVal+1)%2);
	}

	//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
	//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
	//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
	if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - checkerboardAdjustment)) && (yVal > 0) && (yVal < (heightLevel - 1)))
	{
		T prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		T prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		T prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		T prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		T dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		{
			if (checkerboardToUpdate == CHECKERBOARD_PART_1)
			{
				dataMessage[currentDisparity] = dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)];
				prevUMessage[currentDisparity] = messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];
				prevDMessage[currentDisparity] = messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];
				prevLMessage[currentDisparity] = messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];
				prevRMessage[currentDisparity] = messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];
			}
			else //checkerboardToUpdate == CHECKERBOARD_PART_2
			{
				dataMessage[currentDisparity] = dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)];
				prevUMessage[currentDisparity] = messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];
				prevDMessage[currentDisparity] = messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];
				prevLMessage[currentDisparity] = messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];
				prevRMessage[currentDisparity] = messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];
			}
		}

		T currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		T currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		T currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		T currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		//uses the previous message values and data cost to calculate the current message values and store the results
		runBPIterationInOutDataInLocalMemCPU<T>(prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							currentUMessage, currentDMessage, currentLMessage, currentRMessage, (T)disc_k_bp);

		//write the calculated message values to global memory
		for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		{
			indexWriteTo = retrieveIndexInDataAndMessageCPU(xVal, yVal, widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
			if (checkerboardToUpdate == CHECKERBOARD_PART_1)
			{
				messageUDeviceCurrentCheckerboard1[indexWriteTo] = currentUMessage[currentDisparity];
				messageDDeviceCurrentCheckerboard1[indexWriteTo] = currentDMessage[currentDisparity];
				messageLDeviceCurrentCheckerboard1[indexWriteTo] = currentLMessage[currentDisparity];
				messageRDeviceCurrentCheckerboard1[indexWriteTo] = currentRMessage[currentDisparity];
			}
			else //checkerboardToUpdate == CHECKERBOARD_PART_2
			{
				messageUDeviceCurrentCheckerboard2[indexWriteTo] = currentUMessage[currentDisparity];
				messageDDeviceCurrentCheckerboard2[indexWriteTo] = currentDMessage[currentDisparity];
				messageLDeviceCurrentCheckerboard2[indexWriteTo] = currentLMessage[currentDisparity];
				messageRDeviceCurrentCheckerboard2[indexWriteTo] = currentRMessage[currentDisparity];
			}
		}
	}
}


//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
template<typename T>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPU(T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
								T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1, T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
								T* messageUDeviceCurrentCheckerboard2, T* messageDDeviceCurrentCheckerboard2, T* messageLDeviceCurrentCheckerboard2,
								T* messageRDeviceCurrentCheckerboard2, int widthLevel, int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{
	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<T>(widthLevel);

	#pragma omp parallel for
	for (int val = 0; val < (widthLevel / 2)*heightLevel; val++)
	{
		int yVal = val / (widthLevel / 2);
		int xVal = val % (widthLevel / 2);
	/*for (int yVal = 0; yVal < heightLevel; yVal++)
	{
		#pragma omp parallel for
		for (int xVal = 0; xVal < widthLevel / 2; xVal++)
		{*/
			//if (withinImageBoundsCPU(xVal, yVal, widthLevel / 2, heightLevel)) {
				runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<
						T>(dataCostStereoCheckerboard1,
						dataCostStereoCheckerboard2,
						messageUDeviceCurrentCheckerboard1,
						messageDDeviceCurrentCheckerboard1,
						messageLDeviceCurrentCheckerboard1,
						messageRDeviceCurrentCheckerboard1,
						messageUDeviceCurrentCheckerboard2,
						messageDDeviceCurrentCheckerboard2,
						messageLDeviceCurrentCheckerboard2,
						messageRDeviceCurrentCheckerboard2,
						widthCheckerboardCurrentLevel, heightLevel,
						checkerboardPartUpdate, xVal, yVal, 0, disc_k_bp);
			//}
		//}
	}
}


//kernal to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernal works from the point of view of the pixel at the prev level that is being copied to four different places
template<typename T>
void KernelBpStereoCPU::copyPrevLevelToNextLevelBPCheckerboardStereoNoTexturesCPU(
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
		T* messageRDeviceCurrentCheckerboard2, int widthCheckerboardPrevLevel,
		int heightLevelPrev, int widthCheckerboardNextLevel,
		int heightLevelNext, int checkerboardPart)
{
	#pragma omp parallel for
	for (int val = 0; val < widthCheckerboardPrevLevel*heightLevelPrev; val++)
	{
		int yVal = val / widthCheckerboardPrevLevel;
		int xVal = val % widthCheckerboardPrevLevel;
	/*for (int yVal = 0; yVal < heightLevelPrev; yVal++)
	{
		#pragma omp parallel for
		for (int xVal = 0; xVal < widthCheckerboardPrevLevel; xVal++)
		{*/
			/*if (withinImageBoundsCPU(xVal, yVal, widthCheckerboardPrevLevel,
					heightLevelPrev))*/ {
				int heightCheckerboardNextLevel = heightLevelNext;

				int indexCopyTo;
				int indexCopyFrom;

				int checkerboardPartAdjustment;

				T prevValU;
				T prevValD;
				T prevValL;
				T prevValR;

				if (checkerboardPart == CHECKERBOARD_PART_1) {
					checkerboardPartAdjustment = (yVal % 2);
				} else if (checkerboardPart == CHECKERBOARD_PART_2) {
					checkerboardPartAdjustment = ((yVal + 1) % 2);
				}

				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					indexCopyFrom = retrieveIndexInDataAndMessageCPU(xVal, yVal,
							widthCheckerboardPrevLevel, heightLevelPrev,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);

					if (checkerboardPart == CHECKERBOARD_PART_1) {
						prevValU =
								messageUPrevStereoCheckerboard1[indexCopyFrom];
						prevValD =
								messageDPrevStereoCheckerboard1[indexCopyFrom];
						prevValL =
								messageLPrevStereoCheckerboard1[indexCopyFrom];
						prevValR =
								messageRPrevStereoCheckerboard1[indexCopyFrom];
					} else if (checkerboardPart == CHECKERBOARD_PART_2) {
						prevValU =
								messageUPrevStereoCheckerboard2[indexCopyFrom];
						prevValD =
								messageDPrevStereoCheckerboard2[indexCopyFrom];
						prevValL =
								messageLPrevStereoCheckerboard2[indexCopyFrom];
						prevValR =
								messageRPrevStereoCheckerboard2[indexCopyFrom];
					}

					if (withinImageBoundsCPU(xVal * 2 + checkerboardPartAdjustment,
							yVal * 2, widthCheckerboardNextLevel,
							heightCheckerboardNextLevel)) {
						indexCopyTo = retrieveIndexInDataAndMessageCPU(
								(xVal * 2 + checkerboardPartAdjustment),
								(yVal * 2), widthCheckerboardNextLevel,
								heightCheckerboardNextLevel, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES);

						messageUDeviceCurrentCheckerboard1[indexCopyTo] =
								prevValU;
						messageDDeviceCurrentCheckerboard1[indexCopyTo] =
								prevValD;
						messageLDeviceCurrentCheckerboard1[indexCopyTo] =
								prevValL;
						messageRDeviceCurrentCheckerboard1[indexCopyTo] =
								prevValR;

						messageUDeviceCurrentCheckerboard2[indexCopyTo] =
								prevValU;
						messageDDeviceCurrentCheckerboard2[indexCopyTo] =
								prevValD;
						messageLDeviceCurrentCheckerboard2[indexCopyTo] =
								prevValL;
						messageRDeviceCurrentCheckerboard2[indexCopyTo] =
								prevValR;
					}

					if (withinImageBoundsCPU(xVal * 2 + checkerboardPartAdjustment,
							yVal * 2 + 1, widthCheckerboardNextLevel,
							heightCheckerboardNextLevel)) {
						indexCopyTo = retrieveIndexInDataAndMessageCPU(
								(xVal * 2 + checkerboardPartAdjustment),
								(yVal * 2 + 1), widthCheckerboardNextLevel,
								heightCheckerboardNextLevel, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES);

						messageUDeviceCurrentCheckerboard1[indexCopyTo] =
								prevValU;
						messageDDeviceCurrentCheckerboard1[indexCopyTo] =
								prevValD;
						messageLDeviceCurrentCheckerboard1[indexCopyTo] =
								prevValL;
						messageRDeviceCurrentCheckerboard1[indexCopyTo] =
								prevValR;

						messageUDeviceCurrentCheckerboard2[indexCopyTo] =
								prevValU;
						messageDDeviceCurrentCheckerboard2[indexCopyTo] =
								prevValD;
						messageLDeviceCurrentCheckerboard2[indexCopyTo] =
								prevValL;
						messageRDeviceCurrentCheckerboard2[indexCopyTo] =
								prevValR;
					}
				}
			}
		}
	//}
}


//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
template<typename T>
void KernelBpStereoCPU::retrieveOutputDisparityCheckerboardStereoNoTexturesCPU(T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2, T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1, T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1, T* messageUPrevStereoCheckerboard2, T* messageDPrevStereoCheckerboard2, T* messageLPrevStereoCheckerboard2, T* messageRPrevStereoCheckerboard2, float* disparityBetweenImagesDevice, int widthLevel, int heightLevel)
{
		#pragma omp parallel for
		for (int val = 0; val < widthLevel*heightLevel; val++)
			{
				int yVal = val / widthLevel;
				int xVal = val % widthLevel;
				/*for (int yVal = 0; yVal < heightLevel; yVal++)
				{		#pragma omp parallel for
		for (int xVal = 0; xVal < widthLevel; xVal++)
		{*/
			//if (withinImageBoundsCPU(xVal, yVal, widthLevel, heightLevel))
			{
				int widthCheckerboard = getCheckerboardWidthCPU<T>(widthLevel);
				int xValInCheckerboardPart = xVal / 2;

				if (((yVal + xVal) % 2) == 0) //if true, then pixel is from part 1 of the checkerboard; otherwise, it's from part 2
						{
					int checkerboardPartAdjustment = (yVal % 2);

					if ((xVal >= 1) && (xVal < (widthLevel - 1)) && (yVal >= 1)
							&& (yVal < (heightLevel - 1))) {
						// keep track of "best" disparity for current pixel
						int bestDisparity = 0;
						T best_val = INF_BP;
						for (int currentDisparity = 0;
								currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
								currentDisparity++) {
							T val =
									messageUPrevStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
											xValInCheckerboardPart, (yVal + 1),
											widthCheckerboard, heightLevel,
											currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)]
											+ messageDPrevStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
													xValInCheckerboardPart,
													(yVal - 1),
													widthCheckerboard,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)]
											+ messageLPrevStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
													(xValInCheckerboardPart
															+ checkerboardPartAdjustment),
													yVal, widthCheckerboard,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)]
											+ messageRPrevStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
													(xValInCheckerboardPart - 1
															+ checkerboardPartAdjustment),
													yVal, widthCheckerboard,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)]
											+ dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
													xValInCheckerboardPart,
													yVal, widthCheckerboard,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)];

							if (val < (best_val)) {
								best_val = val;
								bestDisparity = currentDisparity;
							}
						}
						disparityBetweenImagesDevice[yVal * widthLevel + xVal] =
								bestDisparity;
					} else {
						disparityBetweenImagesDevice[yVal * widthLevel + xVal] =
								0;
					}
				} else //pixel from part 2 of checkerboard
				{
					int checkerboardPartAdjustment = ((yVal + 1) % 2);

					if ((xVal >= 1) && (xVal < (widthLevel - 1)) && (yVal >= 1)
							&& (yVal < (heightLevel - 1))) {

						// keep track of "best" disparity for current pixel
						int bestDisparity = 0;
						T best_val = INF_BP;
						for (int currentDisparity = 0;
								currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
								currentDisparity++) {
							T val =
									messageUPrevStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
											xValInCheckerboardPart, (yVal + 1),
											widthCheckerboard, heightLevel,
											currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)]
											+ messageDPrevStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
													xValInCheckerboardPart,
													(yVal - 1),
													widthCheckerboard,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)]
											+ messageLPrevStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
													(xValInCheckerboardPart
															+ checkerboardPartAdjustment),
													yVal, widthCheckerboard,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)]
											+ messageRPrevStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
													(xValInCheckerboardPart - 1
															+ checkerboardPartAdjustment),
													yVal, widthCheckerboard,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)]
											+ dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
													xValInCheckerboardPart,
													yVal, widthCheckerboard,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)];

							if (val < (best_val)) {
								best_val = val;
								bestDisparity = currentDisparity;
							}
						}
						disparityBetweenImagesDevice[yVal * widthLevel + xVal] =
								bestDisparity;
					} else {
						disparityBetweenImagesDevice[yVal * widthLevel + xVal] =
								0;
					}
				}
			}
		//}
	}
}
