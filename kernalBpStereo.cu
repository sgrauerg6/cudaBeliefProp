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
#include "kernalBpStereoHeader.cuh"


//checks if the current point is within the image bounds
__device__ bool withinImageBounds(int xVal, int yVal, int width, int height)
{
	return ((xVal >= 0) && (xVal < width) && (yVal >= 0) && (yVal < height));
}


//retrieve the current 1-D index value of the given point at the given disparity in the data cost and message data
__device__ int retrieveIndexInDataAndMessage(int xVal, int yVal, int width, int height, int currentDisparity, int totalNumDispVals, int offsetData)
{
	return RETRIEVE_INDEX_IN_DATA_OR_MESSAGE_ARRAY_EQUATION + offsetData;
}


//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
__device__ void dtStereo(float f[NUM_POSSIBLE_DISPARITY_VALUES]) 
{
	float prev;
	for (int currentDisparity = 1; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		prev = f[currentDisparity-1] + 1.0f;
		if (prev < f[currentDisparity])
			f[currentDisparity] = prev;
	}

	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		prev = f[currentDisparity+1] + 1.0f;
		if (prev < f[currentDisparity])
			f[currentDisparity] = prev;
	}
}




// compute current message
__device__ void msgStereo(float messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], float messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES], 
	float messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], float dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
	float dst[NUM_POSSIBLE_DISPARITY_VALUES])
{
	// aggregate and find min
	float minimum = INF_BP;

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		if (dst[currentDisparity] < minimum)
			minimum = dst[currentDisparity];
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	
	dtStereo(dst);

	// truncate 
	minimum += BPSettingsConstMemStereo.discCostCap;

	// normalize
	float valToNormalize = 0;

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
__global__ void initializeBottomLevelDataStereo(float* dataCostDeviceStereoCheckerboard1, float* dataCostDeviceStereoCheckerboard2)
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


	if (withinImageBounds(xVal, yVal, BPSettingsConstMemStereo.widthImages, BPSettingsConstMemStereo.heightImages))
	{
		

		//make sure that it is possible to check every disparity value
		if ((xVal - NUM_POSSIBLE_DISPARITY_VALUES) >= 0)
		{
			for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
			{
				float currentPixelImage1;
				float currentPixelImage2;

				currentPixelImage1 = tex2D(image1PixelsTextureBPStereo, xVal, yVal);
				currentPixelImage2 = tex2D(image2PixelsTextureBPStereo, xVal - currentDisparity, yVal);

				indexVal = retrieveIndexInDataAndMessage((xVal/2), yVal, (BPSettingsConstMemStereo.widthImages/2), BPSettingsConstMemStereo.heightImages, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);

				//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
				if (((xVal + yVal) % 2) == 0)
				{
					dataCostDeviceStereoCheckerboard1[indexVal] = BPSettingsConstMemStereo.dataWeight * min(abs(currentPixelImage1 - currentPixelImage2), BPSettingsConstMemStereo.dataCostCap);
				}
				else
				{
					dataCostDeviceStereoCheckerboard2[indexVal] = BPSettingsConstMemStereo.dataWeight * min(abs(currentPixelImage1 - currentPixelImage2), BPSettingsConstMemStereo.dataCostCap);
				}

			}
		}
		else
		{
			for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
			{

				indexVal = retrieveIndexInDataAndMessage((xVal/2), yVal, (BPSettingsConstMemStereo.widthImages/2), BPSettingsConstMemStereo.heightImages, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);

				//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
				if (((xVal + yVal) % 2) == 0)
				{
					dataCostDeviceStereoCheckerboard1[indexVal] = 0;
				}
				else
				{
					dataCostDeviceStereoCheckerboard2[indexVal] = 0;
				}

			}
		}
	}
}



//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
__global__ void initializeCurrentLevelDataStereo(float* dataCostDeviceToWriteTo, int widthLevel, int heightLevel, int checkerboardPart, int offsetNum)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int xVal = bx * BLOCK_SIZE_WIDTH_BP + tx;
	int yVal = by * BLOCK_SIZE_HEIGHT_BP + ty;

	if (withinImageBounds(xVal, yVal, widthLevel/2, heightLevel))
	{

		int widthLevelPrevCheckerboard = widthLevel;
		int widthLevelCurrentCheckerboard = widthLevel/2;
		int heightLevelPrev = heightLevel*2;

		//add 1 or 0 to the x-value depending on checkerboard part and row adding to; part with slot at (0, 0) has adjustment of 0 in row 0, 
		//while part with slot at (0, 1) has adjustment of 1 in row 0
		int checkerboardPartAdjustment = 0;

		if (checkerboardPart == CHECKERBOARD_PART_1)
		{
			checkerboardPartAdjustment = (yVal%2);
		}
		else if (checkerboardPart == CHECKERBOARD_PART_2)
		{
			checkerboardPartAdjustment = ((yVal+1)%2);
		}

		//the corresponding x-values at the "lower" level depends on which checkerboard the pixel is in
		int xValPrev = xVal*2 + checkerboardPartAdjustment;

		for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		{
			dataCostDeviceToWriteTo[retrieveIndexInDataAndMessage(xVal, yVal, widthLevelCurrentCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)] = 
				tex1Dfetch(dataCostTexStereoCheckerboard1, retrieveIndexInDataAndMessage(xValPrev, (yVal*2), widthLevelPrevCheckerboard, heightLevelPrev, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetNum)) 
				+ tex1Dfetch(dataCostTexStereoCheckerboard1, retrieveIndexInDataAndMessage(xValPrev, (yVal*2 + 1), widthLevelPrevCheckerboard, heightLevelPrev, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetNum)) 
				+ tex1Dfetch(dataCostTexStereoCheckerboard2,  retrieveIndexInDataAndMessage(xValPrev, (yVal*2), widthLevelPrevCheckerboard, heightLevelPrev, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetNum))
				+ tex1Dfetch(dataCostTexStereoCheckerboard2,  retrieveIndexInDataAndMessage(xValPrev, (yVal*2 + 1), widthLevelPrevCheckerboard, heightLevelPrev, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetNum));
		}
	}
}


//initialize the message values at each pixel of the current level to the default value
__global__ void initializeMessageValsToDefault(float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1, float* messageLDeviceCurrentCheckerboard1, 
												float* messageRDeviceCurrentCheckerboard1, float* messageUDeviceCurrentCheckerboard2, float* messageDDeviceCurrentCheckerboard2, 
												float* messageLDeviceCurrentCheckerboard2, float* messageRDeviceCurrentCheckerboard2, int widthCheckerboardAtLevel, int heightLevel)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int xValInCheckerboard = bx * BLOCK_SIZE_WIDTH_BP + tx;
	int yVal = by * BLOCK_SIZE_HEIGHT_BP + ty;

	if (withinImageBounds(xValInCheckerboard, yVal, widthCheckerboardAtLevel, heightLevel))
	{
		//initialize message values in both checkerboards

		//set the message value at each pixel for each disparity to 0
		for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		{
			messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal, widthCheckerboardAtLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)] = DEFAULT_INITIAL_MESSAGE_VAL;
			messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal, widthCheckerboardAtLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)] = DEFAULT_INITIAL_MESSAGE_VAL;
			messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal, widthCheckerboardAtLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)] = DEFAULT_INITIAL_MESSAGE_VAL;
			messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal, widthCheckerboardAtLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)] = DEFAULT_INITIAL_MESSAGE_VAL;
		}

		//retrieve the previous message value at each movement at each pixel
		for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		{
			messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal, widthCheckerboardAtLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)] = DEFAULT_INITIAL_MESSAGE_VAL;
			messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal, widthCheckerboardAtLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)] = DEFAULT_INITIAL_MESSAGE_VAL;
			messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal, widthCheckerboardAtLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)] = DEFAULT_INITIAL_MESSAGE_VAL;
			messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal, widthCheckerboardAtLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)] = DEFAULT_INITIAL_MESSAGE_VAL;
		}	
	}
}


//device portion of the kernel function to run the current iteration of belief propagation where the input messages and data costs come in as array in local memory
//and the output message values are stored in local memory
__device__ void runBPIterationInOutDataInLocalMem(float prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES], float prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES], float prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES], float prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES], float dataMessage[NUM_POSSIBLE_DISPARITY_VALUES],
								float currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES], float currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES], float currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES], float currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES])
{

			msgStereo(prevUMessage, prevLMessage, prevRMessage,
				dataMessage, currentUMessage);
	
			msgStereo(prevDMessage, prevLMessage, prevRMessage,
				dataMessage, currentDMessage);

			msgStereo(prevUMessage, prevDMessage, prevRMessage,
				dataMessage, currentRMessage);

			msgStereo(prevUMessage, prevDMessage, prevLMessage,
				dataMessage, currentLMessage);

}


//device portion of the kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the 
//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//this function uses local memory to store the message and data values at each disparity in the intermediate step of current message computation
//this function uses linear memory bound to textures to access the current data and message values
__device__ void runBPIterationUsingCheckerboardUpdatesDeviceUseTexBoundAndLocalMem(float* messageUDeviceCurrentCheckerboardOut, float* messageDDeviceCurrentCheckerboardOut, 
													float* messageLDeviceCurrentCheckerboardOut, float* messageRDeviceCurrentCheckerboardOut, 
													int widthLevelCheckerboardPart, int heightLevel, int iterationNum, int xVal, int yVal, int offsetData)
{
	int indexWriteTo;
	int checkerboardAdjustment;

	
	//if iterationNum is even, update checkerboard part 2
	//if iterationNum is odd, update checkerboard part 1
	//checkerboardAdjustment used for indexing into current checkerboard
	if ((iterationNum%2) == 0)
	{
		checkerboardAdjustment = ((yVal+1)%2);
	}

	else
	{
		checkerboardAdjustment = ((yVal)%2);
	}


	//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
	//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
	if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
	{

		float prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		float prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		float prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		float prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		float dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		{
			if ((iterationNum%2) == 0)
			{
				dataMessage[currentDisparity] = tex1Dfetch(dataCostTexStereoCheckerboard2, retrieveIndexInDataAndMessage(xVal, yVal, widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData));
			}
			else
			{
				dataMessage[currentDisparity] = tex1Dfetch(dataCostTexStereoCheckerboard1, retrieveIndexInDataAndMessage(xVal, yVal, widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData));
			}

			prevUMessage[currentDisparity] = tex1Dfetch(messageUPrevTexStereoCheckerboard1, retrieveIndexInDataAndMessage(xVal, (yVal+1), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES));
			prevDMessage[currentDisparity] = tex1Dfetch(messageDPrevTexStereoCheckerboard1, retrieveIndexInDataAndMessage(xVal, (yVal-1), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES));
			prevLMessage[currentDisparity] = tex1Dfetch(messageLPrevTexStereoCheckerboard1, retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), (yVal), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES));
			prevRMessage[currentDisparity] = tex1Dfetch(messageRPrevTexStereoCheckerboard1, retrieveIndexInDataAndMessage((xVal - 1 + checkerboardAdjustment), (yVal), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES));
		}

		float currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		float currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		float currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		float currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		//uses the previous message values and data cost to calculate the current message values and store the results
		runBPIterationInOutDataInLocalMem(prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							currentUMessage, currentDMessage, currentLMessage, currentRMessage);
	

		//write the calculated message values to global memory
		for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		{
			indexWriteTo = retrieveIndexInDataAndMessage(xVal, yVal, widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
			messageUDeviceCurrentCheckerboardOut[indexWriteTo] = currentUMessage[currentDisparity];
			messageDDeviceCurrentCheckerboardOut[indexWriteTo] = currentDMessage[currentDisparity];
			messageLDeviceCurrentCheckerboardOut[indexWriteTo] = currentLMessage[currentDisparity];
			messageRDeviceCurrentCheckerboardOut[indexWriteTo] = currentRMessage[currentDisparity];
		}
	}
	else
	{
		for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		{
			indexWriteTo = retrieveIndexInDataAndMessage(xVal, yVal, widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
			messageUDeviceCurrentCheckerboardOut[indexWriteTo] = 0.0f;
			messageDDeviceCurrentCheckerboardOut[indexWriteTo] = 0.0f;
			messageLDeviceCurrentCheckerboardOut[indexWriteTo] = 0.0f;
			messageRDeviceCurrentCheckerboardOut[indexWriteTo] = 0.0f;
		}
	}
}






//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard" 
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
__global__ void runBPIterationUsingCheckerboardUpdates(float* messageUDeviceCurrentCheckerboardOut, float* messageDDeviceCurrentCheckerboardOut, float* messageLDeviceCurrentCheckerboardOut, 
								float* messageRDeviceCurrentCheckerboardOut, int widthLevel, int heightLevel, int iterationNum, int offsetData)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int xVal = bx * BLOCK_SIZE_WIDTH_BP + tx;
	int yVal = by * BLOCK_SIZE_HEIGHT_BP + ty;

	if (withinImageBounds(xVal, yVal, widthLevel/2, heightLevel))
	{

		
		runBPIterationUsingCheckerboardUpdatesDeviceUseTexBoundAndLocalMem(messageUDeviceCurrentCheckerboardOut, messageDDeviceCurrentCheckerboardOut, 
												messageLDeviceCurrentCheckerboardOut, messageRDeviceCurrentCheckerboardOut, 
												widthLevel/2, heightLevel, iterationNum, xVal, yVal, offsetData);
	}
}




//kernal to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernal works from the point of view of the pixel at the prev level that is being copied to four different places
__global__ void copyPrevLevelToNextLevelBPCheckerboardStereo(float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1, float* messageLDeviceCurrentCheckerboard1, 
															float* messageRDeviceCurrentCheckerboard1, float* messageUDeviceCurrentCheckerboard2, float* messageDDeviceCurrentCheckerboard2, 
															float* messageLDeviceCurrentCheckerboard2, float* messageRDeviceCurrentCheckerboard2, int widthLevel, int heightLevel,
															int checkerboardPart)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int xVal = bx * BLOCK_SIZE_WIDTH_BP + tx;
	int yVal = by * BLOCK_SIZE_HEIGHT_BP + ty;

	if (withinImageBounds(xVal, yVal, widthLevel/2, heightLevel))
	{

		int widthCheckerboard = widthLevel/2;
		int widthCheckerboardNextLevel = widthLevel;
		int heightCheckerboardNextLevel = heightLevel*2;

		int indexCopyTo;
		int indexCopyFrom;


		int checkerboardPartAdjustment;

		float prevValU;
		float prevValD;
		float prevValL;
		float prevValR;

		if (checkerboardPart == CHECKERBOARD_PART_1)
		{
			checkerboardPartAdjustment = (yVal%2);
		}
		else if (checkerboardPart == CHECKERBOARD_PART_2)
		{
			checkerboardPartAdjustment = ((yVal+1)%2);
		}


		for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		{
			indexCopyFrom = retrieveIndexInDataAndMessage(xVal, yVal, widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES); 
			
			if (checkerboardPart == CHECKERBOARD_PART_1)
			{
				prevValU = tex1Dfetch(messageUPrevTexStereoCheckerboard1, indexCopyFrom);
				prevValD = tex1Dfetch(messageDPrevTexStereoCheckerboard1, indexCopyFrom);
				prevValL = tex1Dfetch(messageLPrevTexStereoCheckerboard1, indexCopyFrom);
				prevValR = tex1Dfetch(messageRPrevTexStereoCheckerboard1, indexCopyFrom);
			}
			else if (checkerboardPart == CHECKERBOARD_PART_2)
			{
				prevValU = tex1Dfetch(messageUPrevTexStereoCheckerboard2, indexCopyFrom);
				prevValD = tex1Dfetch(messageDPrevTexStereoCheckerboard2, indexCopyFrom);
				prevValL = tex1Dfetch(messageLPrevTexStereoCheckerboard2, indexCopyFrom);
				prevValR = tex1Dfetch(messageRPrevTexStereoCheckerboard2, indexCopyFrom);
			}

			indexCopyTo = retrieveIndexInDataAndMessage((xVal*2 + checkerboardPartAdjustment), (yVal*2), widthCheckerboardNextLevel, heightCheckerboardNextLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);

			messageUDeviceCurrentCheckerboard1[indexCopyTo] = prevValU;
			messageDDeviceCurrentCheckerboard1[indexCopyTo] = prevValD;
			messageLDeviceCurrentCheckerboard1[indexCopyTo] = prevValL;
			messageRDeviceCurrentCheckerboard1[indexCopyTo] = prevValR;

			messageUDeviceCurrentCheckerboard2[indexCopyTo] = prevValU;
			messageDDeviceCurrentCheckerboard2[indexCopyTo] = prevValD;
			messageLDeviceCurrentCheckerboard2[indexCopyTo] = prevValL;
			messageRDeviceCurrentCheckerboard2[indexCopyTo] = prevValR;

			indexCopyTo = retrieveIndexInDataAndMessage((xVal*2 + checkerboardPartAdjustment), (yVal*2 + 1), widthCheckerboardNextLevel, heightCheckerboardNextLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES); 

			messageUDeviceCurrentCheckerboard1[indexCopyTo] = prevValU;
			messageDDeviceCurrentCheckerboard1[indexCopyTo] = prevValD;
			messageLDeviceCurrentCheckerboard1[indexCopyTo] = prevValL;
			messageRDeviceCurrentCheckerboard1[indexCopyTo] = prevValR;

			messageUDeviceCurrentCheckerboard2[indexCopyTo] = prevValU;
			messageDDeviceCurrentCheckerboard2[indexCopyTo] = prevValD;
			messageLDeviceCurrentCheckerboard2[indexCopyTo] = prevValL;
			messageRDeviceCurrentCheckerboard2[indexCopyTo] = prevValR;
		}
	}
}



//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
__global__ void retrieveOutputDisparityCheckerboardStereo(float* disparityBetweenImagesDevice, int widthLevel, int heightLevel)
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

		int widthCheckerboard = widthLevel/2;
		int xValInCheckerboardPart = xVal/2;

		if (((yVal+xVal) % 2) == 0) //if true, then pixel is from part 1 of the checkerboard; otherwise, it's from part 2
		{
			int	checkerboardPartAdjustment = (yVal%2);

			if ((xVal >= 0) && (xVal <= (widthLevel - 1)) && (yVal >= 0) && (yVal <= (heightLevel - 1)))
			{
				// keep track of "best" disparity for current pixel
				int bestDisparity = 0;
				float best_val = INF_BP;
				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					float val = tex1Dfetch(messageUPrevTexStereoCheckerboard2, retrieveIndexInDataAndMessage(xValInCheckerboardPart, (yVal + 1), widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)) +
						 tex1Dfetch(messageDPrevTexStereoCheckerboard2, retrieveIndexInDataAndMessage(xValInCheckerboardPart, (yVal - 1), widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)) + 
						 tex1Dfetch(messageLPrevTexStereoCheckerboard2, retrieveIndexInDataAndMessage((xValInCheckerboardPart + checkerboardPartAdjustment), yVal, widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)) + 
						 tex1Dfetch(messageRPrevTexStereoCheckerboard2,retrieveIndexInDataAndMessage((xValInCheckerboardPart - 1 + checkerboardPartAdjustment), yVal, widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)) +
						 tex1Dfetch(dataCostTexStereoCheckerboard1, retrieveIndexInDataAndMessage(xValInCheckerboardPart, yVal, widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES));
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

			if ((xVal >= 0) && (xVal <= (widthLevel - 1)) && (yVal >= 0) && (yVal <= (heightLevel - 1)))
			{
				// keep track of "best" disparity for current pixel
				int bestDisparity = 0;
				float best_val = INF_BP;
				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					float val = tex1Dfetch(messageUPrevTexStereoCheckerboard1, retrieveIndexInDataAndMessage(xValInCheckerboardPart, (yVal + 1), widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)) +
						tex1Dfetch(messageDPrevTexStereoCheckerboard1, retrieveIndexInDataAndMessage(xValInCheckerboardPart, (yVal - 1), widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)) + 
						tex1Dfetch(messageLPrevTexStereoCheckerboard1, retrieveIndexInDataAndMessage((xValInCheckerboardPart + checkerboardPartAdjustment), yVal, widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)) + 
						tex1Dfetch(messageRPrevTexStereoCheckerboard1,retrieveIndexInDataAndMessage((xValInCheckerboardPart - 1 + checkerboardPartAdjustment), yVal, widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)) +
						tex1Dfetch(dataCostTexStereoCheckerboard2, retrieveIndexInDataAndMessage(xValInCheckerboardPart, yVal, widthCheckerboard, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES));
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
