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

//Defines the functions to run the CUDA implementation of 2-D Stereo estimation using BP

#include "ProcessOptimizedCPUBP.h"
#include "KernelBpStereoCPU.cpp"

template<typename T>
void ProcessOptimizedCPUBPHelperFuncts::printDataAndMessageValsAtPoint(int xVal, int yVal, T* dataCostDeviceCurrentLevelCheckerboard1, T* dataCostDeviceCurrentLevelCheckerboard2,
		T* messageUDeviceSet0Checkerboard1, T* messageDDeviceSet0Checkerboard1,
		T* messageLDeviceSet0Checkerboard1, T* messageRDeviceSet0Checkerboard1,
		T* messageUDeviceSet0Checkerboard2, T* messageDDeviceSet0Checkerboard2,
		T* messageLDeviceSet0Checkerboard2, T* messageRDeviceSet0Checkerboard2,
		T* messageUDeviceSet1Checkerboard1, T* messageDDeviceSet1Checkerboard1,
		T* messageLDeviceSet1Checkerboard1, T* messageRDeviceSet1Checkerboard1,
		T* messageUDeviceSet1Checkerboard2, T* messageDDeviceSet1Checkerboard2,
		T* messageLDeviceSet1Checkerboard2, T* messageRDeviceSet1Checkerboard2,
		int widthCheckerboard, int heightLevel, int currentCheckerboardSet)
{
	if (currentCheckerboardSet == 0) {
		KernelBpStereoCPU::printDataAndMessageValsAtPointKernelCPU<T>(xVal, yVal,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messageUDeviceSet0Checkerboard1,
				messageDDeviceSet0Checkerboard1,
				messageLDeviceSet0Checkerboard1,
				messageRDeviceSet0Checkerboard1,
				messageUDeviceSet0Checkerboard2,
				messageDDeviceSet0Checkerboard2,
				messageLDeviceSet0Checkerboard2,
				messageRDeviceSet0Checkerboard2, widthCheckerboard,
				heightLevel);
	} else {
		KernelBpStereoCPU::printDataAndMessageValsAtPointKernelCPU<T>(xVal, yVal,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messageUDeviceSet1Checkerboard1,
				messageDDeviceSet1Checkerboard1,
				messageLDeviceSet1Checkerboard1,
				messageRDeviceSet1Checkerboard1,
				messageUDeviceSet1Checkerboard2,
				messageDDeviceSet1Checkerboard2,
				messageLDeviceSet1Checkerboard2,
				messageRDeviceSet1Checkerboard2, widthCheckerboard,
				heightLevel);
	}
}


template<typename T>
void ProcessOptimizedCPUBPHelperFuncts::printDataAndMessageValsToPoint(int xVal, int yVal, T* dataCostDeviceCurrentLevelCheckerboard1, T* dataCostDeviceCurrentLevelCheckerboard2,
		T* messageUDeviceSet0Checkerboard1, T* messageDDeviceSet0Checkerboard1,
		T* messageLDeviceSet0Checkerboard1, T* messageRDeviceSet0Checkerboard1,
		T* messageUDeviceSet0Checkerboard2, T* messageDDeviceSet0Checkerboard2,
		T* messageLDeviceSet0Checkerboard2, T* messageRDeviceSet0Checkerboard2,
		T* messageUDeviceSet1Checkerboard1, T* messageDDeviceSet1Checkerboard1,
		T* messageLDeviceSet1Checkerboard1, T* messageRDeviceSet1Checkerboard1,
		T* messageUDeviceSet1Checkerboard2, T* messageDDeviceSet1Checkerboard2,
		T* messageLDeviceSet1Checkerboard2, T* messageRDeviceSet1Checkerboard2,
		int widthCheckerboard, int heightLevel, int currentCheckerboardSet)
{
	if (currentCheckerboardSet == 0) {
		KernelBpStereoCPU::printDataAndMessageValsToPointKernelCPU<T>(xVal, yVal,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messageUDeviceSet0Checkerboard1,
				messageDDeviceSet0Checkerboard1,
				messageLDeviceSet0Checkerboard1,
				messageRDeviceSet0Checkerboard1,
				messageUDeviceSet0Checkerboard2,
				messageDDeviceSet0Checkerboard2,
				messageLDeviceSet0Checkerboard2,
				messageRDeviceSet0Checkerboard2, widthCheckerboard,
				heightLevel);
	} else {
		KernelBpStereoCPU::printDataAndMessageValsToPointKernelCPU<T>(xVal, yVal,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messageUDeviceSet1Checkerboard1,
				messageDDeviceSet1Checkerboard1,
				messageLDeviceSet1Checkerboard1,
				messageRDeviceSet1Checkerboard1,
				messageUDeviceSet1Checkerboard2,
				messageDDeviceSet1Checkerboard2,
				messageLDeviceSet1Checkerboard2,
				messageRDeviceSet1Checkerboard2, widthCheckerboard,
				heightLevel);
	}
}

//functions directed related to running BP to retrieve the movement between the images

//run the given number of iterations of BP at the current level using the given message values in global device memory
template<typename T>
void ProcessOptimizedCPUBPHelperFuncts::runBPAtCurrentLevel(BPsettings& algSettings, int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
	T* messageUDeviceCheckerboard1, T* messageDDeviceCheckerboard1, T* messageLDeviceCheckerboard1,
	T* messageRDeviceCheckerboard1, T* messageUDeviceCheckerboard2, T* messageDDeviceCheckerboard2, T* messageLDeviceCheckerboard2,
	T* messageRDeviceCheckerboard2, T* dataCostDeviceCheckerboard1,
	T* dataCostDeviceCheckerboard2)
{
	int widthCheckerboard = KernelBpStereoCPU::getCheckerboardWidthCPU<T>(widthLevelActualIntegerSize);

	//at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
	for (int iterationNum = 0; iterationNum < algSettings.numIterations; iterationNum++)
	{
		int checkboardPartUpdate = CHECKERBOARD_PART_2;

		if ((iterationNum % 2) == 0)
		{
			checkboardPartUpdate = CHECKERBOARD_PART_2;
		}
		else
		{
			checkboardPartUpdate = CHECKERBOARD_PART_1;
		}


		KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPU<T>(
						dataCostDeviceCheckerboard1, dataCostDeviceCheckerboard2,
						messageUDeviceCheckerboard1, messageDDeviceCheckerboard1,
						messageLDeviceCheckerboard1, messageRDeviceCheckerboard1,
						messageUDeviceCheckerboard2, messageDDeviceCheckerboard2,
						messageLDeviceCheckerboard2, messageRDeviceCheckerboard2,
						widthLevelActualIntegerSize, heightLevelActualIntegerSize,
						checkboardPartUpdate, algSettings.disc_k_bp);
	}
}

//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
template<typename T>
void ProcessOptimizedCPUBPHelperFuncts::copyMessageValuesToNextLevelDown(int widthLevelActualIntegerSizePrevLevel, int heightLevelActualIntegerSizePrevLevel,
	int widthLevelActualIntegerSizeNextLevel, int heightLevelActualIntegerSizeNextLevel,
	T* messageUDeviceCheckerboard1CopyFrom, T* messageDDeviceCheckerboard1CopyFrom, T* messageLDeviceCheckerboard1CopyFrom,
	T* messageRDeviceCheckerboard1CopyFrom, T* messageUDeviceCheckerboard2CopyFrom, T* messageDDeviceCheckerboard2CopyFrom,
	T* messageLDeviceCheckerboard2CopyFrom, T* messageRDeviceCheckerboard2CopyFrom, T** messageUDeviceCheckerboard1CopyTo,
	T** messageDDeviceCheckerboard1CopyTo, T** messageLDeviceCheckerboard1CopyTo, T** messageRDeviceCheckerboard1CopyTo,
	T** messageUDeviceCheckerboard2CopyTo, T** messageDDeviceCheckerboard2CopyTo, T** messageLDeviceCheckerboard2CopyTo,
	T** messageRDeviceCheckerboard2CopyTo)
{
	int widthCheckerboard = KernelBpStereoCPU::getCheckerboardWidthCPU<T>(widthLevelActualIntegerSizeNextLevel);

#ifndef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

	int totalPossibleMovements = NUM_POSSIBLE_DISPARITY_VALUES;

	//update the number of bytes needed to store each set
	int numDataAndMessageSetInCheckerboardAtLevel = widthCheckerboard * heightLevelActualIntegerSizeNextLevel * totalPossibleMovements;

	//allocate space in the GPU for the message values in the checkerboard set to copy to
	*messageUDeviceCheckerboard1CopyTo = new T[numDataAndMessageSetInCheckerboardAtLevel];
	*messageDDeviceCheckerboard1CopyTo = new T[numDataAndMessageSetInCheckerboardAtLevel];
	*messageLDeviceCheckerboard1CopyTo = new T[numDataAndMessageSetInCheckerboardAtLevel];
	*messageRDeviceCheckerboard1CopyTo = new T[numDataAndMessageSetInCheckerboardAtLevel];

	*messageUDeviceCheckerboard2CopyTo = new T[numDataAndMessageSetInCheckerboardAtLevel];
	*messageDDeviceCheckerboard2CopyTo = new T[numDataAndMessageSetInCheckerboardAtLevel];
	*messageLDeviceCheckerboard2CopyTo = new T[numDataAndMessageSetInCheckerboardAtLevel];
	*messageRDeviceCheckerboard2CopyTo = new T[numDataAndMessageSetInCheckerboardAtLevel];

#endif

	//call the kernal to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
	//storing the current message values
	KernelBpStereoCPU::copyPrevLevelToNextLevelBPCheckerboardStereoNoTexturesCPU<T>(messageUDeviceCheckerboard1CopyFrom, messageDDeviceCheckerboard1CopyFrom,
			messageLDeviceCheckerboard1CopyFrom, messageRDeviceCheckerboard1CopyFrom, messageUDeviceCheckerboard2CopyFrom,
			messageDDeviceCheckerboard2CopyFrom, messageLDeviceCheckerboard2CopyFrom, messageRDeviceCheckerboard2CopyFrom,
			*messageUDeviceCheckerboard1CopyTo, *messageDDeviceCheckerboard1CopyTo, *messageLDeviceCheckerboard1CopyTo,
			*messageRDeviceCheckerboard1CopyTo, *messageUDeviceCheckerboard2CopyTo, *messageDDeviceCheckerboard2CopyTo, *messageLDeviceCheckerboard2CopyTo,
			*messageRDeviceCheckerboard2CopyTo, KernelBpStereoCPU::getCheckerboardWidthCPU<T>(widthLevelActualIntegerSizePrevLevel), (heightLevelActualIntegerSizePrevLevel),
			KernelBpStereoCPU::getCheckerboardWidthCPU<T>(widthLevelActualIntegerSizeNextLevel), heightLevelActualIntegerSizeNextLevel, CHECKERBOARD_PART_1);

	KernelBpStereoCPU::copyPrevLevelToNextLevelBPCheckerboardStereoNoTexturesCPU<T>(messageUDeviceCheckerboard1CopyFrom, messageDDeviceCheckerboard1CopyFrom,
			messageLDeviceCheckerboard1CopyFrom, messageRDeviceCheckerboard1CopyFrom, messageUDeviceCheckerboard2CopyFrom,
			messageDDeviceCheckerboard2CopyFrom, messageLDeviceCheckerboard2CopyFrom, messageRDeviceCheckerboard2CopyFrom,
			*messageUDeviceCheckerboard1CopyTo, *messageDDeviceCheckerboard1CopyTo, *messageLDeviceCheckerboard1CopyTo,
			*messageRDeviceCheckerboard1CopyTo, *messageUDeviceCheckerboard2CopyTo, *messageDDeviceCheckerboard2CopyTo, *messageLDeviceCheckerboard2CopyTo,
			*messageRDeviceCheckerboard2CopyTo, KernelBpStereoCPU::getCheckerboardWidthCPU<T>(widthLevelActualIntegerSizePrevLevel), (heightLevelActualIntegerSizePrevLevel),
			KernelBpStereoCPU::getCheckerboardWidthCPU<T>(widthLevelActualIntegerSizeNextLevel), heightLevelActualIntegerSizeNextLevel, CHECKERBOARD_PART_2);

#ifndef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

	//free the now-copied from computed data of the completed level
	delete [] messageUDeviceCheckerboard1CopyFrom;
	delete [] messageDDeviceCheckerboard1CopyFrom;
	delete [] messageLDeviceCheckerboard1CopyFrom;
	delete [] messageRDeviceCheckerboard1CopyFrom;

	delete [] messageUDeviceCheckerboard2CopyFrom;
	delete [] messageDDeviceCheckerboard2CopyFrom;
	delete [] messageLDeviceCheckerboard2CopyFrom;
	delete [] messageRDeviceCheckerboard2CopyFrom;

#endif
}

//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
template<typename T>
void ProcessOptimizedCPUBPHelperFuncts::initializeDataCosts(float*& image1PixelsDevice, float*& image2PixelsDevice, T* dataCostDeviceCheckerboard1, T* dataCostDeviceCheckerboard2, BPsettings& algSettings)
{
	//initialize the data the the "bottom" of the image pyramid
	KernelBpStereoCPU::initializeBottomLevelDataStereoCPU<T>(image1PixelsDevice,
			image2PixelsDevice, dataCostDeviceCheckerboard1,
			dataCostDeviceCheckerboard2, algSettings.widthImages,
			algSettings.heightImages, algSettings.lambda_bp, algSettings.data_k_bp);
}

//initialize the message values with no previous message values...all message values are set to 0
template<typename T>
void ProcessOptimizedCPUBPHelperFuncts::initializeMessageValsToDefault(T* messageUDeviceSet0Checkerboard1, T* messageDDeviceSet0Checkerboard1, T* messageLDeviceSet0Checkerboard1, T* messageRDeviceSet0Checkerboard1,
												  T* messageUDeviceSet0Checkerboard2, T* messageDDeviceSet0Checkerboard2, T* messageLDeviceSet0Checkerboard2, T* messageRDeviceSet0Checkerboard2,
												  int widthLevel, int heightLevel, int numPossibleMovements)
{
	int widthOfCheckerboard = KernelBpStereoCPU::getCheckerboardWidthCPU<T>(widthLevel);

	//initialize all the message values for each pixel at each possible movement to the default value in the kernal
	KernelBpStereoCPU::initializeMessageValsToDefaultKernelCPU<T>(messageUDeviceSet0Checkerboard1, messageDDeviceSet0Checkerboard1, messageLDeviceSet0Checkerboard1,
												messageRDeviceSet0Checkerboard1, messageUDeviceSet0Checkerboard2, messageDDeviceSet0Checkerboard2, 
												messageLDeviceSet0Checkerboard2, messageRDeviceSet0Checkerboard2, widthOfCheckerboard, heightLevel);
}


template<typename T>
void ProcessOptimizedCPUBPHelperFuncts::initializeDataCurrentLevel(T* dataCostStereoCheckerboard1,
		T* dataCostStereoCheckerboard2, T* dataCostDeviceToWriteToCheckerboard1,
		T* dataCostDeviceToWriteToCheckerboard2,
		int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
		int prevWidthLevelActualIntegerSize,
		int prevHeightLevelActualIntegerSize)
{
	int widthCheckerboard = KernelBpStereoCPU::getCheckerboardWidthCPU<T>(widthLevelActualIntegerSize);

	size_t offsetNum = 0;

	KernelBpStereoCPU::initializeCurrentLevelDataStereoNoTexturesCPU<T>(
			dataCostStereoCheckerboard1,
			dataCostStereoCheckerboard2,
			dataCostDeviceToWriteToCheckerboard1,
			widthLevelActualIntegerSize, heightLevelActualIntegerSize,
			prevWidthLevelActualIntegerSize, prevHeightLevelActualIntegerSize,
			CHECKERBOARD_PART_1, ((int) offsetNum / sizeof(float)));

	KernelBpStereoCPU::initializeCurrentLevelDataStereoNoTexturesCPU<T>(
			dataCostStereoCheckerboard1,
			dataCostStereoCheckerboard2,
			dataCostDeviceToWriteToCheckerboard2,
			widthLevelActualIntegerSize, heightLevelActualIntegerSize,
			prevWidthLevelActualIntegerSize, prevHeightLevelActualIntegerSize,
			CHECKERBOARD_PART_2, ((int) offsetNum / sizeof(float)));
}

template<typename T>
void ProcessOptimizedCPUBPHelperFuncts::retrieveOutputDisparity(T* dataCostDeviceCurrentLevelCheckerboard1, T* dataCostDeviceCurrentLevelCheckerboard2,
		T* messageUDeviceSet0Checkerboard1, T* messageDDeviceSet0Checkerboard1, T* messageLDeviceSet0Checkerboard1, T* messageRDeviceSet0Checkerboard1,
		T* messageUDeviceSet0Checkerboard2, T* messageDDeviceSet0Checkerboard2, T* messageLDeviceSet0Checkerboard2, T* messageRDeviceSet0Checkerboard2,
		T* messageUDeviceSet1Checkerboard1, T* messageDDeviceSet1Checkerboard1, T* messageLDeviceSet1Checkerboard1, T* messageRDeviceSet1Checkerboard1,
		T* messageUDeviceSet1Checkerboard2, T* messageDDeviceSet1Checkerboard2, T* messageLDeviceSet1Checkerboard2, T* messageRDeviceSet1Checkerboard2,
		float* resultingDisparityMapDevice, int widthLevel, int heightLevel, int currentCheckerboardSet)
{
	if (currentCheckerboardSet == 0)
	{
		KernelBpStereoCPU::retrieveOutputDisparityCheckerboardStereoNoTexturesCPU<T>(
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messageUDeviceSet0Checkerboard1,
				messageDDeviceSet0Checkerboard1,
				messageLDeviceSet0Checkerboard1,
				messageRDeviceSet0Checkerboard1,
				messageUDeviceSet0Checkerboard2,
				messageDDeviceSet0Checkerboard2,
				messageLDeviceSet0Checkerboard2,
				messageRDeviceSet0Checkerboard2, resultingDisparityMapDevice,
				widthLevel, heightLevel);
	}
	else
	{
		KernelBpStereoCPU::retrieveOutputDisparityCheckerboardStereoNoTexturesCPU<T>(
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messageUDeviceSet1Checkerboard1,
				messageDDeviceSet1Checkerboard1,
				messageLDeviceSet1Checkerboard1,
				messageRDeviceSet1Checkerboard1,
				messageUDeviceSet1Checkerboard2,
				messageDDeviceSet1Checkerboard2,
				messageLDeviceSet1Checkerboard2,
				messageRDeviceSet1Checkerboard2, resultingDisparityMapDevice,
				widthLevel, heightLevel);
	}
}


//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
//the input images image1PixelsDevice and image2PixelsDevice are stored in the global memory of the GPU
//the output movements resultingDisparityMapDevice is stored in the global memory of the GPU
template<typename T>
DetailedTimings* ProcessOptimizedCPUBP<T>::operator()(float* image1PixelsCompDevice, float* image2PixelsCompDevice, float* resultingDisparityMapCompDevice, BPsettings& algSettings)
{
	//printf("Start opt CPU\n");
	//retrieve the total number of possible movements; this is equal to the number of disparity values 
	int totalPossibleMovements = NUM_POSSIBLE_DISPARITY_VALUES;

	//start at the "bottom level" and work way up to determine amount of space needed to store data costs
	float widthLevel = (float)algSettings.widthImages;
	float heightLevel = (float)algSettings.heightImages;

	//store the "actual" integer size of the width and height of the level since it's not actually
	//possible to work with level with a decimal sizes...the portion of the last row/column is truncated
	//if the width/level size has a decimal
	int widthLevelActualIntegerSize = (int)roundf(widthLevel);
	int heightLevelActualIntegerSize = (int)roundf(heightLevel);

	int halfTotalDataAllLevels = 0;

	//compute "half" the total number of pixels in including every level of the "pyramid"
	//using "half" because the data is split in two using the checkerboard scheme
	for (int levelNum = 0; levelNum < algSettings.numLevels; levelNum++)
	{
		halfTotalDataAllLevels += (KernelBpStereoCPU::getCheckerboardWidthCPU<T>(widthLevelActualIntegerSize)) * (heightLevelActualIntegerSize) * (totalPossibleMovements);
		widthLevel /= 2.0f;
		heightLevel /= 2.0f;

		widthLevelActualIntegerSize = (int)ceil(widthLevel);
		heightLevelActualIntegerSize = (int)ceil(heightLevel);
	}

	//declare and then allocate the space on the device to store the data cost component at each possible movement at each level of the "pyramid"
	//each checkboard holds half of the data
	T* dataCostDeviceCheckerboard1; //checkerboard 1 includes the pixel in slot (0, 0)
	T* dataCostDeviceCheckerboard2;

	T* messageUDeviceCheckerboard1;
	T* messageDDeviceCheckerboard1;
	T* messageLDeviceCheckerboard1;
	T* messageRDeviceCheckerboard1;

	T* messageUDeviceCheckerboard2;
	T* messageDDeviceCheckerboard2;
	T* messageLDeviceCheckerboard2;
	T* messageRDeviceCheckerboard2;

#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

	//printf("ALLOC ALL MEMORY\n");
	dataCostDeviceCheckerboard1 = new T[10*halfTotalDataAllLevels];
	dataCostDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[1*(halfTotalDataAllLevels)]);

	messageUDeviceCheckerboard1 = &(dataCostDeviceCheckerboard1[2*(halfTotalDataAllLevels)]);
	messageDDeviceCheckerboard1 = &(dataCostDeviceCheckerboard1[3*(halfTotalDataAllLevels)]);
	messageLDeviceCheckerboard1 = &(dataCostDeviceCheckerboard1[4*(halfTotalDataAllLevels)]);
	messageRDeviceCheckerboard1 = &(dataCostDeviceCheckerboard1[5*(halfTotalDataAllLevels)]);

	messageUDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[6*(halfTotalDataAllLevels)]);
	messageDDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[7*(halfTotalDataAllLevels)]);
	messageLDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[8*(halfTotalDataAllLevels)]);
	messageRDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[9*(halfTotalDataAllLevels)]);

#else

	dataCostDeviceCheckerboard1 = new T[halfTotalDataAllLevels];
	dataCostDeviceCheckerboard2 = new T[halfTotalDataAllLevels];

#endif


	//now go "back to" the bottom level to initialize the data costs starting at the bottom level and going up the pyramid
	widthLevel = (float)algSettings.widthImages;
	heightLevel = (float)algSettings.heightImages;

	widthLevelActualIntegerSize = (int)roundf(widthLevel);
	heightLevelActualIntegerSize = (int)roundf(heightLevel);

	//printf("INIT DATA COSTS\n");
	//initialize the data cost at the bottom level 
	ProcessOptimizedCPUBPHelperFuncts::initializeDataCosts<T>(
			image1PixelsCompDevice, image2PixelsCompDevice,
			dataCostDeviceCheckerboard1, dataCostDeviceCheckerboard2,
			algSettings);
	//printf("DONE INIT DATA COSTS\n");

	int offsetLevel = 0;

	//set the data costs at each level from the bottom level "up"
	for (int levelNum = 1; levelNum < algSettings.numLevels; levelNum++)
	{
		int prev_level_offset_level = offsetLevel;

		//width is half since each part of the checkboard contains half the values going across
		//retrieve offset where the data starts at the "current level"
		offsetLevel += (KernelBpStereoCPU::getCheckerboardWidthCPU<T>(widthLevelActualIntegerSize)) * (heightLevelActualIntegerSize) * totalPossibleMovements;

		widthLevel /= 2.0f;
		heightLevel /= 2.0f;

		int prevWidthLevelActualIntegerSize = widthLevelActualIntegerSize;
		int prevHeightLevelActualIntegerSize = heightLevelActualIntegerSize;

		widthLevelActualIntegerSize = (int)ceil(widthLevel);
		heightLevelActualIntegerSize = (int)ceil(heightLevel);
		int widthCheckerboard = KernelBpStereoCPU::getCheckerboardWidthCPU<T>(widthLevelActualIntegerSize);

		T* dataCostStereoCheckerboard1 =
				&dataCostDeviceCheckerboard1[prev_level_offset_level];
		T* dataCostStereoCheckerboard2 =
				&dataCostDeviceCheckerboard2[prev_level_offset_level];
		T* dataCostDeviceToWriteToCheckerboard1 =
				&dataCostDeviceCheckerboard1[offsetLevel];
		T* dataCostDeviceToWriteToCheckerboard2 =
				&dataCostDeviceCheckerboard2[offsetLevel];

		//printf("INIT DATA COSTS CURRENT LEVEL\n");
		ProcessOptimizedCPUBPHelperFuncts::initializeDataCurrentLevel<T>(dataCostStereoCheckerboard1,
				dataCostStereoCheckerboard2, dataCostDeviceToWriteToCheckerboard1,
				dataCostDeviceToWriteToCheckerboard2,
				widthLevelActualIntegerSize,
				heightLevelActualIntegerSize, prevWidthLevelActualIntegerSize,
				prevHeightLevelActualIntegerSize);
		//printf("DONE INIT DATA COSTS CURRENT LEVEL\n");
	}

	//declare the space to pass the BP messages
	//need to have two "sets" of checkerboards because
	//the message values at the "higher" level in the image
	//pyramid need copied to a lower level without overwriting
	//values
	T* dataCostDeviceCurrentLevelCheckerboard1;
	T* dataCostDeviceCurrentLevelCheckerboard2;
	T* messageUDeviceSet0Checkerboard1;
	T* messageDDeviceSet0Checkerboard1;
	T* messageLDeviceSet0Checkerboard1;
	T* messageRDeviceSet0Checkerboard1;

	T* messageUDeviceSet0Checkerboard2;
	T* messageDDeviceSet0Checkerboard2;
	T* messageLDeviceSet0Checkerboard2;
	T* messageRDeviceSet0Checkerboard2;

	T* messageUDeviceSet1Checkerboard1;
	T* messageDDeviceSet1Checkerboard1;
	T* messageLDeviceSet1Checkerboard1;
	T* messageRDeviceSet1Checkerboard1;

	T* messageUDeviceSet1Checkerboard2;
	T* messageDDeviceSet1Checkerboard2;
	T* messageLDeviceSet1Checkerboard2;
	T* messageRDeviceSet1Checkerboard2;

	dataCostDeviceCurrentLevelCheckerboard1 = &dataCostDeviceCheckerboard1[offsetLevel];
	dataCostDeviceCurrentLevelCheckerboard2 = &dataCostDeviceCheckerboard2[offsetLevel];

#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

	messageUDeviceSet0Checkerboard1 = &messageUDeviceCheckerboard1[offsetLevel];
	messageDDeviceSet0Checkerboard1 = &messageDDeviceCheckerboard1[offsetLevel];
	messageLDeviceSet0Checkerboard1 = &messageLDeviceCheckerboard1[offsetLevel];
	messageRDeviceSet0Checkerboard1 = &messageRDeviceCheckerboard1[offsetLevel];

	messageUDeviceSet0Checkerboard2 = &messageUDeviceCheckerboard2[offsetLevel];
	messageDDeviceSet0Checkerboard2 = &messageDDeviceCheckerboard2[offsetLevel];
	messageLDeviceSet0Checkerboard2 = &messageLDeviceCheckerboard2[offsetLevel];
	messageRDeviceSet0Checkerboard2 = &messageRDeviceCheckerboard2[offsetLevel];

#else

	//retrieve the number of bytes needed to store the data cost/each set of messages in the checkerboard
	int numDataAndMessageSetInCheckerboardAtLevel = (KernelBpStereoCPU::getCheckerboardWidthCPU<T>(widthLevelActualIntegerSize)) * heightLevelActualIntegerSize * totalPossibleMovements;

	//allocate the space for the message values in the first checkboard set at the current level
	messageUDeviceSet0Checkerboard1 = new T[numDataAndMessageSetInCheckerboardAtLevel];
	messageDDeviceSet0Checkerboard1 = new T[numDataAndMessageSetInCheckerboardAtLevel];
	messageLDeviceSet0Checkerboard1 = new T[numDataAndMessageSetInCheckerboardAtLevel];
	messageRDeviceSet0Checkerboard1 = new T[numDataAndMessageSetInCheckerboardAtLevel];

	messageUDeviceSet0Checkerboard2 = new T[numDataAndMessageSetInCheckerboardAtLevel];
	messageDDeviceSet0Checkerboard2 = new T[numDataAndMessageSetInCheckerboardAtLevel];
	messageLDeviceSet0Checkerboard2 = new T[numDataAndMessageSetInCheckerboardAtLevel];
	messageRDeviceSet0Checkerboard2 = new T[numDataAndMessageSetInCheckerboardAtLevel];

#endif

	//printf("initializeMessageValsToDefault\n");
	//initialize all the BP message values at every pixel for every disparity to 0
	ProcessOptimizedCPUBPHelperFuncts::initializeMessageValsToDefault<T>(messageUDeviceSet0Checkerboard1, messageDDeviceSet0Checkerboard1, messageLDeviceSet0Checkerboard1, messageRDeviceSet0Checkerboard1,
											messageUDeviceSet0Checkerboard2, messageDDeviceSet0Checkerboard2, messageLDeviceSet0Checkerboard2, messageRDeviceSet0Checkerboard2,
											widthLevelActualIntegerSize, heightLevelActualIntegerSize, totalPossibleMovements);
	//printf("DONE initializeMessageValsToDefault\n");


	//alternate between checkerboard sets 0 and 1
	int currentCheckerboardSet = 0;

	//run BP at each level in the "pyramid" starting on top and continuing to the bottom
	//where the final movement values are computed...the message values are passed from
	//the upper level to the lower levels; this pyramid methods causes the BP message values
	//to converge more quickly
	for (int levelNum = algSettings.numLevels - 1; levelNum >= 0; levelNum--)
	{
		//printf("LEVEL: %d\n", levelNum);
		//need to alternate which checkerboard set to work on since copying from one to the other...need to avoid read-write conflict when copying in parallel
		if (currentCheckerboardSet == 0)
		{
			ProcessOptimizedCPUBPHelperFuncts::runBPAtCurrentLevel<T>(algSettings,
					widthLevelActualIntegerSize, heightLevelActualIntegerSize,
					messageUDeviceSet0Checkerboard1,
					messageDDeviceSet0Checkerboard1,
					messageLDeviceSet0Checkerboard1,
					messageRDeviceSet0Checkerboard1,
					messageUDeviceSet0Checkerboard2,
					messageDDeviceSet0Checkerboard2,
					messageLDeviceSet0Checkerboard2,
					messageRDeviceSet0Checkerboard2,
					dataCostDeviceCurrentLevelCheckerboard1,
					dataCostDeviceCurrentLevelCheckerboard2);
		}
		else
		{
			ProcessOptimizedCPUBPHelperFuncts::runBPAtCurrentLevel<T>(algSettings,
					widthLevelActualIntegerSize, heightLevelActualIntegerSize,
					messageUDeviceSet1Checkerboard1,
					messageDDeviceSet1Checkerboard1,
					messageLDeviceSet1Checkerboard1,
					messageRDeviceSet1Checkerboard1,
					messageUDeviceSet1Checkerboard2,
					messageDDeviceSet1Checkerboard2,
					messageLDeviceSet1Checkerboard2,
					messageRDeviceSet1Checkerboard2,
					dataCostDeviceCurrentLevelCheckerboard1,
					dataCostDeviceCurrentLevelCheckerboard2);
		}
		//printf("DONE BP RUN\n");

		//if not at the "bottom level" copy the current message values at the current level to the corresponding slots next level 
		if (levelNum > 0)
		{	
			int prevWidthLevelActualIntegerSize = widthLevelActualIntegerSize;
			int prevHeightLevelActualIntegerSize = heightLevelActualIntegerSize;

			//the "next level" down has double the width and height of the current level
			widthLevel *= 2.0f;
			heightLevel *= 2.0f;

			widthLevelActualIntegerSize = (int)ceil(widthLevel);
			heightLevelActualIntegerSize = (int)ceil(heightLevel);
			int widthCheckerboard = KernelBpStereoCPU::getCheckerboardWidthCPU<T>(widthLevelActualIntegerSize);

			offsetLevel -= widthCheckerboard * heightLevelActualIntegerSize * totalPossibleMovements;

			dataCostDeviceCurrentLevelCheckerboard1 = &dataCostDeviceCheckerboard1[offsetLevel];
			dataCostDeviceCurrentLevelCheckerboard2 = &dataCostDeviceCheckerboard2[offsetLevel];

			//bind messages in the current checkerboard set to the texture to copy to the "other" checkerboard set at the next level 
			if (currentCheckerboardSet == 0)
			{

#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

				messageUDeviceSet1Checkerboard1 = &messageUDeviceCheckerboard1[offsetLevel];
				messageDDeviceSet1Checkerboard1 = &messageDDeviceCheckerboard1[offsetLevel];
				messageLDeviceSet1Checkerboard1 = &messageLDeviceCheckerboard1[offsetLevel];
				messageRDeviceSet1Checkerboard1 = &messageRDeviceCheckerboard1[offsetLevel];

				messageUDeviceSet1Checkerboard2 = &messageUDeviceCheckerboard2[offsetLevel];
				messageDDeviceSet1Checkerboard2 = &messageDDeviceCheckerboard2[offsetLevel];
				messageLDeviceSet1Checkerboard2 = &messageLDeviceCheckerboard2[offsetLevel];
				messageRDeviceSet1Checkerboard2 = &messageRDeviceCheckerboard2[offsetLevel];

#endif

				ProcessOptimizedCPUBPHelperFuncts::copyMessageValuesToNextLevelDown<T>(
						prevWidthLevelActualIntegerSize,
						prevHeightLevelActualIntegerSize,
						widthLevelActualIntegerSize,
						heightLevelActualIntegerSize,
						messageUDeviceSet0Checkerboard1,
						messageDDeviceSet0Checkerboard1,
						messageLDeviceSet0Checkerboard1,
						messageRDeviceSet0Checkerboard1,
						messageUDeviceSet0Checkerboard2,
						messageDDeviceSet0Checkerboard2,
						messageLDeviceSet0Checkerboard2,
						messageRDeviceSet0Checkerboard2,
						(T**)&messageUDeviceSet1Checkerboard1,
						(T**)&messageDDeviceSet1Checkerboard1,
						(T**)&messageLDeviceSet1Checkerboard1,
						(T**)&messageRDeviceSet1Checkerboard1,
						(T**)&messageUDeviceSet1Checkerboard2,
						(T**)&messageDDeviceSet1Checkerboard2,
						(T**)&messageLDeviceSet1Checkerboard2,
						(T**)&messageRDeviceSet1Checkerboard2);

				currentCheckerboardSet = 1;
			}
			else
			{

#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

				messageUDeviceSet0Checkerboard1 = &messageUDeviceCheckerboard1[offsetLevel];
				messageDDeviceSet0Checkerboard1 = &messageDDeviceCheckerboard1[offsetLevel];
				messageLDeviceSet0Checkerboard1 = &messageLDeviceCheckerboard1[offsetLevel];
				messageRDeviceSet0Checkerboard1 = &messageRDeviceCheckerboard1[offsetLevel];

				messageUDeviceSet0Checkerboard2 = &messageUDeviceCheckerboard2[offsetLevel];
				messageDDeviceSet0Checkerboard2 = &messageDDeviceCheckerboard2[offsetLevel];
				messageLDeviceSet0Checkerboard2 = &messageLDeviceCheckerboard2[offsetLevel];
				messageRDeviceSet0Checkerboard2 = &messageRDeviceCheckerboard2[offsetLevel];

#endif

				ProcessOptimizedCPUBPHelperFuncts::copyMessageValuesToNextLevelDown<T>(
						prevWidthLevelActualIntegerSize,
						prevHeightLevelActualIntegerSize,
						widthLevelActualIntegerSize,
						heightLevelActualIntegerSize,
						messageUDeviceSet1Checkerboard1,
						messageDDeviceSet1Checkerboard1,
						messageLDeviceSet1Checkerboard1,
						messageRDeviceSet1Checkerboard1,
						messageUDeviceSet1Checkerboard2,
						messageDDeviceSet1Checkerboard2,
						messageLDeviceSet1Checkerboard2,
						messageRDeviceSet1Checkerboard2,
						(T**)&messageUDeviceSet0Checkerboard1,
						(T**)&messageDDeviceSet0Checkerboard1,
						(T**)&messageLDeviceSet0Checkerboard1,
						(T**)&messageRDeviceSet0Checkerboard1,
						(T**)&messageUDeviceSet0Checkerboard2,
						(T**)&messageDDeviceSet0Checkerboard2,
						(T**)&messageLDeviceSet0Checkerboard2,
						(T**)&messageRDeviceSet0Checkerboard2);

				currentCheckerboardSet = 0;
			}
		}
	}

	ProcessOptimizedCPUBPHelperFuncts::retrieveOutputDisparity<T>(dataCostDeviceCurrentLevelCheckerboard1, dataCostDeviceCurrentLevelCheckerboard2,
			messageUDeviceSet0Checkerboard1, messageDDeviceSet0Checkerboard1, messageLDeviceSet0Checkerboard1, messageRDeviceSet0Checkerboard1,
			messageUDeviceSet0Checkerboard2, messageDDeviceSet0Checkerboard2, messageLDeviceSet0Checkerboard2, messageRDeviceSet0Checkerboard2,
			messageUDeviceSet1Checkerboard1, messageDDeviceSet1Checkerboard1, messageLDeviceSet1Checkerboard1, messageRDeviceSet1Checkerboard1,
			messageUDeviceSet1Checkerboard2, messageDDeviceSet1Checkerboard2, messageLDeviceSet1Checkerboard2, messageRDeviceSet1Checkerboard2,
			resultingDisparityMapCompDevice, widthLevel, heightLevel, currentCheckerboardSet);


#ifndef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

	//printf("ALLOC MULT MEM SEGMENTS\n");

	//free the device storage for the message values used to retrieve the output movement values
	if (currentCheckerboardSet == 0)
	{
		//free device space allocated to message values
		delete [] messageUDeviceSet0Checkerboard1;
		delete [] messageDDeviceSet0Checkerboard1;
		delete [] messageLDeviceSet0Checkerboard1;
		delete [] messageRDeviceSet0Checkerboard1;

		delete [] messageUDeviceSet0Checkerboard2;
		delete [] messageDDeviceSet0Checkerboard2;
		delete [] messageLDeviceSet0Checkerboard2;
		delete [] messageRDeviceSet0Checkerboard2;
	}
	else
	{
		//free device space allocated to message values
		delete [] messageUDeviceSet1Checkerboard1;
		delete [] messageDDeviceSet1Checkerboard1;
		delete [] messageLDeviceSet1Checkerboard1;
		delete [] messageRDeviceSet1Checkerboard1;

		delete [] messageUDeviceSet1Checkerboard2;
		delete [] messageDDeviceSet1Checkerboard2;
		delete [] messageLDeviceSet1Checkerboard2;
		delete [] messageRDeviceSet1Checkerboard2;
	}

	//now free the allocated data space
	delete [] dataCostDeviceCheckerboard1;
	delete [] dataCostDeviceCheckerboard2;

#else

	delete [] dataCostDeviceCheckerboard1;

#endif
}


