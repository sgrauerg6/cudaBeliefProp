/*
 * ProcessBPOnTargetDevice.cpp
 *
 *  Created on: Jun 20, 2019
 *      Author: scott
 */

#include "ProcessBPOnTargetDevice.h"

template<typename T>
ProcessBPOnTargetDevice<T>::ProcessBPOnTargetDevice() {
	// TODO Auto-generated constructor stub

}

template<typename T>
ProcessBPOnTargetDevice<T>::~ProcessBPOnTargetDevice() {
	// TODO Auto-generated destructor stub
}

template<typename T>
DetailedTimings* ProcessBPOnTargetDevice<T>::operator()(float* image1PixelsCompDevice, float* image2PixelsCompDevice, float* resultingDisparityMapCompDevice, BPsettings& algSettings, ProcessBPOnTargetDeviceHelperFuncts<beliefPropProcessingDataType>* processBPHelperFuncts)
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
		halfTotalDataAllLevels += (processBPHelperFuncts->getCheckerboardWidthCPU(widthLevelActualIntegerSize)) * (heightLevelActualIntegerSize) * (totalPossibleMovements);
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
	processBPHelperFuncts->allocateMemoryOnTargetDevice((void**)&dataCostDeviceCheckerboard1, 10*halfTotalDataAllLevels*sizeof(T));
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

	processBPHelperFuncts->allocateMemoryOnTargetDevice((void**)&dataCostDeviceCheckerboard1, halfTotalDataAllLevels*sizeof(T));
	processBPHelperFuncts->allocateMemoryOnTargetDevice((void**)&dataCostDeviceCheckerboard2, halfTotalDataAllLevels*sizeof(T));

#endif


	//now go "back to" the bottom level to initialize the data costs starting at the bottom level and going up the pyramid
	widthLevel = (float)algSettings.widthImages;
	heightLevel = (float)algSettings.heightImages;

	widthLevelActualIntegerSize = (int)roundf(widthLevel);
	heightLevelActualIntegerSize = (int)roundf(heightLevel);

	//printf("INIT DATA COSTS\n");
	//initialize the data cost at the bottom level
	processBPHelperFuncts->initialDataCosts(
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
		offsetLevel += (processBPHelperFuncts->getCheckerboardWidthCPU(widthLevelActualIntegerSize)) * (heightLevelActualIntegerSize) * totalPossibleMovements;

		widthLevel /= 2.0f;
		heightLevel /= 2.0f;

		int prevWidthLevelActualIntegerSize = widthLevelActualIntegerSize;
		int prevHeightLevelActualIntegerSize = heightLevelActualIntegerSize;

		widthLevelActualIntegerSize = (int)ceil(widthLevel);
		heightLevelActualIntegerSize = (int)ceil(heightLevel);
		int widthCheckerboard = processBPHelperFuncts->getCheckerboardWidthCPU(widthLevelActualIntegerSize);

		T* dataCostStereoCheckerboard1 =
				&dataCostDeviceCheckerboard1[prev_level_offset_level];
		T* dataCostStereoCheckerboard2 =
				&dataCostDeviceCheckerboard2[prev_level_offset_level];
		T* dataCostDeviceToWriteToCheckerboard1 =
				&dataCostDeviceCheckerboard1[offsetLevel];
		T* dataCostDeviceToWriteToCheckerboard2 =
				&dataCostDeviceCheckerboard2[offsetLevel];

		//printf("INIT DATA COSTS CURRENT LEVEL\n");
		processBPHelperFuncts->initializeDataCurrentLevel(dataCostStereoCheckerboard1,
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
	int numDataAndMessageSetInCheckerboardAtLevel = (processBPHelperFuncts->getCheckerboardWidthCPU(widthLevelActualIntegerSize)) * heightLevelActualIntegerSize * totalPossibleMovements;

	//allocate the space for the message values in the first checkboard set at the current level
	processBPHelperFuncts->allocateMemoryOnTargetDevice((void**)&messageUDeviceSet0Checkerboard1, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
	processBPHelperFuncts->allocateMemoryOnTargetDevice((void**)&messageDDeviceSet0Checkerboard1, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
	processBPHelperFuncts->allocateMemoryOnTargetDevice((void**)&messageLDeviceSet0Checkerboard1, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
	processBPHelperFuncts->allocateMemoryOnTargetDevice((void**)&messageRDeviceSet0Checkerboard1, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));

	processBPHelperFuncts->allocateMemoryOnTargetDevice((void**)&messageUDeviceSet0Checkerboard2, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
	processBPHelperFuncts->allocateMemoryOnTargetDevice((void**)&messageDDeviceSet0Checkerboard2, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
	processBPHelperFuncts->allocateMemoryOnTargetDevice((void**)&messageLDeviceSet0Checkerboard2, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
	processBPHelperFuncts->allocateMemoryOnTargetDevice((void**)&messageRDeviceSet0Checkerboard2, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));

#endif

	//printf("initializeMessageValsToDefault\n");
	//initialize all the BP message values at every pixel for every disparity to 0
	processBPHelperFuncts->initializeMessageValsToDefault(messageUDeviceSet0Checkerboard1, messageDDeviceSet0Checkerboard1, messageLDeviceSet0Checkerboard1, messageRDeviceSet0Checkerboard1,
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
			processBPHelperFuncts->runBPAtCurrentLevel(algSettings,
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
			processBPHelperFuncts->runBPAtCurrentLevel(algSettings,
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
			int widthCheckerboard = processBPHelperFuncts->getCheckerboardWidthCPU(widthLevelActualIntegerSize);

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

				processBPHelperFuncts->copyMessageValuesToNextLevelDown(
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

				processBPHelperFuncts->copyMessageValuesToNextLevelDown(
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

	processBPHelperFuncts->retrieveOutputDisparity(dataCostDeviceCurrentLevelCheckerboard1, dataCostDeviceCurrentLevelCheckerboard2,
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
		processBPHelperFuncts->freeMemoryOnTargetDevice((void*)messageUDeviceSet0Checkerboard1);
		processBPHelperFuncts->freeMemoryOnTargetDevice((void*)messageDDeviceSet0Checkerboard1);
		processBPHelperFuncts->freeMemoryOnTargetDevice((void*)messageLDeviceSet0Checkerboard1);
		processBPHelperFuncts->freeMemoryOnTargetDevice((void*)messageRDeviceSet0Checkerboard1);

		processBPHelperFuncts->freeMemoryOnTargetDevice((void*)messageUDeviceSet0Checkerboard2);
		processBPHelperFuncts->freeMemoryOnTargetDevice((void*)messageDDeviceSet0Checkerboard2);
		processBPHelperFuncts->freeMemoryOnTargetDevice((void*)messageLDeviceSet0Checkerboard2);
		processBPHelperFuncts->freeMemoryOnTargetDevice((void*)messageRDeviceSet0Checkerboard2);
	}
	else
	{
		//free device space allocated to message values
		processBPHelperFuncts->freeMemoryOnTargetDevice((void*)messageUDeviceSet1Checkerboard1);
		processBPHelperFuncts->freeMemoryOnTargetDevice((void*)messageDDeviceSet1Checkerboard1);
		processBPHelperFuncts->freeMemoryOnTargetDevice((void*)messageLDeviceSet1Checkerboard1);
		processBPHelperFuncts->freeMemoryOnTargetDevice((void*)messageRDeviceSet1Checkerboard1);

		processBPHelperFuncts->freeMemoryOnTargetDevice((void*)messageUDeviceSet1Checkerboard2);
		processBPHelperFuncts->freeMemoryOnTargetDevice((void*)messageDDeviceSet1Checkerboard2);
		processBPHelperFuncts->freeMemoryOnTargetDevice((void*)messageLDeviceSet1Checkerboard2);
		processBPHelperFuncts->freeMemoryOnTargetDevice((void*)messageRDeviceSet1Checkerboard2);
	}

	//now free the allocated data space
	processBPHelperFuncts->freeMemoryOnTargetDevice((void*)dataCostDeviceCheckerboard1);
	processBPHelperFuncts->freeMemoryOnTargetDevice((void*)dataCostDeviceCheckerboard2);


#else

	//now free the allocated data space
	processBPHelperFuncts->freeMemoryOnTargetDevice(dataCostDeviceCheckerboard1);

#endif
}
