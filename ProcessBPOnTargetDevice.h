/*
 * ProcessBPOnTargetDevice.h
 *
 *  Created on: Jun 20, 2019
 *      Author: scott
 */

#ifndef PROCESSBPONTARGETDEVICE_H_
#define PROCESSBPONTARGETDEVICE_H_

#include "DetailedTimings.h"
#include "bpStereoParameters.h"
#include <math.h>
#include <chrono>

#define RUN_DETAILED_TIMING


template<typename T>
class ProcessBPOnTargetDevice {
public:
	ProcessBPOnTargetDevice() { }
	virtual ~ProcessBPOnTargetDevice() { }

		virtual int getCheckerboardWidthTargetDevice(int widthLevelActualIntegerSize)
		{
			return (int)ceil(((float)widthLevelActualIntegerSize) / 2.0);
		}

		virtual void allocateMemoryOnTargetDevice(void** arrayToAllocate, unsigned long numBytesAllocate) = 0;

		virtual void freeMemoryOnTargetDevice(void* arrayToFree) = 0;

		virtual void initializeDataCosts(BPsettings& algSettings, levelProperties& currentLevelProperties,
				float* image1PixelsCompDevice, float* image2PixelsCompDevice, T* dataCostDeviceCheckerboard1,
				T* dataCostDeviceCheckerboard2) = 0;

		virtual void initializeDataCurrentLevel(levelProperties& currentLevelPropertes,
				levelProperties& prevLevelProperties,
				T* dataCostStereoCheckerboard1,
				T* dataCostStereoCheckerboard2,
				T* dataCostDeviceToWriteToCheckerboard1,
				T* dataCostDeviceToWriteToCheckerboard2) = 0;

		virtual void initializeMessageValsToDefault(
				levelProperties& currentLevelPropertes,
				T* messageUDeviceCheckerboard1,
				T* messageDDeviceCheckerboard1,
				T* messageLDeviceCheckerboard1,
				T* messageRDeviceCheckerboard1,
				T* messageUDeviceCheckerboard2,
				T* messageDDeviceCheckerboard2,
				T* messageLDeviceCheckerboard2,
				T* messageRDeviceCheckerboard2) = 0;

		virtual void runBPAtCurrentLevel(BPsettings& algSettings,
				levelProperties& currentLevelPropertes,
				T* dataCostDeviceCurrentLevelCheckerboard1,
				T* dataCostDeviceCurrentLevelCheckerboard2,
				T* messageUDeviceCheckerboard1,
				T* messageDDeviceCheckerboard1,
				T* messageLDeviceCheckerboard1,
				T* messageRDeviceCheckerboard1,
				T* messageUDeviceCheckerboard2,
				T* messageDDeviceCheckerboard2,
				T* messageLDeviceCheckerboard2,
				T* messageRDeviceCheckerboard2) = 0;

		virtual void copyMessageValuesToNextLevelDown(
				levelProperties& currentLevelPropertes,
				levelProperties& nextLevelPropertes,
				T* messageUDeviceCheckerboard1CopyFrom,
				T* messageDDeviceCheckerboard1CopyFrom,
				T* messageLDeviceCheckerboard1CopyFrom,
				T* messageRDeviceCheckerboard1CopyFrom,
				T* messageUDeviceCheckerboard2CopyFrom,
				T* messageDDeviceCheckerboard2CopyFrom,
				T* messageLDeviceCheckerboard2CopyFrom,
				T* messageRDeviceCheckerboard2CopyFrom,
				T* messageUDeviceCheckerboard1CopyTo,
				T* messageDDeviceCheckerboard1CopyTo,
				T* messageLDeviceCheckerboard1CopyTo,
				T* messageRDeviceCheckerboard1CopyTo,
				T* messageUDeviceCheckerboard2CopyTo,
				T* messageDDeviceCheckerboard2CopyTo,
				T* messageLDeviceCheckerboard2CopyTo,
				T* messageRDeviceCheckerboard2CopyTo) = 0;

		virtual void retrieveOutputDisparity(
				int currentCheckerboardSet,
				levelProperties& levelPropertes,
				T* dataCostDeviceCurrentLevelCheckerboard1,
				T* dataCostDeviceCurrentLevelCheckerboard2,
				T* messageUDeviceSet0Checkerboard1,
				T* messageDDeviceSet0Checkerboard1,
				T* messageLDeviceSet0Checkerboard1,
				T* messageRDeviceSet0Checkerboard1,
				T* messageUDeviceSet0Checkerboard2,
				T* messageDDeviceSet0Checkerboard2,
				T* messageLDeviceSet0Checkerboard2,
				T* messageRDeviceSet0Checkerboard2,
				T* messageUDeviceSet1Checkerboard1,
				T* messageDDeviceSet1Checkerboard1,
				T* messageLDeviceSet1Checkerboard1,
				T* messageRDeviceSet1Checkerboard1,
				T* messageUDeviceSet1Checkerboard2,
				T* messageDDeviceSet1Checkerboard2,
				T* messageLDeviceSet1Checkerboard2,
				T* messageRDeviceSet1Checkerboard2,
				float* resultingDisparityMapCompDevice) = 0;

		virtual int getPaddedCheckerboardWidth(int checkerboardWidth)
		{
			if ((checkerboardWidth % NUM_DATA_ALIGN_WIDTH) == 0)
			{
				return checkerboardWidth;
			}
			else
			{
				unsigned int paddedCheckerboardWidth = checkerboardWidth + ((NUM_DATA_ALIGN_WIDTH - checkerboardWidth % NUM_DATA_ALIGN_WIDTH));
				return paddedCheckerboardWidth;
			}
		}

		unsigned long getNumDataForAlignedMemoryAtLevel(unsigned int widthLevelActualIntegerSize, unsigned int heightLevelActualIntegerSize, unsigned int totalPossibleMovements)
		{
			unsigned long numDataAtLevel = ((unsigned long)getPaddedCheckerboardWidth((int)getCheckerboardWidthTargetDevice(widthLevelActualIntegerSize)))
								* ((unsigned long)heightLevelActualIntegerSize) * (unsigned long)totalPossibleMovements;
			unsigned long numBytesAtLevel = numDataAtLevel * sizeof(T);

			if ((numBytesAtLevel % BYTES_ALIGN_MEMORY) == 0)
			{
				return numDataAtLevel;
			}
			else
			{
				numBytesAtLevel += (BYTES_ALIGN_MEMORY - numBytesAtLevel % BYTES_ALIGN_MEMORY);
				unsigned long paddedNumDataAtLevel = numBytesAtLevel / sizeof(T);
				return paddedNumDataAtLevel;
			}
		}

	//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
	//input is images image1Pixels and image1Pixels
	//output is resultingDisparityMap
	DetailedTimings* operator()(float* image1PixelsCompDevice,
			float* image2PixelsCompDevice,
			float* resultingDisparityMapCompDevice, BPsettings& algSettings)
	{

#ifdef RUN_DETAILED_TIMING

		DetailedTimings* timingsPointer = new DetailedTimings;
		double totalTimeBpIters = 0.0;
		double totalTimeCopyData = 0.0;
		double totalTimeCopyDataKernel = 0.0;
		std::chrono::duration<double> diff;

#endif

		//printf("Start opt CPU\n");
		//retrieve the total number of possible movements; this is equal to the number of disparity values
		int totalPossibleMovements = NUM_POSSIBLE_DISPARITY_VALUES;

		unsigned long halfTotalDataAllLevels = 0;
		levelProperties* processingLevelProperties = new levelProperties[algSettings.numLevels];

		//start at the "bottom level" and work way up to determine amount of space needed to store data costs
		int widthLevel = algSettings.widthImages;
		int heightLevel = algSettings.heightImages;

		unsigned long currentOffsetLevel = 0;
		unsigned long* offsetAtLevel = new unsigned long[algSettings.numLevels];
		offsetAtLevel[0] = 0;

		//compute "half" the total number of pixels in including every level of the "pyramid"
		//using "half" because the data is split in two using the checkerboard scheme
		for (int levelNum = 0; levelNum < algSettings.numLevels; levelNum++)
		{
			processingLevelProperties[levelNum].widthLevel =
					widthLevel;
			processingLevelProperties[levelNum].widthCheckerboardLevel =
					getCheckerboardWidthTargetDevice(
							widthLevel);
			processingLevelProperties[levelNum].paddedWidthCheckerboardLevel =
					getPaddedCheckerboardWidth(processingLevelProperties[levelNum].widthCheckerboardLevel);
			processingLevelProperties[levelNum].heightLevel =
					heightLevel;

			if (levelNum > 0)
			{
				//width is half since each part of the checkboard contains half the values going across
				//retrieve offset where the data starts at the "current level"
				currentOffsetLevel += getNumDataForAlignedMemoryAtLevel(
									processingLevelProperties[levelNum - 1].widthLevel,
									processingLevelProperties[levelNum - 1].heightLevel,
									totalPossibleMovements);
				offsetAtLevel[levelNum] = currentOffsetLevel;
			}

			halfTotalDataAllLevels += getNumDataForAlignedMemoryAtLevel(widthLevel, heightLevel, totalPossibleMovements);
			widthLevel = (int) ceil(widthLevel / 2.0);
			heightLevel = (int) ceil(heightLevel / 2.0);
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

#ifdef RUN_DETAILED_TIMING

		auto timeInitSettingsMallocStart = std::chrono::system_clock::now();

#endif

#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

		//printf("ALLOC ALL MEMORY\n");
		allocateMemoryOnTargetDevice((void**)&dataCostDeviceCheckerboard1, 10*halfTotalDataAllLevels*sizeof(T));
		if (dataCostDeviceCheckerboard1 == nullptr)
		{
			printf("Memory alloc failed\n");
		}

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

		allocateMemoryOnTargetDevice((void**) &dataCostDeviceCheckerboard1,
				halfTotalDataAllLevels * sizeof(T));
		allocateMemoryOnTargetDevice((void**) &dataCostDeviceCheckerboard2,
				halfTotalDataAllLevels * sizeof(T));

#endif

#ifdef RUN_DETAILED_TIMING

		auto timeInitSettingsMallocEnd = std::chrono::system_clock::now();

		diff = timeInitSettingsMallocEnd - timeInitSettingsMallocStart;
		double totalTimeInitSettingsMallocStart = diff.count();

		auto timeInitDataCostsStart = std::chrono::system_clock::now();

#endif

		//printf("INIT DATA COSTS\n");
		//initialize the data cost at the bottom level
		initializeDataCosts(algSettings, processingLevelProperties[0], image1PixelsCompDevice, image2PixelsCompDevice,
				dataCostDeviceCheckerboard1, dataCostDeviceCheckerboard2);
		//printf("DONE INIT DATA COSTS\n");

#ifdef RUN_DETAILED_TIMING

		auto timeInitDataCostsEnd = std::chrono::system_clock::now();
		diff = timeInitDataCostsEnd - timeInitDataCostsStart;

		double totalTimeGetDataCostsBottomLevel = diff.count();
		auto timeInitDataCostsHigherLevelsStart =
				std::chrono::system_clock::now();

#endif

		//set the data costs at each level from the bottom level "up"
		for (int levelNum = 1; levelNum < algSettings.numLevels; levelNum++)
		{
			T* dataCostStereoCheckerboard1 =
					&dataCostDeviceCheckerboard1[offsetAtLevel[levelNum - 1]];
			T* dataCostStereoCheckerboard2 =
					&dataCostDeviceCheckerboard2[offsetAtLevel[levelNum - 1]];
			T* dataCostDeviceToWriteToCheckerboard1 =
					&dataCostDeviceCheckerboard1[offsetAtLevel[levelNum]];
			T* dataCostDeviceToWriteToCheckerboard2 =
					&dataCostDeviceCheckerboard2[offsetAtLevel[levelNum]];

			//printf("INIT DATA COSTS CURRENT LEVEL\n");
			initializeDataCurrentLevel(processingLevelProperties[levelNum],
					processingLevelProperties[levelNum - 1],
					dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
					dataCostDeviceToWriteToCheckerboard1,
					dataCostDeviceToWriteToCheckerboard2);
			//printf("DONE INIT DATA COSTS CURRENT LEVEL\n");
		}

		currentOffsetLevel = offsetAtLevel[algSettings.numLevels - 1];

#ifdef RUN_DETAILED_TIMING

		auto timeInitDataCostsHigherLevelsEnd =
				std::chrono::system_clock::now();
		diff = timeInitDataCostsHigherLevelsEnd
				- timeInitDataCostsHigherLevelsStart;

		double totalTimeGetDataCostsHigherLevels = diff.count();

#endif

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

#ifdef RUN_DETAILED_TIMING

		auto timeInitMessageValuesStart = std::chrono::system_clock::now();

#endif

		dataCostDeviceCurrentLevelCheckerboard1 =
				&dataCostDeviceCheckerboard1[currentOffsetLevel];
		dataCostDeviceCurrentLevelCheckerboard2 =
				&dataCostDeviceCheckerboard2[currentOffsetLevel];

#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

		messageUDeviceSet0Checkerboard1 = &messageUDeviceCheckerboard1[currentOffsetLevel];
		messageDDeviceSet0Checkerboard1 = &messageDDeviceCheckerboard1[currentOffsetLevel];
		messageLDeviceSet0Checkerboard1 = &messageLDeviceCheckerboard1[currentOffsetLevel];
		messageRDeviceSet0Checkerboard1 = &messageRDeviceCheckerboard1[currentOffsetLevel];

		messageUDeviceSet0Checkerboard2 = &messageUDeviceCheckerboard2[currentOffsetLevel];
		messageDDeviceSet0Checkerboard2 = &messageDDeviceCheckerboard2[currentOffsetLevel];
		messageLDeviceSet0Checkerboard2 = &messageLDeviceCheckerboard2[currentOffsetLevel];
		messageRDeviceSet0Checkerboard2 = &messageRDeviceCheckerboard2[currentOffsetLevel];

#else

		//retrieve the number of bytes needed to store the data cost/each set of messages in the checkerboard
		int numDataAndMessageSetInCheckerboardAtLevel =
				getNumDataForAlignedMemoryAtLevel(processingLevelProperties[algSettings.numLevels - 1].widthLevel, processingLevelProperties[algSettings.numLevels - 1].heightLevel, totalPossibleMovements);

		//allocate the space for the message values in the first checkboard set at the current level
		allocateMemoryOnTargetDevice((void**) &messageUDeviceSet0Checkerboard1,
				numDataAndMessageSetInCheckerboardAtLevel * sizeof(T));
		allocateMemoryOnTargetDevice((void**) &messageDDeviceSet0Checkerboard1,
				numDataAndMessageSetInCheckerboardAtLevel * sizeof(T));
		allocateMemoryOnTargetDevice((void**) &messageLDeviceSet0Checkerboard1,
				numDataAndMessageSetInCheckerboardAtLevel * sizeof(T));
		allocateMemoryOnTargetDevice((void**) &messageRDeviceSet0Checkerboard1,
				numDataAndMessageSetInCheckerboardAtLevel * sizeof(T));

		allocateMemoryOnTargetDevice((void**) &messageUDeviceSet0Checkerboard2,
				numDataAndMessageSetInCheckerboardAtLevel * sizeof(T));
		allocateMemoryOnTargetDevice((void**) &messageDDeviceSet0Checkerboard2,
				numDataAndMessageSetInCheckerboardAtLevel * sizeof(T));
		allocateMemoryOnTargetDevice((void**) &messageLDeviceSet0Checkerboard2,
				numDataAndMessageSetInCheckerboardAtLevel * sizeof(T));
		allocateMemoryOnTargetDevice((void**) &messageRDeviceSet0Checkerboard2,
				numDataAndMessageSetInCheckerboardAtLevel * sizeof(T));

#endif

#ifdef RUN_DETAILED_TIMING

		auto timeInitMessageValuesKernelTimeStart =
				std::chrono::system_clock::now();

#endif

		//printf("initializeMessageValsToDefault\n");
		//initialize all the BP message values at every pixel for every disparity to 0
		initializeMessageValsToDefault(
				processingLevelProperties[algSettings.numLevels - 1],
				messageUDeviceSet0Checkerboard1,
				messageDDeviceSet0Checkerboard1,
				messageLDeviceSet0Checkerboard1,
				messageRDeviceSet0Checkerboard1,
				messageUDeviceSet0Checkerboard2,
				messageDDeviceSet0Checkerboard2,
				messageLDeviceSet0Checkerboard2,
				messageRDeviceSet0Checkerboard2);
		//printf("DONE initializeMessageValsToDefault\n");

#ifdef RUN_DETAILED_TIMING

		auto timeInitMessageValuesKernelTimeEnd =
				std::chrono::system_clock::now();
		diff = timeInitMessageValuesKernelTimeEnd
				- timeInitMessageValuesKernelTimeStart;

		double totalTimeInitMessageValuesKernelTime = diff.count();

		auto timeInitMessageValuesEnd = std::chrono::system_clock::now();
		diff = timeInitMessageValuesEnd - timeInitMessageValuesStart;

		double totalTimeInitMessageVals = diff.count();

#endif

		//alternate between checkerboard sets 0 and 1
		int currentCheckerboardSet = 0;

		//run BP at each level in the "pyramid" starting on top and continuing to the bottom
		//where the final movement values are computed...the message values are passed from
		//the upper level to the lower levels; this pyramid methods causes the BP message values
		//to converge more quickly
		for (int levelNum = algSettings.numLevels - 1; levelNum >= 0;
				levelNum--)
		{

#ifdef RUN_DETAILED_TIMING

			auto timeBpIterStart = std::chrono::system_clock::now();

#endif

			//printf("LEVEL: %d\n", levelNum);
			//need to alternate which checkerboard set to work on since copying from one to the other...need to avoid read-write conflict when copying in parallel
			if (currentCheckerboardSet == 0) {
				runBPAtCurrentLevel(algSettings, processingLevelProperties[levelNum],
						dataCostDeviceCurrentLevelCheckerboard1,
						dataCostDeviceCurrentLevelCheckerboard2,
						messageUDeviceSet0Checkerboard1,
						messageDDeviceSet0Checkerboard1,
						messageLDeviceSet0Checkerboard1,
						messageRDeviceSet0Checkerboard1,
						messageUDeviceSet0Checkerboard2,
						messageDDeviceSet0Checkerboard2,
						messageLDeviceSet0Checkerboard2,
						messageRDeviceSet0Checkerboard2);
			} else {
				runBPAtCurrentLevel(algSettings, processingLevelProperties[levelNum],
						dataCostDeviceCurrentLevelCheckerboard1,
						dataCostDeviceCurrentLevelCheckerboard2,
						messageUDeviceSet1Checkerboard1,
						messageDDeviceSet1Checkerboard1,
						messageLDeviceSet1Checkerboard1,
						messageRDeviceSet1Checkerboard1,
						messageUDeviceSet1Checkerboard2,
						messageDDeviceSet1Checkerboard2,
						messageLDeviceSet1Checkerboard2,
						messageRDeviceSet1Checkerboard2);
			}

#ifdef RUN_DETAILED_TIMING

			auto timeBpIterEnd = std::chrono::system_clock::now();
			diff = timeBpIterEnd - timeBpIterStart;
			totalTimeBpIters += diff.count();

			auto timeCopyMessageValuesStart = std::chrono::system_clock::now();

#endif

			//printf("DONE BP RUN\n");

			//if not at the "bottom level" copy the current message values at the current level to the corresponding slots next level
			if (levelNum > 0)
			{
				//retrieve offset into allocated memory at next level
				currentOffsetLevel = offsetAtLevel[levelNum - 1];

				dataCostDeviceCurrentLevelCheckerboard1 =
						&dataCostDeviceCheckerboard1[currentOffsetLevel];
				dataCostDeviceCurrentLevelCheckerboard2 =
						&dataCostDeviceCheckerboard2[currentOffsetLevel];

				T* messageUDeviceCheckerboard1CopyFrom;
				T* messageDDeviceCheckerboard1CopyFrom;
				T* messageLDeviceCheckerboard1CopyFrom;
				T* messageRDeviceCheckerboard1CopyFrom;
				T* messageUDeviceCheckerboard2CopyFrom;
				T* messageDDeviceCheckerboard2CopyFrom;
				T* messageLDeviceCheckerboard2CopyFrom;
				T* messageRDeviceCheckerboard2CopyFrom;

				if (currentCheckerboardSet == 0)
				{
					messageUDeviceCheckerboard1CopyFrom = messageUDeviceSet0Checkerboard1;
					messageDDeviceCheckerboard1CopyFrom = messageDDeviceSet0Checkerboard1;
					messageLDeviceCheckerboard1CopyFrom = messageLDeviceSet0Checkerboard1;
					messageRDeviceCheckerboard1CopyFrom = messageRDeviceSet0Checkerboard1;

					messageUDeviceCheckerboard2CopyFrom = messageUDeviceSet0Checkerboard2;
					messageDDeviceCheckerboard2CopyFrom = messageDDeviceSet0Checkerboard2;
					messageLDeviceCheckerboard2CopyFrom = messageLDeviceSet0Checkerboard2;
					messageRDeviceCheckerboard2CopyFrom = messageRDeviceSet0Checkerboard2;
				}
				else
				{
					messageUDeviceCheckerboard1CopyFrom = messageUDeviceSet1Checkerboard1;
					messageDDeviceCheckerboard1CopyFrom = messageDDeviceSet1Checkerboard1;
					messageLDeviceCheckerboard1CopyFrom = messageLDeviceSet1Checkerboard1;
					messageRDeviceCheckerboard1CopyFrom = messageRDeviceSet1Checkerboard1;

					messageUDeviceCheckerboard2CopyFrom = messageUDeviceSet1Checkerboard2;
					messageDDeviceCheckerboard2CopyFrom = messageDDeviceSet1Checkerboard2;
					messageLDeviceCheckerboard2CopyFrom = messageLDeviceSet1Checkerboard2;
					messageRDeviceCheckerboard2CopyFrom = messageRDeviceSet1Checkerboard2;
				}

				T* messageUDeviceCheckerboard1CopyTo;
				T* messageDDeviceCheckerboard1CopyTo;
				T* messageLDeviceCheckerboard1CopyTo;
				T* messageRDeviceCheckerboard1CopyTo;

				T* messageUDeviceCheckerboard2CopyTo;
				T* messageDDeviceCheckerboard2CopyTo;
				T* messageLDeviceCheckerboard2CopyTo;
				T* messageRDeviceCheckerboard2CopyTo;

#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

				messageUDeviceCheckerboard1CopyTo = &messageUDeviceCheckerboard1[currentOffsetLevel];
				messageDDeviceCheckerboard1CopyTo = &messageDDeviceCheckerboard1[currentOffsetLevel];
				messageLDeviceCheckerboard1CopyTo = &messageLDeviceCheckerboard1[currentOffsetLevel];
				messageRDeviceCheckerboard1CopyTo = &messageRDeviceCheckerboard1[currentOffsetLevel];

				messageUDeviceCheckerboard2CopyTo = &messageUDeviceCheckerboard2[currentOffsetLevel];
				messageDDeviceCheckerboard2CopyTo = &messageDDeviceCheckerboard2[currentOffsetLevel];
				messageLDeviceCheckerboard2CopyTo = &messageLDeviceCheckerboard2[currentOffsetLevel];
				messageRDeviceCheckerboard2CopyTo = &messageRDeviceCheckerboard2[currentOffsetLevel];

#else

				int totalPossibleMovements = NUM_POSSIBLE_DISPARITY_VALUES;

				//update the number of bytes needed to store each set
				int numDataAndMessageSetInCheckerboardAtLevel = getNumDataForAlignedMemoryAtLevel(processingLevelProperties[levelNum - 1].widthLevel, processingLevelProperties[levelNum - 1].heightLevel, totalPossibleMovements);

				//allocate space in the GPU for the message values in the checkerboard set to copy to
				allocateMemoryOnTargetDevice((void**) &messageUDeviceCheckerboard1CopyTo, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
				allocateMemoryOnTargetDevice((void**) &messageDDeviceCheckerboard1CopyTo, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
				allocateMemoryOnTargetDevice((void**) &messageLDeviceCheckerboard1CopyTo, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
				allocateMemoryOnTargetDevice((void**) &messageRDeviceCheckerboard1CopyTo, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));

				allocateMemoryOnTargetDevice((void**) &messageUDeviceCheckerboard2CopyTo, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
				allocateMemoryOnTargetDevice((void**) &messageDDeviceCheckerboard2CopyTo, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
				allocateMemoryOnTargetDevice((void**) &messageLDeviceCheckerboard2CopyTo, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
				allocateMemoryOnTargetDevice((void**) &messageRDeviceCheckerboard2CopyTo, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));

#endif

				auto timeCopyMessageValuesKernelStart = std::chrono::system_clock::now();

				copyMessageValuesToNextLevelDown(processingLevelProperties[levelNum],
						processingLevelProperties[levelNum - 1],
						messageUDeviceCheckerboard1CopyFrom,
						messageDDeviceCheckerboard1CopyFrom,
						messageLDeviceCheckerboard1CopyFrom,
						messageRDeviceCheckerboard1CopyFrom,
						messageUDeviceCheckerboard2CopyFrom,
						messageDDeviceCheckerboard2CopyFrom,
						messageLDeviceCheckerboard2CopyFrom,
						messageRDeviceCheckerboard2CopyFrom,
						messageUDeviceCheckerboard1CopyTo,
						messageDDeviceCheckerboard1CopyTo,
						messageLDeviceCheckerboard1CopyTo,
						messageRDeviceCheckerboard1CopyTo,
						messageUDeviceCheckerboard2CopyTo,
						messageDDeviceCheckerboard2CopyTo,
						messageLDeviceCheckerboard2CopyTo,
						messageRDeviceCheckerboard2CopyTo);

				auto timeCopyMessageValuesKernelEnd = std::chrono::system_clock::now();
				diff = timeCopyMessageValuesKernelEnd - timeCopyMessageValuesKernelStart;
				totalTimeCopyDataKernel += diff.count();

				if (currentCheckerboardSet == 0)
				{
					messageUDeviceSet1Checkerboard1 =
							messageUDeviceCheckerboard1CopyTo;
					messageDDeviceSet1Checkerboard1 =
							messageDDeviceCheckerboard1CopyTo;
					messageLDeviceSet1Checkerboard1 =
							messageLDeviceCheckerboard1CopyTo;
					messageRDeviceSet1Checkerboard1 =
							messageRDeviceCheckerboard1CopyTo;

					messageUDeviceSet1Checkerboard2 =
							messageUDeviceCheckerboard2CopyTo;
					messageDDeviceSet1Checkerboard2 =
							messageDDeviceCheckerboard2CopyTo;
					messageLDeviceSet1Checkerboard2 =
							messageLDeviceCheckerboard2CopyTo;
					messageRDeviceSet1Checkerboard2 =
							messageRDeviceCheckerboard2CopyTo;
				}
				else
				{
					messageUDeviceSet0Checkerboard1 =
							messageUDeviceCheckerboard1CopyTo;
					messageDDeviceSet0Checkerboard1 =
							messageDDeviceCheckerboard1CopyTo;
					messageLDeviceSet0Checkerboard1 =
							messageLDeviceCheckerboard1CopyTo;
					messageRDeviceSet0Checkerboard1 =
							messageRDeviceCheckerboard1CopyTo;

					messageUDeviceSet0Checkerboard2 =
							messageUDeviceCheckerboard2CopyTo;
					messageDDeviceSet0Checkerboard2 =
							messageDDeviceCheckerboard2CopyTo;
					messageLDeviceSet0Checkerboard2 =
							messageLDeviceCheckerboard2CopyTo;
					messageRDeviceSet0Checkerboard2 =
							messageRDeviceCheckerboard2CopyTo;
				}

#ifndef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

				//free the now-copied from computed data of the completed level
				freeMemoryOnTargetDevice(messageUDeviceCheckerboard1CopyFrom);
				freeMemoryOnTargetDevice(messageDDeviceCheckerboard1CopyFrom);
				freeMemoryOnTargetDevice(messageLDeviceCheckerboard1CopyFrom);
				freeMemoryOnTargetDevice(messageRDeviceCheckerboard1CopyFrom);

				freeMemoryOnTargetDevice(messageUDeviceCheckerboard2CopyFrom);
				freeMemoryOnTargetDevice(messageDDeviceCheckerboard2CopyFrom);
				freeMemoryOnTargetDevice(messageLDeviceCheckerboard2CopyFrom);
				freeMemoryOnTargetDevice(messageRDeviceCheckerboard2CopyFrom);

#endif
				currentCheckerboardSet = (currentCheckerboardSet + 1) % 2;
			}

#ifdef RUN_DETAILED_TIMING

			auto timeCopyMessageValuesEnd = std::chrono::system_clock::now();
			diff = timeCopyMessageValuesEnd - timeCopyMessageValuesStart;

			totalTimeCopyData += diff.count();

#endif
		}

#ifdef RUN_DETAILED_TIMING

		auto timeGetOutputDisparityStart = std::chrono::system_clock::now();

#endif

		//assume in bottom level when retrieving output disparity
		retrieveOutputDisparity(currentCheckerboardSet,
				processingLevelProperties[0],
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messageUDeviceSet0Checkerboard1,
				messageDDeviceSet0Checkerboard1,
				messageLDeviceSet0Checkerboard1,
				messageRDeviceSet0Checkerboard1,
				messageUDeviceSet0Checkerboard2,
				messageDDeviceSet0Checkerboard2,
				messageLDeviceSet0Checkerboard2,
				messageRDeviceSet0Checkerboard2,
				messageUDeviceSet1Checkerboard1,
				messageDDeviceSet1Checkerboard1,
				messageLDeviceSet1Checkerboard1,
				messageRDeviceSet1Checkerboard1,
				messageUDeviceSet1Checkerboard2,
				messageDDeviceSet1Checkerboard2,
				messageLDeviceSet1Checkerboard2,
				messageRDeviceSet1Checkerboard2,
				resultingDisparityMapCompDevice);

#ifdef RUN_DETAILED_TIMING

		auto timeGetOutputDisparityEnd = std::chrono::system_clock::now();
		diff = timeGetOutputDisparityEnd - timeGetOutputDisparityStart;
		double totalTimeGetOutputDisparity = diff.count();

		auto timeFinalFreeStart = std::chrono::system_clock::now();

#endif

#ifndef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

		//printf("ALLOC MULT MEM SEGMENTS\n");

		//free the device storage for the message values used to retrieve the output movement values
		if (currentCheckerboardSet == 0)
		{
			//free device space allocated to message values
			freeMemoryOnTargetDevice((void*) messageUDeviceSet0Checkerboard1);
			freeMemoryOnTargetDevice((void*) messageDDeviceSet0Checkerboard1);
			freeMemoryOnTargetDevice((void*) messageLDeviceSet0Checkerboard1);
			freeMemoryOnTargetDevice((void*) messageRDeviceSet0Checkerboard1);

			freeMemoryOnTargetDevice((void*) messageUDeviceSet0Checkerboard2);
			freeMemoryOnTargetDevice((void*) messageDDeviceSet0Checkerboard2);
			freeMemoryOnTargetDevice((void*) messageLDeviceSet0Checkerboard2);
			freeMemoryOnTargetDevice((void*) messageRDeviceSet0Checkerboard2);
		}
		else
		{
			//free device space allocated to message values
			freeMemoryOnTargetDevice((void*) messageUDeviceSet1Checkerboard1);
			freeMemoryOnTargetDevice((void*) messageDDeviceSet1Checkerboard1);
			freeMemoryOnTargetDevice((void*) messageLDeviceSet1Checkerboard1);
			freeMemoryOnTargetDevice((void*) messageRDeviceSet1Checkerboard1);

			freeMemoryOnTargetDevice((void*) messageUDeviceSet1Checkerboard2);
			freeMemoryOnTargetDevice((void*) messageDDeviceSet1Checkerboard2);
			freeMemoryOnTargetDevice((void*) messageLDeviceSet1Checkerboard2);
			freeMemoryOnTargetDevice((void*) messageRDeviceSet1Checkerboard2);
		}

		//now free the allocated data space
		freeMemoryOnTargetDevice((void*) dataCostDeviceCheckerboard1);
		freeMemoryOnTargetDevice((void*) dataCostDeviceCheckerboard2);

#else
		//now free the allocated data space
		freeMemoryOnTargetDevice(dataCostDeviceCheckerboard1);

#endif

		delete offsetAtLevel;
		delete processingLevelProperties;

#ifdef RUN_DETAILED_TIMING

		double timeBpItersKernelTotalTime = 0.0;
		auto timeFinalFreeEnd = std::chrono::system_clock::now();

		diff = timeFinalFreeEnd - timeFinalFreeStart;
		double totalTimeFinalFree = diff.count();

		double totalMemoryProcessingTime = 0.0;/*totalTimeInitSettingsMallocStart + totalTimeFinalUnbindFree
		 + (totalTimeInitMessageVals - totalTimeInitMessageValuesKernelTime);
		 //+ (totalTimeCopyData - timeCopyDataKernelTotalTime)
		 //+ (timeBpItersKernelTotalTime - timeBpItersKernelTotalTime);*/
		double totalComputationProcessing = 0.0;/*totalTimeGetDataCostsBottomLevel
		 + totalTimeGetDataCostsHigherLevels
		 + totalTimeInitMessageValuesKernelTime + totalTimeCopyData
		 + timeBpItersKernelTotalTime + totalTimeGetOutputDisparity;*/
		double totalTimed = totalTimeInitSettingsMallocStart
		 + totalTimeGetDataCostsBottomLevel
		 + totalTimeGetDataCostsHigherLevels + totalTimeInitMessageVals
		 + totalTimeBpIters + totalTimeCopyData + totalTimeGetOutputDisparity
		 + totalTimeFinalFree;
		timingsPointer->totalTimeInitSettingsMallocStart.push_back(
				totalTimeInitSettingsMallocStart);
		timingsPointer->totalTimeGetDataCostsBottomLevel.push_back(
				totalTimeGetDataCostsBottomLevel);
		timingsPointer->totalTimeGetDataCostsHigherLevels.push_back(
				totalTimeGetDataCostsHigherLevels);
		timingsPointer->totalTimeInitMessageVals.push_back(
				totalTimeInitMessageVals);
		timingsPointer->totalTimeInitMessageValuesKernelTime.push_back(
				totalTimeInitMessageValuesKernelTime);
		timingsPointer->totalTimeBpIters.push_back(totalTimeBpIters);
		timingsPointer->timeBpItersKernelTotalTime.push_back(
				timeBpItersKernelTotalTime);
		timingsPointer->totalTimeCopyData.push_back(totalTimeCopyData);
		timingsPointer->timeCopyDataKernelTotalTime.push_back(
				totalTimeCopyDataKernel);
		timingsPointer->timeCopyDataMemoryManagementTotalTime.push_back(
				totalTimeCopyData - totalTimeCopyDataKernel);
		timingsPointer->totalTimeGetOutputDisparity.push_back(
				totalTimeGetOutputDisparity);
		timingsPointer->totalTimeFinalFree.push_back(totalTimeFinalFree);
		timingsPointer->totalTimed.push_back(totalTimed);
		timingsPointer->totalMemoryProcessingTime.push_back(
				totalMemoryProcessingTime);
		timingsPointer->totalComputationProcessing.push_back(
				totalComputationProcessing);
		return timingsPointer;

#else

		return nullptr;

#endif
	}
};

#endif /* PROCESSBPONTARGETDEVICE_H_ */
