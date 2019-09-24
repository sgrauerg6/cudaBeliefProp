/*
 * ProcessBPOnTargetDevice.cpp
 *
 *  Created on: Jun 20, 2019
 *      Author: scott
 */

#include "ProcessBPOnTargetDevice.h"

template<typename T>
int ProcessBPOnTargetDevice<T>::getPaddedCheckerboardWidth(int checkerboardWidth)
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

template<typename T>
unsigned long ProcessBPOnTargetDevice<T>::getNumDataForAlignedMemoryAtLevel(unsigned int widthLevelActualIntegerSize, unsigned int heightLevelActualIntegerSize, unsigned int totalPossibleMovements)
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
template<typename T>
DetailedTimings<Runtime_Type_BP> ProcessBPOnTargetDevice<T>::operator()(float* image1PixelsCompDevice,
	float* image2PixelsCompDevice,
	float* resultingDisparityMapCompDevice, const BPsettings& algSettings, unsigned int widthImages, unsigned int heightImages)
{

	DetailedTimings<Runtime_Type_BP> segmentTimings(timingNames_BP);
	double totalTimeBpIters = 0.0;
	double totalTimeCopyData = 0.0;
	double totalTimeCopyDataKernel = 0.0;
	std::chrono::duration<double> diff;

	//retrieve the total number of possible movements; this is equal to the number of disparity values
	int totalPossibleMovements = NUM_POSSIBLE_DISPARITY_VALUES;

	unsigned long halfTotalDataAllLevels = 0;
	std::unique_ptr<levelProperties[]> processingLevelProperties = std::make_unique<levelProperties[]>(algSettings.numLevels);

	//start at the "bottom level" and work way up to determine amount of space needed to store data costs
	int widthLevel = widthImages;
	int heightLevel = heightImages;

	unsigned long currentOffsetLevel = 0;
	std::unique_ptr<unsigned long[]> offsetAtLevel = std::make_unique<unsigned long[]>(algSettings.numLevels);
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
		widthLevel = (int)ceil(widthLevel / 2.0);
		heightLevel = (int)ceil(heightLevel / 2.0);
	}

	//declare and then allocate the space on the device to store the data cost component at each possible movement at each level of the "pyramid"
	//each checkboard holds half of the data
	T* dataCostDeviceCheckerboard0; //checkerboard 1 includes the pixel in slot (0, 0)
	T* dataCostDeviceCheckerboard1;

	checkerboardMessages<T> messagesDeviceCheckerboard0;
	checkerboardMessages<T> messagesDeviceCheckerboard1;

	auto timeInitSettingsMallocStart = std::chrono::system_clock::now();

#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

	//std::cout << "ALLOC ALL MEMORY\n";
	allocateMemoryOnTargetDevice((void**)& dataCostDeviceCheckerboard0, 10 * halfTotalDataAllLevels * sizeof(T));
	if (dataCostDeviceCheckerboard0 == nullptr)
	{
		std::cout << "Memory alloc failed\n";
	}

	dataCostDeviceCheckerboard1 = &(dataCostDeviceCheckerboard0[1 * (halfTotalDataAllLevels)]);

	messagesDeviceCheckerboard0.messagesU = &(dataCostDeviceCheckerboard0[2 * (halfTotalDataAllLevels)]);
	messagesDeviceCheckerboard0.messagesD = &(dataCostDeviceCheckerboard0[3 * (halfTotalDataAllLevels)]);
	messagesDeviceCheckerboard0.messagesL = &(dataCostDeviceCheckerboard0[4 * (halfTotalDataAllLevels)]);
	messagesDeviceCheckerboard0.messagesR = &(dataCostDeviceCheckerboard0[5 * (halfTotalDataAllLevels)]);

	messagesDeviceCheckerboard1.messagesU = &(dataCostDeviceCheckerboard0[6 * (halfTotalDataAllLevels)]);
	messagesDeviceCheckerboard1.messagesD = &(dataCostDeviceCheckerboard0[7 * (halfTotalDataAllLevels)]);
	messagesDeviceCheckerboard1.messagesL = &(dataCostDeviceCheckerboard0[8 * (halfTotalDataAllLevels)]);
	messagesDeviceCheckerboard1.messagesR = &(dataCostDeviceCheckerboard0[9 * (halfTotalDataAllLevels)]);

#else

	allocateMemoryOnTargetDevice((void**)& dataCostDeviceCheckerboard0,
		halfTotalDataAllLevels * sizeof(T));
	allocateMemoryOnTargetDevice((void**)& dataCostDeviceCheckerboard1,
		halfTotalDataAllLevels * sizeof(T));

#endif

	auto timeInitSettingsMallocEnd = std::chrono::system_clock::now();

	diff = timeInitSettingsMallocEnd - timeInitSettingsMallocStart;
	double totalTimeInitSettingsMallocStart = diff.count();
	segmentTimings.addTiming(INIT_SETTINGS_MALLOC, totalTimeInitSettingsMallocStart);

	auto timeInitDataCostsStart = std::chrono::system_clock::now();

	//std::cout << "INIT DATA COSTS\n";
	//initialize the data cost at the bottom level
	initializeDataCosts(algSettings, processingLevelProperties[0], image1PixelsCompDevice, image2PixelsCompDevice,
		dataCostDeviceCheckerboard0, dataCostDeviceCheckerboard1);
	//std::cout << "DONE INIT DATA COSTS\n";

	auto timeInitDataCostsEnd = std::chrono::system_clock::now();
	diff = timeInitDataCostsEnd - timeInitDataCostsStart;
	double totalTimeGetDataCostsBottomLevel = diff.count();
	segmentTimings.addTiming(DATA_COSTS_BOTTOM_LEVEL, totalTimeGetDataCostsBottomLevel);

	auto timeInitDataCostsHigherLevelsStart =
		std::chrono::system_clock::now();

	//set the data costs at each level from the bottom level "up"
	for (int levelNum = 1; levelNum < algSettings.numLevels; levelNum++)
	{
		T* dataCostStereoCheckerboard1 =
			&dataCostDeviceCheckerboard0[offsetAtLevel[levelNum - 1]];
		T* dataCostStereoCheckerboard2 =
			&dataCostDeviceCheckerboard1[offsetAtLevel[levelNum - 1]];
		T* dataCostDeviceToWriteToCheckerboard1 =
			&dataCostDeviceCheckerboard0[offsetAtLevel[levelNum]];
		T* dataCostDeviceToWriteToCheckerboard2 =
			&dataCostDeviceCheckerboard1[offsetAtLevel[levelNum]];

		//std::cout << "INIT DATA COSTS CURRENT LEVEL\n";
		initializeDataCurrentLevel(processingLevelProperties[levelNum],
			processingLevelProperties[levelNum - 1],
			dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
			dataCostDeviceToWriteToCheckerboard1,
			dataCostDeviceToWriteToCheckerboard2);
		//std::cout << "DONE INIT DATA COSTS CURRENT LEVEL\n";
	}

	currentOffsetLevel = offsetAtLevel[algSettings.numLevels - 1];

	auto timeInitDataCostsHigherLevelsEnd =
		std::chrono::system_clock::now();
	diff = timeInitDataCostsHigherLevelsEnd
		- timeInitDataCostsHigherLevelsStart;

	double totalTimeGetDataCostsHigherLevels = diff.count();
	segmentTimings.addTiming(DATA_COSTS_HIGHER_LEVEL, totalTimeGetDataCostsHigherLevels);

	//declare the space to pass the BP messages
	//need to have two "sets" of checkerboards because
	//the message values at the "higher" level in the image
	//pyramid need copied to a lower level without overwriting
	//values
	T* dataCostDeviceCurrentLevelCheckerboard0;
	T* dataCostDeviceCurrentLevelCheckerboard1;
	checkerboardMessages<T> messagesDeviceSet0Checkerboard0;
	checkerboardMessages<T> messagesDeviceSet0Checkerboard1;
	checkerboardMessages<T> messagesDeviceSet1Checkerboard0;
	checkerboardMessages<T> messagesDeviceSet1Checkerboard1;

	auto timeInitMessageValuesStart = std::chrono::system_clock::now();

	dataCostDeviceCurrentLevelCheckerboard0 =
		&dataCostDeviceCheckerboard0[currentOffsetLevel];
	dataCostDeviceCurrentLevelCheckerboard1 =
		&dataCostDeviceCheckerboard1[currentOffsetLevel];

#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

	messagesDeviceSet0Checkerboard0 = retrieveCurrentCheckerboardMessagesFromOffsetIntoAllCheckerboardMessages(messagesDeviceCheckerboard0, currentOffsetLevel);
	messagesDeviceSet0Checkerboard1 = retrieveCurrentCheckerboardMessagesFromOffsetIntoAllCheckerboardMessages(messagesDeviceCheckerboard1, currentOffsetLevel);

#else

	//retrieve the number of bytes needed to store the data cost/each set of messages in the checkerboard
	int numDataAndMessageSetInCheckerboardAtLevel =
		getNumDataForAlignedMemoryAtLevel(processingLevelProperties[algSettings.numLevels - 1].widthLevel, processingLevelProperties[algSettings.numLevels - 1].heightLevel, totalPossibleMovements);

	//allocate the space for the message values in the first checkboard set at the current level
	messagesDeviceSet0Checkerboard0 = allocateMemoryForCheckerboardMessages(numDataAndMessageSetInCheckerboardAtLevel * sizeof(T));
	messagesDeviceSet0Checkerboard1 = allocateMemoryForCheckerboardMessages(numDataAndMessageSetInCheckerboardAtLevel * sizeof(T));

#endif

	auto timeInitMessageValuesKernelTimeStart =
		std::chrono::system_clock::now();

	//std::cout << "initializeMessageValsToDefault\n";
	//initialize all the BP message values at every pixel for every disparity to 0
	initializeMessageValsToDefault(
		processingLevelProperties[algSettings.numLevels - 1],
		messagesDeviceSet0Checkerboard0,
		messagesDeviceSet0Checkerboard1);
	//std::cout << "DONE initializeMessageValsToDefault\n";

	auto timeInitMessageValuesTimeEnd =
		std::chrono::system_clock::now();
	diff = timeInitMessageValuesTimeEnd
		- timeInitMessageValuesKernelTimeStart;

	double totalTimeInitMessageValuesKernelTime = diff.count();
	segmentTimings.addTiming(INIT_MESSAGES_KERNEL, totalTimeInitMessageValuesKernelTime);

	diff = timeInitMessageValuesTimeEnd - timeInitMessageValuesStart;
	double totalTimeInitMessageVals = diff.count();
	segmentTimings.addTiming(INIT_MESSAGES, totalTimeInitMessageVals);

	//alternate between checkerboard sets 0 and 1
	Checkerboard_Parts currentCheckerboardSet = Checkerboard_Parts::CHECKERBOARD_PART_0;

	//run BP at each level in the "pyramid" starting on top and continuing to the bottom
	//where the final movement values are computed...the message values are passed from
	//the upper level to the lower levels; this pyramid methods causes the BP message values
	//to converge more quickly
	for (int levelNum = algSettings.numLevels - 1; levelNum >= 0;
		levelNum--)
	{
		auto timeBpIterStart = std::chrono::system_clock::now();

		//std::cout << "LEVEL: " << levelNum << std::endl;
		//need to alternate which checkerboard set to work on since copying from one to the other...need to avoid read-write conflict when copying in parallel
		if (currentCheckerboardSet == Checkerboard_Parts::CHECKERBOARD_PART_0) {
			runBPAtCurrentLevel(algSettings, processingLevelProperties[levelNum],
				dataCostDeviceCurrentLevelCheckerboard0,
				dataCostDeviceCurrentLevelCheckerboard1,
				messagesDeviceSet0Checkerboard0,
				messagesDeviceSet0Checkerboard1);
		}
		else {
			runBPAtCurrentLevel(algSettings, processingLevelProperties[levelNum],
				dataCostDeviceCurrentLevelCheckerboard0,
				dataCostDeviceCurrentLevelCheckerboard1,
				messagesDeviceSet1Checkerboard0,
				messagesDeviceSet1Checkerboard1);
		}

		auto timeBpIterEnd = std::chrono::system_clock::now();
		diff = timeBpIterEnd - timeBpIterStart;
		totalTimeBpIters += diff.count();

		auto timeCopyMessageValuesStart = std::chrono::system_clock::now();

		//std::cout << "DONE BP RUN\n";

		//if not at the "bottom level" copy the current message values at the current level to the corresponding slots next level
		if (levelNum > 0)
		{
			//retrieve offset into allocated memory at next level
			currentOffsetLevel = offsetAtLevel[levelNum - 1];

			dataCostDeviceCurrentLevelCheckerboard0 =
				&dataCostDeviceCheckerboard0[currentOffsetLevel];
			dataCostDeviceCurrentLevelCheckerboard1 =
				&dataCostDeviceCheckerboard1[currentOffsetLevel];

			checkerboardMessages<T> messagesDeviceCheckerboard0CopyFrom;
			checkerboardMessages<T> messagesDeviceCheckerboard1CopyFrom;

			if (currentCheckerboardSet == Checkerboard_Parts::CHECKERBOARD_PART_0)
			{
				messagesDeviceCheckerboard0CopyFrom = messagesDeviceSet0Checkerboard0;
				messagesDeviceCheckerboard1CopyFrom = messagesDeviceSet0Checkerboard1;
			}
			else
			{
				messagesDeviceCheckerboard0CopyFrom = messagesDeviceSet1Checkerboard0;
				messagesDeviceCheckerboard1CopyFrom = messagesDeviceSet1Checkerboard1;
			}

			checkerboardMessages<T> messagesDeviceCheckerboard0CopyTo;
			checkerboardMessages<T> messagesDeviceCheckerboard1CopyTo;

#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

			messagesDeviceCheckerboard0CopyTo = retrieveCurrentCheckerboardMessagesFromOffsetIntoAllCheckerboardMessages(messagesDeviceCheckerboard0, currentOffsetLevel);
			messagesDeviceCheckerboard1CopyTo = retrieveCurrentCheckerboardMessagesFromOffsetIntoAllCheckerboardMessages(messagesDeviceCheckerboard1, currentOffsetLevel);

#else

			int totalPossibleMovements = NUM_POSSIBLE_DISPARITY_VALUES;

			//update the number of bytes needed to store each set
			int numDataAndMessageSetInCheckerboardAtLevel = getNumDataForAlignedMemoryAtLevel(processingLevelProperties[levelNum - 1].widthLevel, processingLevelProperties[levelNum - 1].heightLevel, totalPossibleMovements);

			//allocate space in the GPU for the message values in the checkerboard set to copy to
			messagesDeviceCheckerboard0CopyTo = allocateMemoryForCheckerboardMessages(numDataAndMessageSetInCheckerboardAtLevel * sizeof(T));
			messagesDeviceCheckerboard1CopyTo = allocateMemoryForCheckerboardMessages(numDataAndMessageSetInCheckerboardAtLevel * sizeof(T));

#endif

			auto timeCopyMessageValuesKernelStart = std::chrono::system_clock::now();

			copyMessageValuesToNextLevelDown(processingLevelProperties[levelNum],
				processingLevelProperties[levelNum - 1],
				messagesDeviceCheckerboard0CopyFrom,
				messagesDeviceCheckerboard1CopyFrom,
				messagesDeviceCheckerboard0CopyTo,
				messagesDeviceCheckerboard1CopyTo);

			auto timeCopyMessageValuesKernelEnd = std::chrono::system_clock::now();
			diff = timeCopyMessageValuesKernelEnd - timeCopyMessageValuesKernelStart;
			totalTimeCopyDataKernel += diff.count();

			if (currentCheckerboardSet == Checkerboard_Parts::CHECKERBOARD_PART_0)
			{
				messagesDeviceSet1Checkerboard0 = messagesDeviceCheckerboard0CopyTo;
				messagesDeviceSet1Checkerboard1 = messagesDeviceCheckerboard1CopyTo;
			}
			else
			{
				messagesDeviceSet0Checkerboard0 = messagesDeviceCheckerboard0CopyTo;
				messagesDeviceSet0Checkerboard1 = messagesDeviceCheckerboard1CopyTo;
			}

#ifndef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

			//free the now-copied from computed data of the completed level
			freeCheckerboardMessagesMemory(messagesDeviceCheckerboard0CopyFrom);
			freeCheckerboardMessagesMemory(messagesDeviceCheckerboard1CopyFrom);

#endif

			//alternate between checkerboard parts 1 and 2
			currentCheckerboardSet = (currentCheckerboardSet == Checkerboard_Parts::CHECKERBOARD_PART_0) ? Checkerboard_Parts::CHECKERBOARD_PART_1 : Checkerboard_Parts::CHECKERBOARD_PART_0;
		}

		auto timeCopyMessageValuesEnd = std::chrono::system_clock::now();
		diff = timeCopyMessageValuesEnd - timeCopyMessageValuesStart;

		totalTimeCopyData += diff.count();
	}

	auto timeGetOutputDisparityStart = std::chrono::system_clock::now();

	//assume in bottom level when retrieving output disparity
	retrieveOutputDisparity(currentCheckerboardSet,
		processingLevelProperties[0],
		dataCostDeviceCurrentLevelCheckerboard0,
		dataCostDeviceCurrentLevelCheckerboard1,
		messagesDeviceSet0Checkerboard0,
		messagesDeviceSet0Checkerboard1,
		messagesDeviceSet1Checkerboard0,
		messagesDeviceSet1Checkerboard1,
		resultingDisparityMapCompDevice);

	auto timeGetOutputDisparityEnd = std::chrono::system_clock::now();
	diff = timeGetOutputDisparityEnd - timeGetOutputDisparityStart;
	double totalTimeGetOutputDisparity = diff.count();
	segmentTimings.addTiming(OUTPUT_DISPARITY, totalTimeGetOutputDisparity);

	auto timeFinalFreeStart = std::chrono::system_clock::now();

#ifndef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

	//std::cout << "ALLOC MULT MEM SEGMENTS\n";

	//free the device storage for the message values used to retrieve the output movement values
	if (currentCheckerboardSet == 0)
	{
		//free device space allocated to message values
		freeCheckerboardMessagesMemory(messagesDeviceSet0Checkerboard0);
		freeCheckerboardMessagesMemory(messagesDeviceSet0Checkerboard1);
	}
	else
	{
		//free device space allocated to message values
		freeCheckerboardMessagesMemory(messagesDeviceSet1Checkerboard0);
		freeCheckerboardMessagesMemory(messagesDeviceSet1Checkerboard1);
	}

	//now free the allocated data space
	freeMemoryOnTargetDevice((void*)dataCostDeviceCheckerboard0);
	freeMemoryOnTargetDevice((void*)dataCostDeviceCheckerboard1);

#else

	//now free the allocated data space
	freeMemoryOnTargetDevice(dataCostDeviceCheckerboard0);

#endif

	auto timeFinalFreeEnd = std::chrono::system_clock::now();

	diff = timeFinalFreeEnd - timeFinalFreeStart;
	double totalTimeFinalFree = diff.count();
	segmentTimings.addTiming(FINAL_FREE, totalTimeFinalFree);

	segmentTimings.addTiming(BP_ITERS, totalTimeBpIters);
	segmentTimings.addTiming(COPY_DATA, totalTimeCopyData);
	segmentTimings.addTiming(COPY_DATA_KERNEL, totalTimeCopyDataKernel);

	double totalTimed = totalTimeInitSettingsMallocStart
			+ totalTimeGetDataCostsBottomLevel
			+ totalTimeGetDataCostsHigherLevels + totalTimeInitMessageVals
			+ totalTimeBpIters + totalTimeCopyData + totalTimeGetOutputDisparity
			+ totalTimeFinalFree;
	segmentTimings.addTiming(TOTAL_TIMED, totalTimed);

	return segmentTimings;
}

#if (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT)

template class ProcessBPOnTargetDevice<float>;

#elif (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_DOUBLE)

template class ProcessBPOnTargetDevice<double>;

#elif (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF)

#ifdef COMPILING_FOR_ARM
template class ProcessBPOnTargetDevice<float16_t>;
#else
template class ProcessBPOnTargetDevice<short>;
#endif //COMPILING_FOR_ARM

#endif
