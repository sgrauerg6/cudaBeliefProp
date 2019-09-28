/*
 * ProcessBPOnTargetDevice.cpp
 *
 *  Created on: Jun 20, 2019
 *      Author: scott
 */

#include "ProcessBPOnTargetDevice.h"

template<typename T, typename U, typename V>
int ProcessBPOnTargetDevice<T, U, V>::getPaddedCheckerboardWidth(int checkerboardWidth)
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

template<typename T, typename U, typename V>
unsigned long ProcessBPOnTargetDevice<T, U, V>::getNumDataForAlignedMemoryAtLevel(unsigned int widthLevelActualIntegerSize, unsigned int heightLevelActualIntegerSize, unsigned int totalPossibleMovements)
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
template<typename T, typename U, typename V>
std::pair<V, DetailedTimings<Runtime_Type_BP>> ProcessBPOnTargetDevice<T, U, V>::operator()(const std::array<V, 2>& imagesOnTargetDevice,
		const BPsettings& algSettings, unsigned int widthImages, unsigned int heightImages)
{
	DetailedTimings<Runtime_Type_BP> segmentTimings(timingNames_BP);
	std::unordered_map<Runtime_Type_BP, std::pair<timingType, timingType>> runtime_start_end_timings;
	double totalTimeBpIters = 0.0;
	double totalTimeCopyData = 0.0;
	double totalTimeCopyDataKernel = 0.0;

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

	//std::cout << "INIT_SETTINGS_MALLOC" << std::endl;
	runtime_start_end_timings[INIT_SETTINGS_MALLOC].first = std::chrono::system_clock::now();

	//declare and then allocate the space on the device to store the data cost component at each possible movement at each level of the "pyramid"
	//each checkboard holds half of the data
	dataCostData<U> dataCostsDeviceCheckerboardAllLevels; //checkerboard 0 includes the pixel in slot (0, 0)

#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

	checkerboardMessages<U> messagesDeviceAllLevels;

	//call function that allocates all data in single array and then set offsets in array for data costs and message data locations
	std::tie(dataCostsDeviceCheckerboardAllLevels, messagesDeviceAllLevels) =
			allocateAndOrganizeDataCostsAndMessageDataAllLevels(halfTotalDataAllLevels);

#else

	dataCostsDeviceCheckerboardAllLevels = allocateMemoryForDataCosts(halfTotalDataAllLevels);

#endif

	runtime_start_end_timings[INIT_SETTINGS_MALLOC].second = runtime_start_end_timings[DATA_COSTS_BOTTOM_LEVEL].first = std::chrono::system_clock::now();
	//std::cout << "DATA_COSTS_BOTTOM_LEVEL" << std::endl;

	//initialize the data cost at the bottom level
	initializeDataCosts(algSettings, processingLevelProperties[0], imagesOnTargetDevice,
			dataCostsDeviceCheckerboardAllLevels);

	runtime_start_end_timings[DATA_COSTS_BOTTOM_LEVEL].second = runtime_start_end_timings[DATA_COSTS_HIGHER_LEVEL].first = std::chrono::system_clock::now();
	//std::cout << "DATA_COSTS_HIGHER_LEVEL" << std::endl;

	//set the data costs at each level from the bottom level "up"
	for (int levelNum = 1; levelNum < algSettings.numLevels; levelNum++)
	{
		dataCostData<U> dataCostsDeviceCheckerboardPrevLevel =
				retrieveCurrentDataCostsFromOffsetIntoAllDataCosts(dataCostsDeviceCheckerboardAllLevels, offsetAtLevel[levelNum - 1]);
		dataCostData<U> dataCostsDeviceCheckerboardWriteTo =
				retrieveCurrentDataCostsFromOffsetIntoAllDataCosts(dataCostsDeviceCheckerboardAllLevels, offsetAtLevel[levelNum]);

		initializeDataCurrentLevel(processingLevelProperties[levelNum],
			processingLevelProperties[levelNum - 1],
			dataCostsDeviceCheckerboardPrevLevel,
			dataCostsDeviceCheckerboardWriteTo);
	}

	runtime_start_end_timings[DATA_COSTS_HIGHER_LEVEL].second = runtime_start_end_timings[INIT_MESSAGES].first = std::chrono::system_clock::now();
	//std::cout << "INIT_MESSAGES" << std::endl;

	currentOffsetLevel = offsetAtLevel[algSettings.numLevels - 1];

	//declare the space to pass the BP messages; need to have two "sets" of checkerboards because the message
	//data at the "higher" level in the image pyramid need copied to a lower level without overwriting values
	std::array<checkerboardMessages<U>, 2> messagesDevice;

	dataCostData<U> dataCostsDeviceCheckerboardCurrentLevel =
			retrieveCurrentDataCostsFromOffsetIntoAllDataCosts(dataCostsDeviceCheckerboardAllLevels, currentOffsetLevel);

#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

	messagesDevice[0] = retrieveCurrentCheckerboardMessagesFromOffsetIntoAllCheckerboardMessages(messagesDeviceAllLevels, currentOffsetLevel);

#else

	//retrieve the number of bytes needed to store the data cost/each set of messages in the checkerboard
	int numDataAndMessageSetInCheckerboardAtLevel =
		getNumDataForAlignedMemoryAtLevel(processingLevelProperties[algSettings.numLevels - 1].widthLevel, processingLevelProperties[algSettings.numLevels - 1].heightLevel, totalPossibleMovements);

	//allocate the space for the message values in the first checkboard set at the current level
	messagesDevice[0] = allocateMemoryForCheckerboardMessages(numDataAndMessageSetInCheckerboardAtLevel);

#endif

	runtime_start_end_timings[INIT_MESSAGES_KERNEL].first = std::chrono::system_clock::now();
	//std::cout << "INIT_MESSAGES_KERNEL" << std::endl;

	//initialize all the BP message values at every pixel for every disparity to 0
	initializeMessageValsToDefault(
		processingLevelProperties[algSettings.numLevels - 1],
		messagesDevice[0]);

	runtime_start_end_timings[INIT_MESSAGES_KERNEL].second = runtime_start_end_timings[INIT_MESSAGES].second = std::chrono::system_clock::now();

	//alternate between checkerboard sets 0 and 1
	enum Checkerboard_Num { CHECKERBOARD_ZERO = 0, CHECKERBOARD_ONE = 1 };
	Checkerboard_Num currentCheckerboardSet = Checkerboard_Num::CHECKERBOARD_ZERO;

	//run BP at each level in the "pyramid" starting on top and continuing to the bottom
	//where the final movement values are computed...the message values are passed from
	//the upper level to the lower levels; this pyramid methods causes the BP message values
	//to converge more quickly
	for (int levelNum = algSettings.numLevels - 1; levelNum >= 0;
		levelNum--)
	{
		std::chrono::duration<double> diff;
		auto timeBpIterStart = std::chrono::system_clock::now();

		//std::cout << "RUN_BP" << std::endl;
		//need to alternate which checkerboard set to work on since copying from one to the other...need to avoid read-write conflict when copying in parallel
		runBPAtCurrentLevel(algSettings, processingLevelProperties[levelNum],
						dataCostsDeviceCheckerboardCurrentLevel,
						messagesDevice[currentCheckerboardSet]);
		//std::cout << "RUN_BP2" << std::endl;

		auto timeBpIterEnd = std::chrono::system_clock::now();
		diff = timeBpIterEnd - timeBpIterStart;
		totalTimeBpIters += diff.count();

		auto timeCopyMessageValuesStart = std::chrono::system_clock::now();

		//if not at the "bottom level" copy the current message values at the current level to the corresponding slots next level
		if (levelNum > 0)
		{
			//retrieve offset into allocated memory at next level
			currentOffsetLevel = offsetAtLevel[levelNum - 1];

			dataCostsDeviceCheckerboardCurrentLevel =
						retrieveCurrentDataCostsFromOffsetIntoAllDataCosts(dataCostsDeviceCheckerboardAllLevels, currentOffsetLevel);

#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

			messagesDevice[(currentCheckerboardSet + 1) % 2] = retrieveCurrentCheckerboardMessagesFromOffsetIntoAllCheckerboardMessages(messagesDeviceAllLevels, currentOffsetLevel);

#else

			int totalPossibleMovements = NUM_POSSIBLE_DISPARITY_VALUES;

			//update the number of bytes needed to store each set
			int numDataAndMessageSetInCheckerboardAtLevel = getNumDataForAlignedMemoryAtLevel(processingLevelProperties[levelNum - 1].widthLevel, processingLevelProperties[levelNum - 1].heightLevel, totalPossibleMovements);

			//allocate space in the GPU for the message values in the checkerboard set to copy to
			messagesDevice[(currentCheckerboardSet + 1) % 2] = allocateMemoryForCheckerboardMessages(numDataAndMessageSetInCheckerboardAtLevel);

#endif

			auto timeCopyMessageValuesKernelStart = std::chrono::system_clock::now();

			//currentCheckerboardSet = index copying data from
			//(currentCheckerboardSet + 1) % 2 = index copying data to
			copyMessageValuesToNextLevelDown(processingLevelProperties[levelNum],
				processingLevelProperties[levelNum - 1],
				messagesDevice[currentCheckerboardSet],
				messagesDevice[(currentCheckerboardSet + 1) % 2]);

			auto timeCopyMessageValuesKernelEnd = std::chrono::system_clock::now();
			diff = timeCopyMessageValuesKernelEnd - timeCopyMessageValuesKernelStart;
			totalTimeCopyDataKernel += diff.count();

#ifndef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

			//free the now-copied from computed data of the completed level
			freeCheckerboardMessagesMemory(messagesDevice[currentCheckerboardSet]);

#endif

			//alternate between checkerboard parts 1 and 2
			currentCheckerboardSet = (currentCheckerboardSet == Checkerboard_Num::CHECKERBOARD_ZERO) ? Checkerboard_Num::CHECKERBOARD_ONE : Checkerboard_Num::CHECKERBOARD_ZERO;
		}

		auto timeCopyMessageValuesEnd = std::chrono::system_clock::now();
		diff = timeCopyMessageValuesEnd - timeCopyMessageValuesStart;

		totalTimeCopyData += diff.count();
	}

	runtime_start_end_timings[OUTPUT_DISPARITY].first = std::chrono::system_clock::now();
	//std::cout << "OUTPUT_DISPARITY" << std::endl;

	//assume in bottom level when retrieving output disparity
	V resultingDisparityMapCompDevice = retrieveOutputDisparity(
		processingLevelProperties[0],
		dataCostsDeviceCheckerboardCurrentLevel,
		messagesDevice[currentCheckerboardSet]);

	runtime_start_end_timings[OUTPUT_DISPARITY].second = runtime_start_end_timings[FINAL_FREE].first = std::chrono::system_clock::now();

#ifndef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

	//free the device storage for the message values used to retrieve the output movement values
	if (currentCheckerboardSet == 0)
	{
		//free device space allocated to message values
		freeCheckerboardMessagesMemory(messagesDevice[0]);
	}
	else
	{
		//free device space allocated to message values
		freeCheckerboardMessagesMemory(messagesDevice[1]);
	}

	//now free the allocated data space
	freeDataCostsMemory(dataCostsDeviceCheckerboardAllLevels);

#else

	//now free the allocated data space; all data in single array when
	//USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT selected
	freeDataCostsAllDataInSingleArray(dataCostsDeviceCheckerboardAllLevels);

#endif

	runtime_start_end_timings[FINAL_FREE].second = std::chrono::system_clock::now();
	//std::cout << "FINAL_FREE" << std::endl;

	//retrieve the timing for each runtime segment and add to vector in timings map
	std::for_each(runtime_start_end_timings.begin(),
			runtime_start_end_timings.end(),
			[&segmentTimings] (const auto& currentRuntimeNameAndTiming) {
				segmentTimings.addTiming(currentRuntimeNameAndTiming.first,
						(timingInSecondsDoublePrecision(currentRuntimeNameAndTiming.second.second - currentRuntimeNameAndTiming.second.first)).count());
			});

	segmentTimings.addTiming(BP_ITERS, totalTimeBpIters);
	segmentTimings.addTiming(COPY_DATA, totalTimeCopyData);
	segmentTimings.addTiming(COPY_DATA_KERNEL, totalTimeCopyDataKernel);

	double totalTimed = segmentTimings.getMedianTiming(INIT_SETTINGS_MALLOC)
			+ segmentTimings.getMedianTiming(DATA_COSTS_BOTTOM_LEVEL)
			+ segmentTimings.getMedianTiming(DATA_COSTS_HIGHER_LEVEL) + segmentTimings.getMedianTiming(INIT_MESSAGES)
			+ totalTimeBpIters + totalTimeCopyData + segmentTimings.getMedianTiming(OUTPUT_DISPARITY)
			+ segmentTimings.getMedianTiming(FINAL_FREE);
	segmentTimings.addTiming(TOTAL_TIMED, totalTimed);

	return std::make_pair(resultingDisparityMapCompDevice, segmentTimings);
}

#if (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT)

template class ProcessBPOnTargetDevice<float, float*, float*>;

#elif (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_DOUBLE)

template class ProcessBPOnTargetDevice<double, double*>;

#elif (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF)

#ifdef COMPILING_FOR_ARM
template class ProcessBPOnTargetDevice<float16_t, float16_t*>;
#else
template class ProcessBPOnTargetDevice<short, short*>;
template class ProcessBPOnTargetDevice<half, half*>;
#endif //COMPILING_FOR_ARM

#endif
