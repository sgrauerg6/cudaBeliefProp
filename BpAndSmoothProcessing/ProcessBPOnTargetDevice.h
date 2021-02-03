/*
 * ProcessBPOnTargetDevice.h
 *
 *  Created on: Jun 20, 2019
 *      Author: scott
 */

#ifndef PROCESSBPONTARGETDEVICE_H_
#define PROCESSBPONTARGETDEVICE_H_

#include <math.h>
#include <chrono>
#include <unordered_map>
#include <memory>
#include <array>
#include <utility>
#include "../ParameterFiles/bpStereoParameters.h"
#include "../ParameterFiles/bpRunSettings.h"
#include "../ParameterFiles/bpStructsAndEnums.h"
#include "../RuntimeTiming/DetailedTimings.h"
#include "../RuntimeTiming/DetailedTimingBPConsts.h"

typedef std::chrono::time_point<std::chrono::system_clock> timingType;
using timingInSecondsDoublePrecision = std::chrono::duration<double>;

template<typename T, typename U, typename V=float*>
class ProcessBPOnTargetDevice {
public:
	ProcessBPOnTargetDevice() { }
	virtual ~ProcessBPOnTargetDevice() { }

	virtual unsigned int getCheckerboardWidthTargetDevice(const unsigned int widthLevelActualIntegerSize) {
		return (unsigned int)ceil(((float)widthLevelActualIntegerSize) / 2.0f);
	}

	virtual void allocateRawMemoryOnTargetDevice(void** arrayToAllocate, const unsigned long numBytesAllocate) = 0;

	virtual U allocateMemoryOnTargetDevice(const unsigned long numData) = 0;

	virtual void freeRawMemoryOnTargetDevice(void* arrayToFree) = 0;

	virtual void freeMemoryOnTargetDevice(U memoryToFree) = 0;

	virtual void initializeDataCosts(const BPsettings& algSettings, const levelProperties& currentLevelProperties,
			const std::array<V, 2>& imagesOnTargetDevice, const dataCostData<U>& dataCostDeviceCheckerboard) = 0;

	virtual void initializeDataCurrentLevel(const levelProperties& currentLevelProperties,
			const levelProperties& prevLevelProperties,
			const dataCostData<U>& dataCostDeviceCheckerboard,
			const dataCostData<U>& dataCostDeviceCheckerboardWriteTo) = 0;

	virtual void initializeMessageValsToDefault(
			const levelProperties& currentLevelProperties,
			const checkerboardMessages<U>& messagesDevice) = 0;

	virtual void runBPAtCurrentLevel(const BPsettings& algSettings,
			const levelProperties& currentLevelProperties,
			const dataCostData<U>& dataCostDeviceCheckerboard,
			const checkerboardMessages<U>& messagesDevice) = 0;

	virtual void copyMessageValuesToNextLevelDown(
			const levelProperties& currentLevelProperties,
			const levelProperties& nextlevelProperties,
			const checkerboardMessages<U>& messagesDeviceCopyFrom,
			const checkerboardMessages<U>& messagesDeviceCopyTo) = 0;

	virtual V retrieveOutputDisparity(
			const levelProperties& levelProperties,
			const dataCostData<U>& dataCostDeviceCheckerboard,
			const checkerboardMessages<U>& messagesDevice) = 0;

	virtual unsigned int getPaddedCheckerboardWidth(const unsigned int checkerboardWidth);

	unsigned long getNumDataForAlignedMemoryAtLevel(const unsigned int widthLevelActualIntegerSize,
			const unsigned int heightLevelActualIntegerSize, const unsigned int totalPossibleMovements);

	virtual void freeCheckerboardMessagesMemory(const checkerboardMessages<U>& checkerboardMessagesToFree)
	{
		std::for_each(checkerboardMessagesToFree.checkerboardMessagesAtLevel.begin(), checkerboardMessagesToFree.checkerboardMessagesAtLevel.end(),
			[this](auto& checkerboardMessagesSet) {
			this->freeMemoryOnTargetDevice(checkerboardMessagesSet); });
	}

	virtual checkerboardMessages<U> allocateMemoryForCheckerboardMessages(const unsigned long numDataAllocatePerMessage)
	{
		checkerboardMessages<U> outputCheckerboardMessages;
		std::for_each(outputCheckerboardMessages.checkerboardMessagesAtLevel.begin(), outputCheckerboardMessages.checkerboardMessagesAtLevel.end(),
			[this, numDataAllocatePerMessage](auto& checkerboardMessagesSet) {
			checkerboardMessagesSet = this->allocateMemoryOnTargetDevice(numDataAllocatePerMessage); });

		return outputCheckerboardMessages;
	}

	virtual checkerboardMessages<U> retrieveCurrentCheckerboardMessagesFromOffsetIntoAllCheckerboardMessages(
			const checkerboardMessages<U>& allCheckerboardMessages, const unsigned long offsetIntoAllCheckerboardMessages)
	{
		checkerboardMessages<U> outputCheckerboardMessages;
		for (unsigned int i = 0; i < outputCheckerboardMessages.checkerboardMessagesAtLevel.size(); i++)
		{
			outputCheckerboardMessages.checkerboardMessagesAtLevel[i] =
				&((allCheckerboardMessages.checkerboardMessagesAtLevel[i])[offsetIntoAllCheckerboardMessages]);
		}

		return outputCheckerboardMessages;
	}

	virtual void freeDataCostsMemory(const dataCostData<U>& dataCostsToFree)
	{
		freeMemoryOnTargetDevice(dataCostsToFree.dataCostCheckerboard0);
		freeMemoryOnTargetDevice(dataCostsToFree.dataCostCheckerboard1);
	}

	virtual dataCostData<U> allocateMemoryForDataCosts(const unsigned long numDataAllocatePerDataCostsCheckerboard)
	{
		dataCostData<U> outputDataCosts;
		outputDataCosts.dataCostCheckerboard0 = allocateMemoryOnTargetDevice(numDataAllocatePerDataCostsCheckerboard);
		outputDataCosts.dataCostCheckerboard1 = allocateMemoryOnTargetDevice(numDataAllocatePerDataCostsCheckerboard);

		return outputDataCosts;
	}

	virtual std::pair<dataCostData<U>, checkerboardMessages<U>> allocateAndOrganizeDataCostsAndMessageDataAllLevels(
			const unsigned long numDataAllocatePerDataCostsMessageDataArray)
	{
		dataCostData<U> dataCostsDeviceCheckerboardAllLevels;
		dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0 = allocateMemoryOnTargetDevice(10u*numDataAllocatePerDataCostsMessageDataArray);
		dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard1 =
				&(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[1 * (numDataAllocatePerDataCostsMessageDataArray)]);

		checkerboardMessages<U> messagesDeviceAllLevels;
		for (unsigned int i = 0; i < messagesDeviceAllLevels.checkerboardMessagesAtLevel.size(); i++) {
			messagesDeviceAllLevels.checkerboardMessagesAtLevel[i] =
					&(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[(i + 2) * (numDataAllocatePerDataCostsMessageDataArray)]);
		}

		return std::make_pair(dataCostsDeviceCheckerboardAllLevels, messagesDeviceAllLevels);
	}

	virtual void freeDataCostsAllDataInSingleArray(const dataCostData<U>& dataCostsToFree)
	{
		freeMemoryOnTargetDevice(dataCostsToFree.dataCostCheckerboard0);
	}

	virtual dataCostData<U> retrieveCurrentDataCostsFromOffsetIntoAllDataCosts(const dataCostData<U>& allDataCosts, const unsigned long offsetIntoAllDataCosts)
	{
		dataCostData<U> outputDataCosts;
		outputDataCosts.dataCostCheckerboard0 = &(allDataCosts.dataCostCheckerboard0[offsetIntoAllDataCosts]);
		outputDataCosts.dataCostCheckerboard1 = &(allDataCosts.dataCostCheckerboard1[offsetIntoAllDataCosts]);

		return outputDataCosts;
	}

	//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
	//input is images image1Pixels and image1Pixels
	//output is resultingDisparityMap
	std::pair<V, DetailedTimings<Runtime_Type_BP>> operator()(const std::array<V, 2>& imagesOnTargetDevice,
			const BPsettings& algSettings, const unsigned int widthImages, const unsigned int heightImages);
};

template<typename T, typename U, typename V>
unsigned int ProcessBPOnTargetDevice<T, U, V>::getPaddedCheckerboardWidth(const unsigned int checkerboardWidth)
{
	//add "padding" to checkerboard width if necessary for alignment
	return ((checkerboardWidth % bp_params::NUM_DATA_ALIGN_WIDTH) == 0) ?
			checkerboardWidth :
			(checkerboardWidth + (bp_params::NUM_DATA_ALIGN_WIDTH - (checkerboardWidth % bp_params::NUM_DATA_ALIGN_WIDTH)));
}

template<typename T, typename U, typename V>
unsigned long ProcessBPOnTargetDevice<T, U, V>::getNumDataForAlignedMemoryAtLevel(const unsigned int widthLevelActualIntegerSize,
		const unsigned int heightLevelActualIntegerSize, const unsigned int totalPossibleMovements)
{
	const unsigned long numDataAtLevel = (unsigned long)getPaddedCheckerboardWidth(getCheckerboardWidthTargetDevice(widthLevelActualIntegerSize))
		* ((unsigned long)heightLevelActualIntegerSize) * (unsigned long)totalPossibleMovements;
	unsigned long numBytesAtLevel = numDataAtLevel * sizeof(T);

	if ((numBytesAtLevel % bp_params::BYTES_ALIGN_MEMORY) == 0) {
		return numDataAtLevel;
	}
	else {
		numBytesAtLevel += (bp_params::BYTES_ALIGN_MEMORY - (numBytesAtLevel % bp_params::BYTES_ALIGN_MEMORY));
		return (numBytesAtLevel / sizeof(T));
	}
}

//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
//input is images image1Pixels and image1Pixels
//output is resultingDisparityMap
template<typename T, typename U, typename V>
std::pair<V, DetailedTimings<Runtime_Type_BP>> ProcessBPOnTargetDevice<T, U, V>::operator()(const std::array<V, 2> & imagesOnTargetDevice,
	const BPsettings& algSettings, const unsigned int widthImages, const unsigned int heightImages)
{
	DetailedTimings<Runtime_Type_BP> segmentTimings(timingNames_BP);
	std::unordered_map<Runtime_Type_BP, std::pair<timingType, timingType>> runtime_start_end_timings;
	double totalTimeBpIters{0.0};
	double totalTimeCopyData{0.0};
	double totalTimeCopyDataKernel{0.0};
	unsigned long halfTotalDataAllLevels{0ul};

	//start at the "bottom level" and work way up to determine amount of space needed to store data costs
	std::unique_ptr<levelProperties[]> processingLevelProperties = std::make_unique<levelProperties[]>(algSettings.numLevels);
	unsigned int widthLevel{widthImages};
	unsigned int heightLevel{heightImages};
	unsigned long currentOffsetLevel{0ul};

	std::unique_ptr<unsigned long[]> offsetAtLevel = std::make_unique<unsigned long[]>(algSettings.numLevels);
	offsetAtLevel[0] = 0;

	//compute "half" the total number of pixels in including every level of the "pyramid"
	//using "half" because the data is split in two using the checkerboard scheme
	for (unsigned int levelNum = 0; levelNum < algSettings.numLevels; levelNum++)
	{
		processingLevelProperties[levelNum].widthLevel = widthLevel;
		processingLevelProperties[levelNum].widthCheckerboardLevel =
			getCheckerboardWidthTargetDevice(widthLevel);
		processingLevelProperties[levelNum].paddedWidthCheckerboardLevel =
			getPaddedCheckerboardWidth(processingLevelProperties[levelNum].widthCheckerboardLevel);
		processingLevelProperties[levelNum].heightLevel = heightLevel;

		if (levelNum > 0)
		{
			//width is half since each part of the checkboard contains half the values going across
			//retrieve offset where the data starts at the "current level"
			currentOffsetLevel += getNumDataForAlignedMemoryAtLevel(
				processingLevelProperties[levelNum - 1].widthLevel,
				processingLevelProperties[levelNum - 1].heightLevel,
				bp_params::NUM_POSSIBLE_DISPARITY_VALUES);
			offsetAtLevel[levelNum] = currentOffsetLevel;
		}

		halfTotalDataAllLevels += getNumDataForAlignedMemoryAtLevel(widthLevel, heightLevel, bp_params::NUM_POSSIBLE_DISPARITY_VALUES);
		widthLevel = (unsigned int)ceil((float)widthLevel / 2.0f);
		heightLevel = (unsigned int)ceil((float)heightLevel / 2.0f);
	}

	runtime_start_end_timings[Runtime_Type_BP::INIT_SETTINGS_MALLOC].first = std::chrono::system_clock::now();

	//declare and allocate the space on the device to store the data cost component at each possible movement at each level of the "pyramid"
	//as well as the message data used for bp processing
	//each checkerboard holds half of the data
	//checkerboard 0 includes the pixel in slot (0, 0)
	dataCostData<U> dataCostsDeviceCheckerboardAllLevels;
	checkerboardMessages<U> messagesDeviceAllLevels;

#ifdef _WIN32
	//assuming that width includes padding
	if /*constexpr*/ (USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT)
#else
	if (USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT)
#endif
	{
		//call function that allocates all data in single array and then set offsets in array for data costs and message data locations
		std::tie(dataCostsDeviceCheckerboardAllLevels, messagesDeviceAllLevels) =
			allocateAndOrganizeDataCostsAndMessageDataAllLevels(halfTotalDataAllLevels);
	}
	else
	{
		dataCostsDeviceCheckerboardAllLevels = allocateMemoryForDataCosts(halfTotalDataAllLevels);
	}

	runtime_start_end_timings[Runtime_Type_BP::INIT_SETTINGS_MALLOC].second = std::chrono::system_clock::now();
	runtime_start_end_timings[Runtime_Type_BP::DATA_COSTS_BOTTOM_LEVEL].first = std::chrono::system_clock::now();

	//initialize the data cost at the bottom level
	initializeDataCosts(algSettings, processingLevelProperties[0], imagesOnTargetDevice,
		dataCostsDeviceCheckerboardAllLevels);

	runtime_start_end_timings[Runtime_Type_BP::DATA_COSTS_BOTTOM_LEVEL].second = std::chrono::system_clock::now();
	runtime_start_end_timings[Runtime_Type_BP::DATA_COSTS_HIGHER_LEVEL].first = std::chrono::system_clock::now();

	//set the data costs at each level from the bottom level "up"
	for (unsigned int levelNum = 1u; levelNum < algSettings.numLevels; levelNum++)
	{
		dataCostData<U> dataCostsDeviceCheckerboardPrevLevel =
			retrieveCurrentDataCostsFromOffsetIntoAllDataCosts(dataCostsDeviceCheckerboardAllLevels, offsetAtLevel[levelNum - 1u]);
		dataCostData<U> dataCostsDeviceCheckerboardWriteTo =
			retrieveCurrentDataCostsFromOffsetIntoAllDataCosts(dataCostsDeviceCheckerboardAllLevels, offsetAtLevel[levelNum]);

		initializeDataCurrentLevel(processingLevelProperties[levelNum], processingLevelProperties[levelNum - 1],
			dataCostsDeviceCheckerboardPrevLevel, dataCostsDeviceCheckerboardWriteTo);
	}

	runtime_start_end_timings[Runtime_Type_BP::DATA_COSTS_HIGHER_LEVEL].second = std::chrono::system_clock::now();
	runtime_start_end_timings[Runtime_Type_BP::INIT_MESSAGES].first = std::chrono::system_clock::now();

	//get offset into data at current processing level of pyramid
	currentOffsetLevel = offsetAtLevel[algSettings.numLevels - 1u];
	dataCostData<U> dataCostsDeviceCheckerboardCurrentLevel =
		retrieveCurrentDataCostsFromOffsetIntoAllDataCosts(dataCostsDeviceCheckerboardAllLevels, currentOffsetLevel);

	//declare the space to pass the BP messages; need to have two "sets" of checkerboards because the message
	//data at the "higher" level in the image pyramid need copied to a lower level without overwriting values
	std::array<checkerboardMessages<U>, 2> messagesDevice;

#ifdef _WIN32
	//assuming that width includes padding
	if /*constexpr*/ (USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT)
#else
	if (USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT)
#endif
	{
		messagesDevice[0] = retrieveCurrentCheckerboardMessagesFromOffsetIntoAllCheckerboardMessages(messagesDeviceAllLevels, currentOffsetLevel);
	}
	else
	{
		//retrieve the number of bytes needed to store the data cost/each set of messages in the checkerboard
		const unsigned int numDataAndMessageSetInCheckerboardAtLevel =
			getNumDataForAlignedMemoryAtLevel(processingLevelProperties[algSettings.numLevels - 1u].widthLevel,
					processingLevelProperties[algSettings.numLevels - 1u].heightLevel, bp_params::NUM_POSSIBLE_DISPARITY_VALUES);

		//allocate the space for the message values in the first checkboard set at the current level
		messagesDevice[0] = allocateMemoryForCheckerboardMessages(numDataAndMessageSetInCheckerboardAtLevel);
	}

	runtime_start_end_timings[Runtime_Type_BP::INIT_MESSAGES_KERNEL].first = std::chrono::system_clock::now();

	//initialize all the BP message values at every pixel for every disparity to 0
	initializeMessageValsToDefault(processingLevelProperties[algSettings.numLevels - 1u], messagesDevice[0]);

	runtime_start_end_timings[Runtime_Type_BP::INIT_MESSAGES_KERNEL].second = std::chrono::system_clock::now();
	runtime_start_end_timings[Runtime_Type_BP::INIT_MESSAGES].second = std::chrono::system_clock::now();

	//alternate between checkerboard sets 0 and 1
	enum Checkerboard_Num { CHECKERBOARD_ZERO = 0, CHECKERBOARD_ONE = 1 };
	Checkerboard_Num currentCheckerboardSet{Checkerboard_Num::CHECKERBOARD_ZERO};

	//run BP at each level in the "pyramid" starting on top and continuing to the bottom
	//where the final movement values are computed...the message values are passed from
	//the upper level to the lower levels; this pyramid methods causes the BP message values
	//to converge more quickly
	for (int levelNum = (int)algSettings.numLevels - 1; levelNum >= 0; levelNum--)
	{
		std::chrono::duration<double> diff;
		const auto timeBpIterStart = std::chrono::system_clock::now();

		//need to alternate which checkerboard set to work on since copying from one to the other...need to avoid read-write conflict when copying in parallel
		runBPAtCurrentLevel(algSettings, processingLevelProperties[(unsigned int)levelNum],
			dataCostsDeviceCheckerboardCurrentLevel, messagesDevice[currentCheckerboardSet]);

		const auto timeBpIterEnd = std::chrono::system_clock::now();
		diff = timeBpIterEnd - timeBpIterStart;
		totalTimeBpIters += diff.count();
		const auto timeCopyMessageValuesStart = std::chrono::system_clock::now();

		//if not at the "bottom level" copy the current message values at the current level to the corresponding slots next level
		if (levelNum > 0)
		{
			//retrieve offset into allocated memory at next level
			currentOffsetLevel = offsetAtLevel[levelNum - 1];

			dataCostsDeviceCheckerboardCurrentLevel =
				retrieveCurrentDataCostsFromOffsetIntoAllDataCosts(dataCostsDeviceCheckerboardAllLevels, currentOffsetLevel);

#ifdef _WIN32
			//assuming that width includes padding
			if /*constexpr*/ (USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT)
#else
			if (USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT)
#endif
			{
				messagesDevice[(currentCheckerboardSet + 1) % 2] =
						retrieveCurrentCheckerboardMessagesFromOffsetIntoAllCheckerboardMessages(messagesDeviceAllLevels, currentOffsetLevel);
			}
			else
			{
				//update the number of bytes needed to store each set
				const unsigned int numDataAndMessageSetInCheckerboardAtLevel =
						getNumDataForAlignedMemoryAtLevel(processingLevelProperties[levelNum - 1].widthLevel,
								                          processingLevelProperties[levelNum - 1].heightLevel,
								                          bp_params::NUM_POSSIBLE_DISPARITY_VALUES);

				//allocate space in the GPU for the message values in the checkerboard set to copy to
				messagesDevice[(currentCheckerboardSet + 1) % 2] = allocateMemoryForCheckerboardMessages(numDataAndMessageSetInCheckerboardAtLevel);
			}

			const auto timeCopyMessageValuesKernelStart = std::chrono::system_clock::now();

			//currentCheckerboardSet = index copying data from
			//(currentCheckerboardSet + 1) % 2 = index copying data to
			copyMessageValuesToNextLevelDown(processingLevelProperties[levelNum], processingLevelProperties[levelNum - 1],
					messagesDevice[currentCheckerboardSet], messagesDevice[(currentCheckerboardSet + 1) % 2]);

			const auto timeCopyMessageValuesKernelEnd = std::chrono::system_clock::now();
			diff = timeCopyMessageValuesKernelEnd - timeCopyMessageValuesKernelStart;
			totalTimeCopyDataKernel += diff.count();

#ifdef _WIN32
			//assuming that width includes padding
			if /*constexpr*/ (!USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT)
#else
			if (!USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT)
#endif
			{
				//free the now-copied from computed data of the completed level
				freeCheckerboardMessagesMemory(messagesDevice[currentCheckerboardSet]);
			}

			//alternate between checkerboard parts 1 and 2
			currentCheckerboardSet = (currentCheckerboardSet == Checkerboard_Num::CHECKERBOARD_ZERO) ?
					Checkerboard_Num::CHECKERBOARD_ONE : Checkerboard_Num::CHECKERBOARD_ZERO;
		}

		const auto timeCopyMessageValuesEnd = std::chrono::system_clock::now();
		diff = timeCopyMessageValuesEnd - timeCopyMessageValuesStart;
		totalTimeCopyData += diff.count();
	}

	runtime_start_end_timings[Runtime_Type_BP::OUTPUT_DISPARITY].first = std::chrono::system_clock::now();

	//assume in bottom level when retrieving output disparity
	const V resultingDisparityMapCompDevice = retrieveOutputDisparity(processingLevelProperties[0],
			dataCostsDeviceCheckerboardCurrentLevel, messagesDevice[currentCheckerboardSet]);

	runtime_start_end_timings[Runtime_Type_BP::OUTPUT_DISPARITY].second = std::chrono::system_clock::now();
	runtime_start_end_timings[Runtime_Type_BP::FINAL_FREE].first = std::chrono::system_clock::now();

#ifdef _WIN32
			//assuming that width includes padding
			if /*constexpr*/ (USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT)
#else
			if (USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT)
#endif
	{
		//now free the allocated data space; all data in single array when
		//USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT set to true
		freeDataCostsAllDataInSingleArray(dataCostsDeviceCheckerboardAllLevels);
	}
	else
	{
		//free the device storage allocated to the message values used to retrieve the output movement values
		freeCheckerboardMessagesMemory(messagesDevice[(currentCheckerboardSet == 0) ? 0 : 1]);

		//now free the allocated data space
		freeDataCostsMemory(dataCostsDeviceCheckerboardAllLevels);
	}

	runtime_start_end_timings[Runtime_Type_BP::FINAL_FREE].second = std::chrono::system_clock::now();

	//retrieve the timing for each runtime segment and add to vector in timings map
	std::for_each(runtime_start_end_timings.begin(), runtime_start_end_timings.end(),
		[&segmentTimings](const auto& currentRuntimeNameAndTiming) {
		segmentTimings.addTiming(currentRuntimeNameAndTiming.first,
			(timingInSecondsDoublePrecision(currentRuntimeNameAndTiming.second.second - currentRuntimeNameAndTiming.second.first)).count());
	});

	segmentTimings.addTiming(Runtime_Type_BP::BP_ITERS, totalTimeBpIters);
	segmentTimings.addTiming(Runtime_Type_BP::COPY_DATA, totalTimeCopyData);
	segmentTimings.addTiming(Runtime_Type_BP::COPY_DATA_KERNEL, totalTimeCopyDataKernel);

	const double totalTimed = segmentTimings.getMedianTiming(Runtime_Type_BP::INIT_SETTINGS_MALLOC)
		+ segmentTimings.getMedianTiming(Runtime_Type_BP::DATA_COSTS_BOTTOM_LEVEL)
		+ segmentTimings.getMedianTiming(Runtime_Type_BP::DATA_COSTS_HIGHER_LEVEL) + segmentTimings.getMedianTiming(Runtime_Type_BP::INIT_MESSAGES)
		+ totalTimeBpIters + totalTimeCopyData + segmentTimings.getMedianTiming(Runtime_Type_BP::OUTPUT_DISPARITY)
		+ segmentTimings.getMedianTiming(Runtime_Type_BP::FINAL_FREE);
	segmentTimings.addTiming(Runtime_Type_BP::TOTAL_TIMED, totalTimed);

	return std::make_pair(resultingDisparityMapCompDevice, segmentTimings);
}


#endif /* PROCESSBPONTARGETDEVICE_H_ */
