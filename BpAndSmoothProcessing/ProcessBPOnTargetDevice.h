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
#include <vector>
#include <array>
#include <utility>
#include "../ParameterFiles/bpStereoParameters.h"
#include "../ParameterFiles/bpRunSettings.h"
#include "../ParameterFiles/bpStructsAndEnums.h"
#include "../RuntimeTiming/DetailedTimings.h"
#include "../RuntimeTiming/DetailedTimingBPConsts.h"
#include "BpUtilFuncts.h"

typedef std::chrono::time_point<std::chrono::system_clock> timingType;
using timingInSecondsDoublePrecision = std::chrono::duration<double>;

template<typename T, typename U, typename V=float*>
class ProcessBPOnTargetDevice {
public:
	ProcessBPOnTargetDevice() { }
	virtual ~ProcessBPOnTargetDevice() { }

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

	virtual void freeCheckerboardMessagesMemory(const checkerboardMessages<U>& checkerboardMessagesToFree)
	{
		std::for_each(checkerboardMessagesToFree.checkerboardMessagesAtLevel_.begin(), checkerboardMessagesToFree.checkerboardMessagesAtLevel_.end(),
			[this](auto& checkerboardMessagesSet) {
			this->freeMemoryOnTargetDevice(checkerboardMessagesSet); });
	}

	virtual checkerboardMessages<U> allocateMemoryForCheckerboardMessages(const unsigned long numDataAllocatePerMessage)
	{
		checkerboardMessages<U> outputCheckerboardMessages;
		std::for_each(outputCheckerboardMessages.checkerboardMessagesAtLevel_.begin(), outputCheckerboardMessages.checkerboardMessagesAtLevel_.end(),
			[this, numDataAllocatePerMessage](auto& checkerboardMessagesSet) {
			checkerboardMessagesSet = this->allocateMemoryOnTargetDevice(numDataAllocatePerMessage); });

		return outputCheckerboardMessages;
	}

	virtual checkerboardMessages<U> retrieveCurrentCheckerboardMessagesFromOffsetIntoAllCheckerboardMessages(
			const checkerboardMessages<U>& allCheckerboardMessages, const unsigned long offsetIntoAllCheckerboardMessages)
	{
		checkerboardMessages<U> outputCheckerboardMessages;
		for (unsigned int i = 0; i < outputCheckerboardMessages.checkerboardMessagesAtLevel_.size(); i++)
		{
			outputCheckerboardMessages.checkerboardMessagesAtLevel_[i] =
				&((allCheckerboardMessages.checkerboardMessagesAtLevel_[i])[offsetIntoAllCheckerboardMessages]);
		}

		return outputCheckerboardMessages;
	}

	virtual void freeDataCostsMemory(const dataCostData<U>& dataCostsToFree)
	{
		freeMemoryOnTargetDevice(dataCostsToFree.dataCostCheckerboard0_);
		freeMemoryOnTargetDevice(dataCostsToFree.dataCostCheckerboard1_);
	}

	virtual dataCostData<U> allocateMemoryForDataCosts(const unsigned long numDataAllocatePerDataCostsCheckerboard)
	{
		dataCostData<U> outputDataCosts;
		outputDataCosts.dataCostCheckerboard0_ = allocateMemoryOnTargetDevice(numDataAllocatePerDataCostsCheckerboard);
		outputDataCosts.dataCostCheckerboard1_ = allocateMemoryOnTargetDevice(numDataAllocatePerDataCostsCheckerboard);

		return outputDataCosts;
	}

	virtual std::pair<dataCostData<U>, checkerboardMessages<U>> allocateAndOrganizeDataCostsAndMessageDataAllLevels(
			const unsigned long numDataAllocatePerDataCostsMessageDataArray)
	{
		dataCostData<U> dataCostsDeviceCheckerboardAllLevels;
		dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0_ = allocateMemoryOnTargetDevice(10u*numDataAllocatePerDataCostsMessageDataArray);
		dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard1_ =
				&(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0_[1 * (numDataAllocatePerDataCostsMessageDataArray)]);

		checkerboardMessages<U> messagesDeviceAllLevels;
		for (unsigned int i = 0; i < messagesDeviceAllLevels.checkerboardMessagesAtLevel_.size(); i++) {
			messagesDeviceAllLevels.checkerboardMessagesAtLevel_[i] =
					&(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0_[(i + 2) * (numDataAllocatePerDataCostsMessageDataArray)]);
		}

		return std::make_pair(dataCostsDeviceCheckerboardAllLevels, messagesDeviceAllLevels);
	}

	virtual void freeDataCostsAllDataInSingleArray(const dataCostData<U>& dataCostsToFree)
	{
		freeMemoryOnTargetDevice(dataCostsToFree.dataCostCheckerboard0_);
	}

	virtual dataCostData<U> retrieveCurrentDataCostsFromOffsetIntoAllDataCosts(const dataCostData<U>& allDataCosts, const unsigned long offsetIntoAllDataCosts)
	{
		dataCostData<U> outputDataCosts;
		outputDataCosts.dataCostCheckerboard0_ = &(allDataCosts.dataCostCheckerboard0_[offsetIntoAllDataCosts]);
		outputDataCosts.dataCostCheckerboard1_ = &(allDataCosts.dataCostCheckerboard1_[offsetIntoAllDataCosts]);

		return outputDataCosts;
	}

	//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
	//input is images image1Pixels and image1Pixels
	//output is resultingDisparityMap
	std::pair<V, DetailedTimings<Runtime_Type_BP>> operator()(const std::array<V, 2>& imagesOnTargetDevice,
			const BPsettings& algSettings, const std::array<unsigned int, 2>& widthHeightImages);
};

//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
//input is images image1Pixels and image1Pixels
//output is resultingDisparityMap
template<typename T, typename U, typename V>
std::pair<V, DetailedTimings<Runtime_Type_BP>> ProcessBPOnTargetDevice<T, U, V>::operator()(const std::array<V, 2> & imagesOnTargetDevice,
	const BPsettings& algSettings, const std::array<unsigned int, 2>& widthHeightImages)
{
	DetailedTimings<Runtime_Type_BP> segmentTimings(timingNames_BP);
	std::unordered_map<Runtime_Type_BP, std::pair<timingType, timingType>> runtime_start_end_timings;
	double totalTimeBpIters{0.0};
	double totalTimeCopyData{0.0};
	double totalTimeCopyDataKernel{0.0};

	//start at the "bottom level" and work way up to determine amount of space needed to store data costs
	std::vector<levelProperties> processingLevelProperties(algSettings.numLevels_);
	std::vector<unsigned long> offsetAtLevel(algSettings.numLevels_);

	//set level properties and offset for bottom level
	processingLevelProperties[0] = levelProperties(widthHeightImages);
	offsetAtLevel[0] = 0;

	//compute level properties and set offset for each data/message array for each level after the bottom level
	for (unsigned int levelNum = 1; levelNum < algSettings.numLevels_; levelNum++)
	{
		//get current level properties from previous level properties
		processingLevelProperties[levelNum] = processingLevelProperties[levelNum-1].getNextLevelProperties();

		//retrieve amount of data used for processing up to current level in each data/message array and set as offset into data at level
		offsetAtLevel[levelNum] = offsetAtLevel[levelNum-1] + bp_util_functs::getNumDataForAlignedMemoryAtLevel<T>(
			{processingLevelProperties[levelNum - 1].widthLevel_, processingLevelProperties[levelNum - 1].heightLevel_},
			bp_params::NUM_POSSIBLE_DISPARITY_VALUES);
	}

	//data for each array at all levels set to data up to final level (corresponds to offset at final level) plus data amount at final level
	const unsigned long dataAllLevelsEachDataMessageArr =
			offsetAtLevel[algSettings.numLevels_-1] + bp_util_functs::getNumDataForAlignedMemoryAtLevel<T>(
					{processingLevelProperties[algSettings.numLevels_-1].widthLevel_, processingLevelProperties[algSettings.numLevels_-1].heightLevel_},
					bp_params::NUM_POSSIBLE_DISPARITY_VALUES);

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
			allocateAndOrganizeDataCostsAndMessageDataAllLevels(dataAllLevelsEachDataMessageArr);
	}
	else
	{
		dataCostsDeviceCheckerboardAllLevels = allocateMemoryForDataCosts(dataAllLevelsEachDataMessageArr);
	}

	runtime_start_end_timings[Runtime_Type_BP::INIT_SETTINGS_MALLOC].second = std::chrono::system_clock::now();
	runtime_start_end_timings[Runtime_Type_BP::DATA_COSTS_BOTTOM_LEVEL].first = std::chrono::system_clock::now();

	//initialize the data cost at the bottom level
	initializeDataCosts(algSettings, processingLevelProperties[0], imagesOnTargetDevice,
		dataCostsDeviceCheckerboardAllLevels);

	runtime_start_end_timings[Runtime_Type_BP::DATA_COSTS_BOTTOM_LEVEL].second = std::chrono::system_clock::now();
	runtime_start_end_timings[Runtime_Type_BP::DATA_COSTS_HIGHER_LEVEL].first = std::chrono::system_clock::now();

	//set the data costs at each level from the bottom level "up"
	for (unsigned int levelNum = 1u; levelNum < algSettings.numLevels_; levelNum++)
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

	//get and use offset into data at current processing level of pyramid
	dataCostData<U> dataCostsDeviceCheckerboardCurrentLevel =
		retrieveCurrentDataCostsFromOffsetIntoAllDataCosts(dataCostsDeviceCheckerboardAllLevels, offsetAtLevel[algSettings.numLevels_ - 1u]);

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
		messagesDevice[0] = retrieveCurrentCheckerboardMessagesFromOffsetIntoAllCheckerboardMessages(
				messagesDeviceAllLevels, offsetAtLevel[algSettings.numLevels_ - 1u]);
	}
	else
	{
		//retrieve the number of bytes needed to store the data cost/each set of messages in the checkerboard
		const unsigned int numDataAndMessageSetInCheckerboardAtLevel = bp_util_functs::getNumDataForAlignedMemoryAtLevel<T>(
				{processingLevelProperties[algSettings.numLevels_ - 1u].widthLevel_, processingLevelProperties[algSettings.numLevels_ - 1u].heightLevel_},
				bp_params::NUM_POSSIBLE_DISPARITY_VALUES);

		//allocate the space for the message values in the first checkboard set at the current level
		messagesDevice[0] = allocateMemoryForCheckerboardMessages(numDataAndMessageSetInCheckerboardAtLevel);
	}

	runtime_start_end_timings[Runtime_Type_BP::INIT_MESSAGES_KERNEL].first = std::chrono::system_clock::now();

	//initialize all the BP message values at every pixel for every disparity to 0
	initializeMessageValsToDefault(processingLevelProperties[algSettings.numLevels_ - 1u], messagesDevice[0]);

	runtime_start_end_timings[Runtime_Type_BP::INIT_MESSAGES_KERNEL].second = std::chrono::system_clock::now();
	runtime_start_end_timings[Runtime_Type_BP::INIT_MESSAGES].second = std::chrono::system_clock::now();

	//alternate between checkerboard sets 0 and 1
	enum Checkerboard_Num { CHECKERBOARD_ZERO = 0, CHECKERBOARD_ONE = 1 };
	Checkerboard_Num currentCheckerboardSet{Checkerboard_Num::CHECKERBOARD_ZERO};

	//run BP at each level in the "pyramid" starting on top and continuing to the bottom
	//where the final movement values are computed...the message values are passed from
	//the upper level to the lower levels; this pyramid methods causes the BP message values
	//to converge more quickly
	for (int levelNum = (int)algSettings.numLevels_ - 1; levelNum >= 0; levelNum--)
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
			//use offset into allocated memory at next level
			dataCostsDeviceCheckerboardCurrentLevel =
				retrieveCurrentDataCostsFromOffsetIntoAllDataCosts(dataCostsDeviceCheckerboardAllLevels, offsetAtLevel[levelNum - 1]);

#ifdef _WIN32
			//assuming that width includes padding
			if /*constexpr*/ (USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT)
#else
			if (USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT)
#endif
			{
				messagesDevice[(currentCheckerboardSet + 1) % 2] =
						retrieveCurrentCheckerboardMessagesFromOffsetIntoAllCheckerboardMessages(messagesDeviceAllLevels, offsetAtLevel[levelNum - 1]);
			}
			else
			{
				//update the number of bytes needed to store each set
				const unsigned int numDataAndMessageSetInCheckerboardAtLevel = bp_util_functs::getNumDataForAlignedMemoryAtLevel<T>(
						{processingLevelProperties[levelNum - 1].widthLevel_, processingLevelProperties[levelNum - 1].heightLevel_},
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

	return {resultingDisparityMapCompDevice, segmentTimings};
}


#endif /* PROCESSBPONTARGETDEVICE_H_ */
