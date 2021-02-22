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

template<typename T, typename U, unsigned int DISP_VALS, typename V=float*>
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

	virtual checkerboardMessages<U> retrieveLevelMessageData(const checkerboardMessages<U>& allCheckerboardMessages, const unsigned long offsetIntoMessages)
	{
		checkerboardMessages<U> outputCheckerboardMessages;
		for (unsigned int i = 0; i < outputCheckerboardMessages.checkerboardMessagesAtLevel_.size(); i++)
		{
			outputCheckerboardMessages.checkerboardMessagesAtLevel_[i] =
				&((allCheckerboardMessages.checkerboardMessagesAtLevel_[i])[offsetIntoMessages]);
		}

		return outputCheckerboardMessages;
	}

	virtual void freeDataCostsMemory(const dataCostData<U>& dataCostsToFree)
	{
		freeMemoryOnTargetDevice(dataCostsToFree.dataCostCheckerboard0_);
		freeMemoryOnTargetDevice(dataCostsToFree.dataCostCheckerboard1_);
	}

	virtual dataCostData<U> allocateMemoryForDataCosts(const unsigned long numDataCostsCheckerboard)
	{
		return {allocateMemoryOnTargetDevice(numDataCostsCheckerboard), allocateMemoryOnTargetDevice(numDataCostsCheckerboard)};
	}

	virtual std::pair<dataCostData<U>, checkerboardMessages<U>> allocateAndOrganizeDataCostsAndMessageDataAllLevels(
			const unsigned long numDataAllocatePerDataCostsMessageDataArray)
	{
		U dataAllLevels = allocateMemoryOnTargetDevice(10u*numDataAllocatePerDataCostsMessageDataArray);
		return organizeDataCostsAndMessageDataAllLevels(dataAllLevels, numDataAllocatePerDataCostsMessageDataArray);
	}

	virtual std::pair<dataCostData<U>, checkerboardMessages<U>> organizeDataCostsAndMessageDataAllLevels(
			U dataAllLevels, const unsigned long numDataAllocatePerDataCostsMessageDataArray)
	{
		dataCostData<U> dataCostsDeviceCheckerboardAllLevels;
		dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0_ = dataAllLevels;
		dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard1_ =
				&(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0_[1 * (numDataAllocatePerDataCostsMessageDataArray)]);

		checkerboardMessages<U> messagesDeviceAllLevels;
		for (unsigned int i = 0; i < messagesDeviceAllLevels.checkerboardMessagesAtLevel_.size(); i++) {
			messagesDeviceAllLevels.checkerboardMessagesAtLevel_[i] =
					&(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0_[(i + 2) * (numDataAllocatePerDataCostsMessageDataArray)]);
		}

		return {dataCostsDeviceCheckerboardAllLevels, messagesDeviceAllLevels};
	}

	virtual void freeDataCostsAllDataInSingleArray(const dataCostData<U>& dataCostsToFree)
	{
		freeMemoryOnTargetDevice(dataCostsToFree.dataCostCheckerboard0_);
	}

	virtual dataCostData<U> retrieveLevelDataCosts(const dataCostData<U>& allDataCosts, const unsigned long offsetIntoAllDataCosts)
	{
		return {&(allDataCosts.dataCostCheckerboard0_[offsetIntoAllDataCosts]), &(allDataCosts.dataCostCheckerboard1_[offsetIntoAllDataCosts])};
	}

	//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
	//input is images image1Pixels and image1Pixels
	//output is resultingDisparityMap
	std::pair<V, DetailedTimings<Runtime_Type_BP>> operator()(const std::array<V, 2>& imagesOnTargetDevice,
			const BPsettings& algSettings, const std::array<unsigned int, 2>& widthHeightImages,
			U allocatedMemForBpProcessingDevice = nullptr);
};

//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
//input is images image1Pixels and image1Pixels
//output is resultingDisparityMap
template<typename T, typename U, unsigned int DISP_VALS, typename V>
std::pair<V, DetailedTimings<Runtime_Type_BP>> ProcessBPOnTargetDevice<T, U, DISP_VALS, V>::operator()(const std::array<V, 2> & imagesOnTargetDevice,
	const BPsettings& algSettings, const std::array<unsigned int, 2>& widthHeightImages, U allocatedMemForBpProcessingDevice)
{
	std::unordered_map<Runtime_Type_BP, std::pair<timingType, timingType>> startEndTimes;
	double totalTimeBpIters{0.0}, totalTimeCopyData{0.0}, totalTimeCopyDataKernel{0.0};

	//start at the "bottom level" and work way up to determine amount of space needed to store data costs
	std::vector<levelProperties> bpLevelProperties;
	bpLevelProperties.reserve(algSettings.numLevels_);

	//set level properties for bottom level that include processing of full image width/height
	bpLevelProperties.push_back(levelProperties(widthHeightImages));

	//compute level properties which includes offset for each data/message array for each level after the bottom level
	for (unsigned int levelNum = 1; levelNum < algSettings.numLevels_; levelNum++) {
		//get current level properties from previous level properties
		bpLevelProperties.push_back(bpLevelProperties[levelNum-1].getNextLevelProperties<T>(DISP_VALS));
	}

	startEndTimes[Runtime_Type_BP::INIT_SETTINGS_MALLOC].first = std::chrono::system_clock::now();

	//declare and allocate the space on the device to store the data cost component at each possible movement at each level of the "pyramid"
	//as well as the message data used for bp processing
	//each checkerboard holds half of the data and checkerboard 0 includes the pixel in slot (0, 0)
	dataCostData<U> dataCostsDeviceAllLevels;
	checkerboardMessages<U> messagesDeviceAllLevels;

	//data for each array at all levels set to data up to final level (corresponds to offset at final level) plus data amount at final level
	const unsigned long dataAllLevelsEachDataMessageArr = bpLevelProperties[algSettings.numLevels_-1].offsetIntoArrays_ +
			bpLevelProperties[algSettings.numLevels_-1].getNumDataInBpArrays<T>(DISP_VALS);

	//assuming that width includes padding
	if constexpr (USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT)
	{
		if constexpr (ALLOCATE_FREE_BP_MEMORY_OUTSIDE_RUNS) {
			std::tie(dataCostsDeviceAllLevels, messagesDeviceAllLevels) =
					organizeDataCostsAndMessageDataAllLevels(allocatedMemForBpProcessingDevice, dataAllLevelsEachDataMessageArr);
		}
		else
		{
			//call function that allocates all data in single array and then set offsets in array for data costs and message data locations
			std::tie(dataCostsDeviceAllLevels, messagesDeviceAllLevels) =
					allocateAndOrganizeDataCostsAndMessageDataAllLevels(dataAllLevelsEachDataMessageArr);
		}
	}
	else
	{
		dataCostsDeviceAllLevels = allocateMemoryForDataCosts(dataAllLevelsEachDataMessageArr);
	}

	startEndTimes[Runtime_Type_BP::INIT_SETTINGS_MALLOC].second = std::chrono::system_clock::now();
	startEndTimes[Runtime_Type_BP::DATA_COSTS_BOTTOM_LEVEL].first = std::chrono::system_clock::now();

	//initialize the data cost at the bottom level
	initializeDataCosts(algSettings, bpLevelProperties[0], imagesOnTargetDevice, dataCostsDeviceAllLevels);

	startEndTimes[Runtime_Type_BP::DATA_COSTS_BOTTOM_LEVEL].second = std::chrono::system_clock::now();
	startEndTimes[Runtime_Type_BP::DATA_COSTS_HIGHER_LEVEL].first = std::chrono::system_clock::now();

	//set the data costs at each level from the bottom level "up"
	for (unsigned int levelNum = 1u; levelNum < algSettings.numLevels_; levelNum++)
	{
		initializeDataCurrentLevel(bpLevelProperties[levelNum], bpLevelProperties[levelNum - 1],
				retrieveLevelDataCosts(dataCostsDeviceAllLevels, bpLevelProperties[levelNum - 1u].offsetIntoArrays_),
				retrieveLevelDataCosts(dataCostsDeviceAllLevels, bpLevelProperties[levelNum].offsetIntoArrays_));
	}

	startEndTimes[Runtime_Type_BP::DATA_COSTS_HIGHER_LEVEL].second = std::chrono::system_clock::now();
	startEndTimes[Runtime_Type_BP::INIT_MESSAGES].first = std::chrono::system_clock::now();

	//get and use offset into data at current processing level of pyramid
	dataCostData<U> dataCostsDeviceCurrentLevel = retrieveLevelDataCosts(
			dataCostsDeviceAllLevels, bpLevelProperties[algSettings.numLevels_ - 1u].offsetIntoArrays_);

	//declare the space to pass the BP messages; need to have two "sets" of checkerboards because the message
	//data at the "higher" level in the image pyramid need copied to a lower level without overwriting values
	std::array<checkerboardMessages<U>, 2> messagesDevice;

	//assuming that width includes padding
	if constexpr (USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT)
	{
		messagesDevice[0] = retrieveLevelMessageData(messagesDeviceAllLevels, bpLevelProperties[algSettings.numLevels_ - 1u].offsetIntoArrays_);
	}
	else
	{
		//allocate the space for the message values in the first checkboard set at the current level
		messagesDevice[0] = allocateMemoryForCheckerboardMessages(
				bpLevelProperties[algSettings.numLevels_ - 1u].getNumDataInBpArrays<T>(DISP_VALS));
	}

	startEndTimes[Runtime_Type_BP::INIT_MESSAGES_KERNEL].first = std::chrono::system_clock::now();

	//initialize all the BP message values at every pixel for every disparity to 0
	initializeMessageValsToDefault(bpLevelProperties[algSettings.numLevels_ - 1u], messagesDevice[0]);

	startEndTimes[Runtime_Type_BP::INIT_MESSAGES_KERNEL].second = std::chrono::system_clock::now();
	startEndTimes[Runtime_Type_BP::INIT_MESSAGES].second = std::chrono::system_clock::now();

	//alternate between checkerboard sets 0 and 1
	enum Checkerboard_Num { CHECKERBOARD_ZERO = 0, CHECKERBOARD_ONE = 1 };
	Checkerboard_Num currCheckerboardSet{Checkerboard_Num::CHECKERBOARD_ZERO};

	//run BP at each level in the "pyramid" starting on top and continuing to the bottom
	//where the final movement values are computed...the message values are passed from
	//the upper level to the lower levels; this pyramid methods causes the BP message values
	//to converge more quickly
	for (int levelNum = (int)algSettings.numLevels_ - 1; levelNum >= 0; levelNum--)
	{
		std::chrono::duration<double> diff;
		const auto timeBpIterStart = std::chrono::system_clock::now();

		//need to alternate which checkerboard set to work on since copying from one to the other...need to avoid read-write conflict when copying in parallel
		runBPAtCurrentLevel(algSettings, bpLevelProperties[(unsigned int)levelNum], dataCostsDeviceCurrentLevel, messagesDevice[currCheckerboardSet]);

		const auto timeBpIterEnd = std::chrono::system_clock::now();
		diff = timeBpIterEnd - timeBpIterStart;
		totalTimeBpIters += diff.count();
		const auto timeCopyMessageValuesStart = std::chrono::system_clock::now();

		//if not at the "bottom level" copy the current message values at the current level to the corresponding slots next level
		if (levelNum > 0)
		{
			//use offset into allocated memory at next level
			dataCostsDeviceCurrentLevel = retrieveLevelDataCosts(dataCostsDeviceAllLevels, bpLevelProperties[levelNum - 1].offsetIntoArrays_);

			//assuming that width includes padding
			if constexpr (USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT)
			{
				messagesDevice[(currCheckerboardSet + 1) % 2] = retrieveLevelMessageData(
						messagesDeviceAllLevels, bpLevelProperties[levelNum - 1].offsetIntoArrays_);
			}
			else
			{
				//allocate space in the GPU for the message values in the checkerboard set to copy to
				messagesDevice[(currCheckerboardSet + 1) % 2] = allocateMemoryForCheckerboardMessages(
						bpLevelProperties[levelNum - 1].getNumDataInBpArrays<T>(DISP_VALS));
			}

			const auto timeCopyMessageValuesKernelStart = std::chrono::system_clock::now();

			//currentCheckerboardSet = index copying data from
			//(currentCheckerboardSet + 1) % 2 = index copying data to
			copyMessageValuesToNextLevelDown(bpLevelProperties[levelNum], bpLevelProperties[levelNum - 1],
					messagesDevice[currCheckerboardSet], messagesDevice[(currCheckerboardSet + 1) % 2]);

			const auto timeCopyMessageValuesKernelEnd = std::chrono::system_clock::now();
			diff = timeCopyMessageValuesKernelEnd - timeCopyMessageValuesKernelStart;
			totalTimeCopyDataKernel += diff.count();

			//assuming that width includes padding
			if constexpr (!USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT)
			{
				//free the now-copied from computed data of the completed level
				freeCheckerboardMessagesMemory(messagesDevice[currCheckerboardSet]);
			}

			//alternate between checkerboard parts 1 and 2
			currCheckerboardSet = (currCheckerboardSet == Checkerboard_Num::CHECKERBOARD_ZERO) ?
					Checkerboard_Num::CHECKERBOARD_ONE : Checkerboard_Num::CHECKERBOARD_ZERO;
		}

		const auto timeCopyMessageValuesEnd = std::chrono::system_clock::now();
		diff = timeCopyMessageValuesEnd - timeCopyMessageValuesStart;
		totalTimeCopyData += diff.count();
	}

	startEndTimes[Runtime_Type_BP::OUTPUT_DISPARITY].first = std::chrono::system_clock::now();

	//assume in bottom level when retrieving output disparity
	const V resultingDisparityMapCompDevice = retrieveOutputDisparity(bpLevelProperties[0],
			dataCostsDeviceCurrentLevel, messagesDevice[currCheckerboardSet]);

	startEndTimes[Runtime_Type_BP::OUTPUT_DISPARITY].second = std::chrono::system_clock::now();
	startEndTimes[Runtime_Type_BP::FINAL_FREE].first = std::chrono::system_clock::now();

	if constexpr (USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT)
	{
		if constexpr (ALLOCATE_FREE_BP_MEMORY_OUTSIDE_RUNS) {
          //do nothing; memory free outside of runs
		}
		else
		{
			//now free the allocated data space; all data in single array when
			//USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT set to true
			freeDataCostsAllDataInSingleArray(dataCostsDeviceAllLevels);
		}
	}
	else
	{
		//free the device storage allocated to the message values used to retrieve the output movement values
		freeCheckerboardMessagesMemory(messagesDevice[(currCheckerboardSet == 0) ? 0 : 1]);

		//now free the allocated data space
		freeDataCostsMemory(dataCostsDeviceAllLevels);
	}

	startEndTimes[Runtime_Type_BP::FINAL_FREE].second = std::chrono::system_clock::now();

	//add timing for each runtime segment to segmentTimings object
	DetailedTimings<Runtime_Type_BP> segmentTimings(timingNames_BP);
	std::for_each(startEndTimes.begin(), startEndTimes.end(),
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
